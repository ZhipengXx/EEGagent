"""One campaign's trial trace, read from disk. The adaptive loop keeps no other state."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRIAL_CONFIG = "trial_config.json"
EVAL_SCORES = "eval_scores.json"
RESEARCH_TRACE = "research_trace.jsonl"


@dataclass
class TrialRecord:
    """What one trial directory shows. Test scores stay out of the planner view."""

    name: str
    index: int
    path: Path
    config: dict[str, Any] | None
    status: str
    epoch: int
    epochs: int
    within_batch_top1: float | None
    fixed_bank_top1: float | None
    fixed_bank_top5: float | None
    test_result: dict[str, Any] | None
    checkpoint: bool
    score_error: str | None
    validation_candidate_count: float | None = None
    duplicate_of: str | None = None
    digest: str | None = None

    def key(self, base: dict[str, Any]) -> tuple[object, ...]:
        """Setting identity. A trial without a config is counted as the base setting."""
        return setting_key(self.config or base)

    def planner_view(self) -> dict[str, Any]:
        """Fields the planner may see. The test result is omitted."""
        return {
            "trial": self.name,
            "status": self.status,
            "setting": self.config or "设置未记录（按默认设置计）",
            "epoch": self.epoch,
            "epochs": self.epochs,
            "fixed_bank_top1": self.fixed_bank_top1,
            "fixed_bank_top5": self.fixed_bank_top5,
            "duplicate_of": self.duplicate_of,
            "score_error": self.score_error,
        }

    def page_view(self) -> dict[str, Any]:
        """Fields the workbench shows. The test result is labelled as unused for selection."""
        row = self.planner_view()
        row["within_batch_top1"] = self.within_batch_top1
        row["test_result"] = self.test_result
        return row


def setting_key(config: dict[str, Any]) -> tuple[object, ...]:
    """Seed, learning rate and weight decay. Other fields stay on the frozen design."""
    return (
        int(config.get("seed", 0)),
        f"{float(config.get('learning_rate', 0.0)):.8g}",
        f"{float(config.get('weight_decay', 0.0)):.8g}",
    )


def write_trial_config(path: Path, config: dict[str, Any]) -> None:
    """Record a trial's setting before its process starts."""
    path.mkdir(parents=True, exist_ok=True)
    (path / TRIAL_CONFIG).write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _locked(path: Path) -> bool:
    lock = path / ".lock"
    if not lock.is_file():
        return False
    with lock.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _validation_candidates(metrics: dict[str, Any] | None, scores: dict[str, Any]) -> float | None:
    """Size of the validation bank. The test bank is a different number and is not used."""
    if metrics is not None:
        count = _float(metrics.get("validation_image_count"))
        if count is not None:
            return count
    return _float(scores.get("candidate_count"))


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _digest(path: Path) -> str | None:
    ckpt = path / "last.ckpt"
    if not ckpt.is_file():
        return None
    stat = ckpt.stat()
    cache_key = (str(ckpt), stat.st_size, stat.st_mtime_ns)
    if cache_key in _DIGESTS:
        return _DIGESTS[cache_key]
    sha = hashlib.sha1()
    with ckpt.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    _DIGESTS[cache_key] = sha.hexdigest()
    return _DIGESTS[cache_key]


def _status(path: Path, metrics: dict[str, Any] | None, status: dict[str, Any] | None) -> str:
    if _locked(path):
        return "running"
    phase = str((status or {}).get("phase") or "")
    if (path / "error.txt").is_file() or phase == "failed":
        return "failed"
    if metrics is not None:
        return "finished"
    if phase in {"loading", "training"}:
        return "interrupted"
    return "unknown"


def _record(path: Path, index: int) -> TrialRecord:
    metrics = _json(path / "metrics.json")
    status = _json(path / "status.json")
    scores = _json(path / EVAL_SCORES) or {}
    config = _json(path / TRIAL_CONFIG)
    source = metrics or {}
    fixed1 = _float(source.get("fixed_bank_top1"))
    fixed5 = _float(source.get("fixed_bank_top5"))
    if fixed1 is None:
        fixed1 = _float(scores.get("fixed_bank_top1"))
        fixed5 = _float(scores.get("fixed_bank_top5"))
    within = _float(source.get("within_batch_top1"))
    if within is None and source.get("metric_name") == "top1":
        within = _float(source.get("primary_metric"))
    test = source.get("test_result") if isinstance(source.get("test_result"), dict) else None
    if test is None and isinstance(scores.get("test_result"), dict):
        test = scores["test_result"]
    return TrialRecord(
        name=path.name,
        index=index,
        path=path,
        config=config,
        status=_status(path, metrics, status),
        epoch=int((status or {}).get("epoch") or 0),
        epochs=int((status or {}).get("epochs") or 0),
        within_batch_top1=within,
        fixed_bank_top1=fixed1,
        fixed_bank_top5=fixed5,
        test_result=test,
        checkpoint=(path / "last.ckpt").is_file(),
        score_error=str(scores["error"]) if scores.get("error") else None,
        validation_candidate_count=_validation_candidates(metrics, scores),
    )


def read_trace(out_dir: Path) -> list[TrialRecord]:
    """Every trials/tN directory, oldest first. Identical legacy checkpoints point at the first copy."""
    root = Path(out_dir) / "trials"
    if not root.is_dir():
        return []
    found: list[TrialRecord] = []
    for path in root.iterdir():
        suffix = path.name[1:]
        if path.is_dir() and path.name.startswith("t") and suffix.isdigit():
            found.append(_record(path, int(suffix)))
    found.sort(key=lambda row: row.index)
    first: dict[str, str] = {}
    for row in found:
        if row.config is not None or row.status != "finished":
            continue
        row.digest = _digest(row.path)
        if row.digest is None:
            continue
        if row.digest in first:
            row.duplicate_of = first[row.digest]
        else:
            first[row.digest] = row.name
    return found


def next_trial_index(trials: list[TrialRecord]) -> int:
    """New trials never reuse a directory."""
    return max((row.index for row in trials), default=0) + 1


def distinct_settings(trials: list[TrialRecord], base: dict[str, Any]) -> int:
    """Count settings that ran or are running. Duplicates of one setting count once."""
    keys = {row.key(base) for row in trials if row.status in {"finished", "running"}}
    return len(keys)


def find_setting(trials: list[TrialRecord], key: tuple[object, ...], base: dict[str, Any]) -> TrialRecord | None:
    """The first finished or running trial with this setting."""
    for row in trials:
        if row.status in {"finished", "running"} and row.key(base) == key:
            return row
    return None


def needs_scoring(trials: list[TrialRecord]) -> list[TrialRecord]:
    """Finished trials with a checkpoint and no validation fixed-bank score, once per checkpoint."""
    return [
        row
        for row in trials
        if row.status == "finished"
        and row.checkpoint
        and row.fixed_bank_top1 is None
        and row.score_error is None
        and row.duplicate_of is None
    ]


def scored(trials: list[TrialRecord]) -> list[TrialRecord]:
    """Trials that carry a validation fixed-bank score. Duplicate copies are skipped."""
    return [row for row in trials if row.duplicate_of is None and row.fixed_bank_top1 is not None]


def local_actions(out_dir: Path) -> list[dict[str, Any]]:
    """Non-training actions recorded for this campaign."""
    path = Path(out_dir) / RESEARCH_TRACE
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_local(out_dir: Path, action: str, findings: dict[str, Any], trial_count: int) -> None:
    """Record one non-training action with the number of finished trials it saw."""
    path = Path(out_dir) / RESEARCH_TRACE
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"action": action, "findings": findings, "trial_count": trial_count}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finished_count(trials: list[TrialRecord]) -> int:
    return sum(1 for row in trials if row.status == "finished")
