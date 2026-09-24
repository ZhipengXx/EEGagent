"""Launch the baseline retrieval entry. A missing file is not a score."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec
from react_agent.eeg_training.protocol import Design, SplitError, probe_blockers, train_command


def child_env() -> dict[str, str]:
    """Point the torch interpreter at this package without installing it there."""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[3])
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (src, current) if part)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def failure_detail(out: Path) -> str:
    """Prefer error.txt, then the last log line, then a generic failure."""
    error = out / "error.txt"
    if error.is_file() and error.read_text(encoding="utf-8").strip():
        return error.read_text(encoding="utf-8").strip()
    log = out / "train.log"
    if log.is_file():
        lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "train_entry_failed"


class UbpRetrievalAdapter:
    """Run one EEG or MEG design in a child process."""

    adapter_id = "ubp_retrieval"
    requires_gpu_cap = True

    def __init__(self, design: Design, data_root: Path, timeout_s: float | None) -> None:
        self.design = design
        self.data_root = data_root
        self.timeout_s = timeout_s
        self.blockers = probe_blockers(design, data_root)
        self.availability = "unverified" if self.blockers else "available"
        self.reason = "; ".join(self.blockers) if self.blockers else "baseline entry probed"

    def probe(self) -> dict[str, object]:
        """Re-check files and the training interpreter."""
        self.blockers = probe_blockers(self.design, self.data_root)
        self.availability = "unverified" if self.blockers else "available"
        self.reason = "; ".join(self.blockers) if self.blockers else "baseline entry probed"
        return {"ok": not self.blockers, "availability": self.availability, "reason": self.reason}

    def run(self, spec: ExperimentSpec, output_dir: str) -> EvidenceBundle:
        """Start train_entry. Refuse before the process when the probe failed."""
        report = self.probe()
        if not report["ok"]:
            raise RuntimeError(str(report["reason"]))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = out / "metrics.json"
        if metrics.exists() and not self.design.test_only:
            raise RuntimeError("metrics already present")
        lock_path = out / ".lock"
        with lock_path.open("w", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("trial lock busy") from exc
            command = train_command(self.design, out, self.data_root)
            proc = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError("per_trial_timeout") from None
            (out / "train.log").write_bytes((stdout or b"") + b"\n" + (stderr or b""))
            if proc.returncode != 0 or not metrics.is_file():
                raise RuntimeError(failure_detail(out))
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        if payload.get("train_validation_overlap"):
            metrics.unlink()
            raise SplitError("overlap_in_metrics")
        fixed = payload.get("metric_name") == "fixed_bank_top1"
        return EvidenceBundle(
            experiment_id=spec.id,
            execution_status="succeeded",
            protocol_status="valid",
            primary_metric=float(payload["primary_metric"]),
            metric_name="fixed_bank_top1" if fixed else "top1",
            evaluator="ubp_fixed_bank" if fixed else "ubp_baseline_within_batch",
            seed=spec.seed,
        )

    def evaluate(self, trial_dir: Path) -> dict[str, object]:
        """Score an existing checkpoint into eval_scores.json. Training does not restart."""
        report = self.probe()
        if not report["ok"]:
            raise RuntimeError(str(report["reason"]))
        out = Path(trial_dir)
        target = out / "eval_scores.json"
        lock_path = out / ".lock"
        with lock_path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("trial lock busy") from exc
            command = train_command(replace(self.design, evaluate_only=True, test_only=False), out, self.data_root)
            proc = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError("per_trial_timeout") from None
            (out / "eval.log").write_bytes((stdout or b"") + b"\n" + (stderr or b""))
        if not target.is_file():
            raise RuntimeError("eval_scores_missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if proc.returncode != 0 or payload.get("error"):
            raise RuntimeError(str(payload.get("error") or "evaluate_failed"))
        return payload
