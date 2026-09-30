"""Job metrics for the workbench. History follows the job directory, not a guessed layout."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.jsonl_read import read_jsonl
from react_agent.eeg_research.agentic.paths import resolve_campaign, resolve_job_dir


def _read(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_unreadable": True, "path": path.name}


def _number(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "missing", "value": None, "raw": None}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"status": "invalid", "value": None, "raw": value}
    if not math.isfinite(number):
        return {"status": "non_finite", "value": None, "raw": value}
    return {"status": "ok", "value": number, "raw": value}


def history_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        epoch = row.get("epoch")
        try:
            epoch_n = int(epoch)
        except (TypeError, ValueError):
            continue
        point = {
            "epoch": epoch_n,
            "train_loss": _number(row.get("train_loss", row.get("loss"))),
            "fixed_bank_top1": _number(row.get("fixed_bank_top1")),
            "fixed_bank_top5": _number(row.get("fixed_bank_top5")),
            "batch_val_top1": _number(row.get("val_top1")),
            "batch_val_top5": _number(row.get("val_top5")),
            "batch_metrics_diagnostic_only": True,
        }
        points.append(point)
    points.sort(key=lambda item: item["epoch"])
    return points


def job_metrics(root: Path, campaign: str, job_id: str) -> dict[str, Any]:
    camp = resolve_campaign(root, campaign)
    if camp is None:
        return {"ok": False, "error": "campaign_missing"}
    job_dir = resolve_job_dir(camp, job_id)
    if job_dir is None:
        return {"ok": False, "error": "job_missing"}
    record = _read(job_dir / "job.json")
    if not isinstance(record, dict):
        record = {"job_id": job_dir.name}
    status = _read(job_dir / "status.json")
    if not isinstance(status, dict):
        status = {}
    identity = _read(job_dir / "evaluation_identity.json")
    frozen = _read(job_dir / "frozen_run_spec.json")
    history = read_jsonl(job_dir / "history.jsonl")
    points = history_points(history["rows"])
    contract = _read(camp / "evaluation_contract.json") or {}
    attempt = None
    candidate_id = record.get("candidate_id")
    if candidate_id:
        attempt_path = camp / "candidates" / str(candidate_id) / "attempt.json"
        loaded = _read(attempt_path)
        if isinstance(loaded, dict):
            attempt = loaded.get("attempt_id")
    epochs_budget = record.get("epochs")
    if epochs_budget is None:
        epochs_budget = status.get("epochs")
    completed = None
    if points:
        completed = max(item["epoch"] for item in points)
    elif status.get("epoch") is not None:
        try:
            completed = int(status.get("epoch"))
        except (TypeError, ValueError):
            completed = None
    return {
        "ok": True,
        "schema_version": "eeg_research.job_metrics.v1",
        "campaign_id": camp.name,
        "candidate_id": candidate_id,
        "attempt_id": attempt,
        "job_id": job_dir.name,
        "seed": record.get("seed") if record.get("seed") is not None else record.get("training_seed"),
        "fidelity": record.get("fidelity"),
        "status": record.get("status") or status.get("phase"),
        "epochs_budget": epochs_budget,
        "epochs_completed": completed,
        "evaluation_contract": {
            "fingerprint": contract.get("fingerprint") if isinstance(contract, dict) else None,
            "research_scope": contract.get("research_scope") if isinstance(contract, dict) else None,
            "final_test_enabled": contract.get("final_test_enabled") if isinstance(contract, dict) else None,
        },
        "evaluation_identity": identity if isinstance(identity, dict) and not identity.get("_unreadable") else None,
        "frozen_run_spec": frozen if isinstance(frozen, dict) and not frozen.get("_unreadable") else None,
        "history": points,
        "history_diagnostics": history["diagnostics"],
        "history_empty": history["empty"],
        "history_truncated": history["truncated"],
        "source": {"job_dir": job_dir.name, "history": "jobs/<job_id>/history.jsonl"},
    }
