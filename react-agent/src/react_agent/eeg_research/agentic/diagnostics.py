"""Planner-facing diagnostic summaries. Full files stay on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_training.diagnostics import UNAVAILABLE, compute_job_diagnostics

CATEGORIES = (
    "data_audit",
    "training_dynamics",
    "retrieval_errors",
    "representation",
    "group_results",
    "intervention_probe",
    "hook_consumption",
    "integrity_cost",
    "image_id_duplicates",
)


def summarize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compact view for the planner. Unavailable stays unavailable."""
    summary = {"schema_version": bundle.get("schema_version"), "items": {}}
    for name in CATEGORIES:
        item = bundle.get(name) or {"status": UNAVAILABLE, "reason": UNAVAILABLE}
        summary["items"][name] = {
            "status": item.get("status") or UNAVAILABLE,
            "reason": item.get("reason"),
        }
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if payload:
            for key in (
                "sample_count",
                "definition",
                "mean_margin",
                "hit_rate",
                "effective_rank",
                "effective_rank_definition",
                "method",
                "last_fixed_bank_top1",
                "epochs",
                "best_epoch",
                "mean_duplicate_rate",
                "query_count",
                "gallery_size",
                "execution_status",
                "mismatches",
            ):
                if key in payload:
                    summary["items"][name][key] = payload[key]
    return summary


def write_job_bundle(job_dir: Path, *, batches: list[list[str]] | None = None) -> dict[str, Any]:
    """Persist the full bundle and a planner summary next to the job."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    bundle = compute_job_diagnostics(job_dir, batches=batches)
    summary = summarize_bundle(bundle)
    (job_dir / "diagnostic_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
