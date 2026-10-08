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


def _training_execution_summary(item: Any) -> dict[str, Any]:
    """Keep observed loss/batch scalars while omitting device/debug receipts."""
    item = item if isinstance(item, dict) else {}
    projected = {"status": item.get("status") or UNAVAILABLE}
    if "reason" in item:
        projected["reason"] = item["reason"]
    payload = item.get("payload")
    if isinstance(payload, dict):
        kept = {key: payload[key] for key in (
            "loss_definition", "loss_direction_reduction", "loss_replica_reduction",
            "negative_sampling_policy", "objective_is_custom", "optimizer_learning_rates",
            "embedding_dtype", "epoch_reduction", "coverage", "limitations",
        ) if key in payload}
        if isinstance(payload.get("batches"), list):
            batches = []
            for batch in payload["batches"]:
                if not isinstance(batch, dict):
                    continue
                observed = {key: batch[key] for key in (
                    "epoch", "batch_index", "batch_position", "global_batch_size",
                    "uniform_logit_reference",
                ) if key in batch}
                replicas = batch.get("replicas")
                if isinstance(replicas, list):
                    observed["local_batch_sizes"] = [replica.get("local_batch_size") for replica in replicas]
                    observed["logit_scales"] = [replica.get("logit_scale") for replica in replicas]
                batches.append(observed)
            kept["batches"] = batches
        projected["payload"] = kept
    return projected


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
                "mean_norm",
                "mean_dimension_variance",
                "last_loss",
                "coverage",
                "scope",
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
            if name == "training_dynamics" and "training_execution" in payload:
                summary["items"][name]["training_execution"] = _training_execution_summary(payload["training_execution"])
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
