"""Job-level diagnostic calculations. Missing inputs are unavailable, not zeros."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_training.hooks import image_id_duplicate_stats

UNAVAILABLE = "unavailable"


def _item(status: str, payload: dict[str, Any] | None = None, reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"status": status}
    if payload is not None:
        row["payload"] = payload
    if reason:
        row["reason"] = reason
    if status == UNAVAILABLE:
        row.setdefault("reason", reason or UNAVAILABLE)
    return row


def training_dynamics(job_dir: Path) -> dict[str, Any]:
    history = job_dir / "history.jsonl"
    if not history.is_file():
        return _item(UNAVAILABLE, reason="history_missing")
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return _item(UNAVAILABLE, reason="history_empty")
    last = rows[-1]
    return _item(
        "observed",
        {
            "epochs": len(rows),
            "last_loss": last.get("train_loss") or last.get("loss"),
            "last_fixed_bank_top1": last.get("fixed_bank_top1"),
            "last_fixed_bank_top5": last.get("fixed_bank_top5"),
            "sample_count": len(rows),
        },
    )


def duplicate_audit(batches: list[list[str]]) -> dict[str, Any]:
    if not batches:
        return _item(UNAVAILABLE, reason="no_train_batches")
    stats = [image_id_duplicate_stats(batch) for batch in batches]
    mean_rate = sum(row["duplicate_rate"] for row in stats) / len(stats)
    return _item(
        "observed",
        {
            "batch_count": len(stats),
            "mean_duplicate_rate": mean_rate,
            "max_same_image": max(row["max_same_image"] for row in stats),
            "batches": stats[:8],
        },
    )


def _load_duplicate_batches(job_dir: Path) -> list[list[str]]:
    path = job_dir / "train_batch_image_ids.jsonl"
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            rows.append([str(item) for item in payload])
    return rows


def compute_job_diagnostics(job_dir: Path, *, batches: list[list[str]] | None = None) -> dict[str, Any]:
    """Fill every DiagnosticBundle category. Absent files stay unavailable."""
    job_dir = Path(job_dir)
    metrics = {}
    metrics_path = job_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    identity = job_dir / "evaluation_identity.json"
    binding = job_dir / "source_binding.json"
    capabilities = job_dir / "capabilities_used.json"
    if batches is None:
        batches = _load_duplicate_batches(job_dir)
    return {
        "schema_version": "eeg_research.diagnostic_bundle.v1",
        "data_audit": _item(UNAVAILABLE, reason="raw_eeg_not_in_job_dir") if not identity.is_file() else _item(
            "observed",
            {
                "evaluation_identity": True,
                "query_count": metrics.get("query_count"),
                "gallery_size": metrics.get("validation_image_count") or metrics.get("candidate_count"),
                "sampling_rate": UNAVAILABLE,
                "repeats_policy": "mean_over_repeats",
                "split_overlap": metrics.get("train_validation_overlap"),
            },
        ),
        "training_dynamics": training_dynamics(job_dir),
        "retrieval_errors": _item(UNAVAILABLE, reason="query_ranks_not_exported"),
        "representation": _item(UNAVAILABLE, reason="embeddings_not_exported"),
        "group_results": _item(UNAVAILABLE, reason="subject_metadata_not_trusted"),
        "intervention_probe": _item(UNAVAILABLE, reason="occlusion_not_requested"),
        "integrity_cost": _item(
            "observed" if binding.is_file() else UNAVAILABLE,
            {
                "source_binding": binding.is_file(),
                "checkpoint": (job_dir / "last.ckpt").is_file(),
                "capabilities_used": capabilities.is_file(),
                "gpu_seconds": None,
            },
            reason=None if binding.is_file() else "binding_missing",
        ),
        "image_id_duplicates": duplicate_audit(batches or []),
    }
