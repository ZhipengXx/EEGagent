"""Approved spec → EffectiveHookConfig written next to the job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "eeg_research.hook_config.v1"


def normalize_hook_config(spec: dict[str, Any] | None, *, spec_ref: str | None = None, spec_hash: str | None = None) -> dict[str, Any]:
    """Expand defaults once. Later readers must not guess missing sections."""
    spec = spec if isinstance(spec, dict) else {}
    payload = {
        "schema_version": SCHEMA,
        "spec_ref": spec_ref or spec.get("experiment_ref") or spec.get("spec_ref"),
        "spec_hash": spec_hash or spec.get("spec_hash"),
        "model": dict(spec.get("model_config") or spec.get("model") or {}),
        "objective": dict(spec.get("objective_config") or spec.get("objective") or {}),
        "transform": dict(spec.get("transform_config") or spec.get("transform") or {}),
        "negative_sampling_policy": spec.get("negative_sampling_policy") or "data_parallel_local",
    }
    payload["config_hash"] = hashlib.sha256(
        json.dumps(
            {key: payload[key] for key in ("model", "objective", "transform", "negative_sampling_policy")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def write_hook_config(job_dir: Path, spec: dict[str, Any] | None, *, spec_ref: str | None = None) -> dict[str, Any]:
    """Persist the effective hook config the trainer will load."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = normalize_hook_config(spec, spec_ref=spec_ref)
    (job_dir / "hook_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
