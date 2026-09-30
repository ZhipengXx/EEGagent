"""Approved spec → EffectiveHookConfig written next to the job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.experiment_gate import CanonicalizationError, canonicalize_experiment_spec
from react_agent.eeg_research.agentic.run_context import intervention_config_hash, run_config_hash

SCHEMA = "eeg_research.hook_config.v2"


class HookConfigError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def normalize_hook_config(
    spec: dict[str, Any] | None,
    *,
    spec_ref: str | None = None,
    spec_hash: str | None = None,
    seed: int | None = None,
    fidelity: str | None = None,
) -> dict[str, Any]:
    """Expand defaults once after canonicalization. Hash is computed on canonical fields."""
    spec = spec if isinstance(spec, dict) else {}
    if spec.get("schema_version") == "eeg_research.frozen_run_spec.v1" or spec.get("hook_spec"):
        hook = spec.get("hook_spec") if isinstance(spec.get("hook_spec"), dict) else spec
        model = dict(hook.get("model") or spec.get("model") or {})
        objective = dict(hook.get("objective") or spec.get("objective") or {})
        transform = dict(hook.get("transform") or spec.get("transform") or {})
        policy = hook.get("negative_sampling_policy") or spec.get("negative_policy") or spec.get("negative_sampling_policy")
        spec_hash = spec_hash or spec.get("spec_hash")
        spec_ref = spec_ref or spec.get("approval_ref") or spec.get("spec_ref")
        intervention = spec.get("intervention_config_hash")
        run_hash = spec.get("run_config_hash")
    else:
        try:
            canonical = canonicalize_experiment_spec(spec)
        except CanonicalizationError as exc:
            raise HookConfigError(exc.reason) from exc
        model = dict(canonical.get("model") or {})
        objective = dict(canonical.get("objective") or {})
        transform = dict(canonical.get("transform") or {})
        policy = canonical.get("negative_sampling_policy") or "data_parallel_local"
        spec_hash = spec_hash or canonical.get("spec_hash") or spec.get("spec_hash")
        spec_ref = spec_ref or canonical.get("experiment_ref") or spec.get("experiment_ref") or spec.get("spec_ref")
        intervention = None
        run_hash = None
    hook = {
        "model": model,
        "objective": objective,
        "transform": transform,
        "negative_sampling_policy": policy or "data_parallel_local",
    }
    intervention = intervention or intervention_config_hash(hook)
    run_hash = run_hash or run_config_hash(hook, seed=seed, fidelity=fidelity)
    payload = {
        "schema_version": SCHEMA,
        "spec_ref": spec_ref,
        "spec_hash": spec_hash,
        "model": model,
        "objective": objective,
        "transform": transform,
        "negative_sampling_policy": hook["negative_sampling_policy"],
        "intervention_config_hash": intervention,
        "run_config_hash": run_hash,
        "config_hash": intervention,
        "seed": seed,
        "fidelity": fidelity,
    }
    if not payload["spec_hash"]:
        payload["spec_hash"] = _digest(
            {key: payload[key] for key in ("model", "objective", "transform", "negative_sampling_policy")}
        )
    return payload


def write_hook_config(
    job_dir: Path,
    spec: dict[str, Any] | None,
    *,
    spec_ref: str | None = None,
    spec_hash: str | None = None,
    seed: int | None = None,
    fidelity: str | None = None,
) -> dict[str, Any]:
    """Persist the effective hook config the trainer will load."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = normalize_hook_config(spec, spec_ref=spec_ref, spec_hash=spec_hash, seed=seed, fidelity=fidelity)
    (job_dir / "hook_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
