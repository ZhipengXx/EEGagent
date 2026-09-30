"""Per-target run bindings and FrozenRunSpec. The current campaign experiment is not a global hook spec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.experiment_gate import ExperimentResolutionError

BASELINE_TARGET = "baseline"
BASELINE_HOOK_SPEC: dict[str, Any] = {
    "model": {},
    "objective": {},
    "transform": {},
}
BINDING_SCHEMA = "eeg_research.candidate_binding.v2"
FROZEN_SCHEMA = "eeg_research.frozen_run_spec.v1"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_approved_binding(
    camp: Path,
    target_id: str,
    spec: dict[str, Any],
    *,
    spec_ref: str | None = None,
    attempt_id: str | None = None,
    source_hash: str | None = None,
) -> dict[str, Any]:
    """Write the approved hook binding for one candidate. Baseline is never overwritten by this."""
    target = str(target_id or "")
    if not target or target == BASELINE_TARGET:
        return dict(BASELINE_HOOK_SPEC)
    dest = Path(camp) / "candidates" / target
    dest.mkdir(parents=True, exist_ok=True)
    record = spec.get("approval_record") if isinstance(spec.get("approval_record"), dict) else None
    binding = {
        "schema_version": BINDING_SCHEMA,
        "target_id": target,
        "model": spec.get("model") if isinstance(spec.get("model"), dict) else {},
        "objective": spec.get("objective") if isinstance(spec.get("objective"), dict) else {},
        "transform": spec.get("transform") if isinstance(spec.get("transform"), dict) else {},
        "negative_sampling_policy": spec.get("negative_sampling_policy"),
        "spec_hash": spec.get("spec_hash"),
        "spec_ref": spec_ref or spec.get("experiment_ref") or (record or {}).get("artifact_ref"),
        "approval_record": record,
        "attempt_id": attempt_id,
        "source_hash": source_hash,
        "parent_candidate_id": spec.get("parent_candidate_id") or "baseline",
        "control_candidate_id": spec.get("control_candidate_id") or spec.get("control_id") or "baseline",
        "intervention": spec.get("principal_intervention") or spec.get("intervention"),
        "hypothesis": spec.get("hypothesis"),
    }
    (dest / "approved_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding="utf-8")
    return binding


def load_approved_binding(camp: Path, target_id: str) -> dict[str, Any] | None:
    path = Path(camp) / "candidates" / str(target_id) / "approved_binding.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _hook_spec_from(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": payload.get("model") if isinstance(payload.get("model"), dict) else {},
        "objective": payload.get("objective") if isinstance(payload.get("objective"), dict) else {},
        "transform": payload.get("transform") if isinstance(payload.get("transform"), dict) else {},
        "negative_sampling_policy": payload.get("negative_sampling_policy"),
        "spec_hash": payload.get("spec_hash"),
        "spec_ref": payload.get("spec_ref"),
    }


def intervention_config_hash(hook: dict[str, Any]) -> str:
    return _digest(
        {
            "model": hook.get("model") or {},
            "objective": hook.get("objective") or {},
            "transform": hook.get("transform") or {},
            "negative_sampling_policy": hook.get("negative_sampling_policy"),
        }
    )


def run_config_hash(hook: dict[str, Any], *, seed: int | None, fidelity: str | None, extra: dict[str, Any] | None = None) -> str:
    body = {
        "model": hook.get("model") or {},
        "objective": hook.get("objective") or {},
        "transform": hook.get("transform") or {},
        "negative_sampling_policy": hook.get("negative_sampling_policy"),
        "seed": seed,
        "fidelity": fidelity,
    }
    if extra:
        body.update(extra)
    return _digest(body)


def build_frozen_run_spec(
    camp: Path,
    target_id: str | None,
    state: dict[str, Any] | None = None,
    *,
    fidelity: str | None = None,
    seed: int | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Immutable execution identity. Missing non-baseline bindings do not fall back to baseline."""
    state = state if isinstance(state, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    target = str(target_id or BASELINE_TARGET)
    if target == BASELINE_TARGET:
        hook = dict(BASELINE_HOOK_SPEC)
        binding = {"target_id": target, **BASELINE_HOOK_SPEC, "spec_hash": protocol.get("baseline_spec_hash") or "baseline"}
        spec_ref = None
        spec_hash = binding.get("spec_hash") or "baseline"
        source_hash = None
        control = "baseline"
    else:
        binding = load_approved_binding(camp, target)
        if binding is None:
            raise ExperimentResolutionError("missing_binding")
        hook = _hook_spec_from(binding)
        spec_ref = binding.get("spec_ref")
        spec_hash = binding.get("spec_hash")
        if not spec_hash:
            raise ExperimentResolutionError("missing_binding")
        source_hash = binding.get("source_hash")
        control = binding.get("control_candidate_id") or "baseline"
    frozen = {
        "schema_version": FROZEN_SCHEMA,
        "target_id": target,
        "target_revision": source_hash or spec_hash,
        "approval_ref": spec_ref,
        "spec_hash": spec_hash,
        "context_hash": (binding.get("approval_record") or {}).get("context_hash") if isinstance(binding.get("approval_record"), dict) else None,
        "source_manifest_hash": source_hash,
        "model": hook.get("model") or {},
        "objective": hook.get("objective") or {},
        "transform": hook.get("transform") or {},
        "recipe": {
            "fidelity": fidelity,
            "seed": seed,
            "negative_sampling_policy": hook.get("negative_sampling_policy") or protocol.get("negative_sampling_policy"),
        },
        "negative_policy": hook.get("negative_sampling_policy") or protocol.get("negative_sampling_policy"),
        "evaluation_hash": protocol.get("fingerprint") or state.get("execution_fingerprint") or state.get("contract_fingerprint"),
        "fidelity": fidelity,
        "seed": seed,
        "control_identity": control,
        "intervention_config_hash": intervention_config_hash(hook),
        "run_config_hash": run_config_hash(hook, seed=seed, fidelity=fidelity),
        "hook_spec": hook,
        "approved_binding": binding,
        "spec_ref": spec_ref,
    }
    return frozen


def resolve_run_context(
    camp: Path,
    target_id: str | None,
    state: dict[str, Any] | None = None,
    *,
    fidelity: str | None = None,
    seed: int | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the FrozenRunSpec for this target. Baseline stays isolated; others require a binding."""
    frozen = build_frozen_run_spec(camp, target_id, state, fidelity=fidelity, seed=seed, protocol=protocol)
    return {
        "target_id": frozen["target_id"],
        "hook_spec": frozen["hook_spec"],
        "spec_ref": frozen.get("spec_ref"),
        "spec_hash": frozen.get("spec_hash"),
        "fidelity": fidelity,
        "seed": seed,
        "approved_binding": frozen.get("approved_binding"),
        "frozen_run_spec": frozen,
        "intervention_config_hash": frozen["intervention_config_hash"],
        "run_config_hash": frozen["run_config_hash"],
    }


def persist_frozen_run_spec(job_dir: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "frozen_run_spec.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    return frozen
