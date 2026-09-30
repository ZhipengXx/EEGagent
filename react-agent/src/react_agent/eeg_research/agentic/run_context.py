"""Per-target run bindings. The current campaign experiment is not a global hook spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved

BASELINE_TARGET = "baseline"
BASELINE_HOOK_SPEC: dict[str, Any] = {
    "model": {},
    "objective": {},
    "transform": {},
}


def persist_approved_binding(
    camp: Path,
    target_id: str,
    spec: dict[str, Any],
    *,
    spec_ref: str | None = None,
) -> dict[str, Any]:
    """Write the approved hook binding for one candidate. Baseline is never overwritten by this."""
    target = str(target_id or "")
    if not target or target == BASELINE_TARGET:
        return dict(BASELINE_HOOK_SPEC)
    dest = Path(camp) / "candidates" / target
    dest.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema_version": "eeg_research.candidate_binding.v1",
        "target_id": target,
        "model": spec.get("model") if isinstance(spec.get("model"), dict) else {},
        "objective": spec.get("objective") if isinstance(spec.get("objective"), dict) else {},
        "transform": spec.get("transform") if isinstance(spec.get("transform"), dict) else {},
        "negative_sampling_policy": spec.get("negative_sampling_policy"),
        "spec_hash": spec.get("spec_hash"),
        "spec_ref": spec_ref,
        "approval_record": True,
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
    }


def resolve_run_context(
    camp: Path,
    target_id: str | None,
    state: dict[str, Any] | None = None,
    *,
    fidelity: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return the hook spec for this target. Baseline and historical candidates stay isolated."""
    state = state if isinstance(state, dict) else {}
    target = str(target_id or BASELINE_TARGET)
    if target == BASELINE_TARGET:
        return {
            "target_id": target,
            "hook_spec": dict(BASELINE_HOOK_SPEC),
            "spec_ref": None,
            "fidelity": fidelity,
            "seed": seed,
            "approved_binding": {"target_id": target, **BASELINE_HOOK_SPEC},
        }
    binding = load_approved_binding(camp, target)
    if binding is not None:
        return {
            "target_id": target,
            "hook_spec": _hook_spec_from(binding),
            "spec_ref": binding.get("spec_ref"),
            "fidelity": fidelity,
            "seed": seed,
            "approved_binding": binding,
        }
    experiment = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
    if target == str(state.get("candidate_id") or "") and experiment_is_approved(experiment):
        return {
            "target_id": target,
            "hook_spec": _hook_spec_from(experiment),
            "spec_ref": state.get("experiment_ref"),
            "fidelity": fidelity,
            "seed": seed,
            "approved_binding": experiment,
        }
    return {
        "target_id": target,
        "hook_spec": dict(BASELINE_HOOK_SPEC),
        "spec_ref": None,
        "fidelity": fidelity,
        "seed": seed,
        "approved_binding": None,
    }
