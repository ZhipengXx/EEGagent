"""Deterministic experiment approval. A draft or LLM status is not permission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

VALIDATOR_VERSION = "eeg_research.experiment_gate.v3"
FIDELITIES = {"pilot", "full"}
KNOWN_HOOKS = {
    "build_encoder",
    "fit_statistics",
    "build_training_transform",
    "build_training_objective",
    "evaluate_only",
}
KNOWN_POLICIES = {"data_parallel_local", "global_batch"}
_ROLE_BLOCKED = {"partial", "failed", "requires_framework_extension"}


class ExperimentDraft(TypedDict, total=False):
    intervention: str
    principal_intervention: str
    hypothesis: Any
    parent_candidate_id: str
    control_candidate_id: str
    initial_fidelity: str
    required_capability_ids: list[str]
    model: dict[str, Any]
    objective: dict[str, Any]
    transform: dict[str, Any]


class ApprovalRecord(TypedDict, total=False):
    approval_record: bool
    spec_hash: str
    validator_version: str
    allowed_actions: list[str]


class ExperimentResolutionError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _spec_hash(spec: dict[str, Any]) -> str:
    body = {
        key: spec[key]
        for key in sorted(spec)
        if key not in {"status", "blocked_reason", "spec_hash", "validator_version", "allowed_actions", "approval_record"}
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execution_hash(spec: dict[str, Any]) -> str:
    body = {
        key: spec.get(key)
        for key in (
            "model",
            "objective",
            "transform",
            "intervention",
            "principal_intervention",
            "negative_sampling_policy",
            "hypothesis",
        )
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _available_capability_ids(context: dict[str, Any] | None) -> set[str]:
    """Only hooks marked available in this context. Disabled names stay unavailable."""
    from react_agent.eeg_research.agentic.capabilities import capability_manifest

    if context is not None and "capabilities" in context:
        manifest = context.get("capabilities") or {}
    else:
        manifest = capability_manifest()
    hooks = manifest.get("hooks") if isinstance(manifest, dict) else {}
    available: set[str] = set()
    for name, row in (hooks or {}).items():
        if not isinstance(row, dict):
            continue
        if row.get("status") == "disabled" or row.get("disabled") is True:
            continue
        if row.get("status") == "available":
            available.add(str(name))
    return available


def experiment_block_reason(spec: dict[str, Any] | None, *, context: dict[str, Any] | None = None) -> str | None:
    """Why this spec cannot be implemented. None means the same validator approved it."""
    if not isinstance(spec, dict) or not spec:
        return "experiment_missing"
    status = str(spec.get("status") or "")
    if spec.get("role_status") in _ROLE_BLOCKED:
        return "experiment_not_approved"
    if spec.get("requires_framework_extension") or status == "requires_framework_extension":
        return "requires_framework_extension"
    if status == "blocked":
        return "experiment_blocked"
    intervention = spec.get("principal_intervention") or spec.get("intervention")
    hypothesis = spec.get("hypothesis") or spec.get("hypothesis_id") or spec.get("engineering_fix_rationale")
    if not intervention:
        return "intervention_missing"
    if not hypothesis:
        return "hypothesis_missing"
    if not spec.get("parent_candidate_id"):
        return "parent_missing"
    fidelity = spec.get("initial_fidelity")
    if fidelity not in FIDELITIES:
        return "fidelity_invalid"
    control = spec.get("control_candidate_id") or spec.get("control_id") or "baseline"
    if control not in {None, "", "baseline"}:
        return "non_baseline_control_unsupported"
    required = [str(item) for item in spec.get("required_capability_ids") or []]
    available = _available_capability_ids(context)
    unknown = [name for name in required if name not in available]
    if unknown:
        return "unknown_capability"
    policy = spec.get("negative_sampling_policy")
    if policy not in (None, "") and policy not in KNOWN_POLICIES:
        return "unknown_negative_policy"
    if status not in {"", "approved", "draft"}:
        return "experiment_not_approved"
    if status != "approved":
        return "experiment_not_approved"
    return None


def approve_experiment(spec: dict[str, Any], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an approval record bound to the exact spec. LLM status is not copied through."""
    copied = dict(spec)
    copied.pop("approval_record", None)
    if copied.get("role_status") in _ROLE_BLOCKED:
        copied["status"] = "draft"
        copied["blocked_reason"] = "experiment_not_approved"
        copied["validator_version"] = VALIDATOR_VERSION
        copied["spec_hash"] = _spec_hash(copied)
        copied["allowed_actions"] = []
        return copied
    if copied.get("requires_framework_extension") or copied.get("status") == "requires_framework_extension":
        copied["status"] = "blocked"
        copied["blocked_reason"] = "requires_framework_extension"
        copied["validator_version"] = VALIDATOR_VERSION
        copied["spec_hash"] = _spec_hash(copied)
        copied["allowed_actions"] = []
        return copied
    if copied.get("status") == "blocked":
        copied["blocked_reason"] = copied.get("blocked_reason") or "experiment_blocked"
        copied["validator_version"] = VALIDATOR_VERSION
        copied["spec_hash"] = _spec_hash(copied)
        copied["allowed_actions"] = []
        return copied
    copied["principal_intervention"] = copied.get("principal_intervention") or copied.get("intervention")
    copied["parent_candidate_id"] = copied.get("parent_candidate_id") or "baseline"
    copied["control_candidate_id"] = copied.get("control_candidate_id") or copied.get("control_id") or "baseline"
    copied["status"] = "approved"
    reason = experiment_block_reason(copied, context=context)
    if reason:
        copied["status"] = "blocked" if reason in {"requires_framework_extension", "unknown_capability", "non_baseline_control_unsupported", "fidelity_invalid"} else "draft"
        copied["blocked_reason"] = reason
        copied["allowed_actions"] = []
    else:
        copied.pop("blocked_reason", None)
        copied["allowed_actions"] = ["implement_candidate", "repair_candidate", "run_pilot", "run_full", "replicate"]
        copied["approval_record"] = True
    copied["validator_version"] = VALIDATOR_VERSION
    copied["spec_hash"] = _spec_hash(copied)
    return copied


def experiment_is_approved(spec: dict[str, Any] | None, *, context: dict[str, Any] | None = None) -> bool:
    if experiment_block_reason(spec, context=context) is not None:
        return False
    if not isinstance(spec, dict):
        return False
    if spec.get("approval_record") is not True:
        return False
    if not spec.get("spec_hash"):
        return False
    if spec.get("role_status") in _ROLE_BLOCKED:
        return False
    return True


def resolve_approved_spec(camp_spec: dict[str, Any] | None, *, stored_hash: str | None = None) -> dict[str, Any] | None:
    """Refuse a cached spec whose content no longer matches the approval hash."""
    if not experiment_is_approved(camp_spec):
        return None
    if stored_hash and camp_spec and stored_hash != camp_spec.get("spec_hash") and stored_hash != _spec_hash(camp_spec):
        return None
    return camp_spec


def _spec_from_artifact(payload: dict[str, Any]) -> dict[str, Any] | None:
    spec = payload.get("experiment_spec")
    if isinstance(spec, dict):
        return spec
    inner = payload.get("payload")
    if isinstance(inner, dict) and isinstance(inner.get("experiment_spec"), dict):
        return inner["experiment_spec"]
    if payload.get("approval_record") is True and payload.get("spec_hash"):
        return payload
    return None


def resolve_approved_experiment(
    camp: Path,
    spec_ref: str | None = None,
    expected_hash: str | None = None,
    target_id: str | None = None,
    attempt_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the registered approval artifact. Campaign state is a cache, not the source of truth."""
    del target_id, attempt_id
    state = state if isinstance(state, dict) else {}
    cached = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
    spec: dict[str, Any] | None = None
    if spec_ref:
        from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact

        try:
            row = resolve_verified_artifact(camp, spec_ref)
        except FileNotFoundError as exc:
            raise ExperimentResolutionError(str(exc) or "approved_spec_missing") from exc
        path = Path(row["path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentResolutionError("approved_spec_unreadable") from exc
        if not isinstance(payload, dict):
            raise ExperimentResolutionError("approved_spec_missing")
        spec = _spec_from_artifact(payload)
        if spec is None:
            raise ExperimentResolutionError("approved_spec_missing")
        if cached is not None and _execution_hash(cached) != _execution_hash(spec):
            raise ExperimentResolutionError("approved_spec_changed")
    elif cached is not None:
        spec = cached
    else:
        raise ExperimentResolutionError("approved_spec_missing")
    if not experiment_is_approved(spec):
        raise ExperimentResolutionError("experiment_not_approved")
    if expected_hash and spec.get("spec_hash") != expected_hash and _spec_hash(spec) != expected_hash:
        raise ExperimentResolutionError("approved_spec_changed")
    return spec
