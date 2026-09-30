"""Deterministic experiment approval. A draft or LLM status is not permission."""

from __future__ import annotations

import hashlib
import json
from typing import Any

VALIDATOR_VERSION = "eeg_research.experiment_gate.v2"
FIDELITIES = {"pilot", "full"}
KNOWN_HOOKS = {
    "build_encoder",
    "fit_statistics",
    "build_training_transform",
    "build_training_objective",
    "evaluate_only",
}
KNOWN_POLICIES = {"data_parallel_local", "global_batch"}


def _spec_hash(spec: dict[str, Any]) -> str:
    body = {key: spec[key] for key in sorted(spec) if key not in {"status", "blocked_reason", "spec_hash", "validator_version", "allowed_actions", "approval_record"}}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_block_reason(spec: dict[str, Any] | None, *, context: dict[str, Any] | None = None) -> str | None:
    """Why this spec cannot be implemented. None means the same validator approved it."""
    if not isinstance(spec, dict) or not spec:
        return "experiment_missing"
    status = str(spec.get("status") or "")
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
    known = set(KNOWN_HOOKS)
    manifest = None if context is None else context.get("capabilities")
    if isinstance(manifest, dict):
        hooks = manifest.get("hooks") if isinstance(manifest.get("hooks"), dict) else {}
        known.update(str(name) for name, row in hooks.items() if isinstance(row, dict) and row.get("status") == "available")
    unknown = [name for name in required if name not in known]
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
    return experiment_block_reason(spec, context=context) is None


def resolve_approved_spec(camp_spec: dict[str, Any] | None, *, stored_hash: str | None = None) -> dict[str, Any] | None:
    """Refuse a cached spec whose content no longer matches the approval hash."""
    if not experiment_is_approved(camp_spec):
        return None
    if stored_hash and camp_spec and stored_hash != camp_spec.get("spec_hash") and stored_hash != _spec_hash(camp_spec):
        return None
    return camp_spec
