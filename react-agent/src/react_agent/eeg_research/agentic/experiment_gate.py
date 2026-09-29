"""One approved experiment spec. A draft is not permission to implement."""

from __future__ import annotations

from typing import Any


def experiment_block_reason(spec: dict[str, Any] | None) -> str | None:
    """Why this spec cannot be implemented. None means the same validator approved it."""
    if not isinstance(spec, dict) or not spec:
        return "experiment_missing"
    status = str(spec.get("status") or "")
    if spec.get("requires_framework_extension") or status == "requires_framework_extension":
        return "requires_framework_extension"
    if status == "blocked":
        return "experiment_blocked"
    intervention = spec.get("principal_intervention") or spec.get("intervention")
    hypothesis = spec.get("hypothesis") or spec.get("hypothesis_id")
    if status != "approved":
        return "experiment_not_approved"
    if not intervention:
        return "intervention_missing"
    if not hypothesis:
        return "hypothesis_missing"
    if not spec.get("parent_candidate_id"):
        return "parent_missing"
    if not spec.get("initial_fidelity"):
        return "fidelity_missing"
    return None


def approve_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the same spec with status approved, or mark it blocked."""
    copied = dict(spec)
    if copied.get("requires_framework_extension") or copied.get("status") == "requires_framework_extension":
        copied["status"] = "blocked"
        copied["blocked_reason"] = "requires_framework_extension"
        return copied
    copied["status"] = "approved"
    copied["principal_intervention"] = copied.get("principal_intervention") or copied.get("intervention")
    reason = experiment_block_reason(copied)
    if reason:
        copied["status"] = "draft" if reason != "requires_framework_extension" else "blocked"
        copied["blocked_reason"] = reason
    return copied


def experiment_is_approved(spec: dict[str, Any] | None) -> bool:
    return experiment_block_reason(spec) is None
