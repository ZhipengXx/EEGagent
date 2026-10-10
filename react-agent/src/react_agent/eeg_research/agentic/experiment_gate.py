"""Deterministic experiment approval. A draft or LLM status is not permission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

VALIDATOR_VERSION = "eeg_research.experiment_gate.v5"
APPROVAL_SCHEMA = "eeg_research.approval_record.v1"
SPEC_SCHEMA = "eeg_research.experiment_spec.v1"
POLICY_VERSION = "eeg_research.confirmation_policy.v1"
FIDELITIES = {"pilot", "full"}
KNOWN_HOOKS = {
    "build_encoder",
    "fit_statistics",
    "build_training_transform",
    "build_training_objective",
    "evaluate_only",
}
KNOWN_POLICIES = {"data_parallel_local", "global_batch"}
_ROLE_BLOCKED = {"partial", "failed", "requires_framework_extension", "blocked"}
_LEGACY_ALIASES = {
    "model_config": "model",
    "objective_config": "objective",
    "transform_config": "transform",
}
_HASH_EXCLUDED = {
    "status",
    "blocked_reason",
    "spec_hash",
    "validator_version",
    "allowed_actions",
    "approval_record",
    "experiment_ref",
    "role_status",
    "schema_version",
}


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
    schema_version: str
    status: str
    spec_hash: str
    context_hash: str
    validator_version: str
    producer_task: str
    producer_attempt: str
    artifact_ref: str
    completion_status: str
    required_capabilities: list[str]
    parent_candidate_id: str
    control_candidate_id: str
    policy_version: str
    allowed_actions: list[str]


class ExperimentResolutionError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CanonicalizationError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _stable_dump(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_dump(value)).hexdigest()


def context_hash(context: dict[str, Any] | None) -> str:
    body = context if isinstance(context, dict) else {}
    identity = {
            "capabilities": body.get("capabilities"),
            "evaluation_contract": body.get("evaluation_contract"),
            "parent_candidate_id": body.get("parent_candidate_id"),
            "control_candidate_id": body.get("control_candidate_id"),
            "budget": body.get("budget"),
            "policy_version": body.get("policy_version") or POLICY_VERSION,
        }
    if "evaluation_mode" in body:
        identity["evaluation_mode"] = body["evaluation_mode"]
    return _digest(identity)


def canonicalize_experiment_spec(spec: dict[str, Any] | None, *, allow_legacy: bool = True) -> dict[str, Any]:
    """Migrate legacy aliases only at this explicit entry. Conflicting values are rejected."""
    if not isinstance(spec, dict):
        raise CanonicalizationError("experiment_missing")
    copied = dict(spec)
    for old, new in _LEGACY_ALIASES.items():
        if old not in copied:
            continue
        if not allow_legacy:
            raise CanonicalizationError("legacy_alias_not_at_entry")
        old_val = copied.pop(old)
        new_val = copied.get(new)
        if old_val in (None, {}):
            continue
        if new_val not in (None, {}, []) and new_val != old_val:
            raise CanonicalizationError("alias_conflict")
        if new_val in (None, {}, []):
            copied[new] = old_val
    for key in ("model", "objective", "transform"):
        value = copied.get(key)
        if value is None:
            copied[key] = {}
        elif not isinstance(value, dict):
            raise CanonicalizationError("config_not_object")
    if "implementation_requirements" in copied:
        from pydantic import ValidationError
        from react_agent.eeg_research.agentic.schemas import ImplementationRequirement
        value = copied["implementation_requirements"]
        if not isinstance(value, list):
            raise CanonicalizationError("implementation_requirements_not_list")
        try:
            copied["implementation_requirements"] = [ImplementationRequirement.model_validate(item).model_dump() for item in value]
        except ValidationError as exc:
            raise CanonicalizationError("implementation_requirement_invalid:" + str(exc.errors()[0]["loc"])) from exc
        ids = [item["requirement_id"] for item in copied["implementation_requirements"]]
        if len(ids) != len(set(ids)):
            raise CanonicalizationError("duplicate_implementation_requirement_id")
    copied["schema_version"] = copied.get("schema_version") or SPEC_SCHEMA
    return copied


def _spec_hash(spec: dict[str, Any]) -> str:
    body = {key: spec[key] for key in sorted(spec) if key not in _HASH_EXCLUDED}
    return _digest(body)


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
    if "implementation_requirements" in spec:
        body["implementation_requirements"] = spec["implementation_requirements"]
    return _digest(body)


def scientific_identity(spec: dict[str, Any] | None) -> dict[str, Any]:
    spec = spec if isinstance(spec, dict) else {}
    identity = {
        "intervention": spec.get("principal_intervention") or spec.get("intervention"),
        "hypothesis": spec.get("hypothesis"),
        "parent_candidate_id": spec.get("parent_candidate_id"),
        "control_candidate_id": spec.get("control_candidate_id") or spec.get("control_id") or "baseline",
        "model": spec.get("model") if isinstance(spec.get("model"), dict) else {},
        "objective": spec.get("objective") if isinstance(spec.get("objective"), dict) else {},
        "transform": spec.get("transform") if isinstance(spec.get("transform"), dict) else {},
        "negative_sampling_policy": spec.get("negative_sampling_policy"),
        # Runtime bindings record an absent optional requirement list as [].
        # That representation must not turn a legal same-design repair into a
        # scientific intervention change; declared non-empty lists stay exact.
        "implementation_requirements": spec.get("implementation_requirements", []),
    }
    return identity


def same_scientific_intervention(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return scientific_identity(left) == scientific_identity(right)


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


def _approval_record(
    *,
    status: str,
    spec_hash: str,
    context: dict[str, Any] | None,
    producer: dict[str, Any] | None,
    spec: dict[str, Any],
    allowed_actions: list[str],
    artifact_ref: str | None = None,
) -> dict[str, Any]:
    producer = producer if isinstance(producer, dict) else {}
    return {
        "schema_version": APPROVAL_SCHEMA,
        "status": status,
        "spec_hash": spec_hash,
        "context_hash": context_hash(context),
        "validator_version": VALIDATOR_VERSION,
        "producer_task": producer.get("task_id") or producer.get("producer_task"),
        "producer_attempt": producer.get("attempt_id") or producer.get("producer_attempt"),
        "artifact_ref": artifact_ref,
        "completion_status": "verified" if status == "approved" else status,
        "required_capabilities": [str(item) for item in spec.get("required_capability_ids") or []],
        "parent_candidate_id": spec.get("parent_candidate_id") or "baseline",
        "control_candidate_id": spec.get("control_candidate_id") or spec.get("control_id") or "baseline",
        "policy_version": (context or {}).get("policy_version") or POLICY_VERSION,
        "allowed_actions": allowed_actions,
    }


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


def _stamp(copied: dict[str, Any], *, status: str, context: dict[str, Any] | None, producer: dict[str, Any] | None, allowed: list[str]) -> dict[str, Any]:
    copied["validator_version"] = VALIDATOR_VERSION
    copied["spec_hash"] = _spec_hash(copied)
    copied["approval_record"] = _approval_record(
        status=status if status != "approved" else "approved",
        spec_hash=copied["spec_hash"],
        context=context,
        producer=producer,
        spec=copied,
        allowed_actions=allowed,
    )
    if status != "approved":
        copied["approval_record"]["status"] = status
        copied["approval_record"]["completion_status"] = status
        copied["approval_record"]["allowed_actions"] = []
    return copied


def executable_draft_problem(spec: dict[str, Any]) -> str | None:
    """New approvals require hook configs and explicit capabilities."""
    descriptive = {"hooks", "note", "note_zh", "description", "reuse", "change_summary"}
    for section in ("model", "objective", "transform"):
        config = spec.get(section) or {}
        if descriptive.intersection(config):
            return "config_metadata_not_executable:" + section
    required = spec.get("required_capability_ids")
    if not isinstance(required, list) or not required:
        return "required_capability_ids_missing"
    return None


def approve_experiment(
    spec: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    camp: Path | None = None,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a runtime ApprovalRecord bound to the exact canonical spec. LLM status is not copied through."""
    try:
        copied = canonicalize_experiment_spec(spec)
    except CanonicalizationError as exc:
        failed = dict(spec) if isinstance(spec, dict) else {}
        failed["status"] = "blocked"
        failed["blocked_reason"] = exc.reason
        failed["allowed_actions"] = []
        return _stamp(failed, status="blocked", context=context, producer=producer, allowed=[])
    copied.pop("approval_record", None)
    copied.pop("experiment_ref", None)
    draft_problem = executable_draft_problem(copied)
    evaluation = (context or {}).get("evaluation_contract") or {}
    mode = (context or {}).get("evaluation_mode") or evaluation.get("evaluation_mode")
    if mode == "loso_method_search" and not any(
        item.get("kind", "implementation") == "implementation" and item.get("core", True)
        for item in copied.get("implementation_requirements") or []
    ):
        draft_problem = "core_implementation_requirements_missing"
    if draft_problem:
        copied["status"] = "blocked"
        copied["blocked_reason"] = draft_problem
        copied["allowed_actions"] = []
        return _stamp(copied, status="blocked", context=context, producer=producer, allowed=[])
    if copied.get("role_status") in _ROLE_BLOCKED:
        copied["status"] = "draft"
        copied["blocked_reason"] = "experiment_not_approved"
        copied["allowed_actions"] = []
        stamped = _stamp(copied, status="draft", context=context, producer=producer, allowed=[])
        return stamped
    if copied.get("requires_framework_extension") or copied.get("status") == "requires_framework_extension":
        copied["status"] = "blocked"
        copied["blocked_reason"] = "requires_framework_extension"
        copied["allowed_actions"] = []
        return _stamp(copied, status="blocked", context=context, producer=producer, allowed=[])
    if copied.get("status") == "blocked":
        copied["blocked_reason"] = copied.get("blocked_reason") or "experiment_blocked"
        copied["allowed_actions"] = []
        return _stamp(copied, status="blocked", context=context, producer=producer, allowed=[])
    copied["principal_intervention"] = copied.get("principal_intervention") or copied.get("intervention")
    copied["parent_candidate_id"] = copied.get("parent_candidate_id") or "baseline"
    copied["control_candidate_id"] = copied.get("control_candidate_id") or copied.get("control_id") or "baseline"
    copied["status"] = "approved"
    reason = experiment_block_reason(copied, context=context)
    if reason:
        copied["status"] = "blocked" if reason in {
            "requires_framework_extension",
            "unknown_capability",
            "non_baseline_control_unsupported",
            "fidelity_invalid",
        } else "draft"
        copied["blocked_reason"] = reason
        copied["allowed_actions"] = []
        stamped = _stamp(copied, status=copied["status"], context=context, producer=producer, allowed=[])
        return stamped
    copied.pop("blocked_reason", None)
    copied["allowed_actions"] = ["implement_candidate", "repair_candidate", "run_pilot", "run_full", "replicate"]
    stamped = _stamp(copied, status="approved", context=context, producer=producer, allowed=copied["allowed_actions"])
    if camp is not None and experiment_is_approved(stamped, context=context):
        ref = register_approved_spec(camp, stamped, producer=producer)
        stamped["experiment_ref"] = ref
        record = stamped["approval_record"]
        if isinstance(record, dict):
            record["artifact_ref"] = ref
    return stamped


def experiment_is_approved(spec: dict[str, Any] | None, *, context: dict[str, Any] | None = None) -> bool:
    if experiment_block_reason(spec, context=context) is not None:
        return False
    if not isinstance(spec, dict):
        return False
    if spec.get("role_status") in _ROLE_BLOCKED:
        return False
    if not spec.get("spec_hash"):
        return False
    record = spec.get("approval_record")
    if record is True:
        return False
    if not isinstance(record, dict):
        return False
    if record.get("status") != "approved":
        return False
    if record.get("spec_hash") and record.get("spec_hash") != spec.get("spec_hash"):
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
    record = payload.get("approval_record")
    if isinstance(record, dict) and payload.get("spec_hash"):
        return payload
    if record is True and payload.get("spec_hash"):
        return payload
    return None


def register_approved_spec(camp: Path, spec: dict[str, Any], *, producer: dict[str, Any] | None = None) -> str:
    """Write the approved spec to the artifact registry. Cache is not the source of truth."""
    from react_agent.eeg_research.agentic.artifacts import register

    camp = Path(camp)
    digest = str(spec.get("spec_hash") or _spec_hash(spec))
    path = camp / "experiments" / f"approved_{digest[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": APPROVAL_SCHEMA,
        "experiment_spec": spec,
        "approval_record": spec.get("approval_record"),
        "spec_hash": digest,
        "validator_version": VALIDATOR_VERSION,
    }
    if not path.is_file():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    producer = producer if isinstance(producer, dict) else {}
    row = register(camp, path, kind="approved_experiment", producer_task_id=producer.get("task_id"))
    return str(row["artifact_id"])


def install_approved_experiment(
    camp: Path,
    state: dict[str, Any],
    spec: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    """Fixture/runtime helper: validate, register, and optionally bind one target."""
    approved = approve_experiment(spec, context=context, camp=camp)
    state["experiment"] = approved
    state["experiment_ref"] = approved.get("experiment_ref")
    if target_id and target_id != "baseline" and experiment_is_approved(approved, context=context):
        from react_agent.eeg_research.agentic.run_context import persist_approved_binding

        persist_approved_binding(camp, target_id, approved, spec_ref=approved.get("experiment_ref"))
    return approved


def resolve_approved_experiment(
    camp: Path,
    spec_ref: str | None = None,
    expected_hash: str | None = None,
    target_id: str | None = None,
    attempt_id: str | None = None,
    state: dict[str, Any] | None = None,
    *,
    action: str | None = None,
) -> dict[str, Any]:
    """Target-first resolver. Campaign state is a cache, not execution authorization."""
    from react_agent.eeg_research.agentic.run_context import load_approved_binding

    state = state if isinstance(state, dict) else {}
    cached = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
    target = str(target_id or "")
    binding = load_approved_binding(camp, target) if target and target != "baseline" else None
    needs_binding = action in {"train", "replicate", "resume", "repair"} or (
        action is None and bool(target) and target != "baseline" and action != "implement"
    )
    if action == "implement":
        needs_binding = False
    if needs_binding and binding is None:
        raise ExperimentResolutionError("missing_binding")
    if binding is not None:
        spec_ref = spec_ref or binding.get("spec_ref")
        expected_hash = binding.get("spec_hash") or expected_hash
        if attempt_id and binding.get("attempt_id") and str(binding["attempt_id"]) != str(attempt_id):
            raise ExperimentResolutionError("attempt_mismatch")
        if action == "repair" and cached is not None and not same_scientific_intervention(binding, cached):
            raise ExperimentResolutionError("repair_changes_scientific_design")
        if action == "repair" and cached is not None and cached.get("status") in {"draft", "blocked", "failed", "partial"}:
            raise ExperimentResolutionError("repair_draft_rejected")
    if not spec_ref:
        if cached is not None and not experiment_is_approved(cached):
            raise ExperimentResolutionError("experiment_not_approved")
        raise ExperimentResolutionError("approved_spec_missing")
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
    if binding is not None and cached is not None and target and str(state.get("candidate_id") or "") not in {"", target}:
        if _execution_hash(cached) != _execution_hash(spec) and action in {"train", "replicate", "resume"}:
            spec = spec
    elif cached is not None and binding is None and _execution_hash(cached) != _execution_hash(spec):
        raise ExperimentResolutionError("approved_spec_changed")
    if not experiment_is_approved(spec):
        raise ExperimentResolutionError("experiment_not_approved")
    if expected_hash and spec.get("spec_hash") != expected_hash and _spec_hash(spec) != expected_hash:
        raise ExperimentResolutionError("approved_spec_changed")
    if binding is not None and binding.get("spec_hash") and spec.get("spec_hash") != binding.get("spec_hash"):
        raise ExperimentResolutionError("binding_changed")
    if binding is not None and cached is not None and action == "repair":
        if not same_scientific_intervention(spec, binding):
            raise ExperimentResolutionError("repair_changes_scientific_design")
    return spec
