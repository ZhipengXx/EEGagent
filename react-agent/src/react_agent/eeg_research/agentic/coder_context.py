"""Deterministic Coder delivery; native tools and execution state stay native.

Every model call is stateless. Current source, exact revision, relevant caller
source and real tool feedback are delivered again when needed. This module
does no file reads, retrieval, model calls, clipping or scientific rewriting.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from .context.representations import deduplicate, resolve_request_pointer
from .context.sources import assert_visible_payload, pointer_escape

VERSION = "coder_implementation_input.v2"

_INTERFACE_LOCAL = frozenset({"schema_version", "design_contract_version", "candidate_id",
    "attempt_id", "operation_id", "spec_hash", "protocol_fingerprint", "execution_fingerprint",
    "contract_fingerprint"})
_PREFLIGHT_LOCAL = frozenset({"schema_version", "protocol_fingerprint", "execution_fingerprint", "gpu"})
_GOAL_CONSTRAINTS = (
    "task_type", "dataset", "research_scope", "evaluation_mode", "evaluation_schedule",
    "allowed_changes", "frozen_components", "checkpoint_policy", "negative_sampling_policy",
    "training_recipe", "recipe", "training_seeds", "precision", "batch_size", "epochs",
    "optimizer", "learning_rate", "weight_decay",
)
_CHECK_STATUS = ("passed", "fresh", "next_tool", "scope", "ready_for_review")
_FEEDBACK = (
    "history", "last_check", "edit_success_context", "edit_failure_context",
    "error_source_context", "failure_source", "runtime_note", "recovery_context",
)
_FEEDBACK_LOCAL = frozenset({"schema_version", "check_fingerprint"})
_ARTIFACT_LOCAL = frozenset({"schema_version", "receipt_id", "request_digest", "sha256", "view_sha256",
    "content_injected", "delivery_status", "delivery_id", "satisfies_development_read_receipt"})


def is_coder_input(role: str, payload: dict[str, Any]) -> bool:
    return role == "candidate_coder" and payload.get("coder_context_version") == VERSION


def _pick(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def _feedback_view(value: Any, current_sha256: str | None) -> Any:
    """Drop local bookkeeping from tool feedback; native feedback files keep it.

    Field-name inventories and check fingerprints carry no task content, and a
    hash equal to the delivered current_file.sha256 only repeats that CAS value.
    Other hashes stay: without a current_file they are the Coder's only base.
    """
    if isinstance(value, list):
        return [_feedback_view(item, current_sha256) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    if "$context_pointer" in value:
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in _FEEDBACK_LOCAL:
            continue
        if key == "keys" and isinstance(item, list) and all(isinstance(name, str) for name in item):
            continue
        if current_sha256 and key in {"sha256", "source_sha256"} and item == current_sha256:
            continue
        result[key] = _feedback_view(item, current_sha256)
    return result


def _expanded(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve only actual request-local pointers before rebuilding the view."""
    def visit(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return visit(resolve_request_pointer(payload, path), value["$context_pointer"])
            return {key: visit(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return visit(payload)


def _remaining(limit: Any, used: Any) -> int | None:
    if type(limit) is not int or type(used) is not int:
        return None
    return max(0, limit - used)


def _execution_state(view: dict[str, Any]) -> dict[str, Any]:
    result = {}
    # Show the actual currently enforceable counters. Session and candidate
    # lineage limits can differ; neither is inferred from the other's counter.
    for prefix, used_key, limit_key in (
        ("session", "attempt_step", "attempt_max_steps"),
        ("operation", "step", "max_steps"),
        ("repair", "repairs_used", "max_repairs"),
    ):
        used, limit = view.get(used_key), view.get(limit_key)
        if type(used) is int and type(limit) is int:
            # step/attempt_step name the impending tool action; they are not
            # counts of already executed tools. Repair usage is an actual count.
            result[prefix] = {"used" if prefix == "repair" else "current_step": used, "limit": limit}
    recovery = view.get("coding_recovery") or {}
    if isinstance(recovery, dict):
        lineage = {}
        for label, maximum, used in (
            ("tool_steps_remaining", "tool_step_limit", "steps_used"),
            ("failed_checks_remaining", "failed_check_limit", "failed_checks_used"),
        ):
            remaining = _remaining(recovery.get(maximum), recovery.get(used))
            if remaining is not None:
                lineage[label] = remaining
        if lineage:
            result["lineage"] = lineage
    check = _pick(view.get("check_status"), _CHECK_STATUS)
    if check:
        result["check"] = check
    progress = _pick(view.get("editing_progress"), (
        "completed_edits", "last_tool", "last_successful_write", "next_tool"))
    if progress:
        result["progress"] = progress
    if isinstance(view.get("check_policy"), dict):
        result["check_policy"] = copy.deepcopy(view["check_policy"])
    return result


def _sources(view: dict[str, Any]) -> dict[str, Any]:
    references = view.get("references") or {}
    ranges = view.get("reference_ranges") or {}
    result: dict[str, Any] = {}
    if isinstance(references, dict) and references:
        result["references"] = {}
        for path, text in references.items():
            # Wrapping a raw source in its original range makes equal exact
            # bodies eligible for the existing request-local object deduper.
            # No file hash is invented from a potentially bounded excerpt.
            result["references"][path] = ({"text": text, **copy.deepcopy(ranges.get(path) or {})}
                if isinstance(text, str) else copy.deepcopy(text))
    available = view.get("available_references")
    if isinstance(available, list):
        result["available"] = copy.deepcopy(available)
    spec = view.get("experiment_spec") or {}
    artifacts = spec.get("artifact_reads") if isinstance(spec, dict) else None
    if isinstance(artifacts, list) and artifacts:
        # These are already selected native pages, not additional reads. Keep
        # the original text/range/gaps while receipts remain in the local log.
        result["artifact_reads"] = [{key: copy.deepcopy(child) for key, child in row.items()
            if key not in _ARTIFACT_LOCAL} if isinstance(row, dict) else copy.deepcopy(row) for row in artifacts]
    return result


def _repair_context(view: dict[str, Any]) -> dict[str, Any]:
    """Review findings can require a repair even when all code checks passed."""
    result = {}
    if view.get("repair_context"):
        result["native_check_repair"] = copy.deepcopy(view["repair_context"])
    spec = view.get("experiment_spec") or {}
    if not isinstance(spec, dict):
        return result
    previous_review = _pick(spec, ("repair_issues", "repair_intervention_coverage",
        "repair_unverified_invariants", "review_mode", "review_focus"))
    if previous_review:
        # Preserve the original review revision/refs as feedback. A prior review
        # is never relabelled as evidence that the current source passed.
        result["previous_implementation_review"] = previous_review
    if spec.get("audit_issues"):
        result["audit_issues"] = copy.deepcopy(spec["audit_issues"])
    return result


def _interface(value: Any) -> dict[str, Any]:
    """Keep scientific interface extensions; remove only known storage fields."""
    if not isinstance(value, dict):
        return {}
    result = {key: copy.deepcopy(child) for key, child in value.items() if key not in _INTERFACE_LOCAL}
    preflight = result.get("multigpu_check")
    if isinstance(preflight, dict):
        visible = {key: copy.deepcopy(child) for key, child in preflight.items() if key not in _PREFLIGHT_LOCAL}
        # Physical allocation is native, but local-batch objective semantics
        # depend on the replica count and actual tail partition.
        if isinstance(preflight.get("gpu"), list):
            visible.setdefault("device_count", len(preflight["gpu"]))
        result["multigpu_check"] = visible
    return result


def _feedback(value: Any) -> Any:
    """Errors and actual tool observations are data, never read capabilities."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    # The system schema is authoritative. Do not repeat the whole tools catalog
    # on format correction, but retain every reported error and actionable hint.
    return {key: copy.deepcopy(child) for key, child in value.items()
        if key not in {"allowed_tool_arguments", "recovery"}}


def project_coder_input(native_payload: dict[str, Any]) -> dict[str, Any]:
    """A pure wire view; never replace the native request or persisted spec."""
    from .approved_design_context import project_approved_design
    from .coder_contract import coder_contract

    assert_visible_payload(native_payload, "candidate_coder")
    if is_coder_input("candidate_coder", native_payload):
        coder_contract(native_payload)  # Missing frozen declarations is a native error.
        return copy.deepcopy(native_payload)
    # A rejected reply can contain arbitrary pointer-looking keys. Quote it
    # before traversing trusted request-local references, exactly as Designer.
    payload = copy.deepcopy(native_payload)
    for field in ("previous_response", "previous", "previous_domain_response"):
        if isinstance(payload.get(field), (dict, list)):
            payload[field] = json.dumps(payload[field], ensure_ascii=False, sort_keys=True)
    view = _expanded(payload)
    result: dict[str, Any] = {
        "coder_context_version": VERSION,
        "approved_design": project_approved_design(view.get("experiment_spec") or {},
            selected_task=view.get("selected_research_task"), interface=view.get("input_spec"),
            effective_hook_config=view.get("effective_hook_config")),
    }
    constraints = _pick(view.get("goal"), _GOAL_CONSTRAINTS)
    if constraints:
        result["approved_design"]["scientific_constraints"] = constraints
    interface = _interface(view.get("input_spec"))
    if interface:
        result["candidate_interface"] = interface
    if view.get("hook_config_runtime_source") is not None:
        result["runtime_source"] = copy.deepcopy(view["hook_config_runtime_source"])
    sources = _sources(view)
    if sources:
        result["sources"] = sources
    # Keep complete-line source and its real CAS hash together. Null means a
    # missing candidate file and is never silently changed into an empty file.
    result["current_file"] = copy.deepcopy(view.get("current_file"))
    if "allowed_write_paths" in view:
        result["allowed_write_paths"] = copy.deepcopy(view["allowed_write_paths"])
    if view.get("implementation_map") is not None:
        result["implementation_map"] = copy.deepcopy(view["implementation_map"])
    execution = _execution_state(view)
    if execution:
        result["execution_state"] = execution
    current = result["current_file"].get("sha256") if isinstance(result["current_file"], dict) else None
    feedback = {key: _feedback_view(view[key], current) for key in _FEEDBACK
        if key in view and view[key] not in (None, [], {})}
    if feedback:
        result["tool_feedback"] = feedback
    repair = _repair_context(view)
    if repair:
        result["repair_context"] = repair
    spec = view.get("experiment_spec") or {}
    spec = spec if isinstance(spec, dict) else {}
    instruction = view.get("current_user_instruction", spec.get("current_user_instruction"))
    if instruction is not None:
        result["current_user_instruction"] = copy.deepcopy(instruction)
    if (view.get("current_user_instruction") is not None and spec.get("current_user_instruction") is not None
            and view["current_user_instruction"] != spec["current_user_instruction"]):
        result.setdefault("repair_context", {})["spec_user_instruction"] = copy.deepcopy(spec["current_user_instruction"])
    hints = {}
    verified = view.get("verified_engineering_repairs") or {}
    if isinstance(verified, dict) and verified.get("items"):
        hints["verified_repairs"] = _pick(verified, ("instruction", "items", "scope"))
    implementation = view.get("relevant_implementation_hints") or {}
    if isinstance(implementation, dict) and implementation.get("notes"):
        hints["implementation_notes"] = copy.deepcopy(implementation["notes"])
    memory = spec.get("retrieved_memory")
    items = memory.get("items") if isinstance(memory, dict) else memory if isinstance(memory, list) else None
    if isinstance(items, list) and items:
        hints["retrieved_memory"] = {"items": copy.deepcopy(items)}
    if hints:
        result["hints"] = hints
    for field in ("previous_response", "previous", "previous_domain_response"):
        if field in view:
            result["previous_response"] = copy.deepcopy(view[field])
            break
    for field in ("previous_response_error", "schema_repair_context", "error_feedback"):
        if view.get(field) is not None:
            result["previous_response_error"] = _feedback(view[field])
            break
    if "previous_response_error" not in result and view.get("schema_error") is not None:
        result["previous_response_error"] = {"kind": "schema_invalid", "detail": copy.deepcopy(view["schema_error"])}
    elif view.get("schema_error") is not None and isinstance(result.get("previous_response_error"), dict):
        correction = result["previous_response_error"]
        if not {"detail", "errors", "validation_error", "validation_errors"}.intersection(correction):
            correction["detail"] = copy.deepcopy(view["schema_error"])
    if view.get("correction_instruction") is not None:
        correction = result.setdefault("previous_response_error", {})
        if isinstance(correction, dict):
            correction.setdefault("instruction", copy.deepcopy(view["correction_instruction"]))
    coder_contract(result)  # One declaration extractor also governs paid recovery.
    return deduplicate(result)
