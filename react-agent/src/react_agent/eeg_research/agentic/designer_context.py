"""Designer science delivery and its separate native assembly inputs.

No retrieval, summarization, model call or text clipping happens here. Native
tasks retain the full context; this projection is only the provider delivery.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any

from .context.representations import deduplicate, resolve_request_pointer
from .context.sources import assert_visible_payload, pointer_escape

VERSION = "designer_scientific_input.v1"

# One evidence boundary shared by projection consumers and reply validation.
# Examples and previous answers are formatting aids, never new source evidence.
DESIGNER_MATERIAL_FIELDS = ("parent_source", "control_source", "selection_basis",
    "method_evidence", "method_evidence_packet", "retrieved_memory", "retrieved_knowledge",
    "analyses", "diagnostics", "verified_development_facts", "verified_encoder_structural_facts",
    "measured_parameter_counts", "parameter_count_comparisons", "method_suite", "method_screening")

_GOAL = ("objective", "task_type", "dataset", "research_scope", "evaluation_mode",
    "evaluation_schedule", "primary_metric", "direction", "allowed_changes",
    "frozen_components", "training_seeds", "checkpoint_policy", "subject_aggregation",
    "required_fold_count", "min_practical_gain_pp", "selection_min_delta",
    "benchmark_subjects", "benchmark_feedback_enabled", "final_test_enabled")
_INTERFACE = ("input", "output", "dataset", "c_num", "channels", "timesteps", "gallery_size",
    "build_encoder", "build_training_objective", "build_training_transform", "fit_statistics",
    "not_available", "runtime_guarantees")
_TASK = ("task_kind", "question", "task_description", "scientific_intent", "evaluation_scope",
    "expected_observation", "completion_condition", "target_candidate_ref")
_SOURCE = ("candidate_id", "intervention", "hypothesis", "source", "source_ref", "source_range",
    "source_truncated", "missing_inputs", "model", "objective", "transform", "recipe",
    "frozen_training_policy", "negative_sampling_policy", "scope", "visibility")
_LOCAL = {"schema_version", "approval_record", "cpu_check_receipt", "verification_artifact",
    "context_delivery", "context_authority", "working_snapshot", "checks", "check",
    "validation_records", "operation_binding", "next_read", "next_reads", "read_capability",
    "read_unit", "json_pointer", "satisfies_development_read_receipt", "task_id", "attempt_id",
    "origin_plan_id", "plan_id", "campaign_id", "input_digest", "input_hash", "role_models",
    "fingerprint", "digest", "sha256"}
_SCIENTIFIC_BODIES = {"model", "objective", "transform", "config", "scientific_parameters",
    "metrics", "scores", "paired_scores", "native_facts", "parameters", "hyperparameters"}


def is_designer_input(role: str, payload: dict[str, Any]) -> bool:
    return role == "experiment_designer" and payload.get("designer_context_version") == VERSION


def _pick(value: Any, names) -> dict:
    return {key: copy.deepcopy(value[key]) for key in names if key in value} if isinstance(value, dict) else {}


def _science(value: Any) -> Any:
    """Keep scientific bodies and provenance, without storage/read machinery."""
    if isinstance(value, list):
        return [_science(item) for item in value if not (isinstance(item, dict)
            and isinstance(item.get("field"), str)
            and (item["field"].endswith(("_hash", "_digest", "_fingerprint"))
                 or item["field"] == "method_revision"))]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {key: copy.deepcopy(child) if key in _SCIENTIFIC_BODIES else _science(child)
        for key, child in value.items()
        if key not in _LOCAL and not key.startswith("_")
        and not key.endswith(("_hash", "_sha256", "_digest", "_fingerprint"))}


def _expanded(view: dict) -> dict:
    # Old frozen requests and idempotent calls may contain exact local pointers.
    # Expand before selecting fields, then rebuild pointers in the final view.
    def visit(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return visit(resolve_request_pointer(view, path), value["$context_pointer"])
            return {key: visit(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return visit(view)


def _source(value: Any) -> dict:
    result = _science(_pick(value, _SOURCE))
    if isinstance(value, dict) and isinstance(value.get("encoder_reference"), dict):
        result["encoder_reference"] = _pick(value["encoder_reference"],
            ("excerpt", "source_ref", "source_range", "scope", "visibility"))
    if isinstance(result.get("frozen_training_policy"), dict):
        result["frozen_training_policy"].pop("interpretation", None)
    return result


def _memory(value: Any) -> dict:
    if not isinstance(value, dict):
        return {"items": _science(value)} if value else {}
    items = value.get("items") or []
    # Absence is meaningful, but embedding models, timings and query hashes are
    # not design evidence. Preserve non-success state rather than inventing a hit.
    result = {"items": _science(items)}
    status = value.get("status")
    if status is not None:
        result["status"] = status
    elif not items:
        if value.get("backend") == "disabled" or value.get("reason") == "disabled":
            result["status"] = "disabled"
        else:
            unavailable = value.get("degraded") or value.get("query_status") in {"failed", "unavailable"}
            result["status"] = "retrieval_unavailable" if unavailable else "no_related_memory"
    return result


def _feedback(value: Any) -> Any:
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = _pick(value, ("kind", "instruction", "finish_reason"))
    rows = value.get("errors", value.get("validation_errors"))
    if isinstance(rows, list):
        result["errors"] = [_pick(row, ("path", "loc", "type", "message", "msg")) for row in rows]
    return result


def _resource_limits(budget: Any) -> dict:
    """Effective method resources, without replaying the owner/API ledger."""
    if not isinstance(budget, dict):
        return {}
    owner = budget.get("owner_budget")
    available = owner.get("owner_available") if isinstance(owner, dict) else {}
    available = available if isinstance(available, dict) else {}
    result = {}
    for field, dimension in (("gpu_seconds_left", "gpu_seconds"), ("training_jobs_left", "training_jobs")):
        known = []
        unknown = False
        for source, key in ((budget, field), (available, dimension)):
            if key not in source:
                continue
            value = source[key]
            if value is None:
                unknown = True
            elif type(value) in (int, float) and math.isfinite(value):
                known.append(value)
        if known:
            result[field] = min(known)
        elif unknown:
            result[field] = None
    return result


def project_designer_input(full_payload: dict[str, Any]) -> dict[str, Any]:
    """Pure, idempotent provider view; never substitute this for native task data."""
    assert_visible_payload(full_payload, "experiment_designer")
    # A rejected answer is quoted text, not trusted transport control. Preserve
    # it exactly as JSON so an invented $context_pointer cannot become a read or
    # invalidate the final provider request's real local pointer closure.
    if isinstance(full_payload.get("previous_response"), (dict, list)):
        full_payload = {**full_payload, "previous_response": json.dumps(
            full_payload["previous_response"], ensure_ascii=False, sort_keys=True)}
    if is_designer_input("experiment_designer", full_payload):
        return copy.deepcopy(full_payload)
    view = _expanded(full_payload)
    result = {"designer_context_version": VERSION}
    goal = _pick(view.get("goal"), _GOAL)
    for section, fields in {
        "screening": ("fold_count", "subjects", "selection", "selection_seed", "training_recipe", "min_mean_gain_pp"),
        "confirmation": ("fold_count", "freeze_method_before_screening", "subject_aggregation"),
    }.items():
        detail = _science(_pick((view.get("goal") or {}).get(section), fields))
        if detail:
            goal[section] = detail
    if goal:
        result["goal"] = goal
    task = _pick(view.get("selected_research_task"), _TASK)
    if task:
        if "task_description" in task:
            task["method"] = task.pop("task_description")
        if "scientific_intent" in task:
            task["rationale"] = task.pop("scientific_intent")
    if task:
        result["selected_research_task"] = _science(task)
    draft = view.get("draft") or {}
    for key in ("initial_fidelity", "negative_sampling_policy"):
        value = view.get(key, draft.get(key))
        if value is not None:
            result[key] = copy.deepcopy(value)
    if not task:
        draft_science = _pick(draft, ("hypothesis", "intervention", "principal_intervention"))
        if draft_science:
            result["draft"] = draft_science
    for key in ("selection_basis", "confirmation_policy", "evaluation_contract", "evaluation_population",
                "contract", "diagnostics", "method_suite", "method_screening", "analyses",
                "verified_development_facts", "verified_encoder_structural_facts", "measured_parameter_counts",
                "parameter_count_comparisons", "method_evidence_packet", "design_revision", "prior_design_brief"):
        if view.get(key) is not None and view[key] != [] and view[key] != {}:
            body = _science(view[key])
            if key in {"measured_parameter_counts", "parameter_count_comparisons"} and isinstance(body, dict):
                body.pop("scope", None)
            if key == "prior_design_brief" and isinstance(body, dict):
                # The task fixes the expected observation; a revision must not echo it.
                body.pop("expected_observation", None)
            if body:
                result[key] = body
    for key in ("evaluation_mode", "scientific_unit", "science_scope", "scope", "cold_start_prior"):
        if key in view and view[key] is not None and (key not in goal or goal[key] != view[key]):
            result[key] = _science(view[key])
    capabilities = view.get("capabilities") or {}
    interface = _pick(view.get("candidate_interface") or capabilities.get("input_spec"), _INTERFACE)
    if interface:
        result["candidate_interface"] = interface
    capability_view = {key: {name: _pick(item, ("status", "default", "when", "description"))
        for name, item in capabilities[key].items()} for key in
        ("hooks", "task_adapters", "negative_sampling_policies") if isinstance(capabilities.get(key), dict)}
    if capability_view:
        result["capabilities"] = capability_view
    for key in ("parent_source", "control_source"):
        if view.get(key):
            result[key] = _source(view[key])
    config = view.get("parent_hook_config")
    if isinstance(config, dict):
        result["parent_hook_config"] = _pick(config, ("model", "objective", "transform", "negative_sampling_policy"))
    # Provenance for method cards says whether the supplied material is a local
    # card, excerpt or actual source. Do not upgrade a path into a full read.
    if view.get("method_evidence"):
        result["method_evidence"] = [_science(_pick(item, ("method_id", "title", "mechanism", "when",
            "assumptions", "failure_modes", "required_inputs", "compatible_capabilities", "evidence_limit",
            "source_refs", "task_fit", "intervention"))) for item in view["method_evidence"]]
        if view.get("method_evidence_origin"):
            result["method_evidence_origin"] = view["method_evidence_origin"]
    for key in ("retrieved_memory", "retrieved_knowledge"):
        if key in view:
            result[key] = _memory(view[key])
    limits = _resource_limits(view.get("budget"))
    if limits:
        result["resource_limits"] = limits
    unavailable = view.get("unavailable_context", view.get("missing_inputs"))
    if unavailable:
        result["unavailable_context"] = copy.deepcopy(unavailable)
    instruction = view.get("current_user_instruction")
    if instruction:
        result["current_user_instruction"] = (_pick(instruction, ("request_zh", "requested_task_kind"))
            if isinstance(instruction, dict) else copy.deepcopy(instruction))
    if view.get("relevant_implementation_hints"):
        hints = view["relevant_implementation_hints"]
        result["relevant_implementation_hints"] = {"notes": [_pick(note,
            ("case_id", "applicability", "instruction")) for note in hints.get("notes", [])]}
    for key in ("illustrative_examples", "current_examples"):
        if view.get(key):
            result[key] = copy.deepcopy(view[key])
            break  # The role-example builder supplies one authoritative example.
    if view.get("previous_response_error") is not None:
        result["previous_response_error"] = _feedback(view["previous_response_error"])
    for key in ("previous", "previous_reply", "previous_response"):
        if view.get(key) is not None:
            result[key] = _science(view[key])
    return deduplicate(result)


def designer_assembly_context(full_payload: dict[str, Any]) -> dict[str, Any]:
    """Frozen native-only adapter inputs; persist outside the provider request."""
    # Assembly, capability checks and recovery share the original context. This
    # copy is a local sidecar, never an additional provider section. In particular
    # preserve exact protocol/config dictionaries rather than guessing which
    # future adapter inputs will be needed from a projected scientific view.
    return copy.deepcopy(full_payload)


def designer_allowed_refs(delivered_payload: dict[str, Any]) -> frozenset[str]:
    """Only real delivered materials, excluding example/correction citations.

    Path-only method cards permit attribution to that card's source; this is not
    a claim that its file contents were read. Scope stays with the supplied card.
    """
    result: set[str] = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"source_ref", "$context_source", "path", "url", "observation_id", "record_id", "method_id"} and isinstance(child, str):
                    result.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    from .review_context import _request_value
    for key in DESIGNER_MATERIAL_FIELDS:
        # Resolve only native material sections. A malformed pointer invented
        # inside an old answer/example must not participate in evidence reads.
        visit(_request_value(delivered_payload, key))
    return frozenset(result)
