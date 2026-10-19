"""Strict Designer answer -> locally completed native design.

The same pure check serves live answers and paid-answer recovery. It performs
no file reads, retrieval, GPU work or model calls and never mutates raw replies.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from pydantic import ValidationError

from .design_contract import (
    DesignContractError, NATIVE_MEASUREMENT_PREFIX, SECTIONS, materialize_design_brief,
    native_measurement_requirements,
)
from .schemas import DesignerBrief, DesignerReply


def _context_value(context: dict, name: str, default=None):
    # Compiled requests may contain request-local pointers. Resolve without I/O.
    from .review_context import _request_value
    value = _request_value(context, name)
    return default if value is None else value


def _aliases(reference: str) -> set[str]:
    aliases = {reference}
    marker = "/src/react_agent/"
    if marker in reference:
        aliases.add(reference.split(marker, 1)[1])
    return aliases


def _delivered_references(context: dict) -> tuple[set[str], dict[str, dict]]:
    """Collect explicit identities and supplied source text, not arbitrary prose."""
    from .designer_context import DESIGNER_MATERIAL_FIELDS, designer_allowed_refs
    refs = {alias for reference in designer_allowed_refs(context) for alias in _aliases(reference)}
    sources: dict[str, dict] = {}
    identity_keys = {"path", "url", "uri", "artifact_id", "evidence_id", "observation_id", "record_id",
                     "method_id", "lesson_id", "knowledge_id", "source_ref", "$context_source", "approval_ref",
                     "analysis_artifact_ref", "evidence_refs", "source_refs", "supporting_episode_ids",
                     "contradicting_episode_ids", "basis_observation_ids"}

    def add(value):
        if isinstance(value, str) and value:
            refs.update(_aliases(value))
        elif isinstance(value, list):
            for item in value:
                add(item)

    def visit(value):
        if isinstance(value, dict):
            ref = value.get("source_ref")
            if isinstance(ref, str) and isinstance(value.get("source", value.get("excerpt")), str):
                for alias in _aliases(ref):
                    sources[alias] = value
            for key, child in value.items():
                if key in identity_keys:
                    add(child)
                if key == "id" and any(name in value for name in ("statement", "mechanism", "observation")):
                    add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    # Correction drafts and examples can contain invented citations. They are
    # previous model output, never additional scientific evidence. Expand only
    # actual material sections so request pointers still work without granting
    # authority to unrelated top-level metadata.
    for key in DESIGNER_MATERIAL_FIELDS:
        visit(_context_value(context, key))
    return refs, sources


def _valid_reference(reference: str, refs: set[str], sources: dict[str, dict]) -> bool:
    if reference in refs:
        return True
    path, separator, symbol = reference.partition("::")
    match = re.fullmatch(r"(.+?)(?::([0-9]+)|#L([0-9]+))?", path)
    if not match:
        return False
    source = sources.get(match.group(1))
    if source is None:
        return False
    text = source.get("source", source.get("excerpt"))
    if separator:
        from .review_context import _symbol_lines
        return bool(_symbol_lines(text, symbol))
    line = match.group(2) or match.group(3)
    # Excerpts do not prove original file line numbers. Cite their path/symbol.
    return "source" in source and line is not None and 1 <= int(line) <= len(text.splitlines())


def _schema_errors(exc: ValidationError) -> list[dict]:
    errors = []
    for row in exc.errors():
        loc = list(row["loc"])
        message = "missing " + str(loc[-1]) if row["type"] == "missing" and loc else row["msg"]
        errors.append({"type": row["type"], "loc": loc, "msg": message})
    return errors


def _native_task(context: dict) -> dict:
    draft = _context_value(context, "draft", {})
    selected = _context_value(context, "selected_research_task", {})
    draft = draft if isinstance(draft, dict) else {}
    selected = selected if isinstance(selected, dict) else {}
    return {**selected, **{key: draft[key] for key in (
        "parent_candidate_id", "control_candidate_id", "initial_fidelity", "negative_sampling_policy") if key in draft}}


def validate_designer_native_context(native_context: dict) -> list[dict]:
    """Check existing frozen binding/fidelity rules before spending an API call."""
    task = _native_task(native_context)
    proposal = task.get("proposal") if isinstance(task.get("proposal"), dict) else task
    parent = _context_value(native_context, "parent_hook_config", {})
    invalid_parent_type = not isinstance(parent, dict)
    parent = parent if isinstance(parent, dict) else {}
    parent_id = str(task.get("parent_candidate_id") or proposal.get("parent_candidate_id")
                    or parent.get("target_id") or "baseline")
    errors = []

    def error(location, reason):
        errors.append({"type": "designer_native_input_invalid", "loc": location, "msg": reason})

    for field in ("draft", "selected_research_task"):
        if not isinstance(_context_value(native_context, field, {}), dict):
            error([field], field + "_invalid")
    if _context_value(native_context, "selected_research_task", {}):
        for field in ("expected_observation", "completion_condition"):
            value = proposal.get(field)
            if not isinstance(value, str) or not value.strip():
                error(["selected_research_task", field], "missing or blank inherited " + field)
    if invalid_parent_type:
        error(["parent_hook_config"], "parent_hook_config_invalid")
    elif parent_id != "baseline" and not all(isinstance(parent.get(section), dict) for section in SECTIONS):
        error(["parent_hook_config"], "parent_hook_config_missing")
    elif any(section in parent and not isinstance(parent[section], dict) for section in SECTIONS):
        error(["parent_hook_config"], "parent_hook_config_invalid")
    if parent.get("target_id") and parent["target_id"] != parent_id:
        error(["parent_hook_config", "target_id"], "parent_hook_config_identity_mismatch")
    scope = proposal.get("evaluation_scope") or task.get("evaluation_scope") or {}
    if not isinstance(scope, dict):
        error(["selected_research_task", "evaluation_scope"], "evaluation_scope_invalid")
    else:
        fidelity = task.get("initial_fidelity") or proposal.get("initial_fidelity")
        fidelity = fidelity or ("pilot" if scope.get("phase") == "pilot" else "full")
        if fidelity not in {"pilot", "full"}:
            error(["draft", "initial_fidelity"], "fidelity_invalid")
    return errors


def _without_analysis_scope(raw: Any) -> tuple[Any, list[dict]]:
    """Drop the one known mirrored key: scope "scientific" on an analysis requirement.

    implementation_requirements carry scope; analysis requirements are always
    scientific, and models copy the sibling key. Any other extra key, or another
    scope value, still fails validation.
    """
    brief = raw.get("design_brief") if isinstance(raw, dict) else None
    rows = brief.get("analysis_requirements") if isinstance(brief, dict) else None
    if not isinstance(rows, list) or not any(isinstance(row, dict) and row.get("scope") == "scientific" for row in rows):
        return raw, []
    normalized, cleaned = [], []
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("scope") == "scientific":
            row = {key: value for key, value in row.items() if key != "scope"}
            normalized.append({"loc": ["design_brief", "analysis_requirements", index, "scope"],
                               "kind": "redundant_scientific_scope_removed"})
        cleaned.append(row)
    return {**raw, "design_brief": {**brief, "analysis_requirements": cleaned}}, normalized


def validate_designer_reply(raw: Any, native_context: dict,
                            delivered_context: dict | None = None) -> dict:
    """Validate only API-owned fields, then add local metadata and frozen facts.

    Return ``reply``, ``errors``, ``native_errors``, ``design_brief`` and
    ``experiment_spec``. Both error lists must be empty to use local outputs. Semantic
    missing_inputs is a valid reported obstruction, not a malformed response.
    Native context is the frozen full task; delivered_context is the actual API
    input and constrains citations. Neither input nor raw is modified.
    """
    result = {"reply": None, "errors": [], "native_errors": validate_designer_native_context(native_context),
              "design_brief": None, "experiment_spec": None}
    raw, normalized = _without_analysis_scope(raw)
    try:
        reply = DesignerReply.model_validate(raw).model_dump()
    except ValidationError as exc:
        result["errors"] = _schema_errors(exc)
        return result
    result["reply"] = reply
    if normalized:
        result["normalizations"] = normalized
    if result["native_errors"]:
        return result
    errors = result["errors"]

    def error(kind, loc, message):
        errors.append({"type": kind, "loc": loc, "msg": message})

    delivered = native_context if delivered_context is None else delivered_context
    refs, sources = _delivered_references(delivered)
    for index, reference in enumerate(reply["evidence_refs"]):
        if not _valid_reference(reference, refs, sources):
            error("designer_evidence_ref_not_delivered", ["evidence_refs", index],
                  reference + " is not a supplied evidence reference; cite supplied materials or use [].")

    content = reply["design_brief"]
    if content is None:
        return result
    task = _native_task(native_context)
    seen = set()
    for field in ("implementation_requirements", "analysis_requirements"):
        for index, requirement in enumerate(content[field]):
            rid = requirement["requirement_id"]
            loc = ["design_brief", field, index, "requirement_id"]
            if rid.startswith(NATIVE_MEASUREMENT_PREFIX):
                error("designer_requirement_id_reserved", loc,
                      "Use an ID outside reserved prefix " + NATIVE_MEASUREMENT_PREFIX)
            if rid in seen:
                error("duplicate_stage_requirement_id", loc, "requirement_id must be unique across stages: " + rid)
            seen.add(rid)
    if (reply["status"] == "completed" and not reply["missing_inputs"]
            and task.get("task_kind", "new_method") == "new_method"
            and not any(row["core"] for row in content["implementation_requirements"])):
        error("designer_core_implementation_missing", ["design_brief", "implementation_requirements"],
              "A completed new_method requires at least one core implementation requirement.")
    if errors:
        return result

    # A valid report of a real obstruction must not be forced to invent an
    # executable design (or spend another API call fixing unavailable inputs).
    if reply["status"] != "completed" or reply["missing_inputs"]:
        return result

    native_brief = copy.deepcopy(content)
    native_brief.update(
        schema_version="eeg_research.design_brief.v2",
        measurement_requirements=native_measurement_requirements(),
        required_capability_ids=list(dict.fromkeys(reply["additional_capability_ids"])),
        method_evidence_refs=copy.deepcopy(reply["evidence_refs"]),
        # The selected task already fixes the expected observation; the Designer only tests it.
        expected_observation=task.get("expected_observation") or "",
        completion_condition=task.get("completion_condition") or "",
    )
    for requirement in native_brief["implementation_requirements"]:
        requirement.update(kind="implementation", binding_stage="design", code_locations=[])
    try:
        # Explicit stage lists ensure the stored contract retains later stages.
        native_brief = DesignerBrief.model_validate(native_brief).model_dump()
        spec = materialize_design_brief(native_brief, task,
                                       _context_value(native_context, "parent_hook_config", {}))
    except (DesignContractError, ValidationError) as exc:
        reason = exc.reason if isinstance(exc, DesignContractError) else str(exc.errors()[0]["loc"])
        if isinstance(exc, DesignContractError) and (reason.startswith("parent_hook_config") or reason == "fidelity_invalid"):
            result["native_errors"].append({"type": "designer_native_input_invalid", "loc": ["parent_hook_config"], "msg": reason})
        else:
            error("design_contract_invalid", ["design_brief"], reason)
        return result
    # Standard hook capabilities are inferred once and written to the stored
    # brief as well as the executable record, so downstream views agree.
    spec["design_brief"]["required_capability_ids"] = copy.deepcopy(spec["required_capability_ids"])
    result.update(design_brief=copy.deepcopy(spec["design_brief"]), experiment_spec=spec)
    return result
