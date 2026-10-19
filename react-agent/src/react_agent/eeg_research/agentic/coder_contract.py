"""Derive the Coder's concrete tool contract from its approved input.

Hashes and artifact metadata belong to the runtime. The Coder supplies code
locations in a flat requirement map; final source checks still resolve them.
"""
from __future__ import annotations

import copy
from typing import Any

from .coder import (RUNTIME_BOUND_WRITE_ARGS, TOOL_ARGUMENTS, bound_implementation_requirements, tool_catalog,
                    validate_tool_request)


def _request_value(payload: dict[str, Any], key: str) -> Any:
    """Read the frozen request, including its exact request-local pointers."""
    from .context.representations import resolve_request_pointer
    from .context.sources import pointer_escape

    def expand(value: Any, pointer: str) -> Any:
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(payload, pointer), pointer)
            return {name: expand(child, pointer + "/" + pointer_escape(name)) for name, child in value.items()}
        if isinstance(value, list):
            return [expand(child, pointer + "/" + str(index)) for index, child in enumerate(value)]
        return value

    try:
        return expand(resolve_request_pointer(payload, "/" + key), "/" + key)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _declarations(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    # A frozen compact request must carry the exact same declaration roster
    # used to generate its system schema. In particular, a missing roster must
    # never silently fall back to the legacy unconstrained map on paid recovery.
    if payload.get("coder_context_version") == "coder_implementation_input.v2":
        design = _request_value(payload, "approved_design")
        rows = design.get("implementation_requirements") if isinstance(design, dict) else None
        if not isinstance(rows, list):
            raise ValueError("coder_native_input_invalid:missing_implementation_requirements")
        identifiers = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("requirement_id"), str)
                    or not row["requirement_id"].strip()
                    or (row.get("probe") is not None and not isinstance(row["probe"], dict))):
                raise ValueError("coder_native_input_invalid:invalid_implementation_requirements")
            identifiers.append(row["requirement_id"])
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("coder_native_input_invalid:duplicate_requirement_id")
        return rows, True
    spec = _request_value(payload, "experiment_spec")
    spec = spec if isinstance(spec, dict) else {}
    experiment = spec.get("experiment") or spec
    known = "implementation_requirements" in experiment or experiment.get("design_contract_version") == "v2"
    rows = experiment.get("implementation_requirements") or []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("requirement_id"), str)], known


_MODULE_PATHS = ("input_module_path", "representation_module_path")


def _probe_schema(requirement: dict[str, Any]) -> dict[str, Any]:
    declared = requirement.get("probe") or {}
    properties = {key: {"const": copy.deepcopy(value)} for key, value in declared.items()}
    for key in _MODULE_PATHS:
        if not declared.get(key):
            properties[key] = {"type": ["string", "null"]}
    return {"type": ["object", "null"], "properties": properties,
            # The existing binder permits an undeclared key only when its
            # supplied value is None; preserve that edge case without inviting
            # the model to invent scientific probe parameters.
            "additionalProperties": {"type": "null"}}


def _map_schema(declarations: list[dict[str, Any]], *, known: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "description": (
        "Flat requirement_id to source_refs/optional probe. Runtime supplies artifact and hash metadata. "
        "Only include bindings being supplied; omission preserves the existing native behavior.")}
    if known:
        result["properties"] = {row["requirement_id"]: {
            "type": "object", "properties": {
                "source_refs": {"type": ["array", "null"], "items": {"type": "string"}},
                "probe": _probe_schema(row),
            }, "additionalProperties": False,
        } for row in declarations}
        result["additionalProperties"] = False
    else:
        # Legacy callers without an approved requirement roster retain their
        # existing generic map argument; a roster is never guessed from history.
        result["additionalProperties"] = True
    return result


def coder_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """One task-specific catalog, JSON Schema and fillable finish template.

    Requirement IDs are allowed properties rather than required properties:
    optional declarations and existing source bindings retain their semantics.
    """
    declarations, known = _declarations(payload)
    mapping_schema = _map_schema(declarations, known=known)
    catalog = tool_catalog()
    branches = []
    for tool, fields in TOOL_ARGUMENTS.items():
        if tool in {"apply_candidate_patch", "edit_candidate_code"}:
            # The runtime binds the write target and its CAS base; the model never sends them.
            fields = {name: spec for name, spec in fields.items() if name not in RUNTIME_BOUND_WRITE_ARGS}
            catalog[tool]["args"] = {name: spec for name, spec in catalog[tool]["args"].items()
                                     if name not in RUNTIME_BOUND_WRITE_ARGS}
        args_schema: dict[str, Any] = {"type": "object"}
        if fields is None:
            args_schema["additionalProperties"] = True
        else:
            properties = {name: {"type": kind} for name, (kind, _required) in fields.items()}
            args_schema.update(properties=properties, additionalProperties=False)
            required = [name for name, (_kind, required) in fields.items() if required]
            if required:
                args_schema["required"] = required
            if tool == "read_code":
                for name in ("start", "end"):
                    properties[name]["minimum"] = 1
            elif tool == "edit_candidate_code":
                properties["edits"].update(minItems=1, maxItems=20, items={"oneOf": [
                    {"type": "object", "properties": {"old": {"type": "string", "minLength": 1},
                                                          "new": {"type": "string"}},
                     "required": ["old", "new"], "additionalProperties": False},
                    {"type": "object", "properties": {"start": {"type": "integer", "minimum": 1},
                                                          "end": {"type": "integer", "minimum": 1},
                                                          "new": {"type": "string"}},
                     "required": ["start", "end", "new"], "additionalProperties": False},
                ]})
            elif tool == "finish_patch":
                properties["implementation_map"] = copy.deepcopy(mapping_schema)
            elif tool == "request_design_revision":
                properties["kind"]["enum"] = ["design_conflict", "scientific_replan"]
                for name in ("requirement_ids", "evidence_refs"):
                    properties[name]["items"] = {"type": "string", "minLength": 1}
        catalog[tool]["args_schema"] = copy.deepcopy(args_schema)
        branches.append({"type": "object", "properties": {"tool": {"const": tool}, "args": args_schema},
                         "required": ["tool", "args"], "additionalProperties": False})
    template = {}
    for requirement in declarations:
        row: dict[str, Any] = {"source_refs": []}
        declared_probe = requirement.get("probe") or {}
        if declared_probe or requirement.get("verification") == "ordered_window_slots":
            pending = {key: "" for key in _MODULE_PATHS if not declared_probe.get(key)}
            if pending:
                row["probe"] = pending
        template[requirement["requirement_id"]] = row
    return {
        "tools": catalog,
        "output_schema": {"title": "CoderToolRequest", "oneOf": branches},
        "finish_patch_template": {"tool": "finish_patch", "args": {"summary": "", "implementation_map": template}},
    }


def validate_coder_request(reply: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Check task-specific shape before execution, without claiming code validity."""
    tool, args, error = validate_tool_request(reply)
    if payload.get("coder_context_version") == "coder_implementation_input.v2":
        try:
            declarations, known = _declarations(payload)
        except ValueError as exc:
            return tool, args, {"ok": False, "error": "coder_native_input_invalid", "detail": str(exc)}
    else:
        declarations, known = [], False
    if error is not None or tool != "finish_patch" or "implementation_map" not in args:
        return tool, args, error
    if payload.get("coder_context_version") != "coder_implementation_input.v2":
        declarations, known = _declarations(payload)
    if not known:
        return tool, args, None
    mapping = args["implementation_map"]
    declared_ids = {row["requirement_id"] for row in declarations}
    if set(mapping) - declared_ids:
        return tool, args, {"ok": False, "error": "implementation_map_unknown_requirement",
                            "allowed_requirement_ids": [row["requirement_id"] for row in declarations],
                            "detail": "Use the flat map in finish_patch_template; runtime writes hash/artifact metadata."}
    for row in mapping.values():
        if not isinstance(row, dict) or set(row) - {"source_refs", "probe"}:
            return tool, args, {"ok": False, "error": "implementation_map_invalid_row"}
        refs = row.get("source_refs") or []
        if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
            return tool, args, {"ok": False, "error": "implementation_map_invalid_source_reference"}
        if row.get("probe") is not None and not isinstance(row["probe"], dict):
            return tool, args, {"ok": False, "error": "implementation_map_invalid_probe"}
    # Reuse the existing semantic binder. It only derives a private copy, does
    # not check source locations and cannot overwrite the approved declaration.
    try:
        bound_implementation_requirements(declarations,
            {"requirements": mapping, "source_sha256": "request", "approved_spec_hash": "request"},
            source_sha256="request", spec_hash="request")
    except (TypeError, ValueError) as exc:
        return tool, args, {"ok": False, "error": str(exc)}
    return tool, args, None
