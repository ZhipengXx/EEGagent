"""Small stage views of an approved contract; stored scientific artifacts stay intact."""
from __future__ import annotations

import ast
import copy
from typing import Any

from .stage_contract import has_stage_contract

_LATER_FIELDS = {"measurement_requirements", "analysis_requirements",
                 "completion_condition", "expected_observation"}
_OPTIONAL_HOOKS = ("fit_statistics", "build_training_transform", "build_training_objective")


def _implementation_spec(spec: dict) -> dict:
    result = {key: copy.deepcopy(value) for key, value in spec.items() if key not in _LATER_FIELDS}
    for key in ("experiment", "design_brief"):
        if isinstance(result.get(key), dict):
            result[key] = _implementation_spec(result[key])
    return result


def _defined_hooks(source: str | None) -> set[str]:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def implementation_stage_payload(payload: dict[str, Any], *, reviewer: bool = True) -> dict[str, Any]:
    """Deliver only implementation acceptance; optional APIs are capabilities, not promises."""
    spec = payload.get("experiment_spec") or {}
    if not has_stage_contract(spec) and payload.get("review_stage") != "implementation":
        return payload
    result = dict(payload)
    experiment = spec.get("experiment") or spec
    result["experiment_spec"] = _implementation_spec(spec)
    result["review_stage"] = "implementation"
    result["stage_acceptance"] = "Current approved implementation requirements, selected hooks and fresh code checks."
    if reviewer:
        required = {"build_encoder", *(experiment.get("required_capability_ids") or [])}
        implemented = _defined_hooks(payload.get("candidate_source"))
        parent = payload.get("parent_source") or {}
        implemented |= _defined_hooks(parent.get("source") if isinstance(parent, dict) else None)
        interface = copy.deepcopy(payload.get("candidate_interface") or {})
        applicability = {}
        for hook in _OPTIONAL_HOOKS:
            applicability[hook] = ("required" if hook in required else
                                   "existing_behavior" if hook in implemented else "unselected_optional")
            if applicability[hook] == "unselected_optional":
                interface.pop(hook, None)
        result["candidate_interface"] = interface
        result["hook_applicability"] = applicability
    return result


def with_selected_task(payload: dict[str, Any], role: str, selected: dict[str, Any]) -> dict[str, Any]:
    """Keep final transport enrichment within the caller's stage view."""
    if role in {"candidate_coder", "candidate_reviewer"}:
        payload = implementation_stage_payload(payload, reviewer=role == "candidate_reviewer")
    proposal = selected.get("proposal") or {}
    fields = ("question", "task_description", "scientific_intent", "evaluation_scope",
              "expected_observation", "completion_condition")
    if payload.get("review_stage") in {"implementation", "analysis"}:
        fields = ("question", "scientific_intent")
    return {**payload, "selected_research_task": {key: copy.deepcopy(proposal[key]) for key in fields if key in proposal},
            "origin_plan_id": selected["plan_id"]}
