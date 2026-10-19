"""New-research planning wire: model science, native task bindings and identities.

Pure functions only. The native caller freezes the directory before transport;
raw provider JSON is never mutated or mistaken for an executable proposal.
"""
from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from .schemas import (CriticWireReply, PlannerCorrectionWire, PlannerGenerationWire,
                      PlannerProposalWire, PlannerTaskProposalWire)

VERSION = "eeg_research.planning_contract.v2"
_ACTIONS = {"screening_2fold": "run_screening", "pilot": "run_pilot", "full": "run_full",
            "confirmation_10fold": "run_confirmation", "diagnostic": "collect_diagnostics"}
_COMPLETION = {
    "new_method": "Complete the selected method implementation, native measurement and analysis; publish its terminal observation and memory handoff. A negative result is a valid completed observation.",
    "validation": "Complete the selected existing-method measurement or diagnostic, analyze the verified evidence and publish the terminal observation and memory handoff.",
    "confirmation": "Measure the frozen existing method under the complete confirmation scope, analyze paired evidence and publish the terminal observation and memory handoff.",
    "diagnosis": "Diagnose existing verified results without new training, then publish the evidence-grounded terminal observation and memory handoff.",
}


def enabled(full: dict | None) -> bool:
    full = full or {}
    goal = full.get("goal") if isinstance(full.get("goal"), dict) else full
    execution = full.get("research_execution") or goal.get("research_execution") or {}
    return execution.get("planning_contract_version") == "v2"


def _error(kind, loc, message):
    return {"type": kind, "loc": list(loc), "msg": message}


def _execution_seed(full: dict) -> int | None:
    # loop._train uses this scalar for run_pilot/run_full. training_seeds is
    # the allowed replication roster, not the seed selector for these actions.
    for source in (full.get("execution_protocol"), full.get("protocol")):
        if isinstance(source, dict):
            for field in ("training_seed", "seed"):
                if type(source.get(field)) is int:
                    return source[field]
    return None


def _scope(full: dict, phase: str) -> dict | None:
    if phase == "diagnostic":
        return {"phase": phase}
    scopes = (full.get("frozen_evaluation_scopes") or {}).get("phases") or {}
    value = scopes.get(phase)
    if isinstance(value, dict) and value.get("phase") == phase:
        if full.get("evaluation_mode") == "loso_method_search":
            expected = 2 if phase == "screening_2fold" else 10
            folds = value.get("folds")
            if (not isinstance(folds, list) or len(folds) != expected
                    or any(not isinstance(fold, str) or not fold.strip() for fold in folds) or len(set(folds)) != expected
                    or type(value.get("seed")) is not int or value["seed"] != 0):
                return None
        value = copy.deepcopy(value)
        if full.get("evaluation_mode") != "loso_method_search":
            seed = _execution_seed(full)
            if seed is not None:
                if "seed" in value and value["seed"] != seed:
                    return None
                value["seed"] = seed
        return value
    if full.get("evaluation_mode") == "loso_method_search":
        return None
    seed = _execution_seed(full)
    if seed is not None:
        return {"phase": phase, "seed": seed}
    # A goal/replication seed roster cannot establish the main run's scalar.
    # The caller must supply a frozen scope or the actual execution protocol.
    return None


def freeze_planning_contract(full: dict) -> dict:
    """Derive only task pairs admitted by the current native task rules."""
    if isinstance(full.get("_planning_contract"), dict):
        return copy.deepcopy(full["_planning_contract"])
    from .action_planning import digest
    from .plan_protocol import validate_task
    from .planner import known_candidate_ids, training_actions
    goal = full.get("goal") or {}
    schedule = full.get("evaluation_schedule") or goal.get("evaluation_schedule") or "full_only"
    actions = set(full.get("available_actions") or [])
    eligible = full.get("eligible_targets") or {}
    permitted = training_actions({**goal, "evaluation_schedule": schedule})
    measurement_phases = ("screening_2fold",) if schedule == "screen_then_confirm" else ("pilot", "full")
    # loop._train admits only run_full/run_screening/run_confirmation for LOSO
    # method suites. The single-target pilot remains a separate valid option.
    if full.get("evaluation_mode") == "loso_method_search" and schedule != "screen_then_confirm":
        measurement_phases = ("full",)
    candidates = known_candidate_ids(full)
    choices = []
    if actions.intersection({"design_experiment", "propose_experiment"}):
        choices.extend(("new_method", None, phase) for phase in measurement_phases
                       if _ACTIONS[phase] in permitted)
    if "collect_diagnostics" in actions:
        for target in sorted(candidates):
            choices.extend((("diagnosis", target, "diagnostic"), ("validation", target, "diagnostic")))
    for phase in measurement_phases:
        if _ACTIONS[phase] in permitted:
            choices.extend(("validation", target, phase) for target in sorted(set(eligible.get(_ACTIONS[phase]) or []))
                           if target in candidates)
    confirmation = "confirmation_10fold" if schedule == "screen_then_confirm" else "full"
    if _ACTIONS[confirmation] in permitted:
        choices.extend(("confirmation", target, confirmation) for target in
                       sorted(set(eligible.get(_ACTIONS[confirmation]) or [])) if target in candidates)
    tasks, errors = [], []
    for kind, target, phase in choices:
        try:
            validate_task({"task_kind": kind, "target_candidate_ref": target, "evaluation_scope": {"phase": phase}}, full)
        except (ValueError, TypeError, KeyError):
            continue
        scope = _scope(full, phase)
        if scope is None:
            problem = _error("planning_native_input_invalid", ["frozen_evaluation_scopes", "phases", phase],
                             "Missing valid frozen evaluation scope for " + phase)
            if problem not in errors:
                errors.append(problem)
            continue
        identity = {"task_kind": kind, "target_candidate_ref": target, "evaluation_scope": scope}
        tasks.append({"task_ref": "task_" + digest(identity)[:16], **identity,
                      "completion_condition": _COMPLETION[kind]})
    if not tasks and not errors:
        errors.append(_error("planning_no_native_tasks", ["tasks"], "No task is eligible under the frozen native state."))
    return {"version": VERSION, "tasks": tasks, "native_errors": errors}


def task_catalog(full: dict) -> list[dict]:
    contract = full.get("_planning_contract")
    if isinstance(contract, dict):
        return copy.deepcopy(contract.get("tasks") or [])
    if isinstance(full.get("fixed_task"), dict):
        return [copy.deepcopy(full["fixed_task"])]
    return copy.deepcopy(full.get("task_options") or [])


def planner_wire_schema(full: dict | None = None, *, correction: bool = False, remaining: int = 4) -> dict:
    full = full or {}
    repair = full.get("plan_repair_request") or {}
    correction = correction or bool(repair)
    if correction and isinstance(repair.get("remaining_slots"), int):
        remaining = min(remaining, repair["remaining_slots"])
    tasks = task_catalog(full)
    model = PlannerTaskProposalWire if len(tasks) > 1 else PlannerProposalWire
    item = model.model_json_schema()
    if len(tasks) > 1:
        item["properties"]["task_ref"]["enum"] = [row["task_ref"] for row in tasks]
    return {"title": "PlannerCorrectionWire" if correction else "PlannerGenerationWire", "type": "object",
            "additionalProperties": False, "required": ["proposals"], "properties": {"proposals": {
                "type": "array", "minItems": 1 if correction else 2, "maxItems": min(4, max(0, remaining)),
                "items": item}}}


def critic_wire_schema() -> dict:
    return CriticWireReply.model_json_schema()


def _validation_errors(exc: ValidationError, prefix=()) -> list[dict]:
    return [_error(row["type"], (*prefix, *row["loc"]),
                   "missing " + str(row["loc"][-1]) if row["type"] == "missing" and row["loc"] else row["msg"])
            for row in exc.errors()]


def scientific_proposal_view(native: dict, full: dict) -> dict:
    """The provider never needs locally duplicated intent/value or protocol fields."""
    result = {"question": native["question"], "method": native["task_description"],
              "rationale": native["scientific_intent"], "expected_observation": native["expected_observation"],
              "implementation_risk": native.get("implementation_risk"),
              "basis_observation_ids": copy.deepcopy(native.get("basis_observation_ids") or []),
              "required_context_refs": copy.deepcopy(native.get("required_context_refs") or [])}
    tasks = task_catalog(full)
    if len(tasks) > 1:
        matched = next((row for row in tasks if row["task_kind"] == native["task_kind"]
            and row.get("target_candidate_ref") == native.get("target_candidate_ref")
            and row["evaluation_scope"] == native["evaluation_scope"]), None)
        if matched is None:
            raise ValueError("proposal_task_not_in_frozen_directory")
        result["task_ref"] = matched["task_ref"]
    return result


def materialize_planner_reply(raw: Any, full: dict, *, stage: str = "generation",
                              existing: list[dict] | None = None) -> dict:
    """Check every supplied field before adding native IDs, task and scope."""
    from .plan_protocol import round_id
    contract = full.get("_planning_contract") or {}
    result = {"proposals": [], "errors": [], "native_errors": copy.deepcopy(contract.get("native_errors") or [])}
    tasks = task_catalog(full)
    if contract.get("version") != VERSION or not tasks:
        if not result["native_errors"]:
            result["native_errors"].append(_error("planning_native_input_invalid", ["_planning_contract"],
                "The native task directory must be frozen before planning."))
    if result["native_errors"]:
        return result
    correction = stage == "correction" or stage.startswith("correction_")
    try:
        outer = (PlannerCorrectionWire if correction else PlannerGenerationWire).model_validate(raw)
        proposal_rows = outer.proposals
    except ValidationError as exc:
        result["errors"] = _validation_errors(exc)
        # Preserve a valid singleton while a later bounded correction supplies
        # a distinct sibling; a failed sibling never discards accepted science.
        if (stage == "generation" and isinstance(raw, dict) and set(raw) == {"proposals"}
                and isinstance(raw["proposals"], list) and len(raw["proposals"]) == 1
                and all(row["loc"] == ["proposals"] and row["type"] == "too_short" for row in result["errors"])):
            proposal_rows = raw["proposals"]
        else:
            return result
    remaining = 4 - len(existing or [])
    if len(proposal_rows) > remaining:
        result["errors"].append(_error("planning_pool_capacity_exceeded", ["proposals"],
            "Return at most " + str(remaining) + " corrected/additional proposals."))
        return result
    model = PlannerTaskProposalWire if len(tasks) > 1 else PlannerProposalWire
    by_ref = {row["task_ref"]: row for row in tasks}
    for index, raw_proposal in enumerate(proposal_rows):
        try:
            wire = model.model_validate(raw_proposal).model_dump()
        except ValidationError as exc:
            result["errors"].extend(_validation_errors(exc, ("proposals", index)))
            continue
        selected = by_ref.get(wire.get("task_ref")) if len(tasks) > 1 else tasks[0]
        if selected is None:
            result["errors"].append(_error("planning_task_ref_unknown", ["proposals", index, "task_ref"],
                                         "Select one supplied task_ref."))
            continue
        invalid = False
        for field in ("basis_observation_ids", "required_context_refs"):
            if len(set(wire[field])) != len(wire[field]):
                invalid = True
                result["errors"].append(_error("planning_reference_duplicate", ["proposals", index, field],
                                             "References must be unique."))
        if invalid:
            continue
        tag = ("c" if stage == "correction" else "c" + stage.removeprefix("correction_") + "p") if correction else "g"
        pid = "proposal_" + round_id(full).removeprefix("research_round_") + "_" + tag + str(index + 1)
        result["proposals"].append({"proposal_id": pid, "task_kind": selected["task_kind"],
            "target_candidate_ref": selected.get("target_candidate_ref"), "question": wire["question"],
            "task_description": wire["method"], "scientific_intent": wire["rationale"],
            "evaluation_scope": copy.deepcopy(selected["evaluation_scope"]),
            "expected_observation": wire["expected_observation"],
            "completion_condition": selected["completion_condition"], "cost_assessment": {},
            "implementation_risk": wire["implementation_risk"], "value_rationale": wire["rationale"],
            "basis_observation_ids": copy.deepcopy(wire["basis_observation_ids"]),
            "required_context_refs": copy.deepcopy(wire["required_context_refs"])})
    return result


def materialize_critic_reply(raw: Any, frozen_pool: dict, full: dict) -> dict:
    result = {"reply": None, "errors": [], "native_errors": []}
    try:
        wire = CriticWireReply.model_validate(raw).model_dump()
    except ValidationError as exc:
        result["errors"] = _validation_errors(exc)
        return result
    ids = [row["proposal_id"] for row in wire["evaluations"]]
    wanted = {row["proposal_id"] for row in frozen_pool.get("proposals") or []}
    if len(ids) != len(set(ids)) or set(ids) != wanted:
        result["errors"].append(_error("planning_critic_coverage_invalid", ["evaluations"],
                                      "Evaluate every frozen proposal exactly once."))
    ranks = [row["rank"] for row in wire["evaluations"]]
    if set(ranks) != set(range(1, len(wanted) + 1)) or len(set(ranks)) != len(ranks):
        result["errors"].append(_error("planning_critic_rank_invalid", ["evaluations"],
                                      "Ranks must be unique and complete from 1 to N."))
    basis = wire["selected_basis_observation_ids"]
    delivered = full.get("_delivered_observation_ids") or set()
    if len(set(basis)) != len(basis) or any(ref not in delivered for ref in basis):
        result["errors"].append(_error("planning_critic_basis_invalid", ["selected_basis_observation_ids"],
                                      "Use unique observation IDs from the actual delivered context."))
    if not result["errors"]:
        result["reply"] = {"protocol_version": "research_plan_critique.v1",
            "proposal_pool_digest": frozen_pool["pool_digest"], **wire,
            "selected_proposal_id": next(row["proposal_id"] for row in wire["evaluations"] if row["rank"] == 1)}
    return result
