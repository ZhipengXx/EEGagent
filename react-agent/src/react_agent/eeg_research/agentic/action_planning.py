"""Independent action drafts and a normalized, source-bound execution envelope.

Rejected model output is audit input. Only this module's validated projection
can become an execution decision; selection never copies an unrelated draft.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable

from pydantic import ValidationError

from .schemas import ActionPlanningConfig, PlannerActionPayload, PlannerDecision, PlannerOption, PlannerProposal


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def planning_config(observation: dict) -> ActionPlanningConfig:
    return ActionPlanningConfig.model_validate(observation.get("action_planning") or
        (observation.get("goal") or {}).get("action_planning") or {})


def planner_round_identity(observation: dict) -> str:
    """A failed reentry cannot buy a new round by spending calls or writing dN.

    Changed verified evidence, target/source/spec/repair, plan or authorized
    scope is a new planning input. Transport errors and prior drafts are not.
    """
    keys = ("goal", "contract", "execution_fingerprint", "contract_fingerprint", "available_actions",
        "eligible_targets", "evidence", "candidate_id", "candidate_ready", "candidates", "candidate_admissions",
        "detached_candidates", "repair_task", "pending_candidate_check", "implementation_target_id",
        "implementation_source", "implementation_lifecycle", "training_target_sources", "artifact_read_index",
        "audit_feedback", "report_draft", "research_questions", "hypothesis", "experiment",
        "action_input_epochs", "verified_development_facts")
    plan = (observation.get("research_plan") or {}).get("plan") or {}
    return "round_" + digest({**{key: observation.get(key) for key in keys}, "research_plan": plan})[:32]


def target_binding(observation: dict, action: str, target: str | None) -> dict:
    """Runtime identities, not the model's claimed executable flag."""
    repair = observation.get("repair_task") or {}
    source = (observation.get("training_target_sources") or {}).get("by_candidate", {}).get(target) or {}
    if target is not None and (observation.get("implementation_source") or {}).get("candidate_id") == target:
        source = observation["implementation_source"]
    row = next((r for r in observation.get("candidates") or [] if r.get("candidate_id") == target), {})
    lifecycle = observation.get("implementation_lifecycle") or {}
    experiment = observation.get("experiment") or {}
    return {
        "candidate_id": target,
        "source_hash": source.get("source_hash") or row.get("source_hash"),
        "spec_hash": source.get("spec_hash") or row.get("spec_hash") or
            (lifecycle.get("spec_hash") if lifecycle.get("target_id") == target else None) or
            (experiment.get("spec_hash") if target == observation.get("implementation_target_id") else None),
        "attempt_id": (repair.get("attempt_id") or row.get("attempt_id")) if repair.get("candidate_id") == target else row.get("attempt_id"),
        "repair_identity": digest(repair) if repair.get("candidate_id") == target and action in {"implement_candidate", "repair_candidate"} else None,
        "execution_fingerprint": observation.get("execution_fingerprint") or (observation.get("contract") or {}).get("execution_fingerprint"),
        "contract_fingerprint": observation.get("contract_fingerprint") or (observation.get("contract") or {}).get("fingerprint"),
    }


def operation_signature(payload: dict) -> str:
    """Questions, labels and rationale cannot create a second operation."""
    binding = payload.get("target_binding") or {}
    operation = payload["action"]
    if operation in {"implement_candidate", "repair_candidate"} and binding.get("repair_identity"):
        operation = "repair_candidate"
    params = {k: payload.get(k) for k in ("hypothesis_draft", "experiment_draft", "read_requests",
        "report_revision", "resolves_issue_ids", "plan_update", "action_depends_on_plan_update", "stop_reason")}
    return digest({"operation": operation, "target_id": payload.get("target_id"), "binding": binding, "parameters": params})


def _payload(option: PlannerOption, raw: dict, observation: dict) -> tuple[dict, str]:
    if option.payload is not None:
        payload = option.payload.model_dump(exclude_none=True)
        origin = "independent_option_payload"
    elif not (observation.get("action_planning") is not None or
            (observation.get("goal") or {}).get("action_planning") is not None or
            (observation.get("goal") or {}).get("evaluation_schedule") == "screen_then_confirm") and raw.get("selected_option_id") == option.option_id and all(
        raw.get(k) == getattr(option, k) for k in ("action", "target_id", "question_id")
    ):
        # Old campaigns can bind the original selected item's own inline body.
        # Never use it for a fallback or an unselected design/read/report draft.
        payload = PlannerActionPayload.model_validate({k: v for k, v in raw.items()
            if k in PlannerActionPayload.model_fields}).model_dump(exclude_none=True)
        origin = "legacy_original_selected_payload"
    else:
        raise ValueError("option_payload_missing")
    for key in ("action", "target_id", "question_id"):
        if payload.get(key) != getattr(option, key):
            raise ValueError("option_payload_identity_mismatch:" + key)
    for key in ("evidence_refs", "required_artifact_refs", "required_read_digests"):
        if payload.get(key, []) != getattr(option, key):
            raise ValueError("option_payload_reference_mismatch:" + key)
    if payload["action"] in {"design_experiment", "propose_experiment"}:
        if not payload.get("hypothesis_draft"):
            raise ValueError("design_hypothesis_missing")
        if not isinstance(payload.get("experiment_draft"), dict) or not payload["experiment_draft"].get("intervention"):
            raise ValueError("design_experiment_draft_missing")
        if option.intervention != payload["experiment_draft"].get("intervention"):
            raise ValueError("selected_option_intervention_mismatch")
    elif option.intervention is not None:
        raise ValueError("option_intervention_requires_design")
    if payload["action"] in {"implement_candidate", "repair_candidate", "run_pilot", "run_full", "run_screening", "run_confirmation", "replicate"} and (
        payload.get("hypothesis_draft") or payload.get("experiment_draft")
    ):
        raise ValueError("bound_target_action_cannot_redesign_scientific_method")
    if payload["action"] == "stop" and not payload.get("stop_reason"):
        raise ValueError("option_stop_reason_missing")
    if payload["action"] == "retrieve_memory" and not payload.get("read_requests") and not (
        observation.get("memory") or observation.get("lessons")
    ):
        raise ValueError("memory_action_has_no_new_input")
    supplied = payload.get("target_binding") or {}
    binding = target_binding(observation, payload["action"], payload.get("target_id"))
    if supplied and supplied != binding:
        raise ValueError("option_target_binding_changed")
    payload["target_binding"] = binding
    if payload["action"] in {"implement_candidate", "repair_candidate"}:
        lifecycle = observation.get("implementation_lifecycle") or {}
        if lifecycle.get("target_id") == payload.get("target_id") and lifecycle.get("status") not in {
            None, "registered_design_verified_for_existing_implementation_gate"
        }:
            raise ValueError("implementation_dependency_invalid:" + str(lifecycle.get("reason") or lifecycle["status"]))
    if payload["action"] in {"run_pilot", "run_full", "replicate", "run_screening", "run_confirmation"}:
        source_view = (observation.get("training_target_sources") or {}).get("by_candidate", {}).get(payload.get("target_id")) or {}
        if source_view.get("missing_inputs"):
            raise ValueError("training_source_dependency_invalid:" + ",".join(source_view["missing_inputs"]))
    if payload["action"] == "repair_candidate" or (
        payload["action"] == "implement_candidate" and binding.get("repair_identity")
    ):
        task = observation.get("repair_task") or {}
        if task.get("candidate_id") != payload.get("target_id") or int(task.get("remaining") or 0) <= 0:
            raise ValueError("repair_target_unavailable")
        if not binding.get("source_hash") or not binding.get("spec_hash") or not (
            task.get("reason") or task.get("issues") or task.get("failure") or task.get("last_error") or task.get("detail")
        ):
            raise ValueError("repair_inputs_missing")
    return payload, origin


def validate_option(raw_option: Any, raw: dict, observation: dict, validator: Callable) -> tuple[dict | None, dict]:
    try:
        option = PlannerOption.model_validate(raw_option)
        if not option.executable:
            raise ValueError("option_declared_nonexecutable")
        payload, origin = _payload(option, raw, observation)
        normalized = option.model_dump(exclude_none=True)
        # Cost is mechanical runtime information; stale model estimates are
        # recorded, then replaced without changing the proposed intervention.
        from .planner import resolve_cost_estimate
        _, cost = resolve_cost_estimate(observation, option.action, option.target_id)
        normalized.update(estimated_cost=cost.get("estimated_cost") or {},
                          cost_basis=cost.get("cost_basis") or [],
                          cost_confidence=cost.get("cost_confidence") or "unknown", payload=payload)
        decision = {**payload, "options": [normalized], "selected_option_id": option.option_id,
                    "selection_rationale": "Runtime validates this independent draft."}
        parsed = validator(decision)
        if not parsed.get("ok"):
            raise ValueError(parsed.get("detail") or "option_runtime_invalid")
        return normalized, {"option_id": option.option_id, "status": "valid", "payload_origin": origin,
            "raw_digest": digest(raw_option), "execution_signature": operation_signature(payload),
            "mechanical_cost_derived": {"submitted": option.estimated_cost.model_dump(),
                                        "runtime": normalized["estimated_cost"]}}
    except ValidationError as exc:
        return None, {"option_id": raw_option.get("option_id") if isinstance(raw_option, dict) else None,
            "status": "rejected", "error": "option_schema_invalid", "errors": exc.errors(include_input=False, include_context=False), "draft": copy.deepcopy(raw_option)}
    except (ValueError, TypeError, KeyError) as exc:
        return None, {"option_id": raw_option.get("option_id") if isinstance(raw_option, dict) else None,
            "status": "rejected", "error": str(exc), "draft": copy.deepcopy(raw_option)}


def normalize_pool(raw: Any, observation: dict, validator: Callable, *, existing: list[dict] | None = None) -> dict:
    try:
        PlannerProposal.model_validate(raw)
    except ValidationError as exc:
        return {"ok": False, "detail": "planner_proposal_invalid", "raw": copy.deepcopy(raw),
                "rejection_records": [{"error": "proposal_schema_invalid", "errors": exc.errors(include_input=False, include_context=False)}]}
    pool = copy.deepcopy(existing or [])
    records = []
    seen = {operation_signature(row["payload"]) for row in pool}
    ids = {row["option_id"] for row in pool}
    for item in raw.get("options") or []:
        normalized, trace = validate_option(item, raw, observation, validator)
        if normalized is not None:
            signature = trace["execution_signature"]
            if signature in seen or normalized["option_id"] in ids:
                trace.update(status="rejected", error="duplicate_execution_operation", draft=copy.deepcopy(item))
            else:
                pool.append(normalized)
                seen.add(signature)
                ids.add(normalized["option_id"])
        records.append(trace)
    if not pool:
        return {"ok": False, "detail": "no_feasible_action", "raw": copy.deepcopy(raw), "valid_options": [],
                "rejection_records": records}
    by_id = {row["option_id"]: row for row in pool}
    preferred = raw.get("selected_option_id")
    preferred_is_unique = sum(isinstance(item, dict) and item.get("option_id") == preferred
                              for item in raw.get("options") or []) == 1
    order = [preferred if preferred_is_unique else None, *(raw.get("ranking") or []), *[row["option_id"] for row in pool]]
    selected = next(by_id[key] for key in order if key in by_id)
    config = planning_config(observation)
    if len(pool) > config.max_options:
        retained = [selected["option_id"]]
        for row in pool:
            if row["option_id"] not in retained and len(retained) < config.max_options:
                retained.append(row["option_id"])
        pool = [row for row in pool if row["option_id"] in retained]
        for trace in records:
            if trace.get("status") == "valid" and trace.get("option_id") not in retained:
                trace.update(included_in_execution_pool=False, omission_reason="configured_max_options")
    execution = {**copy.deepcopy(selected["payload"]), "options": pool, "selected_option_id": selected["option_id"],
        "selection_rationale": raw.get("selection_rationale") or "Deterministic order within the valid runtime pool."}
    PlannerDecision.model_validate(execution)
    return {"ok": True, "action": execution["action"], "reason_zh": execution.get("reason_zh") or execution.get("decision_rationale") or "",
        "raw": copy.deepcopy(raw), "execution": execution, "valid_options": pool,
        "rejection_records": records, "selection_source": "original_selection" if preferred_is_unique and preferred == selected["option_id"] else "deterministic_valid_pool",
        "pool_summary": {"target_options": config.target_options, "valid_count": len(pool),
            "shortfall_reason": ("valid_selection_executes_without_filling_target" if preferred in by_id else
                "bounded_rejections_or_duplicates") if len(pool) < config.target_options else None},
        "raw_digest": digest(raw), "execution_digest": digest(execution)}


def recovery_option(observation: dict) -> dict | None:
    """A real source-bound repair can recover an empty model pool."""
    task = observation.get("repair_task") or {}
    target = task.get("candidate_id")
    if not target or "repair_candidate" not in (observation.get("available_actions") or []):
        return None
    payload = {"action": "repair_candidate", "target_id": target, "evidence_refs": [],
        "decision_rationale": "Repair the recorded engineering failure without changing the approved method."}
    return {"option_id": "runtime_repair_" + str(target), "action": "repair_candidate", "target_id": target,
        "observed_gap": str(task.get("reason") or task.get("detail") or "recorded_candidate_failure"),
        "expected_information": "A new source-bound implementation check and review.",
        "value_rationale": "Existing bounded repair; no new scientific candidate admission.", "payload": payload}


def decide_options(observation: dict, backend: Callable, validator: Callable) -> dict:
    config = planning_config(observation)
    replies, records = [], []
    raw = backend(observation)
    replies.append(copy.deepcopy(raw))
    parsed = normalize_pool(raw, observation, validator)
    records.extend(parsed.get("rejection_records") or [])
    if parsed.get("ok"):
        parsed["proposal_history"] = replies
        return parsed
    recovery = recovery_option(observation)
    if recovery:
        recovered = normalize_pool({"options": [recovery], "selected_option_id": recovery["option_id"]}, observation, validator)
        records.extend(recovered.get("rejection_records") or [])
        if recovered.get("ok"):
            recovered.update(raw=copy.deepcopy(raw), raw_digest=digest(raw), proposal_history=replies,
                             rejection_records=records, selection_source="runtime_bound_repair_recovery")
            return recovered
    for kind, limit in (("repair_failed_options", config.option_repair_calls_per_round),
                        ("replace_unsupported_options", config.replacement_calls_per_round)):
        if not limit:
            continue
        request = {**observation, "option_repair_request": {"kind": kind,
            "failed_options": [r for r in records if r.get("status") == "rejected"],
            "valid_options": [], "instruction": "Return only corrected independent failed options or distinct supported replacements. Keep original IDs for repairs. Do not change approved scientific interventions."}}
        raw = backend(request)
        replies.append(copy.deepcopy(raw))
        parsed = normalize_pool(raw, observation, validator)
        records.extend(parsed.get("rejection_records") or [])
        if parsed.get("ok"):
            parsed.update(proposal_history=replies, rejection_records=records)
            return parsed
    parsed.update(proposal_history=replies, rejection_records=records, bounded_supplement_exhausted=True)
    return parsed
