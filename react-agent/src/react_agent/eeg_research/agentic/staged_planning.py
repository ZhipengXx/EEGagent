"""Generate alternatives, then ask a separate Critic before admitting any work."""
from __future__ import annotations

import copy
import json

from .plan_protocol import (freeze_proposal_pool, normalize_critique,
                            normalize_proposal_pool, proposal_cost, round_id)
from .planning_context import (build_critic_input, build_proposer_input,
                               delivered_observation_ids)


def _response_feedback(parsed, *, role):
    """Only current paths and reasons; input values and native logs stay local."""
    errors = parsed.get("errors") or []
    if not errors:
        errors = [{"loc": [], "type": "proposal_invalid", "msg": row["error"]}
                  for row in parsed.get("validation_records") or [] if row.get("error")]
    if not errors:
        errors = [{"loc": ["proposals"], "type": "pool_incomplete",
                   "msg": "Return enough distinct valid proposals to complete the pool."}]
    return {"kind": role + "_output_invalid", "errors": [
        {"path": "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1")
                                    for part in error.get("loc") or []),
         "type": error.get("type", "invalid"), "message": str(error.get("msg", "invalid"))}
        for error in errors],
        "instruction": "Return the complete corrected JSON matching OUTPUT_SCHEMA. Fix the listed errors and preserve unaffected scientific choices."}


def _delivered_references(proposer_input, full):
    """Use supplied material identities; a stored-but-undelivered artifact is not a citation."""
    from .context.representations import delivered_source_refs
    refs = delivered_source_refs(proposer_input)
    verified = {row.get("artifact_id") for row in full.get("artifact_index") or []
                if row.get("verification_status") == "verified" and row.get("artifact_id")}
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"previous_response", "previous_response_error", "plan_repair_request", "critic_repair_request"}:
                    continue
                if key in {"artifact_id", "analysis_artifact_ref", "source_ref", "evidence_refs", "source_refs"}:
                    values = child if isinstance(child, list) else [child]
                    refs.update(item for item in values if isinstance(item, str) and item in verified)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(proposer_input)
    return sorted(refs)


def _publish_assembly(camp, full, parsed, raw, stage):
    """Derived native proposals stay separate from immutable provider responses."""
    from pathlib import Path
    from .action_planning import digest
    from .loop import _read, _write

    rid = round_id(full)
    record = {"schema_version": "eeg_research.planning_assembly.v2", "round_id": rid,
        "stage": stage, "operation_id": rid + ":" + stage,
        "raw_reply_digest": digest(raw), "contract_digest": digest(full["_planning_contract"]),
        "proposals": copy.deepcopy(parsed.get("valid_proposals") or []),
        "errors": copy.deepcopy(parsed.get("errors") or []),
        "native_errors": copy.deepcopy(parsed.get("native_errors") or [])}
    path = Path(camp) / "research_round_assemblies" / rid / (stage + ".json")
    if path.is_file():
        if _read(path) != record:
            raise ValueError("paid_staged_round_assembly_changed")
    else:
        _write(path, record, exclusive=True)


def _correction_refused(exc):
    return str(exc) in {"planner_round_attempt_limit", "api_operation_attempt_limit", "budget_exhausted",
                        "owner_request_unavailable:llm_calls", "planning_stage_attempt_limit",
                        "planning_role_attempts_exhausted"}


def decide_staged_plan(camp, state, full, delivered, planner, critic, planning_owner,
                       limits, cfg, *, recovered_stages=None, recovery_proof=None):
    from .owner_budget import preview_selected_plan
    from .research_execution import role_api_attempt_limit
    from .staged_planning_recovery import (CRITIC_STAGES, PLANNER_STAGES,
                                           publish_staged_input, publish_staged_raw)
    from .planning_contract import freeze_planning_contract
    if "_planning_contract" not in full:
        full["_planning_contract"] = freeze_planning_contract(full)
    stages = copy.deepcopy(recovered_stages or {})
    rid = round_id(full)
    if full["_planning_contract"].get("native_errors"):
        return {"ok": False, "detail": "planning_native_input_invalid", "valid_proposals": [],
                "errors": [], "native_errors": copy.deepcopy(full["_planning_contract"]["native_errors"]),
                "validation_records": [], "proposal_history": list(stages.values())}

    def request(stage, backend, payload, *, native_pool=None, critic_context=None):
        if stage in stages:
            return copy.deepcopy(stages[stage])
        frozen = ((recovery_proof or {}).get("stage_payloads") or {}).get(stage)
        if frozen is not None:
            payload = copy.deepcopy(frozen)
        bind = getattr(backend, "bind", None)
        if callable(bind):
            bind(operation_id=rid + ":" + stage)
        publish_staged_input(camp, state, full, planning_owner, payload, stage,
                             native_pool=native_pool, critic_context=critic_context)
        raw = (backend(payload, critic_context=critic_context) if critic_context is not None
               else backend(payload))
        publish_staged_raw(camp, state, full, planning_owner, raw, stage=stage)
        stages[stage] = copy.deepcopy(raw)
        return raw

    proposer_input = build_proposer_input(full, delivered)
    full["_delivered_context_refs"] = _delivered_references(proposer_input, full)
    # Snapshots keep ``full`` exact; only validation sees what was delivered.
    checked = {**full, "_delivered_observation_ids": delivered_observation_ids(proposer_input)}
    raw = request("generation", planner, proposer_input)
    parsed = normalize_proposal_pool(raw, checked)
    _publish_assembly(camp, full, parsed, raw, "generation")
    if parsed.get("native_errors"):
        return {**parsed, "proposal_history": list(stages.values())}
    goal = full.get("goal") or full
    planner_limit = min(cfg.max_planner_api_attempts, role_api_attempt_limit(goal, "research_planner"))
    planner_corrections = min(cfg.proposal_correction_calls, planner_limit - 1)
    for stage in PLANNER_STAGES[1:1 + planner_corrections]:
        if parsed["ok"]:
            break
        from .planning_contract import scientific_proposal_view
        repair = {**proposer_input, "previous_response": json.dumps(raw, ensure_ascii=False),
            "previous_response_error": _response_feedback(parsed, role="planner"),
            "plan_repair_request": {"kind": "correct_research_proposals",
                "valid_proposals": [scientific_proposal_view(proposal, full)
                                    for proposal in parsed.get("valid_proposals") or []],
                "remaining_slots": max(0, 4 - len(parsed.get("valid_proposals") or [])),
                "instruction": "Valid proposals are already retained locally. Return only replacement or additional distinct proposals; do not repeat or modify retained proposals, rank, evaluate or select."}}
        from .llm import LlmUnavailable
        try:
            raw = request(stage, planner, repair)
        except LlmUnavailable as exc:
            if not _correction_refused(exc):
                raise
            return {**parsed, "proposal_history": list(stages.values()),
                    "correction_unavailable": {"reason": str(exc), "status": "not_sent"}}
        parsed = normalize_proposal_pool(raw, checked, existing=parsed.get("valid_proposals"), stage=stage)
        _publish_assembly(camp, full, parsed, raw, stage)
        if parsed.get("native_errors"):
            return {**parsed, "proposal_history": list(stages.values())}
    if not parsed["ok"]:
        return {**parsed, "proposal_history": list(stages.values())}

    pool = parsed["pool"]
    feasible, rejected = [], []
    for proposal in pool["proposals"]:
        preview = preview_selected_plan(camp, task_kind=proposal["task_kind"],
            known_cost=proposal_cost(proposal, full)["known"], recovery_limits=limits)
        if preview.get("ok"):
            feasible.append(proposal)
        else:
            rejected.append({"proposal_id": proposal["proposal_id"], "status": "rejected",
                "error": preview.get("reason"), "stage": "complete_task_budget"})
    pool = freeze_proposal_pool(feasible, checked,
        validation_records=pool["validation_records"] + rejected)
    if pool["pool_status"] != "complete":
        return {"ok": False, "detail": "research_pool_" + pool["pool_status"],
            "validation_records": pool["validation_records"], "proposal_history": list(stages.values())}

    from .critic_contract import freeze_critic_context, verify_critic_context
    proof_inputs = (recovery_proof or {}).get("stage_inputs") or {}
    critic_context = None
    for stage in CRITIC_STAGES:
        original = proof_inputs.get(stage)
        if original is not None and (original.get("proposal_pool_digest") != pool["pool_digest"]
                or original.get("proposal_pool") != pool):
            raise ValueError("paid_critic_original_pool_changed")
        if original is not None:
            frozen_context = original.get("critic_context")
            if not isinstance(frozen_context, dict):
                raise ValueError("paid_critic_original_context_missing")
            if critic_context is not None and critic_context != frozen_context:
                raise ValueError("paid_critic_original_context_changed")
            critic_context = copy.deepcopy(frozen_context)
    if critic_context is None:
        critic_context = freeze_critic_context(pool, proposer_input)
    native_errors = verify_critic_context(critic_context, pool)
    if native_errors:
        return {"ok": False, "detail": "critic_native_input_invalid", "native_errors": native_errors,
                "validation_records": [], "proposal_history": list(stages.values())}
    critic_input = build_critic_input(full, proposer_input, pool, critic_context=critic_context)
    raw = request("critic", critic, critic_input, native_pool=pool, critic_context=critic_context)
    result = normalize_critique(raw, pool, checked, critic_context=critic_context)
    if result.get("native_errors"):
        return {**result, "proposal_history": list(stages.values())}
    critic_limit = min(cfg.max_critic_api_attempts, role_api_attempt_limit(goal, "research_critic"))
    for stage in CRITIC_STAGES[1:critic_limit]:
        if result["ok"]:
            break
        repair = {**critic_input,
            "critic_repair_request": {
                "instruction": "Return a complete ordered evaluations list, each frozen proposal_ref exactly once. The first item is preferred. Fix the listed errors without altering or inventing proposals; numeric ranks and selected IDs are filled locally."},
            "previous_response": json.dumps(raw, ensure_ascii=False),
            "previous_response_error": _response_feedback(result, role="critic")}
        from .llm import LlmUnavailable
        try:
            raw = request(stage, critic, repair, native_pool=pool,
                          critic_context=critic_context)
        except LlmUnavailable as exc:
            if not _correction_refused(exc):
                raise
            return {**result, "proposal_history": list(stages.values()),
                    "correction_unavailable": {"reason": str(exc), "status": "not_sent"}}
        result = normalize_critique(raw, pool, checked, critic_context=critic_context)
        if result.get("native_errors"):
            return {**result, "proposal_history": list(stages.values())}
    result["proposal_history"] = list(stages.values())
    if result.get("round"):
        result["round"].update(planning_mode="propose_then_critic",
            proposal_pool_digest=pool["pool_digest"],
            planning_stages=list(stages))
    return result
