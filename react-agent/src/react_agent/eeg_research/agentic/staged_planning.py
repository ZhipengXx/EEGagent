"""Generate alternatives, then ask a separate Critic before admitting any work."""
from __future__ import annotations

import copy

from .plan_protocol import (freeze_proposal_pool, normalize_critique,
                            normalize_proposal_pool, proposal_cost, round_id)


def decide_staged_plan(camp, state, full, delivered, planner, critic, planning_owner,
                       limits, cfg, *, recovered_stages=None, recovery_proof=None):
    from .owner_budget import preview_selected_plan
    from .staged_planning_recovery import publish_staged_raw
    stages = copy.deepcopy(recovered_stages or {})
    rid = round_id(full)

    def request(stage, backend, payload):
        if stage in stages:
            return copy.deepcopy(stages[stage])
        bind = getattr(backend, "bind", None)
        if callable(bind):
            bind(operation_id=rid + ":" + stage)
        raw = backend(payload)
        publish_staged_raw(camp, state, full, planning_owner, raw, stage=stage)
        stages[stage] = copy.deepcopy(raw)
        return raw

    raw = request("generation", planner, {**delivered, "planning_stage": "propose"})
    parsed = normalize_proposal_pool(raw, full)
    if not parsed["ok"] and cfg.proposal_correction_calls:
        raw = request("correction", planner, {**delivered, "planning_stage": "propose",
            "plan_repair_request": {"kind": "correct_research_proposals",
                "valid_proposals": parsed.get("valid_proposals") or [],
                "rejected": parsed.get("validation_records") or [],
                "instruction": "Return 2–4 distinct valid research proposals only. Do not evaluate, rank or select them."}})
        parsed = normalize_proposal_pool(raw, full, existing=parsed.get("valid_proposals"))
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
    pool = freeze_proposal_pool(feasible, full,
        validation_records=pool["validation_records"] + rejected)
    if pool["pool_status"] != "complete":
        return {"ok": False, "detail": "research_pool_" + pool["pool_status"],
            "validation_records": pool["validation_records"], "proposal_history": list(stages.values())}

    proof_inputs = (recovery_proof or {}).get("stage_inputs") or {}
    for stage in ("critic", "critic_correction"):
        original = proof_inputs.get(stage)
        if original is not None and (original.get("proposal_pool_digest") != pool["pool_digest"]
                or original.get("proposal_pool") != pool):
            raise ValueError("paid_critic_original_pool_changed")
    critic_input = {**delivered, "planning_stage": "critic", "proposal_pool": pool,
        "proposal_pool_digest": pool["pool_digest"]}
    raw = request("critic", critic, critic_input)
    result = normalize_critique(raw, pool, full)
    if not result["ok"] and cfg.max_critic_api_attempts > 1:
        raw = request("critic_correction", critic, {**critic_input,
            "critic_repair_request": {"rejected": result["validation_records"],
                "instruction": "Evaluate every frozen proposal exactly once, give unique ranks 1..N, select rank one, and explain that exact choice. Do not alter or invent proposals."}})
        result = normalize_critique(raw, pool, full)
    result["proposal_history"] = list(stages.values())
    if result.get("round"):
        result["round"].update(planning_mode="propose_then_critic",
            proposal_pool_digest=pool["pool_digest"],
            planning_stages=list(stages))
    return result
