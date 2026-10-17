"""Generate alternatives, then ask a separate Critic before admitting any work."""
from __future__ import annotations

import copy

from .plan_protocol import (freeze_proposal_pool, normalize_critique,
                            normalize_proposal_pool, proposal_cost, round_id)
from .planning_context import (build_critic_input, build_proposer_input,
                               delivered_observation_ids, minimal_validation_feedback)


def decide_staged_plan(camp, state, full, delivered, planner, critic, planning_owner,
                       limits, cfg, *, recovered_stages=None, recovery_proof=None):
    from .owner_budget import preview_selected_plan
    from .staged_planning_recovery import publish_staged_input, publish_staged_raw
    stages = copy.deepcopy(recovered_stages or {})
    rid = round_id(full)

    def request(stage, backend, payload, *, native_pool=None):
        if stage in stages:
            return copy.deepcopy(stages[stage])
        bind = getattr(backend, "bind", None)
        if callable(bind):
            bind(operation_id=rid + ":" + stage)
        publish_staged_input(camp, state, full, planning_owner, payload, stage,
                             native_pool=native_pool)
        raw = backend(payload)
        publish_staged_raw(camp, state, full, planning_owner, raw, stage=stage)
        stages[stage] = copy.deepcopy(raw)
        return raw

    proposer_input = build_proposer_input(full, delivered)
    # Snapshots keep ``full`` exact; only validation sees what was delivered.
    checked = {**full, "_delivered_observation_ids": delivered_observation_ids(proposer_input)}
    raw = request("generation", planner, proposer_input)
    parsed = normalize_proposal_pool(raw, checked)
    if not parsed["ok"] and cfg.proposal_correction_calls:
        raw = request("correction", planner, {**proposer_input,
            "plan_repair_request": {"kind": "correct_research_proposals",
                "valid_proposals": parsed.get("valid_proposals") or [],
                "rejected": minimal_validation_feedback(parsed.get("validation_records") or []),
                "instruction": "Keep the valid proposals above; return corrected or additional distinct proposals to complete the pool. Do not evaluate, rank or select them."}})
        parsed = normalize_proposal_pool(raw, checked, existing=parsed.get("valid_proposals"))
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

    proof_inputs = (recovery_proof or {}).get("stage_inputs") or {}
    for stage in ("critic", "critic_correction"):
        original = proof_inputs.get(stage)
        if original is not None and (original.get("proposal_pool_digest") != pool["pool_digest"]
                or original.get("proposal_pool") != pool):
            raise ValueError("paid_critic_original_pool_changed")
    critic_input = build_critic_input(full, proposer_input, pool)
    raw = request("critic", critic, critic_input, native_pool=pool)
    result = normalize_critique(raw, pool, checked)
    if not result["ok"] and cfg.max_critic_api_attempts > 1:
        raw = request("critic_correction", critic, {**critic_input,
            "critic_repair_request": {"rejected": minimal_validation_feedback(result["validation_records"]),
                "instruction": "Evaluate every frozen proposal exactly once, give unique ranks 1..N, select rank one, and explain that exact choice. Do not alter or invent proposals."}},
            native_pool=pool)
        result = normalize_critique(raw, pool, checked)
    result["proposal_history"] = list(stages.values())
    if result.get("round"):
        result["round"].update(planning_mode="propose_then_critic",
            proposal_pool_digest=pool["pool_digest"],
            planning_stages=list(stages))
    return result
