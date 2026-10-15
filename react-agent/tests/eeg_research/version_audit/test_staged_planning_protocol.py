"""Independent planning comparison; local fixtures only, no paid calls or GPU."""
from __future__ import annotations

import copy
from unittest.mock import Mock

import pytest

from react_agent.eeg_research.agentic import loop, owner_budget
from react_agent.eeg_research.agentic.plan_protocol import (
    freeze_proposal_pool, normalize_critique, normalize_proposal_pool, proposal_pool_digest,
)


def observation(**changes):
    return {"goal": {"research_execution": {"version": "single_plan_v1", "target_k": 3,
                "planning_mode": "propose_then_critic"}},
        "available_actions": ["design_experiment"], "evidence": [], "artifact_index": [], **changes}


def proposal(pid):
    return {"proposal_id": pid, "task_kind": "new_method", "target_candidate_ref": None,
        "question": "Compare mechanism " + pid, "task_description": "One bounded study " + pid,
        "scientific_intent": "Independent mechanism " + pid, "evaluation_scope": {"phase": "full", "seed": 0},
        "expected_observation": "Matched measured feedback", "completion_condition": "One observation",
        "required_context_refs": [], "value_rationale": "Discriminate current uncertainty"}


def proposal_draft(*ids):
    return {"protocol_version": "research_proposals.v1", "proposals": [proposal(pid) for pid in ids]}


def pool_and_observation(*ids):
    obs = observation()
    parsed = normalize_proposal_pool(proposal_draft(*ids), obs)
    assert parsed["ok"]
    return parsed["pool"], obs


def critique(pool, ranks=None, selected=None):
    ranks = ranks or [(row["proposal_id"], rank) for rank, row in enumerate(pool["proposals"], 1)]
    winner = next(pid for pid, rank in ranks if rank == 1)
    return {"protocol_version": "research_plan_critique.v1", "proposal_pool_digest": pool["pool_digest"],
        "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Independent comparison " + pid,
                         "implementation_risk": "Bounded engineering uncertainty"} for pid, rank in ranks],
        "selected_proposal_id": selected or winner, "selection_rationale": "Critic selects " + (selected or winner)}


def test_proposer_has_no_self_evaluation_or_selection_authority():
    raw = proposal_draft("A", "B", "C")
    parsed = normalize_proposal_pool(raw, observation())
    assert parsed["ok"] and parsed["pool"]["valid_count"] == 3
    assert set(parsed["pool"]["native_cost_by_id"]) == {"A", "B", "C"}
    assert parsed["pool"]["pool_digest"] == proposal_pool_digest(parsed["pool"])
    assert "evaluations" not in parsed["pool"] and "selected_proposal_id" not in parsed["pool"]
    for field, value in (("evaluations", []), ("selected_proposal_id", "A"), ("selection_rationale", "Pick A")):
        rejected = normalize_proposal_pool({**raw, field: value}, observation())
        assert not rejected["ok"] and rejected["detail"] == "research_proposal_schema_invalid"


def test_native_rejections_and_deduplication_precede_critic():
    invalid = {**proposal("bad"), "target_candidate_ref": "invented"}
    duplicate = {**proposal("A"), "proposal_id": "copy-A"}
    raw = {"protocol_version": "research_proposals.v1", "proposals": [proposal("A"), duplicate, invalid, proposal("B")]}
    parsed = normalize_proposal_pool(raw, observation())
    assert parsed["ok"] and [row["proposal_id"] for row in parsed["pool"]["proposals"]] == ["A", "B"]
    assert parsed["pool"]["shortfall_reason"]
    errors = [row.get("error", "") for row in parsed["pool"]["rejected"]]
    assert "duplicate_research_task" in errors and any("new_method_cannot" in error for error in errors)


def test_valid_alternatives_survive_one_proposal_correction():
    original = normalize_proposal_pool(proposal_draft("A", "B"), observation())
    corrected = normalize_proposal_pool(proposal_draft("B", "C"), observation(), existing=original["valid_proposals"])
    assert corrected["ok"]
    assert [row["proposal_id"] for row in corrected["pool"]["proposals"]] == ["A", "B", "C"]
    assert any(row.get("error") == "duplicate_research_task" for row in corrected["validation_records"])


def test_critic_cannot_choose_from_a_singleton_even_if_legacy_singleton_enabled():
    obs = observation(goal={"research_execution": {"version": "single_plan_v1", "allow_degraded_singleton": True}})
    frozen = freeze_proposal_pool([proposal("A")], obs)
    assert frozen["pool_status"] == "incomplete"
    assert not normalize_critique(critique(frozen), frozen, obs)["ok"]


def test_pre_critic_budget_filter_freezes_a_different_exact_pool():
    original, obs = pool_and_observation("A", "B", "C")
    rejection = {"proposal_id": "A", "status": "rejected", "stage": "complete_task_budget", "error": "task_too_expensive"}
    feasible = freeze_proposal_pool(original["proposals"][1:], obs,
        validation_records=[*original["validation_records"], rejection])
    assert feasible["pool_digest"] != original["pool_digest"]
    assert rejection in feasible["rejected"]
    stale = normalize_critique(critique(original), feasible, obs)
    assert not stale["ok"] and stale["detail"] == "research_critique_pool_digest_mismatch"
    current = normalize_critique(critique(feasible), feasible, obs)
    assert current["ok"] and current["round"]["valid_count"] == 2


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "same_rank", "rank_gap", "non_winner", "unknown_winner"])
def test_critic_requires_complete_unique_comparison_and_its_rank_one_winner(fault):
    pool, obs = pool_and_observation("A", "B", "C")
    raw = critique(pool)
    if fault == "missing":
        raw["evaluations"].pop()
    elif fault == "duplicate":
        raw["evaluations"][2]["proposal_id"] = "B"
    elif fault == "unknown":
        raw["evaluations"][2]["proposal_id"] = "not-in-pool"
    elif fault == "same_rank":
        raw["evaluations"][2]["rank"] = 2
    elif fault == "rank_gap":
        raw["evaluations"][2]["rank"] = 4
    elif fault == "non_winner":
        raw["selected_proposal_id"] = "B"
    else:
        raw["selected_proposal_id"] = "not-in-pool"
    rejected = normalize_critique(raw, pool, obs)
    assert not rejected["ok"] and rejected["selected_proposal"] is None
    assert "round" not in rejected


def test_critic_rationale_and_selection_bind_exactly_without_container_order_fallback():
    pool, obs = pool_and_observation("A", "B", "C")
    raw = critique(pool, [("C", 3), ("A", 2), ("B", 1)])
    result = normalize_critique(raw, pool, obs)
    assert result["ok"] and result["selected_proposal"]["proposal_id"] == "B"
    record = result["round"]
    assert record["selected_proposal_id"] == "B" and record["selection_rationale"] == "Critic selects B"
    assert record["selection_source"] == "independent_critic_selection"
    assert record["planning_mode"] == "propose_then_critic"
    assert record["proposal_pool_digest"] == pool["pool_digest"]
    assert record["evaluation_count"] == record["valid_count"] == 3
    result["selected_proposal"]["question"] = "changed result only"
    assert pool["proposals"][1]["question"] != "changed result only"
    assert raw["selected_proposal_id"] == "B"


@pytest.mark.parametrize("fault", ["proposal", "native_cost", "round", "observation", "rehash_cost"])
def test_critic_rejects_changed_frozen_pool_or_current_observation(fault):
    pool, obs = pool_and_observation("A", "B")
    raw = critique(pool)
    changed = copy.deepcopy(pool)
    if fault == "proposal":
        changed["proposals"][0]["scientific_intent"] = "Unreviewed intervention"
    elif fault in {"native_cost", "rehash_cost"}:
        changed["native_cost_by_id"]["A"]["known"]["training_jobs"] = 999
        if fault == "rehash_cost":
            changed["pool_digest"] = proposal_pool_digest(changed)
            raw["proposal_pool_digest"] = changed["pool_digest"]
    elif fault == "round":
        changed["round_id"] = "another-round"
    else:
        obs["goal"]["question"] = "Another scientific question"
    result = normalize_critique(raw, changed, obs)
    assert not result["ok"] and result["detail"] == "research_proposal_pool_identity_invalid"


def test_native_campaign_planning_functions_cannot_create_work_for_any_alternative(tmp_path, monkeypatch):
    """Run pure staged protocol against a native-created isolated campaign."""
    goal = {"goal_id": "staged_comparison", "max_llm_calls": 20, "max_training_jobs": 2,
        "max_gpu_seconds": 100, "max_candidates": 2,
        "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic", "target_k": 3}}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "staged_contract",
        "task_type": "eeg_image_retrieval"}, request_id="staged_comparison")
    camp = tmp_path / "staged_comparison"
    obs = loop.observation(camp)
    before = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()}
    allocate, native = Mock(side_effect=AssertionError("premature allocation")), Mock(side_effect=AssertionError("premature code"))
    monkeypatch.setattr(owner_budget, "allocate_selected_plan", allocate)
    monkeypatch.setattr(loop, "_apply_action", native)
    parsed = normalize_proposal_pool(proposal_draft("A", "B", "C"), obs)
    assert parsed["ok"]
    result = normalize_critique(critique(parsed["pool"], [("A", 2), ("B", 1), ("C", 3)]), parsed["pool"], obs)
    assert result["ok"] and result["selected_proposal"]["proposal_id"] == "B"
    allocate.assert_not_called()
    native.assert_not_called()
    assert {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()} == before
    assert not list((camp / "jobs").glob("*/launch_intent.json"))
    assert not list((camp / "selected_plans").glob("plan_*.json"))
    assert not (camp / "llm_calls.jsonl").exists()
