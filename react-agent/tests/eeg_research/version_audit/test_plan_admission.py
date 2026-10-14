"""Public synthetic P1-2/P1-3 regressions; no campaign replay or API/GPU."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from react_agent.eeg_research.agentic import loop, selected_plan
from react_agent.eeg_research.agentic.plan_protocol import (
    decide_research_plan, measurement_action, normalize_research_round, proposal_cost,
)
from react_agent.eeg_research.agentic.schemas import ResearchPlanProposal


def proposal(pid, kind="new_method", phase="full", target=None):
    return {"proposal_id": pid, "task_kind": kind, "question": "Question " + pid,
        "task_description": "Task " + pid, "target_candidate_ref": target,
        "scientific_intent": "Independent mechanism " + pid,
        "evaluation_scope": {"phase": phase, "seed": 0},
        "expected_observation": "Native measurement", "completion_condition": "One observation",
        "required_context_refs": [], "value_rationale": "Resolve uncertainty"}


def draft(proposals, evaluated=None):
    ids = [row["proposal_id"] for row in proposals] if evaluated is None else evaluated
    return {"protocol_version": "research_plan_draft.v1", "proposals": proposals,
        "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Information",
                         "implementation_risk": "Unknown"} for rank, pid in enumerate(ids, 1)],
        "selected_proposal_id": ids[0] if ids else None, "selection_rationale": "Explicit comparison"}


def observation(**updates):
    return {"goal": {"research_execution": {"version": "single_plan_v1", "target_k": 2}},
        "available_actions": ["design_experiment"], "candidates": [{"candidate_id": "old"}],
        "eligible_targets": {"run_full": ["baseline", "old"], "run_pilot": ["old"],
                             "run_screening": ["old"], "run_confirmation": ["old"]},
        "evidence": [], "artifact_index": [], **updates}


def test_partial_evaluation_is_incomplete_and_corrected_once():
    raw = draft([proposal("A"), proposal("B")], ["A"])
    result = normalize_research_round(raw, observation())
    assert not result["ok"]
    assert result["round"]["pool_status"] == "incomplete"
    assert result["round"]["valid_count"] == 2
    assert result["round"]["evaluation_count"] == 1
    assert result["round"]["unevaluated_proposal_ids"] == ["B"]
    assert result["round"]["selected_proposal_id"] is None
    backend = Mock(return_value=raw)
    still_incomplete = decide_research_plan(observation(), backend)
    assert not still_incomplete["ok"] and backend.call_count == 2
    repair = backend.call_args_list[1].args[0]["plan_repair_request"]
    assert repair["missing_evaluation_ids"] == ["B"]
    assert {row["proposal_id"] for row in repair["valid_proposals"]} == {"A", "B"}


def test_correction_clarifies_invalid_inline_pointer_and_extra_null_without_weakening_validation():
    bad = proposal("A")
    bad["required_context_refs"] = ["/goal"]
    extra = {**proposal("B"), "business_value": None}
    rejected = normalize_research_round(draft([bad, extra]), observation())
    assert not rejected["ok"] and rejected["round"]["valid_count"] == 0
    backend = Mock(side_effect=[draft([bad, extra]), draft([proposal("A"), proposal("B")])])
    corrected = decide_research_plan(observation(), backend)
    assert corrected["ok"] and backend.call_count == 2
    contract = backend.call_args_list[1].args[0]["plan_repair_request"]["reference_contract"]
    assert "required_context_refs=[]" in contract["inline_context"]
    assert contract["verified_artifact_ids"] == contract["delivered_context_source_refs"] == []
    assert any("/goal" in form for form in contract["invalid_reference_forms"])


def test_every_proposal_evaluated_completes_and_correction_can_complete():
    proposals = [proposal("A"), proposal("B")]
    complete = normalize_research_round(draft(proposals), observation())
    assert complete["ok"] and complete["round"]["pool_status"] == "complete"
    assert complete["round"]["evaluation_count"] == complete["round"]["valid_count"] == 2
    assert complete["round"]["unevaluated_proposal_ids"] == []
    backend = Mock(side_effect=[draft(proposals, ["A"]), draft([], ["A", "B"])])
    corrected = decide_research_plan(observation(), backend)
    assert corrected["ok"] and backend.call_count == 2
    assert corrected["round"]["pool_status"] == "complete"


def test_bad_proposal_is_independent_shortfall():
    obs = observation(goal={"research_execution": {"version": "single_plan_v1", "target_k": 3}})
    result = normalize_research_round(draft([proposal("A"), {"proposal_id": "bad"}, proposal("B")]), obs)
    assert result["ok"] and result["round"]["pool_status"] == "complete"
    assert result["round"]["valid_count"] == result["round"]["evaluation_count"] == 2
    assert result["round"]["shortfall_reason"]
    assert {row["proposal_id"] for row in result["valid_proposals"]} == {"A", "B"}
    assert any(row.get("proposal_id") == "bad" and row["status"] == "rejected"
               for row in result["validation_records"])


@pytest.mark.parametrize("error", ["missing", "bad_rank", "conflicting_rank"])
def test_invalid_evaluation_does_not_fill_coverage(error):
    raw = draft([proposal("A"), proposal("B")])
    if error == "missing":
        raw["evaluations"].pop()
    elif error == "bad_rank":
        raw["evaluations"][1]["rank"] = 0
    else:
        raw["evaluations"][1]["rank"] = 1
    result = normalize_research_round(raw, observation())
    assert not result["ok"] and result["round"]["unevaluated_proposal_ids"] == ["B"]


@pytest.mark.parametrize("kind,phase,target", [
    ("new_method", "diagnostic", None), ("new_method", "confirmation_10fold", None),
    ("diagnosis", "full", "old"), ("confirmation", "pilot", "old"),
    ("validation", "confirmation_10fold", "old"), ("validation", "full", None),
])
def test_matrix_rejects_before_cost_or_dispatch(kind, phase, target):
    bad = proposal("bad", kind, phase, target)
    with pytest.raises(ValueError):
        ResearchPlanProposal.model_validate(bad)
    for project in (proposal_cost, measurement_action):
        with pytest.raises(ValueError):
            project(bad, observation())
    result = normalize_research_round(draft([bad, proposal("good")]), observation())
    assert not result["ok"] and result["round"]["valid_count"] == 1


@pytest.mark.parametrize("kind,phase,target,schedule,action,jobs", [
    ("new_method", "full", None, "full_only", "run_full", None),
    ("new_method", "pilot", None, "full_only", "run_pilot", None),
    ("new_method", "screening_2fold", None, "screen_then_confirm", "run_screening", None),
    ("confirmation", "confirmation_10fold", "old", "screen_then_confirm", "run_confirmation", None),
    ("confirmation", "full", "old", "full_only", "run_full", None),
    ("diagnosis", "diagnostic", "old", "full_only", "collect_diagnostics", 0),
    ("validation", "diagnostic", "old", "screen_then_confirm", "collect_diagnostics", 0),
    ("validation", "full", "old", "full_only", "run_full", None),
])
def test_valid_matrix_cost_and_action_agree(kind, phase, target, schedule, action, jobs):
    item = proposal("valid", kind, phase, target)
    obs = observation(evaluation_schedule=schedule)
    ResearchPlanProposal.model_validate(item)
    assert measurement_action(item, obs) == action
    assert proposal_cost(item, obs)["known"]["training_jobs"] == jobs


@pytest.mark.parametrize("step", ["prepare", "design", "implement", "measure"])
@pytest.mark.parametrize("schedule_mismatch", [False, True])
def test_invalid_persisted_plan_cannot_reach_sdk_or_native(tmp_path, monkeypatch, step, schedule_mismatch):
    from openai.resources.chat.completions import AsyncCompletions
    from react_agent.fmri.llm.deepseek import DeepSeekBackend
    sdk, application, native, designer, coder = [Mock(side_effect=AssertionError("unauthorized call")) for _ in range(5)]
    monkeypatch.setattr(AsyncCompletions, "create", sdk)
    monkeypatch.setattr(DeepSeekBackend, "complete_json", application)
    monkeypatch.setattr(loop, "_apply_action", native)
    pid = "plan_" + "a" * 32
    bad = proposal("bad", phase="full" if schedule_mismatch else "diagnostic")
    row = {"plan_id": pid, "originating_round": "synthetic", "originating_proposal": "bad",
        "proposal": bad, "scientific_intent_digest": "synthetic", "budget_owner": "owner_synthetic",
        "recovery_limits": {"max_engineering_repairs": 1}, "current_step": step}
    loop._write(tmp_path / "selected_plans" / (pid + ".json"), row)
    loop._write(tmp_path / "goal.json", {"evaluation_schedule": "screen_then_confirm" if schedule_mismatch else "full_only"})
    state = {"selected_plan": {"plan_id": pid}}
    original = (tmp_path / "selected_plans" / (pid + ".json")).read_bytes()
    with pytest.raises(ValueError, match="task_.*(mismatch|schedule)"):
        selected_plan.tick_selected_plan(tmp_path, state, Mock(), {"designer": designer, "implement": coder})
    assert (tmp_path / "selected_plans" / (pid + ".json")).read_bytes() == original
    for spy in (sdk, application, native, designer, coder):
        spy.assert_not_called()


def select_with_budget_filter(tmp_path, monkeypatch, raw, obs, rejected_cost=20):
    """Only replace owner transport; execute the native selection/pool code."""
    from react_agent.eeg_research.agentic import api_policy, owner_budget, experiment_observations, action_planning
    obs = copy.deepcopy(obs)
    obs.update(_campaign_path=str(tmp_path), contract_fingerprint="synthetic_budget_contract",
               execution_fingerprint="synthetic_budget_execution")
    loop._write(tmp_path / "goal.json", obs["goal"])
    loop._write(tmp_path / "evaluation_contract.json", {"fingerprint": obs["contract_fingerprint"]})
    obs["action_cost_estimates"] = {"by_action_target": {
        "run_full:baseline": {"estimated_cost": {"training_jobs": 1, "gpu_seconds": 10}},
        "run_full:old": {"estimated_cost": {"training_jobs": 1, "gpu_seconds": 20}},
        "run_full:other": {"estimated_cost": {"training_jobs": 1, "gpu_seconds": 30}},
    }}
    monkeypatch.setattr(loop, "observation", lambda _camp: obs)
    monkeypatch.setattr(experiment_observations, "compact_planner_view", lambda _camp, full: full)
    @contextmanager
    def owner(*args, **kwargs):
        yield "synthetic_planner_owner"
    @contextmanager
    def api_round(*args, **kwargs):
        yield
    monkeypatch.setattr(owner_budget, "planner_owner", owner)
    monkeypatch.setattr(api_policy, "planner_api_round", api_round)
    monkeypatch.setattr(owner_budget, "finish_planner_round", lambda *_args: None)
    monkeypatch.setattr(owner_budget, "preview_selected_plan", lambda _camp, **kwargs: {
        "ok": kwargs["known_cost"]["gpu_seconds"] != rejected_cost, "reason": "synthetic_budget_rejected"})
    allocate = Mock(return_value={"owner_id": "synthetic_selected_owner"})
    monkeypatch.setattr(owner_budget, "allocate_selected_plan", allocate)
    monkeypatch.setattr(action_planning, "target_binding", lambda *_args: {})
    monkeypatch.setattr(loop, "_sync_ledger", lambda *_args: None)
    monkeypatch.setattr(loop, "save_state", lambda *_args: None)
    monkeypatch.setattr(loop, "event", lambda *_args, **kwargs: None)
    monkeypatch.setattr(loop, "persist_failure", lambda _camp, state, **kwargs: state.update(failure=kwargs))
    state = {"goal_id": "synthetic", "evidence": [], "candidates": copy.deepcopy(obs["candidates"]),
             "contract_fingerprint": obs["contract_fingerprint"],
             "execution_fingerprint": obs["execution_fingerprint"]}
    selected_plan._select(tmp_path, state, Mock(return_value=raw))
    stored = next((tmp_path / "research_rounds").glob("*.json"))
    return json.loads(stored.read_text()), allocate, state


@pytest.mark.parametrize("allow_singleton,status,allocations", [(False, "incomplete", 0), (True, "degraded_singleton", 1)])
def test_budget_filter_recounts_feasible_evaluated_pool(tmp_path, monkeypatch, allow_singleton, status, allocations):
    items = [proposal("A", "validation", target="baseline"), proposal("B", "validation", target="old")]
    obs = observation(goal={"research_execution": {"version": "single_plan_v1", "target_k": 2,
        "allow_degraded_singleton": allow_singleton}})
    record, allocate, _ = select_with_budget_filter(tmp_path, monkeypatch, draft(items), obs)
    assert record["pool_status"] == status
    assert record["valid_count"] == record["evaluation_count"] == 1
    assert [row["proposal_id"] for row in record["evaluations"]] == ["A"]
    assert record["shortfall_reason"] == "independent_validation_or_complete_task_budget_rejections"
    assert allocate.call_count == allocations


def test_budget_filter_removes_uncompared_rejection_before_pool_status(tmp_path, monkeypatch):
    items = [proposal("A", "validation", target="baseline"), proposal("B", "validation", target="old"),
             proposal("C", "validation", target="other")]
    obs = observation(candidates=[{"candidate_id": "old"}, {"candidate_id": "other"}],
        eligible_targets={"run_full": ["baseline", "old", "other"]})
    record, allocate, _ = select_with_budget_filter(tmp_path, monkeypatch, draft(items, ["A", "C"]), obs)
    assert record["pool_status"] == "complete"
    assert record["valid_count"] == record["evaluation_count"] == 2
    assert record["unevaluated_proposal_ids"] == [] and record["shortfall_reason"] is None
    allocate.assert_called_once()


def test_saved_round_with_missing_evaluation_cannot_reauthorize_selection(tmp_path, monkeypatch):
    items = [proposal("A", "validation", target="baseline"), proposal("B", "validation", target="other")]
    obs = observation(candidates=[{"candidate_id": "other"}], eligible_targets={"run_full": ["baseline", "other"]})
    record, allocate, state = select_with_budget_filter(tmp_path, monkeypatch, draft(items), obs)
    assert record["pool_status"] == "complete"
    record["evaluations"] = record["evaluations"][:1]
    path = next((tmp_path / "research_rounds").glob("*.json"))
    loop._write(path, record)
    allocate.reset_mock()
    backend = Mock(side_effect=AssertionError("saved round cannot buy another draft"))
    selected_plan._select(tmp_path, state, backend)
    latest = state["latest_research_round"]
    assert latest["pool_status"] == "incomplete"
    assert latest["selected_proposal_id"] is None
    assert latest["shortfall_reason"] == "missing_or_nonunique_proposal_evaluations"
    allocate.assert_not_called()
    backend.assert_not_called()


def test_explicit_pilot_only_goal_rejects_full_before_sdk_or_native(tmp_path, monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions
    sdk, native, designer, coder = [Mock(side_effect=AssertionError("unauthorized call")) for _ in range(4)]
    monkeypatch.setattr(AsyncCompletions, "create", sdk)
    monkeypatch.setattr(loop, "_apply_action", native)
    value = {"allowed_training_actions": ["run_pilot"],
        "research_execution": {"version": "single_plan_v1", "target_k": 2}}
    obs = observation(goal=value)
    raw = draft([proposal("A"), proposal("B")])
    backend = Mock(return_value=raw)
    rejected = decide_research_plan(obs, backend)
    assert not rejected["ok"] and backend.call_count == 2
    assert rejected["round"]["valid_count"] == 0
    assert any("task_training_action_outside_goal:run_full" in row.get("error", "")
               for row in rejected["validation_records"])
    pid = "plan_" + "b" * 32
    loop._write(tmp_path / "goal.json", value)
    loop._write(tmp_path / "selected_plans" / (pid + ".json"), {
        "plan_id": pid, "originating_round": "synthetic", "originating_proposal": "A",
        "proposal": proposal("A"), "scientific_intent_digest": "synthetic",
        "budget_owner": "synthetic", "recovery_limits": {}, "current_step": "design"})
    with pytest.raises(ValueError, match="task_training_action_outside_goal:run_full"):
        selected_plan.tick_selected_plan(tmp_path, {"selected_plan": {"plan_id": pid}}, Mock(),
            {"designer": designer, "implement": coder})
    for spy in (sdk, native, designer, coder):
        spy.assert_not_called()


@pytest.mark.parametrize("kind,phase,schedule,actions,expected", [
    ("new_method", "pilot", "full_only", ["run_pilot"], "run_pilot"),
    ("diagnosis", "diagnostic", "full_only", [], "collect_diagnostics"),
    ("validation", "diagnostic", "full_only", [], "collect_diagnostics"),
    ("new_method", "screening_2fold", "screen_then_confirm", ["run_full"], "run_screening"),
    ("confirmation", "confirmation_10fold", "screen_then_confirm", ["run_full"], "run_confirmation"),
])
def test_explicit_training_scope_uses_native_schedule_aliases(kind, phase, schedule, actions, expected):
    item = proposal("A", kind, phase, target=None if kind == "new_method" else "old")
    obs = observation(goal={"allowed_training_actions": actions, "evaluation_schedule": schedule,
        "research_execution": {"version": "single_plan_v1"}})
    assert measurement_action(item, obs) == expected
    assert proposal_cost(item, obs)


def test_declared_task_risk_is_preserved_as_prediction_and_unknown_extras_still_reject():
    a = proposal("A", "diagnosis", "diagnostic", target="old")
    b = proposal("B", "diagnosis", "diagnostic", target="old")
    a["implementation_risk"] = "Low: read-only hash comparison; absent files remain unverified."
    b["implementation_risk"] = "Moderate: bounded native artifact reads may be unavailable."
    raw = draft([a, b])
    original = copy.deepcopy(raw)
    result = normalize_research_round(raw, observation())
    assert result["ok"] and result["round"]["pool_status"] == "complete"
    assert raw == original
    normalized = {row["proposal_id"]: row for row in result["valid_proposals"]}
    assert normalized["A"]["implementation_risk"] == a["implementation_risk"]
    prediction = proposal_cost(normalized["A"], observation())
    assert prediction["predicted"]["implementation_risk"] == {
        "assessment": a["implementation_risk"], "authority": "model_prediction_not_execution_authority"}
    assert prediction["known"]["training_jobs"] == prediction["known"]["gpu_seconds"] == 0
    invalid = copy.deepcopy(raw)
    invalid["proposals"][0]["invented_authority"] = True
    rejected = normalize_research_round(invalid, observation())
    assert not rejected["ok"] and rejected["round"]["valid_count"] == 1
    assert any("invented_authority" in row.get("error", "") for row in rejected["validation_records"])


@pytest.mark.parametrize("risk", [123, "", "X" * 1001])
def test_task_risk_remains_typed_and_bounded(risk):
    item = {**proposal("A"), "implementation_risk": risk}
    with pytest.raises(ValueError):
        ResearchPlanProposal.model_validate(item)


def test_new_optional_risk_preserves_legacy_plan_publication_digest_and_business_null():
    from react_agent.eeg_research.agentic.action_planning import digest
    from react_agent.eeg_research.agentic.schemas import SelectedPlan
    legacy = {**proposal("A"), "cost_assessment": {"usd": None, "parts": [None]}}
    legacy["evaluation_scope"]["optional"] = None
    normalized = ResearchPlanProposal.model_validate(legacy).model_dump()
    assert normalized == legacy and "implementation_risk" not in normalized
    previous_digest = digest(legacy)
    plan = SelectedPlan(plan_id="plan_" + "c" * 32, originating_round="legacy",
        originating_proposal="A", proposal=legacy, scientific_intent_digest="legacy_intent",
        budget_owner="legacy_owner", recovery_limits={}).model_dump()
    projected = SelectedPlan.model_validate(plan).model_dump()
    assert selected_plan._step_binding(plan)["proposal_digest"] == previous_digest
    assert selected_plan._step_binding(projected) == selected_plan._step_binding(plan)
    assert projected["proposal"]["target_candidate_ref"] is None
    assert projected["proposal"]["evaluation_scope"]["optional"] is None
    assert projected["proposal"]["cost_assessment"] == {"usd": None, "parts": [None]}
    explicit_null = ResearchPlanProposal.model_validate({**legacy, "implementation_risk": None}).model_dump()
    assert "implementation_risk" in explicit_null and explicit_null["implementation_risk"] is None
