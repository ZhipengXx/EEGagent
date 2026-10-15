"""Native selected-task feedback counts, final indexing and requested stop."""
from __future__ import annotations

import pytest

from react_agent.eeg_research.agentic import loop, owner_budget, selected_plan
from react_agent.eeg_research.agentic.iteration_control import maybe_complete_research
from react_agent.eeg_research.agentic.schemas import SelectedPlan


def campaign(tmp_path, limit=2):
    cfg = {"version": "single_plan_v1", "paid_maintenance": "off"}
    if limit is not None:
        cfg["max_completed_rounds"] = limit
    goal = {"goal_id": "native_iterations", "max_training_jobs": 2,
        "max_llm_calls": 100, "max_gpu_seconds": 100, "research_execution": cfg}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "iteration_contract",
        "task_type": "eeg_image_retrieval"}, request_id="native_iterations")
    return tmp_path / "native_iterations"


def terminal(camp, number, status="observation_ready"):
    pid = "plan_" + str(number).zfill(32)
    proposal = {"proposal_id": "diagnose" + str(number), "task_kind": "diagnosis",
        "target_candidate_ref": "baseline", "question": "Inspect feedback " + str(number),
        "task_description": "Inspect existing evidence", "scientific_intent": "Explain the native outcome",
        "evaluation_scope": {"phase": "diagnostic"}, "expected_observation": "Native feedback",
        "completion_condition": "A terminal diagnostic card", "required_context_refs": [],
        "value_rationale": "Resolve an uncertainty"}
    owner = owner_budget.allocate_selected_plan(camp, plan_id=pid, task_kind="diagnosis",
        known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits={})
    plan = SelectedPlan(plan_id=pid, originating_round="round_" + str(number),
        originating_proposal=proposal["proposal_id"], proposal=proposal,
        scientific_intent_digest="synthetic_intent", candidate_id="baseline",
        budget_owner=owner["owner_id"], recovery_limits={}).model_dump()
    selected_plan._receipt(camp, plan, "selected", proposal_id=proposal["proposal_id"],
        budget_owner=owner["owner_id"], pool_status="complete")
    state = loop.load_state(camp)
    selected_plan.save_plan(camp, state, plan)
    return selected_plan._finish(camp, state, plan, status, detail="synthetic failure" if status == "failed" else None)


def protected(camp):
    return {str(p.relative_to(camp)): p.read_bytes() for folder in ("selected_plans", "experiment_observations")
        for p in (camp / folder).rglob("*") if p.is_file()}


def test_native_terminal_iterations_stop_without_another_planner_call(tmp_path):
    camp = campaign(tmp_path)
    first = terminal(camp, 1)
    assert not maybe_complete_research(camp, first)
    # Repairs and numerous internal steps do not produce extra iterations.
    first["step"] = 1000
    assert not maybe_complete_research(camp, first)
    final = terminal(camp, 2, "failed")
    original = protected(camp)
    claims_before = owner_budget.snapshot(camp)["global_spent_and_occupied"]
    assert maybe_complete_research(camp, final)
    assert final["status"] == "finished" and final["completed_research_rounds"] == 2
    report = loop._read(camp / "research_completion.json")
    assert report["completed_rounds"] == 2
    assert [row["status"] for row in report["iterations"]] == ["observation_ready", "failed"]
    assert protected(camp) == original
    assert owner_budget.snapshot(camp)["global_spent_and_occupied"] == claims_before
    assert not (camp / "llm_api_intents.jsonl").exists()
    def forbidden(*args, **kwargs):
        pytest.fail("Iteration completion may not request or dispatch another task")
    assert loop.tick(camp, forbidden, services={"launch": forbidden})["status"] == "finished"
    before = (camp / "research_completion.json").read_bytes()
    assert maybe_complete_research(camp, loop.load_state(camp))
    assert (camp / "research_completion.json").read_bytes() == before


@pytest.mark.parametrize("inflight", [{"live_job": "job"}, {"active_suite": "suite"}, {"queued": [["baseline", "full"]]}])
def test_limit_never_terminates_live_native_work(tmp_path, inflight):
    camp = campaign(tmp_path, 1)
    state = terminal(camp, 1)
    state.update(inflight)
    assert not maybe_complete_research(camp, state)
    assert not (camp / "research_completion.json").exists()


def test_missing_limit_keeps_existing_campaign_behavior(tmp_path):
    camp = campaign(tmp_path, None)
    assert not maybe_complete_research(camp, terminal(camp, 1))


def test_final_memory_fault_retains_task_and_resumes_same_completion(tmp_path, monkeypatch):
    camp = campaign(tmp_path, 1)
    state = terminal(camp, 1)
    goal = loop._read(camp / "goal.json")
    goal["memory"] = {"enabled": True}
    loop._write(camp / "goal.json", goal)
    original = protected(camp)
    queries = []
    def retrieve(_camp, **kwargs):
        queries.append(kwargs)
        return {"degraded": True, "reason": "model unavailable"}
    monkeypatch.setattr("react_agent.eeg_research.agentic.semantic_memory.retrieve", retrieve)
    with pytest.raises(ValueError, match="final_memory_unavailable"):
        maybe_complete_research(camp, state)
    assert state["status"] != "finished" and not (camp / "research_completion.json").exists()
    monkeypatch.setattr("react_agent.eeg_research.agentic.semantic_memory.retrieve",
        lambda *_args, **kwargs: {"backend": "minilm", "degraded": False,
            "indexing": {"indexed": 1}, "fingerprint": "synthetic_backend_fixture"})
    assert maybe_complete_research(camp, state)
    assert queries[0]["require_embedding"] is True
    assert loop._read(camp / "research_completion.json")["final_memory"]["indexing"] == {"indexed": 1}
    assert protected(camp) == original


def test_native_final_embedding_exception_blocks_and_preserves_original_owner(tmp_path, monkeypatch):
    from react_agent.eeg_research.agentic.embedding import EmbeddingUnavailable
    camp = campaign(tmp_path, 1)
    state = terminal(camp, 1)
    goal = loop._read(camp / "goal.json")
    goal["memory"] = {"enabled": True}
    loop._write(camp / "goal.json", goal)
    original = protected(camp)
    continuation = state["selected_plan"]["continuation_owner"]
    before = owner_budget.snapshot(camp, owner_id=continuation)
    def unavailable(*args, **kwargs):
        raise EmbeddingUnavailable("pinned_model_missing")
    monkeypatch.setattr("react_agent.eeg_research.agentic.semantic_memory.retrieve", unavailable)
    def forbidden(*args, **kwargs):
        pytest.fail("A final embedding failure may not purchase another request")
    failed = loop.tick(camp, forbidden)
    assert failed["status"] == "blocked" and failed["failure"]["recoverable"]
    assert "final_memory_unavailable" in failed["detail"]
    assert owner_budget.snapshot(camp, owner_id=continuation) == before
    assert protected(camp) == original
    assert not (camp / "research_completion.json").exists()


def test_native_tick_stops_at_the_tenth_observation_before_any_eleventh_selection(tmp_path):
    camp = campaign(tmp_path, 10)
    for number in range(1, 11):
        state = terminal(camp, number)
    original = protected(camp)
    def forbidden(*args, **kwargs):
        pytest.fail("The eleventh Planner or task may not run")
    result = loop.tick(camp, forbidden, services={"launch": forbidden})
    assert result["status"] == "finished" and result["completed_research_rounds"] == 10
    assert len(list((camp / "selected_plans").glob("plan_*.json"))) == 10
    assert protected(camp) == original
    assert not (camp / "llm_api_intents.jsonl").exists()


def test_unreadable_terminal_card_is_not_a_completed_iteration(tmp_path):
    camp = campaign(tmp_path, 1)
    state = terminal(camp, 1)
    plan = selected_plan.load_plan(camp, state)
    plan["terminal_observation_digest"] = "wrong digest"
    selected_plan.save_plan(camp, state, plan)
    with pytest.raises(ValueError, match="terminal_identity_changed"):
        maybe_complete_research(camp, state)
    assert not (camp / "research_completion.json").exists()
