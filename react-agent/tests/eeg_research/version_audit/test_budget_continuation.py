"""Public owner/continuation/native GPU accounting fixtures; no GPU process."""
from __future__ import annotations

import json
import socket

import dotenv
import pytest

from react_agent.eeg_research.agentic import budget, owner_budget as owners


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    for key in tuple(__import__("os").environ):
        if key.startswith("DEEPSEEK_"):
            monkeypatch.delenv(key, raising=False)
    def refused(*args, **kwargs):
        pytest.fail("public accounting fixture attempted network access")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


def campaign(tmp_path):
    goal = {"max_llm_calls": 20, "max_training_jobs": 4, "max_gpu_seconds": 500,
        "research_execution": {"version": "single_plan_v1", "paid_maintenance": "off",
            "budget": {"envelopes": {"diagnosis": {"role_calls": {"result_analyst": 2},
                "continuation_calls": 2, "input_tokens_per_call": 1000, "output_tokens_per_call": 100}}}}}
    state = {"status": "planning", "max_llm_calls": 20, "max_training_jobs": 4,
             "max_gpu_seconds": 500, "gpu_seconds_left": 500, "training_jobs": 0}
    for name, value in (("goal.json", goal), ("campaign_state.json", state),
                        ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (tmp_path / name).write_text(json.dumps(value))
    return tmp_path, state


def paid_proof(camp, call_id, role):
    claim = owners.claim_request(camp, call_id=call_id, operation="synthetic:" + call_id,
        role=role, input_bound=100, output_bound=50)
    proof = {"input_tokens": 20, "output_tokens": 8, "reasoning_tokens": 5,
             "transport_outcome": "confirmed_response"}
    owners.settle_request(camp, call_id, proof)
    owners.settle_request(camp, call_id, proof)
    return claim


def test_terminal_handoff_two_attempts_and_restart_never_renew_budget(tmp_path):
    camp, _ = campaign(tmp_path)
    selected = owners.allocate_selected_plan(camp, plan_id="p1", task_kind="diagnosis",
        known_cost={}, recovery_limits={})
    with owners.budget_owner(camp, selected["owner_id"]):
        paid_proof(camp, "analyst", "result_analyst")
    handoff = owners.finish_selected_plan(camp, plan_id="p1", observation_digest="terminal:exact")
    assert handoff["allocated"]["llm_calls"] == 2
    assert owners.finish_selected_plan(camp, plan_id="p1", observation_digest="terminal:exact") == handoff
    with pytest.raises(owners.OwnerBudgetError, match="terminal_observation_conflict"):
        owners.finish_selected_plan(camp, plan_id="p1", observation_digest="different")
    with owners.planner_owner(camp, observation_digest="terminal:exact", round_id="next-round") as oid:
        assert oid == handoff["owner_id"]
        paid_proof(camp, "planner-1", "research_planner")
    # Re-entering the exact round models a worker restart, not a new allocation.
    with owners.planner_owner(camp, observation_digest="terminal:exact", round_id="next-round") as restarted:
        assert restarted == oid
        paid_proof(camp, "planner-2", "research_planner")
        with pytest.raises(owners.OwnerBudgetError):
            paid_proof(camp, "planner-3", "research_planner")
    assert owners.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 3
    with pytest.raises(owners.OwnerBudgetError, match="already_bound"):
        with owners.planner_owner(camp, observation_digest="terminal:exact", round_id="new-round"):
            pytest.fail("same continuation was renewed")
    owners.finish_planner_round(camp, oid)
    with pytest.raises(owners.OwnerBudgetError, match="finished"):
        with owners.planner_owner(camp, observation_digest="terminal:exact", round_id="next-round"):
            pytest.fail("finished continuation reopened")
    assert owners.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 3


def test_owner_and_native_gpu_receipt_charge_once_across_restart(tmp_path):
    camp, state = campaign(tmp_path)
    parent = owners.allocate_selected_plan(camp, plan_id="p1", task_kind="validation",
        known_cost={"training_jobs": 1, "gpu_seconds": 100}, recovery_limits={})
    with owners.budget_owner(camp, parent["owner_id"]):
        claim = owners.claim_training(camp, "j1", 50)
        assert owners.claim_training(camp, "j1", 50) == claim
    assert owners.snapshot(camp)["global_spent_and_occupied"]["gpu_seconds"] == 50
    owners.settle_training(camp, "j1", 30)
    owners.settle_training(camp, "j1", 30)
    assert owners.snapshot(camp)["global_spent_and_occupied"]["gpu_seconds"] == 30
    state["training_jobs"] = 1
    assert budget.charge_gpu(camp, state, 30, job_id="j1") is True
    (camp / "campaign_state.json").write_text(json.dumps(state))
    assert budget.charge_gpu(camp, json.loads((camp / "campaign_state.json").read_text()), 30, job_id="j1") is False
    view = budget.snapshot(camp, state)
    assert view["gpu_seconds_used"] == 30
    assert view["owner_budget"]["global_spent_and_occupied"]["gpu_seconds"] == 30
    assert view["owner_budget"]["global_spent_and_occupied"]["training_jobs"] == 1
    with pytest.raises(owners.OwnerBudgetError, match="settlement_identity_conflict"):
        owners.settle_training(camp, "j1", 31)
    assert json.loads((camp / "cost.json").read_text())["settled_job_ids"] == ["j1"]
