"""Planner and Critic spend one finite native allocation at their actual models."""
from __future__ import annotations

import json
import os
import socket
import sqlite3

import pytest

from react_agent.eeg_research.agentic import api_policy, llm, owner_budget
from react_agent.eeg_research.agentic.role_config import resolve_role_config


@pytest.fixture(autouse=True)
def isolated_routing(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for name in list(os.environ):
        if name.startswith(("DEEPSEEK_", "EEG_RESEARCH_PLANNER_", "EEG_RESEARCH_CRITIC_")):
            monkeypatch.delenv(name)
    # Exercise native admission/settlement locally without any provider transport.
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    def no_network(*args, **kwargs):
        pytest.fail("Staged owner regression attempted a real provider request")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)


def campaign(tmp_path, *, staged=True, price="*", reserve=None, continuation=None):
    config = {"input_tokens_per_call": 1000, "output_tokens_per_call": 512}
    if price is not None:
        config["pricing"] = {"source": "synthetic-owner", "date": "2026-10-14", "model": price,
                             "input_per_million": 2, "output_per_million": 3}
    if reserve is not None:
        config["conservative_usd_per_request"] = reserve
    if continuation is not None:
        config["envelopes"] = {"diagnosis": {"continuation_calls": continuation}}
    execution = {"version": "single_plan_v1", "budget": config}
    if staged:
        execution.update(planning_mode="propose_then_critic", max_planner_api_attempts=2,
                         max_critic_api_attempts=2)
    goal = {"max_llm_calls": 40, "max_training_jobs": 4, "max_gpu_seconds": 100,
            "research_execution": execution,
            "role_models": {"research_planner": {"model": "synthetic-flash", "thinking": False, "max_tokens": 128},
                            "research_critic": {"model": "synthetic-pro", "thinking": False, "max_tokens": 256}}}
    if price is not None:
        goal["max_api_usd"] = 5
    for name, value in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                        ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (tmp_path / name).write_text(json.dumps(value))
    return tmp_path


def owners(camp):
    with sqlite3.connect(camp / "resource_owners.sqlite") as conn:
        return {identity: json.loads(body) for identity, body in conn.execute("SELECT id, payload FROM owners")}


def reserve_and_settle(camp, role, call_id, *, operation=None):
    routing = resolve_role_config(camp, role)
    row = api_policy.reserve_operation_attempt(camp, operation=operation or role, call_id=call_id,
        metadata={"role": role, "requested_model": routing.model, "role_config_hash": routing.role_config_hash,
                  "owner_input_token_bound": 500, "owner_output_token_bound": routing.max_tokens})
    price = owner_budget.request_pricing(camp, call_id)
    assert price["model"] == routing.model
    owner_budget.before_transport(camp, call_id=call_id, role=role, input_bound=500,
                                 output_bound=routing.max_tokens, model=routing.model)
    owner_budget.settle_request(camp, call_id, {"provider": "deepseek", "requested_model": routing.model,
        "input_tokens": 100, "output_tokens": 64, "transport_outcome": "confirmed_response",
        "unconfirmed_charge_possible": False})
    return row


@pytest.mark.parametrize("kind", ["bootstrap", "continuation"])
def test_two_stages_claim_and_settle_same_original_owner_without_enlargement(tmp_path, kind):
    camp = campaign(tmp_path)
    if kind == "continuation":
        selected = owner_budget.allocate_selected_plan(camp, plan_id="fixture-plan", task_kind="diagnosis",
            known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits={})
        owner_budget.finish_selected_plan(camp, plan_id="fixture-plan", observation_digest="fixture-observation")
        expected_id = selected["owner_id"] + ":continuation"
    else:
        expected_id = "bootstrap:initial_planner"
    with owner_budget.planner_owner(camp, observation_digest="fixture-observation", round_id="fixture-round") as owner_id:
        assert owner_id == expected_id
        original = owners(camp)[owner_id]["allocated"]
        assert original["llm_calls"] == 4
        assert original["input_tokens"] == 4000 and original["output_tokens"] == 2 * 128 + 2 * 256
        assert original["api_usd"] == pytest.approx((4000 * 2 + (2 * 128 + 2 * 256) * 3) / 1_000_000)
        with api_policy.planner_api_round(camp, "fixture-round", max_attempts=4):
            calls = []
            roles = ("research_planner", "research_critic") if kind == "bootstrap" else ("research_critic", "research_planner")
            for role in roles:
                for index in range(2):
                    calls.append(reserve_and_settle(camp, role, role + str(index)))
                with pytest.raises(llm.LlmUnavailable, match="planning_stage_attempt_limit|planner_round_attempt_limit"):
                    reserve_and_settle(camp, role, role + "excess", operation=role + ":another-operation")
            assert {row["budget_owner"] for row in calls} == {owner_id}
        after = owners(camp)[owner_id]
        assert after["allocated"] == original and after["consumed"]["llm_calls"] == 4
        assert owner_budget.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 4
        assert owner_budget.snapshot(camp)["unknown"] == []


def test_legacy_owner_preserves_two_call_bounds_and_rejects_critic(tmp_path):
    camp = campaign(tmp_path, staged=False)
    with owner_budget.planner_owner(camp, observation_digest="fixture-observation", round_id="legacy-round") as owner_id:
        original = owners(camp)[owner_id]["allocated"]
        assert original["llm_calls"] == 2 and original["output_tokens"] == 2 * 512
        with pytest.raises(owner_budget.OwnerBudgetError, match="continuation_scope_violation"):
            owner_budget.claim_request(camp, call_id="forbidden-critic", operation="critic", role="research_critic",
                input_bound=100, output_bound=256, model="synthetic-pro")
        assert owners(camp)[owner_id]["allocated"] == original


def test_flash_tagged_price_cannot_authorize_pro_before_any_request(tmp_path):
    camp = campaign(tmp_path, price="synthetic-flash")
    with pytest.raises(owner_budget.OwnerBudgetError, match="pricing_provider_or_model_mismatch"):
        with owner_budget.planner_owner(camp, observation_digest="fixture-observation", round_id="price-round"):
            pytest.fail("Pro allocation must reject the Flash-only price before admission")
    assert owners(camp) == {}
    assert not (camp / api_policy.ATTEMPTS).exists()
    assert not (camp / "llm_calls.jsonl").exists()


@pytest.mark.parametrize("calls,expected_output", [(2, 128 + 256), (3, 2 * 128 + 256), (4, 2 * 128 + 2 * 256)])
def test_smaller_continuation_reserves_each_stage_before_optional_correction(tmp_path, calls, expected_output):
    camp = campaign(tmp_path, continuation=calls)
    allocated = owner_budget.allocate_selected_plan(camp, plan_id="fixture-plan", task_kind="diagnosis",
        known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits={})
    bound = owners(camp)[allocated["owner_id"] + ":continuation"]["allocated"]
    assert bound["llm_calls"] == calls and bound["output_tokens"] == expected_output


def test_single_call_continuation_cannot_starve_the_independent_critic(tmp_path):
    camp = campaign(tmp_path, continuation=1)
    with pytest.raises(owner_budget.OwnerBudgetError, match="staged_planning_requires_both_finite_roles"):
        owner_budget.allocate_selected_plan(camp, plan_id="fixture-plan", task_kind="diagnosis",
            known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits={})


def test_unpriced_stages_keep_original_unknown_charge_reserve(tmp_path):
    camp = campaign(tmp_path, price=None, reserve=.25)
    amount = owner_budget._planning_resources(camp, 1000, 512, 4)
    assert amount["api_usd"] == 1 and amount["llm_calls"] == 4
    assert amount["output_tokens"] == 768


def test_explicit_planning_output_envelope_is_not_enlarged_for_pro(tmp_path):
    camp = campaign(tmp_path)
    with pytest.raises(owner_budget.OwnerBudgetError, match="planning_role_output_bound_exceeds_envelope:research_critic"):
        owner_budget._planning_resources(camp, 1000, 128, 4)
