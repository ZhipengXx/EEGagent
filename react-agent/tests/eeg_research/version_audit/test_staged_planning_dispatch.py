"""Native staged selection with two separate offline SDK HTTP requests.

The delivery adapter supplies synthetic local knowledge backed by actual
registered source bytes. This verifies delivery and dispatch, not MiniLM search.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import socket
import sqlite3

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import llm, loop, owner_budget, selected_plan
from react_agent.eeg_research.agentic.context.representations import handle, resolve_request_pointer
from react_agent.eeg_research.agentic.context.sources import ContextSourceRegistry
from react_agent.eeg_research.agentic.plan_protocol import proposal_pool_digest
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


class Crash(BaseException):
    pass


def proposals(source_ref):
    return {"protocol_version": "research_proposals.v1", "proposals": [{
        "proposal_id": pid, "task_kind": "new_method", "target_candidate_ref": None,
        "question": "Does independent mechanism " + pid + " improve a matched comparison?",
        "task_description": "Measure this one declared intervention " + pid,
        "scientific_intent": "Independent mechanism " + pid, "evaluation_scope": {"phase": "full", "seed": 0},
        "expected_observation": "Matched measured feedback", "completion_condition": "One bounded observation",
        "required_context_refs": [source_ref] if pid == "p2" else [],
        "cost_assessment": {}, "value_rationale": "Resolve scientific uncertainty"}
        for pid in ("p1", "p2", "p3")]}


def critique(pool):
    return {"protocol_version": "research_plan_critique.v1", "proposal_pool_digest": pool["pool_digest"],
        "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Independent critic judgement",
            "implementation_risk": "Finite engineering risk"} for pid, rank in (("p3", 3), ("p1", 2), ("p2", 1))],
        "selected_proposal_id": "p2", "selection_rationale": "Select p2 for the strongest information value."}


def journal(camp, name):
    path = camp / name
    return [json.loads(row) for row in path.read_text().splitlines() if row] if path.exists() else []


def database(camp, table):
    with sqlite3.connect((camp / "resource_owners.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        return {identity: json.loads(payload) for identity, payload in db.execute("SELECT id,payload FROM " + table)}


def immutable_snapshot(camp):
    mutable = {"campaign_state.json", "resource_owners.sqlite", "events.jsonl", "ui_events.jsonl", "cost.json"}
    return {str(path.relative_to(camp)): path.read_bytes() for path in camp.rglob("*") if path.is_file()
        and path.name not in mutable and path.suffix != ".sqlite"}


def source_records(camp):
    with sqlite3.connect((camp / "context_sources.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        return {row[0]: row for row in db.execute("SELECT * FROM context_sources")}


@pytest.fixture(params=["deepseek-flash", "deepseek-v4-pro"])
def staged_campaign(tmp_path, monkeypatch, request):
    goal = {"goal_id": "staged_dispatch", "max_llm_calls": 300, "max_training_jobs": 80,
        "max_gpu_seconds": 100000, "max_candidates": 20, "allowed_training_actions": ["run_full"],
        "memory": {"enabled": False},
        "role_models": {
            "research_planner": {"model": request.param, "thinking": True,
                "reasoning_effort": "high", "max_tokens": 32768, "timeout_s": 240},
            "research_critic": {"model": "deepseek-v4-pro", "thinking": True,
                "reasoning_effort": "high", "max_tokens": 32768, "timeout_s": 240}},
        "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic",
            "implementation_review_mode": "single_pass", "target_k": 3,
            "max_planner_api_attempts": 2, "max_critic_api_attempts": 2,
            "budget": {"input_tokens_per_call": 128000, "output_tokens_per_call": 32768,
                "envelopes": {"new_method": {"training_jobs": 2, "gpu_seconds": 100,
                    "engineering_gpu_seconds": 0}}}}}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "staged_dispatch_contract",
        "task_type": "eeg_image_retrieval", "input_shape": [17, 250], "embedding_dim": 1024},
        request_id="staged_dispatch")
    camp = tmp_path / "staged_dispatch"
    registry = ContextSourceRegistry(camp)
    issue = registry.capture({"issue_id": "control_binding", "summary": "Declare matched control before implementation."},
        block_id="selected-review-failure:fixture", kind="checks", role="research_planner")
    knowledge = registry.capture({"lesson": "A matched control makes comparisons attributable to the intervention."},
        block_id="knowledge:matched_control", kind="memory", role="research_planner")
    from react_agent.eeg_research.agentic import experiment_observations
    native_compact = experiment_observations.compact_planner_view
    def with_registered_context(value, fullobs, **kwargs):
        # Cover both selector delivery and ContextService's second compilation.
        # This synthetic adapter never claims an embedding/similarity result.
        delivered = native_compact(value, fullobs, **kwargs)
        delivered["experiment_observations"]["cards"] = [{"body": {"observation_id": "engineering_fixture",
            "content_digest": issue.source_hash, "engineering_failure": {"blocking_issues": [
                {"original_issue": handle(issue)}]}, "source_refs": [{"source_ref": issue.source_ref}]},
            "source_refs": []}]
        delivered["retrieved_knowledge"] = {"status": "ready", "items": [{"body": registry.load(knowledge, role="research_planner"),
            "source_ref": knowledge.source_ref, "source_hash": knowledge.source_hash,
            "scope": "local knowledge prior", "is_scientific_evidence": False}]}
        return delivered
    monkeypatch.setattr(experiment_observations, "compact_planner_view", with_registered_context)
    return camp, issue, knowledge


@pytest.fixture
def sdk(staged_campaign, monkeypatch):
    camp, issue, knowledge = staged_campaign
    clients, requests = [], []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for key in list(os.environ):
        if key.startswith("DEEPSEEK_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    def no_network(*args, **kwargs):
        pytest.fail("Staged dispatch fixture forbids external sockets and DNS")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    def transport(request):
        wire = json.loads(request.content)
        user = json.loads(wire["messages"][1]["content"])
        requests.append(wire)
        assert not list((camp / "selected_plans").glob("plan_*.json"))
        assert not list((camp / "candidates").glob("*/extension/eeg_candidate.py"))
        assert not list((camp / "jobs").glob("*/launch_intent.json"))
        assert not loop.load_state(camp).get("candidate_admissions")
        memory = resolve_request_pointer(user, "/retrieved_knowledge")
        assert memory["items"][0]["body"] == registry_load(camp, knowledge)
        if user["planning_stage"] == "propose":
            expected_model = json.loads((camp / "goal.json").read_text())["role_models"]["research_planner"]["model"]
            assert wire["model"] == expected_model
            assert user["planning_stage"] == "propose"
            assert "research_proposals.v1" in wire["messages"][0]["content"]
            response = proposals(issue.source_ref)
        else:
            assert wire["model"] == "deepseek-v4-pro" and user["planning_stage"] == "critic"
            pool = resolve_request_pointer(user, "/proposal_pool")
            assert pool["pool_digest"] == user["proposal_pool_digest"] == proposal_pool_digest(pool)
            assert [row["proposal_id"] for row in pool["proposals"]] == ["p1", "p2", "p3"]
            assert pool["proposals"][1]["required_context_refs"] == [issue.source_ref]
            assert "research_plan_critique.v1" in wire["messages"][0]["content"]
            response = critique(pool)
        return HTTP.Response(200, json={"id": "offline-staged-" + str(len(requests)), "object": "chat.completion",
            "created": 0, "model": wire["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(response)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})
    class MockBackend(DeepSeekBackend):
        def __init__(self, config):
            super().__init__(config)
            hooks = {key: list(value) for key, value in self._client._client.event_hooks.items()}
            asyncio.run(self._client.close())
            self._client = AsyncOpenAI(api_key="offline-unused", base_url="https://offline.invalid", max_retries=0,
                http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(transport), trust_env=False, event_hooks=hooks))
            self._client._platform = "Linux"
            clients.append(self)
    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", MockBackend)
    yield requests
    for client in clients:
        asyncio.run(client._client.close())


def registry_load(camp, block):
    from react_agent.eeg_research.agentic.paid_research_round import readonly_registry
    return readonly_registry(camp).load(block, role="research_planner")


def backends(camp):
    return llm.role_backend(camp, "research_planner"), llm.role_backend(camp, "research_critic")


def assert_one_selection(camp, state, sdk):
    plans = list((camp / "selected_plans").glob("plan_*.json"))
    assert len(plans) == 1
    plan = selected_plan.load_plan(camp, state)
    assert plan["originating_proposal"] == "p2" and plan["current_step"] == "prepare"
    assert plan["repair_count"] == plan["attempt"] == 0
    assert not state.get("candidate_admissions") and state["training_jobs"] == 0
    assert not list((camp / "candidates").glob("*/extension/eeg_candidate.py"))
    assert not list((camp / "jobs").glob("*/launch_intent.json"))
    calls = journal(camp, "llm_calls.jsonl")
    assert len(sdk) == len(calls) == 2
    assert [row["role"] for row in calls] == ["research_planner", "research_critic"]
    assert {row["budget_owner"] for row in calls} == {"bootstrap:initial_planner"}
    assert all(row["success"] and row["transport_outcome"] == "confirmed_response" for row in calls)
    owner = database(camp, "owners")["bootstrap:initial_planner"]
    assert owner["allocated"]["llm_calls"] == 4
    assert owner["consumed"]["llm_calls"] == 2 and owner["status"] == "released"
    assert len(database(camp, "claims")) == 2
    assert len([row for row in database(camp, "owners").values() if row["kind"] == "selected_plan"]) == 1
    round_record = loop._read(camp / "research_rounds" / (plan["originating_round"] + ".json"))
    assert round_record["selection_source"] == "independent_critic_selection"
    assert round_record["selected_proposal_id"] == "p2"
    assert round_record["selection_rationale"] == "Select p2 for the strongest information value."
    assert round_record["valid_count"] == round_record["evaluation_count"] == 3
    assert round_record["planning_stages"] == ["generation", "critic"]
    return plan


def test_actual_separate_sdk_models_choose_before_any_candidate_work(staged_campaign, sdk):
    camp, _, _ = staged_campaign
    planner, critic = backends(camp)
    state = selected_plan._select(camp, loop.load_state(camp), planner, critic=critic)
    assert_one_selection(camp, state, sdk)
    expected_model = json.loads((camp / "goal.json").read_text())["role_models"]["research_planner"]["model"]
    assert [request["model"] for request in sdk] == [expected_model, "deepseek-v4-pro"]
    assert sdk[0]["messages"][0]["content"] != sdk[1]["messages"][0]["content"]
    assert json.loads(sdk[0]["messages"][1]["content"])["planning_stage"] == "propose"


@pytest.mark.parametrize("stage", ["generation", "critic"])
@pytest.mark.parametrize("window", ["before_raw", "after_raw"])
def test_paid_stage_crash_recovers_same_calls_owner_sources_and_choice(staged_campaign, sdk, monkeypatch, stage, window):
    camp, _, _ = staged_campaign
    planner, critic = backends(camp)
    write = loop._write
    def fault(path, value, **kwargs):
        target = "research_round_raw" in path.parts and value.get("stage") == stage
        if target and window == "before_raw":
            raise Crash(stage + window)
        write(path, value, **kwargs)
        if target and window == "after_raw":
            raise Crash(stage + window)
    with monkeypatch.context() as failure:
        failure.setattr(loop, "_write", fault)
        with pytest.raises(Crash):
            selected_plan._select(camp, loop.load_state(camp), planner, critic=critic)
    original_requests = copy.deepcopy(sdk)
    original_files, original_claims = immutable_snapshot(camp), database(camp, "claims")
    original_sources = source_records(camp)
    original_owner = copy.deepcopy(database(camp, "owners")["bootstrap:initial_planner"])
    def no_context(*args, **kwargs):
        pytest.fail("Paid staged recovery must reuse original context without recapture")
    monkeypatch.setattr(loop, "observation", no_context)
    monkeypatch.setattr("react_agent.eeg_research.agentic.experiment_observations.compact_planner_view", no_context)
    if stage == "critic":
        monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
        def no_api(*args, **kwargs):
            pytest.fail("Already paid Critic recovery must use zero new API calls")
        resumed = selected_plan._select(camp, loop.load_state(camp), no_api, critic=no_api)
    else:
        resumed = selected_plan._select(camp, loop.load_state(camp), planner, critic=critic)
    assert_one_selection(camp, resumed, sdk)
    assert sdk[:len(original_requests)] == original_requests
    for name, content in original_files.items():
        current = (camp / name).read_bytes()
        if stage == "generation" and name.endswith(".jsonl"):
            assert current.startswith(content), name
        else:
            assert current == content, name
    sources = source_records(camp)
    for ref, row in original_sources.items():
        assert sources[ref] == row
    claims = database(camp, "claims")
    for identity, claim in original_claims.items():
        assert claims[identity] == claim
    owner = database(camp, "owners")["bootstrap:initial_planner"]
    for key in ("owner_id", "kind", "allocated", "round_id", "observation_digest", "created_at"):
        assert owner[key] == original_owner[key]
