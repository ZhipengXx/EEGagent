"""Native paid Planner retry/publication recovery with an offline SDK transport."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import socket
import sqlite3

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import llm, loop, plan_protocol, selected_plan
from react_agent.eeg_research.agentic.context.types import digest
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


class Crash(BaseException):
    pass


def reply():
    proposals = [{"proposal_id": pid, "task_kind": "diagnosis", "target_candidate_ref": "baseline",
        "question": "Inspect recorded comparison " + pid, "task_description": "Inspect existing evidence " + pid,
        "scientific_intent": "Explain bounded observation " + pid, "evaluation_scope": {"phase": "diagnostic"},
        "expected_observation": "Recorded diagnostic facts", "completion_condition": "One diagnostic observation",
        "required_context_refs": [], "value_rationale": "Reduce uncertainty", "implementation_risk": "No new training"}
        for pid in ("compare", "coverage", "limits")]
    return {"protocol_version": "research_plan_draft.v1", "proposals": proposals,
        "evaluations": [{"proposal_id": item["proposal_id"], "rank": index, "value_rationale": "Read measured facts",
            "implementation_risk": "bounded"} for index, item in enumerate(proposals, 1)],
        "selected_proposal_id": "compare", "selection_rationale": "Best bounded explanation"}


def campaign(tmp_path):
    goal = {"goal_id": "paid_recovery", "max_training_jobs": 2, "max_llm_calls": 18,
        "max_gpu_seconds": 100, "max_candidates": 2, "allowed_training_actions": ["run_full"],
        "research_execution": {"version": "single_plan_v1", "target_k": 3,
            "max_planner_api_attempts": 2, "proposal_correction_calls": 1}}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "paid_contract",
        "task_type": "eeg_image_retrieval"}, request_id="paid_recovery")
    return tmp_path / "paid_recovery"


@pytest.fixture
def sdk(monkeypatch, request):
    clients, requests = [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def no_network(*args, **kwargs):
        pytest.fail("Paid recovery regression forbids external network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    def transport(request):
        requests.append(json.loads(request.content))
        correction_case = mode == "correction"
        truncated = len(requests) == 1 and not correction_case
        response = reply()
        if correction_case and len(requests) == 1:
            for proposal in response["proposals"]:
                proposal["required_context_refs"] = ["/goal"]
        return HTTP.Response(200, json={"id": "local-paid-fixture", "object": "chat.completion", "created": 0,
            "model": "synthetic-sdk-model", "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop",
                "message": {"role": "assistant", "content": "{" if truncated else json.dumps(response)}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}})
    mode = getattr(request, "param", None)
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


def old_schema(monkeypatch):
    native = plan_protocol.ResearchPlanProposal
    class PreRiskProposal:
        @staticmethod
        def model_validate(value):
            if "implementation_risk" in value:
                raise ValueError("implementation_risk: extra_forbidden in previously deployed schema")
            return native.model_validate(value)
    monkeypatch.setattr(plan_protocol, "ResearchPlanProposal", PreRiskProposal)


def rows(camp, name):
    return [json.loads(line) for line in (camp / name).read_text().splitlines() if line]


def snapshot(camp):
    return {str(path.relative_to(camp)): path.read_bytes() for path in camp.rglob("*") if path.is_file()
        and path.name not in {"campaign_state.json", "resource_owners.sqlite", "events.jsonl", "ui_events.jsonl"}}


def claims(camp):
    with sqlite3.connect((camp / "resource_owners.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        return {identity: json.loads(payload) for identity, payload in db.execute("SELECT id,payload FROM claims")}


def resume_tick(camp, monkeypatch):
    state = loop.load_state(camp)
    state["status"] = "created"
    loop.save_state(camp, state)
    def forbidden(*args, **kwargs):
        pytest.fail("Paid replay must not compile, capture, claim or request again")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
    monkeypatch.setattr(loop, "observation", forbidden)
    monkeypatch.setattr("react_agent.eeg_research.agentic.experiment_observations.compact_planner_view", forbidden)
    monkeypatch.setattr("react_agent.eeg_research.agentic.owner_budget.planner_owner", forbidden)
    return loop.tick(camp, forbidden, services={"launch": forbidden, "designer": forbidden})


def test_native_two_paid_json_attempts_keep_incomplete_and_revalidate_same_round(tmp_path, sdk, monkeypatch):
    camp = campaign(tmp_path)
    backend = llm.role_backend(camp, "research_planner")
    with monkeypatch.context() as previous:
        old_schema(previous)
        blocked = loop.tick(camp, backend)
    assert blocked["status"] == "blocked"
    assert len(sdk) == 2
    paid_rows = rows(camp, "llm_calls.jsonl")
    assert len(paid_rows) == 2 and paid_rows[-1]["success"]
    rid = paid_rows[-1]["planner_round_id"]
    draft = loop._read(camp / "research_round_drafts" / (rid + ".json"))
    assert draft["parsed"]["correction_unavailable"] == {"reason": "planner_round_attempt_limit", "status": "not_sent"}
    assert draft["parsed"]["proposal_history"] == [reply()]
    assert not (camp / "research_rounds" / (rid + ".json")).exists()
    frozen, original_claims = snapshot(camp), claims(camp)
    restored = resume_tick(camp, monkeypatch)
    plan = selected_plan.load_plan(camp, restored)
    assert plan["originating_round"] == rid and plan["current_step"] == "prepare"
    assert plan["repair_count"] == 0 and plan["proposal"]["implementation_risk"] == "No new training"
    assert len(sdk) == len(rows(camp, "llm_calls.jsonl")) == 2
    assert claims(camp) == original_claims
    for name, data in frozen.items():
        assert (camp / name).read_bytes() == data, name
    round_record = loop._read(camp / "research_rounds" / (rid + ".json"))
    assert round_record["pool_status"] == "complete" and round_record["valid_count"] == 3


@pytest.mark.parametrize("window", ["before_raw", "after_raw", "before_draft", "after_draft", "after_round"])
def test_native_publication_crash_replays_without_context_or_budget_changes(tmp_path, sdk, monkeypatch, window):
    camp = campaign(tmp_path)
    backend = llm.role_backend(camp, "research_planner")
    write = loop._write
    def injected(path, value, **kwargs):
        raw = "research_round_raw" in path.parts
        draft = path.parent == camp / "research_round_drafts"
        final = path.parent == camp / "research_rounds"
        if (window == "before_raw" and raw) or (window == "before_draft" and draft):
            raise Crash(window)
        write(path, value, **kwargs)
        if (window == "after_raw" and raw) or (window == "after_draft" and draft) or (window == "after_round" and final):
            raise Crash(window)
    with monkeypatch.context() as faults:
        faults.setattr(loop, "_write", injected)
        with pytest.raises(Crash):
            loop.tick(camp, backend)
    frozen, original_claims = snapshot(camp), claims(camp)
    restored = resume_tick(camp, monkeypatch)
    plan = selected_plan.load_plan(camp, restored)
    assert plan["current_step"] == "prepare" and plan["repair_count"] == 0
    assert len(sdk) == len(rows(camp, "llm_calls.jsonl")) == 2
    assert claims(camp) == original_claims
    for name, data in frozen.items():
        assert (camp / name).read_bytes() == data, name


@pytest.mark.parametrize("tamper", ["reply", "request", "round", "owner", "settlement", "goal", "terminal"])
def test_legacy_paid_replay_refuses_tamper_before_any_context_or_request(tmp_path, sdk, monkeypatch, tamper):
    camp = campaign(tmp_path)
    backend = llm.role_backend(camp, "research_planner")
    write = loop._write
    def lose_raw(path, value, **kwargs):
        if "research_round_raw" in path.parts:
            raise Crash("Legacy paid response without outer publication")
        write(path, value, **kwargs)
    with monkeypatch.context() as faults:
        faults.setattr(loop, "_write", lose_raw)
        with pytest.raises(Crash):
            loop.tick(camp, backend)
    paid = rows(camp, "llm_calls.jsonl")[-1]
    cid = paid["call_id"]
    if tamper in {"reply", "request"}:
        path = camp / ("llm_responses" if tamper == "reply" else "llm_requests") / (cid + ".json")
        body = loop._read(path)
        if tamper == "reply":
            body["reply"]["selection_rationale"] += " changed"
        else:
            body["user"] += " "
        write(path, body)
    elif tamper in {"round", "owner", "settlement"}:
        with sqlite3.connect(camp / "resource_owners.sqlite") as db:
            table, identity = ("owners", paid["budget_owner"]) if tamper in {"round", "owner"} else ("claims", cid)
            body = json.loads(db.execute("SELECT payload FROM " + table + " WHERE id=?", (identity,)).fetchone()[0])
            if tamper == "round":
                body["round_id"] = "research_round_" + "f" * 32
            elif tamper == "owner":
                body["owner_id"] = "different_owner"
            else:
                body["settlement_usage"]["transport_outcome"] = "unknown"
            db.execute("UPDATE " + table + " SET payload=? WHERE id=?", (json.dumps(body), identity))
    elif tamper == "goal":
        body = loop._read(camp / "goal.json")
        body["max_llm_calls"] += 1
        write(camp / "goal.json", body)
    else:
        state = loop.load_state(camp)
        state["terminal_observation_digest"] = "changed_terminal"
        loop.save_state(camp, state)
    frozen, original_claims = snapshot(camp), claims(camp)
    result = resume_tick(camp, monkeypatch)
    assert result["status"] == "blocked" and "paid_round_" in result["detail"]
    assert not list((camp / "selected_plans").glob("plan_*.json"))
    assert len(sdk) == len(rows(camp, "llm_calls.jsonl")) == 2
    assert claims(camp) == original_claims
    for name, data in frozen.items():
        assert (camp / name).read_bytes() == data, name


def test_raw_replay_with_no_registry_does_not_create_a_context_database(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    from react_agent.eeg_research.agentic import owner_budget, paid_research_round
    state = loop.load_state(camp)
    full = {"_campaign_path": str(camp), "goal": loop._read(camp / "goal.json"),
        "available_actions": [], "candidates": [], "candidate_admissions": [], "evidence": [],
        **{key: state.get(key) for key in ("contract_fingerprint", "execution_fingerprint", "candidate_id",
            "candidate_ready", "selected_plan", "terminal_observation_id", "terminal_observation_digest")}}
    with owner_budget.planner_owner(camp, observation_digest=digest(plan_protocol.round_input(full)),
            round_id=plan_protocol.round_id(full)) as owner:
        paid_research_round.publish_raw(camp, state, full, owner, reply())
    assert not (camp / "context_sources.sqlite").exists()
    restored = resume_tick(camp, monkeypatch)
    assert selected_plan.load_plan(camp, restored)["current_step"] == "prepare"
    assert not (camp / "context_sources.sqlite").exists()


@pytest.mark.parametrize("sdk", ["correction"], indirect=True)
@pytest.mark.parametrize("window", ["before_second_raw", "after_second_raw"])
def test_confirmed_correction_tail_survives_its_raw_publication_crash(tmp_path, sdk, monkeypatch, window):
    camp = campaign(tmp_path)
    backend = llm.role_backend(camp, "research_planner")
    write = loop._write
    def crash_second(path, value, **kwargs):
        second = "research_round_raw" in path.parts and value.get("sequence") == 1
        if second and window == "before_second_raw":
            raise Crash(window)
        write(path, value, **kwargs)
        if second and window == "after_second_raw":
            raise Crash(window)
    with monkeypatch.context() as faults:
        faults.setattr(loop, "_write", crash_second)
        with pytest.raises(Crash):
            loop.tick(camp, backend)
    assert len(sdk) == 2
    calls = rows(camp, "llm_calls.jsonl")
    assert all(row["success"] and row["transport_outcome"] == "confirmed_response" for row in calls)
    assert calls[0]["operation_id"].endswith(":generation")
    assert calls[1]["operation_id"].endswith(":correction")
    frozen, original_claims = snapshot(camp), claims(camp)
    restored = resume_tick(camp, monkeypatch)
    plan = selected_plan.load_plan(camp, restored)
    assert plan["current_step"] == "prepare" and plan["originating_round"] == calls[1]["planner_round_id"]
    published = loop._read(camp / "research_rounds" / (plan["originating_round"] + ".json"))
    assert published["pool_status"] == "complete" and len(published["proposal_history"]) == 2
    assert published["proposal_history"][-1] == reply()
    assert len(sdk) == len(rows(camp, "llm_calls.jsonl")) == 2
    assert claims(camp) == original_claims
    for name, data in frozen.items():
        assert (camp / name).read_bytes() == data, name


def test_native_raw_receipt_cannot_replace_settled_reply_even_with_updated_local_digest(tmp_path, sdk, monkeypatch):
    camp = campaign(tmp_path)
    backend = llm.role_backend(camp, "research_planner")
    write = loop._write
    def crash_after_raw(path, value, **kwargs):
        write(path, value, **kwargs)
        if "research_round_raw" in path.parts:
            raise Crash("raw returned")
    with monkeypatch.context() as faults:
        faults.setattr(loop, "_write", crash_after_raw)
        with pytest.raises(Crash):
            loop.tick(camp, backend)
    path = next((camp / "research_round_raw").glob("*/*.json"))
    receipt = loop._read(path)
    receipt["raw"]["selection_rationale"] += " changed after settlement"
    receipt["receipt_digest"] = digest({key: value for key, value in receipt.items() if key != "receipt_digest"})
    write(path, receipt)
    frozen, original_claims = snapshot(camp), claims(camp)
    result = resume_tick(camp, monkeypatch)
    assert result["status"] == "blocked" and "paid_round_returned_response_changed" in result["detail"]
    assert not list((camp / "selected_plans").glob("plan_*.json"))
    assert claims(camp) == original_claims and len(sdk) == 2
    for name, data in frozen.items():
        assert (camp / name).read_bytes() == data, name
