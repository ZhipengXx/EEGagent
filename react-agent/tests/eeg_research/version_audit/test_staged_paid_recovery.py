"""Actual SDK paid stages recover at the raw-write boundary with no extra call."""
from __future__ import annotations

import asyncio
import copy
import json
import socket
import sqlite3

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import llm, loop, plan_protocol
from react_agent.eeg_research.agentic.action_planning import digest
from react_agent.eeg_research.agentic.api_policy import planner_api_round
from react_agent.eeg_research.agentic.owner_budget import planner_owner
from react_agent.eeg_research.agentic.staged_planning_recovery import publish_staged_raw, restore_staged_paid
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


def proposal_reply():
    return {"protocol_version": "research_proposals.v1", "proposals": [
        {"proposal_id": pid, "task_kind": "diagnosis", "target_candidate_ref": "baseline",
         "question": "Inspect measured uncertainty " + pid, "task_description": "Review recorded diagnostics " + pid,
         "scientific_intent": "Explain existing facts " + pid, "evaluation_scope": {"phase": "diagnostic"},
         "expected_observation": "Bounded diagnostic facts", "completion_condition": "One native diagnostic observation",
         "required_context_refs": [], "value_rationale": "Reduce uncertainty", "implementation_risk": "No new code"}
        for pid in ("coverage", "limits", "comparison")]}


def make_camp(tmp_path, *, memory=False):
    goal = {"goal_id": "staged_paid", "max_llm_calls": 20, "max_training_jobs": 2,
        "max_gpu_seconds": 100, "max_candidates": 2, "memory": {"enabled": memory,
            "allow_model_download": False, "context_chars_budget": 6000},
        "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic",
            "target_k": 3, "max_planner_api_attempts": 2, "max_critic_api_attempts": 2}}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "staged_local_contract",
        "task_type": "eeg_image_retrieval"}, request_id="staged-paid-only")
    return tmp_path / "staged_paid"


@pytest.fixture
def sdk(monkeypatch):
    clients, calls = [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def no_network(*args, **kwargs):
        pytest.fail("Staged paid recovery forbids external network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    def transport(request):
        body = json.loads(request.content)
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if payload.get("planning_stage") == "critic":
            pool = payload["proposal_pool"]
            reply = {"protocol_version": "research_plan_critique.v1",
                "proposal_pool_digest": payload["proposal_pool_digest"],
                "evaluations": [{"proposal_id": proposal["proposal_id"], "rank": index,
                    "value_rationale": "Compare current information value", "implementation_risk": "Native read only"}
                    for index, proposal in enumerate(pool["proposals"], 1)],
                "selected_proposal_id": pool["proposals"][0]["proposal_id"],
                "selection_rationale": "First task provides the most useful bounded observation"}
        else:
            reply = proposal_reply()
        return HTTP.Response(200, json={"id": "offline-staged", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": 24, "completion_tokens": 16, "total_tokens": 40}})
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
    yield calls
    for client in clients:
        asyncio.run(client._client.close())


def paid_stages(camp, *, receipts, through="critic"):
    full = loop.observation(camp)
    state = loop.load_state(camp)
    rid = plan_protocol.round_id(full)
    with planner_owner(camp, observation_digest=digest(plan_protocol.round_input(full)), round_id=rid) as owner:
        with planner_api_round(camp, rid, max_attempts=4):
            backend = llm.role_backend(camp, "research_planner")
            backend.bind(operation_id=rid + ":generation")
            generated = backend({**full, "planning_stage": "propose"})
            if receipts:
                publish_staged_raw(camp, state, full, owner, generated, "generation")
            if through == "generation":
                return full, owner, {"generation": generated}
            pool = plan_protocol.freeze_proposal_pool(generated["proposals"], full)
            critic = llm.role_backend(camp, "research_critic")
            critic.bind(operation_id=rid + ":critic")
            compared = critic({**full, "planning_stage": "critic", "proposal_pool": pool,
                              "proposal_pool_digest": pool["pool_digest"]})
            if receipts:
                publish_staged_raw(camp, state, full, owner, compared, "critic")
            return full, owner, {"generation": generated, "critic": compared}


def readonly_snapshot(camp):
    return {str(path.relative_to(camp)): path.read_bytes() for path in camp.rglob("*") if path.is_file()}


@pytest.mark.parametrize("through", ["generation", "critic"])
@pytest.mark.parametrize("receipts", [False, True])
def test_actual_paid_stages_restore_without_capture_owner_changes_or_sdk_call(tmp_path, sdk, monkeypatch, receipts, through):
    camp = make_camp(tmp_path)
    full, owner, stages = paid_stages(camp, receipts=receipts, through=through)
    before = readonly_snapshot(camp)
    count = len(sdk)
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
    monkeypatch.setattr(loop, "observation", lambda *_: pytest.fail("No current observation recompile in recovery"))
    restored = restore_staged_paid(camp, loop.load_state(camp))
    assert restored[1] == owner and restored[2] == stages
    assert plan_protocol.round_input(restored[0]) == plan_protocol.round_input(full)
    assert restored[3]["source"] == "native_confirmed_response"
    first_call = json.loads((camp / "llm_calls.jsonl").read_text().splitlines()[0])
    wire = json.loads(json.loads((camp / "llm_requests" / (first_call["call_id"] + ".json")).read_text())["user"])
    from react_agent.eeg_research.agentic.staged_planning_recovery import _planner_delivery
    assert restored[3]["planner_delivery"] == _planner_delivery(wire)
    assert "planning_stage" not in restored[0] and "proposal_pool" not in restored[0]
    if through == "critic":
        frozen = plan_protocol.freeze_proposal_pool(stages["generation"]["proposals"], restored[0])
        assert restored[3]["stage_inputs"]["critic"]["proposal_pool_digest"] == frozen["pool_digest"]
    assert len(sdk) == count and readonly_snapshot(camp) == before


def test_paid_generation_preserves_actual_cards_and_knowledge_for_first_critic(tmp_path, sdk, monkeypatch):
    import numpy as np
    from react_agent.eeg_research.agentic import experiment_observations as observations, semantic_memory
    from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode
    from react_agent.eeg_research.agentic.owner_budget import budget_owner
    class SyntheticVectors:
        metadata = {"model_id": "synthetic-paid-stage-unit-vectors", "model_revision": "f" * 40}
        fingerprint = "synthetic-paid-stage-vector-cache"
        def __init__(self):
            self.calls = 0
        def encode(self, texts):
            self.calls += 1
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors, [{"input_token_count": len(text.split()), "truncated": False} for text in texts]
    vectors = SyntheticVectors()
    monkeypatch.setattr(semantic_memory, "backend_for", lambda *_a, **_k: vectors)
    camp = make_camp(tmp_path, memory=True)
    observations.ExperimentObservationStore(camp).record({"origin_campaign_id": camp.name,
        "plan_id": "synthetic-previous-failure", "observation_kind": "engineering_failure",
        "execution_status": "failed", "question": "Explain an existing failure before retrying",
        "identity": {"candidate_id": "baseline"}, "native_facts": {"metrics": None, "training_jobs": 0},
        "engineering_failure": {"error": "Synthetic previous shape mismatch", "error_category": "shape_error"},
        "source_refs": []})
    source = camp / "synthetic-prior-source.json"
    source.write_text('{"scope":"development","synthetic":true,"note":"Read diagnostics first"}')
    memory = EpisodeStore(camp)
    saved = memory.persist_episode(episode(task_hash=camp.name, candidate_id="baseline",
        kind="exploratory_result", fidelity="pilot", seed=0, metric=None,
        contract_fingerprint="staged_local_contract", artifact=str(source), episode_id="synthetic-prior"))
    accepted = memory.accept_lessons({"proposed_lessons": [{"statement": "Read bounded diagnostics before selecting a new intervention.",
        "conditions": {"task": camp.name, "evaluation_identity": "staged_local_contract"},
        "supporting_episode_ids": [saved["episode_id"]]}]})
    assert len(accepted["accepted"]) == 1
    full, owner, stages = paid_stages(camp, receipts=False, through="generation")
    before = readonly_snapshot(camp)
    restored = restore_staged_paid(camp, loop.load_state(camp))
    assert readonly_snapshot(camp) == before
    delivery = restored[3]["planner_delivery"]
    def expand(value, root):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(root, value["$context_pointer"]), root)
            return {key: expand(child, root) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child, root) for child in value]
        return value
    delivered = expand(delivery, delivery)
    assert delivered["experiment_observations"]["cards"][0]["current_feedback"] is True
    assert delivered["experiment_observations"]["cards"][0]["body"]["execution_status"] == "failed"
    prior = next(item for item in delivered["retrieved_knowledge"]["items"] if item["record_type"] == "lesson")
    assert "Read bounded diagnostics" in json.dumps(prior)
    assert prior["full_record"]["$context_source"].startswith("context://")
    assert prior["full_record"]["source_hash"]
    pool = plan_protocol.freeze_proposal_pool(stages["generation"]["proposals"], restored[0])
    encode_count = vectors.calls
    with budget_owner(camp, owner, purpose="planning"):
        with planner_api_round(camp, pool["round_id"], max_attempts=4):
            critic = llm.role_backend(camp, "research_critic")
            critic.bind(operation_id=pool["round_id"] + ":critic")
            try:
                result = critic({**delivery, "planning_stage": "critic", "proposal_pool": pool,
                    "proposal_pool_digest": pool["pool_digest"]})
            except llm.LlmUnavailable as exc:
                pytest.fail(str(getattr(exc, "diagnostics", {})))
    assert result["proposal_pool_digest"] == pool["pool_digest"]
    critic_delivery = json.loads(sdk[-1]["messages"][1]["content"])
    critic_delivery = expand(critic_delivery, critic_delivery)
    assert critic_delivery["experiment_observations"] == delivered["experiment_observations"]
    assert critic_delivery["retrieved_knowledge"] == delivered["retrieved_knowledge"]
    assert vectors.calls == encode_count and len(sdk) == 2


def test_injected_raw_receipt_restores_without_provider_journal(tmp_path):
    camp = make_camp(tmp_path)
    full, state = loop.observation(camp), loop.load_state(camp)
    publish_staged_raw(camp, state, full, "bootstrap:initial_planner", proposal_reply(), "generation")
    before = readonly_snapshot(camp)
    restored = restore_staged_paid(camp, state)
    assert restored[2] == {"generation": proposal_reply()}
    assert restored[3]["source"] == "backend_return_receipt"
    assert readonly_snapshot(camp) == before


@pytest.mark.parametrize("tamper", ["response", "claim", "raw", "goal", "source", "revoked_source"])
def test_paid_stage_rejects_changed_proof_instead_of_paying_again(tmp_path, sdk, tamper):
    camp = make_camp(tmp_path)
    full, owner, stages = paid_stages(camp, receipts=True)
    if tamper == "response":
        response = next((camp / "llm_responses").glob("*.json"))
        value = json.loads(response.read_text())
        value["reply"]["protocol_version"] = "changed"
        response.write_text(json.dumps(value))
    elif tamper == "claim":
        with sqlite3.connect(camp / "resource_owners.sqlite") as db:
            identity, serialized = db.execute("SELECT id,payload FROM claims LIMIT 1").fetchone()
            value = json.loads(serialized)
            value["status"] = "unknown"
            db.execute("UPDATE claims SET payload=? WHERE id=?", (json.dumps(value), identity))
    elif tamper == "raw":
        receipt = next((camp / "research_round_raw").rglob("*.json"))
        value = json.loads(receipt.read_text())
        value["raw"]["protocol_version"] = "changed"
        receipt.write_text(json.dumps(value))
    elif tamper == "goal":
        value = json.loads((camp / "goal.json").read_text())
        value["objective"] = "Changed scientific question"
        (camp / "goal.json").write_text(json.dumps(value))
    else:
        ledger = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text().splitlines()]
        manifest = json.loads(open(ledger[-1]["context_sidecars"]["manifest"]).read())
        source = next(row for row in manifest["sources"] if row["block_id"] == "input:/proposal_pool")
        if tamper == "source":
            (camp / source["payload_ref"]).write_text('{"changed":true}')
        else:
            with sqlite3.connect(camp / "context_sources.sqlite") as db:
                db.execute("UPDATE context_sources SET revoked=1 WHERE source_ref=?", (source["source_ref"],))
    before = readonly_snapshot(camp)
    with pytest.raises(ValueError, match="paid_(staged_)?round_"):
        restore_staged_paid(camp, loop.load_state(camp))
    assert len(sdk) == 2 and readonly_snapshot(camp) == before
