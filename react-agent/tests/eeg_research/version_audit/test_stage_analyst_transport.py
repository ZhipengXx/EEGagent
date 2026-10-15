"""Native Analyst task identity and actual SDK delivery stay stage specific."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import llm, roles, selected_plan
from react_agent.eeg_research.agentic.context.budget import compact_schema
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.eeg_research.agentic.runtime_prompts import ROOT
from react_agent.eeg_research.agentic.schemas import LegacyResultAnalysis, ResultAnalysis
from react_agent.eeg_research.agentic.task_ledger import latest
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


def expanded(value, root):
    if isinstance(value, dict):
        if "$context_pointer" in value:
            return expanded(resolve_request_pointer(root, value["$context_pointer"]), root)
        return {key: expanded(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [expanded(child, root) for child in value]
    return value


@pytest.fixture
def analyst_sdk(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    def forbidden(*args, **kwargs):
        pytest.fail("Analyst transport regression attempted network or training")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.start_job", forbidden)
    monkeypatch.setattr("react_agent.eeg_research.agentic.native_patch.implement", forbidden)
    clients, wire = [], []

    def transport(request):
        body = json.loads(request.content)
        wire.append(body)
        raw = json.loads(body["messages"][1]["content"])
        payload = expanded(raw, raw)
        reply = {"execution_assessment": "Verified native two-fold screening completed.",
            "hypothesis_assessment": "weakened", "observations": [{"delta_pp": -1.0}],
            "scope_limits": ["Two fixed development folds and one seed."],
            "evidence_refs": ["ev_native_screening"], "summary_zh": "有效负结果仅适用于两折开发筛查。"}
        if payload.get("review_stage") == "analysis":
            reply["analysis_requirement_coverage"] = [{"requirement_id": "bounded_interpretation",
                "status": "implemented", "detail": "The negative result is interpreted within its measured scope.",
                "evidence_refs": ["ev_native_screening"]}]
        return HTTP.Response(200, json={"id": "offline-stage-analyst", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 80, "total_tokens": 160}})

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
    yield wire
    for client in clients:
        asyncio.run(client.aclose())


def campaign(camp):
    goal = {"max_llm_calls": 4, "memory": {"enabled": False},
        "research_execution": {"version": "single_plan_v1", "design_contract_version": "v2"},
        "role_models": {"result_analyst": {"model": "synthetic-stage-analyst", "thinking": False,
            "max_tokens": 4096}}}
    for name, value in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                       ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (camp / name).write_text(json.dumps(value))
    receipt = camp / "synthetic_measurement.json"
    receipt.write_text('{"scope":"two_fold_development","delta_pp":-1.0}')
    return receipt


@pytest.mark.parametrize("staged", [False, True], ids=["legacy_v2", "staged_v2"])
def test_analyst_wire_prompt_schema_identity_and_completed_task_reuse(tmp_path, analyst_sdk, monkeypatch, staged):
    receipt = campaign(tmp_path)
    proposal = {"question": "What does this paired screening show?", "scientific_intent": "Interpret the measured delta",
        "task_description": "FUTURE_TASK_DESCRIPTION", "expected_observation": "FUTURE_EXPECTED_OBSERVATION",
        "completion_condition": "FUTURE_COMPLETION_CONDITION", "evaluation_scope": {"phase": "screening_2fold"}}
    selected = {"plan_id": "selected_transport_plan", "proposal": proposal}
    monkeypatch.setattr(selected_plan, "load_plan", lambda *args: copy.deepcopy(selected))
    request = {"analysis_trigger": {"kind": "completed_screening"},
        "latest": {"candidate_id": "c1", "fidelity": "full", "evidence_id": "ev_native_screening"},
        "hypothesis": "The candidate improves the paired screening score.",
        "run_identity": {"source_hash": "native_source", "spec_hash": "native_spec", "result_hash": "native_result"},
        "comparison": {"comparable": True, "delta_pp": -1.0},
        "selected_research_task": {key: proposal[key] for key in ("question", "scientific_intent")}}
    if staged:
        request.update(review_stage="analysis", analysis_requirements=[{
            "requirement_id": "bounded_interpretation", "verification": "scope_limits", "core": True,
            "expected_behavior": "Interpret this two-fold negative result without claiming ten-fold evidence."}],
            measurement_review={"review_id": "stage_measurement", "status": "completed",
                "artifact_id": "art_native_measurement", "evidence_refs": ["ev_native_screening"]})
    task = roles.begin_role_task(tmp_path, role="result_analyst", inputs=[receipt], candidate_id="c1", request=request)
    payload = {**request, **{key: task[key] for key in ("task_id", "attempt_id", "input_digest")}}
    backend = llm.role_backend(tmp_path, "result_analyst")
    first = backend(payload)
    assert len(analyst_sdk) == 1 and first["status"] == "completed"
    wire = analyst_sdk[0]
    system = wire["messages"][0]["content"]
    raw = json.loads(wire["messages"][1]["content"])
    delivered = expanded(raw, raw)
    schema, _ = json.JSONDecoder().raw_decode(system.split("OUTPUT_SCHEMA\n", 1)[1].lstrip())
    model = ResultAnalysis if staged else LegacyResultAnalysis
    assert schema == compact_schema(model.model_json_schema())
    assert llm.domain_model_for("result_analyst", camp=tmp_path,
        review_stage="analysis" if staged else None) is model
    mission_root = ROOT.parent / "runtime_v2" if staged else ROOT
    assert (mission_root / "result_analyst.txt").read_text().strip() in system
    assert ("analysis_requirement_coverage" in schema["properties"]) is staged
    assert ("analysis_requirement_coverage" in first["payload"]) is staged
    if staged:
        assert delivered["selected_research_task"] == {key: proposal[key] for key in ("question", "scientific_intent")}
        assert not any(marker in json.dumps(delivered) for marker in (
            "FUTURE_TASK_DESCRIPTION", "FUTURE_EXPECTED_OBSERVATION", "FUTURE_COMPLETION_CONDITION"))
        assert delivered["analysis_requirements"] == request["analysis_requirements"]
        assert first["payload"]["analysis_requirement_coverage"][0]["status"] == "implemented"
    else:
        assert delivered["selected_research_task"] == proposal
    wire_prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
    assert task["recovery_identity"]["prompt_hash"] == first["prompt_hash"] == wire_prompt_hash
    assert task["recovery_identity"]["schema_hash"] == roles.request_digest(request=model.model_json_schema())
    calls = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["prompt_hash"] == wire_prompt_hash
    saved = roles.finish_role_task(tmp_path, task, first, kind="analysis",
        path=tmp_path / "analyses/transport.json", candidate_id="c1")
    assert latest(tmp_path, task["task_id"])["status"] == "completed"
    reused = roles.begin_role_task(tmp_path, role="result_analyst", inputs=[receipt], candidate_id="c1", request=request)
    assert reused["task_id"] == task["task_id"] and reused.get("_cached_role_result")
    repeated = backend({**request, **{key: reused[key] for key in ("task_id", "attempt_id", "input_digest")}})
    assert repeated == saved and len(analyst_sdk) == 1
    assert len((tmp_path / "llm_calls.jsonl").read_text().splitlines()) == 1
