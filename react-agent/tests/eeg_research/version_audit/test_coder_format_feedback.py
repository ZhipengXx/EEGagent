"""Malformed coder markup gets a concrete correction within the existing attempt."""
from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import llm, owner_budget
from react_agent.eeg_research.agentic.coder import validate_tool_request
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.fmri.llm.deepseek import DeepSeekBackend, parse_json_object

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2
VALID_ACTION = {"tool": "read_code", "args": {"path": "reference/train_entry.py"}}
INVALID_DSML = (
    '<｜｜DSML｜｜calls><｜｜DSML｜｜invoke name="read_code">'
    '<｜｜DSML｜｜parameter name="path">reference/train_entry.py</｜｜DSML｜｜parameter>'
    '</｜｜DSML｜｜invoke></｜｜DSML｜｜calls>'
)


@pytest.fixture
def sdk(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    def refuse_network(*args, **kwargs):
        pytest.fail("Coder format regression attempted real network access")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    wire, clients = [], []
    responses = []

    def handle(request):
        body = json.loads(request.content)
        wire.append(body)
        assert responses, "Unexpected additional SDK call"
        content = responses.pop(0)
        return HTTP.Response(200, json={"id": "offline-format", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 256, "total_tokens": 356}})

    class MockBackend(DeepSeekBackend):
        def __init__(self, config):
            super().__init__(config)
            hooks = {key: list(value) for key, value in self._client._client.event_hooks.items()}
            asyncio.run(self._client.close())
            self._client = AsyncOpenAI(api_key="offline-unused", base_url="https://offline.invalid", max_retries=0,
                http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(handle), trust_env=False,
                                             event_hooks=hooks))
            self._client._platform = "Linux"
            clients.append(self)

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", MockBackend)
    yield wire, responses
    for client in clients:
        asyncio.run(client.aclose())


def campaign(path, role="candidate_coder"):
    goal = {"max_llm_calls": 8, "research_execution": {"version": "single_plan_v1"},
        "memory": {"enabled": False}, "role_models": {role: {"model": "synthetic-format",
            "thinking": False, "max_tokens": 8192}}}
    for name, value in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                       ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (path / name).write_text(json.dumps(value))
    return {"candidate_source_presence": "missing", "current_file_delivery": "absent",
            "current_file": None}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_coder_dsml_retry_delivers_specific_feedback_and_preserves_paid_failure(tmp_path, sdk):
    wire, responses = sdk
    malformed = INVALID_DSML + " rejected output " * 300
    responses.extend([malformed, json.dumps(VALID_ACTION)])
    payload = campaign(tmp_path)
    backend = llm.role_backend(tmp_path, "candidate_coder")
    assert backend(payload) == VALID_ACTION

    assert len(wire) == 2
    first = json.loads(wire[0]["messages"][1]["content"])
    assert "previous_response_error" not in first
    retry = json.loads(wire[1]["messages"][1]["content"])
    feedback = resolve_request_pointer(retry, "/previous_response_error")
    assert feedback["kind"] == "invalid_json"
    assert feedback["rejected_format"] == malformed[:2048]
    assert len(feedback["rejected_format"]) <= 2048
    assert feedback["tool_executed"] is False
    assert feedback["example"]["tool"] == "read_code"
    assert set(feedback["example"]) == {"tool", "args"}
    assert validate_tool_request(feedback["example"])[2] is None
    instruction = feedback["instruction"].lower()
    assert "dsml" in instruction and "json" in instruction
    assert "no tool executed" in instruction

    calls = rows(tmp_path / "llm_calls.jsonl")
    attempts = rows(tmp_path / "api_operation_attempts.jsonl")
    assert len(calls) == len(attempts) == 2
    assert calls[0]["success"] is False
    assert calls[0]["failure_kind"] == "format_error"
    assert calls[0]["transport_outcome"] == "confirmed_response"
    assert calls[0]["input_tokens"] == 100 and calls[0]["output_tokens"] == 256
    failed_response = json.loads((tmp_path / "llm_responses" / (calls[0]["call_id"] + ".json")).read_text())
    assert failed_response["validation"] == "format_error"
    assert failed_response["reply"] == malformed[:2048]
    assert calls[1]["application_retry_kind"] == "invalid_json"
    assert all(row["coder_format_feedback_revision"] == llm.CODER_FORMAT_FEEDBACK_REVISION for row in calls)
    assert [row["application_attempt"] for row in calls] == [1, 2]
    assert all(row["max_operation_attempts"] == 2 for row in attempts)
    spent = owner_budget.snapshot(tmp_path)
    assert spent["global_spent_and_occupied"]["llm_calls"] == 2
    assert spent["unknown"] == []

    # A paid, valid second reply is reused natively; format repair does not
    # authorize a new logical coding step or an extra request on replay.
    assert backend(payload) == VALID_ACTION
    assert len(wire) == 2
    assert len(rows(tmp_path / "llm_calls.jsonl")) == 2


def test_repeated_coder_markup_cannot_grow_existing_operation_attempt_limit(tmp_path, sdk):
    wire, responses = sdk
    responses.extend([INVALID_DSML, INVALID_DSML])
    payload = campaign(tmp_path)
    backend = llm.role_backend(tmp_path, "candidate_coder")
    with pytest.raises(llm.LlmUnavailable, match="DeepSeekParseError"):
        backend(payload)
    assert len(wire) == 2
    assert all(row["max_operation_attempts"] == 2 for row in rows(tmp_path / "api_operation_attempts.jsonl"))
    assert all(row["success"] is False for row in rows(tmp_path / "llm_calls.jsonl"))


@pytest.mark.parametrize("raw", [INVALID_DSML, INVALID_DSML + json.dumps(VALID_ACTION),
                                 json.dumps(VALID_ACTION) + INVALID_DSML,
                                 json.dumps(VALID_ACTION) + json.dumps(VALID_ACTION)])
def test_parser_refuses_markup_or_multiple_actions_without_guessing_a_tool(raw):
    with pytest.raises(json.JSONDecodeError):
        parse_json_object(raw)


def test_non_coder_parse_retry_keeps_generic_feedback(tmp_path, sdk):
    wire, responses = sdk
    ready = {"status": "ready", "issues": [], "summary_zh": "Synthetic review."}
    responses.extend([INVALID_DSML, json.dumps(ready)])
    payload = campaign(tmp_path, "candidate_reviewer")
    assert llm.role_backend(tmp_path, "candidate_reviewer")(payload)["status"] == "ready"
    assert len(wire) == 2
    retry = json.loads(wire[1]["messages"][1]["content"])
    feedback = resolve_request_pointer(retry, "/previous_response_error")
    assert feedback["kind"] == "invalid_json"
    assert "rejected_format" not in feedback and "example" not in feedback
