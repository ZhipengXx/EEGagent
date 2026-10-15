"""SDK exception wrapping must preserve the native HTTP admission outcome."""
from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import owner_budget
from react_agent.fmri.config import FmriCheckConfig, LmProfile
from react_agent.fmri.llm.deepseek import (
    DeepSeekBackend, DeepSeekBudgetBoundaryError, DeepSeekCallError, DeepSeekOutputError,
)

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    def refuse_network(*args, **kwargs):
        pytest.fail("SDK compatibility fixture forbids sockets and DNS")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    goal = {"max_llm_calls": 4, "max_api_usd": 1,
        "research_execution": {"version": "single_plan_v1", "budget": {"pricing": {
            "source": "synthetic-compatibility", "date": "2026-10-14",
            "input_per_million": 2, "output_per_million": 3}}}}
    for name, body in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                       ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (tmp_path / name).write_text(json.dumps(body))
    backend = DeepSeekBackend(FmriCheckConfig(deepseek_api_key="synthetic-unused",
        deepseek_base_url="https://offline.invalid", deepseek_trust_env=False,
        fast=LmProfile(model="synthetic-model", max_tokens=32, thinking=False)))
    backend.enable_budget_protocol(tmp_path)
    calls = []
    async def replace_client():
        hooks = {key: list(value) for key, value in backend._client._client.event_hooks.items()}
        await backend._client.close()
        def handle(request):
            calls.append(request)
            raise HTTP.ReadTimeout("synthetic unknown network outcome", request=request)
        backend._client = AsyncOpenAI(api_key="synthetic-unused", base_url="https://offline.invalid", max_retries=0,
            http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(handle), trust_env=False, event_hooks=hooks))
        backend._client._platform = "Linux"
    asyncio.run(replace_client())
    yield tmp_path, goal, backend, calls
    asyncio.run(backend.aclose())


def send(backend):
    return asyncio.run(backend.complete_json(system="Return JSON", user="{}", profile="fast", role="backend_direct"))


@pytest.mark.parametrize("shape", ["boundary_itself", "sdk_nested_cause"])
def test_http_admission_refusal_survives_sdk_exception_shape(transport, monkeypatch, shape):
    camp, goal, backend, calls = transport
    original_hook = backend._budget_http_boundary
    async def nested_hook(request):
        try:
            await original_hook(request)
        except DeepSeekBudgetBoundaryError as error:
            raise RuntimeError("synthetic SDK transport wrapper") from error
    if shape == "sdk_nested_cause":
        backend._client._client.event_hooks["request"] = [nested_hook]
    original_create = backend._client.chat.completions.create
    async def create(**kwargs):
        goal["research_execution"]["budget"]["pricing"]["input_per_million"] = 20
        (camp / "goal.json").write_text(json.dumps(goal))
        try:
            return await original_create(**kwargs)
        except Exception as error:
            if shape == "boundary_itself":
                cause = error
                while cause is not None:
                    if isinstance(cause, DeepSeekBudgetBoundaryError):
                        raise cause
                    cause = cause.__cause__
            raise
    monkeypatch.setattr(backend._client.chat.completions, "create", create)
    with pytest.raises(DeepSeekBudgetBoundaryError, match="pricing_identity_changed") as refused:
        send(backend)
    assert refused.value.definitely_not_sent
    assert calls == []
    state = owner_budget.snapshot(camp)
    assert state["global_spent_and_occupied"]["llm_calls"] == 1
    assert state["global_spent_and_occupied"]["api_usd"] == 0
    assert state["unknown"] == []


def test_actual_sdk_unknown_read_timeout_keeps_its_occupied_claim(transport):
    camp, _goal, backend, calls = transport
    with pytest.raises(DeepSeekCallError) as failed:
        send(backend)
    assert len(calls) == 1
    assert failed.value.usage.transport_outcome == "outcome_unknown"
    assert failed.value.usage.unconfirmed_charge_possible
    state = owner_budget.snapshot(camp)
    assert len(state["unknown"]) == 1
    assert state["global_spent_and_occupied"]["api_usd"] > 0
    assert state["global_spent_and_occupied"]["output_tokens"] == 32
    assert state["global_spent_and_occupied"]["llm_calls"] == 1


def test_actual_sdk_length_failure_retains_complete_final_content_and_usage(transport):
    camp, _goal, backend, calls = transport
    backend.config.fast.strict_final_output = True
    content = json.dumps({"status": "ready", "summary": "synthetic final content " * 400})
    assert len(content) > 2048
    def length_response(request):
        calls.append(request)
        return HTTP.Response(200, json={"id": "synthetic-length", "object": "chat.completion", "created": 0,
            "model": "synthetic-model", "choices": [{"index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": content,
                            "reasoning_content": "synthetic reasoning stays separate from final content"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 32, "total_tokens": 132,
                      "completion_tokens_details": {"reasoning_tokens": 24}}})
    backend._client._client._transport = HTTP.MockTransport(length_response)
    with pytest.raises(DeepSeekOutputError) as truncated:
        send(backend)
    assert truncated.value.raw == content
    usage = truncated.value.usage
    assert not usage.success and usage.status == "failed"
    assert usage.failure_kind == usage.error == "output_truncated"
    assert usage.finish_reason == "length" and usage.transport_outcome == "confirmed_response"
    assert not usage.unconfirmed_charge_possible
    assert usage.input_tokens == 100 and usage.output_tokens == 32 and usage.reasoning_tokens == 24
    assert len(calls) == 1
    state = owner_budget.snapshot(camp)
    assert state["unknown"] == []
    assert state["global_spent_and_occupied"]["llm_calls"] == 1
