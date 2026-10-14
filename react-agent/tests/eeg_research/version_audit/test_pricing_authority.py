"""Frozen USD authority through the actual SDK and local HTTP transport only."""
from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import api_policy, llm, owner_budget as budget
from react_agent.eeg_research.agentic.pricing import estimate_usd, resolve_pricing
from react_agent.fmri.config import FmriCheckConfig, LmProfile
from react_agent.fmri.llm.deepseek import DeepSeekBackend, DeepSeekCallError, DeepSeekConfigError

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_args, **_kwargs: False)
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)
    # Native API admission is enabled solely for this in-process MockTransport.
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    def no_network(*_args, **_kwargs):
        raise AssertionError("pricing regression attempted a real socket")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)


def price(**updates):
    return {"source": "synthetic-regression", "date": "2026-10-14",
        "input_per_million": 2.0, "output_per_million": 3.0,
        "cache_hit_per_million": 0.2, **updates}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def campaign(path, *, pricing=True, cap=1.0, reserve=None):
    cfg = {"pricing": price()} if pricing else {}
    if reserve is not None:
        cfg["conservative_usd_per_request"] = reserve
    write(path / "goal.json", {"max_llm_calls": 10, "max_api_usd": cap,
        "research_execution": {"version": "single_plan_v1", "budget": cfg}})
    write(path / "campaign_state.json", {"status": "planning"})
    write(path / "cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})
    return path


def change_price(camp, **changes):
    goal = json.loads((camp / "goal.json").read_text())
    goal["research_execution"]["budget"]["pricing"].update(changes)
    write(camp / "goal.json", goal)


def env_price(monkeypatch, value):
    for field, name in {"source": "DEEPSEEK_PRICE_SOURCE", "date": "DEEPSEEK_PRICE_DATE",
        "input_per_million": "DEEPSEEK_USD_INPUT_PER_MILLION",
        "output_per_million": "DEEPSEEK_USD_OUTPUT_PER_MILLION",
        "cache_hit_per_million": "DEEPSEEK_USD_CACHE_HIT_PER_MILLION", "model": "DEEPSEEK_PRICE_MODEL"}.items():
        if field in value:
            monkeypatch.setenv(name, str(value[field]))


def response(*, inp=1000, out=100, cache=500, usage=True):
    data = {"id": "local", "object": "chat.completion", "created": 0, "model": "model-A",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true}'}}]}
    if usage:
        data["usage"] = {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out,
                         "prompt_cache_hit_tokens": cache}
    return data


def client(camp, seen, *, result=None, on_http=None):
    backend = DeepSeekBackend(FmriCheckConfig(deepseek_api_key="synthetic-unused",
        deepseek_base_url="https://mock.invalid", deepseek_trust_env=False,
        fast=LmProfile(model="model-A", max_tokens=32, thinking=False)))
    backend.enable_budget_protocol(camp)
    async def replace():
        request_hooks = list(backend._client._client.event_hooks["request"])
        follow_redirects = backend._client._client.follow_redirects
        await backend._client.close()
        def handle(request):
            seen.append(json.loads(request.content))
            if on_http:
                on_http()
            data = response() if result is None else result
            if isinstance(data, Exception):
                raise data
            if isinstance(data, HTTP.Response):
                return data
            return HTTP.Response(200, json=data)
        backend._client = AsyncOpenAI(api_key="synthetic-unused", base_url="https://mock.invalid", max_retries=0,
            http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(handle), trust_env=False,
                event_hooks={"request": request_hooks}, follow_redirects=follow_redirects))
        backend._client._platform = "Linux"
    asyncio.run(replace())
    return backend


def send(backend, *, user="{}"):
    return asyncio.run(backend.complete_json(system="Return JSON", user=user,
        profile="fast", role="backend_direct"))


@pytest.mark.parametrize("mutation", ["price", "model"])
def test_last_transport_boundary_rejects_changes_after_preflight(tmp_path, monkeypatch, mutation):
    camp = campaign(tmp_path)
    seen = []
    backend = client(camp, seen)
    before = budget.before_transport
    def mutate_after_check(*args, **kwargs):
        before(*args, **kwargs)
        if mutation == "price":
            change_price(camp, input_per_million=20.0)
        else:
            backend.config.fast.model = "model-B"
    monkeypatch.setattr(budget, "before_transport", mutate_after_check)
    with pytest.raises(DeepSeekConfigError, match="pricing_identity_changed"):
        send(backend)
    assert seen == []
    claims = budget.snapshot(camp)
    assert claims["global_spent_and_occupied"]["llm_calls"] == 1
    assert claims["global_spent_and_occupied"]["api_usd"] == 0
    # A distinct admission freezes the changed identity; the rejected claim is
    # retained as an attempt and cannot authorize the changed request.
    _, usage = send(backend)
    assert len(seen) == 1
    frozen = budget.request_pricing(camp, usage.call_id)
    assert frozen == resolve_pricing(camp, model=backend.config.fast.model)
    assert seen[0]["model"] == backend.config.fast.model
    assert budget.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 2


def test_cached_charge_survives_lost_outer_receipt_and_late_price_change(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    seen = []
    backend = client(camp, seen)
    original = llm._ledger
    def lost_receipt(*_args, **_kwargs):
        raise RuntimeError("synthetic_outer_receipt_crash")
    monkeypatch.setattr(llm, "_ledger", lost_receipt)
    with pytest.raises(RuntimeError, match="outer_receipt_crash"):
        send(backend)
    intent = json.loads((camp / "llm_api_intents.jsonl").read_text().splitlines()[0])
    identifier = intent["call_id"]
    known = budget.settled_provider_usage(camp, identifier)
    assert known["api_usd"] == pytest.approx(0.0014)
    monkeypatch.setattr(llm, "_ledger", original)
    change_price(camp, input_per_million=200.0)
    api_policy.dispose_unknown_intent(camp, identifier, reason="Synthetic lost receipt", allow_new_request=False)
    assert len(seen) == 1
    recovered = budget.settled_provider_usage(camp, identifier)
    assert recovered["cache_hit_tokens"] == 500
    assert recovered["api_usd"] == pytest.approx(0.0014)
    snapshot = budget.snapshot(camp)
    assert snapshot["global_spent_and_occupied"]["api_usd"] == pytest.approx(0.0014)
    assert snapshot["global_spent_and_occupied"]["llm_calls"] == 1


@pytest.mark.parametrize("mutation", ["price", "http_model"])
def test_sdk_preparation_change_is_checked_at_actual_http_request(tmp_path, monkeypatch, mutation):
    camp = campaign(tmp_path)
    seen = []
    backend = client(camp, seen)
    create = backend._client.chat.completions.create
    async def mutate_inside_sdk(**kwargs):
        if mutation == "price":
            change_price(camp, input_per_million=20)
        else:
            kwargs["model"] = "model-B"
        return await create(**kwargs)
    monkeypatch.setattr(backend._client.chat.completions, "create", mutate_inside_sdk)
    with pytest.raises(DeepSeekConfigError, match="pricing_identity_changed"):
        send(backend)
    assert seen == []
    snap = budget.snapshot(camp)
    assert snap["global_spent_and_occupied"]["llm_calls"] == 1
    assert snap["global_spent_and_occupied"]["api_usd"] == 0


def test_http_redirect_cannot_resend_one_permit_or_refund_sent_claim(tmp_path):
    camp = campaign(tmp_path)
    seen = []
    backend = client(camp, seen, result=HTTP.Response(307, headers={"location": "https://mock.invalid/redirected"}))
    with pytest.raises(DeepSeekCallError):
        send(backend)
    snap = budget.snapshot(camp)
    assert len(seen) == len(snap["unknown"]) == 1
    assert snap["global_spent_and_occupied"]["api_usd"] > 0
    assert snap["global_spent_and_occupied"]["output_tokens"] == 32
    assert snap["global_spent_and_occupied"]["llm_calls"] == 1


def test_actual_http_boundary_rechecks_global_cap(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    seen = []
    backend = client(camp, seen)
    create = backend._client.chat.completions.create
    async def lower_cap_inside_sdk(**kwargs):
        goal = json.loads((camp / "goal.json").read_text())
        goal["max_api_usd"] = 1e-12
        write(camp / "goal.json", goal)
        return await create(**kwargs)
    monkeypatch.setattr(backend._client.chat.completions, "create", lower_cap_inside_sdk)
    with pytest.raises(DeepSeekConfigError, match="global_budget_occupied_exceeds_cap:api_usd"):
        send(backend)
    assert seen == []
    assert budget.snapshot(camp)["global_spent_and_occupied"]["api_usd"] == 0


def test_existing_legacy_goal_keeps_legacy_transport(tmp_path):
    camp = campaign(tmp_path, cap=0)
    goal = json.loads((camp / "goal.json").read_text())
    goal.pop("research_execution")
    write(camp / "goal.json", goal)
    seen = []
    send(client(camp, seen))
    assert len(seen) == 1 and not (camp / "resource_owners.sqlite").exists()


@pytest.mark.parametrize("configuration", ["conflicting_environment", "incomplete", "negative", "nan", "wrong_model", "wrong_provider"])
def test_pricing_conflicts_are_rejected_before_any_http(tmp_path, monkeypatch, configuration):
    camp = campaign(tmp_path)
    if configuration == "conflicting_environment":
        env_price(monkeypatch, price(input_per_million=20))
    elif configuration == "incomplete":
        goal = json.loads((camp / "goal.json").read_text())
        del goal["research_execution"]["budget"]["pricing"]["output_per_million"]
        write(camp / "goal.json", goal)
    elif configuration == "negative":
        change_price(camp, input_per_million=-1)
    elif configuration == "nan":
        change_price(camp, input_per_million=float("nan"))
    elif configuration == "wrong_model":
        change_price(camp, model="model-B")
    else:
        change_price(camp, provider="another-provider")
    seen = []
    backend = client(camp, seen)
    with pytest.raises(llm.LlmUnavailable, match="pricing_"):
        send(backend)
    assert seen == []
    assert budget.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 0


@pytest.mark.parametrize("origin", ["goal", "environment", "matching_both"])
def test_reserve_send_and_settle_use_one_price_identity(tmp_path, monkeypatch, origin):
    camp = campaign(tmp_path, pricing=origin != "environment")
    if origin != "goal":
        env_price(monkeypatch, price())
    seen = []
    _, usage = send(client(camp, seen))
    claim_price = budget.request_pricing(camp, usage.call_id)
    assert claim_price == resolve_pricing(camp, model="model-A")
    assert len(seen) == 1 and seen[0]["model"] == "model-A"
    proof = budget.settled_provider_usage(camp, usage.call_id)
    assert proof["cache_hit_tokens"] == 500
    assert proof["api_usd"] == pytest.approx(estimate_usd(claim_price, input_tokens=1000,
        output_tokens=100, cache_hit_tokens=500))
    ledger = json.loads((camp / "llm_calls.jsonl").read_text().splitlines()[0])
    assert ledger["frozen_pricing"] == claim_price
    assert ledger["pricing_fingerprint"] == claim_price["fingerprint"]
    assert ledger["cost_status"] == "priced"


def test_response_settles_original_price_after_configuration_changes(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    seen = []
    def mutate_after_http():
        change_price(camp, input_per_million=200)
        env_price(monkeypatch, price(input_per_million=300))
    _, usage = send(client(camp, seen, on_http=mutate_after_http))
    proof = budget.settled_provider_usage(camp, usage.call_id)
    assert len(seen) == 1 and proof["api_usd"] == pytest.approx(0.0014)
    assert budget.request_pricing(camp, usage.call_id)["pricing"]["input_per_million"] == 2
    with pytest.raises(llm.LlmUnavailable, match="pricing_configuration_conflict"):
        send(client(camp, seen), user='{"next": true}')
    assert len(seen) == 1


@pytest.mark.parametrize("mutation", ["price", "model", "provider"])
def test_existing_claim_cannot_be_rebound_to_changed_identity(tmp_path, mutation):
    camp = campaign(tmp_path)
    args = {"call_id": "same", "operation": "same-operation", "role": "backend_direct",
        "input_bound": 1000, "output_bound": 100, "model": "model-A"}
    first = budget.claim_request(camp, **args)
    assert budget.claim_request(camp, **args) == first
    if mutation == "price":
        change_price(camp, input_per_million=20)
    elif mutation == "model":
        args["model"] = "model-B"
    else:
        args["provider"] = "another-provider"
    with pytest.raises(budget.OwnerBudgetError, match="claim_identity_conflict|pricing_provider_or_model_mismatch"):
        budget.claim_request(camp, **args)
    assert budget.request_pricing(camp, "same") == first["frozen_pricing"]


@pytest.mark.parametrize("field,value", [("provider", "another-provider"), ("requested_model", "model-B")])
def test_settlement_cannot_change_frozen_provider_or_model(tmp_path, field, value):
    camp = campaign(tmp_path)
    budget.claim_request(camp, call_id="one", operation="op", role="backend_direct", input_bound=1000,
        output_bound=100, model="model-A")
    usage = {"input_tokens": 20, "output_tokens": 8, "transport_outcome": "confirmed_response", field: value}
    with pytest.raises(budget.OwnerBudgetError, match="settlement_provider_or_model_identity_conflict"):
        budget.settle_request(camp, "one", usage)
    assert budget.settled_provider_usage(camp, "one") is None


@pytest.mark.parametrize("reserve,reason", [(None, "unpriced_request_deferred"), (0, "positive_unknown_reserve")])
def test_unknown_price_with_usd_cap_requires_positive_reserve(tmp_path, reserve, reason):
    camp = campaign(tmp_path, pricing=False, reserve=reserve)
    seen = []
    with pytest.raises(llm.LlmUnavailable, match=reason):
        send(client(camp, seen))
    assert seen == []
    assert budget.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 0


def test_unknown_price_without_cap_stays_unpriced(tmp_path):
    camp = campaign(tmp_path, pricing=False, cap=None)
    seen = []
    _, usage = send(client(camp, seen))
    proof = budget.settled_provider_usage(camp, usage.call_id)
    assert proof["api_usd"] is None
    assert budget.request_pricing(camp, usage.call_id)["pricing"] is None
    record = json.loads((camp / "llm_calls.jsonl").read_text().splitlines()[0])
    assert record["api_usd"] is None and record["cost_status"] == "unpriced"
    assert budget.snapshot(camp)["unpriced_call_ids"] == [usage.call_id]
    assert len(seen) == 1


def test_unknown_usage_keeps_conservative_usd_occupied(tmp_path):
    camp = campaign(tmp_path, pricing=False, cap=0.015, reserve=0.01)
    seen = []
    backend = client(camp, seen, result=response(usage=False))
    _, usage = send(backend)
    proof = budget.settled_provider_usage(camp, usage.call_id)
    assert proof["api_usd"] is None
    snap = budget.snapshot(camp)
    assert snap["global_spent_and_occupied"]["api_usd"] == pytest.approx(0.01)
    assert snap["conservative_resource_remainders"]["api_usd"] == pytest.approx(0.01)
    with pytest.raises(llm.LlmUnavailable, match="auxiliary_request_deferred:api_usd"):
        send(backend, user='{"next": true}')
    assert len(seen) == 1


def test_uncertain_http_never_refunds_conservative_claim(tmp_path):
    camp = campaign(tmp_path, pricing=False, reserve=0.01)
    seen = []
    backend = client(camp, seen, result=HTTP.ReadTimeout("local uncertain response"))
    with pytest.raises(DeepSeekCallError):
        send(backend)
    snap = budget.snapshot(camp)
    assert len(seen) == len(snap["unknown"]) == 1
    assert snap["global_spent_and_occupied"]["api_usd"] == pytest.approx(0.01)
    assert snap["global_spent_and_occupied"]["output_tokens"] == 32
    with pytest.raises(llm.LlmUnavailable, match="requires_recovery"):
        send(backend, user='{"next": true}')
    assert len(seen) == 1


def test_cache_rate_above_regular_input_is_reserved_at_upper_bound(tmp_path):
    camp = campaign(tmp_path)
    change_price(camp, cache_hit_per_million=5)
    claim = budget.claim_request(camp, call_id="bound", operation="bound", role="backend_direct",
        input_bound=1000, output_bound=100, model="model-A")
    assert claim["requested"]["api_usd"] == pytest.approx(0.0053)
    usage = {"input_tokens": 1000, "output_tokens": 100, "cache_hit_tokens": 1000,
        "transport_outcome": "confirmed_response"}
    budget.settle_request(camp, "bound", usage)
    budget.settle_request(camp, "bound", usage)
    assert budget.snapshot(camp)["global_spent_and_occupied"]["api_usd"] == pytest.approx(0.0053)


def test_real_provider_overrun_is_recorded_and_blocks_next_http(tmp_path):
    camp = campaign(tmp_path, cap=0.001)
    seen = []
    backend = client(camp, seen, result=response(inp=10000, cache=0))
    _, usage = send(backend)
    snapshot = budget.snapshot(camp)
    assert len(seen) == 1
    assert snapshot["global_spent_and_occupied"]["api_usd"] == pytest.approx(0.0203)
    assert snapshot["global_spent_and_occupied"]["api_usd"] > snapshot["global_caps"]["api_usd"]
    with budget._transaction(camp) as conn:
        assert budget._get(conn, "meta", "overrun")["call_id"] == usage.call_id
    record = json.loads((camp / "cost.json").read_text())
    assert record["api_usd"] == pytest.approx(0.0203)
    with pytest.raises(llm.LlmUnavailable, match="global_budget_occupied_exceeds_cap:api_usd"):
        send(backend, user='{"next": true}')
    assert len(seen) == 1
