"""Opt-in resource owners at the actual provider and native phase boundaries.

Allocation is authority, claims are durable facts. Children partition their
parent; only top-level unused allocations enter the global admission equation.
The pre-existing cost/attempt journals are never reset or refunded.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

VERSION = "single_plan_v1"
DIMENSIONS = ("llm_calls", "input_tokens", "output_tokens", "total_tokens", "api_usd", "training_jobs", "gpu_seconds")
CAPS = {"llm_calls": "max_llm_calls", "input_tokens": "max_input_tokens", "output_tokens": "max_output_tokens", "total_tokens": "max_total_tokens", "api_usd": "max_api_usd", "training_jobs": "max_training_jobs", "gpu_seconds": "max_gpu_seconds"}
OWNER = contextvars.ContextVar("eeg_resource_owner", default=None)
TRANSPORT = contextvars.ContextVar("eeg_provider_claim", default=None)


class OwnerBudgetError(ValueError):
    pass


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def enabled(camp):
    return (_read(Path(camp) / "goal.json").get("research_execution") or {}).get("version") == VERSION


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def resources(value=None):
    value = value or {}
    result = {}
    for key in DIMENSIONS:
        number = value.get(key, 0)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise OwnerBudgetError("budget_resource_invalid:" + key)
        if key not in {"api_usd", "gpu_seconds"} and int(number) != number:
            raise OwnerBudgetError("budget_resource_not_integer:" + key)
        result[key] = number
    return result


def _journal(camp, name):
    path = camp / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _native_ids(camp):
    return {str(row["call_id"]) for name in ("llm_calls.jsonl", "llm_api_intents.jsonl", "api_operation_attempts.jsonl") for row in _journal(camp, name) if row.get("call_id")}


def _native_used(camp):
    cost, state = _read(camp / "cost.json"), _read(camp / "campaign_state.json")
    result = resources({"llm_calls": max(int(cost.get("llm_calls") or 0), int(state.get("llm_calls") or 0), len(_native_ids(camp))),
        "input_tokens": cost.get("input_tokens") or 0, "output_tokens": cost.get("output_tokens") or 0,
        "total_tokens": (cost.get("input_tokens") or 0) + (cost.get("output_tokens") or 0),
        "api_usd": cost.get("api_usd") if isinstance(cost.get("api_usd"), (int, float)) else cost.get("api_usd_known_subtotal") or 0,
        "training_jobs": max(int(cost.get("training_jobs") or 0), int(state.get("training_jobs") or 0), len(list((camp / "jobs").glob("*/launch_intent.json")))),
        "gpu_seconds": cost.get("gpu_seconds_used") or 0})
    return result


@contextlib.contextmanager
def _transaction(camp):
    camp = Path(camp)
    camp.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(camp / "resource_owners.sqlite", timeout=30, isolation_level=None)
    try:
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("CREATE TABLE IF NOT EXISTS owners (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM meta WHERE id='baseline'").fetchone():
            initial=_native_used(camp)
            prior=_read(camp/"cost.json")
            native_rows={str(row.get("call_id") or ("legacy_row_"+str(index))):row for index,row in enumerate(_journal(camp,"llm_calls.jsonl"))}
            coverage={key:max(int(prior.get(key+"_known_call_count") or 0),sum(type(row.get(key)) is int for row in native_rows.values())) for key in ("input_tokens","output_tokens")}
            _put(conn, "meta", "baseline", {"used": initial, "call_ids": sorted(_native_ids(camp)),
                "token_known_counts":coverage,"initial_usage":{key:{field:row.get(field) for field in ("input_tokens","output_tokens","api_usd")} for key,row in native_rows.items()},
                "usd_priced":isinstance(prior.get("api_usd"),(int,float)),"at":time.time()})
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _get(conn, table, identity):
    row = conn.execute("SELECT payload FROM " + table + " WHERE id=?", (identity,)).fetchone()
    return json.loads(row[0]) if row else None


def _put(conn, table, identity, value):
    conn.execute("INSERT INTO " + table + " (id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (identity, json.dumps(value, sort_keys=True)))


def _all(conn, table):
    return [json.loads(row[0]) for row in conn.execute("SELECT payload FROM " + table)]


def _control(camp, *, provider=False):
    if provider and os.environ.get("EEG_EXTERNAL_API_DISABLED", "0").lower() in {"1", "true", "on"}:
        raise OwnerBudgetError("external_api_disabled_by_operator")
    state, control = _read(camp / "campaign_state.json"), _read(camp / "control.json")
    if state.get("status") in {"paused", "cancelled", "completed", "stopped"} or state.get("pause_after_step") or control.get("action") in {"pause", "stop", "cancel"}:
        raise OwnerBudgetError("owner_admission_user_stopped")


def _remaining(owner):
    return {key: max(0, owner["allocated"][key] - owner["consumed"][key]) if owner["status"] in {"active", "handoff"} else 0 for key in DIMENSIONS}


def _available(conn, owner):
    result = _remaining(owner)
    for child in _all(conn, "owners"):
        if child.get("parent_id") == owner["owner_id"]:
            for key, amount in _remaining(child).items():
                result[key] -= amount
    return result


def _snapshot(conn, camp, owner_id=None):
    baseline = _get(conn, "meta", "baseline")
    native, claims = _native_used(camp), _all(conn, "claims")
    used = dict(native)
    # cost.llm_calls already contains native intents. One id enters once.
    baseline_ids = set(baseline["call_ids"])
    owner_actual = dict(baseline["used"])
    reserved_unknown = resources()
    for claim in claims:
        if claim["call_id"] in baseline_ids:
            continue
        actual = claim.get("actual") or resources()
        for key in DIMENSIONS:
            owner_actual[key] += actual[key]
            native_probe_accounted = claim["role"]=="engineering_probe" and key=="gpu_seconds" and (
                "engineering_multigpu_"+claim["operation_id"] in (_read(camp/"cost.json").get("settled_job_ids") or []))
            if not native_probe_accounted:
                reserved_unknown[key] += max(0, claim["committed"][key] - actual[key])
    known = {key:max(used[key],owner_actual[key]) for key in DIMENSIONS}
    conservative = dict(reserved_unknown)
    for key in DIMENSIONS:
        used[key] = known[key] + reserved_unknown[key]
    owners = _all(conn, "owners")
    reserved = resources()
    for row in owners:
        if row.get("parent_id") is None:
            for key, amount in _remaining(row).items():
                reserved[key] += amount
    # Legacy phase authority remains effective; attached plan children are
    # already inside their top-level allocation and are not counted again.
    for receipt in (_read(camp/"phase_reservations.json").get("receipts") or {}).values():
        if receipt.get("status")=="active" and not receipt.get("budget_owner"):
            for key,field in (("llm_calls","remaining_llm_calls"),("training_jobs","remaining_training_jobs"),("gpu_seconds","remaining_gpu_seconds")):
                reserved[key]+=receipt.get(field) or 0
    goal = _read(camp / "goal.json")
    config = ((goal.get("research_execution") or {}).get("budget") or {})
    cap = {key: goal.get(name) if goal.get(name) is not None else config.get(name) for key, name in CAPS.items()}
    for key in ("input_tokens","output_tokens"):
        if cap[key] is not None or cap["total_tokens"] is not None:
            missing=_historical_unknown_count(camp,baseline,key)
            if missing:
                bound=config.get("conservative_"+key+"_per_call")
                if type(bound) is not int or bound<=0:
                    raise OwnerBudgetError("explicit_token_cap_historical_unknown_usage_deferred:"+key)
                used[key]+=missing*bound
                used["total_tokens"]+=missing*bound
                conservative[key]+=missing*bound
                conservative["total_tokens"]+=missing*bound
    if cap["api_usd"] is not None:
        legacy_unpriced = 0 if baseline.get("usd_priced") else _historical_unknown_count(camp,baseline,"api_usd")
        changed_unpriced=[claim for claim in claims if claim["call_id"] not in baseline_ids and claim["requested"]["llm_calls"]>0
            and claim["committed"]["api_usd"]==0 and claim["actual"]["api_usd"]==0
            and claim.get("settlement_usage",{}).get("api_usd") is None and claim.get("usd_bound_kind")!="configured_price"
            and claim.get("settlement_usage",{}).get("transport_outcome")!="definitely_not_sent"]
        unpriced_count=legacy_unpriced+len(changed_unpriced)
        if unpriced_count:
            reserve=config.get("conservative_usd_per_request",(goal.get("api_request_policy") or {}).get("unknown_usd_reserve"))
            if not isinstance(reserve,(int,float)) or not math.isfinite(reserve) or reserve<=0:
                raise OwnerBudgetError("explicit_usd_cap_historical_unpriced_usage_deferred")
            used["api_usd"] += unpriced_count * reserve
            conservative["api_usd"] += unpriced_count * reserve

    unreserved = {key: None if cap[key] is None else max(0, cap[key] - used[key] - reserved[key]) for key in DIMENSIONS}
    request_view=_native_request_observations(camp,claims)
    result = {"global_caps": cap, "global_spent_and_occupied": used, "top_level_unused_allocations": reserved,
        "actual_known_subtotals":known,"conservative_resource_remainders":conservative,
        "unpriced_call_ids":request_view["unpriced"],
        "global_remaining": {key: None if cap[key] is None else max(0, cap[key] - used[key]) for key in DIMENSIONS},
        "unreserved_available": unreserved, "inflight": request_view["inflight"],
        "unknown":request_view["unknown"],"native_pending_call_ids":request_view["native_pending"], "dimension_units": {"api_usd": "USD", "gpu_seconds": "GPU seconds"}}
    if owner_id:
        row = _get(conn, "owners", owner_id)
        result["owner_remaining"] = _remaining(row) if row else resources()
        result["owner_available"] = _available(conn, row) if row else resources()
    return result


def snapshot(camp, state=None, owner_id=None):
    with _transaction(Path(camp)) as conn:
        return _snapshot(conn, Path(camp), owner_id)


def _fit(request, available, prefix):
    for key, amount in request.items():
        if available[key] is not None and amount > available[key] + 1e-9:
            raise OwnerBudgetError(prefix + ":" + key)


def _allocate(conn, camp, owner_id, amount, *, kind, parent_id=None, purpose=None, binding=None):
    identity = {"owner_id": owner_id, "allocated": resources(amount), "kind": kind, "parent_id": parent_id, "purpose": purpose, "binding": binding or {}}
    old = _get(conn, "owners", owner_id)
    if old:
        if old["status"]=="released":
            raise OwnerBudgetError("owner_allocation_already_released")
        if any(old[key] != value for key, value in identity.items()):
            raise OwnerBudgetError("owner_allocation_identity_conflict")
        return old
    _control(camp)
    if parent_id:
        parent = _get(conn, "owners", parent_id)
        if not parent or parent["status"] != "active":
            raise OwnerBudgetError("owner_parent_not_active")
        _fit(identity["allocated"], _available(conn, parent), "owner_child_unavailable")
    else:
        _fit(identity["allocated"], _snapshot(conn, camp)["unreserved_available"], "global_allocation_unavailable")
    row = {**identity, "consumed": resources(), "status": "active", "created_at": time.time()}
    _put(conn, "owners", owner_id, row)
    return row


def allocate_child(camp, parent_owner_id, child_id, amount, *, kind="role", purpose=None):
    with _transaction(Path(camp)) as conn:
        return _allocate(conn, Path(camp), child_id, amount, kind=kind, parent_id=parent_owner_id, purpose=purpose)


def _request_resources(camp, input_bound, output_bound, *, frozen_pricing=None):
    amount = resources({"llm_calls": 1, "input_tokens": input_bound, "output_tokens": output_bound, "total_tokens": input_bound + output_bound})
    goal = _read(camp / "goal.json")
    config = ((goal.get("research_execution") or {}).get("budget") or {})
    from .pricing import resolve_pricing, estimate_usd, PricingConflict
    try:
        frozen_pricing = frozen_pricing or resolve_pricing(camp)
    except PricingConflict as exc:
        raise OwnerBudgetError(str(exc)) from exc
    known_price = estimate_usd(frozen_pricing, input_tokens=input_bound, output_tokens=output_bound, upper_bound=True)
    if known_price is not None:
        amount["api_usd"] = known_price
    else:
        reserve = config.get("conservative_usd_per_request", (goal.get("api_request_policy") or {}).get("unknown_usd_reserve"))
        if reserve is not None:
            amount["api_usd"] = resources({"api_usd": reserve})["api_usd"]
            if amount["api_usd"] <= 0 and (goal.get("max_api_usd") is not None or config.get("max_api_usd") is not None):
                raise OwnerBudgetError("explicit_usd_cap_requires_positive_unknown_reserve")
        elif (goal.get("max_api_usd") if goal.get("max_api_usd") is not None else config.get("max_api_usd")) is not None:
            raise OwnerBudgetError("explicit_usd_cap_unpriced_request_deferred")
    return amount


def _planning_resources(camp, input_bound, output_bound, calls):
    """Reserve both planning stages against their actual routing and finite caps."""
    camp = Path(camp)
    from .research_execution import independent_critic_enabled
    if not independent_critic_enabled(camp):
        return resources({key: calls * value for key, value in
                          _request_resources(camp, input_bound, output_bound).items()})
    goal = _read(camp / "goal.json")
    execution = goal.get("research_execution") or {}
    limits = {"research_planner": min(2, int(execution.get("max_planner_api_attempts", 2))),
              "research_critic": int(execution.get("max_critic_api_attempts", 2))}
    if (any(not 1 <= count <= 2 for count in limits.values())
            or type(calls) is not int or not 2 <= calls <= sum(limits.values())):
        raise OwnerBudgetError("staged_planning_requires_both_finite_roles")
    counts = {role: 1 for role in limits}
    remaining = calls - 2
    for role, limit in limits.items():
        extra = min(remaining, limit - 1)
        counts[role] += extra
        remaining -= extra
    from .role_config import resolve_role_config
    from .pricing import resolve_pricing, PricingConflict
    total = resources()
    for role, count in counts.items():
        resolved = resolve_role_config(camp, role)
        if resolved.max_tokens > output_bound:
            raise OwnerBudgetError("planning_role_output_bound_exceeds_envelope:" + role)
        try:
            price = resolve_pricing(camp, model=resolved.model)
        except PricingConflict as exc:
            raise OwnerBudgetError(str(exc)) from exc
        amount = _request_resources(camp, input_bound, resolved.max_tokens, frozen_pricing=price)
        for key, value in amount.items():
            total[key] += count * value
    return total


def _role_owner(conn, camp, role):
    scope = OWNER.get()
    if scope and scope[0] != str(camp.resolve()):
        raise OwnerBudgetError("owner_campaign_mismatch")
    owner_id = scope[1] if scope else None
    purpose = scope[2] if scope else None
    if not owner_id and scope is None and role not in {"memory_curator", "context_summary", "memory_extraction", "memory_consolidation", "research_planner"}:
        selected = _read(camp / "campaign_state.json").get("selected_plan") or {}
        owner_id = selected.get("budget_owner")
        purpose = "mainline"
    if owner_id:
        row = _get(conn, "owners", owner_id)
        if not row or row["status"] not in {"active", "handoff"}:
            raise OwnerBudgetError("owner_not_active")
        ancestor=row
        while ancestor.get("parent_id"):
            ancestor=_get(conn,"owners",ancestor["parent_id"])
            if not ancestor or ancestor["status"]!="active":
                raise OwnerBudgetError("owner_ancestor_not_active")
        if row["kind"] == "role" and row.get("purpose") != role:
            raise OwnerBudgetError("owner_role_scope_violation")
        if row["kind"] == "role_preparation" and role not in {"memory_curator","context_summary","memory_extraction","memory_consolidation"}:
            raise OwnerBudgetError("owner_preparation_scope_violation")
        if row["kind"] == "continuation":
            from .research_execution import independent_critic_enabled
            planning_roles = {"research_planner", "research_critic"} if independent_critic_enabled(camp) else {"research_planner"}
            if role not in planning_roles or not row.get("round_id"):
                raise OwnerBudgetError("continuation_scope_violation")
        if row["kind"] == "selected_plan":
            if role.startswith("memory_") or role in {"context_summary", "memory_curator"}:
                raise OwnerBudgetError("maintenance_mainline_owner_forbidden")
            role_child = owner_id + ":role:" + role
            if not _get(conn, "owners", role_child):
                raise OwnerBudgetError("role_not_in_selected_chain:" + role)
            owner_id = role_child
        return owner_id
    return None


def claim_request(camp, *, call_id, operation, role, input_bound, output_bound, provider="deepseek", model=None):
    camp = Path(camp)
    if not enabled(camp):
        return None
    from .pricing import resolve_pricing, PricingConflict
    if model is None:
        from .role_config import ROLES, resolve_role_config
        if role in ROLES:
            model = resolve_role_config(camp, role).model
    try:
        frozen_pricing = resolve_pricing(camp, provider=provider, model=model)
    except PricingConflict as exc:
        raise OwnerBudgetError(str(exc)) from exc
    amount = _request_resources(camp, input_bound, output_bound, frozen_pricing=frozen_pricing)
    with _transaction(camp) as conn:
        existing = _get(conn, "claims", call_id)
        if existing:
            if existing["operation_id"] != operation or existing["requested"] != amount or existing["role"] != role or (existing.get("frozen_pricing") and existing["frozen_pricing"] != frozen_pricing):
                raise OwnerBudgetError("claim_identity_conflict")
            return existing
        _control(camp,provider=True)
        snap = _snapshot(conn, camp)
        dispositions={row.get("call_id") for row in _journal(camp,"api_unknown_dispositions.jsonl")
            if row.get("action")=="abandon_response" and row.get("allow_new_request") is True}
        if set(snap["unknown"]).difference(dispositions) or snap["inflight"]:
            raise OwnerBudgetError("unsettled_owner_claim_requires_recovery")
        # Pre-owner native unknowns keep the existing disposition gate.
        cost = _read(camp / "cost.json")
        if cost.get("api_unsettled_call_ids") or cost.get("api_abandoned_call_ids"):
            from .api_policy import unresolved_charge_ids
            if cost.get("api_unsettled_call_ids") or unresolved_charge_ids(camp):
                raise OwnerBudgetError("native_unknown_requires_recovery")
        owner_id = _role_owner(conn, camp, role)
        if role in {"memory_curator","context_summary","memory_extraction","memory_consolidation"}:
            from .research_execution import paid_maintenance_mode
            mode=paid_maintenance_mode(camp)
            if mode=="off":
                raise OwnerBudgetError("paid_maintenance_off_deferred")
            maintenance_owner=_get(conn,"owners",owner_id) if owner_id else None
            if mode=="emergency" and (not maintenance_owner or maintenance_owner["kind"]!="role_preparation"):
                raise OwnerBudgetError("emergency_requires_role_preparation_owner")
            if mode=="batch" and owner_id is not None:
                raise OwnerBudgetError("batch_requires_unreserved_auxiliary")
        _check_global_equation(snap)
        if owner_id:
            owner = _get(conn, "owners", owner_id)
            _fit(amount, _available(conn, owner), "owner_request_unavailable")
            cursor = owner
            while cursor:
                for key, count in amount.items():
                    cursor["consumed"][key] += count
                _put(conn, "owners", cursor["owner_id"], cursor)
                cursor = _get(conn, "owners", cursor["parent_id"]) if cursor.get("parent_id") else None
        else:
            _fit(amount, snap["unreserved_available"], "auxiliary_request_deferred")
        priced=frozen_pricing["pricing"] is not None
        row = {"usd_bound_kind":"configured_price" if priced else "conservative" if amount["api_usd"]>0 else "unpriced",
               "frozen_pricing": frozen_pricing,
               "call_id": call_id, "operation_id": operation, "owner_id": owner_id, "role": role, "requested": amount,
               "committed": amount, "actual": resources({"llm_calls": 1}), "status": "claimed", "created_at": time.time()}
        _put(conn, "claims", call_id, row)
        return row


def _adjust_consumed(conn, row, delta):
    owner = _get(conn, "owners", row["owner_id"]) if row.get("owner_id") else None
    while owner:
        for key, number in delta.items():
            owner["consumed"][key] += number
        _put(conn, "owners", owner["owner_id"], owner)
        owner = _get(conn, "owners", owner["parent_id"]) if owner.get("parent_id") else None


def settle_request(camp, call_id, usage):
    if not enabled(camp):
        return
    camp = Path(camp)
    usage = usage if isinstance(usage, dict) else usage.model_dump()
    with _transaction(camp) as conn:
        row = _get(conn, "claims", call_id)
        if not row:
            raise OwnerBudgetError("settlement_without_claim")
        frozen = row.get("frozen_pricing")
        if frozen and any(usage.get(key) is not None and usage[key] != frozen[field]
                          for key, field in (("provider", "provider"), ("requested_model", "model"))):
            raise OwnerBudgetError("settlement_provider_or_model_identity_conflict")
        if row.get("frozen_pricing", {}).get("pricing") is not None:
            from .pricing import estimate_usd
            computed = estimate_usd(row["frozen_pricing"], input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"), cache_hit_tokens=usage.get("cache_hit_tokens"))
            if computed is not None:
                if usage.get("api_usd") is not None and abs(float(usage["api_usd"]) - computed) > 1e-9:
                    raise OwnerBudgetError("settlement_pricing_identity_conflict")
                usage = {**usage, "api_usd": computed}
        identity = {key: usage.get(key) for key in ("input_tokens", "output_tokens", "api_usd", "transport_outcome", "unconfirmed_charge_possible")}
        if usage.get("cache_hit_tokens") is not None:
            identity["cache_hit_tokens"] = usage["cache_hit_tokens"]
        identity["unconfirmed_charge_possible"] = bool(identity.get("unconfirmed_charge_possible"))
        signature = _hash(identity)
        if row.get("settlement_digest"):
            if row["settlement_digest"] != signature:
                # SDK settlement precedes native pricing; add confirmed price once.
                if row.get("actual", {}).get("api_usd") == 0 and isinstance(usage.get("api_usd"), (int,float)) and row.get("settlement_usage", {}).get("input_tokens") == usage.get("input_tokens") and row.get("settlement_usage", {}).get("output_tokens") == usage.get("output_tokens"):
                    pass
                elif row["status"]=="unknown" and identity["unconfirmed_charge_possible"] and identity["input_tokens"] is None and identity["output_tokens"] is None:
                    # Explicit native abandonment changes disposition, not usage.
                    return
                elif (row["status"]=="settled" and usage.get("charge_usage_source")=="durable_owner_provider_settlement"
                      and row.get("settlement_usage",{}).get("input_tokens")==identity["input_tokens"]
                      and row.get("settlement_usage",{}).get("output_tokens")==identity["output_tokens"]
                      and row.get("settlement_usage",{}).get("cache_hit_tokens")==identity.get("cache_hit_tokens")
                      and row.get("settlement_usage",{}).get("api_usd")==identity["api_usd"]):
                    # A lost outer reply may be explicitly abandoned; exact
                    # persisted provider usage remains a known charge fact.
                    return
                else:
                    raise OwnerBudgetError("settlement_identity_conflict")
            else:
                return
        not_sent = usage.get("transport_outcome") == "definitely_not_sent"
        unknown = usage.get("unconfirmed_charge_possible") or usage.get("transport_outcome") in {"outcome_unknown", "unknown_after_process_exit"}
        actual = resources({"llm_calls": 1})  # durable application attempts retain their historical charge
        committed = dict(row["committed"])
        for key in ("input_tokens", "output_tokens"):
            observed = usage.get(key)
            if type(observed) is int and observed >= 0:
                actual[key] = observed
                committed[key] = observed
            elif not_sent:
                committed[key] = 0
        # Each known component discharges its own remainder. Total tokens
        # always sum input + completion, including partial provider usage.
        actual["total_tokens"] = actual["input_tokens"] + actual["output_tokens"]
        committed["total_tokens"] = committed["input_tokens"] + committed["output_tokens"]
        price = usage.get("api_usd")
        if isinstance(price, (int,float)) and price >= 0 and math.isfinite(price):
            actual["api_usd"] = price
            committed["api_usd"] = price
        elif not_sent:
            committed["api_usd"] = 0
        _adjust_consumed(conn, row, {key: committed[key] - row["committed"][key] for key in DIMENSIONS})
        row.update(actual=actual, committed=committed, settlement_usage=identity,
                   pricing_proof={key:usage[key] for key in ("pricing_source","pricing_date") if usage.get(key) is not None},settlement_digest=signature,
                   status="unknown" if unknown else "settled", settled_at=time.time())
        _put(conn, "claims", call_id, row)
        # Provider overrun is a recorded fact, never clamped to manufacture compliance.
        snap = _snapshot(conn, camp)
        if any(cap is not None and snap["global_spent_and_occupied"][key] + snap["top_level_unused_allocations"][key] > cap + 1e-9 for key, cap in snap["global_caps"].items()):
            _put(conn, "meta", "overrun", {"call_id": call_id, "at": time.time()})


@contextlib.contextmanager
def budget_owner(camp, owner_id, *, purpose="mainline", observation_digest=None):
    token = OWNER.set((str(Path(camp).resolve()), owner_id, purpose, observation_digest))
    try:
        yield owner_id
    finally:
        OWNER.reset(token)


@contextlib.contextmanager
def auxiliary_owner(camp, *, purpose="batch"):
    token = OWNER.set((str(Path(camp).resolve()), None, "auxiliary:" + purpose, None))
    try:
        yield None
    finally:
        OWNER.reset(token)


@contextlib.contextmanager
def transport_claim(camp, call_id):
    token = TRANSPORT.set((str(Path(camp).resolve()), call_id))
    try:
        yield
    finally:
        TRANSPORT.reset(token)


def _verify_request_pricing(camp, claim, *, provider, model):
    frozen = claim.get("frozen_pricing")
    if not frozen:
        return
    from .pricing import resolve_pricing, PricingConflict
    try:
        current = resolve_pricing(camp, provider=provider, model=frozen.get("model") if model is None else model)
    except PricingConflict as exc:
        raise OwnerBudgetError(str(exc)) from exc
    if current != frozen:
        raise OwnerBudgetError("provider_pricing_identity_changed")


def before_transport(camp, *, call_id, role, input_bound, output_bound, provider="deepseek", model=None):
    camp = Path(camp)
    with _transaction(camp) as conn:
        _control(camp,provider=True)
        if _get(conn, "meta", "overrun"):
            raise OwnerBudgetError("provider_budget_overrun_requires_reconciliation")
        _check_global_equation(_snapshot(conn,camp))
        row = _get(conn, "claims", call_id)
        if not row or row["status"] != "claimed":
            raise OwnerBudgetError("provider_transport_claim_not_sendable")
        if row["role"]!=role:
            raise OwnerBudgetError("provider_transport_claim_role_mismatch")
        _verify_request_pricing(camp, row, provider=provider, model=model)
        if input_bound > row["requested"]["input_tokens"] or output_bound > row["requested"]["output_tokens"]:
            raise OwnerBudgetError("provider_bounds_exceed_claim")
        row.update(status="inflight", transport_started_at=time.time())
        _put(conn, "claims", call_id, row)


def current_transport(camp=None):
    value = TRANSPORT.get()
    if value and (camp is None or value[0] == str(Path(camp).resolve())):
        return value
    return None


def request_pricing(camp, call_id):
    """The original claim's price proof also governs late ledger recovery."""
    with _transaction(Path(camp)) as conn:
        row = _get(conn, "claims", call_id)
        return row.get("frozen_pricing") if row else None


def current_campaign():
    scope = OWNER.get()
    return Path(scope[0]) if scope else None


def _envelope(camp, task_kind, known_cost, recovery_limits):
    goal = _read(Path(camp) / "goal.json")
    config = ((goal.get("research_execution") or {}).get("budget") or {})
    configured = (config.get("envelopes") or {}).get(task_kind) or {}
    repairs = int(recovery_limits.get("max_repairs", recovery_limits.get("max_engineering_repairs", goal.get("max_repairs_per_candidate", 2))))
    steps = int(recovery_limits.get("coder_steps", 8))
    reviewer_calls = 1 if (goal.get("research_execution") or {}).get("implementation_review_mode") == "single_pass" else 2
    defaults = {"experiment_designer": 2, "candidate_coder": 2 * steps * (repairs + 1), "candidate_reviewer": reviewer_calls * (repairs + 1), "result_analyst": 2}
    if (goal.get("research_execution") or {}).get("design_contract_version") == "v2":
        defaults["experiment_designer"] = 2 * (repairs + 1)
    if task_kind in {"diagnosis", "validation", "confirmation"}:
        defaults = {"result_analyst": 2}
    feedback_reports = known_cost.get("feedback_report_count",1)
    if type(feedback_reports) is not int or feedback_reports < 1:
        raise OwnerBudgetError("selected_feedback_report_bound_required")
    defaults["result_analyst"] = 2 * feedback_reports
    roles = configured.get("role_calls") or defaults
    if "result_analyst" not in roles or any(type(count) is not int or count < 1 for count in roles.values()):
        raise OwnerBudgetError("selected_chain_requires_bounded_analysis")
    if roles["result_analyst"] < 2 * feedback_reports:
        raise OwnerBudgetError("selected_chain_analysis_envelope_insufficient")
    if task_kind=="new_method" and not {"experiment_designer","candidate_coder","candidate_reviewer","result_analyst"} <= set(roles):
        raise OwnerBudgetError("selected_chain_required_stage_missing")
    input_per = int(configured.get("input_tokens_per_call", config.get("input_tokens_per_call", 128000)))
    output_per = int(configured.get("output_tokens_per_call", config.get("output_tokens_per_call", 32768)))
    role_resources = {role: resources({key: value * count for key, value in _request_resources(Path(camp), input_per, output_per).items()}) for role, count in roles.items()}
    for row in role_resources.values():
        row["training_jobs"] = row["gpu_seconds"] = 0
    known = known_cost.get("resources", known_cost)
    jobs = known.get("training_jobs") if known.get("training_jobs") is not None else configured.get("training_jobs")
    gpu = known.get("gpu_seconds") if known.get("gpu_seconds") is not None else configured.get("gpu_seconds")
    if task_kind == "diagnosis":
        jobs = gpu = 0
    if type(jobs) is not int or jobs < 0 or not isinstance(gpu, (int,float)) or not math.isfinite(gpu) or gpu < 0 or (jobs > 0 and gpu <= 0):
        raise OwnerBudgetError("selected_chain_measurement_cost_unknown")
    preparation = {}
    for role,count in (configured.get("preparation_calls") or {}).items():
        if role not in roles or type(count) is not int or not 1 <= count <= 2:
            raise OwnerBudgetError("role_preparation_allocation_invalid")
        preparation[role] = resources({key:value * count for key,value in _request_resources(Path(camp),input_per,output_per).items()})
    total = resources({key: sum(row[key] for row in list(role_resources.values()) + list(preparation.values())) for key in DIMENSIONS})
    engineering_gpu=configured.get("engineering_gpu_seconds")
    if engineering_gpu is None:
        engineering_gpu=0
        common=_read(Path(camp)/"method_search_manifest.json")
        if task_kind=="new_method" and goal.get("evaluation_mode")=="loso_method_search":
            cards=common.get("gpu") or goal.get("gpu") or []
            if cards:
                engineering_gpu=120 * len(cards) * (repairs+2)
            elif jobs>0:
                raise OwnerBudgetError("engineering_gpu_envelope_unknown")
    engineering=resources({"gpu_seconds":engineering_gpu})
    total.update(training_jobs=jobs, gpu_seconds=gpu+engineering["gpu_seconds"])
    from .research_execution import planning_call_budget
    planning_calls = planning_call_budget(goal)
    continuation_calls = int(configured.get("continuation_calls", planning_calls))
    if not 1 <= continuation_calls <= planning_calls:
        raise OwnerBudgetError("continuation_must_be_finite")
    continuation = _planning_resources(camp, input_per, output_per, continuation_calls)
    return total, role_resources, continuation, preparation, engineering


def allocate_selected_plan(camp, *, plan_id, task_kind, known_cost, recovery_limits):
    camp = Path(camp)
    if not enabled(camp):
        raise OwnerBudgetError("owner_protocol_not_enabled")
    total, roles, continuation, preparation, engineering = _envelope(camp, task_kind, known_cost, recovery_limits)
    owner_id = "plan:" + plan_id
    amount = resources({key: total[key] + continuation[key] for key in DIMENSIONS})
    with _transaction(camp) as conn:
        _control(camp)
        owner = _allocate(conn, camp, owner_id, amount, kind="selected_plan", binding={"plan_id": plan_id, "task_kind": task_kind, "recovery_limits": recovery_limits})
        for role, budget in roles.items():
            _allocate(conn, camp, owner_id + ":role:" + role, budget, kind="role", parent_id=owner_id, purpose=role)
        if engineering["gpu_seconds"]:
            _allocate(conn,camp,owner_id+":engineering_gpu",engineering,kind="engineering_gpu",parent_id=owner_id,purpose="engineering_multigpu")
        for role,budget in preparation.items():
            _allocate(conn,camp,owner_id+":prepare:"+role,budget,kind="role_preparation",parent_id=owner_id,purpose=role)
        _allocate(conn, camp, owner_id + ":continuation", continuation, kind="continuation", parent_id=owner_id, purpose="research_planner")
        return owner


def finish_selected_plan(camp, *, plan_id, observation_digest):
    if not observation_digest:
        raise OwnerBudgetError("terminal_observation_digest_required")
    camp = Path(camp)
    owner_id = "plan:" + plan_id
    with _transaction(camp) as conn:
        row = _get(conn, "owners", owner_id)
        continuation = _get(conn, "owners", owner_id + ":continuation")
        if not row or not continuation:
            raise OwnerBudgetError("selected_plan_owner_not_found")
        if row.get("terminal_observation_digest"):
            if row["terminal_observation_digest"] != observation_digest:
                raise OwnerBudgetError("terminal_observation_conflict")
            return continuation
        descendants={owner_id}
        children=_all(conn,"owners")
        changed=True
        while changed:
            changed=False
            for child in children:
                if child.get("parent_id") in descendants and child["owner_id"] not in descendants:
                    descendants.add(child["owner_id"]);changed=True
        pending = [claim for claim in _all(conn, "claims") if claim.get("owner_id") in descendants and claim["status"] in {"claimed", "inflight", "unknown"}]
        if pending:
            raise OwnerBudgetError("owner_release_requires_reconciliation")
        # Move the unspent continuation into a top-level owner in one transaction.
        remaining = _remaining(continuation)
        for child in children:
            if child["owner_id"] in descendants and child["owner_id"] not in {owner_id,continuation["owner_id"]}:
                child["status"] = "released"
                _put(conn, "owners", child["owner_id"], child)
        row.update(status="released", terminal_observation_digest=observation_digest, released_at=time.time())
        continuation.update(parent_id=None, allocated=remaining, consumed=resources(), status="handoff", observation_digest=observation_digest)
        _put(conn, "owners", owner_id, row)
        _put(conn, "owners", continuation["owner_id"], continuation)
        return continuation


@contextlib.contextmanager
def planner_owner(camp, *, observation_digest, round_id):
    camp = Path(camp)
    if not enabled(camp):
        yield None
        return
    with _transaction(camp) as conn:
        matches = [row for row in _all(conn, "owners") if row["kind"] == "continuation" and row.get("observation_digest") == observation_digest]
        if matches:
            owner = matches[0]
            if owner.get("round_id") not in {None, round_id}:
                raise OwnerBudgetError("continuation_already_bound_to_round")
            if owner["status"] == "released":
                raise OwnerBudgetError("continuation_round_already_finished")
            owner.update(round_id=round_id, status="active")
            _put(conn, "owners", owner["owner_id"], owner)
        else:
            if any(row["kind"] == "selected_plan" for row in _all(conn, "owners")):
                raise OwnerBudgetError("continuation_observation_not_found")
            config=((_read(camp/"goal.json").get("research_execution") or {}).get("budget") or {})
            planning=(config.get("envelopes") or {}).get("planning") or {}
            from .research_execution import planning_call_budget
            amount = _planning_resources(camp,
                int(planning.get("input_tokens_per_call", config.get("input_tokens_per_call", 128000))),
                int(planning.get("output_tokens_per_call", config.get("output_tokens_per_call", 32768))),
                planning_call_budget(camp))
            owner = _allocate(conn, camp, "bootstrap:initial_planner", amount, kind="continuation", binding={"observation_digest": observation_digest})
            if owner["status"] == "released":
                raise OwnerBudgetError("bootstrap_round_already_finished")
            if owner.get("round_id") not in {None, round_id}:
                raise OwnerBudgetError("bootstrap_already_bound_to_round")
            owner.update(round_id=round_id, status="active", observation_digest=observation_digest)
            _put(conn, "owners", owner["owner_id"], owner)
    with budget_owner(camp, owner["owner_id"], purpose="planning", observation_digest=observation_digest):
        yield owner["owner_id"]


def input_token_bound(messages, request_fields=None, budget_report=None):
    """UTF-8 byte ceiling, optionally bounded by the compiled provider limit.

    This is a conservative allocation bound, not the compiler's token estimate.
    Reasoning is already included in provider completion/output usage.
    """
    structured = {key: value for key, value in (request_fields or {}).items() if key in {"tools", "tool_choice", "response_format"}}
    bound = len(json.dumps(messages, ensure_ascii=False).encode()) + len(json.dumps(structured, ensure_ascii=False).encode()) + 128
    ceiling = (budget_report or {}).get("provider_input_tokens")
    return min(bound, ceiling) if type(ceiling) is int and ceiling > 0 else bound


def native_cost_floor(camp):
    """Preserve historical usage not represented by complete legacy journals."""
    if not enabled(camp):
        return None
    with _transaction(Path(camp)) as conn:
        baseline = _get(conn, "meta", "baseline")
        result = dict(baseline["used"])
        ids = set(baseline["call_ids"])
        result["llm_calls"] += len(_native_ids(Path(camp)) - ids)
        rows = {row["call_id"]: row for row in _journal(Path(camp), "llm_calls.jsonl") if row.get("call_id")}
        initial=baseline.get("initial_usage") or {}
        for call_id,row in rows.items():
            for key in ("input_tokens", "output_tokens", "api_usd"):
                if isinstance(row.get(key), (int,float)):
                    before=(initial.get(call_id) or {}).get(key) if call_id in ids else 0
                    result[key] += max(0,row[key]-(before if isinstance(before,(int,float)) else 0))
        return result


def finish_planner_round(camp, owner_id):
    """Unused correction allocation is released only after a completed round."""
    with _transaction(Path(camp)) as conn:
        row = _get(conn, "owners", owner_id)
        if not row or row["kind"] != "continuation":
            raise OwnerBudgetError("planner_owner_not_continuation")
        if any(claim.get("owner_id") == owner_id and claim["status"] in {"claimed", "inflight", "unknown"} for claim in _all(conn,"claims")):
            raise OwnerBudgetError("continuation_release_requires_reconciliation")
        row["status"] = "released"
        _put(conn,"owners",owner_id,row)


def current_owner(camp):
    scope = OWNER.get()
    if not scope or scope[0] != str(Path(camp).resolve()) or not scope[1]:
        return None
    with _transaction(Path(camp)) as conn:
        return _get(conn,"owners",scope[1])


def reserve_phase_owner(camp, receipt_id, amount, replace_owner_id=None):
    with _transaction(Path(camp)) as conn:
        owner_id = _role_owner_for_phase(conn,Path(camp))
        if replace_owner_id:
            old=_get(conn,"owners",replace_owner_id)
            if not old or old.get("parent_id")!=owner_id:
                raise OwnerBudgetError("replacement_phase_owner_mismatch")
            if any(claim.get("owner_id")==replace_owner_id and claim["status"] in {"claimed","inflight","unknown"} for claim in _all(conn,"claims")):
                raise OwnerBudgetError("replacement_phase_requires_reconciliation")
            old["status"]="released"
            _put(conn,"owners",replace_owner_id,old)
        return _allocate(conn,Path(camp),owner_id+":phase:"+receipt_id,amount,
            kind="phase",parent_id=owner_id,purpose=receipt_id)


def _role_owner_for_phase(conn,camp):
    scope = OWNER.get()
    selected = _read(camp / "campaign_state.json").get("selected_plan") or {}
    owner_id = scope[1] if scope and scope[0] == str(camp.resolve()) else selected.get("budget_owner")
    owner = _get(conn,"owners",owner_id) if owner_id else None
    if not owner or owner["kind"] != "selected_plan" or owner["status"] != "active":
        raise OwnerBudgetError("measurement_selected_owner_required")
    return owner_id


def claim_training(camp, job_id, gpu_estimate, phase_owner_id=None):
    camp = Path(camp)
    if not enabled(camp):
        return
    call_id = "job:"+job_id
    amount=resources({"training_jobs":1,"gpu_seconds":gpu_estimate})
    with _transaction(camp) as conn:
        old=_get(conn,"claims",call_id)
        if old:
            if old["requested"]!=amount:
                raise OwnerBudgetError("training_claim_identity_conflict")
            return old
        _control(camp)
        owner_id = phase_owner_id or _role_owner_for_phase(conn,camp)
        row=_get(conn,"owners",owner_id)
        if not row or row["status"]!="active":
            raise OwnerBudgetError("training_owner_not_active")
        _fit(amount,_available(conn,row),"training_owner_unavailable")
        claim={"call_id":call_id,"operation_id":job_id,"owner_id":owner_id,"role":"training",
            "requested":amount,"committed":amount,"actual":resources({"training_jobs":1}),"status":"inflight","created_at":time.time()}
        _adjust_consumed(conn,claim,amount)
        _put(conn,"claims",call_id,claim)
        return claim


def settle_training(camp,job_id,gpu_seconds):
    if not enabled(camp):
        return
    with _transaction(Path(camp)) as conn:
        claim=_get(conn,"claims","job:"+job_id)
        if not claim:
            return
        if claim["status"]=="settled":
            if claim["actual"]["gpu_seconds"]!=gpu_seconds:
                raise OwnerBudgetError("training_settlement_identity_conflict")
            return
        committed=resources({"training_jobs":1,"gpu_seconds":gpu_seconds})
        _adjust_consumed(conn,claim,{key:committed[key]-claim["committed"][key] for key in DIMENSIONS})
        claim.update(committed=committed,actual=committed,status="settled",settled_at=time.time())
        _put(conn,"claims",claim["call_id"],claim)


def release_phase_owner(camp,owner_id):
    with _transaction(Path(camp)) as conn:
        row=_get(conn,"owners",owner_id)
        if not row:
            raise OwnerBudgetError("phase_owner_not_found")
        if any(claim.get("owner_id")==owner_id and claim["status"] in {"claimed","inflight","unknown"} for claim in _all(conn,"claims")):
            raise OwnerBudgetError("phase_owner_release_requires_reconciliation")
        row["status"]="released"
        _put(conn,"owners",owner_id,row)


@contextlib.contextmanager
def role_preparation(camp,role):
    """Only explicitly allocated emergency preparation can call maintenance."""
    from .research_execution import paid_maintenance_mode
    if not enabled(camp) or paid_maintenance_mode(camp)!="emergency":
        yield
        return
    selected=_read(Path(camp)/"campaign_state.json").get("selected_plan") or {}
    scope=OWNER.get()
    parent=scope[1] if scope and scope[0]==str(Path(camp).resolve()) else selected.get("budget_owner")
    prep_id=None
    with _transaction(Path(camp)) as conn:
        row=_get(conn,"owners",parent) if parent else None
        if row and row["kind"]=="selected_plan":
            candidate=row["owner_id"]+":prepare:"+role
            child=_get(conn,"owners",candidate)
            if child and child["status"]=="active":
                prep_id=candidate
    if prep_id:
        with budget_owner(camp,prep_id,purpose="role_preparation"):
            yield
    else:
        yield


def preview_selected_plan(camp, *, task_kind, known_cost, recovery_limits):
    """No owner is allocated for losing proposals; compare complete envelopes."""
    try:
        total,roles,continuation,preparation,engineering=_envelope(camp,task_kind,known_cost,recovery_limits)
        requested=resources({key:total[key]+continuation[key] for key in DIMENSIONS})
        with _transaction(Path(camp)) as conn:
            _control(Path(camp))
            _fit(requested,_snapshot(conn,Path(camp))["unreserved_available"],"global_allocation_unavailable")
        return {"ok":True,"resources":requested,"roles":roles,"continuation":continuation,"preparation":preparation,"engineering_gpu":engineering}
    except OwnerBudgetError as exc:
        return {"ok":False,"reason":str(exc)}



def _check_global_equation(snap):
    for key,cap in snap["global_caps"].items():
        if cap is not None and snap["global_spent_and_occupied"][key]+snap["top_level_unused_allocations"][key]>cap+1e-9:
            raise OwnerBudgetError("global_budget_occupied_exceeds_cap:"+key)


def mark_transport_dispatched(camp, call_id, *, provider="deepseek", model=None):
    """The sole HTTP boundary consumes the claim's one transport permit once."""
    with _transaction(Path(camp)) as conn:
        _control(Path(camp),provider=True)
        claim=_get(conn,"claims",call_id)
        if not claim or claim["status"]!="inflight" or claim.get("dispatched_at"):
            raise OwnerBudgetError("provider_transport_permit_already_used_or_missing")
        _verify_request_pricing(Path(camp), claim, provider=provider, model=model)
        _check_global_equation(_snapshot(conn,Path(camp)))
        claim["dispatched_at"]=time.time()
        _put(conn,"claims",call_id,claim)


def mark_http_dispatched(camp, call_id, *, provider, model):
    """Recheck frozen authority on the SDK's completed HTTP request."""
    camp = Path(camp)
    with _transaction(camp) as conn:
        claim = _get(conn, "claims", call_id)
        if claim and claim.get("http_dispatched_at"):
            raise OwnerBudgetError("provider_http_transport_permit_already_used")
        if not claim or claim["status"] != "inflight" or not claim.get("dispatched_at"):
            raise OwnerBudgetError("provider_http_transport_permit_missing")
        _control(camp, provider=True)
        _verify_request_pricing(camp, claim, provider=provider, model=model)
        _check_global_equation(_snapshot(conn, camp))
        claim["http_dispatched_at"] = time.time()
        _put(conn, "claims", call_id, claim)


def admission_available(camp, role):
    """The caller's spendable balance, alongside protected global allocations."""
    with _transaction(Path(camp)) as conn:
        owner_id=_role_owner(conn,Path(camp),role)
        snap=_snapshot(conn,Path(camp),owner_id)
        available=snap.get("owner_available") if owner_id else snap["unreserved_available"]
        return {"owner_id":owner_id,"available":available,"snapshot":snap}


def training_launch_admission(camp,job_id,n_gpu,*,now):
    """No training entrypoint can consume another task's GPU allocation."""
    if not enabled(camp):
        return None
    with _transaction(Path(camp)) as conn:
        _control(Path(camp))
        _check_global_equation(_snapshot(conn,Path(camp)))
        claim=_get(conn,"claims","job:"+job_id)
        if not claim or claim["role"]!="training" or claim["status"]!="inflight" or claim.get("process_dispatched_at"):
            raise OwnerBudgetError("training_launch_permit_already_used_or_missing")
        if claim["committed"]["gpu_seconds"]<=0:
            raise OwnerBudgetError("training_gpu_bound_required")
        claim["process_dispatched_at"]=now
        _put(conn,"claims",claim["call_id"],claim)
        return {"budget_owner":claim["owner_id"],"gpu_seconds_bound":claim["committed"]["gpu_seconds"],
                "deadline_at":now+claim["committed"]["gpu_seconds"]/max(1,n_gpu)}



def claim_engineering_gpu(camp,probe_id,gpu_bound):
    if not enabled(camp):
        return None
    camp=Path(camp)
    with _transaction(camp) as conn:
        old=_get(conn,"claims","engineering:"+probe_id)
        if old:
            if old["requested"]["gpu_seconds"]!=gpu_bound:
                raise OwnerBudgetError("engineering_probe_bound_conflict")
            return old
        _control(camp)
        parent=_role_owner_for_phase(conn,camp)
        owner_id=parent+":engineering_gpu"
        owner=_get(conn,"owners",owner_id)
        amount=resources({"gpu_seconds":gpu_bound})
        if not owner or owner["status"]!="active":
            raise OwnerBudgetError("engineering_gpu_owner_required")
        _fit(amount,_available(conn,owner),"engineering_gpu_owner_unavailable")
        _check_global_equation(_snapshot(conn,camp))
        claim={"call_id":"engineering:"+probe_id,"operation_id":probe_id,"owner_id":owner_id,"role":"engineering_probe",
            "requested":amount,"committed":amount,"actual":resources(),"status":"inflight","created_at":time.time()}
        _adjust_consumed(conn,claim,amount)
        _put(conn,"claims",claim["call_id"],claim)
        return claim


def settle_engineering_gpu(camp,probe_id,gpu_seconds,*,uncertain=False):
    if not enabled(camp):
        return
    with _transaction(Path(camp)) as conn:
        row=_get(conn,"claims","engineering:"+probe_id)
        if not row:
            raise OwnerBudgetError("engineering_settlement_without_claim")
        final=resources({"gpu_seconds":gpu_seconds})
        if row["status"] in {"settled","unknown"}:
            if row["committed"]!=final:
                raise OwnerBudgetError("engineering_settlement_identity_conflict")
            return
        _adjust_consumed(conn,row,{key:final[key]-row["committed"][key] for key in DIMENSIONS})
        row.update(committed=final,actual=resources() if uncertain else final,status="unknown" if uncertain else "settled",settled_at=time.time())
        _put(conn,"claims",row["call_id"],row)



def _historical_unknown_count(camp,baseline,dimension):
    """Late confirmed usage discharges only its original conservative remainder."""
    ids=set(baseline["call_ids"])
    initial=baseline.get("initial_usage") or {}
    current={row["call_id"]:row for row in _journal(camp,"llm_calls.jsonl") if row.get("call_id")}
    tracked_unknown=sum(not isinstance((current.get(identity) or {}).get(dimension),(int,float))
        and (current.get(identity) or {}).get("transport_outcome")!="definitely_not_sent" for identity in ids)
    opaque=max(0,baseline["used"]["llm_calls"]-len(ids))
    initial_tracked_known=sum(isinstance((initial.get(identity) or {}).get(dimension),(int,float)) for identity in ids)
    initial_coverage=baseline.get("token_known_counts",{}).get(dimension,0) if dimension!="api_usd" else sum(isinstance(row.get(dimension),(int,float)) for row in initial.values())
    opaque_known=max(0,initial_coverage-initial_tracked_known)
    tracked_initial_sum=sum((initial.get(identity) or {}).get(dimension) or 0 for identity in ids)
    # A positive persisted known subtotal proves at least one known anonymous
    # historical call, even when older journals omitted its stable id.
    if not opaque_known and baseline["used"][dimension]>tracked_initial_sum:
        opaque_known=1
    return tracked_unknown+max(0,opaque-opaque_known)


def engineering_launch_admission(camp,probe_id):
    """One claimed engineering check grants one GPU-child launch permit."""
    if not enabled(camp):
        return None
    with _transaction(Path(camp)) as conn:
        _control(Path(camp))
        _check_global_equation(_snapshot(conn,Path(camp)))
        claim=_get(conn,"claims","engineering:"+probe_id)
        if not claim or claim["status"]!="inflight" or claim.get("gpu_child_dispatched_at"):
            raise OwnerBudgetError("engineering_gpu_launch_permit_already_used_or_missing")
        owner=_get(conn,"owners",claim["owner_id"])
        if not owner or owner["status"]!="active":
            raise OwnerBudgetError("engineering_gpu_owner_not_active")
        claim["gpu_child_dispatched_at"]=time.time()
        _put(conn,"claims",claim["call_id"],claim)
        return claim



def settled_provider_usage(camp,call_id):
    """Known charge proof survives an outer receipt crash; it is not a reply."""
    if not enabled(camp):
        return None
    with _transaction(Path(camp)) as conn:
        claim=_get(conn,"claims",call_id)
        usage=(claim or {}).get("settlement_usage") or {}
        if claim and claim["status"]=="settled" and usage.get("transport_outcome")=="confirmed_response":
            return {**usage,**(claim.get("pricing_proof") or {})}
    return None



def _native_request_observations(camp,claims):
    """Expose existing native occupancy without creating a second charge."""
    intents={str(row["call_id"]) for name in ("llm_api_intents.jsonl","api_operation_attempts.jsonl")
        for row in _journal(camp,name) if row.get("call_id")}
    settled={str(row["call_id"]):row for row in _journal(camp,"llm_calls.jsonl") if row.get("call_id")}
    owned={row["call_id"]:row for row in claims}
    native_ids=intents|set(settled)
    disposition_ids={str(row["call_id"]) for row in _journal(camp,"api_unknown_dispositions.jsonl")
        if row.get("call_id") and row.get("action")=="abandon_response"}
    # A disposition retains its unknown charge; it ends waiting for that reply.
    known_dispositions=disposition_ids.intersection(native_ids|set(owned))
    native_unknown={identity for identity,row in settled.items() if row.get("unconfirmed_charge_possible")
        or row.get("transport_outcome") in {"outcome_unknown","unknown_after_process_exit"}}
    pending=intents.difference(settled).difference(known_dispositions)
    inflight={identity for identity in pending if (owned.get(identity) or {}).get("status")!="settled"}|{row["call_id"] for row in claims if row["status"] in {"claimed","inflight"}}
    unknown=native_unknown|known_dispositions|{row["call_id"] for row in claims if row["status"]=="unknown"}
    unpriced=set()
    for identity in native_ids|set(owned):
        claim=owned.get(identity)
        if claim and claim["requested"]["llm_calls"]==0:
            continue
        source=settled.get(identity) or ((claim or {}).get("settlement_usage") or {})
        if source.get("transport_outcome")=="definitely_not_sent" or ((claim or {}).get("settlement_usage") or {}).get("transport_outcome")=="definitely_not_sent":
            continue
        # SDK usage can be durable before its outer native receipt is written.
        price=source.get("api_usd")
        if not isinstance(price,(int,float)):
            price=((claim or {}).get("settlement_usage") or {}).get("api_usd")
        if not isinstance(price,(int,float)):
            unpriced.add(identity)
    return {"inflight":sorted(inflight),"unknown":sorted(unknown),"unpriced":sorted(unpriced),"native_pending":sorted(pending)}
