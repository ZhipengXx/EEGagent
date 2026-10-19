"""Recover already returned Planner drafts without compiling or buying a request.

The legacy path requires the original wire, confirmed native settlement and the
exact immutable history sources. Registry connections are explicitly read only;
constructing the ordinary registry would itself publish database schema.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from .action_planning import digest
from .plan_protocol import round_id, round_input


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _rows(camp, name):
    path = camp / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def readonly_registry(camp):
    from .context.sources import ContextSourceRegistry
    registry = object.__new__(ContextSourceRegistry)
    registry.camp = Path(camp).resolve()
    registry.campaign_id = registry.camp.name
    registry.root = registry.camp / "context_sources"
    registry.db_path = registry.camp / "context_sources.sqlite"
    if not registry.db_path.is_file():
        raise ValueError("paid_round_context_registry_missing")
    registry._db = lambda: sqlite3.connect(registry.db_path.as_uri() + "?mode=ro", uri=True)
    return registry


def _science_rows(rows):
    return [{key: row.get(key) for key in ("evidence_id", "candidate_id", "kind", "source_hash", "spec_hash", "result_hash")}
            for row in rows or []]


def state_binding(camp, state):
    from .loop import _planner_goal
    from .native_authorization import effective_goal
    return {"goal": _planner_goal(effective_goal(camp)),
        **{key: copy.deepcopy(state.get(key)) for key in ("contract_fingerprint", "execution_fingerprint",
            "terminal_observation_id", "terminal_observation_digest", "selected_plan", "candidates",
            "candidate_id", "candidate_ready", "candidate_admissions", "detached_candidates")},
        "evidence": _science_rows(state.get("evidence"))}


def _verify_current(camp, state, full):
    binding = state_binding(camp, state)
    for key in ("goal", "contract_fingerprint", "execution_fingerprint", "terminal_observation_id",
                "terminal_observation_digest", "selected_plan", "candidate_id", "candidate_ready"):
        if full.get(key) != binding.get(key):
            raise ValueError("paid_round_current_identity_changed:" + key)
    if (full.get("candidates") or []) != (state.get("candidates") or []):
        raise ValueError("paid_round_current_identity_changed:candidates")
    if (full.get("candidate_admissions") or []) != (state.get("candidate_admissions") or []):
        raise ValueError("paid_round_current_identity_changed:candidate_admissions")
    current = {row["evidence_id"]: row for row in binding["evidence"]}
    for row in _science_rows(full.get("evidence")):
        recorded = current.get(row["evidence_id"])
        # Native job reconciliation may enrich previously absent identity
        # fields. It may not replace any identity delivered to this Planner.
        if recorded is None or any(value is not None and recorded.get(key) != value for key, value in row.items()):
            raise ValueError("paid_round_current_identity_changed:evidence")
    contract = _read(camp / "evaluation_contract.json")
    if contract.get("fingerprint") != full.get("contract_fingerprint"):
        raise ValueError("paid_round_contract_changed")
    # The frozen candidate list alone cannot hide changed source bytes.
    for row in full.get("candidates") or []:
        source_hash = row.get("source_hash") or (row.get("check") or {}).get("source_sha256")
        if source_hash:
            path = camp / "candidates" / str(row["candidate_id"]) / "extension" / "eeg_candidate.py"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
                raise ValueError("paid_round_candidate_source_changed")


def _wire_observation(camp, wire, *, role="research_planner"):
    """Expand original delivered history references, never capture new sources."""
    from .context.representations import resolve_request_pointer, validate_pointer_closure
    from .context.sources import pointer_escape
    valid, _ = validate_pointer_closure(wire)
    if not valid:
        raise ValueError("paid_round_request_pointer_changed")
    def expand(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(wire, path), value["$context_pointer"])
            return {key: expand(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return value
    full = expand(wire)
    references = set()
    for row in wire.get("history_navigation") or []:
        ref, field = row.get("$context_source"), row.get("field")
        if not ref:
            continue
        references.add(ref)
        if field and "/" not in field:
            registry = readonly_registry(camp)
            source = registry.get(ref, role=role)
            if source.block_id != "planner-history:" + field or source.source_hash != row.get("source_hash"):
                raise ValueError("paid_round_history_identity_changed")
            full[field] = registry.load(source, role=role)
    for card in (wire.get("experiment_observations") or {}).get("cards") or []:
        references.update(row.get("source_ref") for row in card.get("source_refs") or [] if row.get("source_ref"))
    # compact_planner_view adds delivery cards after the native round was bound.
    full.pop("experiment_observations", None)
    full["_delivered_context_refs"] = references
    full["_paid_replay_readonly"] = True
    return full


def _one(rows, call_id, name):
    found = [row for row in rows if row.get("call_id") == call_id]
    if len(found) != 1:
        raise ValueError("paid_round_" + name + "_not_unique")
    return found[0]


def _verify_call(camp, row, owner, claims, journals, *, allowed_roles=None, operations=None, state=None):
    from .context.types import digest as wire_digest
    call_id = row["call_id"]
    intent = _one(journals["llm_api_intents.jsonl"], call_id, "intent")
    attempt = _one(journals["api_operation_attempts.jsonl"], call_id, "attempt")
    request = _read(camp / "llm_requests" / (call_id + ".json"))
    response = _read(camp / "llm_responses" / (call_id + ".json"))
    rid = owner["round_id"]
    allowed_roles = {"research_planner"} if allowed_roles is None else set(allowed_roles)
    operations = {rid + ":generation", rid + ":correction"} if operations is None else set(operations)
    if (row.get("success") is not True or row.get("transport_outcome") != "confirmed_response"
            or response.get("validation") != "valid" or row.get("planner_round_id") != rid
            or row.get("role") not in allowed_roles or row.get("operation_id") not in operations):
        raise ValueError("paid_round_response_not_confirmed")
    for source in (intent, attempt):
        for key in ("role", "operation_id", "planner_round_id", "budget_owner", "resource_claim_id",
                    "request_trace_ref", "request_system_sha256", "request_user_sha256", "role_config_hash", "prompt_hash"):
            if source.get(key) != row.get(key):
                raise ValueError("paid_round_native_identity_changed:" + key)
    if row.get("budget_owner") != owner["owner_id"] or row.get("resource_claim_id") != call_id:
        raise ValueError("paid_round_owner_changed")
    reference = Path(str(row.get("request_trace_ref") or ""))
    if not reference.is_absolute():
        reference = camp / reference
    if reference.resolve() != (camp / "llm_requests" / (call_id + ".json")).resolve():
        raise ValueError("paid_round_request_ref_changed")
    for key in ("call_id", "role", "prompt_hash", "request_trace_ref", "request_system_sha256", "request_user_sha256"):
        if response.get(key) != row.get(key):
            raise ValueError("paid_round_response_identity_changed:" + key)
    for key in ("call_id", "role", "prompt_hash", "role_config_hash"):
        if request.get(key) != row.get(key):
            raise ValueError("paid_round_request_identity_changed:" + key)
    for field in ("system", "user"):
        sha = hashlib.sha256(str(request.get(field, "")).encode("utf-8")).hexdigest()
        if sha != request.get(field + "_sha256") or sha != row.get("request_" + field + "_sha256"):
            raise ValueError("paid_round_request_sha_changed")
    if wire_digest({"system": request["system"], "user": request["user"],
                    "transport_fields": request.get("transport_fields") or {}}) != row.get("context_request_digest"):
        raise ValueError("paid_round_prepared_request_changed")
    if intent.get("context_request_digest") != row.get("context_request_digest"):
        raise ValueError("paid_round_intent_request_changed")
    if wire_digest(response.get("reply")) != row.get("response_reply_sha256"):
        raise ValueError("paid_round_reply_sha_changed")
    claim = claims.get(call_id) or {}
    usage = claim.get("settlement_usage") or {}
    if (claim.get("call_id") != call_id or claim.get("status") != "settled"
            or claim.get("owner_id") != owner["owner_id"] or claim.get("role") != row["role"]
            or claim.get("operation_id") != row["operation_id"]
            or usage.get("transport_outcome") != "confirmed_response" or usage.get("unconfirmed_charge_possible") is not False
            or claim.get("settlement_digest") != wire_digest(usage)):
        raise ValueError("paid_round_settlement_not_confirmed")
    for key in ("input_tokens", "output_tokens"):
        if usage.get(key) != row.get(key) or (claim.get("actual") or {}).get(key) != row.get(key):
            raise ValueError("paid_round_settlement_usage_changed")
    if (claim.get("actual") or {}).get("llm_calls") != 1:
        raise ValueError("paid_round_settlement_call_count_changed")
    wire = json.loads(request["user"])
    from .staged_planning_recovery import PLANNING_CONTEXT_VERSIONS, _input_snapshot
    if wire.get("planning_context_version") in PLANNING_CONTEXT_VERSIONS:
        full, _ = _input_snapshot(camp, wire, row, owner, state=state)
    else:
        full = _wire_observation(camp, wire, role=row["role"])
    if round_id(full) != rid:
        raise ValueError("paid_round_original_input_changed")
    raw = copy.deepcopy(response["reply"])
    normalization = response.get("normalization") or {}
    if normalization:
        from .planner_context import normalize_input_echoes
        normalized, removed = normalize_input_echoes(raw, full)
        if normalization != {"kind": "identical_read_only_input_echoes_removed", "removed_fields": removed,
                "normalized_reply": normalized, "raw_model_reply_preserved": True}:
            raise ValueError("paid_round_normalization_changed")
        raw = normalized
    return full, raw, {"call_id": call_id, "operation_id": row["operation_id"],
        "context_request_digest": row["context_request_digest"], "response_reply_sha256": row["response_reply_sha256"]}


