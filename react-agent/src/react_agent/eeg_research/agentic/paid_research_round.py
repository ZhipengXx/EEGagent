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
from .plan_protocol import normalize_research_round, round_id, round_input


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


def publish_raw(camp, state, full, owner, raw, *, correction=False):
    """The backend-return boundary precedes all proposal validation and release."""
    from .loop import _write
    rid = round_id(full)
    folder = camp / "research_round_raw" / rid
    sequence = len(list(folder.glob("*.json")))
    receipt = {"schema_version": "eeg_research.paid_round_raw.v1", "round_id": rid,
        "sequence": sequence, "operation_id": rid + (":correction" if correction else ":generation"),
        "planning_owner": owner, "input_observation_digest": digest(round_input(full)),
        "round_input": round_input(full), "observation": copy.deepcopy(full),
        "state_binding": state_binding(camp, state), "raw": copy.deepcopy(raw)}
    receipt["receipt_digest"] = digest(receipt)
    _write(folder / (str(sequence).zfill(3) + ".json"), receipt, exclusive=True)


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


def _wire_observation(camp, wire):
    """Expand original delivered history references, never capture new sources."""
    from .context.representations import resolve_request_pointer, validate_pointer_closure
    from .context.sources import pointer_escape
    valid, _ = validate_pointer_closure(wire)
    if not valid:
        raise ValueError("paid_round_request_pointer_changed")
    def expand(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return copy.deepcopy(resolve_request_pointer(wire, path))
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
            source = registry.get(ref, role="research_planner")
            if source.block_id != "planner-history:" + field or source.source_hash != row.get("source_hash"):
                raise ValueError("paid_round_history_identity_changed")
            full[field] = registry.load(source, role="research_planner")
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


def _verify_call(camp, row, owner, claims, journals):
    from .context.types import digest as wire_digest
    call_id = row["call_id"]
    intent = _one(journals["llm_api_intents.jsonl"], call_id, "intent")
    attempt = _one(journals["api_operation_attempts.jsonl"], call_id, "attempt")
    request = _read(camp / "llm_requests" / (call_id + ".json"))
    response = _read(camp / "llm_responses" / (call_id + ".json"))
    rid = owner["round_id"]
    if (row.get("success") is not True or row.get("transport_outcome") != "confirmed_response"
            or response.get("validation") != "valid" or row.get("planner_round_id") != rid
            or row.get("operation_id") not in {rid + ":generation", rid + ":correction"}):
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
            or claim.get("owner_id") != owner["owner_id"] or claim.get("role") != "research_planner"
            or claim.get("operation_id") != row["operation_id"]
            or usage.get("transport_outcome") != "confirmed_response" or usage.get("unconfirmed_charge_possible") is not False
            or claim.get("settlement_digest") != wire_digest(usage)):
        raise ValueError("paid_round_settlement_not_confirmed")
    for key in ("input_tokens", "output_tokens"):
        if usage.get(key) != row.get(key) or (claim.get("actual") or {}).get(key) != row.get(key):
            raise ValueError("paid_round_settlement_usage_changed")
    if (claim.get("actual") or {}).get("llm_calls") != 1:
        raise ValueError("paid_round_settlement_call_count_changed")
    full = _wire_observation(camp, json.loads(request["user"]))
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


def restore_paid(camp, state):
    """Return a proved paid round, or None before ordinary context collection."""
    owner_id = (state.get("selected_plan") or {}).get("continuation_owner") or "bootstrap:initial_planner"
    returned = None
    for folder in sorted((camp / "research_round_raw").glob("research_round_*")):
        receipts = [_read(path) for path in sorted(folder.glob("*.json"))]
        if not receipts or receipts[0].get("planning_owner") != owner_id:
            continue
        first = receipts[0]
        binding = state_binding(camp, state)
        if first.get("state_binding") != binding:
            raise ValueError("paid_round_current_binding_changed")
        full = copy.deepcopy(first["observation"])
        for sequence, receipt in enumerate(receipts):
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
            if (receipt.get("receipt_digest") != digest(unsigned) or receipt.get("sequence") != sequence
                    or receipt.get("schema_version") != "eeg_research.paid_round_raw.v1"
                    or receipt.get("round_id") != round_id(full) or receipt.get("round_input") != round_input(full)
                    or receipt.get("input_observation_digest") != digest(round_input(full))
                    or receipt.get("state_binding") != binding or receipt.get("planning_owner") != owner_id
                    or receipt.get("observation") != first["observation"]
                    or receipt.get("operation_id") not in {round_id(full) + ":generation", round_id(full) + ":correction"}):
                raise ValueError("paid_round_raw_receipt_changed")
        _verify_current(camp, state, full)
        full["_paid_replay_readonly"] = True
        returned = full, owner_id, [row["raw"] for row in receipts], {"source": "backend_return_receipt"}
        break
    db_path = camp / "resource_owners.sqlite"
    if not db_path.is_file():
        return returned
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        found = db.execute("SELECT payload FROM owners WHERE id=?", (owner_id,)).fetchone()
        owner = json.loads(found[0]) if found else None
        claims = {identity: json.loads(payload) for identity, payload in db.execute("SELECT id,payload FROM claims")}
    if not owner or not owner.get("round_id"):
        if returned:
            raise ValueError("paid_round_planning_owner_missing")
        return None
    rid = owner["round_id"]
    if returned and rid != round_id(returned[0]):
        raise ValueError("paid_round_planning_owner_binding_changed")
    if (camp / "research_rounds" / (rid + ".json")).is_file():
        return returned
    journals = {name: _rows(camp, name) for name in ("llm_calls.jsonl", "llm_api_intents.jsonl", "api_operation_attempts.jsonl")}
    paid = [row for row in journals["llm_calls.jsonl"] if row.get("budget_owner") == owner_id
            and row.get("role") == "research_planner" and row.get("success") is True
            and _read(camp / "llm_responses" / (str(row["call_id"]) + ".json")).get("validation") == "valid"]
    if not paid:
        # Local injected backends have no provider ledger. A native attempted
        # request cannot use that local receipt path to excuse missing proof.
        if ((returned and any(row.get("budget_owner") == owner_id for row in journals["llm_api_intents.jsonl"]))
                or any(row.get("budget_owner") == owner_id and row.get("success") is True
                       for row in journals["llm_calls.jsonl"])):
            raise ValueError("paid_round_confirmed_response_missing")
        return returned
    if (owner.get("kind") != "continuation" or owner.get("status") not in {"active", "released"}
            or (state.get("terminal_observation_digest") and owner.get("observation_digest") != state["terminal_observation_digest"])):
        raise ValueError("paid_round_planning_owner_binding_changed")
    history, proofs, full = [], [], None
    for row in paid:
        _one(journals["llm_calls.jsonl"], row["call_id"], "settlement")
        view, raw, proof = _verify_call(camp, row, owner, claims, journals)
        _verify_current(camp, state, view)
        if full is not None and round_input(full) != round_input(view):
            raise ValueError("paid_round_response_inputs_conflict")
        full = view
        history.append(raw)
        proofs.append(proof)
    if owner.get("observation_digest") != (state.get("terminal_observation_digest") or digest(round_input(full))):
        raise ValueError("paid_round_owner_observation_changed")
    if returned:
        if len(history) < len(receipts) or round_input(returned[0]) != round_input(full):
            raise ValueError("paid_round_returned_response_proof_missing")
        for receipt, raw, proof in zip(receipts, history, proofs):
            if receipt["raw"] != raw or receipt["operation_id"] != proof["operation_id"]:
                raise ValueError("paid_round_returned_response_changed")
        # A correction can settle immediately before its outer raw receipt is
        # written. Its existing native response is the durable tail; no receipt
        # or original response is replaced to complete this same round.
    return full, owner_id, history, {"source": "native_confirmed_response", "calls": proofs}


def revalidate_history(full, history):
    parsed, records, valid = None, [], None
    for raw in history:
        parsed = normalize_research_round(raw, full, existing=valid)
        valid = parsed.get("valid_proposals")
        records += parsed.get("validation_records") or []
    parsed["proposal_history"] = copy.deepcopy(history)
    parsed["validation_records"] = records
    if parsed.get("round"):
        parsed["round"]["validation_records"] = records
    return parsed
