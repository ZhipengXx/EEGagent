"""Read-only recovery of paid proposal and independent comparison stages."""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from . import paid_research_round as paid
from .action_planning import digest
from .plan_protocol import proposal_pool_digest, round_id, round_input

STAGE_ROLES = {"generation": "research_planner", "correction": "research_planner",
               "critic": "research_critic", "critic_correction": "research_critic"}
STAGE_FIELDS = frozenset({"proposal_pool", "proposal_pool_digest", "critic_repair_request",
    "plan_repair_request", "schema_error", "schema_repair_context", "previous",
    "correction_instruction", "staged_planning_stage", "planning_stage"})


def _original_observation(view):
    return {key: copy.deepcopy(value) for key, value in view.items() if key not in STAGE_FIELDS}


def _planner_delivery(wire):
    """Expand only request-local deduplication; retain delivery cards/handles."""
    from .context.representations import resolve_request_pointer, validate_pointer_closure
    from .context.sources import pointer_escape
    if not validate_pointer_closure(wire)[0]:
        raise ValueError("paid_staged_round_request_pointer_changed")
    def expand(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(wire, path), path)
            return {key: expand(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return expand(wire)


def publish_staged_raw(camp, state, full, owner, raw, stage):
    """Publish the returned stage before validation; never replace older bytes."""
    from .loop import _write
    camp = Path(camp)
    if stage not in STAGE_ROLES:
        raise ValueError("paid_staged_round_unknown_stage")
    rid = round_id(full)
    folder = camp / "research_round_raw" / rid
    sequence = len(list(folder.glob("*.json")))
    receipt = {"schema_version": "eeg_research.paid_round_raw.v1", "round_id": rid,
        "sequence": sequence, "stage": stage, "role": STAGE_ROLES[stage],
        "operation_id": rid + ":" + stage, "planning_owner": owner,
        "input_observation_digest": digest(round_input(full)), "round_input": round_input(full),
        "observation": copy.deepcopy(full), "state_binding": paid.state_binding(camp, state),
        "raw": copy.deepcopy(raw)}
    receipt["receipt_digest"] = digest(receipt)
    _write(folder / (str(sequence).zfill(3) + ".json"), receipt, exclusive=True)


def _receipt_stages(camp, state, owner_id):
    binding = paid.state_binding(camp, state)
    for folder in sorted((camp / "research_round_raw").glob("research_round_*")):
        receipts = [paid._read(path) for path in sorted(folder.glob("*.json"))]
        if not receipts or receipts[0].get("planning_owner") != owner_id or not receipts[0].get("stage"):
            continue
        if receipts[0].get("state_binding") != binding:
            raise ValueError("paid_staged_round_current_binding_changed")
        full = copy.deepcopy(receipts[0]["observation"])
        rid = round_id(full)
        stages = {}
        for sequence, receipt in enumerate(receipts):
            stage = receipt.get("stage")
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
            if (stage not in STAGE_ROLES or stage in stages or receipt.get("sequence") != sequence
                    or receipt.get("receipt_digest") != digest(unsigned)
                    or receipt.get("schema_version") != "eeg_research.paid_round_raw.v1"
                    or receipt.get("round_id") != rid or receipt.get("operation_id") != rid + ":" + stage
                    or receipt.get("role") != STAGE_ROLES[stage] or receipt.get("planning_owner") != owner_id
                    or receipt.get("state_binding") != binding or receipt.get("round_input") != round_input(full)
                    or receipt.get("input_observation_digest") != digest(round_input(full))
                    or receipt.get("observation") != receipts[0]["observation"]):
                raise ValueError("paid_staged_round_raw_receipt_changed")
            stages[stage] = copy.deepcopy(receipt["raw"])
        paid._verify_current(camp, state, full)
        full["_paid_replay_readonly"] = True
        return full, stages, receipts
    return None


def _stages_order(stages):
    if stages and "generation" not in stages:
        raise ValueError("paid_staged_round_generation_missing")
    if "critic_correction" in stages and "critic" not in stages:
        raise ValueError("paid_staged_round_critic_missing")


def _verify_sources(camp, row):
    """Verify the original delivered source bytes without recapturing them."""
    manifest_ref = (row.get("context_sidecars") or {}).get("manifest")
    path = Path(str(manifest_ref or ""))
    if (not path.is_absolute() or not path.is_relative_to(camp / "context_deliveries")
            or path.parent.name != row.get("context_request_digest") or not path.is_file()):
        raise ValueError("paid_staged_round_delivery_manifest_missing")
    manifest = paid._read(path)
    if manifest.get("campaign_id") != camp.name or manifest.get("role") != row["role"]:
        raise ValueError("paid_staged_round_delivery_identity_changed")
    sources = manifest.get("sources") or []
    dependencies = [{"block_id": source["block_id"], "source_ref": source["source_ref"],
        "source_hash": source["source_hash"], "revision": source["source_revision"],
        "visibility": source["visibility"]} for source in sources]
    if manifest.get("source_dependency_digest") != digest(dependencies):
        raise ValueError("paid_staged_round_source_dependencies_changed")
    registry = paid.readonly_registry(camp)
    for source in sources:
        # Budget/control delivery heads can advance between the paid stages.
        # Their original bytes remain immutable; current scientific identities
        # are separately checked against native Goal, source and state.
        try:
            block = registry.get(source["source_ref"], role=row["role"], require_current=False)
            if block.source_hash != source["source_hash"] or block.campaign_id != camp.name:
                raise ValueError("source_identity_changed")
            registry.load(block, role=row["role"], require_current=False)
        except (ValueError, KeyError, OSError, PermissionError) as exc:
            raise ValueError("paid_staged_round_source_changed") from exc


def restore_staged_paid(camp, state):
    """Return original inputs/raw stages with native proof, without new calls."""
    camp = Path(camp).resolve()
    owner_id = (state.get("selected_plan") or {}).get("continuation_owner") or "bootstrap:initial_planner"
    returned = _receipt_stages(camp, state, owner_id)
    journals = {name: paid._rows(camp, name) for name in
        ("llm_calls.jsonl", "llm_api_intents.jsonl", "api_operation_attempts.jsonl")}
    relevant = [row for row in journals["llm_calls.jsonl"] if row.get("budget_owner") == owner_id
        and row.get("role") in set(STAGE_ROLES.values())]
    provider_journal = any(row.get("budget_owner") == owner_id for rows in journals.values() for row in rows)
    db_path = camp / "resource_owners.sqlite"
    if not db_path.is_file():
        if provider_journal:
            raise ValueError("paid_staged_round_planning_owner_missing")
        if not returned:
            return None
        full, stages, _ = returned
        _stages_order(stages)
        return full, owner_id, stages, {"source": "backend_return_receipt", "stage_inputs": {}}
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        found = db.execute("SELECT payload FROM owners WHERE id=?", (owner_id,)).fetchone()
        owner = json.loads(found[0]) if found else None
        claims = {identity: json.loads(payload) for identity, payload in db.execute("SELECT id,payload FROM claims")}
    if not owner or not owner.get("round_id"):
        if provider_journal:
            raise ValueError("paid_staged_round_planning_owner_missing")
        if returned:
            full, stages, _ = returned
            _stages_order(stages)
            return full, owner_id, stages, {"source": "backend_return_receipt", "stage_inputs": {}}
        return None
    rid = owner["round_id"]
    if returned and round_id(returned[0]) != rid:
        raise ValueError("paid_staged_round_planning_owner_binding_changed")
    successful = [row for row in relevant if row.get("success") is True and
        paid._read(camp / "llm_responses" / (str(row["call_id"]) + ".json")).get("validation") == "valid"]
    if not successful:
        if returned and provider_journal:
            raise ValueError("paid_staged_round_confirmed_response_missing")
        if not returned:
            return None
        full, stages, _ = returned
        _stages_order(stages)
        return full, owner_id, stages, {"source": "backend_return_receipt", "stage_inputs": {}}
    if (owner.get("kind") != "continuation" or owner.get("status") not in {"active", "released"}
            or state.get("terminal_observation_digest") and owner.get("observation_digest") != state["terminal_observation_digest"]):
        raise ValueError("paid_staged_round_planning_owner_binding_changed")
    full, stages, proofs, stage_inputs, planner_delivery = None, {}, [], {}, None
    for row in successful:
        paid._one(journals["llm_calls.jsonl"], row["call_id"], "settlement")
        operation = str(row.get("operation_id") or "")
        stage = operation[len(rid) + 1:] if operation.startswith(rid + ":") else ""
        if stage not in STAGE_ROLES or stage in stages or row.get("role") != STAGE_ROLES[stage]:
            raise ValueError("paid_staged_round_stage_identity_changed")
        view, raw, proof = paid._verify_call(camp, row, owner, claims, journals,
            allowed_roles={STAGE_ROLES[stage]}, operations={rid + ":" + stage})
        _verify_sources(camp, row)
        original = _original_observation(view)
        paid._verify_current(camp, state, original)
        if full is not None and round_input(full) != round_input(original):
            raise ValueError("paid_staged_round_response_inputs_conflict")
        if full is None:
            full = original
        if stage == "generation":
            request = paid._read(camp / "llm_requests" / (row["call_id"] + ".json"))
            planner_delivery = _planner_delivery(json.loads(request["user"]))
        if stage in {"critic", "critic_correction"}:
            pool = view.get("proposal_pool")
            pool_digest = view.get("proposal_pool_digest")
            if not isinstance(pool, dict) or not pool_digest or pool.get("pool_digest") != pool_digest:
                raise ValueError("paid_staged_round_critic_pool_missing")
            if proposal_pool_digest(pool) != pool_digest:
                raise ValueError("paid_staged_round_critic_pool_changed")
            if "critic" in stage_inputs and stage_inputs["critic"] != {"proposal_pool": pool, "proposal_pool_digest": pool_digest}:
                raise ValueError("paid_staged_round_critic_inputs_conflict")
            stage_inputs[stage] = {"proposal_pool": copy.deepcopy(pool), "proposal_pool_digest": pool_digest}
        stages[stage] = raw
        proofs.append({**proof, "stage": stage, "role": row["role"]})
    _stages_order(stages)
    if owner.get("observation_digest") != (state.get("terminal_observation_digest") or digest(round_input(full))):
        raise ValueError("paid_staged_round_owner_observation_changed")
    if returned:
        original_full, original_stages, receipts = returned
        if round_input(original_full) != round_input(full):
            raise ValueError("paid_staged_round_returned_response_proof_missing")
        for stage, raw in original_stages.items():
            if stage not in stages or raw != stages[stage]:
                raise ValueError("paid_staged_round_returned_response_changed")
        full = original_full
    full["_paid_replay_readonly"] = True
    return full, owner_id, stages, {"source": "native_confirmed_response", "calls": proofs,
                                  "stage_inputs": stage_inputs, "planner_delivery": planner_delivery}
