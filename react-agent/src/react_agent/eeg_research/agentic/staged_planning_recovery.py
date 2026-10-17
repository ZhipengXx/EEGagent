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
PLANNING_CONTEXT_VERSION = "research_planning_input.v1"
INPUT_SCHEMA_VERSION = "eeg_research.paid_round_input.v1"


def _native_observation(full):
    native = copy.deepcopy(full)
    native.pop("_paid_replay_readonly", None)
    return native


def publish_staged_input(camp, state, full, owner, payload, stage, *, native_pool=None):
    """Freeze native inputs locally before invoking a lean Planner or Critic.

    The model receives only ``payload``. The full observation and frozen pool
    remain outside its prompt, available even if the process exits after the
    paid response has settled but before the outer raw receipt is written.
    """
    from .loop import _write
    if stage not in STAGE_ROLES:
        raise ValueError("paid_staged_round_unknown_stage")
    if payload.get("planning_context_version") != PLANNING_CONTEXT_VERSION:
        return None  # Historical full-wire stages retain their original path.
    rid = round_id(full)
    if stage in {"critic", "critic_correction"}:
        from .planning_context import critic_pool_input
        if (not isinstance(native_pool, dict) or native_pool.get("round_id") != rid
                or native_pool.get("pool_digest") != proposal_pool_digest(native_pool)
                or payload.get("proposal_pool_digest") != native_pool["pool_digest"]
                or payload.get("proposal_pool") != critic_pool_input(native_pool)):
            raise ValueError("paid_staged_round_critic_pool_changed")
    elif native_pool is not None:
        raise ValueError("paid_staged_round_unexpected_pool")
    receipt = {"schema_version": INPUT_SCHEMA_VERSION,
        "planning_context_version": PLANNING_CONTEXT_VERSION,
        "round_id": rid, "stage": stage, "role": STAGE_ROLES[stage],
        "operation_id": rid + ":" + stage, "planning_owner": owner,
        "input_observation_digest": digest(round_input(full)), "round_input": round_input(full),
        "observation": _native_observation(full), "state_binding": paid.state_binding(camp, state),
        "model_input": copy.deepcopy(payload), "native_pool": copy.deepcopy(native_pool)}
    receipt["receipt_digest"] = digest(receipt)
    path = Path(camp) / "research_round_inputs" / rid / (stage + ".json")
    if path.is_file():
        if paid._read(path) != receipt:
            raise ValueError("paid_staged_round_input_snapshot_changed")
    else:
        _write(path, receipt, exclusive=True)
    return receipt


def _input_snapshot(camp, wire, row, owner, *, state=None):
    """Bind the paid wire to its separately frozen native observation."""
    rid = owner["round_id"]
    operation = str(row.get("operation_id") or "")
    stage = operation[len(rid) + 1:] if operation.startswith(rid + ":") else ""
    if stage not in STAGE_ROLES:
        raise ValueError("paid_staged_round_stage_identity_changed")
    receipt = paid._read(Path(camp) / "research_round_inputs" / rid / (stage + ".json"))
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    full, model_input = receipt.get("observation"), receipt.get("model_input")
    if (receipt.get("schema_version") != INPUT_SCHEMA_VERSION
            or receipt.get("planning_context_version") != PLANNING_CONTEXT_VERSION
            or receipt.get("receipt_digest") != digest(unsigned)
            or receipt.get("round_id") != rid or receipt.get("operation_id") != operation
            or receipt.get("stage") != stage or receipt.get("role") != row.get("role")
            or receipt.get("planning_owner") != owner.get("owner_id")
            or not isinstance(full, dict) or not isinstance(model_input, dict)
            or model_input.get("planning_context_version") != PLANNING_CONTEXT_VERSION
            or round_id(full) != rid or receipt.get("round_input") != round_input(full)
            or receipt.get("input_observation_digest") != digest(round_input(full))):
        raise ValueError("paid_staged_round_input_snapshot_changed")
    if state is not None and receipt.get("state_binding") != paid.state_binding(camp, state):
        raise ValueError("paid_staged_round_current_binding_changed")
    # Format/schema retries may add measured error feedback. They may not
    # replace any scientific input or supply a different candidate pool.
    added = set(wire) - set(model_input)
    if (added - {"previous_response_error"}
            or any(wire.get(key) != value for key, value in model_input.items())
            or ("previous_response_error" in added and not isinstance(wire["previous_response_error"], dict))):
        raise ValueError("paid_staged_round_delivered_input_changed")
    full = copy.deepcopy(full)
    pool = receipt.get("native_pool")
    if stage in {"critic", "critic_correction"}:
        from .planning_context import critic_pool_input
        if (not isinstance(pool, dict) or pool.get("round_id") != rid
                or pool.get("pool_digest") != proposal_pool_digest(pool)
                or model_input.get("proposal_pool_digest") != pool["pool_digest"]
                or model_input.get("proposal_pool") != critic_pool_input(pool)):
            raise ValueError("paid_staged_round_critic_pool_changed")
        full.update(proposal_pool=copy.deepcopy(pool), proposal_pool_digest=pool["pool_digest"])
    elif pool is not None:
        raise ValueError("paid_staged_round_unexpected_pool")
    full["_paid_replay_readonly"] = True
    return full, receipt


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
        "observation": _native_observation(full), "state_binding": paid.state_binding(camp, state),
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


def _returned_proof(camp, state, owner_id, returned):
    """Preserve local injected-backend delivery without pretending it is paid."""
    full, stages, _ = returned
    rid = round_id(full)
    proof = {"source": "backend_return_receipt", "stage_inputs": {}}
    for stage in stages:
        path = camp / "research_round_inputs" / rid / (stage + ".json")
        if not path.is_file():
            continue  # Historical full-wire injected backends have no snapshot.
        model_input = paid._read(path).get("model_input") or {}
        native, _ = _input_snapshot(camp, model_input,
            {"operation_id": rid + ":" + stage, "role": STAGE_ROLES[stage]},
            {"round_id": rid, "owner_id": owner_id}, state=state)
        if _native_observation(_original_observation(native)) != _native_observation(full):
            raise ValueError("paid_staged_round_response_inputs_conflict")
        if stage == "generation":
            proof["planner_delivery"] = _planner_delivery(model_input)
        if stage in {"critic", "critic_correction"}:
            proof["stage_inputs"][stage] = {key: copy.deepcopy(native[key])
                for key in ("proposal_pool", "proposal_pool_digest")}
    return proof


def _local_input_proof(camp, state, owner):
    """Recover unfinished stages' inputs without treating them as responses."""
    rid = owner["round_id"]
    folder = camp / "research_round_inputs" / rid
    if not folder.is_dir():
        return None
    if (owner.get("kind") != "continuation" or owner.get("status") not in {"active", "released"}):
        raise ValueError("paid_staged_round_planning_owner_binding_changed")
    full = None
    proof = {"source": "local_pre_call_input", "stage_inputs": {}, "stage_payloads": {}}
    for stage in STAGE_ROLES:
        path = folder / (stage + ".json")
        if not path.is_file():
            continue
        model_input = paid._read(path).get("model_input") or {}
        native, _ = _input_snapshot(camp, model_input,
            {"operation_id": rid + ":" + stage, "role": STAGE_ROLES[stage]}, owner, state=state)
        original = _original_observation(native)
        paid._verify_current(camp, state, original)
        if full is not None and _native_observation(full) != _native_observation(original):
            raise ValueError("paid_staged_round_response_inputs_conflict")
        full = original if full is None else full
        proof["stage_payloads"][stage] = copy.deepcopy(model_input)
        if stage == "generation":
            proof["planner_delivery"] = _planner_delivery(model_input)
        if stage in {"critic", "critic_correction"}:
            proof["stage_inputs"][stage] = {key: copy.deepcopy(native[key])
                for key in ("proposal_pool", "proposal_pool_digest")}
    if full is None:
        return None
    if owner.get("observation_digest") != (state.get("terminal_observation_digest") or digest(round_input(full))):
        raise ValueError("paid_staged_round_owner_observation_changed")
    return full, proof


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
        return full, owner_id, stages, _returned_proof(camp, state, owner_id, returned)
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
            return full, owner_id, stages, _returned_proof(camp, state, owner_id, returned)
        return None
    rid = owner["round_id"]
    if returned and round_id(returned[0]) != rid:
        raise ValueError("paid_staged_round_planning_owner_binding_changed")
    successful = [row for row in relevant if row.get("success") is True and
        paid._read(camp / "llm_responses" / (str(row["call_id"]) + ".json")).get("validation") == "valid"]
    local_inputs = _local_input_proof(camp, state, owner)
    if not successful:
        if returned and provider_journal:
            raise ValueError("paid_staged_round_confirmed_response_missing")
        if not returned:
            if local_inputs:
                full, proof = local_inputs
                if "generation" not in proof["stage_payloads"]:
                    raise ValueError("paid_staged_round_generation_missing")
                return full, owner_id, {}, proof
            return None
        full, stages, _ = returned
        _stages_order(stages)
        proof = _returned_proof(camp, state, owner_id, returned)
        if local_inputs:
            local_full, local_proof = local_inputs
            if _native_observation(local_full) != _native_observation(full):
                raise ValueError("paid_staged_round_response_inputs_conflict")
            proof["stage_payloads"] = local_proof["stage_payloads"]
            proof["stage_inputs"].update(local_proof["stage_inputs"])
            if local_proof.get("planner_delivery") is not None:
                proof["planner_delivery"] = local_proof["planner_delivery"]
        return full, owner_id, stages, proof
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
            allowed_roles={STAGE_ROLES[stage]}, operations={rid + ":" + stage}, state=state)
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
    stage_payloads = {}
    if local_inputs:
        local_full, local_proof = local_inputs
        if _native_observation(local_full) != _native_observation(full):
            raise ValueError("paid_staged_round_response_inputs_conflict")
        for stage, value in local_proof["stage_inputs"].items():
            if stage in stage_inputs and stage_inputs[stage] != value:
                raise ValueError("paid_staged_round_critic_inputs_conflict")
            stage_inputs[stage] = value
        stage_payloads = local_proof["stage_payloads"]
    return full, owner_id, stages, {"source": "native_confirmed_response", "calls": proofs,
        "stage_inputs": stage_inputs, "planner_delivery": planner_delivery, "stage_payloads": stage_payloads}
