"""Explicit, append-only recovery scopes; historical allocations stay unchanged."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

JOURNAL = "research_authorizations.jsonl"
HEAD = "research_authorization_head.json"
CAPS = ("max_llm_calls", "max_training_jobs", "max_gpu_seconds", "max_api_usd")


class AuthorizationError(ValueError):
    pass


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.is_file() else []


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: Path) -> str:
    from .verification_cache import file_hash
    return file_hash(path, _stream_sha)


def _call_proofs(camp: Path, ids: list[str]) -> list[dict]:
    calls = {row["call_id"]: row for row in _rows(camp / "llm_calls.jsonl")}
    attempts = {row["call_id"]: row for row in _rows(camp / "api_operation_attempts.jsonl")}
    proofs = []
    for call_id in ids:
        call = calls.get(call_id)
        if (not call or call.get("unconfirmed_charge_possible")
                or call.get("transport_outcome") in {"outcome_unknown", "unknown_after_process_exit"}):
            raise AuthorizationError("recovery_authorization_requires_settled_calls")
        proofs.append({"call_id": call_id, "call_sha256": _digest(call),
            "attempt_sha256": _digest(attempts.get(call_id)),
            "request_sha256": _sha(camp / "llm_requests" / (call_id + ".json"))})
    return proofs


def current_authorization(camp: Path) -> dict | None:
    """Read the journal authority. The replaceable head is never authority."""
    rows = _rows(camp / JOURNAL)
    if not rows:
        if (camp / HEAD).exists():
            raise AuthorizationError("research_authorization_journal_missing")
        return None
    previous = None
    for receipt in rows:
        unsigned = {key: value for key, value in receipt.items()
                    if key not in {"authorization_hash", "memory_recovery_turn"}}
        if (receipt.get("previous_authorization_hash") != previous
                or receipt.get("authorization_hash") != _digest(unsigned)):
            raise AuthorizationError("research_authorization_receipt_changed")
        previous = receipt["authorization_hash"]
    if len(rows) != 1:
        raise AuthorizationError("research_recovery_scope_already_allocated")
    receipt = rows[-1]
    goal = _read(camp / "goal.json")
    if (receipt.get("campaign_root") != str(camp.resolve())
            or receipt.get("original_goal_sha256") != _sha(camp / "goal.json")
            or receipt.get("global_caps") != {key: goal.get(key) for key in CAPS}
            or receipt.get("original_max_candidates") != goal.get("max_candidates")
            or receipt.get("max_candidates") != 20
            or receipt.get("planner_recovery_max_calls") != 2
            or receipt.get("memory_recovery_max_calls") != 2
            or receipt.get("memory_recovery_turn") != receipt["parent_memory_turn_id"]
                + ":authorization:" + receipt["authorization_hash"]):
        raise AuthorizationError("research_authorization_scope_mismatch")
    if _sha(camp / "method_search_manifest.json") != receipt["scientific_manifest_sha256"]:
        raise AuthorizationError("research_authorization_scientific_manifest_changed")
    admissions = _rows(camp / "candidate_admissions.jsonl")
    count = receipt["candidate_admissions_used"]
    if _digest(admissions[:count]) != receipt["candidate_admissions_prefix_sha256"]:
        raise AuthorizationError("research_authorization_old_admissions_changed")
    old_ids = receipt["parent_planner_call_ids"] + receipt["parent_memory_call_ids"]
    if _call_proofs(camp, old_ids) != receipt["parent_call_proofs"]:
        raise AuthorizationError("research_authorization_old_calls_changed")
    return receipt


def effective_goal(camp: Path) -> dict:
    goal = _read(camp / "goal.json")
    receipt = current_authorization(camp)
    if receipt:
        goal = {**goal, "max_candidates": receipt["max_candidates"],
            "research_authorization_ref": receipt["authorization_hash"]}
    return goal


def apply_authorized_scope(camp: Path, state: dict) -> None:
    receipt = current_authorization(camp)
    if receipt:
        state["max_candidates"] = receipt["max_candidates"]
        state["research_authorization_ref"] = receipt["authorization_hash"]
        state["candidate_scope"] = {"max_candidates": receipt["max_candidates"],
            "original_max_candidates": receipt["original_max_candidates"],
            "authorization_ref": receipt["authorization_hash"]}
    elif state.get("research_authorization_ref"):
        raise AuthorizationError("research_authorization_projection_without_receipt")


def authorized_maintenance_turn(camp: Path, base_turn_id: str) -> tuple[str, str] | None:
    receipt = current_authorization(camp)
    if receipt and base_turn_id == receipt["parent_memory_turn_id"]:
        return receipt["memory_recovery_turn"], receipt["authorization_hash"]
    return None


def _memory_calls(camp: Path, turn: str) -> list[str]:
    with sqlite3.connect("file:" + str((camp / "memory.sqlite").resolve()) + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT calls FROM context_maintenance_turns WHERE turn_id=?", (turn,)).fetchone()
        digests = [item[0] for item in db.execute(
            "SELECT input_digest FROM context_maintenance_jobs WHERE last_turn_id=?", (turn,))]
    if not row or row[0] != 2 or not digests:
        raise AuthorizationError("research_recovery_requires_exhausted_parent_memory_turn")
    ids = []
    for attempt in _rows(camp / "api_operation_attempts.jsonl"):
        path = camp / "llm_requests" / (attempt["call_id"] + ".json")
        if not path.is_file():
            continue
        request = _read(path)
        if (request.get("operation") == "context_maintenance:summary"
                and any(digest in request.get("user", "") for digest in digests)):
            ids.append(attempt["call_id"])
    if len(ids) != 2:
        raise AuthorizationError("research_recovery_parent_memory_call_proof_missing")
    return ids


def authorize_research_recovery(camp: Path, *, authorization_id: str,
        max_candidates: int = 20, planner_recovery_max_calls: int = 2,
        memory_recovery_max_calls: int = 2, parent_planner_round_id: str | None = None,
        parent_memory_turn_id: str = "native_closed:11") -> dict:
    """Native explicit entry; never clears a block or creates an API attempt."""
    if not authorization_id.strip() or (max_candidates, planner_recovery_max_calls, memory_recovery_max_calls) != (20, 2, 2):
        raise AuthorizationError("research_recovery_requires_explicit_20_2_2_authorization")
    with (camp / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = current_authorization(camp)
        if existing:
            if existing["authorization_id"] != authorization_id:
                raise AuthorizationError("research_recovery_scope_already_allocated")
            return existing
        from .budget_inheritance import delegated_successor
        from .candidate_lifecycle import require_reconciled, sync
        from .method_suite import manifest, verify_dependencies
        from .schemas import GoalSpec
        goal = _read(camp / "goal.json")
        GoalSpec.model_validate(goal)
        state = _read(camp / "campaign_state.json")
        control = _read(camp / "control.json") if (camp / "control.json").is_file() else {}
        failure = state.get("failure") or {}
        if (state.get("status") != "blocked" or state.get("detail") != "planner_round_attempt_limit"
                or failure.get("phase") != "planner" or failure.get("recoverable") is not True
                or failure.get("error_type") != "planner_round_attempt_limit"
                or state.get("pause_after_step") or control.get("action") in {"pause", "stop"}
                or delegated_successor(camp)):
            raise AuthorizationError("research_recovery_requires_quiescent_planner_limit_block")
        if any(state.get(key) for key in ("live_job", "active_suite", "unknown_process",
                                        "pending_process_intent", "pending_candidate_check")):
            raise AuthorizationError("research_recovery_requires_quiescent_campaign")
        require_reconciled(camp, state)
        science = manifest(camp)
        if not science.get("framework_dependencies"):
            raise AuthorizationError("research_recovery_scientific_dependencies_missing")
        verify_dependencies(science["framework_dependencies"])
        sync(camp, state)
        admissions = _rows(camp / "candidate_admissions.jsonl")
        used = len(admissions) + len(state.get("inherited_candidate_admissions") or [])
        if goal.get("max_candidates") != 2 or used != 2 or state.get("max_candidates") != 2:
            raise AuthorizationError("research_recovery_requires_two_consumed_candidate_slots")
        attempts = _rows(camp / "api_operation_attempts.jsonl")
        settled = _rows(camp / "llm_calls.jsonl")
        settled_ids = {row["call_id"] for row in settled}
        reserved = {row["call_id"] for row in [*attempts, *_rows(camp / "llm_api_intents.jsonl")]}
        cost = _read(camp / "cost.json")
        if (not reserved.issubset(settled_ids) or any(row.get("unconfirmed_charge_possible")
                or row.get("transport_outcome") in {"outcome_unknown", "unknown_after_process_exit"}
                for row in settled) or cost.get("api_unsettled_call_ids") or cost.get("api_uncertain_calls")):
            raise AuthorizationError("research_recovery_requires_all_api_settled")
        round_id = parent_planner_round_id or next((row.get("planner_round_id")
            for row in reversed(attempts) if row.get("planner_round_id")), None)
        planner_ids = [row["call_id"] for row in attempts if row.get("planner_round_id") == round_id]
        if not round_id or len(planner_ids) != 6:
            raise AuthorizationError("research_recovery_requires_six_consumed_parent_planner_calls")
        decisions = [row["decision_id"] for row in state.get("decisions") or []]
        if parent_memory_turn_id != "native_closed:" + str(len(decisions)):
            raise AuthorizationError("research_recovery_parent_memory_turn_not_current_closed_group")
        memory_ids = _memory_calls(camp, parent_memory_turn_id)
        receipt = {"schema_version": "eeg_research.recovery_authorization.v1",
            "authorization_id": authorization_id, "campaign_root": str(camp.resolve()),
            "original_goal_sha256": _sha(camp / "goal.json"),
            "global_caps": {key: goal.get(key) for key in CAPS},
            "original_max_candidates": 2, "max_candidates": 20,
            "candidate_admissions_used": len(admissions), "candidate_slots_used": used,
            "candidate_admissions_prefix_sha256": _digest(admissions),
            "parent_planner_round_id": round_id, "parent_planner_call_ids": planner_ids,
            "planner_recovery_max_calls": 2, "decision_ids_at_authorization": decisions,
            "parent_memory_turn_id": parent_memory_turn_id, "parent_memory_call_ids": memory_ids,
            "memory_recovery_max_calls": 2, "parent_call_proofs": _call_proofs(camp, planner_ids + memory_ids),
            "spent_at_authorization": {key: cost.get(key) for key in ("llm_calls", "training_jobs", "gpu_seconds_used")},
            "scientific_manifest_sha256": _sha(camp / "method_search_manifest.json"),
            "scientific_dependencies": science["framework_dependencies"],
            "previous_authorization_hash": None, "authorized_at": time.time()}
        receipt["authorization_hash"] = _digest(receipt)
        receipt["memory_recovery_turn"] = parent_memory_turn_id + ":authorization:" + receipt["authorization_hash"]
        with (camp / JOURNAL).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary = (camp / HEAD).with_suffix(".tmp")
        temporary.write_text(json.dumps({"authorization_hash": receipt["authorization_hash"]}))
        temporary.replace(camp / HEAD)
        from .loop import load_state, save_state, event
        state = load_state(camp)
        save_state(camp, state)
        event(camp, "research_recovery_authorized", authorization_id=authorization_id,
              authorization_hash=receipt["authorization_hash"], max_candidates=20,
              planner_recovery_max_calls=2, memory_recovery_max_calls=2)
        return receipt
