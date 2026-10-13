"""Immutable Planner decisions with append-only control-state receipts."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .action_planning import digest
from .artifacts import request_digest

LEDGER = "decision_receipts.jsonl"
_PRIORITY = {"executed": 1, "rejected": 2, "detached": 3}


def _original(camp: Path, decision_id: str) -> tuple[dict, str]:
    path = camp / "decisions" / (decision_id + ".json")
    content = path.read_bytes()
    body = json.loads(content)
    record = body.get("decision") or {}
    if record.get("decision_id") != decision_id:
        raise ValueError("decision_content_identity_conflict:" + decision_id)
    for field, payload in (("raw_digest", "raw"), ("execution_digest", "execution")):
        if record.get(field) and request_digest(request=body.get(payload)) != record[field]:
            raise ValueError("decision_" + payload + "_identity_conflict:" + decision_id)
    return body, hashlib.sha256(content).hexdigest()


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    previous = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unsigned = {key: value for key, value in row.items() if key != "receipt_hash"}
        if (row.get("previous_receipt_hash") != previous or digest(unsigned) != row.get("receipt_hash")
                or row.get("status") not in _PRIORITY):
            raise ValueError("decision_receipt_identity_conflict")
        rows.append(row)
        previous = row["receipt_hash"]
    return rows


def append_receipt(camp: Path, decision: dict | str, *, status: str, reason: str = "",
                   source: str = "", result: Any = None, task_refs: list[str] | None = None) -> dict:
    """Record one native outcome; retries adopt the same semantic receipt."""
    if status not in _PRIORITY:
        raise ValueError("decision_receipt_status_invalid")
    decision_id = str(decision.get("decision_id") if isinstance(decision, dict) else decision)
    body, original_hash = _original(camp, decision_id)
    original = body["decision"]
    if isinstance(decision, dict):
        for key in ("raw_digest", "execution_digest", "campaign_id"):
            if decision.get(key) is not None and decision.get(key) != original.get(key):
                raise ValueError("decision_receipt_input_identity_conflict:" + decision_id)
    identity = {"decision_id": decision_id, "decision_sha256": original_hash, "status": status,
                "reason": reason, "source": source, "result": copy.deepcopy(result),
                "task_refs": sorted(set(task_refs or []))}
    receipt_id = "decision_receipt_" + digest(identity)[:24]
    path = camp / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with (camp / ".decision_receipts.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = _rows(path)
        if any(row["decision_id"] == decision_id and row["decision_sha256"] != original_hash for row in rows):
            raise ValueError("decision_receipt_original_changed:" + decision_id)
        for row in rows:
            if row["receipt_id"] == receipt_id:
                return row
        prior = [row for row in rows if row["decision_id"] == decision_id]
        if prior and _PRIORITY[status] < max(_PRIORITY[row["status"]] for row in prior):
            raise ValueError("decision_receipt_terminal_status_cannot_replay:" + decision_id)
        row = {"schema_version": "eeg_research.decision_receipt.v1", **identity,
               "receipt_id": receipt_id, "created_at": time.time(),
               "raw_digest": original.get("raw_digest"), "execution_digest": original.get("execution_digest"),
               "previous_receipt_hash": rows[-1]["receipt_hash"] if rows else None}
        row["receipt_hash"] = digest(row)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row


def _project(record: dict, receipts: list[dict], original_hash: str) -> dict:
    record = copy.deepcopy(record)
    for row in receipts:
        if row["decision_sha256"] != original_hash:
            raise ValueError("decision_receipt_original_changed:" + str(record.get("decision_id")))
    if not receipts:
        return record
    # A detached/rejected action remains consumed if a lower-priority legacy
    # status is encountered on restart. Within a priority, the latest wins.
    chosen = max(enumerate(receipts), key=lambda item: (_PRIORITY[item[1]["status"]], item[0]))[1]
    record.update(consumed=True, decision_receipt=chosen["receipt_hash"], receipt_status=chosen["status"])
    if chosen["status"] == "executed":
        record["executed"] = True
    elif chosen["status"] == "rejected":
        record.update(ok=False, executed=False, rejected_before_execution=True, detail=chosen["reason"])
    else:
        result = chosen.get("result") if isinstance(chosen.get("result"), dict) else {}
        record.update(executed=True, detached=True, detach_receipt=result.get("detach_receipt"),
                      detail=chosen["reason"])
    return record


def read_projected_decision(camp: Path, decision_id: str) -> dict:
    body, original_hash = _original(camp, decision_id)
    rows = [row for row in _rows(camp / LEDGER) if row["decision_id"] == decision_id]
    body["decision"] = _project(body["decision"], rows, original_hash)
    return body


def project_decisions(camp: Path, state: dict) -> None:
    """Overlay receipts on the mutable campaign projection, never its sources."""
    rows = _rows(camp / LEDGER)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["decision_id"], []).append(row)
    for record in state.get("decisions") or []:
        decision_id = str(record.get("decision_id"))
        if decision_id not in grouped:
            continue
        _, original_hash = _original(camp, decision_id)
        record.update(_project(record, grouped[decision_id], original_hash))
