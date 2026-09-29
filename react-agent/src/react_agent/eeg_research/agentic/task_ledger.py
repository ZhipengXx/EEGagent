"""Typed task ledger. Recovery must match task_id, attempt_id, and input_digest."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.identity import result_matches

LEDGER = "task_ledger.jsonl"
STATUSES = {"pending", "running", "completed", "failed", "superseded", "cancelled"}


def _append(camp: Path, row: dict[str, Any]) -> dict[str, Any]:
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / LEDGER).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def create_task(
    camp: Path,
    *,
    role: str,
    input_artifact_refs: list[str],
    input_digest: str,
    expected_output_schema: str,
    candidate_id: str | None = None,
    attempt_id: str | None = None,
    supersedes_task_id: str | None = None,
    depends_on: list[str] | None = None,
    budget_reservation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one task. A revised task must name the task it supersedes."""
    row = {
        "task_id": f"task_{uuid.uuid4().hex[:12]}",
        "role": role,
        "input_artifact_refs": list(input_artifact_refs),
        "input_digest": input_digest,
        "expected_output_schema": expected_output_schema,
        "status": "pending",
        "attempt_id": attempt_id or uuid.uuid4().hex[:12],
        "candidate_id": candidate_id,
        "supersedes_task_id": supersedes_task_id,
        "depends_on": list(depends_on or []),
        "budget_reservation": budget_reservation or {},
        "created_at": time.time(),
    }
    if supersedes_task_id:
        mark(camp, supersedes_task_id, "superseded")
    return _append(camp, row)


def mark(camp: Path, task_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
    """Append a status update. The ledger is append-only."""
    if status not in STATUSES:
        raise ValueError(f"unknown_task_status:{status}")
    row = {"task_id": task_id, "status": status, "at": time.time(), **fields}
    return _append(camp, row)


def latest(camp: Path, task_id: str) -> dict[str, Any] | None:
    """Return the newest row for a task id."""
    path = camp / LEDGER
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") == task_id:
            found = row if found is None else {**found, **row}
    return found


def reusable(
    payload: Any,
    *,
    task_id: str,
    attempt_id: str,
    input_digest: str,
    candidate_id: str | None = None,
) -> bool:
    """True only when the stored result belongs to this task attempt and digest.

    Results from another candidate, another attempt, or a changed input are not reused.
    """
    if not isinstance(payload, dict) or not input_digest or not task_id or not attempt_id:
        return False
    if payload.get("task_id") != task_id:
        return False
    if candidate_id is not None and payload.get("candidate_id") not in {None, candidate_id}:
        return False
    if payload.get("attempt_id") != attempt_id:
        return False
    if payload.get("input_digest") != input_digest:
        return False
    return True


def identity_reusable(payload: Any, *, candidate_id: str, attempt_id: str, phase: str, input_hash: str) -> bool:
    """Compatibility wrapper around attempt isolation."""
    return result_matches(
        payload,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        phase=phase,
        input_hash=input_hash,
    )
