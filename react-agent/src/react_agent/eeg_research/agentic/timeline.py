"""Paginated research timeline. Identities are not invented from nearby timestamps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.jsonl_read import read_jsonl
from react_agent.eeg_research.agentic.paths import resolve_campaign
from react_agent.eeg_research.agentic.ui_events import LLM_ROLES
from react_agent.eeg_research.agentic.view import ACTION_ZH

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


def _read(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _stamp(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _item(
    *,
    event_id: str,
    event_type: str,
    timestamp: float,
    source: str,
    role: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    row = {
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "source": source,
        "role": role,
        "status": status,
        "summary": summary,
        "task_id": None,
        "call_id": None,
        "attempt_id": None,
        "candidate_id": None,
        "job_id": None,
        "parent_event_id": None,
        "artifact_refs": [],
        "reason_zh": None,
    }
    row.update(fields)
    return row


def _collect(camp: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        key = str(row.get("event_id") or "")
        if not key or key in seen:
            return
        seen.add(key)
        items.append(row)

    state = _read(camp / "campaign_state.json") or {}
    for row in state.get("decisions") or []:
        if not isinstance(row, dict):
            continue
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            continue
        action = str(row.get("action") or "")
        add(
            _item(
                event_id=f"decision:{decision_id}",
                event_type="decision",
                timestamp=_stamp(row, "at", "created_at", "started_at"),
                source="campaign_state.decisions",
                role="research_planner",
                status="ok" if row.get("ok") else "failed",
                summary=ACTION_ZH.get(action, action),
                action=action,
                action_zh=ACTION_ZH.get(action, action),
                decision_id=decision_id,
                candidate_id=row.get("candidate_id"),
                job_id=row.get("job_id"),
                reason_zh=row.get("reason_zh"),
                executed=row.get("executed"),
            )
        )

    events = read_jsonl(camp / "events.jsonl")
    diagnostics.extend(events["diagnostics"])
    for index, row in enumerate(events["rows"]):
        kind = str(row.get("event") or row.get("kind") or "event")
        stamp = _stamp(row, "at", "timestamp")
        identity = f"event:{kind}:{stamp}:{index}"
        add(
            _item(
                event_id=identity,
                event_type=kind,
                timestamp=stamp,
                source="events.jsonl",
                status=row.get("status"),
                summary=kind,
                candidate_id=row.get("candidate_id"),
                job_id=row.get("job_id"),
                detail=row.get("reason") or row.get("detail"),
            )
        )

    llm = read_jsonl(camp / "llm_calls.jsonl")
    diagnostics.extend(llm["diagnostics"])
    for row in llm["rows"]:
        call_id = str(row.get("call_id") or "")
        if not call_id:
            continue
        success = row.get("success")
        add(
            _item(
                event_id=f"llm:{call_id}:{'ok' if success else 'fail'}",
                event_type="llm_call_finished" if success else "llm_call_failed",
                timestamp=_stamp(row, "started_at", "timestamp"),
                source="llm_calls.jsonl",
                role=row.get("role"),
                status="completed" if success else "failed",
                summary=str(row.get("role") or "llm"),
                call_id=call_id,
                candidate_id=row.get("candidate_id"),
                job_id=row.get("job_id"),
                error=row.get("error"),
                elapsed_seconds=row.get("elapsed_seconds"),
            )
        )

    tasks = read_jsonl(camp / "task_ledger.jsonl")
    diagnostics.extend(tasks["diagnostics"])
    for index, row in enumerate(tasks["rows"]):
        task_id = str(row.get("task_id") or "")
        status = str(row.get("status") or "")
        if not task_id:
            continue
        add(
            _item(
                event_id=f"task:{task_id}:{status}:{index}",
                event_type=f"task_{status}" if status else "task",
                timestamp=_stamp(row, "at", "created_at", "timestamp"),
                source="task_ledger.jsonl",
                role=row.get("role"),
                status=status or None,
                summary=f"{row.get('role') or 'task'} {status}".strip(),
                task_id=task_id,
                attempt_id=row.get("attempt_id"),
                candidate_id=row.get("candidate_id"),
                artifact_refs=[row["artifact_id"]] if row.get("artifact_id") else [],
            )
        )

    ui = read_jsonl(camp / "ui_events.jsonl")
    diagnostics.extend(ui["diagnostics"])
    for row in ui["rows"]:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        add(
            _item(
                event_id=event_id,
                event_type=str(row.get("event_type") or "ui"),
                timestamp=_stamp(row, "timestamp", "at"),
                source="ui_events.jsonl",
                role=row.get("role"),
                status=row.get("status"),
                summary=row.get("summary") or row.get("event_type"),
                task_id=row.get("task_id"),
                call_id=row.get("call_id"),
                attempt_id=row.get("attempt_id"),
                candidate_id=row.get("candidate_id"),
                job_id=row.get("job_id"),
                parent_event_id=row.get("parent_event_id"),
                artifact_refs=list(row.get("artifact_refs") or []),
                tool=row.get("tool"),
                error=row.get("error"),
            )
        )

    jobs_root = camp / "jobs"
    if jobs_root.is_dir():
        for job_dir in sorted(path for path in jobs_root.iterdir() if path.is_dir()):
            record = _read(job_dir / "job.json")
            if not isinstance(record, dict):
                continue
            job_id = str(record.get("job_id") or job_dir.name)
            add(
                _item(
                    event_id=f"job:{job_id}",
                    event_type="training_job",
                    timestamp=_stamp(record, "started_at", "at"),
                    source="jobs/<job_id>/job.json",
                    role=None,
                    status=record.get("status"),
                    summary=f"Training worker · {record.get('fidelity') or job_id}",
                    candidate_id=record.get("candidate_id"),
                    job_id=job_id,
                    seed=record.get("seed") if record.get("seed") is not None else record.get("training_seed"),
                    fidelity=record.get("fidelity"),
                )
            )

    items.sort(key=lambda row: (float(row.get("timestamp") or 0), str(row.get("event_id"))))
    return items, diagnostics


def _role_activity(items: list[dict[str, Any]], *, live_starts: bool) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for row in items:
        role = row.get("role")
        if role not in LLM_ROLES:
            continue
        latest[str(role)] = row
    open_calls: set[str] = set()
    finished: set[str] = set()
    for row in items:
        call_id = row.get("call_id")
        if not call_id:
            continue
        if row.get("event_type") == "llm_call_started":
            open_calls.add(str(call_id))
        if row.get("event_type") in {"llm_call_finished", "llm_call_failed"}:
            finished.add(str(call_id))
            open_calls.discard(str(call_id))
    roles = []
    for name in sorted(LLM_ROLES):
        row = latest.get(name)
        status = "not_called"
        if row is None:
            status = "not_called"
        elif row.get("event_type") == "llm_call_started" and live_starts:
            status = "active"
        elif row.get("event_type") in {"llm_call_failed", "task_failed"}:
            status = "failed"
        elif row.get("status") in {"pending"}:
            status = "queued"
        elif row.get("status") in {"partial", "blocked", "waiting_for_evidence", "superseded", "failed"}:
            status = str(row["status"])
        elif row.get("event_type") in {"llm_call_finished", "task_completed", "decision"}:
            status = "completed"
        elif row.get("status") in {"completed", "ok"}:
            status = "completed"
        else:
            status = "unknown" if not live_starts else str(row.get("status") or "unknown")
        if not live_starts and status == "active":
            status = "unknown"
        roles.append(
            {
                "role": name,
                "status": status,
                "event_id": None if row is None else row.get("event_id"),
                "call_id": None if row is None else row.get("call_id"),
                "timestamp": None if row is None else row.get("timestamp"),
            }
        )
    return {
        "roles": roles,
        "activity_available": live_starts,
        "open_call_ids": sorted(open_calls - finished) if live_starts else [],
    }


def campaign_timeline(root: Path, campaign: str, *, cursor: str = "", limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    camp = resolve_campaign(root, campaign)
    if camp is None:
        return {"ok": False, "error": "campaign_missing"}
    try:
        size = int(limit)
    except (TypeError, ValueError):
        size = DEFAULT_LIMIT
    size = max(1, min(MAX_LIMIT, size))
    try:
        offset = int(cursor or "0")
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)
    items, diagnostics = _collect(camp)
    live_starts = any(row.get("event_type") == "llm_call_started" for row in items)
    page = items[offset : offset + size]
    next_offset = offset + size
    has_more = next_offset < len(items)
    return {
        "ok": True,
        "schema_version": "eeg_research.timeline.v1",
        "campaign_id": camp.name,
        "events": page,
        "cursor": str(offset),
        "next_cursor": str(next_offset) if has_more else None,
        "has_more": has_more,
        "total": len(items),
        "diagnostics": diagnostics,
        "activity": _role_activity(items, live_starts=live_starts),
        "null_policy": {
            "missing": "field omitted or null",
            "unknown": "legacy record without a start event",
            "unlinked": "no identity shared; timestamps are not used to invent edges",
        },
    }
