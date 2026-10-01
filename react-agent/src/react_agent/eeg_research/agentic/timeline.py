"""Paginated research timeline. Identities are not invented from nearby timestamps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.jsonl_read import read_jsonl
from react_agent.eeg_research.agentic.paths import resolve_campaign
from react_agent.eeg_research.agentic.ui_events import LLM_ROLES
from react_agent.eeg_research.agentic.view import ACTION_ZH, worker_health
from react_agent.eeg_research.agentic.paths import safe_name

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
    events = read_jsonl(camp / "events.jsonl")
    decision_events = {str(row["decision_id"]): row for row in events["rows"]
                       if row.get("event") == "decision" and row.get("decision_id")}
    evidence = {str(row["evidence_id"]): row for row in state.get("evidence") or [] if row.get("evidence_id")}
    decisions = [row for row in state.get("decisions") or [] if isinstance(row, dict)]
    for index, row in enumerate(decisions):
        if not isinstance(row, dict):
            continue
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            continue
        action = str(row.get("action") or "")
        raw_path = camp / "decisions" / (decision_id + ".json") if safe_name(decision_id) else None
        saved = _read(raw_path) if raw_path and raw_path.resolve().is_relative_to(camp.resolve()) else None
        raw = saved.get("raw") if isinstance(saved, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        logged = decision_events.get(decision_id, {})
        refs = raw.get("evidence_ids") or row.get("evidence_ids") or raw.get("evidence_refs") or []
        refs = [ref for ref in refs if isinstance(ref, str)][:20] if isinstance(refs, list) else []
        cited = []
        for ref in refs:
            found = evidence.get(ref)
            if found:
                cited.append({key: found.get(key) for key in ("evidence_id", "kind", "candidate_id", "job_id",
                    "fidelity", "evaluation_valid", "fixed_bank_top1", "delta_vs_control_pp", "reason",
                    "analysis_artifact_id", "summary") if key in found and key != "summary"})
            else:
                cited.append({"evidence_id": ref, "available": False})
        next_row = decisions[index + 1] if index + 1 < len(decisions) else None
        artifact_refs = list(row.get("artifact_refs") or [])
        artifact_refs += [item["analysis_artifact_id"] for item in cited if item.get("analysis_artifact_id")]
        required = raw.get("required_artifact_refs") or row.get("required_artifact_refs") or []
        add(
            _item(
                event_id=f"decision:{decision_id}",
                event_type="decision",
                timestamp=_stamp(logged, "at", "timestamp") or _stamp(row, "at", "created_at", "started_at"),
                source="campaign_state.decisions",
                role="research_planner",
                status="failed" if not row.get("ok") else "pending" if row.get("executed") is False else "executed" if row.get("executed") is True else "ok",
                summary=ACTION_ZH.get(action, action),
                action=action,
                action_zh=ACTION_ZH.get(action, action),
                decision_id=decision_id,
                candidate_id=row.get("candidate_id") or raw.get("target_id"),
                job_id=row.get("job_id"),
                reason_zh=row.get("reason_zh"),
                executed=row.get("executed"),
                observation=raw.get("observed_gap") or raw.get("observation"),
                rationale=raw.get("decision_rationale") or row.get("reason_zh"),
                expected_information=raw.get("expected_information") or row.get("expected_information"),
                question_id=raw.get("question_id") or row.get("question_id"),
                stop_reason=raw.get("stop_reason"),
                evidence=cited,
                evidence_refs=refs,
                artifact_refs=list(dict.fromkeys(ref for ref in artifact_refs if isinstance(ref, str))),
                required_artifact_refs=required[:12] if isinstance(required, list) else [],
                previous_recorded_decision=None if index == 0 else {"event_id": "decision:" + str(decisions[index - 1].get("decision_id")),
                    "decision_id": decisions[index - 1].get("decision_id"), "action": decisions[index - 1].get("action")},
                next_recorded_decision=None if next_row is None else {"event_id": "decision:" + str(next_row.get("decision_id")),
                    "decision_id": next_row.get("decision_id"), "action": next_row.get("action")},
            )
        )

    diagnostics.extend(events["diagnostics"])
    for index, row in enumerate(events["rows"]):
        kind = str(row.get("event") or row.get("kind") or "event")
        if kind == "decision" and row.get("decision_id") in decision_events:
            # The state and append-only event record share an actual decision ID.
            # Project that decision once with its recorded timestamp and payload.
            if any(str(item.get("decision_id")) == str(row["decision_id"]) for item in decisions):
                continue
        stamp = _stamp(row, "at", "timestamp")
        identity = f"event:{kind}:{stamp}:{index}"
        add(
            _item(
                event_id=identity,
                event_type="decision_log" if kind == "decision" else kind,
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
    task_identity: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(tasks["rows"]):
        task_id = str(row.get("task_id") or "")
        status = str(row.get("status") or "")
        if not task_id:
            continue
        identity = {**task_identity.get(task_id, {}), **row}
        task_identity[task_id] = identity
        add(
            _item(
                event_id=f"task:{task_id}:{status}:{index}",
                event_type=f"task_{status}" if status else "task",
                timestamp=_stamp(row, "at", "created_at", "timestamp"),
                source="task_ledger.jsonl",
                role=identity.get("role"),
                status=status or None,
                summary=f"{row.get('role') or 'task'} {status}".strip(),
                task_id=task_id,
                attempt_id=identity.get("attempt_id"),
                candidate_id=identity.get("candidate_id"),
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


def _role_activity(items: list[dict[str, Any]], *, live_starts: bool,
                   campaign_status: str = "", disconnected: bool = False) -> dict[str, Any]:
    """Current calls and last outcomes are separate; completion is not activity."""
    latest: dict[str, dict[str, Any]] = {}
    starts: dict[str, dict[str, Any]] = {}
    closed: set[str] = set()
    for row in items:
        role = row.get("role")
        if role in LLM_ROLES:
            latest[str(role)] = row
        call_id = row.get("call_id")
        if not call_id:
            continue
        if row.get("event_type") == "llm_call_started":
            starts[str(call_id)] = row
        elif row.get("event_type") in {"llm_call_finished", "llm_call_failed"}:
            closed.add(str(call_id))
    open_rows = {key: row for key, row in starts.items() if key not in closed}
    terminal = campaign_status in {"finished", "cancelled", "failed"}
    unexpected_disconnect = disconnected and campaign_status not in {"finished", "cancelled", "failed", "paused", "blocked"}
    inactive = terminal or campaign_status in {"paused", "blocked", "interrupted"} or unexpected_disconnect
    roles = []
    for name in sorted(LLM_ROLES):
        last = latest.get(name)
        active = [row for row in open_rows.values() if row.get("role") == name]
        running = max(active, key=lambda row: row["timestamp"]) if active else None
        row = running or last
        last_status = None if last is None else last.get("status")
        if last_status == "ok" or (last and last.get("event_type") == "decision" and last_status != "failed"):
            last_status = "completed"
        if row is None:
            status = "not_called"
        elif running and live_starts and not inactive:
            status = "active"
        elif running and (unexpected_disconnect or campaign_status == "interrupted"):
            status = "interrupted"
        elif running and not terminal:
            status = "paused" if campaign_status == "paused" else "blocked" if campaign_status == "blocked" else "unknown"
        elif terminal:
            status = "idle"
        elif last_status in {"pending", "running"}:
            status = "queued" if last_status == "pending" else "unknown"
        elif last_status in {"failed", "partial", "blocked", "waiting_for_evidence"}:
            status = str(last_status)
        else:
            status = "idle"
        roles.append({"role": name, "status": status, "last_status": last_status,
                      "event_id": None if row is None else row.get("event_id"),
                      "call_id": None if row is None else row.get("call_id"),
                      "timestamp": None if row is None else row.get("timestamp")})
    return {"roles": roles, "activity_available": live_starts,
            "campaign_status": "interrupted" if unexpected_disconnect else campaign_status,
            "open_call_ids": sorted(open_rows) if live_starts and not inactive else []}


def campaign_timeline(root: Path, campaign: str, *, cursor: str = "", limit: int = DEFAULT_LIMIT, focus: str = "") -> dict[str, Any]:
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
    if focus:
        position = next((index for index, row in enumerate(items) if row["event_id"] == focus), None)
        if position is None:
            return {"ok": False, "error": "event_missing"}
        offset = max(0, position - size // 2)
    live_starts = any(row.get("event_type") == "llm_call_started" for row in items)
    registry = read_jsonl(camp / "artifact_registry.jsonl")
    record_index = []
    for row in registry["rows"][-48:]:
        path = Path(str(row.get("path") or "")).resolve()
        if path.suffix != ".json" or not path.is_relative_to(camp.resolve()):
            continue
        linked = next((item for item in reversed(items) if item.get("task_id") == row.get("producer_task_id")
                       and row.get("producer_task_id") and row.get("artifact_id") in (item.get("artifact_refs") or [])), None)
        record_index.append({"artifact_id": row.get("artifact_id"), "kind": row.get("kind"),
            "filename": path.name, "candidate_id": row.get("candidate_id"),
            "event_id": None if linked is None else linked["event_id"]})
    page = [{**row, "position": offset + index} for index, row in enumerate(items[offset : offset + size])]
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
        "artifact_index": record_index,
        "activity": _role_activity(items, live_starts=live_starts,
            campaign_status=str((_read(camp / "campaign_state.json") or {}).get("status") or ""),
            disconnected=worker_health(camp, _read(camp / "campaign_state.json") or {}).get("worker_disconnected") is True),
        "null_policy": {
            "missing": "field omitted or null",
            "unknown": "legacy record without a start event",
            "unlinked": "no identity shared; timestamps are not used to invent edges",
        },
    }
