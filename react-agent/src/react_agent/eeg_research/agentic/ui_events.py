"""Append-only UI events. Writing here must not touch the cost ledger."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

UI_EVENT_VERSION = "eeg_research.ui_event.v1"
UI_EVENTS = "ui_events.jsonl"
LLM_ROLES = {
    "research_planner",
    "research_librarian",
    "experiment_designer",
    "candidate_coder",
    "candidate_reviewer",
    "result_analyst",
    "memory_curator",
    "result_auditor",
}


def append_ui_event(
    camp: Path,
    event_type: str,
    *,
    role: str | None = None,
    task_id: str | None = None,
    call_id: str | None = None,
    attempt_id: str | None = None,
    candidate_id: str | None = None,
    job_id: str | None = None,
    parent_event_id: str | None = None,
    artifact_refs: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Record one UI-visible event. Never increments llm_calls or charges budget."""
    row = {
        "schema_version": UI_EVENT_VERSION,
        "event_id": f"ue_{uuid.uuid4().hex[:12]}",
        "event_type": event_type,
        "timestamp": time.time(),
        "campaign_id": camp.name,
        "role": role,
        "task_id": task_id,
        "call_id": call_id,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "parent_event_id": parent_event_id,
        "artifact_refs": list(artifact_refs or []),
    }
    for key, value in fields.items():
        if value is not None:
            row[key] = value
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / UI_EVENTS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return row
