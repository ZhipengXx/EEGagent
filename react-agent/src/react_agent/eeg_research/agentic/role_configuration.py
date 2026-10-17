"""Apply future Coder/Reviewer settings without rewriting started work."""
from __future__ import annotations

import json
from pathlib import Path

from .role_config import ALIASES, FIELDS, resolve_role_config

CONFIGURABLE_ROLES = frozenset({"candidate_coder", "candidate_reviewer"})
ACTIVE_TASK_STATUSES = frozenset({"pending", "running", "waiting_for_evidence"})


def _require_stage_boundary(camp: Path, state: dict) -> None:
    selected = state.get("selected_plan") or {}
    if selected.get("current_step") in {"prepare", "design", "design_revision"} and selected.get("budget_owner"):
        # Its old allocation already includes the following Coder/Reviewer.
        # Future plans will reserve the new profile when they are admitted.
        raise ValueError("role_configuration_requires_new_plan_for_coding_budget")
    if any(state.get(key) for key in (
        "live_job", "active_suite", "queued", "unknown_process", "pending_process_intent",
        "pending_candidate_check", "repair_task",
    )) or selected.get("current_step") in {"implement", "repair", "check"}:
        raise ValueError("role_configuration_requires_stage_boundary")
    # Fold updates before examining status: a completed task's original pending
    # row must not prevent configuring the following, not-yet-created task.
    tasks = {}
    ledger = camp / "task_ledger.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tasks.setdefault(row["task_id"], {}).update(row)
    if any(row.get("role") in CONFIGURABLE_ROLES and row.get("status") in ACTIVE_TASK_STATUSES
           for row in tasks.values()):
        raise ValueError("role_configuration_has_started_coding_task")
    # A request reserved before a process exit is still started work, even when
    # there is no corresponding task-ledger row yet.
    cost_path = camp / "cost.json"
    if cost_path.is_file() and json.loads(cost_path.read_text(encoding="utf-8")).get("api_unsettled_call_ids"):
        raise ValueError("role_configuration_has_unsettled_api_request")


def apply_role_configuration(camp: Path, state: dict, overrides: dict, *, reason: str) -> dict:
    """Called under the native worker lock; changes only future role routing.

    Goal/scientific artifacts, completed role receipts, and paid request records
    keep their original configuration. Existing overrides for other roles and
    unmentioned fields are retained.
    """
    from .loop import _read, _write, event

    camp = Path(camp)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("role_configuration_reason_required")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("role_configuration_overrides_required")
    requested = {}
    for name, fields in overrides.items():
        role = ALIASES.get(name, name)
        if role not in CONFIGURABLE_ROLES:
            raise ValueError("role_configuration_only_coder_and_reviewer")
        if (role in requested or not isinstance(fields, dict) or not fields
                or set(fields).difference(FIELDS)):
            raise ValueError("invalid_role_config_fields")
        requested[role] = fields
    _require_stage_boundary(camp, state)
    path = camp / "role_config_overrides.json"
    previous = _read(path)
    if not isinstance(previous, dict):
        raise ValueError("invalid_role_config_overrides")
    updated = dict(previous)
    before = {}
    effective = {}
    for role, fields in requested.items():
        before[role] = resolve_role_config(camp, role).model_dump()
        current = previous.get(role) or {}
        if not isinstance(current, dict):
            raise ValueError("invalid_role_config_fields")
        merged = {**current, **fields}
        # An explicit thinking change has the same inheritance semantics as a
        # fresh resolve; do not retain an opposite effort from an older layer.
        if fields.get("thinking") is False and "reasoning_effort" not in fields:
            merged["reasoning_effort"] = None
        elif fields.get("thinking") is True and merged.get("reasoning_effort") is None:
            if "reasoning_effort" not in fields:
                merged["reasoning_effort"] = "high"
        effective[role] = resolve_role_config(camp, role, merged).model_dump()
        updated[role] = merged
    changed = updated != previous
    if changed:
        _write(path, updated)
        event(camp, "role_configuration_updated", reason=reason.strip(),
              scope="future_unstarted_coder_and_reviewer_tasks", requested=requested,
              previous=before, effective=effective)
    return {"changed": changed, "scope": "future_unstarted_coder_and_reviewer_tasks",
            "overrides_path": str(path), "roles": effective}
