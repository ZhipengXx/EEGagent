"""Deterministic stop rules for the V1.4 diagnostic goal."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.schemas import QuestionRecord

_DONE = {"answered", "described", "not_applicable"}


def evaluate_stop(
    ledger: list[QuestionRecord] | list[dict[str, Any]],
    tickets: list[dict[str, Any]],
    *,
    budget_ok: bool,
    executable_actions: list[str],
    planning_failed: bool = False,
) -> dict[str, Any]:
    """Decide whether a stop is legal. Does not invent quality flags."""
    rows = [_as_dict(q) for q in ledger]
    required_open = [
        q["question_id"]
        for q in rows
        if q.get("required") and q.get("status") not in _DONE and q.get("status") != "blocked"
    ]
    blocked = [q["question_id"] for q in rows if q.get("required") and q.get("status") == "blocked"]
    open_tickets = [t for t in tickets if t.get("status") == "open"]
    if planning_failed and (required_open or open_tickets):
        return {
            "allow_stop": True,
            "screening_decision": "abstain",
            "coverage": "partial",
            "stop_reason": "planning_blocked",
            "missing_question_ids": required_open,
        }
    if required_open or open_tickets:
        if executable_actions and budget_ok:
            return {
                "allow_stop": False,
                "stop_rejected": True,
                "missing_question_ids": required_open or [t.get("question_id") for t in open_tickets],
                "reason": "executable_followup_remains",
            }
        return {
            "allow_stop": True,
            "screening_decision": "abstain",
            "coverage": "partial",
            "stop_reason": "abstain",
            "missing_question_ids": required_open,
        }
    if blocked:
        return {
            "allow_stop": True,
            "screening_decision": "blocked",
            "coverage": "partial",
            "stop_reason": "blocked",
            "missing_question_ids": blocked,
        }
    return {
        "allow_stop": True,
        "screening_decision": "pass_configured",
        "coverage": "complete",
        "stop_reason": "configured_checks_complete",
        "missing_question_ids": [],
        "reference_quality": "not_assessed",
    }


def _as_dict(row: QuestionRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return row.model_dump()
