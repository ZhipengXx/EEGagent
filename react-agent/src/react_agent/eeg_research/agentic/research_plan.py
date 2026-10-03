"""Versioned research plan. plan_update is validated, persisted, and consumed next round."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.schemas import PLAN_VERSION_NAME, SCHEMA_VERSION

QUESTION_STATES = {"open", "answered", "inconclusive", "blocked"}
PLAN_FILE = "plan.json"
PLAN_DEPENDENT_ACTIONS = {"design_experiment", "propose_experiment", "implement_candidate", "run_pilot", "run_full", "replicate", "stop"}


class PlanError(ValueError):
    """Raised when a plan update cannot be applied."""


def _version_number(value: Any) -> int:
    """Integer plan version. A missing or non-numeric value is a rejected update."""
    if isinstance(value, bool) or value is None or value == "":
        raise PlanError("plan_version_missing")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlanError("plan_version_missing") from exc


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def init_plan(camp: Path, goal: dict[str, Any], *, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create plan version 1. Existing plans are left untouched."""
    existing = load_plan(camp)
    if existing:
        return existing
    questions = [
        {
            "question_id": "q1",
            "text": str(goal.get("objective") or "改进指定协议下的检索"),
            "status": "open",
        }
    ]
    plan = {
        "schema_version": PLAN_VERSION_NAME,
        "campaign_schema": SCHEMA_VERSION,
        "plan_version": 1,
        "goal_id": goal.get("goal_id"),
        "objective": goal.get("objective"),
        "milestones": ["freeze_contract", "baseline", "diagnosed_intervention", "matched_comparison"],
        "research_questions": questions,
        "active_hypothesis_id": None,
        "candidate_pool": [],
        "pending_comparisons": [],
        "reserved_confirmation": {
            "target_pairs": None if goal.get("confirmation_target_pairs") is None else int(goal.get("confirmation_target_pairs") or 0),
            "training_seeds": list((protocol or {}).get("training_seeds") or goal.get("training_seeds") or []),
        },
        "stopping_policy": {
            "max_training_jobs": goal.get("max_training_jobs"),
            "max_llm_calls": goal.get("max_llm_calls"),
            "max_gpu_seconds": goal.get("max_gpu_seconds"),
            "no_progress_is_not_score_tie": True,
        },
        "revisions": [],
        "last_consumed_decision_id": None,
        "accepted_evidence_ids": [],
        "audit_issues": [],
    }
    _persist(camp, plan)
    return plan


def load_plan(camp: Path) -> dict[str, Any]:
    """Return the current plan, or an empty dict when none exists."""
    payload = _read(camp / PLAN_FILE)
    return payload if isinstance(payload, dict) and payload.get("plan_version") else {}


def _id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def normalize_plan_update(
    update: dict[str, Any],
    decision: dict[str, Any] | None = None,
    *,
    known_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Copy cited evidence and question aliases into the fields apply_update reads.

    Does not invent evidence. Unknown cited ids stay so apply_update can reject them.
    """
    if not isinstance(update, dict):
        raise PlanError("plan_update_not_object")
    copied = dict(update)
    decision = decision if isinstance(decision, dict) else {}
    del known_evidence_ids
    for field in ("updated_questions", "question_updates"):
        if copied.get(field) is not None and not isinstance(copied[field], list):
            raise PlanError("bad_question_updates")

    evidence = _id_list(copied.get("based_on_evidence_ids"))
    if not evidence:
        cited: list[str] = []
        cited.extend(_id_list(decision.get("evidence_ids")))
        cited.extend(_id_list(decision.get("evidence_refs")))
        cited.extend(_id_list(copied.get("new_evidence_ids")))
        for row in copied.get("updated_questions") or []:
            if isinstance(row, dict):
                cited.extend(_id_list(row.get("evidence_refs") or row.get("evidence_ids")))
        seen: set[str] = set()
        evidence = []
        for item in cited:
            if item in seen:
                continue
            seen.add(item)
            evidence.append(item)
        copied["based_on_evidence_ids"] = evidence

    affected = _id_list(copied.get("affected_question_ids") or copied.get("affected_task_ids"))
    if not affected:
        found: list[str] = []
        if decision.get("question_id"):
            found.append(str(decision["question_id"]))
        for row in list(copied.get("updated_questions") or []) + list(copied.get("question_updates") or []):
            if isinstance(row, dict) and row.get("question_id"):
                found.append(str(row["question_id"]))
        copied["affected_question_ids"] = list(dict.fromkeys(found))

    if not copied.get("question_updates") and isinstance(copied.get("updated_questions"), list):
        mapped: list[dict[str, Any]] = []
        for row in copied["updated_questions"]:
            if not isinstance(row, dict) or not row.get("question_id"):
                continue
            change: dict[str, Any] = {"question_id": row["question_id"]}
            if row.get("status"):
                change["status"] = row["status"]
            note = row.get("note") or row.get("note_zh")
            if note:
                change["note"] = note
            mapped.append(change)
        copied["question_updates"] = mapped
    return copied


def _persist(camp: Path, plan: dict[str, Any]) -> None:
    _write(camp / PLAN_FILE, plan)
    archive = camp / "plan_versions" / f"v{int(plan['plan_version'])}.json"
    _write(archive, plan)


def apply_update(
    camp: Path,
    update: dict[str, Any],
    *,
    known_evidence_ids: set[str],
    current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and persist a planner plan_update. Concurrent versions are rejected."""
    plan = current or load_plan(camp)
    next_plan = validate_update(plan, update, known_evidence_ids=known_evidence_ids)
    _persist(camp, next_plan)
    return next_plan


def validate_update(
    plan: dict[str, Any],
    update: dict[str, Any],
    *,
    known_evidence_ids: set[str],
) -> dict[str, Any]:
    """Validate an update without writing, so planner repair uses the same rules."""
    if not plan:
        raise PlanError("plan_missing")
    if not isinstance(update, dict):
        raise PlanError("plan_update_not_object")
    based = update.get("based_on_plan_version")
    if based is None:
        based = update.get("plan_version")
    if _version_number(based) != _version_number(plan.get("plan_version")):
        raise PlanError("plan_version_conflict")
    evidence_ids = update.get("based_on_evidence_ids") or []
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise PlanError("plan_update_needs_evidence")
    if any(not isinstance(item, str) for item in evidence_ids):
        raise PlanError("plan_update_unknown_evidence")
    unknown = [item for item in evidence_ids if item not in known_evidence_ids]
    if unknown:
        raise PlanError("plan_update_unknown_evidence")
    affected = update.get("affected_question_ids") or update.get("affected_task_ids") or []
    if not isinstance(affected, list) or not affected or any(not isinstance(item, str) for item in affected):
        raise PlanError("plan_update_needs_affected")
    next_plan = json.loads(json.dumps(plan))
    next_plan["plan_version"] = int(plan["plan_version"]) + 1
    if update.get("active_hypothesis_id") is not None:
        next_plan["active_hypothesis_id"] = update.get("active_hypothesis_id")
    if isinstance(update.get("candidate_pool"), list):
        next_plan["candidate_pool"] = update["candidate_pool"]
    if isinstance(update.get("pending_comparisons"), list):
        next_plan["pending_comparisons"] = update["pending_comparisons"]
    if isinstance(update.get("reserved_confirmation"), dict):
        next_plan["reserved_confirmation"] = update["reserved_confirmation"]
    questions = {row.get("question_id"): dict(row) for row in next_plan.get("research_questions") or []}
    changes = update.get("question_updates") or []
    if not isinstance(changes, list) or any(not isinstance(item, dict) for item in changes):
        raise PlanError("bad_question_updates")
    for change in changes:
        qid = change.get("question_id")
        status = change.get("status")
        if not isinstance(qid, str) or qid not in questions:
            raise PlanError("unknown_question")
        if not isinstance(status, str) or status not in QUESTION_STATES:
            raise PlanError("bad_question_status")
        questions[qid]["status"] = status
        if change.get("note"):
            questions[qid]["note"] = change["note"]
    next_plan["research_questions"] = list(questions.values())
    next_plan.setdefault("revisions", []).append(
        {
            "from_version": plan["plan_version"],
            "to_version": next_plan["plan_version"],
            "based_on_evidence_ids": evidence_ids,
            "affected": affected,
            "reason_zh": update.get("reason_zh") or "",
        }
    )
    next_plan["accepted_evidence_ids"] = sorted(set(plan.get("accepted_evidence_ids") or []) | set(evidence_ids))
    return next_plan


def consume_for_planner(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Planner view: current plan plus material changes since the last accepted decision."""
    plan = load_plan(camp)
    last_id = plan.get("last_consumed_decision_id")
    decisions = list(state.get("decisions") or [])
    new_decisions = []
    seen = last_id is None
    for row in decisions:
        if not seen:
            seen = row.get("decision_id") == last_id
            continue
        if row.get("ok"):
            new_decisions.append(
                {
                    "decision_id": row.get("decision_id"),
                    "action": row.get("action"),
                    "reason_zh": row.get("reason_zh"),
                    "evidence_ids": row.get("evidence_ids") or [],
                }
            )
    evidence_ids = [row.get("evidence_id") for row in state.get("evidence") or [] if row.get("evidence_id")]
    new_evidence = [eid for eid in evidence_ids if eid not in set(plan.get("accepted_evidence_ids") or [])]
    return {
        "plan": {
            key: plan.get(key)
            for key in (
                "plan_version",
                "objective",
                "milestones",
                "research_questions",
                "active_hypothesis_id",
                "candidate_pool",
                "pending_comparisons",
                "reserved_confirmation",
                "audit_issues",
                "stopping_policy",
            )
        },
        "changes_since_last_plan": {
            "new_decisions": new_decisions[-12:],
            "new_evidence_ids": new_evidence[-20:],
        },
    }


def mark_consumed(camp: Path, decision_id: str) -> None:
    """Record which decision the current plan has already been shown against."""
    plan = load_plan(camp)
    if not plan:
        return
    plan["last_consumed_decision_id"] = decision_id
    _write(camp / PLAN_FILE, plan)
