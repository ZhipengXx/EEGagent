"""Explicit operator cancellation of one unmeasured selected task.

The caller holds the campaign worker lock. Cancellation uses the ordinary
terminal observation and budget handoff, leaving the campaign and candidates
available for a subsequent Planner decision.
"""
from __future__ import annotations

from pathlib import Path


def cancel_selected_plan(camp: Path, state: dict, reason: str,
                         expected_plan_id: str | None = None) -> dict:
    from .candidate_lifecycle import require_reconciled
    from .experiment_observations import ExperimentObservationStore
    from .loop import _read
    from .owner_budget import snapshot
    from .research_execution import single_plan_enabled
    from .selected_plan import TERMINAL, _finish, _measurement, load_plan

    camp = Path(camp)
    if not reason.strip():
        raise ValueError("selected_plan_cancel_reason_required")
    if not single_plan_enabled(camp):
        raise ValueError("selected_plan_cancel_requires_native_protocol")
    if (state.get("status") in {"cancelled", "finished"}
            or _read(camp / "control.json").get("action") == "stop"):
        raise ValueError("selected_plan_cancel_cannot_reopen_campaign")
    plan = load_plan(camp, state)
    if plan is None:
        raise ValueError("selected_plan_cancel_target_missing")
    if expected_plan_id is not None and expected_plan_id != plan["plan_id"]:
        raise ValueError("selected_plan_cancel_target_changed")
    if (state.get("live_job") or state.get("active_suite") or state.get("queued")
            or state.get("pending_candidate_check")):
        raise ValueError("selected_plan_cancel_requires_quiescent_task")
    require_reconciled(camp, state)
    budget = snapshot(camp)
    if budget.get("inflight") or budget.get("unknown"):
        raise ValueError("selected_plan_cancel_requires_claim_reconciliation")

    if plan["status"] == "cancelled":
        records = ExperimentObservationStore(camp).records(
            observation_id=plan.get("terminal_observation_id"), limit=1)
        if not records:
            raise ValueError("selected_plan_cancel_terminal_observation_missing")
        card = records[0]["body"]
        if (card["plan_id"] != plan["plan_id"] or card["execution_status"] != "cancelled"
                or card["content_digest"] != plan.get("terminal_observation_digest")):
            raise ValueError("selected_plan_cancel_terminal_observation_changed")
        # _finish is idempotent after publication and can recover the handoff
        # if the process exited between its native observation and budget steps.
        return _finish(camp, state, plan, "cancelled", detail=reason.strip())

    if plan["status"] in TERMINAL:
        raise ValueError("selected_plan_cancel_task_already_terminal")
    if plan["current_step"] not in {"prepare", "design", "implement", "measure"}:
        raise ValueError("selected_plan_cancel_requires_unmeasured_boundary")
    if (plan.get("measurement_evidence_id") or plan.get("analysis_artifact_ref")
            or _measurement(state, plan) is not None):
        raise ValueError("selected_plan_cancel_measurement_already_available")
    return _finish(camp, state, plan, "cancelled", detail=reason.strip())
