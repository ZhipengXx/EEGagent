"""Retry a terminal design handoff without reopening its released owner."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .action_planning import digest
from .artifacts import file_digest
from .schemas import SelectedPlan

REVISION = "eeg_research.designer_contract_retry.v1"
_CONTRACT_ERRORS = ("scientific_parameter_config_mismatch", "design_brief_invalid",
    "parent_hook_config_missing", "parent_hook_config_identity_mismatch",
    "parent_hook_config_invalid", "fidelity_invalid", "completed_design_requires_brief")


def is_contract_failure(failure: dict) -> bool:
    return str(failure.get("error_type") or "").split(":", 1)[0] in _CONTRACT_ERRORS


def is_retry_plan(plan: dict | None) -> bool:
    return bool(plan and any(row.get("kind") == "designer_contract_retry"
                            for row in plan.get("step_receipts") or []))


def _prepare(camp: Path, state: dict, parent: dict, artifact: dict, prior: dict,
             reason: str) -> dict:
    from .loop import _read, _write
    from .selected_plan import _receipt_row
    failure = (prior.get("payload") or {}).get("design_failure") or {}
    if (not is_contract_failure(failure) or prior.get("status") != "blocked"
            or parent.get("status") != "failed" or parent.get("current_step") != "design"
            or parent.get("candidate_id") or not parent.get("terminal_observation_id")
            or is_retry_plan(parent)):
        raise ValueError("designer_retry_requires_terminal_contract_failure")
    task_id = failure.get("task_id")
    calls = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text().splitlines() if line.strip()]
    paid = [row for row in calls if row.get("task_id") == task_id
            and row.get("role") == "experiment_designer" and row.get("success") is True
            and row.get("transport_outcome") == "confirmed_response"]
    if not paid:
        raise ValueError("designer_retry_requires_paid_brief")
    call = paid[-1]
    response_path = camp / "llm_responses" / (call["call_id"] + ".json")
    response = _read(response_path)
    reply = response.get("reply") or {}
    brief = reply.get("design_brief") or (reply.get("payload") or {}).get("design_brief")
    if (not isinstance(brief, dict) or digest(reply) != call.get("response_reply_sha256")
            or response.get("task_id") != task_id
            or artifact.get("producer_task_id") != task_id):
        raise ValueError("designer_retry_paid_brief_changed")
    record = _read(camp / "research_rounds" / (parent["originating_round"] + ".json"))
    proposal = next((row for row in record.get("proposals") or []
                     if row.get("proposal_id") == parent["originating_proposal"]), None)
    if proposal != parent["proposal"] or record.get("selected_proposal_id") != parent["originating_proposal"]:
        raise ValueError("designer_retry_selected_proposal_changed")
    retry_id = "plan_" + digest({"parent_plan_id": parent["plan_id"],
        "failure_sha256": artifact["sha256"], "revision": REVISION})[:32]
    path = camp / "designer_retries" / (retry_id + ".json")
    if path.is_file():
        return _read(path)
    plan = SelectedPlan(plan_id=retry_id, originating_round=parent["originating_round"],
        originating_proposal=parent["originating_proposal"], proposal=parent["proposal"],
        scientific_intent_digest=parent["scientific_intent_digest"], budget_owner="plan:" + retry_id,
        recovery_limits=parent["recovery_limits"], input_evidence_ids=parent["input_evidence_ids"],
        status="preparing", current_step="design").model_dump()
    feedback = {**copy.deepcopy(failure), "parent_plan_id": parent["plan_id"],
        "prior_artifact_id": artifact["artifact_id"], "source_call_id": call["call_id"],
        "previous_design_brief": copy.deepcopy(brief),
        "correction_instruction": "Correct the recorded native design-contract error in this prior brief. "
            "Preserve the selected research task and its scientific intervention. Return the complete corrected design brief."}
    receipt = _receipt_row(plan, "designer_contract_retry", parent_plan_id=parent["plan_id"],
        failure_artifact_id=artifact["artifact_id"], failure_sha256=artifact["sha256"],
        source_call_id=call["call_id"], paid_response_sha256=file_digest(response_path),
        revision=REVISION, reason=reason)
    prepared = {"schema_version": REVISION, "plan": plan, "receipt": receipt,
        "known_cost": record["native_cost_by_id"][parent["originating_proposal"]]["known"],
        "feedback": feedback}
    _write(path, prepared, exclusive=True)
    return prepared


def retry_contract_design(camp: Path, state: dict, *, reason: str, designer,
                          artifact: dict | None = None, prior: dict | None = None) -> dict:
    """Dispatch exactly the design tick of one deterministic retry plan."""
    from .loop import _read
    from .owner_budget import allocate_selected_plan
    from .selected_plan import (_append_plan_receipt, load_plan, save_plan, tick_selected_plan)
    camp = Path(camp)
    current = load_plan(camp, state)
    if is_retry_plan(current):
        prepared = _read(camp / "designer_retries" / (current["plan_id"] + ".json"))
        if not prepared:
            raise ValueError("designer_retry_record_missing")
    else:
        if current is None or artifact is None or prior is None:
            raise ValueError("designer_retry_requires_failed_selected_plan")
        prepared = _prepare(camp, state, current, artifact, prior, reason)
    plan_id = prepared["plan"]["plan_id"]
    existing = _read(camp / "selected_plans" / (plan_id + ".json"))
    plan = (load_plan(camp, {"selected_plan": {"plan_id": plan_id}}) if existing
            else copy.deepcopy(prepared["plan"]))
    if plan["current_step"] != "design" or plan["status"] in {"failed", "cancelled", "needs_replan"}:
        return _result(state, plan, prepared, reused=True)
    allocate_selected_plan(camp, plan_id=plan_id, task_kind=plan["proposal"]["task_kind"],
        known_cost=prepared["known_cost"], recovery_limits=plan["recovery_limits"])
    _append_plan_receipt(camp, plan, prepared["receipt"])
    _retry_rejected_guard(camp, plan)
    feedback = copy.deepcopy(prepared["feedback"])
    previous = state.get("design_failure") or {}
    feedback.update(recoverable=True,
        scientific_epoch=copy.deepcopy(previous.get("scientific_epoch", [
            row.get("evidence_id") for row in state.get("evidence") or []
            if row.get("evaluation_valid") is True and row.get("fidelity") in {"pilot", "full"}])),
        attempts_in_epoch=int(previous.get("attempts_in_epoch") or 1))
    state["design_failure"] = feedback
    # The old blocked spec remains failed until the new native design replaces
    # it. Clearing this flag makes available_actions treat it as pending work.
    state.update(status="created", experiment_failed=True)
    state.pop("failure", None)
    state.pop("detail", None)
    save_plan(camp, state, plan)
    tick_selected_plan(camp, state, None, {"designer": designer})
    return _result(state, load_plan(camp, state), prepared, reused=False)


def _retry_rejected_guard(camp: Path, plan: dict) -> None:
    """Keep one pre-execution refusal and publish one fresh native intent."""
    from .decision_receipts import read_projected_decision
    from .selected_plan import _receipt
    guards = [row for row in plan["step_receipts"]
              if row.get("kind") == "designer_contract_guard_retry"]
    if guards:
        # Adopt an appended receipt if a process stopped before save_plan.
        plan["attempt"] = max(plan["attempt"], guards[-1]["next_attempt"])
    pending = next((row for row in reversed(plan["step_receipts"])
                    if row.get("kind") == "step_intent" and row.get("step") == "design"
                    and row.get("attempt") == plan["attempt"]), None)
    if pending is None:
        return
    record = read_projected_decision(camp, pending["decision_id"])["decision"]
    if (record.get("rejected_before_execution") is not True or record.get("executed") is not False
            or record.get("detail") != "action_unavailable:design_experiment"):
        return
    if guards:
        raise ValueError("designer_contract_guard_retry_exhausted")
    plan["attempt"] += 1
    _receipt(camp, plan, "designer_contract_guard_retry", rejected_decision_id=pending["decision_id"],
             reason=record["detail"], next_attempt=plan["attempt"])


def _result(state: dict, plan: dict, prepared: dict, *, reused: bool) -> dict:
    return {"status": (state.get("experiment") or {}).get("status"),
        "experiment_ref": state.get("experiment_ref"), "retry_plan_id": plan["plan_id"],
        "parent_plan_id": prepared["receipt"]["parent_plan_id"],
        "prior_artifact_id": prepared["receipt"]["failure_artifact_id"],
        "source_call_id": prepared["receipt"]["source_call_id"],
        "current_step": plan["current_step"], "reused": reused}
