"""Resume a reviewed implementation stopped by the old default session quota."""
from __future__ import annotations

import copy
from pathlib import Path

from .action_planning import digest
from .artifacts import file_digest

REVISION = "eeg_research.coder_session_quota_correction.v1"


def _prepare(camp: Path, state: dict, plan: dict, reason: str) -> dict:
    from . import candidate_lifecycle as lifecycle
    from .loop import _read
    from .identity import source_hash
    from .native_patch import _check_fingerprint
    from .experiment_gate import resolve_approved_experiment
    from .selected_plan import scientific_projection
    goal = _read(camp / "goal.json")
    config = goal.get("coding_recovery")
    candidate_id = plan.get("candidate_id")
    if (not isinstance(config, dict) or config.get("max_tool_steps") is not None
            or plan["current_step"] != "repair" or plan["status"] != "blocked"
            or plan.get("terminal_observation_id") or not candidate_id):
        raise ValueError("coder_retry_requires_default_session_quota_failure")
    lifecycle.sync(camp, state)
    lifecycle.require_reconciled(camp, state)
    detached = (state.get("detached_candidates") or {}).get(candidate_id) or {}
    failed = detached.get("candidate_failure") or {}
    recovery = _read(camp / "candidates" / candidate_id / "coding_recovery.json")
    if (detached.get("reason") != "step_limit" or failed.get("status") != "implementation_failed"
            or failed.get("detail") != "step_limit" or detached.get("suite_id")
            or detached.get("invalid_job_failure") or detached.get("suite_engineering_failure")
            or failed.get("steps") != recovery.get("steps_used")
            or recovery.get("steps_used") != recovery.get("tool_step_limit")
            or int(recovery.get("failed_checks_used", 0)) >= int(recovery.get("failed_check_limit", 0))):
        raise ValueError("coder_retry_requires_recorded_session_quota_failure")
    workspace = camp / "candidates" / candidate_id
    attempt = _read(workspace / "attempt.json")
    spec = _read(workspace / "spec.json")
    review_path = workspace / ("review.json." + str(attempt.get("supersedes_attempt_id") or ""))
    review = _read(review_path)
    issues = review.get("issues") or []
    if (attempt.get("candidate_id") != candidate_id or not attempt.get("attempt_id")
            or not (attempt.get("operations") or {}).get("implement_candidate")
            or review.get("status") != "needs_fix" or review.get("candidate_id") != candidate_id
            or review.get("attempt_id") != attempt.get("supersedes_attempt_id")
            or not issues or any(row.get("kind") != "code_error" for row in issues)
            or spec.get("repair_issues") != issues):
        raise ValueError("coder_retry_requires_native_reviewer_repair")
    approved = resolve_approved_experiment(camp, target_id=candidate_id,
        attempt_id=attempt["attempt_id"], state={}, action="repair")
    if (approved.get("spec_hash") != plan.get("spec_hash")
            or review.get("approved_spec_hash") != plan.get("spec_hash")
            or (spec.get("experiment") or {}).get("spec_hash") != plan.get("spec_hash")
            or digest(scientific_projection(approved)) != plan.get("approved_scientific_digest")):
        raise ValueError("coder_retry_approved_spec_changed")
    source = source_hash(workspace)
    cpu = _read(workspace / "cpu_checks.json")
    full = _read(workspace / "checks.json")
    if (not source or cpu.get("ok") is not True or cpu.get("source_sha256") != source
            or cpu.get("check_scope") != "cpu"
            or cpu.get("check_fingerprint") != _check_fingerprint(workspace, cpu.get("python"), cpu_only=True)
            or (failed.get("check") or {}).get("source_sha256") != source
            or full.get("source_sha256") == source):
        raise ValueError("coder_retry_requires_fresh_cpu_only_repair")
    from .decision_receipts import read_projected_decision
    intents = [row for row in plan["step_receipts"] if row.get("kind") == "step_intent"
               and row.get("step") == "repair" and row.get("decision_id") in detached.get("decision_ids", [])]
    if not intents:
        raise ValueError("coder_retry_failed_native_step_missing")
    decision = read_projected_decision(camp, intents[-1]["decision_id"])["decision"]
    if decision.get("detached") is not True or decision.get("origin_plan_id") != plan["plan_id"]:
        raise ValueError("coder_retry_failed_native_step_changed")
    repair = {"candidate_id": candidate_id, "attempt_id": attempt["attempt_id"],
        "remaining": 1, "used": plan["repair_count"], "issues": copy.deepcopy(issues),
        "intervention_coverage": copy.deepcopy(review.get("intervention_coverage")),
        "unverified_invariants": copy.deepcopy(review.get("unverified_invariants") or []),
        "previous_review": copy.deepcopy((spec.get("review_focus") or {}).get("previous_review")),
        "review_focus": copy.deepcopy(spec.get("review_focus") or {"issues": issues})}
    return lifecycle._append(camp / lifecycle.FRAMEWORK_CORRECTIONS, {
        "schema_version": REVISION, "candidate_id": candidate_id, "plan_id": plan["plan_id"],
        "plan_attempt": plan["attempt"], "detach_receipt": detached["receipt_hash"],
        "failed_decision_id": decision["decision_id"], "attempt_id": attempt["attempt_id"],
        "source_hash": source, "spec_hash": plan["spec_hash"], "repair_task": repair,
        "review_sha256": file_digest(review_path), "reason": reason})


def retry_coder_stage(camp: Path, state: dict, *, reason: str, services: dict) -> dict:
    """Correct the old detach projection, then execute one native repair tick."""
    from . import candidate_lifecycle as lifecycle
    from .loop import _read
    from .identity import source_hash
    from .experiment_gate import resolve_approved_experiment
    from .run_context import load_approved_binding
    from .selected_plan import load_plan, save_plan, tick_selected_plan, _receipt
    from .coder_submission_recovery import REVISION as REFERENCE_REVISION, preview_initial_submission_recovery
    camp = Path(camp)
    if (not reason.strip() or state.get("evaluation_mode") != "loso_method_search"
            or state.get("status") in {"paused", "cancelled"} or state.get("live_job")
            or state.get("active_suite") or state.get("queued")):
        raise ValueError("coder_retry_requires_quiescent_method_campaign_and_reason")
    plan = load_plan(camp, state)
    if plan is None:
        raise ValueError("coder_retry_selected_plan_missing")
    prior = [row for row in lifecycle._rows(camp / lifecycle.FRAMEWORK_CORRECTIONS)
             if row.get("schema_version") in {REVISION, REFERENCE_REVISION} and row.get("plan_id") == plan["plan_id"]]
    if prior:
        correction = prior[-1]
    elif plan["current_step"] == "implement":
        correction = lifecycle._append(camp / lifecycle.FRAMEWORK_CORRECTIONS,
            preview_initial_submission_recovery(camp, state, plan, reason))
    else:
        correction = _prepare(camp, state, plan, reason)
    lifecycle.sync(camp, state)
    candidate_id = correction["candidate_id"]
    if lifecycle.is_detached(state, candidate_id):
        raise ValueError("coder_retry_new_detachment_requires_new_diagnosis")
    if (correction["schema_version"] == REFERENCE_REVISION and plan["current_step"] == "implement"
            and plan["status"] == "blocked" and plan["attempt"] == correction["plan_attempt"]):
        plan.update(current_step="repair", status="repairing")
    if plan["current_step"] != "repair" or plan["status"] in {"failed", "cancelled", "needs_replan", "observation_ready"}:
        return _result(state, plan, correction, reused=True)
    lifecycle.require_reconciled(camp, state)
    dispatched = any(row.get("kind") == "step_intent" and row.get("step") == "repair"
                     and row.get("attempt") == correction["plan_attempt"] for row in plan["step_receipts"])
    if plan["attempt"] == correction["plan_attempt"] and not dispatched:
        workspace = camp / "candidates" / candidate_id
        attempt = _read(workspace / "attempt.json")
        if (attempt.get("attempt_id") != correction["attempt_id"]
                or source_hash(workspace) != correction["source_hash"]
                or plan.get("spec_hash") != correction["spec_hash"]):
            raise ValueError("coder_retry_prepared_inputs_changed")
        approved = resolve_approved_experiment(camp, target_id=candidate_id,
            attempt_id=correction["attempt_id"], state={}, action="repair")
        binding = load_approved_binding(camp, candidate_id)
        state.update(experiment=approved, experiment_ref=binding["spec_ref"],
            hypothesis=approved.get("hypothesis"), pending_candidate_id=candidate_id,
            candidate_ready=False, experiment_failed=False, status="created",
            repair_task=copy.deepcopy(correction["repair_task"]))
        state.pop("failure", None)
        state.pop("detail", None)
        for row in state.get("candidates") or []:
            if row.get("candidate_id") == candidate_id:
                row.update(lifecycle_status="implementation_recovery",
                           framework_correction_receipt=correction["receipt_hash"])
        if not any(row.get("kind") == "coder_stage_retry" for row in plan["step_receipts"]):
            _receipt(camp, plan, "coder_stage_retry", correction_receipt=correction["receipt_hash"],
                detach_receipt=correction["detach_receipt"], failed_decision_id=correction["failed_decision_id"],
                attempt_id=correction["attempt_id"], source_hash=correction["source_hash"])
        save_plan(camp, state, plan)
    elif dispatched and state.get("status") == "blocked":
        failure = state.get("failure") or {}
        if failure.get("recoverable") is True and failure.get("phase") in {
                "implement_candidate", "repair_candidate", "review_candidate", "resources"}:
            state["status"] = "created"
            state.pop("failure", None)
            state.pop("detail", None)
            save_plan(camp, state, plan)
    tick_selected_plan(camp, state, None, services)
    return _result(state, load_plan(camp, state), correction, reused=False)


def _result(state: dict, plan: dict, correction: dict, *, reused: bool) -> dict:
    return {"plan_id": plan["plan_id"], "candidate_id": correction["candidate_id"],
        "current_step": plan["current_step"], "status": plan["status"],
        "candidate_ready": bool(state.get("candidate_ready")),
        "correction_receipt": correction["receipt_hash"], "reused": reused}
