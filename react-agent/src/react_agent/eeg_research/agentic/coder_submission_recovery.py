"""Restore a paid finish submission rejected by the old symbol resolver."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from .action_planning import digest

REVISION = "eeg_research.coder_source_reference_correction.v1"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def preview_initial_submission_recovery(camp: Path, state: dict, plan: dict, reason: str) -> dict:
    """Verify the old paid action without executing it or changing artifacts."""
    from . import candidate_lifecycle as lifecycle
    from .coder_contract import validate_coder_request
    from .decision_receipts import read_projected_decision
    from .experiment_gate import resolve_approved_experiment
    from .identity import source_hash
    from .loop import _read
    from .native_patch import _check_fingerprint, _short
    from .review_context import _symbol_line, valid_source_reference
    from .selected_plan import _measurement, scientific_projection

    camp = Path(camp)
    goal = _read(camp / "goal.json")
    config = goal.get("coding_recovery")
    candidate_id = plan.get("candidate_id")
    limit = min(int(plan.get("recovery_limits", {}).get("max_engineering_repairs", 2)),
                int(state.get("max_repairs_per_candidate", 2)))
    if (not reason.strip() or not isinstance(config, dict) or config.get("max_tool_steps") is not None
            or plan.get("current_step") != "implement" or plan.get("status") != "blocked"
            or plan.get("terminal_observation_id") or not candidate_id or plan.get("repair_count", 0) >= limit
            or state.get("live_job") or state.get("active_suite") or state.get("queued")
            or state.get("pending_candidate_check") or plan.get("measurement_evidence_id")
            or plan.get("analysis_artifact_ref") or _measurement(state, plan) is not None):
        raise ValueError("coder_retry_requires_unmeasured_initial_submission_failure")
    lifecycle.sync(camp, state)
    lifecycle.require_reconciled(camp, state)
    detached = (state.get("detached_candidates") or {}).get(candidate_id) or {}
    failed = detached.get("candidate_failure") or {}
    workspace = camp / "candidates" / candidate_id
    recovery = _read(workspace / "coding_recovery.json")
    sessions = recovery.get("tool_sessions") or []
    rows = _rows(workspace / "coder_log.jsonl")
    last = rows[-1] if rows else {}
    if (detached.get("reason") != "coding_lineage_budget_exhausted"
            or failed.get("status") != "implementation_failed" or failed.get("detail") != detached.get("reason")
            or detached.get("suite_id") or detached.get("invalid_job_failure") or detached.get("suite_engineering_failure")
            or recovery.get("candidate_id") != candidate_id or len(sessions) != 1 or sessions[0].get("kind") != "initial"
            or failed.get("steps") != recovery.get("steps_used")
            or recovery.get("steps_used") != recovery.get("tool_step_limit")
            or last.get("step") != recovery.get("steps_used") or last.get("tool") != "finish_patch"
            or (last.get("result") or {}).get("error") != "implementation_map_invalid_source_reference"
            or recovery.get("failed_checks_used", 0) >= recovery.get("failed_check_limit", 0)):
        raise ValueError("coder_retry_requires_recorded_initial_reference_failure")
    attempt = _read(workspace / "attempt.json")
    binding = recovery.get("operation_binding") or {}
    spec = _read(workspace / "spec.json")
    source = source_hash(workspace)
    cpu = _read(workspace / "cpu_checks.json")
    full = _read(workspace / "checks.json")
    if (not source or attempt.get("candidate_id") != candidate_id or not attempt.get("attempt_id")
            or binding.get("attempt_id") != attempt["attempt_id"] or binding.get("candidate_id") != candidate_id
            or binding.get("operation_id") != (attempt.get("operations") or {}).get("implement_candidate")
            or cpu.get("ok") is not True or cpu.get("check_scope") != "cpu"
            or cpu.get("source_sha256") != source or (failed.get("check") or {}).get("source_sha256") != source
            or cpu.get("check_fingerprint") != _check_fingerprint(workspace, cpu.get("python"), cpu_only=True)
            or full.get("source_sha256") == source):
        raise ValueError("coder_retry_requires_unchanged_cpu_checked_initial_source")
    approved = resolve_approved_experiment(camp, target_id=candidate_id,
        attempt_id=attempt["attempt_id"], state={}, action="repair")
    if (approved.get("spec_hash") != plan.get("spec_hash")
            or (spec.get("experiment") or {}).get("spec_hash") != plan.get("spec_hash")
            or digest(scientific_projection(approved)) != plan.get("approved_scientific_digest")):
        raise ValueError("coder_retry_approved_spec_changed")

    operation = binding["operation_id"] + ":coder_step:" + str(last["step"])
    paid = [row for row in _rows(camp / "llm_calls.jsonl") if row.get("role") == "candidate_coder"
        and row.get("candidate_id") == candidate_id and row.get("attempt_id") == attempt["attempt_id"]
        and row.get("operation_id") == operation and row.get("success") is True
        and row.get("transport_outcome") == "confirmed_response" and row.get("finish_reason") == "stop"]
    if not paid:
        raise ValueError("coder_retry_paid_finish_response_missing")
    call = paid[-1]
    response_path = camp / "llm_responses" / (str(call["call_id"]) + ".json")
    response = _read(response_path)
    reply = response.get("reply")
    if (response.get("validation") != "valid" or response.get("candidate_id") != candidate_id
            or response.get("attempt_id") != attempt["attempt_id"] or digest(reply) != call.get("response_reply_sha256")):
        raise ValueError("coder_retry_paid_finish_response_changed")
    tool, args, error = validate_coder_request(reply, {"experiment_spec": spec})
    if tool != "finish_patch" or error is not None or _short(args) != last.get("args"):
        raise ValueError("coder_retry_paid_finish_action_changed")
    text = (workspace / "extension/eeg_candidate.py").read_text()
    allowed = {str(workspace / "extension/eeg_candidate.py"), "extension/eeg_candidate.py"}
    assignment_lines = {node.lineno for node in ast.walk(ast.parse(text)) if isinstance(node, (ast.Assign, ast.AnnAssign))}
    assignment_refs = []
    mapping = args.get("implementation_map") or {}
    for row in mapping.values():
        for reference in row.get("source_refs") or []:
            if not valid_source_reference(reference, allowed, text):
                raise ValueError("coder_retry_source_reference_still_invalid")
            path, separator, symbol = reference.partition("::")
            if separator and path in allowed and _symbol_line(text, symbol) in assignment_lines:
                assignment_refs.append(reference)
    if not assignment_refs:
        raise ValueError("coder_retry_requires_assignment_reference_parser_correction")
    intents = [row for row in plan["step_receipts"] if row.get("kind") == "step_intent"
        and row.get("step") == "implement" and row.get("decision_id") in detached.get("decision_ids", [])]
    if not intents:
        raise ValueError("coder_retry_failed_native_step_missing")
    decision = read_projected_decision(camp, intents[-1]["decision_id"])["decision"]
    if decision.get("detached") is not True or decision.get("origin_plan_id") != plan["plan_id"]:
        raise ValueError("coder_retry_failed_native_step_changed")
    issue = {"kind": "code_error", "category": "framework_source_reference_parser_corrected",
        "severity": "blocking", "detail": "The framework now resolves existing assignment symbols. Complete the unchanged pending submission.",
        "smallest_correction": "Reuse the paid pending finish_patch submission; retain the approved source and scientific method.",
        "evidence_refs": assignment_refs, "pending_submission": copy.deepcopy(args)}
    return {"schema_version": REVISION, "candidate_id": candidate_id, "plan_id": plan["plan_id"],
        "plan_attempt": plan["attempt"], "detach_receipt": detached["receipt_hash"],
        "failed_decision_id": decision["decision_id"], "attempt_id": attempt["attempt_id"],
        "source_hash": source, "spec_hash": plan["spec_hash"], "paid_call_id": call["call_id"],
        "paid_response_sha256": call["response_reply_sha256"], "assignment_references": assignment_refs,
        "repair_task": {"candidate_id": candidate_id, "remaining": 1, "used": plan["repair_count"] + 1,
            "issues": [issue], "initial_implementation_review": True,
            "review_focus": {"issues": [issue]}}, "reason": reason.strip()}
