"""Runtime-only, current-state exhaustion; resource reconciliation is a wait."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .action_planning import digest, recovery_option, validate_option


def state_identity(observation: dict) -> str:
    return digest({key: observation.get(key) for key in ("goal", "contract", "execution_fingerprint",
        "contract_fingerprint", "budget", "available_actions", "eligible_targets", "candidate_id",
        "candidate_ready", "candidates", "candidate_admissions", "detached_candidates", "repair_task",
        "pending_candidate_check", "implementation_lifecycle", "implementation_source", "training_target_sources",
        "artifact_read_index", "artifact_read_budget", "audit_feedback", "report_draft", "evidence")})


def legal_action_checks(observation: dict, validator) -> list[dict]:
    """Probe the actual legal operations, not coarse names alone."""
    from .planner import action_target_choices
    checks = []
    for action in observation.get("available_actions") or []:
        if action == "stop":
            continue
        targets, null_allowed = action_target_choices(observation, action)
        pairs = [*([None] if null_allowed else []), *targets]
        if not pairs:
            checks.append({"action": action, "feasible": False, "reason": "no_runtime_target"})
            continue
        for target in pairs:
            payload = {"action": action, "target_id": target}
            if action in {"design_experiment", "propose_experiment"}:
                # A legal unoccupied design slot is a real path; we cannot prove
                # scientific exhaustion by fabricating a hypothesis for it.
                checks.append({"action": action, "target_id": target, "feasible": True, "reason": "new_design_admission_available"})
                continue
            if action == "revise_report":
                checks.append({"action": action, "target_id": target, "feasible": bool((observation.get("audit_feedback") or {}).get("unresolved_issue_ids")), "reason": "report_revision_inputs"})
                continue
            if action == "retrieve_memory":
                index = [row for row in observation.get("artifact_index") or [] if row.get("verification_status") == "verified"]
                feasible = bool(index and (observation.get("artifact_read_budget") or {}).get("remaining_reads", 0) > 0)
                checks.append({"action": action, "target_id": target, "feasible": feasible or bool(observation.get("memory")), "reason": "bounded_unread_evidence_or_memory"})
                continue
            option = {"option_id": "runtime_probe", "action": action, "target_id": target,
                "observed_gap": "Current runtime feasibility", "expected_information": "Required operation output",
                "value_rationale": "Runtime probes only an existing legal operation.", "payload": payload}
            valid, record = validate_option(option, {"options": [option]}, observation, validator)
            checks.append({"action": action, "target_id": target, "feasible": valid is not None,
                           "reason": record.get("error") or "runtime_validator_passed"})
    return checks


def waiting_reason(observation: dict) -> str | None:
    if observation.get("resource_wait"):
        return "resources_or_intent_require_reconciliation"
    if observation.get("pending_candidate_check"):
        return "engineering_check_pending"
    lifecycle = observation.get("implementation_lifecycle") or {}
    if lifecycle.get("status") == "design_resolution_failed" and any(token in str(lifecycle.get("reason") or "")
            for token in ("changed", "identity", "mismatch", "invalid")):
        return "global_dependency_identity_requires_reconciliation"
    state = observation.get("controller_state") or {}
    failure = state.get("failure") or observation.get("failure") or {}
    text = str(failure.get("error_type") or failure.get("detail") or state.get("detail") or "")
    if any(part in text for part in ("gpu_busy", "gpu_resource", "unknown_process", "unsettled_api", "intent_requires")):
        return "resources_or_intent_require_reconciliation"
    if state.get("live_job") or state.get("api_unsettled_call_ids"):
        return "resources_or_intent_require_reconciliation"
    return None


def issue(camp: Path, observation: dict, records: list, validator) -> dict | None:
    if waiting_reason(observation) or (observation.get("budget") or {}).get("llm_calls_left", 1) <= 0:
        return None
    checks = legal_action_checks(observation, validator)
    if any(row["feasible"] for row in checks):
        return None
    body = {"schema_version": "eeg_research.planner_exhaustion.v1", "campaign_id": (observation.get("goal") or {}).get("goal_id"),
        "state_identity": state_identity(observation), "legal_action_checks": checks,
        "rejection_records_digest": digest(records), "bounded_supplement_exhausted": True,
        "termination_reason": "no_feasible_action", "created_at": time.time()}
    body["receipt_hash"] = digest(body)
    root = camp / "planner_exhaustion"
    root.mkdir(exist_ok=True)
    path = root / (body["receipt_hash"] + ".json")
    if not path.exists():
        path.write_text(json.dumps({"receipt": body, "rejection_records": records}, ensure_ascii=False, indent=2))
    return body


def verify(camp: Path, receipt: dict, observation: dict, validator) -> str | None:
    if not isinstance(receipt, dict):
        return "exhaustion_receipt_missing"
    if receipt.get("receipt_hash") != digest({key: value for key, value in receipt.items() if key != "receipt_hash"}):
        return "exhaustion_receipt_forged"
    path = camp / "planner_exhaustion" / (str(receipt.get("receipt_hash")) + ".json")
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return "exhaustion_receipt_not_runtime_issued"
    if (stored.get("receipt") != receipt or digest(stored.get("rejection_records")) != receipt.get("rejection_records_digest")
            or not receipt.get("bounded_supplement_exhausted")):
        return "exhaustion_receipt_forged"
    if receipt.get("state_identity") != state_identity(observation):
        return "exhaustion_receipt_stale"
    if waiting_reason(observation):
        return "exhaustion_requires_resource_reconciliation"
    checks = legal_action_checks(observation, validator)
    if any(row["feasible"] for row in checks):
        return "exhaustion_new_feasible_action"
    return None
