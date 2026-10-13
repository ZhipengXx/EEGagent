"""Append-only scientific admission and explicit failed-branch termination."""
from __future__ import annotations

import json
import os
import time
import copy
import math
from pathlib import Path

from .action_planning import digest

ADMISSIONS = "candidate_admissions.jsonl"
DETACHMENTS = "candidate_detachments.jsonl"


def enabled(goal: dict) -> bool:
    return goal.get("evaluation_schedule") == "screen_then_confirm" or goal.get("action_planning") is not None


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    previous = None
    for row in rows:
        unsigned = {k: v for k, v in row.items() if k != "receipt_hash"}
        if row.get("previous_receipt_hash") != previous or digest(unsigned) != row.get("receipt_hash"):
            raise ValueError("candidate_lineage_ledger_identity_changed:" + path.name)
        previous = row["receipt_hash"]
    return rows


def _append(path: Path, row: dict) -> dict:
    rows = _rows(path)
    row = {**copy.deepcopy(row), "previous_receipt_hash": rows[-1]["receipt_hash"] if rows else None}
    row["receipt_hash"] = digest(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def sync(camp: Path, state: dict) -> None:
    if not enabled(_read(camp / "goal.json")) and state.get("candidate_admission_protocol") != "approval_allocation.v2":
        return
    rows = _rows(camp / ADMISSIONS)
    if not rows and any(row.get("candidate_id") not in {None, "baseline"} for row in state.get("candidates") or []):
        raise ValueError("scientific_candidate_admission_inheritance_required")
    ids = {row.get("admission_id") for row in rows}
    if any(row.get("admission_id") not in ids for row in state.get("candidate_admissions") or []):
        raise ValueError("candidate_admission_ledger_missing")
    state.update(candidate_admission_protocol="approval_allocation.v2", candidate_admissions=rows)
    goal = _read(camp / "goal.json")
    inherited = []
    if goal.get("historical_budget_receipt_hash") or (camp / "budget_inheritance.json").is_file():
        from .budget_inheritance import validate_inheritance
        if not (camp / "budget_inheritance.json").is_file():
            raise ValueError("scientific_admission_inheritance_receipt_missing")
        receipt = validate_inheritance(camp)
        quota = receipt.get("scientific_admission_history")
        if (not isinstance(quota, dict) or quota.get("schema_version") != "eeg_research.scientific_admission_history.v1"
                or goal.get("inherited_scientific_admission_receipt_hash") != receipt["receipt_hash"]
                or goal.get("inherited_scientific_admissions") != quota.get("admissions")):
            raise ValueError("scientific_admission_history_proof_required_for_new_protocol")
        inherited = quota["admissions"]
    elif goal.get("inherited_scientific_admissions") or goal.get("inherited_scientific_admission_receipt_hash"):
        raise ValueError("scientific_admission_inheritance_receipt_missing")
    state["inherited_candidate_admissions"] = inherited
    detachments = _rows(camp / DETACHMENTS)
    state["detached_candidates"] = {row["candidate_id"]: row for row in detachments}


def admit(camp: Path, state: dict, candidate_id: str, spec: dict) -> dict | None:
    """Approval + allocation consumes a slot permanently; repairs reuse it."""
    if not enabled(_read(camp / "goal.json")):
        return None
    from .experiment_gate import experiment_is_approved
    if not experiment_is_approved(spec) or not spec.get("spec_hash"):
        raise ValueError("scientific_admission_requires_approved_spec")
    sync(camp, state)
    method_id = digest({"spec_hash": spec["spec_hash"], "hypothesis": spec.get("hypothesis"),
                        "intervention": spec.get("intervention")})
    existing = next((row for row in state["candidate_admissions"]
                     if row["candidate_id"] == candidate_id and row["method_identity"] == method_id), None)
    if existing:
        if is_detached(state, candidate_id):
            raise ValueError("detached_candidate_cannot_reset_scientific_lineage")
        state["pending_candidate_id"] = candidate_id
        return existing
    if any(row["method_identity"] == method_id for row in state["candidate_admissions"]):
        raise ValueError("scientific_method_already_admitted_use_existing_lineage")
    if len(state["candidate_admissions"]) + len(state.get("inherited_candidate_admissions") or []) >= int(state.get("max_candidates", 4)):
        raise ValueError("scientific_candidate_admission_budget_exhausted")
    row = _append(camp / ADMISSIONS, {"schema_version": "eeg_research.scientific_admission.v2",
        "admission_id": "admission_" + digest([candidate_id, method_id])[:20],
        "candidate_id": candidate_id, "method_identity": method_id, "spec_hash": spec["spec_hash"],
        "experiment_ref": state.get("experiment_ref") or spec.get("experiment_ref"),
        "campaign_id": state.get("goal_id"), "execution_fingerprint": state.get("execution_fingerprint"),
        "committed_at": time.time(), "refund_on_failure": False})
    sync(camp, state)
    state["pending_candidate_id"] = candidate_id
    return row


def is_detached(state: dict, candidate_id: str | None) -> bool:
    return bool(candidate_id and candidate_id in (state.get("detached_candidates") or {}))


def require_reconciled(camp: Path, state: dict) -> None:
    from .jobs import _alive
    if state.get("live_job") or state.get("unknown_process") or state.get("pending_process_intent"):
        raise ValueError("candidate_detach_requires_process_reconciliation")
    for directory in (camp / "jobs").glob("*"):
        path = directory / "job.json"
        if (directory / "launch_intent.json").is_file() and not path.is_file():
            raise ValueError("candidate_detach_requires_process_reconciliation")
        if not path.is_file():
            continue
        record = _read(path)
        child = {"pid": record.get("child_pid"), "proc_start": record.get("child_proc_start")}
        if record.get("status") not in {"finished", "failed", "cancelled", "deadline", "invalid"} or _alive(record) or _alive(child):
            raise ValueError("candidate_detach_requires_process_reconciliation")
        if record.get("status") == "invalid":
            # Native metrics rejection is terminal scientific invalidity. It
            # may release a failed branch only after its physical job and GPU
            # cost have genuinely settled; neither fact becomes successful.
            job_id = str(record.get("job_id") or directory.name)
            cost = _read(camp / "cost.json")
            ended, started, used = record.get("ended_at"), record.get("started_at"), record.get("gpu_seconds")
            numeric = lambda value: type(value) in (int, float) and math.isfinite(value)
            if (job_id != directory.name or not isinstance(record.get("pid"), int)
                    or record["pid"] <= 0 or not record.get("proc_start")
                    or not numeric(ended) or not numeric(started) or ended < started
                    or not numeric(used) or used < 0
                    or job_id not in set(cost.get("settled_job_ids") or [])
                    or not numeric(cost.get("gpu_seconds_used")) or cost["gpu_seconds_used"] < used):
                raise ValueError("candidate_detach_requires_invalid_job_settlement")
            settlement = _read(directory / "settlement.json")
            if settlement and (settlement.get("status") != "complete" or settlement.get("job_id") != job_id):
                raise ValueError("candidate_detach_requires_invalid_job_settlement")
            from .budget import phase_reservation
            # A suite can settle through its phase receipt instead of the
            # ordinary per-job settlement file. Outstanding consumed jobs
            # retain their reservation until native phase settlement.
            if state.get("active_suite"):
                progress = _read(camp / "suites" / state["active_suite"] / "suite_state.json")
                if progress.get("phase_receipt_id"):
                    reservation = phase_reservation(camp, progress["phase_receipt_id"])
                    if job_id in set((reservation or {}).get("consumed_job_ids") or []) and job_id not in set((reservation or {}).get("settled_phase_job_ids") or []):
                        raise ValueError("candidate_detach_requires_invalid_job_settlement")
    for intent in (camp / "engineering_checks" / "multigpu").glob("*/intent.json"):
        process = _read(intent.parent / "process.json")
        receipt = _read(intent.parent / "receipt.json")
        child = {"pid": process.get("child_pid"), "proc_start": process.get("child_proc_start")}
        if (process.get("status") not in {"finished", "failed", "deadline"}
                or receipt.get("status") not in {"finished", "failed", "deadline"}
                or _alive(process) or _alive(child)):
            raise ValueError("candidate_detach_requires_probe_reconciliation")
    intents = [json.loads(line) for line in (camp / "llm_api_intents.jsonl").read_text().splitlines() if line.strip()] if (camp / "llm_api_intents.jsonl").is_file() else []
    calls = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text().splitlines() if line.strip()] if (camp / "llm_calls.jsonl").is_file() else []
    if {r["call_id"] for r in intents} - {r.get("call_id") for r in calls} or state.get("api_unsettled_call_ids"):
        raise ValueError("candidate_detach_requires_api_intent_reconciliation")
    from .api_policy import unresolved_charge_ids
    if unresolved_charge_ids(camp):
        raise ValueError("candidate_detach_requires_api_intent_reconciliation")
    failure = state.get("failure") or {}
    if failure.get("phase") == "recovery" or "identity" in str(failure.get("error_type") or ""):
        raise ValueError("candidate_detach_cannot_bypass_global_identity_failure")


def detach_failed_candidate(camp: Path, state: dict, candidate_id: str, reason: str) -> dict:
    """Terminate one quiescent failed branch, preserving its bytes and charges."""
    if not reason.strip():
        raise ValueError("candidate_detach_reason_required")
    sync(camp, state)
    if is_detached(state, candidate_id):
        return state["detached_candidates"][candidate_id]
    require_reconciled(camp, state)
    if (state.get("pending_candidate_check") or {}).get("candidate_id") == candidate_id:
        raise ValueError("candidate_detach_check_pending_is_resource_wait")
    row = next((r for r in state.get("candidates") or [] if r.get("candidate_id") == candidate_id), None)
    if candidate_id == "baseline":
        raise ValueError("baseline_cannot_be_detached")
    suite_id = state.get("active_suite")
    suite_failure = None
    invalid_failure = None
    if suite_id:
        suite = _read(camp / "suites" / suite_id / "suite_manifest.json")
        progress = _read(camp / "suites" / suite_id / "suite_state.json")
        if (suite.get("method", {}).get("candidate_id") == candidate_id
                and progress.get("scheduling_status") == "engineering_failed"
                and (progress.get("failure") or {}).get("scientific_negative") is False):
            suite_failure = progress.get("failure")
        if suite.get("method", {}).get("candidate_id") == candidate_id:
            for fold_id, job_id in (progress.get("jobs") or {}).items():
                job = _read(camp / "jobs" / job_id / "job.json")
                if (job.get("candidate_id") == candidate_id and job.get("status") == "invalid"
                        and (job.get("result") or {}).get("evaluation_valid") is False
                        and fold_id not in set(progress.get("completed_folds") or [])):
                    invalid_failure = {"kind": "invalid_training_acceptance", "job_id": job_id, "fold_id": fold_id,
                        "job_status": "invalid", "reason": (job.get("result") or {}).get("reason"), "scientific_negative": False}
                    suite_failure = suite_failure or invalid_failure
                    break
    if row and not suite_failure:
        for path in (camp / "jobs").glob("*/job.json"):
            job = _read(path)
            frozen = _read(path.parent / "frozen_run_spec.json")
            if (job.get("candidate_id") == candidate_id and job.get("status") == "invalid"
                    and (job.get("result") or {}).get("evaluation_valid") is False
                    and row.get("source_hash") and job.get("source_hash") == row["source_hash"]
                    and row.get("spec_hash") and frozen.get("spec_hash") == row["spec_hash"]):
                invalid_failure = {"kind": "invalid_training_acceptance", "job_id": path.parent.name,
                    "job_status": "invalid", "reason": (job.get("result") or {}).get("reason"), "scientific_negative": False}
                break
    if row is None or (row.get("status") not in {"implementation_failed", "review_needs_fix", "review_blocked", "abandoned", "requires_framework_extension", "failed"} and not suite_failure and not invalid_failure):
        raise ValueError("candidate_detach_requires_local_failed_candidate")
    repair = state.get("repair_task") or {}
    if repair.get("candidate_id") == candidate_id and int(repair.get("remaining") or 0) > 0:
        raise ValueError("candidate_detach_has_bounded_repair_remaining")
    from .task_ledger import _append as append_task
    tasks = {}
    ledger = camp / "task_ledger.jsonl"
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            if line.strip():
                task = json.loads(line)
                tasks[task["task_id"]] = {**tasks.get(task["task_id"], {}), **task}
    task_ids = [key for key, task in tasks.items() if task.get("candidate_id") == candidate_id
                and task.get("status") not in {"completed", "failed", "superseded", "cancelled"}]
    decision_ids = []
    from .decision_receipts import project_decisions
    project_decisions(camp, state)
    for decision in state.get("decisions") or []:
        path = camp / "decisions" / (str(decision.get("decision_id")) + ".json")
        body = _read(path)
        execution = body.get("execution") or body.get("raw") or {}
        pending_branch = state.get("candidate_id") == candidate_id or state.get("pending_candidate_id") == candidate_id
        implicit_branch_action = execution.get("target_id") is None and pending_branch and execution.get("action") in {
            "design_experiment", "propose_experiment", "implement_candidate", "repair_candidate"}
        if decision.get("executed") is False and not decision.get("consumed") and (execution.get("target_id") == candidate_id or implicit_branch_action):
            decision_ids.append(decision["decision_id"])
    receipt = _append(camp / DETACHMENTS, {"schema_version": "eeg_research.candidate_detach.v1",
        "candidate_id": candidate_id, "reason": reason, "candidate_failure": row,
        "suite_engineering_failure": suite_failure,
        "invalid_job_failure": invalid_failure,
        "active_selector_snapshot": {key: state.get(key) for key in (
            "active_issue_ids", "active_artifact_refs", "active_read_digests", "active_implementation_target_id")},
        "repair_task": repair if repair.get("candidate_id") == candidate_id else None,
        "task_ids": task_ids, "decision_ids": decision_ids, "suite_id": state.get("active_suite"),
        "budget_snapshot": {"training_jobs": state.get("training_jobs"), "llm_calls": state.get("llm_calls"),
                            "gpu_seconds_left": state.get("gpu_seconds_left")}, "created_at": time.time()})
    # Receipt is committed first; recovery can finish these projections after a crash.
    sync(camp, state)
    apply_detachments(camp, state)
    for task_id in task_ids:
        append_task(camp, {"task_id": task_id, "status": "cancelled", "detach_receipt": receipt["receipt_hash"], "at": time.time()})
    return receipt


def apply_detachments(camp: Path, state: dict) -> None:
    """Idempotently project committed receipts; old decisions never replay."""
    for target, receipt in (state.get("detached_candidates") or {}).items():
        affected = (state.get("candidate_id") == target or state.get("pending_candidate_id") == target
                    or (state.get("repair_task") or {}).get("candidate_id") == target
                    or bool(receipt.get("suite_id") and state.get("active_suite") == receipt["suite_id"]))
        from .task_ledger import latest, mark
        for task_id in receipt["task_ids"]:
            task = latest(camp, task_id) or {}
            if task.get("status") not in {"completed", "failed", "superseded", "cancelled"}:
                mark(camp, task_id, "cancelled", detach_receipt=receipt["receipt_hash"])
        if receipt.get("suite_id") and receipt.get("suite_engineering_failure"):
            from .screening import detach_phase
            detach_phase(camp, state, receipt["suite_id"], receipt["receipt_hash"])
        for row in state.get("candidates") or []:
            if row.get("candidate_id") == target:
                row["lifecycle_status"] = "abandoned"
                row["detach_receipt"] = receipt["receipt_hash"]
        for decision in state.get("decisions") or []:
            if decision.get("decision_id") in receipt["decision_ids"] and decision.get("detach_receipt") != receipt["receipt_hash"]:
                path = camp / "decisions" / (decision["decision_id"] + ".json")
                if path.is_file():
                    from .decision_receipts import append_receipt, project_decisions
                    append_receipt(camp, decision, status="detached", reason=receipt["reason"],
                                   source="candidate_lifecycle.apply_detachments",
                                   result={"candidate_id": target, "detach_receipt": receipt["receipt_hash"]},
                                   task_refs=receipt["task_ids"])
                    project_decisions(camp, state)
                else:
                    # Legacy projections can omit the decision artifact. Keep
                    # their consumed state without manufacturing raw bytes.
                    decision.update(executed=True, detached=True, consumed=True, detach_receipt=receipt["receipt_hash"])
        if (state.get("repair_task") or {}).get("candidate_id") == target:
            state["repair_task"] = None
        if state.get("candidate_id") == target or state.get("pending_candidate_id") == target:
            for key in ("candidate_id", "pending_candidate_id", "experiment", "experiment_ref", "hypothesis", "execution_spec",
                        "active_implementation_target_id", "pending_candidate_check"):
                state.pop(key, None)
            state["candidate_ready"] = False
            state["experiment_failed"] = False
        if affected:
            state.update(active_issue_ids=[], active_artifact_refs=[], active_read_digests=[], active_implementation_target_id=None)
        if state.get("active_suite") == receipt.get("suite_id") and receipt.get("suite_id"):
            state["active_suite"] = None
            state["method_jobs_reserved"] = 0
            state["gpu_seconds_reserved"] = max(0.0, float(state.get("gpu_seconds_reserved") or 0) - float(state.get("method_gpu_reservation") or 0))
            state["method_gpu_reservation"] = 0
        state["queued"] = [item for item in state.get("queued") or [] if item[0] != target]
        if affected and (state.get("failure") or {}).get("phase") in {"implement_candidate", "repair_candidate", "coder", "review"}:
            state.pop("failure", None)
            state.pop("detail", None)
        if affected and state.get("status") not in {"paused", "cancelled", "finished"}:
            state["status"] = "planning"
