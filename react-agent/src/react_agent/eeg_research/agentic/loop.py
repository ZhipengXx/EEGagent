"""Campaign loop. HTTP creates the campaign. The worker advances it one step at a time."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.budget import charge_gpu, snapshot as budget_snapshot
from react_agent.eeg_research.agentic.coder import apply_candidate_patch, finish_patch, run_candidate_check
from react_agent.eeg_research.agentic.comparison import compare_runs, matched_control
from react_agent.eeg_research.agentic.contract import public_contract
from react_agent.eeg_research.agentic.execution_protocol import load_protocol, next_unused_training_seed
from react_agent.eeg_research.agentic.llm import LlmUnavailable
from react_agent.eeg_research.agentic.memory import EpisodeStore, episode, query_lessons, retrieve
from react_agent.eeg_research.agentic.planner import STOP_REASONS, audit_completed, available_actions, completion_block_reason, decide, evidence_count
from react_agent.eeg_research.agentic.promotion import promotion_decision
from react_agent.eeg_research.agentic.research_plan import (
    PLAN_DEPENDENT_ACTIONS,
    PlanError,
    apply_update,
    consume_for_planner,
    init_plan,
    mark_consumed,
    normalize_plan_update,
)
from react_agent.eeg_research.agentic.runner import accept_job, comparable
from react_agent.eeg_research.agentic.schemas import SCHEMA_VERSION

_TERMINAL = {"paused", "finished", "blocked", "cancelled"}
DEFAULT_MAX_GPU_SECONDS = 48 * 3600


def _planner_goal(goal: dict[str, Any]) -> dict[str, Any]:
    """Keep research switches. Drop held-out test scores so the planner cannot steer on them."""
    public: dict[str, Any] = {}
    for key, value in goal.items():
        if key == "final_test_enabled":
            public[key] = value
            continue
        if key == "test_result" or key.startswith("test_result"):
            continue
        if key.startswith("final_test_"):
            continue
        public[key] = value
    from react_agent.eeg_research.agentic.handoffs import development_view

    return development_view(public)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)  # atomic create; never replace an existing decision
        finally:
            tmp_path.unlink(missing_ok=True)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def report_dependency_hash(state: dict[str, Any]) -> str:
    """Hash claims plus every settled evidence identity. New rows make a prior audit stale."""
    import hashlib

    experiment = state.get("experiment") if isinstance(state.get("experiment"), dict) else {}
    report = state.get("report_draft") if isinstance(state.get("report_draft"), dict) else {}
    actual_files = []
    for row in report.get("dependency_manifest") or []:
        if not isinstance(row, dict) or row.get("scope") != "development":
            continue
        path = Path(str(row.get("path") or ""))
        actual_files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
    deps = {
        "evidence": [
            {
                "evidence_id": row.get("evidence_id") or row.get("job_id"),
                "spec_hash": row.get("spec_hash")
                or ((row.get("experiment") or {}).get("spec_hash") if isinstance(row.get("experiment"), dict) else None),
                "source_hash": row.get("source_hash"),
                "config_hash": row.get("config_hash"),
                "metrics": {
                    "fixed_bank_top1": row.get("fixed_bank_top1"),
                    "evaluation_valid": row.get("evaluation_valid"),
                },
                "comparison": row.get("comparison"),
                "diagnostics": row.get("diagnostics") or row.get("diagnostic_ref"),
                "confirmation": row.get("promotion") or row.get("confirmation"),
                "result_hash": row.get("result_hash"),
            }
            for row in state.get("evidence") or [] if row.get("kind") != "audit"
        ],
        "report_hash": report.get("report_hash") or state.get("audit_report_hash"),
        "dependency_manifest_hash": report.get("dependency_manifest_hash"),
        "file_hashes": state.get("report_file_hashes") or report.get("file_hashes"),
        "claims": report.get("claims") or state.get("audit_claims") or [],
        "experiment_spec_hash": experiment.get("spec_hash"),
        "confirmation_policy_hash": state.get("confirmation_policy_hash"),
        "actual_dependency_files": actual_files,
    }
    return hashlib.sha256(json.dumps(deps, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def refresh_audit_freshness(state: dict[str, Any]) -> str:
    current = report_dependency_hash(state)
    audited = state.get("audited_report_hash")
    if audited != current and state.get("audit_status") in {"pass", "revise", "block", "partial", "failed", "blocked"}:
        state["audit_status"] = "stale"
        state["audit_fresh"] = False
    return current


def event(camp: Path, kind: str, **fields: Any) -> None:
    row = {"at": time.time(), "event": kind, **fields}
    with (camp / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def create_campaign(
    root: Path,
    *,
    goal: dict[str, Any],
    contract: dict[str, Any],
    request_id: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Idempotent create. The same request id does not open a second campaign."""
    index = root / "request_index.json"
    known = _read(index)
    if request_id in known:
        return _read(Path(known[request_id]) / "campaign_state.json")
    camp = root / goal["goal_id"]
    existing = _read(camp / "campaign_state.json")
    if existing:
        known[request_id] = str(camp)
        _write(index, known)
        return existing
    goal = dict(goal)
    goal.setdefault("planner_mode", "compare_options")
    if goal["planner_mode"] not in {"single_action", "compare_options"}:
        raise ValueError("planner_mode_invalid")
    state = {
        "planner_mode": goal["planner_mode"],
        "goal_id": goal["goal_id"],
        "request_id": request_id,
        "status": "created",
        "contract_fingerprint": contract["fingerprint"],
        "execution_fingerprint": None if protocol is None else protocol.get("fingerprint"),
        "evidence": [],
        "memory": [],
        "training_jobs": 0,
        "llm_calls": 0,
        "max_training_jobs": int(goal.get("max_training_jobs", 20)),
        "max_llm_calls": int(goal.get("max_llm_calls", 300)),
        "max_candidates": int(goal.get("max_candidates", 4)),
        "max_gpu_seconds": float(goal.get("max_gpu_seconds", DEFAULT_MAX_GPU_SECONDS)),
        "gpu_seconds_left": float(goal.get("max_gpu_seconds", DEFAULT_MAX_GPU_SECONDS)),
        "gpu_seconds_reserved": 0.0,
        "llm_calls_left": int(goal.get("max_llm_calls", 300)),
        "max_repairs_per_candidate": int(goal.get("max_repairs_per_candidate", 2)),
        "require_audit_before_completion": bool(goal.get("require_audit_before_completion", False)),
        "allowed_training_actions": list(goal.get("allowed_training_actions", ["run_pilot", "run_full", "replicate"])),
        "training_seeds": list(goal.get("training_seeds") or (protocol or {}).get("training_seeds") or []),
        "confirmation_policy_hash": None,
        "schema_version": SCHEMA_VERSION,
        "baseline_jobs": 0,
        "hypothesis": None,
        "experiment": None,
        "candidate_ready": False,
        "candidate_id": None,
        "candidates": [],
        "live_job": None,
        "decisions": [],
        "pause_after_step": False,
    }
    from react_agent.eeg_research.agentic.confirmation_policy import (
        ConfirmationPolicyError,
        freeze_confirmation_policy,
        write_confirmation_policy,
    )

    try:
        policy = freeze_confirmation_policy(goal, protocol)
    except ConfirmationPolicyError:
        raise
    state["training_seeds"] = list(policy.get("training_seeds") or state.get("training_seeds") or [])
    state["confirmation_policy_hash"] = policy.get("policy_hash")
    _write(camp / "goal.json", goal)
    _write(camp / "resolved_goal.json", goal)
    _write(camp / "evaluation_contract.json", contract)
    if protocol is not None:
        if policy.get("seeds_declared") and policy.get("training_seeds"):
            protocol = dict(protocol)
            protocol["training_seeds"] = list(policy["training_seeds"])
        _write(camp / "execution_protocol.json", protocol)
    write_confirmation_policy(camp, policy)
    _write(camp / "campaign_state.json", state)
    known[request_id] = str(camp)
    _write(index, known)
    event(camp, "created", fingerprint=contract["fingerprint"])
    init_plan(camp, goal, protocol=protocol)
    return state


def load_state(camp: Path) -> dict[str, Any]:
    return _read(camp / "campaign_state.json")


def _decision_number(stem: str) -> int:
    digits = stem[1:] if stem.startswith("d") else stem
    return int(digits) if digits.isdigit() else 0


def incomplete_candidate_id(camp: Path, state: dict[str, Any]) -> str | None:
    """Candidate directory that has a spec but never entered state or evidence.

    A coder log does not drop the id. Resume still has to finish that same candidate.
    """
    recorded = {row.get("candidate_id") for row in state.get("candidates") or []}
    evidenced = {row.get("candidate_id") for row in state.get("evidence") or []}
    root = camp / "candidates"
    if not root.is_dir():
        return None
    names = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not (name.startswith("c") and name[1:].isdigit()):
            continue
        if name in recorded or name in evidenced:
            continue
        if (path / "spec.json").is_file():
            names.append(name)
    if not names:
        return None
    return sorted(names, key=lambda name: int(name[1:]))[0]


def persist_failure(
    camp: Path,
    state: dict[str, Any],
    *,
    phase: str,
    error_type: str,
    detail: str,
    recoverable: bool,
) -> None:
    """Record a blocked phase. Pause and stop are left untouched."""
    if state.get("status") in {"paused", "cancelled"}:
        return
    state["status"] = "blocked"
    state["detail"] = detail
    state["failure"] = {"phase": phase, "recoverable": recoverable, "error_type": error_type}
    event(
        camp,
        "llm_unavailable" if recoverable else "worker_error",
        error_type=error_type,
        phase=phase,
        recoverable=recoverable,
    )


def _sync_ledger(camp: Path, state: dict[str, Any]) -> None:
    """Use the cost ledger when it exists. A missing file does not reset the state counter."""
    if not (camp / "cost.json").is_file():
        return
    ledger = _read(camp / "cost.json").get("llm_calls")
    if isinstance(ledger, int):
        state["llm_calls"] = ledger
        state["llm_calls_left"] = int(state.get("max_llm_calls", 100)) - ledger


def _unexecuted_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    """Only an explicit executed false is unfinished. Older decisions omit the field."""
    decisions = state.get("decisions") or []
    if not decisions:
        return None
    last = decisions[-1]
    if last.get("ok") and last.get("executed") is False:
        return last
    return None


def _decision_raw(camp: Path, decision_id: str) -> dict[str, Any]:
    payload = _read(camp / "decisions" / f"{decision_id}.json")
    raw = payload.get("raw")
    return raw if isinstance(raw, dict) else {}


def _reject_plan_decision(camp: Path, record: dict[str, Any], detail: str) -> None:
    """Keep a rejected decision in the audit trail without replaying its action."""
    raw = _decision_raw(camp, str(record["decision_id"]))
    record.update(ok=False, detail=detail, rejected_before_execution=True)
    _write(camp / "decisions" / f"{record['decision_id']}.json", {"decision": record, "raw": raw})


def _mark_executed(state: dict[str, Any]) -> None:
    if state.get("status") in {"blocked", "paused", "cancelled"}:
        return
    decisions = state.get("decisions") or []
    if decisions and decisions[-1].get("ok"):
        decisions[-1]["executed"] = True


def pending_repair(state: dict[str, Any]) -> dict[str, Any] | None:
    """Same-candidate repair with remaining attempts. This is not a new scientific candidate."""
    task = state.get("repair_task")
    if not isinstance(task, dict):
        return None
    if int(task.get("remaining") or 0) <= 0:
        return None
    if not task.get("candidate_id"):
        return None
    return task


def pending_implement(camp: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    """Last decision asked for a candidate that was never finished."""
    decisions = state.get("decisions") or []
    if not decisions:
        return None
    last = decisions[-1]
    if last.get("action") != "implement_candidate" or not last.get("ok"):
        return None
    if incomplete_candidate_id(camp, state) is None:
        return None
    return last


def _result_hash(result: dict[str, Any]) -> str:
    import hashlib

    body = {
        "job_id": result.get("job_id"),
        "candidate_id": result.get("candidate_id"),
        "seed": result.get("seed"),
        "fidelity": result.get("fidelity"),
        "fixed_bank_top1": result.get("fixed_bank_top1"),
        "evaluation_valid": result.get("evaluation_valid"),
        "checkpoint_id": result.get("checkpoint_id"),
        "source_hash": result.get("source_hash"),
        "config_hash": result.get("config_hash"),
        "spec_hash": result.get("spec_hash"),
        "contract_fingerprint": result.get("contract_fingerprint") or result.get("execution_fingerprint"),
    }
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _next_decision_id(camp: Path, state: dict[str, Any]) -> str:
    numbers = [_decision_number(str(row.get("decision_id") or "")) for row in state.get("decisions") or []]
    numbers += [_decision_number(path.stem) for path in (camp / "decisions").glob("d*.json")]
    return "d" + str(max(numbers, default=0) + 1)


def _recover_decision_reads(camp: Path, state: dict[str, Any]) -> None:
    """Coordinate verified persisted decisions and append-only read receipts."""
    from react_agent.eeg_research.agentic.artifacts import lookup, request_digest
    from react_agent.eeg_research.agentic.handoffs import read_history, read_digest, verify_read_receipt, rebuild_read_projection, _receipt_id
    from react_agent.eeg_research.agentic.schemas import PlannerDecision
    history = read_history(camp)
    known = {row.get("decision_id"): row for row in state.get("decisions") or []}
    if len(known) != len(state.get("decisions") or []) or None in known:
        raise ValueError("decision_state_identity_conflict")
    paths = sorted((camp / "decisions").glob("d*.json"), key=lambda path: _decision_number(path.stem))
    persisted_ids = {path.stem for path in paths} | set(known)
    for receipt in history:
        if receipt.get("receipt_id"):
            if receipt["receipt_id"] != _receipt_id(receipt):
                raise ValueError("artifact_read_receipt_identity_changed")
            if receipt.get("decision_id") and receipt["decision_id"] not in persisted_ids:
                raise ValueError("artifact_read_receipt_decision_missing")
    for path in paths:
        if _decision_number(path.stem) < 1:
            continue
        body = _read(path)
        record, raw = body.get("decision"), body.get("raw")
        if not isinstance(record, dict) or record.get("decision_id") != path.stem:
            raise ValueError("decision_file_identity_conflict:" + path.stem)
        previous = known.get(path.stem)
        if previous:
            keys = ("action", "ok", "read_requests", "required_artifact_refs", "required_read_digests",
                    "selected_option_id", "resolves_issue_ids", "raw_digest", "campaign_id")
            if any(previous[key] != record.get(key) for key in keys if key in previous):
                raise ValueError("decision_state_identity_conflict:" + path.stem)
        if record.get("decision_schema_version"):
            if (record.get("decision_schema_version") != "eeg_research.decision.v2"
                    or record.get("campaign_id") != state.get("goal_id")
                    or record.get("raw_digest") != request_digest(request=raw)):
                raise ValueError("decision_content_identity_conflict:" + path.stem)
        legacy_completion = None
        validation_raw = raw
        if not record.get("decision_schema_version"):
            # V1 non-read resumes already have independent local completion /
            # workspace markers. Reuse those two existing recovery paths; a
            # bare JSON decision still cannot authorize an orphan action.
            if record.get("action") == "propose_experiment":
                marker = camp / "hypotheses" / ("h" + str(_decision_number(path.stem)) + ".json")
                saved = _read(marker) if marker.is_file() and marker.resolve().is_relative_to(camp.resolve()) else {}
                if isinstance(saved.get("hypothesis"), dict) and isinstance(saved.get("experiment"), dict):
                    legacy_completion = marker
                    validation_raw = {**(raw or {}), "action": "propose_experiment"}
            elif record.get("action") == "implement_candidate" and record.get("executed") is not False:
                candidate = incomplete_candidate_id(camp, state)
                marker = camp / "candidates" / str(candidate) / "spec.json"
                if (candidate and marker.is_file() and marker.resolve().is_relative_to(camp.resolve())
                        and isinstance(_read(marker), dict) and state.get("hypothesis") and state.get("experiment")):
                    legacy_completion = marker
                    validation_raw = {"action": "implement_candidate"}
        if record.get("ok"):
            if not isinstance(validation_raw, dict):
                if previous is None and state.get("status") in _TERMINAL:
                    limit = "legacy_decision_raw_missing:" + path.stem
                    if limit not in state.setdefault("decision_recovery_limits", []):
                        state["decision_recovery_limits"].append(limit)
                    continue
                raise ValueError("decision_raw_missing:" + path.stem)
            # The existing single_action parser accepts opaque inline-coder
            # extension fields. Their full bytes remain bound by raw_digest;
            # validate the shared control schema without discarding the raw body.
            parsed = PlannerDecision.model_validate({key: value for key, value in validation_raw.items()
                                                     if key in PlannerDecision.model_fields})
            if parsed.action != record.get("action") or record.get("read_requests", []) != validation_raw.get("read_requests", []):
                raise ValueError("decision_action_identity_conflict:" + path.stem)
        if previous is None:
            row = dict(record)
            if legacy_completion is not None:
                from react_agent.eeg_research.agentic.artifacts import file_digest
                row["legacy_recovery_source"] = {"path": str(legacy_completion), "sha256": file_digest(legacy_completion)}
            if row.get("ok") and row.get("executed") is False and not (row.get("action") == "retrieve_memory" and raw.get("read_requests")) and legacy_completion is None and not record.get("decision_schema_version"):
                if state.get("status") not in _TERMINAL:
                    raise ValueError("orphan_non_read_decision_requires_reconciliation:" + path.stem)
            state.setdefault("decisions", []).append(row)
            known[path.stem] = row
            previous = row
            if row.get("executed") is True and row.get("action") == "retrieve_memory" and raw.get("read_requests"):
                state["_recovered_completed_read_id"] = path.stem
        elif record.get("executed") is True and previous.get("executed") is False:
            previous["executed"] = True
            if previous.get("action") == "retrieve_memory" and raw and raw.get("read_requests"):
                state["_recovered_completed_read_id"] = path.stem
        if record.get("action") != "retrieve_memory" or not raw or not raw.get("read_requests"):
            continue
        expected = {read_digest(request, (lookup(camp, str(request.get("artifact_id"))) or {}).get("sha256"))
                    for request in raw["read_requests"]}
        receipts = [row for row in history if row.get("decision_id") == path.stem]
        seen = set()
        for receipt in receipts:
            digest = receipt.get("request_digest")
            if digest not in expected or digest in seen or str(receipt.get("epoch")) != str(record.get("evidence_count")):
                raise ValueError("decision_read_receipt_identity_conflict:" + path.stem)
            seen.add(digest)
            # Completed historical reads may become unavailable. Revalidate all
            # pending or state-lagged reads before they can authorize recovery.
            if previous.get("executed") is False or state.get("_recovered_completed_read_id") == path.stem:
                verify_read_receipt(camp, receipt)
    state["decisions"] = sorted(state.get("decisions") or [], key=lambda row: _decision_number(str(row.get("decision_id") or "")))
    rebuild_read_projection(camp, state)
    if state.get("_recovered_completed_read_id"):
        for row in state.get("artifact_read_results") or []:
            row["replayed"] = True


def rebuild_campaign_projection(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Rebuild evidence/memory/counts from on-disk RunRecords when outer state lagged."""
    try:
        _recover_decision_reads(camp, state)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if state.get("status") in _TERMINAL:
            state["decision_recovery_error"] = str(exc)
        else:
            persist_failure(camp, state, phase="recovery", error_type="decision_read_recovery_failed",
                            detail=str(exc), recoverable=False)
        return state
    jobs_root = Path(camp) / "jobs"
    if not jobs_root.is_dir():
        return state
    for job_dir in sorted(path for path in jobs_root.iterdir() if path.is_dir()):
        job_path = job_dir / "job.json"
        metrics_path = job_dir / "metrics.json"
        run_path = job_dir / "run_record.json"
        if not job_path.is_file() and not run_path.is_file() and not metrics_path.is_file():
            continue
        record = _read(job_path) if job_path.is_file() else {}
        if record.get("status") == "running":
            continue
        run_record = _read(run_path) if run_path.is_file() else {}
        metrics = _read(metrics_path) if metrics_path.is_file() else {}
        result = run_record.get("result") if isinstance(run_record.get("result"), dict) else {}
        if not result:
            result = dict(metrics)
        if not result and not record:
            continue
        job_id = str(record.get("job_id") or run_record.get("job_id") or job_dir.name)
        _record_job(
            camp,
            state,
            {
                "job_id": job_id,
                "candidate_id": record.get("candidate_id") or result.get("candidate_id") or run_record.get("candidate_id"),
                "seed": record.get("seed") if record.get("seed") is not None else result.get("seed"),
                "status": record.get("status") or run_record.get("status") or "finished",
                "gpu_seconds": record.get("gpu_seconds") or 0,
                "source_hash": record.get("source_hash") or (record.get("manifest") or {}).get("entry_sha256"),
                "result": result,
            },
        )
    return state


def align_interrupt(camp: Path) -> dict[str, Any]:
    """Use the same verified recovery entry point without replaying an action."""
    state = load_state(camp)
    previous_status = state.get("status")
    rebuild_campaign_projection(camp, state)
    if previous_status in _TERMINAL:
        state["status"] = previous_status
    save_state(camp, state)
    return load_state(camp)


def request_control(camp: Path, action: str) -> None:
    """User controls live in their own file so a running worker cannot overwrite them."""
    _write(camp / "control.json", {"action": action, "at": time.time()})


def _apply_control(camp: Path, state: dict[str, Any]) -> None:
    control = _read(camp / "control.json")
    action = control.get("action")
    if action == "pause":
        state["pause_after_step"] = True
    elif action == "stop":
        state["status"] = "cancelled"
        state["stop_reason"] = "user_cancelled"
    elif action == "resume":
        state["pause_after_step"] = False


def save_state(camp: Path, state: dict[str, Any]) -> None:
    _apply_control(camp, state)
    from react_agent.eeg_research.agentic.audit_context import audit_feedback
    state["report_support_status"] = audit_feedback(camp, state)["report_support_status"]
    if state.get("pause_after_step") and state.get("status") not in _TERMINAL and not state.get("live_job"):
        state["status"] = "paused"
    _write(camp / "campaign_state.json", state)


def _dev_row(row: dict[str, Any]) -> dict[str, Any]:
    """Allowlist view of one evidence row."""
    keep = (
        "evidence_id",
        "candidate_id",
        "fidelity",
        "evaluation_valid",
        "reason",
        "fixed_bank_top1",
        "fixed_bank_top5",
        "gallery_size",
        "delta_vs_control_pp",
        "control_id",
        "curve",
        "summary",
        "kind",
        "seed",
        "training_seed",
        "failures",
        "diagnostic_ref",
        "diagnostics",
        "comparison",
        "comparison_status",
        "job_status",
        "promotion",
        "local_only",
        "status",
    )
    return {key: row.get(key) for key in keep if key in row}


def candidate_implementation_target(camp: Path, state: dict[str, Any]) -> str:
    """Reuse the worker's actual candidate allocation for planner target binding."""
    repair = pending_repair(state)
    if repair:
        return str(repair["candidate_id"])
    unfinished = incomplete_candidate_id(camp, state)
    if unfinished:
        return unfinished
    index = len(state.get("candidates") or []) + 1
    while (camp / "candidates" / f"c{index}").exists():
        index += 1
    return f"c{index}"


def observation(camp: Path) -> dict[str, Any]:
    state = load_state(camp)
    refresh_audit_freshness(state)
    contract = public_contract(_read(camp / "evaluation_contract.json"))
    from react_agent.eeg_research.agentic.planner import blocked_actions, eligible_targets

    evidence = [_dev_row(row) for row in state.get("evidence") or []]
    from react_agent.eeg_research.agentic.interface import candidate_interface

    protocol = load_protocol(camp)
    trainable = ["baseline"]
    for row in state.get("candidates") or []:
        if row.get("status") == "ready" and row.get("candidate_id") and row["candidate_id"] not in trainable:
            trainable.append(row["candidate_id"])
    if state.get("candidate_ready") and state.get("candidate_id") and state["candidate_id"] not in trainable:
        trainable.append(state["candidate_id"])
    latest_job = next((row for row in reversed(state.get("evidence") or []) if row.get("job_dir") or row.get("fidelity")), None)
    from react_agent.eeg_research.agentic.capabilities import capability_manifest
    from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy
    from react_agent.eeg_research.agentic.run_context import load_approved_binding

    goal = _read(camp / "goal.json")
    policy = load_confirmation_policy(camp, goal)
    experiment = state.get("experiment") if isinstance(state.get("experiment"), dict) else {}
    parent_id = experiment.get("parent_candidate_id") or "baseline"
    control_id = experiment.get("control_candidate_id") or "baseline"
    from react_agent.eeg_research.agentic.handoffs import candidate_context, development_view, load_analysis_views

    from react_agent.eeg_research.agentic.audit_context import audit_feedback
    from react_agent.eeg_research.agentic.handoffs import (
        artifact_index, bounded_history, bounded_control_view, verified_read_context, read_history, read_epoch, READS_PER_EVIDENCE_EPOCH,
    )
    feedback = audit_feedback(camp, state)
    analysis_views = load_analysis_views(camp, state)
    index = artifact_index(camp, state)
    reads = read_history(camp)
    remaining_reads = max(0, READS_PER_EVIDENCE_EPOCH - sum(row.get("epoch") == read_epoch(state) for row in reads))
    state["_has_development_artifacts"] = any(row["verification_status"] == "verified" for row in index)
    state["_read_remaining"] = remaining_reads
    state["_has_open_report_issues"] = bool((state.get("report_draft") or {}).get("claims") and any(
        row["status"] == "open" and row.get("claim_id") for row in feedback["issues"]))
    state["_implementation_target"] = candidate_implementation_target(camp, state)
    actions = available_actions(state)
    public = development_view({
        "goal": _planner_goal(goal),
        "planner_mode": goal.get("planner_mode", "single_action"),
        "audit_feedback": feedback,
        "report_support_status": feedback["report_support_status"],
        "report_draft": {key: (state.get("report_draft") or {}).get(key) for key in ("report_hash", "report_ref", "dependency_manifest_hash", "claims")},
        "artifact_index": index,
        "artifact_reads": verified_read_context(camp, state),
        "artifact_read_budget": {"remaining_reads": remaining_reads, "epoch": read_epoch(state)},
        "_artifact_read_digests": [row.get("request_digest") for row in reads],
        "artifact_read_index": [{key: row.get(key) for key in ("artifact_id", "request_digest", "receipt_id", "start", "end", "status")}
                                for row in reads],

        "require_audit_before_completion": bool(goal.get("require_audit_before_completion", False)),
        "audit_status": state.get("audit_status") or "pending",
        "audit_fresh": state.get("audit_fresh") is True,
        "completion_audit_ready": audit_completed(state),
        "audited_report_hash": state.get("audited_report_hash"),
        "confirmation_policy": policy,
        "confirmation_policy_hash": (policy or {}).get("policy_hash") or state.get("confirmation_policy_hash"),
        "parent_source": candidate_context(camp, str(parent_id)),
        "control_source": candidate_context(camp, str(control_id)),
        "analyses": analysis_views,
        "method_evidence_packet": state.get("method_evidence_packet"),
        "contract": contract,
        "candidate_interface": candidate_interface(protocol),
        "evidence": evidence,
        "candidates": state.get("candidates") or [],
        "hypothesis": state.get("hypothesis"),
        "experiment": state.get("experiment"),
        "candidate_ready": state.get("candidate_ready"),
        "candidate_id": state.get("candidate_id"),
        "available_actions": actions,
        "eligible_targets": eligible_targets(state),
        "implementation_target_id": state["_implementation_target"],
        "blocked_actions": blocked_actions(state),
        "trainable_ids": trainable,
        "budget": budget_snapshot(camp, state),
        "capabilities": capability_manifest(),
        "research_plan": consume_for_planner(camp, state),
        "memory": retrieve(
            state.get("memory") or [],
            task_hash=str(state.get("goal_id")),
            fingerprint=str(state.get("contract_fingerprint")),
        ),
        "lessons": query_lessons(
            {
                "task": state.get("goal_id"),
                "evaluation_identity": state.get("contract_fingerprint"),
            },
            EpisodeStore(camp).list_lessons(),
        ),
        "recent_decisions": (state.get("decisions") or [])[-4:],
        "last_local_result": state.get("last_local_result"),
        "latest_comparison": None if latest_job is None else latest_job.get("comparison"),
        "latest_diagnostics": None if latest_job is None else latest_job.get("diagnostics"),
        "_known_evidence_ids": [row.get("evidence_id") for row in evidence],
    })
    limits = {}
    for key in ("evidence", "memory", "lessons", "analyses"):
        public[key], limits[key] = bounded_history(public[key])
    public["evidence_index"] = [{key: row.get(key) for key in ("evidence_id", "candidate_id", "kind", "fidelity", "evaluation_valid")}
                                for row in evidence]
    public["memory_index"] = [{key: row.get(key) for key in ("episode_id", "candidate_id", "artifact", "retrieval")}
                              for row in retrieve(state.get("memory") or [], task_hash=str(state.get("goal_id")), fingerprint=str(state.get("contract_fingerprint")))]
    plan = (public.get("research_plan") or {}).get("plan") or {}
    hypothesis_evidence = []
    for view in analysis_views:
        payload = view.get("payload")
        if view.get("verification_status") != "verified" or view.get("completion_status") != "completed" or not isinstance(payload, dict):
            continue
        assessment = {"artifact_id": view.get("artifact_id"), "content_hash": view.get("content_hash"),
                      "candidate_id": view.get("candidate_id"), "evidence_id": view.get("evidence_id"),
                      "authority": "llm_interpretation", "hypothesis_assessment": payload.get("hypothesis_assessment"),
                      "evidence_refs": payload.get("evidence_refs") or [], "context_limits": {}}
        for key in ("prediction_checks", "competing_explanations", "evidence_gaps", "suggested_next_actions"):
            assessment[key], assessment["context_limits"][key] = bounded_history(payload.get(key) or [], budget=4000)
        hypothesis_evidence.append(assessment)
    public["controller_state"] = {
        "research_questions": plan.get("research_questions") or [],
        "hypothesis": development_view(state.get("hypothesis")),
        "hypothesis_evidence": hypothesis_evidence,
        "candidate_states": [{key: row.get(key) for key in ("candidate_id", "status", "parent_id", "attempt_id")}
                             for row in state.get("candidates") or []],
        "next_comparisons": plan.get("pending_comparisons") or [],
        "open_audit_issue_ids": feedback["unresolved_issue_ids"],
        "remaining_resources": public["budget"],
        "reserved_confirmation": {"plan": plan.get("reserved_confirmation"), "resource_status": "plan_information_only"},
        "interpretation_authority": "analyst_interpretation_is_not_runtime_confirmation",
    }
    for key in ("audit_feedback", "hypothesis", "experiment", "latest_diagnostics", "latest_comparison",
                "method_evidence_packet", "recent_decisions", "last_local_result", "research_plan", "report_draft", "controller_state"):
        public[key], limits[key] = bounded_control_view(public[key])
    public["context_limits"] = {"history": limits, "omitted_content": "use registered artifact_index IDs and read_requests", "control_identities_retained": True}
    from react_agent.eeg_research.agentic.planner import cost_estimates
    public["action_cost_estimates"] = cost_estimates(state, public, camp=camp)
    return public


Services = dict[str, Callable[..., Any]]


def tick(camp: Path, backend: Any, runner: Any | None = None, services: Services | None = None) -> dict[str, Any]:
    """One research step. A live job is reconciled first and never double-started."""
    services = services or {}
    state = load_state(camp)
    rebuild_campaign_projection(camp, state)
    if state.get("status") in _TERMINAL:
        if (state.get("failure") or {}).get("phase") == "recovery":
            save_state(camp, state)
        return state
    if state.pop("_recovered_completed_read_id", None):
        _finish_step(camp, state)
        return state
    if state.get("live_job"):
        settle = services.get("settle")
        if settle is None:
            state["status"] = "training"
            save_state(camp, state)
            return state
        record = settle(camp, state["live_job"])
        if record.get("status") == "running":
            state["status"] = "training"
            state["heartbeat"] = record.get("heartbeat")
            save_state(camp, state)
            return state
        _record_job(camp, state, record)
        state["live_job"] = None
        state["status"] = "analyzing"
        save_state(camp, state)
        analyst = services.get("analyze")
        if analyst is not None:
            analyst(camp, state)
            save_state(camp, state)
        if state.get("pause_after_step"):
            state["status"] = "paused"
            save_state(camp, state)
        return state
    _sync_ledger(camp, state)
    queued = state.get("queued") or []
    if queued and services.get("launch") is not None:
        item = queued.pop(0)
        candidate_id, fidelity = item[0], item[1]
        training_seed = item[2] if len(item) > 2 else None
        state["queued"] = queued
        _launch(camp, state, services, candidate_id, fidelity, training_seed=training_seed)
        _finish_step(camp, state)
        return state
    pending = _unexecuted_decision(state)
    if pending is not None and (state.get("failure") or {}).get("phase") == "plan_update":
        _reject_plan_decision(camp, pending, "plan_update_blocks_action")
        save_state(camp, state)
        pending = None
    if pending is not None and state.get("status") not in {"paused", "cancelled"}:
        _apply_action(camp, state, str(pending.get("action") or ""), pending, runner, services, observation(camp))
        _finish_step(camp, state)
        return state
    if pending_implement(camp, state) is not None and services.get("implement") is not None:
        if state.get("status") not in {"paused", "cancelled"}:
            state["status"] = "planning"
            services["implement"](camp, state)
            _finish_step(camp, state)
            return state
    if state.get("llm_calls_left", 1) <= 0:
        state["status"] = "blocked"
        state["detail"] = "budget_exhausted"
        save_state(camp, state)
        return state
    obs = observation(camp)
    try:
        from react_agent.eeg_research.agentic.handoffs import record_read_delivery, prepare_read_delivery, _read_journal_records
        from react_agent.eeg_research.agentic.artifacts import request_digest
        reserved_id = _next_decision_id(camp, state)
        def planner_request(request):
            rows = request.get("artifact_reads") or []
            prepare_read_delivery(rows)
            prefix = reserved_id + ":planner_call:"
            attempt = 1 + sum(row.get("consumer") == "research_planner" and
                              str(row.get("consumer_request_id") or "").startswith(prefix)
                              for row in _read_journal_records(camp))
            record_read_delivery(camp, rows, consumer="research_planner", request_id=prefix + str(attempt),
                                 input_digest=request_digest(request=request))
            return backend(request)
        decision = decide(obs, planner_request)
    except LlmUnavailable as exc:
        _sync_ledger(camp, state)
        persist_failure(camp, state, phase="planner", error_type=str(exc), detail=str(exc), recoverable=True)
        save_state(camp, state)
        return state
    except Exception as exc:
        persist_failure(
            camp,
            state,
            phase="planner",
            error_type=type(exc).__name__,
            detail=f"{type(exc).__name__}: {exc}",
            recoverable=False,
        )
        save_state(camp, state)
        return state
    _sync_ledger(camp, state)
    raw_decision = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
    cited = raw_decision.get("evidence_ids") or raw_decision.get("evidence_refs") or []
    record = {
        "decision_id": _next_decision_id(camp, state),
        "decision_schema_version": "eeg_research.decision.v2",
        "campaign_id": state["goal_id"],
        "raw_digest": request_digest(request=decision.get("raw")),
        "action": decision.get("action"),
        "ok": decision.get("ok"),
        "reason_zh": decision.get("reason_zh"),
        "expected_information": raw_decision.get("expected_information"),
        "required_artifact_refs": raw_decision.get("required_artifact_refs") or [],
        "required_read_digests": raw_decision.get("required_read_digests") or [],
        "options": raw_decision.get("options") or [],
        "selected_option_id": raw_decision.get("selected_option_id"),
        "selection_rationale": raw_decision.get("selection_rationale"),
        "resolves_issue_ids": raw_decision.get("resolves_issue_ids") or [],
        "read_requests": raw_decision.get("read_requests") or [],
        "budget_snapshot": obs.get("budget"),
        "question_id": raw_decision.get("question_id"),
        "evidence_ids": [item for item in cited if isinstance(item, str)] if isinstance(cited, list) else [],
        "detail": decision.get("detail"),
        "evidence_count": evidence_count(state),
        "executed": bool(decision.get("ok")) is False,
    }
    state.setdefault("decisions", []).append(record)
    _write(camp / "decisions" / f"{record['decision_id']}.json", {"decision": record, "raw": decision.get("raw")}, exclusive=True)
    event(camp, "decision", **record)
    if not decision.get("ok"):
        state["status"] = "blocked"
        state["detail"] = decision.get("detail")
        state["failure"] = {
            "phase": "plan_update" if decision.get("plan_update_error") else "planner",
            "recoverable": True,
            "error_type": "plan_update_rejected" if decision.get("plan_update_error") else str(decision.get("detail") or "schema"),
        }
        if decision.get("plan_update_error"):
            event(camp, "plan_update_rejected", detail=decision["plan_update_error"], decision_id=record["decision_id"])
        save_state(camp, state)
        return state
    raw_update = raw_decision.get("plan_update")
    update_failed = False
    if isinstance(raw_update, dict):
        known = {str(row.get("evidence_id")) for row in state.get("evidence") or [] if row.get("evidence_id")}
        try:
            apply_update(
                camp,
                normalize_plan_update(raw_update, raw_decision, known_evidence_ids=known),
                known_evidence_ids=known,
            )
        except PlanError as exc:
            update_failed = True
            event(camp, "plan_update_rejected", detail=str(exc), decision_id=record["decision_id"])
        except Exception as exc:
            update_failed = True
            event(
                camp,
                "plan_update_rejected",
                detail=f"{type(exc).__name__}: {exc}",
                decision_id=record["decision_id"],
            )
    depends = bool(raw_decision.get("action_depends_on_plan_update"))
    if update_failed and depends and str(decision.get("action") or "") in PLAN_DEPENDENT_ACTIONS:
        _reject_plan_decision(camp, record, "plan_update_blocks_action")
        persist_failure(
            camp,
            state,
            phase="plan_update",
            error_type="plan_update_rejected",
            detail="plan_update_blocks_action",
            recoverable=True,
        )
        _finish_step(camp, state)
        return state
    mark_consumed(camp, record["decision_id"])
    if (state.get("failure") or {}).get("phase") in {"planner", "plan_update"}:
        state.pop("failure", None)
        state.pop("detail", None)
    _apply_action(camp, state, str(decision.get("action") or ""), decision, runner, services, obs)
    _finish_step(camp, state)
    return state


def _finish_step(camp: Path, state: dict[str, Any]) -> None:
    _sync_ledger(camp, state)
    _mark_executed(state)
    if state.get("decisions"):
        record = state["decisions"][-1]
        path = camp / "decisions" / f"{record['decision_id']}.json"
        if path.is_file() and record.get("executed") is True:
            body = _read(path)
            from react_agent.eeg_research.agentic.artifacts import request_digest
            if ((body.get("decision") or {}).get("decision_id") != record["decision_id"]
                    or (record.get("raw_digest") and request_digest(request=body.get("raw")) != record["raw_digest"])):
                raise ValueError("decision_content_identity_conflict:" + record["decision_id"])
            body["decision"] = dict(record)
            _write(path, body)
    if state.get("pause_after_step") and state.get("status") not in _TERMINAL and not state.get("live_job"):
        state["status"] = "paused"
    save_state(camp, state)


def _apply_action(
    camp: Path,
    state: dict[str, Any],
    action: str,
    decision: dict[str, Any],
    runner: Any,
    services: Services,
    obs: dict[str, Any],
) -> None:
    """Run one already chosen action. Does not allocate a new decision id."""
    state["status"] = "planning"
    raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else _decision_raw(camp, str(decision.get("decision_id") or ""))
    if obs.get("planner_mode") == "compare_options" or any(raw.get(key) for key in ("options", "read_requests", "resolves_issue_ids", "report_revision")) or action == "revise_report":
        from react_agent.eeg_research.agentic.planner import _parse, validate_comparison_and_reads
        live = observation(camp)
        # Reconcile paid calls and persisted resource/target changes before
        # execution. A smaller remaining budget is never overwritten by a stale
        # in-memory planning snapshot.
        current = load_state(camp)
        if current.get("status") in _TERMINAL:
            state["status"] = current["status"]
            state["detail"] = "execution_stopped_by_current_campaign_state"
            return
        for key in ("max_llm_calls", "max_training_jobs", "max_gpu_seconds", "gpu_seconds_left"):
            if current.get(key) is not None and state.get(key) is not None:
                state[key] = min(state[key], current[key])
        state["training_jobs"] = max(int(state.get("training_jobs") or 0), int(current.get("training_jobs") or 0))
        _sync_ledger(camp, state)
        live["budget"] = budget_snapshot(camp, state)
        scope = {**current, **{key: state.get(key) for key in ("max_llm_calls", "llm_calls", "llm_calls_left",
                    "max_training_jobs", "training_jobs", "max_gpu_seconds", "gpu_seconds_left")},
                 "_implementation_target": live.get("implementation_target_id"),
                 "_has_development_artifacts": any(row.get("verification_status") == "verified" for row in live.get("artifact_index") or []),
                 "_read_remaining": live["artifact_read_budget"]["remaining_reads"],
                 "_has_open_report_issues": "revise_report" in live["available_actions"]}
        live["available_actions"] = available_actions(scope)
        from react_agent.eeg_research.agentic.planner import eligible_targets
        live["eligible_targets"] = eligible_targets(scope)
        candidate = {**raw, "action": action}
        parsed = _parse(candidate, live["available_actions"], set(live.get("_known_evidence_ids") or []),
                        set(live.get("trainable_ids") or []), live.get("eligible_targets"))
        if action == "retrieve_memory" and raw.get("read_requests"):
            from react_agent.eeg_research.agentic.handoffs import read_history, read_digest, verify_read_receipt, READS_PER_EVIDENCE_EPOCH
            current_id = decision.get("decision_id") or (state.get("decisions") or [{}])[-1].get("decision_id")
            same_decision = [row for row in read_history(camp) if row.get("decision_id") == current_id]
            by_id = {row["artifact_id"]: row for row in live.get("artifact_index") or []}
            requested = {read_digest(row, (by_id.get(row.get("artifact_id")) or {}).get("content_hash")) for row in raw["read_requests"]}
            try:
                stored_digests = {verify_read_receipt(camp, row)["request_digest"] for row in same_decision}
                if not stored_digests <= requested:
                    raise ValueError("decision_read_receipt_identity_conflict")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                persist_failure(camp, state, phase="recovery", error_type="decision_read_recovery_failed", detail=str(exc), recoverable=False)
                return
            if stored_digests:
                live["_artifact_read_digests"] = [digest for digest in live["_artifact_read_digests"] if digest not in stored_digests]
                epoch = str((state.get("decisions") or [{}])[-1].get("evidence_count"))
                used = sum(str(row.get("epoch")) == epoch for row in read_history(camp))
                live["artifact_read_budget"]["remaining_reads"] = max(0, READS_PER_EVIDENCE_EPOCH - used) + len(stored_digests)
                if "retrieve_memory" not in live["available_actions"]:
                    live["available_actions"].append("retrieve_memory")
                parsed = _parse(candidate, live["available_actions"], set(live.get("_known_evidence_ids") or []),
                                set(live.get("trainable_ids") or []), live.get("eligible_targets"))
        detail = None if parsed.get("ok") else parsed.get("detail")
        if detail is None:
            detail = validate_comparison_and_reads(candidate, live)
        if detail:
            persist_failure(camp, state, phase="action", error_type="execution_revalidation_failed", detail=detail, recoverable=True)
            return
        if state.get("decisions"):
            state["decisions"][-1]["execution_budget_snapshot"] = live["budget"]
    state["active_issue_ids"] = list(raw.get("resolves_issue_ids") or [])
    state["active_artifact_refs"] = list(raw.get("required_artifact_refs") or [])
    state["active_read_digests"] = list(raw.get("required_read_digests") or [])
    state["active_implementation_target_id"] = raw.get("target_id") if action in {"implement_candidate", "repair_candidate"} else None
    if action == "stop":
        reason = raw.get("stop_reason") or decision.get("stop_reason")
        if reason in {None, ""}:
            reason = "blocked"
        if reason not in STOP_REASONS:
            persist_failure(
                camp,
                state,
                phase="stop",
                error_type="stop_reason_invalid",
                detail=f"stop_reason_invalid:{reason}",
                recoverable=True,
            )
            return
        refresh_audit_freshness(state)
        goal = _read(camp / "goal.json")
        gate = {**state, "require_audit_before_completion": bool(goal.get("require_audit_before_completion", False))}
        if completion_block_reason(gate, reason):
            persist_failure(camp, state, phase="stop", error_type="audit_required_before_goal_addressed",
                            detail="audit_required_before_goal_addressed", recoverable=True)
            return
        state["status"] = "finished"
        state["execution_status"] = "completed"
        state["research_outcome"] = state.get("research_outcome") or "not_evaluated"
        refresh_audit_freshness(state)
        state["audit_status"] = state.get("audit_status") or "pending"
        state["termination_reason"] = reason
        state["stop_reason"] = reason
    elif action == "inspect_data":
        state["data_audit"] = _inspect(camp)
        _append_derived(state, {"evidence_id": f"ev_audit_{len(state['evidence']) + 1}", "kind": "data_audit", "summary": state["data_audit"]})
    elif action == "retrieve_memory":
        state["memory_hits"] = [row.get("candidate_id") for row in obs.get("memory") or []]
        if raw.get("read_requests"):
            from react_agent.eeg_research.agentic.handoffs import consume_artifact_reads
            state["last_local_result"] = consume_artifact_reads(camp, state, raw["read_requests"],
                decision.get("decision_id") or (state.get("decisions") or [{}])[-1].get("decision_id"))
    elif action == "revise_report":
        try:
            _revise_report(camp, state, raw)
        except (OSError, ValueError, KeyError) as exc:
            persist_failure(camp, state, phase="report_revision", error_type=type(exc).__name__,
                            detail=str(exc), recoverable=True)
    elif action == "retrieve_methods":
        _retrieve_methods(camp, state, raw, services)
    elif action == "diagnose_results":
        _append_derived(state, _diagnose(camp, state))
    elif action == "collect_diagnostics":
        _collect_diagnostics(camp, state)
    elif action == "design_experiment":
        _design_experiment(camp, state, decision, raw, services)
    elif action == "propose_experiment":
        _propose_experiment(camp, state, decision, raw)
    elif action == "repair_candidate":
        _repair_candidate(camp, state, services)
    elif action == "audit_result":
        refresh_audit_freshness(state)
        if audit_completed(state):
            state["last_local_result"] = {"kind": "audit", "new_information": False,
                                          "reason": "fresh_audit_reused"}
            event(camp, "audit_reused", audited_dependency_hash=state["audited_report_hash"])
        else:
            _audit_result(camp, state, services)
    elif action == "implement_candidate":
        from react_agent.eeg_research.agentic.experiment_gate import ExperimentResolutionError, resolve_approved_experiment

        repairing = bool(state.get("repair_task") and int((state.get("repair_task") or {}).get("remaining") or 0) > 0)
        spec = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
        repair = state.get("repair_task") if isinstance(state.get("repair_task"), dict) else {}
        target_id = repair.get("candidate_id") if repairing else state.get("candidate_id")
        try:
            resolved = resolve_approved_experiment(
                camp,
                spec_ref=state.get("experiment_ref"),
                expected_hash=spec.get("spec_hash") if isinstance(spec, dict) and not repairing else None,
                target_id=target_id,
                attempt_id=repair.get("attempt_id"),
                state=state,
                action="repair" if repairing else "implement",
            )
            if repairing:
                state["execution_spec"] = resolved
            else:
                state["experiment"] = resolved
        except ExperimentResolutionError as exc:
            persist_failure(
                camp,
                state,
                phase="implement_candidate",
                error_type=exc.reason,
                detail=exc.reason,
                recoverable=True,
            )
            if exc.reason in {
                "experiment_not_approved",
                "approved_spec_missing",
                "approved_spec_changed",
                "missing_binding",
            }:
                state["experiment_failed"] = True
            return
        implementer = services.get("implement")
        if implementer is None:
            _implement_inline(camp, state, decision if decision.get("raw") else {"raw": raw, "reason_zh": decision.get("reason_zh")})
        else:
            implementer(camp, state)
    elif action in {"run_pilot", "run_full", "replicate"}:
        _train(camp, state, action, runner, services, raw)


def _propose_experiment(camp: Path, state: dict[str, Any], decision: dict[str, Any], raw: dict[str, Any]) -> None:
    path = camp / "hypotheses" / f"h{len(state.get('decisions') or [])}.json"
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            persist_failure(
                camp,
                state,
                phase="propose_experiment",
                error_type="hypothesis_unreadable",
                detail="hypothesis_unreadable",
                recoverable=False,
            )
            return
        state["hypothesis"] = saved.get("hypothesis")
        state["experiment"] = saved.get("experiment") or {}
    else:
        from react_agent.eeg_research.agentic.experiment_gate import approve_experiment

        state["hypothesis"] = raw.get("hypothesis_draft") or {"mechanism": decision.get("reason_zh")}
        draft = dict(raw.get("experiment_draft") or {"initial_fidelity": "pilot"})
        draft["parent_candidate_id"] = draft.get("parent_candidate_id") or "baseline"
        draft["hypothesis"] = state["hypothesis"]
        state["experiment"] = approve_experiment(draft, camp=camp)
        state["experiment_ref"] = state["experiment"].get("experiment_ref")
        _write(path, {"hypothesis": state["hypothesis"], "experiment": state["experiment"]})
    from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved

    state["candidate_ready"] = False
    state["experiment_failed"] = not experiment_is_approved(state.get("experiment") if isinstance(state.get("experiment"), dict) else None)


def _retrieve_methods(camp: Path, state: dict[str, Any], raw: dict[str, Any], services: Services) -> None:
    from react_agent.eeg_research.agentic.knowledge import retrieve_methods
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task
    from react_agent.eeg_research.agentic.schemas import ROLE_RESULT_VERSION

    query = str(raw.get("query") or (state.get("hypothesis") or {}).get("mechanism") or "")
    packed = retrieve_methods(query)
    packed["local_only"] = True
    packed["online"] = False
    inputs = [camp / "goal.json", camp / "evaluation_contract.json"]
    task = begin_role_task(camp, role="research_librarian", inputs=inputs, request={"query": query, "retrieval": packed})
    payload = {
        "schema_version": ROLE_RESULT_VERSION,
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "input_digest": task["input_digest"],
        "status": "completed",
        "search_scope": packed["search_scope"],
        "local_only": True,
        "method_cards": packed["hits"],
        "sources": packed["sources"],
        "summary_zh": f"本地方法卡 {len(packed['hits'])} 条，未联网检索",
    }
    librarian = services.get("librarian")
    if librarian is not None:
        try:
            reply = librarian({**payload, "task_id": task["task_id"], "input_digest": task["input_digest"], "online_retrieval": False})
            if isinstance(reply, dict):
                reply_status = str(reply.get("status") or "completed")
                if reply_status in {"partial", "failed", "blocked"}:
                    payload["status"] = reply_status
                payload["summary_zh"] = str(reply.get("summary_zh") or payload["summary_zh"])
                payload["payload"] = reply
                body = reply.get("payload") if isinstance(reply.get("payload"), dict) else reply
                state["method_evidence_packet"] = body
            payload["local_only"] = True
        except LlmUnavailable:
            payload["status"] = "partial"
            payload["summary_zh"] = "文献角色不可用，仅返回本地方法卡"
    path = camp / "knowledge" / "method_hits" / f"{task['task_id']}.json"
    envelope = finish_role_task(camp, task, payload, kind="method_hits", path=path)
    pointer = {
        "latest_artifact_id": (envelope.get("artifact_refs") or [None])[-1],
        "path": str(path),
        "immutable": False,
    }
    _write(camp / "knowledge" / "method_hits_latest.json", pointer)
    state["method_hits"] = packed["hits"]
    _append_derived(
        state,
        {
            "evidence_id": f"ev_methods_{len(state['evidence']) + 1}",
            "kind": "method_hits",
            "local_only": True,
            "summary": {"n": len(packed["hits"]), "local_only": True, "query": query},
        },
    )


def _design_context(camp: Path, state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Verified inputs the designer actually receives. Missing pieces stay listed."""
    from react_agent.eeg_research.agentic.budget import snapshot as budget_snapshot
    from react_agent.eeg_research.agentic.capabilities import capability_manifest
    from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy

    missing: list[str] = []
    from react_agent.eeg_research.agentic.handoffs import candidate_context, development_view, load_analysis_views, matched_diagnostics

    parent = str(spec.get("parent_candidate_id") or "baseline")
    control = str(spec.get("control_candidate_id") or "baseline")
    diagnostics = matched_diagnostics(state, parent)
    if diagnostics is None:
        missing.append("diagnostics")
    methods = state.get("method_hits") or []
    if not methods:
        missing.append("method_evidence")
    contract_path = camp / "evaluation_contract.json"
    contract = _read(contract_path) if contract_path.is_file() else {}
    if not contract:
        missing.append("evaluation_contract")
    from react_agent.eeg_research.agentic.audit_context import audit_feedback
    from react_agent.eeg_research.agentic.handoffs import selected_read_context
    return development_view({
        "artifact_reads": selected_read_context(camp, state),
        "audit_issues": [row for row in audit_feedback(camp, state)["issues"] if row["issue_id"] in (state.get("active_issue_ids") or [])],
        "hypothesis": spec.get("hypothesis") or state.get("hypothesis"),
        "goal": _planner_goal(_read(camp / "goal.json")),
        "confirmation_policy": load_confirmation_policy(camp),
        "diagnostics": diagnostics,
        "method_evidence": methods,
        "method_evidence_packet": state.get("method_evidence_packet"),
        "parent_candidate_id": spec.get("parent_candidate_id"),
        "control_candidate_id": spec.get("control_candidate_id") or "baseline",
        "parent_source": candidate_context(camp, parent),
        "control_source": candidate_context(camp, control),
        "analyses": load_analysis_views(camp, state, target=parent),
        "cold_start_prior": diagnostics is None,
        "capability_status_semantics": "available=wired and usable; verified=false=no execution receipt yet, not missing",
        "evaluation_contract": {
            "fingerprint": contract.get("fingerprint"),
            "research_scope": contract.get("research_scope"),
            "primary_metric": contract.get("primary_metric"),
        },
        "capabilities": capability_manifest(load_protocol(camp)),
        "budget": budget_snapshot(camp, state),
        "missing_inputs": missing,
    })


def _design_experiment(camp: Path, state: dict[str, Any], decision: dict[str, Any], raw: dict[str, Any], services: Services) -> None:
    from react_agent.eeg_research.agentic.experiment_gate import approve_experiment, experiment_is_approved
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task

    spec = raw.get("experiment_draft") or state.get("experiment") or {"initial_fidelity": "pilot"}
    if not isinstance(spec, dict):
        spec = {"initial_fidelity": "pilot"}
    spec = dict(spec)
    for key in ("spec_hash", "approval_record", "allowed_actions", "blocked_reason", "validator_version", "role_status", "experiment_ref"):
        spec.pop(key, None)
    spec["status"] = "draft"
    spec["parent_candidate_id"] = spec.get("parent_candidate_id") or "baseline"
    spec["hypothesis"] = raw.get("hypothesis_draft") or spec.get("hypothesis") or state.get("hypothesis")
    designer = services.get("designer")
    inputs = [camp / "goal.json"]
    if (camp / "evaluation_contract.json").is_file():
        inputs.append(camp / "evaluation_contract.json")
    context = _design_context(camp, state, spec)
    task = begin_role_task(
        camp,
        role="experiment_designer",
        inputs=inputs,
        request={"draft": spec, **context},
        artifacts=[{"artifact_id": row["artifact_id"], "kind": "selected_development_read", "sha256": row.get("sha256")}
                   for row in context.get("artifact_reads") or [] if row.get("status") == "read"],
    )
    payload: dict[str, Any] = {"status": "completed", "experiment_spec": spec, "summary_zh": "已写出 ExperimentSpec"}
    blocked = False
    design_failed = False
    if designer is not None:
        try:
            from react_agent.eeg_research.agentic.handoffs import record_read_delivery
            record_read_delivery(camp, context.get("artifact_reads") or [], consumer="experiment_designer",
                                 request_id=task["task_id"], input_digest=task["input_digest"])
            reply = designer(
                {
                    "task_id": task["task_id"],
                    "attempt_id": task["attempt_id"],
                    "input_digest": task["input_digest"],
                    "draft": spec,
                    **context,
                }
            )
            spec_reply = None
            if isinstance(reply, dict):
                spec_reply = reply.get("experiment_spec")
                inner = reply.get("payload")
                if spec_reply is None and isinstance(inner, dict):
                    spec_reply = inner.get("experiment_spec")
                payload["summary_zh"] = str(reply.get("summary_zh") or payload["summary_zh"])
                status = str(reply.get("status") or "")
                if status in {"partial", "failed", "requires_framework_extension", "blocked"} or reply.get("requires_framework_extension"):
                    blocked = True
                    design_failed = True
                    spec["role_status"] = status or "requires_framework_extension"
                    spec["status"] = "requires_framework_extension" if status in {"requires_framework_extension", "blocked"} or reply.get("requires_framework_extension") else "draft"
                    spec["missing_capability"] = reply.get("required_capability_ids") or reply.get("missing_inputs")
            if isinstance(spec_reply, dict):
                spec_reply = dict(spec_reply)
                if spec_reply.get("status") == "approved":
                    spec_reply["status"] = "draft"
                spec_reply.pop("approval_record", None)
                spec = {**spec, **spec_reply}
        except LlmUnavailable:
            payload["status"] = "partial"
            payload["summary_zh"] = "设计角色不可用，草稿未批准"
            spec["role_status"] = "partial"
            spec["status"] = "draft"
            design_failed = True
    if design_failed or blocked:
        spec["status"] = "blocked" if blocked and spec.get("role_status") in {"requires_framework_extension", "blocked"} else "draft"
        spec["blocked_reason"] = spec.get("blocked_reason") or ("requires_framework_extension" if blocked else "experiment_not_approved")
        spec["approval_record"] = None
        spec = approve_experiment(spec, camp=camp, context=context, producer=task)
        blocked = True
    else:
        spec = approve_experiment(spec, camp=camp, context=context, producer=task)
        blocked = not experiment_is_approved(spec)
    payload["experiment_spec"] = spec
    payload["status"] = "blocked" if blocked else payload["status"]
    path = camp / "experiments" / f"spec_{task['task_id']}.json"
    envelope = finish_role_task(camp, task, payload, kind="experiment_spec", path=path)
    stored = envelope.get("experiment_spec")
    if not isinstance(stored, dict):
        inner = envelope.get("payload")
        stored = inner.get("experiment_spec") if isinstance(inner, dict) else spec
    state["experiment"] = stored
    state["experiment_ref"] = (
        (stored.get("experiment_ref") if isinstance(stored, dict) else None)
        or (spec.get("experiment_ref") if isinstance(spec, dict) else None)
        or (envelope.get("artifact_refs") or [None])[-1]
    )
    state["hypothesis"] = stored.get("hypothesis") if isinstance(stored, dict) else state.get("hypothesis")
    state["candidate_ready"] = False
    state["experiment_failed"] = blocked
    _write(
        camp / "hypotheses" / f"h{len(state.get('decisions') or [])}.json",
        {"hypothesis": state["hypothesis"], "experiment": stored, "experiment_ref": state["experiment_ref"]},
    )


def _repair_candidate(camp: Path, state: dict[str, Any], services: Services) -> None:
    repair = state.get("repair_task") if isinstance(state.get("repair_task"), dict) else None
    if not repair or int(repair.get("remaining") or 0) <= 0:
        persist_failure(
            camp,
            state,
            phase="repair_candidate",
            error_type="no_repair_task",
            detail="repair_candidate_requires_existing_repair_task",
            recoverable=True,
        )
        return
    implementer = services.get("implement")
    if implementer is None:
        persist_failure(
            camp,
            state,
            phase="repair_candidate",
            error_type="implementer_missing",
            detail="repair_uses_existing_implement_path",
            recoverable=True,
        )
        return
    implementer(camp, state)


def _collect_diagnostics(camp: Path, state: dict[str, Any]) -> None:
    from react_agent.eeg_research.agentic.diagnostics import write_job_bundle

    jobs = [row for row in state.get("evidence") or [] if row.get("job_dir")]
    if not jobs:
        _append_derived(
            state,
            {
                "evidence_id": f"ev_diag_bundle_{len(state['evidence']) + 1}",
                "kind": "diagnostic_bundle",
                "status": "unavailable",
                "diagnostic_ref": None,
                "summary": {"reason": "no_job_dir"},
            },
        )
        return
    latest = jobs[-1]
    summary = write_job_bundle(Path(str(latest["job_dir"])))
    latest["diagnostics"] = summary
    latest["diagnostic_ref"] = str(Path(str(latest["job_dir"])) / "diagnostic_summary.json")
    _append_derived(
        state,
        {
            "evidence_id": f"ev_diag_bundle_{len(state['evidence']) + 1}",
            "kind": "diagnostic_bundle",
            "status": "ready",
            "diagnostic_ref": latest["diagnostic_ref"],
            "summary": summary,
        },
    )


def _audit_result(camp: Path, state: dict[str, Any], services: Services) -> None:
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task

    latest = next((row for row in reversed(state.get("evidence") or []) if row.get("comparison") or row.get("job_dir")), None)
    claims = []
    verdict = "PASS"
    if latest is None:
        verdict = "REVISE"
        claims.append({"claim": "no_job_result", "status": "unsupported"})
    else:
        if latest.get("comparison") and latest["comparison"].get("comparable"):
            claims.append({"claim": "comparison_comparable", "status": "supported", "ref": latest.get("evidence_id")})
        else:
            claims.append({"claim": "comparison_comparable", "status": "unsupported", "ref": latest.get("evidence_id")})
            verdict = "REVISE"
        if not latest.get("diagnostics"):
            claims.append({"claim": "diagnostics_present", "status": "unsupported"})
            verdict = "REVISE" if verdict != "BLOCK" else verdict
    import hashlib

    from react_agent.eeg_research.agentic.handoffs import development_view, load_analysis_views
    from react_agent.eeg_research.agentic.artifacts import file_digest

    manifest = []
    for row in state.get("evidence") or []:
        job_dir = row.get("job_dir")
        if not job_dir:
            continue
        job_path = Path(str(job_dir))
        try:
            job_path.resolve().relative_to(camp.resolve())
        except ValueError:
            continue
        for name in ("run_record.json", "metrics.json", "selected_checkpoint.json", "diagnostic_summary.json",
                     "frozen_run_spec.json", "source_binding.json", "hook_consumed.json"):
            path = job_path / name
            if not path.is_file():
                continue
            contents = json.loads(path.read_text(encoding="utf-8"))
            manifest.append({"path": str(path), "kind": name.removesuffix(".json"), "content_hash": file_digest(path),
                "scope": "development", "payload": development_view(contents)})
    from react_agent.eeg_research.agentic.artifacts import lookup, verify
    for row in state.get("evidence") or []:
        ref = row.get("analysis_artifact_id")
        if row.get("kind") != "analysis" or not ref:
            continue
        artifact = lookup(camp, str(ref))
        ok, reason = verify(camp, str(ref))
        if not ok:
            claims.append({"claim": "analysis_dependency_valid", "status": "unsupported", "ref": ref, "reason": reason})
            verdict = "REVISE" if verdict != "BLOCK" else verdict
        if artifact:
            path = Path(artifact["path"])
            if path.is_file() and path.resolve().is_relative_to(camp.resolve()):
                manifest.append({"path": str(path), "kind": "analysis", "content_hash": file_digest(path),
                    "declared_hash": artifact["sha256"], "verification_status": "verified" if ok else "invalid",
                    "scope": "development", "payload": development_view(json.loads(path.read_text())) if ok else None})
    for name in ("goal.json", "evaluation_contract.json", "confirmation_policy.json"):
        path = camp / name
        if path.is_file():
            manifest.append({"path": str(path), "kind": name.removesuffix(".json"), "content_hash": file_digest(path),
                "scope": "development", "payload": development_view(json.loads(path.read_text()))})
    for claim in claims:
        claim["claim_id"] = claim.get("claim_id") or claim["claim"]
    report_claims = [dict(claim) for claim in claims]
    for claim in report_claims:
        revision = (state.get("report_claim_revisions") or {}).get(claim["claim_id"])
        if revision:
            claim.update(revision_operation=revision["operation"], statement=revision.get("statement"),
                         scope_limits=revision.get("scope_limits") or [], evidence_refs=revision.get("evidence_refs") or [])
    report_draft = {
        "schema_version": "eeg_research.report_draft.v1", "scope": "development",
        "claims": report_claims, "latest": development_view(latest),
        "analyses": load_analysis_views(camp, state),
        "scope_limits": ["pilot evidence is not confirmed superiority", "final holdout excluded", "same-provider audit is not independent replication"],
        "dependency_manifest": manifest,
    }
    report_draft["dependency_manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    report_draft["report_hash"] = hashlib.sha256(json.dumps(report_draft, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    report_draft["report_ref"] = "report:" + report_draft["report_hash"]
    state["report_draft"] = report_draft

    from react_agent.eeg_research.agentic.audit_context import AUDIT_SCHEMA, IDENTITY_KEYS, reusable_audit, sync_issues
    dependency_snapshot = report_dependency_hash(state)
    cached = reusable_audit(camp, report_hash=report_draft["report_hash"], manifest_hash=report_draft["dependency_manifest_hash"],
                            report_ref=report_draft["report_ref"], dependency_hash=dependency_snapshot)
    if cached is not None:
        _accept_audit_projection(camp, state, cached["artifact_id"], cached["payload"])
        sync_issues(camp, state)
        state["last_local_result"] = {"kind": "audit", "new_information": False, "reason": "verified_audit_recovered"}
        return
    report_hash = report_draft["report_hash"]
    evidence_id = f"ev_audit_result_{len(state['evidence']) + 1}"
    payload = {
        "audit_schema_version": AUDIT_SCHEMA,
        "audited_report_ref": report_draft["report_ref"],
        "dependency_manifest_hash": report_draft["dependency_manifest_hash"],
        "report_claims": report_claims,
        "source_evidence_ids": [row.get("evidence_id") for row in state.get("evidence") or [] if row.get("evidence_id")],
        "evidence_id": evidence_id,
        "model_audit_status": "not_run", "identity_verified": False,
        "model_audit_missing_inputs": ["auditor_not_run"],
        "status": "completed",
        "verdict": verdict,
        "deterministic_verdict": verdict,
        "claims": claims,
        "report_hash": report_hash,
        "summary_zh": "审计基于已落盘的 comparison/diagnostics，未训练模型",
    }
    from react_agent.eeg_research.agentic.audit_context import audit_feedback
    from react_agent.eeg_research.agentic.handoffs import bounded_control_view
    feedback_view, feedback_limits = bounded_control_view(audit_feedback(camp, state))
    request = {
        "latest": development_view(latest), "draft": report_draft, "report_draft": report_draft,
        "deterministic_audit": {"authority": "runtime_integrity_precheck", "verdict": verdict,
                                "claims": claims, "scope": "development"},
        "dependency_manifest": manifest, "analyses": report_draft["analyses"], "claims": report_claims,
        "report_hash": report_hash, "dependency_manifest_hash": report_draft["dependency_manifest_hash"],
        "audited_report_ref": report_draft["report_ref"], "audited_dependency_hash": dependency_snapshot,
        "audit_feedback": feedback_view,
        "context_limits": {"audit_feedback": feedback_limits},
    }
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"], request=request)
    request = {**request, "task_id": task["task_id"], "attempt_id": task["attempt_id"], "input_digest": task["input_digest"]}
    auditor = services.get("auditor")
    if auditor is not None:
        from react_agent.eeg_research.agentic.roles import bind_role_output, RoleResultError
        from react_agent.eeg_research.agentic.schemas import AuditReport
        from pydantic import ValidationError
        try:
            reply = auditor(request)
            bound = bind_role_output(reply, task=task)
            body = bound.get("payload") if isinstance(bound.get("payload"), dict) else bound
            wrapper_statuses = {str(bound.get("status") or "completed")}
            while isinstance(body.get("payload"), dict) and not body.get("verdict"):
                wrapper_statuses.add(str(body.get("status") or "completed"))
                body = body["payload"]
            model_verdict = body.get("verdict")
            payload["auditor_claims"] = body.get("claims") or []
            for key in ("open_issues", "required_corrections", "review_limits", "resolved_issues"):
                payload[key] = body.get(key) or []
            payload["summary_zh"] = str(body.get("summary_zh") or payload["summary_zh"])
            payload["model_identity"] = {key: body.get(key) for key in IDENTITY_KEYS}
            missing = [key for key in IDENTITY_KEYS if body.get(key) != payload[key]]
            statuses = wrapper_statuses | {str(body.get("status") or "completed")}
            if statuses != {"completed"}:
                missing.append("model_audit_not_completed")
            domain = {key: value for key, value in body.items() if key in AuditReport.model_fields}
            try:
                AuditReport.model_validate(domain)
            except ValidationError:
                missing.append("audit_domain_invalid")
            payload["identity_verified"] = not missing
            payload["model_audit_missing_inputs"] = missing
            payload["model_audit_status"] = "partial" if missing else "completed"
            if missing:
                payload["status"] = "partial"
            if verdict in {"REVISE", "BLOCK"} and model_verdict == "PASS":
                payload["model_verdict_ignored"] = "PASS"
            if not missing and model_verdict in {"PASS", "REVISE", "BLOCK"}:
                ranking = {"PASS": 0, "REVISE": 1, "BLOCK": 2}
                payload["verdict"] = max((verdict, model_verdict), key=ranking.__getitem__)
        except (LlmUnavailable, RoleResultError, ValueError, TypeError) as exc:
            payload.update(status="partial", model_audit_status="partial", identity_verified=False,
                           model_audit_missing_inputs=[str(exc)])
    deterministic_issues = [{"claim_id": claim["claim_id"], "category": "integrity", "severity": "blocking",
        "problem": "Deterministic check failed: " + claim["claim"],
        "required_correction": "Supply valid development evidence for this check; a model verdict cannot override it",
        "evidence_refs": [claim["ref"]] if claim.get("ref") else []}
        for claim in claims if claim.get("status") == "unsupported"]
    payload["open_issues"] = deterministic_issues + list(payload.get("open_issues") or [])
    # Cache comparison and the immutable result bind the same pre-call snapshot.
    # Derived audit rows are excluded by report_dependency_hash.
    payload["audited_dependency_hash"] = dependency_snapshot
    path = camp / "audits" / f"{task['task_id']}.json"
    envelope = finish_role_task(camp, task, payload, kind="audit", path=path)
    _accept_audit_projection(camp, state, envelope["artifact_refs"][-1], payload)
    sync_issues(camp, state)


def _accept_audit_projection(camp: Path, state: dict[str, Any], artifact_id: str, payload: dict[str, Any]) -> None:
    """Recover the immutable result without another task, write or charge."""
    state["audit_status"] = str(payload["status"]) if payload.get("status") in {"partial", "failed", "blocked"} else {
        "PASS": "pass", "REVISE": "revise", "BLOCK": "block"}.get(str(payload["verdict"]), "unavailable")
    state["audit_report_hash"] = payload["report_hash"]
    state["audit_claims"] = payload.get("claims") or []
    state["audit_fresh"] = True
    state["latest_audit_artifact_id"] = artifact_id
    if not any(row.get("audit_artifact_id") == artifact_id for row in state.get("evidence") or []):
        state.setdefault("evidence", []).append({"evidence_id": payload["evidence_id"], "kind": "audit",
            "audit_artifact_id": artifact_id, "summary": {key: payload.get(key) for key in (
                "verdict", "claims", "auditor_claims", "report_hash", "model_verdict_ignored",
                "open_issues", "required_corrections", "review_limits", "identity_verified", "model_audit_status")}})
    state["audited_report_hash"] = payload.get("audited_dependency_hash")
    state["audit_fresh"] = bool(state["audited_report_hash"] and state["audited_report_hash"] == report_dependency_hash(state))
    refresh_audit_freshness(state)


def _revise_report(camp: Path, state: dict[str, Any], raw: dict[str, Any]) -> None:
    """Revise only claim wording/scope/refs; immutable measurements stay untouched."""
    from react_agent.eeg_research.agentic.schemas import ReportClaimRevision
    from react_agent.eeg_research.agentic.audit_context import audit_feedback
    claims = {row.get("claim_id"): row for row in (state.get("report_draft") or {}).get("claims") or []}
    known = {row.get("evidence_id") for row in state.get("evidence") or []}
    issues = audit_feedback(camp, state)["issues"]
    linked = set(raw.get("resolves_issue_ids") or [])
    if not raw.get("report_revision") or not linked or not linked <= {row["issue_id"] for row in issues if row["status"] == "open"}:
        raise ValueError("report_revision_requires_existing_open_issue")
    revisions = [ReportClaimRevision.model_validate(row).model_dump() for row in raw["report_revision"]]
    for row in revisions:
        if row["claim_id"] not in claims or not set(row["evidence_refs"]) <= known:
            raise ValueError("report_revision_unknown_claim_or_evidence")
        if not any(issue["issue_id"] in linked and issue.get("claim_id") == row["claim_id"] for issue in issues):
            raise ValueError("report_revision_issue_claim_mismatch")
    state.setdefault("report_claim_revisions", {}).update({row["claim_id"]: row for row in revisions})
    draft = json.loads(json.dumps(state["report_draft"]))
    for row in draft["claims"]:
        revision = state["report_claim_revisions"].get(row.get("claim_id"))
        if revision:
            row.update(revision_operation=revision["operation"], statement=revision.get("statement"),
                       scope_limits=revision["scope_limits"], evidence_refs=revision["evidence_refs"])
    import hashlib
    material = {key: value for key, value in draft.items() if key not in {"report_hash", "report_ref"}}
    draft["report_hash"] = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()
    draft["report_ref"] = "report:" + draft["report_hash"]
    state["report_draft"] = draft
    refresh_audit_freshness(state)
    state["last_local_result"] = {"kind": "report_revision", "claim_ids": [row["claim_id"] for row in revisions],
                                  "issues_closed": False, "requires_reaudit": True}
    event(camp, "report_claims_revised", claim_ids=[row["claim_id"] for row in revisions], issue_ids=sorted(linked))


def _append_derived(state: dict[str, Any], row: dict[str, Any]) -> None:
    """Skip a derived row whose content matches the last row of its kind."""
    body = {key: value for key, value in row.items() if key != "evidence_id"}
    for old in reversed(state.get("evidence") or []):
        if old.get("kind") == row.get("kind"):
            if {key: value for key, value in old.items() if key != "evidence_id"} == body:
                state["last_local_result"] = {"kind": row.get("kind"), "new_information": False, "same_as": old.get("evidence_id")}
                return
            break
    state["evidence"].append(row)
    state["last_local_result"] = {"kind": row.get("kind"), "new_information": True, "evidence_id": row.get("evidence_id")}


def _inspect(camp: Path) -> dict[str, Any]:
    contract = _read(camp / "evaluation_contract.json")
    roles: dict[str, int] = {}
    missing = 0
    for row in contract.get("files") or []:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
        if not Path(row["path"]).is_file():
            missing += 1
    return {
        "research_scope": contract.get("research_scope"),
        "val_mode": contract.get("val_mode"),
        "files_by_role": roles,
        "missing_files": missing,
        "feature_caches_present": all(Path(path).is_file() for path in contract.get("feature_caches") or []),
    }


def _train_log_tail(path: Path, *, limit: int = 4000) -> str:
    """Last characters of a job log. Missing files are empty, not an error."""
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > limit:
        return text[-limit:]
    return text


def _diagnose(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Curve summary from history files, plus the tail of jobs that never wrote one."""
    curves = []
    failures = []
    for row in state.get("evidence") or []:
        job = row.get("job_dir")
        if not job:
            continue
        job_path = Path(job)
        history = job_path / "history.jsonl"
        points: list[dict[str, Any]] = []
        if history.is_file():
            points = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        if points:
            best = max(points, key=lambda item: float(item.get("fixed_bank_top1") or 0.0))
            curves.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "fidelity": row.get("fidelity"),
                    "epochs": len(points),
                    "best_epoch": best.get("epoch"),
                    "best_fixed_bank_top1": best.get("fixed_bank_top1"),
                    "first_train_loss": points[0].get("train_loss"),
                    "last_train_loss": points[-1].get("train_loss"),
                    "fixed_bank_trend": [round(float(item.get("fixed_bank_top1") or 0.0), 5) for item in points],
                }
            )
            continue
        tail = _train_log_tail(job_path / "train.log")
        failed = row.get("evaluation_valid") is False or row.get("job_status") in {"failed", "invalid"}
        if not failed and not tail:
            continue
        failures.append(
            {
                "candidate_id": row.get("candidate_id"),
                "fidelity": row.get("fidelity"),
                "reason": row.get("reason"),
                "job_status": row.get("job_status"),
                "train_log_tail": tail,
            }
        )
    status = "ready" if curves or failures else "unavailable"
    payload: dict[str, Any] = {
        "evidence_id": f"ev_diag_{len(state['evidence']) + 1}",
        "kind": "learning_profile",
        "status": status,
        "curve": curves,
    }
    if failures:
        payload["failures"] = failures
    return payload


def _implement_inline(camp: Path, state: dict[str, Any], decision: dict[str, Any]) -> None:
    """Direct patch from one reply. Used by synthetic tests; the worker uses NativePatch."""
    from react_agent.eeg_research.agentic.interface import candidate_interface
    from react_agent.eeg_research.agentic.lineage import materialize

    candidate_id = candidate_implementation_target(camp, state)
    workspace = camp / "candidates" / candidate_id
    raw = decision.get("raw") or {}
    content = raw.get("content")
    if not content:
        state["detail"] = "implementation_failed"
        return
    workspace.mkdir(parents=True, exist_ok=True)
    materialize(camp, workspace, state.get("experiment") or {})
    protocol = load_protocol(camp)
    (workspace / "input_spec.json").write_text(json.dumps(candidate_interface(protocol), ensure_ascii=False, indent=2), encoding="utf-8")
    applied = apply_candidate_patch(workspace, raw.get("relative") or "extension/eeg_candidate.py", content, raw.get("expected_base_hash") or "")
    if not applied.get("ok"):
        state["detail"] = applied.get("error")
        return
    checked = run_candidate_check(workspace)
    (workspace / "checks.json").write_text(json.dumps(checked, ensure_ascii=False), encoding="utf-8")
    state["candidates"].append({"candidate_id": candidate_id, "status": "checked" if checked.get("ok") else "implementation_failed"})
    if not checked.get("ok"):
        state["detail"] = "implementation_failed"
        return
    finished = finish_patch(workspace, raw.get("reason_zh") or "candidate")
    state["candidate_ready"] = bool(finished.get("ok"))
    state["candidate_id"] = candidate_id
    state["status"] = "checking"
    if state["candidate_ready"] and isinstance(state.get("experiment"), dict):
        from react_agent.eeg_research.agentic.run_context import persist_approved_binding

        persist_approved_binding(
            camp,
            candidate_id,
            state.get("execution_spec") if isinstance(state.get("execution_spec"), dict) else state["experiment"],
            spec_ref=state.get("experiment_ref"),
        )


def _pair_record(result: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_pp": comparison.get("delta_pp"),
        "fidelity": result.get("fidelity"),
        "seed": result.get("seed"),
        "run_id": result.get("evidence_id") or result.get("job_id"),
        "job_id": result.get("job_id"),
        "control_run_id": comparison.get("control_run_id"),
        "checkpoint_id": result.get("checkpoint_id"),
        "evaluation_valid": result.get("evaluation_valid"),
        "source_hash": result.get("source_hash"),
        "config_hash": result.get("config_hash"),
        "approval_ref": result.get("experiment_ref"),
        "approval_hash": result.get("spec_hash"),
    }


SETTLEMENT_STAGES = (
    "validated_result",
    "cost_committed",
    "episode_committed",
    "evidence_committed",
    "diagnostics_committed",
    "comparison_committed",
    "promotion_committed",
    "audit_marked",
)


def _load_settlement(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "settlement.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload:
            return payload
    return {
        "schema_version": "eeg_research.settlement.v1",
        "job_id": job_dir.name,
        "completed": [],
        "status": "pending",
    }


def _save_settlement(job_dir: Path, settlement: dict[str, Any]) -> None:
    done = set(settlement.get("completed") or [])
    settlement["status"] = "complete" if set(SETTLEMENT_STAGES) <= done else "partial"
    _write(job_dir / "settlement.json", settlement)


def _record_job(camp: Path, state: dict[str, Any], record: dict[str, Any]) -> None:
    job_id = str(record.get("job_id") or "")
    job_dir = camp / "jobs" / job_id
    settlement = _load_settlement(job_dir)
    existing = next(
        (
            row
            for row in state.get("evidence") or []
            if row.get("job_id") == job_id or row.get("evidence_id") == f"ev_{job_id}"
        ),
        None,
    )
    incoming = dict(record.get("result") or {})
    incoming_hash = _result_hash({**incoming, "job_id": job_id, "candidate_id": record.get("candidate_id") or incoming.get("candidate_id")}) if incoming else None
    if existing is not None and existing.get("result_hash") and incoming_hash and incoming and incoming_hash != existing["result_hash"]:
        material = {key: incoming.get(key) for key in ("fixed_bank_top1", "evaluation_valid", "checkpoint_id", "source_hash", "config_hash") if key in incoming}
        existing_material = {key: existing.get(key) for key in material}
        if material and existing_material != material:
            _write(
                job_dir / "job_result_conflict.json",
                {
                    "schema_version": "eeg_research.job_result_conflict.v1",
                    "job_id": job_id,
                    "existing_hash": existing["result_hash"],
                    "incoming_hash": incoming_hash,
                },
            )
            persist_failure(
                camp,
                state,
                phase="settle",
                error_type="job_result_conflict",
                detail="job_result_conflict",
                recoverable=False,
            )
            return
    comparison_path = camp / "comparisons" / f"{(existing or {}).get('evidence_id') or incoming.get('evidence_id') or f'ev_{job_id}'}.json"
    completed = list(settlement.get("completed") or [])
    if settlement.get("status") == "complete":
        if not (job_dir / "diagnostic_summary.json").is_file() and "diagnostics_committed" in completed:
            completed.remove("diagnostics_committed")
            settlement["status"] = "partial"
        if not comparison_path.is_file() and "comparison_committed" in completed:
            completed.remove("comparison_committed")
            if "promotion_committed" in completed:
                completed.remove("promotion_committed")
            settlement["status"] = "partial"
        settlement["completed"] = completed
    if (
        existing is not None
        and settlement.get("status") == "complete"
        and (job_dir / "diagnostic_summary.json").is_file()
        and comparison_path.is_file()
        and incoming_hash in {None, existing.get("result_hash")}
    ):
        if existing not in (state.get("evidence") or []):
            state.setdefault("evidence", []).append(existing)
        return
    if existing is None:
        result = incoming
    else:
        if incoming:
            for key, value in incoming.items():
                if value is not None:
                    if key == "evidence_id" and existing.get("evidence_id"):
                        continue
                    existing[key] = value
        result = existing
    result["evidence_id"] = result.get("evidence_id") or f"ev_{job_id or record.get('candidate_id')}"
    result["job_id"] = job_id
    result["candidate_id"] = record.get("candidate_id") or result.get("candidate_id")
    result["job_dir"] = str(job_dir)
    result["seed"] = record.get("seed") if record.get("seed") is not None else result.get("seed")
    result["training_seed"] = result.get("seed")
    result["job_status"] = record.get("status") or result.get("job_status")
    result["source_hash"] = result.get("source_hash") or record.get("source_hash") or (record.get("manifest") or {}).get("entry_sha256")
    if (job_dir / "last.ckpt").is_file() and not result.get("checkpoint_id"):
        from react_agent.eeg_research.agentic.artifacts import file_digest

        result["checkpoint_id"] = file_digest(job_dir / "last.ckpt")
    hook_path = job_dir / "hook_config.json"
    if hook_path.is_file():
        hook = _read(hook_path)
        result["config_hash"] = hook.get("config_hash") or hook.get("intervention_config_hash")
        result["intervention_config_hash"] = hook.get("intervention_config_hash") or result.get("config_hash")
        result["run_config_hash"] = hook.get("run_config_hash")
        result["spec_hash"] = result.get("spec_hash") or hook.get("spec_hash")
        result["experiment_ref"] = result.get("experiment_ref") or hook.get("spec_ref")
    frozen_path = job_dir / "frozen_run_spec.json"
    if frozen_path.is_file():
        frozen = _read(frozen_path)
        result["spec_hash"] = result.get("spec_hash") or frozen.get("spec_hash")
        result["experiment_ref"] = result.get("experiment_ref") or frozen.get("approval_ref")
        result["intervention_config_hash"] = result.get("intervention_config_hash") or frozen.get("intervention_config_hash")
        result["run_config_hash"] = result.get("run_config_hash") or frozen.get("run_config_hash")
    if result.get("evaluation_valid"):
        result["contract_fingerprint"] = result.get("execution_fingerprint") or state.get("contract_fingerprint")
    from react_agent.eeg_research.agentic.run_context import load_approved_binding

    binding = load_approved_binding(camp, str(result.get("candidate_id") or ""))
    if binding:
        result["experiment_ref"] = binding.get("spec_ref") or result.get("experiment_ref")
        result["spec_hash"] = binding.get("spec_hash") or result.get("spec_hash")
    elif state.get("candidate_id") == result.get("candidate_id"):
        result["hypothesis"] = state.get("hypothesis")
        result["experiment_ref"] = result.get("experiment_ref") or state.get("experiment_ref")
        result["spec_hash"] = result.get("spec_hash") or (
            (state.get("experiment") or {}).get("spec_hash") if isinstance(state.get("experiment"), dict) else None
        )
    result["result_hash"] = result.get("result_hash") or _result_hash(result)
    _write(job_dir / "run_record.json", {"schema_version": "eeg_research.run_record.v1", "result": result, "result_hash": result["result_hash"]})

    completed = list(settlement.get("completed") or [])

    def mark(stage: str) -> None:
        if stage not in completed:
            completed.append(stage)
        settlement["completed"] = completed
        _save_settlement(job_dir, settlement)

    if "validated_result" not in completed:
        mark("validated_result")
    if "cost_committed" not in completed:
        charge_gpu(camp, state, float(record.get("gpu_seconds") or 0), job_id=job_id)
        cost_path = camp / "cost.json"
        cost = _read(cost_path)
        cost["training_jobs"] = int(state.get("training_jobs", 0))
        cost.setdefault("api_usd", None)
        _write(cost_path, cost)
        mark("cost_committed")
    if existing is None and not any(row.get("job_id") == job_id for row in state.get("evidence") or []):
        state.setdefault("evidence", []).append(result)
    if "episode_committed" not in completed:
        store = EpisodeStore(camp)
        ep = episode(
            task_hash=str(state.get("goal_id")),
            candidate_id=str(result.get("candidate_id")),
            kind="exploratory_result" if result.get("evaluation_valid") else "implementation_failure",
            fidelity=str(result.get("fidelity")),
            seed=int(record.get("seed") or result.get("seed") or 0),
            metric=result.get("fixed_bank_top1") if result.get("evaluation_valid") else None,
            contract_fingerprint=str(state.get("contract_fingerprint")),
            artifact=str(job_dir / "metrics.json") if result.get("evaluation_valid") else str(job_dir),
            job_id=job_id,
        )
        stored = store.persist_episode(ep)
        if not any(item.get("episode_id") == stored.get("episode_id") or item.get("job_id") == job_id for item in state.get("memory") or []):
            state.setdefault("memory", []).append(stored)
        mark("episode_committed")
    else:
        store = EpisodeStore(camp)
        for item in store.list_episodes():
            if str(item.get("job_id") or "") == job_id and not any(
                row.get("episode_id") == item.get("episode_id") or row.get("job_id") == job_id for row in state.get("memory") or []
            ):
                state.setdefault("memory", []).append(item)
    if "evidence_committed" not in completed:
        mark("evidence_committed")
    if "diagnostics_committed" not in completed or not (job_dir / "diagnostic_summary.json").is_file():
        from react_agent.eeg_research.agentic.diagnostics import write_job_bundle

        diagnostics = write_job_bundle(job_dir)
        result["diagnostics"] = diagnostics
        result["diagnostic_ref"] = str(job_dir / "diagnostic_summary.json")
        mark("diagnostics_committed")
    else:
        diagnostics = result.get("diagnostics") or {}
    if "comparison_committed" not in completed:
        protocol = load_protocol(camp)
        others = [row for row in state.get("evidence") or [] if row is not result]
        control = matched_control(others, result, protocol)
        comparison = compare_runs(candidate=result, control=control, protocol=protocol)
        result["comparison_status"] = "comparable" if comparison.get("comparable") else comparison.get("reason")
        result["comparison"] = comparison
        if comparison.get("comparable") and comparison.get("delta_pp") is not None:
            result["delta_vs_control_pp"] = comparison["delta_pp"]
            result["control_id"] = comparison.get("control_run_id")
            result["pair_record"] = _pair_record(result, comparison)
        else:
            result.pop("delta_vs_control_pp", None)
            result.pop("control_id", None)
        mark("comparison_committed")
    else:
        comparison = result.get("comparison") or {}
    if "promotion_committed" not in completed:
        goal = _read(camp / "goal.json")
        paired_records = [
            row.get("pair_record")
            for row in state.get("evidence") or []
            if row.get("candidate_id") == result.get("candidate_id")
            and isinstance(row.get("pair_record"), dict)
            and row.get("evaluation_valid")
        ]
        from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy

        policy = load_confirmation_policy(camp, goal)
        promo = promotion_decision(
            comparison=comparison if isinstance(comparison, dict) else {},
            goal=goal,
            paired_records=paired_records,
            fidelity=str(result.get("fidelity") or ""),
            policy=policy,
        )
        result["promotion"] = promo
        _write(
            camp / "comparisons" / f"{result['evidence_id']}.json",
            {"comparison": comparison, "promotion": promo, "diagnostics": result.get("diagnostics") or diagnostics},
        )
        if promo.get("status") == "confirmed":
            confirmation = {
                "schema_version": "eeg_research.confirmation_record.v1",
                "policy_hash": (policy or {}).get("policy_hash"),
                "status": "confirmed",
                "pair_refs": [row.get("run_id") for row in paired_records if isinstance(row, dict)],
                "aggregate": promo,
                "limitations": ["development_scope_only"],
            }
            confirm_path = camp / "confirmations" / f"{result['evidence_id']}.json"
            _write(confirm_path, confirmation)
            result["confirmation_ref"] = str(confirm_path)
        mark("promotion_committed")
    if "audit_marked" not in completed:
        if state.get("audit_status") in {"pass", "revise", "block"}:
            state["audit_status"] = "stale"
            state["audit_fresh"] = False
        mark("audit_marked")
    event(camp, "job_settled", job_id=record.get("job_id"), status=record.get("status"), valid=result.get("evaluation_valid"))


def _launch(
    camp: Path,
    state: dict[str, Any],
    services: Services,
    candidate_id: str,
    fidelity: str,
    *,
    training_seed: int | None = None,
) -> None:
    """Start one job, or adopt a job.json already on disk. Never start a second process for the same id."""
    job_index = int(state["training_jobs"]) + 1
    job_id = f"j{job_index}_{candidate_id}_{fidelity}"
    job_dir = camp / "jobs" / job_id
    if job_dir.exists() and not (job_dir / "job.json").is_file():
        persist_failure(
            camp,
            state,
            phase="train",
            error_type="training_start_unconfirmed",
            detail="training_start_unconfirmed",
            recoverable=False,
        )
        return
    if (job_dir / "job.json").is_file():
        record = _read(job_dir / "job.json")
        recorded = record.get("candidate_id")
        if recorded and recorded != candidate_id:
            persist_failure(
                camp,
                state,
                phase="train",
                error_type="job_identity_mismatch",
                detail="job_identity_mismatch",
                recoverable=False,
            )
            return
        if (record.get("status") or "running") != "running":
            persist_failure(
                camp,
                state,
                phase="train",
                error_type=str(record.get("detail") or record.get("status") or "job_not_running"),
                detail=str(record.get("detail") or record.get("status") or "job_not_running"),
                recoverable=False,
            )
            return
        state["training_jobs"] = job_index
        state["live_job"] = job_id
        state["status"] = "training"
        return
    if training_seed is not None:
        state["pending_training_seed"] = training_seed
    record = services["launch"](camp, state, job_id, candidate_id, fidelity)
    state.pop("pending_training_seed", None)
    if (record.get("status") or "running") != "running":
        persist_failure(
            camp,
            state,
            phase="train",
            error_type=str(record.get("detail") or record.get("status") or "launch_blocked"),
            detail=str(record.get("detail") or record.get("status") or "launch_blocked"),
            recoverable=False,
        )
        return
    if candidate_id == "baseline":
        state["baseline_jobs"] = int(state.get("baseline_jobs") or 0) + 1
    state["training_jobs"] = job_index
    state["live_job"] = record["job_id"]
    state["status"] = "training"
    event(camp, "job_started", job_id=record["job_id"], candidate_id=candidate_id, fidelity=fidelity, seed=training_seed)


def _reviewed_ids(state: dict[str, Any]) -> set[str]:
    names = {"baseline"}
    for row in state.get("candidates") or []:
        if row.get("status") == "ready" and row.get("candidate_id"):
            names.add(str(row["candidate_id"]))
    if state.get("candidate_ready") and state.get("candidate_id"):
        names.add(str(state["candidate_id"]))
    return names


def _train(
    camp: Path,
    state: dict[str, Any],
    action: str,
    runner: Any,
    services: Services,
    raw: dict[str, Any] | None = None,
) -> None:
    from react_agent.eeg_research.agentic.planner import training_actions

    if action not in training_actions(_read(camp / "goal.json")):
        persist_failure(camp, state, phase="train", error_type="training_action_outside_goal",
                        detail=f"training_action_outside_goal:{action}", recoverable=False)
        return
    if state.get("gpu_seconds_left", 0) <= 0 or state["training_jobs"] >= state["max_training_jobs"]:
        state["status"] = "blocked"
        state["detail"] = "budget_exhausted"
        return
    fidelity = "pilot" if action == "run_pilot" else "full"
    raw = raw or {}
    reviewed = _reviewed_ids(state)
    requested = raw.get("target_id")
    candidate_id = requested or state.get("candidate_id")
    if requested and requested not in reviewed:
        persist_failure(
            camp,
            state,
            phase="train",
            error_type="unknown_target",
            detail=f"unknown_target:{requested}",
            recoverable=False,
        )
        return
    if not candidate_id:
        persist_failure(
            camp,
            state,
            phase="train",
            error_type="target_id_missing",
            detail="target_id_missing",
            recoverable=False,
        )
        return
    if candidate_id != "baseline":
        from react_agent.eeg_research.agentic.experiment_gate import ExperimentResolutionError, resolve_approved_experiment

        spec = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
        from react_agent.eeg_research.agentic.run_context import load_approved_binding

        binding = load_approved_binding(camp, candidate_id)
        try:
            resolve_approved_experiment(
                camp,
                spec_ref=(binding or {}).get("spec_ref") or state.get("experiment_ref"),
                expected_hash=(binding or {}).get("spec_hash"),
                target_id=candidate_id,
                state=state,
                action="replicate" if action == "replicate" else "train",
            )
        except ExperimentResolutionError as exc:
            persist_failure(
                camp,
                state,
                phase="train",
                error_type=exc.reason,
                detail=exc.reason,
                recoverable=True,
            )
            return
    protocol = load_protocol(camp) or {}
    from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy

    policy = load_confirmation_policy(camp, _read(camp / "goal.json"))
    if policy and policy.get("seeds_declared"):
        protocol = dict(protocol)
        protocol["training_seeds"] = list(policy.get("training_seeds") or [])
    used = {
        int(row.get("seed") or 0)
        for row in state.get("evidence") or []
        if row.get("candidate_id") == candidate_id and row.get("fidelity") == fidelity and row.get("evaluation_valid")
    }
    if action == "replicate":
        training_seed = next_unused_training_seed(protocol, used)
        if training_seed is None:
            persist_failure(
                camp,
                state,
                phase="train",
                error_type="no_unused_training_seed",
                detail="no_unused_training_seed",
                recoverable=False,
            )
            return
    else:
        training_seed = int(protocol.get("training_seed", protocol.get("seed") or 0))
    job_index = state["training_jobs"] + 1
    job_dir = camp / "jobs" / f"j{job_index}"
    if job_dir.exists() and not (job_dir / "job.json").is_file() and not (job_dir / "metrics.json").is_file():
        persist_failure(
            camp,
            state,
            phase="train",
            error_type="training_start_unconfirmed",
            detail="training_start_unconfirmed",
            recoverable=False,
        )
        return
    if services.get("launch") is not None:
        has_control = any(
            row.get("candidate_id") == "baseline"
            and row.get("fidelity") == fidelity
            and row.get("evaluation_valid")
            and int(row.get("seed") or 0) == int(training_seed)
            for row in state.get("evidence") or []
        )
        if candidate_id != "baseline" and not has_control:
            if state["training_jobs"] + 2 > state["max_training_jobs"]:
                persist_failure(
                    camp,
                    state,
                    phase="train",
                    error_type="unmatched_control",
                    detail="unmatched_control",
                    recoverable=True,
                )
                return
            queued = list(state.get("queued") or [])
            queued.append([candidate_id, fidelity, training_seed])
            state["queued"] = queued
            _launch(camp, state, services, "baseline", fidelity, training_seed=training_seed)
        else:
            _launch(camp, state, services, candidate_id, fidelity, training_seed=training_seed)
        return
    job = camp / "jobs" / f"j{job_index}"
    job.mkdir(parents=True, exist_ok=True)
    from react_agent.eeg_research.agentic.hook_config import write_hook_config
    from react_agent.eeg_research.agentic.run_context import resolve_run_context

    run_context = resolve_run_context(camp, candidate_id, state, fidelity=fidelity, seed=training_seed, protocol=protocol)
    write_hook_config(
        job,
        run_context.get("frozen_run_spec") or run_context["hook_spec"],
        spec_ref=run_context.get("spec_ref"),
        spec_hash=run_context.get("spec_hash"),
        seed=training_seed,
        fidelity=fidelity,
    )
    if run_context.get("frozen_run_spec"):
        from react_agent.eeg_research.agentic.run_context import persist_frozen_run_spec

        persist_frozen_run_spec(job, run_context["frozen_run_spec"])
    manifest = _read(camp / "candidates" / candidate_id / "source_manifest.json")
    if runner is None:
        state["status"] = "blocked"
        state["detail"] = "no_job_launcher"
        return
    result = runner(job, manifest, fidelity)
    state["training_jobs"] = job_index
    accepted = accept_job(job, manifest, fidelity, protocol=load_protocol(camp))
    _record_job(
        camp,
        state,
        {
            "job_id": job.name,
            "candidate_id": candidate_id,
            "seed": result.get("seed", training_seed),
            "status": "finished" if accepted.get("evaluation_valid") else "invalid",
            "gpu_seconds": result.get("gpu_seconds", 0),
            "result": accepted,
        },
    )
    state["evidence"][-1]["evidence_id"] = f"ev_{job_index}"
    others = [row for row in state["evidence"][:-1] if comparable(row, state["evidence"][-1])]
    if others and state["evidence"][-1].get("evaluation_valid"):
        state["evidence"][-1]["delta"] = state["evidence"][-1]["fixed_bank_top1"] - others[0]["fixed_bank_top1"]
    state["status"] = "analyzing"
