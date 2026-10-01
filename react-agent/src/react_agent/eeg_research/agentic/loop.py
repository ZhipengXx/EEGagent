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
from react_agent.eeg_research.agentic.planner import STOP_REASONS, available_actions, decide, evidence_count
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
    return public


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def report_dependency_hash(state: dict[str, Any]) -> str:
    """Hash claims plus every settled evidence identity. New rows make a prior audit stale."""
    import hashlib

    experiment = state.get("experiment") if isinstance(state.get("experiment"), dict) else {}
    report = state.get("report_draft") if isinstance(state.get("report_draft"), dict) else {}
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
            for row in state.get("evidence") or []
        ],
        "report_hash": report.get("report_hash") or state.get("audit_report_hash"),
        "dependency_manifest_hash": report.get("dependency_manifest_hash") or state.get("audited_report_hash"),
        "file_hashes": state.get("report_file_hashes") or report.get("file_hashes"),
        "claims": report.get("claims") or state.get("audit_claims") or [],
        "experiment_spec_hash": experiment.get("spec_hash"),
        "confirmation_policy_hash": state.get("confirmation_policy_hash"),
    }
    return hashlib.sha256(json.dumps(deps, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def refresh_audit_freshness(state: dict[str, Any]) -> str:
    current = report_dependency_hash(state)
    audited = state.get("audited_report_hash")
    if audited and audited != current and state.get("audit_status") in {"pass", "revise", "block"}:
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
    state = {
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


def rebuild_campaign_projection(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Rebuild evidence/memory/counts from on-disk RunRecords when outer state lagged."""
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
    """Merge decision files the state does not yet list. Does not call the planner or touch the ledger.

    paused, cancelled, and finished stay as they are. Stop and pause are not rewritten here.
    """
    state = load_state(camp)
    status = state.get("status")
    known = {row.get("decision_id") for row in state.get("decisions") or []}
    folder = camp / "decisions"
    added = False
    if folder.is_dir():
        paths = sorted(folder.glob("d*.json"), key=lambda path: _decision_number(path.stem))
        for path in paths:
            payload = _read(path)
            record = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            if not record:
                continue
            decision_id = str(record.get("decision_id") or path.stem)
            if decision_id in known:
                continue
            row = dict(record)
            row["decision_id"] = decision_id
            state.setdefault("decisions", []).append(row)
            known.add(decision_id)
            added = True
    rebuild_campaign_projection(camp, state)
    if added:
        state["status"] = status
        save_state(camp, state)
        state = load_state(camp)
        if status in {"paused", "cancelled", "finished"}:
            state["status"] = status
            _write(camp / "campaign_state.json", state)
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


def observation(camp: Path) -> dict[str, Any]:
    state = load_state(camp)
    contract = public_contract(_read(camp / "evaluation_contract.json"))
    from react_agent.eeg_research.agentic.planner import blocked_actions, eligible_targets

    actions = available_actions(state)
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
    parent_binding = None if parent_id == "baseline" else load_approved_binding(camp, str(parent_id))
    control_binding = None if control_id == "baseline" else load_approved_binding(camp, str(control_id))
    return {
        "goal": _planner_goal(goal),
        "confirmation_policy": policy,
        "confirmation_policy_hash": (policy or {}).get("policy_hash") or state.get("confirmation_policy_hash"),
        "parent_source": {
            "candidate_id": parent_id,
            "spec_hash": None if parent_binding is None else parent_binding.get("spec_hash"),
            "recipe": None if parent_binding is None else {key: parent_binding.get(key) for key in ("model", "objective", "transform")},
        },
        "control_source": {
            "candidate_id": control_id,
            "spec_hash": None if control_binding is None else control_binding.get("spec_hash"),
            "recipe": None if control_binding is None else {key: control_binding.get(key) for key in ("model", "objective", "transform")},
        },
        "contract": contract,
        "candidate_interface": candidate_interface(protocol),
        "evidence": evidence,
        "candidates": state.get("candidates") or [],
        "hypothesis": state.get("hypothesis"),
        "experiment": state.get("experiment"),
        "candidate_ready": state.get("candidate_ready"),
        "available_actions": actions,
        "eligible_targets": eligible_targets(state),
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
    }


Services = dict[str, Callable[..., Any]]


def tick(camp: Path, backend: Any, runner: Any | None = None, services: Services | None = None) -> dict[str, Any]:
    """One research step. A live job is reconciled first and never double-started."""
    services = services or {}
    state = load_state(camp)
    rebuild_campaign_projection(camp, state)
    if state.get("status") in _TERMINAL:
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
        decision = decide(obs, backend)
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
        "decision_id": f"d{len(state.get('decisions') or []) + 1}",
        "action": decision.get("action"),
        "ok": decision.get("ok"),
        "reason_zh": decision.get("reason_zh"),
        "evidence_ids": [item for item in cited if isinstance(item, str)] if isinstance(cited, list) else [],
        "detail": decision.get("detail"),
        "evidence_count": evidence_count(state),
        "executed": bool(decision.get("ok")) is False,
    }
    state.setdefault("decisions", []).append(record)
    _write(camp / "decisions" / f"{record['decision_id']}.json", {"decision": record, "raw": decision.get("raw")})
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
    task = begin_role_task(camp, role="research_librarian", inputs=inputs)
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
                payload["summary_zh"] = str(reply.get("summary_zh") or payload["summary_zh"])
                payload["payload"] = reply
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

    missing: list[str] = []
    diagnostics = None
    for row in reversed(state.get("evidence") or []):
        if row.get("diagnostics") or row.get("diagnostic_ref"):
            diagnostics = {"summary": row.get("diagnostics"), "ref": row.get("diagnostic_ref"), "evidence_id": row.get("evidence_id")}
            break
    if diagnostics is None:
        missing.append("diagnostics")
    methods = state.get("method_hits") or []
    if not methods:
        missing.append("method_evidence")
    contract_path = camp / "evaluation_contract.json"
    contract = _read(contract_path) if contract_path.is_file() else {}
    if not contract:
        missing.append("evaluation_contract")
    return {
        "hypothesis": spec.get("hypothesis") or state.get("hypothesis"),
        "diagnostics": diagnostics,
        "method_evidence": methods,
        "parent_candidate_id": spec.get("parent_candidate_id"),
        "control_candidate_id": spec.get("control_candidate_id") or "baseline",
        "evaluation_contract": {
            "fingerprint": contract.get("fingerprint"),
            "research_scope": contract.get("research_scope"),
            "primary_metric": contract.get("primary_metric"),
        },
        "capabilities": capability_manifest(),
        "budget": budget_snapshot(camp, state),
        "missing_inputs": missing,
    }


def _design_experiment(camp: Path, state: dict[str, Any], decision: dict[str, Any], raw: dict[str, Any], services: Services) -> None:
    from react_agent.eeg_research.agentic.experiment_gate import approve_experiment, experiment_is_approved
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task

    spec = raw.get("experiment_draft") or state.get("experiment") or {"initial_fidelity": "pilot"}
    if not isinstance(spec, dict):
        spec = {"initial_fidelity": "pilot"}
    spec = dict(spec)
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
        request={
            "draft": spec,
            "diagnostics": context.get("diagnostics"),
            "method_evidence": context.get("method_evidence"),
            "budget": context.get("budget"),
            "capabilities": context.get("capabilities"),
            "parent_candidate_id": context.get("parent_candidate_id"),
            "control_candidate_id": context.get("control_candidate_id"),
            "evaluation_contract": context.get("evaluation_contract"),
        },
    )
    payload: dict[str, Any] = {"status": "completed", "experiment_spec": spec, "summary_zh": "已写出 ExperimentSpec"}
    blocked = False
    design_failed = False
    if designer is not None:
        try:
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

    report_hash = hashlib.sha256(json.dumps({"claims": claims, "latest": latest}, default=str, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    payload = {
        "status": "completed",
        "verdict": verdict,
        "deterministic_verdict": verdict,
        "claims": claims,
        "report_hash": report_hash,
        "summary_zh": "审计基于已落盘的 comparison/diagnostics，未训练模型",
    }
    task = begin_role_task(
        camp,
        role="result_auditor",
        inputs=[camp / "goal.json"],
        request={"claims": claims, "report_hash": report_hash, "latest": latest},
    )
    auditor = services.get("auditor")
    if auditor is not None:
        try:
            reply = auditor(
                {
                    "task_id": task["task_id"],
                    "attempt_id": task["attempt_id"],
                    "input_digest": task["input_digest"],
                    "latest": latest,
                    "draft": payload,
                    "claims": claims,
                    "report_hash": report_hash,
                }
            )
            if isinstance(reply, dict):
                model_verdict = reply.get("verdict") or (reply.get("payload") or {}).get("verdict")
                payload["auditor_claims"] = reply.get("claims") or (reply.get("payload") or {}).get("claims") or []
                payload["open_issues"] = reply.get("open_issues") or []
                payload["required_corrections"] = reply.get("required_corrections") or []
                payload["summary_zh"] = str(reply.get("summary_zh") or payload["summary_zh"])
                if verdict in {"REVISE", "BLOCK"} and model_verdict == "PASS":
                    payload["verdict"] = verdict
                    payload["model_verdict_ignored"] = "PASS"
                elif model_verdict in {"REVISE", "BLOCK", "PASS"}:
                    payload["verdict"] = model_verdict
        except LlmUnavailable:
            payload["status"] = "partial"
            payload["summary_zh"] = "审计角色不可用，保留确定性核查"
    path = camp / "audits" / f"{task['task_id']}.json"
    finish_role_task(camp, task, payload, kind="audit", path=path)
    state["audit_status"] = {"PASS": "pass", "REVISE": "revise", "BLOCK": "block"}.get(str(payload["verdict"]), "unavailable")
    state["audit_report_hash"] = report_hash
    state["audit_claims"] = claims
    state["audit_fresh"] = True
    _append_derived(
        state,
        {
            "evidence_id": f"ev_audit_result_{len(state['evidence']) + 1}",
            "kind": "audit",
            "summary": {
                "verdict": payload["verdict"],
                "claims": claims,
                "auditor_claims": payload.get("auditor_claims") or [],
                "report_hash": report_hash,
                "model_verdict_ignored": payload.get("model_verdict_ignored"),
            },
        },
    )
    state["audited_report_hash"] = report_dependency_hash(state)


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

    candidate_id = f"c{len(state.get('candidates') or []) + 1}"
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
