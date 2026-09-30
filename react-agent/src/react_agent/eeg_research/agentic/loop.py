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
    PlanError,
    apply_update,
    consume_for_planner,
    init_plan,
    mark_consumed,
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
    _write(camp / "goal.json", goal)
    _write(camp / "resolved_goal.json", goal)
    _write(camp / "evaluation_contract.json", contract)
    if protocol is not None:
        _write(camp / "execution_protocol.json", protocol)
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
    return {
        "goal": _planner_goal(_read(camp / "goal.json")),
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
    record = {
        "decision_id": f"d{len(state.get('decisions') or []) + 1}",
        "action": decision.get("action"),
        "ok": decision.get("ok"),
        "reason_zh": decision.get("reason_zh"),
        "evidence_ids": (decision.get("raw") or {}).get("evidence_ids") or [],
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
            "phase": "planner",
            "recoverable": True,
            "error_type": str(decision.get("detail") or "schema"),
        }
        save_state(camp, state)
        return state
    raw_decision = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
    raw_update = raw_decision.get("plan_update")
    update_failed = False
    if isinstance(raw_update, dict):
        known = {str(row.get("evidence_id")) for row in state.get("evidence") or [] if row.get("evidence_id")}
        try:
            apply_update(camp, raw_update, known_evidence_ids=known)
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
    dependent = {"design_experiment", "propose_experiment", "implement_candidate", "run_pilot", "run_full", "replicate", "stop"}
    if update_failed and depends and str(decision.get("action") or "") in dependent:
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
        from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved

        repairing = bool(state.get("repair_task") and int((state.get("repair_task") or {}).get("remaining") or 0) > 0)
        candidate_root = camp / "candidates"
        already_started = candidate_root.is_dir() and any(candidate_root.iterdir())
        spec = state.get("experiment") if isinstance(state.get("experiment"), dict) else None
        if not repairing and not already_started and not experiment_is_approved(spec):
            persist_failure(
                camp,
                state,
                phase="implement_candidate",
                error_type="experiment_not_approved",
                detail="experiment_not_approved",
                recoverable=True,
            )
            state["experiment_failed"] = True
            return
        if not repairing and spec and spec.get("spec_hash"):
            from react_agent.eeg_research.agentic.experiment_gate import _spec_hash

            if _spec_hash(spec) != spec.get("spec_hash"):
                persist_failure(
                    camp,
                    state,
                    phase="implement_candidate",
                    error_type="approved_spec_changed",
                    detail="approved_spec_changed",
                    recoverable=True,
                )
                return
        ref = state.get("experiment_ref")
        if not repairing and ref:
            from react_agent.eeg_research.agentic.artifacts import verify as verify_artifact

            ok, reason = verify_artifact(camp, str(ref))
            if not ok:
                persist_failure(
                    camp,
                    state,
                    phase="implement_candidate",
                    error_type=reason,
                    detail=reason,
                    recoverable=True,
                )
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
        state["experiment"] = approve_experiment(draft)
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
        request={"draft": spec, "diagnostics": context.get("diagnostics"), "method_evidence": context.get("method_evidence")},
    )
    payload: dict[str, Any] = {"status": "completed", "experiment_spec": spec, "summary_zh": "已写出 ExperimentSpec"}
    blocked = False
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
                if status in {"requires_framework_extension", "blocked"} or reply.get("requires_framework_extension"):
                    blocked = True
                    spec["status"] = "requires_framework_extension"
                    spec["missing_capability"] = reply.get("required_capability_ids") or reply.get("missing_inputs")
            if isinstance(spec_reply, dict):
                spec = {**spec, **spec_reply}
        except LlmUnavailable:
            payload["status"] = "partial"
            payload["summary_zh"] = "设计角色不可用，使用 planner 草稿"
    if not blocked:
        spec = approve_experiment(spec)
        blocked = not experiment_is_approved(spec)
    else:
        spec["status"] = "blocked"
        spec["blocked_reason"] = "requires_framework_extension"
    payload["experiment_spec"] = spec
    payload["status"] = "blocked" if blocked else payload["status"]
    path = camp / "experiments" / f"spec_{task['task_id']}.json"
    envelope = finish_role_task(camp, task, payload, kind="experiment_spec", path=path)
    stored = envelope.get("experiment_spec")
    if not isinstance(stored, dict):
        inner = envelope.get("payload")
        stored = inner.get("experiment_spec") if isinstance(inner, dict) else spec
    state["experiment"] = stored
    state["experiment_ref"] = (envelope.get("artifact_refs") or [None])[-1]
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
    state["audited_report_hash"] = report_hash
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


def _record_job(camp: Path, state: dict[str, Any], record: dict[str, Any]) -> None:
    job_id = str(record.get("job_id") or "")
    if job_id and any(row.get("job_id") == job_id or row.get("evidence_id") == f"ev_{job_id}" for row in state.get("evidence") or []):
        return
    result = dict(record.get("result") or {})
    result["evidence_id"] = f"ev_{job_id or record.get('candidate_id')}"
    result["job_id"] = job_id
    result["candidate_id"] = record.get("candidate_id")
    result["job_dir"] = str(camp / "jobs" / job_id)
    result["seed"] = record.get("seed")
    result["training_seed"] = record.get("seed")
    result["job_status"] = record.get("status")
    result["source_hash"] = record.get("source_hash") or (record.get("manifest") or {}).get("entry_sha256")
    if (Path(result["job_dir"]) / "last.ckpt").is_file():
        from react_agent.eeg_research.agentic.artifacts import file_digest

        result["checkpoint_id"] = file_digest(Path(result["job_dir"]) / "last.ckpt")
    hook_path = Path(result["job_dir"]) / "hook_config.json"
    if hook_path.is_file():
        hook = _read(hook_path)
        result["config_hash"] = hook.get("config_hash")
    if result.get("evaluation_valid"):
        result["contract_fingerprint"] = result.get("execution_fingerprint") or state.get("contract_fingerprint")
    charge_gpu(camp, state, float(record.get("gpu_seconds") or 0), job_id=job_id)
    if state.get("candidate_id") == result.get("candidate_id"):
        result["hypothesis"] = state.get("hypothesis")
        result["experiment"] = state.get("experiment")
        result["experiment_ref"] = state.get("experiment_ref")
        result["spec_hash"] = (state.get("experiment") or {}).get("spec_hash") if isinstance(state.get("experiment"), dict) else None
    cost_path = camp / "cost.json"
    cost = _read(cost_path)
    cost["training_jobs"] = int(state.get("training_jobs", 0))
    cost.setdefault("api_usd", None)
    _write(cost_path, cost)
    store = EpisodeStore(camp)
    ep = episode(
        task_hash=str(state.get("goal_id")),
        candidate_id=str(result.get("candidate_id")),
        kind="exploratory_result" if result.get("evaluation_valid") else "implementation_failure",
        fidelity=str(result.get("fidelity")),
        seed=int(record.get("seed") or 0),
        metric=result.get("fixed_bank_top1") if result.get("evaluation_valid") else None,
        contract_fingerprint=str(state.get("contract_fingerprint")),
        artifact=str(Path(result["job_dir"]) / "metrics.json") if result.get("evaluation_valid") else str(result["job_dir"]),
        job_id=job_id,
    )
    stored = store.persist_episode(ep)
    if not any(item.get("episode_id") == stored.get("episode_id") or item.get("job_id") == job_id for item in state.get("memory") or []):
        state.setdefault("memory", []).append(stored)
    state.setdefault("evidence", []).append(result)
    if state.get("audit_status") in {"pass", "revise", "block"}:
        state["audit_status"] = "stale"
        state["audit_fresh"] = False
    protocol = load_protocol(camp)
    control = matched_control(state["evidence"][:-1], result, protocol)
    comparison = compare_runs(candidate=result, control=control, protocol=protocol)
    result["comparison_status"] = "comparable" if comparison.get("comparable") else comparison.get("reason")
    result["comparison"] = comparison
    if comparison.get("comparable") and comparison.get("delta_pp") is not None:
        result["delta_vs_control_pp"] = comparison["delta_pp"]
        result["control_id"] = comparison.get("control_run_id")
    else:
        result.pop("delta_vs_control_pp", None)
        result.pop("control_id", None)
    from react_agent.eeg_research.agentic.diagnostics import write_job_bundle

    diagnostics = write_job_bundle(Path(result["job_dir"]))
    result["diagnostics"] = diagnostics
    result["diagnostic_ref"] = str(Path(result["job_dir"]) / "diagnostic_summary.json")
    goal = _read(camp / "goal.json")
    if comparison.get("comparable") and comparison.get("delta_pp") is not None:
        result["pair_record"] = _pair_record(result, comparison)
    paired_records = [
        row.get("pair_record")
        for row in state["evidence"]
        if row.get("candidate_id") == result.get("candidate_id")
        and isinstance(row.get("pair_record"), dict)
        and row.get("evaluation_valid")
    ]
    promo = promotion_decision(
        comparison=comparison,
        goal=goal,
        paired_records=paired_records,
        fidelity=str(result.get("fidelity") or ""),
    )
    result["promotion"] = promo
    _write(camp / "comparisons" / f"{result['evidence_id']}.json", {"comparison": comparison, "promotion": promo, "diagnostics": diagnostics})
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
    protocol = load_protocol(camp) or {}
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

    write_hook_config(job, state.get("experiment") if isinstance(state.get("experiment"), dict) else {}, spec_ref=state.get("experiment_ref"))
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
