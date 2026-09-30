"""Background worker. One process per campaign; training end triggers the next decision."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic import jobs
from react_agent.eeg_research.agentic.contract import public_contract
from react_agent.eeg_research.agentic.execution_protocol import load_protocol, resolve_worker_design
from react_agent.eeg_research.agentic.identity import ensure_attempt, operation_id, result_matches, rotate_attempt, source_hash
from react_agent.eeg_research.agentic.interface import candidate_interface
from react_agent.eeg_research.agentic.lineage import LineageError, materialize
from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
from react_agent.eeg_research.agentic.loop import (
    _sync_ledger,
    align_interrupt,
    event,
    incomplete_candidate_id,
    load_state,
    persist_failure,
    save_state,
    tick,
)
from react_agent.eeg_research.agentic.native_patch import RecoveryBlocked, implement, review
from react_agent.eeg_training.protocol import data_root

_TERMINAL = {"paused", "finished", "blocked", "cancelled"}
CODER_STEPS = 8
IMPLEMENT_ATTEMPTS = 3


def _lesson_proposal(reply: Any) -> dict[str, Any]:
    """Read lessons from a RoleResult envelope or a bare curator object."""
    if not isinstance(reply, dict):
        return {}
    inner = reply.get("payload")
    if isinstance(inner, dict) and ("proposed_lessons" in inner or "summary_zh" in inner):
        return inner
    return reply


def worker_lock_held(camp: Path) -> bool:
    """True when another worker still owns the campaign lock."""
    lock_path = camp / "worker.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _block_phase(
    camp_dir: Path,
    state: dict[str, Any],
    *,
    phase: str,
    error_type: str,
    detail: str,
    recoverable: bool,
) -> None:
    persist_failure(
        camp_dir,
        state,
        phase=phase,
        error_type=error_type,
        detail=detail,
        recoverable=recoverable,
    )


def _attach(backend: Any, **fields: Any) -> None:
    bind = getattr(backend, "bind", None)
    if callable(bind):
        bind(**fields)
    try:
        backend.identity = dict(fields)
    except Exception:
        return


def _stored_review(workspace: Path, *, candidate_id: str, attempt_id: str, input_hash: str) -> dict[str, Any] | None:
    """Return a review only when it belongs to this attempt and this source hash."""
    path = workspace / "review.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecoveryBlocked("review_result_unreadable") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {"ready", "needs_fix", "blocked"}:
        raise RecoveryBlocked("review_result_unreadable")
    if not result_matches(
        payload,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        phase="review_candidate",
        input_hash=input_hash,
    ):
        return None
    return payload


def _review_or_reuse(
    camp_dir: Path,
    workspace: Path,
    spec: dict[str, Any],
    summary: dict[str, Any],
    reviewer: Any,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Use a stored review, or call the reviewer once. None means the campaign is already blocked."""
    attempt = ensure_attempt(workspace, str(workspace.name))
    digest = source_hash(workspace)
    try:
        stored = _stored_review(
            workspace,
            candidate_id=str(workspace.name),
            attempt_id=str(attempt["attempt_id"]),
            input_hash=digest,
        )
    except RecoveryBlocked as exc:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type="RecoveryBlocked",
            detail=str(exc),
            recoverable=False,
        )
        return None
    if stored is not None:
        return stored
    cost_path = camp_dir / "cost.json"
    used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
    if int(state.get("max_llm_calls", 100)) - int(used) <= 0:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type="budget_exhausted",
            detail="budget_exhausted",
            recoverable=True,
        )
        return None
    identity = {
        "candidate_id": workspace.name,
        "attempt_id": attempt["attempt_id"],
        "phase": "review_candidate",
        "operation_id": operation_id(workspace, workspace.name, "review_candidate"),
        "input_hash": digest,
    }
    _attach(reviewer, **identity)
    try:
        return review(workspace, spec, summary, reviewer, getattr(reviewer, "model", ""), identity=identity)
    except LlmUnavailable as exc:
        _block_phase(camp_dir, state, phase="review_candidate", error_type=str(exc), detail=str(exc), recoverable=True)
        return None
    except Exception as exc:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type=type(exc).__name__,
            detail=f"{type(exc).__name__}: {exc}",
            recoverable=False,
        )
        return None


def build_services(camp: Path) -> dict[str, Any]:
    coder = role_backend(camp, "candidate_coder")
    reviewer = role_backend(camp, "candidate_reviewer")
    analyst = role_backend(camp, "result_analyst")
    contract = public_contract(json.loads((camp / "evaluation_contract.json").read_text(encoding="utf-8")))
    summary = {key: contract.get(key) for key in ("research_scope", "primary_metric", "val_mode", "fingerprint")}

    def launch(camp_dir: Path, state: dict[str, Any], job_id: str, candidate_id: str, fidelity: str) -> dict[str, Any]:
        design = resolve_worker_design(camp_dir, state)
        if design is None:
            save_state(camp_dir, state)
            return {"job_id": job_id, "status": "blocked", "detail": state.get("detail")}
        extension = None if candidate_id == "baseline" else camp_dir / "candidates" / candidate_id / "extension"
        root = Path(design.data_root) if design.data_root else data_root()
        return jobs.start_job(
            camp_dir / "jobs" / job_id,
            design,
            candidate_id=candidate_id,
            extension=extension,
            fidelity=fidelity,
            root=root,
            protocol_path=camp_dir / "execution_protocol.json",
        )

    def settle(camp_dir: Path, job_id: str) -> dict[str, Any]:
        return jobs.reconcile(camp_dir / "jobs" / job_id)

    def do_implement(camp_dir: Path, state: dict[str, Any]) -> None:
        if state.get("status") in {"paused", "cancelled"}:
            return
        repair = state.get("repair_task") if isinstance(state.get("repair_task"), dict) else None
        repairing = bool(repair and int(repair.get("remaining") or 0) > 0 and repair.get("candidate_id"))
        candidate_id = None
        if repairing:
            candidate_id = str(repair["candidate_id"])
        if candidate_id is None:
            candidate_id = incomplete_candidate_id(camp_dir, state)
        if candidate_id is None:
            index = len(state.get("candidates") or []) + 1
            while (camp_dir / "candidates" / f"c{index}").exists():
                index += 1
            candidate_id = f"c{index}"
        workspace = camp_dir / "candidates" / candidate_id
        spec = {"hypothesis": state.get("hypothesis"), "experiment": state.get("experiment")}
        if repairing:
            spec["repair_issues"] = repair.get("issues") or []
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        protocol = load_protocol(camp_dir)
        (workspace / "input_spec.json").write_text(
            json.dumps(candidate_interface(protocol), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            materialize(camp_dir, workspace, state.get("experiment") or {})
        except LineageError as exc:
            _block_phase(
                camp_dir,
                state,
                phase="implement_candidate",
                error_type="LineageError",
                detail=str(exc),
                recoverable=False,
            )
            return
        evidence_id = f"ev_impl_{candidate_id}"
        if not repairing and any(row.get("evidence_id") == evidence_id for row in state.get("evidence") or []):
            return
        max_calls = int(state.get("max_llm_calls", 100))

        def calls_left() -> int:
            cost_path = camp_dir / "cost.json"
            used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
            return max_calls - int(used)

        impl_path = workspace / "implementation.json"
        if repairing:
            attempt = rotate_attempt(workspace, candidate_id)
        else:
            attempt = ensure_attempt(workspace, candidate_id)
        _attach(
            coder,
            candidate_id=candidate_id,
            attempt_id=attempt["attempt_id"],
            phase="implement_candidate",
            operation_id=operation_id(workspace, candidate_id, "implement_candidate"),
            input_hash=source_hash(workspace),
        )
        outcome: dict[str, Any] | None = None
        if impl_path.is_file() and not repairing:
            outcome = json.loads(impl_path.read_text(encoding="utf-8"))
        else:
            last_llm: LlmUnavailable | None = None
            for _attempt in range(IMPLEMENT_ATTEMPTS):
                try:
                    outcome = implement(
                        workspace,
                        spec,
                        coder,
                        max_steps=CODER_STEPS,
                        calls_left=calls_left,
                        reserve=2,
                        resume_repair=repairing,
                    )
                    break
                except RecoveryBlocked as exc:
                    _block_phase(
                        camp_dir,
                        state,
                        phase="implement_candidate",
                        error_type="RecoveryBlocked",
                        detail=str(exc),
                        recoverable=False,
                    )
                    return
                except LlmUnavailable as exc:
                    last_llm = exc
                except Exception as exc:
                    _block_phase(
                        camp_dir,
                        state,
                        phase="implement_candidate",
                        error_type=type(exc).__name__,
                        detail=f"{type(exc).__name__}: {exc}",
                        recoverable=False,
                    )
                    return
            if outcome is None:
                error_type = str(last_llm) if last_llm is not None else "LlmUnavailable"
                _block_phase(camp_dir, state, phase="implement_candidate", error_type=error_type, detail=error_type, recoverable=True)
                return
            if outcome.get("status") == "ready_for_review":
                attempt = ensure_attempt(workspace, candidate_id)
                impl_path.write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "attempt_id": attempt["attempt_id"],
                            "phase": "implement_candidate",
                            "operation_id": operation_id(workspace, candidate_id, "implement_candidate"),
                            "input_hash": source_hash(workspace),
                            "status": outcome["status"],
                            "steps": outcome.get("steps"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        row = {"candidate_id": candidate_id, "status": outcome["status"], "steps": outcome.get("steps")}
        if outcome["status"] == "ready_for_review":
            verdict = _review_or_reuse(camp_dir, workspace, spec, summary, reviewer, state)
            if verdict is None:
                return
            row["review"] = verdict["status"]
            row["review_format_failed"] = verdict.get("format_failed")
            if verdict["status"] == "ready":
                state["candidate_ready"] = True
                state["candidate_id"] = candidate_id
                state["repair_task"] = None
                row["status"] = "ready"
                if isinstance(state.get("experiment"), dict):
                    from react_agent.eeg_research.agentic.run_context import persist_approved_binding

                    persist_approved_binding(camp_dir, candidate_id, state["experiment"], spec_ref=state.get("experiment_ref"))
            elif verdict["status"] == "needs_fix":
                used = int((repair or {}).get("used") or 0)
                limit = int(state.get("max_repairs_per_candidate", 2))
                if used < limit:
                    state["repair_task"] = {
                        "candidate_id": candidate_id,
                        "issues": verdict.get("issues") or [],
                        "remaining": 1,
                        "used": used + 1,
                    }
                else:
                    state["repair_task"] = None
                    state["experiment_failed"] = True
                row["status"] = "review_needs_fix"
            else:
                state["repair_task"] = None
                row["status"] = f"review_{verdict['status']}"
        if row["status"] != "ready":
            repairing_now = bool(
                isinstance(state.get("repair_task"), dict)
                and int((state.get("repair_task") or {}).get("remaining") or 0) > 0
            )
            if not repairing_now:
                state["experiment_failed"] = True
        existing = next(
            (index for index, item in enumerate(state.get("candidates") or []) if item.get("candidate_id") == candidate_id),
            None,
        )
        if existing is None:
            state.setdefault("candidates", []).append(row)
        else:
            state["candidates"][existing] = row
        evidence_row = {
            "evidence_id": evidence_id,
            "kind": "implementation",
            "candidate_id": candidate_id,
            "summary": row,
        }
        replaced = False
        for index, item in enumerate(state.get("evidence") or []):
            if item.get("evidence_id") == evidence_id:
                state["evidence"][index] = evidence_row
                replaced = True
                break
        if not replaced:
            state["evidence"].append(evidence_row)
        event(camp_dir, "implemented", **row)

    def analyze(camp_dir: Path, state: dict[str, Any]) -> None:
        latest = state["evidence"][-1]
        comparison = latest.get("comparison")
        diagnostics = latest.get("diagnostics")
        if not latest.get("evaluation_valid"):
            return
        bound_hypothesis = latest.get("hypothesis")
        bound_experiment = latest.get("experiment")
        if bound_hypothesis is None:
            for row in state.get("candidates") or []:
                if row.get("candidate_id") == latest.get("candidate_id") and row.get("hypothesis"):
                    bound_hypothesis = row.get("hypothesis")
                    bound_experiment = row.get("experiment")
        payload = {
            "latest": {key: latest.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "gallery_size", "delta_vs_control_pp", "control_id", "seed")},
            "comparison": comparison,
            "diagnostics": diagnostics,
            "hypothesis": bound_hypothesis,
            "experiment": bound_experiment,
            "hypothesis_binding_missing": bound_hypothesis is None,
            "controls": [
                {key: row.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "seed")}
                for row in state["evidence"]
                if row.get("candidate_id") == "baseline" and row.get("evaluation_valid")
            ],
        }
        billed = 0
        try:
            reply = analyst(payload)
            billed += 1
        except LlmUnavailable as exc:
            billed += 1
            reply = {"status": "failed", "role_failed": True, "hypothesis_assessment": None, "summary_zh": f"分析未完成：{exc}"}
        target = camp_dir / "analyses" / f"{latest['evidence_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"reply": reply, "comparison": comparison, "diagnostics": diagnostics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state["evidence"].append(
            {
                "evidence_id": f"ev_analysis_{latest['evidence_id']}",
                "kind": "analysis",
                "candidate_id": latest.get("candidate_id"),
                "summary": {key: reply.get(key) for key in ("hypothesis_assessment", "summary_zh", "suggested_next_actions")},
            }
        )
        from react_agent.eeg_research.agentic.memory import EpisodeStore
        from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task

        store = EpisodeStore(camp_dir)
        try:
            curator = role_backend(camp, "memory_curator")
            task = begin_role_task(camp_dir, role="memory_curator", inputs=[camp_dir / "goal.json"])
            try:
                proposal = curator(
                    {
                        "episodes": store.list_episodes(),
                        "task_id": task["task_id"],
                        "attempt_id": task["attempt_id"],
                        "input_digest": task["input_digest"],
                    }
                )
                billed += 1
            except LlmUnavailable:
                billed += 1
                store.mark_pending_curation("curator_unavailable")
            else:
                accepted = store.accept_lessons(_lesson_proposal(proposal))
                finish_role_task(
                    camp_dir,
                    task,
                    {"status": "completed", "summary_zh": "已提议条件化经验", "accepted": accepted},
                    kind="lessons",
                    path=camp_dir / "memory" / f"lessons_{task['task_id']}.json",
                )
        except LlmUnavailable:
            store.mark_pending_curation("curator_unavailable")
        except Exception:
            store.mark_pending_curation("curator_failed")
        _sync_ledger(camp_dir, state)
        if not (camp_dir / "cost.json").is_file():
            state["llm_calls"] = int(state.get("llm_calls", 0)) + billed
            state["llm_calls_left"] = int(state.get("llm_calls_left", 0)) - billed

    def _lazy(role: str):
        def call(payload: dict[str, Any]) -> dict[str, Any]:
            return role_backend(camp, role)(payload)

        return call

    return {
        "launch": launch,
        "settle": settle,
        "implement": do_implement,
        "analyze": analyze,
        "librarian": _lazy("research_librarian"),
        "designer": _lazy("experiment_designer"),
        "auditor": _lazy("result_auditor"),
    }


def run_worker(camp: Path, *, poll_seconds: float = 30.0, max_ticks: int = 200) -> dict[str, Any]:
    """Advance until a terminal state. A second worker for the same campaign exits at once."""
    lock_path = camp / "worker.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"status": "worker_already_running"}
    (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
    state = load_state(camp)
    if resolve_worker_design(camp, state) is None:
        if state.get("status") not in {"paused", "cancelled"}:
            save_state(camp, state)
        return state
    align_interrupt(camp)
    state = load_state(camp)
    if state.get("status") in _TERMINAL:
        return state
    try:
        planner = role_backend(camp, "research_planner")
        services = build_services(camp)
    except LlmUnavailable as exc:
        state = load_state(camp)
        if state.get("status") not in {"paused", "cancelled"}:
            state["status"] = "blocked"
            state["detail"] = str(exc)
            save_state(camp, state)
        return state
    for tick_index in range(max_ticks):
        state = tick(camp, planner, services=services)
        (camp / "worker.json").write_text(
            json.dumps({"pid": os.getpid(), "heartbeat": time.time(), "tick": tick_index, "resumable": bool(state.get("live_job"))}),
            encoding="utf-8",
        )
        if state.get("status") in _TERMINAL:
            break
        if state.get("status") == "training":
            time.sleep(poll_seconds)
    else:
        state = load_state(camp)
        if state.get("live_job"):
            (camp / "worker.json").write_text(
                json.dumps({"pid": os.getpid(), "heartbeat": time.time(), "resumable": True, "detail": "worker_tick_limit"}),
                encoding="utf-8",
            )
            event(camp, "worker_suspended", reason="tick_limit", live_job=state.get("live_job"))
    return state
