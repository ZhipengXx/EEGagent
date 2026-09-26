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
from react_agent.eeg_research.agentic.execution_protocol import resolve_worker_design
from react_agent.eeg_research.agentic.identity import ensure_attempt, operation_id, result_matches, source_hash
from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
from react_agent.eeg_research.agentic.loop import (
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
        return jobs.start_job(
            camp_dir / "jobs" / job_id,
            design,
            candidate_id=candidate_id,
            extension=extension,
            fidelity=fidelity,
            root=data_root(),
            protocol_path=camp_dir / "execution_protocol.json",
        )

    def settle(camp_dir: Path, job_id: str) -> dict[str, Any]:
        return jobs.reconcile(camp_dir / "jobs" / job_id)

    def do_implement(camp_dir: Path, state: dict[str, Any]) -> None:
        if state.get("status") in {"paused", "cancelled"}:
            return
        candidate_id = incomplete_candidate_id(camp_dir, state)
        if candidate_id is None:
            index = len(state.get("candidates") or []) + 1
            while (camp_dir / "candidates" / f"c{index}").exists():
                index += 1
            candidate_id = f"c{index}"
        workspace = camp_dir / "candidates" / candidate_id
        spec = {"hypothesis": state.get("hypothesis"), "experiment": state.get("experiment")}
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_id = f"ev_impl_{candidate_id}"
        if any(row.get("evidence_id") == evidence_id for row in state.get("evidence") or []):
            return
        max_calls = int(state.get("max_llm_calls", 100))

        def calls_left() -> int:
            cost_path = camp_dir / "cost.json"
            used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
            return max_calls - int(used)

        impl_path = workspace / "implementation.json"
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
        if impl_path.is_file():
            outcome = json.loads(impl_path.read_text(encoding="utf-8"))
        else:
            last_llm: LlmUnavailable | None = None
            for _attempt in range(IMPLEMENT_ATTEMPTS):
                try:
                    outcome = implement(workspace, spec, coder, max_steps=CODER_STEPS, calls_left=calls_left, reserve=2)
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
                row["status"] = "ready"
            else:
                row["status"] = f"review_{verdict['status']}"
        if row["status"] != "ready":
            state["experiment_failed"] = True
        state.setdefault("candidates", []).append(row)
        state["evidence"].append(
            {
                "evidence_id": evidence_id,
                "kind": "implementation",
                "candidate_id": candidate_id,
                "summary": row,
            }
        )
        event(camp_dir, "implemented", **row)

    def analyze(camp_dir: Path, state: dict[str, Any]) -> None:
        latest = state["evidence"][-1]
        if not latest.get("evaluation_valid") or latest.get("candidate_id") == "baseline":
            return
        payload = {
            "latest": {key: latest.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "gallery_size", "delta_vs_control_pp", "control_id", "seed")},
            "hypothesis": state.get("hypothesis"),
            "controls": [
                {key: row.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "seed")}
                for row in state["evidence"]
                if row.get("candidate_id") == "baseline" and row.get("evaluation_valid")
            ],
        }
        try:
            reply = analyst(payload)
        except LlmUnavailable as exc:
            reply = {"hypothesis_assessment": "not_tested", "summary_zh": f"分析未完成：{exc}"}
        state["llm_calls"] = int(state.get("llm_calls", 0)) + 1
        state["llm_calls_left"] = int(state.get("llm_calls_left", 0)) - 1
        target = camp_dir / "analyses" / f"{latest['evidence_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reply, ensure_ascii=False, indent=2), encoding="utf-8")
        state["evidence"].append(
            {
                "evidence_id": f"ev_analysis_{latest['evidence_id']}",
                "kind": "analysis",
                "candidate_id": latest.get("candidate_id"),
                "summary": {key: reply.get(key) for key in ("hypothesis_assessment", "summary_zh", "suggested_next_actions")},
            }
        )

    return {"launch": launch, "settle": settle, "implement": do_implement, "analyze": analyze}


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
    for _ in range(max_ticks):
        state = tick(camp, planner, services=services)
        (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": time.time()}), encoding="utf-8")
        if state.get("status") in _TERMINAL:
            break
        if state.get("status") == "training":
            time.sleep(poll_seconds)
    return state
