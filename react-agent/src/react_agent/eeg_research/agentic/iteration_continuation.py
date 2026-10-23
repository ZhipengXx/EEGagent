"""Explicit continuation of a completed study without rewriting its results."""
from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path


def _receipts(camp: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((Path(camp) / "research_continuations").glob("*.json"))]


def current_continuation(camp: Path) -> dict | None:
    rows = _receipts(camp)
    return rows[-1] if rows else None


def completion_path(camp: Path) -> Path:
    receipt = current_continuation(camp)
    return Path(camp) / (receipt["completion_path"] if receipt else "research_completion.json")


def planning_continuation_owner(camp: Path, observation_digest: str | None,
                                fallback: str | None = None) -> str | None:
    receipt = current_continuation(camp)
    if receipt and receipt["observation_digest"] == observation_digest:
        return receipt["planner_owner_id"]
    return fallback


def apply_continuation_goal(camp: Path, goal: dict) -> dict:
    receipt = current_continuation(camp)
    if not receipt:
        return goal
    return {**goal, "research_execution": {
        **goal.get("research_execution", {}), "max_completed_rounds": receipt["requested_rounds"]},
        "objective": goal.get("objective", "") + (
            f"\nThe current execution target supersedes the original iteration count: "
            f"continue this same study for {receipt['additional_rounds']} additional iterations "
            f"({receipt['requested_rounds']} total), using its existing observations and memory.")}


def apply_continuation_scope(camp: Path, state: dict) -> None:
    receipt = current_continuation(camp)
    if receipt:
        state["research_execution"] = {**state.get("research_execution", {}),
            "max_completed_rounds": receipt["requested_rounds"]}
        state["research_completion_ref"] = receipt["completion_path"]


def _activate(camp: Path, state: dict, receipt: dict) -> None:
    from .loop import event, save_state
    # Retrying an old request may recover its publication, never reset later work.
    if (state.get("status") == "finished"
            and state.get("stop_reason") == "requested_iterations_completed"
            and state.get("completed_research_rounds") == receipt["completed_rounds_before"]):
        state["status"] = "created"
        state.pop("stop_reason", None)
        apply_continuation_scope(camp, state)
        save_state(camp, state)
        event(camp, "research_continued", request_id=receipt["request_id"],
              previous_completed_rounds=receipt["completed_rounds_before"],
              requested_rounds=receipt["requested_rounds"])


def continue_research(camp: Path, *, request_id: str, additional_rounds: int) -> dict:
    """Prepare a bounded continuation; the ordinary start command runs it."""
    from .loop import _read, _write, load_state
    from .owner_budget import allocate_research_continuation, snapshot
    from .research_execution import single_plan_enabled
    from .budget_inheritance import delegated_successor

    camp = Path(camp)
    if not request_id.strip() or type(additional_rounds) is not int or additional_rounds < 1:
        raise ValueError("continuation_requires_request_id_and_positive_additional_rounds")
    with (camp / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("continuation_requires_stopped_worker") from exc
        rows = _receipts(camp)
        existing = next((row for row in rows if row["request_id"] == request_id), None)
        state = load_state(camp)
        if existing:
            if existing["additional_rounds"] != additional_rounds:
                raise ValueError("continuation_request_conflict")
            if existing == rows[-1]:
                _activate(camp, state, existing)
            return existing
        control = _read(camp / "control.json")
        if (not single_plan_enabled(camp) or state.get("status") != "finished"
                or state.get("stop_reason") != "requested_iterations_completed"
                or state.get("pause_after_step") or control.get("action") in {"pause", "stop"}
                or delegated_successor(camp)):
            raise ValueError("continuation_requires_completed_research")
        if any(state.get(key) for key in ("live_job", "active_suite", "queued",
                "pending_candidate_check", "unknown_process", "pending_process_intent")):
            raise ValueError("continuation_requires_quiescent_research")
        source = completion_path(camp)
        completion = _read(source)
        count = state.get("completed_research_rounds")
        pointer = state.get("selected_plan") or {}
        if (type(count) is not int or count < 1 or completion.get("completed_rounds") != count
                or completion.get("requested_rounds") != count
                or len(completion.get("iterations") or []) != count
                or not pointer.get("terminal_observation_digest")
                or not pointer.get("continuation_owner")):
            raise ValueError("continuation_completion_not_reconciled")
        budget = snapshot(camp)
        if any(budget.get(key) for key in ("inflight", "unknown", "native_pending_call_ids")):
            raise ValueError("continuation_requires_settled_api")
        key = hashlib.sha256(request_id.encode()).hexdigest()[:24]
        target = count + additional_rounds
        owner_id = "continuation:extension:" + key
        allocate_research_continuation(camp, owner_id=owner_id,
            observation_digest=pointer["terminal_observation_digest"])
        receipt = {"schema_version": "eeg_research.iteration_continuation.v1",
            "request_id": request_id, "sequence": len(rows) + 1,
            "additional_rounds": additional_rounds, "completed_rounds_before": count,
            "requested_rounds": target, "source_completion_path": str(source.relative_to(camp)),
            "source_completion_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "completion_path": f"research_completions/rounds_{target:06d}_{key}.json",
            "planner_owner_id": owner_id,
            "observation_digest": pointer["terminal_observation_digest"],
            "observation_id": pointer["terminal_observation_id"],
            "plan_id": pointer["plan_id"], "created_at": time.time()}
        _write(camp / "research_continuations" / f"{len(rows) + 1:06d}_{key}.json",
               receipt, exclusive=True)
        _activate(camp, state, receipt)
        return receipt
