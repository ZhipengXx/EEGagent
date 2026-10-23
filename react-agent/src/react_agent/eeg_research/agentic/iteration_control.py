"""Finish a requested number of native research iterations after memory capture."""
from __future__ import annotations

from pathlib import Path


def maybe_complete_research(camp: Path, state: dict) -> bool:
    from .loop import _read, _write, event, save_state
    from .research_execution import research_goal

    goal = research_goal(camp)
    limit = (goal.get("research_execution") or {}).get("max_completed_rounds")
    if limit is None or state.get("live_job") or state.get("active_suite") or state.get("queued"):
        return False
    pointer = state.get("selected_plan") or {}
    if not pointer.get("terminal_observation_id"):
        return False
    # Handoff must finish under the original plan before its unused continuation
    # can be released. The ordinary executor recovers that boundary first.
    if not pointer.get("continuation_owner"):
        return False
    plans = [_read(path) for path in sorted((camp / "selected_plans").glob("plan_*.json"))]
    plans = [plan for plan in plans if plan.get("terminal_observation_id")
             and plan.get("status") in {"observation_ready", "failed", "needs_replan"}]
    if len(plans) < limit:
        return False
    from .experiment_observations import ExperimentObservationStore

    store = ExperimentObservationStore(camp)
    completed = []
    observations = set()
    for plan in plans:
        records = store.records(observation_id=plan["terminal_observation_id"], limit=1)
        if not records:
            raise ValueError("research_iteration_terminal_observation_unreadable")
        card = records[0]["body"]
        if (card["plan_id"] != plan["plan_id"]
                or card["content_digest"] != plan["terminal_observation_digest"]
                or card["execution_status"] != plan["status"]
                or card["observation_id"] in observations):
            raise ValueError("research_iteration_terminal_identity_changed")
        observations.add(card["observation_id"])
        completed.append({"plan_id": plan["plan_id"], "round_id": plan["originating_round"],
            "proposal_id": plan["originating_proposal"], "task_kind": plan["proposal"]["task_kind"],
            "candidate_id": plan.get("candidate_id"), "repair_count": plan["repair_count"],
            "status": plan["status"], "observation_id": card["observation_id"],
            "observation_digest": card["content_digest"], "recorded_at": records[0]["navigation"]["created_at"]})
    completed.sort(key=lambda row: (row["recorded_at"], row["plan_id"]))
    indexing = None
    if (goal.get("memory") or {}).get("enabled"):
        from .embedding import EmbeddingUnavailable
        from .semantic_memory import embedding_health, planner_query, retrieve
        # Index the final feedback even though no eleventh Planner request is
        # created. A real similarity failure remains a recoverable failure.
        try:
            memory = retrieve(camp, query=planner_query(camp, state), state=state,
                              role="research_planner", require_embedding=True)
        except EmbeddingUnavailable as exc:
            raise ValueError("research_iteration_final_memory_unavailable:" + str(exc)) from exc
        if memory.get("degraded"):
            raise ValueError("research_iteration_final_memory_unavailable:" + str(memory.get("reason")))
        indexing = embedding_health(memory)
    from .iteration_continuation import completion_path
    path = completion_path(camp)
    existing = _read(path)
    if existing:
        if existing.get("iterations") != completed or existing.get("requested_rounds") != limit:
            raise ValueError("research_iteration_completion_changed")
    else:
        _write(path, {"schema_version": "eeg_research.iteration_completion.v1",
            "requested_rounds": limit, "completed_rounds": len(completed),
            "iterations": completed, "final_memory": indexing,
            "stop_reason": "requested_iterations_completed"}, exclusive=True)
    from .owner_budget import finish_planner_round
    finish_planner_round(camp, pointer["continuation_owner"])
    state.update(status="finished", stop_reason="requested_iterations_completed",
                 completed_research_rounds=len(completed))
    state["research_completion_ref"] = str(path.relative_to(camp))
    state.pop("failure", None)
    state.pop("detail", None)
    if not existing:
        event(camp, "research_iterations_completed", completed_rounds=len(completed), requested_rounds=limit)
    save_state(camp, state)
    return True
