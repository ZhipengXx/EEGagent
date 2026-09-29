"""Planner. Availability is computed here, not promised by the prompt."""

from __future__ import annotations

from typing import Any, Callable

ACTIONS = (
    "inspect_data",
    "retrieve_memory",
    "retrieve_methods",
    "diagnose_results",
    "collect_diagnostics",
    "design_experiment",
    "propose_experiment",
    "implement_candidate",
    "repair_candidate",
    "run_pilot",
    "run_full",
    "replicate",
    "audit_result",
    "stop",
)

STOP_REASONS = (
    "goal_addressed",
    "no_supported_next_experiment",
    "no_progress",
    "budget_exhausted",
    "blocked",
    "user_cancelled",
)

Backend = Callable[[dict[str, Any]], dict[str, Any]]


def _has(evidence: list[dict[str, Any]], candidate_id: Any, fidelity: str) -> bool:
    return any(
        row.get("candidate_id") == candidate_id and row.get("fidelity") == fidelity and row.get("evaluation_valid")
        for row in evidence
    )


_LOCAL = {
    "inspect_data",
    "retrieve_memory",
    "retrieve_methods",
    "diagnose_results",
    "collect_diagnostics",
    "design_experiment",
    "propose_experiment",
    "audit_result",
}
IMPLEMENT_CALLS = 8 + 2 + 1
"""Coder steps, reviewer with one repair, and the planner call that asks for the pilot."""


_DERIVED = {"data_audit", "learning_profile"}


def submitted_candidates(state: dict[str, Any]) -> int:
    """Scientific candidates that produced a patch. Baseline is counted separately."""
    unwritten = {"implementation_failed", "requires_framework_extension"}
    return sum(
        1
        for row in state.get("candidates") or []
        if row.get("status") not in unwritten and row.get("candidate_id") != "baseline"
    )


def evidence_count(state: dict[str, Any]) -> int:
    """Rows that can change a decision. Re-derived summaries do not count."""
    return sum(1 for row in state.get("evidence") or [] if row.get("kind") not in _DERIVED)


def _seen_without_new_evidence(state: dict[str, Any], action: str) -> bool:
    """The same local action at the same substantive evidence count adds nothing."""
    count = evidence_count(state)
    for row in reversed(state.get("decisions") or []):
        if row.get("action") == action and row.get("ok"):
            return row.get("evidence_count") == count
    return False


def available_actions(state: dict[str, Any]) -> list[str]:
    if state.get("gpu_seconds_left", 1) <= 0 and state.get("llm_calls_left", 1) <= 0:
        return ["stop"]
    evidence = state.get("evidence") or []
    actions = ["inspect_data", "retrieve_memory", "retrieve_methods", "stop"]
    pending_experiment = bool(state.get("experiment")) and not state.get("candidate_ready") and not state.get("experiment_failed")
    if not pending_experiment:
        actions.append("propose_experiment")
        actions.append("design_experiment")
    if any(row.get("job_dir") or row.get("fidelity") for row in evidence):
        actions.append("diagnose_results")
        actions.append("collect_diagnostics")
        actions.append("audit_result")
    calls_left = int(state.get("llm_calls_left", 1_000))
    if state.get("repair_task") and int((state.get("repair_task") or {}).get("remaining") or 0) > 0:
        if calls_left >= IMPLEMENT_CALLS:
            actions.append("implement_candidate")
            actions.append("repair_candidate")
    elif (
        pending_experiment
        and submitted_candidates(state) < int(state.get("max_candidates", 4))
        and calls_left >= IMPLEMENT_CALLS
    ):
        actions.append("implement_candidate")
    actions = [name for name in actions if name not in _LOCAL or not _seen_without_new_evidence(state, name)]
    room = int(state.get("training_jobs", 0)) < int(state.get("max_training_jobs", 0)) and float(state.get("gpu_seconds_left", 1)) > 0
    current = state.get("candidate_id")
    if room and state.get("candidate_ready"):
        if not _has(evidence, current, "pilot"):
            actions.append("run_pilot")
        elif not _has(evidence, current, "full"):
            actions.append("run_full")
    if room and current and _has(evidence, current, "full"):
        used = {
            int(row.get("seed") or 0)
            for row in evidence
            if row.get("candidate_id") == current and row.get("fidelity") == "full" and row.get("evaluation_valid")
        }
        declared = [int(item) for item in state.get("training_seeds") or []]
        if declared and any(seed not in used for seed in declared):
            actions.append("replicate")
    return actions


def decide(observation: dict[str, Any], backend: Backend, *, repairs: int = 0) -> dict[str, Any]:
    """Ask once. One schema repair is allowed. A second failure blocks."""
    allowed = observation.get("available_actions") or []
    known = set(observation.get("_known_evidence_ids") or [])
    trainable = set(observation.get("trainable_ids") or ["baseline"])
    reply = backend(observation)
    parsed = _parse(reply, allowed, known, trainable)
    if parsed.get("ok"):
        return parsed
    if repairs >= 1:
        return {"ok": False, "status": "blocked", "detail": parsed.get("detail")}
    repaired = backend({"schema_error": parsed.get("detail"), "previous": reply, "available_actions": allowed})
    second = _parse(repaired, allowed, known, trainable)
    if second.get("ok"):
        second["repairs"] = 1
        return second
    return {"ok": False, "status": "blocked", "detail": second.get("detail"), "repairs": 1}


def _parse(reply: Any, allowed: list[str], known: set[str], trainable: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(reply, dict):
        return {"ok": False, "detail": "not_json"}
    action = reply.get("action")
    if action not in ACTIONS or (allowed and action not in allowed):
        return {"ok": False, "detail": f"unknown_action:{action}"}
    evidence_ids = reply.get("evidence_ids") or []
    if not isinstance(evidence_ids, list) or any(item not in known for item in evidence_ids):
        return {"ok": False, "detail": "unknown_evidence"}
    if action == "propose_experiment" and not isinstance(reply.get("hypothesis_draft"), dict):
        return {"ok": False, "detail": "hypothesis_draft_missing"}
    if action == "stop":
        reason = reply.get("stop_reason")
        if reason not in {None, ""} and reason not in STOP_REASONS:
            return {"ok": False, "detail": f"stop_reason_invalid:{reason}"}
    if action in {"run_pilot", "run_full", "replicate"}:
        target = reply.get("target_id")
        allowed_targets = trainable or set()
        if not isinstance(target, str) or not target:
            return {"ok": False, "detail": "target_id_missing"}
        if allowed_targets and target not in allowed_targets:
            return {"ok": False, "detail": f"unknown_target:{target}"}
    return {"ok": True, "action": action, "reason_zh": reply.get("reason_zh") or "", "raw": reply}
