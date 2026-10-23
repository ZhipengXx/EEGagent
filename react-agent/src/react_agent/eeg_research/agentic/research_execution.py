"""Explicit selection of the research-task execution protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

SINGLE_PLAN_VERSION = "single_plan_v1"
SCIENCE_MAX_FIXES = 5
SCIENCE_MAX_ATTEMPTS = 1 + SCIENCE_MAX_FIXES


def research_goal(value: Mapping | Path | str) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_dir():
        path = path / "goal.json"
    goal = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if path.name == "goal.json" and goal:
        from .iteration_continuation import apply_continuation_goal
        return apply_continuation_goal(path.parent, goal)
    return goal


def single_plan_enabled(value: Mapping | Path | str) -> bool:
    config = research_goal(value).get("research_execution") or {}
    return isinstance(config, Mapping) and config.get("version") == SINGLE_PLAN_VERSION


def independent_critic_enabled(value: Mapping | Path | str) -> bool:
    # Single-plan research always proposes, then has an independent Critic select.
    return single_plan_enabled(value)


def role_api_attempt_limit(value: Mapping | Path | str, role: str, default: int = 2) -> int:
    """One role's total requests, including its initial request and corrections.

    New scientific wires allow five corrections. Explicit role/global limits
    remain authoritative; the global policy's *implicit* two-attempt default
    continues to govern the other roles without blocking the new wires.
    """
    goal = research_goal(value)
    config = goal.get("research_execution") or {}
    fields = {"research_planner": "max_planner_api_attempts",
              "research_critic": "max_critic_api_attempts",
              "experiment_designer": "max_designer_api_attempts"}
    field = fields.get(role)
    implicit = SCIENCE_MAX_ATTEMPTS if field and single_plan_enabled(goal) else default
    limit = config.get(field, implicit) if field else default
    if type(limit) is not int or not 1 <= limit <= SCIENCE_MAX_ATTEMPTS:
        raise ValueError("invalid_role_api_attempt_limit:" + role)
    policy = goal.get("api_request_policy") or {}
    if "max_attempts_per_operation" in policy:
        from .api_policy import ApiRequestPolicy
        limit = min(limit, ApiRequestPolicy.model_validate(policy).max_attempts_per_operation)
    return limit


def planning_call_budget(value: Mapping | Path | str) -> int:
    """The two planning stages share one finite native round allocation."""
    goal = research_goal(value)
    if not independent_critic_enabled(goal):
        return 2
    return sum(role_api_attempt_limit(goal, role) for role in ("research_planner", "research_critic"))


def paid_maintenance_mode(value: Mapping | Path | str) -> str:
    goal = research_goal(value)
    mode = goal.get("paid_maintenance")
    execution = goal.get("research_execution") or {}
    nested = execution.get("paid_maintenance") if isinstance(execution, Mapping) else None
    if mode is not None and nested is not None and mode != nested:
        raise ValueError("conflicting_paid_maintenance_policy")
    if mode is None:
        mode = nested
    if mode is None:
        return "off" if single_plan_enabled(goal) else "legacy"
    if mode not in {"off", "emergency", "batch"}:
        raise ValueError("invalid_paid_maintenance_policy")
    return mode
