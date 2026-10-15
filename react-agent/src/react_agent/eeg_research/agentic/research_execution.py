"""Explicit selection of the research-task execution protocol."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

SINGLE_PLAN_VERSION = "single_plan_v1"


def research_goal(value: Mapping | Path | str) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_dir():
        path = path / "goal.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def single_plan_enabled(value: Mapping | Path | str) -> bool:
    config = research_goal(value).get("research_execution") or {}
    return isinstance(config, Mapping) and config.get("version") == SINGLE_PLAN_VERSION


def independent_critic_enabled(value: Mapping | Path | str) -> bool:
    config = research_goal(value).get("research_execution") or {}
    return single_plan_enabled(value) and config.get("planning_mode") == "propose_then_critic"


def planning_call_budget(value: Mapping | Path | str) -> int:
    """The two planning stages share one finite native round allocation."""
    goal = research_goal(value)
    config = goal.get("research_execution") or {}
    if not independent_critic_enabled(goal):
        return 2
    return min(2, int(config.get("max_planner_api_attempts", 2))) + int(config.get("max_critic_api_attempts", 2))


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
