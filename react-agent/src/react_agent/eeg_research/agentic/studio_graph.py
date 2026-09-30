"""Thin LangGraph wrapper around one research tick.

Studio shows load -> step -> summarize. A finished campaign is read and not advanced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from react_agent.eeg_research.agentic.llm import role_backend
from react_agent.eeg_research.agentic.loop import _TERMINAL, load_state, tick
from react_agent.eeg_research.agentic.planner import available_actions
from react_agent.eeg_research.agentic.worker import build_services

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = str(_REPO_ROOT / "runs" / "eeg_research_v18")
DEFAULT_CAMPAIGN = "eeg_ui_b37bc8404b9b"


class StudioState(TypedDict, total=False):
    """Studio-facing view of one campaign step."""

    root: str
    campaign: str
    max_ticks: int
    campaign_status: str
    disk_state: dict[str, Any]
    ticks_run: int
    studio_summary: dict[str, Any]


def apply_studio_defaults(state: StudioState) -> StudioState:
    """Fill an empty Studio form. Does not create a campaign."""
    merged: StudioState = {
        "root": DEFAULT_ROOT,
        "campaign": DEFAULT_CAMPAIGN,
        "max_ticks": 1,
    }
    for key, value in state.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value  # type: ignore[literal-required]
    try:
        ticks = int(merged.get("max_ticks") or 1)
    except (TypeError, ValueError):
        ticks = 1
    merged["max_ticks"] = ticks if ticks >= 1 else 1
    return merged


def _campaign_dir(state: StudioState) -> Path:
    return Path(state["root"]) / state["campaign"]


def load_node(state: StudioState) -> StudioState:
    merged = apply_studio_defaults(state)
    camp = _campaign_dir(merged)
    disk = load_state(camp) if (camp / "campaign_state.json").is_file() else {}
    return {
        **merged,
        "campaign_status": str(disk.get("status") or "missing"),
        "disk_state": disk,
        "ticks_run": 0,
    }


def _after_load(state: StudioState) -> str:
    if state.get("campaign_status") in _TERMINAL or state.get("campaign_status") == "missing":
        return "summarize"
    return "step"


def step_node(state: StudioState) -> StudioState:
    """One or more existing ticks. Terminal campaigns never reach this node."""
    camp = _campaign_dir(state)
    planner = role_backend(camp, "research_planner")
    services = build_services(camp)
    current = dict(state.get("disk_state") or load_state(camp))
    ran = 0
    for _ in range(int(state.get("max_ticks") or 1)):
        if current.get("status") in _TERMINAL:
            break
        current = tick(camp, planner, services=services)
        ran += 1
        if current.get("status") in _TERMINAL:
            break
    return {
        "ticks_run": ran,
        "campaign_status": str(current.get("status") or "missing"),
        "disk_state": current,
    }


def summarize_node(state: StudioState) -> StudioState:
    disk = state.get("disk_state") or {}
    decisions = disk.get("decisions") or []
    last = decisions[-1] if decisions else {}
    summary = {
        "status": disk.get("status") or state.get("campaign_status"),
        "last_decision": last.get("action"),
        "candidate_id": disk.get("candidate_id"),
        "live_job": disk.get("live_job"),
        "available_actions": available_actions(disk) if disk else [],
        "training_jobs": disk.get("training_jobs"),
        "llm_calls": disk.get("llm_calls"),
        "max_ticks": int(state.get("max_ticks") or 1),
        "ticks_run": int(state.get("ticks_run") or 0),
    }
    return {"studio_summary": summary}


builder = StateGraph(StudioState)
builder.add_node("load", load_node)
builder.add_node("step", step_node)
builder.add_node("summarize", summarize_node)
builder.add_edge("__start__", "load")
builder.add_conditional_edges(
    "load",
    _after_load,
    {"step": "step", "summarize": "summarize"},
)
builder.add_edge("step", "summarize")
builder.add_edge("summarize", END)
graph = builder.compile(name="EEG Research")
