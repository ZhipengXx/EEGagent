"""LangGraph export for the fMRI check loop."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.fmri.loop import (
    LoopRuntime,
    execute_tool,
    finalize,
    ingest,
    run_required_checks,
    select_action,
    update_evidence,
    validate_action,
)
from react_agent.fmri.state import CheckState

_RUNTIMES: dict[str, LoopRuntime] = {}


def bind_runtime(run_id: str, runtime: LoopRuntime) -> None:
    """Attach a LoopRuntime for graph nodes of this run."""
    _RUNTIMES[run_id] = runtime


def _rt(state: CheckState) -> LoopRuntime:
    run_id = state.get("run_id") or ""
    runtime = _RUNTIMES.get(run_id)
    if runtime is None:
        raise RuntimeError("fmri_check runtime missing; use CLI or bind_runtime")
    return runtime


async def ingest_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Ingest node."""
    return await ingest(state, _rt(state))


async def required_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Required checks node."""
    return await run_required_checks(state, _rt(state))


def evidence_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Evidence update node."""
    return update_evidence(state, _rt(state))


async def select_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Policy node."""
    return await select_action(state, _rt(state))


async def validate_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Action validator node."""
    return await validate_action(state, _rt(state))


async def execute_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Single-tool executor."""
    return await execute_tool(state, _rt(state))


async def finalize_node(state: CheckState, runtime: Runtime[Any] | None = None) -> dict[str, Any]:
    """Report writer."""
    return await finalize(state, _rt(state))


def _route(state: CheckState) -> str:
    return str(state.get("route") or "finalize")


def _after_required(state: CheckState) -> Literal["update_evidence", "finalize"]:
    return "finalize" if state.get("route") == "finalize" else "update_evidence"


def _after_select(state: CheckState) -> Literal["validate_action", "finalize"]:
    return "finalize" if state.get("route") == "finalize" else "validate_action"


def _after_validate(state: CheckState) -> Literal["execute_tool", "select_action", "finalize"]:
    route = state.get("route")
    if route == "execute":
        return "execute_tool"
    if route == "select":
        return "select_action"
    return "finalize"


builder = StateGraph(CheckState)
builder.add_node("ingest", ingest_node)
builder.add_node("run_required_checks", required_node)
builder.add_node("update_evidence", evidence_node)
builder.add_node("select_action", select_node)
builder.add_node("validate_action", validate_node)
builder.add_node("execute_tool", execute_node)
builder.add_node("finalize", finalize_node)
builder.add_edge("__start__", "ingest")
builder.add_edge("ingest", "run_required_checks")
builder.add_conditional_edges("run_required_checks", _after_required)
builder.add_edge("update_evidence", "select_action")
builder.add_conditional_edges("select_action", _after_select)
builder.add_conditional_edges("validate_action", _after_validate)
builder.add_edge("execute_tool", "update_evidence")
builder.add_edge("finalize", "__end__")

graph = builder.compile(name="fMRI Check")
