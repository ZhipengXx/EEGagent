"""Build a planner-facing tool catalog from the live registry."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.schemas import SampleSpec, ToolResult
from react_agent.fmri.tools.base import RunContext
from react_agent.fmri.tools.registry import ToolRegistry


def tool_catalog(
    registry: ToolRegistry,
    *,
    sample: SampleSpec,
    context: RunContext,
    results: list[ToolResult],
    enabled: list[str],
    completed: set[str],
) -> list[dict[str, Any]]:
    """Static metadata plus live readiness."""
    prior_failed = {
        r.tool_name for r in results if r.execution_status in {"error", "skipped"}
    }
    rows: list[dict[str, Any]] = []
    for spec in registry.specs():
        if spec.kind == "generator":
            continue
        if spec.name not in enabled:
            continue
        try:
            tool = registry.get(spec.name)
        except KeyError:
            continue
        avail = tool.availability(sample, context)
        missing_evidence = [
            name for name in spec.requires_evidence if name not in {
                r.tool_name for r in results if r.execution_status == "success"
            }
        ]
        blocked_by_failed = [name for name in spec.requires_evidence if name in prior_failed]
        ready = (
            avail.status == "available"
            and spec.name not in completed
            and not missing_evidence
            and not blocked_by_failed
        )
        rows.append(
            {
                "tool_id": spec.name,
                "tool_version": spec.version,
                "stage_hint": spec.stage_hint,
                "scope": spec.scope,
                "answers_questions": spec.answers_questions,
                "requires_resources": spec.requires_resources or spec.requires,
                "requires_evidence": spec.requires_evidence,
                "expected_outputs": spec.expected_outputs or spec.produces,
                "can_affect_verdict": spec.can_affect_verdict,
                "cost_hint": spec.cost_hint or spec.cost_class,
                "applicability": avail.status,
                "unavailable_reason": avail.reason,
                "ready": ready,
                "already_completed": spec.name in completed,
                "unsupported_claims": spec.unsupported_claims,
            }
        )
    return rows
