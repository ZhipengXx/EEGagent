"""Reject illegal plans before they become executable."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.planning.guards import REGISTERED_METRICS, eval_guard
from react_agent.fmri.planning.schemas import PlanProposal, PlanStep


class PlanValidationError(ValueError):
    """Illegal PlanProposal."""


def validate_proposal(
    proposal: PlanProposal,
    *,
    catalog: list[dict[str, Any]],
    required_questions: list[str],
    remaining_lm_calls: int,
    remaining_tool_calls: int,
    evidence_ids: set[str],
    resources: dict[str, Any],
    run_goal: str = "quality_screening",
    has_reference_score: bool = False,
    completed_tools: set[str] | None = None,
    ledger: dict[str, Any] | None = None,
) -> None:
    """Raise PlanValidationError when the proposal is not executable."""
    completed_tools = completed_tools or set()
    known_tools = {row["tool_id"]: row for row in catalog}
    step_ids = [step.step_id for step in proposal.steps]
    if len(step_ids) != len(set(step_ids)):
        raise PlanValidationError("duplicate_step_id")
    if proposal.next_step_id and proposal.next_step_id not in step_ids and proposal.steps:
        raise PlanValidationError("next_step_unknown")
    if _has_cycle(proposal.steps):
        raise PlanValidationError("dependency_cycle")
    planned_questions = {step.question_id for step in proposal.steps}
    answered = _answered_questions(ledger, catalog, completed_tools)
    if proposal.stop_request is None:
        missing_required = [
            qid
            for qid in required_questions
            if qid not in planned_questions and qid not in answered
        ]
        if missing_required and not proposal.resource_gaps:
            raise PlanValidationError(f"required_question_dropped:{missing_required[0]}")
    if remaining_tool_calls < 0 or remaining_lm_calls < 0:
        raise PlanValidationError("budget_exhausted")
    optional_count = sum(
        1 for step in proposal.steps if step.question_id not in required_questions
    )
    if optional_count > remaining_tool_calls:
        raise PlanValidationError("budget_over_optional_tools")
    for step in proposal.steps:
        _validate_step(
            step,
            known_tools=known_tools,
            evidence_ids=evidence_ids,
            resources=resources,
            run_goal=run_goal,
            has_reference_score=has_reference_score,
            all_ids=set(step_ids),
            completed_tools=completed_tools,
        )
        if step.tool_id in completed_tools or known_tools.get(step.tool_id, {}).get(
            "already_completed"
        ):
            step.status = "completed"


def _answered_questions(
    ledger: dict[str, Any] | None,
    catalog: list[dict[str, Any]],
    completed_tools: set[str],
) -> set[str]:
    """Required questions already closed by evidence, not still untested."""
    answered: set[str] = set()
    for qid, rec in (ledger or {}).items():
        if isinstance(rec, dict) and rec.get("status") not in {None, "untested"}:
            answered.add(qid)
    for row in catalog:
        if row.get("already_completed") or row.get("tool_id") in completed_tools:
            answered.update(row.get("answers_questions") or [])
    return answered


def _validate_step(
    step: PlanStep,
    *,
    known_tools: dict[str, dict[str, Any]],
    evidence_ids: set[str],
    resources: dict[str, Any],
    run_goal: str,
    has_reference_score: bool,
    all_ids: set[str],
    completed_tools: set[str],
) -> None:
    if step.tool_id not in known_tools:
        raise PlanValidationError(f"unknown_tool:{step.tool_id}")
    spec = known_tools[step.tool_id]
    already_done = step.tool_id in completed_tools or spec.get("already_completed") is True
    if not already_done:
        if spec.get("applicability") not in {None, "available"} and not step.depends_on:
            raise PlanValidationError(f"tool_not_ready:{step.tool_id}")
        if spec.get("ready") is False and not step.depends_on:
            raise PlanValidationError(f"tool_not_ready:{step.tool_id}")
    for dep in step.depends_on:
        if dep not in all_ids:
            raise PlanValidationError(f"unknown_dependency:{dep}")
    for ref in step.supporting_evidence_ids:
        prefix = ":".join(str(ref).split(":")[:2])
        if evidence_ids and prefix not in evidence_ids and ref not in evidence_ids:
            raise PlanValidationError(f"invalid_evidence_ref:{ref}")
    if step.when is not None:
        _validate_guard(step.when)
    if step.tool_id == "cortex_mae":
        feature = step.expected_evidence_type in {"embedding", "feature_extraction"}
        if run_goal == "quality_screening" and not has_reference_score and not feature:
            raise PlanValidationError("cortex_mae_requires_reference_or_feature_goal")
    for bind in step.resource_binding_ids:
        rec = resources.get(bind)
        if rec is False or (isinstance(rec, dict) and rec.get("ready") is False):
            raise PlanValidationError(f"resource_not_ready:{bind}")


def _validate_guard(guard) -> None:
    if guard.op == "metric_compare":
        if not guard.metric_path or guard.metric_path not in REGISTERED_METRICS:
            raise PlanValidationError(f"unregistered_metric:{guard.metric_path}")
        if not guard.threshold_ref:
            raise PlanValidationError("metric_compare_needs_threshold_ref")
    for child in guard.args:
        _validate_guard(child)


def _has_cycle(steps: list[PlanStep]) -> bool:
    deps = {step.step_id: list(step.depends_on) for step in steps}
    seen: set[str] = set()
    stack: set[str] = set()

    def visit(node: str) -> bool:
        if node in stack:
            return True
        if node in seen:
            return False
        stack.add(node)
        for child in deps.get(node, []):
            if visit(child):
                return True
        stack.remove(node)
        seen.add(node)
        return False

    return any(visit(step.step_id) for step in steps)


def ready_steps(
    proposal: PlanProposal,
    *,
    ledger: dict[str, Any],
    resources: dict[str, Any],
    findings: set[str],
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> list[PlanStep]:
    """Steps whose dependencies finished and guard is true."""
    status = {step.step_id: step.status for step in proposal.steps}
    ready: list[PlanStep] = []
    for step in proposal.steps:
        if step.status != "pending":
            continue
        if any(status.get(dep) != "completed" for dep in step.depends_on):
            continue
        value = eval_guard(
            step.when,
            ledger=ledger,
            resources=resources,
            steps=status,
            findings=findings,
            metrics=metrics,
            thresholds=thresholds,
        )
        if value == "true":
            ready.append(step)
    return ready
