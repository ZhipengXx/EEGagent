"""Evaluate finite Guard expressions. Missing values are unknown, not false."""

from __future__ import annotations

from typing import Any, Literal

from react_agent.fmri.planning.schemas import Guard

GuardValue = Literal["true", "false", "unknown"]

REGISTERED_METRICS = {
    "basic_statistics.temporal_change_ratio",
    "basic_statistics.max_frame_diff_ratio",
    "basic_statistics.mean",
    "basic_statistics.std",
    "gray_control_contrast.overall_delta_rms",
    "gray_control_contrast.comparable",
    "cortex_mae.embedding_available",
    "cortex_mae.reference_score_assessed",
    "cortex_mae.padded",
}


def eval_guard(
    guard: Guard | None,
    *,
    ledger: dict[str, Any],
    resources: dict[str, Any],
    steps: dict[str, str],
    findings: set[str],
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> GuardValue:
    """Return true/false/unknown. None guard is true."""
    if guard is None:
        return "true"
    op = guard.op
    if op == "and":
        return _combine_and([
            eval_guard(
                item,
                ledger=ledger,
                resources=resources,
                steps=steps,
                findings=findings,
                metrics=metrics,
                thresholds=thresholds,
            )
            for item in guard.args
        ])
    if op == "or":
        return _combine_or([
            eval_guard(
                item,
                ledger=ledger,
                resources=resources,
                steps=steps,
                findings=findings,
                metrics=metrics,
                thresholds=thresholds,
            )
            for item in guard.args
        ])
    if op == "not":
        if not guard.args:
            return "unknown"
        inner = eval_guard(
            guard.args[0],
            ledger=ledger,
            resources=resources,
            steps=steps,
            findings=findings,
            metrics=metrics,
            thresholds=thresholds,
        )
        if inner == "true":
            return "false"
        if inner == "false":
            return "true"
        return "unknown"
    if op == "question_open":
        rec = ledger.get(guard.question_id or "")
        if not rec:
            return "unknown"
        return "true" if rec.get("status") == "untested" else "false"
    if op == "resource_ready":
        bind = resources.get(guard.binding_id or "")
        if bind is None:
            return "unknown"
        ready = bind is True or (isinstance(bind, dict) and bind.get("ready") is True)
        return "true" if ready else "false"
    if op == "step_status_is":
        status = steps.get(guard.step_id or "")
        if status is None:
            return "unknown"
        return "true" if status == guard.status else "false"
    if op == "finding_present":
        if not guard.finding_code:
            return "unknown"
        return "true" if guard.finding_code in findings else "false"
    if op == "metric_compare":
        return _metric_compare(guard, metrics, thresholds)
    return "unknown"


def _metric_compare(guard: Guard, metrics: dict[str, Any], thresholds: dict[str, float]) -> GuardValue:
    path = guard.metric_path or ""
    if path not in REGISTERED_METRICS:
        return "unknown"
    if not guard.threshold_ref or guard.threshold_ref not in thresholds:
        return "unknown"
    value = _lookup(metrics, path)
    if value is None or not isinstance(value, (int, float)):
        return "unknown"
    thresh = thresholds[guard.threshold_ref]
    cmp = guard.comparator or "gt"
    if cmp == "gt":
        ok = value > thresh
    elif cmp == "ge":
        ok = value >= thresh
    elif cmp == "lt":
        ok = value < thresh
    elif cmp == "le":
        ok = value <= thresh
    else:
        ok = value == thresh
    return "true" if ok else "false"


def _lookup(metrics: dict[str, Any], path: str) -> Any:
    cur: Any = metrics
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _combine_and(values: list[GuardValue]) -> GuardValue:
    if any(v == "false" for v in values):
        return "false"
    if any(v == "unknown" for v in values):
        return "unknown"
    return "true"


def _combine_or(values: list[GuardValue]) -> GuardValue:
    if any(v == "true" for v in values):
        return "true"
    if any(v == "unknown" for v in values):
        return "unknown"
    return "false"
