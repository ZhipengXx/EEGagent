"""Rule and hybrid next-action policies."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.schemas import Decision, SampleSpec, ToolResult
from react_agent.fmri.tools.base import RunContext
from react_agent.fmri.tools.registry import ToolRegistry


def _success(results: list[ToolResult], name: str) -> ToolResult | None:
    for result in results:
        if result.tool_name == name and result.execution_status == "success":
            return result
    return None


def _flags(results: list[ToolResult]) -> set[str]:
    codes: set[str] = set()
    for result in results:
        if result.execution_status != "success":
            continue
        for finding in result.findings:
            if finding.severity in {"flag", "error"}:
                codes.add(finding.code)
    return codes


def _refs_for(result: ToolResult | None, field: str) -> list[str]:
    if result is None:
        return []
    return [f"{result.tool_name}:{result.result_id}:{field}"]


def filter_candidates(
    *,
    registry: ToolRegistry,
    sample: SampleSpec,
    context: RunContext,
    config: FmriCheckConfig,
    results: list[ToolResult],
    completed: set[str],
) -> list[str]:
    """Hard-filter executable optional tools (required already done)."""
    names = registry.executable(
        sample,
        context,
        enabled=config.enabled_tools,
        completed=completed,
    )
    return [n for n in names if n not in config.required_checks]


def rule_decision(
    *,
    sample: SampleSpec,
    results: list[ToolResult],
    candidates: list[str],
    required: list[str],
    completed: set[str],
) -> Decision:
    """Fully offline policy."""
    validate = _success(results, "validate_input")
    flags = _flags(results)
    if "malformed" in flags or (
        validate
        and validate.metrics.get("valid_for_numeric_checks") is False
        and any(f.code == "malformed" for f in validate.findings)
    ):
        return Decision(
            action="stop",
            stop_reason="invalid_input",
            reason="Input contract failed; no further tools.",
            evidence_refs=_refs_for(validate, "metrics.valid_for_numeric_checks"),
            source="rule",
        )
    for name in required:
        if name not in completed:
            if name not in candidates and name not in required:
                continue
            return Decision(
                action="run_tool",
                tool_name=name,
                reason=f"Required check {name} has not completed.",
                source="rule",
            )
    stats = _success(results, "basic_statistics")
    if "temporal_diagnostics" in candidates and flags & {
        "low_temporal_change",
        "temporal_spike",
        "degenerate_signal",
    }:
        return Decision(
            action="run_tool",
            tool_name="temporal_diagnostics",
            reason="Temporal heuristic flags are present and the diagnostic is available.",
            evidence_refs=_refs_for(stats, "metrics.temporal_change_ratio"),
            question_to_resolve="Are adjacent-frame changes localized?",
            expected_observation="lag-1 and adjacent-diff summary",
            source="rule",
        )
    for name, reason in (
        (
            "gray_control_contrast",
            "An explicit gray control is bound; describe Y-G change.",
        ),
        (
            "stimulus_temporal_profile",
            "Describe when the (contrast) RMS changes.",
        ),
        (
            "surface_roi_profile",
            "A prepared surface atlas is available.",
        ),
        (
            "cross_image_specificity",
            "A fixed cohort panel is available for output diversity.",
        ),
        (
            "roi_summary",
            "An explicit ROI map is available for spatial summary.",
        ),
        (
            "reference_distribution",
            "A frozen reference exists for relative (possibly uncalibrated) deviation.",
        ),
        (
            "cortex_mae",
            "CortexMAE-P weights and Schaefer-400 are available.",
        ),
        (
            "semantic_consistency",
            "Independent CLIP embeddings are available for RSA.",
        ),
    ):
        if name in candidates:
            return Decision(
                action="run_tool",
                tool_name=name,
                reason=reason,
                source="rule",
            )
    if flags:
        return Decision(
            action="stop",
            stop_reason="flagged_findings",
            reason="Flags remain but no remaining candidate can localize them.",
            source="rule",
        )
    return Decision(
        action="stop",
        stop_reason="configured_checks_complete",
        reason="Required checks finished with no remaining informative candidate.",
        source="rule",
    )


def choose_profile(candidates: list[str], config: FmriCheckConfig, upgraded: bool) -> str:
    """Select fast vs reasoning without letting the LM change profiles."""
    if (
        not upgraded
        and len(candidates) >= config.hybrid.candidate_count_for_reasoning
    ):
        return "reasoning"
    return "fast"


def validate_decision(
    decision: Decision,
    *,
    candidates: list[str],
    results: list[ToolResult],
    completed: set[str],
    required_unanswered: list[str] | None = None,
    allow_abstain: bool = False,
) -> tuple[bool, str | None]:
    """Reject illegal LM/rule actions before execution."""
    if decision.action == "stop":
        if required_unanswered:
            if allow_abstain and decision.stop_reason in {
                "abstain",
                "degraded_execution",
                "no_candidates",
                "required_questions_blocked",
                "max_lm_calls",
                "max_tool_calls",
                "max_rounds",
            }:
                return True, None
            return False, "required_questions_unanswered"
        return True, None
    if decision.tool_name not in candidates:
        return False, "tool_not_in_candidates"
    tool = decision.tool_name or ""
    if tool in completed:
        return False, "repeat_tool"
    if decision.tool_args:
        for key, value in decision.tool_args.items():
            if key in {"code", "shell", "url", "path"} or isinstance(value, str) and (
                value.startswith("http") or "/" in value or "import " in value
            ):
                return False, "illegal_tool_args"
    allowed_ids = {
        f"{r.tool_name}:{r.result_id}"
        for r in results
        if r.execution_status == "success"
    }
    for ref in decision.evidence_refs:
        prefix = ":".join(ref.split(":")[:2])
        if decision.evidence_refs and prefix not in allowed_ids:
            return False, f"invalid_evidence_ref:{ref}"
    return True, None
