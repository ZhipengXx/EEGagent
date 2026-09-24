"""Heuristic follow-up tickets. They request inspection and do not flag quality."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.schemas import ToolResult


def derive_tickets(results: list[ToolResult], config: FmriCheckConfig) -> list[dict[str, Any]]:
    """Open or close routing tickets from actual contrast metrics."""
    if not config.diagnostic.enabled:
        return []
    rule = config.diagnostic.routing.contrast_rms_step_ratio
    tickets: list[dict[str, Any]] = []
    contrast = _contrast(results)
    if rule.enabled and contrast is not None:
        metrics = contrast.metrics or {}
        ratio = metrics.get("max_step_over_median")
        median = metrics.get("median_step")
        amplitude = metrics.get("max_step")
        status = "open"
        reason = None
        if ratio is None or median is None or float(median) <= 1e-8:
            status = "dismissed_with_evidence"
            reason = "denominator_unstable"
        elif float(ratio) < float(rule.threshold):
            status = "dismissed_with_evidence"
            reason = "below_heuristic_threshold"
        if _targeted_answers(results, metrics.get("ras_v1") or {}):
            status = "characterized"
            reason = "targeted_description_recorded"
        if status == "open" or reason == "targeted_description_recorded" or reason == "denominator_unstable":
            if status != "dismissed_with_evidence" or reason == "denominator_unstable":
                tickets.append(
                    {
                        "trigger_id": "contrast_rms_step_ratio",
                        "rule_id": config.diagnostic.routing.profile,
                        "rule_version": "exploratory_v1",
                        "question_id": rule.question,
                        "source_execution_id": contrast.result_id,
                        "metric_path": "max_step_over_median",
                        "observed_value": ratio,
                        "amplitude": amplitude,
                        "denominator": median,
                        "threshold_ref": rule.threshold,
                        "basis": rule.basis,
                        "suggested_tool_capabilities": [
                            "temporal_diagnostics:targeted:contrast"
                        ],
                        "priority": 1,
                        "decision_effect": "none",
                        "status": status if status != "dismissed_with_evidence" else "dismissed_with_evidence",
                        "unresolved_reason": reason,
                        "question_resolved": status == "characterized",
                        "defect_confirmed": False,
                    }
                )
        elif status == "dismissed_with_evidence" and reason == "below_heuristic_threshold":
            pass
    return tickets


def _contrast(results: list[ToolResult]) -> ToolResult | None:
    for row in results:
        if row.tool_name == "gray_control_contrast" and row.execution_status == "success":
            if row.applicability == "applicable":
                return row
    return None


def _targeted_answers(results: list[ToolResult], ras: dict[str, Any]) -> bool:
    peak = (ras.get("max_A") or {}) if isinstance(ras, dict) else {}
    want = peak.get("from_frame")
    for row in results:
        if row.tool_name != "temporal_diagnostics":
            continue
        metrics = row.metrics or {}
        if metrics.get("mode") != "targeted" or metrics.get("signal_mode") != "contrast":
            continue
        if want is None or metrics.get("from_frame") == want:
            return row.execution_status == "success"
    return False
