"""Question contracts for the image16 numeric diagnostic profile."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import DEEP_SCREEN_QUESTIONS, FmriCheckConfig
from react_agent.fmri.diagnostic.routing import derive_tickets
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import QuestionRecord, SampleSpec, ToolResult

CONTRACTS: list[dict[str, Any]] = [
    {
        "question_id": "input_contract",
        "required": True,
        "accepted_signal_modes": ["raw"],
        "required_metric_keys": ["T", "V"],
        "note": "Shape and basic numeric contract.",
    },
    {
        "question_id": "gray_comparability",
        "required": True,
        "accepted_signal_modes": ["contrast"],
        "required_metric_keys": ["overall_delta_rms"],
        "note": "Explicit matched gray contrast.",
    },
    {
        "question_id": "stimulus_temporal_description",
        "required": True,
        "accepted_signal_modes": ["contrast"],
        "required_metric_keys": ["ras_v1"],
        "note": "Event-window description of contrast R/A/S.",
    },
    {
        "question_id": "coarse_spatial_description",
        "required": True,
        "accepted_signal_modes": ["contrast"],
        "required_metric_keys": ["hemisphere_rms"],
        "note": "Cheap vertex/hemisphere scan of contrast.",
    },
    {
        "question_id": "triggered_followups",
        "required": True,
        "accepted_signal_modes": ["contrast"],
        "required_metric_keys": [],
        "note": "Open routing tickets must be characterized or blocked.",
    },
    {
        "question_id": "reference_quality",
        "required": False,
        "accepted_signal_modes": ["contrast", "raw"],
        "required_metric_keys": [],
        "note": "Optional calibrated reference.",
    },
]


DEEP_NOTES = {
    "roi": "Deep screening: ROI profile of the contrast.",
    "specificity": "Deep screening: cross-image specificity.",
    "calibrated_reference": "Deep screening: position in the reference distribution.",
    "image_fmri_rsa": "Deep screening: image and fMRI similarity structure.",
}


def _deep_record(
    question_id: str,
    tool: str,
    results: list[ToolResult],
    config: FmriCheckConfig,
    gray: bool,
) -> QuestionRecord:
    """A missing resource is recorded as blocked. It is never counted as a pass."""
    runs = [r for r in results if r.tool_name == tool]
    status = "unassessed"
    refs: list[str] = []
    reason: str | None = "not_run"
    required = True
    if gray:
        status = "not_applicable"
        required = False
        reason = "gray_control_sample"
    elif runs:
        last = runs[-1]
        refs = [f"{last.tool_name}:{last.result_id}"]
        if last.execution_status == "success" and last.applicability != "not_applicable":
            status = "described"
            reason = None
        else:
            status = "blocked"
            reason = last.error or last.skip_reason or "tool_not_applicable"
    return QuestionRecord(
        question_id=question_id,
        status=status,  # type: ignore[arg-type]
        tool_name=tool,
        required=required,
        required_by_profile=required,
        accepted_signal_modes=["contrast", "raw"],
        required_metric_keys=[],
        compatible_protocol=config.diagnostic.coverage_profile,
        evidence_refs=refs,
        evidence_ids=refs,
        note=DEEP_NOTES.get(question_id),
        unresolved_reason=reason,
        answer_type="described" if status == "described" else "none",
    )


def _by_name(results: list[ToolResult], name: str) -> list[ToolResult]:
    return [r for r in results if r.tool_name == name and r.execution_status == "success"]


def _has_signal(result: ToolResult, mode: str) -> bool:
    prov = result.signal_provenance or {}
    metric_mode = (result.metrics or {}).get("signal_mode") or (result.metrics or {}).get("mode")
    return prov.get("signal_mode") == mode or metric_mode == mode


def build_diagnostic_ledger(
    *,
    sample: SampleSpec,
    results: list[ToolResult],
    config: FmriCheckConfig,
    resolver: ResourceResolver,
) -> list[QuestionRecord]:
    """Map evidence onto V1.4 contracts. Tool name alone is not enough."""
    gray = resolver.is_gray_control(sample)
    tickets = derive_tickets(results, config)
    open_tickets = [t for t in tickets if t.get("status") == "open"]
    blocked_tickets = [t for t in tickets if t.get("status") == "blocked"]
    rows: list[QuestionRecord] = []
    for spec in CONTRACTS:
        qid = spec["question_id"]
        required = bool(spec["required"])
        status = "unassessed"
        refs: list[str] = []
        reason = None
        if qid == "input_contract":
            ok = _by_name(results, "validate_input") and _by_name(results, "basic_statistics")
            if ok:
                status = "answered"
                refs = [f"{r.tool_name}:{r.result_id}" for r in ok]
        elif qid == "gray_comparability":
            if gray:
                status = "not_applicable"
                required = False
                reason = "gray_control_sample"
            else:
                hit = [
                    r
                    for r in _by_name(results, "gray_control_contrast")
                    if r.applicability == "applicable" and (r.metrics or {}).get("comparable")
                ]
                if hit:
                    status = "answered"
                    refs = [f"{hit[-1].tool_name}:{hit[-1].result_id}"]
        elif qid == "stimulus_temporal_description":
            if gray:
                status = "not_applicable"
                required = False
            else:
                hit = [
                    r
                    for r in _by_name(results, "stimulus_temporal_profile")
                    if _has_signal(r, "contrast") and "ras_v1" in (r.metrics or {})
                ]
                if hit:
                    status = "described"
                    refs = [f"{hit[-1].tool_name}:{hit[-1].result_id}"]
                elif _by_name(results, "temporal_diagnostics"):
                    reason = "raw_temporal_does_not_satisfy_contrast_contract"
        elif qid == "coarse_spatial_description":
            if gray:
                status = "not_applicable"
                required = False
            else:
                hit = [
                    r
                    for r in _by_name(results, "surface_spatial_sanity")
                    if (r.metrics or {}).get("mode") == "summary" and _has_signal(r, "contrast")
                ]
                if hit:
                    status = "described"
                    refs = [f"{hit[-1].tool_name}:{hit[-1].result_id}"]
        elif qid == "triggered_followups":
            if not tickets:
                status = "answered"
                reason = "no_open_ticket"
            elif open_tickets:
                status = "unassessed"
                reason = "open_followup"
                refs = [t["source_execution_id"] for t in open_tickets]
            elif blocked_tickets:
                status = "blocked"
                reason = "followup_blocked"
            else:
                status = "described"
                reason = "characterized_not_a_defect"
        elif qid == "reference_quality":
            status = "unassessed"
            reason = "no_compatible_calibrated_reference"
        rows.append(
            QuestionRecord(
                question_id=qid,
                status=status,  # type: ignore[arg-type]
                required=required,
                required_by_profile=required,
                accepted_signal_modes=list(spec["accepted_signal_modes"]),
                required_metric_keys=list(spec["required_metric_keys"]),
                compatible_protocol=config.diagnostic.coverage_profile,
                evidence_refs=refs,
                evidence_ids=refs,
                note=spec["note"],
                unresolved_reason=reason,
                answer_type="described" if status == "described" else "none",
            )
        )
    if config.diagnostic.screen_depth == "deep":
        for question_id, tool in DEEP_SCREEN_QUESTIONS:
            rows.append(_deep_record(question_id, tool, results, config, gray))
    return rows
