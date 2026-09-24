"""Question ledger for tribe_diagnostic / numeric_minimal profiles."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import QuestionRecord, SampleSpec, ToolResult

QUESTION_DEFS = [
    ("input_mapping", "Is the input / time / space mapping usable?", "validate_input"),
    ("gray_contrast", "Is there a change relative to the paired gray control?", "gray_control_contrast"),
    ("temporal", "When does the change appear?", "stimulus_temporal_profile"),
    ("roi", "Where on cortex does the change sit?", "surface_roi_profile"),
    ("specificity", "Do different images produce nearly the same output?", "cross_image_specificity"),
    ("calibrated_reference", "Is a calibrated distribution check available?", "reference_distribution"),
    ("cortexmae_embedding", "Where does this sit in a learned fMRI representation?", "cortex_mae"),
    ("image_fmri_rsa", "Do image and fMRI similarity structures align?", "semantic_consistency"),
]


def _result(results: list[ToolResult], name: str) -> ToolResult | None:
    for row in results:
        if row.tool_name == name:
            return row
    return None


def _status_for(result: ToolResult | None) -> str:
    if result is None:
        return "untested"
    if result.applicability == "not_applicable":
        return "not_applicable"
    if result.execution_status == "skipped":
        return "unresolved"
    if result.execution_status != "success":
        return "unresolved"
    if any(f.decision_effect == "flag" for f in result.findings):
        return "flagged"
    return "described"


def build_question_ledger(
    *,
    sample: SampleSpec,
    results: list[ToolResult],
    config: FmriCheckConfig,
    resolver: ResourceResolver,
) -> list[QuestionRecord]:
    """Map current evidence onto the diagnostic questions."""
    if getattr(config, "diagnostic", None) and config.diagnostic.enabled:
        from react_agent.fmri.diagnostic.contracts import build_diagnostic_ledger

        return build_diagnostic_ledger(
            sample=sample, results=results, config=config, resolver=resolver
        )
    profile = config.check_profile.name
    required_ids = set(config.check_profile.required_questions)
    if profile == "tribe_diagnostic" and not required_ids:
        required_ids = {"input_mapping", "gray_contrast"}
    if config.planning.enabled:
        if config.planning.require_gray_question_for_non_control:
            required_ids.add("gray_contrast")
        if config.planning.require_temporal_description_for_non_control:
            required_ids.add("temporal")
    rows: list[QuestionRecord] = []
    for qid, text, tool in QUESTION_DEFS:
        result = _result(results, tool)
        required = qid in required_ids
        if qid == "gray_contrast" and resolver.is_gray_control(sample):
            rows.append(
                QuestionRecord(
                    question_id=qid,
                    status="not_applicable",
                    tool_name=tool,
                    required=False,
                    note="gray control sample",
                )
            )
            continue
        if qid == "input_mapping":
            result = _result(results, "validate_input")
        if qid == "temporal":
            result = _result(results, "stimulus_temporal_profile") or _result(
                results, "temporal_diagnostics"
            )
        status = _status_for(result)
        refs = [f"{result.tool_name}:{result.result_id}"] if result else []
        answer_type = "none"
        if status == "described":
            answer_type = "described"
        rows.append(
            QuestionRecord(
                question_id=qid,
                status=status,  # type: ignore[arg-type]
                tool_name=tool,
                evidence_refs=refs,
                required=required,
                note=text,
                answer_type=answer_type,
            )
        )
    return rows


def unanswered_required(ledger: list[QuestionRecord]) -> list[str]:
    """Required questions still untested."""
    return [q.question_id for q in ledger if q.required and q.status == "untested"]


def ledger_as_dict(ledger: list[QuestionRecord]) -> dict[str, Any]:
    return {q.question_id: q.model_dump() for q in ledger}


def candidate_briefs(
    names: list[str],
    registry,
) -> list[dict[str, Any]]:
    """Observation payload for every executable candidate."""
    out = []
    for name in names:
        spec = registry.get(name).spec
        out.append(
            {
                "tool_name": name,
                "question": spec.question,
                "cost_class": spec.cost_class,
                "estimated_cpu_seconds": spec.estimated_cpu_seconds,
                "valid_claims": spec.valid_claims,
                "unsupported_claims": spec.unsupported_claims,
                "scope": spec.scope,
            }
        )
    return out
