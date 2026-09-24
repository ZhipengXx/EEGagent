"""JSON / Markdown / JSONL reporting. Never invents extra checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from react_agent.fmri.jsonutil import jsonable
from react_agent.fmri.schemas import CLAIM_SCOPE, SCHEMA_VERSION, FinalReport, Finding, ToolResult


def collect_findings(results: list[ToolResult]) -> list[Finding]:
    """Flatten findings from executed tools."""
    out: list[Finding] = []
    for result in results:
        out.extend(result.findings)
    return out


def _decision_effect(finding: Finding) -> str:
    return finding.decision_effect or "none"


def verdict_from_results(
    results: list[ToolResult],
    *,
    required: list[str],
    invalid_input: bool,
    unresolved: list[str],
    unavailable: list[str],
    required_questions_unanswered: bool = False,
    reference_unassessed: bool = False,
) -> tuple[str, str, str]:
    """Deterministic verdict, coverage_status, default stop_reason."""
    success_names = {
        r.tool_name for r in results if r.execution_status == "success"
    }
    quality_findings = [
        f
        for r in results
        if r.execution_status == "success"
        for f in r.findings
        if f.code != "valid_for_numeric_checks" and _decision_effect(f) in {"flag", "block"}
    ]
    hard_errors = [
        f
        for f in quality_findings
        if _decision_effect(f) == "block" or f.code == "malformed"
    ]
    required_done = all(name in success_names for name in required)
    coverage = "complete" if required_done and not reference_unassessed else "partial"
    if not required_done:
        coverage = "partial"

    if invalid_input or hard_errors:
        return "invalid_input", "partial", "invalid_input"
    if quality_findings:
        return "flagged", coverage, "flagged_findings"
    if not required_done:
        return "inconclusive", "partial", "required_checks_incomplete"
    if required_questions_unanswered:
        return "inconclusive", "partial", "required_questions_unanswered"
    if unresolved and unavailable:
        return "inconclusive", "partial", "unresolved_without_tools"
    return "passed_configured_checks", coverage, "configured_checks_complete"


def _reference_unassessed(results: list[ToolResult]) -> bool:
    for result in results:
        if result.tool_name != "reference_distribution":
            continue
        if result.execution_status == "skipped":
            return True
        status = (result.metrics or {}).get("calibration_status")
        if status in {"insufficient", "engineering_smoke"}:
            return True
        if result.applicability == "not_applicable":
            return True
    return False


def build_final_report(
    *,
    sample_id: str,
    safe_sample_id: str,
    results: list[ToolResult],
    required: list[str],
    invalid_input: bool,
    unresolved: list[str],
    unavailable: list[str],
    need_score: dict[str, Any],
    stop_reason: str,
    fallback_used: bool,
    cost: dict[str, Any],
    lm_summary: dict[str, Any] | None = None,
    coverage_by_question: dict[str, Any] | None = None,
    evidence_scopes: list[str] | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
    lm_decision_count: int | None = None,
    fallback_count: int | None = None,
    non_fallback_lm_successes: int | None = None,
    check_profile: str | None = None,
    required_questions_unanswered: bool = False,
    screening_decision: str | None = None,
    degraded_execution: bool = False,
    degraded_reason: str | None = None,
    requested_policy: str | None = None,
    resolved_policy: str | None = None,
    requested_backend: str | None = None,
    resolved_backend: str | None = None,
    analysis_goal: str | None = None,
    coverage_profile: str | None = None,
    coverage_by_dimension: dict[str, Any] | None = None,
    depth_reached: str | None = None,
    followups: list[dict[str, Any]] | None = None,
    biological_validity: str | None = None,
    upgrade_notes: list[str] | None = None,
) -> FinalReport:
    """Assemble the canonical report from already-run evidence."""
    ref_open = _reference_unassessed(results)
    verdict, coverage, default_stop = verdict_from_results(
        results,
        required=required,
        invalid_input=invalid_input,
        unresolved=unresolved,
        unavailable=unavailable,
        required_questions_unanswered=required_questions_unanswered,
        reference_unassessed=ref_open,
    )
    findings = collect_findings(results)
    metrics_table: dict[str, Any] = {}
    executed = []
    skipped = []
    scopes = list(evidence_scopes or ["numeric"])
    limitations = [
        "Numeric screening passing does not prove that the image-conditioned "
        "real fMRI response is correct.",
        f"claim_scope={CLAIM_SCOPE}",
        "passed_configured_checks does not mean every tool ran.",
    ]
    executions: list[dict[str, Any]] = []
    buckets: dict[str, list[Any]] = {}
    for result in results:
        executions.append(
            {
                "execution_id": result.result_id,
                "tool_name": result.tool_name,
                "execution_key": result.execution_key,
                "signal_mode": (result.signal_provenance or {}).get("signal_mode")
                or (result.metrics or {}).get("signal_mode"),
                "mode": (result.metrics or {}).get("mode"),
                "status": result.execution_status,
                "provenance": result.signal_provenance,
            }
        )
        if result.execution_status == "success":
            executed.append(result.tool_name)
            buckets.setdefault(result.tool_name, []).append(result.metrics)
        if result.execution_status == "skipped":
            skipped.append(result.tool_name)
        limitations.extend(result.limitations)
        family = {
            "gray_control_contrast": "control_relative",
            "stimulus_temporal_profile": "temporal_descriptive",
            "temporal_diagnostics": "temporal_descriptive",
            "surface_roi_profile": "surface_descriptive",
            "roi_summary": "surface_descriptive",
            "cross_image_specificity": "cross_sample_descriptive",
            "cortex_mae": "learned_representation_descriptive",
            "semantic_consistency": "image_fmri_rsa_descriptive",
        }.get(result.tool_name)
        if family and family not in scopes:
            scopes.append(family)
    for name, rows in buckets.items():
        metrics_table[name] = rows[0] if len(rows) == 1 else {"executions": rows}
    return FinalReport(
        schema_version=SCHEMA_VERSION,
        sample_id=sample_id,
        safe_sample_id=safe_sample_id,
        verdict=verdict,  # type: ignore[arg-type]
        coverage_status=coverage,  # type: ignore[arg-type]
        claim_scope=CLAIM_SCOPE,
        stop_reason=stop_reason or default_stop,
        unresolved_questions=unresolved,
        unavailable_checks=unavailable,
        evidence_confidence=None,
        recommended_training_weight=None,
        need_score=need_score,
        findings=findings,
        metrics_table=jsonable(metrics_table),
        executed_tools=executed,
        skipped_tools=skipped,
        fallback_used=fallback_used,
        limitations=list(dict.fromkeys(limitations)),
        lm_summary=lm_summary,
        cost=jsonable(cost),
        coverage_by_question=coverage_by_question or {},
        evidence_scopes=scopes,
        unassessed_claims=[
            "stimulus_conditioned_biological_correctness",
            "cortex_mae_embedding_reference",
            "image_fmri_semantic_consistency",
        ],
        candidate_count=candidate_count,
        selected_count=selected_count,
        executed_count=len(executed),
        skipped_count=len(skipped),
        lm_decision_count=lm_decision_count,
        fallback_count=fallback_count,
        non_fallback_lm_successes=non_fallback_lm_successes,
        check_profile=check_profile,
        screening_decision=_screening_decision(
            verdict,
            stop_reason=stop_reason or default_stop,
            required_unanswered=required_questions_unanswered,
            invalid_input=invalid_input,
            explicit=screening_decision,
        ),
        screening_coverage=coverage_by_question or {},
        degraded_execution=degraded_execution,
        degraded_reason=degraded_reason,
        requested_policy=requested_policy,
        resolved_policy=resolved_policy,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        analysis_goal=analysis_goal,
        coverage_scope="configured_required_numeric_questions" if analysis_goal else None,
        coverage_profile=coverage_profile,
        coverage_by_dimension=coverage_by_dimension or {},
        depth_reached=depth_reached,
        followups=followups or [],
        executions=executions,
        biological_validity=biological_validity,
        upgrade_notes=upgrade_notes or [],
    )


def _screening_decision(
    verdict: str,
    *,
    stop_reason: str,
    required_unanswered: bool,
    invalid_input: bool,
    explicit: str | None,
) -> str:
    if explicit:
        return explicit
    if invalid_input or verdict == "invalid_input":
        return "blocked"
    if verdict == "flagged":
        return "flagged"
    if required_unanswered or stop_reason in {
        "abstain",
        "degraded_execution",
        "no_candidates",
        "required_questions_unanswered",
        "required_questions_blocked",
    }:
        return "abstain"
    if verdict == "passed_configured_checks":
        return "pass_configured"
    return "abstain"


def write_report_json(path: Path, report: FinalReport) -> None:
    """Write machine-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(), handle, indent=2)


def write_report_md(path: Path, report: FinalReport) -> None:
    """Write a human-readable Markdown summary. Numbers come from the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# fMRI check report: {report.sample_id}",
        "",
        f"- verdict: `{report.verdict}`",
        f"- screening_decision: `{report.screening_decision}`",
        f"- coverage: `{report.coverage_status}`",
        f"- analysis_goal: `{report.analysis_goal}`",
        f"- coverage_scope: `{report.coverage_scope}`",
        f"- claim_scope: `{report.claim_scope}`",
        f"- stop_reason: {report.stop_reason}",
        f"- fallback_used: {report.fallback_used}",
        f"- evidence_confidence: {report.evidence_confidence}",
        f"- recommended_training_weight: {report.recommended_training_weight}",
        "",
        "## Findings",
    ]
    if not report.findings:
        lines.append("- none")
    for finding in report.findings:
        lines.append(
            f"- `{finding.code}` ({finding.severity}): {finding.message}"
        )
    lines += ["", "## Metrics"]
    for tool, metrics in report.metrics_table.items():
        lines.append(f"### {tool}")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key == "rois" or key == "deviations":
                    lines.append(f"- `{key}`: (see JSON)")
                else:
                    lines.append(f"- `{key}`: {value}")
        else:
            lines.append(f"- {metrics}")
    lines += ["", "## Limitations"]
    for item in report.limitations:
        lines.append(f"- {item}")
    if report.lm_summary:
        lines += ["", "## LM summary (explanatory only)", json.dumps(report.lm_summary, indent=2)]
    lines += ["", "## Cost", json.dumps(report.cost, indent=2)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSONL event. Safe if the run dies mid-loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write batch summary.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "verdict",
        "coverage",
        "stop_reason",
        "checks",
        "tool_calls",
        "lm_calls",
        "input_tokens",
        "output_tokens",
        "api_usd",
        "elapsed",
        "fallback_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
