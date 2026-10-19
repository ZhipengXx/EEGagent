"""Staged analysis contract: native facts, the 4-key Analyst reply, delivery check, record.

The program owns numbers, execution state, identities and coverage; the Analyst
owns findings, hypothesis judgement, limitations and the summary. One rule set
serves live replies, recovered replies and the stage review. Pure functions:
no file I/O and no model calls.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

RECORD_VERSION = "eeg_research.analysis_record.v1"
# Envelope/runtime keys around a stored payload; never part of the Analyst reply.
_STRIP = frozenset({"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "status",
    "artifact_refs", "candidate_id", "method_benchmark_origin", "method_screening_origin"})
_TEXT = ("observation", "interpretation", "implication")


def _pick(value, fields):
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def allowed_refs(measurement_review: dict) -> list[str]:
    """The only references a finding may cite: the measurement review and its native refs."""
    review = measurement_review if isinstance(measurement_review, dict) else {}
    refs = [review.get("artifact_id"), review.get("path"), *(review.get("evidence_refs") or [])]
    return list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref))


def _measurement_score(score: dict) -> dict:
    """Scientific evaluation conditions, without the scorer's storage receipts."""
    result = _pick(score, ("candidate_id", "fold_id", "held_out_subject", "train_subjects", "seed", "fidelity"))
    metrics = score.get("metrics") or {}
    result["metrics"] = _pick(metrics, ("top1", "top5", "query_count", "candidate_count"))
    identity = metrics.get("inference_identity") or {}
    conditions = _pick(identity, ("eval_mode", "augmentation", "evaluator", "input_geometry", "tie_policy"))
    if conditions:
        result["evaluation_conditions"] = conditions
    return result


def _paired_evaluation_conditions(screening: dict) -> list[dict]:
    """Report exact recorded input equalities, never infer unavailable identities."""
    controls = {row.get("fold_id"): row for row in screening.get("baseline_scores") or []}
    result = []
    for score in screening.get("scores") or []:
        control = controls.get(score.get("fold_id"))
        if not control:
            continue
        left = (score.get("metrics") or {}).get("inference_identity") or {}
        right = (control.get("metrics") or {}).get("inference_identity") or {}
        if not left or not right:
            continue
        same = {}
        for label, key in (("held_out_files", "held_out_files"), ("test_image_features", "test_image_cache"),
                           ("evaluated_inputs", "actual_inputs")):
            if left.get(key) is not None and right.get(key) is not None:
                same[label] = left[key] == right[key]
        result.append({"fold_id": score.get("fold_id"), "same_recorded_inputs": same})
    return result


def build_analysis_facts(payload: dict) -> dict:
    """Project native facts once; nothing is invented except the baseline mean of paired scores."""
    latest = payload.get("latest") or {}
    screening = payload.get("method_screening") if isinstance(payload.get("method_screening"), dict) else None
    review = payload.get("measurement_review") or {}
    facts = {"run_identity": _pick(latest, ("evidence_id", "candidate_id", "fidelity", "seed", "phase", "suite_id")),
        "execution": {"evaluation_valid": latest.get("evaluation_valid"),
                      "measurement_review_status": review.get("status")}}
    if screening is not None:
        measurement = _pick(screening, ("paired_scores", "screening_mean_top1", "screening_mean_top5",
            "screening_mean_delta_pp", "min_mean_gain_pp", "screening_result", "promotion_decision"))
        baselines = [row.get("baseline_top1") for row in screening.get("paired_scores") or []]
        if baselines and all(isinstance(value, (int, float)) for value in baselines):
            measurement["baseline_mean_top1"] = sum(baselines) / len(baselines)
        facts["execution"]["screening_result"] = screening.get("screening_result")
        # Previously these scientific conditions lived only in the duplicated
        # method_screening subtree. Keep them when that subtree is removed.
        if screening.get("scores"):
            measurement["scores"] = [_measurement_score(row) for row in screening["scores"]]
        if screening.get("baseline_scores"):
            measurement["baseline_scores"] = [_measurement_score(row) for row in screening["baseline_scores"]]
        conditions = _paired_evaluation_conditions(screening)
        if conditions:
            measurement["paired_evaluation_conditions"] = conditions
    elif isinstance(payload.get("method_suite"), dict):
        suite = payload["method_suite"]
        measurement = _pick(suite, ("mean_top1", "mean_top5", "std_top1", "std_top5", "ddof", "coverage"))
        if suite.get("scores"):
            measurement["scores"] = [_measurement_score(row) for row in suite["scores"]]
        if "std_top1" in measurement or "std_top5" in measurement:
            measurement["variation_unit"] = "across_subject_folds"
        if payload.get("comparison"):
            measurement["comparison"] = copy.deepcopy(payload["comparison"])
    else:
        measurement = {**_pick(latest, ("fixed_bank_top1", "delta_vs_control_pp", "control_id")),
                       **({"comparison": copy.deepcopy(payload["comparison"])} if payload.get("comparison") else {})}
    facts["measurement"] = measurement
    coverage = _pick(payload.get("verified_run_coverage"), ("complete_fold_count", "required_fold_count",
        "full_required_fold_count", "independent_seed_count", "replicated"))
    coverage.update(_pick(screening, ("screening_fold_ids", "screening_subjects")))
    if "independent_seed_count" in payload:
        coverage.setdefault("independent_seed_count", payload["independent_seed_count"])
    facts["coverage"] = coverage
    # Not "scope": context capture treats a top-level "scope" as a visibility label.
    context = _pick(payload, ("science_scope", "scientific_unit", "evaluation_population", "limitations"))
    if screening is not None and "science_scope" in screening:
        context["science_scope"] = copy.deepcopy(screening["science_scope"])
    facts["evaluation_context"] = context
    refs = allowed_refs(review)
    evidence = payload.get("analysis_evidence") or {}
    if evidence:
        facts["training_runs"] = copy.deepcopy(evidence.get("training_runs") or [])
        facts["source_catalog"] = copy.deepcopy(evidence.get("source_catalog") or [])
        # Bind complete source identities locally. The provider projection
        # delivers only scientific facts and short references.
        facts["source_bindings"] = copy.deepcopy(evidence.get("source_bindings") or {})
        refs.extend(row["source_ref"] for row in facts["source_catalog"]
                    if isinstance(row, dict) and row.get("source_ref") in facts["source_bindings"])
        # New scientific delivery cites native evidence/artifact IDs and the
        # collector's short refs. Receipt paths stay in the local review/bindings.
        refs = [ref for ref in refs if not Path(ref).is_absolute()]
    diagnostics = {key: copy.deepcopy(payload[key]) for key in
        ("objective_semantic_findings", "objective_effectiveness_findings") if payload.get(key)}
    if diagnostics:
        facts["scientific_diagnostics"] = diagnostics
    facts["allowed_evidence_refs"] = list(dict.fromkeys(refs))
    return facts


def check_inputs(payload: dict) -> tuple[list[dict], list[str]]:
    """Requirements and allowed refs from a live or compiled (pointer-bearing) request."""
    from .review_context import _request_value
    requirements = _request_value(payload, "analysis_requirements") or []
    facts = _request_value(payload, "analysis_facts") or {}
    return [row for row in requirements if isinstance(row, dict)], list(facts.get("allowed_evidence_refs") or [])


def is_record(body: Any) -> bool:
    return isinstance(body, dict) and body.get("record_version") == RECORD_VERSION


def parse_analysis_reply(body: dict) -> tuple[dict, dict | None]:
    """A fresh reply, or the stored record a cached task returns (only its reply part is validated)."""
    from .schemas import AnalystReply
    if is_record(body):
        return AnalystReply.model_validate(body["analysis"]).model_dump(), body
    return AnalystReply.model_validate({key: value for key, value in body.items() if key not in _STRIP}).model_dump(), None


def _blank(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def analysis_delivery_check(reply: dict, requirements: list[dict], allowed: list[str]) -> dict:
    """Structural delivery of the analysis, not proof that its science is correct."""
    errors = []
    def error(kind, loc, msg):
        errors.append({"type": kind, "loc": loc, "msg": msg})
    findings = reply.get("findings") if isinstance(reply.get("findings"), list) else []
    if not findings:
        error("analysis_findings_missing", ["findings"], "At least one finding is required; explain insufficient evidence in a finding.")
    if _blank(reply.get("summary_zh")):
        error("analysis_summary_missing", ["summary_zh"],
            "summary_zh is required: intervention, key numbers, hypothesis assessment, scope and unknowns.")
    for index, item in enumerate(reply.get("limitations") or []):
        if _blank(item):
            error("analysis_text_blank", ["limitations", index], "Remove blank limitations; [] is valid.")
    declared = {row.get("requirement_id"): row for row in requirements}
    allowed_set = set(allowed)
    answered, refs_ok = {}, {}
    for index, finding in enumerate(findings):
        complete = True
        for key in _TEXT:
            if _blank(finding.get(key)):
                complete = False
                error("analysis_text_blank", ["findings", index, key], f"{key} must be non-empty text.")
        if finding.get("next_step") is not None and _blank(finding.get("next_step")):
            error("analysis_text_blank", ["findings", index, "next_step"], "Use null when there is no next step.")
        valid = True
        for position, ref in enumerate(finding.get("evidence_refs") or []):
            if ref not in allowed_set:
                valid = False
                error("analysis_evidence_ref_not_delivered", ["findings", index, "evidence_refs", position],
                      f"{ref} is not a delivered evidence reference; use only: {sorted(allowed_set)}")
        for position, rid in enumerate(dict.fromkeys(finding.get("requirement_ids") or [])):
            if rid not in declared:
                error("analysis_requirement_id_unknown", ["findings", index, "requirement_ids", position],
                      f"{rid} is not an analysis_requirements ID; use only: {sorted(declared)}")
                continue
            answered.setdefault(rid, []).append((index, complete))
            refs_ok[rid] = refs_ok.get(rid, True) and valid
    coverage, missing = [], []
    for rid, row in declared.items():
        rows = answered.get(rid, [])
        present = any(complete for _, complete in rows)
        refs_valid = bool(rows) and refs_ok.get(rid, False)
        coverage.append({"requirement_id": rid, "finding_indices": [index for index, _ in rows],
                         "response_present": present, "references_valid": refs_valid})
        if row.get("core", True) and not (present and refs_valid):
            missing.append(rid)
            if not rows:
                error("analysis_core_requirement_unanswered", ["findings"],
                      f"No finding lists core requirement {rid}: {row.get('expected_behavior', '')}")
    return {"status": "incomplete" if errors else "complete", "requirement_coverage": coverage,
            "missing_requirement_ids": missing, "errors": errors}


def failure_check(requirements: list[dict], reason: str) -> dict:
    """Delivery report when no valid reply exists (role or format failure)."""
    return {"status": "incomplete", "requirement_coverage": [], "errors": [{"type": "analyst_reply_unavailable", "msg": reason}],
            "missing_requirement_ids": [row.get("requirement_id") for row in requirements if row.get("core", True)]}


def materialize_analysis_record(facts: dict, reply: dict | None, check: dict, *, summary_zh: str | None = None) -> dict:
    """Bind native facts, the Analyst reply and its delivery check; refs are the valid cited union."""
    allowed = set(facts.get("allowed_evidence_refs") or [])
    refs = [ref for finding in (reply or {}).get("findings") or [] for ref in finding.get("evidence_refs") or []
            if ref in allowed]
    return {"record_version": RECORD_VERSION, "native_facts": copy.deepcopy(facts), "analysis": copy.deepcopy(reply),
            "delivery_check": copy.deepcopy(check), "evidence_refs": list(dict.fromkeys(refs)),
            "summary_zh": (reply or {}).get("summary_zh") or summary_zh or ""}


def analysis_memory_view(record: dict) -> dict:
    """Science view for Observation, Designer and reports; the delivery check stays local."""
    reply = record.get("analysis") or {}
    findings = copy.deepcopy(reply.get("findings") or [])
    return {"hypothesis_assessment": reply.get("hypothesis_assessment"), "summary_zh": record.get("summary_zh") or "",
            "findings": findings, "limitations": copy.deepcopy(reply.get("limitations") or []),
            "next_steps": [item["next_step"] for item in findings if item.get("next_step")],
            "evidence_refs": copy.deepcopy(record.get("evidence_refs") or []),
            "native_facts": copy.deepcopy(record.get("native_facts") or {})}
