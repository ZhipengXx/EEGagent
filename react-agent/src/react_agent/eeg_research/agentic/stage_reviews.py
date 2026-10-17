"""Native measurement acceptance and source-bound analysis coverage.

The caller supplies the suite result reconstructed by the existing native
screening/suite verifier. No model, training or implementation check runs here.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .action_planning import digest
from .artifacts import file_digest, lookup, register
from .stage_contract import requirements_for_stage


def _save(camp: Path, report: dict) -> dict:
    """Publish one deterministic review and reuse its exact registered bytes."""
    report = copy.deepcopy(report)
    report["review_id"] = "stage_review_" + digest(report)[:32]
    path = Path(camp) / "stage_reviews" / (report["review_id"] + ".json")
    from .loso_study import write_once
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise ValueError("stage_review_identity_changed")
    else:
        write_once(path, report)
    artifact_id = "art_" + report["review_id"]
    artifact = lookup(camp, artifact_id)
    if artifact is None:
        artifact = register(camp, path, kind=report["stage"] + "_review",
            candidate_id=report["binding"]["candidate_id"], artifact_id=artifact_id)
    elif artifact["sha256"] != file_digest(path):
        raise ValueError("stage_review_artifact_changed")
    return {**report, "artifact_id": artifact_id, "path": str(path.resolve()), "sha256": artifact["sha256"]}


def review_measurement(camp: Path, state: dict, latest: dict, experiment: dict, *,
                       verified_result: dict | None = None) -> dict:
    """Accept only the exact completed native receipt for this source/spec."""
    if latest.get("kind") not in {"method_screening", "method_suite"}:
        return _review_single_measurement(camp, state, latest, experiment)
    if verified_result is None:
        if latest.get("kind") == "method_screening":
            from .screening import read_screening
            verified_result = read_screening(camp, latest["suite_id"])
        else:
            from .method_suite import read_aggregate
            verified_result = read_aggregate(camp, latest["suite_id"])
    requirements = requirements_for_stage(experiment, "measurement")
    receipt = Path(latest["suite_ref"])
    receipt_sha = file_digest(receipt)
    wrapper = json.loads(receipt.read_text(encoding="utf-8"))
    result_key = "screening" if latest.get("kind") == "method_screening" else "aggregate"
    native = verified_result
    binding = {key: latest.get(key) for key in ("evidence_id", "candidate_id", "source_hash", "spec_hash",
        "suite_id", "suite_hash", "method_revision", "seed", "fidelity", "result_hash")}
    binding.update(receipt_ref=str(receipt.resolve()), receipt_sha256=receipt_sha)
    identity_ok = bool(latest.get("evaluation_valid") is True and latest.get("source_hash")
        and latest.get("spec_hash") == experiment.get("spec_hash")
        and latest.get("result_hash") == receipt_sha and wrapper.get(result_key) == native
        and latest.get(result_key) == native
        and all(native.get(key) == latest.get(key) for key in ("candidate_id", "suite_hash", "method_revision", "seed", "fidelity")))
    complete = native.get("status") == "complete"
    required = native.get("required_fold_count")
    covered = native.get("coverage")
    paired = (len(native.get("paired_scores") or []) == required if result_key == "screening"
        else (latest.get("comparison") or {}).get("comparable") is True)
    coverage_ok = bool(complete and type(required) is int and required > 0 and covered == required
        and len(native.get("scores") or []) == required
        and (latest.get("candidate_id") == "baseline" or paired))
    metric = native.get("screening_mean_top1" if result_key == "screening" else "mean_top1")
    metric_ok = bool(complete and type(metric) in (int, float) and math.isfinite(metric) and 0 <= metric <= 1)
    outcomes = {
        "run_identity": (identity_ok, "Native receipt matches the accepted candidate, source, spec and result bytes."),
        "paired_coverage": (identity_ok and coverage_ok, "Native completed coverage includes the required matched folds."),
        "metric_integrity": (identity_ok and metric_ok, "Native verifier reconstructed valid finite metrics and their aggregation; a negative delta remains valid."),
    }
    refs = [str(latest["evidence_id"]), str(receipt.resolve()), *(latest.get("artifact_refs") or [])]
    rows = []
    for requirement in requirements:
        ok, detail = outcomes[requirement["verification"]]
        rows.append({"requirement_id": requirement["requirement_id"],
            "status": "implemented" if ok else "unverified", "detail": detail if ok else
                "The declared measurement condition lacks an exact verified native receipt.",
            "evidence_refs": refs if ok else []})
    missing = [requirement["requirement_id"] for requirement, row in zip(requirements, rows)
        if requirement.get("core", True) and row["status"] != "implemented"]
    return _save(camp, {"schema_version": "eeg_research.stage_review.v1", "stage": "measurement",
        "authority": "native_verified_measurement", "status": "completed" if identity_ok and complete and not missing else "blocked",
        "binding": binding, "requirements_digest": digest(requirements), "requirement_coverage": rows,
        "blocking_requirement_ids": missing, "evidence_refs": refs})


def analysis_stage_payload(payload: dict, experiment: dict, measurement_review: dict) -> dict:
    """Deliver scientific inputs and verified measurements, without code tasks."""
    keys = ("analysis_trigger", "latest", "comparison", "diagnostics", "hypothesis", "run_identity",
        "development_feedback", "controls", "evaluation_population", "independent_seed_count",
        "candidate_seed_coverage", "matched_run_history", "verified_run_coverage", "verified_development_facts",
        "verified_training_diagnostics", "verified_encoder_structural_facts", "objective_effectiveness_findings",
        "objective_semantic_findings", "method_suite", "method_screening", "evaluation_mode",
        "primary_metric", "scientific_unit", "science_scope", "limitations", "benchmark_permission")
    result = {key: copy.deepcopy(payload[key]) for key in keys if key in payload}
    selected = payload.get("selected_research_task") or {}
    scientific = {key: copy.deepcopy(selected[key]) for key in ("question", "scientific_intent") if key in selected}
    result.update(review_stage="analysis", scientific_question=scientific.get("question"),
        scientific_intent=scientific.get("scientific_intent") or experiment.get("scientific_intent"),
        analysis_requirements=requirements_for_stage(experiment, "analysis"),
        measurement_review=copy.deepcopy(measurement_review),
        evidence_refs=list(dict.fromkeys([measurement_review["artifact_id"], measurement_review["path"],
            *measurement_review["evidence_refs"]])))
    if scientific:
        result["selected_research_task"] = scientific
    if isinstance(payload.get("hypothesis_binding"), dict):
        result["hypothesis_binding"] = {key: copy.deepcopy(payload["hypothesis_binding"][key])
            for key in ("status", "source_hash", "spec_hash", "artifact_ref", "artifact_sha256", "missing_inputs")
            if key in payload["hypothesis_binding"]}
    return result


def _review_single_measurement(camp: Path, state: dict, latest: dict, experiment: dict) -> dict:
    from .comparison import compare_runs, matched_control
    from .execution_protocol import load_protocol
    from .runner import accept_job
    job = Path(latest["job_dir"])
    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    protocol = load_protocol(camp)
    native = accept_job(job, record.get("manifest"), latest["fidelity"], protocol=protocol)
    frozen = json.loads((job / "frozen_run_spec.json").read_text(encoding="utf-8"))
    source = json.loads((job / "source_binding.json").read_text(encoding="utf-8"))
    identity_ok = bool(record.get("status") == "finished" and native.get("evaluation_valid") is True
        and record.get("result") == native and record.get("candidate_id") == latest.get("candidate_id")
        and frozen.get("spec_hash") == latest.get("spec_hash") == experiment.get("spec_hash")
        and source.get("file_sha256") == latest.get("source_hash")
        and frozen.get("seed") == latest.get("seed")
        and native.get("fixed_bank_top1") == latest.get("fixed_bank_top1"))
    declared_controls = [row for row in state.get("evidence") or []
        if row.get("evidence_id") == latest.get("control_id")]
    control = matched_control(declared_controls, latest, protocol)
    comparison = compare_runs(candidate=latest, control=control, protocol=protocol)
    paired = latest.get("candidate_id") == "baseline"
    control_binding, control_refs = {}, []
    if control and control.get("job_dir") and comparison.get("comparable"):
        control_job = Path(control["job_dir"])
        control_receipt = control_job / "job.json"
        control_record = json.loads(control_receipt.read_text(encoding="utf-8"))
        accepted = accept_job(control_job, control_record.get("manifest"), control["fidelity"], protocol=protocol)
        paired = bool(control_record.get("status") == "finished" and accepted.get("evaluation_valid") is True
            and control_record.get("result") == accepted and accepted.get("fixed_bank_top1") == control.get("fixed_bank_top1"))
        if paired:
            control_binding = {"control_id": control["evidence_id"], "control_receipt_ref": str(control_receipt.resolve()),
                "control_receipt_sha256": file_digest(control_receipt), "control_accepted_result_digest": digest(accepted)}
            control_refs = [control["evidence_id"], str(control_receipt.resolve()), *(control.get("artifact_refs") or [])]
    metric = native.get("fixed_bank_top1")
    finite = type(metric) in (int, float) and math.isfinite(metric) and 0 <= metric <= 1
    conditions = {"run_identity": identity_ok, "paired_coverage": identity_ok and paired,
                  "metric_integrity": identity_ok and finite}
    receipt = job / "job.json"
    refs = [latest["evidence_id"], str(receipt.resolve()), *(latest.get("artifact_refs") or []), *control_refs]
    requirements = requirements_for_stage(experiment, "measurement")
    rows = [{"requirement_id": row["requirement_id"],
        "status": "implemented" if conditions[row["verification"]] else "unverified",
        "detail": "Native accepted finished run verifies this condition." if conditions[row["verification"]]
            else "The declared condition lacks a matching accepted finished run.",
        "evidence_refs": refs if conditions[row["verification"]] else []} for row in requirements]
    missing = [req["requirement_id"] for req, row in zip(requirements, rows)
               if req.get("core", True) and row["status"] != "implemented"]
    binding = {key: latest.get(key) for key in ("evidence_id", "candidate_id", "job_id", "source_hash", "spec_hash",
        "seed", "fidelity", "execution_fingerprint")}
    binding.update(receipt_ref=str(receipt.resolve()), receipt_sha256=file_digest(receipt),
                   accepted_result_digest=digest(native), **control_binding)
    return _save(camp, {"schema_version": "eeg_research.stage_review.v1", "stage": "measurement",
        "authority": "native_verified_measurement", "status": "completed" if identity_ok and not missing else "blocked",
        "binding": binding, "requirements_digest": digest(requirements), "requirement_coverage": rows,
        "blocking_requirement_ids": missing, "evidence_refs": refs})


def analysis_reply_errors(body: dict, payload: dict) -> list[dict]:
    """Format errors the Analyst can correct in-call instead of producing a degraded artifact.

    Reference/ID shape would otherwise make check_analysis_coverage block the stage;
    an empty summary_zh would otherwise reach the Observation and the next Planner.
    Marking a requirement unverified is a scientific judgement and is not reported.
    """
    review = payload.get("measurement_review")
    if payload.get("review_stage") != "analysis" or not isinstance(review, dict):
        return []
    allowed = {review.get("artifact_id"), review.get("path"), *(review.get("evidence_refs") or [])} - {None}
    declared = {row["requirement_id"] for row in payload.get("analysis_requirements") or []}
    errors, seen = [], set()
    if not str(body.get("summary_zh") or "").strip():
        errors.append({"type": "analysis_summary_missing", "loc": ["summary_zh"],
            "msg": "summary_zh is required: intervention, main result with exact numbers, hypothesis assessment, scope and unknowns"})
    for index, row in enumerate(body.get("analysis_requirement_coverage") or []):
        rid = row.get("requirement_id")
        if rid in seen or (declared and rid not in declared):
            errors.append({"type": "analysis_requirement_id_invalid", "loc": ["analysis_requirement_coverage", index, "requirement_id"],
                "msg": f"{rid} must be a unique ID from analysis_requirements"})
        seen.add(rid)
        if row.get("status") != "implemented":
            continue
        for ref in row.get("evidence_refs") or []:
            if ref not in allowed:
                errors.append({"type": "analysis_evidence_ref_not_delivered", "loc": ["analysis_requirement_coverage", index, "evidence_refs"],
                    "msg": f"{ref} is not a delivered evidence reference; use only: {sorted(allowed)}"})
    return errors


def check_analysis_coverage(reply: dict, experiment: dict, measurement_review: dict) -> dict:
    """Bind the Analyst's stage claims to the delivered measurement evidence."""
    requirements = requirements_for_stage(experiment, "analysis")
    rows = reply.get("analysis_requirement_coverage") or []
    allowed = {measurement_review["artifact_id"], measurement_review["path"], *measurement_review["evidence_refs"]}
    ids = [row["requirement_id"] for row in rows]
    declared = {row["requirement_id"] for row in requirements}
    invalid = len(ids) != len(set(ids)) or bool(set(ids) - declared)
    missing = []
    for requirement in requirements:
        row = next((item for item in rows if item["requirement_id"] == requirement["requirement_id"]), None)
        supported = bool(row and row["status"] == "implemented" and row.get("evidence_refs")
            and all(ref in allowed for ref in row["evidence_refs"]))
        if requirement.get("core", True) and not supported:
            missing.append(requirement["requirement_id"])
    return {"schema_version": "eeg_research.stage_review.v1", "stage": "analysis",
        "authority": "analyst_coverage_bound_to_native_measurement",
        "status": "completed" if measurement_review["status"] == "completed" and not invalid and not missing else "blocked",
        "binding": copy.deepcopy(measurement_review["binding"]),
        "measurement_review_id": measurement_review["review_id"], "requirements_digest": digest(requirements),
        "requirement_coverage": copy.deepcopy(rows), "blocking_requirement_ids": missing,
        "invalid_coverage_ids": invalid}


def save_analysis_review(camp: Path, review: dict, *, artifact_id: str) -> dict:
    artifact = lookup(camp, artifact_id)
    if not artifact or artifact["sha256"] != file_digest(Path(artifact["path"])):
        raise ValueError("analysis_stage_artifact_unavailable")
    report = copy.deepcopy(review)
    report["binding"].update(analysis_artifact_id=artifact_id, analysis_sha256=artifact["sha256"])
    return _save(camp, report)
