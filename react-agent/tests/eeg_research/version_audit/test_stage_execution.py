"""Stage acceptance on synthetic native receipts; no model, data or GPU calls."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic import handoffs, loop, method_suite, screening, selected_plan, worker
from react_agent.eeg_research.agentic.artifacts import file_digest, register
from react_agent.eeg_research.agentic.stage_reviews import (
    analysis_stage_payload, check_analysis_coverage, review_measurement,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")  # Only the injected Analyst is callable.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def forbidden(*args, **kwargs):
        pytest.fail("Stage acceptance attempted network, training or an implementation check")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(worker, "implement", forbidden)
    monkeypatch.setattr(worker.jobs, "start_job", forbidden)


def native_receipt(tmp_path):
    camp = tmp_path / "stage_campaign"
    camp.mkdir()
    experiment = {"design_contract_version": "v2", "spec_hash": "approved-stage-spec",
        "hypothesis": "The candidate can change the paired screening delta.",
        "implementation_requirements": [{"requirement_id": "code_only", "expected_behavior": "SOURCE_SECRET"}],
        "measurement_requirements": [
            {"requirement_id": "identity", "verification": "run_identity", "expected_behavior": "Use the approved source/spec."},
            {"requirement_id": "pairs", "verification": "paired_coverage", "expected_behavior": "Both selected matched folds exist."},
            {"requirement_id": "metrics", "verification": "metric_integrity", "expected_behavior": "Use native finite metrics."}],
        "analysis_requirements": [{"requirement_id": "explain", "verification": "scope_limits",
            "expected_behavior": "Interpret a negative screening within its two-fold scope."}],
        "model": {"SOURCE_SECRET": "HOOK_SECRET"}, "completion_condition": "FUTURE_TASK_SECRET"}
    result = {"status": "complete", "candidate_id": "c1", "suite_id": "synthetic_suite", "suite_hash": "suite-hash",
        "method_revision": "revision", "seed": 0, "fidelity": "full", "phase": "screening_2fold",
        "coverage": 2, "required_fold_count": 2, "screening_fold_ids": ["fold_07", "fold_10"],
        "screening_subjects": ["sub-07", "sub-10"], "selection_manifest_hash": "selection-hash",
        "baseline_suite_hash": "baseline-hash", "screening_result": "not_promoted", "scope": "two_fold_development",
        "science_scope": {"independent_final_test": False, "scope": "two_fold_development"},
        "screening_mean_top1": 0.05, "screening_mean_top5": 0.1, "screening_mean_delta_pp": -4.5,
        "scores": [{"fold_id": "fold_07", "held_out_subject": "sub-07", "seed": 0, "fidelity": "full"},
                   {"fold_id": "fold_10", "held_out_subject": "sub-10", "seed": 0, "fidelity": "full"}],
        "paired_scores": [{"subject": "sub-07", "candidate_top1": 0.04, "baseline_top1": 0.09, "delta_pp": -5.0},
                          {"subject": "sub-10", "candidate_top1": 0.06, "baseline_top1": 0.10, "delta_pp": -4.0}]}
    for score in result["scores"]:
        score.update(train_subjects=["sub-01"], checkpoint_hash="checkpoint", freeze_ref="native-freeze", freeze_sha256="freeze-sha")
    receipt = camp / "suites/synthetic_suite/screening.json"
    loop._write(receipt, {"screening": result, "score_dependencies": []})
    artifact = register(camp, receipt, kind="method_screening", artifact_id="art_native_screening")
    latest = {"evidence_id": "ev_native_screening", "kind": "method_screening", "candidate_id": "c1",
        "suite_id": "synthetic_suite", "suite_hash": "suite-hash", "method_revision": "revision",
        "source_hash": "accepted-source-hash", "spec_hash": experiment["spec_hash"], "seed": 0, "fidelity": "full",
        "evaluation_valid": True, "execution_fingerprint": "execution", "suite_ref": str(receipt),
        "result_hash": file_digest(receipt), "screening": result, "contract_fingerprint": "contract",
        "artifact_refs": [artifact["artifact_id"]]}
    return camp, experiment, result, latest


def test_measurement_receipt_is_deterministic_bound_and_negative_is_valid(tmp_path):
    camp, spec, native, latest = native_receipt(tmp_path)
    first = review_measurement(camp, {}, latest, spec, verified_result=native)
    second = review_measurement(camp, {}, latest, spec, verified_result=native)
    assert first == second and first["status"] == "completed"
    assert len(first["requirement_coverage"]) == 3
    assert first["binding"]["source_hash"] == latest["source_hash"]
    assert first["binding"]["spec_hash"] == spec["spec_hash"]
    assert first["binding"]["receipt_sha256"] == file_digest(Path(latest["suite_ref"]))
    assert len((camp / "artifact_registry.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("change", ["spec", "receipt", "coverage"])
def test_measurement_core_failure_does_not_create_completed_acceptance(tmp_path, change):
    camp, spec, native, latest = native_receipt(tmp_path)
    if change == "spec":
        spec["spec_hash"] = "different-spec"
    elif change == "receipt":
        Path(latest["suite_ref"]).write_text('{"screening": {}}')
    else:
        native["paired_scores"] = native["paired_scores"][:1]
    result = review_measurement(camp, {}, latest, spec, verified_result=native)
    assert result["status"] == "blocked" and result["blocking_requirement_ids"]


def test_empty_stage_lists_do_not_bypass_native_receipt_identity(tmp_path):
    camp, spec, native, latest = native_receipt(tmp_path)
    spec.update(measurement_requirements=[], analysis_requirements=[], spec_hash="wrong-spec")
    measured = review_measurement(camp, {}, latest, spec, verified_result=native)
    assert measured["status"] == "blocked" and measured["requirement_coverage"] == []
    assert check_analysis_coverage({}, spec, measured)["status"] == "blocked"


def test_complete_ten_fold_receipt_uses_the_same_measurement_stage(tmp_path):
    camp, spec, native, latest = native_receipt(tmp_path)
    aggregate = {**native, "coverage": 10, "required_fold_count": 10, "mean_top1": 0.05,
        "scores": [{"fold_id": f"fold_{i:02d}"} for i in range(1, 11)]}
    receipt = camp / "suites/synthetic_suite/aggregate.json"
    loop._write(receipt, {"aggregate": aggregate, "score_dependencies": []})
    artifact = register(camp, receipt, kind="method_suite", artifact_id="art_native_aggregate")
    latest.update(kind="method_suite", suite_ref=str(receipt), result_hash=file_digest(receipt),
        aggregate=aggregate, comparison={"comparable": True}, artifact_refs=[artifact["artifact_id"]])
    assert review_measurement(camp, {}, latest, spec, verified_result=aggregate)["status"] == "completed"


@pytest.mark.parametrize("fidelity,mismatch", [("full", None), ("pilot", None),
    ("pilot", "seed"), ("pilot", "fidelity"), ("pilot", "protocol"), ("pilot", "declared_control")])
def test_single_run_stage_reuses_native_accept_job_without_training(tmp_path, monkeypatch, fidelity, mismatch):
    camp, spec, _, _ = native_receipt(tmp_path)
    from react_agent.eeg_research.agentic import execution_protocol, runner
    protocol = {"fingerprint": "single-run-protocol", "split_seed": 0,
        "gallery_image_ids": ["image1"], "validation_image_ids": ["image1"], "train_query_ids": ["train1"]}
    monkeypatch.setattr(execution_protocol, "load_protocol", lambda *_: protocol)
    rows, accepted_by_job = [], {}
    for name, target, metric in (("j1", "baseline", 0.1), ("j2", "c1", 0.05)):
        job = camp / "jobs" / name
        run_fidelity = "full" if target == "baseline" and mismatch == "fidelity" else fidelity
        seed = 1 if target == "baseline" and mismatch == "seed" else 0
        fingerprint = "other-protocol" if target == "baseline" and mismatch == "protocol" else protocol["fingerprint"]
        accepted = {"evaluation_valid": True, "fixed_bank_top1": metric, "fidelity": run_fidelity}
        loop._write(job / "job.json", {"status": "finished", "candidate_id": target,
            "manifest": {"entry_sha256": "accepted-source-hash"}, "result": accepted})
        loop._write(job / "frozen_run_spec.json", {"spec_hash": spec["spec_hash"], "seed": seed})
        loop._write(job / "source_binding.json", {"file_sha256": "accepted-source-hash"})
        accepted_by_job[job] = accepted
        rows.append({"evidence_id": "ev_" + name, "candidate_id": target, "job_id": name, "job_dir": str(job),
            "source_hash": "accepted-source-hash", "spec_hash": spec["spec_hash"], "seed": seed,
            "fidelity": run_fidelity, "evaluation_valid": True, "fixed_bank_top1": metric,
            "execution_fingerprint": fingerprint, "gallery_size": 200})
    rows[-1]["control_id"] = rows[0]["evidence_id"]
    if mismatch == "declared_control":
        # The valid baseline in evidence cannot replace the explicitly named pair.
        rows[-1]["control_id"] = "ev_missing_declared_control"
    calls = []
    def accept(job, manifest, fidelity, *, protocol):
        calls.append(job)
        assert manifest["entry_sha256"] == "accepted-source-hash" and fidelity == accepted_by_job[job]["fidelity"]
        return accepted_by_job[job]
    monkeypatch.setattr(runner, "accept_job", accept)
    reviewed = review_measurement(camp, {"evidence": rows}, rows[-1], spec)
    assert (reviewed["status"] == "completed") is (mismatch is None)
    assert len(calls) == (2 if mismatch is None else 1)
    if mismatch:
        assert reviewed["blocking_requirement_ids"] == ["pairs"]
        assert "control_receipt_sha256" not in reviewed["binding"]
    else:
        assert reviewed["binding"]["control_id"] == rows[0]["evidence_id"]
        assert reviewed["binding"]["control_receipt_sha256"] == file_digest(camp / "jobs/j1/job.json")
        assert rows[0]["evidence_id"] in reviewed["evidence_refs"]
        assert str((camp / "jobs/j1/job.json").resolve()) in reviewed["evidence_refs"]
    assert reviewed["binding"]["receipt_ref"] == str((camp / "jobs/j2/job.json").resolve())


@pytest.mark.parametrize("coverage", ["missing", "foreign", "valid_negative"])
def test_analysis_acceptance_requires_its_own_bound_coverage(tmp_path, coverage):
    camp, spec, native, latest = native_receipt(tmp_path)
    measured = review_measurement(camp, {}, latest, spec, verified_result=native)
    payload = analysis_stage_payload({"experiment": spec, "candidate_context": {"source": "SOURCE_SECRET"},
        "selected_research_task": {"question": "What does this negative screening show?",
            "scientific_intent": "Compare paired outcomes", "completion_condition": "FUTURE_TASK_SECRET"},
        "method_screening": native, "hypothesis": spec["hypothesis"]}, spec, measured)
    assert payload["review_stage"] == "analysis" and payload["analysis_requirements"] == spec["analysis_requirements"]
    assert all(text not in json.dumps(payload) for text in ("SOURCE_SECRET", "HOOK_SECRET", "FUTURE_TASK_SECRET"))
    reply = {"hypothesis_assessment": "weakened"}
    if coverage != "missing":
        reply["analysis_requirement_coverage"] = [{"requirement_id": "explain", "status": "implemented",
            "detail": "Negative two-fold development result does not establish general failure.",
            "evidence_refs": [latest["evidence_id"] if coverage == "valid_negative" else "foreign-result"]}]
    reviewed = check_analysis_coverage(reply, spec, measured)
    assert (reviewed["status"] == "completed") is (coverage == "valid_negative")


@pytest.mark.parametrize("coverage", ["valid", "missing", "foreign", "legacy", "single_missing"])
def test_native_worker_uses_one_analyst_and_only_complete_analysis_can_close(tmp_path, monkeypatch, coverage):
    camp, spec, native, latest = native_receipt(tmp_path)
    single_run = coverage == "single_missing"
    if single_run:
        coverage = "missing"
    if coverage == "legacy":
        spec.pop("measurement_requirements")
        spec.pop("analysis_requirements")
    goal = {"goal_id": camp.name, "evaluation_mode": "loso_method_search", "evaluation_schedule": "screen_then_confirm",
        "max_llm_calls": 20, "max_training_jobs": 4, "max_gpu_seconds": 100, "memory": {"enabled": False},
        "training_seeds": [0], "research_execution": {"version": "single_plan_v1", "design_contract_version": "v2",
            "paid_maintenance": "off"}}
    state = {"goal_id": camp.name, "status": "created", "evaluation_mode": "loso_method_search", "training_jobs": 0,
        "llm_calls": 0, "max_llm_calls": 20, "max_training_jobs": 4, "max_gpu_seconds": 100,
        "contract_fingerprint": "contract", "candidates": [{"candidate_id": "c1"}], "evidence": [latest],
        "experiment": spec, "memory": [], "decisions": [], "training_seeds": [0]}
    for name, value in (("goal.json", goal), ("campaign_state.json", state),
                        ("evaluation_contract.json", {"fingerprint": "contract"}),
                        ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        loop._write(camp / name, value)
    monkeypatch.setattr(method_suite, "verified_suite_records", lambda *args: [])
    monkeypatch.setattr(screening, "verified_screening_records", lambda *args: [latest])
    native_reads = []
    def read_native(*args):
        native_reads.append(args)
        assert file_digest(Path(latest["suite_ref"])) == latest["result_hash"]
        return copy.deepcopy(native)
    monkeypatch.setattr(screening, "read_screening", read_native)
    if single_run:
        from react_agent.eeg_research.agentic import analysis_dependencies, development_feedback, execution_protocol, runner
        # Use the older action flow: it has no single-plan return before curator.
        goal.pop("research_execution")
        goal["evaluation_mode"] = state["evaluation_mode"] = "single_target"
        spec["measurement_requirements"] = [row for row in spec["measurement_requirements"]
            if row["verification"] != "paired_coverage"]
        job = camp / "jobs/j1"
        accepted = {"evaluation_valid": True, "fixed_bank_top1": 0.05, "fidelity": "full"}
        loop._write(job / "job.json", {"status": "finished", "candidate_id": "c1", "manifest": {}, "result": accepted})
        loop._write(job / "frozen_run_spec.json", {"spec_hash": spec["spec_hash"], "seed": 0})
        loop._write(job / "source_binding.json", {"file_sha256": latest["source_hash"]})
        latest.update(kind="run", job_id="j1", job_dir=str(job), fixed_bank_top1=0.05)
        loop._write(camp / "goal.json", goal)
        def accept(*args, **kwargs):
            native_reads.append(args)
            return copy.deepcopy(accepted)
        monkeypatch.setattr(runner, "accept_job", accept)
        monkeypatch.setattr(execution_protocol, "load_protocol", lambda *args: {})
        monkeypatch.setattr(development_feedback, "ensure_development_feedback", lambda *args: {"status": "verified"})
        monkeypatch.setattr(development_feedback, "_ensure_report", lambda *args: None)
        monkeypatch.setattr(analysis_dependencies, "analysis_targets", lambda *args: [latest])
    monkeypatch.setattr(worker, "load_protocol", lambda *args: {})
    monkeypatch.setattr(handoffs, "candidate_context", lambda *args: {"source": "SOURCE_SECRET", "hooks": "HOOK_SECRET"})
    monkeypatch.setattr(handoffs, "analysis_experiment_binding", lambda *args: {"status": "verified",
        "experiment": spec, "hypothesis": spec["hypothesis"]})
    monkeypatch.setattr(handoffs, "verified_encoder_structural_facts", lambda *args, **kwargs: [])
    from react_agent.eeg_research.agentic import measurement_context, research_progress
    monkeypatch.setattr(measurement_context, "evaluation_population_context", lambda *args: {})
    for name in ("verified_training_diagnostic_facts", "objective_effectiveness_findings", "objective_semantic_findings", "verified_diagnostic_facts"):
        monkeypatch.setattr(research_progress, name, lambda *args, **kwargs: [])
    proposal = {"question": "What does this negative screening show?", "scientific_intent": "Interpret paired two-fold outcomes",
        "evaluation_scope": {"phase": "screening_2fold"}, "expected_observation": "FUTURE_TASK_SECRET",
        "completion_condition": "FUTURE_TASK_SECRET"}
    monkeypatch.setattr(selected_plan, "load_plan", lambda *args: {"plan_id": "selected-stage-plan", "candidate_id": "c1", "proposal": proposal})
    calls = []
    def analyst(payload):
        calls.append(copy.deepcopy(payload))
        if coverage == "legacy":
            assert "review_stage" not in payload and "measurement_review" not in payload
            assert "candidate_context" in payload and "completion_condition" in payload["selected_research_task"]
        else:
            assert payload["measurement_review"]["status"] == "completed"
            assert all(text not in json.dumps(payload) for text in ("SOURCE_SECRET", "HOOK_SECRET", "FUTURE_TASK_SECRET"))
        reply = {"hypothesis_assessment": "weakened", "summary_zh": "负筛查结果仅适用于本次两折。"}
        if coverage not in {"missing", "legacy"}:
            reply["analysis_requirement_coverage"] = [{"requirement_id": "explain", "status": "implemented",
                "detail": "Only two fixed development folds were measured.",
                "evidence_refs": [latest["evidence_id"] if coverage == "valid" else "foreign-result"]}]
        return reply
    backend_roles = []
    def backend(_camp, role, **kwargs):
        backend_roles.append(role)
        return analyst
    monkeypatch.setattr(worker, "role_backend", backend)
    worker.build_services(camp)["analyze"](camp, state)
    assert len(calls) == 1 and native_reads
    assert backend_roles == ["result_analyst"]
    analysis = next(row for row in state["evidence"] if row.get("kind") == "analysis")
    assert (analysis["status"] == "completed") is (coverage in {"valid", "legacy"})
    assert bool(selected_plan._analysis_ref(state, latest["evidence_id"])) is (coverage in {"valid", "legacy"})
    if coverage == "legacy":
        assert "analysis_stage_review_ref" not in analysis and not (camp / "stage_reviews").exists()
        return
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    review_artifact = resolve_verified_artifact(camp, analysis["analysis_stage_review_ref"])
    saved = json.loads(Path(review_artifact["path"]).read_text())
    assert saved["binding"]["analysis_artifact_id"] == analysis["analysis_artifact_id"]
    assert saved["binding"]["source_hash"] == latest["source_hash"]
    assert saved["binding"]["spec_hash"] == latest["spec_hash"]
    from react_agent.eeg_research.agentic import llm
    from react_agent.eeg_research.agentic.embedding import memory_config
    analysis_artifact = resolve_verified_artifact(camp, analysis["analysis_artifact_id"])
    envelope = json.loads(Path(analysis_artifact["path"]).read_text())
    system = llm.system_prompt("result_analyst", camp=camp, memory=memory_config(camp), review_stage="analysis")
    assert envelope["prompt_hash"] == hashlib.sha256(system.encode()).hexdigest()[:16]
    assert state["training_jobs"] == 0
    assert ([path.name for path in (camp / "jobs").iterdir()] == ["j1"] if single_run else not (camp / "jobs").exists())
