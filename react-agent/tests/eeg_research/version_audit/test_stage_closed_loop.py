"""Three-stage acceptance through real SDK transport and native synthetic CPU jobs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import test_closed_loop_evaluation as closed_loop
from test_closed_loop_evaluation import sdk_transport

from react_agent.eeg_research.agentic import closed_loop_evaluation as evaluation
from react_agent.eeg_research.agentic import loop
from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact


def test_staged_sdk_mock_cpu_experiment_closes_and_delivers_next_planner(tmp_path, monkeypatch, sdk_transport):
    import torch
    torch.set_num_threads(1)
    original_goal = evaluation._goal
    original_response = closed_loop.response
    delivered = []

    def staged_goal(max_api_calls):
        goal = original_goal(max_api_calls)
        goal["research_execution"].update(design_contract_version="v2", implementation_review_mode="single_pass")
        goal["role_models"]["candidate_reviewer"].update(thinking=False, reasoning_effort=None)
        return goal

    def staged_response(role, payload):
        delivered.append((role, payload))
        if role == "experiment_designer":
            return {"status": "completed", "design_brief": {
                "hypothesis": "Synthetic temporal means give an executable controlled encoder.",
                "principal_intervention": "Average the temporal axis before linear projection.",
                "hook_changes": {"model": {"mode": "replace", "config": {"z_dim": 1024}},
                                 "objective": {"mode": "inherit"}, "transform": {"mode": "inherit"}},
                "required_capability_ids": ["build_encoder"],
                "implementation_requirements": [{"requirement_id": "temporal_mean", "core": True,
                    "expected_behavior": "Average the temporal axis before linear projection.",
                    "verification": "source_review"}],
                "measurement_requirements": [
                    {"requirement_id": "run_identity", "expected_behavior": "Verify the accepted source and spec.",
                     "verification": "run_identity"},
                    {"requirement_id": "matched_coverage", "expected_behavior": "Verify matched native pilot coverage.",
                     "verification": "paired_coverage"},
                    {"requirement_id": "native_metrics", "expected_behavior": "Use native valid finite metrics.",
                     "verification": "metric_integrity"}],
                "analysis_requirements": [
                    {"requirement_id": "grounded", "expected_behavior": "Ground observations in measured evidence.",
                     "verification": "evidence_grounding"},
                    {"requirement_id": "paired", "expected_behavior": "Interpret the matched pilot comparison.",
                     "verification": "paired_comparison"},
                    {"requirement_id": "scope", "expected_behavior": "Restrict conclusions to synthetic pilot infrastructure.",
                     "verification": "scope_limits"},
                    {"requirement_id": "hypothesis", "expected_behavior": "State what the pilot tests and does not test.",
                     "verification": "hypothesis_assessment"}],
                "expected_observation": "One matched native synthetic pilot measurement.",
                "completion_condition": "One measured and analyzed native observation, without an efficacy claim.",
            }}
        reply = original_response(role, payload)
        if role == "candidate_coder" and reply.get("tool") == "finish_patch":
            reply["args"]["implementation_map"] = {"temporal_mean": {"source_refs": [
                "extension/eeg_candidate.py::TemporalMeanEncoder.forward",
                "extension/eeg_candidate.py::EEGCandidate.build_encoder"]}}
        if role == "result_analyst":
            assert payload["review_stage"] == "analysis"
            assert payload["measurement_review"]["status"] == "completed"
            reply["analysis_requirement_coverage"] = [{"requirement_id": row["requirement_id"],
                "status": "implemented", "detail": "The accepted matched pilot is synthetic infrastructure evidence only.",
                "evidence_refs": [payload["measurement_review"]["artifact_id"]]}
                for row in payload["analysis_requirements"]]
        return reply

    monkeypatch.setattr(evaluation, "_goal", staged_goal)
    monkeypatch.setattr(closed_loop, "response", staged_response)
    prepared, wire = sdk_transport
    report = evaluation.run_evaluation(tmp_path / "staged-loop", real_api=False, _sdk_mock=True,
        max_api_calls=18, max_ticks=2000, wall_timeout=180, poll_seconds=.1)
    assert report["ok"], report["failure"]
    assert report["actual_external_API_attempts"] == report["physical_GPU_jobs"] == report["physical_GPU_seconds"] == 0
    assert report["terminal_plan"]["status"] == "observation_ready"
    assert report["terminal_original_readback_verified"] and report["terminal_delivered_to_next_Planner"]
    assert report["next_selected_plan"]["current_step"] == "prepare" and not report["second_task_dispatched"]
    counts = Counter(role for role, _ in delivered)
    assert counts == {"research_planner": 2, "experiment_designer": 1, "candidate_coder": 2,
                      "candidate_reviewer": 1, "result_analyst": 1}
    assert report["native_API_attempts"] == len(prepared) == len(wire) == 7
    for request, body in zip(prepared, wire):
        assert request.verify() and body["messages"] == list(request.messages)
    for role, payload in delivered:
        if role in {"candidate_coder", "candidate_reviewer"}:
            assert payload["review_stage"] == "implementation"
            experiment = payload["experiment_spec"].get("experiment") or payload["experiment_spec"]
            assert {row["requirement_id"] for row in experiment["implementation_requirements"]} == {"temporal_mean"}
            assert "measurement_requirements" not in experiment and "analysis_requirements" not in experiment
        elif role == "result_analyst":
            assert "candidate_context" not in payload and "implementation_requirements" not in payload
            assert "source" not in payload

    camp = Path(report["campaign"])
    state = loop.load_state(camp)
    plan = report["terminal_plan"]
    measurement = next(row for row in state["evidence"] if row.get("evidence_id") == plan["measurement_evidence_id"])
    analysis = next(row for row in state["evidence"] if row.get("kind") == "analysis"
                    and row.get("run_evidence_id") == measurement["evidence_id"])
    assert analysis["status"] == analysis["analysis_stage_status"] == "completed"
    reviews = {}
    for stage, ref in (("measurement", analysis["measurement_review_ref"]),
                       ("analysis", analysis["analysis_stage_review_ref"])):
        artifact = resolve_verified_artifact(camp, ref)
        saved = json.loads(Path(artifact["path"]).read_text())
        assert saved["stage"] == stage and saved["status"] == "completed"
        assert saved["binding"]["source_hash"] == measurement["source_hash"]
        assert saved["binding"]["spec_hash"] == measurement["spec_hash"]
        assert saved["binding"]["evidence_id"] == measurement["evidence_id"]
        assert not saved["blocking_requirement_ids"]
        reviews[stage] = saved
    assert reviews["measurement"]["authority"] == "native_verified_measurement"
    assert len(reviews["measurement"]["requirement_coverage"]) == 3
    assert len(reviews["analysis"]["requirement_coverage"]) == 4
    assert reviews["analysis"]["measurement_review_id"] == reviews["measurement"]["review_id"]

    assert len(report["jobs"]) == 2
    assert {row["candidate_id"] for row in report["jobs"]} == {"baseline", plan["candidate_id"]}
    for row in report["jobs"]:
        assert row["status"] == "finished" and row["native_accepted_result"]["evaluation_valid"]
        job = camp / "jobs" / row["job_id"]
        assert (job / "frozen_run_spec.json").is_file() and (job / "source_binding.json").is_file()
    source = camp / "candidates" / plan["candidate_id"]
    assert json.loads((source / "review.json").read_text())["status"] == "ready"
    assert json.loads((source / "implementation_map.json").read_text())["approved_spec_hash"] == plan["spec_hash"]
    assert state["training_jobs"] == 2 and not state.get("live_job")
