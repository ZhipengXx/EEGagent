"""Native terminal review failures reach the next Planner without measured claims."""
from __future__ import annotations

import json

import pytest

from test_reviewer_output_revalidation import fixture_campaign, sdk
from react_agent.eeg_research.agentic import experiment_observations as cards, loop, native_patch, selected_plan
from react_agent.eeg_research.agentic.binding import file_sha256
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, ReadRequest, Ready, canonical


REQUIREMENT = {"requirement_id": "matched_control_arm_and_fold_identity", "kind": "implementation", "core": True,
    "expected_behavior": "A baseline arm is trained and scored in this task on the same folds before approval, with per-fold scores recorded.",
    "code_locations": ["EEGCandidate.build_encoder"], "verification": "source_review"}
WHY = "No training occurred: training_jobs 0; there are no candidate/control fold scores or trained module receipts. Those require a native run after review."


def blocked_review(tmp_path, monkeypatch):
    camp = fixture_campaign(tmp_path, requirement=REQUIREMENT)
    workspace = camp / "candidates/c1"
    state = loop.load_state(camp)
    spec = loop._read(workspace / "spec.json")
    attempt = loop._read(workspace / "attempt.json")
    class Reviewer:
        def __call__(self, request):
            identity = request["requirement_evidence_identity"]
            return {"status": "ready", "issues": [], "review_limits": "No measured outcome before a native run.",
                "summary_zh": "训练前无法提供匹配两臂的逐折结果。", "intervention_coverage": {"principal": {"status": "implemented"}},
                "requirement_coverage": [{"requirement_id": REQUIREMENT["requirement_id"], "status": "unverified",
                    "source_hash": identity["source_hash"], "source_refs": [identity["source_ref"]],
                    "check_ref": identity["check_ref"], "check_sha256": identity["check_sha256"], "detail": WHY}]}
    verdict = native_patch.review(workspace, spec, {}, Reviewer(), "offline", identity={
        "candidate_id": "c1", "attempt_id": attempt["attempt_id"], "operation_id": "original_review_operation",
        "phase": "review_candidate", "input_hash": file_sha256(workspace / "extension/eeg_candidate.py")})
    assert verdict["status"] == "blocked" and verdict["blocking_remaining"] == 1
    state.update(experiment_failed=True, candidate_ready=False, status="created")
    state["candidates"][0]["status"] = "review_blocked"
    loop.save_state(camp, state)
    plan = selected_plan.load_plan(camp, state)
    assert not plan.get("source_hash")  # A failed implementation never reached measure.
    return camp, state, plan


def test_native_failed_plan_delivers_the_actual_source_bound_blocker_to_prepared_planner(tmp_path, sdk, monkeypatch):
    camp, state, plan = blocked_review(tmp_path, monkeypatch)
    workspace = camp / "candidates/c1"
    inputs = [workspace / name for name in ("spec.json", "checks.json", "review.json", "attempt.json", "implementation.json", "extension/eeg_candidate.py")]
    inputs += [camp / name for name in ("goal.json", "cost.json", "llm_calls.jsonl", "llm_api_intents.jsonl", "api_operation_attempts.jsonl")]
    originals = {path: path.read_bytes() for path in inputs}
    selected_plan._finish(camp, state, plan, "failed", detail="engineering_implementation_failed")
    saved = loop._read(camp / "experiment_observations" / (state["terminal_observation_id"] + ".json"))
    failure = saved["engineering_failure"]
    issue = failure["blocking_issues"][0]
    assert issue["requirement_id"] == REQUIREMENT["requirement_id"]
    assert issue["kind"] == "core_implementation_unverified" and issue["error_class"] == "semantic_evidence_missing_or_stale"
    assert issue["stage"] == "review_candidate" and issue["coverage_status"] == "unverified"
    assert issue["coverage_detail"] == WHY and issue["expected"] == REQUIREMENT["expected_behavior"]
    assert issue["source_hash"] == file_sha256(workspace / "extension/eeg_candidate.py")
    assert failure["verified_failure_source"]["check_sha256"] == file_sha256(workspace / "checks.json")
    assert saved["native_facts"]["metrics"] is None and saved["observation_kind"] == "engineering_failure"
    assert saved["source_refs"] and len(sdk) == 2 and state["training_jobs"] == 0
    store = cards.ExperimentObservationStore(camp)
    assert store.records(limit=1, observation_id=saved["observation_id"])[0]["body"] == saved
    handle = issue["original_issue"]
    read = store.registry.read(ReadRequest(**handle["next_read"]), role="research_planner")
    assert read.status == "ready" and read.content["actual"]["detail"] == WHY
    context = {**cards.observation_conditions(saved), "terminal_observation_id": saved["observation_id"]}
    result = ContextService(camp).compile(CompileRequest(camp.name, "research_planner", {
        "current_task_context": context, "terminal_observation_id": saved["observation_id"],
        "selected_plan": {"plan_id": plan["plan_id"], "terminal_observation_id": saved["observation_id"]}}, system="Choose the next task against native failed feedback."))
    assert isinstance(result, Ready) and result.prepared.verify()
    delivered = result.prepared.payload["experiment_observations"]["cards"][0]
    assert delivered["current_feedback"] is True
    assert delivered["body"]["engineering_failure"]["blocking_issues"][0]["coverage_detail"] == WHY
    assert len(canonical(delivered)) < 16000 and len(sdk) == 2
    for path, value in originals.items():
        assert path.read_bytes() == value, path


@pytest.mark.parametrize("changed", ["source", "checks", "spec", "attempt"])
def test_failure_feedback_does_not_adopt_a_stale_review_identity(tmp_path, sdk, monkeypatch, changed):
    camp, state, plan = blocked_review(tmp_path, monkeypatch)
    workspace = camp / "candidates/c1"
    if changed == "source":
        path = workspace / "extension/eeg_candidate.py"
        path.write_text(path.read_text() + "\n ")
    else:
        path = workspace / ({"checks": "checks.json", "spec": "spec.json", "attempt": "attempt.json"}[changed])
        payload = loop._read(path)
        if changed == "checks":
            payload["check_fingerprint"] = "stale"
        elif changed == "spec":
            payload["experiment"]["implementation_requirements"][0]["expected_behavior"] = "different promise"
        else:
            payload["attempt_id"] = "different_attempt"
        loop._write(path, payload)
    saved = cards.record_experiment_observation(camp, state, plan, "failed")
    assert saved["native_facts"]["metrics"] is None and not saved["engineering_failure"].get("blocking_issues")
    assert saved["engineering_failure"]["failure_feedback_status"] == "current_review_identity_mismatch"
    assert not saved["source_refs"] and len(sdk) == 2
