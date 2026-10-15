"""Exercise stage separation through native review and the actual mocked SDK wire."""
import copy
import json

import pytest

from react_agent.eeg_research.agentic import llm, native_patch
from react_agent.eeg_research.agentic.binding import file_sha256
from react_agent.eeg_research.agentic.implementation_semantics import requirements_hash
from react_agent.eeg_research.agentic.interface import candidate_interface
from react_agent.eeg_research.agentic.stage_context import implementation_stage_payload, with_selected_task
from test_single_pass_review import campaign, sdk  # real transport fixture; external network forbidden
from test_reviewer_output_revalidation import resolve


def staged_spec():
    return {"design_contract_version": "v2", "spec_hash": "approved-fixture",
        "required_capability_ids": ["build_encoder"], "intervention": "Test encoder",
        "implementation_requirements": [{"requirement_id": "encoder", "kind": "implementation",
            "core": True, "expected_behavior": "Provide the approved encoder.", "verification": "source_review",
            "code_locations": ["EEGCandidate.build_encoder"]}],
        "measurement_requirements": [{"requirement_id": "paired_folds", "core": True,
            "expected_behavior": "Complete frozen paired folds.", "verification": "paired_coverage"}],
        "analysis_requirements": [{"requirement_id": "interpret", "core": True,
            "expected_behavior": "Interpret the measured hypothesis.", "verification": "hypothesis_assessment"}],
        "completion_condition": "Finish experiment and analysis",
        "design_brief": {"measurement_requirements": [{"requirement_id": "paired_folds"}],
                         "analysis_requirements": [{"requirement_id": "interpret"}],
                         "completion_condition": "Finish experiment and analysis"}}


def test_native_implementation_review_can_finish_before_any_training_receipt(tmp_path, monkeypatch):
    workspace = tmp_path / "candidates/c1"
    entry = workspace / "extension/eeg_candidate.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("class EEGCandidate:\n    def build_encoder(self):\n        return 1\n")
    experiment = staged_spec()
    spec = {"experiment": experiment}
    original = copy.deepcopy(spec)
    (workspace / "spec.json").write_text(json.dumps(spec))
    (workspace / "input_spec.json").write_text(json.dumps(candidate_interface(None)))
    checks = {"ok": True, "source_sha256": file_sha256(entry),
        "check_fingerprint": native_patch._check_fingerprint(workspace, None),
        "implementation_semantics": {"requirements_hash": requirements_hash(experiment["implementation_requirements"])}}
    (workspace / "checks.json").write_text(json.dumps(checks))
    monkeypatch.setattr(native_patch, "_parameter_census", lambda *_: {})
    observed = []
    def reviewer(payload):
        observed.append(payload)
        identity = payload["requirement_evidence_identity"]
        return {"status": "ready", "issues": [], "summary_zh": "代码要求满足，可训练。",
            "intervention_coverage": {"encoder": {"status": "implemented", "source_refs": [str(entry)]}},
            "requirement_coverage": [{"requirement_id": "encoder", "status": "implemented",
                "source_hash": identity["source_hash"], "check_sha256": identity["check_sha256"],
                "source_refs": [str(entry)], "check_ref": identity["check_ref"], "detail": "The approved encoder exists."}]}
    result = native_patch.review(workspace, spec, {}, reviewer, "offline")
    assert result["status"] == "ready", result
    assert result["review_stage"] == "implementation"
    assert len(observed) == 1
    assert observed[0]["hook_applicability"]["fit_statistics"] == "unselected_optional"
    assert "fit_statistics" not in observed[0]["candidate_interface"]
    wire = json.dumps(observed[0])
    assert "paired_folds" not in wire and '"interpret"' not in wire
    assert "Finish experiment and analysis" not in wire
    assert spec == original
    assert not (tmp_path / "measurement_reviews").exists()


@pytest.mark.parametrize("source,required,expected", [
    ("", [], "unselected_optional"),
    ("", ["fit_statistics"], "required"),
    ("class Base:\n    def fit_statistics(self, encoder, stats):\n        pass\n", [], "existing_behavior"),
])
def test_optional_hook_contract_respects_selected_and_inherited_behavior(source, required, expected):
    experiment = staged_spec()
    experiment["required_capability_ids"] += required
    payload = {"experiment_spec": experiment, "candidate_interface": candidate_interface(None),
               "parent_source": {"source": source}}
    result = implementation_stage_payload(payload)
    assert result["hook_applicability"]["fit_statistics"] == expected
    assert ("fit_statistics" in result["candidate_interface"]) == (expected != "unselected_optional")
    assert implementation_stage_payload(result) == result


def test_stage_scope_survives_selected_plan_enrichment_at_actual_sdk_boundary(tmp_path, sdk, monkeypatch):
    wire, _ = sdk
    payload = campaign(tmp_path, "single_pass")
    goal = json.loads((tmp_path / "goal.json").read_text())
    goal["research_execution"]["design_contract_version"] = "v2"
    (tmp_path / "goal.json").write_text(json.dumps(goal))
    payload.update(experiment_spec={"experiment": staged_spec()}, candidate_interface=candidate_interface(None))
    selected = {"plan_id": "plan_fixture", "proposal": {"question": "Encoder question",
        "scientific_intent": "Encoder mechanism", "task_description": "Implement, train, interpret",
        "completion_condition": "Publish future fold measurements", "expected_observation": "Gain accuracy"}}
    monkeypatch.setattr("react_agent.eeg_research.agentic.selected_plan.load_plan", lambda *_: selected)
    assert llm.role_backend(tmp_path, "candidate_reviewer")(payload)["status"] == "ready"
    assert len(wire) == 1
    actual = json.loads(wire[0]["messages"][1]["content"])
    actual = resolve(actual, actual)
    assert actual["review_stage"] == "implementation"
    assert actual["selected_research_task"] == {"question": "Encoder question", "scientific_intent": "Encoder mechanism"}
    assert "measurement_requirements" not in json.dumps(actual)
    assert "analysis_requirements" not in json.dumps(actual)
    assert "fit_statistics" not in actual["candidate_interface"]


def test_analysis_enrichment_retains_only_scientific_question_and_intent():
    selected = {"plan_id": "fixture", "proposal": {"question": "Q", "scientific_intent": "S",
        "completion_condition": "Full experiment", "expected_observation": "Gain", "task_description": "All stages"}}
    result = with_selected_task({"review_stage": "analysis", "analysis_requirements": []}, "result_analyst", selected)
    assert result["selected_research_task"] == {"question": "Q", "scientific_intent": "S"}


def test_legacy_inputs_keep_original_scope():
    payload = {"experiment_spec": {"design_contract_version": "v2", "implementation_requirements": []},
               "candidate_interface": candidate_interface(None)}
    assert implementation_stage_payload(payload) is payload
    selected = {"plan_id": "fixture", "proposal": {"completion_condition": "Legacy full task"}}
    assert with_selected_task(payload, "candidate_reviewer", selected)["selected_research_task"] == selected["proposal"]


def test_staged_analysis_mission_is_scoped_without_single_plan_mode(tmp_path):
    (tmp_path / "goal.json").write_text('{"max_llm_calls": 0, "memory": {"enabled": false}}')
    old = llm.system_prompt("result_analyst", camp=tmp_path)
    staged = llm.system_prompt("result_analyst", camp=tmp_path, review_stage="analysis")
    assert "Own Implementation review" not in staged
    assert "assess only analysis_requirements" in staged
    assert "analysis_requirement_coverage" in staged
    assert "analysis_requirement_coverage" not in old
