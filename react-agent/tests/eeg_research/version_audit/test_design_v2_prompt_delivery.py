"""Exercise the actual versioned request compiler without provider calls."""
import json

import pytest

from react_agent.eeg_research.agentic.cli import goal
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
from react_agent.eeg_research.agentic.llm import domain_model_for, system_prompt
from react_agent.eeg_research.agentic.role_examples import add_role_examples
from react_agent.eeg_research.agentic.runtime_prompts import conditional_hints, runtime_mission
from react_agent.eeg_research.agentic.schemas import ExperimentDesignBriefResult, ResearchExecutionConfig


@pytest.fixture
def camp(tmp_path):
    configured = goal("design-v2-local")
    configured.update(max_llm_calls=0, max_training_jobs=0, max_gpu_seconds=0, memory={"enabled": False})
    (tmp_path / "goal.json").write_text(json.dumps(configured))
    (tmp_path / "campaign_state.json").write_text('{"status":"implementing"}')
    (tmp_path / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval"}')
    return tmp_path


@pytest.mark.parametrize("role, delivered", [
    ("experiment_designer", "without guessing future code_locations"),
    ("candidate_coder", "request_design_revision"),
    ("candidate_reviewer", "smallest_correction and validation_expectation"),
])
def test_actual_v2_prepared_system_contains_its_workflow(camp, role, delivered):
    payload = {"design_contract_version": "v2", "selected_research_task": {"task_description": "Synthetic local task"},
        "experiment_spec": {"model": {}, "objective": {}, "transform": {}}}
    compiled = ContextService(camp).compile(CompileRequest(camp.name, role, payload,
        system=system_prompt(role, camp=camp)))
    assert isinstance(compiled, Ready), compiled
    assert compiled.prepared.verify() and compiled.prepared.budget_report["ready_budget_ok"]
    assert delivered in compiled.prepared.system
    assert len(runtime_mission(role, design_contract_version="v2")) < len(runtime_mission(role))
    assert not (camp / "llm_calls.jsonl").exists()


def test_v2_designer_schema_and_example_match_at_real_boundary(camp):
    assert domain_model_for("experiment_designer", camp=camp) is ExperimentDesignBriefResult
    system = system_prompt("experiment_designer", camp=camp)
    encoded = system.split("\nOUTPUT_SCHEMA\n", 1)[1].split("\nReturn one JSON object.", 1)[0]
    schema = json.loads(encoded)
    assert "design_brief" in schema["properties"]
    assert "experiment_spec" not in schema["properties"]
    payload = add_role_examples({}, "experiment_designer", execution_version="single_plan_v1", design_contract_version="v2")
    response = payload["illustrative_examples"]["examples"][0]["next_response"]
    result = ExperimentDesignBriefResult.model_validate(response)
    assert result.design_brief.hook_changes.objective.mode == "inherit"
    assert result.design_brief.implementation_requirements[0].code_locations == []


def test_legacy_goal_remains_legacy_without_implicit_new_version(tmp_path):
    execution = ResearchExecutionConfig.model_validate({"version": "single_plan_v1"}).model_dump()
    assert "design_contract_version" not in execution
    (tmp_path / "goal.json").write_text(json.dumps({"research_execution": execution}))
    system = system_prompt("experiment_designer", camp=tmp_path)
    assert "Record core implementation promises with real code locations" in system
    assert "without guessing future code_locations" not in system


def test_current_experiment_spec_selects_existing_loss_hint():
    payload = {"experiment_spec": {"objective": {"name": "positive_mass_infonce"}}}
    before = json.dumps(payload, sort_keys=True)
    hints = conditional_hints("candidate_coder", payload)
    assert hints["notes"][0]["case_id"] == "positive_mass_infonce"
    assert json.dumps(payload, sort_keys=True) == before


def test_v2_uses_actual_context_instead_of_repeated_toy_implementations():
    payload = {"experiment_spec": {"model": {}}, "last_check": {"ok": False, "error": "shape"}}
    assert add_role_examples(payload, "candidate_coder", design_contract_version="v2") is payload
    assert add_role_examples(payload, "candidate_reviewer", design_contract_version="v2") is payload


@pytest.mark.parametrize("version, calls", [("v1", 2), ("v2", 6)])
def test_design_revision_calls_reserved_in_existing_plan_envelope(tmp_path, version, calls):
    from react_agent.eeg_research.agentic.owner_budget import _envelope
    (tmp_path / "goal.json").write_text(json.dumps({"research_execution": {
        "version": "single_plan_v1", "design_contract_version": version,
        "implementation_review_mode": "single_pass"}}))
    _, roles, _, _, _ = _envelope(tmp_path, "new_method", {"training_jobs": 0, "gpu_seconds": 0}, {"max_repairs": 2})
    assert roles["experiment_designer"]["llm_calls"] == calls


def test_explicit_designer_call_bound_is_preserved(tmp_path):
    from react_agent.eeg_research.agentic.owner_budget import _envelope
    (tmp_path / "goal.json").write_text(json.dumps({"research_execution": {
        "version": "single_plan_v1", "design_contract_version": "v2", "budget": {"envelopes": {
            "new_method": {"role_calls": {"experiment_designer": 2, "candidate_coder": 1,
                "candidate_reviewer": 1, "result_analyst": 2}}}}}}))
    _, roles, _, _, _ = _envelope(tmp_path, "new_method", {"training_jobs": 0, "gpu_seconds": 0}, {"max_repairs": 2})
    assert roles["experiment_designer"]["llm_calls"] == 2
