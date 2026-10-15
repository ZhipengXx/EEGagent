"""Native scientific handoff; no model calls, EEG data or GPU execution."""

import copy
import json

import pytest
from pydantic import ValidationError

from react_agent.eeg_research.agentic.design_contract import (
    DesignContractError, design_contract_version, materialize_design_brief,
    scientific_design_projection,
)
from react_agent.eeg_research.agentic.experiment_gate import (
    CanonicalizationError, approve_experiment, canonicalize_experiment_spec,
    same_scientific_intervention,
)
from react_agent.eeg_research.agentic.hook_config import normalize_hook_config
from react_agent.eeg_research.agentic.plan_protocol import scientific_projection
from react_agent.eeg_research.agentic.run_context import BASELINE_HOOK_SPEC
from react_agent.eeg_research.agentic.schemas import (
    DesignerBrief, ExperimentDesignBriefResult, ExperimentDesignResult,
    ImplementationIssue, ImplementationRequirement, LegacyExperimentDesignResult, ResearchExecutionConfig,
)


def brief():
    return {
        "hypothesis": "Temporal attention preserves useful temporal structure.",
        "principal_intervention": "Apply temporal attention before temporal pooling.",
        "scientific_parameters": {"d_model": 128, "heads": 4},
        "hook_changes": {
            "model": {"mode": "replace", "config": {"d_model": 128, "heads": 4}},
            "objective": {"mode": "inherit"}, "transform": {"mode": "inherit"},
        },
        "implementation_requirements": [
            {"requirement_id": "attention_before_pool", "expected_behavior": "Attention operates before pooling."},
            {"requirement_id": "parent_hooks", "scope": "engineering",
             "expected_behavior": "Use the inherited objective and transform config."},
        ],
    }


def task(**extra):
    return {"plan_id": "plan_test", "parent_candidate_id": "baseline",
            "scientific_intent": "Test temporal attention while keeping supervision fixed.",
            "evaluation_scope": {"phase": "screening_2fold", "folds": ["sub-07", "sub-10"], "seed": 0},
            **extra}


def test_inherited_baseline_configs_reach_real_hooks_without_protocol_metadata():
    pytest.importorskip("torch")
    from react_agent.eeg_research.agentic.baseline import EEGCandidate
    from react_agent.eeg_training.model import contrastive_loss
    from react_agent.eeg_training.train_entry import _build_hook

    draft = materialize_design_brief(brief(), task(), BASELINE_HOOK_SPEC)
    context = {"evaluation_mode": "loso_method_search", "capabilities": {
        "hooks": {"build_encoder": {"status": "available"}}}}
    approved = approve_experiment(draft, context=context)
    assert approved["status"] == "approved", approved
    hook = normalize_hook_config(approved)
    assert hook["objective"] == {} and hook["transform"] == {}
    assert draft["evaluation_scope"]["folds"] == ["sub-07", "sub-10"]
    candidate = EEGCandidate()
    assert _build_hook(candidate, "build_training_objective", {}, hook["objective"]) is contrastive_loss
    assert _build_hook(candidate, "build_training_transform", {}, hook["transform"]) is None


def test_inherit_uses_parent_binding_and_replace_is_complete():
    parent = {"target_id": "c4", "model": {"d_model": 32, "legacy_width": 99},
              "objective": {"temperature": 0.125}, "transform": {"offset": 0.5},
              "spec_hash": "parent_spec", "source_hash": "parent_source"}
    original = copy.deepcopy(parent)
    draft = materialize_design_brief(brief(), task(parent_candidate_id="c4"), parent)
    assert draft["model"] == {"d_model": 128, "heads": 4}
    assert draft["objective"] == {"temperature": 0.125}
    assert draft["transform"] == {"offset": 0.5}
    assert draft["inheritance"]["source_hash"] == "parent_source"
    draft["objective"]["temperature"] = 0.25
    assert parent == original


@pytest.mark.parametrize("parent", [None, {"model": {}, "objective": {}},
                                      {"target_id": "c8", "model": {}, "objective": {}, "transform": {}}])
def test_nonbaseline_inherit_does_not_silently_fall_back(parent):
    with pytest.raises(DesignContractError, match="parent_hook_config"):
        materialize_design_brief(brief(), task(parent_candidate_id="c4"), parent)


@pytest.mark.parametrize("change", [{"mode": "inherit", "config": {}}, {"mode": "replace"}])
def test_hook_mode_has_one_meaning(change):
    incoming = brief()
    incoming["hook_changes"]["objective"] = change
    with pytest.raises(DesignContractError, match="design_brief_invalid"):
        materialize_design_brief(incoming, task(), BASELINE_HOOK_SPEC)


@pytest.mark.parametrize("parameters", [{"d_model": 256}, {"model": {"d_model": 256}},
                                         {"model.d_model": 256}])
def test_parameter_and_executable_config_cannot_disagree(parameters):
    incoming = brief()
    incoming["scientific_parameters"] = parameters
    with pytest.raises(DesignContractError, match="scientific_parameter_config_mismatch"):
        materialize_design_brief(incoming, task(), BASELINE_HOOK_SPEC)


def test_designer_declares_behavior_and_coder_locations_remain_separate():
    parsed = ExperimentDesignBriefResult.model_validate({"status": "completed", "design_brief": brief()})
    assert parsed.design_brief.implementation_requirements[0].code_locations == []
    spec = materialize_design_brief(parsed.design_brief, task(), BASELINE_HOOK_SPEC)
    before = copy.deepcopy(spec)
    assert canonicalize_experiment_spec(spec)["implementation_requirements"][0]["binding_stage"] == "design"
    implemented = copy.deepcopy(spec)
    implemented["implementation_requirements"][0]["code_locations"] = ["Encoder.forward"]
    assert scientific_projection(implemented) == scientific_projection(spec)
    assert spec == before
    wrong_version = {**spec, "design_contract_version": "v1"}
    with pytest.raises(CanonicalizationError, match="design_declaration_requires_v2_contract"):
        canonicalize_experiment_spec(wrong_version)


def test_ordered_probe_locations_can_be_bound_after_design_without_changing_semantics():
    incoming = brief()
    incoming["implementation_requirements"][0].update(
        verification="ordered_window_slots", probe={"slot_count": 3})
    spec = materialize_design_brief(incoming, task(), BASELINE_HOOK_SPEC)
    declared = spec["implementation_requirements"][0]
    assert declared["probe"]["input_module_path"] == ""
    bound = copy.deepcopy(spec)
    bound["implementation_requirements"][0]["probe"].update(
        input_module_path="pool", representation_module_path="projection")
    assert same_scientific_intervention(spec, bound)
    bound["implementation_requirements"][0]["probe"]["slot_count"] = 1
    assert not same_scientific_intervention(spec, bound)
    with pytest.raises(ValidationError, match="bound_modules"):
        ImplementationRequirement.model_validate({**declared, "binding_stage": "bound", "code_locations": ["Encoder.forward"]})


def test_engineering_revisions_preserve_identity_but_scientific_requirements_are_frozen():
    original = materialize_design_brief(brief(), task(), BASELINE_HOOK_SPEC)
    revised = copy.deepcopy(original)
    revised["implementation_requirements"][1]["expected_behavior"] = "Pass the exact inherited empty config."
    revised["design_brief"]["applicability"] = "Clarified interface documentation."
    assert scientific_design_projection(original) == scientific_projection(revised)
    for mutation in ("behavior", "scope", "removed", "model", "parameter", "hook_mode", "hypothesis"):
        changed = copy.deepcopy(revised)
        if mutation == "behavior":
            changed["implementation_requirements"][0]["expected_behavior"] = "Pool before attention."
        elif mutation == "scope":
            changed["implementation_requirements"][0]["scope"] = "engineering"
        elif mutation == "removed":
            changed["implementation_requirements"].pop(0)
        elif mutation == "model":
            changed["model"]["heads"] = 8
        elif mutation == "parameter":
            changed["scientific_parameters"]["heads"] = 8
        elif mutation == "hook_mode":
            changed["hook_modes"]["objective"] = "replace"
        else:
            changed["hypothesis"] = "Test a different scientific mechanism."
        assert not same_scientific_intervention(original, changed), mutation


def test_old_contract_shapes_and_projection_are_unchanged():
    requirement = {"requirement_id": "projection", "expected_behavior": "Project the EEG.",
                   "code_locations": ["Encoder.forward"]}
    normalized = ImplementationRequirement.model_validate(requirement).model_dump()
    assert normalized == {**requirement, "kind": "implementation", "core": True,
                          "verification": "source_review", "probe": None}
    legacy = {"intervention": "projection", "hypothesis": "projection helps", "model": {},
              "objective": {}, "transform": {}, "parent_candidate_id": "baseline", "initial_fidelity": "pilot",
              "required_capability_ids": ["build_encoder"], "implementation_requirements": [normalized]}
    canonical = canonicalize_experiment_spec(legacy)
    assert canonical == {**legacy, "schema_version": "eeg_research.experiment_spec.v1"}
    assert scientific_projection(legacy) == {key: legacy.get(key) for key in (
        "intervention", "principal_intervention", "hypothesis", "model", "objective", "transform",
        "parent_candidate_id", "control_candidate_id", "initial_fidelity", "implementation_requirements",
        "required_capability_ids", "evaluation_scope")}
    assert "design_contract_version" not in ResearchExecutionConfig.model_validate({"version": "single_plan_v1"}).model_dump()
    assert "design_brief" not in ExperimentDesignResult.model_validate({"status": "completed", "experiment_spec": legacy}).model_dump()
    assert ImplementationIssue.model_validate({"severity": "blocking", "category": "legacy"}).model_dump() == {
        "severity": "blocking", "category": "legacy", "blocking": None}


def test_v2_wire_only_accepts_the_brief_contract_and_version_is_frozen(tmp_path):
    assert "experiment_spec" not in ExperimentDesignBriefResult.model_json_schema()["properties"]
    assert "design_brief" not in LegacyExperimentDesignResult.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        ExperimentDesignBriefResult.model_validate({"status": "completed", "experiment_spec": {}})
    for enabled in (False, True):
        goal = {"research_execution": {"version": "single_plan_v1"}}
        if enabled:
            goal["research_execution"]["design_contract_version"] = "v2"
        (tmp_path / "goal.json").write_text(json.dumps(goal))
        expected = "v2" if enabled else "v1"
        assert design_contract_version(tmp_path) == design_contract_version(str(tmp_path)) == expected
        assert design_contract_version(goal) == design_contract_version({"goal": goal}) == expected
