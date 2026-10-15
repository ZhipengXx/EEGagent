"""Stage requirements preserve legacy identities and survive native bindings."""

import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from react_agent.eeg_research.agentic.design_contract import materialize_design_brief
from react_agent.eeg_research.agentic.experiment_gate import (
    CanonicalizationError, _execution_hash, _spec_hash, approve_experiment,
    canonicalize_experiment_spec, resolve_approved_experiment,
)
from react_agent.eeg_research.agentic.plan_protocol import digest, scientific_projection
from react_agent.eeg_research.agentic.run_context import (
    BASELINE_HOOK_SPEC, build_frozen_run_spec, persist_approved_binding,
)
from react_agent.eeg_research.agentic.schemas import (
    AnalysisRequirement, DesignerBrief, LegacyExperimentDesignResult, LegacyResultAnalysis, MeasurementRequirement,
    ResultAnalysis, StageRequirementCoverage,
)
from react_agent.eeg_research.agentic.stage_contract import (
    has_stage_contract, normalize_stage_requirements, requirements_for_stage,
)


def brief():
    return {
        "hypothesis": "Temporal filtering changes transfer.",
        "principal_intervention": "Replace encoder with temporal convolutions.",
        "hook_changes": {
            "model": {"mode": "replace", "config": {"width": 32}},
            "objective": {"mode": "inherit"}, "transform": {"mode": "inherit"},
        },
        "implementation_requirements": [
            {"requirement_id": "temporal_path", "expected_behavior": "Apply convolution before pooling."}],
    }


def task():
    return {"parent_candidate_id": "baseline", "scientific_intent": "Temporal filtering with fixed supervision.",
            "evaluation_scope": {"phase": "screening_2fold", "folds": ["fold_07", "fold_10"], "seed": 0}}


def staged_brief():
    value = brief()
    value["measurement_requirements"] = [{"requirement_id": "paired_folds",
        "expected_behavior": "Obtain paired coverage on the frozen selected folds.", "verification": "paired_coverage"}]
    value["analysis_requirements"] = [{"requirement_id": "bounded_conclusion",
        "expected_behavior": "State the scope of the observed comparison.", "verification": "scope_limits"}]
    return value


def materialize(value):
    return materialize_design_brief(value, task(), BASELINE_HOOK_SPEC)


def test_legacy_v2_hashes_match_prechange_head_snapshot():
    # Captured by independently loading the four original HEAD modules.
    value = materialize(brief())
    normalized = canonicalize_experiment_spec(value)
    assert _spec_hash(normalized) == "99ce63634fbf4535ef562dd6318212ca949560ce623b4a5401e2ed4f5cda203c"
    assert digest(scientific_projection(normalized)) == "b611d36f67c00454f36df22ce04a6d4db729bf6cfa0aadc32f72f6b57b3c9c8e"
    for field in ("measurement_requirements", "analysis_requirements"):
        assert field not in value and field not in value["design_brief"]
        assert field not in scientific_projection(normalized)
    assert not has_stage_contract(value)


def test_legacy_analysis_defaults_are_not_removed_or_extended():
    assert ResultAnalysis.model_validate({"summary_zh": "No trained measurements yet."}).model_dump() == {
        "execution_assessment": None, "hypothesis_assessment": None, "observations": [],
        "prediction_checks": [], "interpretations": [], "competing_explanations": [],
        "evidence_gaps": [], "suggested_next_actions": [], "scope_limits": [],
        "evidence_refs": [], "summary_zh": "No trained measurements yet."}
    explicit = ResultAnalysis.model_validate({"analysis_requirement_coverage": []}).model_dump()
    assert explicit["analysis_requirement_coverage"] == []
    schema = LegacyExperimentDesignResult.model_json_schema()
    assert "measurement_requirements" not in schema["$defs"]["DesignerExperimentSpec"]["properties"]


def test_legacy_analyst_wire_schema_matches_original_snapshot():
    # Captured from original ResultAnalysis at baseline 924ddc4, independently
    # compared against that model before this test was made independent of git.
    legacy_schema = LegacyResultAnalysis.model_json_schema()
    encoded = json.dumps(legacy_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(encoded.encode()).hexdigest() == "78b08b3ee436ae37adf512fee1b27285222d735a486827ed8ab7fd612177cfba"
    assert legacy_schema["title"] == "ResultAnalysis"
    assert "analysis_requirement_coverage" not in legacy_schema["properties"]
    assert "StageRequirementCoverage" not in legacy_schema.get("$defs", {})
    for payload in ({}, {"summary_zh": "Result pending.", "hypothesis_assessment": "not_tested"}):
        assert LegacyResultAnalysis.model_validate(payload).model_dump() == ResultAnalysis.model_validate(payload).model_dump()
    with pytest.raises(ValidationError):
        LegacyResultAnalysis.model_validate({"analysis_requirement_coverage": []})
    with pytest.raises(ValidationError):
        LegacyResultAnalysis.model_validate({"hypothesis_assessment": "invented_status"})


@pytest.mark.parametrize("field", ["measurement_requirements", "analysis_requirements"])
def test_explicit_empty_stage_is_stable_and_distinct_from_absence(field):
    value = brief()
    value[field] = []
    parsed = DesignerBrief.model_validate(value)
    assert parsed.model_dump()[field] == []
    spec = materialize(parsed)
    assert spec[field] == [] and spec["design_brief"][field] == []
    assert scientific_projection(spec)[field] == []
    assert has_stage_contract(spec) and has_stage_contract({"experiment": spec})
    assert _spec_hash(spec) != _spec_hash(materialize(brief()))
    assert canonicalize_experiment_spec(canonicalize_experiment_spec(spec)) == canonicalize_experiment_spec(spec)


@pytest.mark.parametrize("model,verification", [
    (MeasurementRequirement, "scope_limits"), (AnalysisRequirement, "run_identity")])
def test_predicates_cannot_be_assigned_to_another_stage(model, verification):
    with pytest.raises(ValidationError):
        model.model_validate({"requirement_id": "bad", "expected_behavior": "Check evidence.",
                              "verification": verification})


@pytest.mark.parametrize("extra", [{"code_locations": ["Encoder.forward"]}, {"threshold": 99},
                                  {"probe": {}}, {"stage": "implementation"}])
def test_measurement_predicates_have_no_code_binding_or_model_invented_threshold(extra):
    with pytest.raises(ValidationError):
        MeasurementRequirement.model_validate({"requirement_id": "measurement", "expected_behavior": "Verify the run.",
                                                "verification": "run_identity", **extra})


@pytest.mark.parametrize("collision", ["implementation", "measurement"])
def test_ids_are_unique_across_all_stages_at_both_entries(collision):
    value = staged_brief()
    value["analysis_requirements"][0]["requirement_id"] = "temporal_path" if collision == "implementation" else "paired_folds"
    with pytest.raises(ValidationError, match="duplicate_stage_requirement_id"):
        DesignerBrief.model_validate(value)
    spec = materialize(staged_brief())
    spec["analysis_requirements"] = value["analysis_requirements"]
    with pytest.raises(CanonicalizationError, match="duplicate_stage_requirement_id"):
        canonicalize_experiment_spec(spec)


def test_v1_cannot_accept_stage_requirements_its_scientific_identity_would_ignore():
    spec = materialize(staged_brief())
    spec.pop("design_contract_version")
    spec["implementation_requirements"] = []
    with pytest.raises(CanonicalizationError, match="stage_requirements_require_v2_contract"):
        canonicalize_experiment_spec(spec)


@pytest.mark.parametrize("field,change", [
    ("measurement_requirements", {"expected_behavior": "Require different measurement evidence."}),
    ("measurement_requirements", {"core": False}),
    ("measurement_requirements", {"verification": "metric_integrity"}),
    ("analysis_requirements", {"expected_behavior": "Assert a different scientific conclusion."}),
    ("analysis_requirements", {"verification": "hypothesis_assessment"}),
])
def test_acceptance_changes_are_part_of_scientific_and_execution_identity(field, change):
    original = canonicalize_experiment_spec(materialize(staged_brief()))
    modified = copy.deepcopy(original)
    modified[field][0].update(change)
    assert scientific_projection(modified) != scientific_projection(original)
    assert _execution_hash(modified) != _execution_hash(original)
    assert _spec_hash(modified) != _spec_hash(original)
    removed = copy.deepcopy(original)
    removed.pop(field)
    assert scientific_projection(removed) != scientific_projection(original)


def test_native_approval_binding_resolver_and_frozen_run_keep_each_stage(tmp_path):
    context = {"capabilities": {"hooks": {"build_encoder": {"status": "available"}}}}
    approved = approve_experiment(materialize(staged_brief()), camp=tmp_path, context=context)
    assert approved["status"] == "approved", approved
    binding = persist_approved_binding(tmp_path, "c1", approved, source_hash="candidate_source")
    resolved = resolve_approved_experiment(tmp_path, target_id="c1", action="repair", state={"experiment": approved})
    frozen = build_frozen_run_spec(tmp_path, "c1", seed=0, fidelity="full")
    assert has_stage_contract(binding) and has_stage_contract(frozen)
    for value in (binding, resolved):
        assert scientific_projection(value) == scientific_projection(approved)
    for field in ("measurement_requirements", "analysis_requirements"):
        assert approved[field] == binding[field] == resolved[field] == frozen[field]
    assert resolved["objective"] == {} and resolved["transform"] == {}
    phase = requirements_for_stage({"experiment": approved}, "measurement")
    phase[0]["expected_behavior"] = "Local view only."
    assert approved["measurement_requirements"][0]["expected_behavior"] != "Local view only."


def test_analysis_coverage_uses_stage_evidence_and_rejects_duplicate_ids():
    coverage = {"requirement_id": "bounded_conclusion", "status": "implemented",
                "detail": "Uses the measured two-fold evidence and states its scope.",
                "evidence_refs": ["ev_paired_screening"]}
    parsed = ResultAnalysis.model_validate({"analysis_requirement_coverage": [coverage]})
    assert parsed.model_dump()["analysis_requirement_coverage"] == [coverage]
    with pytest.raises(ValidationError, match="duplicate_analysis_requirement_coverage_id"):
        ResultAnalysis.model_validate({"analysis_requirement_coverage": [coverage, coverage]})
    with pytest.raises(ValidationError, match="needs_evidence_refs"):
        StageRequirementCoverage.model_validate({**coverage, "evidence_refs": []})
    assert StageRequirementCoverage.model_validate({**coverage, "status": "unverified", "evidence_refs": []})
    assert normalize_stage_requirements(materialize(brief())) == {}
