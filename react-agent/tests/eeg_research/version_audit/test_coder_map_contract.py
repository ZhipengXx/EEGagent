"""Published map examples survive wire preparation and native map validation."""

import copy
import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.coder import validate_tool_request
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
from react_agent.eeg_research.agentic.llm import system_prompt
from react_agent.eeg_research.agentic.native_patch import _store_implementation_map
from react_agent.eeg_research.agentic.runtime_prompts import runtime_mission


def published_example(mission):
    text = mission.split("implementation_map shape: ", 1)[1]
    result, _ = json.JSONDecoder().raw_decode(text)
    return result


def test_map_shape_and_reference_formats_reach_actual_v2_prepared_coder(tmp_path):
    goal = {"goal_id": tmp_path.name, "research_execution": {
        "version": "single_plan_v1", "design_contract_version": "v2"}, "memory": {"enabled": False}}
    (tmp_path / "goal.json").write_text(json.dumps(goal))
    (tmp_path / "campaign_state.json").write_text('{"status":"created"}')
    (tmp_path / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval"}')
    result = ContextService(tmp_path, maintenance_callback=lambda **_: pytest.fail("No model calls")).compile(
        CompileRequest(tmp_path.name, "candidate_coder", {"current_file": None, "last_check": None},
                       system=system_prompt("candidate_coder", camp=tmp_path)))
    assert isinstance(result, Ready), result
    assert result.prepared.verify() and result.prepared.budget_report["ready_budget_ok"]
    mission = result.prepared.system
    example = published_example(mission)
    assert example == {"<exact requirement_id>": {
        "source_refs": ["extension/eeg_candidate.py::EEGCandidate.build_encoder"]}}
    assert "Row keys are source_refs and optional probe" in mission
    assert "runtime writes artifact/hash metadata" in mission
    assert "existing single line (extension/eeg_candidate.py:1)" in mission
    assert not (tmp_path / "llm_calls.jsonl").exists()


@pytest.mark.parametrize("reference", ["extension/eeg_candidate.py:1",
                                      "extension/eeg_candidate.py::EEGCandidate.build_encoder"])
def test_published_map_example_is_accepted_without_mutating_the_approved_spec(tmp_path, reference):
    source = tmp_path / "extension/eeg_candidate.py"
    source.parent.mkdir()
    source.write_text("class EEGCandidate:\n    def build_encoder(self, input_spec, model_config=None):\n        return input_spec\n")
    spec = {"experiment": {"design_contract_version": "v2", "spec_hash": "synthetic_approved_spec",
                           "implementation_requirements": [{"requirement_id": "encoder_interface",
                               "binding_stage": "design", "expected_behavior": "Provide the encoder hook."}]}}
    original = copy.deepcopy(spec)
    row = published_example(runtime_mission("candidate_coder", design_contract_version="v2"))["<exact requirement_id>"]
    row["source_refs"] = [reference]
    mapping = {"encoder_interface": row}
    tool, args, error = validate_tool_request({"tool": "finish_patch", "args": {
        "summary": "Mapped the implemented interface.", "implementation_map": mapping}})
    assert error is None and tool == "finish_patch"
    result = _store_implementation_map(tmp_path, spec, args["implementation_map"])
    assert result["ok"], result
    assert spec == original
    stored = json.loads((tmp_path / "implementation_map.json").read_text())
    assert stored["approved_spec_hash"] == "synthetic_approved_spec"
    assert stored["requirements"] == mapping


@pytest.mark.parametrize("reference", ["extension/eeg_candidate.py:1-3", "extension/eeg_candidate.py:99",
                                      "extension/eeg_candidate.py::EEGCandidate.missing", "reference/parent.py:1"])
def test_prompt_patch_does_not_widen_reference_validation(tmp_path, reference):
    source = tmp_path / "extension/eeg_candidate.py"
    source.parent.mkdir()
    source.write_text("class EEGCandidate:\n    def build_encoder(self, input_spec, model_config=None):\n        return input_spec\n")
    result = _store_implementation_map(tmp_path, {"experiment": {
        "spec_hash": "synthetic", "implementation_requirements": [{"requirement_id": "encoder_interface"}]}},
        {"encoder_interface": {"source_refs": [reference]}})
    assert result == {"ok": False, "error": "implementation_map_invalid_source_reference"}
    assert not (tmp_path / "implementation_map.json").exists()


def test_legacy_coder_mission_is_unchanged():
    root = Path(__file__).resolve().parents[3] / "src/react_agent/eeg_research/agentic/prompts/runtime_v1"
    expected = (root / "shared_contract.txt").read_text() + "\n\n" + (root / "candidate_coder.txt").read_text()
    assert runtime_mission("candidate_coder") == expected
    assert "implementation_map shape:" not in expected
