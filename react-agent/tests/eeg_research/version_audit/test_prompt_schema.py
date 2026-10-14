"""One provider schema with independent native proposal admission preserved."""
import json

import pytest

from react_agent.eeg_research.agentic.llm import system_prompt
from react_agent.eeg_research.agentic.schemas import ResearchPlanProposal, ResearchRoundDraft


def test_planner_prompt_has_one_exact_proposal_schema():
    prompt = system_prompt("research_planner", camp={"research_execution": {"version": "single_plan_v1"}})
    assert prompt.count("\nOUTPUT_SCHEMA\n") == 1
    assert "INDEPENDENT_PROPOSAL_SCHEMA" not in prompt
    assert "required_context_refs is []" in prompt
    assert "delivered context:// source references" in prompt
    assert "null placeholders such as business_value" in prompt
    encoded = prompt.split("\nOUTPUT_SCHEMA\n", 1)[1].split("\nReturn one JSON object.", 1)[0]
    schema = json.loads(encoded)
    items = schema["properties"]["proposals"]["items"]
    native = ResearchPlanProposal.model_json_schema()
    assert set(items["properties"]) == set(native["properties"])
    assert items["required"] == native["required"]
    for field in ("proposal_id", "task_kind", "scientific_intent", "task_description"):
        expected = {key: value for key, value in native["properties"][field].items()
                    if key not in {"title", "description"}}
        assert items["properties"][field] == expected
    scope = items["properties"]["evaluation_scope"]
    assert scope["required"] == ["phase"]
    assert scope["properties"]["phase"]["enum"] == native["properties"]["evaluation_scope"]["properties"]["phase"]["enum"]


def test_raw_invalid_sibling_survives_outer_parse_for_independent_validation():
    raw = {"protocol_version": "research_plan_draft.v1", "proposals": [{"proposal_id": "invalid"}],
           "evaluations": [], "selected_proposal_id": None, "selection_rationale": "Validate independently."}
    parsed = ResearchRoundDraft.model_validate(raw)
    assert parsed.proposals == raw["proposals"]


def test_single_plan_coder_real_prepared_request_has_native_tool_schema(tmp_path, monkeypatch):
    from react_agent.eeg_research.agentic.coder import CoderToolRequest, TOOL_ARGUMENTS, tool_catalog
    from react_agent.eeg_research.agentic.context.service import ContextService
    from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: False)
    value = {"research_execution": {"version": "single_plan_v1"}, "memory": {"enabled": False}}
    (tmp_path / "goal.json").write_text(json.dumps(value))
    (tmp_path / "campaign_state.json").write_text(json.dumps({"status": "implementing"}))
    payload = {"tools": tool_catalog(), "current_file": None, "last_check": None,
        "history": [], "allowed_write_paths": ["extension/eeg_candidate.py"],
        "input_spec": {"business_array": [None, {"optional": None}]}}
    system = system_prompt("candidate_coder", camp=tmp_path)
    result = ContextService(tmp_path).compile(CompileRequest(tmp_path.name, "candidate_coder", payload, system=system))
    assert isinstance(result, Ready), result
    prepared = result.prepared
    assert prepared.verify() and prepared.budget_report["ready_budget_ok"]
    assert prepared.system.count("\nOUTPUT_SCHEMA\n") == 1
    encoded = prepared.system.split("\nOUTPUT_SCHEMA\n", 1)[1].split("\nReturn one JSON object.", 1)[0]
    schema = json.loads(encoded)
    assert schema != {}
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["tool", "args"]
    assert set(schema["properties"]) == {"tool", "args"}
    assert schema["properties"]["tool"]["enum"] == list(TOOL_ARGUMENTS)
    assert schema["properties"]["args"]["type"] == "object"
    assert prepared.system.count('"enum":') == 1
    assert prepared.payload["tools"] == tool_catalog()
    assert prepared.payload["current_file"] is None and prepared.payload["last_check"] is None
    assert prepared.payload["input_spec"]["business_array"] == [None, {"optional": None}]
    assert CoderToolRequest.model_validate({"tool": "read_code", "args": {"path": "reference/train_entry.py"}}).tool == "read_code"


@pytest.mark.parametrize("wrapper", ["json_object", "tool_call"])
def test_coder_published_envelope_and_native_validator_reject_type_wrapper(wrapper):
    from pydantic import ValidationError
    from react_agent.eeg_research.agentic.coder import CoderToolRequest, validate_tool_request
    reply = {"type": wrapper, "tool": "read_code", "args": {"path": "reference/train_entry.py"}}
    original = json.dumps(reply, sort_keys=True)
    with pytest.raises(ValidationError):
        CoderToolRequest.model_validate(reply)
    _, _, error = validate_tool_request(reply)
    assert error["error"] == "invalid_tool_request" and error["forbidden_fields"] == ["type"]
    assert json.dumps(reply, sort_keys=True) == original


def test_designer_coder_config_contract_survives_actual_prepared_schema_compaction(tmp_path):
    """Constructor/metadata obligations are delivered prose, not stripped annotations."""
    import hashlib
    from react_agent.eeg_research.agentic.context.service import ContextService
    from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
    from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready, canonical, digest
    from react_agent.eeg_research.agentic.context.sources import ContextSourceRegistry
    from react_agent.eeg_research.agentic.runtime_prompts import runtime_mission

    camp = tmp_path / "synthetic-config-prompt-contract"
    camp.mkdir()
    (camp / "goal.json").write_text(json.dumps({"goal_id": camp.name, "objective": "Synthetic prompt delivery only",
        "max_llm_calls": 0, "research_execution": {"version": "single_plan_v1", "paid_maintenance": "off"},
        "memory": {"enabled": False}}))
    (camp / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval"}')
    approved_model = {"c_num": 17, "finite_output_required": True, "hook": "build_encoder",
        "input_shape": [None, 17, 250], "interface_version": "synthetic_interface_v1",
        "layers": "temporal mean then linear projection", "output_shape": [None, 1024],
        "timesteps": [0, 250], "z_dim": 1024}
    requirements = [{"requirement_id": "native_projection", "kind": "implementation", "core": True,
        "expected_behavior": "Use the declared temporal aggregation and projection.",
        "code_locations": ["EEGCandidate.build_encoder"], "verification": "source_review"},
        {"requirement_id": "scientific_prediction", "kind": "scientific_hypothesis", "core": True,
         "expected_behavior": "An effect remains an unmeasured prediction until the evaluator returns evidence."}]
    spec = {"model": approved_model, "intervention": "Synthetic temporal aggregation",
            "implementation_requirements": requirements}
    source = "class EEGCandidate:\n    def build_encoder(self, input_spec, model_config=None):\n        return input_spec, model_config\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    payload = {"approved_spec": spec, "experiment": spec,
        "input_spec": {"c_num": 17, "timesteps": [0, 250], "output_dim": 1024},
        "implementation_source": source, "last_check": {"ok": False, "source_sha256": source_hash,
            "error": "unknown_model_config", "rejected_keys": list(approved_model),
            "scope": "synthetic failed check delivery; no implementation approval"}}
    original = canonical(payload)
    def expand(value, root):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(root, value["$context_pointer"]), root)
            return {key: expand(child, root) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child, root) for child in value]
        return value
    expected = {
        "experiment_designer": ("model mapping contains only actual constructor hyperparameters",
            "input_spec/candidate_interface", "Put architecture descriptions in intervention/hypothesis",
            "Scientific improvement is a hypothesis, not an implementation gate."),
        "candidate_coder": ("Support the complete approved model_config", "actual caller",
            "do not invent a rejection whitelist", "inspect the actual automatic check output",
            "A patch submission or intended check success is not a passing receipt.",
            "mechanism or hypothesis requires needs_replan"),
    }
    for role, obligations in expected.items():
        result = ContextService(camp).compile(CompileRequest(camp.name, role, payload,
            system=system_prompt(role, camp=camp), max_output_tokens=16384))
        assert isinstance(result, Ready), result
        assert result.prepared.verify() and result.prepared.budget_report["ready_budget_ok"]
        mission = result.prepared.system.split("\nOUTPUT_SCHEMA\n", 1)[0]
        assert all(obligation in mission for obligation in obligations)
        assert "Record core implementation promises with real code locations" in mission if role == "experiment_designer" else "fresh checks are satisfied" in mission
        delivered = expand(result.prepared.payload, result.prepared.payload)
        assert delivered["approved_spec"]["model"] == approved_model
        assert delivered["approved_spec"]["implementation_requirements"] == requirements
        assert delivered["last_check"]["ok"] is False and delivered["last_check"]["rejected_keys"] == list(approved_model)
        registered = next(row for row in result.prepared.dependencies if row["block_id"] == "input:/approved_spec")
        registry = ContextSourceRegistry(camp)
        block = registry.get(registered["source_ref"], role=role)
        assert block.source_hash == registered["source_hash"] == digest(spec)
        assert registry.load(block, role=role) == spec
        assert len(runtime_mission(role)) < (3500 if role == "candidate_coder" else 2500)
    assert canonical(payload) == original
    assert not (camp / "llm_calls.jsonl").exists()


def test_development_filter_distinguishes_json_null_from_private_omission():
    import copy
    from react_agent.eeg_research.agentic.handoffs import development_view
    from react_agent.eeg_research.agentic.context.types import digest
    visible = {"model": {"input_shape": [None, 17, 250], "output_shape": [None, 1024],
        "nested": [[None, 0, False, None], None]}, "target_candidate_ref": None, "metrics": None}
    original = copy.deepcopy(visible)
    assert development_view(visible) == original
    assert digest(development_view(visible)) == digest(original) and visible == original
    mixed = {"rows": [None, {"scope": "final_test", "private_metric": 42}, None, visible, 0, False],
             "api_key": "synthetic-private-key", "safe_null": None}
    assert development_view(mixed) == {"rows": [None, None, visible, 0, False], "safe_null": None}
    assert development_view({"scope": "secret", "private_metric": 42}) is None
