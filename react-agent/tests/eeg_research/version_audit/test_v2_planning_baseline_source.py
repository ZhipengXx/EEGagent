"""The real baseline implementation reaches both staged planning roles."""

import copy
import json

import pytest

from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.eeg_research.agentic.experiment_observations import compact_planner_view
from react_agent.eeg_research.agentic.handoffs import candidate_context
from react_agent.eeg_research.agentic.llm import system_prompt


def campaign(tmp_path, version):
    goal = {"goal_id": "baseline-planning-" + version, "objective": "Propose new EEG encoders from the actual baseline.",
            "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic",
                                   "design_contract_version": version},
            "memory": {"enabled": False}}
    (tmp_path / "goal.json").write_text(json.dumps(goal))
    (tmp_path / "campaign_state.json").write_text('{"status":"planning"}')
    (tmp_path / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval","fingerprint":"synthetic"}')
    baseline = candidate_context(tmp_path, "baseline")
    return {"goal": goal, "parent_source": baseline, "control_source": copy.deepcopy(baseline),
            "contract": {"task_type": "eeg_image_retrieval", "fingerprint": "synthetic"},
            "candidates": [], "evidence": [], "budget": {"llm_calls_left": 0}}


def expand(payload):
    def visit(value):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return visit(resolve_request_pointer(payload, value["$context_pointer"]))
            return {key: visit(child) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child) for child in value]
        return value
    return visit(payload)


@pytest.mark.parametrize("role,stage", [("research_planner", "propose"), ("research_critic", "critic")])
def test_v2_real_baseline_survives_actual_prepared_request_for_both_roles(tmp_path, role, stage):
    full = campaign(tmp_path, "v2")
    original = copy.deepcopy(full)
    delivered = compact_planner_view(tmp_path, full)
    assert full == original
    assert "parent_source" not in delivered and "control_source" not in delivered
    assert delivered["baseline_source"] == full["parent_source"]
    payload = {**delivered, "planning_stage": stage}
    if role == "research_critic":
        payload.update(proposal_pool={"proposals": []}, proposal_pool_digest="synthetic_pool")
    result = ContextService(tmp_path, maintenance_callback=lambda **_: pytest.fail("No model maintenance")).compile(
        CompileRequest(tmp_path.name, role, payload, system=system_prompt(role, camp=tmp_path), max_output_tokens=16384))
    assert isinstance(result, Ready), result
    assert result.prepared.verify() and result.prepared.budget_report["ready_budget_ok"]
    actual = expand(json.loads(result.prepared.user))
    baseline = actual["baseline_source"]
    assert baseline["source"] == full["parent_source"]["source"]
    assert baseline["source_hash"] == full["parent_source"]["source_hash"]
    assert "class EEGCandidate" in baseline["source"]
    assert baseline["encoder_reference"] == full["parent_source"]["encoder_reference"]
    assert "class EEGProjectLayer" in baseline["encoder_reference"]["excerpt"]
    assert not (tmp_path / "llm_calls.jsonl").exists()


def test_v1_still_archives_baseline_and_v2_delivery_is_idempotent(tmp_path):
    full = campaign(tmp_path, "v1")
    old = compact_planner_view(tmp_path, full)
    assert "baseline_source" not in old and "parent_source" not in old and "control_source" not in old
    assert {row["field"] for row in old["history_navigation"]} >= {"parent_source", "control_source"}
    goal = full["goal"]
    goal["research_execution"]["design_contract_version"] = "v2"
    (tmp_path / "goal.json").write_text(json.dumps(goal))
    current = compact_planner_view(tmp_path, full)
    repeated = compact_planner_view(tmp_path, current)
    assert repeated["baseline_source"] == current["baseline_source"]
    assert repeated["history_navigation"] == current["history_navigation"]
