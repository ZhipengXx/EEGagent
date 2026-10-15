"""Independent pre-code Critic configuration and actual bounded delivery."""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from react_agent.eeg_research.agentic.role_config import resolve_role_config
from react_agent.eeg_research.agentic.runtime_prompts import runtime_mission
from react_agent.eeg_research.agentic.schemas import (
    GoalSpec, ResearchExecutionConfig, ResearchPlanCritique, ResearchProposalDraft,
)


def test_legacy_execution_and_reviewer_identity_stay_unchanged():
    config = ResearchExecutionConfig.model_validate({"version": "single_plan_v1"})
    assert config.planning_mode == "combined"
    assert config.implementation_review_mode == "legacy"
    assert config.max_critic_api_attempts == 2
    assert config.model_dump() == {
        "version": "single_plan_v1", "target_k": 3, "allow_degraded_singleton": False,
        "proposal_correction_calls": 1, "max_planner_api_attempts": 2,
        "max_engineering_repairs": 2, "max_completed_rounds": None,
        "paid_maintenance": None, "budget": {},
    }
    reviewer = resolve_role_config({"research_execution": config.model_dump()}, "candidate_reviewer", environ={})
    old = {"model": "deepseek-flash", "thinking": True, "reasoning_effort": "high",
           "max_tokens": 16384, "timeout_s": 180.0, "base_url": "https://api.deepseek.com",
           "request_protocol": "eeg_json_final_stop.v1"}
    expected = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert reviewer.role_config_hash == expected
    assert reviewer.sources == {name: "default" for name in (
        "model", "thinking", "reasoning_effort", "max_tokens", "timeout_s")}


def test_explicit_staged_settings_and_critic_alias_validate_in_goal():
    goal = GoalSpec.model_validate({"goal_id": "staged-config-only", "max_llm_calls": 0,
        "max_training_jobs": 0, "max_gpu_seconds": 0,
        "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic",
            "implementation_review_mode": "single_pass", "max_critic_api_attempts": 1},
        "role_models": {"critic": {"model": "synthetic-pro"}}})
    configured = goal.model_dump()["research_execution"]
    assert configured["planning_mode"] == "propose_then_critic"
    assert configured["implementation_review_mode"] == "single_pass"
    assert configured["max_critic_api_attempts"] == 1
    critic = resolve_role_config(goal.model_dump(), "critic", environ={})
    assert critic.role == "research_critic" and critic.model == "synthetic-pro"
    assert critic.thinking and critic.reasoning_effort == "high"
    assert critic.max_tokens == 16384 and critic.timeout_s == 240
    assert resolve_role_config({}, "research_critic", environ={}).model == "deepseek-v4-pro"


def test_staged_planner_defaults_to_pro_thinking_with_legacy_and_override_compatibility():
    goal = {"research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic"}}
    planner = resolve_role_config(goal, "planner", environ={})
    assert planner.model == "deepseek-v4-pro" and planner.thinking
    assert planner.reasoning_effort == "high" and planner.max_tokens == 32768
    assert planner.timeout_s == 240 and planner.sources["model"] == "staged_planner_default"
    assert resolve_role_config({}, "planner", environ={}).model == "deepseek-flash"
    assert resolve_role_config({"research_execution": {"version": "single_plan_v1"}},
                               "planner", environ={}).model == "deepseek-flash"
    explicit = {**goal, "role_models": {"planner": {"model": "synthetic-planner", "thinking": False}}}
    planner = resolve_role_config(explicit, "planner", environ={})
    assert planner.model == "synthetic-planner" and not planner.thinking
    assert planner.reasoning_effort is None and planner.sources["model"] == "goal"


def test_standard_new_goal_pins_both_planning_roles_above_legacy_flash_environment():
    from react_agent.eeg_research.agentic.cli import goal
    submitted = goal("pro-thinking-config")
    env = {"DEEPSEEK_FAST_MODEL": "deepseek-flash", "DEEPSEEK_THINKING": "false"}
    for role in ("research_planner", "research_critic"):
        configured = resolve_role_config(submitted, role, environ=env)
        assert configured.model == "deepseek-v4-pro" and configured.thinking
        assert configured.reasoning_effort == "high" and configured.sources["model"] == "goal"


@pytest.mark.parametrize("attempts", [0, 3])
def test_critic_attempt_limit_is_bounded(attempts):
    with pytest.raises(ValidationError):
        ResearchExecutionConfig.model_validate({"version": "single_plan_v1", "max_critic_api_attempts": attempts})


def test_single_pass_reviewer_defaults_allow_explicit_model_profile_overrides():
    goal = {"research_execution": {"version": "single_plan_v1", "implementation_review_mode": "single_pass"}}
    reviewer = resolve_role_config(goal, "reviewer", environ={})
    assert reviewer.model == "deepseek-flash" and reviewer.thinking is False
    assert reviewer.reasoning_effort is None and reviewer.max_tokens == 8192
    assert reviewer.sources["max_tokens"] == "single_pass_default"
    explicit = {**goal, "role_models": {"reviewer": {"model": "synthetic-reviewer", "thinking": True,
                    "reasoning_effort": "high", "max_tokens": 12000}}}
    reviewer = resolve_role_config(explicit, "candidate_reviewer", environ={})
    assert reviewer.model == "synthetic-reviewer" and reviewer.thinking and reviewer.max_tokens == 12000
    reviewer = resolve_role_config(explicit, "candidate_reviewer", {"thinking": False, "max_tokens": 4096}, environ={})
    assert reviewer.model == "synthetic-reviewer" and not reviewer.thinking
    assert reviewer.reasoning_effort is None and reviewer.max_tokens == 4096


def test_generation_schema_disallows_self_selection_but_keeps_invalid_siblings():
    proposals = [{"proposal_id": "independently-invalid"}, {"proposal_id": "another-invalid"}]
    raw = {"protocol_version": "research_proposals.v1", "proposals": proposals}
    assert ResearchProposalDraft.model_validate(raw).proposals == proposals
    for field, value in {"evaluations": [], "selected_proposal_id": "p1", "selection_rationale": "Self-selected"}.items():
        with pytest.raises(ValidationError):
            ResearchProposalDraft.model_validate({**raw, field: value})
    schema = ResearchProposalDraft.model_json_schema()
    assert set(schema["properties"]) == {"protocol_version", "proposals"}
    assert schema["properties"]["proposals"]["minItems"] == 2
    assert "scientific_intent" in schema["properties"]["proposals"]["items"]["properties"]


def test_critique_schema_binds_pool_and_separate_evaluations():
    raw = {"protocol_version": "research_plan_critique.v1", "proposal_pool_digest": "actual-pool",
        "evaluations": [{"proposal_id": "p1", "rank": 1, "value_rationale": "Higher information value",
            "implementation_risk": "Small wrapper change"},
            {"proposal_id": "p2", "rank": 2, "value_rationale": "Less supported hypothesis",
            "implementation_risk": "Larger architecture change"}],
        "selected_proposal_id": "p1", "selection_rationale": "Prefer p1 for bounded feedback."}
    assert ResearchPlanCritique.model_validate(raw).model_dump() == raw
    with pytest.raises(ValidationError):
        ResearchPlanCritique.model_validate({**raw, "proposals": []})
    with pytest.raises(ValidationError):
        ResearchPlanCritique.model_validate({**raw, "evaluations": raw["evaluations"][:1]})


def test_actual_critic_prepared_request_delivers_complete_pool_and_conditional_memory(tmp_path, monkeypatch):
    from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
    from react_agent.eeg_research.agentic.context.service import ContextService
    from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready, canonical, digest
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    goal = {"research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic"},
            "memory": {"enabled": False}}
    (tmp_path / "goal.json").write_text(json.dumps(goal))
    (tmp_path / "campaign_state.json").write_text('{"status":"planning"}')
    (tmp_path / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval"}')
    proposals = [{"proposal_id": f"p{number}", "question": "Compare a bounded intervention",
        "scientific_intent": (f"Conditional hypothesis {number}. " * 100),
        "value_rationale": (f"Competing rationale {number}. " * 100)} for number in (1, 2)]
    pool = {"proposals": proposals, "native_cost_by_id": {"p1": {"training_jobs": 2}, "p2": {"training_jobs": 2}}}
    pool_digest = digest(pool)
    pool["pool_digest"] = pool_digest
    payload = {"goal": goal, "proposal_pool": pool, "proposal_pool_digest": pool_digest,
        "experiment_observations": [{"status": "failed", "training_jobs": 0}],
        "retrieved_knowledge": {"hits": [{"lesson": "Historical prior only",
            "applicability": "Different intervention; no current measured gain", "similarity": 0.41}]}}
    original = canonical(payload)
    system = runtime_mission("research_critic") + "\nOUTPUT_SCHEMA\n" + json.dumps(ResearchPlanCritique.model_json_schema())
    compiled = ContextService(tmp_path).compile(CompileRequest(tmp_path.name, "research_critic", payload,
        system=system, max_output_tokens=16384, allow_semantic_compaction=False))
    assert isinstance(compiled, Ready), compiled
    assert compiled.prepared.verify() and compiled.prepared.budget_report["ready_budget_ok"]
    def expand(value, root):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(root, value["$context_pointer"]), root)
            return {key: expand(child, root) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child, root) for child in value]
        return value
    delivered = expand(compiled.prepared.payload, compiled.prepared.payload)
    assert delivered["proposal_pool"] == pool
    assert delivered["proposal_pool_digest"] == pool_digest
    assert delivered["retrieved_knowledge"]["hits"][0]["similarity"] == 0.41
    assert delivered["experiment_observations"][0]["status"] == "failed"
    assert "Do not change proposals" in compiled.prepared.system
    assert "select the rank-1 proposal" in compiled.prepared.system
    assert canonical(payload) == original
    assert not (tmp_path / "llm_calls.jsonl").exists()


def test_stage_missions_separate_generation_comparison_and_coding():
    proposer = runtime_mission("research_proposer")
    critic = runtime_mission("research_critic")
    assert "Do not rank, evaluate, select a proposal" in proposer
    assert "before any Designer or Coder is called" in proposer
    assert "every proposal" in critic and "before any detailed design or coding" in critic
    assert "exact delivered proposal_pool_digest" in critic
    assert len(proposer) < 3000 and len(critic) < 3000
