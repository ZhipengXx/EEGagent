"""Default single-target task bounds are allocation caps, never predictions."""
from __future__ import annotations

import json
import os

import pytest

from react_agent.eeg_research.agentic.cli import goal
from react_agent.eeg_research.agentic.plan_protocol import proposal_cost
from react_agent.eeg_research.agentic import owner_budget as budget


@pytest.fixture(autouse=True)
def no_deployment_configuration(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_args, **_kwargs: False)
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)


def synthetic_campaign(path, value, *, spent_jobs=0):
    (path / "goal.json").write_text(json.dumps(value))
    (path / "campaign_state.json").write_text(json.dumps({"status": "planning", "training_jobs": spent_jobs}))
    (path / "cost.json").write_text(json.dumps({"llm_calls": 0, "training_jobs": spent_jobs}))
    return path


def cost(value):
    return proposal_cost({"task_kind": "new_method", "scientific_intent": "Unknown new model",
        "evaluation_scope": {"phase": "full"}}, {"goal": value,
        "action_cost_estimates": {"run_full:baseline": {"estimated_cost": {
            "training_jobs": 1, "gpu_seconds": 999999}, "cost_confidence": "historical"}}})


def test_default_goal_admits_unknown_method_with_complete_runtime_bound(tmp_path):
    value = goal("synthetic_default")
    native = cost(value)
    assert native["known"] == {"training_jobs": None, "gpu_seconds": None}
    bound = native["runtime_allocation_bound"]
    assert bound["training_jobs"] == 2
    assert bound["gpu_seconds"] == pytest.approx(2 * value["max_gpu_seconds"] / value["max_training_jobs"])
    assert bound["authority"] == "maximum_runtime_allocation_not_cost_estimate"
    assert native["confidence"] == "unknown"
    camp = synthetic_campaign(tmp_path, value)
    preview = budget.preview_selected_plan(camp, task_kind="new_method", known_cost=native["known"], recovery_limits={})
    assert preview["ok"]
    assert preview["resources"]["training_jobs"] == 2
    assert preview["resources"]["gpu_seconds"] == bound["gpu_seconds"]


def test_runtime_allocation_bounds_two_native_jobs_and_per_gpu_deadline(tmp_path):
    value = goal("synthetic_native_bounds")
    camp = synthetic_campaign(tmp_path, value)
    owner = budget.allocate_selected_plan(camp, plan_id="bounded", task_kind="new_method",
        known_cost=cost(value)["known"], recovery_limits={})
    gpu_per_job = owner["allocated"]["gpu_seconds"] / owner["allocated"]["training_jobs"]
    with budget.budget_owner(camp, owner["owner_id"]):
        for job_id in ("matched_baseline", "candidate"):
            claim = budget.claim_training(camp, job_id, gpu_per_job)
            assert claim["requested"]["training_jobs"] == 1
            permit = budget.training_launch_admission(camp, job_id, 2, now=100)
            assert permit["deadline_at"] == pytest.approx(100 + gpu_per_job / 2)
            budget.settle_training(camp, job_id, 20)
        with pytest.raises(budget.OwnerBudgetError, match="training_owner_unavailable:training_jobs"):
            budget.claim_training(camp, "unapproved_third", gpu_per_job)
    snapshot = budget.snapshot(camp)
    assert snapshot["global_spent_and_occupied"]["training_jobs"] == 2
    assert snapshot["global_spent_and_occupied"]["gpu_seconds"] == 40


def test_remaining_global_budget_must_fit_entire_cold_start_measurement(tmp_path):
    value = goal("synthetic_nearly_spent")
    camp = synthetic_campaign(tmp_path, value, spent_jobs=value["max_training_jobs"] - 1)
    preview = budget.preview_selected_plan(camp, task_kind="new_method", known_cost=cost(value)["known"], recovery_limits={})
    assert not preview["ok"] and preview["reason"] == "global_allocation_unavailable:training_jobs"


def test_existing_unconfigured_goal_is_not_silently_granted_an_envelope(tmp_path):
    value = goal("synthetic_existing")
    value["research_execution"].pop("budget", None)
    camp = synthetic_campaign(tmp_path, value)
    native = cost(value)
    assert "runtime_allocation_bound" not in native
    assert native["known"]["gpu_seconds"] is None
    preview = budget.preview_selected_plan(camp, task_kind="new_method", known_cost=native["known"], recovery_limits={})
    assert not preview["ok"] and preview["reason"] == "selected_chain_measurement_cost_unknown"
