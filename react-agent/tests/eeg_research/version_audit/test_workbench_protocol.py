"""Workbench route protocol propagation and read-only campaign projections."""
from __future__ import annotations

import json

import pytest

from react_agent.eeg_research.agentic import cli, loop, view
from react_agent.eeg_training import launch
from react_agent.eeg_training.protocol import Design
from react_agent.fmri import workbench


@pytest.mark.parametrize("requested,expected", [(None, "single_plan_v1"),
    ("single_plan_v1", "single_plan_v1"), ("legacy", "legacy")])
def test_ui_run_passes_explicit_execution_choice(monkeypatch, requested, expected):
    design = Design("eeg", "intra-subject", "sub-01", policy="agentic")
    monkeypatch.setattr(launch, "parse_design", lambda form: design)
    monkeypatch.setattr(launch, "launch_design", lambda *args: {"ok": True, "blockers": []})
    seen = []
    def open_run(design, payload, **kwargs):
        seen.append(kwargs)
        return {**payload, "execution_version": kwargs["execution_version"]}
    monkeypatch.setattr(cli, "open_agentic_run", open_run)
    form = {"action": "run", "request_id": "ui-choice"}
    if requested is not None:
        form["execution_version"] = requested
    status, result = workbench.submit_retrieval(form)
    assert status == 200 and result["execution_version"] == expected
    assert seen[0]["execution_version"] == expected and seen[0]["request_id"] == "ui-choice"


def test_invalid_ui_choice_refuses_before_launch(monkeypatch):
    monkeypatch.setattr(launch, "parse_design", lambda form: Design("eeg", "intra-subject", "sub-01", policy="agentic"))
    monkeypatch.setattr(launch, "launch_design", lambda *args: pytest.fail("invalid choice reached launch"))
    status, result = workbench.submit_retrieval({"action": "run", "execution_version": "other"})
    assert status == 400 and result["blockers"] == ["invalid_execution_version"]


@pytest.mark.parametrize("version", ["legacy", "single_plan_v1"])
def test_ui_status_projects_actual_frozen_protocol_without_mutation(tmp_path, version):
    goal = {"goal_id": "visible", "max_llm_calls": 20, "max_training_jobs": 0, "max_gpu_seconds": 0}
    if version == "single_plan_v1":
        goal["research_execution"] = {"version": version, "paid_maintenance": "off"}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "ui-synthetic", "task_type": "eeg_image_retrieval"}, request_id="visible")
    camp = tmp_path / "visible"
    state = loop.load_state(camp)
    state["selected_plan"] = {"plan_id": "synthetic_selected", "current_step": "implement"}
    loop.save_state(camp, state)
    originals = {name: (camp / name).read_bytes() for name in ("goal.json", "campaign_state.json", "confirmation_policy.json")}
    summary = view.campaign_summary(camp)
    detail = view.campaign_view(camp, include_source=False)
    for result in (summary, detail):
        assert result["execution_version"] == version
        assert result["paid_maintenance"] == ("off" if version == "single_plan_v1" else "legacy")
        assert result["selected_plan"]["plan_id"] == "synthetic_selected"
    assert all((camp / name).read_bytes() == body for name, body in originals.items())
    if version == "legacy":
        assert json.loads((camp / "goal.json").read_text()).get("research_execution") is None
