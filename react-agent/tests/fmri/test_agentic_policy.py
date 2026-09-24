"""Planned policy loop: mock plan, abstain, no forced CortexMAE."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from react_agent.fmri.config import load_config
from react_agent.fmri.llm.mock import MockBackend
from react_agent.fmri.loop import run_sample
from react_agent.fmri.schemas import SampleSpec


def _plan(
    steps: list[dict],
    *,
    gaps: list[str] | None = None,
    stop_request: dict | None = None,
) -> dict:
    return {
        "schema_version": "plan.v1",
        "phase": "create",
        "goal": "quality_screening",
        "claim_scope": "numeric_consistency_only",
        "steps": [
            {
                "step_id": row["step_id"],
                "question_id": row["question_id"],
                "tool_id": row["tool_id"],
                "depends_on": row.get("depends_on") or [],
                "brief_reason": row.get("reason") or "planned",
                "expected_evidence_type": row.get("expected_evidence_type"),
                "supporting_evidence_ids": [],
                "memory_ids": [],
                "resource_binding_ids": [],
            }
            for row in steps
        ],
        "next_step_id": steps[0]["step_id"] if steps else None,
        "unresolved_questions": [s["question_id"] for s in steps],
        "resource_gaps": gaps or [],
        "stop_request": stop_request,
        "memory_usage": [],
    }


def _tiny_spec(tmp_path: Path) -> SampleSpec:
    arr = np.linspace(0.1, 0.2, 16 * 32, dtype=np.float32).reshape(16, 32)
    path = tmp_path / "tiny.npy"
    np.save(path, arr)
    return SampleSpec(
        sample_id="tiny16",
        fmri_path=str(path),
        time_axis=0,
        sampling_interval_s=1.0,
        spatial_representation="surface",
        space_name="fsaverage5",
        normalization="none_raw_signed",
        generation_profile_id="static_gray4_image1_gray11_tribev2_v1",
        expected_shape=(16, 32),
    )


def test_planned_executes_l1_not_cortexmae(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.planning.enabled = True
    cfg.planning.require_gray_question_for_non_control = False
    cfg.planning.require_temporal_description_for_non_control = True
    cfg.required_checks = ["validate_input", "basic_statistics"]
    cfg.enabled_tools = [
        "validate_input",
        "basic_statistics",
        "stimulus_temporal_profile",
        "cortex_mae",
    ]
    cfg.memory.enabled = True
    cfg.memory.mode = "read_write"
    cfg.memory.path = str(tmp_path / "mem.sqlite3")
    cfg.memory.namespace = "unit"
    cfg.memory.curator_enabled = False
    spec = _tiny_spec(tmp_path)
    mock = MockBackend(
        plans=[
            _plan(
                [
                    {
                        "step_id": "s1",
                        "question_id": "temporal",
                        "tool_id": "stimulus_temporal_profile",
                    }
                ]
            )
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "planned",
            config=cfg,
            backend="mock",
            policy="planned",
            mock=mock,
        )
    )
    assert "cortex_mae" not in report["executed_tools"]
    assert "stimulus_temporal_profile" in report["executed_tools"]
    assert report.get("resolved_policy") == "planned"
    assert report["cost"]["lm_calls"] >= 1
    assert report["cost"]["accepted_plan_count"] >= 1
    assert report["cost"].get("llm_calls_attempted", 0) >= 1


def test_planned_abstains_when_required_blocked(tmp_path: Path) -> None:
    cfg = load_config()
    cfg.planning.enabled = True
    cfg.planning.require_gray_question_for_non_control = True
    cfg.planning.require_temporal_description_for_non_control = False
    cfg.required_checks = ["validate_input", "basic_statistics"]
    cfg.enabled_tools = ["validate_input", "basic_statistics"]
    cfg.check_profile.required_questions = ["input_mapping", "gray_contrast"]
    cfg.basic_statistics.enable_demo_temporal_heuristics = False
    spec = _tiny_spec(tmp_path)
    mock = MockBackend(
        plans=[
            _plan([], gaps=["gray_control"]),
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "abstain",
            config=cfg,
            backend="mock",
            policy="planned",
            mock=mock,
        )
    )
    assert report["screening_decision"] == "abstain"
    assert report["verdict"] != "passed_configured_checks"


def test_planned_accepts_completed_validate_input(tmp_path: Path) -> None:
    """Planner may re-list L0 validate_input after it already succeeded."""
    cfg = load_config()
    cfg.planning.enabled = True
    cfg.planning.require_gray_question_for_non_control = False
    cfg.planning.require_temporal_description_for_non_control = True
    cfg.required_checks = ["validate_input", "basic_statistics"]
    cfg.enabled_tools = [
        "validate_input",
        "basic_statistics",
        "stimulus_temporal_profile",
    ]
    spec = _tiny_spec(tmp_path)
    mock = MockBackend(
        plans=[
            _plan(
                [
                    {
                        "step_id": "s0",
                        "question_id": "input_mapping",
                        "tool_id": "validate_input",
                    },
                    {
                        "step_id": "s1",
                        "question_id": "temporal",
                        "tool_id": "stimulus_temporal_profile",
                    },
                ]
            )
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "replay_l0",
            config=cfg,
            backend="mock",
            policy="planned",
            mock=mock,
        )
    )
    assert report["stop_reason"] != "degraded_execution"
    assert "stimulus_temporal_profile" in report["executed_tools"]
    assert report["cost"]["accepted_plan_count"] >= 1


def test_planned_omits_answered_input_mapping(tmp_path: Path) -> None:
    """L0 already answered input_mapping; plan need not re-list it."""
    cfg = load_config()
    cfg.planning.enabled = True
    cfg.planning.require_gray_question_for_non_control = False
    cfg.planning.require_temporal_description_for_non_control = True
    cfg.check_profile.required_questions = ["input_mapping", "temporal"]
    cfg.memory.curator_enabled = False
    cfg.required_checks = ["validate_input", "basic_statistics"]
    cfg.enabled_tools = [
        "validate_input",
        "basic_statistics",
        "stimulus_temporal_profile",
    ]
    spec = _tiny_spec(tmp_path)
    long_note = (
        "input_mapping is already answered by validate_input. "
        "The single plausible remaining gap is calibrated_reference."
    )
    mock = MockBackend(
        plans=[
            _plan(
                [
                    {
                        "step_id": "s1",
                        "question_id": "temporal",
                        "tool_id": "stimulus_temporal_profile",
                    }
                ],
                stop_request={"reason": long_note, "screening_hint": "abstain"},
            )
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "omit_l0",
            config=cfg,
            backend="mock",
            policy="planned",
            mock=mock,
        )
    )
    assert report["stop_reason"] == "configured_checks_complete"
    assert " " not in report["stop_reason"]
    assert "stimulus_temporal_profile" in report["executed_tools"]
    assert report["cost"]["accepted_plan_count"] == 1
    assert report["cost"]["lm_calls"] == 1


def test_rule_zero_lm_calls(tmp_path: Path) -> None:
    cfg = load_config()
    spec = _tiny_spec(tmp_path)
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "rule",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    assert report["cost"]["lm_calls"] == 0
    assert report["cost"].get("planner_calls_succeeded", 0) == 0
