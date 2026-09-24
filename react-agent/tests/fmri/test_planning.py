"""Plan validator, Guard DSL, and policy resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from react_agent.fmri.config import load_config, resolve_policy_backend
from react_agent.fmri.planning.guards import eval_guard
from react_agent.fmri.planning.schemas import Guard, PlanProposal, PlanStep
from react_agent.fmri.planning.validator import PlanValidationError, validate_proposal


def _catalog(*names: str, ready: bool = True) -> list[dict]:
    return [
        {
            "tool_id": name,
            "ready": ready,
            "applicability": "available" if ready else "unavailable",
        }
        for name in names
    ]


def test_rule_still_forces_none() -> None:
    cfg = load_config()
    policy, backend = resolve_policy_backend(cfg, policy="rule", backend="deepseek")
    assert policy == "rule"
    assert backend == "none"


def test_planned_not_rewritten_to_none() -> None:
    cfg = load_config()
    cfg.deepseek_api_key = "dummy"
    policy, backend = resolve_policy_backend(cfg, policy="planned", backend="deepseek")
    assert policy == "planned"
    assert backend == "deepseek"


def test_planned_without_key_fails() -> None:
    cfg = load_config()
    cfg.deepseek_api_key = None
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        resolve_policy_backend(cfg, policy="planned", backend="deepseek")


def test_reject_unknown_tool_and_code_guard() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s1", question_id="gray_contrast", tool_id="not_a_tool"),
        ],
        next_step_id="s1",
    )
    with pytest.raises(PlanValidationError, match="unknown_tool"):
        validate_proposal(
            proposal,
            catalog=_catalog("gray_control_contrast"),
            required_questions=["gray_contrast"],
            remaining_lm_calls=3,
            remaining_tool_calls=3,
            evidence_ids=set(),
            resources={"gray_control": True},
        )


def test_reject_dependency_cycle() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="a", question_id="gray_contrast", tool_id="gray_control_contrast", depends_on=["b"]),
            PlanStep(step_id="b", question_id="temporal", tool_id="stimulus_temporal_profile", depends_on=["a"]),
        ],
        next_step_id="a",
    )
    with pytest.raises(PlanValidationError, match="dependency_cycle"):
        validate_proposal(
            proposal,
            catalog=_catalog("gray_control_contrast", "stimulus_temporal_profile"),
            required_questions=["gray_contrast", "temporal"],
            remaining_lm_calls=3,
            remaining_tool_calls=3,
            evidence_ids=set(),
            resources={"gray_control": True},
        )


def test_reject_cortex_mae_without_reference_or_feature_goal() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(
                step_id="s1",
                question_id="cortexmae_embedding",
                tool_id="cortex_mae",
                expected_evidence_type="quality",
            )
        ],
        next_step_id="s1",
        resource_gaps=["reference"],
    )
    with pytest.raises(PlanValidationError, match="cortex_mae_requires_reference"):
        validate_proposal(
            proposal,
            catalog=_catalog("cortex_mae"),
            required_questions=[],
            remaining_lm_calls=3,
            remaining_tool_calls=3,
            evidence_ids=set(),
            resources={},
            run_goal="quality_screening",
            has_reference_score=False,
        )


def test_guard_unknown_not_false() -> None:
    guard = Guard(op="metric_compare", metric_path="basic_statistics.mean", comparator="gt", threshold_ref="missing")
    assert eval_guard(
        guard,
        ledger={},
        resources={},
        steps={},
        findings=set(),
        metrics={},
        thresholds={},
    ) == "unknown"


def test_required_question_cannot_be_dropped() -> None:
    proposal = PlanProposal(goal="screen", steps=[], next_step_id=None)
    with pytest.raises(PlanValidationError, match="required_question_dropped"):
        validate_proposal(
            proposal,
            catalog=_catalog("gray_control_contrast"),
            required_questions=["gray_contrast"],
            remaining_lm_calls=3,
            remaining_tool_calls=3,
            evidence_ids=set(),
            resources={"gray_control": True},
        )


def test_answered_required_question_not_dropped() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s1", question_id="gray_contrast", tool_id="gray_control_contrast"),
        ],
        next_step_id="s1",
    )
    catalog = [
        {
            "tool_id": "validate_input",
            "ready": False,
            "applicability": "available",
            "already_completed": True,
            "answers_questions": ["input_mapping"],
        },
        {
            "tool_id": "gray_control_contrast",
            "ready": True,
            "applicability": "available",
            "answers_questions": ["gray_contrast"],
        },
    ]
    validate_proposal(
        proposal,
        catalog=catalog,
        required_questions=["input_mapping", "gray_contrast"],
        remaining_lm_calls=3,
        remaining_tool_calls=3,
        evidence_ids=set(),
        resources={"gray_control": True},
        completed_tools={"validate_input"},
        ledger={"input_mapping": {"status": "described"}},
    )


def test_answered_required_via_ledger_only() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s1", question_id="temporal", tool_id="stimulus_temporal_profile"),
        ],
        next_step_id="s1",
    )
    validate_proposal(
        proposal,
        catalog=_catalog("stimulus_temporal_profile"),
        required_questions=["input_mapping", "temporal"],
        remaining_lm_calls=3,
        remaining_tool_calls=3,
        evidence_ids=set(),
        resources={},
        ledger={"input_mapping": {"status": "described"}},
    )


def test_completed_l0_tool_not_rejected() -> None:
    catalog = [
        {
            "tool_id": "validate_input",
            "ready": False,
            "applicability": "available",
            "already_completed": True,
        },
        {
            "tool_id": "gray_control_contrast",
            "ready": True,
            "applicability": "available",
        },
    ]
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s0", question_id="input_mapping", tool_id="validate_input"),
            PlanStep(step_id="s1", question_id="gray_contrast", tool_id="gray_control_contrast"),
        ],
        next_step_id="s1",
    )
    validate_proposal(
        proposal,
        catalog=catalog,
        required_questions=["input_mapping", "gray_contrast"],
        remaining_lm_calls=3,
        remaining_tool_calls=3,
        evidence_ids=set(),
        resources={"gray_control": True},
        completed_tools={"validate_input"},
    )
    assert proposal.steps[0].status == "completed"
    assert proposal.steps[1].status == "pending"


def test_completed_l0_via_completed_tools_only() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s0", question_id="input_mapping", tool_id="validate_input"),
            PlanStep(step_id="s1", question_id="gray_contrast", tool_id="gray_control_contrast"),
        ],
        next_step_id="s1",
    )
    validate_proposal(
        proposal,
        catalog=_catalog("validate_input", "gray_control_contrast", ready=False),
        required_questions=["input_mapping", "gray_contrast"],
        remaining_lm_calls=3,
        remaining_tool_calls=3,
        evidence_ids=set(),
        resources={"gray_control": True},
        completed_tools={"validate_input", "gray_control_contrast"},
    )
    assert all(step.status == "completed" for step in proposal.steps)


def test_incomplete_not_ready_still_rejected() -> None:
    proposal = PlanProposal(
        goal="screen",
        steps=[
            PlanStep(step_id="s0", question_id="input_mapping", tool_id="validate_input"),
        ],
        next_step_id="s0",
    )
    with pytest.raises(PlanValidationError, match="tool_not_ready:validate_input"):
        validate_proposal(
            proposal,
            catalog=_catalog("validate_input", ready=False),
            required_questions=["input_mapping"],
            remaining_lm_calls=3,
            remaining_tool_calls=3,
            evidence_ids=set(),
            resources={},
        )


def test_plan_rejected_progress_line_shows_error() -> None:
    from react_agent.fmri.progress import format_progress_line

    assert format_progress_line(
        {"type": "plan.rejected", "error": "tool_not_ready:validate_input"}
    ) == "[plan] rejected tool_not_ready:validate_input"


def test_plan_validation_error_not_logged_as_api_failed(tmp_path: Path) -> None:
    from react_agent.fmri.planning.policy import _api_failure, _validation_failure
    from react_agent.fmri.progress import Progress, bind_progress

    events = tmp_path / "events.jsonl"
    sink: list[dict] = []
    with bind_progress(Progress(events_path=events, sink=sink, print_cli=False)):
        decision = _validation_failure(
            PlanValidationError("tool_not_ready:validate_input"),
            events,
        )
        again = _api_failure(
            object(),
            {},
            PlanValidationError("tool_not_ready:validate_input"),
            events,
        )
    kinds = [row["type"] for row in sink]
    assert "api.failed" not in kinds
    assert kinds.count("plan.rejected") == 2
    assert all(
        row.get("line") == "[plan] rejected tool_not_ready:validate_input"
        for row in sink
        if row["type"] == "plan.rejected"
    )
    assert decision.stop_reason == "degraded_execution"
    assert "Plan failed validation" in decision.reason
    assert "API" not in decision.reason
    assert again.stop_reason == "degraded_execution"


def test_stop_reason_stays_short_code() -> None:
    from react_agent.fmri.progress import format_progress_line

    long_note = (
        "input_mapping is already answered by validate_input:abc. "
        "No further spend is justified."
    )
    assert format_progress_line(
        {
            "type": "decision",
            "via": "planned",
            "decision": {
                "action": "stop",
                "stop_reason": "configured_checks_complete",
                "reason": long_note,
            },
        }
    ) == "[decision] planned stop configured_checks_complete"
    assert format_progress_line(
        {
            "type": "stop",
            "verdict": "passed_configured_checks",
            "stop_reason": "configured_checks_complete",
        }
    ) == "[stop] passed_configured_checks configured_checks_complete"
