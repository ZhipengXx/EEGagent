"""V1.7 split, fixed-bank scores, budget, and controller choices."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from react_agent.eeg_research.controller import (
    PlanningError,
    ResearchController,
    StaleState,
    action_for_evidence,
    clip_raw,
    open_adaptive,
    parse_planner_reply,
    planner_context,
    retrieve_memory,
)
from react_agent.eeg_research.executor import ResearchBudget
from react_agent.eeg_training.launch import run_adaptive_loop, task_card
from react_agent.eeg_training.fixed_bank import fixed_bank_accuracy
from react_agent.eeg_training.protocol import Design, SplitError, protocol_label, resolve_run_gpu, validate_design


def test_all_subjects_are_image_holdout_and_missing_holdout_blocks() -> None:
    pooled = Design("eeg", "inter-subject", "all", training_strategy="pooled_subjects")
    validate_design(pooled)
    assert protocol_label(pooled) == "多被试合训 · 图像留出验证"
    blocked = Design(
        "eeg",
        "inter-subject",
        "all",
        generalization_target="held_out_subject",
    )
    with pytest.raises(SplitError, match="held_out_subject_missing"):
        validate_design(blocked)


def test_fixed_bank_ignores_chunk_size_and_uses_image_ids() -> None:
    queries = [("q1", [1.0, 0.0]), ("q2", [0.0, 1.0])]
    bank = [("img-a", [1.0, 0.0]), ("img-b", [0.0, 1.0]), ("img-c", [0.2, 0.2])]
    positives = {"q1": {"img-a", "img-c"}, "q2": {"img-b"}}
    whole = fixed_bank_accuracy(queries, bank, positives, chunk_size=2)
    chunked = fixed_bank_accuracy(queries, bank, positives, chunk_size=1)
    assert whole == chunked
    assert whole["fixed_bank_top1"] == 1.0


def test_gpu_charge_scales_with_card_count_and_does_not_reset() -> None:
    one = ResearchBudget(max_trials=6, max_lm_calls=12, max_execution_attempts=4, gpu_seconds=100.0, per_trial_timeout_s=10.0)
    two = ResearchBudget(max_trials=6, max_lm_calls=12, max_execution_attempts=4, gpu_seconds=100.0, per_trial_timeout_s=10.0)
    assert two.charge_allocated_gpu(2, 4.0) == 2 * one.charge_allocated_gpu(1, 4.0)
    one.charge_allocated_gpu(1, 4.0)
    assert one.gpu_seconds_used == 8.0
    with pytest.raises(Exception, match="gpu_budget_exhausted"):
        one.charge_allocated_gpu(1, 100.0)


def test_observations_choose_different_actions_and_missing_key_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert action_for_evidence("positive_mapping_error", trials_used=1, max_trials=6, memory_hit=False) == "analyze_retrieval_errors"
    assert action_for_evidence("repeat_no_gain", trials_used=1, max_trials=6, memory_hit=True) == "stop_research"
    assert action_for_evidence("new_evidence", trials_used=2, max_trials=6, memory_hit=False) == "train_candidate"
    assert action_for_evidence("no_legal_question", trials_used=1, max_trials=6, memory_hit=False) == "stop_research"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert "不会退回固定两试" in open_adaptive()
    memory = retrieve_memory(
        [
            {"memory_id": 1, "task_hash": "task", "metric": "fixed_bank_top1"},
            {"memory_id": 2, "task_hash": "task", "metric": "fixed_bank_top1", "test_result": 0.2},
            {"memory_id": 3, "task_hash": "other", "metric": "fixed_bank_top1"},
        ],
        task_hash="task",
        metric="fixed_bank_top1",
    )
    assert memory["memory_used"] == [1]
    assert memory["cold_start"] is False
    assert os.environ.get("DEEPSEEK_API_KEY") is None


def test_empty_gpu_does_not_mean_every_card(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = Design("eeg", "intra-subject", "sub-01", gpu_mode="explicit")
    with pytest.raises(SplitError, match="gpu_not_selected"):
        resolve_run_gpu(explicit)
    monkeypatch.setattr(
        "react_agent.eeg_training.protocol.list_gpus",
        lambda: [{"index": 1, "memory_free_mb": 10}, {"index": 3, "memory_free_mb": 90}],
    )
    bound = resolve_run_gpu(Design("eeg", "intra-subject", "sub-01", gpu_mode="auto_one"))
    assert bound.gpu == (3,)


def test_stale_state_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    controller = ResearchController(tmp_path)
    first = controller.decide(
        campaign_id="task",
        observation="inspect",
        trials_used=0,
        max_trials=6,
        memory_rows=[],
        expected_version=0,
    )
    assert first["status"] == "planning_blocked"
    assert first["legacy_fixed"] is False
    with pytest.raises(StaleState, match="stale_state_version"):
        controller.decide(
            campaign_id="task",
            observation="new_evidence",
            trials_used=2,
            max_trials=6,
            memory_rows=[],
            expected_version=0,
        )


def _adaptive_budget() -> ResearchBudget:
    return ResearchBudget(
        max_trials=6,
        max_lm_calls=12,
        max_execution_attempts=4,
        gpu_seconds=10.0,
        per_trial_timeout_s=10.0,
    )


def test_illegal_plan_does_not_start_training_or_fall_back(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    started: list[int] = []
    monkeypatch.setattr(
        "react_agent.eeg_training.launch._start_one_trial",
        lambda *_args, **_kwargs: started.append(1),
    )
    design = Design("eeg", "intra-subject", "sub-01", policy="adaptive", gpu=(0,))
    outcome = run_adaptive_loop(
        design,
        tmp_path,
        _adaptive_budget(),
        task_card(design),
        tmp_path,
        backend=lambda _context: {"action": "delete_data"},
    )
    assert started == []
    assert outcome["started"] is False
    research = outcome["research"]
    assert research["status"] == "planning_blocked"
    assert research["legacy_fixed"] is False
    assert research["raw"]["action"] == "delete_data"
    assert "delete_data" in research["detail"]
    assert "正在请求规划" in outcome["notes"]


def test_chinese_label_and_nested_action_map_to_legal_ids() -> None:
    prompt = (
        Path(__file__).resolve().parents[2] / "src/react_agent/eeg_research/prompts/controller.txt"
    ).read_text(encoding="utf-8")
    for key in ("inspect_task_data", "train_candidate", "stop_research", "learning_rate", "weight_decay"):
        assert key in prompt
    assert "Simplified Chinese" in prompt
    assert "already completed" in prompt
    assert "tried_settings" in prompt
    assert "twice or half" in prompt
    assert "status running" in prompt
    assert "0.005" in prompt
    trained = parse_planner_reply({"action": "训练", "reason": "需要一条基线"})
    assert trained["action"] == "train_candidate"
    assert trained["action_label"] == "训练"
    nested = parse_planner_reply({"decision": {"action": "停止"}, "reason": "没有新证据"})
    assert nested["action"] == "stop_research"
    forwarded = parse_planner_reply({"next_action": "inspect_task_data", "reason": "先看划分"})
    assert forwarded["action"] == "inspect_task_data"
    with pytest.raises(PlanningError, match="实际动作"):
        parse_planner_reply({"summary": "先看看数据"})
    with pytest.raises(PlanningError, match="delete_data"):
        parse_planner_reply({"action": "delete_data"})


def test_blocked_reply_keeps_a_clipped_raw_object(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    controller = ResearchController(tmp_path)
    huge = "x" * 5000
    blocked = controller.decide(
        campaign_id="task",
        observation="inspect",
        trials_used=0,
        max_trials=6,
        memory_rows=[],
        backend=lambda _context: {"action": "delete_data", "reason": huge},
    )
    assert blocked["status"] == "planning_blocked"
    assert blocked["legacy_fixed"] is False
    assert isinstance(blocked["raw"], str)
    assert len(blocked["raw"].encode("utf-8")) <= 2048
    assert "delete_data" in blocked["raw"]
    assert clip_raw({"action": "停止"}) == {"action": "停止"}


def test_prior_curve_stays_a_record_until_a_new_trial(tmp_path) -> None:
    from react_agent.eeg_training.train_entry import read_train_status

    camp = tmp_path / "camp"
    camp.mkdir()
    (camp / "status.json").write_text(json.dumps({"phase": "finished", "epoch": 23, "epochs": 50}), encoding="utf-8")
    (camp / "history.jsonl").write_text(
        '{"epoch": 1, "train_loss": 1.0, "val_top1": 0.2, "val_top5": 0.4}\n',
        encoding="utf-8",
    )
    (camp / "metrics.json").write_text(
        json.dumps({"primary_metric": 0.3630694088088461, "test_result": None}),
        encoding="utf-8",
    )
    (camp / "research_state.json").write_text(
        json.dumps({"phase": "planned", "status": "planned", "action": "inspect_task_data", "action_label": "查看数据", "reason": "冷启动", "legacy_fixed": False}),
        encoding="utf-8",
    )
    status = read_train_status(tmp_path, "camp")
    assert status["prior_curve"] is True
    assert status["phase"] == "finished"
    script = Path(__file__).resolve().parents[2] / "src/react_agent/fmri/workbench_static/app.js"
    assert "已有训练记录" in script.read_text(encoding="utf-8")
    trial = camp / "trials" / "t1"
    trial.mkdir(parents=True)
    (trial / "history.jsonl").write_text(
        '{"epoch": 1, "train_loss": 1.0, "val_top1": 0.2, "val_top5": 0.4}\n',
        encoding="utf-8",
    )
    assert read_train_status(tmp_path, "camp")["prior_curve"] is False
    later = camp / "trials" / "t3"
    later.mkdir()
    (later / "status.json").write_text(json.dumps({"phase": "training", "epoch": 9, "epochs": 50}), encoding="utf-8")
    assert read_train_status(tmp_path, "camp")["epoch"] == 9
    assert json.loads((camp / "metrics.json").read_text(encoding="utf-8"))["primary_metric"] == 0.3630694088088461


def test_prompt_context_omits_test_scores() -> None:
    context = planner_context(
        campaign_id="task",
        observation={"validation_metric": 0.2, "test_result": 0.9},
        trials_used=1,
        max_trials=6,
        memory_rows=[{"memory_id": 2, "task_hash": "task", "metric": "fixed_bank_top1", "test_result": 0.9}],
    )
    assert "test_result" not in context["observation"]
    assert context["memory"]["memory_used"] == []
    assert context["memory"]["excluded"][0]["reason"] == "test_result"
