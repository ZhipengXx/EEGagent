"""The adaptive loop reads the trial trace from disk. No live DeepSeek, no GPU."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from react_agent.eeg_research.executor import ResearchBudget
from react_agent.eeg_research.trace import next_trial_index, read_trace
from react_agent.eeg_training.fixed_bank import FixedBankTally, fixed_bank_accuracy
from react_agent.eeg_training.launch import run_adaptive_loop, task_card
from react_agent.eeg_training.protocol import Design

DESIGN = Design("eeg", "intra-subject", "sub-01", policy="adaptive", gpu=(0,))


def _budget() -> ResearchBudget:
    return ResearchBudget(
        max_trials=6,
        max_lm_calls=12,
        max_execution_attempts=4,
        gpu_seconds=10.0,
        per_trial_timeout_s=10.0,
    )


def _finished(path: Path, *, ckpt: bytes, metrics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "status.json").write_text(json.dumps({"phase": "finished", "epoch": 3, "epochs": 50}), encoding="utf-8")
    (path / "last.ckpt").write_bytes(ckpt)
    (path / "history.jsonl").write_text(
        '{"epoch": 1, "train_loss": 1.0, "val_top1": 0.2, "val_top5": 0.4, "fixed_bank_top1": 0.3, "fixed_bank_top5": 0.6}\n',
        encoding="utf-8",
    )


def _fake_trainer(calls: list[str], score: float = 0.3):
    def run(trial_design, out_dir, index, *_args, **_kwargs):
        calls.append(f"t{index}")
        _finished(
            Path(out_dir) / "trials" / f"t{index}",
            ckpt=f"ckpt-{index}".encode(),
            metrics={
                "primary_metric": score,
                "metric_name": "fixed_bank_top1",
                "fixed_bank_top1": score,
                "fixed_bank_top5": 0.6,
                "within_batch_top1": 0.5,
                "test_result": {"top1": 0.91, "top5": 0.99},
            },
        )

    return run


def _script(*replies):
    seen: list[dict] = []

    def backend(context):
        seen.append(context)
        index = min(len(seen), len(replies)) - 1
        return replies[index]

    return backend, seen


def test_inspect_then_train_reads_the_trace(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    root_metrics = {"primary_metric": 0.3630694088088461, "metric_name": "top1", "test_result": None}
    (tmp_path / "metrics.json").write_text(json.dumps(root_metrics), encoding="utf-8")
    backend, seen = _script(
        {"action": "inspect_task_data", "reason": "先看数据"},
        {"action": "train_candidate", "reason": "需要一条基线"},
        {"action": "stop_research", "reason": "先停下来"},
    )
    outcome = run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend)
    assert calls == ["t1"]
    assert outcome["started"] is True
    assert outcome["research"]["action"] == "stop_research"
    assert outcome["research"]["legacy_fixed"] is False
    assert "inspect_task_data" not in seen[1]["legal_actions"]
    assert seen[1]["observation"]["checked"]["inspect_task_data"]["train_files"] == 1
    after = seen[2]
    assert after["observation"]["trials"][0]["fixed_bank_top1"] == 0.3
    assert after["observation"]["validation_metric"] == 0.3
    assert after["memory"]["memory_used"] == ["t1"]
    assert "inspect_task_data" in after["legal_actions"]
    blob = json.dumps(seen, ensure_ascii=False)
    assert "0.3630694088088461" not in blob
    assert "0.91" not in blob
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == root_metrics
    config = json.loads((tmp_path / "trials" / "t1" / "trial_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 0


def test_repeated_inspect_stops_without_training(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    backend, seen = _script({"action": "inspect_task_data", "reason": "再看一次"})
    outcome = run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend)
    assert calls == []
    assert len(seen) == 3
    assert outcome["started"] is False
    assert outcome["research"]["detail"] == "查看数据这一步已经做过。"


def test_existing_setting_is_reused_on_a_later_click(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    first, _ = _script(
        {"action": "train_candidate", "reason": "基线"},
        {"action": "stop_research", "reason": "停"},
    )
    run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=first)
    assert calls == ["t1"]

    later, seen = _script(
        {"action": "train_candidate", "reason": "基线"},
        {"action": "train_candidate", "reason": "改 weight decay", "changes": {"weight_decay": 0.01}},
        {"action": "stop_research", "reason": "停"},
    )
    outcome = run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=later)
    assert calls == ["t1", "t2"]
    assert seen[0]["trials_used"] == 1
    assert "这个设置已有结果（t1），直接复用，不再训练。" in outcome["notes"]
    assert outcome["trials_used"] == 2
    config = json.loads((tmp_path / "trials" / "t2" / "trial_config.json").read_text(encoding="utf-8"))
    assert config["weight_decay"] == 0.01


def test_replicate_takes_the_next_unused_seed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    backend, _ = _script(
        {"action": "train_candidate", "reason": "基线"},
        {"action": "replicate_candidate", "reason": "换 seed"},
        {"action": "stop_research", "reason": "停"},
    )
    run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend)
    assert calls == ["t1", "t2"]
    config = json.loads((tmp_path / "trials" / "t2" / "trial_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 1


def test_legacy_duplicates_are_scored_once_and_not_retrained(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    legacy = {"primary_metric": 0.4118990126144455, "metric_name": "top1", "test_result": None}
    for index in range(1, 5):
        _finished(tmp_path / "trials" / f"t{index}", ckpt=b"same", metrics=legacy)
    before = (tmp_path / "trials" / "t1" / "metrics.json").read_text(encoding="utf-8")
    scored: list[str] = []

    def scorer(_design, trial_dir, _base):
        scored.append(Path(trial_dir).name)
        payload = {
            "fixed_bank_top1": 0.35,
            "fixed_bank_top5": 0.7,
            "candidate_count": 1654,
            "test_result": {"top1": 0.8, "top5": 0.9, "candidate_count": 200},
        }
        (Path(trial_dir) / "eval_scores.json").write_text(json.dumps(payload), encoding="utf-8")

    backend, seen = _script(
        {"action": "train_candidate", "reason": "基线"},
        {"action": "stop_research", "reason": "停"},
    )
    outcome = run_adaptive_loop(
        DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend, scorer=scorer
    )
    assert scored == ["t1"]
    assert calls == []
    assert outcome["trials_used"] == 1
    assert seen[0]["observation"]["duplicate_trials"] == ["t2", "t3", "t4"]
    assert seen[0]["observation"]["validation_metric"] == 0.35
    setting = seen[0]["observation"]["trials"][0]["setting"]
    assert setting["learning_rate"] == pytest.approx(1e-4)
    tried = seen[0]["observation"]["tried_settings"][0]
    assert tried["learning_rate"] == pytest.approx(1e-4)
    assert tried["weight_decay"] == pytest.approx(1e-4)
    assert tried["validation_candidate_count"] == 1654
    blob = json.dumps(seen, ensure_ascii=False)
    assert "0.8" not in blob
    assert "within_batch" not in blob
    assert "0.4118990126144455" not in blob
    assert (tmp_path / "trials" / "t1" / "metrics.json").read_text(encoding="utf-8") == before
    assert next_trial_index(read_trace(tmp_path)) == 5


def test_scoring_failure_is_recorded_and_planning_continues(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    _finished(tmp_path / "trials" / "t1", ckpt=b"x", metrics={"primary_metric": 0.4, "metric_name": "top1"})

    def scorer(*_args):
        raise RuntimeError("cache_missing")

    backend, seen = _script({"action": "stop_research", "reason": "停"})
    outcome = run_adaptive_loop(
        DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend, scorer=scorer
    )
    assert len(seen) == 1
    assert seen[0]["observation"]["trials"][0]["score_error"] == "cache_missing"
    assert outcome["research"]["action"] == "stop_research"


def test_running_trial_blocks_another_train(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("react_agent.eeg_training.launch._start_one_trial", _fake_trainer(calls))
    _finished(
        tmp_path / "trials" / "t1",
        ckpt=b"done",
        metrics={"primary_metric": 0.04, "metric_name": "fixed_bank_top1", "fixed_bank_top1": 0.04},
    )
    running = tmp_path / "trials" / "t2"
    running.mkdir()
    (running / "trial_config.json").write_text(
        json.dumps({"seed": 0, "learning_rate": 0.0002, "weight_decay": 0.0001, "epochs": 50}),
        encoding="utf-8",
    )
    (running / "status.json").write_text(
        json.dumps({"phase": "training", "epoch": 2, "epochs": 50}),
        encoding="utf-8",
    )
    lock = (running / ".lock").open("a", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    backend, seen = _script(
        {"action": "train_candidate", "reason": "再改学习率", "changes": {"learning_rate": 0.0004}}
    )
    outcome = run_adaptive_loop(DESIGN, tmp_path, _budget(), task_card(DESIGN), tmp_path, backend=backend)
    lock.close()
    assert calls == []
    assert not (tmp_path / "trials" / "t3").exists()
    assert outcome["started"] is False
    assert "已有试验在训练，这一轮不再启动新训练。" in outcome["notes"]
    assert seen[0]["observation"]["tried_settings"][1]["status"] == "running"
    assert seen[0]["observation"]["tried_settings"][1]["learning_rate"] == pytest.approx(0.0002)


def test_interrupted_trial_is_not_a_finished_setting(tmp_path) -> None:
    trial = tmp_path / "trials" / "t2"
    trial.mkdir(parents=True)
    (trial / "status.json").write_text(json.dumps({"phase": "training", "epoch": 9, "epochs": 50}), encoding="utf-8")
    rows = read_trace(tmp_path)
    assert rows[0].status == "interrupted"
    assert next_trial_index(rows) == 3


def test_tensor_bank_matches_the_reference_and_ignores_batch_size() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(0)
    bank = torch.randn(7, 5, generator=generator)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3, 6])
    queries = bank[labels] + 0.8 * torch.randn(10, 5, generator=generator)
    reference = fixed_bank_accuracy(
        [(f"q{i}", queries[i].tolist()) for i in range(10)],
        [(f"img{j}", bank[j].tolist()) for j in range(7)],
        {f"q{i}": {f"img{int(labels[i])}"} for i in range(10)},
    )
    for size in (1, 3, 10):
        tally = FixedBankTally(bank, labels)
        for start in range(0, 10, size):
            tally.add(queries[start : start + size])
        result = tally.result()
        assert result["fixed_bank_top1"] == pytest.approx(reference["fixed_bank_top1"])
        assert result["fixed_bank_top5"] == pytest.approx(reference["fixed_bank_top5"])
        assert result["candidate_count"] == 7.0
