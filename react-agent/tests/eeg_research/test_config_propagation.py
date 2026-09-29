"""CLI/workbench Goal values must reach the training argv. Field presence is not enough."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from react_agent.eeg_research.agentic.execution_protocol import (
    build_execution_protocol,
    command_matches,
    effective_config,
    project_command,
)
from react_agent.eeg_training.protocol import Design


def _images(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "train" / "train.pt").write_text(json.dumps(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]), encoding="utf-8")
    (tmp_path / "test" / "test.pt").write_text(json.dumps(["z"]), encoding="utf-8")


def test_non_default_goal_reaches_the_child_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    _images(tmp_path)
    data_root = tmp_path / "custom_root"
    data_root.mkdir()
    (data_root / "train").mkdir()
    (data_root / "test").mkdir()
    (data_root / "train" / "train.pt").write_text(json.dumps(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]), encoding="utf-8")
    (data_root / "test" / "test.pt").write_text(json.dumps(["z"]), encoding="utf-8")
    design = Design(
        dataset="meg",
        exp_setting="intra-subject",
        subject="sub-03",
        epochs=12,
        seed=9,
        batch_size=32,
        lr=3e-4,
        weight_decay=0.0,
        gpu=(2, 3),
        data_root=str(data_root),
        train_dir=str(data_root / "train"),
        test_dir=str(data_root / "test"),
        training_strategy="pooled_subjects",
        policy="agentic",
    )
    protocol = build_execution_protocol(design, data_root)
    command = project_command(protocol, "full")
    config = effective_config(protocol, "full")
    assert command_matches(command, protocol, "full")
    assert command[command.index("--dataset") + 1] == "meg"
    assert command[command.index("--exp-setting") + 1] == "intra-subject"
    assert command[command.index("--subject") + 1] == "sub-03"
    assert Path(command[command.index("--data-root") + 1]) == data_root
    assert command[command.index("--gpu") + 1] == "2,3"
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--lr") + 1] == "0.0003"
    assert command[command.index("--weight-decay") + 1] == "0"
    assert command[command.index("--negative-policy") + 1] == "data_parallel_local"
    assert config["weight_decay"] == 0.0
    assert config["gpu"] == [2, 3]
    assert config["dataset"] == "meg"
    assert config["subject"] == "sub-03"
    assert protocol["input_geometry"]["c_num"] == 271
    assert protocol["input_geometry"]["timesteps"] == [0, 201]
    recorded = {"command": command, "effective_config": config}
    assert recorded["effective_config"]["lr"] == 3e-4
    assert recorded["command"][recorded["command"].index("--weight-decay") + 1] == "0"
