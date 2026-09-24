"""Protocol choices, split isolation, and the refusal to invent a score."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from react_agent.eeg_research.adapters.ubp_retrieval import UbpRetrievalAdapter, child_env
from react_agent.eeg_research.schemas import ExperimentSpec
from react_agent.eeg_training.launch import _budget, launch_design, parse_design, trial_steps
from react_agent.eeg_training.protocol import (
    FULL_EEG_CHANNELS,
    Design,
    SplitError,
    holdout_image_ids,
    list_subjects,
    parse_gpu_list,
    parse_nvidia_smi,
    parse_stop,
    parse_subject_selection,
    split_plan,
    subjects_for,
    train_command,
    validate_design,
)
from react_agent.eeg_training.train_entry import (
    append_history,
    begin_history,
    limit_visible_gpus,
    read_train_status,
    score_held_out,
    write_status,
)
from react_agent.fmri import workbench


def test_four_designs_keep_held_out_test_out_of_fit(tmp_path: Path) -> None:
    designs = [
        Design("eeg", "intra-subject", "sub-01"),
        Design("eeg", "inter-subject", "sub-10"),
        Design("meg", "intra-subject", "sub-04"),
        Design("meg", "inter-subject", "sub-02"),
    ]
    assert len(subjects_for("eeg")) == 10
    assert len(subjects_for("meg")) == 4
    assert len(FULL_EEG_CHANNELS) == 63
    assert len(set(FULL_EEG_CHANNELS)) == 63
    for design in designs:
        plan = split_plan(tmp_path, design)
        assert set(plan.forbidden_files).isdisjoint(plan.train_files)
        assert set(plan.forbidden_files).isdisjoint(plan.val_files)
        assert plan.forbidden_files[0].name == "test.pt"
        if design.exp_setting == "intra-subject":
            assert design.subject in plan.forbidden_files[0].parts
            assert plan.val_mode == "train_holdout"
        else:
            assert design.subject in plan.train_files[0].parts
            assert all(design.subject not in path.parts for path in plan.forbidden_files)
            assert plan.val_mode == "other_subjects_test"
    with pytest.raises(SplitError, match="bad_subject"):
        validate_design(Design("meg", "intra-subject", "sub-01,,sub-02"))


def test_intra_holdout_stays_disjoint_from_test() -> None:
    train = [f"img-{index}" for index in range(20)]
    test = ["test-img"]
    kept, validation = holdout_image_ids(train, test, 3)
    assert set(kept).isdisjoint(validation)
    assert set(validation).isdisjoint(test)
    assert set(kept) | set(validation) == set(train)
    assert holdout_image_ids(train, test, 3)[0] == kept
    with pytest.raises(SplitError, match="train_test_overlap"):
        holdout_image_ids(["shared"], ["shared"], 0)


def test_missing_entry_does_not_write_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EEG_TRAIN_PYTHON", raising=False)
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    design = Design("eeg", "intra-subject", "sub-01")
    out = tmp_path / "camp"
    payload = launch_design(design, out, "run", tmp_path / "empty")
    assert payload["started"] is False
    assert payload["blockers"]
    assert payload["test_result"] is None
    assert payload["api_usd"] is None
    assert not list(out.rglob("metrics.json"))
    dry = launch_design(design, out, "dry_run", tmp_path / "empty")
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    cost = json.loads((out / "cost.json").read_text(encoding="utf-8"))
    assert dry["campaign_written"] is True
    assert comparison["test_result"] is None
    assert cost["api_usd"] is None
    assert not list(out.rglob("metrics.json"))


def test_workbench_form_dry_run_stays_in_research_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workbench, "RESEARCH", tmp_path)
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    monkeypatch.delenv("EEG_TRAIN_PYTHON", raising=False)
    code, payload = workbench.submit_retrieval(
        {
            "dataset": "meg",
            "exp_setting": "inter-subject",
            "subject": "sub-01",
            "epochs": "1",
            "seed": "0",
            "action": "dry_run",
        }
    )
    assert code == 200
    assert payload["started"] is False
    assert payload["dataset"] == "meg"
    assert payload["exp_setting"] == "inter-subject"
    static = Path(__file__).resolve().parents[2] / "src" / "react_agent" / "fmri" / "workbench_static"
    page = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    assert "检索实验" in page
    assert "被试内" in script
    assert "data-dataset" in script
    assert "训练集目录" in script
    assert "batch size" in script
    assert 'type="checkbox"' in script
    assert "不会占用全部卡" in script
    assert "发现数据与设备」后选择" in script
    assert "全部" in script
    assert "subject-checks" in script
    assert "data-subject" not in script
    assert "GPU 秒数上限" in script
    assert "28800" in script
    assert "progress-track" in script
    assert "检查数据" in script
    assert ">输出<" in script or "输出" in script
    assert not list(tmp_path.rglob("metrics.json"))


def test_blank_dirs_stay_automatic_and_custom_dirs_are_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    automatic = split_plan(tmp_path, Design("eeg", "intra-subject", "sub-01"))
    assert automatic.val_mode == "train_holdout"
    assert automatic.forbidden_files[0].name == "test.pt"
    assert set(automatic.forbidden_files).isdisjoint(automatic.train_files)
    custom = Design(
        "eeg",
        "intra-subject",
        "sub-01",
        train_dir=str(tmp_path / "train_set"),
        test_dir=str(tmp_path / "test_set"),
        batch_size=32,
        lr=0.002,
        gpu=(0, 2),
    )
    plan = split_plan(tmp_path, custom)
    assert plan.val_mode == "custom_train_holdout"
    assert plan.train_files == (tmp_path / "train_set" / "train.pt",)
    assert plan.forbidden_files == (tmp_path / "test_set" / "test.pt",)
    assert set(plan.forbidden_files).isdisjoint(plan.train_files)
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    out = tmp_path / "out"
    payload = launch_design(custom, out, "run", tmp_path)
    assert payload["started"] is False
    assert any("未找到" in item for item in payload["blockers"])
    assert not list(out.rglob("metrics.json"))
    command = train_command(custom, out, tmp_path)
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--lr") + 1] == "0.002"
    assert command[command.index("--gpu") + 1] == "0,2"
    assert "--train-dir" in command
    rows = parse_nvidia_smi("0, NVIDIA A100, 40000\n1, NVIDIA A100, 80\nbad line\n")
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[1]["memory_free_mb"] == 80
    assert parse_nvidia_smi("not a gpu table") == []
    with pytest.raises(SplitError, match="一起填写"):
        validate_design(Design("eeg", "intra-subject", "sub-01", train_dir=str(tmp_path)))


def test_gpu_selection_is_a_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    empty = Design("eeg", "intra-subject", "sub-01")
    command = train_command(empty, tmp_path / "out", tmp_path)
    assert "--gpu" not in command
    listed = Design("eeg", "intra-subject", "sub-01", gpu=(0, 2))
    command = train_command(listed, tmp_path / "out", tmp_path)
    assert command[command.index("--gpu") + 1] == "0,2"
    assert parse_gpu_list("") == ()
    assert parse_gpu_list("0, 2") == (0, 2)
    assert parse_design({"gpu": ""}).gpu == ()
    assert parse_design({"gpu": "0,2"}).gpu == (0, 2)
    with pytest.raises(SplitError, match="bad_gpu"):
        parse_gpu_list("0,,2")
    with pytest.raises(SplitError, match="bad_gpu"):
        parse_gpu_list("-1")
    with pytest.raises(SplitError, match="bad_gpu"):
        validate_design(Design("eeg", "intra-subject", "sub-01", gpu=(-1,)))


def _subject_file(root: Path, name: str) -> None:
    folder = root / "things-eeg" / "Preprocessed_data_250Hz_whiten" / name
    folder.mkdir(parents=True)
    (folder / "train.pt").write_bytes(b"")


def test_subjects_come_from_the_scan_and_pool_into_one_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert list_subjects(tmp_path, "eeg") == []
    _subject_file(tmp_path, "sub-07")
    _subject_file(tmp_path, "sub-01")
    (tmp_path / "things-eeg" / "Preprocessed_data_250Hz_whiten" / "notes").mkdir()
    assert list_subjects(tmp_path, "eeg") == ["sub-01", "sub-07"]
    code, payload = workbench.submit_retrieval(
        {"action": "discover", "dataset": "eeg", "data_root": str(tmp_path), "subject": ""}
    )
    assert code == 200
    assert payload["started"] is False
    assert payload["discovered"] is True
    assert payload["subjects"] == ["sub-01", "sub-07"]
    assert "subjects" not in payload["options"]["datasets"][0]
    intra = split_plan(tmp_path, Design("eeg", "intra-subject", "sub-01,sub-07"))
    assert intra.train_subjects == ("sub-01", "sub-07")
    assert intra.val_files == intra.train_files
    assert {path.parent.name for path in intra.forbidden_files} == {"sub-01", "sub-07"}
    assert set(intra.forbidden_files).isdisjoint(intra.train_files)
    inter = split_plan(tmp_path, Design("eeg", "inter-subject", "sub-01"))
    assert inter.val_mode == "other_subjects_test"
    assert inter.forbidden_files[0].parent.name == "sub-07"
    assert set(inter.forbidden_files).isdisjoint(inter.train_files)
    assert set(inter.forbidden_files).isdisjoint(inter.val_files)
    pooled = split_plan(tmp_path, Design("eeg", "inter-subject", "all"))
    assert pooled.val_mode == "all_subjects_holdout"
    assert pooled.val_files == pooled.train_files
    assert {path.parent.name for path in pooled.forbidden_files} == {"sub-01", "sub-07"}
    assert set(pooled.forbidden_files).isdisjoint(pooled.train_files)
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    command = train_command(Design("eeg", "intra-subject", "sub-01,sub-07"), tmp_path / "out", tmp_path)
    assert command[command.index("--subject") + 1] == "sub-01,sub-07"
    assert Design("eeg", "intra-subject", "all").campaign_id().endswith("all_s0")
    assert "sub-01_sub-07" in Design("eeg", "intra-subject", "sub-01,sub-07").campaign_id()
    out = tmp_path / "camp"
    refused = launch_design(Design("eeg", "intra-subject", "sub-01,sub-07"), out, "run", tmp_path)
    assert refused["started"] is False
    assert not list(out.rglob("metrics.json"))
    assert parse_subject_selection("all") == "all"
    assert parse_subject_selection("") == ""
    with pytest.raises(SplitError, match="bad_subject"):
        parse_subject_selection("sub-01,,sub-07")
    with pytest.raises(SplitError, match="bad_subject"):
        parse_subject_selection("../sub-01")
    with pytest.raises(SplitError, match="bad_subject"):
        parse_subject_selection("all,sub-01")
    with pytest.raises(SplitError, match="no_subject"):
        validate_design(Design("eeg", "intra-subject", ""))
    validate_design(Design("eeg", "intra-subject", "", train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "test")))


def test_console_log_reports_the_command_and_refuses_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    design = Design("eeg", "intra-subject", "sub-01")
    out = tmp_path / "out"
    dry = launch_design(design, out, "dry_run", tmp_path)
    assert dry["started"] is False
    assert any(line.startswith("命令：") for line in dry["log"])
    assert dry["progress"]["found"] <= dry["progress"]["total"]
    assert dry["progress"]["total"] > 0
    run = launch_design(design, out / "run", "run", tmp_path)
    assert run["started"] is False
    assert any(line == "GPU 秒数上限：28800" for line in run["log"])
    assert not any("没有 GPU 秒数上限" in line for line in run["log"])
    assert not list(out.rglob("metrics.json"))
    empty = launch_design(Design("eeg", "intra-subject", ""), tmp_path / "bad", "run", tmp_path)
    assert any("还没有选择被试" in line for line in empty["log"])
    assert empty["progress"] is None


def test_gpu_seconds_field_overrides_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EEG_GPU_SECONDS", raising=False)
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    blank = parse_design({"subject": "sub-01", "gpu_seconds": ""})
    assert blank.gpu_seconds is None
    assert _budget(blank).gpu_seconds == 28800
    assert _budget(blank).per_trial_timeout_s == 28800
    missing = launch_design(blank, tmp_path / "missing", "run", tmp_path)
    assert missing["started"] is False
    assert missing["gpu_seconds"] == 28800
    assert not list((tmp_path / "missing").rglob("metrics.json"))
    monkeypatch.setenv("EEG_GPU_SECONDS", "5")
    chosen = parse_design({"subject": "sub-01", "gpu_seconds": "120"})
    assert chosen.gpu_seconds == 120
    assert _budget(chosen).gpu_seconds == 120
    logged = launch_design(chosen, tmp_path / "chosen", "dry_run", tmp_path)
    assert any(line == "GPU 秒数上限：120" for line in logged["log"])
    assert not any("没有 GPU 秒数上限" in line for line in logged["log"])
    with pytest.raises(SplitError, match="bad_budget"):
        parse_design({"subject": "sub-01", "gpu_seconds": "-1"})
    with pytest.raises(SplitError, match="bad_budget"):
        parse_design({"subject": "sub-01", "gpu_seconds": "abc"})


def test_child_pythonpath_includes_src_and_log_line_is_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = str(Path(child_env.__code__.co_filename).resolve().parents[3])
    assert src in child_env()["PYTHONPATH"].split(os.pathsep)
    captured: dict[str, object] = {}

    class _Proc:
        returncode = 1

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            return b"header\nModuleNotFoundError: No module named 'react_agent'\n", b""

    def _popen(command: list[str], stdout: object = None, stderr: object = None, env: dict[str, str] | None = None) -> _Proc:
        captured["command"] = command
        captured["env"] = env
        return _Proc()

    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)
    monkeypatch.setattr("react_agent.eeg_research.adapters.ubp_retrieval.subprocess.Popen", _popen)
    design = Design("eeg", "intra-subject", "sub-01")
    adapter = UbpRetrievalAdapter(design, tmp_path, 10.0)
    monkeypatch.setattr(adapter, "probe", lambda: {"ok": True})
    spec = ExperimentSpec(
        id="t",
        campaign_id="c",
        hypothesis_id="h",
        model_id="ubp_eeg_project",
        profile_id="profile_baseline",
        frozen_task_hash="hash",
        seed=0,
        resolved_config_hash="x",
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(RuntimeError, match="No module named 'react_agent'"):
        adapter.run(spec, str(tmp_path / "out"))
    env = captured["env"]
    assert isinstance(env, dict)
    assert src in str(env["PYTHONPATH"]).split(os.pathsep)
    assert not (tmp_path / "out" / "metrics.json").is_file()


def test_train_entry_import_does_not_load_graph() -> None:
    sys.modules.pop("react_agent.graph", None)
    importlib.import_module("react_agent.eeg_training.train_entry")
    assert "react_agent.graph" not in sys.modules


def test_checked_gpu_indexes_follow_nvidia_smi_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    limit_visible_gpus(Design("eeg", "intra-subject", "sub-01", gpu=(0, 5, 8)))
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,5,8"
    assert child_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    limit_visible_gpus(Design("eeg", "intra-subject", "sub-01"))
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_history_keeps_only_recorded_epochs(tmp_path: Path) -> None:
    out = tmp_path / "camp"
    out.mkdir()
    begin_history(out, 50)
    assert read_train_status(tmp_path, "camp")["history"] == []
    assert read_train_status(tmp_path, "camp")["phase"] == "loading"
    append_history(out, 1, 1.5, 0.2, 0.4)
    write_status(out, "training", 1, 50)
    payload = read_train_status(tmp_path, "camp")
    assert payload["epoch"] == 1
    assert payload["history"] == [{"epoch": 1, "train_loss": 1.5, "val_top1": 0.2, "val_top5": 0.4}]
    assert not (out / "metrics.json").is_file()
    begin_history(out, 50)
    assert read_train_status(tmp_path, "camp")["history"] == []
    assert read_train_status(tmp_path, "../camp") is None
    assert read_train_status(tmp_path, "missing") is None


def test_stop_condition_chooses_the_chain_without_starting(tmp_path: Path) -> None:
    assert parse_stop("") == "chain_early"
    assert parse_design({"subject": "sub-01", "stop": "single_full"}).stop == "single_full"
    with pytest.raises(SplitError, match="bad_stop"):
        parse_design({"subject": "sub-01", "stop": "forever"})
    out = tmp_path / "camp"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"primary_metric": 0.36, "test_result": None}), encoding="utf-8")
    chained = trial_steps(Design("eeg", "inter-subject", "all", stop="chain_early"), out)
    assert [step["action"] for step in chained] == ["skip", "train"]
    assert chained[1]["weight_decay"] == 0.01
    single = trial_steps(Design("eeg", "inter-subject", "all", stop="single_full"), out)
    assert len(single) == 1
    assert single[0]["action"] == "skip"
    command = train_command(
        Design("eeg", "intra-subject", "sub-01", stop="single_early", weight_decay=1e-4),
        out,
        tmp_path,
    )
    assert command[command.index("--stop") + 1] == "single_early"
    assert "--test-only" not in command
    assert score_held_out(Design("eeg", "intra-subject", "sub-01"), tmp_path, out) is None
    stored = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert stored["test_result"] is None
    assert stored["primary_metric"] == 0.36
