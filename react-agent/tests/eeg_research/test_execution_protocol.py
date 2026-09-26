"""Execution protocol checks. They do not call DeepSeek or start GPU training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.execution_protocol import (
    ProtocolError,
    apply_final_test_policy,
    build_execution_protocol,
    command_matches,
    design_from_protocol,
    identity_digest,
    project_command,
    resolve_worker_design,
    unsupported_agentic_reason,
)
from react_agent.eeg_research.agentic.loop import create_campaign
from react_agent.eeg_research.agentic.runner import accept_job, comparable
from react_agent.eeg_training.protocol import Design


def _images(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir(exist_ok=True)
    (tmp_path / "test").mkdir(exist_ok=True)
    (tmp_path / "train" / "train.pt").write_text(json.dumps(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]), encoding="utf-8")
    (tmp_path / "test" / "test.pt").write_text(json.dumps(["z"]), encoding="utf-8")


def _design(tmp_path: Path, **kwargs) -> Design:
    _images(tmp_path)
    fields = dict(
        dataset="eeg",
        exp_setting="inter-subject",
        subject="all",
        epochs=40,
        seed=3,
        train_dir=str(tmp_path / "train"),
        test_dir=str(tmp_path / "test"),
        data_root=str(tmp_path),
        training_strategy="pooled_subjects",
        policy="agentic",
    )
    fields.update(kwargs)
    return Design(**fields)


def _python(monkeypatch) -> None:
    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)


def test_selection_freezes_meg_and_intra(tmp_path: Path) -> None:
    design = _design(tmp_path, dataset="meg", exp_setting="intra-subject", subject="sub-02", seed=4)
    protocol = build_execution_protocol(design, tmp_path)
    assert protocol["dataset"] == "meg"
    assert protocol["exp_setting"] == "intra-subject"
    assert protocol["subject"] == "sub-02"
    assert protocol["seed"] == 4
    assert protocol["data_root"] == str(tmp_path)
    assert protocol["full_epochs"] == 40
    assert protocol["final_test_enabled"] is False
    assert protocol["dataset"] != "eeg" or protocol["subject"] != "all"


def test_missing_samples_do_not_write_an_empty_protocol(tmp_path: Path) -> None:
    design = Design("eeg", "intra-subject", "sub-01", data_root=str(tmp_path), policy="agentic")
    with pytest.raises(ProtocolError):
        build_execution_protocol(design, tmp_path)
    assert not (tmp_path / "execution_protocol.json").exists()


def test_reject_per_subject_and_hidden_override(monkeypatch, tmp_path: Path) -> None:
    from react_agent.fmri import workbench

    per = _design(tmp_path, training_strategy="per_subject", subject="sub-01")
    reason = unsupported_agentic_reason(per)
    assert reason and "每个被试" in reason
    hidden = _design(tmp_path, generalization_target="held_out_subject", subject="all")
    hidden_reason = unsupported_agentic_reason(hidden)
    assert hidden_reason and "不会进入训练命令" in hidden_reason

    monkeypatch.setattr(
        "react_agent.eeg_training.launch.launch_design",
        lambda *args, **kwargs: {"ok": True, "started": False, "blockers": [], "log": []},
    )
    opened: list[str] = []
    monkeypatch.setattr(
        "react_agent.eeg_research.agentic.cli.open_agentic_run",
        lambda *args, **kwargs: opened.append("opened") or {"started": True},
    )
    _status, payload = workbench.submit_retrieval(
        {
            "action": "run",
            "policy": "agentic",
            "training_strategy": "per_subject",
            "subject": "sub-01",
            "dataset": "eeg",
            "exp_setting": "intra-subject",
            "request_id": "nope",
        }
    )
    assert any("每个被试" in item for item in payload["blockers"])
    assert opened == []
    assert "代码级研究已启动" not in " ".join(payload.get("log") or [])


def test_accept_selection_keeps_the_submitted_task(monkeypatch) -> None:
    from react_agent.fmri import workbench

    seen: dict[str, Design] = {}

    def launch(design, *_args, **_kwargs):
        seen["design"] = design
        return {"ok": True, "started": False, "blockers": [], "log": []}

    def opener(design, payload, **_kwargs):
        seen["opened"] = design
        return {**payload, "started": True, "campaign_id": "kept", "agentic_campaign": "kept"}

    monkeypatch.setattr("react_agent.eeg_training.launch.launch_design", launch)
    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.open_agentic_run", opener)
    _status, payload = workbench.submit_retrieval(
        {
            "action": "run",
            "policy": "agentic",
            "training_strategy": "pooled_subjects",
            "dataset": "meg",
            "exp_setting": "intra-subject",
            "subject": "sub-03",
            "seed": "9",
            "epochs": "12",
            "request_id": "yes",
        }
    )
    assert payload["agentic_campaign"] == "kept"
    assert seen["opened"].dataset == "meg"
    assert seen["opened"].exp_setting == "intra-subject"
    assert seen["opened"].subject == "sub-03"
    assert seen["opened"].seed == 9


def test_full_and_pilot_commands_follow_the_protocol(tmp_path: Path, monkeypatch) -> None:
    _python(monkeypatch)
    design = _design(tmp_path, seed=5, epochs=22, subject="sub-01", exp_setting="intra-subject")
    protocol = build_execution_protocol(design, tmp_path)
    full = project_command(protocol, "full")
    pilot = project_command(protocol, "pilot")
    assert command_matches(full, protocol, "full")
    assert command_matches(pilot, protocol, "pilot")
    assert "--epochs" in full and full[full.index("--epochs") + 1] == "22"
    assert pilot[pilot.index("--epochs") + 1] == "3"
    assert pilot[pilot.index("--stop") + 1] == "single_full"
    assert full[full.index("--seed") + 1] == "5"
    assert full[full.index("--subject") + 1] == "sub-01"
    assert Path(full[full.index("--data-root") + 1]) == tmp_path


def test_mismatch_is_not_comparable(tmp_path: Path, monkeypatch) -> None:
    _python(monkeypatch)
    design = _design(tmp_path)
    protocol = build_execution_protocol(design, tmp_path)
    command = project_command(protocol, "full")
    job = tmp_path / "camp" / "jobs" / "j1"
    job.mkdir(parents=True)
    (tmp_path / "camp" / "execution_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    binding = {"module": "eeg_candidate", "entry_sha256": "abc", "file_sha256": "abc"}
    (job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate", "file_sha256": "abc"}), encoding="utf-8")
    (job / "metrics.json").write_text(
        json.dumps({"fixed_bank_top1": 0.2, "validation_image_count": 1, "validation_identity": protocol["validation_identity"]}),
        encoding="utf-8",
    )
    (job / "job.json").write_text(
        json.dumps({"command": command, "execution_fingerprint": protocol["fingerprint"], "fidelity": "full"}),
        encoding="utf-8",
    )
    good = accept_job(job, binding, "full", protocol=protocol)
    assert good["evaluation_valid"] is True
    bad_job = tmp_path / "camp" / "jobs" / "j2"
    bad_job.mkdir()
    (bad_job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate", "file_sha256": "abc"}), encoding="utf-8")
    (bad_job / "metrics.json").write_text(
        json.dumps({"fixed_bank_top1": 0.2, "validation_identity": "nope"}),
        encoding="utf-8",
    )
    (bad_job / "job.json").write_text(
        json.dumps({"command": ["python", "--dataset", "eeg"], "execution_fingerprint": protocol["fingerprint"]}),
        encoding="utf-8",
    )
    bad = accept_job(bad_job, binding, "full", protocol=protocol)
    assert bad["evaluation_valid"] is False
    assert bad["reason"] == "protocol_mismatch"
    assert "contract_fingerprint" not in bad
    assert comparable(good, bad) is False
    peer = dict(good)
    assert comparable(good, peer) is True


def test_recovery_reads_protocol_and_blocks_when_missing(tmp_path: Path) -> None:
    design = _design(tmp_path, dataset="meg", subject="sub-04", exp_setting="intra-subject", seed=8)
    protocol = build_execution_protocol(design, tmp_path)
    camp = tmp_path / "camp"
    camp.mkdir()
    (camp / "execution_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    state: dict = {"status": "created"}
    restored = resolve_worker_design(camp, state)
    assert restored is not None
    assert restored.dataset == "meg"
    assert restored.subject == "sub-04"
    assert restored.seed == 8
    assert restored.exp_setting == "intra-subject"
    (camp / "execution_protocol.json").unlink()
    missing = resolve_worker_design(camp, state)
    assert missing is None
    assert state["status"] == "blocked"
    assert state["detail"] == "execution_protocol_missing"
    assert design_from_protocol(protocol).dataset != "eeg" or design.subject != "all"


def test_identity_changes_with_seed(tmp_path: Path) -> None:
    left = build_execution_protocol(_design(tmp_path, seed=1), tmp_path)
    right = build_execution_protocol(_design(tmp_path, seed=2), tmp_path)
    assert left["validation_image_ids"] != right["validation_image_ids"]
    assert left["fingerprint"] != right["fingerprint"]
    assert left["validation_identity"] != right["validation_identity"]
    same_count = dict(left)
    same_count["validation_identity"] = identity_digest(["other"])
    assert same_count["validation_identity"] != left["validation_identity"]


def test_final_test_stays_empty() -> None:
    payload = apply_final_test_policy({"test_result": 0.9, "fixed_bank_top1": 0.1}, False)
    assert payload["test_result"] is None


def test_goal_scope_follows_design(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.cli import goal
    from react_agent.eeg_research.agentic.contract import freeze_contract, research_scope

    design = _design(tmp_path, exp_setting="inter-subject", subject="sub-01")
    protocol = build_execution_protocol(design, tmp_path)
    contract = freeze_contract(design, tmp_path)
    contract["execution_fingerprint"] = protocol["fingerprint"]
    written = goal("goal", design)
    assert written["research_scope"] == research_scope(design)
    assert written["final_test_enabled"] is False
    create_campaign(tmp_path / "root", goal=written, contract=contract, request_id="g", protocol=protocol)
    camp = tmp_path / "root" / "goal"
    assert json.loads((camp / "goal.json").read_text(encoding="utf-8"))["research_scope"] == contract["research_scope"]
    assert json.loads((camp / "resolved_goal.json").read_text(encoding="utf-8"))["research_scope"] == contract["research_scope"]
    saved = json.loads((camp / "execution_protocol.json").read_text(encoding="utf-8"))
    assert saved["fingerprint"] == contract["execution_fingerprint"]
