"""Worker failure recovery with a fake LLM. No API, GPU, or training process."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
from react_agent.eeg_research.agentic.llm import LlmUnavailable, _ledger
from react_agent.eeg_research.agentic.loop import align_interrupt, create_campaign, load_state, save_state
from react_agent.eeg_training.protocol import Design

APP_JS = Path(__file__).resolve().parents[2] / "src" / "react_agent" / "fmri" / "workbench_static" / "app.js"


def _campaign(tmp_path: Path) -> Path:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "train" / "train.pt").write_text(json.dumps(["a", "b", "c", "d"]), encoding="utf-8")
    (tmp_path / "test" / "test.pt").write_text(json.dumps(["z"]), encoding="utf-8")
    design = Design(
        "eeg",
        "inter-subject",
        "all",
        train_dir=str(tmp_path / "train"),
        test_dir=str(tmp_path / "test"),
        data_root=str(tmp_path),
        policy="agentic",
    )
    protocol = build_execution_protocol(design, tmp_path)
    contract = {
        "fingerprint": "contract-fp",
        "execution_fingerprint": protocol["fingerprint"],
        "final_test_enabled": False,
        "research_scope": "pooled_subject_retrieval",
        "files": [],
    }
    goal = {
        "goal_id": "recovery",
        "research_scope": "pooled_subject_retrieval",
        "final_test_enabled": False,
        "max_training_jobs": 10,
        "max_llm_calls": 100,
        "max_gpu_seconds": 100,
        "max_candidates": 4,
    }
    root = tmp_path / "root"
    create_campaign(root, goal=goal, contract=contract, request_id="req", protocol=protocol)
    return root / "recovery"


def _install_backend(monkeypatch, *, fail: BaseException) -> dict[str, int]:
    ticks = {"planner": 0, "coder": 0}

    def factory(_camp: Path, role: str):
        def call(_payload: dict) -> dict:
            if role == "candidate_coder":
                ticks["coder"] += 1
                if isinstance(fail, LlmUnavailable):
                    _ledger(_camp, {"success": False, "error": str(fail), "role": role})
                raise fail
            if role == "research_planner":
                ticks["planner"] += 1
                if ticks["planner"] == 1:
                    return {
                        "action": "propose_experiment",
                        "reason_zh": "先提出实验",
                        "evidence_ids": [],
                        "hypothesis_draft": {"mechanism": "时序池化"},
                        "experiment_draft": {"initial_fidelity": "pilot"},
                    }
                return {"action": "implement_candidate", "reason_zh": "编写候选", "evidence_ids": []}
            return {}

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)
    return ticks


def _cost(camp: Path) -> dict:
    path = camp / "cost.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _assert_counts(camp: Path, *, status: str, candidates: int, training_jobs: int) -> None:
    from react_agent.eeg_research.agentic.view import campaign_view

    state = load_state(camp)
    view = campaign_view(camp)
    assert state["status"] == status
    assert view["status"] == status
    assert len(state.get("candidates") or []) == candidates
    assert state.get("training_jobs") == training_jobs
    assert not any(row.get("evaluation_valid") for row in state.get("evidence") or [])


def test_invalid_json_during_implement_blocks_without_a_new_candidate(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    ticks = _install_backend(monkeypatch, fail=LlmUnavailable("DeepSeekParseError"))
    from react_agent.eeg_research.agentic.worker import run_worker

    run_worker(camp, poll_seconds=0, max_ticks=3)
    state = load_state(camp)
    _assert_counts(camp, status="blocked", candidates=0, training_jobs=0)
    assert state["status"] != "planning"
    assert "DeepSeekParseError" in str(state.get("detail") or "")
    assert state["failure"]["phase"] == "implement_candidate"
    assert state["failure"]["recoverable"] is True
    assert ticks["coder"] == 3
    assert _cost(camp)["llm_failures"] == 3
    assert _cost(camp)["llm_calls"] == 3
    assert "llm_unavailable" in (camp / "events.jsonl").read_text(encoding="utf-8")
    assert (camp / "candidates" / "c1").is_dir()
    assert not (camp / "candidates" / "c2").exists()
    assert not (camp / "candidates" / "c1" / "coder_log.jsonl").exists()


def test_program_error_is_not_labeled_as_parse_failure(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    ticks = _install_backend(monkeypatch, fail=RuntimeError("disk full"))
    from react_agent.eeg_research.agentic.worker import run_worker

    run_worker(camp, poll_seconds=0, max_ticks=3)
    state = load_state(camp)
    _assert_counts(camp, status="blocked", candidates=0, training_jobs=0)
    assert ticks["coder"] == 1
    assert state["failure"]["recoverable"] is False
    assert state["failure"]["error_type"] == "RuntimeError"
    assert "DeepSeekParseError" not in str(state.get("detail") or "")
    assert "RuntimeError" in str(state.get("detail") or "")
    assert _cost(camp).get("llm_failures", 0) == 0


def test_missing_or_zombie_worker_is_interrupted_and_a_sleeping_one_is_not(tmp_path: Path, monkeypatch) -> None:
    from react_agent.eeg_research.agentic.view import campaign_view

    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    state["detail"] = ""
    save_state(camp, state)
    (camp / "worker.json").write_text(json.dumps({"pid": 1 << 30, "heartbeat": 0}), encoding="utf-8")
    missing = campaign_view(camp)
    assert missing["status"] == "interrupted"
    assert missing["status"] != "planning"
    assert "退出" in str(missing["detail"])
    assert load_state(camp)["status"] == "planning"

    (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": 0}), encoding="utf-8")
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs._process_state", lambda _pid: "Z")
    zombie = campaign_view(camp)
    assert zombie["status"] == "interrupted"
    assert load_state(camp)["status"] == "planning"

    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs._process_state", lambda _pid: "S")
    sleeping = campaign_view(camp)
    assert sleeping["status"] == "planning"
    assert "死亡" not in str(sleeping.get("detail") or "")

    state = load_state(camp)
    state["status"] = "paused"
    save_state(camp, state)
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs._process_state", lambda _pid: "Z")
    paused = campaign_view(camp)
    assert paused["status"] == "paused"
    assert load_state(camp)["status"] == "paused"
    assert 'interrupted: "已中断"' in APP_JS.read_text(encoding="utf-8")


def test_resume_merges_the_uncommitted_decision_and_reuses_the_candidate(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    state["llm_calls"] = 2
    state["training_jobs"] = 0
    state["hypothesis"] = {"mechanism": "时序池化"}
    state["experiment"] = {"initial_fidelity": "pilot"}
    state["candidates"] = [
        {"candidate_id": "c1", "status": "review_needs_fix"},
        {"candidate_id": "c2", "status": "review_needs_fix"},
    ]
    state["evidence"] = [
        {"evidence_id": "ev_impl_c1", "kind": "implementation", "candidate_id": "c1"},
        {"evidence_id": "ev_impl_c2", "kind": "implementation", "candidate_id": "c2"},
    ]
    state["decisions"] = [
        {"decision_id": f"d{index}", "action": "propose_experiment", "ok": True} for index in range(1, 7)
    ]
    save_state(camp, state)
    (camp / "cost.json").write_text(
        json.dumps({"llm_calls": 2, "llm_failures": 0, "gpu_seconds_used": 0.0}),
        encoding="utf-8",
    )
    for name in ("c1", "c2", "c3"):
        folder = camp / "candidates" / name
        folder.mkdir(parents=True)
        (folder / "spec.json").write_text("{}", encoding="utf-8")
    record = {"decision_id": "d7", "action": "implement_candidate", "ok": True, "reason_zh": "编写候选"}
    (camp / "decisions").mkdir()
    (camp / "decisions" / "d7.json").write_text(json.dumps({"decision": record}), encoding="utf-8")
    ticks = {"planner": 0, "coder": 0}

    def factory(_camp: Path, role: str):
        def call(_payload: dict) -> dict:
            if role == "candidate_coder":
                ticks["coder"] += 1
                _ledger(_camp, {"success": False, "error": "DeepSeekParseError", "role": role})
                raise LlmUnavailable("DeepSeekParseError")
            if role == "research_planner":
                ticks["planner"] += 1
                return {"action": "stop", "reason_zh": "不应再要决策", "evidence_ids": []}
            return {}

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)

    def spawn(root: Path, campaign: str, _poll: float) -> int:
        from react_agent.eeg_research.agentic.worker import run_worker

        run_worker(root / campaign, poll_seconds=0, max_ticks=3)
        return os.getpid()

    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.spawn_worker", spawn)
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp.parent), "--poll-seconds", "0"]) == 0
    saved = load_state(camp)
    ids = [row["decision_id"] for row in saved["decisions"]]
    assert ids == [f"d{index}" for index in range(1, 8)]
    assert len(ids) == len(set(ids))
    assert ticks["planner"] == 0
    assert ticks["coder"] == 3
    _assert_counts(camp, status="blocked", candidates=2, training_jobs=0)
    assert len(saved["evidence"]) == 2
    assert (camp / "candidates" / "c3").is_dir()
    assert not (camp / "candidates" / "c4").exists()
    assert _cost(camp)["llm_calls"] == 5
    assert _cost(camp)["llm_failures"] == 3


def test_second_resume_while_locked_does_not_start_another_worker(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    state["training_jobs"] = 0
    state["candidates"] = [{"candidate_id": "c1", "status": "review_needs_fix"}]
    state["decisions"] = [{"decision_id": "d1", "action": "propose_experiment", "ok": True}]
    save_state(camp, state)
    handle = (camp / "worker.lock").open("a", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spawned = {"n": 0}

    def spawn(*_args, **_kwargs):
        spawned["n"] += 1
        raise AssertionError("second worker")

    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.spawn_worker", spawn)
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp.parent)]) == 0
    assert spawned["n"] == 0
    saved = load_state(camp)
    assert saved["status"] == "planning"
    assert len(saved["decisions"]) == 1
    assert len(saved["candidates"]) == 1
    assert saved["training_jobs"] == 0
    handle.close()


def test_resume_with_a_live_job_only_reconciles(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "training"
    state["live_job"] = "j1_c1_pilot"
    state["training_jobs"] = 1
    state["candidates"] = [{"candidate_id": "c1", "status": "ready"}]
    save_state(camp, state)
    job = camp / "jobs" / "j1_c1_pilot"
    job.mkdir(parents=True)
    command = ["python", "-m", "react_agent.eeg_training.train_entry"]
    (job / "job.json").write_text(json.dumps({"command": command, "status": "running"}), encoding="utf-8")
    before = (job / "job.json").read_text(encoding="utf-8")

    def reconcile(job_dir: Path) -> dict:
        return {"status": "running", "job_id": job_dir.name}

    def start_job(*_args, **_kwargs):
        raise AssertionError("training command started")

    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.reconcile", reconcile)
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.start_job", start_job)
    monkeypatch.setattr(
        "react_agent.eeg_research.agentic.worker.role_backend",
        lambda *_args, **_kwargs: (lambda _payload: {}),
    )

    def spawn(root: Path, campaign: str, poll: float) -> int:
        from react_agent.eeg_research.agentic.worker import run_worker

        run_worker(root / campaign, poll_seconds=poll, max_ticks=2)
        return os.getpid()

    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.spawn_worker", spawn)
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp.parent), "--poll-seconds", "0"]) == 0
    saved = load_state(camp)
    assert saved["training_jobs"] == 1
    assert saved["live_job"] == "j1_c1_pilot"
    assert saved["status"] == "training"
    assert (job / "job.json").read_text(encoding="utf-8") == before
    assert len(list((camp / "jobs").iterdir())) == 1


def test_pause_and_stop_survive_align_and_the_worker(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.worker import run_worker

    camp = _campaign(tmp_path)
    for status in ("paused", "cancelled"):
        state = load_state(camp)
        state["status"] = status
        state["training_jobs"] = 0
        save_state(camp, state)
        align_interrupt(camp)
        run_worker(camp, poll_seconds=0, max_ticks=2)
        assert load_state(camp)["status"] == status
        assert load_state(camp)["training_jobs"] == 0
