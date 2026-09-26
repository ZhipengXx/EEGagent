"""Crash boundaries after the first worker-failure recovery. Fake LLM only."""

from __future__ import annotations

import json
import os
from pathlib import Path

from react_agent.eeg_research.agentic.binding import file_sha256
from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
from react_agent.eeg_research.agentic.llm import LlmUnavailable, _ledger
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state
from react_agent.eeg_training.protocol import Design


def _campaign(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    camp = root / "recovery"
    (camp / "cost.json").write_text(json.dumps({"llm_calls": 4, "llm_failures": 0}), encoding="utf-8")
    return camp


def _install(monkeypatch, calls):
    def factory(_camp: Path, role: str):
        def call(payload: dict) -> dict:
            calls[role] = calls.get(role, 0) + 1
            calls["last"] = (role, payload)
            behaviour = calls["behaviour"]
            if callable(behaviour):
                return behaviour(role, payload)
            raise AssertionError(role)

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)


def _ready_candidate(camp: Path, *, review: dict | None = None, implementation: bool = True) -> None:
    workspace = camp / "candidates" / "c1"
    (workspace / "extension").mkdir(parents=True)
    (workspace / "spec.json").write_text("{}", encoding="utf-8")
    entry = workspace / "extension" / "eeg_candidate.py"
    entry.write_text("x = 1\n", encoding="utf-8")
    (workspace / "checks.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    if implementation:
        (workspace / "implementation.json").write_text(
            json.dumps({"candidate_id": "c1", "status": "ready_for_review", "steps": 2}),
            encoding="utf-8",
        )
    if review is not None:
        (workspace / "review.json").write_text(json.dumps(review), encoding="utf-8")
    state = load_state(camp)
    state["status"] = "planning"
    state["hypothesis"] = {"mechanism": "池化"}
    state["experiment"] = {"initial_fidelity": "pilot"}
    state["decisions"] = [
        {"decision_id": "d1", "action": "implement_candidate", "ok": True, "executed": False, "reason_zh": "编写"}
    ]
    save_state(camp, state)


def _run(camp: Path) -> dict:
    from react_agent.eeg_research.agentic.worker import run_worker

    return run_worker(camp, poll_seconds=0, max_ticks=1)


def test_review_llm_failure_keeps_the_candidate_and_skips_the_coder(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _ready_candidate(camp)
    def behaviour(role, _payload):
        if role == "candidate_reviewer":
            raise LlmUnavailable("DeepSeekParseError")
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    state = _run(camp)
    assert state["status"] == "blocked"
    assert state["failure"]["phase"] == "review_candidate"
    assert state["failure"]["recoverable"] is True
    assert "DeepSeekParseError" in str(state["detail"])
    assert state.get("candidates") == []
    assert state.get("evidence") == []
    assert calls.get("candidate_coder", 0) == 0
    assert calls.get("research_planner", 0) == 0
    assert (camp / "candidates" / "c1" / "extension" / "eeg_candidate.py").is_file()
    assert not (camp / "candidates" / "c2").exists()
    assert json.loads((camp / "cost.json").read_text(encoding="utf-8"))["llm_calls"] == 4


def test_review_program_error_is_not_a_parse_failure_and_resume_does_not_retry(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _ready_candidate(camp)

    def behaviour(role, _payload):
        if role == "candidate_reviewer":
            raise RuntimeError("disk full")
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    state = _run(camp)
    assert state["failure"]["phase"] == "review_candidate"
    assert state["failure"]["recoverable"] is False
    assert state["failure"]["error_type"] == "RuntimeError"
    assert "DeepSeekParseError" not in str(state["detail"])
    assert "RuntimeError" in str(state["detail"])
    spawned = {"n": 0}

    def spawn(*_args, **_kwargs):
        spawned["n"] += 1
        raise AssertionError("retried")

    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.spawn_worker", spawn)
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp.parent)]) == 0
    assert spawned["n"] == 0
    assert load_state(camp)["status"] == "blocked"
    assert calls.get("candidate_coder", 0) == 0


def test_review_file_is_reused_and_a_real_rejection_is_recorded_once(tmp_path: Path, monkeypatch) -> None:
    from react_agent.eeg_research.agentic.identity import ensure_attempt, source_hash

    camp = _campaign(tmp_path)
    _ready_candidate(camp)
    workspace = camp / "candidates" / "c1"
    attempt = ensure_attempt(workspace, "c1")
    stored = {
        "status": "needs_fix",
        "summary_zh": "还要改",
        "issues": [{"title": "形状"}],
        "candidate_id": "c1",
        "attempt_id": attempt["attempt_id"],
        "phase": "review_candidate",
        "input_hash": source_hash(workspace),
    }
    (workspace / "review.json").write_text(json.dumps(stored), encoding="utf-8")
    def behaviour(role, _payload):
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    state = _run(camp)
    assert calls.get("candidate_reviewer", 0) == 0
    assert calls.get("candidate_coder", 0) == 0
    assert len(state["candidates"]) == 1
    assert state["candidates"][0]["candidate_id"] == "c1"
    assert state["candidates"][0]["status"] == "review_needs_fix"
    assert state["experiment_failed"] is True
    assert [row["evidence_id"] for row in state["evidence"]] == ["ev_impl_c1"]
    assert not state.get("failure")
    saved = load_state(camp)
    saved["decisions"][-1]["executed"] = False
    saved["status"] = "planning"
    save_state(camp, saved)
    again = _run(camp)
    assert len(again["candidates"]) == 1
    assert len(again["evidence"]) == 1


def test_reviewer_rejection_uses_the_existing_status(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _ready_candidate(camp)

    def behaviour(role, _payload):
        if role == "candidate_reviewer":
            return {"status": "needs_fix", "summary_zh": "否决", "issues": [{"title": "形状不对", "severity": "blocking"}]}
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    state = _run(camp)
    assert calls["candidate_reviewer"] == 1
    assert calls.get("candidate_coder", 0) == 0
    assert state["candidates"][0]["status"] == "review_needs_fix"
    assert state["status"] != "blocked"
    assert len(state["evidence"]) == 1


def test_planner_parse_failure_can_be_asked_again_without_skipping(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    save_state(camp, state)
    seen = {"n": 0}

    def behaviour(role, _payload):
        if role != "research_planner":
            raise AssertionError(role)
        seen["n"] += 1
        if seen["n"] == 1:
            _ledger(camp, {"success": False, "error": "DeepSeekParseError", "role": role})
            raise LlmUnavailable("DeepSeekParseError")
        return {"action": "stop", "reason_zh": "停", "evidence_ids": []}

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    first = _run(camp)
    assert first["status"] == "blocked"
    assert first["failure"]["phase"] == "planner"
    assert first["failure"]["recoverable"] is True
    assert first["failure"]["error_type"] == "DeepSeekParseError"
    assert first["decisions"] == []
    assert not (camp / "decisions").exists()
    assert json.loads((camp / "cost.json").read_text(encoding="utf-8"))["llm_calls"] == 5

    def spawn(root: Path, campaign: str, _poll: float) -> int:
        from react_agent.eeg_research.agentic.worker import run_worker

        run_worker(root / campaign, poll_seconds=0, max_ticks=1)
        return os.getpid()

    monkeypatch.setattr("react_agent.eeg_research.agentic.cli.spawn_worker", spawn)
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp.parent), "--poll-seconds", "0"]) == 0
    saved = load_state(camp)
    assert [row["decision_id"] for row in saved["decisions"]] == ["d1"]
    assert saved["decisions"][0]["action"] == "stop"
    assert saved["llm_calls"] == 5
    assert json.loads((camp / "cost.json").read_text(encoding="utf-8"))["llm_calls"] == 5


def test_planner_schema_failure_is_not_executed_and_runtime_errors_stay_blocked(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    save_state(camp, state)

    def behaviour(role, _payload):
        if role != "research_planner":
            raise AssertionError(role)
        _ledger(camp, {"success": True, "role": role})
        return {"action": "not_an_action"}

    _install(monkeypatch, {"behaviour": behaviour})
    schema = _run(camp)
    assert schema["status"] == "blocked"
    assert schema["failure"]["error_type"] == "unknown_action:not_an_action"
    assert schema["failure"]["error_type"] != "DeepSeekParseError"
    assert len(schema["decisions"]) == 1
    assert schema["decisions"][0]["ok"] is False
    assert schema["hypothesis"] is None
    assert schema["llm_calls"] == json.loads((camp / "cost.json").read_text(encoding="utf-8"))["llm_calls"]

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    camp2 = _campaign(runtime_root)

    def crash(role, _payload):
        if role == "research_planner":
            raise RuntimeError("disk full")
        raise AssertionError(role)

    _install(monkeypatch, {"behaviour": crash})
    crashed = _run(camp2)
    assert crashed["failure"]["recoverable"] is False
    assert crashed["failure"]["error_type"] == "RuntimeError"
    assert "DeepSeekParseError" not in str(crashed["detail"])
    assert crashed["decisions"] == []
    spawned = {"n": 0}
    monkeypatch.setattr(
        "react_agent.eeg_research.agentic.cli.spawn_worker",
        lambda *_args, **_kwargs: spawned.__setitem__("n", spawned["n"] + 1),
    )
    from react_agent.eeg_research.agentic.cli import main

    assert main(["resume", "--campaign", "recovery", "--root", str(camp2.parent)]) == 0
    assert spawned["n"] == 0
    assert load_state(camp2)["status"] == "blocked"


def test_unexecuted_decision_is_replayed_once(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    marker = {"hypothesis": {"mechanism": "已落盘"}, "experiment": {"initial_fidelity": "pilot", "parent_candidate_id": "baseline"}}
    (camp / "hypotheses").mkdir()
    (camp / "hypotheses" / "h1.json").write_text(json.dumps(marker), encoding="utf-8")
    record = {
        "decision_id": "d1",
        "action": "propose_experiment",
        "ok": True,
        "executed": False,
        "reason_zh": "提出",
    }
    (camp / "decisions").mkdir()
    (camp / "decisions" / "d1.json").write_text(
        json.dumps({"decision": record, "raw": {"hypothesis_draft": {"mechanism": "不该重写"}}}),
        encoding="utf-8",
    )
    def behaviour(role, _payload):
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    state = _run(camp)
    assert calls.get("research_planner", 0) == 0
    assert [row["decision_id"] for row in state["decisions"]] == ["d1"]
    assert state["hypothesis"] == {"mechanism": "已落盘"}
    assert json.loads((camp / "hypotheses" / "h1.json").read_text(encoding="utf-8")) == marker
    assert len(list((camp / "hypotheses").glob("h*.json"))) == 1


def test_training_job_json_is_adopted_and_a_bare_directory_blocks(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    state["candidate_ready"] = True
    state["candidate_id"] = "c1"
    state["decisions"] = [{"decision_id": "d1", "action": "run_pilot", "ok": True, "executed": False}]
    save_state(camp, state)
    job = camp / "jobs" / "j1_baseline_pilot"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"status": "running", "command": ["python"]}), encoding="utf-8")

    def start_job(*_args, **_kwargs):
        raise AssertionError("training command started")

    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.start_job", start_job)
    def _refuse(role, _payload):
        raise AssertionError(role)

    _install(monkeypatch, {"behaviour": _refuse})
    adopted = _run(camp)
    assert adopted["live_job"] == "j1_baseline_pilot"
    assert adopted["training_jobs"] == 1
    assert adopted["status"] == "training"
    assert [row["decision_id"] for row in adopted["decisions"]] == ["d1"]

    camp2 = _campaign(tmp_path / "bare")
    state2 = load_state(camp2)
    state2["status"] = "planning"
    state2["candidate_id"] = "c1"
    state2["decisions"] = [{"decision_id": "d1", "action": "run_pilot", "ok": True, "executed": False}]
    save_state(camp2, state2)
    (camp2 / "jobs" / "j1_baseline_pilot").mkdir(parents=True)
    blocked = _run(camp2)
    assert blocked["status"] == "blocked"
    assert blocked["detail"] == "training_start_unconfirmed"
    assert blocked["failure"]["recoverable"] is False
    assert blocked["training_jobs"] == 0


def test_coder_log_resumes_the_same_candidate_from_the_next_step(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _ready_candidate(camp, implementation=False)
    workspace = camp / "candidates" / "c1"
    entry = workspace / "extension" / "eeg_candidate.py"
    digest = file_sha256(entry)
    row = {
        "step": 1,
        "tool": "apply_candidate_patch",
        "result": {"ok": True, "sha256": digest, "auto_check": {"ok": True}},
    }
    (workspace / "coder_log.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    steps: list[int] = []

    def behaviour(role, payload):
        if role == "candidate_coder":
            steps.append(int(payload["step"]))
            return {"tool": "finish_patch", "args": {"summary": "继续"}}
        if role == "candidate_reviewer":
            return {"status": "ready", "summary_zh": "通过", "issues": []}
        raise AssertionError(role)

    calls: dict = {"behaviour": behaviour}
    _install(monkeypatch, calls)
    before = entry.read_text(encoding="utf-8")
    state = _run(camp)
    assert steps == [2]
    assert entry.read_text(encoding="utf-8") == before
    assert state["candidates"][0]["candidate_id"] == "c1"
    assert [row["evidence_id"] for row in state["evidence"]] == ["ev_impl_c1"]
    assert not (camp / "candidates" / "c2").exists()
    assert state["training_jobs"] == 0
    assert json.loads((camp / "cost.json").read_text(encoding="utf-8"))["llm_calls"] == 4
    assert state["llm_calls"] == 4


def test_finished_coder_log_does_not_call_the_coder_again(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _ready_candidate(camp, implementation=False)
    workspace = camp / "candidates" / "c1"
    entry = workspace / "extension" / "eeg_candidate.py"
    digest = file_sha256(entry)
    lines = [
        {"step": 1, "tool": "apply_candidate_patch", "result": {"ok": True, "sha256": digest, "auto_check": {"ok": True}}},
        {"step": 2, "tool": "finish_patch", "result": {"ok": True, "manifest": {"module": "eeg_candidate"}}},
    ]
    (workspace / "coder_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    (workspace / "source_manifest.json").write_text(json.dumps({"module": "eeg_candidate"}), encoding="utf-8")

    def behaviour(role, _payload):
        if role == "candidate_coder":
            raise AssertionError("coder reran")
        if role == "candidate_reviewer":
            return {"status": "ready", "summary_zh": "通过", "issues": []}
        raise AssertionError(role)

    _install(monkeypatch, {"behaviour": behaviour})
    state = _run(camp)
    assert state["candidates"][0]["candidate_id"] == "c1"
    assert len(state["evidence"]) == 1


def test_coder_log_mismatches_block_without_a_tool_rerun(tmp_path: Path, monkeypatch) -> None:
    def prepare(root: Path, name: str) -> Path:
        camp = _campaign(root)
        _ready_candidate(camp, implementation=False)
        return camp

    tail = prepare(tmp_path / "tail", "tail")
    workspace = tail / "candidates" / "c1"
    (workspace / "coder_log.jsonl").write_text('{"step": 1, "tool": "read_code", "result": {"ok": true}', encoding="utf-8")
    def _refuse(role, _payload):
        raise AssertionError(role)

    _install(monkeypatch, {"behaviour": _refuse})
    blocked = _run(tail)
    assert blocked["detail"] == "coder_log_tail_incomplete"
    assert blocked["failure"]["recoverable"] is False
    assert json.loads((tail / "cost.json").read_text(encoding="utf-8"))["llm_calls"] == 4

    mismatch = prepare(tmp_path / "hash", "hash")
    space = mismatch / "candidates" / "c1"
    entry = space / "extension" / "eeg_candidate.py"
    (space / "coder_log.jsonl").write_text(
        json.dumps({"step": 1, "tool": "apply_candidate_patch", "result": {"ok": True, "sha256": "deadbeef", "auto_check": {"ok": True}}}) + "\n",
        encoding="utf-8",
    )
    assert file_sha256(entry) != "deadbeef"
    hashed = _run(mismatch)
    assert hashed["detail"] == "unlogged_code_change"
    assert hashed.get("candidates") in ([], None) or hashed.get("candidates") == []

    manifest = prepare(tmp_path / "manifest", "manifest")
    folder = manifest / "candidates" / "c1"
    source = folder / "extension" / "eeg_candidate.py"
    digest = file_sha256(source)
    (folder / "coder_log.jsonl").write_text(
        json.dumps({"step": 1, "tool": "apply_candidate_patch", "result": {"ok": True, "sha256": digest, "auto_check": {"ok": True}}}) + "\n",
        encoding="utf-8",
    )
    (folder / "source_manifest.json").write_text("{}", encoding="utf-8")
    unfinished = _run(manifest)
    assert unfinished["detail"] == "unlogged_finish"
    assert unfinished["training_jobs"] == 0
    assert not (manifest / "candidates" / "c2").exists()
