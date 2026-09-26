"""Candidate, attempt, and campaign isolation. Fake LLM only."""

from __future__ import annotations

import json
from pathlib import Path

from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
from react_agent.eeg_research.agentic.identity import ensure_attempt, new_call_row, source_hash
from react_agent.eeg_research.agentic.llm import _ledger
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state
from react_agent.eeg_training.protocol import Design


def _campaign(tmp_path: Path, goal_id: str = "recovery") -> Path:
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
        "goal_id": goal_id,
        "research_scope": "pooled_subject_retrieval",
        "final_test_enabled": False,
        "max_training_jobs": 10,
        "max_llm_calls": 100,
        "max_gpu_seconds": 100,
        "max_candidates": 4,
    }
    root = tmp_path / "root"
    create_campaign(root, goal=goal, contract=contract, request_id=goal_id, protocol=protocol)
    camp = root / goal_id
    (camp / "cost.json").write_text(json.dumps({"llm_calls": 4, "llm_failures": 0}), encoding="utf-8")
    return camp


def _install(monkeypatch, behaviour) -> dict:
    seen: dict = {"calls": []}

    def factory(_camp: Path, role: str):
        def call(payload: dict) -> dict:
            seen["calls"].append({"role": role, "identity": getattr(call, "identity", None), "payload": payload})
            return behaviour(role, payload, call)

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)
    return seen


def _ready(camp: Path, candidate_id: str, *, implementation: bool = True) -> Path:
    workspace = camp / "candidates" / candidate_id
    (workspace / "extension").mkdir(parents=True)
    (workspace / "spec.json").write_text("{}", encoding="utf-8")
    (workspace / "extension" / "eeg_candidate.py").write_text(f"# {candidate_id}\nx = 1\n", encoding="utf-8")
    (workspace / "checks.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    if implementation:
        attempt = ensure_attempt(workspace, candidate_id)
        (workspace / "implementation.json").write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "attempt_id": attempt["attempt_id"],
                    "phase": "implement_candidate",
                    "input_hash": source_hash(workspace),
                    "status": "ready_for_review",
                    "steps": 2,
                }
            ),
            encoding="utf-8",
        )
    return workspace


def _open_implement(camp: Path, candidate_id: str = "c3") -> None:
    state = load_state(camp)
    state["status"] = "planning"
    state["hypothesis"] = {"mechanism": "池化"}
    state["experiment"] = {"initial_fidelity": "pilot"}
    state["candidates"] = [
        {"candidate_id": "c1", "status": "implementation_failed"},
        {"candidate_id": "c2", "status": "review_needs_fix"},
    ]
    state["evidence"] = [
        {"evidence_id": "ev_impl_c1", "kind": "implementation", "candidate_id": "c1"},
        {"evidence_id": "ev_impl_c2", "kind": "implementation", "candidate_id": "c2"},
    ]
    state["decisions"] = [
        {"decision_id": "d1", "action": "implement_candidate", "ok": True, "executed": False, "reason_zh": "编写"}
    ]
    save_state(camp, state)
    _ready(camp, "c1", implementation=False)
    _ready(camp, "c2")
    (camp / "candidates" / "c2" / "review.json").write_text(
        json.dumps({"status": "needs_fix", "summary_zh": "旧", "issues": [], "candidate_id": "c2"}),
        encoding="utf-8",
    )
    _ready(camp, candidate_id)


def _run(camp: Path) -> dict:
    from react_agent.eeg_research.agentic.worker import run_worker

    return run_worker(camp, poll_seconds=0, max_ticks=1)


def _cost(camp: Path) -> int:
    return int(json.loads((camp / "cost.json").read_text(encoding="utf-8")).get("llm_calls") or 0)


def test_c2_review_success_does_not_block_c3(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _open_implement(camp)
    _ledger(
        camp,
        new_call_row(role="candidate_reviewer", success=True, candidate_id="c2", attempt_id="old", phase="review_candidate"),
    )
    before = _cost(camp)
    c1 = (camp / "candidates" / "c1" / "extension" / "eeg_candidate.py").read_bytes()
    c2 = (camp / "candidates" / "c2" / "extension" / "eeg_candidate.py").read_bytes()

    def behaviour(role, _payload, call):
        if role == "candidate_reviewer":
            assert call.identity["candidate_id"] == "c3"
            _ledger(camp, new_call_row(success=True, role=role, **call.identity))
            return {"status": "needs_fix", "summary_zh": "改 c3", "issues": [{"title": "形状", "severity": "blocking"}]}
        raise AssertionError(role)

    seen = _install(monkeypatch, behaviour)
    state = _run(camp)
    assert state["status"] != "blocked" or state.get("detail") != "review_result_missing"
    assert state["failure"].get("error_type") != "review_result_missing" if state.get("failure") else True
    assert [row["candidate_id"] for row in state["candidates"]] == ["c1", "c2", "c3"]
    assert not (camp / "candidates" / "c4").exists()
    assert (camp / "candidates" / "c1" / "extension" / "eeg_candidate.py").read_bytes() == c1
    assert (camp / "candidates" / "c2" / "extension" / "eeg_candidate.py").read_bytes() == c2
    assert sum(1 for row in seen["calls"] if row["role"] == "candidate_coder") == 0
    assert _cost(camp) == before + 1
    saved = load_state(camp)
    saved["decisions"][-1]["executed"] = False
    saved["status"] = "planning"
    save_state(camp, saved)
    again = _run(camp)
    assert [row["candidate_id"] for row in again["candidates"]] == ["c1", "c2", "c3"]
    assert [row["evidence_id"] for row in again["evidence"]] == ["ev_impl_c1", "ev_impl_c2", "ev_impl_c3"]
    assert again["training_jobs"] == 0
    assert _cost(camp) == before + 1


def test_restart_keeps_attempt_and_uses_a_new_call_id(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _open_implement(camp)
    workspace = camp / "candidates" / "c3"
    attempt_before = json.loads((workspace / "attempt.json").read_text(encoding="utf-8"))["attempt_id"]
    phase = {"n": 0}

    def behaviour(role, _payload, call):
        if role != "candidate_reviewer":
            raise AssertionError(role)
        phase["n"] += 1
        _ledger(camp, new_call_row(success=phase["n"] > 1, role=role, **(call.identity or {})))
        if phase["n"] == 1:
            from react_agent.eeg_research.agentic.llm import LlmUnavailable

            raise LlmUnavailable("DeepSeekParseError")
        return {"status": "ready", "summary_zh": "通过", "issues": []}

    _install(monkeypatch, behaviour)
    first = _run(camp)
    assert first["status"] == "blocked"
    assert json.loads((workspace / "attempt.json").read_text(encoding="utf-8"))["attempt_id"] == attempt_before
    assert not (workspace / "review.json").exists()
    state = load_state(camp)
    state["status"] = "planning"
    save_state(camp, state)
    second = _run(camp)
    assert second["candidates"][-1]["candidate_id"] == "c3"
    assert json.loads((workspace / "attempt.json").read_text(encoding="utf-8"))["attempt_id"] == attempt_before
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text().splitlines() if line.strip()]
    review_rows = [row for row in rows if row.get("phase") == "review_candidate" and row.get("candidate_id") == "c3"]
    assert len(review_rows) == 2
    assert review_rows[0]["call_id"] != review_rows[1]["call_id"]
    assert review_rows[0]["attempt_id"] == review_rows[1]["attempt_id"] == attempt_before
    assert _cost(camp) == 4 + 2


def test_two_campaigns_do_not_share_review_or_jobs(tmp_path: Path, monkeypatch) -> None:
    left = _campaign(tmp_path / "left", "left")
    right = _campaign(tmp_path / "right", "right")
    _open_implement(left)
    _open_implement(right)
    _ledger(left, new_call_row(role="candidate_reviewer", success=True, candidate_id="c2", phase="review_candidate"))
    left_cost = _cost(left)
    (left / "jobs" / "j1_c2_pilot").mkdir(parents=True)
    (left / "jobs" / "j1_c2_pilot" / "job.json").write_text(json.dumps({"candidate_id": "c2", "status": "running"}), encoding="utf-8")

    def behaviour(role, _payload, call):
        if role == "candidate_reviewer":
            assert call.identity["candidate_id"] == "c3"
            _ledger(right, new_call_row(success=True, role=role, **call.identity))
            return {"status": "ready", "summary_zh": "只属于 right", "issues": []}
        raise AssertionError(role)

    _install(monkeypatch, behaviour)
    state = _run(right)
    assert state["candidates"][-1]["candidate_id"] == "c3"
    assert _cost(left) == left_cost
    assert not (right / "jobs" / "j1_c2_pilot").exists()
    assert "left" not in (right / "llm_calls.jsonl").read_text(encoding="utf-8") if (right / "llm_calls.jsonl").is_file() else True
    left_state = load_state(left)
    assert [row["candidate_id"] for row in left_state["candidates"]] == ["c1", "c2"]


def test_matching_review_is_committed_once_and_a_mismatch_is_redone(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _open_implement(camp)
    workspace = camp / "candidates" / "c3"
    attempt = ensure_attempt(workspace, "c3")
    digest = source_hash(workspace)
    (workspace / "review.json").write_text(
        json.dumps(
            {
                "status": "needs_fix",
                "summary_zh": "已落盘",
                "issues": [{"title": "形状", "severity": "blocking"}],
                "candidate_id": "c3",
                "attempt_id": attempt["attempt_id"],
                "phase": "review_candidate",
                "input_hash": digest,
            }
        ),
        encoding="utf-8",
    )

    def behaviour(role, _payload, _call):
        raise AssertionError(role)

    _install(monkeypatch, behaviour)
    state = _run(camp)
    assert [row["evidence_id"] for row in state["evidence"]][-1] == "ev_impl_c3"
    assert state["candidates"][-1]["status"] == "review_needs_fix"
    assert _cost(camp) == 4

    camp2 = _campaign(tmp_path / "mismatch", "mismatch")
    _open_implement(camp2)
    space = camp2 / "candidates" / "c3"
    other = ensure_attempt(space, "c3")
    (space / "review.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "summary_zh": "旧代码",
                "issues": [],
                "candidate_id": "c3",
                "attempt_id": other["attempt_id"],
                "phase": "review_candidate",
                "input_hash": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    def redo(role, _payload, call):
        if role == "candidate_reviewer":
            _ledger(camp2, new_call_row(success=True, role=role, **call.identity))
            return {"status": "ready", "summary_zh": "新代码", "issues": []}
        raise AssertionError(role)

    _install(monkeypatch, redo)
    redone = _run(camp2)
    saved = json.loads((space / "review.json").read_text(encoding="utf-8"))
    assert saved["input_hash"] == source_hash(space)
    assert saved["input_hash"] != "deadbeef"
    assert redone["candidates"][-1]["candidate_id"] == "c3"
    assert _cost(camp2) == 5


def test_job_identity_mismatch_does_not_start_training(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    state = load_state(camp)
    state["status"] = "planning"
    state["candidate_ready"] = True
    state["candidate_id"] = "c1"
    state["decisions"] = [{"decision_id": "d1", "action": "run_pilot", "ok": True, "executed": False}]
    save_state(camp, state)
    job = camp / "jobs" / "j1_baseline_pilot"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"candidate_id": "c9", "status": "running"}), encoding="utf-8")

    def start_job(*_args, **_kwargs):
        raise AssertionError("training command started")

    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.start_job", start_job)
    _install(monkeypatch, lambda role, _payload, _call: (_ for _ in ()).throw(AssertionError(role)))
    blocked = _run(camp)
    assert blocked["status"] == "blocked"
    assert blocked["detail"] == "job_identity_mismatch"
    assert blocked["training_jobs"] == 0


def test_review_budget_exhausted_stays_on_c3(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _open_implement(camp)
    state = load_state(camp)
    state["max_llm_calls"] = 4
    save_state(camp, state)

    def behaviour(role, _payload, _call):
        raise AssertionError(role)

    _install(monkeypatch, behaviour)
    blocked = _run(camp)
    assert blocked["status"] == "blocked"
    assert blocked["detail"] == "budget_exhausted"
    assert blocked["failure"]["phase"] == "review_candidate"
    assert [row["candidate_id"] for row in blocked["candidates"]] == ["c1", "c2"]
    assert not (camp / "candidates" / "c4").exists()
    assert _cost(camp) == 4


def test_coder_log_tail_still_blocks_without_a_new_candidate(tmp_path: Path, monkeypatch) -> None:
    camp = _campaign(tmp_path)
    _open_implement(camp, "c3")
    workspace = camp / "candidates" / "c3"
    (workspace / "implementation.json").unlink()
    (workspace / "coder_log.jsonl").write_text('{"step": 1, "tool": "read_code", "result": {"ok": true}', encoding="utf-8")

    def behaviour(role, _payload, _call):
        raise AssertionError(role)

    _install(monkeypatch, behaviour)
    blocked = _run(camp)
    assert blocked["detail"] == "coder_log_tail_incomplete"
    assert not (camp / "candidates" / "c4").exists()
    assert [row["candidate_id"] for row in blocked["candidates"]] == ["c1", "c2"]
