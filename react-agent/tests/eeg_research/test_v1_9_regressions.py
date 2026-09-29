"""V1.9.1/1.9.2 contract regressions. Synthetic only: no live API, no GPU Goal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend, _ledger
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, observation, save_state, tick
from react_agent.eeg_research.agentic.roles import RoleResultError, begin_role_task, validate_role_result, wrap_role_result
from react_agent.eeg_research.agentic.schemas import ROLE_RESULT_VERSION
from react_agent.eeg_research.agentic.task_ledger import latest
from react_agent.eeg_training.protocol import Design
from react_agent.fmri.schemas import LMUsage


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


def _goal(**kwargs) -> dict:
    fields = {
        "goal_id": "goal",
        "max_training_jobs": 4,
        "max_llm_calls": 40,
        "max_gpu_seconds": 100,
        "max_candidates": 4,
    }
    fields.update(kwargs)
    return fields


def _open_campaign(tmp_path: Path, *, request_id: str = "v19"):
    from react_agent.eeg_research.agentic.contract import freeze_contract

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal=_goal(), contract=contract, request_id=request_id)
    return tmp_path / "goal", None


def _usage() -> LMUsage:
    return LMUsage(
        provider="deepseek",
        requested_model="deepseek-flash",
        response_model="deepseek-flash",
        finish_reason="stop",
        input_tokens=4,
        output_tokens=8,
    )


def test_observation_keeps_final_test_enabled(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="obs-goal")
    goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
    goal["final_test_enabled"] = False
    goal["test_result"] = {"top1": 0.91}
    goal["final_test_top1"] = 0.91
    (camp / "goal.json").write_text(json.dumps(goal, ensure_ascii=False), encoding="utf-8")
    obs = observation(camp)
    assert obs["goal"]["final_test_enabled"] is False
    assert "test_result" not in obs["goal"]
    assert "final_test_top1" not in obs["goal"]


def test_omitted_stop_reason_is_blocked_not_invented(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="stop-omit")
    state = tick(camp, lambda _obs: {"action": "stop", "reason_zh": "停", "evidence_ids": []})
    assert state["status"] == "finished"
    assert state["stop_reason"] == "blocked"
    assert state["stop_reason"] != "no_supported_next_experiment"


def test_analyze_bills_analyst_and_curator(tmp_path: Path, monkeypatch) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="analyze-bill")
    state = load_state(camp)
    state["evidence"] = [
        {
            "evidence_id": "ev_1",
            "candidate_id": "c1",
            "fidelity": "pilot",
            "evaluation_valid": True,
            "fixed_bank_top1": 0.12,
            "comparison": {"comparable": True, "delta_pp": 1.5, "reason": "matched_control"},
            "diagnostics": {"schema_version": "eeg_research.diagnostic_bundle.v1"},
        }
    ]
    save_state(camp, state)

    def factory(_camp: Path, role: str):
        def call(_payload: dict) -> dict:
            _ledger(_camp, {"success": True, "role": role, "input_tokens": 1, "output_tokens": 1})
            if role == "result_analyst":
                return {"hypothesis_assessment": "inconclusive", "summary_zh": "分析"}
            if role == "memory_curator":
                return {"proposed_lessons": [], "summary_zh": "无新课"}
            return {"summary_zh": "ok"}

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)
    from react_agent.eeg_research.agentic.worker import build_services

    services = build_services(camp)
    services["analyze"](camp, state)
    assert state["evidence"][0]["comparison"]["comparable"] is True
    cost = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
    assert cost["llm_calls"] == 2
    assert state["llm_calls"] == 2
    assert state["llm_calls_left"] == int(state["max_llm_calls"]) - 2


def test_curator_failure_keeps_comparison_and_marks_pending(tmp_path: Path, monkeypatch) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="curator-fail")
    state = load_state(camp)
    comparison = {"comparable": True, "delta_pp": 0.4, "reason": "matched_control"}
    state["evidence"] = [
        {
            "evidence_id": "ev_1",
            "candidate_id": "c1",
            "fidelity": "full",
            "evaluation_valid": True,
            "fixed_bank_top1": 0.2,
            "comparison": comparison,
        }
    ]
    save_state(camp, state)

    def factory(_camp: Path, role: str):
        def call(_payload: dict) -> dict:
            _ledger(_camp, {"success": role != "memory_curator", "role": role})
            if role == "memory_curator":
                raise LlmUnavailable("DeepSeekParseError")
            return {"hypothesis_assessment": "inconclusive", "summary_zh": "分析"}

        call.model = "fake"  # type: ignore[attr-defined]
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)
    from react_agent.eeg_research.agentic.memory import EpisodeStore
    from react_agent.eeg_research.agentic.worker import build_services

    services = build_services(camp)
    services["analyze"](camp, state)
    assert state["evidence"][0]["comparison"] == comparison
    assert EpisodeStore(camp).pending_curation() is True
    cost = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
    assert cost["llm_calls"] == 2


def test_validate_role_result_rejects_wrong_identity(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="envelope")
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"])
    wrapped = wrap_role_result({"status": "completed", "summary_zh": "占位", "verdict": "REVISE"}, task=task)
    assert validate_role_result(wrapped, task)["task_id"] == task["task_id"]
    with pytest.raises(RoleResultError, match="identity_mismatch"):
        validate_role_result({**wrapped, "task_id": "task_other"}, task)
    with pytest.raises(RoleResultError, match="identity_mismatch"):
        validate_role_result({**wrapped, "input_digest": "deadbeef"}, task)


def test_bind_wraps_partial_envelope_and_dict_refs() -> None:
    from react_agent.eeg_research.agentic.roles import bind_role_output

    task = {"task_id": "task_a", "attempt_id": "att1", "input_digest": "abc"}
    wrapped = bind_role_output(
        {
            "schema_version": ROLE_RESULT_VERSION,
            "summary_zh": "部分信封",
            "artifact_refs": [{"ref": "placeholder", "note": "no training"}],
            "proposed_lessons": [],
        },
        task=task,
    )
    assert wrapped["task_id"] == "task_a"
    assert wrapped["attempt_id"] == "att1"
    assert wrapped["input_digest"] == "abc"
    assert wrapped["artifact_refs"] == ["placeholder"]
    assert wrapped["proposed_lessons"] == []


def test_role_backend_rejects_mismatched_envelope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    camp, _protocol = _open_campaign(tmp_path, request_id="llm-envelope")
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"])

    class Fake:
        def __init__(self, _config) -> None:
            return

        async def complete_json(self, **_kwargs):
            return (
                {
                    "schema_version": ROLE_RESULT_VERSION,
                    "task_id": "task_other",
                    "attempt_id": task["attempt_id"],
                    "input_digest": task["input_digest"],
                    "status": "completed",
                    "summary_zh": "错身份",
                },
                _usage(),
            )

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    backend = role_backend(camp, "result_auditor")
    with pytest.raises(LlmUnavailable, match="identity_mismatch"):
        backend(
            {
                "task_id": task["task_id"],
                "attempt_id": task["attempt_id"],
                "input_digest": task["input_digest"],
            }
        )


def test_evaluate_export_prints_complete_argv(tmp_path: Path, monkeypatch, capsys) -> None:
    from react_agent.eeg_research.agentic.cli import main
    from react_agent.eeg_research.agentic.export import pack_candidate
    from react_agent.eeg_training import train_entry

    job = tmp_path / "job"
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate"}), encoding="utf-8")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "pack"
    pack_candidate(job, dest)
    seen: dict[str, list[str] | None] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(train_entry, "main", fake_main)
    code = main(["evaluate-export", "--pack", str(dest), "--data-root", str(tmp_path)])
    assert code == 0
    assert seen == {}
    printed = json.loads(capsys.readouterr().out)
    assert printed["evaluate_only"] is True
    assert printed["final_test"] is False
    assert printed["out"] == str(dest / "eval_only")
    argv = printed["argv"]
    assert "--evaluate-only" in argv
    assert "--test-only" not in argv
    assert "--dataset" in argv
    assert "--data-root" in argv
    assert "--out" in argv
    assert str(dest / "eval_only") in argv
    flags = argv[argv.index("react_agent.eeg_training.train_entry") + 1 :]
    train_entry.main(flags)
    assert seen["argv"] is not None
    assert "--evaluate-only" in seen["argv"]
    assert "--test-only" not in seen["argv"]


def test_truncated_read_same_range_is_repeated_then_next_range_works(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import implement

    workspace = tmp_path / "ws"
    replies = [
        {"tool": "read_code", "args": {"path": "reference/baseline.py", "start": 1, "end": 10}},
        {"tool": "read_code", "args": {"path": "reference/baseline.py", "start": 1, "end": 10}},
        {"tool": "read_code", "args": {"path": "reference/baseline.py", "start": 11, "end": 20}},
    ]

    def backend(_request):
        return replies.pop(0)

    implement(workspace, {}, backend, max_steps=3, python="python")
    log = [json.loads(line) for line in (workspace / "coder_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[0]["result"]["truncated"] is True
    assert log[1]["result"]["error"] == "repeated_read"
    assert log[2]["result"]["ok"] is True
    assert log[2]["result"].get("truncated") in {False, True}
    assert log[2]["result"]["text"]


def test_design_experiment_writes_spec_and_ledger(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="design")

    def designer(payload: dict) -> dict:
        assert payload["task_id"]
        assert payload["attempt_id"]
        return {
            "experiment_spec": {"initial_fidelity": "pilot", "intervention": "temporal_pooling"},
            "summary_zh": "写出草稿",
        }

    state = tick(
        camp,
        lambda _obs: {
            "action": "design_experiment",
            "reason_zh": "设计",
            "evidence_ids": [],
            "hypothesis_draft": {"mechanism": "时域池化"},
            "experiment_draft": {"initial_fidelity": "pilot"},
        },
        services={"designer": designer},
    )
    specs = list((camp / "experiments").glob("spec_*.json"))
    assert specs
    payload = json.loads(specs[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eeg_research.role_result.v1"
    assert latest(camp, payload["task_id"])["status"] == "completed"
    assert state["experiment"]["intervention"] == "temporal_pooling"


def test_collect_diagnostics_writes_bundle(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="diag")
    job = camp / "jobs" / "j1"
    job.mkdir(parents=True)
    state = load_state(camp)
    state["evidence"] = [
        {
            "evidence_id": "ev_1",
            "candidate_id": "c1",
            "fidelity": "pilot",
            "evaluation_valid": True,
            "job_dir": str(job),
        }
    ]
    save_state(camp, state)
    tick(camp, lambda _obs: {"action": "collect_diagnostics", "reason_zh": "诊断", "evidence_ids": ["ev_1"]})
    assert (job / "diagnostic_summary.json").is_file()
    assert (job / "diagnostic_bundle.json").is_file()
    saved = load_state(camp)
    rows = [row for row in saved["evidence"] if row.get("kind") == "diagnostic_bundle"]
    assert rows
    assert rows[0]["diagnostic_ref"]


def test_repair_candidate_uses_implement_and_ledger(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, request_id="repair")
    state = load_state(camp)
    state["repair_task"] = {"candidate_id": "c1", "remaining": 1}
    state["llm_calls_left"] = 40
    state["candidates"] = [{"candidate_id": "c1", "status": "review_needs_fix"}]
    save_state(camp, state)
    called: list[str] = []

    def implement(camp_dir: Path, current: dict) -> None:
        from react_agent.eeg_research.agentic.roles import finish_role_task

        called.append(current["repair_task"]["candidate_id"])
        task = begin_role_task(camp_dir, role="candidate_coder", inputs=[camp_dir / "goal.json"], candidate_id="c1")
        path = camp_dir / "candidates" / "c1" / "repair.json"
        finish_role_task(camp_dir, task, {"status": "completed", "summary_zh": "修复"}, kind="repair", path=path, candidate_id="c1")
        current["repair_task"]["remaining"] = 0

    tick(
        camp,
        lambda _obs: {"action": "repair_candidate", "reason_zh": "修", "evidence_ids": []},
        services={"implement": implement},
    )
    assert called == ["c1"]
    payload = json.loads((camp / "candidates" / "c1" / "repair.json").read_text(encoding="utf-8"))
    assert latest(camp, payload["task_id"])["status"] == "completed"


def test_role_backend_wraps_domain_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    camp, _protocol = _open_campaign(tmp_path, request_id="wrap-domain")
    task = begin_role_task(camp, role="memory_curator", inputs=[camp / "goal.json"])

    class Fake:
        def __init__(self, _config) -> None:
            return

        async def complete_json(self, **_kwargs):
            return ({"proposed_lessons": [], "summary_zh": "无课"}, _usage())

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    reply = role_backend(camp, "memory_curator")(
        {
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "episodes": [],
        }
    )
    assert reply["schema_version"] == ROLE_RESULT_VERSION
    assert reply["task_id"] == task["task_id"]
    assert reply["proposed_lessons"] == []
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["success"] is True
