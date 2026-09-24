"""Synthetic checks for the code-level research loop. They are not real training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from react_agent.eeg_research.agentic.binding import binding_accepts
from react_agent.eeg_research.agentic.coder import apply_candidate_patch, read_code
from react_agent.eeg_research.agentic.contract import freeze_contract, research_scope
from react_agent.eeg_research.agentic.loop import create_campaign, tick
from react_agent.eeg_research.agentic.memory import confirmation, retrieve
from react_agent.eeg_research.agentic.planner import available_actions, decide
from react_agent.eeg_research.agentic.runner import comparable
from react_agent.eeg_training.protocol import Design

_CANDIDATE = '''
from react_agent.eeg_research.agentic.baseline import EEGCandidate as Baseline

class EEGCandidate(Baseline):
    candidate_id = "c1"
'''


def _design(tmp_path: Path) -> Design:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    return Design("eeg", "inter-subject", "all", train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "test"), policy="agentic")


def test_all_subject_scope_is_pooled(tmp_path: Path) -> None:
    design = _design(tmp_path)
    assert research_scope(design) == "pooled_subject_retrieval"
    contract = freeze_contract(design, tmp_path)
    assert contract["final_test_enabled"] is False
    assert contract["fingerprint"]
    assert "0.363" not in json.dumps(contract["files"])


def test_illegal_json_blocks_without_legacy_fallback() -> None:
    outcome = decide({"available_actions": ["stop", "inspect_data"]}, lambda _obs: "nope")  # type: ignore[arg-type]
    assert outcome["status"] == "blocked"
    assert outcome["repairs"] == 1
    assert "legacy_fixed" not in outcome


def test_patch_and_final_test_reads_are_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "cand"
    refused = apply_candidate_patch(workspace, "src/react_agent/eeg_training/protocol.py", "x = 1\n", "")
    assert refused["error"] == "patch_refused"
    assert read_code(workspace, "extension/final_test/test.pt")["error"] == "read_refused"


def test_patch_needs_the_current_hash_and_check_runs_in_a_child(tmp_path: Path, monkeypatch) -> None:
    import sys

    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_research.agentic.coder import run_candidate_check

    workspace = tmp_path / "cand"
    first = apply_candidate_patch(workspace, "extension/eeg_candidate.py", _CANDIDATE, "")
    assert first["ok"] is True
    stale = apply_candidate_patch(workspace, "extension/eeg_candidate.py", _CANDIDATE, "")
    assert stale["error"] == "base_hash_mismatch"
    entry = workspace / "extension" / "eeg_candidate.py"
    again = apply_candidate_patch(workspace, "extension/eeg_candidate.py", _CANDIDATE, file_sha256(entry))
    assert again["ok"] is True
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-leak")
    result = run_candidate_check(workspace, python=sys.executable)
    assert result["ok"] is True, result
    assert result["shape"] == [4, 1024]
    assert result["loaded_from_workspace"] is True
    assert result["checkpoint_round_trip"] is True
    assert result["params_with_grad"] > 0


def test_candidate_env_has_no_api_key(monkeypatch) -> None:
    from react_agent.eeg_research.agentic.runner import research_env

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("HF_TOKEN", "t")
    env = research_env(Path("/tmp/ext"))
    assert "DEEPSEEK_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert env["EEG_FINAL_TEST"] == "0"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(Path("/tmp/ext").resolve())
    assert env["EEG_CANDIDATE_PATH"] == str(Path("/tmp/ext").resolve())
    relative = research_env(Path("candidates/c10/extension"))
    assert Path(relative["EEG_CANDIDATE_PATH"]).is_absolute()


def test_unloaded_candidate_cannot_enter_evidence(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "source_binding.json").write_text(json.dumps({"module": "react_agent.eeg_training.model", "file_sha256": "abc"}), encoding="utf-8")
    ok, reason = binding_accepts(job, {"module": "eeg_candidate", "entry_sha256": "def"})
    assert ok is False
    assert reason == "candidate_not_loaded"


def test_pilot_and_other_galleries_are_not_ranked() -> None:
    full = {"evaluation_valid": True, "fidelity": "full", "contract_fingerprint": "p", "gallery_size": 1654}
    pilot = dict(full, fidelity="pilot")
    other = dict(full, gallery_size=200)
    assert comparable(full, pilot) is False
    assert comparable(full, other) is False
    assert comparable(full, dict(full)) is True


def test_repeat_request_does_not_open_a_second_campaign(tmp_path: Path) -> None:
    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    goal = {"goal_id": "goal", "max_training_jobs": 2, "max_llm_calls": 4, "max_gpu_seconds": 10}
    first = create_campaign(tmp_path, goal=goal, contract=contract, request_id="same")
    second = create_campaign(tmp_path, goal=goal, contract=contract, request_id="same")
    assert first["goal_id"] == second["goal_id"]
    assert len(list((tmp_path / "goal").glob("campaign_state.json"))) == 1


def test_code_candidate_then_new_evidence_changes_the_next_action(tmp_path: Path) -> None:
    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    goal = {"goal_id": "goal", "max_training_jobs": 4, "max_llm_calls": 40, "max_gpu_seconds": 100}
    create_campaign(tmp_path, goal=goal, contract=contract, request_id="req")
    camp = tmp_path / "goal"
    script = [
        {"action": "propose_experiment", "reason_zh": "先提出投影改动", "evidence_ids": [], "experiment_draft": {"initial_fidelity": "pilot"}, "hypothesis_draft": {"mechanism": "投影尺度"}},
        {"action": "implement_candidate", "reason_zh": "写入候选", "evidence_ids": [], "relative": "extension/eeg_candidate.py", "content": _CANDIDATE, "expected_base_hash": ""},
        {"action": "run_pilot", "reason_zh": "小规模训练", "evidence_ids": []},
    ]

    def backend(obs):
        if obs.get("schema_error"):
            return {"action": "stop", "reason_zh": "停", "evidence_ids": []}
        return script.pop(0)

    def runner(job: Path, manifest: dict, fidelity: str):
        (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.02, "validation_image_count": 10}), encoding="utf-8")
        (job / "source_binding.json").write_text(
            json.dumps({"module": manifest["module"], "file_sha256": manifest["entry_sha256"]}),
            encoding="utf-8",
        )
        return {"gpu_seconds": 3, "seed": 0}

    tick(camp, backend, runner)
    tick(camp, backend, runner)
    state = tick(camp, backend, runner)
    assert state["candidate_ready"] is True
    assert state["evidence"][0]["fidelity"] == "pilot"
    assert state["evidence"][0]["evaluation_valid"] is True
    before = available_actions({"evidence": [], "gpu_seconds_left": 10, "llm_calls_left": 3, "max_training_jobs": 4, "training_jobs": 0})
    after = available_actions(state)
    assert "diagnose_results" not in before
    assert "diagnose_results" in after


def test_zero_budget_does_not_start_training(tmp_path: Path) -> None:
    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    goal = {"goal_id": "goal", "max_training_jobs": 1, "max_llm_calls": 3, "max_gpu_seconds": 0}
    create_campaign(tmp_path, goal=goal, contract=contract, request_id="budget")
    state = tick(tmp_path / "goal", lambda _obs: {"action": "run_pilot", "reason_zh": "想训练", "evidence_ids": []})
    assert state["status"] == "blocked"
    assert state["training_jobs"] == 0


def test_memory_hides_test_and_keeps_levels() -> None:
    rows = retrieve(
        [
            {"task_hash": "a", "contract_fingerprint": "p", "candidate_id": "c", "evidence_level": "exploratory_result"},
            {"task_hash": "b", "contract_fingerprint": "q", "candidate_id": "d", "test_result": 0.2},
            {"task_hash": "other", "contract_fingerprint": "z", "candidate_id": "e", "evidence_level": "exploratory_result"},
        ],
        task_hash="a",
        fingerprint="p",
    )
    assert all("test_result" not in row for row in rows)
    assert rows[0]["retrieval"] == "comparable"
    assert rows[1]["retrieval"] == "analogy_only"
    assert confirmation([0.01, 0.02]) == "provisional_improvement"
    assert confirmation([0.01, 0.02, 0.03]) == "replicated_improvement"


def test_agentic_click_leaves_the_old_campaign_untouched(tmp_path: Path) -> None:
    from react_agent.eeg_training.launch import launch_design

    design = _design(tmp_path)
    old = tmp_path / "runs" / "eeg_research" / design.campaign_id()
    old.mkdir(parents=True)
    for name in ("events.jsonl", "comparison.json", "cost.json", "report.md", "metrics.json"):
        (old / name).write_text(f"old {name}", encoding="utf-8")
    before = {path.name: path.read_text(encoding="utf-8") for path in old.iterdir()}
    payload = launch_design(design, old, "run", tmp_path)
    after = {path.name: path.read_text(encoding="utf-8") for path in old.iterdir()}
    assert after == before
    assert payload["started"] is False
    script = Path(__file__).resolve().parents[2] / "src/react_agent/fmri/workbench_static/app.js"
    text = script.read_text(encoding="utf-8")
    assert 'value="legacy_fixed"' in text
    assert 'value="agentic"' in text
    assert "不使用两试早停" in text


def test_ui_agentic_run_writes_a_new_campaign(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.cli import open_agentic_run

    v1 = Path(__file__).resolve().parents[2] / "runs/eeg_research_v18/eeg_retrieval_research_v1/campaign_state.json"
    before = v1.read_bytes()
    design = _design(tmp_path)
    spawned: list[str] = []

    def spawn(_root, campaign, _poll):
        spawned.append(campaign)
        return 1

    root = tmp_path / "v18"
    payload = {"ok": True, "started": False, "blockers": [], "log": []}
    first = open_agentic_run(design, payload, request_id="click-1", root=root, spawn=spawn)
    camp = root / first["campaign_id"]
    assert first["started"] is True
    assert first["campaign_id"] != "eeg_retrieval_research_v1"
    assert (camp / "campaign_state.json").is_file()
    contract = json.loads((camp / "evaluation_contract.json").read_text(encoding="utf-8"))
    assert contract["final_test_enabled"] is False
    again = open_agentic_run(design, payload, request_id="click-1", root=root, spawn=spawn)
    assert again["campaign_id"] == first["campaign_id"]
    assert spawned == [first["campaign_id"]]
    blocked = open_agentic_run(design, {"ok": False, "started": False, "blockers": ["missing"], "log": []}, request_id="click-2", root=root, spawn=spawn)
    assert blocked["started"] is False
    assert spawned == [first["campaign_id"]]
    assert v1.read_bytes() == before


def test_live_job_is_reconciled_without_a_planner_call(tmp_path: Path) -> None:
    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    goal = {"goal_id": "goal", "max_training_jobs": 4, "max_llm_calls": 8, "max_gpu_seconds": 100}
    create_campaign(tmp_path, goal=goal, contract=contract, request_id="svc")
    camp = tmp_path / "goal"
    from react_agent.eeg_research.agentic.loop import load_state, save_state

    state = load_state(camp)
    state.update({"candidate_ready": True, "candidate_id": "c1", "experiment": {"initial_fidelity": "pilot"}})
    save_state(camp, state)
    calls: list[str] = []
    launched: list[str] = []
    phase = {"running": True}

    def planner(obs):
        calls.append("plan")
        return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": []}

    def launch(_camp, _state, job_id, candidate_id, fidelity):
        launched.append(candidate_id)
        return {"job_id": job_id}

    def settle(_camp, job_id):
        if phase["running"]:
            return {"status": "running", "job_id": job_id}
        candidate = "baseline" if "baseline" in job_id else "c1"
        score = 0.004 if candidate == "baseline" else 0.003
        return {
            "status": "finished",
            "job_id": job_id,
            "candidate_id": candidate,
            "seed": 0,
            "gpu_seconds": 5,
            "result": {"evaluation_valid": True, "fidelity": "pilot", "fixed_bank_top1": score, "gallery_size": 1654},
        }

    services = {"launch": launch, "settle": settle}
    tick(camp, planner, services=services)
    assert launched == ["baseline"]
    tick(camp, planner, services=services)
    assert calls == ["plan"]
    phase["running"] = False
    tick(camp, planner, services=services)
    tick(camp, planner, services=services)
    assert launched == ["baseline", "c1"]
    state = tick(camp, planner, services=services)
    assert calls == ["plan"]
    rows = [row for row in state["evidence"] if row.get("candidate_id") == "c1"]
    assert rows[0]["delta_vs_control_pp"] == -0.1
    assert "run_pilot" not in available_actions(state)
    assert any(row["kind"] == "implementation_failure" or row["kind"] == "exploratory_result" for row in state["memory"])


def test_repeated_local_action_needs_new_evidence() -> None:
    state = {
        "evidence": [{"evidence_id": "ev_1", "fidelity": "pilot", "evaluation_valid": True, "candidate_id": "baseline"}],
        "decisions": [{"action": "diagnose_results", "ok": True, "evidence_count": 1}],
        "gpu_seconds_left": 10,
        "llm_calls_left": 20,
        "max_training_jobs": 4,
        "training_jobs": 1,
    }
    assert "diagnose_results" not in available_actions(state)
    state["evidence"].append({"evidence_id": "ev_2", "fidelity": "pilot", "evaluation_valid": True, "candidate_id": "c1"})
    assert "diagnose_results" in available_actions(state)
    state["experiment"] = {"initial_fidelity": "pilot"}
    assert "propose_experiment" not in available_actions(state)
    assert "implement_candidate" in available_actions(state)
    state["experiment_failed"] = True
    assert "propose_experiment" in available_actions(state)
    state["decisions"] = [
        {"action": "propose_experiment", "ok": True, "evidence_count": 2},
        {"action": "propose_experiment", "ok": True, "evidence_count": 1},
    ]
    assert "propose_experiment" in available_actions(state)


def test_coder_loop_patches_checks_and_finishes(tmp_path: Path) -> None:
    import sys

    from react_agent.eeg_research.agentic.native_patch import implement

    seen: list[dict] = []
    replies = [
        {"tool": "apply_candidate_patch", "args": {"path": "extension/eeg_candidate.py", "content": "raise ImportError('broken')\n", "expected_base_hash": ""}},
        None,
        {"tool": "finish_patch", "args": {"summary": "baseline subclass"}},
    ]

    def backend(request):
        seen.append(request)
        reply = replies.pop(0)
        if reply is None:
            current = request["current_file"]["sha256"]
            return {"tool": "apply_candidate_patch", "args": {"path": "extension/eeg_candidate.py", "content": _CANDIDATE, "expected_base_hash": current}}
        return reply

    outcome = implement(tmp_path / "ws", {"hypothesis": {}}, backend, python=sys.executable)
    assert outcome["status"] == "ready_for_review"
    assert seen[0]["input_spec"]["c_num"] == 17
    assert seen[1]["last_check"]["ok"] is False
    assert seen[2]["last_check"]["ok"] is True
    assert (tmp_path / "ws" / "source_manifest.json").is_file()


def test_pause_survives_a_worker_save(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import load_state, request_control, save_state

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal={"goal_id": "goal"}, contract=contract, request_id="p")
    camp = tmp_path / "goal"
    stale = load_state(camp)
    request_control(camp, "pause")
    stale["status"] = "planning"
    save_state(camp, stale)
    assert load_state(camp)["status"] == "paused"


def test_repeated_diagnosis_adds_no_evidence(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import load_state, save_state

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal={"goal_id": "goal", "max_llm_calls": 10}, contract=contract, request_id="d")
    camp = tmp_path / "goal"
    job = tmp_path / "job"
    job.mkdir()
    (job / "history.jsonl").write_text('{"epoch": 1, "train_loss": 2.0, "fixed_bank_top1": 0.01}\n', encoding="utf-8")
    state = load_state(camp)
    state["evidence"].append({"evidence_id": "ev_j1", "candidate_id": "baseline", "fidelity": "pilot", "evaluation_valid": True, "job_dir": str(job)})
    save_state(camp, state)
    backend = lambda _obs: {"action": "diagnose_results", "reason_zh": "看曲线", "evidence_ids": []}  # noqa: E731
    first = tick(camp, backend)
    assert len(first["evidence"]) == 2
    assert "diagnose_results" not in available_actions(first)


_STATS_CANDIDATE = '''
import torch
from torch import nn
from react_agent.eeg_research.agentic.baseline import EEGCandidate as Baseline


class _Norm(nn.Module):
    def __init__(self, inner, c_num):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.zeros(1, c_num, 1))
        self.register_buffer("std", torch.ones(1, c_num, 1))

    def forward(self, x):
        return self.inner((x - self.mean) / self.std)


class EEGCandidate(Baseline):
    candidate_id = "stats"

    def build_encoder(self, input_spec, model_config=None):
        return _Norm(super().build_encoder(input_spec, model_config), int(input_spec["c_num"]))

    def fit_statistics(self, encoder, stats):
        encoder.mean.copy_(torch.tensor(stats["mean"]).view(1, -1, 1))
        encoder.std.copy_(torch.tensor(stats["std"]).view(1, -1, 1))
'''


def test_train_statistics_hook_is_checked_and_uses_train_only(tmp_path: Path) -> None:
    import sys

    import torch

    from react_agent.eeg_research.agentic.coder import run_candidate_check
    from react_agent.eeg_training.train_entry import train_channel_statistics

    workspace = tmp_path / "cand"
    apply_candidate_patch(workspace, "extension/eeg_candidate.py", _STATS_CANDIDATE, "")
    result = run_candidate_check(workspace, python=sys.executable)
    assert result["ok"] is True, result
    assert result["uses_train_statistics"] is True
    assert result["statistics_stored_as_buffers"] is True
    loader = [{"eeg": torch.ones(2, 3, 4)}, {"eeg": 3 * torch.ones(1, 3, 4)}]
    stats = train_channel_statistics(loader)
    assert stats["source"] == "train_files"
    assert stats["subject_ids"] is None
    assert stats["values_per_channel"] == 12
    assert abs(stats["mean"][0] - (8 * 1 + 4 * 3) / 12) < 1e-9


def test_reviewer_gets_guarantees_and_one_repair(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import review

    workspace = tmp_path / "cand"
    apply_candidate_patch(workspace, "extension/eeg_candidate.py", _CANDIDATE, "")
    seen: list[dict] = []
    replies = [{"verdict": "ok"}, {"status": "ready", "issues": [], "summary_zh": "可以试跑"}]

    def backend(payload):
        seen.append(payload)
        return replies.pop(0)

    result = review(workspace, {"hypothesis": {}}, {"fingerprint": "p"}, backend, "m")
    assert result["status"] == "ready"
    assert result["schema_repaired"] is True
    assert "runtime_guarantees" in seen[0]
    assert "schema_error" in seen[1]
    assert "score" not in result
    broken = review(workspace, {}, {}, lambda _p: {"verdict": "ok"}, "m")
    assert broken["status"] == "blocked"
    assert broken["format_failed"] is True


def test_coder_stops_before_the_budget_and_planner_reserves_calls(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import implement

    left = {"n": 4}

    def backend(_request):
        left["n"] -= 1
        return {"tool": "list_project_files", "args": {}}

    outcome = implement(tmp_path / "ws", {}, backend, calls_left=lambda: left["n"], reserve=2)
    assert outcome["detail"] == "budget_exhausted"
    assert left["n"] == 2
    state = {"experiment": {"initial_fidelity": "pilot"}, "llm_calls_left": 5, "gpu_seconds_left": 10}
    assert "implement_candidate" not in available_actions(state)
    state["llm_calls_left"] = 20
    assert "implement_candidate" in available_actions(state)
    state["max_candidates"] = 4
    state["candidates"] = [
        {"candidate_id": "c1", "status": "implementation_failed"},
        {"candidate_id": "c5", "status": "requires_framework_extension"},
        {"candidate_id": "c6", "status": "review_blocked"},
        {"candidate_id": "c7", "status": "review_blocked"},
    ]
    assert "implement_candidate" in available_actions(state)
    state["candidates"].append({"candidate_id": "c8", "status": "ready"})
    assert "implement_candidate" not in available_actions(state)


def test_coder_must_write_after_two_reads_and_extension_is_answered_once(tmp_path: Path) -> None:
    import sys

    from react_agent.eeg_research.agentic.native_patch import implement

    seen: list[dict] = []
    replies = [
        {"tool": "list_project_files", "args": {}},
        {"tool": "list_project_files", "args": {}},
        {"tool": "list_project_files", "args": {}},
        {"tool": "requires_framework_extension", "args": {"missing_interface": "need custom module"}},
        {"tool": "apply_candidate_patch", "args": {"path": "extension/eeg_candidate.py", "content": _CANDIDATE, "expected_base_hash": ""}},
        {"tool": "finish_patch", "args": {"summary": "ok"}},
    ]

    def backend(request):
        seen.append(request)
        return replies.pop(0)

    outcome = implement(tmp_path / "ws", {}, backend, python=sys.executable)
    assert outcome["status"] == "ready_for_review"
    log = [json.loads(line) for line in (tmp_path / "ws" / "coder_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[2]["result"]["error"] == "write_required"
    assert log[3]["result"]["error"] == "interface_already_available"


def test_retrieval_owns_scale_when_encoder_has_none() -> None:
    import torch
    from torch import nn

    from react_agent.eeg_training.model import EEGProjectLayer, LocalRetrieval

    class Bare(nn.Module):
        def forward(self, inputs):
            return inputs.mean(dim=-1).repeat(1, 1024)[:, :1024]

    eeg = torch.zeros(4, 17, 250)
    image = torch.ones(4, 1024)
    bare = LocalRetrieval(Bare())
    loss, top1, top5 = bare(eeg, image)
    assert torch.isfinite(loss)
    assert bare.logit_scale.requires_grad
    baseline = LocalRetrieval(EEGProjectLayer(1024, 17, [0, 250]))
    assert not hasattr(baseline, "logit_scale")
    base_loss, _, _ = baseline(eeg, image)
    assert torch.isfinite(base_loss)


def test_c10_runtime_opinions_do_not_block_training(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import review

    training_budget = {
        "severity": "blocking",
        "title": "延长的训练预算未在候选实现中体现，规格要求的最关键变量无法由本候选保证",
        "evidence": "required_controls 要求对照使用相同延长训练。candidate_source 无任何训练循环、epoch 数或调度相关代码。",
        "impact": "若训练时长由外部默认值决定，实验无法对假设做出可判别的检验。",
    }
    interface_assert = {
        "severity": "blocking",
        "title": "接口断言层未按规格落地，只存在于 encoder 内部而非输入/输出契约层",
        "evidence": "规格要求最前置加显式接口断言：校验输入为 [batch,17,250] float32、输出为 [batch,1024] 有限值。",
        "impact": "check 中的形状结果不能证明运行时会拦截键名错误。",
    }
    workspace = tmp_path / "c10"
    apply_candidate_patch(workspace, "extension/eeg_candidate.py", _CANDIDATE, "")
    (workspace / "checks.json").write_text("{}", encoding="utf-8")

    def backend(_payload):
        return {
            "status": "needs_fix",
            "issues": [training_budget, interface_assert],
            "summary_zh": "两条阻塞意见都在运行时职责内。",
        }

    result = review(workspace, {"hypothesis": {}}, {"fingerprint": "p"}, backend, "m")
    assert result["status"] == "ready"
    assert result["model_status"] == "needs_fix"
    assert result["blocking_remaining"] == 0
    assert result["runtime_issues_downgraded"] == 2
    assert all(row["severity"] == "non_blocking" for row in result["issues"])

    def still_blocked(_payload):
        return {
            "status": "needs_fix",
            "issues": [
                training_budget,
                {
                    "severity": "blocking",
                    "title": "输出维度写成 512，与契约 1024 不符",
                    "evidence": "Linear 的 out_features 是 512。",
                    "impact": "检查要求的 [batch, 1024] 无法满足。",
                },
            ],
            "summary_zh": "还有一个代码缺陷。",
        }

    (tmp_path / "bad").mkdir()
    kept = review(tmp_path / "bad", {"hypothesis": {}}, {}, still_blocked, "m")
    assert kept["status"] == "needs_fix"
    assert kept["blocking_remaining"] == 1
    assert kept["issues"][1]["severity"] == "blocking"


def _job_record(pid: int, *, started_at: float) -> dict:
    from react_agent.eeg_research.agentic.jobs import _proc_start

    return {
        "job_id": "j1",
        "pid": pid,
        "proc_start": _proc_start(pid),
        "status": "running",
        "started_at": started_at,
        "gpu": [0],
        "fidelity": "pilot",
        "manifest": {"module": "m", "entry_sha256": "abc"},
    }


def _write_finished_artifacts(job: Path, record: dict) -> None:
    job.mkdir()
    (job / "metrics.json").write_text(
        json.dumps({"fixed_bank_top1": 0.004, "validation_image_count": 1654}),
        encoding="utf-8",
    )
    (job / "source_binding.json").write_text(
        json.dumps({"module": "m", "file_sha256": "abc"}),
        encoding="utf-8",
    )
    (job / "status.json").write_text(
        json.dumps({"phase": "finished", "epoch": 3, "epochs": 3}),
        encoding="utf-8",
    )
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")


def test_zombie_child_is_reconciled_and_reaped(tmp_path: Path) -> None:
    import time

    from react_agent.eeg_research.agentic.jobs import reconcile, stop_job

    pid = os.fork()
    if pid == 0:
        os._exit(0)
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except OSError:
            state = ""
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        os.waitpid(pid, 0)
        raise AssertionError("child did not become a zombie")

    job = tmp_path / "j1"
    _write_finished_artifacts(job, _job_record(pid, started_at=time.time() - 5))
    settled = reconcile(job)
    assert settled["status"] == "finished"
    assert settled["result"]["evaluation_valid"] is True
    assert settled["result"]["fixed_bank_top1"] == 0.004
    again = reconcile(job)
    assert again["status"] == "finished"
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited = 0
    assert waited == 0

    spare = os.fork()
    if spare == 0:
        os._exit(0)
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            state = Path(f"/proc/{spare}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except OSError:
            state = ""
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        os.waitpid(spare, 0)
        raise AssertionError("child did not become a zombie")
    other = tmp_path / "j2"
    _write_finished_artifacts(other, _job_record(spare, started_at=time.time() - 5))
    stopped = stop_job(other)
    assert stopped["status"] == "running"
    assert json.loads((other / "job.json").read_text(encoding="utf-8"))["status"] == "running"
    try:
        waited, _status = os.waitpid(spare, os.WNOHANG)
    except ChildProcessError:
        waited = 0
    assert waited == 0


def test_running_child_is_not_settled_early(tmp_path: Path) -> None:
    import signal
    import subprocess
    import time

    from react_agent.eeg_research.agentic.jobs import _alive, reconcile

    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        job = tmp_path / "j1"
        record = _job_record(proc.pid, started_at=time.time())
        _write_finished_artifacts(job, record)
        settled = reconcile(job)
        assert settled["status"] == "running"
        assert json.loads((job / "job.json").read_text(encoding="utf-8"))["status"] == "running"
        assert _alive(record) is True
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)


def test_new_planner_does_not_stop_on_a_small_gap() -> None:
    prompt = Path(__file__).resolve().parents[2] / "src/react_agent/eeg_research/agentic/prompts/research_planner.txt"
    text = prompt.read_text(encoding="utf-8")
    assert "0.005" not in text
    assert "Do not stop because the best two scores are close" in text
