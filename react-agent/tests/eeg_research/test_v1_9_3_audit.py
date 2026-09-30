"""R01–R08 and the call-chain checks from the V1.9.2 audit. No API and no GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range, verify
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, tick
from react_agent.eeg_research.agentic.roles import RoleResultError, begin_role_task, bind_role_output, finish_role_task
from react_agent.eeg_research.agentic.schemas import ROLE_RESULT_VERSION


def _goal(**kwargs) -> dict:
    fields = {"goal_id": "goal", "max_training_jobs": 4, "max_llm_calls": 40, "max_gpu_seconds": 100, "max_candidates": 4}
    fields.update(kwargs)
    return fields


def _camp(tmp_path: Path, request_id: str = "audit") -> Path:
    create_campaign(tmp_path, goal=_goal(), contract={"fingerprint": "fp"}, request_id=request_id)
    return tmp_path / "goal"


def test_r01_finished_role_artifact_verifies_immediately(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "r01")
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"])
    path = camp / "audits" / "once.json"
    envelope = finish_role_task(camp, task, {"status": "completed", "summary_zh": "一次写完"}, kind="audit", path=path)
    artifact_id = envelope["artifact_refs"][-1]
    ok, reason = verify(camp, artifact_id)
    assert ok is True
    assert reason == "ok"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ok, reason = verify(camp, artifact_id)
    assert ok is False
    assert reason == "artifact_hash_mismatch"


def test_r02_wrong_attempt_is_not_rebound(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "r02")
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"])
    with pytest.raises(RoleResultError, match="identity_mismatch"):
        bind_role_output(
            {
                "schema_version": ROLE_RESULT_VERSION,
                "task_id": task["task_id"],
                "attempt_id": "att_old",
                "input_digest": "digest_old",
                "status": "completed",
                "summary_zh": "旧回复",
            },
            task=task,
        )
    wrapped = bind_role_output({"status": "completed", "summary_zh": "业务载荷", "verdict": "REVISE"}, task=task)
    assert wrapped["task_id"] == task["task_id"]
    assert wrapped["attempt_id"] == task["attempt_id"]


def test_r03_registered_spec_matches_state(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "r03")

    def designer(payload: dict) -> dict:
        assert payload["diagnostics"] is None or "missing_inputs" in payload
        assert "budget" in payload
        assert "capabilities" in payload
        assert "evaluation_contract" in payload
        return {"experiment_spec": {"initial_fidelity": "pilot", "intervention": "temporal_pooling"}, "summary_zh": "修订"}

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
    spec_path = next((camp / "experiments").glob("spec_*.json"))
    body = json.loads(spec_path.read_text(encoding="utf-8"))
    stored = body.get("experiment_spec") or body["payload"]["experiment_spec"]
    assert stored["intervention"] == "temporal_pooling"
    assert stored == state["experiment"]
    assert stored["status"] == "approved"


def test_r04_extension_blocks_implement(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "r04")

    def designer(_payload: dict) -> dict:
        return {"status": "requires_framework_extension", "required_capability_ids": ["custom_attention"], "summary_zh": "缺接口"}

    state = tick(
        camp,
        lambda _obs: {
            "action": "design_experiment",
            "reason_zh": "设计",
            "evidence_ids": [],
            "hypothesis_draft": {"mechanism": "注意力"},
            "experiment_draft": {"initial_fidelity": "pilot", "intervention": "attention"},
        },
        services={"designer": designer},
    )
    assert state["experiment_failed"] is True
    from react_agent.eeg_research.agentic.planner import available_actions

    assert "implement_candidate" not in available_actions(state)


def test_r05_audit_pass_cannot_override_revise(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "r05")
    state = tick(
        camp,
        lambda _obs: {"action": "audit_result", "reason_zh": "审计", "evidence_ids": []},
        services={"auditor": lambda _payload: {"verdict": "PASS", "claims": [{"claim_id": "c", "status": "supported"}], "summary_zh": "通过"}},
    )
    rows = [row for row in state["evidence"] if row.get("kind") == "audit"]
    assert rows[-1]["summary"]["verdict"] == "REVISE"
    assert rows[-1]["summary"]["model_verdict_ignored"] == "PASS"
    assert rows[-1]["summary"]["auditor_claims"]


def test_r06_lesson_rejects_empty_conditions_missing_counterexample_and_forged_effect(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode

    store = EpisodeStore(tmp_path)
    saved = store.persist_episode(
        episode(task_hash="g", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=0, metric=0.1, contract_fingerprint="fp", artifact="a")
    )
    result = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "supporting_episode_ids": [saved["episode_id"]],
                    "contradicting_episode_ids": ["ep_missing"],
                    "conditions": None,
                    "observed_effect": 999,
                    "evidence_level": "exploratory_result",
                }
            ]
        }
    )
    assert result["accepted"] == []
    reasons = result["rejected"][0]["reasons"]
    assert "episode_missing" in reasons
    assert "conditions_missing" in reasons
    assert "forged_effect" in reasons


def test_r07_pilot_is_not_confirmation() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 1.2, "reason": "matched_control"},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 3},
        paired_deltas_pp=[0.2, 0.2, 1.2],
        fidelity="pilot",
    )
    assert promo["status"] != "confirmed"
    assert promo["tier"] != "confirmation"


def test_r08_mean_below_one_pp_is_not_confirmed() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 1.2, "reason": "matched_control"},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 3},
        paired_deltas_pp=[0.2, 0.2, 1.2],
        fidelity="full",
    )
    assert promo["status"] != "confirmed"
    assert promo["tier"] != "confirmation"


def test_full_identity_pairs_can_confirm() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    records = [
        {
            "delta_pp": 1.4,
            "fidelity": "full",
            "seed": seed,
            "run_id": f"run_{seed}",
            "control_run_id": f"base_{seed}",
            "checkpoint_id": f"ckpt_{seed}",
            "source_hash": "src",
            "config_hash": "cfg",
            "approval_ref": "art_exp",
            "evaluation_valid": True,
        }
        for seed in (0, 1, 2)
    ]
    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 1.4, "reason": "matched_control"},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 3, "training_seeds": [0, 1, 2]},
        paired_records=records,
        fidelity="full",
    )
    assert promo["status"] == "confirmed"
    assert promo["tier"] == "confirmation"


def test_hook_config_and_objective_typeerror_are_not_swallowed() -> None:
    import torch

    from react_agent.eeg_training.hooks import HookConfigError, call_configured, compute_objective, resolve_negative_policy
    from react_agent.eeg_training.model import contrastive_loss

    def builder(payload):
        assert payload["sentinel"] == 7
        return payload["sentinel"]

    builder.accepted_config_keys = {"sentinel"}
    assert call_configured(builder, {"sentinel": 7}, kind="model") == 7
    with pytest.raises(HookConfigError):
        call_configured(builder, {"sentinel": 7, "extra": 1}, kind="model")

    def internal(eeg_z, img_z, scale, positives=None):
        assert positives == ["img"]
        raise TypeError("inside")

    eeg = torch.zeros(2, 2)
    img = torch.zeros(2, 2)
    with pytest.raises(TypeError, match="inside"):
        compute_objective(internal, eeg, img, torch.tensor(1.0), ["img"])

    class Candidate:
        negative_sampling_policy = None

    assert resolve_negative_policy(Candidate(), contrastive_loss) == "data_parallel_local"
    Candidate.negative_sampling_policy = "global_batch"
    assert resolve_negative_policy(Candidate(), object()) == "global_batch"


def test_negative_policy_mismatch_is_not_one_objective_effect() -> None:
    from react_agent.eeg_research.agentic.comparison import compare_runs

    protocol = {"schema_version": "eeg_research.v1.9", "split_seed": 1, "validation_image_ids": ["a"], "fingerprint": "p"}
    result = compare_runs(
        candidate={"evidence_id": "c", "candidate_id": "c1", "evaluation_valid": True, "fidelity": "full", "seed": 1, "negative_sampling_policy": "global_batch", "fixed_bank_top1": 0.2},
        control={"evidence_id": "b", "candidate_id": "baseline", "evaluation_valid": True, "fidelity": "full", "seed": 1, "negative_sampling_policy": "data_parallel_local", "fixed_bank_top1": 0.1},
        protocol=protocol,
    )
    assert result["comparable"] is False
    assert result["reason"] == "negative_policy_mismatch"


def test_other_subjects_gallery_is_the_validation_images(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
    from react_agent.eeg_training.protocol import Design

    meg = tmp_path / "things-meg" / "Preprocessed_data"
    for name, images in (("sub-01", ["a", "b"]), ("sub-02", ["b", "c"])):
        folder = meg / name
        folder.mkdir(parents=True)
        (folder / "train.pt").write_text(json.dumps({"img": images}), encoding="utf-8")
        (folder / "test.pt").write_text(json.dumps({"img": ["c"]}), encoding="utf-8")
    design = Design(
        dataset="meg",
        exp_setting="inter-subject",
        subject="sub-01",
        epochs=2,
        seed=1,
        data_root=str(tmp_path),
        training_strategy="pooled_subjects",
        policy="agentic",
    )
    protocol = build_execution_protocol(design, tmp_path)
    assert protocol["gallery_image_ids"]
    assert set(protocol["positive_map"][qid] for qid in protocol["validation_query_ids"]) <= set(protocol["gallery_image_ids"])


def test_retrieval_and_representation_match_direct_numbers(tmp_path: Path) -> None:
    from react_agent.eeg_training.diagnostics import compute_job_diagnostics

    job = tmp_path / "job"
    job.mkdir()
    (job / "retrieval_queries.jsonl").write_text(
        json.dumps({"query_id": "q", "positive_rank": 2, "positive_score": 0.4, "best_negative_score": 0.1, "k": 1}) + "\n",
        encoding="utf-8",
    )
    (job / "embeddings.json").write_text(json.dumps({"vectors": [[1.0, 0.0], [0.0, 1.0]]}), encoding="utf-8")
    bundle = compute_job_diagnostics(job)
    assert bundle["retrieval_errors"]["status"] == "observed"
    assert bundle["retrieval_errors"]["payload"]["mean_margin"] == pytest.approx(0.3)
    assert bundle["representation"]["status"] == "observed"
    assert bundle["representation"]["payload"]["effective_rank"] == pytest.approx(1.0)
    assert "covariance" in bundle["representation"]["payload"]["effective_rank_definition"]
    assert bundle["group_results"]["status"] == "unavailable"


def test_analysis_uses_the_candidate_hypothesis_and_api_failure_is_not_not_tested(tmp_path: Path, monkeypatch) -> None:
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    from react_agent.eeg_research.agentic.worker import build_services

    camp = _camp(tmp_path, "t11")
    state = load_state(camp)
    state["hypothesis"] = {"mechanism": "当前 c2"}
    state["evidence"] = [
        {
            "evidence_id": "ev_1",
            "candidate_id": "c1",
            "fidelity": "full",
            "evaluation_valid": True,
            "fixed_bank_top1": 0.2,
            "hypothesis": {"mechanism": "历史 c1"},
            "comparison": {"comparable": True, "delta_pp": 0.4},
        }
    ]
    seen = {}

    def factory(_camp: Path, role: str):
        def call(payload: dict) -> dict:
            if role == "result_analyst":
                seen["hypothesis"] = payload.get("hypothesis")
                raise LlmUnavailable("DeepSeekParseError")
            return {"proposed_lessons": [], "summary_zh": "无"}

        call.model = "fake"
        return call

    monkeypatch.setattr("react_agent.eeg_research.agentic.worker.role_backend", factory)
    build_services(camp)["analyze"](camp, state)
    assert seen["hypothesis"]["mechanism"] == "历史 c1"
    analysis = [row for row in state["evidence"] if row.get("kind") == "analysis"][-1]
    assert analysis["summary"]["hypothesis_assessment"] is None
    assert state["evidence"][0]["comparison"]["comparable"] is True


def test_lessons_retrieve_compatible_and_analogy(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode, query_lessons

    store = EpisodeStore(tmp_path)
    saved = store.persist_episode(
        episode(task_hash="g", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=1, metric=0.2, contract_fingerprint="fp", artifact="a", job_id="j1")
    )
    again = store.persist_episode(
        episode(task_hash="g", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=1, metric=0.9, contract_fingerprint="fp", artifact="b", job_id="j1")
    )
    assert again["episode_id"] == saved["episode_id"]
    (tmp_path / "comparisons").mkdir()
    (tmp_path / "comparisons" / "cmp_1.json").write_text(json.dumps({"comparison": {"delta_pp": 1.0}}), encoding="utf-8")
    accepted = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "supporting_episode_ids": [saved["episode_id"]],
                    "conditions": {"task": "eeg", "evaluation_identity": "fp", "negative_sampling_policy": "data_parallel_local"},
                    "evidence_level": "exploratory_result",
                    "comparison_refs": ["cmp_1"],
                    "observed_effect": 1.0,
                }
            ]
        }
    )
    assert accepted["accepted"]
    found = query_lessons({"task": "eeg", "evaluation_identity": "fp", "negative_sampling_policy": "data_parallel_local"}, store.list_lessons())
    assert found["compatible_evidence"]
    other = query_lessons({"task": "eeg", "evaluation_identity": "other"}, store.list_lessons())
    assert other["analogy_only"]
    assert not other["compatible_evidence"]


def test_historical_candidate_training_is_not_granted_to_the_current_one() -> None:
    from react_agent.eeg_research.agentic.planner import available_actions, eligible_targets

    state = {
        "gpu_seconds_left": 10,
        "llm_calls_left": 40,
        "max_training_jobs": 4,
        "training_jobs": 0,
        "candidate_id": "c2",
        "candidate_ready": False,
        "candidates": [{"candidate_id": "c1", "status": "ready"}, {"candidate_id": "c2", "status": "implementation_failed"}],
        "evidence": [{"candidate_id": "c1", "fidelity": "pilot", "evaluation_valid": True}],
    }
    targets = eligible_targets(state)
    assert "c1" in targets["run_full"]
    assert "c2" not in targets["run_full"]
    assert "c2" not in targets["run_pilot"]
    assert "run_full" in available_actions(state)


def test_plan_update_failure_blocks_a_dependent_stop(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "plan-dep")
    state = tick(
        camp,
        lambda _obs: {
            "action": "stop",
            "reason_zh": "停",
            "evidence_ids": [],
            "stop_reason": "blocked",
            "action_depends_on_plan_update": True,
            "plan_update": {"notes_zh": "没有版本"},
        },
    )
    assert state["status"] != "finished"
    assert state["detail"] == "plan_update_blocks_action"


def test_parser_retry_cannot_pass_the_last_llm_call(tmp_path: Path, monkeypatch) -> None:
    from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
    from react_agent.fmri.llm.deepseek import DeepSeekParseError
    from react_agent.fmri.schemas import LMUsage

    camp = tmp_path / "budget"
    camp.mkdir()
    (camp / "goal.json").write_text(json.dumps({"max_llm_calls": 1}), encoding="utf-8")
    hits = {"n": 0}

    class Fake:
        def __init__(self, _config) -> None:
            return

        async def complete_json(self, **_kwargs):
            hits["n"] += 1
            raise DeepSeekParseError("x", LMUsage(provider="deepseek", requested_model="m", response_model="m", finish_reason="stop", input_tokens=1, output_tokens=1))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    backend = role_backend(camp, "result_analyst")
    with pytest.raises(LlmUnavailable):
        backend({"ping": True})
    assert hits["n"] == 1


def test_evaluate_pack_checkpoint_is_not_the_output_directory(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.export import evaluate_only_argv, pack_candidate

    job = tmp_path / "job"
    workspace = tmp_path / "ws" / "extension"
    workspace.mkdir(parents=True)
    (workspace / "eeg_candidate.py").write_text("class EEGCandidate:\n    pass\n", encoding="utf-8")
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate", "class_file": "/old/workspace/eeg_candidate.py"}), encoding="utf-8")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "pack"
    pack_candidate(job, dest, workspace=tmp_path / "ws")
    binding = json.loads((dest / "source_binding.json").read_text(encoding="utf-8"))
    assert binding["class_file"] == "extension/eeg_candidate.py"
    assert binding["workspace"] == "."
    argv = evaluate_only_argv(dest, out=tmp_path / "fresh")
    assert str(tmp_path / "fresh") in argv
    assert str(dest / "last.ckpt") in argv
    assert argv[argv.index("--out") + 1] != argv[argv.index("--checkpoint") + 1]


def test_second_artifact_version_does_not_change_the_first_hash(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "versions")
    first_task = begin_role_task(camp, role="research_planner", inputs=[camp / "goal.json"])
    first = finish_role_task(
        camp,
        first_task,
        {"role": "research_planner", "summary_zh": "第一版", "notes_zh": ["一"]},
        kind="plan",
        path=camp / "plans" / "first.json",
    )
    second_task = begin_role_task(camp, role="research_planner", inputs=[camp / "goal.json"])
    second = finish_role_task(
        camp,
        second_task,
        {"role": "research_planner", "summary_zh": "第二版", "notes_zh": ["二"]},
        kind="plan",
        path=camp / "plans" / "second.json",
    )
    first_id = first["artifact_refs"][-1]
    second_id = second["artifact_refs"][-1]
    first_hash = lookup(camp, first_id)["sha256"]
    assert verify(camp, first_id) == (True, "ok")
    (camp / "plans" / "second.json").write_bytes((camp / "plans" / "second.json").read_bytes() + b" ")
    with pytest.raises(FileNotFoundError):
        read_verified_range(camp, second_id)
    with pytest.raises(FileNotFoundError):
        read_verified_range(camp, "art_missing")
    assert lookup(camp, first_id)["sha256"] == first_hash
    assert verify(camp, first_id) == (True, "ok")


def test_designer_sees_diagnostics_and_draft_without_intervention_stays_unapproved(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.experiment_gate import approve_experiment
    from react_agent.eeg_research.agentic.loop import save_state

    camp = _camp(tmp_path, "design-context")
    state = load_state(camp)
    state["evidence"] = [
        {
            "evidence_id": "ev_diag",
            "candidate_id": "c1",
            "diagnostics": {"retrieval_errors": {"status": "observed"}},
            "diagnostic_ref": "jobs/j1/diagnostic_summary.json",
        }
    ]
    save_state(camp, state)
    seen: dict = {}

    def designer(payload: dict) -> dict:
        seen["diagnostics"] = payload["diagnostics"]
        return {
            "experiment_spec": {
                "initial_fidelity": "pilot",
                "intervention": "projection_scale",
                "hypothesis": "h1",
                "parent_candidate_id": "baseline",
            },
            "summary_zh": "看见诊断",
        }

    tick(
        camp,
        lambda _obs: {"action": "design_experiment", "reason_zh": "设计", "evidence_ids": ["ev_diag"]},
        services={"designer": designer},
    )
    assert seen["diagnostics"]["summary"]["retrieval_errors"]["status"] == "observed"
    blocked = approve_experiment(
        {"initial_fidelity": "pilot", "hypothesis": "h1", "parent_candidate_id": "baseline"}
    )
    assert blocked["status"] != "approved"
    assert blocked["blocked_reason"] == "intervention_missing"


def test_independent_inspect_still_runs_when_plan_update_fails(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "inspect-plan")
    state = tick(
        camp,
        lambda _obs: {
            "action": "inspect_data",
            "reason_zh": "先看数据",
            "evidence_ids": [],
            "action_depends_on_plan_update": False,
            "plan_update": {"notes_zh": "没有版本号"},
        },
    )
    assert "research_scope" in state["data_audit"]
    assert any(row.get("kind") == "data_audit" for row in state["evidence"])
    assert state["status"] == "planning"


def test_duplicate_pairs_count_once_and_a_null_bar_cannot_confirm() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    def row(seed: int, run: str) -> dict:
        return {
            "delta_pp": 2.0,
            "fidelity": "full",
            "seed": seed,
            "run_id": run,
            "control_run_id": "base",
            "checkpoint_id": f"ckpt-{run}",
            "source_hash": "src",
            "config_hash": "cfg",
            "approval_ref": "art_exp",
            "evaluation_valid": True,
        }

    records = [row(0, "a"), row(1, "b"), row(2, "c"), row(0, "a")]
    counted = promotion_decision(
        comparison={"comparable": True, "delta_pp": 2.0},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 3, "training_seeds": [0, 1, 2]},
        fidelity="full",
        paired_records=records,
    )
    assert counted["paired_n"] == 3
    assert counted["status"] == "confirmed"
    open_bar = promotion_decision(
        comparison={"comparable": True, "delta_pp": 2.0},
        goal={"min_practical_gain_pp": None, "confirmation_target_pairs": 3},
        fidelity="full",
        paired_records=records[:3],
    )
    assert open_bar["status"] != "confirmed"
    assert open_bar["tier"] != "confirmation"


def test_negative_result_can_pass_and_a_new_report_retires_the_old_audit(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import save_state

    camp = _camp(tmp_path, "negative-audit")
    first = tick(
        camp,
        lambda _obs: {"action": "audit_result", "reason_zh": "先审计空结果", "evidence_ids": []},
        services={"auditor": lambda _payload: {"verdict": "PASS", "summary_zh": "模型想通过"}},
    )
    first_hash = first["audit_report_hash"]
    first_file = next((camp / "audits").glob("*.json"))
    first_bytes = first_file.read_bytes()
    state = load_state(camp)
    state["evidence"].append(
        {
            "evidence_id": "ev_neg",
            "kind": "job",
            "candidate_id": "c1",
            "fidelity": "full",
            "delta_pp": -1.5,
            "comparison": {"comparable": True, "delta_pp": -1.5},
            "diagnostics": {"retrieval_errors": {"status": "observed"}},
            "diagnostic_ref": "jobs/j1/diagnostic_summary.json",
        }
    )
    save_state(camp, state)
    second = tick(
        camp,
        lambda _obs: {"action": "audit_result", "reason_zh": "审计负结果", "evidence_ids": ["ev_neg"]},
        services={"auditor": lambda _payload: {"verdict": "PASS", "summary_zh": "负结果成立"}},
    )
    assert second["audit_status"] == "pass"
    assert second["audit_report_hash"] != first_hash
    assert second.get("research_outcome") != "improved_provisional"
    assert first_file.read_bytes() == first_bytes
    latest_file = next(
        path
        for path in (camp / "audits").glob("*.json")
        if second["audit_report_hash"] in path.read_text(encoding="utf-8")
    )
    latest = json.loads(latest_file.read_text(encoding="utf-8"))
    body = latest.get("payload") or latest
    assert body["verdict"] == "PASS"
    assert body["report_hash"] == second["audit_report_hash"]


def test_self_declared_checkpoint_optional_and_query_mismatch_stay_invalid(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.runner import accept_job

    job = tmp_path / "job"
    job.mkdir()
    (job / "metrics.json").write_text(
        json.dumps({"fixed_bank_top1": 0.2, "validation_image_count": 1, "query_count": 1, "checkpoint_optional": True}),
        encoding="utf-8",
    )
    protocol = {
        "validation_image_ids": ["img-a"],
        "validation_query_ids": ["q1", "q2"],
        "fingerprint": "fp",
    }
    missing = accept_job(job, {"module": "m", "entry_sha256": "abc"}, "full", protocol=protocol)
    assert missing["evaluation_valid"] is False
    assert missing["reason"] == "checkpoint_missing"
    (job / "last.ckpt").write_bytes(b"ckpt")
    mismatched = accept_job(job, {"module": "m", "entry_sha256": "abc"}, "full", protocol=protocol)
    assert mismatched["evaluation_valid"] is False
    assert mismatched["reason"] == "query_count_mismatch"


def test_unknown_model_config_and_non_scalar_loss_are_rejected() -> None:
    import torch

    from react_agent.eeg_research.agentic.baseline import EEGCandidate
    from react_agent.eeg_training.hooks import scalar_loss

    with pytest.raises(ValueError, match="unknown_model_config"):
        EEGCandidate().build_encoder({"c_num": 17, "timesteps": [0, 250]}, {"extra_width": 8})
    with pytest.raises(RuntimeError, match="loss_not_scalar"):
        scalar_loss(torch.zeros(2))


def test_same_job_is_charged_once_and_a_live_child_is_not_marked_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from react_agent.eeg_research.agentic.budget import charge_gpu
    from react_agent.eeg_research.agentic.jobs import stop_job

    camp = _camp(tmp_path, "budget-job")
    state = load_state(camp)
    assert charge_gpu(camp, state, 3, job_id="j1") is True
    assert charge_gpu(camp, state, 9, job_id="j1") is False
    used = json.loads((camp / "cost.json").read_text(encoding="utf-8"))["gpu_seconds_used"]
    assert used == 3
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    job = tmp_path / "live-job"
    job.mkdir()
    from react_agent.eeg_research.agentic.jobs import _proc_start

    record = {
        "job_id": "j-live",
        "pid": proc.pid,
        "proc_start": _proc_start(proc.pid),
        "status": "running",
        "started_at": 1,
        "gpu": [0],
    }
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.os.killpg", lambda *_args, **_kwargs: None)
    try:
        stopped = stop_job(job)
        assert stopped["status"] == "running"
        assert stopped["cancel_requested"] is True
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_stop_keeps_execution_research_and_audit_apart(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "status-split")
    state = tick(camp, lambda _obs: {"action": "stop", "reason_zh": "先停下", "evidence_ids": [], "stop_reason": "blocked"})
    assert state["status"] == "finished"
    assert state["execution_status"] == "completed"
    assert state["research_outcome"] == "not_evaluated"
    assert state["audit_status"] == "pending"
    assert state["research_outcome"] != "improved_confirmed"


def test_stub_trajectory_consumes_the_pilot_evidence(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.contract import freeze_contract
    from react_agent.eeg_training.protocol import Design

    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    design = Design("eeg", "inter-subject", "all", train_dir=str(tmp_path / "train"), test_dir=str(tmp_path / "test"), policy="agentic")
    create_campaign(
        tmp_path,
        goal=_goal(max_training_jobs=4, max_gpu_seconds=100),
        contract=freeze_contract(design, tmp_path),
        request_id="t16",
    )
    camp = tmp_path / "goal"
    candidate = "from react_agent.eeg_research.agentic.baseline import EEGCandidate as Baseline\n\nclass EEGCandidate(Baseline):\n    candidate_id = 'c1'\n"
    phase = {"n": 0}
    consumed: dict[str, str] = {}

    def backend(obs: dict) -> dict:
        phase["n"] += 1
        if phase["n"] == 1:
            return {
                "action": "design_experiment",
                "reason_zh": "提出改动",
                "evidence_ids": [],
                "experiment_draft": {
                    "initial_fidelity": "pilot",
                    "intervention": "projection_scale",
                    "hypothesis": "投影尺度",
                    "parent_candidate_id": "baseline",
                },
            }
        if phase["n"] == 2:
            return {
                "action": "implement_candidate",
                "reason_zh": "写入候选",
                "evidence_ids": [],
                "relative": "extension/eeg_candidate.py",
                "content": candidate,
                "expected_base_hash": "",
            }
        if phase["n"] == 3:
            return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": [], "target_id": "c1"}
        pilot_ids = [row["evidence_id"] for row in obs["evidence"] if row.get("fidelity") == "pilot"]
        assert pilot_ids
        consumed["evidence_id"] = pilot_ids[-1]
        assert "diagnose_results" in obs["available_actions"]
        return {"action": "diagnose_results", "reason_zh": "看试跑", "evidence_ids": [pilot_ids[-1]]}

    def runner(job: Path, manifest: dict, fidelity: str) -> dict:
        (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.02, "validation_image_count": 10}), encoding="utf-8")
        (job / "source_binding.json").write_text(
            json.dumps({"module": manifest["module"], "file_sha256": manifest["entry_sha256"]}),
            encoding="utf-8",
        )
        return {"gpu_seconds": 1, "seed": 0}

    def designer(payload: dict) -> dict:
        return {"experiment_spec": payload["draft"], "summary_zh": "批准试跑"}

    states = [tick(camp, backend, runner, services={"designer": designer}) for _ in range(4)]
    assert states[0].get("experiment", {}).get("status") == "approved"
    assert states[1]["candidate_ready"] is True
    assert states[2]["evidence"][-1]["fidelity"] == "pilot"
    decision = states[3]["decisions"][-1]
    assert decision["action"] == "diagnose_results"
    assert consumed["evidence_id"] in decision["evidence_ids"]
    assert any(row.get("evidence_id") == consumed["evidence_id"] and row.get("fidelity") == "pilot" for row in states[3]["evidence"])
