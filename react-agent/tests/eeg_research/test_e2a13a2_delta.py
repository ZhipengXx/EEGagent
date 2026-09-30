"""Acceptance checks for the e2a13a2 incremental revision. Temporary fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.experiment_gate import (
    ExperimentResolutionError,
    approve_experiment,
    experiment_is_approved,
    resolve_approved_experiment,
)
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state, tick


def _goal(**kwargs) -> dict:
    fields = {"goal_id": "goal", "max_training_jobs": 8, "max_llm_calls": 40, "max_gpu_seconds": 100, "max_candidates": 4}
    fields.update(kwargs)
    return fields


def _camp(tmp_path: Path, request_id: str = "delta") -> Path:
    create_campaign(tmp_path, goal=_goal(), contract={"fingerprint": "fp"}, request_id=request_id)
    return tmp_path / "goal"


def _draft(**kwargs) -> dict:
    spec = {
        "intervention": "projection_scale",
        "hypothesis": {"mechanism": "投影缩放"},
        "parent_candidate_id": "baseline",
        "initial_fidelity": "pilot",
    }
    spec.update(kwargs)
    return spec


def test_d01_disabled_known_hook_cannot_be_approved() -> None:
    context = {
        "capabilities": {
            "hooks": {
                "build_encoder": {"status": "available"},
                "fit_statistics": {"status": "available"},
                "build_training_transform": {"status": "disabled"},
                "build_training_objective": {"status": "available"},
                "evaluate_only": {"status": "available"},
            }
        }
    }
    blocked = approve_experiment(_draft(required_capability_ids=["build_training_transform"]), context=context)
    assert blocked.get("approval_record") is not True
    assert experiment_is_approved(blocked, context=context) is False
    assert blocked["blocked_reason"] == "unknown_capability"


def test_d02_self_declared_approved_cannot_implement(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "d02")
    state = load_state(camp)
    state["experiment"] = {
        "status": "approved",
        "intervention": "projection_scale",
        "hypothesis": {"mechanism": "自报批准"},
        "parent_candidate_id": "baseline",
        "initial_fidelity": "pilot",
    }
    state["decisions"] = [{"decision_id": "d1", "action": "implement_candidate", "ok": True, "executed": False, "reason_zh": "写"}]
    save_state(camp, state)
    from react_agent.eeg_research.agentic.planner import available_actions

    assert "implement_candidate" not in available_actions(state)
    seen = tick(camp, lambda _obs: {"action": "stop", "reason_zh": "停", "evidence_ids": [], "stop_reason": "blocked"})
    assert seen.get("experiment_failed") is True
    assert seen.get("failure", {}).get("error_type") == "experiment_not_approved"
    assert experiment_is_approved(seen.get("experiment")) is False


def test_d03_baseline_does_not_inherit_current_experiment(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.run_context import resolve_run_context

    camp = _camp(tmp_path, "d03")
    state = load_state(camp)
    state["candidate_id"] = "c1"
    state["experiment"] = approve_experiment(
        _draft(model={"drop_proj": 0.1}, transform={"input_noise_std": 0.05}, initial_fidelity="full")
    )
    ctx = resolve_run_context(camp, "baseline", state, fidelity="full", seed=0)
    assert ctx["hook_spec"]["model"] == {}
    assert ctx["hook_spec"]["transform"] == {}
    assert ctx["hook_spec"]["objective"] == {}


def test_d04_undeclared_seeds_cannot_confirm() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    records = [
        {
            "delta_pp": 2.0,
            "fidelity": "full",
            "seed": seed,
            "run_id": f"run_{seed}",
            "control_run_id": "base",
            "checkpoint_id": f"ckpt_{seed}",
            "source_hash": "src",
            "config_hash": "cfg",
            "approval_ref": "art",
            "evaluation_valid": True,
        }
        for seed in (0, 1, 2)
    ]
    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 2.0},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 3, "training_seeds": [99, 100, 101]},
        paired_records=records,
        fidelity="full",
    )
    assert promo["status"] != "confirmed"
    assert promo["paired_n"] == 0


def test_d05_partial_settlement_writes_missing_diagnostics(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import _record_job

    camp = _camp(tmp_path, "d05")
    state = load_state(camp)
    job = camp / "jobs" / "jpartial"
    job.mkdir(parents=True)
    (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.2}), encoding="utf-8")
    state["evidence"] = [
        {
            "evidence_id": "ev_jpartial",
            "job_id": "jpartial",
            "candidate_id": "c1",
            "evaluation_valid": True,
            "fidelity": "full",
            "job_dir": str(job),
        }
    ]
    _record_job(
        camp,
        state,
        {
            "job_id": "jpartial",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 0,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.2},
        },
    )
    assert (job / "diagnostic_summary.json").is_file()
    assert (job / "settlement.json").is_file()
    assert json.loads((job / "settlement.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert sum(1 for row in state["evidence"] if row.get("job_id") == "jpartial") == 1


def test_d06_empty_comparison_file_is_rejected(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode

    store = EpisodeStore(tmp_path)
    stored = store.persist_episode(episode(task_hash="t", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=0, metric=0.1, contract_fingerprint="fp", artifact="a", job_id="j1"))
    (tmp_path / "comparisons").mkdir()
    (tmp_path / "comparisons" / "cmp_empty.json").write_text("{}", encoding="utf-8")
    rejected = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "statement": "空对照不能当证据",
                    "uncertainty": "high",
                    "supporting_episode_ids": [stored["episode_id"]],
                    "conditions": {"task": "eeg"},
                    "comparison_refs": ["cmp_empty"],
                    "observed_effect": 1.0,
                    "evidence_level": "exploratory_result",
                }
            ]
        }
    )
    assert rejected["accepted"] == []
    assert rejected["rejected"]


def test_d06_lesson_keeps_statement_and_uncertainty(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode

    store = EpisodeStore(tmp_path)
    stored = store.persist_episode(episode(task_hash="t", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=0, metric=0.1, contract_fingerprint="fp", artifact="a", job_id="j1"))
    accepted = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "statement": "池化改善检索",
                    "uncertainty": "medium",
                    "supporting_episode_ids": [stored["episode_id"]],
                    "conditions": {"task": "eeg", "split": "inter-subject"},
                    "evidence_level": "exploratory_result",
                }
            ]
        }
    )
    lesson = accepted["accepted"][0]
    assert lesson["statement"] == "池化改善检索"
    assert lesson["uncertainty"] == "medium"
    assert lesson["requested_evidence_level"] == "exploratory_result"


def test_d07_new_incomparable_row_stales_audit(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import refresh_audit_freshness
    from react_agent.eeg_research.agentic.view import campaign_view

    camp = _camp(tmp_path, "d07")
    first = tick(
        camp,
        lambda _obs: {
            "action": "audit_result",
            "reason_zh": "审",
            "evidence_ids": [],
        },
    )
    assert first.get("audit_status") in {"pass", "revise", "block"}
    hashed = first.get("audited_report_hash")
    assert hashed
    first["evidence"] = list(first.get("evidence") or []) + [
        {
            "evidence_id": "ev_incomp",
            "job_id": "jincomp",
            "candidate_id": "c9",
            "evaluation_valid": True,
            "fidelity": "full",
            "comparison_status": "unmatched_control",
        }
    ]
    save_state(camp, first)
    refresh_audit_freshness(first)
    assert first["audit_status"] == "stale"
    view = campaign_view(camp)
    assert view["audit_status"] == "stale"


def test_d08_job_command_rewrite_adds_pack_checkpoint(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.export import evaluate_only_argv, pack_candidate

    job = tmp_path / "job"
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    (job / "job.json").write_text(
        json.dumps(
            {
                "command": ["python", "-m", "react_agent.eeg_training.train_entry", "--out", "/old", "--test-only"],
                "epochs": 1,
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "pack"
    pack_candidate(job, dest)
    argv = evaluate_only_argv(dest, out=tmp_path / "fresh", data_root=tmp_path)
    assert "--checkpoint" in argv
    assert argv[argv.index("--checkpoint") + 1] == str(dest / "last.ckpt")
    assert "--evaluate-only" in argv
    assert "--test-only" not in argv


def test_d09_d10_cpu_toy_evaluator_and_unique_params(tmp_path: Path) -> None:
    import torch

    from react_agent.eeg_training.diagnostics import compute_job_diagnostics, write_validation_artifacts
    from react_agent.eeg_training.train_entry import unique_trainable_parameters

    param = torch.nn.Parameter(torch.ones(3), requires_grad=True)
    unique = unique_trainable_parameters([param, param])
    assert len(unique) == 1
    queries = [("img-a", [1.0, 0.0]), ("img-b", [0.0, 1.0])]
    bank = [("img-a", [1.0, 0.0]), ("img-b", [0.0, 1.0])]
    written = write_validation_artifacts(tmp_path, queries, bank, {"img-a": {"img-a"}, "img-b": {"img-b"}})
    assert written["query_rows"] == 2
    bundle = compute_job_diagnostics(tmp_path)
    assert bundle["retrieval_errors"]["status"] == "observed"
    assert bundle["retrieval_errors"]["payload"]["sample_count"] == 2
    assert bundle["representation"]["status"] == "observed"
    assert (tmp_path / "retrieval_queries.jsonl").is_file()
    assert (tmp_path / "embeddings.json").is_file()


def test_d11_state_mutation_does_not_override_artifact(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.artifacts import register

    camp = _camp(tmp_path, "d11")
    approved = approve_experiment(_draft(model={}))
    path = camp / "experiments" / "spec.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"experiment_spec": approved}, ensure_ascii=False), encoding="utf-8")
    row = register(camp, path, kind="experiment_spec")
    mutated = dict(approved)
    mutated["model"] = {"drop_proj": 0.1}
    state = {"experiment": mutated, "experiment_ref": row["artifact_id"]}
    with pytest.raises(ExperimentResolutionError, match="approved_spec_changed"):
        resolve_approved_experiment(camp, spec_ref=row["artifact_id"], expected_hash=approved["spec_hash"], state=state)
    resolved = resolve_approved_experiment(camp, spec_ref=row["artifact_id"], expected_hash=approved["spec_hash"], state={"experiment": approved})
    assert resolved.get("model") in ({}, None) or resolved.get("model") == {}


def test_d12_designer_digest_includes_budget(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.roles import begin_role_task

    camp = _camp(tmp_path, "d12")
    first = begin_role_task(
        camp,
        role="experiment_designer",
        inputs=[camp / "goal.json"],
        request={"draft": {"intervention": "a"}, "diagnostics": {"mean_margin": 0.1}, "budget": {"gpu_seconds_left": 100}},
    )
    second = begin_role_task(
        camp,
        role="experiment_designer",
        inputs=[camp / "goal.json"],
        request={"draft": {"intervention": "a"}, "diagnostics": {"mean_margin": 0.1}, "budget": {"gpu_seconds_left": 1}},
    )
    assert first["input_digest"] != second["input_digest"]


def test_cited_evidence_refs_fill_plan_update_and_design_runs(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.research_plan import load_plan

    camp = _camp(tmp_path, "plan-alias")
    tick(camp, lambda _obs: {"action": "inspect_data", "reason_zh": "看数据", "evidence_ids": []})
    state = tick(
        camp,
        lambda _obs: {
            "action": "design_experiment",
            "reason_zh": "设计基线",
            "evidence_refs": ["ev_audit_1"],
            "question_id": "q1",
            "action_depends_on_plan_update": True,
            "hypothesis_draft": {"mechanism": "基线参考"},
            "experiment_draft": {
                "intervention": "baseline_reference",
                "parent_candidate_id": "baseline",
                "initial_fidelity": "pilot",
                "hypothesis": {"mechanism": "基线参考"},
            },
            "plan_update": {
                "based_on_plan_version": 1,
                "updated_questions": [
                    {
                        "question_id": "q1",
                        "status": "open",
                        "note_zh": "先建立固定画廊基线",
                        "evidence_refs": [],
                    }
                ],
            },
        },
    )
    assert state.get("detail") != "plan_update_blocks_action"
    assert state.get("failure", {}).get("error_type") != "plan_update_rejected"
    assert load_plan(camp)["plan_version"] == 2
    assert load_plan(camp)["accepted_evidence_ids"] == ["ev_audit_1"]
    assert isinstance(state.get("experiment"), dict)


def test_plan_update_without_cited_evidence_still_blocks(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.research_plan import PlanError, apply_update, init_plan, normalize_plan_update

    camp = tmp_path / "empty-plan"
    camp.mkdir()
    init_plan(camp, {"goal_id": "g", "objective": "改进检索"})
    empty = normalize_plan_update(
        {"based_on_plan_version": 1, "affected_question_ids": ["q1"]},
        {},
        known_evidence_ids={"ev_1"},
    )
    with pytest.raises(PlanError, match="plan_update_needs_evidence"):
        apply_update(camp, empty, known_evidence_ids={"ev_1"})
    unknown = normalize_plan_update(
        {"based_on_plan_version": 1, "affected_question_ids": ["q1"]},
        {"evidence_refs": ["ev_fake"]},
        known_evidence_ids={"ev_1"},
    )
    with pytest.raises(PlanError, match="plan_update_unknown_evidence"):
        apply_update(camp, unknown, known_evidence_ids={"ev_1"})


def test_already_started_candidates_do_not_bypass_approval(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "started")
    (camp / "candidates" / "c1").mkdir(parents=True)
    (camp / "candidates" / "c1" / "note.txt").write_text("started", encoding="utf-8")
    state = load_state(camp)
    state["experiment"] = _draft(status="approved")
    state["decisions"] = [{"decision_id": "d1", "action": "implement_candidate", "ok": True, "executed": False, "reason_zh": "写"}]
    save_state(camp, state)
    seen = tick(camp, lambda _obs: {"action": "stop", "reason_zh": "停", "evidence_ids": [], "stop_reason": "blocked"})
    assert seen.get("experiment_failed") is True
    assert seen.get("failure", {}).get("error_type") == "experiment_not_approved"
