"""Consumer-chain checks for the systematic revision. Temporary fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.experiment_gate import approve_experiment, experiment_is_approved
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state, tick


def _goal(**kwargs) -> dict:
    fields = {"goal_id": "goal", "max_training_jobs": 8, "max_llm_calls": 40, "max_gpu_seconds": 100, "max_candidates": 4}
    fields.update(kwargs)
    return fields


def _camp(tmp_path: Path, request_id: str = "sys") -> Path:
    create_campaign(tmp_path, goal=_goal(), contract={"fingerprint": "fp"}, request_id=request_id)
    return tmp_path / "goal"


def test_t01_blocked_banana_and_unknown_capability_are_not_approved() -> None:
    blocked = approve_experiment(
        {
            "status": "blocked",
            "intervention": "projection_scale",
            "hypothesis": "h",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "pilot",
            "required_capability_ids": ["banana_hook"],
        }
    )
    banana = approve_experiment(
        {
            "intervention": "projection_scale",
            "hypothesis": "h",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "banana",
        }
    )
    unknown = approve_experiment(
        {
            "intervention": "projection_scale",
            "hypothesis": "h",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "pilot",
            "required_capability_ids": ["telepathy_loss"],
        }
    )
    assert experiment_is_approved(blocked) is False
    assert experiment_is_approved(banana) is False
    assert experiment_is_approved(unknown) is False
    assert banana["blocked_reason"] == "fidelity_invalid"
    assert unknown["blocked_reason"] == "unknown_capability"


def test_t02_approved_spec_hash_is_consumed_by_implement_gate(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t02")
    approved = approve_experiment(
        {
            "intervention": "projection_scale",
            "hypothesis": "dropout",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "pilot",
        }
    )
    state = load_state(camp)
    state["experiment"] = approved
    save_state(camp, state)
    seen = {}

    def backend(obs):
        seen["experiment"] = obs.get("experiment")
        return {"action": "stop", "reason_zh": "停", "evidence_ids": [], "stop_reason": "blocked"}

    tick(camp, backend)
    assert seen["experiment"]["spec_hash"] == approved["spec_hash"]
    assert experiment_is_approved(seen["experiment"]) is True


def test_t04_designer_digest_changes_when_diagnostics_change(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.roles import begin_role_task

    camp = _camp(tmp_path, "t04")
    first = begin_role_task(
        camp,
        role="experiment_designer",
        inputs=[camp / "goal.json"],
        request={"draft": {"intervention": "a"}, "diagnostics": {"mean_margin": 0.1}},
    )
    second = begin_role_task(
        camp,
        role="experiment_designer",
        inputs=[camp / "goal.json"],
        request={"draft": {"intervention": "a"}, "diagnostics": {"mean_margin": 0.9}},
    )
    assert first["input_digest"] != second["input_digest"]


def test_t07_approved_hook_config_reaches_job_and_trainer_reader(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.hook_config import write_hook_config
    from react_agent.eeg_training.train_entry import _hook_section

    spec = approve_experiment(
        {
            "intervention": "projection_scale",
            "hypothesis": "h",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "pilot",
            "model_config": {"drop_proj": 0.1},
            "objective_config": {"sentinel": 3},
            "transform_config": {"noise": 0.0},
        }
    )
    job = tmp_path / "job"
    written = write_hook_config(job, spec, spec_ref="art_spec")
    assert _hook_section(job, "model")["drop_proj"] == 0.1
    assert _hook_section(job, "objective")["sentinel"] == 3
    assert _hook_section(job, "transform")["noise"] == 0.0
    assert written["config_hash"]
    assert json.loads((job / "hook_config.json").read_text(encoding="utf-8"))["config_hash"] == written["config_hash"]


def test_t13_and_t15_retrieval_and_rank_one_spectrum() -> None:
    from react_agent.eeg_training.diagnostics import representation_from_vectors, retrieval_from_rows, retrieval_rows_from_scores

    queries = [("q1", [1.0, 0.0]), ("q2", [0.2, 1.0])]
    bank = [("pos", [1.0, 0.0]), ("neg", [0.0, 1.0])]
    positives = {"q1": {"pos"}, "q2": {"pos"}}
    rows = retrieval_rows_from_scores(queries, bank, positives, k=1)
    bundle = retrieval_from_rows(rows)
    assert bundle["status"] == "observed"
    assert bundle["payload"]["rows"][0]["positive_rank"] == 1
    assert bundle["payload"]["mean_margin"] == pytest.approx((rows[0]["margin"] + rows[1]["margin"]) / 2)
    rank_one = representation_from_vectors([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [3.0, 6.0, 9.0, 12.0]])
    assert rank_one["status"] == "observed"
    assert rank_one["payload"]["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    constant = representation_from_vectors([[1.0, 1.0], [1.0, 1.0]])
    assert constant["status"] == "undefined"
    assert constant["payload"]["effective_rank"] is None


def test_t16_summary_keeps_scientific_numbers(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.diagnostics import write_job_bundle

    job = tmp_path / "job"
    job.mkdir()
    (job / "retrieval_queries.jsonl").write_text(
        json.dumps({"query_id": "q", "positive_rank": 2, "positive_score": 0.4, "best_negative_score": 0.1, "k": 1}) + "\n",
        encoding="utf-8",
    )
    (job / "embeddings.json").write_text(json.dumps({"vectors": [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]}), encoding="utf-8")
    summary = write_job_bundle(job)
    assert summary["items"]["retrieval_errors"]["mean_margin"] == pytest.approx(0.3)
    assert summary["items"]["retrieval_errors"]["sample_count"] == 1
    assert summary["items"]["representation"]["effective_rank"] == pytest.approx(1.0, abs=1e-6)


def test_t18_t19_record_job_builds_pair_records_and_dedupes_seed(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import _record_job

    camp = _camp(tmp_path, "t18")
    (camp / "execution_protocol.json").write_text(
        json.dumps(
            {
                "fingerprint": "fp",
                "split_seed": 1,
                "validation_image_ids": ["a"],
                "gallery_image_ids": ["a"],
                "schema_version": "eeg_research.v1.9",
                "validation_query_ids": ["q"],
            }
        ),
        encoding="utf-8",
    )
    goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
    goal["min_practical_gain_pp"] = 1.0
    goal["confirmation_target_pairs"] = 3
    (camp / "goal.json").write_text(json.dumps(goal), encoding="utf-8")
    state = load_state(camp)
    state["contract_fingerprint"] = "fp"
    state["candidate_id"] = "c1"
    state["experiment"] = approve_experiment(
        {"intervention": "projection_scale", "hypothesis": "h", "parent_candidate_id": "baseline", "initial_fidelity": "full"}
    )

    def settle(job_id: str, candidate: str, seed: int, score: float) -> None:
        job = camp / "jobs" / job_id
        job.mkdir(parents=True, exist_ok=True)
        (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": score}), encoding="utf-8")
        _record_job(
            camp,
            state,
            {
                "job_id": job_id,
                "candidate_id": candidate,
                "seed": seed,
                "status": "finished",
                "gpu_seconds": 1,
                "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": score, "execution_fingerprint": "fp"},
            },
        )

    settle("b0", "baseline", 0, 0.006)
    settle("c0", "c1", 0, 0.02)
    settle("c0b", "c1", 0, 0.03)
    settle("b1", "baseline", 1, 0.006)
    settle("c1", "c1", 1, 0.02)
    settle("b2", "baseline", 2, 0.006)
    settle("c2", "c1", 2, 0.02)
    replay = len(state["evidence"])
    settle("c2", "c1", 2, 0.02)
    assert len(state["evidence"]) == replay
    last = next(row for row in reversed(state["evidence"]) if row.get("candidate_id") == "c1" and row.get("seed") == 2)
    assert last["pair_record"]["run_id"]
    assert last["promotion"]["paired_n"] == 3
    assert last["promotion"]["status"] == "confirmed"


def test_t24_empty_or_fake_lesson_cannot_confirm(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore

    store = EpisodeStore(tmp_path)
    rejected = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "supporting_episode_ids": [],
                    "conditions": {"task": "eeg"},
                    "comparison_refs": ["missing"],
                    "observed_effect": 999,
                    "evidence_level": "confirmed_result",
                }
            ]
        }
    )
    assert rejected["accepted"] == []
    reasons = rejected["rejected"][0]["reasons"]
    assert "supporting_missing" in reasons
    assert "forged_effect" not in reasons or "comparison_ref_missing" in reasons or "confirmation_missing" in reasons


def test_t28_new_result_marks_current_audit_stale(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import _record_job

    camp = _camp(tmp_path, "t28")
    state = load_state(camp)
    state["audit_status"] = "pass"
    state["audit_report_hash"] = "old"
    state["audit_fresh"] = True
    job = camp / "jobs" / "jnew"
    job.mkdir(parents=True)
    _record_job(
        camp,
        state,
        {
            "job_id": "jnew",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 1,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.1},
        },
    )
    assert state["audit_status"] == "stale"
    assert state["audit_fresh"] is False


def test_t08_build_hook_rejects_unknown_keys_without_merging_geometry() -> None:
    from react_agent.eeg_training.hooks import HookConfigError
    from react_agent.eeg_training.train_entry import _build_hook

    class Candidate:
        def build_encoder(self, geometry, config=None):
            assert "c_num" not in (config or {})
            return {"geometry": geometry, "config": config}

        build_encoder.accepted_config_keys = {"drop_proj"}

    built = _build_hook(Candidate(), "build_encoder", {"c_num": 17, "timesteps": [0, 250]}, {"drop_proj": 0.2})
    assert built["config"]["drop_proj"] == 0.2
    with pytest.raises(HookConfigError, match="unknown"):
        _build_hook(Candidate(), "build_encoder", {"c_num": 17, "timesteps": [0, 250]}, {"drop_proj": 0.2, "extra_width": 8})


def test_t31_pack_copies_hook_config_and_keeps_checkpoint_off_fresh_out(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.export import evaluate_only_argv, pack_candidate

    job = tmp_path / "job"
    workspace = tmp_path / "ws" / "extension"
    workspace.mkdir(parents=True)
    (workspace / "eeg_candidate.py").write_text("class EEGCandidate:\n    pass\n", encoding="utf-8")
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "hook_config.json").write_text(json.dumps({"model": {"drop_proj": 0.1}, "config_hash": "abc"}), encoding="utf-8")
    (job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate", "class_file": "/old/ws/eeg_candidate.py"}), encoding="utf-8")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "pack"
    pack_candidate(job, dest, workspace=tmp_path / "ws")
    assert (dest / "hook_config.json").is_file()
    argv = evaluate_only_argv(dest, out=tmp_path / "fresh")
    assert str(dest / "last.ckpt") in argv
    assert argv[argv.index("--checkpoint") + 1] != str(tmp_path / "fresh")


def test_t33_independent_inspect_still_runs(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t33")
    state = tick(
        camp,
        lambda _obs: {
            "action": "inspect_data",
            "reason_zh": "看数据",
            "evidence_ids": [],
            "action_depends_on_plan_update": False,
            "plan_update": {"notes_zh": "缺版本"},
        },
    )
    assert any(row.get("kind") == "data_audit" for row in state["evidence"])
