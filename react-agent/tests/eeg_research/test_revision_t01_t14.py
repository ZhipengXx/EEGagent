"""Consumer regressions T01–T14. Helpers alone do not count as a passing chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.experiment_gate import (
    CanonicalizationError,
    ExperimentResolutionError,
    approve_experiment,
    canonicalize_experiment_spec,
    experiment_is_approved,
    install_approved_experiment,
    resolve_approved_experiment,
)
from react_agent.eeg_research.agentic.hook_config import HookConfigError, normalize_hook_config, write_hook_config
from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state, tick, _record_job, rebuild_campaign_projection
from react_agent.eeg_research.agentic.llm import LlmUnavailable
from react_agent.eeg_research.agentic.run_context import persist_approved_binding, resolve_run_context


def _goal(**kwargs) -> dict:
    fields = {"goal_id": "goal", "max_training_jobs": 8, "max_llm_calls": 40, "max_gpu_seconds": 100, "max_candidates": 4}
    fields.update(kwargs)
    return fields


def _camp(tmp_path: Path, request_id: str = "rev") -> Path:
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


def test_t01_alias_configs_canonicalize_and_conflict_rejects(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t01")
    job = camp / "jobs" / "jalias"
    migrated = canonicalize_experiment_spec(
        _draft(model_config={"drop_proj": 0.3}, objective_config={"temperature": 0.07}, transform_config={"input_noise_std": 0.1})
    )
    assert migrated["model"]["drop_proj"] == 0.3
    assert migrated["objective"]["temperature"] == 0.07
    assert migrated["transform"]["input_noise_std"] == 0.1
    approved = approve_experiment(migrated, camp=camp)
    assert approved.get("spec_hash")
    hook = write_hook_config(job, approved, spec_ref=approved.get("experiment_ref"), spec_hash=approved.get("spec_hash"))
    assert hook["spec_hash"] == approved["spec_hash"]
    assert hook["model"]["drop_proj"] == 0.3
    assert hook["intervention_config_hash"]
    assert hook["config_hash"] == hook["intervention_config_hash"]
    with pytest.raises((CanonicalizationError, HookConfigError)):
        normalize_hook_config(_draft(model={"drop_proj": 0.1}, model_config={"drop_proj": 0.9}))


def test_t02_cached_boolean_and_fake_hash_rejected_by_implement(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t02")
    state = load_state(camp)
    state["experiment"] = {
        "status": "approved",
        "approval_record": True,
        "spec_hash": "deadbeef",
        "intervention": "projection_scale",
        "hypothesis": {"mechanism": "假批准"},
        "parent_candidate_id": "baseline",
        "initial_fidelity": "pilot",
    }
    state["decisions"] = [{"decision_id": "d1", "action": "implement_candidate", "ok": True, "executed": False, "reason_zh": "写"}]
    save_state(camp, state)
    seen = tick(camp, lambda _obs: {"action": "stop", "reason_zh": "停", "evidence_ids": [], "stop_reason": "blocked"})
    assert experiment_is_approved(seen.get("experiment")) is False
    assert seen.get("failure", {}).get("error_type") in {"experiment_not_approved", "approved_spec_missing"}
    with pytest.raises(ExperimentResolutionError):
        resolve_approved_experiment(camp, spec_ref=None, expected_hash="deadbeef", state=state, action="implement")


def test_t03_repair_draft_and_cross_target_rejected(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t03")
    state = load_state(camp)
    approved = install_approved_experiment(camp, state, _draft(model={"drop_proj": 0.2}), target_id="c1")
    persist_approved_binding(camp, "c1", approved, spec_ref=approved.get("experiment_ref"))
    state["experiment"] = _draft(status="draft", intervention="other_mechanism", hypothesis={"mechanism": "换机制"})
    state["repair_task"] = {"candidate_id": "c1", "remaining": 1, "attempt_id": "a1"}
    save_state(camp, state)
    with pytest.raises(ExperimentResolutionError):
        resolve_approved_experiment(
            camp,
            spec_ref=approved.get("experiment_ref"),
            target_id="c1",
            state=state,
            action="repair",
        )
    legal = dict(approved)
    state["experiment"] = legal
    resolved = resolve_approved_experiment(
        camp,
        spec_ref=approved.get("experiment_ref"),
        target_id="c1",
        state=state,
        action="repair",
    )
    assert resolved["spec_hash"] == approved["spec_hash"]


def test_t04_current_c2_cannot_authorize_c1(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t04")
    state = load_state(camp)
    c1 = install_approved_experiment(camp, state, _draft(model={"drop_proj": 0.2}, intervention="c1_drop"), target_id="c1")
    c2 = install_approved_experiment(camp, state, _draft(model={"drop_proj": 0.8}, intervention="c2_drop"), target_id="c2")
    state["candidate_id"] = "c2"
    state["experiment"] = c2
    state["experiment_ref"] = c2.get("experiment_ref")
    with pytest.raises(ExperimentResolutionError, match="missing_binding"):
        resolve_approved_experiment(camp, spec_ref=c2.get("experiment_ref"), target_id="c9", state=state, action="train")
    resolved = resolve_approved_experiment(camp, target_id="c1", state=state, action="train")
    assert resolved["spec_hash"] == c1["spec_hash"]
    ctx = resolve_run_context(camp, "c1", state, fidelity="full", seed=0)
    assert ctx["hook_spec"]["model"]["drop_proj"] == 0.2
    with pytest.raises(ExperimentResolutionError, match="missing_binding"):
        resolve_run_context(camp, "c_missing", state, fidelity="full", seed=0)
    baseline = resolve_run_context(camp, "baseline", state, fidelity="full", seed=0)
    assert baseline["hook_spec"]["model"] == {}


def test_t05_designer_failure_does_not_approve(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t05")

    def fail(_obs):
        raise LlmUnavailable("down")

    seen = tick(
        camp,
        lambda _obs: {
            "action": "design_experiment",
            "reason_zh": "设计",
            "evidence_ids": [],
            "hypothesis_draft": {"mechanism": "池化"},
            "experiment_draft": _draft(),
        },
        services={"designer": fail},
    )
    assert experiment_is_approved(seen.get("experiment")) is False
    assert seen.get("experiment_failed") is True

    def partial(_payload):
        return {"status": "partial", "experiment_spec": _draft(), "summary_zh": "半成品"}

    camp2 = _camp(tmp_path / "p", "t05b")
    seen2 = tick(
        camp2,
        lambda _obs: {
            "action": "design_experiment",
            "reason_zh": "设计",
            "evidence_ids": [],
            "hypothesis_draft": {"mechanism": "池化"},
            "experiment_draft": _draft(),
        },
        services={"designer": partial},
    )
    assert experiment_is_approved(seen2.get("experiment")) is False


def test_t06_journal_complete_recovers_after_unsaved_state(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t06")
    state = load_state(camp)
    job = camp / "jobs" / "jrec"
    job.mkdir(parents=True)
    (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.2}), encoding="utf-8")
    _record_job(
        camp,
        state,
        {
            "job_id": "jrec",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 1,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.2},
        },
    )
    assert json.loads((job / "settlement.json").read_text(encoding="utf-8"))["status"] == "complete"
    fresh = load_state(camp)
    assert not any(row.get("job_id") == "jrec" for row in fresh.get("evidence") or [])
    rebuild_campaign_projection(camp, fresh)
    assert sum(1 for row in fresh["evidence"] if row.get("job_id") == "jrec") == 1
    memory_n = len(fresh.get("memory") or [])
    rebuild_campaign_projection(camp, fresh)
    assert sum(1 for row in fresh["evidence"] if row.get("job_id") == "jrec") == 1
    assert len(fresh.get("memory") or []) == memory_n


def test_t07_missing_comparison_rebuilds_and_hash_conflict_rejected(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "t07")
    state = load_state(camp)
    job = camp / "jobs" / "jhash"
    job.mkdir(parents=True)
    (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.2}), encoding="utf-8")
    _record_job(
        camp,
        state,
        {
            "job_id": "jhash",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 0,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.2, "source_hash": "s1"},
        },
    )
    comparison = camp / "comparisons" / "ev_jhash.json"
    assert comparison.is_file()
    comparison.unlink()
    _record_job(
        camp,
        state,
        {
            "job_id": "jhash",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 0,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.2, "source_hash": "s1"},
        },
    )
    assert comparison.is_file()
    _record_job(
        camp,
        state,
        {
            "job_id": "jhash",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 0,
            "result": {"evaluation_valid": True, "fidelity": "full", "fixed_bank_top1": 0.9, "source_hash": "s2", "checkpoint_id": "other"},
        },
    )
    assert (job / "job_result_conflict.json").is_file()
    assert state.get("failure", {}).get("error_type") == "job_result_conflict"


def test_t08_confirmation_requires_declared_pairs(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.confirmation_policy import freeze_confirmation_policy
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    policy = freeze_confirmation_policy({"training_seeds": [0, 1], "confirmation_target_pairs": 2, "min_practical_gain_pp": 1.0})
    records = [
        {
            "delta_pp": 2.0,
            "fidelity": "full",
            "seed": 0,
            "run_id": "r0",
            "control_run_id": "b0",
            "checkpoint_id": "c0",
            "source_hash": "src",
            "config_hash": "cfg",
            "approval_ref": "art",
            "evaluation_valid": True,
        },
        {
            "delta_pp": 2.0,
            "fidelity": "pilot",
            "seed": 1,
            "run_id": "r1",
            "control_run_id": "b1",
            "checkpoint_id": "c1",
            "source_hash": "src",
            "config_hash": "cfg",
            "approval_ref": "art",
            "evaluation_valid": True,
        },
    ]
    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 2.0},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 2, "training_seeds": [0, 1]},
        paired_records=records,
        fidelity="full",
        policy=policy,
    )
    assert promo["status"] != "confirmed"
    records[1]["fidelity"] = "full"
    promo2 = promotion_decision(
        comparison={"comparable": True, "delta_pp": 2.0},
        goal={"min_practical_gain_pp": 1.0, "confirmation_target_pairs": 2, "training_seeds": [0, 1]},
        paired_records=records,
        fidelity="full",
        policy=policy,
    )
    assert promo2["status"] == "confirmed"


def test_t09_same_id_content_change_stales_audit(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import refresh_audit_freshness

    camp = _camp(tmp_path, "t09")
    first = tick(camp, lambda _obs: {"action": "audit_result", "reason_zh": "审", "evidence_ids": []})
    hashed = first.get("audited_report_hash")
    first["evidence"] = [
        {
            "evidence_id": "ev_same",
            "job_id": "jsame",
            "config_hash": "cfg-a",
            "fixed_bank_top1": 0.1,
            "comparison": {"comparable": True, "delta_pp": 1.0},
        }
    ]
    save_state(camp, first)
    refresh_audit_freshness(first)
    first["audited_report_hash"] = hashed
    first["audit_status"] = "pass"
    first["evidence"][0]["fixed_bank_top1"] = 0.4
    refresh_audit_freshness(first)
    assert first["audit_status"] == "stale"


def test_t10_lesson_roundtrip_and_bogus_comparison(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, canonicalize_lesson_proposal, episode

    store = EpisodeStore(tmp_path)
    stored = store.persist_episode(
        episode(task_hash="t", candidate_id="c1", kind="exploratory_result", fidelity="full", seed=0, metric=0.1, contract_fingerprint="fp", artifact="a", job_id="j1")
    )
    aliased = canonicalize_lesson_proposal(
        {
            "claim": "池化改善检索",
            "confidence": "medium",
            "requested_level": "exploratory_result",
            "supporting_episode_ids": [stored["episode_id"]],
            "conditions": {"task": "eeg"},
        }
    )
    assert aliased["statement"] == "池化改善检索"
    assert aliased["requested_evidence_level"] == "exploratory_result"
    accepted = store.accept_lessons({"proposed_lessons": [aliased]})
    assert accepted["accepted"][0]["requested_evidence_level"] == "exploratory_result"
    (tmp_path / "comparisons").mkdir()
    (tmp_path / "comparisons" / "cmp_bogus.json").write_text("{}", encoding="utf-8")
    rejected = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "statement": "伪造对照",
                    "uncertainty": "high",
                    "supporting_episode_ids": [stored["episode_id"]],
                    "conditions": {"task": "eeg"},
                    "comparison_refs": ["cmp_bogus"],
                    "observed_effect": 1.0,
                    "requested_evidence_level": "exploratory_result",
                }
            ]
        }
    )
    assert rejected["accepted"] == []


def test_t11_pack_hash_mismatch_fails(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.export import pack_candidate, pack_is_rebuildable

    job = tmp_path / "job"
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    (job / "source_binding.json").write_text(json.dumps({"class_file": "extension/eeg_candidate.py", "workspace": "."}), encoding="utf-8")
    dest = tmp_path / "pack"
    pack_candidate(job, dest)
    ok, _reason = pack_is_rebuildable(dest)
    assert ok is False or (dest / "extension" / "eeg_candidate.py").is_file() is False
    (dest / "extension").mkdir(exist_ok=True)
    (dest / "extension" / "eeg_candidate.py").write_text("class EEGCandidate:\n    pass\n", encoding="utf-8")
    pack_candidate(job, dest, workspace=tmp_path)
    (dest / "last.ckpt").write_bytes(b"changed")
    ok, reason = pack_is_rebuildable(dest)
    assert ok is False
    assert "hash_mismatch" in reason or "source_missing" in reason


def test_t12_tie_zero_and_rank_interval() -> None:
    from react_agent.eeg_training.diagnostics import retrieval_rows_from_scores
    from react_agent.eeg_training.fixed_bank import fixed_bank_accuracy, score_query

    queries = [("q1", [0.0, 0.0])]
    bank = [("img-a", [1.0, 0.0]), ("img-b", [0.0, 1.0])]
    scored = score_query([0.0, 0.0], bank, {"img-a"}, k=1)
    assert scored["zero_norm_query"] is True
    assert scored["top_k_hit"] is False
    assert scored["min_rank"] != "hit"
    acc = fixed_bank_accuracy(queries, bank, {"q1": {"img-a"}})
    assert acc["fixed_bank_top1"] == 0.0
    tied = retrieval_rows_from_scores(
        [("q2", [1.0, 1.0])],
        [("img-a", [1.0, 1.0]), ("img-b", [1.0, 1.0])],
        {"q2": {"img-a", "img-b"}},
    )
    assert tied[0]["all_ties"] is True
    assert tied[0]["top_k_hit"] is False
    assert tied[0]["rank_interval"][0] <= tied[0]["rank_interval"][1]


def test_t13_selected_checkpoint_identity_in_artifacts(tmp_path: Path) -> None:
    from react_agent.eeg_training.diagnostics import write_validation_artifacts

    queries = [("sub-1::img-a", [1.0, 0.0]), ("sub-1::img-b", [0.0, 1.0])]
    bank = [("img-a", [1.0, 0.0]), ("img-b", [0.0, 1.0])]
    written = write_validation_artifacts(
        tmp_path,
        queries,
        bank,
        {"sub-1::img-a": {"img-a"}, "sub-1::img-b": {"img-b"}},
        checkpoint_id="best.ckpt",
        sample_seed=7,
    )
    assert written["checkpoint_id"] == "best.ckpt"
    payload = json.loads((tmp_path / "embeddings.json").read_text(encoding="utf-8"))
    assert payload["source"] == "selected_checkpoint"
    rows = (tmp_path / "retrieval_queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert "sub-1::img-a" in rows[0]


def test_t14_goal_seeds_shared_by_policy_planner_executor(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy
    from react_agent.eeg_research.agentic.execution_protocol import next_unused_training_seed
    from react_agent.eeg_research.agentic.planner import eligible_targets

    create_campaign(
        tmp_path,
        goal=_goal(training_seeds=[0, 1, 2], confirmation_target_pairs=3),
        contract={"fingerprint": "fp"},
        request_id="t14",
        protocol={"fingerprint": "p", "training_seeds": [9], "schema_version": "eeg_research.v1.9", "split_seed": 0},
    )
    camp = tmp_path / "goal"
    policy = load_confirmation_policy(camp)
    assert policy is not None
    assert policy["training_seeds"] == [0, 1, 2]
    proto = json.loads((camp / "execution_protocol.json").read_text(encoding="utf-8"))
    assert proto["training_seeds"] == [0, 1, 2]
    state = load_state(camp)
    assert state["training_seeds"] == [0, 1, 2]
    state["candidate_ready"] = True
    state["candidate_id"] = "c1"
    state["candidates"] = [{"candidate_id": "c1", "status": "ready"}]
    state["evidence"] = [
        {"candidate_id": "c1", "fidelity": "pilot", "evaluation_valid": True, "seed": 0},
        {"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 0},
    ]
    state["training_jobs"] = 1
    targets = eligible_targets(state)
    assert "c1" in targets["replicate"]
    used = {0}
    assert next_unused_training_seed(proto, used) == 1
    used = {0, 1, 2}
    assert next_unused_training_seed(proto, used) is None
    state["evidence"].append({"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 1})
    state["evidence"].append({"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 2})
    assert eligible_targets(state)["replicate"] == []


def test_role_schema_rejects_permissions_and_identity_mismatch(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.roles import RoleResultError, begin_role_task, bind_role_output, wrap_role_result
    from react_agent.eeg_research.agentic.schemas import ROLE_RESULT_VERSION

    camp = _camp(tmp_path, "roles")
    task = begin_role_task(camp, role="experiment_designer", inputs=[camp / "goal.json"], request={"draft": {}})
    ok = wrap_role_result(
        {
            "schema_version": ROLE_RESULT_VERSION,
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "status": "completed",
            "summary_zh": "ok",
        },
        task=task,
    )
    assert ok["task_id"] == task["task_id"]
    with pytest.raises(RoleResultError):
        wrap_role_result({"status": "completed", "allowed_actions": ["train"]}, task=task)
    with pytest.raises(RoleResultError):
        bind_role_output(
            {
                "schema_version": ROLE_RESULT_VERSION,
                "task_id": "other",
                "attempt_id": task["attempt_id"],
                "input_digest": task["input_digest"],
                "status": "completed",
            },
            task=task,
        )


def test_train_device_cpu_resolver() -> None:
    import os

    from react_agent.eeg_training.train_entry import resolve_train_device

    os.environ["EEG_TRAIN_DEVICE"] = "cpu"
    try:
        device = resolve_train_device("cpu")
        assert str(device) == "cpu"
    finally:
        os.environ.pop("EEG_TRAIN_DEVICE", None)
