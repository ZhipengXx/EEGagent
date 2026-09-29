"""V1.9.0 contracts, plans, ledgers, seeds, and config propagation. No live API or GPU."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.artifacts import input_digest, register, verify
from react_agent.eeg_research.agentic.execution_protocol import (
    ProtocolError,
    build_execution_protocol,
    command_matches,
    design_from_protocol,
    effective_config,
    project_command,
)
from react_agent.eeg_research.agentic.loop import _train, create_campaign, load_state, observation, save_state, tick
from react_agent.eeg_research.agentic.planner import available_actions, decide
from react_agent.eeg_research.agentic.research_plan import PlanError, apply_update, init_plan, load_plan
from react_agent.eeg_research.agentic.runner import accept_job
from react_agent.eeg_research.agentic.task_ledger import create_task, reusable
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


def test_weight_decay_zero_is_kept(tmp_path: Path) -> None:
    protocol = build_execution_protocol(_design(tmp_path, weight_decay=0.0), tmp_path)
    assert protocol["weight_decay"] == 0.0
    restored = design_from_protocol(protocol)
    assert restored.weight_decay == 0.0


def test_training_seed_does_not_change_split(tmp_path: Path) -> None:
    design = _design(tmp_path, seed=1)
    split = build_execution_protocol(design, tmp_path, split_seed=1, training_seed=1)
    other = build_execution_protocol(design, tmp_path, split_seed=1, training_seed=99, training_seeds=[1, 99])
    assert split["validation_image_ids"] == other["validation_image_ids"]
    assert split["validation_identity"] == other["validation_identity"]
    assert split["split_seed"] == other["split_seed"] == 1
    assert other["training_seed"] == 99
    assert split["train_query_ids"]
    assert "::" in split["train_query_ids"][0]


def test_query_ids_include_subject(tmp_path: Path) -> None:
    protocol = build_execution_protocol(_design(tmp_path, subject="sub-02"), tmp_path)
    assert all(item.startswith("sub-02::") for item in protocol["train_query_ids"])
    assert protocol["positive_map"][protocol["validation_query_ids"][0]] in protocol["validation_image_ids"]


def test_plan_update_is_versioned_and_consumed(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    camp.mkdir()
    goal = {"goal_id": "g", "objective": "改进检索", "max_training_jobs": 4, "max_llm_calls": 10, "max_gpu_seconds": 100}
    plan = init_plan(camp, goal)
    assert plan["plan_version"] == 1
    with pytest.raises(PlanError, match="plan_update_needs_evidence"):
        apply_update(camp, {"based_on_plan_version": 1, "affected_question_ids": ["q1"]}, known_evidence_ids={"ev_1"})
    updated = apply_update(
        camp,
        {
            "based_on_plan_version": 1,
            "based_on_evidence_ids": ["ev_1"],
            "affected_question_ids": ["q1"],
            "question_updates": [{"question_id": "q1", "status": "answered"}],
            "reason_zh": "基线已建立",
        },
        known_evidence_ids={"ev_1"},
    )
    assert updated["plan_version"] == 2
    assert (camp / "plan_versions" / "v1.json").is_file()
    assert load_plan(camp)["research_questions"][0]["status"] == "answered"
    with pytest.raises(PlanError, match="plan_version_conflict"):
        apply_update(
            camp,
            {
                "based_on_plan_version": 1,
                "based_on_evidence_ids": ["ev_1"],
                "affected_question_ids": ["q1"],
            },
            known_evidence_ids={"ev_1"},
        )


def test_task_ledger_rejects_cross_attempt_reuse(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    camp.mkdir()
    task = create_task(
        camp,
        role="candidate_coder",
        input_artifact_refs=["art_1"],
        input_digest="abc",
        expected_output_schema="role_result.v1",
        candidate_id="c1",
        attempt_id="att1",
    )
    payload = {"task_id": task["task_id"], "attempt_id": "att1", "input_digest": "abc", "candidate_id": "c1"}
    assert reusable(payload, task_id=task["task_id"], attempt_id="att1", input_digest="abc", candidate_id="c1")
    assert not reusable(payload, task_id=task["task_id"], attempt_id="att2", input_digest="abc", candidate_id="c1")
    assert not reusable(payload, task_id=task["task_id"], attempt_id="att1", input_digest="zzz", candidate_id="c1")
    assert not reusable(payload, task_id=task["task_id"], attempt_id="att1", input_digest="abc", candidate_id="c2")


def test_artifact_hash_mismatch_is_not_handoff(tmp_path: Path) -> None:
    camp = tmp_path / "camp"
    camp.mkdir()
    path = camp / "note.txt"
    path.write_text("one", encoding="utf-8")
    row = register(camp, path, kind="note", candidate_id="c1")
    ok, reason = verify(camp, row["artifact_id"])
    assert ok is True
    path.write_text("two", encoding="utf-8")
    ok, reason = verify(camp, row["artifact_id"])
    assert ok is False
    assert reason == "artifact_hash_mismatch"
    other = tmp_path / "other.txt"
    other.write_text("one", encoding="utf-8")
    assert input_digest([path]) != input_digest([other]) or path.read_text() != other.read_text()


def test_accept_job_rejects_nan(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": math.nan}), encoding="utf-8")
    (job / "source_binding.json").write_text(json.dumps({"module": "m", "file_sha256": "x"}), encoding="utf-8")
    rejected = accept_job(job, {"module": "m", "entry_sha256": "x"}, "full")
    assert rejected["evaluation_valid"] is False
    assert rejected["reason"] == "score_not_finite"


def test_failures_reach_the_planner_observation(tmp_path: Path) -> None:
    design = _design(tmp_path)
    from react_agent.eeg_research.agentic.contract import freeze_contract

    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal={"goal_id": "goal", "max_training_jobs": 2, "max_llm_calls": 8, "max_gpu_seconds": 10}, contract=contract, request_id="obs")
    camp = tmp_path / "goal"
    from react_agent.eeg_research.agentic.loop import load_state, save_state

    state = load_state(camp)
    job = camp / "jobs" / "j1"
    job.mkdir(parents=True)
    (job / "train.log").write_text("cuda:0 vs cpu logit_scale", encoding="utf-8")
    state["evidence"] = [
        {
            "evidence_id": "ev_1",
            "candidate_id": "c1",
            "fidelity": "pilot",
            "evaluation_valid": False,
            "reason": "failed",
            "job_dir": str(job),
            "job_status": "failed",
        }
    ]
    save_state(camp, state)
    tick(camp, lambda _obs: {"action": "diagnose_results", "reason_zh": "看失败", "evidence_ids": ["ev_1"]})
    obs = observation(camp)
    profile = [row for row in obs["evidence"] if row.get("kind") == "learning_profile"][0]
    assert profile["failures"]
    assert "cuda:0" in profile["failures"][0]["train_log_tail"]


def test_target_id_is_required_for_training() -> None:
    outcome = decide(
        {"available_actions": ["run_pilot", "stop"], "trainable_ids": ["c1"], "_known_evidence_ids": []},
        lambda _obs: {"action": "run_pilot", "reason_zh": "跑", "evidence_ids": []},
    )
    assert outcome["ok"] is False
    assert "target_id" in str(outcome["detail"])


def test_baseline_does_not_consume_candidate_slots() -> None:
    state = {
        "experiment": {"initial_fidelity": "pilot"},
        "llm_calls_left": 40,
        "gpu_seconds_left": 10,
        "max_candidates": 1,
        "candidates": [{"candidate_id": "baseline", "status": "ready"}],
    }
    assert "implement_candidate" in available_actions(state)
    state["candidates"].append({"candidate_id": "c1", "status": "ready"})
    assert "implement_candidate" not in available_actions(state)


def test_unmatched_control_is_not_a_comparison() -> None:
    from react_agent.eeg_research.agentic.comparison import compare_runs

    result = compare_runs(
        candidate={"evidence_id": "ev_c", "candidate_id": "c1", "evaluation_valid": True, "fidelity": "full", "seed": 1},
        control=None,
        protocol={"schema_version": "eeg_research.v1.9", "split_seed": 1, "validation_image_ids": ["a"], "fingerprint": "p"},
    )
    assert result["comparable"] is False
    assert result["reason"] == "unmatched_control"


def test_null_gain_bar_is_not_invented() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 0.4, "reason": "matched_control"},
        goal={"min_practical_gain_pp": None, "confirmation_target_pairs": 3},
        paired_deltas_pp=[0.4],
    )
    assert promo["min_practical_gain_pp"] is None
    assert promo["confirmation"] == "provisional_improvement"


def test_method_cards_are_local() -> None:
    from react_agent.eeg_research.agentic.knowledge import retrieve_methods

    hits = retrieve_methods("projection scale contrastive")
    assert hits["local_only"] is True
    assert hits["hits"][0]["id"] == "projection_scale"
    assert hits["hits"][0]["retrieval"] == "local_only"


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


def _open_campaign(tmp_path: Path, *, with_protocol: bool = False, request_id: str = "v19"):
    from react_agent.eeg_research.agentic.contract import freeze_contract

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    protocol = build_execution_protocol(design, tmp_path) if with_protocol else None
    create_campaign(tmp_path, goal=_goal(), contract=contract, request_id=request_id, protocol=protocol)
    return tmp_path / "goal", protocol


def _ready(camp: Path, candidate_id: str = "c1") -> None:
    state = load_state(camp)
    state.update(
        {
            "candidate_ready": True,
            "candidate_id": candidate_id,
            "experiment": {"initial_fidelity": "pilot"},
            "candidates": [{"candidate_id": candidate_id, "status": "ready"}],
        }
    )
    save_state(camp, state)


def _settle_scores(phase: dict, scores: dict[str, float], *, fingerprint: str | None = None):
    def settle(_camp, job_id: str):
        if phase["running"]:
            return {"status": "running", "job_id": job_id}
        candidate = "baseline" if "baseline" in job_id else "c1"
        result = {
            "evaluation_valid": True,
            "fidelity": "pilot",
            "fixed_bank_top1": scores[candidate],
            "gallery_size": 1,
        }
        if fingerprint:
            result["execution_fingerprint"] = fingerprint
        return {
            "status": "finished",
            "job_id": job_id,
            "candidate_id": candidate,
            "seed": phase.get("seed", 0),
            "gpu_seconds": 5,
            "result": result,
        }

    return settle


def test_incomparable_evidence_has_no_delta(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, with_protocol=False, request_id="no-delta")
    _ready(camp)
    launched: list[str] = []
    phase = {"running": True, "seed": 0}

    def planner(_obs):
        return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": [], "target_id": "c1"}

    def launch(_camp, _state, job_id, candidate_id, fidelity):
        launched.append(candidate_id)
        return {"job_id": job_id}

    services = {"launch": launch, "settle": _settle_scores(phase, {"baseline": 0.004, "c1": 0.003})}
    tick(camp, planner, services=services)
    assert launched == ["baseline"]
    phase["running"] = False
    tick(camp, planner, services=services)
    tick(camp, planner, services=services)
    tick(camp, planner, services=services)
    state = load_state(camp)
    rows = [row for row in state["evidence"] if row.get("candidate_id") == "c1"]
    assert rows
    assert rows[0]["comparison_status"] == "protocol_missing"
    assert "delta_vs_control_pp" not in rows[0]
    assert rows[0].get("control_id") in (None, "")


def test_matched_control_writes_comparable_delta(tmp_path: Path) -> None:
    camp, protocol = _open_campaign(tmp_path, with_protocol=True, request_id="delta")
    assert protocol is not None
    _ready(camp)
    launched: list[str] = []
    phase = {"running": True, "seed": int(protocol["training_seed"])}
    fingerprint = protocol["fingerprint"]

    def planner(_obs):
        return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": [], "target_id": "c1"}

    def launch(_camp, _state, job_id, candidate_id, fidelity):
        launched.append(candidate_id)
        return {"job_id": job_id}

    services = {
        "launch": launch,
        "settle": _settle_scores(phase, {"baseline": 0.10, "c1": 0.12}, fingerprint=fingerprint),
    }
    tick(camp, planner, services=services)
    phase["running"] = False
    tick(camp, planner, services=services)
    tick(camp, planner, services=services)
    tick(camp, planner, services=services)
    state = load_state(camp)
    rows = [row for row in state["evidence"] if row.get("candidate_id") == "c1"]
    assert rows
    assert rows[0]["comparison_status"] == "comparable"
    assert rows[0]["delta_vs_control_pp"] == 2.0
    assert rows[0]["comparison"]["comparable"] is True


def test_blocked_launch_does_not_take_a_training_slot(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, with_protocol=True, request_id="blocked-launch")
    _ready(camp)

    def planner(_obs):
        return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": [], "target_id": "c1"}

    def launch(_camp, _state, job_id, candidate_id, fidelity):
        return {"job_id": job_id, "status": "blocked", "detail": "protocol_mismatch"}

    state = tick(camp, planner, services={"launch": launch, "settle": lambda *_a, **_k: {"status": "blocked"}})
    assert state["training_jobs"] == 0
    assert state.get("live_job") is None
    assert state["status"] == "blocked"
    assert state["detail"] == "protocol_mismatch"
    assert state.get("evidence") == []


def test_start_job_protocol_mismatch_does_not_spawn(tmp_path: Path, monkeypatch) -> None:
    import sys
    from dataclasses import replace

    from react_agent.eeg_research.agentic.jobs import start_job

    monkeypatch.setenv("EEG_TRAIN_PYTHON", sys.executable)

    def boom(*_args, **_kwargs):
        raise AssertionError("Popen")

    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs.subprocess.Popen", boom)
    camp, protocol = _open_campaign(tmp_path, with_protocol=True, request_id="mismatch-job")
    assert protocol is not None
    design = design_from_protocol(protocol)
    mismatched = replace(design, batch_size=int(design.batch_size) + 8)
    job = camp / "jobs" / "j1"
    record = start_job(
        job,
        mismatched,
        candidate_id="baseline",
        extension=None,
        fidelity="pilot",
        root=tmp_path,
        protocol_path=camp / "execution_protocol.json",
    )
    assert record["status"] == "blocked"
    assert record["detail"] == "protocol_mismatch"
    saved = json.loads((job / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked"
    assert not (job / "train.log").exists()


def test_repair_does_not_open_a_new_experiment() -> None:
    state = {
        "experiment": {"initial_fidelity": "pilot"},
        "llm_calls_left": 40,
        "gpu_seconds_left": 10,
        "max_candidates": 4,
        "candidates": [{"candidate_id": "c1", "status": "review_needs_fix"}],
        "repair_task": {"candidate_id": "c1", "remaining": 1, "used": 1, "issues": []},
        "candidate_ready": False,
    }
    actions = available_actions(state)
    assert "implement_candidate" in actions
    assert "propose_experiment" not in actions


def test_unmatched_control_appends_to_existing_queue(tmp_path: Path) -> None:
    camp, protocol = _open_campaign(tmp_path, with_protocol=True, request_id="queued")
    _ready(camp)
    state = load_state(camp)
    state["queued"] = [["c2", "full", 99]]
    launched: list[str] = []

    def launch(_camp, _state, job_id, candidate_id, fidelity):
        launched.append(candidate_id)
        return {"job_id": job_id}

    _train(camp, state, "run_pilot", None, {"launch": launch}, {"target_id": "c1"})
    assert launched == ["baseline"]
    assert state["queued"][0] == ["c2", "full", 99]
    assert state["queued"][1][0] == "c1"
    assert state["queued"][1][1] == "pilot"
    if protocol is not None:
        assert state["queued"][1][2] == int(protocol["training_seed"])


def test_inline_runner_respects_protocol_checkpoint(tmp_path: Path) -> None:
    camp, _protocol = _open_campaign(tmp_path, with_protocol=True, request_id="inline-ckpt")
    _ready(camp)
    workspace = camp / "candidates" / "c1"
    workspace.mkdir(parents=True)
    (workspace / "source_manifest.json").write_text(
        json.dumps({"module": "eeg_candidate", "entry_sha256": "abc"}),
        encoding="utf-8",
    )

    def planner(_obs):
        return {"action": "run_pilot", "reason_zh": "试跑", "evidence_ids": [], "target_id": "c1"}

    def runner(job: Path, manifest: dict, fidelity: str):
        (job / "metrics.json").write_text(
            json.dumps({"fixed_bank_top1": 0.02, "validation_image_count": 1}),
            encoding="utf-8",
        )
        (job / "source_binding.json").write_text(
            json.dumps({"module": manifest["module"], "file_sha256": manifest["entry_sha256"]}),
            encoding="utf-8",
        )
        return {"gpu_seconds": 1, "seed": 0}

    state = tick(camp, planner, runner)
    assert state["evidence"]
    assert state["evidence"][-1]["evaluation_valid"] is False
    assert state["evidence"][-1]["reason"] == "checkpoint_missing"


def test_deadline_sends_term_then_kill(tmp_path: Path, monkeypatch) -> None:
    import signal
    import time

    from react_agent.eeg_research.agentic import jobs as jobs_mod
    from react_agent.eeg_research.agentic.jobs import reconcile

    monkeypatch.setattr(jobs_mod, "TERM_GRACE_S", 0.0)
    monkeypatch.setattr(jobs_mod, "_alive", lambda _record: True)
    sent: list[int] = []
    monkeypatch.setattr(jobs_mod.os, "killpg", lambda _pid, sig: sent.append(sig))
    job = tmp_path / "j1"
    job.mkdir()
    record = {
        "job_id": "j1",
        "pid": 4242,
        "proc_start": "1",
        "status": "running",
        "started_at": time.time() - 10,
        "deadline_at": time.time() - 1,
        "gpu": [0],
        "fidelity": "pilot",
        "manifest": {"module": "m", "entry_sha256": "abc"},
    }
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")
    first = reconcile(job)
    assert first["status"] == "running"
    assert first.get("term_sent_at")
    assert sent == [signal.SIGTERM]
    second = reconcile(job)
    assert second["status"] == "cancelled"
    assert second["detail"] == "deadline"
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_other_subjects_test_does_not_freeze_empty_queries(tmp_path: Path) -> None:
    meg = tmp_path / "things-meg" / "Preprocessed_data"
    for name in ("sub-01", "sub-02"):
        (meg / name).mkdir(parents=True)
        (meg / name / "train.pt").write_text("", encoding="utf-8")
        (meg / name / "test.pt").write_text("", encoding="utf-8")
    design = Design(
        dataset="meg",
        exp_setting="inter-subject",
        subject="sub-01",
        epochs=12,
        seed=2,
        data_root=str(tmp_path),
        training_strategy="pooled_subjects",
        policy="agentic",
    )
    with pytest.raises(ProtocolError):
        build_execution_protocol(design, tmp_path)
    assert not (tmp_path / "execution_protocol.json").exists()


def test_unavailable_diagnostics_are_not_empty_fakes(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.diagnostics import CATEGORIES, summarize_bundle
    from react_agent.eeg_training.diagnostics import UNAVAILABLE, compute_job_diagnostics

    bundle = compute_job_diagnostics(tmp_path / "empty_job")
    for name in CATEGORIES:
        assert name in bundle
        assert bundle[name]["status"] in {"observed", UNAVAILABLE}
        if bundle[name]["status"] == UNAVAILABLE:
            assert bundle[name].get("reason")
            assert bundle[name].get("payload") in (None, {}) or "reason" in bundle[name]
    summary = summarize_bundle(bundle)
    assert summary["items"]["retrieval_errors"]["status"] == UNAVAILABLE


def test_librarian_is_local_only_and_records_a_task(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.contract import freeze_contract
    from react_agent.eeg_research.agentic.loop import create_campaign, load_state, tick
    from react_agent.eeg_research.agentic.task_ledger import latest

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal=_goal(), contract=contract, request_id="lib")
    camp = tmp_path / "goal"
    tick(camp, lambda _obs: {"action": "retrieve_methods", "reason_zh": "查方法", "evidence_ids": [], "query": "temporal encoder"})
    state = load_state(camp)
    hits = [row for row in state["evidence"] if row.get("kind") == "method_hits"]
    assert hits
    assert hits[0]["local_only"] is True
    payload = json.loads((camp / "knowledge" / "method_hits.json").read_text(encoding="utf-8"))
    assert payload["local_only"] is True
    assert payload["schema_version"] == "eeg_research.role_result.v1"
    assert latest(camp, payload["task_id"])["status"] == "completed"


def test_replicate_is_not_confirmation() -> None:
    from react_agent.eeg_research.agentic.promotion import promotion_decision

    promo = promotion_decision(
        comparison={"comparable": True, "delta_pp": 1.2, "reason": "matched_control"},
        goal={"min_practical_gain_pp": None, "confirmation_target_pairs": 3},
        paired_deltas_pp=[1.2, 0.9],
        fidelity="full",
    )
    assert promo["replicate_status"] == "replicated_seed"
    assert promo["confirmation"] == "provisional_improvement"
    assert promo["status"] == "provisional"
    assert promo["tier"] != "confirmation"


def test_lesson_cannot_rewrite_episode(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode

    store = EpisodeStore(tmp_path)
    saved = store.persist_episode(
        episode(
            task_hash="g",
            candidate_id="c1",
            kind="exploratory_result",
            fidelity="full",
            seed=0,
            metric=0.1,
            contract_fingerprint="fp",
            artifact="a",
        )
    )
    with pytest.raises(PermissionError, match="episode_immutable"):
        store.update_episode(saved["episode_id"], primary_metric=0.99)
    rejected = store.accept_lessons(
        {
            "proposed_lessons": [
                {
                    "supporting_episode_ids": [saved["episode_id"]],
                    "evidence_level": "confirmed_result",
                    "observed_effect": "win",
                }
            ]
        }
    )
    assert rejected["rejected"]
    assert rejected["rejected"][0]["reason"] == "evidence_level_exceeds_runs"
    assert store.list_episodes()[0]["primary_metric"] == 0.1


def test_export_pack_is_evaluate_only(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.cli import main
    from react_agent.eeg_research.agentic.export import pack_candidate, pack_is_rebuildable

    job = tmp_path / "job"
    job.mkdir()
    (job / "last.ckpt").write_bytes(b"ckpt")
    (job / "source_binding.json").write_text(json.dumps({"module": "eeg_candidate"}), encoding="utf-8")
    (job / "evaluation_identity.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "pack"
    manifest = pack_candidate(job, dest)
    assert manifest["evaluate_only"] is True
    assert manifest["final_test"] is False
    ok, reason = pack_is_rebuildable(dest)
    assert ok is True
    assert reason == "ok"
    code = main(["evaluate-export", "--pack", str(dest)])
    assert code == 0


def test_validate_goal_does_not_spawn() -> None:
    from react_agent.eeg_research.agentic.cli import load_goal_file, main
    from react_agent.eeg_research.agentic.planner import decide

    path = Path(__file__).resolve().parents[2] / "configs" / "goals" / "eeg_retrieval_v1_9.yaml"
    goal = load_goal_file(path)
    assert goal["max_training_jobs"] == 12
    assert goal["max_llm_calls"] == 120
    assert goal["max_gpu_seconds"] == 28800
    blocked = decide(
        {"available_actions": ["stop"], "_known_evidence_ids": []},
        lambda _obs: {"action": "invent", "reason_zh": "x", "evidence_ids": []},
    )
    assert blocked["ok"] is False
    assert "unknown_action" in str(blocked["detail"])
    code = main(["validate-goal", "--goal", str(path)])
    assert code == 0


def test_view_splits_pilot_and_full(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.contract import freeze_contract
    from react_agent.eeg_research.agentic.loop import create_campaign, load_state, save_state
    from react_agent.eeg_research.agentic.view import campaign_view

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal=_goal(), contract=contract, request_id="view")
    camp = tmp_path / "goal"
    state = load_state(camp)
    state["evidence"] = [
        {
            "evidence_id": "ev_p",
            "candidate_id": "c1",
            "fidelity": "pilot",
            "evaluation_valid": True,
            "fixed_bank_top1": 0.4,
            "delta_vs_control_pp": 5.0,
            "comparison": {"comparable": True},
            "comparison_status": "comparable",
        },
        {
            "evidence_id": "ev_f",
            "candidate_id": "c1",
            "fidelity": "full",
            "evaluation_valid": True,
            "fixed_bank_top1": 0.2,
            "delta_vs_control_pp": 1.0,
            "comparison": {"comparable": True},
            "comparison_status": "comparable",
            "promotion": {"tier": "full", "confirmation": "provisional_improvement"},
        },
    ]
    save_state(camp, state)
    view = campaign_view(camp)
    assert len(view["pilot_rows"]) == 1
    assert len(view["full_rows"]) == 1
    assert view["best_full"]["evidence_id"] == "ev_f"
    assert view["best_full"]["delta_vs_control_pp"] == 1.0
    assert "gpu_seconds_reserved" in view["budget"]
    assert view["budget"]["api_usd"] is None


def test_comparison_then_diagnostics_reach_observation(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.contract import freeze_contract
    from react_agent.eeg_research.agentic.loop import _record_job, create_campaign, load_state, observation, save_state

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(tmp_path, goal=_goal(), contract=contract, request_id="diag")
    camp = tmp_path / "goal"
    job = camp / "jobs" / "j1_c1_pilot"
    job.mkdir(parents=True)
    (job / "metrics.json").write_text(json.dumps({"fixed_bank_top1": 0.02, "validation_image_count": 2}), encoding="utf-8")
    (job / "history.jsonl").write_text(json.dumps({"epoch": 1, "loss": 1.2, "fixed_bank_top1": 0.02}) + "\n", encoding="utf-8")
    state = load_state(camp)
    _record_job(
        camp,
        state,
        {
            "job_id": "j1_c1_pilot",
            "candidate_id": "c1",
            "seed": 0,
            "status": "finished",
            "gpu_seconds": 1,
            "result": {"evaluation_valid": True, "fidelity": "pilot", "fixed_bank_top1": 0.02},
        },
    )
    save_state(camp, state)
    row = state["evidence"][-1]
    assert row["comparison"]
    assert row["diagnostics"]
    assert row["diagnostic_ref"]
    assert (job / "diagnostic_bundle.json").is_file()
    obs = observation(camp)
    assert obs["latest_comparison"]
    assert obs["latest_diagnostics"]
    assert any("diagnostic_ref" in item for item in obs["evidence"])


def test_truncated_read_allows_next_range(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.coder import read_code

    workspace = tmp_path / "ws"
    (workspace / "extension").mkdir(parents=True)
    (workspace / "extension" / "eeg_candidate.py").write_text("\n".join(f"line_{i}" for i in range(1, 40)) + "\n", encoding="utf-8")
    first = read_code(workspace, "extension/eeg_candidate.py", 1, 10)
    assert first["truncated"] is True
    assert first["next_range"]["start"] == 11
    second = read_code(workspace, "extension/eeg_candidate.py", first["next_range"]["start"], first["next_range"]["end"])
    assert second["ok"] is True
    assert "line_11" in second["text"]


def test_stop_reason_must_be_structured() -> None:
    from react_agent.eeg_research.agentic.planner import decide

    bad = decide(
        {"available_actions": ["stop"], "_known_evidence_ids": []},
        lambda _obs: {"action": "stop", "stop_reason": "feels_done", "reason_zh": "停", "evidence_ids": []},
    )
    assert bad["ok"] is False
    assert "stop_reason_invalid" in str(bad["detail"])
    good = decide(
        {"available_actions": ["stop"], "_known_evidence_ids": []},
        lambda _obs: {"action": "stop", "stop_reason": "no_progress", "reason_zh": "停", "evidence_ids": []},
    )
    assert good["ok"] is True

