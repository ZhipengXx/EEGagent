"""Live DeepSeek JSON smoke for auditor and curator. Skips without DEEPSEEK_API_KEY.

Does not spawn a worker, start a campaign loop, or run GPU training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from react_agent.eeg_research.agentic.llm import role_backend
from react_agent.eeg_research.agentic.loop import create_campaign
from react_agent.eeg_research.agentic.memory import EpisodeStore, episode
from react_agent.eeg_research.agentic.roles import begin_role_task, validate_role_result
from react_agent.eeg_research.agentic.worker import _lesson_proposal
from react_agent.eeg_training.protocol import Design

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

pytestmark = pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY missing")


def _images(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir(exist_ok=True)
    (tmp_path / "test").mkdir(exist_ok=True)
    (tmp_path / "train" / "train.pt").write_text(json.dumps(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]), encoding="utf-8")
    (tmp_path / "test" / "test.pt").write_text(json.dumps(["z"]), encoding="utf-8")


def _design(tmp_path: Path) -> Design:
    _images(tmp_path)
    return Design(
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


def _camp(tmp_path: Path, request_id: str) -> Path:
    from react_agent.eeg_research.agentic.contract import freeze_contract

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(
        tmp_path,
        goal={
            "goal_id": "goal",
            "max_training_jobs": 4,
            "max_llm_calls": 4,
            "max_gpu_seconds": 100,
            "max_candidates": 4,
        },
        contract=contract,
        request_id=request_id,
    )
    return tmp_path / "goal"


def test_live_result_auditor_placeholder_artifacts(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "live-auditor")
    task = begin_role_task(camp, role="result_auditor", inputs=[camp / "goal.json"])
    payload = {
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "input_digest": task["input_digest"],
        "constraint": "Do not train. Placeholder artifacts only.",
        "artifacts": [{"kind": "placeholder", "note": "no checkpoint, no metrics, no training"}],
        "latest": None,
        "draft": {
            "verdict": "REVISE",
            "claims": [{"claim": "no_job_result", "status": "unsupported"}],
            "summary_zh": "无训练产物",
        },
    }
    reply = role_backend(camp, "result_auditor")(payload)
    envelope = validate_role_result(reply, task)
    assert envelope["task_id"] == task["task_id"]
    assert envelope["input_digest"] == task["input_digest"]
    assert envelope["schema_version"] == "eeg_research.role_result.v1"
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["success"] is True
    assert rows[-1]["role"] == "result_auditor"
    assert not (camp / "worker.lock").exists()


def test_live_memory_curator_cannot_edit_episode(tmp_path: Path) -> None:
    camp = _camp(tmp_path, "live-curator")
    store = EpisodeStore(camp)
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
    before = json.dumps(store.list_episodes()[0], sort_keys=True)
    task = begin_role_task(camp, role="memory_curator", inputs=[camp / "goal.json"])
    reply = role_backend(camp, "memory_curator")(
        {
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "episodes": store.list_episodes(),
            "constraint": "Do not edit episode values. Propose lessons only.",
        }
    )
    envelope = validate_role_result(reply, task)
    assert envelope["task_id"] == task["task_id"]
    store.accept_lessons(_lesson_proposal(reply))
    after = json.dumps(store.list_episodes()[0], sort_keys=True)
    assert after == before
    assert store.list_episodes()[0]["episode_id"] == saved["episode_id"]
    assert store.list_episodes()[0]["primary_metric"] == 0.1
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["success"] is True
    assert rows[-1]["role"] == "memory_curator"
    assert not (camp / "worker.lock").exists()


def _camp_open(tmp_path: Path, request_id: str, *, calls: int = 24) -> Path:
    from react_agent.eeg_research.agentic.contract import freeze_contract

    design = _design(tmp_path)
    contract = freeze_contract(design, tmp_path)
    create_campaign(
        tmp_path,
        goal={
            "goal_id": "goal",
            "max_training_jobs": 4,
            "max_llm_calls": calls,
            "max_gpu_seconds": 100,
            "max_candidates": 4,
        },
        contract=contract,
        request_id=request_id,
    )
    return tmp_path / "goal"


def _assert_ledger(camp: Path, role: str) -> None:
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[-1]["success"] is True
    assert rows[-1]["role"] == role
    assert all(row["role"] == role for row in rows)
    assert not (camp / "worker.lock").exists()
    assert not list((camp / "jobs").glob("*/job.json")) if (camp / "jobs").exists() else True


def _body(reply: dict) -> dict:
    inner = reply.get("payload")
    return inner if isinstance(inner, dict) else reply


def test_live_planner_action_matches_the_current_task(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.loop import observation
    from react_agent.eeg_research.agentic.planner import decide

    camp = _camp_open(tmp_path, "live-planner")
    obs = observation(camp)
    result = decide(obs, role_backend(camp, "research_planner"))
    assert result["ok"] is True
    assert result["action"] in obs["available_actions"]
    _assert_ledger(camp, "research_planner")


def test_live_librarian_packet_is_local(tmp_path: Path) -> None:
    camp = _camp_open(tmp_path, "live-librarian")
    task = begin_role_task(camp, role="research_librarian", inputs=[camp / "goal.json"])
    reply = role_backend(camp, "research_librarian")(
        {
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "question": "Which EEG-to-image retrieval objective stays inside the current local contrastive hook?",
            "search_scope": "supplied_contract_only",
            "sources": [],
            "constraint": "No web. local_only must be true. Do not invent a numeric gain.",
        }
    )
    envelope = validate_role_result(reply, task)
    body = _body(envelope)
    assert body.get("local_only") is True
    assert isinstance(body.get("summary_zh"), str) and body["summary_zh"]
    assert isinstance(body.get("method_cards", []), list)
    _assert_ledger(camp, "research_librarian")


def test_live_designer_spec_is_classified_by_the_gate(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.experiment_gate import approve_experiment

    camp = _camp_open(tmp_path, "live-designer")
    task = begin_role_task(camp, role="experiment_designer", inputs=[camp / "goal.json"])
    reply = role_backend(camp, "experiment_designer")(
        {
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "draft": {
                "initial_fidelity": "pilot",
                "parent_candidate_id": "baseline",
                "hypothesis": "A smaller projection dropout changes retrieval without touching the evaluator.",
            },
            "diagnostics": None,
            "missing_inputs": ["diagnostics"],
            "capabilities": {"hooks": {"verified": False}},
            "constraint": "One principal intervention. Do not approve yourself. Do not read final holdout.",
        }
    )
    envelope = validate_role_result(reply, task)
    body = _body(envelope)
    status = str(body.get("status") or "")
    if status in {"requires_framework_extension", "blocked"} or body.get("requires_framework_extension"):
        assert isinstance(body.get("summary_zh"), str)
    else:
        spec = body.get("experiment_spec")
        assert isinstance(spec, dict)
        classified = approve_experiment(spec)
        assert classified["status"] in {"approved", "draft", "blocked"}
    _assert_ledger(camp, "experiment_designer")


def test_live_reviewer_status_is_a_known_verdict(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import review

    camp = _camp_open(tmp_path, "live-reviewer")
    workspace = tmp_path / "workspace"
    (workspace / "extension").mkdir(parents=True)
    (workspace / "extension" / "eeg_candidate.py").write_text(
        "from react_agent.eeg_research.agentic.baseline import EEGCandidate as Baseline\n\nclass EEGCandidate(Baseline):\n    candidate_id = 'c1'\n",
        encoding="utf-8",
    )
    (workspace / "checks.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    backend = role_backend(camp, "candidate_reviewer")
    result = review(
        workspace,
        {"intervention": "projection_scale", "hypothesis": "dropout", "parent_candidate_id": "baseline"},
        {"research_scope": "pooled_subject_retrieval", "primary_metric": "validation.fixed_gallery_top1"},
        backend,
        getattr(backend, "model", ""),
    )
    assert result["format_failed"] is False
    assert result["status"] in {"ready", "needs_fix", "blocked"}
    _assert_ledger(camp, "candidate_reviewer")


def test_live_analyst_keeps_hypothesis_assessment(tmp_path: Path) -> None:
    camp = _camp_open(tmp_path, "live-analyst")
    reply = role_backend(camp, "result_analyst")(
        {
            "latest": {
                "evidence_id": "ev_1",
                "candidate_id": "c1",
                "fidelity": "pilot",
                "fixed_bank_top1": 0.02,
                "seed": 0,
            },
            "comparison": {"comparable": False, "reason": "unmatched_control"},
            "diagnostics": None,
            "hypothesis": "projection dropout",
            "experiment": {"intervention": "projection_scale", "parent_candidate_id": "baseline"},
            "hypothesis_binding_missing": False,
            "constraint": "Do not invent a holdout score. Missing diagnostics stay unknown.",
        }
    )
    assert isinstance(reply, dict)
    assert "hypothesis_assessment" in reply
    assert isinstance(reply.get("summary_zh"), str) and reply["summary_zh"]
    assert isinstance(reply.get("observations", []), list)
    _assert_ledger(camp, "result_analyst")


def test_live_coder_finishes_or_fails_explicitly(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.native_patch import implement

    camp = _camp_open(tmp_path, "live-coder", calls=16)
    workspace = tmp_path / "coder-workspace"
    backend = role_backend(camp, "candidate_coder")
    outcome = implement(
        workspace,
        {
            "intervention": "projection_scale",
            "hypothesis": "Keep the baseline encoder and change only drop_proj.",
            "parent_candidate_id": "baseline",
            "initial_fidelity": "pilot",
        },
        backend,
        max_steps=4,
        max_repairs=1,
    )
    assert outcome["status"] in {"ready_for_review", "implementation_failed", "requires_framework_extension"}
    if outcome["status"] == "implementation_failed":
        assert outcome.get("detail") in {"step_limit", "repair_limit", "budget_exhausted"}
    rows = [json.loads(line) for line in (camp / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["role"] == "candidate_coder" and row["success"] is True for row in rows)
    assert not (camp / "worker.lock").exists()
