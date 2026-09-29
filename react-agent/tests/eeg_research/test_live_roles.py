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
