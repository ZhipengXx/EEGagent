"""Explicit demo campaign for the workbench. Not a real experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEMO_CAMPAIGN = "demo_ui_workspace"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_demo_campaign(root: Path) -> Path:
    """Synthetic workspace used by tests and ?demo=1. Does not read user runs."""
    camp = root / DEMO_CAMPAIGN
    camp.mkdir(parents=True, exist_ok=True)
    now = time.time()
    _write(
        camp / "goal.json",
        {
            "goal_id": DEMO_CAMPAIGN,
            "objective": "Improve EEG–image retrieval under a matched protocol (demo)",
            "max_training_jobs": 8,
            "max_llm_calls": 40,
            "max_gpu_seconds": 3600,
        },
    )
    _write(
        camp / "evaluation_contract.json",
        {
            "fingerprint": "demo-contract",
            "research_scope": "pooled_subject_retrieval",
            "final_test_enabled": False,
        },
    )
    _write(camp / "cost.json", {"llm_calls": 8, "llm_failures": 0, "api_usd": None, "gpu_seconds_used": 12.0})
    decisions = []
    actions = [
        "inspect_data",
        "retrieve_methods",
        "design_experiment",
        "implement_candidate",
        "repair_candidate",
        "run_pilot",
        "diagnose_results",
        "run_full",
        "replicate",
        "audit_result",
        "retrieve_memory",
        "stop",
    ]
    for index, action in enumerate(actions, start=1):
        decisions.append(
            {
                "decision_id": f"d{index}",
                "action": action,
                "ok": action != "stop",
                "executed": True,
                "at": now + index,
                "candidate_id": "c01",
                "reason_zh": f"观察：步骤 {index}。假设：时序池化。下一步：继续。",
            }
        )
    state = {
        "status": "analyzing",
        "detail": "",
        "execution_status": "analyzing",
        "research_outcome": "not_evaluated",
        "audit_status": "pending",
        "live_job": None,
        "training_jobs": 2,
        "llm_calls": 8,
        "gpu_seconds_left": 3500,
        "pause_after_step": False,
        "hypothesis": {"mechanism": "temporal pooling", "statement": "时序池化是否改善固定 gallery Top-1"},
        "candidates": [{"candidate_id": "c01", "status": "pilot_complete"}],
        "decisions": decisions,
        "evidence": [
            {
                "evidence_id": "ev_pilot",
                "candidate_id": "c01",
                "job_id": "j_pilot",
                "fidelity": "pilot",
                "evaluation_valid": True,
                "fixed_bank_top1": 0.362,
                "delta_vs_control_pp": 2.4,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
                "seed": 0,
            },
            {
                "evidence_id": "ev_zero",
                "candidate_id": "c01",
                "job_id": "j_full_zero",
                "fidelity": "full",
                "evaluation_valid": True,
                "fixed_bank_top1": 0.40,
                "delta_vs_control_pp": 0,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
                "seed": 0,
                "promotion": {"tier": "full", "confirmation": "none"},
            },
            {
                "evidence_id": "ev_neg",
                "candidate_id": "c02",
                "job_id": "j_full_neg",
                "fidelity": "full",
                "evaluation_valid": True,
                "fixed_bank_top1": 0.38,
                "delta_vs_control_pp": -2,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
                "seed": 0,
            },
            {
                "evidence_id": "ev_pilot_not_full",
                "candidate_id": "c03",
                "job_id": "j_other_pilot",
                "fidelity": "pilot",
                "evaluation_valid": True,
                "fixed_bank_top1": 0.9,
                "delta_vs_control_pp": 50,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
            },
            {
                "evidence_id": "ev_invalid",
                "candidate_id": "c04",
                "job_id": "j_invalid",
                "fidelity": "full",
                "evaluation_valid": False,
                "fixed_bank_top1": 0.99,
                "delta_vs_control_pp": 9,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
            },
            {
                "evidence_id": "ev_base",
                "candidate_id": "baseline",
                "job_id": "j_base",
                "fidelity": "full",
                "evaluation_valid": True,
                "fixed_bank_top1": 0.40,
                "delta_vs_control_pp": 0,
                "comparison": {"comparable": True},
                "comparison_status": "comparable",
            },
        ],
    }
    _write(camp / "campaign_state.json", state)
    _write(camp / "events.jsonl", json.dumps({"at": now, "event": "created"}) + "\n")
    workspace = camp / "candidates" / "c01"
    _write(workspace / "extension" / "eeg_candidate.py", "def build_encoder():\n    return None\n" + ("# demo padding\n" * 200))
    _write(workspace / "spec.json", {"parent_candidate_id": "baseline"})
    _write(workspace / "lineage.json", {"parent_candidate_id": "baseline", "parent_is_not_control": True})
    _write(workspace / "attempt.json", {"attempt_id": "att_demo", "candidate_id": "c01"})
    _write(workspace / "review.json", {"summary_zh": "演示审查通过"})
    job = camp / "jobs" / "j_pilot"
    _write(
        job / "job.json",
        {
            "job_id": "j_pilot",
            "candidate_id": "c01",
            "fidelity": "pilot",
            "seed": 0,
            "epochs": 3,
            "status": "finished",
            "started_at": now + 6,
        },
    )
    history = [
        {"epoch": 1, "train_loss": 2.10, "val_top1": 0.1, "val_top5": 0.2, "fixed_bank_top1": 0.320, "fixed_bank_top5": 0.5},
        {"epoch": 2, "train_loss": 0.0, "val_top1": 0.12, "val_top5": 0.22, "fixed_bank_top1": 0.348, "fixed_bank_top5": 0.55},
        {"epoch": 3, "train_loss": 1.61, "val_top1": 0.14, "val_top5": 0.24},
    ]
    (job / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    finished = camp / "jobs" / "j_full_zero"
    _write(
        finished / "job.json",
        {
            "job_id": "j_full_zero",
            "candidate_id": "c01",
            "fidelity": "full",
            "seed": 0,
            "epochs": 3,
            "status": "finished",
            "started_at": now + 8,
        },
    )
    (finished / "history.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss": 1.0, "val_top1": 0.2, "val_top5": 0.3, "fixed_bank_top1": 0.40, "fixed_bank_top5": 0.6}) + "\n",
        encoding="utf-8",
    )
    other = root / "other_campaign"
    other.mkdir(parents=True, exist_ok=True)
    _write(other / "campaign_state.json", {"status": "paused", "decisions": [], "candidates": [], "evidence": []})
    _write(other / "goal.json", {"objective": "other demo campaign"})
    _write(
        other / "jobs" / "secret_job" / "job.json",
        {"job_id": "secret_job", "candidate_id": "cx", "fidelity": "full", "status": "finished"},
    )
    (other / "jobs" / "secret_job" / "history.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss": 9.0, "val_top1": 0.9, "val_top5": 0.9, "fixed_bank_top1": 0.9, "fixed_bank_top5": 0.9}) + "\n",
        encoding="utf-8",
    )
    calls = [
        {"call_id": "call_done", "role": "experiment_designer", "success": True, "started_at": now + 3},
        {"call_id": "call_analyst", "role": "result_analyst", "success": True, "started_at": now + 7},
    ]
    (camp / "llm_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")
    from react_agent.eeg_research.agentic.ui_events import append_ui_event

    leftover_events = camp / "ui_events.jsonl"
    if leftover_events.is_file():
        leftover_events.unlink()
    append_ui_event(camp, "llm_call_started", role="result_analyst", call_id="call_live", status="active")
    append_ui_event(camp, "llm_call_finished", role="experiment_designer", call_id="call_done", status="completed")
    leftover_worker = camp / "worker.json"
    if leftover_worker.is_file():
        leftover_worker.unlink()
    return camp
