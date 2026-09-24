"""Read-only page view of code-level campaigns. Test results are not part of it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.cli import DEFAULT_ROOT, status_view

ACTION_ZH = {
    "inspect_data": "核对数据",
    "retrieve_memory": "检索经验",
    "diagnose_results": "诊断结果",
    "propose_experiment": "提出实验",
    "implement_candidate": "编写候选代码",
    "run_pilot": "小规模试跑",
    "run_full": "完整训练",
    "replicate": "重复 seed",
    "stop": "停止",
}


def _read(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def campaign_view(camp: Path) -> dict[str, Any]:
    view = status_view(camp)
    goal = _read(camp / "goal.json") or {}
    contract = _read(camp / "evaluation_contract.json") or {}
    cost = _read(camp / "cost.json") or {}
    state = _read(camp / "campaign_state.json") or {}
    candidates = []
    for row in state.get("candidates") or []:
        workspace = camp / "candidates" / str(row.get("candidate_id"))
        entry = workspace / "extension" / "eeg_candidate.py"
        review = _read(workspace / "review.json") or {}
        candidates.append(
            {
                **row,
                "source": entry.read_text(encoding="utf-8")[:6000] if entry.is_file() else None,
                "review_summary": review.get("summary_zh"),
                "spec": _read(workspace / "spec.json"),
            }
        )
    rows = []
    for item in state.get("evidence") or []:
        if item.get("fidelity") is None:
            continue
        rows.append(
            {
                "evidence_id": item.get("evidence_id"),
                "candidate_id": item.get("candidate_id"),
                "fidelity": item.get("fidelity"),
                "fixed_bank_top1": item.get("fixed_bank_top1"),
                "delta_vs_control_pp": item.get("delta_vs_control_pp"),
                "evaluation_valid": item.get("evaluation_valid"),
                "reason": item.get("reason"),
                "seed": item.get("seed"),
            }
        )
    analyses = [
        item.get("summary") for item in state.get("evidence") or [] if item.get("kind") == "analysis"
    ]
    decisions = [
        {**row, "action_zh": ACTION_ZH.get(str(row.get("action")), row.get("action"))}
        for row in (state.get("decisions") or [])[-8:]
    ]
    live = state.get("live_job")
    live_status = _read(camp / "jobs" / str(live) / "status.json") if live else None
    return {
        "campaign_id": camp.name,
        "objective": goal.get("objective"),
        "research_scope": contract.get("research_scope"),
        "scope_zh": "多被试合训 · 图像留出验证（不是未见被试）" if contract.get("research_scope") == "pooled_subject_retrieval" else contract.get("research_scope"),
        "allowed_changes": goal.get("allowed_changes"),
        "status": view["status"],
        "detail": view.get("detail"),
        "pause_after_step": state.get("pause_after_step"),
        "hypothesis": state.get("hypothesis"),
        "live_job": live,
        "live_progress": live_status,
        "decisions": decisions,
        "candidates": candidates,
        "experiments": rows,
        "analyses": analyses,
        "budget": {
            "training_jobs": state.get("training_jobs"),
            "max_training_jobs": state.get("max_training_jobs"),
            "llm_calls": cost.get("llm_calls"),
            "max_llm_calls": state.get("max_llm_calls"),
            "gpu_seconds_used": cost.get("gpu_seconds_used"),
            "gpu_seconds_left": state.get("gpu_seconds_left"),
            "api_usd": cost.get("api_usd"),
        },
    }


def agentic_status(root: Path | None = None, campaign: str = "") -> dict[str, Any]:
    base = root or DEFAULT_ROOT
    if not base.is_dir():
        return {"campaigns": []}
    names = [campaign] if campaign else sorted(path.name for path in base.iterdir() if (path / "campaign_state.json").is_file())
    return {"campaigns": [campaign_view(base / name) for name in names if (base / name / "campaign_state.json").is_file()]}


def control(action: str, campaign: str, root: Path | None = None) -> dict[str, Any]:
    """Pause after this step, resume, or stop the current job. Refresh does not call this."""
    from react_agent.eeg_research.agentic.cli import main

    if action not in {"pause", "resume", "stop"}:
        return {"ok": False, "error": "bad_action"}
    base = root or DEFAULT_ROOT
    if not (base / campaign / "campaign_state.json").is_file():
        return {"ok": False, "error": "campaign_missing"}
    code = main([action, "--campaign", campaign, "--root", str(base)])
    return {"ok": code == 0, "campaign": campaign_view(base / campaign)}
