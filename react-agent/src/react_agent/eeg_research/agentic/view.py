"""Read-only page view of code-level campaigns. Test results are not part of it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.budget import snapshot as budget_snapshot
from react_agent.eeg_research.agentic.cli import DEFAULT_ROOT, status_view

ACTION_ZH = {
    "inspect_data": "核对数据",
    "retrieve_memory": "检索经验",
    "retrieve_methods": "检索方法卡",
    "diagnose_results": "诊断结果",
    "collect_diagnostics": "收集诊断包",
    "design_experiment": "设计实验",
    "propose_experiment": "提出实验",
    "implement_candidate": "编写候选代码",
    "repair_candidate": "修复候选",
    "run_pilot": "小规模试跑",
    "run_full": "完整训练",
    "replicate": "重复 seed",
    "audit_result": "审计结果",
    "stop": "停止",
}


def _read(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _created_at(camp: Path) -> float:
    events = camp / "events.jsonl"
    if events.is_file():
        with events.open(encoding="utf-8") as handle:
            line = handle.readline().strip()
        if line:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and row.get("at") is not None:
                return float(row["at"])
    state = camp / "campaign_state.json"
    if state.is_file():
        return state.stat().st_mtime
    return 0.0


def _updated_at(camp: Path) -> float:
    state = camp / "campaign_state.json"
    if state.is_file():
        return state.stat().st_mtime
    return 0.0


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
        comparable = bool((item.get("comparison") or {}).get("comparable")) or item.get("comparison_status") == "comparable"
        rows.append(
            {
                "evidence_id": item.get("evidence_id"),
                "candidate_id": item.get("candidate_id"),
                "fidelity": item.get("fidelity"),
                "fixed_bank_top1": item.get("fixed_bank_top1"),
                "delta_vs_control_pp": item.get("delta_vs_control_pp"),
                "evaluation_valid": item.get("evaluation_valid"),
                "comparable": comparable,
                "reason": item.get("reason"),
                "seed": item.get("seed"),
                "promotion_tier": (item.get("promotion") or {}).get("tier"),
                "confirmation": (item.get("promotion") or {}).get("confirmation"),
            }
        )
    pilot_rows = [row for row in rows if row.get("fidelity") == "pilot"]
    full_rows = [row for row in rows if row.get("fidelity") == "full"]
    ranked_full = [
        row
        for row in full_rows
        if row.get("evaluation_valid") and row.get("comparable") and row.get("candidate_id") != "baseline"
    ]
    ranked_full.sort(key=lambda row: (row.get("delta_vs_control_pp") is not None, row.get("delta_vs_control_pp") or -1e9), reverse=True)
    best_full = ranked_full[0] if ranked_full else None
    analyses = [
        item.get("summary") for item in state.get("evidence") or [] if item.get("kind") == "analysis"
    ]
    decisions = [
        {**row, "action_zh": ACTION_ZH.get(str(row.get("action")), row.get("action"))}
        for row in (state.get("decisions") or [])[-8:]
    ]
    live = state.get("live_job")
    live_status = _read(camp / "jobs" / str(live) / "status.json") if live else None
    budget = budget_snapshot(camp, state)
    return {
        "campaign_id": camp.name,
        "created_at": _created_at(camp),
        "updated_at": _updated_at(camp),
        "objective": goal.get("objective"),
        "research_scope": contract.get("research_scope"),
        "scope_zh": "多被试合训 · 图像留出验证（不是未见被试）" if contract.get("research_scope") == "pooled_subject_retrieval" else contract.get("research_scope"),
        "allowed_changes": goal.get("allowed_changes"),
        "status": view["status"],
        "detail": view.get("detail"),
        "stop_reason": state.get("stop_reason"),
        "pause_after_step": state.get("pause_after_step"),
        "hypothesis": state.get("hypothesis"),
        "live_job": live,
        "live_progress": live_status,
        "decisions": decisions,
        "candidates": candidates,
        "experiments": rows,
        "pilot_rows": pilot_rows,
        "full_rows": full_rows,
        "best_full": best_full,
        "analyses": analyses,
        "budget": {
            "training_jobs": budget.get("training_jobs_used"),
            "max_training_jobs": budget.get("training_jobs_limit"),
            "llm_calls": budget.get("llm_calls_used"),
            "max_llm_calls": budget.get("llm_calls_limit"),
            "gpu_seconds_used": budget.get("gpu_seconds_used"),
            "gpu_seconds_reserved": budget.get("gpu_seconds_reserved"),
            "gpu_seconds_left": budget.get("gpu_seconds_left"),
            "api_usd": cost.get("api_usd"),
        },
    }


def agentic_status(root: Path | None = None, campaign: str = "") -> dict[str, Any]:
    base = root or DEFAULT_ROOT
    if not base.is_dir():
        return {"campaigns": []}
    names = [campaign] if campaign else sorted(path.name for path in base.iterdir() if (path / "campaign_state.json").is_file())
    rows = [campaign_view(base / name) for name in names if (base / name / "campaign_state.json").is_file()]
    rows.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return {"campaigns": rows}


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
