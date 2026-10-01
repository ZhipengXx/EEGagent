"""Read-only page view of code-level campaigns. Test results are not part of it."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic import jobs as jobs_mod
from react_agent.eeg_research.agentic.budget import snapshot as budget_snapshot
from react_agent.eeg_research.agentic.cli import DEFAULT_ROOT, status_view
from react_agent.eeg_research.agentic.paths import resolve_campaign, resolve_candidate_dir, safe_name

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

SOURCE_PREVIEW_CHARS = 6000


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


def finite_delta(value: Any) -> float | None:
    """None and non-finite values are missing. Zero is a real delta."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def rank_full_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valid, comparable, non-baseline Full rows. 0 pp ranks above -2 pp."""
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("evaluation_valid"):
            continue
        if not row.get("comparable"):
            continue
        if row.get("candidate_id") == "baseline":
            continue
        if row.get("fidelity") != "full":
            continue
        delta = finite_delta(row.get("delta_vs_control_pp"))
        if delta is None:
            continue
        ranked = dict(row)
        ranked["delta_vs_control_pp"] = delta
        ranked["beats_control"] = delta > 0
        eligible.append(ranked)
    eligible.sort(key=lambda row: row["delta_vs_control_pp"], reverse=True)
    return eligible


def worker_health(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Read-only process projection. An old heartbeat is not a dead worker."""
    worker = {}
    path = camp / "worker.json"
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            worker = loaded
    try:
        pid = int(worker.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    process_state = jobs_mod._process_state(pid) if pid > 0 else None
    if pid <= 0:
        liveness = "unknown" if not worker else "missing"
    elif process_state is None:
        liveness = "unknown"
    elif process_state in {"R", "S", "D"}:
        liveness = "alive"
    elif process_state == "Z":
        liveness = "zombie"
    else:
        liveness = "exited"
    disconnected = bool(pid > 0 and jobs_mod.worker_disconnected(pid))
    return {
        "pid": pid or None,
        "process_state": process_state,
        "process_liveness": liveness,
        "last_progress_at": worker.get("heartbeat"),
        "worker_disconnected": disconnected,
        "note": "heartbeat updates after a tick; a blocking LLM call can look stale while the process is alive",
    }


def _experiment_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in state.get("evidence") or []:
        if item.get("fidelity") is None:
            continue
        comparable = bool((item.get("comparison") or {}).get("comparable")) or item.get("comparison_status") == "comparable"
        rows.append(
            {
                "evidence_id": item.get("evidence_id"),
                "candidate_id": item.get("candidate_id"),
                "job_id": item.get("job_id"),
                "fidelity": item.get("fidelity"),
                "fixed_bank_top1": item.get("fixed_bank_top1"),
                "delta_vs_control_pp": item.get("delta_vs_control_pp"),
                "evaluation_valid": item.get("evaluation_valid"),
                "comparable": comparable,
                "reason": item.get("reason"),
                "seed": item.get("seed"),
                "attempt_id": item.get("attempt_id"),
                "parent_candidate_id": item.get("parent_candidate_id"),
                "promotion_tier": (item.get("promotion") or {}).get("tier"),
                "confirmation": (item.get("promotion") or {}).get("confirmation"),
            }
        )
    return rows


def _candidate_rows(camp: Path, state: dict[str, Any], *, include_source: bool) -> list[dict[str, Any]]:
    rows = []
    for row in state.get("candidates") or []:
        candidate_id = str(row.get("candidate_id") or "")
        workspace = camp / "candidates" / candidate_id
        entry = workspace / "extension" / "eeg_candidate.py"
        review = _read(workspace / "review.json") or {}
        spec = _read(workspace / "spec.json") or {}
        lineage = _read(workspace / "lineage.json") or {}
        attempt = _read(workspace / "attempt.json") or {}
        parent_id = None
        if isinstance(spec, dict):
            parent_id = spec.get("parent_candidate_id")
        if parent_id is None and isinstance(lineage, dict):
            parent_id = lineage.get("parent_candidate_id")
        item = {
            **row,
            "review_summary": review.get("summary_zh") if isinstance(review, dict) else None,
            "spec": spec if isinstance(spec, dict) else None,
            "parent_candidate_id": parent_id,
            "lineage": lineage if isinstance(lineage, dict) else None,
            "attempt_id": attempt.get("attempt_id") if isinstance(attempt, dict) else None,
            "source": None,
            "source_loaded": include_source,
            "source_available": entry.is_file(),
            "source_truncated": False,
            "source_chars": 0,
        }
        if include_source and entry.is_file():
            text = entry.read_text(encoding="utf-8")
            item["source_chars"] = len(text)
            item["source_truncated"] = len(text) > SOURCE_PREVIEW_CHARS
            item["source"] = text[:SOURCE_PREVIEW_CHARS]
        rows.append(item)
    return rows


def _job_index(camp: Path) -> list[dict[str, Any]]:
    root = camp / "jobs"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        record = _read(path / "job.json") or {}
        if not isinstance(record, dict):
            record = {}
        rows.append(
            {
                "job_id": record.get("job_id") or path.name,
                "candidate_id": record.get("candidate_id"),
                "fidelity": record.get("fidelity"),
                "seed": record.get("seed") if record.get("seed") is not None else record.get("training_seed"),
                "status": record.get("status"),
                "epochs": record.get("epochs"),
            }
        )
    return rows


def campaign_summary(camp: Path) -> dict[str, Any]:
    view = status_view(camp)
    goal = _read(camp / "goal.json") or {}
    contract = _read(camp / "evaluation_contract.json") or {}
    state = _read(camp / "campaign_state.json") or {}
    return {
        "campaign_id": camp.name,
        "created_at": _created_at(camp),
        "updated_at": _updated_at(camp),
        "objective": goal.get("objective"),
        "status": view["status"],
        "detail": view.get("detail"),
        "execution_status": state.get("execution_status") or view["status"],
        "research_scope": contract.get("research_scope"),
        "live_job": state.get("live_job"),
        "termination_reason": state.get("termination_reason") or state.get("stop_reason"),
    }


def campaign_view(camp: Path, *, include_source: bool = True) -> dict[str, Any]:
    view = status_view(camp)
    goal = _read(camp / "goal.json") or {}
    contract = _read(camp / "evaluation_contract.json") or {}
    cost = _read(camp / "cost.json") or {}
    state = _read(camp / "campaign_state.json") or {}
    candidates = _candidate_rows(camp, state, include_source=include_source)
    rows = _experiment_rows(state)
    by_candidate = {item.get("candidate_id"): item for item in candidates}
    for row in rows:
        candidate = by_candidate.get(row.get("candidate_id"))
        if candidate and not row.get("parent_candidate_id"):
            row["parent_candidate_id"] = candidate.get("parent_candidate_id")
        if candidate and not row.get("attempt_id"):
            row["attempt_id"] = candidate.get("attempt_id")
    pilot_rows = [row for row in rows if row.get("fidelity") == "pilot"]
    full_rows = [row for row in rows if row.get("fidelity") == "full"]
    ranked_full = rank_full_rows(full_rows)
    best_full = ranked_full[0] if ranked_full else None
    if best_full is not None:
        best_full = dict(best_full)
        if not best_full.get("beats_control"):
            best_full["rank_note"] = "尚未优于对照"
            best_full["research_success"] = False
        else:
            best_full["rank_note"] = None
            best_full["research_success"] = False
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
    from react_agent.eeg_research.agentic.loop import refresh_audit_freshness

    refresh_audit_freshness(state)
    audit_status = state.get("audit_status") or "pending"
    health = worker_health(camp, state)
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
        "execution_status": state.get("execution_status") or view["status"],
        "research_outcome": state.get("research_outcome") or "not_evaluated",
        "audit_status": audit_status,
        "termination_reason": state.get("termination_reason") or state.get("stop_reason"),
        "stop_reason": state.get("stop_reason"),
        "pause_after_step": state.get("pause_after_step"),
        "hypothesis": state.get("hypothesis"),
        "live_job": live,
        "live_progress": live_status,
        "jobs": _job_index(camp),
        "health": health,
        "decisions": decisions,
        "decision_count": len(state.get("decisions") or []),
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


def candidate_source(root: Path, campaign: str, candidate_id: str, *, preview: bool = False) -> dict[str, Any]:
    camp = resolve_campaign(root, campaign)
    if camp is None:
        return {"ok": False, "error": "campaign_missing"}
    workspace = resolve_candidate_dir(camp, candidate_id)
    if workspace is None:
        return {"ok": False, "error": "candidate_missing"}
    entry = workspace / "extension" / "eeg_candidate.py"
    if not entry.is_file():
        return {"ok": True, "candidate_id": workspace.name, "source": None, "source_truncated": False, "source_chars": 0}
    text = entry.read_text(encoding="utf-8")
    if preview:
        return {
            "ok": True,
            "candidate_id": workspace.name,
            "source": text[:SOURCE_PREVIEW_CHARS],
            "source_truncated": len(text) > SOURCE_PREVIEW_CHARS,
            "source_chars": len(text),
        }
    return {
        "ok": True,
        "candidate_id": workspace.name,
        "source": text,
        "source_truncated": False,
        "source_chars": len(text),
    }


def agentic_status(root: Path | None = None, campaign: str = "") -> dict[str, Any]:
    base = root or DEFAULT_ROOT
    if not base.is_dir():
        return {"campaigns": []}
    if campaign:
        camp = resolve_campaign(base, campaign)
        if camp is None:
            return {"campaigns": []}
        return {"campaigns": [campaign_view(camp, include_source=False)]}
    names = sorted(path.name for path in base.iterdir() if (path / "campaign_state.json").is_file())
    rows = [campaign_summary(base / name) for name in names if safe_name(name)]
    rows.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return {"campaigns": rows}


def control(action: str, campaign: str, root: Path | None = None) -> dict[str, Any]:
    """Pause after this step, resume, or stop the current job. Refresh does not call this."""
    from react_agent.eeg_research.agentic.cli import main

    if action not in {"pause", "resume", "stop"}:
        return {"ok": False, "error": "bad_action"}
    base = root or DEFAULT_ROOT
    if resolve_campaign(base, campaign) is None:
        return {"ok": False, "error": "campaign_missing"}
    code = main([action, "--campaign", campaign, "--root", str(base)])
    return {"ok": code == 0, "campaign": campaign_view(base / campaign)}
