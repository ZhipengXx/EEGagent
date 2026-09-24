"""Background worker. One process per campaign; training end triggers the next decision."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic import jobs
from react_agent.eeg_research.agentic.contract import public_contract
from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
from react_agent.eeg_research.agentic.loop import event, load_state, save_state, tick
from react_agent.eeg_research.agentic.native_patch import implement, review
from react_agent.eeg_training.protocol import Design, data_root

_TERMINAL = {"paused", "finished", "blocked", "cancelled"}
CODER_STEPS = 8


def _design(state: dict[str, Any]) -> Design:
    gpu = tuple(state.get("gpu") or (0,))
    return Design("eeg", "inter-subject", "all", gpu=gpu, gpu_seconds=float(state.get("gpu_seconds_left") or 1), policy="agentic")


def build_services(camp: Path) -> dict[str, Any]:
    coder = role_backend(camp, "candidate_coder")
    reviewer = role_backend(camp, "candidate_reviewer")
    analyst = role_backend(camp, "result_analyst")
    contract = public_contract(json.loads((camp / "evaluation_contract.json").read_text(encoding="utf-8")))
    summary = {key: contract.get(key) for key in ("research_scope", "primary_metric", "val_mode", "fingerprint")}

    def launch(camp_dir: Path, state: dict[str, Any], job_id: str, candidate_id: str, fidelity: str) -> dict[str, Any]:
        extension = None if candidate_id == "baseline" else camp_dir / "candidates" / candidate_id / "extension"
        return jobs.start_job(
            camp_dir / "jobs" / job_id,
            _design(state),
            candidate_id=candidate_id,
            extension=extension,
            fidelity=fidelity,
            root=data_root(),
        )

    def settle(camp_dir: Path, job_id: str) -> dict[str, Any]:
        return jobs.reconcile(camp_dir / "jobs" / job_id)

    def do_implement(camp_dir: Path, state: dict[str, Any]) -> None:
        index = len(state.get("candidates") or []) + 1
        while (camp_dir / "candidates" / f"c{index}").exists():
            index += 1
        candidate_id = f"c{index}"
        workspace = camp_dir / "candidates" / candidate_id
        spec = {"hypothesis": state.get("hypothesis"), "experiment": state.get("experiment")}
        (workspace).mkdir(parents=True, exist_ok=True)
        (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        max_calls = int(state.get("max_llm_calls", 100))

        def calls_left() -> int:
            cost_path = camp_dir / "cost.json"
            used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
            return max_calls - int(used)

        outcome = implement(workspace, spec, coder, max_steps=CODER_STEPS, calls_left=calls_left, reserve=2)
        state["llm_calls"] = int(state.get("llm_calls", 0)) + int(outcome.get("steps") or 0)
        state["llm_calls_left"] = int(state.get("llm_calls_left", 0)) - int(outcome.get("steps") or 0)
        row = {"candidate_id": candidate_id, "status": outcome["status"], "steps": outcome.get("steps")}
        if outcome["status"] == "ready_for_review":
            verdict = review(workspace, spec, summary, reviewer, getattr(reviewer, "model", ""))
            row["review"] = verdict["status"]
            row["review_format_failed"] = verdict["format_failed"]
            if verdict["status"] == "ready":
                state["candidate_ready"] = True
                state["candidate_id"] = candidate_id
                row["status"] = "ready"
            else:
                row["status"] = f"review_{verdict['status']}"
        if row["status"] != "ready":
            state["experiment_failed"] = True
        state.setdefault("candidates", []).append(row)
        state["evidence"].append(
            {
                "evidence_id": f"ev_impl_{candidate_id}",
                "kind": "implementation",
                "candidate_id": candidate_id,
                "summary": row,
            }
        )
        event(camp_dir, "implemented", **row)

    def analyze(camp_dir: Path, state: dict[str, Any]) -> None:
        latest = state["evidence"][-1]
        if not latest.get("evaluation_valid") or latest.get("candidate_id") == "baseline":
            return
        payload = {
            "latest": {key: latest.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "gallery_size", "delta_vs_control_pp", "control_id", "seed")},
            "hypothesis": state.get("hypothesis"),
            "controls": [
                {key: row.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "seed")}
                for row in state["evidence"]
                if row.get("candidate_id") == "baseline" and row.get("evaluation_valid")
            ],
        }
        try:
            reply = analyst(payload)
        except LlmUnavailable as exc:
            reply = {"hypothesis_assessment": "not_tested", "summary_zh": f"分析未完成：{exc}"}
        state["llm_calls"] = int(state.get("llm_calls", 0)) + 1
        state["llm_calls_left"] = int(state.get("llm_calls_left", 0)) - 1
        target = camp_dir / "analyses" / f"{latest['evidence_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reply, ensure_ascii=False, indent=2), encoding="utf-8")
        state["evidence"].append(
            {
                "evidence_id": f"ev_analysis_{latest['evidence_id']}",
                "kind": "analysis",
                "candidate_id": latest.get("candidate_id"),
                "summary": {key: reply.get(key) for key in ("hypothesis_assessment", "summary_zh", "suggested_next_actions")},
            }
        )

    return {"launch": launch, "settle": settle, "implement": do_implement, "analyze": analyze}


def run_worker(camp: Path, *, poll_seconds: float = 30.0, max_ticks: int = 200) -> dict[str, Any]:
    """Advance until a terminal state. A second worker for the same campaign exits at once."""
    lock_path = camp / "worker.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"status": "worker_already_running"}
    (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
    try:
        planner = role_backend(camp, "research_planner")
        services = build_services(camp)
    except LlmUnavailable as exc:
        state = load_state(camp)
        state["status"] = "blocked"
        state["detail"] = str(exc)
        save_state(camp, state)
        return state
    state = load_state(camp)
    if state.get("status") == "paused":
        state["status"] = "created"
        state["pause_after_step"] = False
        save_state(camp, state)
    for _ in range(max_ticks):
        state = tick(camp, planner, services=services)
        (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": time.time()}), encoding="utf-8")
        if state.get("status") in _TERMINAL:
            break
        if state.get("status") == "training":
            time.sleep(poll_seconds)
    return state
