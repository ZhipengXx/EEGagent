"""Campaign loop. HTTP creates the campaign. The worker advances it one step at a time."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.coder import apply_candidate_patch, finish_patch, run_candidate_check
from react_agent.eeg_research.agentic.contract import public_contract
from react_agent.eeg_research.agentic.memory import episode, retrieve
from react_agent.eeg_research.agentic.planner import available_actions, decide, evidence_count
from react_agent.eeg_research.agentic.runner import accept_job, comparable

_TERMINAL = {"paused", "finished", "blocked", "cancelled"}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def event(camp: Path, kind: str, **fields: Any) -> None:
    row = {"at": time.time(), "event": kind, **fields}
    with (camp / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def create_campaign(
    root: Path,
    *,
    goal: dict[str, Any],
    contract: dict[str, Any],
    request_id: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Idempotent create. The same request id does not open a second campaign."""
    index = root / "request_index.json"
    known = _read(index)
    if request_id in known:
        return _read(Path(known[request_id]) / "campaign_state.json")
    camp = root / goal["goal_id"]
    existing = _read(camp / "campaign_state.json")
    if existing:
        known[request_id] = str(camp)
        _write(index, known)
        return existing
    state = {
        "goal_id": goal["goal_id"],
        "request_id": request_id,
        "status": "created",
        "contract_fingerprint": contract["fingerprint"],
        "execution_fingerprint": None if protocol is None else protocol.get("fingerprint"),
        "evidence": [],
        "memory": [],
        "training_jobs": 0,
        "llm_calls": 0,
        "max_training_jobs": int(goal.get("max_training_jobs", 10)),
        "max_llm_calls": int(goal.get("max_llm_calls", 100)),
        "max_candidates": int(goal.get("max_candidates", 4)),
        "gpu_seconds_left": float(goal.get("max_gpu_seconds", 28800)),
        "llm_calls_left": int(goal.get("max_llm_calls", 100)),
        "hypothesis": None,
        "experiment": None,
        "candidate_ready": False,
        "candidate_id": None,
        "candidates": [],
        "live_job": None,
        "decisions": [],
        "pause_after_step": False,
    }
    _write(camp / "goal.json", goal)
    _write(camp / "resolved_goal.json", goal)
    _write(camp / "evaluation_contract.json", contract)
    if protocol is not None:
        _write(camp / "execution_protocol.json", protocol)
    _write(camp / "campaign_state.json", state)
    known[request_id] = str(camp)
    _write(index, known)
    event(camp, "created", fingerprint=contract["fingerprint"])
    return state


def load_state(camp: Path) -> dict[str, Any]:
    return _read(camp / "campaign_state.json")


def _decision_number(stem: str) -> int:
    digits = stem[1:] if stem.startswith("d") else stem
    return int(digits) if digits.isdigit() else 0


def incomplete_candidate_id(camp: Path, state: dict[str, Any]) -> str | None:
    """Directory with a spec and no coder log that never entered state or evidence."""
    recorded = {row.get("candidate_id") for row in state.get("candidates") or []}
    evidenced = {row.get("candidate_id") for row in state.get("evidence") or []}
    root = camp / "candidates"
    if not root.is_dir():
        return None
    names = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not (name.startswith("c") and name[1:].isdigit()):
            continue
        if name in recorded or name in evidenced:
            continue
        if (path / "spec.json").is_file() and not (path / "coder_log.jsonl").is_file():
            names.append(name)
    if not names:
        return None
    return sorted(names, key=lambda name: int(name[1:]))[0]


def pending_implement(camp: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    """Last decision asked for a candidate that was never finished."""
    decisions = state.get("decisions") or []
    if not decisions:
        return None
    last = decisions[-1]
    if last.get("action") != "implement_candidate" or not last.get("ok"):
        return None
    if incomplete_candidate_id(camp, state) is None:
        return None
    return last


def align_interrupt(camp: Path) -> dict[str, Any]:
    """Merge decision files the state does not yet list. Does not call the planner or touch the ledger.

    paused, cancelled, and finished stay as they are. Stop and pause are not rewritten here.
    """
    state = load_state(camp)
    status = state.get("status")
    known = {row.get("decision_id") for row in state.get("decisions") or []}
    folder = camp / "decisions"
    added = False
    if folder.is_dir():
        paths = sorted(folder.glob("d*.json"), key=lambda path: _decision_number(path.stem))
        for path in paths:
            payload = _read(path)
            record = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            if not record:
                continue
            decision_id = str(record.get("decision_id") or path.stem)
            if decision_id in known:
                continue
            row = dict(record)
            row["decision_id"] = decision_id
            state.setdefault("decisions", []).append(row)
            known.add(decision_id)
            added = True
    if added:
        state["status"] = status
        save_state(camp, state)
        state = load_state(camp)
        if status in {"paused", "cancelled", "finished"}:
            state["status"] = status
            _write(camp / "campaign_state.json", state)
    return load_state(camp)


def request_control(camp: Path, action: str) -> None:
    """User controls live in their own file so a running worker cannot overwrite them."""
    _write(camp / "control.json", {"action": action, "at": time.time()})


def _apply_control(camp: Path, state: dict[str, Any]) -> None:
    control = _read(camp / "control.json")
    action = control.get("action")
    if action == "pause":
        state["pause_after_step"] = True
    elif action == "stop":
        state["status"] = "cancelled"
    elif action == "resume":
        state["pause_after_step"] = False


def save_state(camp: Path, state: dict[str, Any]) -> None:
    _apply_control(camp, state)
    if state.get("pause_after_step") and state.get("status") not in _TERMINAL and not state.get("live_job"):
        state["status"] = "paused"
    _write(camp / "campaign_state.json", state)


def _dev_row(row: dict[str, Any]) -> dict[str, Any]:
    """Allowlist view of one evidence row."""
    keep = (
        "evidence_id",
        "candidate_id",
        "fidelity",
        "evaluation_valid",
        "reason",
        "fixed_bank_top1",
        "fixed_bank_top5",
        "gallery_size",
        "delta_vs_control_pp",
        "control_id",
        "curve",
        "summary",
        "kind",
        "seed",
    )
    return {key: row.get(key) for key in keep if key in row}


def observation(camp: Path) -> dict[str, Any]:
    state = load_state(camp)
    contract = public_contract(_read(camp / "evaluation_contract.json"))
    actions = available_actions(state)
    evidence = [_dev_row(row) for row in state.get("evidence") or []]
    from react_agent.eeg_research.agentic.interface import CANDIDATE_INTERFACE

    return {
        "goal": {key: value for key, value in _read(camp / "goal.json").items() if "test" not in key},
        "contract": contract,
        "candidate_interface": CANDIDATE_INTERFACE,
        "evidence": evidence,
        "candidates": state.get("candidates") or [],
        "hypothesis": state.get("hypothesis"),
        "experiment": state.get("experiment"),
        "candidate_ready": state.get("candidate_ready"),
        "available_actions": actions,
        "budget": {
            "training_jobs_left": int(state.get("max_training_jobs", 0)) - int(state.get("training_jobs", 0)),
            "gpu_seconds_left": state.get("gpu_seconds_left"),
            "llm_calls_left": state.get("llm_calls_left"),
        },
        "memory": retrieve(
            state.get("memory") or [],
            task_hash=str(state.get("goal_id")),
            fingerprint=str(state.get("contract_fingerprint")),
        ),
        "recent_decisions": (state.get("decisions") or [])[-4:],
        "last_local_result": state.get("last_local_result"),
        "_known_evidence_ids": [row.get("evidence_id") for row in evidence],
    }


Services = dict[str, Callable[..., Any]]


def tick(camp: Path, backend: Any, runner: Any | None = None, services: Services | None = None) -> dict[str, Any]:
    """One research step. A live job is reconciled first and never double-started."""
    services = services or {}
    state = load_state(camp)
    if state.get("status") in _TERMINAL:
        return state
    if state.get("live_job"):
        settle = services.get("settle")
        if settle is None:
            state["status"] = "training"
            save_state(camp, state)
            return state
        record = settle(camp, state["live_job"])
        if record.get("status") == "running":
            state["status"] = "training"
            state["heartbeat"] = record.get("heartbeat")
            save_state(camp, state)
            return state
        _record_job(camp, state, record)
        state["live_job"] = None
        state["status"] = "analyzing"
        save_state(camp, state)
        analyst = services.get("analyze")
        if analyst is not None:
            analyst(camp, state)
            save_state(camp, state)
        if state.get("pause_after_step"):
            state["status"] = "paused"
            save_state(camp, state)
        return state
    ledger = _read(camp / "cost.json").get("llm_calls")
    if isinstance(ledger, int):
        state["llm_calls"] = ledger
        state["llm_calls_left"] = int(state.get("max_llm_calls", 100)) - ledger
    queued = state.get("queued") or []
    if queued and services.get("launch") is not None:
        candidate_id, fidelity = queued.pop(0)
        state["queued"] = queued
        _launch(camp, state, services, candidate_id, fidelity)
        save_state(camp, state)
        return state
    if pending_implement(camp, state) is not None and services.get("implement") is not None:
        if state.get("status") not in {"paused", "cancelled"}:
            state["status"] = "planning"
            services["implement"](camp, state)
            if state.get("pause_after_step") and state["status"] not in _TERMINAL and not state.get("live_job"):
                state["status"] = "paused"
            save_state(camp, state)
            return state
    if state.get("llm_calls_left", 1) <= 0:
        state["status"] = "blocked"
        state["detail"] = "budget_exhausted"
        save_state(camp, state)
        return state
    obs = observation(camp)
    state["llm_calls"] = int(state.get("llm_calls", 0)) + 1
    state["llm_calls_left"] = int(state.get("llm_calls_left", 1)) - 1
    decision = decide(obs, backend)
    if decision.get("repairs"):
        state["llm_calls"] += 1
        state["llm_calls_left"] -= 1
    record = {
        "decision_id": f"d{len(state.get('decisions') or []) + 1}",
        "action": decision.get("action"),
        "ok": decision.get("ok"),
        "reason_zh": decision.get("reason_zh"),
        "evidence_ids": (decision.get("raw") or {}).get("evidence_ids") or [],
        "detail": decision.get("detail"),
        "evidence_count": evidence_count(state),
    }
    state["decisions"].append(record)
    _write(camp / "decisions" / f"{record['decision_id']}.json", {"decision": record, "raw": decision.get("raw")})
    event(camp, "decision", **record)
    if not decision.get("ok"):
        state["status"] = "blocked"
        state["detail"] = decision.get("detail")
        save_state(camp, state)
        return state
    action = decision["action"]
    state["status"] = "planning"
    if action == "stop":
        state["status"] = "finished"
        state["stop_reason"] = (decision.get("raw") or {}).get("stop_reason") or decision.get("reason_zh")
    elif action == "inspect_data":
        state["data_audit"] = _inspect(camp)
        _append_derived(state, {"evidence_id": f"ev_audit_{len(state['evidence']) + 1}", "kind": "data_audit", "summary": state["data_audit"]})
    elif action == "retrieve_memory":
        state["memory_hits"] = [row.get("candidate_id") for row in obs["memory"]]
    elif action == "diagnose_results":
        _append_derived(state, _diagnose(camp, state))
    elif action == "propose_experiment":
        raw = decision.get("raw") or {}
        state["hypothesis"] = raw.get("hypothesis_draft") or {"mechanism": decision.get("reason_zh")}
        state["experiment"] = raw.get("experiment_draft") or {"initial_fidelity": "pilot"}
        state["experiment"]["parent_candidate_id"] = state["experiment"].get("parent_candidate_id") or "baseline"
        state["candidate_ready"] = False
        state["experiment_failed"] = False
        _write(camp / "hypotheses" / f"h{len(state['decisions'])}.json", {"hypothesis": state["hypothesis"], "experiment": state["experiment"]})
    elif action == "implement_candidate":
        implementer = services.get("implement")
        if implementer is None:
            _implement_inline(camp, state, decision)
        else:
            implementer(camp, state)
    elif action in {"run_pilot", "run_full", "replicate"}:
        _train(camp, state, action, runner, services)
    if state.get("pause_after_step") and state["status"] not in _TERMINAL and not state.get("live_job"):
        state["status"] = "paused"
    save_state(camp, state)
    return state


def _append_derived(state: dict[str, Any], row: dict[str, Any]) -> None:
    """Skip a derived row whose content matches the last row of its kind."""
    body = {key: value for key, value in row.items() if key != "evidence_id"}
    for old in reversed(state.get("evidence") or []):
        if old.get("kind") == row.get("kind"):
            if {key: value for key, value in old.items() if key != "evidence_id"} == body:
                state["last_local_result"] = {"kind": row.get("kind"), "new_information": False, "same_as": old.get("evidence_id")}
                return
            break
    state["evidence"].append(row)
    state["last_local_result"] = {"kind": row.get("kind"), "new_information": True, "evidence_id": row.get("evidence_id")}


def _inspect(camp: Path) -> dict[str, Any]:
    contract = _read(camp / "evaluation_contract.json")
    roles: dict[str, int] = {}
    missing = 0
    for row in contract.get("files") or []:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
        if not Path(row["path"]).is_file():
            missing += 1
    return {
        "research_scope": contract.get("research_scope"),
        "val_mode": contract.get("val_mode"),
        "files_by_role": roles,
        "missing_files": missing,
        "feature_caches_present": all(Path(path).is_file() for path in contract.get("feature_caches") or []),
    }


def _diagnose(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Curve summary from history files. No checkpoint means unavailable."""
    curves = []
    for row in state.get("evidence") or []:
        job = row.get("job_dir")
        if not job:
            continue
        history = Path(job) / "history.jsonl"
        if not history.is_file():
            continue
        points = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not points:
            continue
        best = max(points, key=lambda item: float(item.get("fixed_bank_top1") or 0.0))
        curves.append(
            {
                "candidate_id": row.get("candidate_id"),
                "fidelity": row.get("fidelity"),
                "epochs": len(points),
                "best_epoch": best.get("epoch"),
                "best_fixed_bank_top1": best.get("fixed_bank_top1"),
                "first_train_loss": points[0].get("train_loss"),
                "last_train_loss": points[-1].get("train_loss"),
                "fixed_bank_trend": [round(float(item.get("fixed_bank_top1") or 0.0), 5) for item in points],
            }
        )
    status = "ready" if curves else "unavailable"
    return {"evidence_id": f"ev_diag_{len(state['evidence']) + 1}", "kind": "learning_profile", "status": status, "curve": curves}


def _implement_inline(camp: Path, state: dict[str, Any], decision: dict[str, Any]) -> None:
    """Direct patch from one reply. Used by synthetic tests; the worker uses NativePatch."""
    candidate_id = f"c{len(state.get('candidates') or []) + 1}"
    workspace = camp / "candidates" / candidate_id
    raw = decision.get("raw") or {}
    content = raw.get("content")
    if not content:
        state["detail"] = "implementation_failed"
        return
    applied = apply_candidate_patch(workspace, raw.get("relative") or "extension/eeg_candidate.py", content, raw.get("expected_base_hash") or "")
    if not applied.get("ok"):
        state["detail"] = applied.get("error")
        return
    checked = run_candidate_check(workspace)
    (workspace / "checks.json").write_text(json.dumps(checked, ensure_ascii=False), encoding="utf-8")
    state["candidates"].append({"candidate_id": candidate_id, "status": "checked" if checked.get("ok") else "implementation_failed"})
    if not checked.get("ok"):
        state["detail"] = "implementation_failed"
        return
    finished = finish_patch(workspace, raw.get("reason_zh") or "candidate")
    state["candidate_ready"] = bool(finished.get("ok"))
    state["candidate_id"] = candidate_id
    state["status"] = "checking"


def _record_job(camp: Path, state: dict[str, Any], record: dict[str, Any]) -> None:
    result = dict(record.get("result") or {})
    result["evidence_id"] = f"ev_{record.get('job_id')}"
    result["candidate_id"] = record.get("candidate_id")
    result["job_dir"] = str(camp / "jobs" / str(record.get("job_id")))
    result["seed"] = record.get("seed")
    result["job_status"] = record.get("status")
    if result.get("evaluation_valid"):
        result["contract_fingerprint"] = result.get("execution_fingerprint") or state.get("contract_fingerprint")
    state["gpu_seconds_left"] = float(state.get("gpu_seconds_left", 0)) - float(record.get("gpu_seconds") or 0)
    cost_path = camp / "cost.json"
    cost = _read(cost_path)
    cost["gpu_seconds_used"] = float(cost.get("gpu_seconds_used", 0.0)) + float(record.get("gpu_seconds") or 0)
    cost["training_jobs"] = int(state.get("training_jobs", 0))
    cost.setdefault("api_usd", None)
    _write(cost_path, cost)
    if result.get("evaluation_valid"):
        controls = [
            row
            for row in state["evidence"]
            if row.get("candidate_id") == "baseline"
            and row.get("fidelity") == result.get("fidelity")
            and row.get("evaluation_valid")
            and row.get("gallery_size") == result.get("gallery_size")
            and row.get("seed") == result.get("seed")
        ]
        if controls and result.get("candidate_id") != "baseline":
            result["control_id"] = controls[-1]["evidence_id"]
            result["delta_vs_control_pp"] = round(100 * (result["fixed_bank_top1"] - controls[-1]["fixed_bank_top1"]), 4)
        state["memory"].append(
            episode(
                task_hash=str(state.get("goal_id")),
                candidate_id=str(result.get("candidate_id")),
                kind="exploratory_result",
                fidelity=str(result.get("fidelity")),
                seed=int(record.get("seed") or 0),
                metric=result.get("fixed_bank_top1"),
                contract_fingerprint=str(state.get("contract_fingerprint")),
                artifact=str(Path(result["job_dir"]) / "metrics.json"),
            )
        )
    else:
        state["memory"].append(
            episode(
                task_hash=str(state.get("goal_id")),
                candidate_id=str(result.get("candidate_id")),
                kind="implementation_failure",
                fidelity=str(result.get("fidelity")),
                seed=int(record.get("seed") or 0),
                metric=None,
                contract_fingerprint=str(state.get("contract_fingerprint")),
                artifact=str(result["job_dir"]),
            )
        )
    state["evidence"].append(result)
    event(camp, "job_settled", job_id=record.get("job_id"), status=record.get("status"), valid=result.get("evaluation_valid"))


def _launch(camp: Path, state: dict[str, Any], services: Services, candidate_id: str, fidelity: str) -> None:
    """Start one job. The control runs at the same fidelity before its candidate."""
    job_index = int(state["training_jobs"]) + 1
    job_id = f"j{job_index}_{candidate_id}_{fidelity}"
    record = services["launch"](camp, state, job_id, candidate_id, fidelity)
    state["training_jobs"] = job_index
    state["live_job"] = record["job_id"]
    state["status"] = "training"
    event(camp, "job_started", job_id=record["job_id"], candidate_id=candidate_id, fidelity=fidelity)


def _train(camp: Path, state: dict[str, Any], action: str, runner: Any, services: Services) -> None:
    if state.get("gpu_seconds_left", 0) <= 0 or state["training_jobs"] >= state["max_training_jobs"]:
        state["status"] = "blocked"
        state["detail"] = "budget_exhausted"
        return
    fidelity = "pilot" if action == "run_pilot" else "full"
    candidate_id = state.get("candidate_id") or "c1"
    job_index = state["training_jobs"] + 1
    if services.get("launch") is not None:
        has_control = any(
            row.get("candidate_id") == "baseline" and row.get("fidelity") == fidelity and row.get("evaluation_valid")
            for row in state.get("evidence") or []
        )
        if not has_control and state["training_jobs"] + 2 <= state["max_training_jobs"]:
            state["queued"] = [[candidate_id, fidelity]]
            _launch(camp, state, services, "baseline", fidelity)
        else:
            _launch(camp, state, services, candidate_id, fidelity)
        return
    job = camp / "jobs" / f"j{job_index}"
    job.mkdir(parents=True, exist_ok=True)
    manifest = _read(camp / "candidates" / candidate_id / "source_manifest.json")
    if runner is None:
        state["status"] = "blocked"
        state["detail"] = "no_job_launcher"
        return
    result = runner(job, manifest, fidelity)
    state["training_jobs"] = job_index
    accepted = accept_job(job, manifest, fidelity)
    _record_job(
        camp,
        state,
        {
            "job_id": job.name,
            "candidate_id": candidate_id,
            "seed": result.get("seed", 0),
            "status": "finished" if accepted.get("evaluation_valid") else "invalid",
            "gpu_seconds": result.get("gpu_seconds", 0),
            "result": accepted,
        },
    )
    state["evidence"][-1]["evidence_id"] = f"ev_{job_index}"
    others = [row for row in state["evidence"][:-1] if comparable(row, state["evidence"][-1])]
    if others and state["evidence"][-1].get("evaluation_valid"):
        state["evidence"][-1]["delta"] = state["evidence"][-1]["fixed_bank_top1"] - others[0]["fixed_bank_top1"]
    state["status"] = "analyzing"
