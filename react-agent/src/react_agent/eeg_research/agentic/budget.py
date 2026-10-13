"""Campaign resource ledger. Attempts charge; JSON repair does not bypass the cap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COST = "cost.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def snapshot(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Used, reserved, and remaining resources. Missing files do not reset counters."""
    cost = _read(camp / COST)
    gpu_limit = float(state.get("max_gpu_seconds") or cost.get("max_gpu_seconds") or 0.0)
    gpu_used = float(cost.get("gpu_seconds_used") or 0.0)
    gpu_reserved = float(state.get("gpu_seconds_reserved") or 0.0)
    llm_limit = int(state.get("max_llm_calls") or 0)
    llm_used = int(cost.get("llm_calls") if isinstance(cost.get("llm_calls"), int) else state.get("llm_calls") or 0)
    jobs_limit = int(state.get("max_training_jobs") or 0)
    jobs_used = int(state.get("training_jobs") or cost.get("training_jobs") or 0)
    gpu_left = float(state.get("gpu_seconds_left") if state.get("gpu_seconds_left") is not None else max(0.0, gpu_limit - gpu_used - gpu_reserved))
    result = {
        "gpu_seconds_used": gpu_used,
        "gpu_seconds_reserved": gpu_reserved,
        "gpu_seconds_left": gpu_left,
        "gpu_seconds_limit": gpu_limit,
        "llm_calls_used": llm_used,
        "llm_calls_left": llm_limit - llm_used,
        "llm_calls_limit": llm_limit,
        "training_jobs_used": jobs_used,
        "training_jobs_left": jobs_limit - jobs_used,
        "training_jobs_limit": jobs_limit,
        "baseline_jobs": int(state.get("baseline_jobs") or 0),
    }
    from .owner_budget import enabled, snapshot as owner_snapshot, current_owner
    if enabled(camp):
        active = current_owner(camp)
        pointer = state.get("selected_plan") or {}
        owner_id = active.get("owner_id") if active else pointer.get("continuation_owner") or pointer.get("budget_owner")
        result["owner_budget"] = owner_snapshot(camp,state,owner_id=owner_id)
    return result


def reserve_gpu(state: dict[str, Any], seconds: float) -> bool:
    """Reserve GPU seconds for a comparison request. False when the remainder cannot pay."""
    left = float(state.get("gpu_seconds_left") or 0.0) - float(state.get("gpu_seconds_reserved") or 0.0)
    if seconds > left:
        return False
    state["gpu_seconds_reserved"] = float(state.get("gpu_seconds_reserved") or 0.0) + seconds
    return True


def release_gpu(state: dict[str, Any], seconds: float) -> None:
    """Drop a reservation after settle or cancel."""
    state["gpu_seconds_reserved"] = max(0.0, float(state.get("gpu_seconds_reserved") or 0.0) - seconds)


def charge_gpu(camp: Path, state: dict[str, Any], seconds: float, *, job_id: str | None = None) -> bool:
    """Move spent seconds from reservation into the used ledger. One job settles once."""
    cost = _read(camp / COST)
    settled = [str(item) for item in cost.get("settled_job_ids") or []]
    if job_id and job_id in settled:
        return False
    used = max(0.0, float(seconds))
    release_gpu(state, used)
    state["gpu_seconds_left"] = float(state.get("gpu_seconds_left") or 0.0) - used
    cost["gpu_seconds_used"] = float(cost.get("gpu_seconds_used") or 0.0) + used
    cost["training_jobs"] = int(state.get("training_jobs") or 0)
    if job_id:
        cost["settled_job_ids"] = settled + [job_id]
    cost.setdefault("api_usd", None)
    if job_id:
        from .owner_budget import settle_training
        settle_training(camp, job_id, used)
    _write(camp / COST, cost)
    return True


def wall_deadline(gpu_seconds_left: float, n_gpu: int, *, now: float) -> float | None:
    """Wall-clock deadline from remaining GPU-seconds and assigned cards."""
    cards = max(1, int(n_gpu))
    remaining = float(gpu_seconds_left)
    if remaining <= 0:
        return now
    return now + remaining / cards


def reserve_phase(camp: Path, state: dict[str, Any], request: dict[str, Any], *, replace_receipt_id: str | None = None) -> dict[str, Any]:
    """Atomically admit a bounded phase. Reservations are separate from charges.

    The campaign worker is single-writer; a file lock also protects admission
    against a second CLI. Actual usage is always recovered from the cost ledger.
    """
    import fcntl
    from react_agent.eeg_research.agentic.loso_study import identity
    required = {"phase", "candidate_suite_id", "suite_hash", "method_revision",
                "selection_manifest_hash", "training_jobs", "gpu_seconds", "llm_calls", "fold_work"}
    if not required <= request.keys():
        raise ValueError("phase_reservation_request_incomplete")
    jobs, seconds, calls = request["training_jobs"], request["gpu_seconds"], request["llm_calls"]
    import math
    if type(jobs) is not int or jobs < 0 or type(calls) is not int or calls < 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("phase_reservation_cost_invalid")
    path = camp / "phase_reservations.json"
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(path) or {"schema_version": "eeg_research.phase_reservations.v1", "receipts": {}}
        receipt_id = "phase_" + identity(request)
        existing = ledger["receipts"].get(receipt_id)
        if existing and existing["status"] == "active":
            return existing
        if existing:
            raise ValueError("phase_reservation_already_released")
        if state.get("status") in {"paused", "cancelled", "completed"} or state.get("pause_after_step"):
            raise ValueError("phase_admission_user_stopped")
        cost = _read(camp / COST)
        job_ids = {path.parent.name for pattern in ("*/launch_intent.json", "*/job.json")
                   for path in (camp / "jobs").glob(pattern)}
        used_jobs = max(int(state.get("training_jobs", 0)), int(cost.get("training_jobs", 0)), len(job_ids))
        used_calls = max(int(state.get("llm_calls", 0)), int(cost.get("llm_calls", 0)))
        used_gpu = float(cost.get("gpu_seconds_used", 0))
        previous = ledger["receipts"].get(replace_receipt_id) if replace_receipt_id else None
        if replace_receipt_id and (not previous or previous["status"] != "active"
                or set(previous["consumed_job_ids"]) - set(previous.get("settled_phase_job_ids", []))):
            raise ValueError("phase_re_admission_requires_settled_previous_receipt")
        active = [row for key, row in ledger["receipts"].items() if row["status"] == "active" and key != replace_receipt_id]
        job_reserved = sum(row["remaining_training_jobs"] for row in active)
        gpu_reserved = sum(row["remaining_gpu_seconds"] for row in active)
        call_reserved = sum(row["remaining_llm_calls"] for row in active)
        from .owner_budget import enabled, reserve_phase_owner
        owner_phase = None
        if enabled(camp):
            owner_phase = reserve_phase_owner(camp, receipt_id, {"training_jobs":jobs,"gpu_seconds":seconds},
                replace_owner_id=(previous or {}).get("budget_owner"))
        if not owner_phase and used_jobs + job_reserved + jobs > int(state["max_training_jobs"]):
            raise ValueError("phase_training_job_reservation_unavailable")
        if not owner_phase and used_gpu + gpu_reserved + seconds > float(state["max_gpu_seconds"]):
            raise ValueError("phase_gpu_reservation_unavailable")
        if not owner_phase and used_calls + call_reserved + calls > int(state["max_llm_calls"]):
            raise ValueError("phase_feedback_api_reservation_unavailable")
        receipt = {"receipt_id": receipt_id, "request": request, "request_hash": identity(request),
                   "status": "active", "remaining_training_jobs": jobs, "remaining_gpu_seconds": seconds,
                   "remaining_llm_calls": calls, "consumed_job_ids": [], "released_reason": None,
                   "job_gpu_estimates": {}, "settled_phase_job_ids": [],
                   "usage_at_admission": {"training_jobs": used_jobs, "llm_calls": used_calls, "gpu_seconds": used_gpu}}
        if owner_phase:
            receipt.update(budget_owner=owner_phase["owner_id"], allocation_kind="selected_plan_child",
                feedback_api_authority="selected_plan_role_allocations")
        ledger["receipts"][receipt_id] = receipt
        if previous:
            previous.update(status="released", released_reason="superseded_by_explicit_failed_process_retry",
                            remaining_training_jobs=0, remaining_gpu_seconds=0.0, remaining_llm_calls=0,
                            superseded_by_receipt_id=receipt_id)
        _write(path, ledger)
        state["phase_reservation_id"] = receipt_id
        state["gpu_seconds_reserved"] = gpu_reserved + seconds
        state["method_jobs_reserved"] = job_reserved + jobs
        return receipt


def phase_reservation(camp: Path, receipt_id: str) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.loso_study import identity
    row = (_read(camp / "phase_reservations.json").get("receipts") or {}).get(receipt_id)
    if not row or row["request_hash"] != identity(row["request"]) or receipt_id != "phase_" + row["request_hash"]:
        raise ValueError("phase_reservation_receipt_invalid")
    return row


def consume_phase_job(camp: Path, receipt_id: str, job_id: str, gpu_estimate: float) -> None:
    """Consume an admission slot at persisted launch intent, once per job."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if job_id in row["consumed_job_ids"]:
            return
        if row["status"] != "active" or row["remaining_training_jobs"] <= 0:
            raise ValueError("phase_launch_without_available_reservation")
        from .owner_budget import enabled, claim_training
        if enabled(camp):
            claim_training(camp, job_id, gpu_estimate, phase_owner_id=row.get("budget_owner"))
        row["consumed_job_ids"].append(job_id)
        row["remaining_training_jobs"] -= 1
        # Keep a running/unknown process's GPU reservation until native settle.
        row.setdefault("job_gpu_estimates", {})[job_id] = gpu_estimate
        ledger["receipts"][receipt_id] = row
        _write(camp / "phase_reservations.json", ledger)


def settle_phase_job(camp: Path, state: dict[str, Any], receipt_id: str, job_id: str) -> None:
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if job_id not in row["consumed_job_ids"]:
            raise ValueError("phase_job_settlement_missing_intent")
        if job_id not in row.setdefault("settled_phase_job_ids", []):
            row["remaining_gpu_seconds"] = max(0.0, row["remaining_gpu_seconds"] - row["job_gpu_estimates"][job_id])
            row["settled_phase_job_ids"].append(job_id)
            ledger["receipts"][receipt_id] = row
            _write(camp / "phase_reservations.json", ledger)
        state["gpu_seconds_reserved"] = sum(item["remaining_gpu_seconds"] for item in ledger["receipts"].values() if item["status"] == "active")


def settle_phase_feedback(camp: Path, receipt_id: str, evidence_id: str, allocation: int) -> None:
    """Discharge a completed method report's allocation, without altering fees."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        completed = row.setdefault("feedback_completed_evidence_ids", [])
        if evidence_id in completed:
            return
        if row["status"] != "active":
            raise ValueError("phase_feedback_reservation_released")
        completed.append(evidence_id)
        row["remaining_llm_calls"] = max(0, row["remaining_llm_calls"] - allocation)
        ledger["receipts"][receipt_id] = row
        _write(camp / "phase_reservations.json", ledger)


def release_phase(camp: Path, state: dict[str, Any], receipt_id: str, reason: str) -> dict[str, Any]:
    """Release only unused allocation. Never change API/job/GPU charges."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if row["status"] == "active":
            unsettled = set(row["consumed_job_ids"]) - set(row.get("settled_phase_job_ids", []))
            if unsettled:
                raise ValueError("phase_release_requires_process_cost_reconciliation")
            if row.get("budget_owner"):
                from .owner_budget import release_phase_owner
                release_phase_owner(camp,row["budget_owner"])
            row.update(status="released", released_reason=reason, remaining_training_jobs=0,
                       remaining_gpu_seconds=0.0, remaining_llm_calls=0)
            ledger["receipts"][receipt_id] = row
            _write(camp / "phase_reservations.json", ledger)
        active = [item for item in ledger["receipts"].values() if item["status"] == "active"]
        state["gpu_seconds_reserved"] = sum(item["remaining_gpu_seconds"] for item in active)
        state["method_jobs_reserved"] = sum(item["remaining_training_jobs"] for item in active)
        if state.get("phase_reservation_id") == receipt_id:
            state["phase_reservation_id"] = None
        return row
