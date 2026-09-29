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
    return {
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
    _write(camp / COST, cost)
    return True


def wall_deadline(gpu_seconds_left: float, n_gpu: int, *, now: float) -> float | None:
    """Wall-clock deadline from remaining GPU-seconds and assigned cards."""
    cards = max(1, int(n_gpu))
    remaining = float(gpu_seconds_left)
    if remaining <= 0:
        return now
    return now + remaining / cards
