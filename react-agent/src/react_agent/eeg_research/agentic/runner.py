"""One training job. The child does not inherit API keys."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.adapters.ubp_retrieval import child_env
from react_agent.eeg_research.agentic.binding import binding_accepts
from react_agent.eeg_training.protocol import Design, train_command


_SECRET_KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY")


def research_env(candidate_extension: Path | None) -> dict[str, str]:
    env = child_env()
    for key in list(env):
        upper = key.upper()
        if key in _SECRET_KEYS or "API_KEY" in upper or "SECRET" in upper or "TOKEN" in upper:
            env.pop(key, None)
    env["EEG_FINAL_TEST"] = "0"
    if candidate_extension is not None:
        extension_dir = str(Path(candidate_extension).resolve())
        env["PYTHONPATH"] = os.pathsep.join(part for part in (extension_dir, env.get("PYTHONPATH", "")) if part)
        env["EEG_CANDIDATE_PATH"] = extension_dir
        env["EEG_CANDIDATE_MODULE"] = "eeg_candidate"
    return env


def comparable(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Same contract and full fidelity. Pilot and unknown source stay out of the ranking."""
    return (
        left.get("evaluation_valid") is True
        and right.get("evaluation_valid") is True
        and left.get("fidelity") == "full"
        and right.get("fidelity") == "full"
        and left.get("contract_fingerprint")
        and left.get("contract_fingerprint") == right.get("contract_fingerprint")
        and left.get("gallery_size") == right.get("gallery_size")
    )


def accept_job(job_dir: Path, manifest: dict[str, Any] | None, fidelity: str) -> dict[str, Any]:
    """Read metrics only after the source binding matches. NaN is not a score of zero."""
    metrics_path = job_dir / "metrics.json"
    if not metrics_path.is_file():
        return {"evaluation_valid": False, "reason": "metrics_missing", "fidelity": fidelity}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = payload.get("fixed_bank_top1")
    if not isinstance(score, (int, float)):
        return {"evaluation_valid": False, "reason": "score_missing", "fidelity": fidelity, "execution_succeeded": True}
    if manifest is None:
        return {"evaluation_valid": False, "reason": "manifest_missing", "fidelity": fidelity}
    ok, reason = binding_accepts(job_dir, manifest)
    record = {
        "evaluation_valid": ok,
        "reason": reason,
        "fidelity": fidelity,
        "fixed_bank_top1": float(score) if ok else None,
        "gallery_size": payload.get("validation_image_count"),
        "execution_succeeded": True,
        "implementation_failure": False,
    }
    if payload.get("error") in {"oom", "nan"}:
        record["evaluation_valid"] = False
        record["fixed_bank_top1"] = None
        record["reason"] = str(payload["error"])
        record["implementation_failure"] = True
    return record


def launch_command(design: Design, out_dir: Path, root: Path) -> list[str]:
    return train_command(design, out_dir, root)


def charge_gpu(started: float, n_gpu: int) -> float:
    return max(0.0, time.time() - started) * max(1, n_gpu)


Runner = Callable[[Design, Path, dict[str, str]], dict[str, Any]]
