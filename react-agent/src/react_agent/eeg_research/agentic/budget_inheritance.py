"""Native, single-successor accounting for a paused legacy study.

Historical usage is referenced once; the successor's cost.json contains only
new consumption. No historical goal, protocol, or ledger is rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path
from react_agent.eeg_research.agentic.loso_study import (
    CAPS, campaign_path, digest, identity, manifest, read, usage, write_once,
)


def verified_history(study: Path) -> dict:
    study = study.resolve()
    body = manifest(study)
    dependencies = []
    for fold in body["folds"]:
        camp = campaign_path(study, fold)
        path = camp / "campaign_state.json"
        if not path.exists():
            continue
        state = read(path)
        if state.get("live_job") or state.get("status") not in {"paused", "blocked", "finished", "cancelled"}:
            raise ValueError("historical_study_not_quiescent")
        from react_agent.eeg_research.agentic.loso_study import worker_alive
        if worker_alive(camp):
            raise ValueError("historical_worker_still_alive")
        intents_path = camp / "llm_api_intents.jsonl"
        if intents_path.exists():
            intents = {json.loads(line)["call_id"] for line in intents_path.read_text().splitlines() if line.strip()}
            calls_path = camp / "llm_calls.jsonl"
            calls = {json.loads(line)["call_id"] for line in calls_path.read_text().splitlines() if line.strip()} if calls_path.exists() else set()
            if intents - calls:
                raise ValueError("historical_api_cost_uncertain")
        for dep in [path, camp / "goal.json", camp / "execution_protocol.json",
                    camp / "cost.json", camp / "llm_calls.jsonl", intents_path, *sorted((camp / "jobs").glob("*/job.json"))]:
            if dep.exists():
                if dep.name == "job.json" and read(dep).get("status") == "running":
                    raise ValueError("historical_job_unsettled")
                if dep.name == "job.json":
                    from react_agent.eeg_research.agentic.jobs import _alive
                    if _alive(read(dep)):
                        raise ValueError("historical_training_process_still_alive")
                dependencies.append({"path": str(dep), "sha256": digest(dep)})
    dependencies.append({"path": str(study / "study_manifest.json"), "sha256": digest(study / "study_manifest.json")})
    used = usage(study, body)
    remaining = {key: CAPS[key] - used[key] for key in CAPS}
    if any(value < 0 for value in remaining.values()):
        raise ValueError("historical_authorization_exhausted")
    result = {"schema_version": "eeg_research.budget_inheritance.v1", "source_study": str(study),
              "authorization_caps": CAPS, "historical_usage": used, "remaining": remaining,
              "dependencies": dependencies, "accounting": "historical_reference_plus_successor_incremental"}
    result["receipt_hash"] = identity(result)
    return result


def derive_limits(goal: dict, receipt: dict) -> dict:
    result = dict(goal)
    for field, resource in (("max_llm_calls", "llm_calls"), ("max_training_jobs", "training_jobs"),
                            ("max_gpu_seconds", "gpu_seconds")):
        result[field] = min(result[field], receipt["remaining"][resource])
    result["historical_budget_receipt_hash"] = receipt["receipt_hash"]
    return result


def estimate_suite(receipt: dict, n_gpu: int) -> tuple[float, list[dict]]:
    """Conservative ten-full projection of actual historical occupied GPU time."""
    estimates, basis = [], []
    for dep in receipt["dependencies"]:
        path = Path(dep["path"])
        if path.name != "job.json":
            continue
        record = read(path)
        metrics_path = path.parent / "metrics.json"
        if record.get("fidelity") != "full" or not metrics_path.exists():
            continue
        epochs = read(metrics_path).get("last_epoch")
        seconds = record.get("gpu_seconds")
        cards = len(record.get("gpu") or [])
        if not epochs or not seconds or not cards:
            continue
        # Global batch remains fixed. Fewer cards can increase wall time, so
        # reducing the assigned card count must not discount measured GPU work.
        # More cards retain the conservative occupied-wall-time projection.
        projected = float(seconds) / int(epochs) * max(1.0, n_gpu / cards) * 50 * 10 * 1.25
        estimates.append(projected)
        basis.append({"job_ref": str(path), "job_sha256": digest(path), "metrics_ref": str(metrics_path),
                      "metrics_sha256": digest(metrics_path), "actual_epochs": epochs, "actual_gpu_seconds": seconds,
                      "historical_gpu_count": cards, "new_gpu_count": n_gpu,
                      "projection_gpu_seconds": projected,
                      "rule": "max_gpu_seconds_per_epoch*max(1,new_cards/old_cards)*50*10*1.25",
                      "limitation": "Historical projection, not measured speedup; fixed global batch, local-negative cost and new candidate runtime can differ."})
    if not estimates:
        raise ValueError("method_suite_conservative_cost_basis_missing")
    return max(estimates), basis


def claim_successor(camp: Path, receipt: dict) -> None:
    source = Path(receipt["source_study"])
    claim = source.parent / "budget_successions" / (identity({"source": str(source)}) + ".json")
    link = {"source_study": str(source), "successor": str(camp.resolve()), "receipt_hash": receipt["receipt_hash"]}
    if claim.exists():
        if read(claim) != link:
            raise ValueError("historical_budget_already_has_successor")
    else:
        write_once(claim, link)
    target = camp / "budget_inheritance.json"
    if target.exists():
        if read(target) != receipt:
            raise ValueError("budget_inheritance_receipt_changed")
    else:
        write_once(target, receipt)


def validate_inheritance(camp: Path) -> dict:
    receipt = read(camp / "budget_inheritance.json")
    for dependency in receipt["dependencies"]:
        if digest(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError("historical_budget_source_changed")
    body = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if identity(body) != receipt["receipt_hash"]:
        raise ValueError("budget_inheritance_receipt_invalid")
    if receipt["authorization_caps"] != CAPS or any(
            receipt["remaining"][key] != CAPS[key] - receipt["historical_usage"][key] for key in CAPS):
        raise ValueError("budget_inheritance_authorization_mismatch")
    claim_successor(camp, receipt)
    return receipt
