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


def _authorized_caps(additional: dict | None = None) -> tuple[dict, dict]:
    """Apply only explicit native authorization increments; retain legacy caps."""
    import math
    supplied = {} if additional is None else additional
    if not isinstance(supplied, dict) or set(supplied) - set(CAPS):
        raise ValueError("invalid_additional_budget_authorization")
    extras = {key: supplied.get(key, 0) for key in CAPS}
    for key, value in extras.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid_additional_budget_authorization")
        if key in {"llm_calls", "training_jobs"} and not isinstance(value, int):
            raise ValueError("invalid_additional_budget_authorization")
    return {key: CAPS[key] + extras[key] for key in CAPS}, extras


def verified_history(study: Path, *, additional_authorization: dict | None = None) -> dict:
    if (study / "budget_inheritance.json").is_file() and (study / "method_search_manifest.json").is_file():
        _, extras = _authorized_caps(additional_authorization)
        if any(extras.values()):
            raise ValueError("method_handoff_cannot_add_authorization")
        return verified_method_history(study)
    caps, extras = _authorized_caps(additional_authorization)
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
    remaining = {key: caps[key] - used[key] for key in CAPS}
    if any(value < 0 for value in remaining.values()):
        raise ValueError("historical_authorization_exhausted")
    extended = any(extras.values())
    result = {"schema_version": "eeg_research.budget_inheritance.v2" if extended else "eeg_research.budget_inheritance.v1",
              "source_study": str(study), "authorization_caps": caps, "historical_usage": used, "remaining": remaining,
              "dependencies": dependencies, "accounting": "historical_reference_plus_successor_incremental"}
    if extended:
        result["additional_authorization"] = extras
    result["receipt_hash"] = identity(result)
    return result


def derive_limits(goal: dict, receipt: dict) -> dict:
    result = dict(goal)
    for field, resource in (("max_llm_calls", "llm_calls"), ("max_training_jobs", "training_jobs"),
                            ("max_gpu_seconds", "gpu_seconds")):
        result[field] = min(result[field], receipt["remaining"][resource])
    result["historical_budget_receipt_hash"] = receipt["receipt_hash"]
    for field, limit in receipt.get("research_limits", {}).items():
        result[field] = min(result.get(field, limit), limit)
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


def successor_claim(source: Path) -> Path:
    source = source.resolve()
    return source.parent / "budget_successions" / (identity({"source": str(source)}) + ".json")


def delegated_successor(source: Path) -> str | None:
    claim = successor_claim(source)
    return read(claim)["successor"] if claim.is_file() else None


def claim_successor(camp: Path, receipt: dict) -> None:
    source = Path(receipt["source_study"]).resolve()
    if source == camp.resolve():
        raise ValueError("budget_successor_cannot_be_source")
    if receipt.get("source_kind") == "method_campaign":
        import fcntl
        # The parent cannot resume between validating its final consumption and
        # publishing the exclusive successor. The lock is a native control.
        with (source / "worker.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("historical_worker_lock_held") from exc
            for dependency in receipt["dependencies"]:
                if digest(Path(dependency["path"])) != dependency["sha256"]:
                    raise ValueError("historical_budget_source_changed")
            _claim_successor(camp, receipt)
    else:
        _claim_successor(camp, receipt)


def _claim_successor(camp: Path, receipt: dict) -> None:
    source = Path(receipt["source_study"]).resolve()
    claim = successor_claim(source)
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


def validate_inheritance(camp: Path, *, allow_delegated: bool = False, _seen: set | None = None) -> dict:
    camp = camp.resolve()
    seen = set() if _seen is None else set(_seen)
    if camp in seen:
        raise ValueError("budget_inheritance_cycle")
    seen.add(camp)
    if not allow_delegated and delegated_successor(camp):
        raise ValueError("campaign_budget_delegated_to_successor")
    receipt = read(camp / "budget_inheritance.json")
    for dependency in receipt["dependencies"]:
        if digest(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError("historical_budget_source_changed")
    body = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if identity(body) != receipt["receipt_hash"]:
        raise ValueError("budget_inheritance_receipt_invalid")
    version = receipt.get("schema_version")
    if version == "eeg_research.budget_inheritance.v1" and "additional_authorization" not in receipt:
        expected_caps = CAPS
    elif version == "eeg_research.budget_inheritance.v2" and "additional_authorization" in receipt:
        expected_caps, extras = _authorized_caps(receipt["additional_authorization"])
        if extras != receipt["additional_authorization"] or not any(extras.values()):
            raise ValueError("budget_inheritance_authorization_mismatch")
    elif version == "eeg_research.budget_inheritance.v3" and receipt.get("source_kind") == "method_campaign":
        parent = Path(receipt["source_study"])
        upstream = validate_inheritance(parent, allow_delegated=True, _seen=seen)
        own = _method_consumption(parent)
        expected_caps = upstream["authorization_caps"]
        if (receipt.get("upstream_receipt_hash") != upstream["receipt_hash"]
                or receipt.get("source_incremental_usage") != own
                or receipt["historical_usage"] != {key: upstream["historical_usage"][key] + own[key] for key in CAPS}
                or receipt.get("research_limits") != _research_limits(parent)):
            raise ValueError("budget_inheritance_authorization_mismatch")
    else:
        raise ValueError("budget_inheritance_authorization_mismatch")
    if receipt["authorization_caps"] != expected_caps or any(
            receipt["remaining"][key] != expected_caps[key] - receipt["historical_usage"][key] for key in CAPS):
        raise ValueError("budget_inheritance_authorization_mismatch")
    claim_successor(camp, receipt)
    return receipt


def _research_limits(camp: Path) -> dict:
    goal = read(camp / "goal.json")
    return {field: goal.get(field, default) for field, default in
            (("max_candidates", 2), ("max_repairs_per_candidate", 2),
             ("max_concurrent_training_jobs", 1))}


def _method_consumption(camp: Path) -> dict:
    """Read settled incremental costs; never reset or reconcile them to zero."""
    import math
    from react_agent.eeg_research.agentic.jobs import _alive
    state, cost = read(camp / "campaign_state.json"), read(camp / "cost.json")
    if (state.get("status") not in {"paused", "blocked", "finished", "cancelled"}
            or state.get("live_job") or state.get("gpu_seconds_reserved", 0)):
        raise ValueError("historical_study_not_quiescent")
    if state.get("candidates"):
        raise ValueError("resource_handoff_requires_no_candidates")
    calls_path, intents_path = camp / "llm_calls.jsonl", camp / "llm_api_intents.jsonl"
    calls = {json.loads(line)["call_id"] for line in calls_path.read_text().splitlines() if line.strip()} \
        if calls_path.is_file() else set()
    intents = {json.loads(line)["call_id"] for line in intents_path.read_text().splitlines() if line.strip()} \
        if intents_path.is_file() else set()
    abandoned = cost.get("api_abandoned_call_ids", [])
    if (intents - calls or cost.get("api_unsettled_call_ids")
            or cost.get("api_uncertain_calls", 0) != len(abandoned)):
        raise ValueError("historical_api_cost_uncertain")
    settled_rows = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()] \
        if calls_path.is_file() else []
    unknown = {row["call_id"] for row in settled_rows
               if row.get("transport_outcome") == "unknown_after_process_exit"
               and row.get("success") is False and row.get("budget_charge_calls") == 1}
    if set(abandoned) != unknown or len(abandoned) != len(unknown):
        raise ValueError("historical_abandoned_api_accounting_invalid")
    charged_calls = cost.get("llm_calls")
    # Native GPU-only campaigns omit API fields before their first reservation.
    # Zero is justified only by an explicit zero state and empty durable ledgers.
    if ("llm_calls" not in cost and not (calls | intents)
            and type(state.get("llm_calls")) is int and state["llm_calls"] == 0):
        charged_calls = 0
    if charged_calls != len(calls | intents) or state.get("llm_calls") != charged_calls:
        raise ValueError("historical_api_accounting_unreconciled")
    for directory in (camp / "jobs").glob("*"):
        path = directory / "job.json"
        if (directory / "launch_intent.json").is_file() and not path.is_file():
            raise ValueError("historical_job_unsettled")
        if not path.is_file():
            continue
        record = read(path)
        if (_alive(record) or record.get("status") == "running"
                or record.get("job_id") not in cost.get("settled_job_ids", [])):
            raise ValueError("historical_job_unsettled")
    for path in (camp / "engineering_checks").rglob("process.json"):
        record = read(path)
        child = {"pid": record.get("child_pid"), "proc_start": record.get("child_proc_start")}
        if (_alive(record) or _alive(child) or record.get("status") == "running"
                or not (path.parent / "receipt.json").is_file()):
            raise ValueError("historical_engineering_probe_unsettled")
    if state.get("training_jobs") != cost.get("training_jobs"):
        raise ValueError("historical_training_accounting_unreconciled")
    used = {"llm_calls": charged_calls, "training_jobs": cost["training_jobs"],
            "gpu_seconds": cost["gpu_seconds_used"]}
    if any(type(used[key]) is not int for key in ("llm_calls", "training_jobs")):
        raise ValueError("invalid_historical_usage")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in used.values()):
        raise ValueError("invalid_historical_usage")
    return used


def verified_method_history(camp: Path) -> dict:
    """A new native campaign can change resources only with a new matched control."""
    import fcntl
    from react_agent.eeg_research.agentic.loso_study import worker_alive
    camp = camp.resolve()
    with (camp / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("historical_worker_lock_held") from exc
        if worker_alive(camp):
            raise ValueError("historical_worker_still_alive")
        upstream = validate_inheritance(camp, allow_delegated=True)
        goal = read(camp / "goal.json")
        if (goal.get("evaluation_mode") != "loso_method_search"
                or goal.get("historical_budget_receipt_hash") != upstream["receipt_hash"]):
            raise ValueError("method_budget_source_identity_mismatch")
        own = _method_consumption(camp)
        used = {key: upstream["historical_usage"][key] + own[key] for key in CAPS}
        caps = upstream["authorization_caps"]
        remaining = {key: caps[key] - used[key] for key in CAPS}
        if any(value < 0 for value in remaining.values()):
            raise ValueError("historical_authorization_exhausted")
        paths = [camp / name for name in ("budget_inheritance.json", "goal.json",
                 "campaign_state.json", "cost.json", "execution_protocol.json",
                 "method_search_manifest.json", "llm_calls.jsonl", "llm_api_intents.jsonl")]
        paths.extend(sorted((camp / "jobs").glob("*/job.json")))
        paths.extend(sorted((camp / "engineering_checks").rglob("*.json")))
        dependencies = {row["path"]: row for row in upstream["dependencies"]}
        dependencies.update({str(p): {"path": str(p), "sha256": digest(p)} for p in paths if p.is_file()})
        result = {"schema_version": "eeg_research.budget_inheritance.v3",
            "source_kind": "method_campaign", "source_study": str(camp),
            "upstream_receipt_hash": upstream["receipt_hash"],
            "authorization_caps": caps, "historical_usage": used,
            "source_incremental_usage": own, "remaining": remaining,
            "research_limits": _research_limits(camp),
            "dependencies": list(dependencies.values()),
            "accounting": "transitive_historical_reference_plus_successor_incremental"}
        result["receipt_hash"] = identity(result)
        return result
