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
    result["scientific_admission_history"] = _legacy_scientific_admission_scope(study)
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
    from react_agent.eeg_research.agentic.candidate_lifecycle import enabled
    if enabled(goal):
        quota = receipt.get("scientific_admission_history")
        if not isinstance(quota, dict) or quota.get("schema_version") != "eeg_research.scientific_admission_history.v1":
            raise ValueError("scientific_admission_history_proof_required_for_new_protocol")
        result["inherited_scientific_admissions"] = quota["admissions"]
        result["inherited_scientific_admission_receipt_hash"] = receipt["receipt_hash"]
    return result


FRESH_VERSION = "eeg_research.fresh_budget_authorization.v1"


def _fresh_allocation(goal: dict) -> dict:
    import math
    allocation = {resource: goal[field] for field, resource in
                  (("max_llm_calls", "llm_calls"), ("max_training_jobs", "training_jobs"),
                   ("max_gpu_seconds", "gpu_seconds"))}
    for key, value in allocation.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or (key in {"llm_calls", "training_jobs"} and type(value) is not int)):
            raise ValueError("invalid_fresh_budget_authorization")
    return allocation


def fresh_authorization(camp: Path, goal: dict, source: Path, *, authorization_id: str) -> dict:
    """Bind a new explicit allocation; old spending remains read-only history.

    This is separate from a resource handoff: none of the old unused budget,
    scientific allocations, candidate results or repair attempts are imported.
    The new campaign must start from its own matched control and new proposals.
    """
    camp, source = camp.resolve(), source.resolve()
    if camp == source or not authorization_id.strip():
        raise ValueError("fresh_authorization_requires_new_campaign_and_identity")
    if goal.get("evaluation_mode") != "loso_method_search":
        raise ValueError("fresh_authorization_requires_method_search")
    if not (source / "method_search_manifest.json").is_file():
        raise ValueError("fresh_authorization_requires_verified_method_history")
    allocation = _fresh_allocation(goal)
    if any(goal.get(field) for field in ("historical_budget_receipt_hash", "inherited_scientific_admissions",
                                         "inherited_scientific_admission_receipt_hash")):
        raise ValueError("fresh_authorization_cannot_import_prior_campaign")
    historical = verified_method_history(source, history_only=True)
    return _fresh_receipt(camp, source, allocation, historical, authorization_id)


def _fresh_receipt(camp: Path, source: Path, allocation: dict, historical: dict, authorization_id: str) -> dict:
    authorization = {"schema_version": FRESH_VERSION, "authorization_id": authorization_id,
        "campaign": str(camp), "new_allocation_caps": allocation,
        "scope": "new_campaign_new_scientific_admissions",
        "historical_remaining_transferred": False, "historical_results_imported": False}
    result = {"schema_version": FRESH_VERSION, "source_kind": "fresh_method_campaign_allocation",
        "source_study": str(source), "historical_snapshot": historical,
        "historical_usage": historical["historical_usage"],
        "authorization_caps": {key: historical["historical_usage"][key] + allocation[key] for key in CAPS},
        "remaining": allocation, "new_allocation_caps": allocation, "authorization": authorization,
        "authorization_hash": identity(authorization), "dependencies": historical["dependencies"],
        "scientific_admission_history": {"schema_version": "eeg_research.scientific_admission_history.v1",
            "scope": "fresh_authorized_method_search", "admissions": [], "used": 0,
            "refund_on_failure": False, "legacy_per_fold_results_imported": False,
            "scientific_results_imported": False},
        "accounting": "verified_old_spending_plus_explicit_new_allocation_no_old_remaining_transfer"}
    result["receipt_hash"] = identity(result)
    return result


def _claim_fresh_authorization(camp: Path, receipt: dict) -> None:
    authorization = receipt["authorization"]
    if authorization.get("campaign") != str(camp.resolve()):
        raise ValueError("fresh_authorization_campaign_mismatch")
    # One named authorization cannot allocate its budget to two campaigns.
    claim = camp.parent / "fresh_budget_authorizations" / (identity({
        "authorization_id": authorization["authorization_id"]}) + ".json")
    for target, content in ((claim, authorization),
                            (camp / "fresh_budget_authorization.json", authorization),
                            (camp / "budget_inheritance.json", receipt)):
        if target.is_file():
            if read(target) != content:
                raise ValueError("fresh_budget_authorization_already_bound")
        else:
            write_once(target, content)


def _validate_fresh_authorization(camp: Path, receipt: dict, seen: set) -> dict:
    allocation = receipt.get("new_allocation_caps")
    authorization = receipt.get("authorization")
    if (not isinstance(allocation, dict) or set(allocation) != set(CAPS)
            or not isinstance(authorization, dict)
            or identity(authorization) != receipt.get("authorization_hash")
            or authorization.get("new_allocation_caps") != allocation
            or authorization.get("campaign") != str(camp.resolve())
            or receipt.get("source_kind") != "fresh_method_campaign_allocation"
            or receipt.get("remaining") != allocation):
        raise ValueError("fresh_budget_authorization_invalid")
    source = Path(receipt["source_study"])
    if source.resolve() in seen:
        raise ValueError("budget_inheritance_cycle")
    historical = verified_method_history(source, history_only=True, _seen=seen)
    if historical != receipt.get("historical_snapshot"):
        raise ValueError("historical_budget_source_changed")
    goal = read(camp / "goal.json")
    # Reconstruct all fresh semantics from the immutable native goal/source;
    # rehashing a modified receipt cannot grant more budget or import results.
    if goal.get("evaluation_mode") != "loso_method_search":
        raise ValueError("fresh_authorization_requires_method_search")
    expected = _fresh_receipt(camp, source, _fresh_allocation(goal), historical,
                              authorization["authorization_id"])
    if receipt != expected or goal.get("historical_budget_receipt_hash") != receipt["receipt_hash"]:
        raise ValueError("fresh_budget_authorization_invalid")
    if read(camp / "fresh_budget_authorization.json") != authorization:
        raise ValueError("fresh_budget_authorization_changed")
    return receipt["authorization_caps"]


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
    if receipt.get("schema_version") == FRESH_VERSION:
        _claim_fresh_authorization(camp, receipt)
        return
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


def validate_inheritance(camp: Path, *, allow_delegated: bool = False, _seen: set | None = None,
                         _claim: bool = True) -> dict:
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
        upstream = validate_inheritance(parent, allow_delegated=True, _seen=seen, _claim=_claim)
        own = _method_consumption(parent)
        expected_caps = upstream["authorization_caps"]
        if (receipt.get("upstream_receipt_hash") != upstream["receipt_hash"]
                or receipt.get("source_incremental_usage") != own
                or receipt["historical_usage"] != {key: upstream["historical_usage"][key] + own[key] for key in CAPS}
                or receipt.get("research_limits") != _research_limits(parent)):
            raise ValueError("budget_inheritance_authorization_mismatch")
        if "scientific_admission_history" in receipt and receipt["scientific_admission_history"] != _scientific_admission_history(parent, upstream):
            raise ValueError("scientific_admission_history_source_changed")
    elif version == FRESH_VERSION:
        expected_caps = _validate_fresh_authorization(camp, receipt, seen)
    else:
        raise ValueError("budget_inheritance_authorization_mismatch")
    if version in {"eeg_research.budget_inheritance.v1", "eeg_research.budget_inheritance.v2"} and "scientific_admission_history" in receipt:
        if receipt["scientific_admission_history"] != _legacy_scientific_admission_scope(Path(receipt["source_study"])):
            raise ValueError("scientific_admission_history_scope_mismatch")
    if receipt["authorization_caps"] != expected_caps or any(
            receipt["remaining"][key] != expected_caps[key] - receipt["historical_usage"][key] for key in CAPS):
        raise ValueError("budget_inheritance_authorization_mismatch")
    if _claim:
        claim_successor(camp, receipt)
    return receipt


def _research_limits(camp: Path) -> dict:
    goal = read(camp / "goal.json")
    return {field: goal.get(field, default) for field, default in
            (("max_candidates", 2), ("max_repairs_per_candidate", 2),
             ("max_concurrent_training_jobs", 1))}


def _legacy_scientific_admission_scope(study: Path) -> dict:
    return {"schema_version": "eeg_research.scientific_admission_history.v1",
        "scope": "legacy_per_fold_resource_only", "admissions": [], "used": 0,
        "refund_on_failure": False, "legacy_per_fold_results_imported": False, "scientific_results_imported": False,
        "scope_proof_ref": str(study / "study_manifest.json"), "scope_proof_sha256": digest(study / "study_manifest.json")}


def _scientific_admission_history(camp: Path, upstream: dict) -> dict:
    """Carry allocated method slots, including failed, unwritten and detached IDs.

    Original per-fold studies are another scientific unit; their spending is
    carried in resource CAPS, never imported as frozen method-search results.
    Old method receipts have a signed abandoned allocation list even when the
    latest candidate row no longer contains a check or successful patch.
    """
    from react_agent.eeg_research.agentic.candidate_lifecycle import _rows, ADMISSIONS
    state = read(camp / "campaign_state.json")
    if "candidates" not in state:
        raise ValueError("scientific_admission_history_allocation_proof_missing")
    inherited = []
    prior = upstream.get("scientific_admission_history")
    if isinstance(prior, dict):
        inherited = list(prior.get("admissions") or [])
        if prior.get("used") != len({row.get("quota_identity") for row in inherited}):
            raise ValueError("scientific_admission_history_invalid")
    elif upstream.get("source_kind") == "method_campaign":
        ids = upstream.get("abandoned_untrained_candidates")
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("scientific_admission_history_allocation_proof_missing")
        inherited = [{"quota_identity": identity({"origin": upstream["source_study"], "candidate_id": value}),
            "origin_campaign": upstream["source_study"], "origin_candidate_id": value,
            "proof_kind": "signed_legacy_method_allocation_receipt", "proof_receipt_hash": upstream["receipt_hash"]}
            for value in dict.fromkeys(ids) if value != "baseline"]
    rows = _rows(camp / ADMISSIONS)
    own = []
    if rows:
        own = [{"quota_identity": identity({"origin": str(camp.resolve()), "admission_id": row["admission_id"]}),
            "origin_campaign": str(camp.resolve()), "origin_candidate_id": row["candidate_id"],
            "origin_admission_id": row["admission_id"], "spec_hash": row.get("spec_hash"),
            "proof_kind": "native_approval_allocation_ledger", "proof_ref": str(camp / ADMISSIONS),
            "proof_sha256": digest(camp / ADMISSIONS), "proof_receipt_hash": row["receipt_hash"]} for row in rows]
    else:
        # Conservative migration of native V1 allocations. Checks and current
        # status never refund an already allocated scientific method slot.
        ids = set()
        for row in state["candidates"]:
            value = row.get("candidate_id") if isinstance(row, dict) else row
            if not isinstance(value, str) or not value:
                raise ValueError("scientific_admission_history_allocation_proof_missing")
            if value != "baseline":
                ids.add(value)
        for directory in (camp / "candidates").glob("*"):
            if directory.is_dir() and (directory / "spec.json").is_file() and directory.name != "baseline":
                ids.add(directory.name)
        task_path = camp / "task_ledger.jsonl"
        if task_path.is_file():
            for line in task_path.read_text().splitlines():
                row = json.loads(line) if line.strip() else {}
                if row.get("role") in {"candidate_coder", "candidate_reviewer"} and row.get("candidate_id") not in {None, "baseline"}:
                    ids.add(row["candidate_id"])
        own = [{"quota_identity": identity({"origin": str(camp.resolve()), "candidate_id": value}),
            "origin_campaign": str(camp.resolve()), "origin_candidate_id": value,
            "proof_kind": "legacy_native_allocated_candidate_conservative", "proof_ref": str(camp / "campaign_state.json"),
            "proof_sha256": digest(camp / "campaign_state.json")} for value in sorted(ids)]
    admissions = {row["quota_identity"]: row for row in [*inherited, *own]}
    return {"schema_version": "eeg_research.scientific_admission_history.v1", "scope": "frozen_method_candidate_quota",
        "admissions": list(admissions.values()), "used": len(admissions), "refund_on_failure": False,
        "legacy_per_fold_results_imported": False, "scientific_results_imported": False}


def untrained_candidates(camp: Path, state: dict) -> list[str] | None:
    """Candidate IDs that never reached training, or None if any candidate has jobs or suites.

    A resource handoff carries budget, never candidate performance results.
    Untrained candidates retain engineering records and API spending. Their
    allocated scientific quota is carried separately and is never refunded.
    """
    ids = [str(row.get("candidate_id") if isinstance(row, dict) else row) for row in state.get("candidates") or []]
    for path in (camp / "jobs").glob("*/job.json"):
        candidate = read(path).get("candidate_id")
        if candidate and candidate != "baseline":
            return None
    if any(not suite.name.startswith("baseline_") for suite in (camp / "suites").glob("*") if suite.is_dir()):
        return None
    return ids


def _method_consumption(camp: Path, *, require_no_candidates: bool = True) -> dict:
    """Read settled incremental costs; never reset or reconcile them to zero."""
    import math
    from react_agent.eeg_research.agentic.jobs import _alive
    state, cost = read(camp / "campaign_state.json"), read(camp / "cost.json")
    if (state.get("status") not in {"paused", "blocked", "finished", "cancelled"}
            or state.get("live_job") or state.get("gpu_seconds_reserved", 0)):
        raise ValueError("historical_study_not_quiescent")
    if require_no_candidates and untrained_candidates(camp, state) is None:
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


def verified_method_history(camp: Path, *, history_only: bool = False, _seen: set | None = None) -> dict:
    """A new native campaign can change resources only with a new matched control."""
    import fcntl
    from react_agent.eeg_research.agentic.loso_study import worker_alive
    camp = camp.resolve()
    # A fresh allocation reads old records without creating or touching a lock.
    from contextlib import nullcontext
    lock = camp / "worker.lock"
    handle_context = lock.open("rb") if history_only and lock.is_file() else (
        nullcontext(None) if history_only else lock.open("a"))
    with handle_context as handle:
        try:
            if handle is not None:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("historical_worker_lock_held") from exc
        if worker_alive(camp):
            raise ValueError("historical_worker_still_alive")
        upstream = validate_inheritance(camp, allow_delegated=True, _claim=not history_only, _seen=_seen)
        goal = read(camp / "goal.json")
        if (goal.get("evaluation_mode") != "loso_method_search"
                or goal.get("historical_budget_receipt_hash") != upstream["receipt_hash"]):
            raise ValueError("method_budget_source_identity_mismatch")
        own = _method_consumption(camp, require_no_candidates=not history_only)
        used = {key: upstream["historical_usage"][key] + own[key] for key in CAPS}
        caps = upstream["authorization_caps"]
        remaining = {key: caps[key] - used[key] for key in CAPS}
        if not history_only and any(value < 0 for value in remaining.values()):
            raise ValueError("historical_authorization_exhausted")
        paths = [camp / name for name in ("budget_inheritance.json", "goal.json",
                 "campaign_state.json", "cost.json", "execution_protocol.json",
                 "method_search_manifest.json", "llm_calls.jsonl", "llm_api_intents.jsonl")]
        paths.extend(sorted((camp / "jobs").glob("*/job.json")))
        if history_only:
            paths.extend(sorted((camp / "jobs").glob("*/metrics.json")))
        paths.extend(sorted((camp / "engineering_checks").rglob("*.json")))
        paths.extend(sorted((camp / "candidates").rglob("spec.json")))
        paths.extend(path for path in (camp / "candidate_admissions.jsonl", camp / "candidate_detachments.jsonl",
                     camp / "task_ledger.jsonl") if path.is_file())
        dependencies = {row["path"]: row for row in upstream["dependencies"]}
        dependencies.update({str(p): {"path": str(p), "sha256": digest(p)} for p in paths if p.is_file()})
        result = {"schema_version": "eeg_research.budget_inheritance.v3",
            "source_kind": "method_campaign", "source_study": str(camp),
            "upstream_receipt_hash": upstream["receipt_hash"],
            "authorization_caps": caps, "historical_usage": used,
            "source_incremental_usage": own, "remaining": remaining,
            "research_limits": _research_limits(camp),
            "scientific_admission_history": _scientific_admission_history(camp, upstream),
            "abandoned_untrained_candidates": untrained_candidates(camp, read(camp / "campaign_state.json")),
            "dependencies": list(dependencies.values()),
            "accounting": "transitive_historical_reference_plus_successor_incremental"}
        result["receipt_hash"] = identity(result)
        return result
