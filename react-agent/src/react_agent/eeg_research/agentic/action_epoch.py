"""Stable action inputs, independent of model/UI history projections."""
from __future__ import annotations
import hashlib
import json

DIAGNOSTIC_ACTIONS = frozenset({"collect_diagnostics", "diagnose_results"})
SOURCE_KEYS = ("evidence_id", "kind", "candidate_id", "job_id", "job_dir", "suite_id",
    "suite_ref", "suite_hash", "source_hash", "method_revision", "result_hash", "config_hash",
    "protocol_hash", "seed", "fidelity", "phase", "scope", "coverage", "evaluation_valid", "reason")

def action_input_epoch(state: dict, action: str) -> str:
    supplied = (state.get("action_input_epochs") or {}).get(action)
    if supplied:
        return supplied
    rows = []
    for row in state.get("evidence") or []:
        if row.get("kind") in {"method_screening", "method_suite", "run"} or row.get("job_dir"):
            rows.append({key: row[key] for key in SOURCE_KEYS if key in row})
        elif action not in DIAGNOSTIC_ACTIONS and row.get("kind") in {"diagnostic_bundle", "learning_profile", "training_diagnostic"}:
            rows.append({key: row[key] for key in (*SOURCE_KEYS, "source_epoch", "artifact_refs", "status") if key in row})
    body = {"version": "suite_diagnostics.v1", "action": action,
            "inputs": sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()

def action_epochs(state: dict) -> dict:
    return {action: action_input_epoch(state, action) for action in sorted(DIAGNOSTIC_ACTIONS)}
