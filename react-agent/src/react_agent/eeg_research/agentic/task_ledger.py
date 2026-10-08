"""Typed task ledger. Recovery must match task_id, attempt_id, and input_digest."""

from __future__ import annotations

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.identity import result_matches

LEDGER = "task_ledger.jsonl"
STATUSES = {"pending", "running", "waiting_for_evidence", "completed", "partial", "blocked", "failed", "superseded", "cancelled"}


def _append(camp: Path, row: dict[str, Any]) -> dict[str, Any]:
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / LEDGER).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def create_task(
    camp: Path,
    *,
    role: str,
    input_artifact_refs: list[str],
    input_digest: str,
    expected_output_schema: str,
    candidate_id: str | None = None,
    attempt_id: str | None = None,
    supersedes_task_id: str | None = None,
    depends_on: list[str] | None = None,
    budget_reservation: dict[str, Any] | None = None,
    recovery_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one task. A revised task must name the task it supersedes."""
    row = {
        "task_id": f"task_{uuid.uuid4().hex[:12]}",
        "role": role,
        "input_artifact_refs": list(input_artifact_refs),
        "input_digest": input_digest,
        "expected_output_schema": expected_output_schema,
        "status": "pending",
        "attempt_id": attempt_id or uuid.uuid4().hex[:12],
        "candidate_id": candidate_id,
        "supersedes_task_id": supersedes_task_id,
        "depends_on": list(depends_on or []),
        "budget_reservation": budget_reservation or {},
        "created_at": time.time(),
    }
    if recovery_identity is not None:
        row["recovery_identity"] = recovery_identity
        row["recovery_identity_digest"] = recovery_identity_digest(recovery_identity)
    if supersedes_task_id:
        mark(camp, supersedes_task_id, "superseded")
    return _append(camp, row)


def mark(camp: Path, task_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
    """Append a status update. The ledger is append-only."""
    if status not in STATUSES:
        raise ValueError(f"unknown_task_status:{status}")
    row = {"task_id": task_id, "status": status, "at": time.time(), **fields}
    return _append(camp, row)


def latest(camp: Path, task_id: str) -> dict[str, Any] | None:
    """Return the newest row for a task id."""
    path = camp / LEDGER
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_id") == task_id:
            found = row if found is None else {**found, **row}
    return found


def reusable(
    payload: Any,
    *,
    task_id: str,
    attempt_id: str,
    input_digest: str,
    candidate_id: str | None = None,
) -> bool:
    """True only when the stored result belongs to this task attempt and digest.

    Results from another candidate, another attempt, or a changed input are not reused.
    """
    if not isinstance(payload, dict) or not input_digest or not task_id or not attempt_id:
        return False
    if payload.get("task_id") != task_id:
        return False
    if candidate_id is not None and payload.get("candidate_id") not in {None, candidate_id}:
        return False
    if payload.get("attempt_id") != attempt_id:
        return False
    if payload.get("input_digest") != input_digest:
        return False
    return True


def identity_reusable(payload: Any, *, candidate_id: str, attempt_id: str, phase: str, input_hash: str) -> bool:
    """Compatibility wrapper around attempt isolation."""
    return result_matches(
        payload,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        phase=phase,
        input_hash=input_hash,
    )


RECOVERY_IDENTITY_VERSION = "eeg_research.role_recovery_identity.v1"
_RECOVERY_REQUIRED = frozenset({
    "role", "action", "target_id", "source", "config", "approval", "prompt_hash",
    "schema_hash", "evaluator", "evidence", "memory_snapshot", "legal_actions",
    "budget_feasibility", "request_semantics", "feedback",
})


def recovery_identity_digest(identity: dict[str, Any]) -> str:
    """Hash explicit current semantic/authorization identities, never a summary.

    ``budget_feasibility`` describes whether actions remain executable, rather
    than remaining dollars/ticks/PIDs. ``request_semantics`` includes the actual
    hypothesis/intervention/role requirement. ``feedback`` retains schema errors,
    semantic rejections and new artifact-page digests so legitimate recovery is
    a new input. Callers must supply even inapplicable fields explicitly as null.
    """
    if not isinstance(identity, dict) or _RECOVERY_REQUIRED.difference(identity):
        raise ValueError("incomplete_role_recovery_identity")
    if not identity.get("role") or not identity.get("action") or not identity.get("prompt_hash") or not identity.get("schema_hash"):
        raise ValueError("invalid_role_recovery_identity")
    if not isinstance(identity["legal_actions"], list) or not isinstance(identity["budget_feasibility"], dict):
        raise ValueError("invalid_role_recovery_authorization")
    from react_agent.eeg_research.agentic.artifacts import stable_request
    # Free-form metadata (summary, tick/PID, raw usage) cannot create a new input.
    # Scientific narrative belongs only in the explicit request_semantics field.
    normalized = stable_request({key: identity[key] for key in _RECOVERY_REQUIRED})
    normalized["legal_actions"] = sorted(normalized["legal_actions"])
    encoded = json.dumps({"version": RECOVERY_IDENTITY_VERSION, "identity": normalized},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def task_snapshots(camp: Path) -> list[dict[str, Any]]:
    """Fold the append-only ledger; retain each original task and receipt."""
    path = camp / LEDGER
    if not path.is_file():
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if task_id:
            rows[str(task_id)] = {**rows.get(str(task_id), {}), **row}
    return list(rows.values())


def _legacy_prompt_provenance(camp: Path, task: dict[str, Any],
                              envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Recover an omitted Analyst prompt only from exact valid transport traces.

    Older worker code stripped the bound envelope before saving the normalized
    domain body. The request still binds task/attempt/digest and the valid model
    response must normalize to precisely the registered saved result. Missing or
    mismatched proof is a cache miss, not a fabricated prompt/receipt update.
    """
    if task.get("role") != "result_analyst":
        return None
    calls = camp / "llm_calls.jsonl"
    if not calls.is_file():
        return None
    from react_agent.eeg_research.agentic.schemas import ResultAnalysis

    def normalized(value: Any) -> dict[str, Any]:
        body = value.get("payload") if isinstance(value.get("payload"), dict) else value
        excluded = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "artifact_refs"}
        if "status" not in ResultAnalysis.model_fields:
            excluded.add("status")
        return ResultAnalysis.model_validate({k: v for k, v in body.items() if k not in excluded}).model_dump()

    try:
        saved = normalized(envelope)
    except (ValueError, TypeError, AttributeError):
        return None
    for line in reversed(calls.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != task["role"] or row.get("success") is not True or not row.get("call_id"):
            continue
        request_path = camp / "llm_requests" / (str(row["call_id"]) + ".json")
        response_path = camp / "llm_responses" / (str(row["call_id"]) + ".json")
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            user = json.loads(request["user"])
            if any(user.get(k) != task.get(k) for k in ("task_id", "attempt_id", "input_digest")):
                continue
            system_sha = hashlib.sha256(request["system"].encode("utf-8")).hexdigest()
            user_sha = hashlib.sha256(request["user"].encode("utf-8")).hexdigest()
            if (request.get("role") != task["role"] or response.get("validation") != "valid"
                    or row.get("prompt_hash") != system_sha[:16]
                    or request.get("system_sha256") != system_sha
                    or request.get("user_sha256") != user_sha
                    or row.get("request_system_sha256") != system_sha
                    or row.get("request_user_sha256") != user_sha
                    or normalized(response.get("reply")) != saved):
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        return {"prompt_hash": row["prompt_hash"], "call_id": row["call_id"],
                "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "scope": "Exact task-bound valid historical request/response normalized to the verified saved Analyst result; original receipt is unchanged."}
    return None


def find_completed_task(camp: Path, *, role: str,
                        recovery_identity: dict[str, Any],
                        candidate_id: str | None = None,
                        input_digest: str | None = None) -> dict[str, Any] | None:
    """Return a still-verified completed receipt for this exact semantic input.

    Older tasks without a recovery identity are usable only for an exact original
    request digest and matching prompt/schema; no historical authorization is
    inferred. A failed/partial/stale/superseded task or modified result is never a
    cached success. The caller still validates current executable gates/freshness.
    No task, API, check, job, or score is created by this read-only lookup.
    """
    digest = recovery_identity_digest(recovery_identity)
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.roles import RoleResultError, validate_role_result
    for task in reversed(task_snapshots(camp)):
        if task.get("role") != role or task.get("status") != "completed":
            continue
        if candidate_id is not None and task.get("candidate_id") != candidate_id:
            continue
        old_digest = task.get("recovery_identity_digest")
        if old_digest != digest and not (
            old_digest is None and input_digest and task.get("input_digest") == input_digest
        ):
            continue
        artifact_id = task.get("artifact_id")
        if not artifact_id:
            continue
        try:
            artifact = resolve_verified_artifact(camp, str(artifact_id))
            stored = json.loads(Path(str(artifact["path"])).read_text(encoding="utf-8"))
            envelope = validate_role_result(stored, task)
        except (OSError, ValueError, KeyError, TypeError, RoleResultError):
            continue
        if envelope.get("status") != "completed":
            continue
        prompt_provenance = None
        prompt_hash = envelope.get("prompt_hash")
        if not prompt_hash and old_digest is None:
            prompt_provenance = _legacy_prompt_provenance(camp, task, envelope)
            prompt_hash = (prompt_provenance or {}).get("prompt_hash")
        if prompt_hash != recovery_identity["prompt_hash"]:
            continue
        # The persisted prompt hash covers the schema injected in that system
        # prompt; legacy tasks also retain their envelope schema version.
        if old_digest is None and task.get("expected_output_schema") != envelope.get("schema_version"):
            continue
        return {"task": task, "envelope": envelope, "artifact": artifact,
                "provenance": {"original_task_id": task["task_id"],
                    "original_attempt_id": task.get("attempt_id"),
                    "original_input_digest": task.get("input_digest"),
                    "artifact_id": artifact_id, "sha256": artifact["sha256"],
                    "legacy_prompt_provenance": prompt_provenance,
                    "recovery_identity_digest": digest,
                    "scope": "Verified original completed task reused; original identity/receipt preserved; current gates must be revalidated."}}
    return None


def record_task_reuse(camp: Path, cached: dict[str, Any]) -> dict[str, Any]:
    """Append a provenance event without inventing a new successful attempt."""
    task = cached["task"]
    return mark(camp, str(task["task_id"]), "completed", reused_at=time.time(),
                reuse_provenance=cached["provenance"])


def no_progress_repetitions(camp: Path, *, recovery_identity: dict[str, Any]) -> int:
    """Count finished attempts for the same progress-sensitive input.

    Summary wording and supervision timestamps are absent from this explicit
    identity. Changed source, permissions, schema feedback, artifact pages or
    intervention therefore reset the count. Running/pending tasks do not count.
    Existing caller-specific parse/repair caps remain the authority.
    """
    digest = recovery_identity_digest(recovery_identity)
    return sum(row.get("recovery_identity_digest") == digest and
               row.get("status") in {"completed", "partial", "blocked", "failed"}
               for row in task_snapshots(camp))
