"""Role envelopes and TaskLedger handoff. Recovery matches task, attempt, and digest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from react_agent.eeg_research.agentic.artifacts import input_digest, register
from react_agent.eeg_research.agentic.schemas import ROLE_RESULT_VERSION, RoleResult
from react_agent.eeg_research.agentic.task_ledger import create_task, mark, reusable

LONG_JSON_ROLES = {
    "candidate_coder",
    "candidate_reviewer",
    "result_analyst",
    "research_planner",
    "research_librarian",
    "experiment_designer",
    "memory_curator",
    "result_auditor",
}
ENVELOPE_ROLES = {
    "research_librarian",
    "experiment_designer",
    "memory_curator",
    "result_auditor",
}


class RoleResultError(ValueError):
    """The role output is not a valid envelope for this task."""


_IDENTITY_FIELDS = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash"}


def _string_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if not isinstance(value, list):
        return refs
    for item in value:
        if isinstance(item, str) and item:
            refs.append(item)
        elif isinstance(item, dict):
            for key in ("artifact_id", "ref", "path", "id"):
                found = item.get(key)
                if isinstance(found, str) and found:
                    refs.append(found)
                    break
    return refs


def _coerce_role_fields(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    if "artifact_refs" in copied:
        copied["artifact_refs"] = _string_refs(copied.get("artifact_refs"))
    if "evidence_refs" in copied:
        copied["evidence_refs"] = _string_refs(copied.get("evidence_refs"))
    return copied


def _domain_body(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _IDENTITY_FIELDS}


def wrap_role_result(
    payload: dict[str, Any],
    *,
    task: dict[str, Any],
    prompt_hash: str = "",
    status: str | None = None,
) -> dict[str, Any]:
    """Attach the task identity. A missing envelope is wrapped, not invented as success."""
    payload = _coerce_role_fields(payload)
    if payload.get("schema_version") == ROLE_RESULT_VERSION and payload.get("task_id"):
        envelope = dict(payload)
    else:
        envelope = {
            "schema_version": ROLE_RESULT_VERSION,
            "task_id": task["task_id"],
            "attempt_id": task["attempt_id"],
            "input_digest": task["input_digest"],
            "prompt_hash": prompt_hash,
            "status": status or payload.get("status") or "completed",
            "artifact_refs": list(payload.get("artifact_refs") or []),
            "evidence_refs": list(payload.get("evidence_refs") or []),
            "summary_zh": str(payload.get("summary_zh") or ""),
            "payload": payload,
        }
    envelope["task_id"] = task["task_id"]
    envelope["attempt_id"] = task["attempt_id"]
    envelope["input_digest"] = task["input_digest"]
    envelope.setdefault("schema_version", ROLE_RESULT_VERSION)
    envelope["artifact_refs"] = _string_refs(envelope.get("artifact_refs"))
    envelope["evidence_refs"] = _string_refs(envelope.get("evidence_refs"))
    try:
        RoleResult.model_validate(envelope)
    except ValidationError as exc:
        raise RoleResultError("role_result_invalid") from exc
    return envelope


def validate_role_result(payload: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Reject a result that belongs to another task, attempt, or digest."""
    if not isinstance(payload, dict):
        raise RoleResultError("role_result_not_object")
    payload = _coerce_role_fields(payload)
    if payload.get("schema_version") != ROLE_RESULT_VERSION:
        raise RoleResultError("role_result_schema")
    if not reusable(
        payload,
        task_id=str(task["task_id"]),
        attempt_id=str(task["attempt_id"]),
        input_digest=str(task["input_digest"]),
        candidate_id=task.get("candidate_id"),
    ):
        raise RoleResultError("role_result_identity_mismatch")
    try:
        return RoleResult.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise RoleResultError("role_result_invalid") from exc


def task_identity_from_payload(payload: Any, bound: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Collect task identity from the request body or a bound call."""
    bound = bound or {}
    if not isinstance(payload, dict):
        return None
    task_id = payload.get("task_id") or bound.get("task_id")
    attempt_id = payload.get("attempt_id") or bound.get("attempt_id")
    input_digest_value = payload.get("input_digest") or bound.get("input_digest")
    if not task_id or not attempt_id or not input_digest_value:
        return None
    return {
        "task_id": str(task_id),
        "attempt_id": str(attempt_id),
        "input_digest": str(input_digest_value),
        "candidate_id": payload.get("candidate_id") or bound.get("candidate_id"),
    }


def bind_role_output(reply: Any, *, task: dict[str, Any], prompt_hash: str = "") -> dict[str, Any]:
    """Validate a claimed envelope or wrap a domain object.

    A different task_id is not reused. A schema_version with missing or partial identity is wrapped.
    """
    if not isinstance(reply, dict):
        raise RoleResultError("role_result_not_object")
    reply = _coerce_role_fields(reply)
    claimed = reply.get("task_id")
    if reply.get("schema_version") == ROLE_RESULT_VERSION and claimed and claimed != task["task_id"]:
        raise RoleResultError("role_result_identity_mismatch")
    matched = (
        reply.get("schema_version") == ROLE_RESULT_VERSION
        and claimed == task["task_id"]
        and str(reply.get("attempt_id") or "") == str(task["attempt_id"])
        and str(reply.get("input_digest") or "") == str(task["input_digest"])
    )
    if matched:
        try:
            envelope = validate_role_result(reply, task)
        except RoleResultError:
            envelope = wrap_role_result(_domain_body(reply), task=task, prompt_hash=prompt_hash)
    else:
        body = _domain_body(reply) if reply.get("schema_version") == ROLE_RESULT_VERSION else reply
        envelope = wrap_role_result(body, task=task, prompt_hash=prompt_hash)
    inner = envelope.get("payload")
    if isinstance(inner, dict):
        for key, value in inner.items():
            envelope.setdefault(key, value)
    return envelope


def begin_role_task(
    camp: Path,
    *,
    role: str,
    inputs: list[Path],
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Open a ledger task whose digest covers the listed input files."""
    digest = input_digest(inputs)
    refs = [str(path) for path in inputs]
    return create_task(
        camp,
        role=role,
        input_artifact_refs=refs,
        input_digest=digest,
        expected_output_schema=ROLE_RESULT_VERSION,
        candidate_id=candidate_id,
    )


def finish_role_task(
    camp: Path,
    task: dict[str, Any],
    payload: dict[str, Any],
    *,
    kind: str,
    path: Path,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Write the envelope, register the artifact, and mark the task completed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = wrap_role_result(payload, task=task)
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact = register(camp, path, kind=kind, producer_task_id=task["task_id"], candidate_id=candidate_id)
    envelope.setdefault("artifact_refs", []).append(artifact["artifact_id"])
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    mark(camp, task["task_id"], "completed", artifact_id=artifact["artifact_id"], path=str(path))
    return envelope
