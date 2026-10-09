"""Role envelopes and TaskLedger handoff. Recovery matches task, attempt, and digest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from react_agent.eeg_research.agentic.artifacts import allocate_artifact_id, input_digest, register, request_digest, verify
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
    "result_analyst",
    "research_librarian",
    "experiment_designer",
    "memory_curator",
    "result_auditor",
}


class RoleResultError(ValueError):
    """The role output is not a valid envelope for this task."""


_IDENTITY_FIELDS = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash"}
_FORBIDDEN_PERMISSION_FIELDS = {"allowed_actions", "approval_record", "permissions", "execute"}


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


def _reject_permission_fields(payload: dict[str, Any]) -> None:
    found = [key for key in _FORBIDDEN_PERMISSION_FIELDS if key in payload]
    if found:
        raise RoleResultError("role_result_permissions")


def wrap_role_result(
    payload: dict[str, Any],
    *,
    task: dict[str, Any],
    prompt_hash: str = "",
    status: str | None = None,
) -> dict[str, Any]:
    """Attach the task identity. A missing envelope is wrapped, not invented as success."""
    payload = _coerce_role_fields(payload)
    _reject_permission_fields(payload)
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


def _claimed(reply: dict[str, Any], field: str) -> str | None:
    value = reply.get(field)
    if value in (None, ""):
        return None
    return str(value)


def bind_role_output(reply: Any, *, task: dict[str, Any], prompt_hash: str = "") -> dict[str, Any]:
    """Validate a claimed envelope or wrap a domain object.

    Any explicit identity field must match this attempt. A mismatch is not stripped and rebound.
    A business payload with no identity is wrapped by the runtime.
    """
    if not isinstance(reply, dict):
        raise RoleResultError("role_result_not_object")
    reply = _coerce_role_fields(reply)
    claimed_task = _claimed(reply, "task_id")
    claimed_attempt = _claimed(reply, "attempt_id")
    claimed_digest = _claimed(reply, "input_digest")
    claimed_candidate = _claimed(reply, "candidate_id")
    if claimed_task is not None and claimed_task != str(task["task_id"]):
        raise RoleResultError("role_result_identity_mismatch")
    if claimed_attempt is not None and claimed_attempt != str(task["attempt_id"]):
        raise RoleResultError("role_result_identity_mismatch")
    if claimed_digest is not None and claimed_digest != str(task["input_digest"]):
        raise RoleResultError("role_result_identity_mismatch")
    expected_candidate = task.get("candidate_id")
    if claimed_candidate is not None and expected_candidate not in (None, "") and claimed_candidate != str(expected_candidate):
        raise RoleResultError("role_result_identity_mismatch")
    explicit = (claimed_task, claimed_attempt, claimed_digest)
    if any(item is not None for item in explicit) and not all(item is not None for item in explicit):
        raise RoleResultError("role_result_identity_mismatch")
    matched = (
        reply.get("schema_version") == ROLE_RESULT_VERSION
        and claimed_task == str(task["task_id"])
        and claimed_attempt == str(task["attempt_id"])
        and claimed_digest == str(task["input_digest"])
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
    request: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    recovery_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a ledger task whose digest covers the business request and input files."""
    if request is not None and request.get("artifact_reads"):
        from react_agent.eeg_research.agentic.handoffs import prepare_read_delivery
        prepare_read_delivery(request["artifact_reads"])
    digest = (
        request_digest(request=request, artifacts=artifacts, paths=inputs)
        if request is not None or artifacts
        else input_digest(inputs)
    )
    refs = [str(path) for path in inputs]
    refs.extend(str(row["artifact_id"]) for row in artifacts or [] if isinstance(row, dict) and row.get("artifact_id"))
    # The request hash remains exact. Recovery additionally names the current
    # source, approval, prompt/schema, evidence, memory and action feasibility.
    # A clock tick or remaining-call decrement alone is not a new decision.
    if request is not None or recovery_identity is not None:
        from react_agent.eeg_research.agentic.artifacts import input_dependency_manifest
        dependencies = input_dependency_manifest(camp, paths=inputs, artifacts=artifacts)
        recovery_identity = {**(recovery_identity or role_recovery_identity(
            camp, role, candidate_id, request or {}, digest)), "input_dependencies": dependencies}
        from react_agent.eeg_research.agentic.task_ledger import find_completed_task, no_progress_repetitions
        cached = find_completed_task(camp, role=role, recovery_identity=recovery_identity,
                                     candidate_id=candidate_id, input_digest=digest)
        if cached and "input_dependencies" in (cached["task"].get("recovery_identity") or {}):
            task = dict(cached["task"])
            task["_cached_role_result"] = cached
            return task
        if no_progress_repetitions(camp, recovery_identity=recovery_identity) >= 3:
            from react_agent.eeg_research.agentic.llm import LlmUnavailable
            raise LlmUnavailable("role_no_progress_retry_limit")
    return create_task(
        camp,
        role=role,
        input_artifact_refs=refs,
        input_digest=digest,
        expected_output_schema=ROLE_RESULT_VERSION,
        candidate_id=candidate_id,
        recovery_identity=recovery_identity,
    )


def role_recovery_identity(camp: Path, role: str, target: str | None, request: dict, digest: str) -> dict:
    import hashlib
    from react_agent.eeg_research.agentic.llm import system_prompt, domain_model_for
    from react_agent.eeg_research.agentic.embedding import memory_config
    from react_agent.eeg_research.agentic.run_context import load_approved_binding
    from react_agent.eeg_research.agentic.artifacts import request_digest
    memory = memory_config(camp)
    system = system_prompt(role, memory=memory)
    model = domain_model_for(role, memory)
    schema = model.model_json_schema() if model else {"envelope": ROLE_RESULT_VERSION}
    binding = load_approved_binding(camp, str(target)) if target else None
    source = camp / "candidates" / str(target) / "extension" / "eeg_candidate.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    budget = request.get("budget") or {}
    def semantic_view(value):
        if isinstance(value, dict):
            return {key: semantic_view(item) for key, item in value.items() if key not in {
                "heartbeat", "heartbeat_at", "last_polled_at", "tick", "elapsed", "elapsed_seconds",
                "timestamp", "recorded_at", "at", "created_at", "updated_at", "started_at", "ended_at",
                "pid", "worker_pid", "supervisor_pid", "budget", "remaining_resources",
                "task_id", "attempt_id", "input_digest", "_context_budget",
                "summary", "summary_zh", "summary_en", "role_context_projection", "planner_context_projection",
                "llm_calls", "llm_calls_left", "input_tokens", "output_tokens", "reasoning_tokens",
                "gpu_seconds_used", "gpu_seconds_left", "gpu_seconds_reserved", "training_jobs_left"}}
        if isinstance(value, list):
            return [semantic_view(item) for item in value]
        return value
    return {"input_dependencies": [], "role": role, "action": request.get("action") or request.get("analysis_trigger") or role,
            "target_id": target, "source": source_hash or request.get("source_hash") or request.get("run_identity"),
            "config": {"hooks": {key: (binding or {}).get(key) or {} for key in ("model", "objective", "transform")},
                       "scientific_request": semantic_view(request.get("experiment") or request.get("contract") or request.get("effective_config"))},
            "approval": binding, "prompt_hash": hashlib.sha256(system.encode()).hexdigest()[:16],
            "schema_hash": request_digest(request=schema), "evaluator": request.get("evaluation_population") or request.get("contract"),
            "evidence": semantic_view(request.get("artifact_reads") or request.get("evidence") or request.get("run_artifacts") or []),
            "memory_snapshot": semantic_view(request.get("retrieved_memory") or request.get("memory") or request.get("lessons")),
            "legal_actions": request.get("available_actions") or request.get("allowed_actions") or [],
            "budget_feasibility": {"can_call": budget.get("llm_calls_left", 1) > 0,
                                   "can_implement": budget.get("llm_calls_left", 11) >= 11,
                                   "can_train_suite": budget.get("training_jobs_left", 10) >= 10,
                                   "gpu_available": budget.get("gpu_seconds_left", 1) > 0},
            "request_semantics": request_digest(request=semantic_view(request)),
            "feedback": request.get("schema_error") or request.get("previous_design_failure") or request.get("repair_issues")}


def finish_role_task(
    camp: Path,
    task: dict[str, Any],
    payload: dict[str, Any],
    *,
    kind: str,
    path: Path,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Write the final envelope once, then register those exact bytes."""
    try:
        assert_role_task_dependencies(camp, task)
    except RoleResultError as exc:
        # Never rewrite a reused historical successful receipt. For a new task,
        # dependency loss must still terminate the attempt and its retry count.
        if task.get("_cached_role_result"):
            raise
        if payload.get("status", "completed") not in {"failed", "blocked"}:
            mark(camp, task["task_id"], "failed", role=task.get("role"),
                 attempt_id=task.get("attempt_id"), dependency_error=str(exc))
            raise
        payload = {"status": "failed", "prompt_hash": payload.get("prompt_hash", ""),
                   "summary_zh": str(payload.get("summary_zh") or str(exc)),
                   "dependency_error": str(exc)}
    cached = task.get("_cached_role_result")
    if cached:
        from react_agent.eeg_research.agentic.task_ledger import find_completed_task
        verified = find_completed_task(camp, role=task["role"], recovery_identity=task["recovery_identity"],
                                       candidate_id=task.get("candidate_id"), input_digest=task["input_digest"])
        if not verified or verified["task"]["task_id"] != task["task_id"]:
            raise RoleResultError("completed_role_cache_changed")
        return verified["envelope"]
    if not payload.get("prompt_hash"):
        calls = camp / "llm_calls.jsonl"
        if calls.exists():
            for line in reversed(calls.read_text(encoding="utf-8").splitlines()):
                row = json.loads(line)
                if row.get("task_id") == task["task_id"] and row.get("role") == task["role"] and row.get("success") is True:
                    payload = {**payload, "prompt_hash": row.get("prompt_hash", "")}
                    break
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_id = allocate_artifact_id()
    envelope = wrap_role_result(payload, task=task, prompt_hash=str(payload.get("prompt_hash") or ""))
    refs = list(envelope.get("artifact_refs") or [])
    if artifact_id not in refs:
        refs.append(artifact_id)
    envelope["artifact_refs"] = refs
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact = register(
        camp,
        path,
        kind=kind,
        producer_task_id=task["task_id"],
        candidate_id=candidate_id,
        artifact_id=artifact_id,
    )
    ok, reason = verify(camp, artifact["artifact_id"])
    if not ok:
        raise RoleResultError(reason)
    task_status = str(envelope.get("status") or "failed")
    if task_status == "requires_framework_extension":
        task_status = "blocked"
    if task_status not in {"completed", "partial", "blocked", "failed"}:
        task_status = "partial"
    mark(camp, task["task_id"], task_status, role=task.get("role"), attempt_id=task.get("attempt_id"),
         candidate_id=candidate_id or task.get("candidate_id"), artifact_id=artifact["artifact_id"], path=str(path))
    return envelope


def assert_role_task_dependencies(camp: Path, task: dict[str, Any]) -> None:
    """Reject a stale task at consumption without rewriting its original receipt."""
    from react_agent.eeg_research.agentic.artifacts import verify_input_dependencies
    identity = task.get("recovery_identity") or {}
    if "input_dependencies" not in identity:
        if task.get("_cached_role_result"):
            raise RoleResultError("completed_role_cache_dependency_proof_missing")
        return
    if not verify_input_dependencies(camp, identity["input_dependencies"]):
        raise RoleResultError("role_task_input_dependencies_changed")
