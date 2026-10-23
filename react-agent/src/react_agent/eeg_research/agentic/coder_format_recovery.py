"""A fixed child operation for a revised coder JSON correction.

Only conclusively rejected final responses qualify. This helper never executes
or interprets their markup, refunds spend, or changes any tool-step counter.

Two cases qualify, each for exactly one non-nesting child:
- revised feedback: both parent attempts were format errors under an older
  feedback revision, so the current feedback is new information;
- provider degradation: both parent attempts used the current revision and
  the provider returned no final content or leaked non-JSON markup, before
  any tool step ran. Without this child the step stays "recoverable" with no
  admissible request left.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .api_policy import ATTEMPTS, operation_id
from .context.representations import resolve_request_pointer
from .context.types import digest


# Confirmed provider outputs with no usable final JSON; no tool call was executed.
PROVIDER_DEGRADATION = frozenset({("DeepSeekParseError", "format_error"),
                                  ("DeepSeekOutputError", "missing_final_content")})
CHILD_SUFFIXES = (":format_feedback:", ":provider_retry:")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _frozen_step(payload: dict) -> int:
    version = payload.get("coder_context_version")
    if version == "coder_implementation_input.v2":
        pointer = "/execution_state/operation/current_step"
    elif version is None:
        pointer = "/step"
    else:
        raise ValueError("unsupported_coder_context_version")
    step = resolve_request_pointer(payload, pointer)
    if type(step) is not int or step < 1:
        raise ValueError("invalid_frozen_coder_step")
    return step


def _existing_child(camp: Path, operation: str, rows: list[dict],
                    revisions: set[str]) -> tuple[str, str, str] | None:
    """An established child keeps its identity across feedback revisions."""
    children = {}
    child_rows = []
    for row in rows:
        child = row.get("operation_id")
        if not isinstance(child, str):
            continue
        for suffix in CHILD_SUFFIXES:
            prefix = operation + suffix
            if not child.startswith(prefix):
                continue
            revision_hash = child[len(prefix):]
            if len(revision_hash) != 16 or any(char not in "0123456789abcdef" for char in revision_hash):
                raise ValueError("invalid_or_nested_coder_format_child")
            children[child] = (suffix, revision_hash)
            child_rows.append(row)
    if not children:
        return None
    if len(children) != 1:
        raise ValueError("ambiguous_coder_format_child")
    child, (suffix, revision_hash) = next(iter(children.items()))
    known_revisions = {value for value in revisions if isinstance(value, str) and value}
    for row in child_rows:
        records = [row]
        call_id = row.get("call_id")
        if isinstance(call_id, str) and call_id and Path(call_id).name == call_id:
            path = camp / "llm_requests" / (call_id + ".json")
            if path.is_file():
                records.append(json.loads(path.read_text()))
        for record in records:
            recovery = record.get("coder_format_recovery") or {}
            if not isinstance(recovery, dict):
                raise ValueError("invalid_coder_format_child_metadata")
            if (recovery.get("operation_id", child) != child
                    or recovery.get("parent_operation_id", operation) != operation):
                raise ValueError("coder_format_child_parent_mismatch")
            for value in (recovery.get("feedback_revision"), record.get("coder_format_feedback_revision")):
                if isinstance(value, str) and value:
                    known_revisions.add(value)
    matching = [value for value in known_revisions if _sha(value)[:16] == revision_hash]
    if len(matching) != 1:
        raise ValueError("coder_format_child_revision_proof_missing")
    return child, suffix, matching[0]


def coder_format_recovery(camp: Path, operation: str, *, payload: dict,
                          bound: dict, revision: str) -> dict | None:
    """Return the same bounded child on every reentry, or leave the parent alone.

    The caller uses this operation for both paid-reply recovery and transport.
    Its ordinary owner/global admission remains authoritative for all requests.
    """
    try:
        return _recovery(Path(camp), operation, payload, bound, revision)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None  # Incomplete proof does not create a new request allowance.


def _recovery(camp: Path, operation: str, payload: dict, bound: dict,
              revision: str) -> dict | None:
    step = payload.get("step")
    candidate = bound.get("candidate_id")
    if (not revision or any(suffix in operation for suffix in CHILD_SUFFIXES) or not candidate
            or Path(candidate).name != candidate or not bound.get("attempt_id")
            or type(step) is not int or step < 1
            or operation != operation_id("candidate_coder", payload, bound, "")):
        return None
    all_attempts = _rows(camp / ATTEMPTS)
    all_calls = _rows(camp / "llm_calls.jsonl")
    all_intents = _rows(camp / "llm_api_intents.jsonl")
    attempts = [row for row in all_attempts if row.get("operation_id") == operation]
    if (len(attempts) != 2 or [row.get("operation_attempt") for row in attempts] != [1, 2]
            or any(row.get("max_operation_attempts") != 2 or row.get("role") != "candidate_coder"
                   for row in attempts)):
        return None
    call_ids = [row["call_id"] for row in attempts]
    if len(set(call_ids)) != 2:
        return None
    calls = [row for row in all_calls if row.get("operation_id") == operation]
    if len(calls) != 2 or {row["call_id"] for row in calls} != set(call_ids):
        return None
    if any(row.get("operation_id") == operation and row.get("call_id") not in call_ids
           for row in all_intents):
        return None
    calls = [next(row for row in calls if row["call_id"] == call_id) for call_id in call_ids]
    semantics = operation_id("candidate_coder", payload,
                             {key: value for key, value in bound.items() if key != "operation_id"}, "")
    revisions = {call.get("coder_format_feedback_revision", "eeg_research.coder_json_feedback.v1") for call in calls}
    existing = _existing_child(camp, operation, all_attempts + all_calls + all_intents,
                               revisions | {revision})
    if existing is not None:
        child_operation, child_suffix, revision = existing
    if revision not in revisions:
        kinds, suffix = {("DeepSeekParseError", "format_error")}, ":format_feedback:"
    elif revisions == {revision}:
        kinds, suffix = PROVIDER_DEGRADATION, ":provider_retry:"
    else:
        return None  # Mixed revisions prove neither case.
    if existing is not None and child_suffix != suffix:
        return None
    proof = []
    for attempt, call in zip(attempts, calls):
        if (call.get("operation_id") != operation or call.get("role") != "candidate_coder"
                or call.get("success") is not False or (call.get("error"), call.get("failure_kind")) not in kinds
                or call.get("finish_reason") != "stop"
                or call.get("transport_outcome") != "confirmed_response"
                or call.get("unconfirmed_charge_possible") is not False
                or call.get("role_request_semantics_digest") != semantics):
            return None
        request_path = camp / "llm_requests" / (call["call_id"] + ".json")
        response_path = camp / "llm_responses" / (call["call_id"] + ".json")
        request = json.loads(request_path.read_text())
        response = json.loads(response_path.read_text())
        for key in ("candidate_id", "attempt_id", "input_hash", "input_digest"):
            if key in bound and any(row.get(key) != bound[key] for row in (call, request, response)):
                return None
        if any(row.get("call_id") != call["call_id"] or row.get("role") != "candidate_coder"
               for row in (request, response)):
            return None
        system_sha, user_sha = _sha(request["system"]), _sha(request["user"])
        if (request.get("system_sha256") != system_sha or request.get("user_sha256") != user_sha
                or any(row.get("request_system_sha256") != system_sha
                       or row.get("request_user_sha256") != user_sha for row in (attempt, call, response))
                or any(row.get("prompt_hash") != system_sha[:16] for row in (attempt, call, request, response))
                or any(attempt.get(key) != call.get(key) for key in ("attempt_id", "input_digest", "budget_owner"))
                or request.get("role_config_hash") != call.get("role_config_hash")):
            return None
        fields = request.get("transport_fields")
        if (fields is None or digest({"system": request["system"], "user": request["user"],
                                     "transport_fields": fields}) != call.get("context_request_digest")):
            return None
        frozen = json.loads(request["user"])
        if (_frozen_step(frozen) != step or response.get("validation") != call.get("failure_kind")
                or not isinstance(response.get("reply"), str) or not response["reply"]
                or response["reply"] != call.get("raw_excerpt")):
            return None
        # The exact paid semantics digest above binds the original business
        # payload, including source, spec, caller and tool history.
        proof.append({"call_id": call["call_id"], "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                      "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                      "settled_call_sha256": digest(call)})
    if calls[0].get("budget_owner") != calls[1].get("budget_owner"):
        return None
    workspace = camp / "candidates" / candidate
    recovery = json.loads((workspace / "coding_recovery.json").read_text())
    saved = recovery.get("operation_binding") or {}
    if (recovery.get("candidate_id") != candidate or recovery.get("steps_used") != step - 1
            or any(saved.get(key) != value for key, value in bound.items())
            or any(row.get("step", 0) >= step for row in _rows(workspace / "coder_log.jsonl"))):
        return None
    return {"operation_id": child_operation if existing is not None else operation + suffix + _sha(revision)[:16],
            "parent_operation_id": operation, "feedback_revision": revision, "max_attempts": 2,
            "budget_owner": calls[-1].get("budget_owner"), "parent_call_ids": call_ids,
            "rejected_format": calls[-1]["raw_excerpt"][:2048], "provenance": proof}
