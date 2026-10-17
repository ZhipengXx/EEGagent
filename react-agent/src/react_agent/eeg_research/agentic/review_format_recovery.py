"""One fixed format-repair operation for a formerly single-attempt review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .api_policy import ATTEMPTS, operation_id
from .context.types import digest

REVISION = "eeg_research.review_json_feedback.v1"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def operation_attempt_limit(camp: Path, operation: str, proposed: int = 2) -> int:
    """An old one-attempt parent cannot silently become a two-attempt parent."""
    bounds = [row["max_operation_attempts"] for row in _rows(Path(camp) / ATTEMPTS)
              if row.get("operation_id") == operation]
    return min([proposed, *bounds])


def review_format_recovery(camp: Path, operation: str, *, payload: dict, bound: dict,
                           config_hash: str) -> dict | None:
    """Preserve the paid failure; bind at most one child to its unchanged input."""
    try:
        return _recovery(Path(camp), operation, payload, bound, config_hash)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _recovery(camp: Path, operation: str, payload: dict, bound: dict,
              config_hash: str) -> dict | None:
    if (":review_format:" in operation or not bound.get("candidate_id")
            or not bound.get("attempt_id") or not bound.get("operation_id")):
        return None
    attempts = [row for row in _rows(camp / ATTEMPTS) if row.get("operation_id") == operation]
    if (len(attempts) != 1 or attempts[0].get("operation_attempt") != 1
            or attempts[0].get("max_operation_attempts") != 1
            or attempts[0].get("role") != "candidate_reviewer"):
        return None
    attempt = attempts[0]
    calls = [row for row in _rows(camp / "llm_calls.jsonl") if row.get("operation_id") == operation]
    if len(calls) != 1 or calls[0].get("call_id") != attempt.get("call_id"):
        return None
    call = calls[0]
    if (call.get("role") != "candidate_reviewer" or call.get("success") is not False
            or call.get("error") != "DeepSeekParseError" or call.get("failure_kind") != "format_error"
            or call.get("finish_reason") != "stop" or call.get("transport_outcome") != "confirmed_response"
            or call.get("unconfirmed_charge_possible") is not False
            or call.get("role_config_hash") != config_hash):
        return None
    semantics = operation_id("candidate_reviewer", payload,
        {key: value for key, value in bound.items() if key != "operation_id"}, "")
    if call.get("role_request_semantics_digest") != semantics:
        return None
    if any(row.get("operation_id") == operation and row.get("call_id") != call["call_id"]
           for row in _rows(camp / "llm_api_intents.jsonl")):
        return None
    request_path = camp / "llm_requests" / (call["call_id"] + ".json")
    response_path = camp / "llm_responses" / (call["call_id"] + ".json")
    request, response = json.loads(request_path.read_text()), json.loads(response_path.read_text())
    for key in ("candidate_id", "attempt_id", "input_hash", "input_digest"):
        if key in bound and any(row.get(key) != bound[key] for row in (call, request, response)):
            return None
    if any(row.get("call_id") != call["call_id"] or row.get("role") != "candidate_reviewer"
           for row in (request, response)):
        return None
    system_sha, user_sha = (hashlib.sha256(request[key].encode()).hexdigest() for key in ("system", "user"))
    if (request.get("system_sha256") != system_sha or request.get("user_sha256") != user_sha
            or any(row.get("request_system_sha256") != system_sha or row.get("request_user_sha256") != user_sha
                   or row.get("prompt_hash") != system_sha[:16] for row in (attempt, call, response))
            or request.get("prompt_hash") != system_sha[:16]
            or request.get("role_config_hash") != config_hash
            or attempt.get("budget_owner") != call.get("budget_owner")):
        return None
    fields = request.get("transport_fields")
    if (fields is None or digest({"system": request["system"], "user": request["user"],
            "transport_fields": fields}) != call.get("context_request_digest")
            or response.get("validation") != "format_error"
            or not isinstance(response.get("reply"), str) or not response["reply"]
            or response["reply"] != call.get("raw_excerpt")):
        return None
    return {"operation_id": operation + ":review_format:json", "parent_operation_id": operation,
        "feedback_revision": REVISION, "max_attempts": 2, "budget_owner": call.get("budget_owner"),
        "parent_call_ids": [call["call_id"]], "rejected_format": response["reply"][:2048],
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "settled_call_sha256": digest(call)}
