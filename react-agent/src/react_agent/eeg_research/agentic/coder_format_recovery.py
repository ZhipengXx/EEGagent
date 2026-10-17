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
from .context.types import digest


# Confirmed provider outputs with no usable final JSON; no tool call was executed.
PROVIDER_DEGRADATION = frozenset({("DeepSeekParseError", "format_error"),
                                  ("DeepSeekOutputError", "missing_final_content")})


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def coder_format_recovery(camp: Path, operation: str, *, payload: dict,
                          bound: dict, revision: str) -> dict | None:
    """Return the same bounded child on every reentry, or leave the parent alone.

    The caller uses this operation for both paid-reply recovery and transport.
    Its ordinary owner/global admission remains authoritative for all requests.
    """
    try:
        return _recovery(Path(camp), operation, payload, bound, revision)
    except (OSError, ValueError, KeyError, TypeError):
        return None  # Incomplete proof does not create a new request allowance.


def _recovery(camp: Path, operation: str, payload: dict, bound: dict,
              revision: str) -> dict | None:
    step = payload.get("step")
    candidate = bound.get("candidate_id")
    if (not revision or ":format_feedback:" in operation or ":provider_retry:" in operation or not candidate
            or Path(candidate).name != candidate or not bound.get("attempt_id")
            or type(step) is not int or step < 1
            or operation != operation_id("candidate_coder", payload, bound, "")):
        return None
    attempts = [row for row in _rows(camp / ATTEMPTS) if row.get("operation_id") == operation]
    if (len(attempts) != 2 or [row.get("operation_attempt") for row in attempts] != [1, 2]
            or any(row.get("max_operation_attempts") != 2 or row.get("role") != "candidate_coder"
                   for row in attempts)):
        return None
    call_ids = [row["call_id"] for row in attempts]
    if len(set(call_ids)) != 2:
        return None
    calls = [row for row in _rows(camp / "llm_calls.jsonl") if row.get("operation_id") == operation]
    if len(calls) != 2 or {row["call_id"] for row in calls} != set(call_ids):
        return None
    if any(row.get("operation_id") == operation and row.get("call_id") not in call_ids
           for row in _rows(camp / "llm_api_intents.jsonl")):
        return None
    calls = [next(row for row in calls if row["call_id"] == call_id) for call_id in call_ids]
    semantics = operation_id("candidate_coder", payload,
                             {key: value for key, value in bound.items() if key != "operation_id"}, "")
    revisions = {call.get("coder_format_feedback_revision", "eeg_research.coder_json_feedback.v1") for call in calls}
    if revision not in revisions:
        kinds, suffix = {("DeepSeekParseError", "format_error")}, ":format_feedback:"
    elif revisions == {revision}:
        kinds, suffix = PROVIDER_DEGRADATION, ":provider_retry:"
    else:
        return None  # Mixed revisions prove neither case.
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
        if (frozen.get("step") != step or response.get("validation") != call.get("failure_kind")
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
    return {"operation_id": operation + suffix + _sha(revision)[:16],
            "parent_operation_id": operation, "feedback_revision": revision, "max_attempts": 2,
            "budget_owner": calls[-1].get("budget_owner"), "parent_call_ids": call_ids,
            "rejected_format": calls[-1]["raw_excerpt"][:2048], "provenance": proof}
