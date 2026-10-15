"""Adopt a confirmed role reply at the original paid-call identity.

The caller supplies its normal domain/reference validator. Recovery supplies
the original frozen request, and never grants an action or scientific gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from .types import digest


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _error(reason: str):
    from react_agent.eeg_research.agentic.roles import RoleResultError
    return RoleResultError(reason)


def require_current_prompt(task: dict | None, prompt_hash: str) -> None:
    """A pending original attempt cannot send the current, different system."""
    if not task:
        return
    expected = (task.get("recovery_identity") or {}).get("prompt_hash")
    if expected and expected != prompt_hash:
        raise _error("started_role_prompt_changed_requires_child_task")


def has_confirmed_reply(camp: Path, task: dict) -> bool:
    """Only select an original task; complete transport proof is checked later."""
    calls = [row for row in _rows(Path(camp) / "llm_calls.jsonl")
             if row.get("task_id") == task["task_id"]]
    return bool(calls and calls[-1].get("success") is True
                and calls[-1].get("transport_outcome") == "confirmed_response"
                and (Path(camp) / "llm_responses" / (str(calls[-1].get("call_id")) + ".json")).is_file())


def unresolved_task_intents(camp: Path, task: dict) -> bool:
    settled = {row.get("call_id") for row in _rows(Path(camp) / "llm_calls.jsonl")}
    return any(row.get("task_id") == task["task_id"] and row.get("call_id") not in settled
               for name in ("llm_api_intents.jsonl", "api_operation_attempts.jsonl")
               for row in _rows(Path(camp) / name))


def recover_role_reply(camp: Path, *, role: str, resolved: Any,
                       validate_reply: Callable[[dict, dict, str], dict],
                       task: dict | None = None, operation_id: str | None = None,
                       identity: dict | None = None,
                       request_semantics_digest: str | None = None) -> dict | None:
    """Return a validated original reply, or block a paid-but-unproven reply.

    Nothing is returned for a request that has never reached transport. Known
    unsuccessful attempts remain available to the existing bounded retry path;
    unknown transport and confirmed replies with missing proof cannot be resent.
    Non-envelope roles use their persisted operation/candidate/attempt binding.
    """
    from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies
    from react_agent.eeg_research.agentic.task_ledger import mark, recovery_identity_digest
    from react_agent.eeg_research.agentic.llm import transport_options
    from .sources import ContextSourceRegistry

    camp = Path(camp)
    identity = identity or {}
    calls = _rows(camp / "llm_calls.jsonl")
    intents = _rows(camp / "llm_api_intents.jsonl") + _rows(camp / "api_operation_attempts.jsonl")
    if task:
        if task.get("role") != role:
            raise _error("role_recovery_task_role_mismatch")
        assert_role_task_dependencies(camp, task)
        recovery = task.get("recovery_identity") or {}
        if ("input_dependencies" not in recovery
                or recovery_identity_digest(recovery) != task.get("recovery_identity_digest")):
            raise _error("role_recovery_dependency_proof_missing")
        matches = lambda row: row.get("task_id") == task["task_id"]
    else:
        if not operation_id:
            return None
        matches = lambda row: (row.get("operation_id") == operation_id and row.get("role") == role
            and all(row.get(key) == identity.get(key) for key in ("candidate_id", "attempt_id", "input_hash") if identity.get(key)))
    def intent_matches(row):
        if task:
            return matches(row)
        # Older operation/intent journals omitted candidate and input_hash.
        # Their exact call id and request hashes still link to the fully bound
        # request/response/cost row. Missing fields do not identify a new source.
        return (row.get("operation_id") == operation_id and row.get("role") == role
            and all(row.get(key) == identity[key] for key in ("candidate_id", "attempt_id", "input_hash")
                    if identity.get(key) and key in row))
    selected = [row for row in calls if matches(row)]
    unsettled = [row for row in intents if intent_matches(row) and row.get("call_id") not in {item.get("call_id") for item in calls}]
    if unsettled:
        raise _error("unsettled_api_intent_requires_recovery")
    if not selected:
        return None
    call = selected[-1]
    if call.get("unconfirmed_charge_possible") or call.get("transport_outcome") in {"outcome_unknown", "unknown_after_process_exit"}:
        dispositions = [row for row in _rows(camp / "api_unknown_dispositions.jsonl")
                        if row.get("call_id") == call.get("call_id")]
        if (dispositions and dispositions[-1].get("action") == "abandon_response"
                and dispositions[-1].get("allow_new_request") is True):
            return None  # Explicit native authorization, with the old charge retained.
        raise _error("unknown_api_charge_requires_disposition")
    if call.get("success") is not True or call.get("transport_outcome") != "confirmed_response":
        return None  # Never look behind this newest unsuccessful attempt.
    if task is None and request_semantics_digest is not None:
        if not call.get("role_request_semantics_digest"):
            raise _error("role_recovery_request_semantics_proof_missing")
        if call["role_request_semantics_digest"] != request_semantics_digest:
            return None  # A genuinely revised request uses its remaining native allowance.
    call_id = str(call.get("call_id") or "")
    try:
        request_path = camp / "llm_requests" / (call_id + ".json")
        response_path = camp / "llm_responses" / (call_id + ".json")
        traced = json.loads(request_path.read_text(encoding="utf-8"))
        response = json.loads(response_path.read_text(encoding="utf-8"))
        frozen_user = json.loads(traced["user"])
        system_hash = hashlib.sha256(traced["system"].encode()).hexdigest()
        user_hash = hashlib.sha256(traced["user"].encode()).hexdigest()
        expected = task or identity
        for key in ("task_id", "attempt_id", "input_digest", "candidate_id", "input_hash"):
            if expected.get(key) is not None and any(row.get(key) != expected[key] for row in (call, traced, response)):
                raise _error("role_recovery_identity_mismatch")
        if task and any(frozen_user.get(key) != task[key] for key in ("task_id", "attempt_id", "input_digest")):
            raise _error("role_recovery_request_identity_mismatch")
        if any(row.get("call_id") != call_id or row.get("role") != role for row in (traced, response)):
            raise _error("role_recovery_trace_identity_mismatch")
        if (traced.get("system_sha256") != system_hash or call.get("request_system_sha256") != system_hash
                or traced.get("user_sha256") != user_hash or call.get("request_user_sha256") != user_hash
                or call.get("prompt_hash") != system_hash[:16]
                or traced.get("prompt_hash") != system_hash[:16] or response.get("prompt_hash") != system_hash[:16]):
            raise _error("role_recovery_request_hash_mismatch")
        if task and system_hash[:16] != (task.get("recovery_identity") or {}).get("prompt_hash"):
            raise _error("role_recovery_original_prompt_mismatch")
        if (call.get("requested_model") != resolved.model or call.get("role_config_hash") != resolved.role_config_hash
                or traced.get("role_config_hash") != resolved.role_config_hash
                or traced.get("role_model_config") != resolved.semantics()
                or call.get("role_model_config") != resolved.semantics()):
            raise _error("role_recovery_config_mismatch")
        fields = traced.get("transport_fields") or traced.get("context_transport_fields") or transport_options(resolved.profile())
        if fields != transport_options(resolved.profile()):
            raise _error("role_recovery_transport_config_mismatch")
        if digest({"system": traced["system"], "user": traced["user"], "transport_fields": fields}) != call.get("context_request_digest"):
            raise _error("role_recovery_prepared_digest_mismatch")
        if role == "candidate_coder" and response.get("validation") == "coder_tool_request_invalid":
            reply = response.get("reply")
            if not isinstance(reply, dict) or digest(reply) != call.get("response_reply_sha256"):
                raise _error("role_recovery_response_hash_missing_or_changed")
            # Known invalid tool arguments executed nothing. Preserve their spend
            # and let the existing operation allowance govern the next attempt.
            return None
        revalidate_review_schema = (role == "candidate_reviewer"
                                    and response.get("validation") == "domain_schema_invalid")
        if (response.get("validation") not in {"pending", "valid"} and not revalidate_review_schema
                or not isinstance(response.get("reply"), dict)):
            raise _error("role_recovery_reply_invalid")
        reply = response["reply"]
        if digest(reply) != call.get("response_reply_sha256"):
            raise _error("role_recovery_response_hash_missing_or_changed")
        if call.get("finish_reason") != "stop" or not isinstance(call.get("input_tokens"), int):
            raise _error("role_recovery_transport_completion_proof_missing")
        if not any(row.get("call_id") == call_id and intent_matches(row)
                   and row.get("request_system_sha256") == system_hash
                   and row.get("request_user_sha256") == user_hash for row in intents):
            raise _error("role_recovery_paid_intent_missing")
        manifest_ref = (call.get("context_sidecars") or {}).get("manifest")
        if not manifest_ref:
            raise _error("role_recovery_source_manifest_missing")
        manifest = json.loads(Path(manifest_ref).read_text(encoding="utf-8"))
        registry = ContextSourceRegistry(camp)
        source_rows = []
        for source in manifest.get("sources", []):
            block = registry.get(source["source_ref"], role=role,
                                 require_current=str(source["block_id"]).startswith("closed:"))
            registry.load(block, role=role, require_current=str(source["block_id"]).startswith("closed:"))
            if block.source_hash != source["source_hash"]:
                raise _error("role_recovery_source_changed")
            source_rows.append({"block_id": block.block_id, "source_ref": block.source_ref,
                "source_hash": block.source_hash, "revision": block.source_revision, "visibility": list(block.visibility)})
        if digest(source_rows) != manifest.get("source_dependency_digest"):
            raise _error("role_recovery_source_manifest_changed")
        result = validate_reply(reply, frozen_user, system_hash[:16])
        if not isinstance(result, dict):
            raise _error("role_recovery_domain_validator_invalid")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from react_agent.eeg_research.agentic.roles import RoleResultError
        if isinstance(exc, RoleResultError):
            raise
        raise _error("role_recovery_proof_or_domain_invalid:" + type(exc).__name__) from exc
    receipt = {"schema_version": "eeg.role_response_recovery.v1", "role": role,
        "call_id": call_id, "task_id": (task or {}).get("task_id"),
        "attempt_id": (task or identity).get("attempt_id"), "input_digest": (task or identity).get("input_digest"),
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
        "prompt_hash": system_hash[:16], "context_request_digest": call["context_request_digest"],
        "source_dependency_digest": manifest["source_dependency_digest"],
        "api_calls_added": 0, "cost_preserved": True, "recorded_at": time.time()}
    if revalidate_review_schema:
        receipt["original_validation"] = response["validation"]
        receipt["revalidated_by"] = "current_review_schema_and_source_references"
    existing = _rows(camp / "role_response_recovery.jsonl")
    if not any(row.get("call_id") == call_id and row.get("task_id") == receipt["task_id"] for row in existing):
        with (camp / "role_response_recovery.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if task:
            mark(camp, task["task_id"], task.get("status", "pending"),
                 recovered_transport_call_id=call_id, recovered_response_sha256=receipt["response_sha256"])
    return result
