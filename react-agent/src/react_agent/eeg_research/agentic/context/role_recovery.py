"""Adopt a confirmed role reply at the original paid-call identity.

The caller supplies its normal domain/reference validator. Recovery supplies
the original frozen request, and never grants an action or scientific gate.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .types import digest


@dataclass(frozen=True)
class PendingRoleCorrection:
    """A proven paid answer that still needs its existing bounded correction.

    This is not an adopted answer or a new operation authorization. The caller
    revalidates ``reply`` against ``frozen_payload`` and resumes the original
    operation after ``prior_application_attempts`` already reserved requests.
    """

    reply: dict
    frozen_payload: dict
    frozen_system: str
    frozen_prompt_hash: str
    operation_id: str
    prior_application_attempts: int
    call_id: str
    validation: str
    context_sidecars: dict


def _pending_designer_correction(camp: Path, call: dict, response: dict,
                                traced: dict, frozen_payload: dict,
                                calls: list[dict]) -> PendingRoleCorrection:
    """Keep durable operation attempts, including non-domain failures, spent."""
    from ..research_execution import role_api_attempt_limit

    operation = call.get("operation_id")
    attempts = [row for row in _rows(camp / "api_operation_attempts.jsonl")
                if operation and row.get("operation_id") == operation]
    if not operation or not any(row.get("call_id") == call.get("call_id") for row in attempts):
        raise _error("role_recovery_operation_attempt_proof_missing")
    ids = {row.get("call_id") for row in attempts}
    ids.update(row.get("call_id") for row in calls if row.get("operation_id") == operation)
    ids.discard(None)
    role = call.get("role", "experiment_designer")
    limit = min(role_api_attempt_limit(camp, role),
                *(int(row.get("max_operation_attempts", 2)) for row in attempts))
    if len(ids) >= limit:
        raise _error("reviewer_correction_attempts_exhausted" if role == "candidate_reviewer"
                     else "analysis_correction_attempts_exhausted" if role == "result_analyst"
                     else "designer_correction_attempts_exhausted")
    sidecars = copy.deepcopy(call.get("context_sidecars") or {})
    if call.get("designer_assembly_ref"):
        sidecars["designer_assembly_ref"] = copy.deepcopy(call["designer_assembly_ref"])
    return PendingRoleCorrection(
        reply=copy.deepcopy(response["reply"]),
        frozen_payload=copy.deepcopy(frozen_payload),
        frozen_system=traced["system"],
        frozen_prompt_hash=traced["prompt_hash"],
        operation_id=operation,
        prior_application_attempts=len(ids),
        call_id=call["call_id"],
        validation=str(response.get("validation") or "pending"),
        context_sidecars=sidecars,
    )


def _verify_designer_assembly(camp: Path, call: dict, traced: dict,
                             response: dict, expected: dict) -> None:
    """Scientific wire omits identity; immutable local assembly keeps it."""
    ref = call.get("designer_assembly_ref")
    if (not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256")
            or any(row.get("designer_assembly_ref") != ref for row in (traced, response))):
        raise _error("role_recovery_designer_assembly_proof_missing")
    path = Path(ref["path"]).resolve()
    if not path.is_relative_to((camp / "designer_inputs").resolve()):
        raise _error("role_recovery_designer_assembly_path_invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise _error("role_recovery_designer_assembly_hash_mismatch")
    snapshot = json.loads(raw)
    if (not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != "eeg_research.designer_assembly.v1"
            or not isinstance(snapshot.get("assembly_context"), dict)
            or snapshot.get("operation_id") != call.get("operation_id")
            or any(snapshot.get(key) != expected.get(key)
                   for key in ("task_id", "attempt_id", "input_digest")
                   if expected.get(key) is not None)):
        raise _error("role_recovery_designer_assembly_identity_mismatch")


def _designer_domain_error(exc: Exception) -> bool:
    """Do not turn an identity or source validation failure into a retry."""
    from pydantic import ValidationError
    from ..roles import RoleResultError

    return isinstance(exc, ValidationError) or (
        isinstance(exc, RoleResultError)
        and str(exc) in {"recovered_designer_contract_invalid", "designer_output_invalid"})


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
                       request_semantics_digest: str | None = None,
                       allow_designer_correction: bool = False,
                       allow_reviewer_correction: bool = False,
                       allow_analysis_correction: bool = False) -> dict | PendingRoleCorrection | None:
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
        designer_scientific_wire = (role == "experiment_designer"
                                   and frozen_user.get("designer_context_version") == "designer_scientific_input.v1")
        from .compiler import is_analysis_scientific_input, project_analysis_scientific_input
        analyst_scientific_wire = is_analysis_scientific_input(role, frozen_user)
        if designer_scientific_wire:
            _verify_designer_assembly(camp, call, traced, response, task or call)
        elif role == "candidate_reviewer" and frozen_user.get("reviewer_context_version") == "reviewer_implementation_input.v2":
            from ..reviewer_transport import verify_review_delivery
            verify_review_delivery(camp, call, traced, response, frozen_user)
        elif not analyst_scientific_wire and task and any(
                frozen_user.get(key) != task[key] for key in ("task_id", "attempt_id", "input_digest")):
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
        revalidate_designer = (allow_designer_correction and designer_scientific_wire
                              and response.get("validation") in {
                                  "pending", "valid", "domain_schema_invalid",
                                  "design_contract_invalid", "designer_output_invalid"})
        revalidate_reviewer = (allow_reviewer_correction and role == "candidate_reviewer"
                              and frozen_user.get("reviewer_context_version") == "reviewer_implementation_input.v2"
                              and response.get("validation") in {"pending", "valid", "reviewer_output_invalid"})
        revalidate_analysis = (allow_analysis_correction and analyst_scientific_wire
                               and response.get("validation") in {"pending", "valid", "domain_schema_invalid", "analysis_reply_invalid"})
        if (response.get("validation") not in {"pending", "valid"}
                and not revalidate_review_schema and not revalidate_designer and not revalidate_reviewer
                and not revalidate_analysis
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
        analyst_native = None
        for source in manifest.get("sources", []):
            block = registry.get(source["source_ref"], role=role,
                                 require_current=str(source["block_id"]).startswith("closed:"))
            source_payload = registry.load(block, role=role,
                require_current=str(source["block_id"]).startswith("closed:"))
            if block.source_hash != source["source_hash"]:
                raise _error("role_recovery_source_changed")
            source_rows.append({"block_id": block.block_id, "source_ref": block.source_ref,
                "source_hash": block.source_hash, "revision": block.source_revision, "visibility": list(block.visibility)})
            if analyst_scientific_wire and str(source["block_id"]).startswith("analyst-native:"):
                if analyst_native is not None:
                    raise _error("role_recovery_analysis_native_ambiguous")
                analyst_native = source_payload
        if digest(source_rows) != manifest.get("source_dependency_digest"):
            raise _error("role_recovery_source_manifest_changed")
        if analyst_scientific_wire:
            if not isinstance(analyst_native, dict):
                raise _error("role_recovery_analysis_native_missing")
            if task and any(analyst_native.get(key) != task[key]
                            for key in ("task_id", "attempt_id", "input_digest")):
                raise _error("role_recovery_analysis_native_identity_mismatch")
            original = {key: value for key, value in frozen_user.items()
                        if key not in {"previous_response_error", "previous_response"}}
            if project_analysis_scientific_input(analyst_native) != original:
                raise _error("role_recovery_analysis_science_changed")
        try:
            result = validate_reply(reply, frozen_user, system_hash[:16])
        except Exception as exc:
            if revalidate_designer and _designer_domain_error(exc):
                return _pending_designer_correction(camp, call, response, traced, frozen_user, calls)
            if revalidate_reviewer and str(exc) == "recovered_reviewer_contract_invalid":
                return _pending_designer_correction(camp, call, response, traced, frozen_user, calls)
            if revalidate_analysis:
                from pydantic import ValidationError
                if isinstance(exc, ValidationError) or str(exc) == "recovered_analysis_reply_invalid":
                    # Source and paid-call proof above have already passed.
                    # Resume the full original native binding, never a current
                    # caller's reconstructed or newly retrieved input.
                    return _pending_designer_correction(camp, call, response, traced, analyst_native, calls)
            raise
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
    if revalidate_designer and response.get("validation") not in {"pending", "valid"}:
        receipt["original_validation"] = response["validation"]
        receipt["revalidated_by"] = "current_designer_contract_and_frozen_input"
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
