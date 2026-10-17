"""A bounded Reviewer output retry proved by the original native transport."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .context.types import digest

OVERRIDE = {"candidate_reviewer": {"max_tokens": 32768}}


def validate_override(value):
    if value and (value != OVERRIDE or type(value["candidate_reviewer"]["max_tokens"]) is not int):
        raise ValueError("review_revalidation_only_allows_candidate_reviewer_max_tokens_32768")
    return value or {}


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _rows(camp, name):
    path = camp / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _one(rows, call_id):
    found = [row for row in rows if row.get("call_id") == call_id]
    if len(found) != 1:
        raise ValueError("review_truncation_transport_proof_not_unique")
    return found[0]


def truncated_proof(camp, state, candidate_id):
    """Read proof only; a failure label grants no new request or allocation."""
    failure = state.get("failure") or {}
    if (failure.get("phase") != "review_candidate" or failure.get("error_type") != "output_truncated"
            or failure.get("recoverable") is not True or state.get("detail") != "output_truncated"):
        raise ValueError("review_revalidation_requires_confirmed_output_truncation")
    return _original_proof(camp, state, candidate_id)


def _original_proof(camp, state, candidate_id):
    from .binding import file_sha256
    from .native_patch import _check_fingerprint
    from .implementation_semantics import requirements_hash
    from .experiment_gate import resolve_approved_experiment
    from .selected_plan import load_plan
    from .plan_protocol import scientific_projection
    from .role_config import resolve_role_config
    from .context.representations import resolve_request_pointer
    camp = Path(camp).resolve()
    plan = load_plan(camp, state)
    workspace = camp / "candidates" / candidate_id
    attempt = _read(workspace / "attempt.json")
    source, check_path = workspace / "extension/eeg_candidate.py", workspace / "checks.json"
    checks, spec = _read(check_path), _read(workspace / "spec.json")
    impl = _read(workspace / "implementation.json")
    candidate = next((row for row in state.get("candidates") or [] if row.get("candidate_id") == candidate_id), {})
    if (not plan or plan.get("candidate_id") != candidate_id or plan.get("status") not in {"blocked", "implementing"}
            or plan.get("current_step") not in {"implement", "repair"}
            or attempt.get("candidate_id") != candidate_id or not attempt.get("attempt_id")
            or checks.get("ok") is not True or checks.get("source_sha256") != file_sha256(source)
            or checks.get("check_fingerprint") != _check_fingerprint(workspace, None)
            or (checks.get("implementation_semantics") or {}).get("requirements_hash") != requirements_hash(
                (spec.get("experiment") or {}).get("implementation_requirements") or [])
            or candidate.get("status") != "ready_for_review"
            or impl.get("status") != "ready_for_review" or impl.get("input_hash") != file_sha256(source)
            or impl.get("attempt_id") != attempt["attempt_id"]
            or (spec.get("experiment") or {}).get("spec_hash") != (state.get("experiment") or {}).get("spec_hash")):
        raise ValueError("review_truncation_current_candidate_identity_changed")
    if (_read(workspace / "input_spec.json").get("multigpu_check") or {}).get("enabled"):
        probe = checks.get("multigpu_check") or {}
        if probe.get("ok") is not True or (probe.get("result") or {}).get("source_sha256") != file_sha256(source):
            raise ValueError("review_truncation_current_multigpu_check_missing")
    approved = resolve_approved_experiment(camp, target_id=candidate_id, state=state, action="implement",
        spec_ref=state.get("experiment_ref"), expected_hash=(state.get("experiment") or {}).get("spec_hash"))
    if plan.get("approved_scientific_digest") != digest(scientific_projection(approved)):
        raise ValueError("review_truncation_approved_science_changed")
    operation = (attempt.get("operations") or {}).get("review_candidate")
    owner_id = plan["budget_owner"] + ":role:candidate_reviewer"
    calls = [row for row in _rows(camp, "llm_calls.jsonl") if row.get("operation_id") == operation
             and row.get("role") == "candidate_reviewer"]
    attempts, intents = _rows(camp, "api_operation_attempts.jsonl"), _rows(camp, "llm_api_intents.jsonl")
    old_attempts = [row for row in attempts if row.get("operation_id") == operation]
    if not operation or len(calls) != 2 or {row["call_id"] for row in calls} != {row["call_id"] for row in old_attempts}:
        raise ValueError("review_truncation_requires_two_settled_original_attempts")
    with sqlite3.connect((camp / "resource_owners.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        owners = {key: json.loads(value) for key, value in db.execute("SELECT id,payload FROM owners")}
        claims = {key: json.loads(value) for key, value in db.execute("SELECT id,payload FROM claims")}
    owner = owners.get(owner_id) or {}
    if (owner.get("parent_id") != plan["budget_owner"] or owner.get("purpose") != "candidate_reviewer"
            or owner.get("status") != "active"):
        raise ValueError("review_truncation_original_owner_changed")
    resolved = resolve_role_config(camp, "candidate_reviewer")
    if resolved.max_tokens != 16384:
        raise ValueError("review_truncation_original_output_budget_changed")
    proofs = []
    for call in calls:
        cid = call["call_id"]
        request_path, response_path = camp / "llm_requests" / (cid + ".json"), camp / "llm_responses" / (cid + ".json")
        request, response = _read(request_path), _read(response_path)
        intent, reserved = _one(intents, cid), _one(attempts, cid)
        claim, usage = claims.get(cid) or {}, (claims.get(cid) or {}).get("settlement_usage") or {}
        if (call.get("success") is not False or call.get("failure_kind") != "output_truncated"
                or call.get("finish_reason") != "length" or call.get("transport_outcome") != "confirmed_response"
                or call.get("unconfirmed_charge_possible") is not False or call.get("budget_owner") != owner_id
                or call.get("resource_claim_id") != cid
                or call.get("candidate_id") != candidate_id or call.get("attempt_id") != attempt["attempt_id"]
                or call.get("input_hash") != file_sha256(source) or response.get("validation") != "output_truncated"
                or request.get("role_model_config") != resolved.semantics()
                or call.get("role_config_hash") != resolved.role_config_hash
                or claim.get("call_id") != cid or claim.get("role") != "candidate_reviewer"
                or claim.get("status") != "settled" or claim.get("owner_id") != owner_id
                or (claim.get("actual") or {}).get("llm_calls") != 1
                or (claim.get("committed") or {}).get("llm_calls") != 1
                or claim.get("operation_id") != operation or usage.get("transport_outcome") != "confirmed_response"
                or usage.get("unconfirmed_charge_possible") is not False or claim.get("settlement_digest") != digest(usage)):
            raise ValueError("review_truncation_original_call_not_confirmed")
        for record in (intent, reserved):
            for key in ("operation_id", "budget_owner", "resource_claim_id", "role_config_hash", "request_trace_ref",
                        "request_system_sha256", "request_user_sha256", "role"):
                if record.get(key) != call.get(key):
                    raise ValueError("review_truncation_original_request_identity_changed")
        for key in ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "input_hash"):
            if request.get(key) != call.get(key) or response.get(key) != call.get(key):
                raise ValueError("review_truncation_original_response_identity_changed")
        for field in ("system", "user"):
            sha = hashlib.sha256(str(request.get(field, "")).encode()).hexdigest()
            if sha != call.get("request_" + field + "_sha256") or sha != request.get(field + "_sha256"):
                raise ValueError("review_truncation_original_wire_changed")
        if digest({"system": request["system"], "user": request["user"], "transport_fields": request.get("transport_fields") or {}}) != call.get("context_request_digest"):
            raise ValueError("review_truncation_original_prepared_request_changed")
        raw = response.get("reply")
        if not isinstance(raw, str) or raw[:2048] != call.get("raw_excerpt"):
            raise ValueError("review_truncation_original_paid_prefix_changed")
        wire = json.loads(request["user"])
        for key, value in (("source_hash", file_sha256(source)), ("check_sha256", file_sha256(check_path))):
            if resolve_request_pointer(wire, "/requirement_evidence_identity/" + key) != value:
                raise ValueError("review_truncation_original_evidence_changed")
        for field in ("input_tokens", "output_tokens"):
            if usage.get(field) != call.get(field) or (claim.get("actual") or {}).get(field) != call.get(field):
                raise ValueError("review_truncation_original_usage_changed")
        proofs.append({"call_id": cid, "request_sha256": file_sha256(request_path),
            "response_sha256": file_sha256(response_path), "settlement_digest": claim["settlement_digest"]})
    return {"schema_version": "eeg_research.review_output_revalidation.v1", "candidate_id": candidate_id,
        "attempt_id": attempt["attempt_id"], "plan_id": plan["plan_id"], "budget_owner": owner_id,
        "parent_operation_id": operation, "source_hash": file_sha256(source), "check_sha256": file_sha256(check_path),
        "spec_hash": approved["spec_hash"], "goal_sha256": file_sha256(camp / "goal.json"), "original_calls": proofs,
        "original_role_config": resolved.semantics(),
        "role_config": resolve_role_config(camp, "candidate_reviewer", {"max_tokens": 32768}).semantics()}


def authorize(camp, proof, reason):
    from .loop import _read, _write
    identity = digest(proof)
    record = {**proof, "authorization_id": "review_output_" + identity[:32],
        "operation_id": "review_revalidation_" + identity[:32], "reason": reason,
        "authority": "same_owner_bounded_output_retry_no_new_allocation"}
    path = Path(camp) / "review_revalidations" / (record["authorization_id"] + ".json")
    if path.is_file():
        saved = _read(path)
        if {key: value for key, value in saved.items() if key != "reason"} != {key: value for key, value in record.items() if key != "reason"}:
            raise ValueError("review_output_revalidation_authorization_changed")
        return saved
    _write(path, record, exclusive=True)
    return record


def authorized_config(camp, workspace, state, original_config_hash):
    """Use an explicitly revalidated review on ordinary resume at Goal routing."""
    from .binding import file_sha256
    from .identity import source_hash
    attempt = _read(workspace / "attempt.json")
    plan = state.get("selected_plan") or {}
    for path in (Path(camp) / "review_revalidations").glob("review_output_*.json"):
        record = _read(path)
        if (record.get("candidate_id") != workspace.name or record.get("attempt_id") != attempt.get("attempt_id")
                or record.get("plan_id") != plan.get("plan_id")
                or (record.get("original_role_config") or {}).get("role_config_hash") != original_config_hash):
            continue
        proof = {key: value for key, value in record.items() if key not in {"authorization_id", "operation_id", "reason", "authority"}}
        if (record.get("authorization_id") != "review_output_" + digest(proof)[:32]
                or record.get("operation_id") != "review_revalidation_" + digest(proof)[:32]
                or record.get("source_hash") != source_hash(workspace)
                or record.get("check_sha256") != file_sha256(workspace / "checks.json")
                or record.get("goal_sha256") != file_sha256(Path(camp) / "goal.json")):
            raise ValueError("review_revalidation_authorized_evidence_changed")
        for call in record["original_calls"]:
            cid = call["call_id"]
            if (call["request_sha256"] != file_sha256(Path(camp) / "llm_requests" / (cid + ".json"))
                    or call["response_sha256"] != file_sha256(Path(camp) / "llm_responses" / (cid + ".json"))):
                raise ValueError("review_revalidation_original_paid_evidence_changed")
        return record
    return None


def paid_review_proof(camp, state, candidate_id):
    """Verify a stopped paid reply under the existing 32k authorization, read only."""
    from .binding import file_sha256
    from .role_config import resolve_role_config
    from .llm import transport_options
    from .selected_plan import load_plan
    from .schemas import ImplementationReview
    from .review_context import review_reference_errors
    from .paid_research_round import readonly_registry
    from .context.representations import resolve_request_pointer, validate_pointer_closure
    from .context.sources import pointer_escape
    from .plan_protocol import scientific_projection
    camp = Path(camp).resolve()
    workspace = camp / "candidates" / candidate_id
    original_config = resolve_role_config(camp, "candidate_reviewer")
    authorization = authorized_config(camp, workspace, state, original_config.role_config_hash)
    if not authorization:
        return None
    failure = state.get("failure") or {}
    if (failure.get("phase") != "review_candidate" or failure.get("error_type") != "domain_schema_invalid"
            or failure.get("recoverable") is not True or state.get("detail") != "domain_schema_invalid"):
        return None
    proof = _original_proof(camp, state, candidate_id)
    saved_proof = {key: value for key, value in authorization.items()
                   if key not in {"authorization_id", "operation_id", "reason", "authority"}}
    if proof != saved_proof:
        raise ValueError("review_paid_recovery_authorized_scope_changed")
    resolved = resolve_role_config(camp, "candidate_reviewer", {"max_tokens": 32768})
    if resolved.semantics() != authorization["role_config"]:
        raise ValueError("review_paid_recovery_profile_changed")
    operation = authorization["operation_id"]
    calls = [row for row in _rows(camp, "llm_calls.jsonl") if row.get("operation_id") == operation]
    attempts = [row for row in _rows(camp, "api_operation_attempts.jsonl") if row.get("operation_id") == operation]
    intents = [row for row in _rows(camp, "llm_api_intents.jsonl") if row.get("operation_id") == operation]
    if (len(calls) != 2 or {row["call_id"] for row in calls} != {row["call_id"] for row in attempts}
            or {row["call_id"] for row in calls} != {row["call_id"] for row in intents}):
        raise ValueError("review_paid_recovery_requires_original_two_confirmed_attempts")
    with sqlite3.connect((camp / "resource_owners.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        claims = {key: json.loads(value) for key, value in db.execute("SELECT id,payload FROM claims")}
    plan = load_plan(camp, state)
    evidence = []
    for call in calls:
        cid = call["call_id"]
        request_path, response_path = camp / "llm_requests" / (cid + ".json"), camp / "llm_responses" / (cid + ".json")
        request, response = _read(request_path), _read(response_path)
        claim = claims.get(cid) or {}
        usage = claim.get("settlement_usage") or {}
        if (call.get("role") != "candidate_reviewer" or call.get("success") is not True
                or call.get("finish_reason") != "stop" or call.get("transport_outcome") != "confirmed_response"
                or call.get("unconfirmed_charge_possible") is True or call.get("budget_owner") != proof["budget_owner"]
                or call.get("resource_claim_id") != cid or call.get("candidate_id") != candidate_id
                or call.get("attempt_id") != proof["attempt_id"] or call.get("input_hash") != proof["source_hash"]
                or response.get("validation") != "review_evidence_reference_invalid"
                or call.get("role_config_hash") != resolved.role_config_hash
                or request.get("role_config_hash") != resolved.role_config_hash
                or request.get("role_model_config") != resolved.semantics()
                or call.get("role_model_config") != resolved.semantics()
                or request.get("transport_fields") != transport_options(resolved.profile())
                or claim.get("call_id") != cid or claim.get("role") != "candidate_reviewer"
                or claim.get("status") != "settled" or claim.get("owner_id") != proof["budget_owner"]
                or claim.get("operation_id") != operation or usage.get("transport_outcome") != "confirmed_response"
                or usage.get("unconfirmed_charge_possible") is not False or claim.get("settlement_digest") != digest(usage)
                or (claim.get("actual") or {}).get("llm_calls") != 1 or (claim.get("committed") or {}).get("llm_calls") != 1):
            raise ValueError("review_paid_recovery_call_not_confirmed")
        for record in (_one(attempts, cid), _one(intents, cid)):
            for key in ("role", "operation_id", "budget_owner", "resource_claim_id", "role_config_hash", "prompt_hash",
                        "request_trace_ref", "request_system_sha256", "request_user_sha256"):
                if record.get(key) != call.get(key):
                    raise ValueError("review_paid_recovery_native_request_identity_changed")
        if _one(intents, cid).get("context_request_digest") != call.get("context_request_digest"):
            raise ValueError("review_paid_recovery_native_prepared_request_changed")
        for key in ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "input_hash"):
            if request.get(key) != call.get(key) or response.get(key) != call.get(key):
                raise ValueError("review_paid_recovery_trace_identity_changed")
        for field in ("system", "user"):
            sha = hashlib.sha256(str(request.get(field, "")).encode()).hexdigest()
            if sha != call.get("request_" + field + "_sha256") or sha != request.get(field + "_sha256"):
                raise ValueError("review_paid_recovery_original_wire_changed")
        if (call.get("prompt_hash") != request["system_sha256"][:16]
                or digest({"system": request["system"], "user": request["user"],
                           "transport_fields": request["transport_fields"]}) != call.get("context_request_digest")
                or not isinstance(response.get("reply"), dict)
                or digest(response["reply"]) != call.get("response_reply_sha256")):
            raise ValueError("review_paid_recovery_request_or_reply_sha_changed")
        for field in ("input_tokens", "output_tokens"):
            if usage.get(field) != call.get(field) or (claim.get("actual") or {}).get(field) != call.get(field):
                raise ValueError("review_paid_recovery_settlement_usage_changed")
        wire = json.loads(request["user"])
        if not validate_pointer_closure(wire)[0]:
            raise ValueError("review_paid_recovery_original_pointer_changed")
        def expand(value, path=""):
            if isinstance(value, dict):
                if "$context_pointer" in value:
                    return expand(resolve_request_pointer(wire, path), value["$context_pointer"])
                return {key: expand(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
            if isinstance(value, list):
                return [expand(child, path + "/" + str(index)) for index, child in enumerate(value)]
            return value
        frozen = expand(wire)
        identity = frozen.get("requirement_evidence_identity") or {}
        if (identity.get("source_hash") != proof["source_hash"] or identity.get("check_sha256") != proof["check_sha256"]
                or frozen.get("candidate_source") != (workspace / "extension/eeg_candidate.py").read_text()
                or frozen.get("origin_plan_id") != proof["plan_id"]
                or frozen.get("selected_research_task") != {key: plan["proposal"][key] for key in (
                    "question", "task_description", "scientific_intent", "evaluation_scope", "expected_observation", "completion_condition")
                    if key in plan["proposal"]}
                or digest(scientific_projection((frozen.get("experiment_spec") or {}).get("experiment") or {}))
                    != plan.get("approved_scientific_digest")):
            raise ValueError("review_paid_recovery_original_scientific_evidence_changed")
        manifest_ref = (call.get("context_sidecars") or {}).get("manifest")
        if not manifest_ref:
            raise ValueError("review_paid_recovery_delivery_manifest_missing")
        manifest = _read(Path(manifest_ref))
        registry, source_rows = readonly_registry(camp), []
        for source in manifest.get("sources", []):
            current = str(source["block_id"]).startswith("closed:")
            block = registry.get(source["source_ref"], role="candidate_reviewer", require_current=current)
            registry.load(block, role="candidate_reviewer", require_current=current)
            if block.source_hash != source["source_hash"]:
                raise ValueError("review_paid_recovery_delivery_source_changed")
            source_rows.append({"block_id": block.block_id, "source_ref": block.source_ref,
                "source_hash": block.source_hash, "revision": block.source_revision, "visibility": list(block.visibility)})
        if digest(source_rows) != manifest.get("source_dependency_digest"):
            raise ValueError("review_paid_recovery_delivery_manifest_changed")
        evidence.append({"call_id": cid, "request_sha256": file_sha256(request_path),
            "response_sha256": file_sha256(response_path), "settlement_digest": claim["settlement_digest"]})
    # Use the latest paid correction; preserve its judgment and every source ref.
    reply = response["reply"]
    ImplementationReview.model_validate(reply)
    if review_reference_errors(reply, frozen):
        raise ValueError("review_paid_recovery_original_references_still_invalid")
    return {"authorization": authorization, "proof": proof, "reply": reply, "request": frozen,
        "receipt": {"schema_version": "eeg_research.paid_review_recovery.v1", "authorization_id": authorization["authorization_id"],
            "operation_id": operation, "call_id": call["call_id"], "original_calls": evidence,
            "request_digest": call["context_request_digest"], "reply_digest": call["response_reply_sha256"],
            "api_calls_added": 0, "training_jobs_added": 0}}


def record_paid_review(camp, recovered):
    from .loop import _write
    receipt = recovered["receipt"]
    path = Path(camp) / "review_revalidations" / (receipt["call_id"] + ".recovered.json")
    if path.is_file():
        if _read(path) != receipt:
            raise ValueError("review_paid_recovery_receipt_changed")
    else:
        _write(path, receipt, exclusive=True)
