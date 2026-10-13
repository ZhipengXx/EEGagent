"""Durable application attempts, shared Planner round limits and unknown charges."""
from __future__ import annotations

import contextlib
import contextvars
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

ROUND = contextvars.ContextVar("eeg_planner_api_round", default=None)
ROUND_AUTHORIZATION = contextvars.ContextVar("eeg_planner_round_authorization", default=None)
ATTEMPTS = "api_operation_attempts.jsonl"
AUTHORIZATION_BINDINGS = "planner_authorization_bindings.jsonl"


class ApiRequestPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_attempts_per_operation: int = Field(default=2, ge=1, le=2)
    max_attempts_per_planner_round: int = Field(default=6, ge=1, le=100)
    unknown_usd_reserve: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    allow_output_budget_adjustment: bool = False
    max_output_tokens: int = Field(default=65536, ge=1, le=131072)
    output_retry_timeout_s: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)
    transport_backoff_s: float = Field(default=.25, ge=0, le=60, allow_inf_nan=False)


def validate_api_request_policy(value: dict) -> dict:
    return ApiRequestPolicy.model_validate(value).model_dump()


def api_request_policy(camp: Path) -> ApiRequestPolicy:
    path = camp / "goal.json"
    goal = json.loads(path.read_text()) if path.is_file() else {}
    return ApiRequestPolicy.model_validate(goal.get("api_request_policy") or {})


@contextlib.contextmanager
def planner_api_round(camp: Path, round_id: str, max_attempts: int = 6):
    with authorized_planner_round(camp, round_id, max_attempts=max_attempts):
        yield


def _authorization_completion(camp: Path, authorization: dict) -> str | None:
    """A created decision or legacy inline flag cannot complete recovery."""
    from .decision_receipts import read_projected_decision
    path = camp / "campaign_state.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    before = set(authorization["decision_ids_at_authorization"])
    for decision in state.get("decisions") or []:
        decision_id = str(decision.get("decision_id") or "")
        if not decision_id or decision_id in before or decision.get("ok") is not True or decision.get("executed") is not True:
            continue
        projected = read_projected_decision(camp, decision_id)["decision"]
        if (projected.get("ok") is True and projected.get("executed") is True
                and projected.get("receipt_status") == "executed" and projected.get("consumed") is True
                and projected.get("decision_receipt") == decision.get("decision_receipt")):
            return decision_id
    return None


def _bind_authorized_round(camp: Path, requested_round: str, authorization: dict) -> dict:
    identity = {"authorization_id": authorization["authorization_id"],
        "authorization_hash": authorization["authorization_hash"],
        "parent_planner_round_id": authorization["parent_planner_round_id"],
        "parent_planner_call_ids": list(authorization["parent_planner_call_ids"]),
        "max_recovery_calls": min(2, int(authorization["planner_recovery_max_calls"]))}
    child = "authorized_round_" + hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()[:32]
    path = camp / AUTHORIZATION_BINDINGS
    with (camp / "api_operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
        for row in rows:
            if row.get("authorization_id") == identity["authorization_id"]:
                if any(row.get(key) != value for key, value in identity.items()) or row.get("child_planner_round_id") != child:
                    raise ValueError("planner_authorization_binding_conflict")
                return row
        row = {"schema_version": "eeg.planner_authorization_binding.v1", **identity,
            "child_planner_round_id": child, "initial_requested_round_id": requested_round,
            "created_at": time.time()}
        with path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        return row


@contextlib.contextmanager
def authorized_planner_round(camp: Path, round_id: str, max_attempts: int = 6):
    if not round_id or not 1 <= max_attempts <= 100:
        raise ValueError("invalid_planner_api_round")
    max_attempts = min(max_attempts, api_request_policy(camp).max_attempts_per_planner_round)
    from .native_authorization import current_authorization
    authorization = current_authorization(camp)
    binding = None
    if authorization is not None:
        binding = _bind_authorized_round(camp, round_id, authorization)
        if _authorization_completion(camp, authorization) is None:
            round_id = binding["child_planner_round_id"]
            max_attempts = min(max_attempts, binding["max_recovery_calls"])
        else:
            binding = None
    token = ROUND.set((str(camp.resolve()), round_id, max_attempts))
    authorization_token = ROUND_AUTHORIZATION.set(binding)
    try:
        yield
    finally:
        ROUND_AUTHORIZATION.reset(authorization_token)
        ROUND.reset(token)


@contextlib.contextmanager
def authorized_memory_api_scope(camp: Path, *, base_turn_id: str, recovery_turn_id: str,
                                authorization_ref: str):
    """Only the separately authorized memory turn may leave Planner's quota."""
    from .native_authorization import current_authorization
    authorization = current_authorization(camp)
    if (authorization is None or authorization_ref != authorization["authorization_hash"]
            or base_turn_id != authorization["parent_memory_turn_id"]
            or recovery_turn_id != authorization["memory_recovery_turn"]):
        raise ValueError("memory_authorization_scope_mismatch")
    token = ROUND.set(None)
    authorization_token = ROUND_AUTHORIZATION.set(None)
    try:
        yield
    finally:
        ROUND_AUTHORIZATION.reset(authorization_token)
        ROUND.reset(token)


def operation_id(role: str, payload: dict, bound: dict, config_hash: str) -> str:
    """A reentry of the same step/request retains its attempt allowance."""
    if bound.get("operation_id"):
        prefix = str(bound["operation_id"])
        if role == "candidate_coder":
            step = payload.get("step")
            if not isinstance(step, int) or step < 1:
                raise ValueError("coder_api_operation_step_required")
            return prefix + ":coder_step:" + str(step)
        return prefix
    def semantic(value):
        if isinstance(value, dict):
            return {key: semantic(item) for key, item in value.items() if key not in {
                "task_id", "attempt_id", "input_digest", "heartbeat", "timestamp", "recorded_at",
                "elapsed_seconds", "budget", "role_context_projection", "planner_context_projection"}}
        return [semantic(item) for item in value] if isinstance(value, list) else value
    # Task IDs can change following an exception. Semantic request/step/source
    # identity remains the authority, with routing configuration explicitly bound.
    value = {"role": role, "request": semantic(payload), "binding": semantic(bound), "config": config_hash}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), default=str).encode()).hexdigest()
    return "op_" + digest


def reserve_operation_attempt(camp: Path, *, operation: str, call_id: str,
                              max_attempts: int = 2, reserve_usd: float | None = None,
                              metadata: dict | None = None) -> dict:
    """Write before transport, including crashes before the main intent journal."""
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    if not 1 <= max_attempts <= 2:
        raise ValueError("invalid_operation_attempt_limit")
    max_attempts = min(max_attempts, api_request_policy(camp).max_attempts_per_operation)
    camp.mkdir(parents=True, exist_ok=True)
    round_ = ROUND.get()
    if round_ and round_[0] != str(camp.resolve()):
        raise LlmUnavailable("planner_api_round_campaign_mismatch")
    authorization = ROUND_AUTHORIZATION.get()
    if authorization is not None and (metadata or {}).get("role") != "research_planner":
        raise LlmUnavailable("planner_recovery_scope_role_mismatch")
    with (camp / "api_operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = camp / ATTEMPTS
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
        same_call = next((row for row in rows if row.get("call_id") == call_id), None)
        if same_call:
            if same_call.get("operation_id") != operation:
                raise LlmUnavailable("api_call_operation_identity_conflict")
            if any((metadata or {}).get(key) is not None and same_call.get(key) != (metadata or {}).get(key)
                   for key in ("role","requested_model","role_config_hash","request_system_sha256","request_user_sha256","input_digest")):
                raise LlmUnavailable("api_call_request_identity_conflict")
            return same_call
        used = sum(row["operation_id"] == operation for row in rows)
        if used >= max_attempts:
            raise LlmUnavailable("api_operation_attempt_limit", diagnostics={"operation_id": operation, "actual_attempts": used})
        if round_ and sum(row.get("planner_round_id") == round_[1] for row in rows) >= round_[2]:
            raise LlmUnavailable("planner_round_attempt_limit", diagnostics={"planner_round_id": round_[1]})
        safe_metadata = {key: (metadata or {}).get(key) for key in ("role", "task_id", "attempt_id", "input_digest",
            "prompt_hash", "requested_model", "role_config_hash", "role_model_config", "request_trace_ref",
            "request_system_sha256", "request_user_sha256", "research_authorization_ref", "parent_planner_round_id",
            "original_maintenance_turn_id", "maintenance_authorization_ref")}
        row = {**safe_metadata, "operation_id": operation, "call_id": call_id, "operation_attempt": used + 1,
               "planner_round_id": round_[1] if round_ else None,
               "max_operation_attempts": max_attempts, "max_round_attempts": round_[2] if round_ else None,
               "unknown_usd_reserve": reserve_usd, "recorded_at": time.time(), "status": "reserved_before_transport"}
        if authorization is not None:
            row.update(research_authorization_ref=authorization["authorization_hash"],
                       parent_planner_round_id=authorization["parent_planner_round_id"])
        from .owner_budget import enabled, claim_request, OwnerBudgetError
        if enabled(camp):
            report = (metadata or {}).get("context_budget_report") or {}
            fields = (metadata or {}).get("context_transport_fields") or {}
            input_bound = (metadata or {}).get("owner_input_token_bound") or report.get("model_tokens_with_safety_margin") or report.get("safe_input_tokens")
            if input_bound is None:
                input_bound = max(int(report.get("model_tokens") or 0), int((metadata or {}).get("owner_input_token_bound") or 0))
            output_bound = fields.get("max_tokens", (metadata or {}).get("owner_output_token_bound"))
            if type(input_bound) is not int or input_bound < 1 or type(output_bound) is not int or output_bound < 1:
                raise LlmUnavailable("owner_request_token_bounds_required")
            try:
                claim = claim_request(camp, call_id=call_id, operation=operation, role=(metadata or {}).get("role") or "backend_direct",
                    input_bound=input_bound, output_bound=output_bound)
            except OwnerBudgetError as exc:
                raise LlmUnavailable(str(exc)) from exc
            row.update(budget_owner=claim["owner_id"], resource_claim_id=call_id,
                owner_input_token_bound=input_bound, owner_output_token_bound=output_bound)
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return row


def unresolved_charge_ids(camp: Path) -> set[str]:
    path = camp / "llm_calls.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    unknown = {row["call_id"] for row in rows if row.get("unconfirmed_charge_possible") or row.get("transport_outcome") in {"outcome_unknown", "unknown_after_process_exit"}}
    path = camp / "api_unknown_dispositions.jsonl"
    dispositions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    permitted = {row["call_id"] for row in dispositions if row.get("action") == "abandon_response" and row.get("allow_new_request") is True}
    return unknown.difference(permitted)


def dispose_unknown_intent(camp: Path, call_id: str, *, reason: str,
                           allow_new_request: bool = False) -> dict:
    """Explicit abandonment never refunds attempts or unknown cost/reservation."""
    from react_agent.eeg_research.agentic.llm import refresh_method_api_cost, _ledger
    from react_agent.eeg_research.agentic.loso_study import worker_alive
    if not reason.strip():
        raise ValueError("api_unknown_disposition_reason_required")
    if worker_alive(camp):
        raise ValueError("api_unknown_disposition_worker_still_alive")
    cost = refresh_method_api_cost(camp)
    rows = [json.loads(line) for line in (camp / "llm_api_intents.jsonl").read_text().splitlines() if line.strip()] if (camp / "llm_api_intents.jsonl").is_file() else []
    intent = next((row for row in rows if row.get("call_id") == call_id), None)
    if not intent:
        path = camp / ATTEMPTS
        operations = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
        intent = next((row for row in operations if row.get("call_id") == call_id), None)
    if not intent:
        raise ValueError("api_unknown_intent_not_found")
    if (camp / "llm_responses" / (call_id + ".json")).is_file():
        raise ValueError("api_unknown_durable_response_requires_adoption")
    if call_id in cost.get("api_unsettled_call_ids", []):
        row = {key: value for key, value in intent.items() if key not in {"status", "uncertain_if_unsettled"}}
        row.update(success=False, error="abandoned_unknown_response", transport_outcome="unknown_after_process_exit",
                   unconfirmed_charge_possible=True, input_tokens=None, output_tokens=None, reasoning_tokens=None,
                   budget_charge_calls=1, cost_status="uncertain")
        _ledger(camp, row)
    elif call_id not in unresolved_charge_ids(camp):
        raise ValueError("api_intent_not_unresolved")
    receipt = {"call_id": call_id, "action": "abandon_response", "reason": reason,
               "allow_new_request": allow_new_request, "unknown_charge_retained": True,
               "call_charge_retained": True, "at": time.time()}
    with (camp / "api_unknown_dispositions.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt
