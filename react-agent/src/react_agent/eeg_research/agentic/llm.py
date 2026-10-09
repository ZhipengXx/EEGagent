"""Role calls through the existing DeepSeek client. Every call is written to the cost ledger."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.identity import new_call_row
from react_agent.eeg_research.agentic.roles import ENVELOPE_ROLES
from react_agent.eeg_research.agentic.ui_events import append_ui_event

PROMPTS = Path(__file__).resolve().parent / "prompts"
PARSE_ATTEMPTS = 3
CODER_MAX_TOKENS = 16384
FAST_MAX_TOKENS = 4096
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
RAW_EXCERPT_LIMIT = 2048
API_INTENTS = "llm_api_intents.jsonl"


class LlmUnavailable(RuntimeError):
    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def domain_model_for(role: str, memory=None):
    from react_agent.eeg_research.agentic.schemas import DOMAIN_OUTPUT_MODELS

    if role == "memory_curator" and memory is not None and memory.enabled and memory.skills_enabled:
        from react_agent.eeg_research.agentic.skill_memory import SkillLessonProposal
        return SkillLessonProposal
    return DOMAIN_OUTPUT_MODELS.get(role)


def system_prompt(role: str, *, memory=None) -> str:
    from react_agent.eeg_research.agentic.schemas import role_output_schema

    shared = (PROMPTS / "shared_contract.txt").read_text(encoding="utf-8")
    mission = (PROMPTS / f"{role}.txt").read_text(encoding="utf-8")
    schema = role_output_schema(role)
    if memory is not None and memory.enabled:
        extension = {"research_planner": "retrieved_memory", "experiment_designer": "retrieved_procedural_hints",
                     "candidate_coder": "retrieved_procedural_hints"}.get(role)
        if role == "memory_curator" and memory.skills_enabled:
            extension = "procedural_skill_extension"
            schema = json.dumps(domain_model_for(role, memory).model_json_schema(), ensure_ascii=False)
        if extension:
            mission += "\n\n" + (PROMPTS / (extension + ".txt")).read_text(encoding="utf-8")
    return shared + "\n\n" + mission + "\n\nOUTPUT_SCHEMA\n" + schema + "\nReturn one JSON object only.\n"


def _method_api_accounting(camp: Path) -> bool:
    path = camp / "goal.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("evaluation_mode") == "loso_method_search"


def _write_cost(camp: Path, cost: dict[str, Any]) -> None:
    target = camp / "cost.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def refresh_method_api_cost(camp: Path) -> dict[str, Any]:
    """Reconcile native intent reservations and settled rows after a crash.

    One reserved application attempt always consumes a call. If transport may
    have occurred before persistence, the unmatched call remains uncertain and
    is never relabeled zero/exactly-once. Tokens remain unknown until settled.
    This opt-in reconciliation leaves GPU/external-budget fields intact.
    """
    cost_path = camp / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.is_file() else {}
    if not _method_api_accounting(camp):
        return cost
    intents = {}
    path = camp / API_INTENTS
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("call_id") or row.get("status") != "reserved_before_transport":
                    raise LlmUnavailable("api_intent_ledger_invalid")
                intents[str(row["call_id"])] = row
    settled = {}
    calls = camp / "llm_calls.jsonl"
    if calls.is_file():
        for index, line in enumerate(calls.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                row = json.loads(line)
                key = str(row.get("call_id") or ("legacy_row_" + str(index)))
                if key in settled and settled[key] != row:
                    raise LlmUnavailable("call_id_ledger_conflict")
                settled[key] = row
    pending = sorted(set(intents).difference(settled))
    cost["llm_calls"] = len(set(settled).union(intents))
    cost["llm_failures"] = sum(row.get("success") is not True for row in settled.values())
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        known = [row[key] for row in settled.values() if isinstance(row.get(key), int)]
        if known or key in cost:
            cost[key] = sum(known)
        cost[key + "_known_call_count"] = len(known)
    cost.update(api_unsettled_call_ids=pending, api_uncertain_calls=len(pending),
                api_intent_accounting="application_attempt_reserved_before_transport_v1",
                api_usd=None, pricing_source=None,
                cost_status="uncertain" if pending else "unpriced")
    cost.setdefault("gpu_seconds_used", 0.0)
    _write_cost(camp, cost)
    return cost


def _reserve_api_intent(camp: Path, row: dict[str, Any]) -> None:
    if not _method_api_accounting(camp):
        return
    camp.mkdir(parents=True, exist_ok=True)
    # Durability precedes transport. A crash between fsync and cost.json rename
    # is reconciled from the append-only intent on the next native budget read.
    intent = {name: row.get(name) for name in
              ("call_id", "role", "task_id", "attempt_id", "input_digest", "prompt_hash",
               "request_trace_ref", "request_system_sha256", "request_user_sha256")}
    if row.get("operation"):
        intent["operation"] = row["operation"]
    intent.update(status="reserved_before_transport", recorded_at=time.time(),
                  uncertain_if_unsettled=True)
    with (camp / API_INTENTS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(intent, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    refresh_method_api_cost(camp)


def _ledger(camp: Path, row: dict[str, Any]) -> None:
    camp.mkdir(parents=True, exist_ok=True)
    method_mode = _method_api_accounting(camp)
    ledger_path = camp / "llm_calls.jsonl"
    if method_mode and row.get("call_id") and ledger_path.is_file():
        prior = next((json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and json.loads(line).get("call_id") == row["call_id"]), None)
        if prior is not None:
            if prior != row:
                raise LlmUnavailable("call_id_ledger_conflict")
            refresh_method_api_cost(camp)
            return
    with (camp / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if method_mode:
            handle.flush()
            os.fsync(handle.fileno())
    if method_mode:
        refresh_method_api_cost(camp)
        return
    cost_path = camp / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.is_file() else {}
    cost["llm_calls"] = int(cost.get("llm_calls", 0)) + 1
    cost["llm_failures"] = int(cost.get("llm_failures", 0)) + (0 if row.get("success") else 1)
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        value = row.get(key)
        if isinstance(value, int):
            cost[key] = int(cost.get(key, 0)) + value
    cost["api_usd"] = None
    cost["pricing_source"] = None
    cost["cost_status"] = "unpriced"
    cost.setdefault("gpu_seconds_used", 0.0)
    cost_path.write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_trace(camp: Path, row: dict[str, Any], *, system: str,
                   user: str, max_tokens: int) -> dict[str, Any]:
    """Persist the exact application-level role input before its API attempt.

    Transport configuration and credentials are deliberately not serialized.
    This is the complete_json input, not a claim about raw HTTP wire bytes.
    """
    folder = camp / "llm_requests"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / (str(row["call_id"]) + ".json")
    system_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    user_sha = hashlib.sha256(user.encode("utf-8")).hexdigest()
    identity = {key: row.get(key) for key in
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_digest", "input_hash", "requested_model")}
    if row.get("operation"):
        identity["operation"] = row["operation"]
    record = {**identity, "system": system, "user": user,
              "system_sha256": system_sha, "user_sha256": user_sha,
              "profile": "fast", "max_tokens": max_tokens,
              "status": "prepared_before_transport", "recorded_at": time.time(),
              "scope": "Exact application-level system/user strings supplied to complete_json; no transport credentials or HTTP wire-body claim."}
    record["context_measurement"] = {"system_chars": len(system), "user_chars": len(user),
                                     "total_application_chars": len(system) + len(user)}
    temp = path.with_suffix(".json.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return {"request_trace_ref": str(path), "request_system_sha256": system_sha,
            "request_user_sha256": user_sha}


def _response_trace(camp: Path, row: dict[str, Any], reply: Any, *, validation: str,
                    schema_errors: Any = None, normalization: Any = None) -> None:
    """Keep the exact model response and validation outcome, without credentials."""
    folder = camp / "llm_responses"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (str(row["call_id"]) + ".json")
    identity = {key: row.get(key) for key in
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_digest", "input_hash",
                 "request_trace_ref", "request_system_sha256", "request_user_sha256")}
    if row.get("operation"):
        identity["operation"] = row["operation"]
    record = {**identity, "reply": reply, "validation": validation,
              "schema_errors": schema_errors, "recorded_at": time.time()}
    if normalization:
        record["normalization"] = normalization
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _fail_row(row: dict[str, Any], exc: BaseException, started: float) -> dict[str, Any]:
    row.update({"success": False, "error": type(exc).__name__, "cost_status": "uncertain", "elapsed_seconds": round(time.time() - started, 3)})
    usage = getattr(exc, "usage", None)
    if usage is not None:
        row.update(
            {
                "response_model": usage.response_model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "finish_reason": usage.finish_reason,
                "provider_error": usage.error,
            }
        )
    causes = []
    cause = exc.__cause__
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    if causes:
        row["error_causes"] = causes
    raw = getattr(exc, "raw", None)
    if isinstance(raw, str):
        row["raw_excerpt"] = raw[:RAW_EXCERPT_LIMIT]
    return row


def _check_api_budget(camp: Path) -> None:
    """Apply the same native intent reconciliation to roles and compaction."""
    goal_path = camp / "goal.json"
    if not goal_path.is_file():
        return
    try:
        goal = json.loads(goal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        goal = {}
    used = 0
    cost_path = camp / "cost.json"
    if cost_path.is_file():
        try:
            used = int(json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            used = 0
    if goal.get("evaluation_mode") == "loso_method_search":
        reconciled = refresh_method_api_cost(camp)
        used = int(reconciled.get("llm_calls") or 0)
        if reconciled.get("api_unsettled_call_ids"):
            raise LlmUnavailable("unsettled_api_intent_requires_recovery",
                diagnostics={"api_unsettled_call_ids": reconciled["api_unsettled_call_ids"],
                             "cost_status": "uncertain"})
    limit = goal.get("max_llm_calls")
    if isinstance(limit, int) and used >= limit:
        raise LlmUnavailable("budget_exhausted")


def _invoke_compaction(camp: Path, client: Any, loop: Any, *, model: str, max_tokens: int,
                       system: str, payload: dict, task: dict, validate: Callable,
                       context_budget_chars: int) -> dict:
    """One bounded delivery operation through the existing durable API ledger."""
    from react_agent.fmri.llm.deepseek import DeepSeekParseError
    from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies
    error = None
    for attempt in range(2):
        assert_role_task_dependencies(camp, task)
        request = dict(payload)
        if error is not None:
            request["summary_validation_error"] = error
        user = json.dumps(request, ensure_ascii=False, default=str, separators=(",", ":"))
        if len(system) + len(user) > context_budget_chars:
            raise LlmUnavailable("compaction_context_budget_exceeded")
        _check_api_budget(camp)
        started = time.time()
        row = new_call_row(role="memory_curator", operation="handoff_compaction",
            prompt_hash=hashlib.sha256(system.encode()).hexdigest()[:16],
            requested_model=model, started_at=started,
            **{key: task[key] for key in ("task_id", "attempt_id", "input_digest")})
        row.update(application_attempt=attempt + 1,
                   application_retry_kind="compaction_validation" if error else None)
        try:
            row.update(_request_trace(camp, row, system=system, user=user, max_tokens=max_tokens))
            _reserve_api_intent(camp, row)
            reply, usage = loop.run_until_complete(client.complete_json(
                system=system, user=user, profile="fast", role="memory_curator"))
        except DeepSeekParseError as exc:
            _ledger(camp, _fail_row(row, exc, started))
            _response_trace(camp, row, getattr(exc, "raw", None), validation="invalid_json")
            error = "Return one complete JSON object matching OUTPUT_SCHEMA."
            continue
        except Exception as exc:
            _ledger(camp, _fail_row(row, exc, started))
            raise LlmUnavailable(type(exc).__name__) from exc
        row.update(success=True, response_model=usage.response_model,
                   input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   reasoning_tokens=usage.reasoning_tokens, finish_reason=usage.finish_reason,
                   elapsed_seconds=round(time.time() - started, 3))
        _ledger(camp, row)
        _response_trace(camp, row, reply, validation="pending")
        try:
            checked = validate(reply)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:3000]
            _response_trace(camp, row, reply, validation="compaction_invalid", schema_errors=error)
            continue
        _response_trace(camp, row, reply, validation="valid")
        return checked
    raise LlmUnavailable("compaction_validation_failed", diagnostics={"reason": error})


def role_backend(camp: Path, role: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a callable that sends one structured input to DeepSeek for this role."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LlmUnavailable("DEEPSEEK_API_KEY missing")
    from react_agent.fmri.config import FmriCheckConfig
    from react_agent.fmri.llm.deepseek import DeepSeekBackend, DeepSeekParseError

    config = FmriCheckConfig(
        deepseek_api_key=key,
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        deepseek_trust_env=os.environ.get("DEEPSEEK_TRUST_ENV", "true").strip().lower() not in {"0", "false", "no"},
    )
    if os.environ.get("DEEPSEEK_FAST_MODEL"):
        config.fast.model = os.environ["DEEPSEEK_FAST_MODEL"]
    floor = CODER_MAX_TOKENS if role in LONG_JSON_ROLES else FAST_MAX_TOKENS
    config.fast.max_tokens = max(int(config.fast.max_tokens or 0), floor)
    from react_agent.eeg_research.agentic.embedding import memory_config
    memory = memory_config(camp)
    system = system_prompt(role, memory=memory)
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]
    loop = asyncio.new_event_loop()
    client = DeepSeekBackend(config)
    bound: dict[str, Any] = {}

    def bind(**fields: Any) -> None:
        bound.clear()
        bound.update({key: value for key, value in fields.items() if value is not None})

    def prepare_context(payload: dict[str, Any], *, inputs: list[Path]) -> dict:
        # Called before begin_role_task: the consumer digest binds the actual
        # semantic summary and its original dependencies, not an invisible view.
        if (role != "memory_curator" or payload.get("evaluation_mode") != "loso_method_search"
                or os.environ.get("EEG_HANDOFF_COMPACTION", "0").lower() not in {"1", "true", "on"}):
            return payload
        from react_agent.eeg_research.agentic.handoffs import compact_role_context
        from react_agent.eeg_research.agentic.role_examples import add_role_examples

        def fits(value: dict) -> bool:
            # Reserve exact-length task identity strings used by the native
            # ledger. Include system/schema/examples and the final JSON bytes.
            probe = {**value, "task_id": "task_" + "0" * 12,
                     "attempt_id": "0" * 12, "input_digest": "0" * 64}
            probe = compact_role_context(add_role_examples(probe, role, memory),
                                         role, system_chars=len(system))
            return bool(probe["role_context_projection"]["fits_budget"])

        if fits(payload):
            return payload
        from react_agent.eeg_research.agentic.compaction import compact_for_curator, protected_context
        if not fits(protected_context(payload)):
            raise LlmUnavailable("handoff_compaction_protected_budget_exceeded")

        def invoke(**kwargs):
            return _invoke_compaction(camp, client, loop, model=config.fast.model,
                                      max_tokens=int(config.fast.max_tokens), **kwargs)

        prepared = compact_for_curator(camp, payload, inputs=inputs,
                                        model=config.fast.model, invoke=invoke)
        if not fits(prepared):
            raise LlmUnavailable("handoff_compaction_output_budget_exceeded")
        return prepared

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        task = None
        if role in ENVELOPE_ROLES:
            from react_agent.eeg_research.agentic.roles import task_identity_from_payload

            task = task_identity_from_payload(payload, bound)
            if task is None:
                raise LlmUnavailable("role_result_task_missing")
            from react_agent.eeg_research.agentic.task_ledger import (
                find_completed_task, latest, record_task_reuse, recovery_identity_digest,
            )
            persisted = latest(camp, str(task["task_id"]))
            recovery = (persisted or {}).get("recovery_identity")
            if persisted and str((recovery or {}).get("action") or "").startswith("handoff_compaction:"):
                raise LlmUnavailable("delivery_operation_task_not_role_task")
            if persisted and persisted.get("role") == role and persisted.get("input_digest") == task.get("input_digest"):
                from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies, RoleResultError
                if persisted.get("status") == "completed" and "input_dependencies" not in (recovery or {}):
                    raise LlmUnavailable("completed_role_cache_dependency_proof_missing")
                try:
                    assert_role_task_dependencies(camp, persisted)
                except RoleResultError as exc:
                    raise LlmUnavailable(str(exc)) from exc
            # Reuse only an explicit current identity established by the native
            # begin-task consumer. A legacy empty prompt/digest is not guessed.
            if (persisted and persisted.get("role") == role and isinstance(recovery, dict)
                    and persisted.get("input_digest") == task.get("input_digest")
                    and recovery.get("prompt_hash") == prompt_hash
                    and persisted.get("recovery_identity_digest") == recovery_identity_digest(recovery)):
                cached = find_completed_task(camp, role=role, recovery_identity=recovery,
                                             candidate_id=persisted.get("candidate_id"),
                                             input_digest=task.get("input_digest"))
                if cached and cached["task"]["task_id"] == task["task_id"]:
                    record_task_reuse(camp, cached)
                    return cached["envelope"]
        last: DeepSeekParseError | None = None
        schema_error = None
        previous_domain = None
        schema_feedback = None
        for _attempt in range(PARSE_ATTEMPTS):
            call_payload = payload
            if last is not None:
                call_payload = {
                    **payload,
                    "previous_response_error": {
                        "kind": "invalid_json",
                        "finish_reason": getattr(last.usage, "finish_reason", None),
                        "instruction": "Return one complete JSON object matching OUTPUT_SCHEMA. Escape Python newlines and quotes inside content. Keep the implementation concise; do not omit requested interventions.",
                    },
                }
            if schema_error is not None:
                call_payload = {**call_payload, "schema_error": schema_error, "previous": previous_domain,
                                "schema_repair_context": schema_feedback,
                                "correction_instruction": "Return a complete corrected JSON object using only OUTPUT_SCHEMA fields. Input metadata is read-only, not output. Remove every reported forbidden field; correct every reported missing/invalid field. Keep the task, legal actions, evidence identities and business context unchanged."}
            from react_agent.eeg_research.agentic.role_examples import add_role_examples
            call_payload = add_role_examples(call_payload, role, memory)
            from react_agent.eeg_research.agentic.handoffs import compact_role_context
            call_payload = compact_role_context(call_payload, role, system_chars=len(system))
            projection = call_payload["role_context_projection"]
            if not projection["fits_budget"]:
                raise LlmUnavailable("role_context_budget_requires_artifact_pages",
                                     diagnostics={"role_context_projection": projection})
            _check_api_budget(camp)
            started = time.time()
            row = new_call_row(
                role=role,
                prompt_hash=prompt_hash,
                requested_model=config.fast.model,
                started_at=started,
                **bound,
            )
            if task is not None:
                row.update({name: task.get(name) for name in ("task_id", "attempt_id", "input_digest")})
                row.setdefault("input_hash", task.get("input_digest"))
            row.update(application_attempt=_attempt + 1,
                       application_retry_kind=("domain_schema" if schema_error is not None else
                                               "invalid_json" if last is not None else None),
                       role_context_revision=projection["revision"],
                       role_context_original_user_chars=projection["original_user_chars"],
                       role_context_projected_total_chars=projection["projected_total_chars"],
                       role_context_budget_chars=projection["budget_chars"])
            append_ui_event(
                camp,
                "llm_call_started",
                role=role,
                call_id=row.get("call_id"),
                candidate_id=bound.get("candidate_id"),
                task_id=bound.get("task_id"),
                attempt_id=bound.get("attempt_id"),
                job_id=bound.get("job_id"),
                status="active",
            )
            try:
                user_text = json.dumps(call_payload, ensure_ascii=False, default=str, separators=(",", ":"))
                row.update(_request_trace(camp, row, system=system, user=user_text,
                                          max_tokens=int(config.fast.max_tokens)))
                _reserve_api_intent(camp, row)
                reply, usage = loop.run_until_complete(
                    client.complete_json(
                        system=system,
                        user=user_text,
                        profile="fast",
                        role=role,
                    )
                )
            except DeepSeekParseError as exc:
                last = exc
                _ledger(camp, _fail_row(row, exc, started))
                _response_trace(camp, row, getattr(exc, "raw", None), validation="invalid_json")
                append_ui_event(
                    camp,
                    "llm_parse_retry",
                    role=role,
                    call_id=row.get("call_id"),
                    candidate_id=bound.get("candidate_id"),
                    status="failed",
                    error=type(exc).__name__,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                _ledger(camp, _fail_row(row, exc, started))
                append_ui_event(
                    camp,
                    "llm_call_failed",
                    role=role,
                    call_id=row.get("call_id"),
                    candidate_id=bound.get("candidate_id"),
                    status="failed",
                    error=type(exc).__name__,
                )
                raise LlmUnavailable(type(exc).__name__) from exc
            row.update(
                {
                    "success": True,
                    "response_model": usage.response_model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "finish_reason": usage.finish_reason,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            _ledger(camp, row)
            _response_trace(camp, row, reply, validation="pending")
            parsed = reply if isinstance(reply, dict) else {"_not_object": reply}
            normalization = None
            if role == "research_planner":
                from react_agent.eeg_research.agentic.planner_context import normalize_input_echoes
                parsed, removed = normalize_input_echoes(parsed, payload)
                if removed:
                    normalization = {"kind": "identical_read_only_input_echoes_removed",
                                     "removed_fields": removed, "normalized_reply": parsed,
                                     "raw_model_reply_preserved": True}
            bound_reply = None
            if role in ENVELOPE_ROLES:
                from react_agent.eeg_research.agentic.roles import RoleResultError, bind_role_output
                try:
                    bound_reply = bind_role_output(parsed, task=task, prompt_hash=prompt_hash)
                except RoleResultError as exc:
                    _response_trace(camp, row, reply, validation="role_identity_invalid", schema_errors=str(exc))
                    append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                    status="failed", error=str(exc))
                    raise LlmUnavailable(str(exc)) from exc
            if role != "candidate_coder":
                from pydantic import ValidationError
                domain_model = domain_model_for(role, memory)
                body = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else parsed
                identity_keys = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "artifact_refs", "candidate_id"}
                if domain_model is not None:
                    body = {key: value for key, value in body.items() if key not in identity_keys
                            and (key != "status" or "status" in domain_model.model_fields)}
                    try:
                        domain_model.model_validate(body)
                    except ValidationError as exc:
                        schema_error = json.dumps(exc.errors(include_input=False), ensure_ascii=False, default=str)[:3000]
                        _response_trace(camp, row, reply, validation="domain_schema_invalid",
                                        schema_errors=exc.errors(include_input=False))
                        previous_domain = body
                        errors = exc.errors(include_input=False)
                        schema_feedback = {
                            "allowed_top_level_fields": sorted(domain_model.model_fields),
                            "forbidden_field_paths": [list(row["loc"]) for row in errors if row["type"] == "extra_forbidden"],
                            "missing_field_paths": [list(row["loc"]) for row in errors if row["type"] == "missing"],
                            "validation_errors": errors[:12],
                        }
                        last = None
                        append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                        status="failed", error="domain_schema_invalid")
                        continue
            _response_trace(camp, row, reply, validation="valid", normalization=normalization)
            append_ui_event(camp, "llm_call_finished", role=role, call_id=row.get("call_id"),
                            candidate_id=bound.get("candidate_id"), status="completed")
            if bound_reply is not None:
                return bound_reply
            return parsed
        raise LlmUnavailable("domain_schema_invalid" if schema_error is not None else
                             (type(last).__name__ if last is not None else "DeepSeekParseError"),
                             diagnostics={"domain_schema_errors": schema_error} if schema_error is not None else {}) from last

    call.model = config.fast.model  # type: ignore[attr-defined]
    call.bind = bind  # type: ignore[attr-defined]
    call.prepare_context = prepare_context  # type: ignore[attr-defined]
    return call
