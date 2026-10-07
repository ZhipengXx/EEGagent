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


def _ledger(camp: Path, row: dict[str, Any]) -> None:
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cost_path = camp / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.is_file() else {}
    cost["llm_calls"] = int(cost.get("llm_calls", 0)) + 1
    cost["llm_failures"] = int(cost.get("llm_failures", 0)) + (0 if row.get("success") else 1)
    for key in ("input_tokens", "output_tokens"):
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
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_hash", "requested_model")}
    record = {**identity, "system": system, "user": user,
              "system_sha256": system_sha, "user_sha256": user_sha,
              "profile": "fast", "max_tokens": max_tokens,
              "status": "prepared_before_transport", "recorded_at": time.time(),
              "scope": "Exact application-level system/user strings supplied to complete_json; no transport credentials or HTTP wire-body claim."}
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
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_hash",
                 "request_trace_ref", "request_system_sha256", "request_user_sha256")}
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

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        task = None
        if role in ENVELOPE_ROLES:
            from react_agent.eeg_research.agentic.roles import task_identity_from_payload

            task = task_identity_from_payload(payload, bound)
            if task is None:
                raise LlmUnavailable("role_result_task_missing")
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
            goal_path = camp / "goal.json"
            if goal_path.is_file():
                try:
                    goal = json.loads(goal_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    goal = {}
                limit = goal.get("max_llm_calls")
                cost_path = camp / "cost.json"
                used = 0
                if cost_path.is_file():
                    try:
                        used = int(json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls") or 0)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        used = 0
                if isinstance(limit, int) and used >= limit:
                    raise LlmUnavailable("budget_exhausted")
            started = time.time()
            row = new_call_row(
                role=role,
                prompt_hash=prompt_hash,
                requested_model=config.fast.model,
                started_at=started,
                **bound,
            )
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
    return call
