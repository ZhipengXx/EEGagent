"""DeepSeek JSON-decision backend via the OpenAI-compatible SDK."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from pydantic import ValidationError

from react_agent.fmri.config import FmriCheckConfig, LmProfile
from react_agent.fmri.prompts import DECISION_SCHEMA_TEXT, DECISION_SYSTEM, SUMMARY_SYSTEM
from react_agent.fmri.schemas import Decision, LMUsage

RAW_EXCERPT_LIMIT = 2048
_DSML_BLOCK = re.compile(r"<｜｜DSML｜｜[\s\S]*?(?:</｜｜DSML｜｜\s*calls>|$)", re.MULTILINE)
_PREFERRED_KEYS = ("tool", "action", "schema_version")


def message_text(message: Any) -> str:
    """Only the final answer is executable; reasoning is never final JSON."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return content if isinstance(content, str) else ""


def _legacy_message_text(message: Any) -> str:
    """Preserve the existing non-strict fMRI entrypoint's output convention."""
    content = message_text(message)
    if content.strip():
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and isinstance(message, dict):
        reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return content if isinstance(content, str) else ""


def strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_dsml(text: str) -> str:
    """Drop DeepSeek tool-call markup so a stray {} inside it is not the result."""
    return _DSML_BLOCK.sub("", text)


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, offset = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            found.append(payload)
        index = max(offset, start + 1)
    return found


def _choose_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = [item for item in objects if any(key in item for key in _PREFERRED_KEYS)]
    if preferred:
        return preferred[-1]
    return objects[-1] if objects else None


def parse_json_object(text: str) -> dict[str, Any]:
    """Read one JSON object. Fences are ignored. A second object or tool markup is refused.

    Scanning for any later object would execute a different command than the first one.
    """
    cleaned = strip_json_fence(text).strip()
    if _DSML_BLOCK.search(cleaned):
        raise json.JSONDecodeError("trailing_tool_markup", cleaned, 0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise exc
    if not isinstance(payload, dict):
        raise TypeError("json is not an object")
    return payload


def raw_excerpt(text: str, limit: int = RAW_EXCERPT_LIMIT) -> str:
    return (text or "")[:limit]


class DeepSeekConfigError(RuntimeError):
    """Raised when DeepSeek is requested without credentials."""


class DeepSeekBackend:
    """Call DeepSeek chat completions with JSON object output."""

    name = "deepseek"

    def __init__(self, config: FmriCheckConfig) -> None:
        if not config.deepseek_api_key:
            raise DeepSeekConfigError(
                "DEEPSEEK_API_KEY missing; refuse to fake an online call"
            )
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        self.config = config
        # EEG opts in at its application boundary. The existing fMRI callers
        # retain their transport and parsing behavior.
        self._requires_prepared_context = False
        self._budget_campaign = None
        self._client = AsyncOpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(trust_env=config.deepseek_trust_env),
        )

    def _profile(self, name: str) -> LmProfile:
        return self.config.reasoning if name == "reasoning" else self.config.fast

    async def aclose(self) -> None:
        await self._client.close()

    def enable_budget_protocol(self, camp) -> None:
        """Bind EEG's opt-in campaign even for backend entrypoints called directly."""
        from pathlib import Path
        self._budget_campaign = Path(camp)

    async def _complete(self, *, messages: list[dict[str, str]], profile: str,
                        request_fields: dict | None = None, role: str = "backend_direct",
                        budget_report: dict | None = None) -> tuple[str, LMUsage]:
        from react_agent.eeg_research.agentic import owner_budget as budget
        camp = self._budget_campaign or budget.current_campaign()
        if camp is None or not budget.enabled(camp):
            return await self._complete_transport(messages=messages, profile=profile, request_fields=request_fields)
        from uuid import uuid4
        from react_agent.eeg_research.agentic.api_policy import reserve_operation_attempt
        from react_agent.eeg_research.agentic.llm import _reserve_api_intent, _ledger, _price_usage
        spec = self._profile(profile)
        bounds = budget.input_token_bound(messages, request_fields, budget_report)
        current = budget.current_transport(camp)
        direct = current is None
        call_id = current[1] if current else uuid4().hex[:12]
        direct_row = None
        if direct:
            import hashlib
            operation = "backend_" + hashlib.sha256(json.dumps({"messages": messages,
                "model":spec.model,"max_tokens":spec.max_tokens,"fields":request_fields},sort_keys=True).encode()).hexdigest()
            direct_row = {"call_id":call_id,"role":role,"requested_model":spec.model,
                "owner_input_token_bound":bounds,"owner_output_token_bound":spec.max_tokens,
                "operation_id":operation,"started_at":time.time()}
            direct_row.update(reserve_operation_attempt(camp, operation=operation, call_id=call_id, metadata=direct_row))
            _reserve_api_intent(camp, direct_row)
        try:
            budget.before_transport(camp, call_id=call_id, role=role, input_bound=bounds, output_bound=spec.max_tokens)
        except budget.OwnerBudgetError as exc:
            not_sent = {"transport_outcome":"definitely_not_sent","unconfirmed_charge_possible":False,"success":False,"error":str(exc)}
            budget.settle_request(camp,call_id,not_sent)
            if direct_row is not None:
                _ledger(camp,{**direct_row,**not_sent})
            raise DeepSeekConfigError(str(exc)) from exc
        try:
            with budget.transport_claim(camp,call_id):
                text, usage = await self._complete_transport(messages=messages, profile=profile, request_fields=request_fields)
        except (Exception, asyncio.CancelledError) as exc:
            usage = getattr(exc,"usage",None)
            if usage is not None:
                row = usage.model_dump()
            elif isinstance(exc,DeepSeekConfigError):
                row = {"transport_outcome":"definitely_not_sent","unconfirmed_charge_possible":False}
            else:
                row = {"transport_outcome":"outcome_unknown","unconfirmed_charge_possible":True}
            _price_usage(row,camp=camp)
            budget.settle_request(camp, call_id, row)
            if direct_row is not None:
                _ledger(camp, {**direct_row,**row,"call_id":call_id})
            raise
        usage.call_id = call_id
        row = usage.model_dump()
        _price_usage(row,camp=camp)
        budget.settle_request(camp, call_id, row)
        if direct_row is not None:
            _ledger(camp, {**direct_row,**row,"call_id":call_id})
        return text,usage

    async def _complete_transport(self, *, messages: list[dict[str,str]], profile: str,
                        request_fields: dict | None = None) -> tuple[str, LMUsage]:
        spec = self._profile(profile)
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "stream": False,
            "timeout": spec.timeout_s,
            "max_tokens": spec.max_tokens,
        }
        if spec.wall_clock_deadline:
            # Use the SDK's HTTP module (some deployments vendor httpx2).
            from openai import _base_client
            http_module = getattr(_base_client, "httpx", None) or _base_client.httpx2
            kwargs["timeout"] = http_module.Timeout(connect=spec.connect_timeout_s,
                read=spec.read_timeout_s or spec.timeout_s, write=spec.write_timeout_s,
                pool=spec.pool_timeout_s)
        if spec.supports_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if spec.supports_temperature and spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        extra: dict[str, Any] = {}
        if spec.supports_thinking:
            # Some models enable thinking by default. Omitting the field does
            # not honor a fast profile that explicitly sets thinking=False.
            extra["thinking"] = {"type": "enabled" if spec.thinking else "disabled"}
            if spec.thinking and spec.reasoning_effort:
                kwargs["reasoning_effort"] = spec.reasoning_effort
        if extra:
            kwargs["extra_body"] = extra
        if request_fields is not None:
            for name in ("model", "max_tokens", "stream", "response_format", "temperature", "reasoning_effort", "extra_body"):
                if name in request_fields and kwargs.get(name) != request_fields[name]:
                    raise DeepSeekConfigError("prepared_transport_field_changed:" + name)
            for name in ("tools", "tool_choice"):
                if name in request_fields:
                    kwargs[name] = request_fields[name]
        from react_agent.eeg_research.agentic import owner_budget as budget
        camp=self._budget_campaign or budget.current_campaign()
        if camp is not None and budget.enabled(camp):
            permit=budget.current_transport(camp)
            if permit is None:
                raise DeepSeekConfigError("provider_transport_without_owner_claim")
            try:
                budget.mark_transport_dispatched(camp,permit[1])
            except budget.OwnerBudgetError as exc:
                raise DeepSeekConfigError(str(exc)) from exc
        started = time.perf_counter()
        try:
            request = self._client.chat.completions.create(**kwargs)
            response = await asyncio.wait_for(request, timeout=spec.timeout_s) if spec.wall_clock_deadline else await request
        except (Exception, asyncio.CancelledError) as exc:  # cancellation still has possible cost
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            # An HTTP error is a confirmed response; absent a response, delivery
            # outcome is unknown. Only known pre-send local errors are uncharged.
            from openai import _base_client
            http_module = getattr(_base_client, "httpx", None) or _base_client.httpx2
            cause = exc.__cause__ or exc
            not_sent = isinstance(cause, (http_module.ConnectError, http_module.ConnectTimeout, http_module.PoolTimeout))
            unconfirmed = status is None and not not_sent
            kind = "timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else "cancelled" if isinstance(exc, asyncio.CancelledError) else "authentication_error" if status in {401, 403} else "rate_limited" if status == 429 else "provider_error" if status and status >= 500 else "transport_error"
            usage = LMUsage(
                provider="deepseek",
                requested_model=spec.model,
                profile=profile,
                thinking=spec.thinking,
                reasoning_effort=spec.reasoning_effort if spec.thinking else None,
                http_status=status, failure_kind=kind,
                transport_outcome="definitely_not_sent" if not_sent else "outcome_unknown" if unconfirmed else "confirmed_response",
                success=False,
                error=f"{type(exc).__name__}",
                elapsed_seconds=time.perf_counter() - started,
                unconfirmed_charge_possible=unconfirmed,
            )
            raise DeepSeekCallError(usage) from exc

        choice = response.choices[0] if response.choices else None
        content = (message_text if spec.strict_final_output else _legacy_message_text)(choice.message) if choice else ""
        raw_usage = getattr(response, "usage", None)
        usage = LMUsage(
            provider="deepseek",
            requested_model=spec.model,
            response_model=getattr(response, "model", None),
            profile=profile,
            thinking=spec.thinking,
            reasoning_effort=spec.reasoning_effort if spec.thinking else None,
            input_tokens=_get(raw_usage, "prompt_tokens"),
            output_tokens=_get(raw_usage, "completion_tokens"),
            reasoning_tokens=_first_int(_nested(raw_usage, "completion_tokens_details", "reasoning_tokens"), _get(raw_usage, "reasoning_tokens")),
            cache_hit_tokens=_first_int(_get(raw_usage, "prompt_cache_hit_tokens"), _nested(raw_usage, "prompt_tokens_details", "cached_tokens")),
            cache_miss_tokens=_get(raw_usage, "prompt_cache_miss_tokens"),
            elapsed_seconds=time.perf_counter() - started,
            success=True,
            finish_reason=getattr(choice, "finish_reason", None) if choice else None,
            transport_outcome="confirmed_response", token_count_source="provider" if raw_usage else None,
        )
        if spec.strict_final_output:
            reason = usage.finish_reason
            failure = "output_truncated" if reason == "length" else "non_stop_finish_reason" if reason != "stop" else "unexpected_tool_calls" if getattr(choice.message, "tool_calls", None) else "missing_final_content" if not content.strip() else None
            if failure:
                usage.success, usage.status, usage.failure_kind, usage.error = False, "failed", failure, failure
                raise DeepSeekOutputError(raw_excerpt(content), usage)
        return content, usage

    def enable_context_protocol(self) -> None:
        """Require a frozen, measured request on EEG application calls."""
        self._requires_prepared_context = True

    async def complete_prepared_json(self, prepared, *, profile: str, role: str):
        """Send the exact messages measured by the EEG context compiler."""
        return await self.complete_json(system=prepared.system, user=prepared.user,
                                        profile=profile, role=role, prepared=prepared)

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        profile: str,
        role: str,
        prepared=None,
    ) -> tuple[dict[str, Any], LMUsage]:
        """Generic JSON object call used by planner/curator/probe."""
        from uuid import uuid4

        if self._requires_prepared_context and prepared is None:
            raise DeepSeekConfigError("eeg_prepared_request_required")
        if prepared is not None:
            if (not prepared.verify() or prepared.system != system or prepared.user != user
                    or not prepared.budget_report.get("ready_budget_ok")):
                raise DeepSeekConfigError("invalid_prepared_context")
            spec = self._profile(profile)
            fields = prepared.transport_fields
            if fields.get("model") != spec.model or fields.get("max_tokens") != spec.max_tokens:
                raise DeepSeekConfigError("prepared_transport_profile_changed")
            messages = list(prepared.messages)
            if messages != [{"role": "system", "content": system}, {"role": "user", "content": user}]:
                raise DeepSeekConfigError("prepared_messages_changed")
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if prepared is None:
            text, usage = await self._complete(messages=messages, profile=profile, role=role)
        else:
            text, usage = await self._complete(messages=messages, profile=profile,
                                                request_fields=prepared.transport_fields, role=role,
                                                budget_report=prepared.budget_report)
        usage.call_id = usage.call_id or uuid4().hex[:12]
        usage.role = role
        usage.status = "succeeded" if usage.success else "failed"
        try:
            payload = parse_json_object(text)
        except (json.JSONDecodeError, TypeError) as exc:
            usage.error = f"invalid_json:{type(exc).__name__}"
            usage.failure_kind = "format_error"
            usage.success = False
            usage.status = "failed"
            raise DeepSeekParseError(raw_excerpt(text), usage) from exc
        return payload, usage

    async def decide(
        self,
        observation: dict[str, Any],
        candidate_tools: list[str],
        profile: str,
    ) -> tuple[Decision, LMUsage]:
        """Ask for one JSON action."""
        user = (
            DECISION_SCHEMA_TEXT
            + "\nCandidates: "
            + json.dumps(candidate_tools)
            + "\nObservation:\n"
            + json.dumps(observation)
        )
        text, usage = await self._complete(
            messages=[
                {"role": "system", "content": DECISION_SYSTEM},
                {"role": "user", "content": user},
            ],
            profile=profile,
        )
        try:
            payload = parse_json_object(text)
            decision = Decision.model_validate(
                {**payload, "source": "hybrid", "profile": profile}
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            usage.error = f"invalid_json_decision:{type(exc).__name__}"
            usage.success = False
            raise DeepSeekParseError(raw_excerpt(text), usage) from exc
        return decision, usage

    async def summarize(
        self, report: dict[str, Any], profile: str
    ) -> tuple[dict[str, Any], LMUsage]:
        """Ask for explanatory JSON only."""
        user = json.dumps(
            {
                "verdict": report.get("verdict"),
                "stop_reason": report.get("stop_reason"),
                "findings": report.get("findings"),
                "executed_tools": report.get("executed_tools"),
                "limitations": report.get("limitations"),
            }
        )
        text, usage = await self._complete(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user},
            ],
            profile="fast",
        )
        try:
            payload = parse_json_object(text)
        except (json.JSONDecodeError, TypeError) as exc:
            usage.error = f"invalid_json_summary:{type(exc).__name__}"
            usage.success = False
            raise DeepSeekParseError(raw_excerpt(text), usage) from exc
        return payload, usage


class DeepSeekCallError(RuntimeError):
    """Transport/API failure that already has an LMUsage row."""

    def __init__(self, usage: LMUsage) -> None:
        super().__init__(usage.error or "deepseek_call_failed")
        self.usage = usage


class DeepSeekParseError(RuntimeError):
    """Model returned non-schema JSON."""

    def __init__(self, raw: str, usage: LMUsage) -> None:
        super().__init__("deepseek_parse_failed")
        self.raw = raw
        self.usage = usage


class DeepSeekOutputError(DeepSeekParseError):
    """A provider response exists but is not a complete final answer."""


def _get(obj: Any, name: str) -> int | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return int(value) if isinstance(value, int) else None


def _first_int(*values) -> int | None:
    return next((value for value in values if isinstance(value, int)), None)


def _nested(obj: Any, a: str, b: str) -> int | None:
    mid = getattr(obj, a, None) if obj is not None else None
    if mid is None and isinstance(obj, dict):
        mid = obj.get(a)
    return _get(mid, b)
