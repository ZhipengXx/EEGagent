"""DeepSeek JSON-decision backend via the OpenAI-compatible SDK."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from react_agent.fmri.config import FmriCheckConfig, LmProfile
from react_agent.fmri.prompts import DECISION_SCHEMA_TEXT, DECISION_SYSTEM, SUMMARY_SYSTEM
from react_agent.fmri.schemas import Decision, LMUsage


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
        from openai import AsyncOpenAI

        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            max_retries=0,
        )

    def _profile(self, name: str) -> LmProfile:
        return self.config.reasoning if name == "reasoning" else self.config.fast

    async def _complete(self, *, messages: list[dict[str, str]], profile: str) -> tuple[str, LMUsage]:
        spec = self._profile(profile)
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "stream": False,
            "timeout": spec.timeout_s,
            "max_tokens": spec.max_tokens,
        }
        if spec.supports_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if spec.supports_temperature and spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        extra: dict[str, Any] = {}
        if spec.supports_thinking and spec.thinking:
            extra["thinking"] = {"type": "enabled"}
            if spec.reasoning_effort:
                extra["reasoning_effort"] = spec.reasoning_effort
        if extra:
            kwargs["extra_body"] = extra
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            unconfirmed = status not in {401, 403}
            usage = LMUsage(
                provider="deepseek",
                requested_model=spec.model,
                profile=profile,
                thinking=spec.thinking,
                success=False,
                error=f"{type(exc).__name__}",
                elapsed_seconds=time.perf_counter() - started,
                unconfirmed_charge_possible=unconfirmed,
            )
            raise DeepSeekCallError(usage) from exc

        content = response.choices[0].message.content or ""
        raw_usage = getattr(response, "usage", None)
        usage = LMUsage(
            provider="deepseek",
            requested_model=spec.model,
            response_model=getattr(response, "model", None),
            profile=profile,
            thinking=spec.thinking,
            input_tokens=_get(raw_usage, "prompt_tokens"),
            output_tokens=_get(raw_usage, "completion_tokens"),
            reasoning_tokens=_nested(raw_usage, "completion_tokens_details", "reasoning_tokens"),
            cache_hit_tokens=_nested(raw_usage, "prompt_tokens_details", "cached_tokens"),
            elapsed_seconds=time.perf_counter() - started,
            success=True,
        )
        return content, usage

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        profile: str,
        role: str,
    ) -> tuple[dict[str, Any], LMUsage]:
        """Generic JSON object call used by planner/curator/probe."""
        from uuid import uuid4

        text, usage = await self._complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            profile=profile,
        )
        usage.call_id = uuid4().hex[:12]
        usage.role = role
        usage.status = "succeeded" if usage.success else "failed"
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError("json is not an object")
        except (json.JSONDecodeError, TypeError) as exc:
            usage.error = f"invalid_json:{type(exc).__name__}"
            usage.success = False
            usage.status = "failed"
            raise DeepSeekParseError(text, usage) from exc
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
            payload = json.loads(text)
            decision = Decision.model_validate(
                {**payload, "source": "hybrid", "profile": profile}
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            usage.error = f"invalid_json_decision:{type(exc).__name__}"
            usage.success = False
            raise DeepSeekParseError(text, usage) from exc
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
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError("summary is not an object")
        except (json.JSONDecodeError, TypeError) as exc:
            usage.error = f"invalid_json_summary:{type(exc).__name__}"
            usage.success = False
            raise DeepSeekParseError(text, usage) from exc
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


def _get(obj: Any, name: str) -> int | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return int(value) if isinstance(value, int) else None


def _nested(obj: Any, a: str, b: str) -> int | None:
    mid = getattr(obj, a, None) if obj is not None else None
    if mid is None and isinstance(obj, dict):
        mid = obj.get(a)
    return _get(mid, b)
