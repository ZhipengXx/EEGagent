"""DeepSeek JSON-decision backend via the OpenAI-compatible SDK."""

from __future__ import annotations

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
    """Prefer content. Flash sometimes leaves the object in reasoning_content."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str) and content.strip():
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
        if spec.supports_thinking:
            # Some models enable thinking by default. Omitting the field does
            # not honor a fast profile that explicitly sets thinking=False.
            extra["thinking"] = {"type": "enabled" if spec.thinking else "disabled"}
            if spec.thinking and spec.reasoning_effort:
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

        choice = response.choices[0]
        content = message_text(choice.message)
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
            finish_reason=getattr(choice, "finish_reason", None),
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
            payload = parse_json_object(text)
        except (json.JSONDecodeError, TypeError) as exc:
            usage.error = f"invalid_json:{type(exc).__name__}"
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
