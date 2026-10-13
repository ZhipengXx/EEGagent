"""Render once and account for application characters separately from tokens."""
from __future__ import annotations

import math
from typing import Any

from .types import BudgetConfig, CompileRequest, RENDER_VERSION, canonical

TOKENIZER_VERSION = "unicode_utf8_conservative_estimate.v1"


def token_count(text: str, tokenizer=None) -> tuple[int, str, str]:
    if tokenizer is not None:
        result = tokenizer(text)
        if isinstance(result, dict):
            return int(result["tokens"]), str(result.get("precision", "estimate")), str(result.get("version", "provided"))
        if isinstance(result, int):
            return result, "exact_tokenizer_not_billing", "provided"
        return len(result), "exact_tokenizer_not_billing", "provided"
    # Non-ASCII may require multiple model tokens. This is labelled an estimate,
    # including a separate safety multiplier; it is never a provider guarantee.
    ascii_chars = sum(ord(char) < 128 for char in text)
    estimate = math.ceil(ascii_chars / 3 + (len(text) - ascii_chars) * 1.5)
    return estimate, "estimate", TOKENIZER_VERSION


def provider_window(request: CompileRequest) -> tuple[int, int]:
    configured = request.budget.provider_context_tokens
    # Explicitly versioned conservative model policy, overridden for small models.
    defaults = {"deepseek": 128_000, "openai": 128_000, "anthropic": 200_000}
    window = configured or defaults.get(request.provider, 32_768)
    input_limit = request.budget.provider_input_tokens or window
    return window, input_limit


def measure(system: str, user: str, request: CompileRequest,
            transport_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = transport_fields or {}
    # Compatibility definition: actual system + actual user Unicode characters.
    effective = system + user
    envelope = canonical({"messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}], **fields})
    tokens, precision, version = token_count(effective, request.tokenizer)
    structured = {key: fields[key] for key in ("tools", "response_format") if key in fields}
    structured_tokens, _, _ = token_count(canonical(structured), request.tokenizer) if structured else (0, precision, version)
    # Message protocol overhead is model policy overhead, not HTTP tokenization.
    overhead = 16
    safety = math.ceil((tokens + structured_tokens + overhead) * request.budget.token_safety_factor)
    window, input_limit = provider_window(request)
    reserve = request.max_output_tokens + request.budget.reasoning_reserve_tokens
    ceiling = request.budget.hard_chars if request.mode == "repair" else request.budget.fallback_chars
    application_ok = len(effective) < ceiling if request.mode == "repair" else len(effective) <= ceiling
    return {
        "version": "eeg.context.budget.v1", "render_version": RENDER_VERSION,
        "application_count_definition": "len(actual_system)+len(actual_user)",
        "chars": len(effective), "system_chars": len(system), "user_chars": len(user),
        "utf8_bytes": len(effective.encode("utf-8")),
        "transport_envelope_chars": len(envelope), "transport_envelope_utf8_bytes": len(envelope.encode("utf-8")),
        "model_tokens": tokens + structured_tokens + overhead,
        "token_precision": precision, "tokenizer_version": version,
        "usage_calibration": getattr(request.tokenizer, "calibration_metadata", None),
        "token_safety_factor": request.budget.token_safety_factor,
        "safe_input_tokens": safety, "max_output_tokens": request.max_output_tokens,
        "reasoning_reserve_tokens": request.budget.reasoning_reserve_tokens,
        "provider_context_tokens": window, "provider_input_tokens": input_limit,
        "provider_budget_ok": safety <= input_limit and safety + reserve <= window,
        "target_chars": request.budget.target_chars, "high_chars": request.budget.high_chars,
        "fallback_chars": request.budget.fallback_chars, "hard_chars": request.budget.hard_chars,
        "active_ceiling_chars": ceiling, "application_budget_ok": application_ok,
        "repair_hard_boundary": "strict_lt" if request.mode == "repair" else None,
        "ready_budget_ok": application_ok and safety <= input_limit and safety + reserve <= window,
    }


def compact_schema(schema: Any) -> Any:
    """Drop schema annotations, never recurse into business defaults/constants.

    A property actually named title/description remains a property. Only keys at
    JSON-schema nodes are annotations. Values under enum/default/const are opaque.
    """
    if not isinstance(schema, dict):
        return schema
    result = {}
    child_schema_maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
    child_schema_values = {"items", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames"}
    child_schema_lists = {"allOf", "anyOf", "oneOf", "prefixItems"}
    for key, value in schema.items():
        if key in {"title", "description", "$comment", "examples"}:
            continue
        if key in child_schema_maps and isinstance(value, dict):
            result[key] = {name: compact_schema(child) for name, child in value.items()}
        elif key in child_schema_values and isinstance(value, dict):
            result[key] = compact_schema(value)
        elif key in child_schema_lists and isinstance(value, list):
            result[key] = [compact_schema(child) for child in value]
        else:
            result[key] = value
    return result
