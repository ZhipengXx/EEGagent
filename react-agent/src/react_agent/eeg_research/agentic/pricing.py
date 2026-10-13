"""One deployment-supplied price identity for reservation and settlement."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


class PricingConflict(ValueError):
    pass


def _normalize(value, *, label):
    if not value:
        return None
    if not value.get("source") or not value.get("date") or any(value.get(key) is None for key in ("input_per_million", "output_per_million")):
        raise PricingConflict("incomplete_pricing_configuration:" + label)
    result = {key: str(value[key]) for key in ("source", "date")}
    result.update(provider=str(value.get("provider") or "deepseek"), model=value.get("model"))
    for key in ("input_per_million", "output_per_million", "cache_hit_per_million"):
        if value.get(key) is None:
            continue
        number = float(value[key])
        if not math.isfinite(number) or number < 0:
            raise PricingConflict("invalid_pricing_configuration:" + label)
        result[key] = number
    return result


def resolve_pricing(goal_or_camp, *, provider="deepseek", model=None, environ=None):
    env = os.environ if environ is None else environ
    goal = goal_or_camp if isinstance(goal_or_camp, dict) else json.loads((Path(goal_or_camp) / "goal.json").read_text())
    configured = ((goal.get("research_execution") or {}).get("budget") or {}).get("pricing") or {}
    raw_env = {key: env[name] for key, name in {
        "source": "DEEPSEEK_PRICE_SOURCE", "date": "DEEPSEEK_PRICE_DATE",
        "input_per_million": "DEEPSEEK_USD_INPUT_PER_MILLION", "output_per_million": "DEEPSEEK_USD_OUTPUT_PER_MILLION",
        "cache_hit_per_million": "DEEPSEEK_USD_CACHE_HIT_PER_MILLION", "model": "DEEPSEEK_PRICE_MODEL",
    }.items() if env.get(name) is not None}
    goal_price = _normalize(configured, label="goal")
    env_price = _normalize(raw_env, label="environment")
    if goal_price is not None and env_price is not None and goal_price != env_price:
        raise PricingConflict("pricing_configuration_conflict")
    price = goal_price or env_price
    if price is not None:
        if price["provider"] != provider or (model is not None and price.get("model") not in {None, "*", model}):
            raise PricingConflict("pricing_provider_or_model_mismatch")
        price = {**price, "request_provider": provider, "request_model": model}
    identity = {"pricing": price, "provider": provider, "model": model}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"pricing": price, "fingerprint": fingerprint, "provider": provider, "model": model}


def estimate_usd(frozen, *, input_tokens, output_tokens, cache_hit_tokens=0, upper_bound=False):
    price = frozen.get("pricing")
    if price is None or type(input_tokens) is not int or type(output_tokens) is not int:
        return None
    input_rate = price["input_per_million"]
    cache_rate = price.get("cache_hit_per_million", input_rate)
    if upper_bound:
        input_rate = max(input_rate, cache_rate)
        cache_hit_tokens = 0
    hit = cache_hit_tokens if type(cache_hit_tokens) is int and 0 <= cache_hit_tokens <= input_tokens else 0
    return ((input_tokens - hit) * input_rate + hit * cache_rate + output_tokens * price["output_per_million"]) / 1_000_000
