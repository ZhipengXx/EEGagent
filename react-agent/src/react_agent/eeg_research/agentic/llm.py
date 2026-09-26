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

PROMPTS = Path(__file__).resolve().parent / "prompts"


class LlmUnavailable(RuntimeError):
    """No key, or the call failed. Callers must not substitute a scripted reply."""


def system_prompt(role: str) -> str:
    shared = (PROMPTS / "shared_contract.txt").read_text(encoding="utf-8")
    return shared + "\n\n" + (PROMPTS / f"{role}.txt").read_text(encoding="utf-8")


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


def role_backend(camp: Path, role: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a callable that sends one structured input to DeepSeek for this role."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LlmUnavailable("DEEPSEEK_API_KEY missing")
    from react_agent.fmri.config import FmriCheckConfig
    from react_agent.fmri.llm.deepseek import DeepSeekBackend

    config = FmriCheckConfig(
        deepseek_api_key=key,
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
    if os.environ.get("DEEPSEEK_FAST_MODEL"):
        config.fast.model = os.environ["DEEPSEEK_FAST_MODEL"]
    config.fast.max_tokens = max(int(config.fast.max_tokens or 0), 4096)
    system = system_prompt(role)
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]
    loop = asyncio.new_event_loop()
    client = DeepSeekBackend(config)
    bound: dict[str, Any] = {}

    def bind(**fields: Any) -> None:
        bound.clear()
        bound.update({key: value for key, value in fields.items() if value is not None})

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        row = new_call_row(
            role=role,
            prompt_hash=prompt_hash,
            requested_model=config.fast.model,
            started_at=started,
            **bound,
        )
        try:
            reply, usage = loop.run_until_complete(
                client.complete_json(
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False, default=str),
                    profile="fast",
                    role=role,
                )
            )
        except Exception as exc:  # noqa: BLE001
            row.update({"success": False, "error": type(exc).__name__, "cost_status": "uncertain"})
            _ledger(camp, row)
            raise LlmUnavailable(type(exc).__name__) from exc
        row.update(
            {
                "success": True,
                "response_model": usage.response_model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        _ledger(camp, row)
        return reply if isinstance(reply, dict) else {"_not_object": reply}

    call.model = config.fast.model  # type: ignore[attr-defined]
    call.bind = bind  # type: ignore[attr-defined]
    return call
