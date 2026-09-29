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
    """No key, or the call failed. Callers must not substitute a scripted reply."""


def system_prompt(role: str) -> str:
    from react_agent.eeg_research.agentic.schemas import role_output_schema

    shared = (PROMPTS / "shared_contract.txt").read_text(encoding="utf-8")
    mission = (PROMPTS / f"{role}.txt").read_text(encoding="utf-8")
    return shared + "\n\n" + mission + "\n\nOUTPUT_SCHEMA\n" + role_output_schema(role) + "\nReturn one JSON object only.\n"


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
            }
        )
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
    )
    if os.environ.get("DEEPSEEK_FAST_MODEL"):
        config.fast.model = os.environ["DEEPSEEK_FAST_MODEL"]
    floor = CODER_MAX_TOKENS if role in LONG_JSON_ROLES else FAST_MAX_TOKENS
    config.fast.max_tokens = max(int(config.fast.max_tokens or 0), floor)
    system = system_prompt(role)
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]
    loop = asyncio.new_event_loop()
    client = DeepSeekBackend(config)
    bound: dict[str, Any] = {}

    def bind(**fields: Any) -> None:
        bound.clear()
        bound.update({key: value for key, value in fields.items() if value is not None})

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        last: DeepSeekParseError | None = None
        for _attempt in range(PARSE_ATTEMPTS):
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
            try:
                reply, usage = loop.run_until_complete(
                    client.complete_json(
                        system=system,
                        user=json.dumps(payload, ensure_ascii=False, default=str),
                        profile="fast",
                        role=role,
                    )
                )
            except DeepSeekParseError as exc:
                last = exc
                _ledger(camp, _fail_row(row, exc, started))
                continue
            except Exception as exc:  # noqa: BLE001
                _ledger(camp, _fail_row(row, exc, started))
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
            parsed = reply if isinstance(reply, dict) else {"_not_object": reply}
            if role in ENVELOPE_ROLES:
                from react_agent.eeg_research.agentic.roles import RoleResultError, bind_role_output, task_identity_from_payload

                task = task_identity_from_payload(payload, bound)
                if task is None:
                    raise LlmUnavailable("role_result_task_missing")
                try:
                    return bind_role_output(parsed, task=task, prompt_hash=prompt_hash)
                except RoleResultError as exc:
                    raise LlmUnavailable(str(exc)) from exc
            return parsed
        raise LlmUnavailable(type(last).__name__ if last is not None else "DeepSeekParseError") from last

    call.model = config.fast.model  # type: ignore[attr-defined]
    call.bind = bind  # type: ignore[attr-defined]
    return call
