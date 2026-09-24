"""Call counting, reservations, and simple USD estimates."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import Budgets
from react_agent.fmri.schemas import LMUsage


def empty_ledger() -> dict[str, Any]:
    """Create a fresh per-sample ledger."""
    return {
        "tool_calls": 0,
        "lm_calls": 0,
        "lm_attempts": 0,
        "lm_successes": 0,
        "lm_retries": 0,
        "lm_repairs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": None,
        "api_usd": None,
        "reservations": 0,
        "unconfirmed_charges": 0,
        "generation_requests": 0,
        "new_tribe_predictions": 0,
        "planner_calls_succeeded": 0,
        "accepted_plan_count": 0,
        "create_plan_attempts": 0,
        "create_plan_accepted": 0,
        "repair_attempts": 0,
        "repaired_plan_accepted": 0,
        "revise_plan_attempts": 0,
        "revise_plan_accepted": 0,
        "provider_calls_succeeded": 0,
        "curator_calls": 0,
        "llm_calls_attempted": 0,
        "llm_calls_succeeded": 0,
    }


class BudgetError(RuntimeError):
    """Raised when a preflight check fails."""


class BudgetLedger:
    """In-memory budget tracker for one sample, plus optional batch caps."""

    def __init__(self, budgets: Budgets, batch: dict[str, int] | None = None) -> None:
        self.budgets = budgets
        self.data = empty_ledger()
        self.batch = batch if batch is not None else {"lm_calls": 0}
        self.calls: list[dict[str, Any]] = []

    def preflight_tool(self) -> None:
        """Refuse another tool call when the cap is reached."""
        if self.data["tool_calls"] >= self.budgets.max_tool_calls:
            raise BudgetError("max_tool_calls exhausted")

    def record_tool(self) -> None:
        """Count a real tool execution (cache hits do not call this)."""
        self.data["tool_calls"] += 1

    def preflight_lm(self) -> None:
        """Refuse another LM call including repairs, retries, and summaries."""
        if self.data["lm_calls"] >= self.budgets.max_lm_calls:
            raise BudgetError("max_lm_calls exhausted")
        if (
            self.budgets.max_batch_lm_calls is not None
            and self.batch["lm_calls"] >= self.budgets.max_batch_lm_calls
        ):
            raise BudgetError("batch max_lm_calls exhausted")
        if self.budgets.max_batch_usd is not None and self.budgets.usd_input_per_million is None:
            raise BudgetError("USD cap configured without prices")
        self.data["reservations"] += 1

    def record_lm(self, usage: LMUsage, *, retry: bool = False, repair: bool = False) -> None:
        """Settle one LM attempt."""
        self.calls.append(
            {
                "role": usage.role or "unknown",
                "call_id": usage.call_id,
                "provider": usage.provider,
                "requested_model": usage.requested_model,
                "response_model": usage.response_model,
                "success": usage.success,
                "error": usage.error,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "elapsed_seconds": usage.elapsed_seconds,
                "retry": retry,
                "repair": repair,
            }
        )
        self.data["lm_attempts"] += 1
        self.data["lm_calls"] += 1
        self.data["llm_calls_attempted"] += 1
        self.batch["lm_calls"] += 1
        if retry:
            self.data["lm_retries"] += 1
        if repair:
            self.data["lm_repairs"] += 1
        if usage.success:
            self.data["lm_successes"] += 1
            self.data["llm_calls_succeeded"] += 1
            self.data["provider_calls_succeeded"] += 1
            if usage.role in {"planner", "replanner"}:
                self.data["planner_calls_succeeded"] += 1
            if usage.role == "planner":
                self.data["create_plan_attempts"] += 1
            if usage.role == "replanner":
                self.data["revise_plan_attempts"] += 1
        if repair:
            self.data["repair_attempts"] += 1
        if usage.role == "curator":
            self.data["curator_calls"] += 1
        if usage.unconfirmed_charge_possible or usage.input_tokens is None:
            self.data["unconfirmed_charges"] += 1
        if usage.input_tokens is not None:
            self.data["input_tokens"] += usage.input_tokens
        if usage.output_tokens is not None:
            self.data["output_tokens"] += usage.output_tokens
        usd = estimate_usd(usage, self.budgets)
        usage.api_usd = usd
        if usd is not None:
            current = self.data["api_usd"]
            self.data["api_usd"] = (0.0 if current is None else current) + usd

    def preflight_generation(self, new_predictions: int) -> None:
        """Refuse a generation request that would exceed prediction caps."""
        if (
            self.budgets.max_generation_requests is not None
            and self.data["generation_requests"] >= self.budgets.max_generation_requests
        ):
            raise BudgetError("max_generation_requests exhausted")
        if self.budgets.max_new_tribe_predictions is not None:
            used = int(self.data["new_tribe_predictions"])
            if used + int(new_predictions) > self.budgets.max_new_tribe_predictions:
                raise BudgetError("max_new_tribe_predictions exhausted")

    def record_generation(self, new_predictions: int) -> None:
        """Count one generation request and any new TRIBE forwards."""
        self.data["generation_requests"] += 1
        self.data["new_tribe_predictions"] += int(new_predictions)

    def record_accepted_plan(self, *, phase: str = "create", repaired: bool = False) -> None:
        """Count a validator-accepted plan (create or revise)."""
        self.data["accepted_plan_count"] += 1
        if phase == "revise":
            self.data["revise_plan_accepted"] += 1
        else:
            self.data["create_plan_accepted"] += 1
        if repaired:
            self.data["repaired_plan_accepted"] += 1

    def snapshot(self) -> dict[str, Any]:
        """Copy ledger numbers."""
        return dict(self.data)


def estimate_usd(usage: LMUsage, budgets: Budgets) -> float | None:
    """Compute USD if prices and token counts are both present.

    Reasoning tokens are not added again when already inside output_tokens.
    """
    if budgets.usd_input_per_million is None or budgets.usd_output_per_million is None:
        return None
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    inp = usage.input_tokens / 1_000_000 * budgets.usd_input_per_million
    out = usage.output_tokens / 1_000_000 * budgets.usd_output_per_million
    return inp + out
