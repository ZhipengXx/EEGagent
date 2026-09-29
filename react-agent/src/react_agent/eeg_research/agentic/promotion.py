"""Promotion rules. A null threshold is not filled in by the model."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_research.agentic.memory import confirmation


def promotion_decision(
    *,
    comparison: dict[str, Any],
    goal: dict[str, Any],
    paired_deltas_pp: list[float] | None = None,
) -> dict[str, Any]:
    """Return a code-level promotion record. LLM must not invent a numeric bar."""
    bar = goal.get("min_practical_gain_pp")
    target_pairs = int(goal.get("confirmation_target_pairs") or 3)
    if not comparison.get("comparable"):
        return {
            "status": "not_compared",
            "reason": comparison.get("reason") or "unmatched_control",
            "min_practical_gain_pp": bar,
            "confirmation": None,
        }
    delta_pp = comparison.get("delta_pp")
    if bar is not None and delta_pp is not None and float(delta_pp) < float(bar):
        return {
            "status": "below_threshold",
            "reason": "min_practical_gain_not_met",
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": None,
        }
    if paired_deltas_pp:
        level = confirmation(paired_deltas_pp, target=target_pairs, margin_pp=float(bar or 0.0))
        return {
            "status": "provisional" if level.startswith("provisional") or level.startswith("no_") else "confirmed",
            "reason": level,
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": level,
            "paired_n": len(paired_deltas_pp),
        }
    return {
        "status": "observed",
        "reason": "single_seed",
        "min_practical_gain_pp": bar,
        "delta_pp": delta_pp,
        "confirmation": None,
    }
