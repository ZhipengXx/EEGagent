"""Promotion rules. A null threshold is not filled in by the model."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_research.agentic.memory import confirmation


def _tier(fidelity: str | None, confirmation_level: str | None) -> str:
    if confirmation_level == "replicated_improvement":
        return "confirmation"
    if fidelity == "smoke" or fidelity == "check":
        return "smoke"
    if fidelity == "pilot":
        return "pilot"
    if fidelity == "full":
        return "full"
    return fidelity or "observed"


def promotion_decision(
    *,
    comparison: dict[str, Any],
    goal: dict[str, Any],
    paired_deltas_pp: list[float] | None = None,
    fidelity: str | None = None,
) -> dict[str, Any]:
    """Return a code-level promotion record. LLM must not invent a numeric bar.

    A green replicate is not a green confirmation. confirmation_target_pairs is the
    pre-declared paired-seed policy; fewer pairs stay provisional.
    """
    bar = goal.get("min_practical_gain_pp")
    target_pairs = int(goal.get("confirmation_target_pairs") or 3)
    paired = list(paired_deltas_pp or [])
    replicate_n = len(paired)
    if not comparison.get("comparable"):
        return {
            "status": "not_compared",
            "reason": comparison.get("reason") or "unmatched_control",
            "min_practical_gain_pp": bar,
            "confirmation": None,
            "tier": _tier(fidelity, None),
            "replicate_status": "recorded" if replicate_n > 1 else "single_or_none",
            "confirmation_target_pairs": target_pairs,
            "paired_n": replicate_n,
        }
    delta_pp = comparison.get("delta_pp")
    if bar is not None and delta_pp is not None and float(delta_pp) < float(bar):
        return {
            "status": "below_threshold",
            "reason": "min_practical_gain_not_met",
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": None,
            "tier": _tier(fidelity, None),
            "replicate_status": "recorded" if replicate_n > 1 else "single_or_none",
            "confirmation_target_pairs": target_pairs,
            "paired_n": replicate_n,
        }
    if paired:
        level = confirmation(paired, target=target_pairs, margin_pp=float(bar or 0.0))
        confirmed = level == "replicated_improvement"
        return {
            "status": "confirmed" if confirmed else "provisional",
            "reason": level,
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": level,
            "tier": _tier(fidelity, level),
            "replicate_status": "replicated_seed" if replicate_n > 1 else "single_seed",
            "confirmation_target_pairs": target_pairs,
            "paired_n": replicate_n,
        }
    return {
        "status": "observed",
        "reason": "single_seed",
        "min_practical_gain_pp": bar,
        "delta_pp": delta_pp,
        "confirmation": None,
        "tier": _tier(fidelity, None),
        "replicate_status": "single_seed",
        "confirmation_target_pairs": target_pairs,
        "paired_n": 0,
    }
