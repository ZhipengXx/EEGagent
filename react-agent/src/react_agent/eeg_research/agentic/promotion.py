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


def _paired_rows(
    paired_deltas_pp: list[float] | None,
    paired_records: list[dict[str, Any]] | None,
    fidelity: str | None,
) -> list[dict[str, Any]]:
    if paired_records:
        return [dict(row) for row in paired_records if isinstance(row, dict)]
    return [{"delta_pp": float(value), "fidelity": fidelity, "legacy_float": True} for value in (paired_deltas_pp or [])]


def _confirmation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full runs only. One predeclared training seed is one independent pair."""
    kept: list[dict[str, Any]] = []
    seen_tokens: set[tuple[Any, ...]] = set()
    seen_seeds: set[Any] = set()
    seen_checkpoints: set[Any] = set()
    for row in rows:
        fidelity = str(row.get("fidelity") or "")
        if fidelity in {"pilot", "check", "smoke"}:
            continue
        if fidelity and fidelity != "full":
            continue
        if row.get("evaluation_valid") is False:
            continue
        seed = row.get("seed")
        if seed is not None and seed in seen_seeds:
            continue
        checkpoint = row.get("checkpoint_id")
        if checkpoint not in (None, "") and checkpoint in seen_checkpoints:
            continue
        token = (row.get("run_id") or row.get("job_id"), seed, checkpoint)
        if any(token) and token in seen_tokens:
            continue
        if any(token):
            seen_tokens.add(token)
        if seed is not None:
            seen_seeds.add(seed)
        if checkpoint not in (None, ""):
            seen_checkpoints.add(checkpoint)
        kept.append(row)
    return kept


def promotion_decision(
    *,
    comparison: dict[str, Any],
    goal: dict[str, Any],
    paired_deltas_pp: list[float] | None = None,
    fidelity: str | None = None,
    paired_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a code-level promotion record. LLM must not invent a numeric bar.

    Pilot runs and bare float lists never become confirmation. The paired set, not the
    last delta, decides the aggregate. Percentage points are compared as percentage points.
    """
    bar = goal.get("min_practical_gain_pp")
    target_pairs = int(goal.get("confirmation_target_pairs") or 0)
    has_rule = goal.get("confirmation_target_pairs") is not None
    rows = _confirmation_rows(_paired_rows(paired_deltas_pp, paired_records, fidelity))
    deltas = [float(row["delta_pp"]) for row in rows if row.get("delta_pp") is not None]
    replicate_n = len(deltas)
    identity_ready = bool(rows) and all(
        not row.get("legacy_float") and row.get("run_id") and row.get("control_run_id") and row.get("seed") is not None
        for row in rows
    )
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
    if deltas and has_rule and target_pairs > 0:
        margin = 0.0 if bar is None else float(bar)
        level = confirmation(deltas, target=target_pairs, margin_pp=margin)
        confirmed = (
            level == "replicated_improvement"
            and fidelity == "full"
            and identity_ready
            and bar is not None
        )
        status = "confirmed" if confirmed else "provisional"
        if bar is not None and level == "no_confirmed_improvement":
            status = "below_threshold"
        return {
            "status": status,
            "reason": level,
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": level,
            "tier": "confirmation" if confirmed else _tier(fidelity, None),
            "replicate_status": "replicated_seed" if replicate_n > 1 else "single_seed",
            "confirmation_target_pairs": target_pairs,
            "paired_n": replicate_n,
        }
    if not deltas and bar is not None and delta_pp is not None and float(delta_pp) < float(bar):
        return {
            "status": "below_threshold",
            "reason": "min_practical_gain_not_met",
            "min_practical_gain_pp": bar,
            "delta_pp": delta_pp,
            "confirmation": None,
            "tier": _tier(fidelity, None),
            "replicate_status": "single_or_none",
            "confirmation_target_pairs": target_pairs,
            "paired_n": 0,
        }
    return {
        "status": "observed",
        "reason": "single_seed" if not deltas else "confirmation_rule_missing",
        "min_practical_gain_pp": bar,
        "delta_pp": delta_pp,
        "confirmation": None,
        "tier": _tier(fidelity, None),
        "replicate_status": "replicated_seed" if replicate_n > 1 else "single_seed",
        "confirmation_target_pairs": target_pairs,
        "paired_n": replicate_n,
    }
