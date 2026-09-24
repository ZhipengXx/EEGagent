"""Compatibility filter and bounded ranking. Memory cannot change thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parent / "rules" / "procedural_v1.md"

COMPAT_FIELDS = (
    "generation_profile_id",
    "space_name",
    "normalization",
    "checkpoint_hash",
    "t_len",
)


def load_procedural_rules() -> str:
    return RULES_PATH.read_text(encoding="utf-8")


def retrieve_bundle(
    items: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    max_items: int = 6,
    exclude_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Filter then rank. Cross-profile numeric cases are excluded."""
    exclude = exclude_sample_ids or set()
    usable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    prior_runs: list[dict[str, Any]] = []
    for item in items:
        if item.get("sample_id") in exclude:
            prior_runs.append(item)
            excluded.append({**item, "exclude_reason": "same_sample_not_independent_reference"})
            continue
        if item.get("execution_domain") == "synthetic" and query.get("execution_domain") == "real":
            excluded.append({**item, "exclude_reason": "domain_mismatch"})
            continue
        missing = [f for f in COMPAT_FIELDS if query.get(f) and not item.get(f)]
        if missing:
            excluded.append({**item, "exclude_reason": f"missing_metadata:{missing[0]}"})
            continue
        mismatch = [
            f
            for f in COMPAT_FIELDS
            if query.get(f) is not None
            and item.get(f) is not None
            and str(query.get(f)) != str(item.get(f))
        ]
        if mismatch:
            excluded.append({**item, "exclude_reason": f"incompatible:{mismatch[0]}"})
            continue
        usable.append(item)

    def _rank(item: dict[str, Any]) -> tuple[int, str]:
        status = item.get("verification_status") or "unverified"
        priority = {
            "confirmed_issue": 0,
            "overturned": 1,
            "confirmed_clear": 2,
            "inconclusive": 3,
            "unverified": 4,
        }.get(status, 5)
        return (priority, str(item.get("created_at") or ""))

    usable.sort(key=_rank)
    selected = usable[:max_items]
    effects = []
    for item in selected:
        status = item.get("verification_status") or "unverified"
        effect = "reminder" if status == "unverified" else "ordering"
        effects.append(
            {
                "memory_id": item.get("episode_id"),
                "effect": effect,
                "verification_status": status,
            }
        )
    return {
        "rules": load_procedural_rules(),
        "items": _summaries(selected),
        "excluded": [
            {"episode_id": e.get("episode_id"), "reason": e.get("exclude_reason")}
            for e in excluded[:12]
        ],
        "effects": effects,
        "hit_count": len(selected),
        "prior_runs": _summaries(prior_runs[:2]),
        "path_changed": False,
        "path_effect": "被检索但未影响路径" if selected or prior_runs else "no_hits",
    }


def _summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        out.append(
            {
                "episode_id": item.get("episode_id"),
                "sample_id": item.get("sample_id"),
                "generation_profile_id": item.get("generation_profile_id"),
                "t_len": item.get("t_len"),
                "screening_decision": item.get("screening_decision"),
                "verification_status": item.get("verification_status"),
                "plan_sequence": item.get("plan_sequence"),
                "key_metrics": item.get("key_metrics"),
                "findings": item.get("findings"),
            }
        )
    return out
