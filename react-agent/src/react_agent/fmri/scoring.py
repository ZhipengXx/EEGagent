"""Heuristic need_score. Not a risk probability or medical judgement."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.schemas import ToolResult


def compute_need_score(
    results: list[ToolResult],
    config: FmriCheckConfig,
    *,
    completed: set[str],
) -> dict[str, Any]:
    """Compute A/M/D heuristic priority components.

    Unobserved A/D are null, not 0. V1 leaves D disabled (w_d=0).
    """
    required = list(config.required_checks)
    missing_required = [name for name in required if name not in completed]
    m_val = (len(missing_required) / len(required)) if required else 0.0

    flag_codes = []
    error_codes = []
    for result in results:
        if result.execution_status != "success":
            continue
        for finding in result.findings:
            if finding.severity == "error":
                error_codes.append(finding.code)
            if finding.severity == "flag":
                flag_codes.append(finding.code)

    a_observed = any(
        r.tool_name == "basic_statistics" and r.execution_status == "success"
        for r in results
    )
    a_val: float | None
    if not a_observed:
        a_val = None
    elif error_codes:
        a_val = 1.0
    elif flag_codes:
        a_val = min(1.0, 0.35 * len(set(flag_codes)))
    else:
        a_val = 0.0

    d_val = None  # V1: no reliable conflict definition
    weights = {"A": config.score.w_a, "M": config.score.w_m, "D": config.score.w_d}
    observed = {"A": a_val is not None, "M": True, "D": False}
    usable = {
        key: weights[key]
        for key, seen in observed.items()
        if seen and weights[key] > 0
    }
    total_w = sum(usable.values()) or 1.0
    components = {"A": a_val, "M": m_val, "D": d_val}
    need = 0.0
    for key, weight in usable.items():
        value = components[key]
        if value is None:
            continue
        need += (weight / total_w) * float(value)

    return {
        "kind": "heuristic_priority",
        "need_score": need,
        "components": components,
        "observed_mask": observed,
        "weights": weights,
        "renormalized_weights": {k: w / total_w for k, w in usable.items()},
        "missing_required": missing_required,
        "flag_codes": sorted(set(flag_codes)),
        "error_codes": sorted(set(error_codes)),
        "note": "Engineering heuristic; not calibrated on real brain data.",
    }
