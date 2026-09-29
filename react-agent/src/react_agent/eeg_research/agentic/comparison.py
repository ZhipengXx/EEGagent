"""Deterministic comparison. A missing matched control is not a completed comparison."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_research.agentic.schemas import Comparison, comparison_key, protocol_is_comparable


def matched_control(
    evidence: list[dict[str, Any]],
    result: dict[str, Any],
    protocol: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the same-protocol, same-fidelity, same-training-seed baseline row."""
    ok, _reason = protocol_is_comparable(protocol)
    if not ok:
        return None
    seed = result.get("training_seed", result.get("seed"))
    fidelity = result.get("fidelity")
    fingerprint = result.get("execution_fingerprint") or result.get("contract_fingerprint")
    for row in reversed(evidence):
        if row.get("candidate_id") != "baseline":
            continue
        if not row.get("evaluation_valid"):
            continue
        if row.get("fidelity") != fidelity:
            continue
        if int(row.get("seed") or 0) != int(seed or 0):
            continue
        row_fp = row.get("execution_fingerprint") or row.get("contract_fingerprint")
        if fingerprint and row_fp and row_fp != fingerprint:
            continue
        return row
    return None


def compare_runs(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any] | None,
    protocol: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a Comparison dict. Feasibility-only jobs are not marked compared."""
    comparable_protocol, reason = protocol_is_comparable(protocol)
    if candidate.get("candidate_id") == "baseline":
        payload = Comparison(
            control_run_id=None,
            candidate_run_id=str(candidate.get("evidence_id") or ""),
            comparable=False,
            reason="baseline_not_compared",
            fidelity=str(candidate.get("fidelity") or ""),
            training_seed=None if candidate.get("seed") is None else int(candidate.get("seed") or 0),
        )
        return payload.model_dump()
    if not candidate.get("evaluation_valid"):
        payload = Comparison(
            control_run_id=None,
            candidate_run_id=str(candidate.get("evidence_id") or ""),
            comparable=False,
            reason="candidate_invalid",
            fidelity=str(candidate.get("fidelity") or ""),
        )
        return payload.model_dump()
    if not comparable_protocol:
        payload = Comparison(
            control_run_id=None,
            candidate_run_id=str(candidate.get("evidence_id") or ""),
            comparable=False,
            reason=reason or "historical_incomparable",
            fidelity=str(candidate.get("fidelity") or ""),
        )
        return payload.model_dump()
    if control is None:
        payload = Comparison(
            control_run_id=None,
            candidate_run_id=str(candidate.get("evidence_id") or ""),
            comparable=False,
            reason="unmatched_control",
            fidelity=str(candidate.get("fidelity") or ""),
            training_seed=None if candidate.get("seed") is None else int(candidate.get("seed") or 0),
        )
        return payload.model_dump()
    delta = None
    if candidate.get("fixed_bank_top1") is not None and control.get("fixed_bank_top1") is not None:
        delta = float(candidate["fixed_bank_top1"]) - float(control["fixed_bank_top1"])
    payload = Comparison(
        control_run_id=str(control.get("evidence_id") or ""),
        candidate_run_id=str(candidate.get("evidence_id") or ""),
        comparable=True,
        same_seed_delta=delta,
        fidelity=str(candidate.get("fidelity") or ""),
        training_seed=int(candidate.get("seed") or 0),
        evidence_level="observed",
        difference_sources=["source_hash"] if candidate.get("candidate_id") != "baseline" else [],
        reason="matched_control",
    )
    data = payload.model_dump()
    data["comparison_key"] = comparison_key(protocol or {}, str(candidate.get("fidelity") or ""))
    data["delta_pp"] = None if delta is None else round(100 * delta, 4)
    return data
