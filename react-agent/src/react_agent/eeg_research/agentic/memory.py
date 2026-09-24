"""Research memory. Episodes are written by the runtime. The curator cannot promote them."""

from __future__ import annotations

from typing import Any


def evidence_level(kind: str) -> str:
    if kind == "implementation_failure":
        return "implementation_failure"
    if kind == "paired_seed_result":
        return "paired_seed_result"
    if kind == "exploratory_result":
        return "exploratory_result"
    return "exploratory_result"


def episode(
    *,
    task_hash: str,
    candidate_id: str,
    kind: str,
    fidelity: str,
    seed: int,
    metric: float | None,
    contract_fingerprint: str,
    artifact: str,
) -> dict[str, Any]:
    """One deterministic episode. Test metrics are not accepted."""
    return {
        "task_hash": task_hash,
        "candidate_id": candidate_id,
        "kind": kind,
        "evidence_level": evidence_level(kind),
        "fidelity": fidelity,
        "seed": seed,
        "primary_metric": metric,
        "contract_fingerprint": contract_fingerprint,
        "artifact": artifact,
        "scope": "development",
    }


def retrieve(entries: list[dict[str, Any]], *, task_hash: str, fingerprint: str) -> list[dict[str, Any]]:
    """Compatible rows stay comparable. Other protocols are analogy only."""
    rows = []
    for entry in entries:
        if "test" in json_keys(entry):
            continue
        copied = dict(entry)
        if entry.get("task_hash") != task_hash or entry.get("contract_fingerprint") != fingerprint:
            copied["retrieval"] = "analogy_only"
        else:
            copied["retrieval"] = "comparable"
        rows.append(copied)
    return rows


def json_keys(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(key) + " " + json_keys(item) for key, item in value.items())
    if isinstance(value, list):
        return " ".join(json_keys(item) for item in value)
    return ""


def confirmation(pairs: list[float], *, target: int = 3, margin_pp: float = 0.0) -> str:
    """Paired seed rule. Fewer than the target stays provisional."""
    if len(pairs) < target:
        return "provisional_improvement" if pairs and sum(pairs) / len(pairs) > margin_pp / 100 else "no_confirmed_improvement"
    positive = sum(1 for value in pairs if value > 0)
    mean = sum(pairs) / len(pairs)
    if mean > margin_pp / 100 and 3 * positive >= 2 * len(pairs):
        return "replicated_improvement"
    return "no_confirmed_improvement"
