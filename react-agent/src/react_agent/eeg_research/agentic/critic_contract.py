"""Compact Critic wire, frozen aliases and native-only result assembly.

The paid response stays unchanged. Every reference is interpreted against the
same pre-call context, including correction and recovery after interruption.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .schemas import CriticComparisonWire, CriticHistoryComparisonWire, ResearchPlanCritiqueV2

VERSION = "eeg_research.critic_contract.v2"
_CONTEXT_KEYS = {"version", "pool_digest", "alias_to_id", "delivered_observation_ids"}


def enabled(full_or_goal: dict | None) -> bool:
    full = full_or_goal or {}
    goal = full.get("goal") if isinstance(full.get("goal"), dict) else full
    execution = full.get("research_execution") or goal.get("research_execution") or {}
    return (execution.get("critic_contract_version") == "v2"
            and execution.get("planning_contract_version") == "v2"
            and execution.get("planning_mode", "propose_then_critic") == "propose_then_critic")


def _error(kind: str, loc, message: str) -> dict:
    return {"type": kind, "loc": list(loc), "msg": message}


def _pool_identity(pool: dict) -> tuple[dict[str, str], list[dict]]:
    from .plan_protocol import proposal_pool_digest

    errors = []
    rows = pool.get("proposals")
    ids = [row.get("proposal_id") if isinstance(row, dict) else None for row in rows] if isinstance(rows, list) else []
    if (not 2 <= len(ids) <= 4 or any(not isinstance(pid, str) or not pid.strip() for pid in ids)
            or len(set(pid for pid in ids if isinstance(pid, str))) != len(ids)
            or pool.get("pool_status") != "complete" or pool.get("valid_count") != len(ids)):
        errors.append(_error("critic_native_pool_invalid", ["proposals"],
                             "The Critic requires a complete frozen pool with unique native proposal IDs."))
    try:
        valid_digest = isinstance(pool.get("pool_digest"), str) and pool["pool_digest"] == proposal_pool_digest(pool)
    except (KeyError, TypeError, ValueError):
        valid_digest = False
    if not valid_digest:
        errors.append(_error("critic_native_pool_identity_invalid", ["pool_digest"],
                             "The frozen proposal pool identity is missing or changed."))
    aliases = {"p" + str(index): pid for index, pid in enumerate(ids, 1)} if not errors else {}
    return aliases, errors


def freeze_critic_context(pool: dict, delivered_view: dict) -> dict:
    """Freeze aliases from the final feasible pool and only actually sent history."""
    from .planning_context import delivered_observation_ids

    aliases, errors = _pool_identity(pool)
    if errors:
        raise ValueError(errors[0]["type"])
    return {"version": VERSION, "pool_digest": pool["pool_digest"], "alias_to_id": aliases,
            "delivered_observation_ids": sorted(delivered_observation_ids(delivered_view))}


def verify_critic_context(context: Any, pool: dict) -> list[dict]:
    """Local identity failures must never become requests for model correction."""
    aliases, errors = _pool_identity(pool)
    if not isinstance(context, dict) or set(context) != _CONTEXT_KEYS:
        return [*errors, _error("critic_native_context_invalid", ["critic_context"],
                               "The original frozen Critic context is required.")]
    if (context.get("version") != VERSION or context.get("pool_digest") != pool.get("pool_digest")
            or context.get("alias_to_id") != aliases):
        errors.append(_error("critic_native_context_identity_invalid", ["critic_context"],
                             "Critic aliases must match the original frozen proposal pool."))
    ids = context.get("delivered_observation_ids")
    if (not isinstance(ids, list) or any(not isinstance(item, str) or not item.strip() for item in ids)
            or len(set(item for item in ids if isinstance(item, str))) != len(ids)):
        errors.append(_error("critic_native_context_history_invalid", ["critic_context", "delivered_observation_ids"],
                             "The frozen delivered observation identities are invalid."))
    return errors


def critic_wire_schema(context: dict | None = None) -> dict:
    """Construction-time schema may be generic; actual calls supply frozen context."""
    history = bool(context and context.get("delivered_observation_ids"))
    schema = (CriticHistoryComparisonWire if history else CriticComparisonWire).model_json_schema()
    if context is None:
        return schema
    aliases = list(context["alias_to_id"])
    schema["properties"]["evaluations"].update(minItems=len(aliases), maxItems=len(aliases))
    schema["$defs"]["CriticAssessmentWire"]["properties"]["proposal_ref"]["enum"] = aliases
    if history:
        schema["properties"]["selected_basis_observation_ids"]["items"]["enum"] = list(context["delivered_observation_ids"])
    return schema


def materialize_critic_reply(raw: Any, pool: dict, full: dict | None = None, *,
                              critic_context: dict | None = None) -> dict:
    """Validate every API field, then attach deterministic native metadata."""
    result = {"reply": None, "errors": [], "native_errors": verify_critic_context(critic_context, pool)}
    if result["native_errors"]:
        return result
    history_ids = critic_context["delivered_observation_ids"]
    model = CriticHistoryComparisonWire if history_ids else CriticComparisonWire
    try:
        wire = model.model_validate(raw).model_dump()
    except ValidationError as exc:
        result["errors"] = [_error(row["type"], row["loc"],
            "missing " + str(row["loc"][-1]) if row["type"] == "missing" and row["loc"] else row["msg"])
            for row in exc.errors()]
        return result
    aliases = critic_context["alias_to_id"]
    refs = [row["proposal_ref"] for row in wire["evaluations"]]
    for index, ref in enumerate(refs):
        if ref not in aliases:
            result["errors"].append(_error("critic_proposal_ref_unknown", ["evaluations", index, "proposal_ref"],
                                          "Select one supplied proposal_ref."))
        elif ref in refs[:index]:
            result["errors"].append(_error("critic_proposal_ref_duplicate", ["evaluations", index, "proposal_ref"],
                                          "Evaluate each supplied proposal_ref exactly once."))
    if set(refs) != set(aliases) or len(refs) != len(aliases):
        result["errors"].append(_error("critic_evaluation_coverage_invalid", ["evaluations"],
                                      "Return every supplied proposal exactly once in preference order."))
    basis = wire.get("selected_basis_observation_ids", [])
    if len(set(basis)) != len(basis) or any(item not in history_ids for item in basis):
        result["errors"].append(_error("critic_basis_not_delivered", ["selected_basis_observation_ids"],
                                      "Select unique observation IDs from the delivered history, or return []."))
    if result["errors"]:
        return result
    native = {"protocol_version": "research_plan_critique.v2", "proposal_pool_digest": pool["pool_digest"],
        "evaluations": [{"proposal_id": aliases[row["proposal_ref"]], "rank": rank,
                         "assessment": row["assessment"]} for rank, row in enumerate(wire["evaluations"], 1)],
        "selected_proposal_id": aliases[refs[0]], "selection_rationale": wire["selection_rationale"],
        "selected_basis_observation_ids": basis}
    result["reply"] = ResearchPlanCritiqueV2.model_validate(native).model_dump()
    return result
