"""One frozen ConfirmationPolicy per campaign. Planner, executor and promotion share it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.schemas import SCHEMA_VERSION

POLICY_SCHEMA = "eeg_research.confirmation_policy.v1"
POLICY_NAME = "confirmation_policy.json"


class ConfirmationPolicyError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def declared_seeds(goal: dict[str, Any] | None, protocol: dict[str, Any] | None = None) -> list[int] | None:
    """None means undeclared. An explicit empty list stays empty."""
    goal = goal if isinstance(goal, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    if "training_seeds" in goal and goal.get("training_seeds") is not None:
        seeds = goal.get("training_seeds")
        if seeds == []:
            return []
        return [int(item) for item in seeds]
    if goal.get("declared_training_seeds") is not None:
        return [int(item) for item in goal["declared_training_seeds"]]
    proto_seeds = protocol.get("training_seeds")
    if proto_seeds:
        return [int(item) for item in proto_seeds]
    return None


def freeze_confirmation_policy(
    goal: dict[str, Any] | None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one policy after the goal is parsed. Unsatisfiable pair counts fail at create."""
    goal = goal if isinstance(goal, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    seeds = declared_seeds(goal, protocol)
    target = goal.get("confirmation_target_pairs")
    if target is not None and seeds is not None and int(target) > len(seeds):
        raise ConfirmationPolicyError("unsatisfiable_target_pairs")
    body = {
        "schema_version": POLICY_SCHEMA,
        "training_seeds": list(seeds or []),
        "seeds_declared": seeds is not None,
        "target_pairs": None if target is None else int(target),
        "require_full_fidelity": True,
        "min_practical_gain_pp": goal.get("min_practical_gain_pp"),
        "control_identity": goal.get("control_candidate_id") or protocol.get("control_candidate_id") or "baseline",
        "evaluation_version": protocol.get("schema_version") or SCHEMA_VERSION,
        "evaluation_hash": protocol.get("fingerprint"),
        "aggregate": "mean_majority",
        "split_seed": protocol.get("split_seed"),
        "diagnostic_sampling_seed": goal.get("diagnostic_sampling_seed", protocol.get("split_seed")),
    }
    if goal.get("evaluation_mode") == "loso_method_search":
        body.update(evaluation_mode="loso_method_search", aggregate="paired_complete_loso_suites",
                    unit="paired_complete_method_suites", required_fold_count=10,
                    benchmark_feedback_enabled=goal.get("benchmark_feedback_enabled") is True,
                    replicated=False, statistical_significance="not_established")
    hashed = {key: value for key, value in body.items() if key != "policy_hash"}
    body["policy_hash"] = _digest(hashed)
    return body


def write_confirmation_policy(camp: Path, policy: dict[str, Any]) -> dict[str, Any]:
    path = Path(camp) / POLICY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return policy


def load_confirmation_policy(camp: Path | None = None, goal: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if camp is not None:
        path = Path(camp) / POLICY_NAME
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("policy_hash"):
                return payload
    if goal is not None:
        try:
            return freeze_confirmation_policy(goal, None)
        except ConfirmationPolicyError:
            return None
    return None


def policy_training_seeds(policy: dict[str, Any] | None) -> list[int] | None:
    if not isinstance(policy, dict):
        return None
    if not policy.get("seeds_declared"):
        return None
    return [int(item) for item in policy.get("training_seeds") or []]
