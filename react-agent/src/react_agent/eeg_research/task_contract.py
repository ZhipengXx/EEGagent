"""Freeze a task card and reject split leakage before any trial."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from react_agent.eeg_research.schemas import TaskCard


class ContractError(ValueError):
    """Raised when a task card cannot enter auto-research."""


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def freeze_task(card: TaskCard) -> TaskCard:
    """Fill split hashes and the campaign task hash."""
    data = card.model_dump()
    data["train_split_hash"] = data["train_split_hash"] or _digest(sorted(card.train_ids))
    data["validation_split_hash"] = data["validation_split_hash"] or _digest(
        sorted(card.validation_ids)
    )
    data["test_split_hash"] = data["test_split_hash"] or _digest(sorted(card.test_ids))
    data["frozen_task_hash"] = ""
    frozen = TaskCard.model_validate(data)
    identity = frozen.model_dump()
    identity.pop("frozen_task_hash")
    return frozen.model_copy(update={"frozen_task_hash": _digest(identity)})


def validate_splits(card: TaskCard) -> None:
    """Refuse overlap or any use of test for model selection."""
    train = set(card.train_ids)
    validation = set(card.validation_ids)
    test = set(card.test_ids)
    if not train or not validation:
        raise ContractError("missing_split")
    if train & validation:
        raise ContractError("train_validation_overlap")
    if train & test or validation & test:
        raise ContractError("test_split_overlap")
    if card.uses_test_for_tuning:
        raise ContractError("test_used_for_tuning")
    if card.research_scope != "eeg_task_validation":
        raise ContractError("research_scope_mismatch")


def protocol_fingerprint(card: TaskCard) -> dict[str, Any]:
    """Return the fields that must match before two scores are compared."""
    return {
        "frozen_task_hash": card.frozen_task_hash,
        "candidate_bank_hash": card.candidate_bank_hash,
        "target_feature_hash": card.target_feature_hash,
        "candidate_count": card.candidate_count,
        "similarity": card.similarity,
        "trial_aggregation": card.trial_aggregation,
        "metric_implementation_hash": card.metric_implementation_hash,
        "training_fidelity": card.training_fidelity,
        "primary_metric": card.primary_metric,
    }
