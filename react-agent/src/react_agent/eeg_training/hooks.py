"""Train-only transform and objective helpers. Sentinel tests run on CPU."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from react_agent.eeg_training.model import contrastive_loss


def image_id_duplicate_stats(image_ids: list[str]) -> dict[str, Any]:
    """Count same-image collisions in one training batch. IDs never enter the encoder."""
    total = len(image_ids)
    counts = Counter(str(item) for item in image_ids)
    extras = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "batch_size": total,
        "unique_image_ids": len(counts),
        "duplicate_rate": 0.0 if total == 0 else extras / total,
        "max_same_image": max(counts.values()) if counts else 0,
        "same_image_positives": extras,
    }


def is_custom_objective(objective: Any) -> bool:
    """True when the candidate replaced the baseline contrastive loss."""
    if objective is None:
        return False
    if objective is contrastive_loss:
        return False
    name = getattr(objective, "__name__", "")
    if name == "contrastive_loss" and not hasattr(objective, "parameters"):
        return False
    return True


def objective_parameters(objective: Any) -> list[Any]:
    """Learnable objective weights that must enter the optimizer."""
    parameters = getattr(objective, "parameters", None)
    if callable(parameters):
        return [item for item in parameters() if getattr(item, "requires_grad", False)]
    return []


def apply_train_transform(transform: Callable | None, eeg, counters: dict[str, int]):
    """Apply a train-only EEG transform. Validation must not call this."""
    if transform is None:
        return eeg
    counters["transform_train_calls"] = int(counters.get("transform_train_calls") or 0) + 1
    return transform(eeg)


def note_eval_without_transform(counters: dict[str, int]) -> None:
    """Record that eval skipped the stochastic transform."""
    counters["transform_eval_calls"] = int(counters.get("transform_eval_calls") or 0)


def compute_objective(objective: Any, eeg_z, img_z, scale, positives: list[str] | None = None):
    """Return a finite scalar loss. Baseline contrastive ignores extra positives."""
    counters = getattr(objective, "_call_counts", None)
    if isinstance(counters, dict):
        counters["objective_train_calls"] = int(counters.get("objective_train_calls") or 0) + 1
    if not is_custom_objective(objective):
        return contrastive_loss(eeg_z, img_z, scale)
    try:
        return objective(eeg_z, img_z, scale, positives)
    except TypeError:
        return objective(eeg_z, img_z, scale)


def capabilities_used_payload(
    *,
    module: str,
    hooks: dict[str, bool],
    parameter_counts: dict[str, int],
    counters: dict[str, int],
    negative_policy: str,
    config_hash: str,
) -> dict[str, Any]:
    """Record which hooks actually ran. Missing hooks stay false, not invented."""
    return {
        "module": module,
        "hooks": hooks,
        "parameter_counts": parameter_counts,
        "transform_train_calls": int(counters.get("transform_train_calls") or 0),
        "transform_eval_calls": int(counters.get("transform_eval_calls") or 0),
        "objective_train_calls": int(counters.get("objective_train_calls") or 0),
        "negative_sampling_policy": negative_policy,
        "config_hash": config_hash,
    }
