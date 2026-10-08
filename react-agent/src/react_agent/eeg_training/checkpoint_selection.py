"""Select finite development checkpoints independently of the patience monitor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from react_agent.eeg_training.protocol import SplitError


@dataclass
class CheckpointSelection:
    """Keep raw best, saved best and significant-progress anchor separate."""

    policy: str = "legacy_min_delta"
    min_delta: float = 0.001
    raw_best: float | None = None
    saved_best: float | None = None
    patience_anchor: float | None = None
    best_epoch: int | None = None
    stall: int = 0

    def observe(self, top1: float, epoch: int) -> bool:
        """Return whether this finite, strictly better epoch should be saved."""
        value = float(top1)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SplitError("non_finite_or_invalid_development_top1")
        if self.policy not in {"legacy_min_delta", "strict_best"}:
            raise SplitError("bad_checkpoint_policy")
        if self.raw_best is None or value > self.raw_best:
            self.raw_best = value
        significant = self.patience_anchor is None or value > self.patience_anchor + self.min_delta
        if significant:
            self.patience_anchor = value
            self.stall = 0
        else:
            self.stall += 1
        threshold = self.min_delta if self.policy == "legacy_min_delta" else 0.0
        save = self.saved_best is None or value > self.saved_best + threshold
        if save:
            self.saved_best = value
            self.best_epoch = int(epoch)
        return save


def validate_training_completion(
    metrics: dict[str, Any], selected: dict[str, Any], protocol: dict[str, Any], fidelity: str,
) -> tuple[bool, str]:
    """Refuse shortened or misselected method-search full evidence."""
    if protocol.get("evaluation_mode", "single_target") != "loso_method_search":
        return True, "legacy_completion_policy"
    if fidelity != "full":
        return False, "method_suite_requires_full"
    try:
        full_epochs = int(protocol.get("full_epochs", 0))
        metrics_epochs = int(metrics.get("last_epoch") or 0)
        selected_epochs = int(selected.get("last_epoch") or 0)
    except (TypeError, ValueError):
        return False, "method_suite_training_epochs_invalid"
    if full_epochs != 50:
        return False, "method_suite_requires_50_epochs"
    overrides = (protocol.get("fidelity_overrides") or {}).get("full") or {}
    if overrides.get("epochs") != 50 or overrides.get("stop") != "single_full" or protocol.get("checkpoint_policy") != "strict_best":
        return False, "method_suite_training_policy_mismatch"
    if metrics_epochs != 50 or selected_epochs != 50:
        return False, "method_suite_training_incomplete"
    if selected.get("policy") != "strict_best" or selected.get("stop_reason") != "epochs_completed":
        return False, "method_suite_checkpoint_policy_mismatch"
    try:
        raw = float(selected["raw_best_top1"])
        saved = float(selected["saved_top1"])
        score = float(metrics["fixed_bank_top1"])
        epoch = int(selected["best_epoch"])
    except (TypeError, ValueError, KeyError):
        return False, "method_suite_checkpoint_selection_missing"
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (raw, saved, score)):
        return False, "method_suite_checkpoint_score_invalid"
    if raw != saved or saved != score or not 1 <= epoch <= 50 or metrics.get("selected_checkpoint_epoch") != epoch:
        return False, "method_suite_checkpoint_selection_mismatch"
    return True, "complete_fixed_50_epoch_training"
