"""Baseline candidate. The wrapper must load the same encoder the trainer already uses."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_training.model import EEGProjectLayer, contrastive_loss


class EEGCandidate:
    """Frozen baseline plugin. New candidates subclass this in their own workspace."""

    candidate_id = "baseline"

    def build_encoder(self, input_spec: dict[str, Any], model_config: dict[str, Any] | None = None) -> EEGProjectLayer:
        """Project EEG [batch, channels, time] to the image feature size."""
        config = model_config or {}
        return EEGProjectLayer(
            z_dim=int(config.get("z_dim", 1024)),
            c_num=int(input_spec["c_num"]),
            timesteps=list(input_spec["timesteps"]),
            drop_proj=float(config.get("drop_proj", 0.3)),
        )

    def build_training_objective(self, objective_config: dict[str, Any] | None = None) -> Any:
        """Return the symmetric contrastive loss. The image target stays the frozen cache."""
        del objective_config
        return contrastive_loss

    def build_training_transform(self, transform_config: dict[str, Any] | None = None) -> Any:
        """Baseline applies no extra train-only transform."""
        del transform_config
        return None
