"""Attach protocol identity to an evidence bundle."""

from __future__ import annotations

from react_agent.eeg_research.schemas import EvidenceBundle, TaskCard


def bind_protocol(bundle: EvidenceBundle, card: TaskCard) -> EvidenceBundle:
    """Copy frozen comparison fields onto a successful validation bundle."""
    if bundle.execution_status != "succeeded":
        return bundle
    return bundle.model_copy(
        update={
            "candidate_bank_hash": card.candidate_bank_hash,
            "target_feature_hash": card.target_feature_hash,
            "fidelity": card.training_fidelity,
            "metric_name": card.primary_metric,
        }
    )
