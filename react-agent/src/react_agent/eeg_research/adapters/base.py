"""Adapter protocol. Training entry points stay behind this boundary."""

from __future__ import annotations

from typing import Any, Protocol

from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec


class TrainingAdapter(Protocol):
    """Run one registered profile. Implementations must not accept shell text."""

    adapter_id: str
    availability: str
    reason: str
    requires_gpu_cap: bool

    def probe(self) -> dict[str, Any]:
        """Check the local entry without starting training."""

    def run(self, spec: ExperimentSpec, output_dir: str) -> EvidenceBundle:
        """Execute one trial and return validation metrics only."""
