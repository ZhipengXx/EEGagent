"""Placeholder for the missing local EEG retrieval trainer."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec

MISSING = (
    "No EEG training entry exists in react-agent. "
    "Need a retrieval trainer, split files, environment, and GPU cap before a real trial."
)


class UnavailableRetrievalAdapter:
    """Refuse to launch training or invent a score."""

    adapter_id = "unavailable_retrieval"
    availability = "unavailable"
    reason = MISSING
    requires_gpu_cap = True

    def probe(self) -> dict[str, Any]:
        """Record the missing entry without touching a GPU."""
        return {"ok": False, "availability": "unavailable", "reason": self.reason}

    def run(self, spec: ExperimentSpec, output_dir: str) -> EvidenceBundle:
        """Always refuse. A missing trainer is not a failed scientific result."""
        raise RuntimeError(self.reason)
