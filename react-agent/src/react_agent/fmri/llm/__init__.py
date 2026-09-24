"""LM backend protocol."""

from __future__ import annotations

from typing import Any, Protocol

from react_agent.fmri.schemas import Decision, LMUsage


class LMBackend(Protocol):
    """Vendor-agnostic decision/summary adapter."""

    name: str

    async def decide(
        self,
        observation: dict[str, Any],
        candidate_tools: list[str],
        profile: str,
    ) -> tuple[Decision, LMUsage]:
        """Return a structured Decision plus usage."""

    async def summarize(
        self,
        report: dict[str, Any],
        profile: str,
    ) -> tuple[dict[str, Any], LMUsage]:
        """Return an explanatory summary that must not change verdicts."""
