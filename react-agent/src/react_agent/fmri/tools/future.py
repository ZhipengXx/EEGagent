"""Unused P2 variants. They never return fabricated pass results."""

from __future__ import annotations

from typing import Any

from react_agent.fmri.schemas import Availability, CostEstimate, Finding, SampleSpec, ToolSpec
from react_agent.fmri.tools.base import RunContext, make_result, timed

UNUSED_FLAT_SPEC = ToolSpec(
    name="cortex_mae_flat",
    version="0.0.0",
    description="Reserved CortexMAE-F (flat map). Not wired in this increment.",
    level="fine",
    issue_types=["representation_prior"],
    enabled=False,
    disabled_reason=(
        "unused_variant: CortexMAE-F needs a verified fsaverage5→flat-map projector. "
        "This increment only wires CortexMAE-P."
    ),
    cost_class="expensive",
)

UNUSED_VOLUME_SPEC = ToolSpec(
    name="cortex_mae_volume",
    version="0.0.0",
    description="Reserved CortexMAE-V (volume). Not wired in this increment.",
    level="fine",
    issue_types=["representation_prior"],
    enabled=False,
    disabled_reason=(
        "unused_variant: CortexMAE-V needs a verified surface→MNI cortex projector. "
        "This increment only wires CortexMAE-P."
    ),
    cost_class="expensive",
)


class DisabledTool:
    """Registered but never executable placeholder."""

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Always disabled."""
        return Availability(status="disabled", reason=self.spec.disabled_reason)

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """No cost because it cannot run."""
        return CostEstimate(cost_class="expensive")

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Return skipped, never success/passed."""
        return make_result(
            self.spec,
            status="skipped",
            skip_reason=self.spec.disabled_reason,
            findings=[
                Finding(
                    code="unavailable",
                    severity="info",
                    message=self.spec.disabled_reason or "disabled",
                    evidence_refs=[],
                )
            ],
            elapsed=timed() - timed(),
        )
