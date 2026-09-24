"""Simple in-process tool registry."""

from __future__ import annotations

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.schemas import SampleSpec, ToolSpec
from react_agent.fmri.tools.base import CheckTool, RunContext
from react_agent.fmri.tools.control_contrast import GrayControlContrastTool
from react_agent.fmri.tools.cortex_mae import CortexMaeTool
from react_agent.fmri.tools.future import DisabledTool, UNUSED_FLAT_SPEC, UNUSED_VOLUME_SPEC
from react_agent.fmri.tools.numeric import BasicStatisticsTool, TemporalDiagnosticsTool, ValidateInputTool
from react_agent.fmri.tools.reference import ReferenceDistributionTool
from react_agent.fmri.tools.semantic import SemanticConsistencyTool
from react_agent.fmri.tools.spatial import RoiSummaryTool
from react_agent.fmri.tools.specificity import CrossImageSpecificityTool
from react_agent.fmri.tools.surface_roi import SurfaceRoiProfileTool
from react_agent.fmri.tools.surface_spatial import SurfaceSpatialSanityTool
from react_agent.fmri.tools.temporal_profile import StimulusTemporalProfileTool


class ToolRegistry:
    """Name -> tool mapping with enablement from config."""

    def __init__(self, tools: list[CheckTool]) -> None:
        self._tools = {t.spec.name: t for t in tools}

    def get(self, name: str) -> CheckTool:
        """Return a registered tool or raise."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unregistered tool {name}") from exc

    def specs(self) -> list[ToolSpec]:
        """All registered specs."""
        return [t.spec for t in self._tools.values()]

    def executable(
        self,
        sample: SampleSpec,
        context: RunContext,
        enabled: list[str],
        completed: set[str],
    ) -> list[str]:
        """Filter registered, enabled, available tools not yet completed."""
        names: list[str] = []
        for name, tool in self._tools.items():
            if getattr(tool.spec, "kind", "check") == "generator":
                continue
            if not tool.spec.enabled:
                continue
            if name not in enabled:
                continue
            diagnostic = bool(getattr(context.config, "diagnostic", None) and context.config.diagnostic.enabled)
            repeatable_diag = diagnostic and name in {
                "temporal_diagnostics",
                "surface_spatial_sanity",
                "surface_roi_profile",
            }
            if name in completed and not tool.spec.repeatable and not repeatable_diag:
                continue
            avail = tool.availability(sample, context)
            if avail.status != "available":
                continue
            missing_deps = [
                dep
                for dep in tool.spec.dependencies + [
                    p for p in tool.spec.preconditions if p in self._tools
                ]
                if dep not in completed and dep in self._tools
            ]
            if missing_deps:
                continue
            prior = list(context.prior_results or [])
            failed = {
                row.get("tool_name")
                for row in prior
                if isinstance(row, dict) and row.get("execution_status") in {"error", "skipped"}
            }
            if any(ev in failed for ev in tool.spec.requires_evidence):
                continue
            names.append(name)
        return names


def build_registry(config: FmriCheckConfig | None = None) -> ToolRegistry:
    """Construct the V1 registry including disabled future tools."""
    _ = config
    tools: list[CheckTool] = [
        ValidateInputTool(),
        BasicStatisticsTool(),
        TemporalDiagnosticsTool(),
        RoiSummaryTool(),
        ReferenceDistributionTool(),
        GrayControlContrastTool(),
        StimulusTemporalProfileTool(),
        SurfaceSpatialSanityTool(),
        SurfaceRoiProfileTool(),
        CrossImageSpecificityTool(),
        CortexMaeTool(),
        SemanticConsistencyTool(),
        DisabledTool(UNUSED_FLAT_SPEC),
        DisabledTool(UNUSED_VOLUME_SPEC),
    ]
    return ToolRegistry(tools)
