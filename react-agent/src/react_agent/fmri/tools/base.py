"""Check-tool protocol, specs, and execution helpers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.data import ArrayHandle
from react_agent.fmri.jsonutil import jsonable
from react_agent.fmri.schemas import (
    Availability,
    CostEstimate,
    Coverage,
    Finding,
    SampleSpec,
    ToolResult,
    ToolSpec,
)


@dataclass
class RunContext:
    """Per-sample runtime passed into tools. Not stored in LangGraph state."""

    config: FmriCheckConfig
    sample: SampleSpec
    array_handle: ArrayHandle | None
    base_dir: str
    artifact_dir: str
    fingerprint: str
    prior_results: list[dict[str, Any]] | None = None


class CheckTool(Protocol):
    """Executable inspection tool."""

    spec: ToolSpec

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Return whether the tool can run."""

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Return a cheap cost estimate."""

    async def run(
        self, sample: SampleSpec, args: dict[str, Any], context: RunContext
    ) -> ToolResult:
        """Execute the check."""


def new_result_id(tool_name: str) -> str:
    """Allocate a stable-enough result id."""
    return f"{tool_name}_{uuid.uuid4().hex[:8]}"


def make_result(
    spec: ToolSpec,
    *,
    status: str,
    findings: list[Finding] | None = None,
    metrics: dict[str, Any] | None = None,
    coverage: Coverage | None = None,
    limitations: list[str] | None = None,
    elapsed: float | None = None,
    error: str | None = None,
    skip_reason: str | None = None,
    metric_notes: dict[str, str] | None = None,
    artifacts: list | None = None,
    applicability: str | None = None,
    diagnostic_question: str | None = None,
    answer_summary: str | None = None,
    calibration_status: str | None = None,
    decision_effect: str | None = None,
    resource_fingerprints: dict[str, str] | None = None,
    signal_provenance: dict[str, Any] | None = None,
    execution_key: str | None = None,
) -> ToolResult:
    """Construct a JSON-safe ToolResult."""
    return ToolResult(
        tool_name=spec.name,
        tool_version=spec.version,
        result_id=new_result_id(spec.name),
        execution_status=status,  # type: ignore[arg-type]
        findings=findings or [],
        metrics=jsonable(metrics or {}),
        coverage=coverage,
        limitations=limitations or [],
        artifacts=artifacts or [],
        elapsed_seconds=elapsed,
        gpu_seconds=None,
        cost_estimate=CostEstimate(cost_class=spec.cost_class),
        error=error,
        cache_hit=False,
        skip_reason=skip_reason,
        metric_notes=metric_notes or {},
        applicability=applicability,  # type: ignore[arg-type]
        diagnostic_question=diagnostic_question or spec.question,
        answer_summary=answer_summary,
        calibration_status=calibration_status,  # type: ignore[arg-type]
        decision_effect=decision_effect,  # type: ignore[arg-type]
        resource_fingerprints=resource_fingerprints or {},
        signal_provenance=signal_provenance,
        execution_key=execution_key,
    )


def timed() -> float:
    """Return a monotonic timestamp."""
    return time.perf_counter()


def load_tv(context: RunContext) -> np.ndarray:
    """Load the [T, V] array or raise."""
    if context.array_handle is None:
        raise RuntimeError("no array handle")
    return context.array_handle.load()
