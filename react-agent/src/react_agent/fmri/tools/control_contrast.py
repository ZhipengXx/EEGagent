"""Gray-control contrast: what changes versus the explicit gray video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.jsonutil import finite_or_none
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import (
    ArtifactRef,
    Availability,
    CostEstimate,
    Coverage,
    Finding,
    SampleSpec,
    ToolSpec,
)
from react_agent.fmri.tools.base import RunContext, load_tv, make_result, timed
from react_agent.fmri.tools.shared import peak_stats, rms_curve

CONTRAST_SPEC = ToolSpec(
    name="gray_control_contrast",
    version="1.1.0",
    description="Compare a sample prediction to an explicitly bound gray-control series.",
    level="medium",
    issue_types=["control_mismatch", "control_missing", "low_control_relative_change"],
    preconditions=["validate_input"],
    cost_class="medium",
    scope="pair",
    question="Is there a change relative to the paired gray control?",
    requires=["gray_control"],
    produces=["contrast_array", "response_curve"],
    evidence_family="control_relative",
    valid_claims=["model_output_differs_from_gray_input"],
    unsupported_claims=["snr", "p_value", "statistically_significant_activation"],
    estimated_cpu_seconds=0.2,
    stage_hint="L1",
    answers_questions=["gray_contrast"],
    requires_resources=["gray_control"],
    expected_outputs=["contrast_array", "response_curve"],
    can_affect_verdict=False,
    cost_hint="medium",
)


class GrayControlContrastTool:
    """Y - G on signed raw values. One gray video is not physiological noise."""

    spec = CONTRAST_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need an array; gray itself is later marked not_applicable."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.is_gray_control(sample):
            return Availability(status="available")
        if resolver.resolve_control(sample) is None:
            return Availability(status="unavailable", reason="no_gray_control")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """One subtract and RMS."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.2)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Write a contrast artifact and RMS curve."""
        started = timed()
        limitations = [
            "A single gray video is not a physiological noise estimate.",
            "No SNR, p-value, or significant-activation claim.",
            "Small response is descriptive unless a calibrated flag is enabled.",
        ]
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.is_gray_control(sample):
            return make_result(
                self.spec,
                status="success",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="gray_control_self",
                        severity="info",
                        message="Gray-control sample: contrast versus itself is not applicable.",
                        decision_effect="none",
                    )
                ],
                metrics={"not_applicable": True, "reason": "self_control"},
                limitations=limitations,
                elapsed=timed() - started,
            )
        control = resolver.resolve_control(sample)
        if control is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_gray_control",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="control_missing",
                        severity="warning",
                        message="No explicit gray_control path/id is configured.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        y = np.asarray(load_tv(context), dtype=np.float64)
        g = np.asarray(control.array, dtype=np.float64)
        if g.shape != y.shape:
            return make_result(
                self.spec,
                status="success",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="control_mismatch",
                        severity="warning",
                        message=f"Control shape {list(g.shape)} != sample {list(y.shape)}.",
                        decision_effect="none",
                    )
                ],
                metrics={"comparable": False},
                limitations=limitations,
                elapsed=timed() - started,
            )
        ok, reasons = resolver.compatibility_report(sample, control)
        if "control_is_self" in reasons:
            return make_result(
                self.spec,
                status="success",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="control_is_self",
                        severity="warning",
                        message="Bound control is the same array as the sample.",
                        decision_effect="none",
                    )
                ],
                metrics={"comparable": False, "compat_reasons": reasons},
                limitations=limitations,
                elapsed=timed() - started,
            )
        if not ok:
            return make_result(
                self.spec,
                status="success",
                applicability="partial",
                findings=[
                    Finding(
                        code="control_mismatch",
                        severity="warning",
                        message="Control pairing is not verified: " + "; ".join(reasons),
                        decision_effect="none",
                    )
                ],
                metrics={"comparable": False, "compat_reasons": reasons},
                limitations=limitations
                + ["Unverified pairing cannot be treated as a validated contrast effect."],
                elapsed=timed() - started,
            )

        delta = y - g
        curve = rms_curve(delta)
        g_rms = rms_curve(g)
        overall = float(np.sqrt(np.nanmean(delta**2)))
        max_r = float(np.nanmax(curve)) if curve.size else 0.0
        peak = int(np.nanargmax(curve)) if curve.size else None
        denom = float(np.sqrt(np.nanmean(g**2)))
        normalized = finite_or_none(overall / denom) if denom > 1e-12 else None
        stats = peak_stats(curve)
        from react_agent.fmri.diagnostic.metrics import contrast_series

        ras = contrast_series(delta)
        alignment = (sample.resources.time_alignment_status if sample.resources else None) == "verified"
        peak_time = None
        if alignment and peak is not None and sample.sampling_interval_s:
            peak_time = finite_or_none(peak * sample.sampling_interval_s)

        art_dir = Path(context.artifact_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        contrast_path = art_dir / "contrast_minus_gray.npy"
        curve_path = art_dir / "response_rms.json"
        np.save(contrast_path, delta.astype(np.float32))
        curve_path.write_text(
            __import__("json").dumps(
                {
                    "response_rms": [finite_or_none(float(x)) for x in curve],
                    "control_rms": [finite_or_none(float(x)) for x in g_rms],
                    "weighting": "equal_vertex",
                }
            ),
            encoding="utf-8",
        )

        return make_result(
            self.spec,
            status="success",
            applicability="applicable",
            findings=[
                Finding(
                    code="control_contrast_described",
                    severity="info",
                    message=(
                        f"overall_delta_rms={overall:.4g}; peak_frame={peak}. "
                        "Describes model-output difference only."
                    ),
                    decision_effect="none",
                )
            ],
            metrics={
                "overall_delta_rms": finite_or_none(overall),
                "max_response_rms": finite_or_none(max_r),
                "peak_frame": peak,
                "peak_time_s": peak_time,
                "normalized_delta_over_control_rms": normalized,
                "normalization_denominator": finite_or_none(denom),
                "weighting": "equal_vertex",
                "comparable": True,
                "compat_reasons": reasons,
                "signal_mode": "contrast",
                "ras_v1": ras,
                **stats,
            },
            signal_provenance={
                "signal_mode": "contrast",
                "metric_definition_version": "ras_v1",
                "control_artifact_id": control.fingerprint,
                "T": int(y.shape[0]),
                "normalization_id": sample.normalization,
                "surface_space": sample.space_name,
            },
            coverage=Coverage(
                description="Y-G RMS over vertices",
                axes=["time", "space"],
                n_time=int(y.shape[0]),
                n_space=int(y.shape[1]),
            ),
            artifacts=[
                ArtifactRef(name="contrast_minus_gray", path=str(contrast_path), kind="npy"),
                ArtifactRef(name="response_rms", path=str(curve_path), kind="json"),
            ],
            limitations=limitations,
            resource_fingerprints={"gray_control": control.fingerprint},
            elapsed=timed() - started,
        )
