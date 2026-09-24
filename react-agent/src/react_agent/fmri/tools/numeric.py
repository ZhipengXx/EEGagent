"""Numeric inspection tools: input validation, summaries, temporal diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from react_agent.fmri.data import inspect_array
from react_agent.fmri.jsonutil import finite_or_none
from react_agent.fmri.schemas import Availability, CostEstimate, Coverage, Finding, SampleSpec, ToolSpec
from react_agent.fmri.tools.base import RunContext, load_tv, make_result, timed

VALIDATE_SPEC = ToolSpec(
    name="validate_input",
    version="1.0.0",
    description="Check file readability, 2D numeric shape, axes, and finite values.",
    level="coarse",
    issue_types=["malformed", "incomplete_metadata", "degenerate_signal"],
    preconditions=[],
    cost_class="cheap",
    stage_hint="L0",
    answers_questions=["input_mapping"],
    can_affect_verdict=True,
    cost_hint="cheap",
)

STATS_SPEC = ToolSpec(
    name="basic_statistics",
    version="1.0.0",
    description="Summarize shape, finite ratio, moments, and cheap temporal change.",
    level="coarse",
    issue_types=["degenerate_signal", "low_temporal_change", "temporal_spike"],
    preconditions=["validate_input"],
    cost_class="cheap",
    stage_hint="L0",
    answers_questions=["input_mapping"],
    can_affect_verdict=True,
    cost_hint="cheap",
)

TEMPORAL_SPEC = ToolSpec(
    name="temporal_diagnostics",
    version="1.0.0",
    description="Lag-1 correlation, adjacent diffs, and peak-frame description.",
    level="medium",
    issue_types=["temporal_structure", "temporal_spike", "insufficient_length"],
    preconditions=["validate_input", "basic_statistics"],
    cost_class="medium",
    stage_hint="L1",
    answers_questions=["temporal"],
    cost_hint="medium",
)


class ValidateInputTool:
    """Mandatory cheap input contract check."""

    spec = VALIDATE_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Always available."""
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Cheap local IO."""
        return CostEstimate(cost_class="cheap", estimated_seconds=0.05)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Inspect the array file and metadata completeness."""
        started = timed()
        from pathlib import Path

        handle, info = inspect_array(sample, Path(context.base_dir))
        findings: list[Finding] = []
        metrics: dict[str, Any] = dict(info)
        limitations: list[str] = []
        notes: dict[str, str] = {}

        if handle is None:
            code = str(info.get("error", "undecodable"))
            findings.append(
                Finding(
                    code="malformed",
                    severity="error",
                    message=f"Input array is not usable: {code}",
                    evidence_refs=[],
                )
            )
            metrics["valid_for_numeric_checks"] = False
            return make_result(
                self.spec,
                status="success",
                findings=findings,
                metrics=metrics,
                coverage=Coverage(description="file inspect failed"),
                elapsed=timed() - started,
            )

        context.array_handle = handle
        data = handle.load()
        finite_mask = np.isfinite(data)
        finite_ratio = float(finite_mask.mean())
        metrics.update(
            {
                "T": handle.shape_tv[0],
                "V": handle.shape_tv[1],
                "dtype": handle.dtype,
                "finite_ratio": finite_ratio,
                "valid_for_numeric_checks": bool(finite_ratio == 1.0),
            }
        )
        if finite_ratio < 1.0:
            n_nan = int(np.isnan(data).sum())
            n_inf = int(np.isinf(data).sum())
            metrics["n_nan"] = n_nan
            metrics["n_inf"] = n_inf
            notes["finite_ratio"] = "NaN/Inf converted to null in JSON metrics elsewhere"
            findings.append(
                Finding(
                    code="malformed",
                    severity="error",
                    message=f"Array contains non-finite values (nan={n_nan}, inf={n_inf}).",
                    evidence_refs=[],
                )
            )
            metrics["valid_for_numeric_checks"] = False

        missing: list[str] = []
        if sample.sampling_interval_s is None:
            missing.append("sampling_interval_s")
        if sample.spatial_representation in {None, "unknown"} and not sample.space_name:
            missing.append("spatial_metadata")
        if not sample.generation_profile_id:
            missing.append("generation_profile_id")
        if missing:
            metrics["incomplete_metadata_fields"] = missing
            findings.append(
                Finding(
                    code="incomplete_metadata",
                    severity="warning",
                    message="Metadata incomplete: " + ", ".join(missing),
                    evidence_refs=[],
                )
            )
            limitations.append(
                "Missing metadata is not a signal error; it only limits later tools."
            )

        finite = data[np.isfinite(data)]
        if finite.size and np.nanstd(finite) == 0:
            metrics["degenerate_signal"] = True
            findings.append(
                Finding(
                    code="degenerate_signal",
                    severity="flag",
                    message="All finite values are constant (including all-zero).",
                    evidence_refs=[],
                )
            )
        else:
            metrics["degenerate_signal"] = False

        if not findings:
            findings.append(
                Finding(
                    code="valid_for_numeric_checks",
                    severity="info",
                    message="Array is a finite 2D numeric series and can be summarized.",
                    evidence_refs=[],
                )
            )

        return make_result(
            self.spec,
            status="success",
            findings=findings,
            metrics=metrics,
            coverage=Coverage(
                description="file and dtype/shape contract",
                axes=["time", "space"],
                n_time=handle.shape_tv[0],
                n_space=handle.shape_tv[1],
            ),
            limitations=limitations,
            elapsed=timed() - started,
            metric_notes=notes,
        )


class BasicStatisticsTool:
    """Mandatory cheap numeric summary."""

    spec = STATS_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need a readable array."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Cheap CPU summary."""
        return CostEstimate(cost_class="cheap", estimated_seconds=0.1)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Compute summaries; do not treat mean~0 as an error."""
        started = timed()
        params = context.config.basic_statistics
        data = np.asarray(load_tv(context), dtype=np.float64)
        t_len, v_len = data.shape
        finite = np.isfinite(data)
        finite_ratio = float(finite.mean())
        values = data[finite]
        limitations = [
            "Spatial mean of frame-to-frame diffs can hide local anomalies.",
            "Mean near 0 after normalization is not an error.",
        ]
        findings: list[Finding] = []
        notes: dict[str, str] = {}

        def _stat(func) -> float | None:
            if values.size == 0:
                return None
            return finite_or_none(func(values))

        quantiles = {}
        for q in params.quantiles:
            quantiles[str(q)] = (
                finite_or_none(np.quantile(values, q)) if values.size else None
            )

        abs_tol = params.near_constant.abs_tol
        rel_tol = params.near_constant.rel_tol
        col_std = np.nanstd(data, axis=0)
        col_mean = np.nanmean(data, axis=0)
        near = (col_std <= abs_tol) | (col_std <= rel_tol * np.maximum(np.abs(col_mean), 1e-12))
        near_frac = float(np.mean(near)) if v_len else None

        spatial_mean = np.nanmean(data, axis=1)
        diffs = np.abs(np.diff(spatial_mean)) if t_len > 1 else np.array([])
        sm_std = float(np.nanstd(spatial_mean)) if t_len else 0.0
        temporal_change_ratio = (
            finite_or_none(float(np.mean(diffs)) / (sm_std + 1e-8)) if diffs.size else None
        )
        med_diff = float(np.median(diffs)) if diffs.size else 0.0
        max_diff_ratio = (
            finite_or_none(float(np.max(diffs)) / (med_diff + 1e-8)) if diffs.size else None
        )
        time_var = np.nanvar(data, axis=0)
        space_var = np.nanvar(data, axis=1)

        metrics = {
            "T": t_len,
            "V": v_len,
            "dtype": str(data.dtype),
            "finite_ratio": finite_ratio,
            "mean": _stat(np.mean),
            "std": _stat(np.std),
            "min": _stat(np.min),
            "max": _stat(np.max),
            "quantiles": quantiles,
            "zero_fraction": finite_or_none(float(np.mean(data == 0))),
            "near_constant_spatial_fraction": near_frac,
            "near_constant_abs_tol": abs_tol,
            "near_constant_rel_tol": rel_tol,
            "time_var_mean": finite_or_none(float(np.nanmean(time_var))),
            "space_var_mean": finite_or_none(float(np.nanmean(space_var))),
            "temporal_change_ratio": temporal_change_ratio,
            "max_frame_diff_ratio": max_diff_ratio,
            "temporal_change_definition": (
                "mean(|diff(spatial_mean_t)|) / (std(spatial_mean_t)+1e-8)"
            ),
            "max_frame_diff_definition": (
                "max(|diff(spatial_mean_t)|) / (median(|diff|)+1e-8); heuristic"
            ),
        }

        if near_frac is not None and near_frac >= 0.99:
            findings.append(
                Finding(
                    code="degenerate_signal",
                    severity="flag",
                    message="Nearly all spatial series are constant within configured tolerances.",
                    evidence_refs=[],
                )
            )

        if params.enable_demo_temporal_heuristics:
            notes["temporal_heuristics"] = "demo heuristic; not a physiological threshold"
            low = params.demo_temporal_change_low
            spike = params.demo_temporal_spike_ratio
            if (
                low is not None
                and temporal_change_ratio is not None
                and temporal_change_ratio < low
            ):
                findings.append(
                    Finding(
                        code="low_temporal_change",
                        severity="flag",
                        message=(
                            "Cheap temporal-change pre-screen is below the demo heuristic."
                        ),
                        evidence_refs=[],
                    )
                )
            if (
                spike is not None
                and max_diff_ratio is not None
                and max_diff_ratio > spike
            ):
                findings.append(
                    Finding(
                        code="temporal_spike",
                        severity="flag",
                        message="Adjacent-frame jump exceeds the demo heuristic ratio.",
                        evidence_refs=[],
                    )
                )

        return make_result(
            self.spec,
            status="success",
            findings=findings,
            metrics=metrics,
            coverage=Coverage(
                description="global numeric summary",
                axes=["time", "space"],
                n_time=t_len,
                n_space=v_len,
            ),
            limitations=limitations,
            elapsed=timed() - started,
            metric_notes=notes,
        )


class TemporalDiagnosticsTool:
    """Optional medium temporal description. Not an HRF test."""

    spec = TEMPORAL_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need an array."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Medium CPU scan."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.2)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Describe lag-1 correlation and adjacent diffs."""
        if args.get("mode") == "targeted":
            return await self._targeted(sample, args, context)
        started = timed()
        params = context.config.temporal_diagnostics
        data = np.asarray(load_tv(context), dtype=np.float64)
        t_len, v_len = data.shape
        limitations = [
            "Short series: no resting-state bands, functional connectivity, or p-values.",
            "No automatic HRF lag test; alignment metadata was not reinterpreted.",
        ]
        if t_len < params.min_length:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="insufficient_length",
                findings=[
                    Finding(
                        code="insufficient_length",
                        severity="warning",
                        message=f"T={t_len} < min_length={params.min_length}",
                        evidence_refs=[],
                    )
                ],
                metrics={"T": t_len, "min_length": params.min_length},
                limitations=limitations,
                elapsed=timed() - started,
            )

        # lag-1 per vertex; undefined if either window is constant
        x0 = data[:-1]
        x1 = data[1:]
        std0 = np.nanstd(x0, axis=0)
        std1 = np.nanstd(x1, axis=0)
        defined = (std0 > 0) & (std1 > 0)
        n_undefined = int((~defined).sum())
        corrs = np.full(v_len, np.nan)
        if defined.any():
            c0 = x0[:, defined] - np.nanmean(x0[:, defined], axis=0)
            c1 = x1[:, defined] - np.nanmean(x1[:, defined], axis=0)
            num = np.nansum(c0 * c1, axis=0)
            den = np.sqrt(np.nansum(c0**2, axis=0) * np.nansum(c1**2, axis=0))
            corrs[defined] = num / np.maximum(den, 1e-12)
        lag1_mean = finite_or_none(float(np.nanmean(corrs))) if defined.any() else None

        spatial_mean = np.nanmean(data, axis=1)
        diffs = np.abs(np.diff(spatial_mean))
        peak_frame = int(np.nanargmax(np.abs(spatial_mean)))
        peak_diff_frame = int(np.nanargmax(diffs)) + 1 if diffs.size else None
        med_diff = float(np.median(diffs)) if diffs.size else 0.0
        max_diff_ratio = (
            finite_or_none(float(np.max(diffs)) / (med_diff + 1e-8)) if diffs.size else None
        )

        metrics: dict[str, Any] = {
            "T": t_len,
            "V": v_len,
            "lag1_corr_mean": lag1_mean,
            "lag1_undefined_count": n_undefined,
            "adjacent_diff_mean": finite_or_none(float(np.mean(diffs))),
            "adjacent_diff_max": finite_or_none(float(np.max(diffs))) if diffs.size else None,
            "max_diff_ratio": max_diff_ratio,
            "peak_abs_mean_frame": peak_frame,
            "peak_diff_frame": peak_diff_frame,
            "peak_diff_from_frame": (peak_diff_frame - 1) if peak_diff_frame is not None else None,
            "peak_diff_to_frame": peak_diff_frame,
            "threshold_source": params.threshold_source,
            "signal_mode": "raw",
        }
        if sample.sampling_interval_s:
            metrics["peak_abs_mean_s"] = finite_or_none(
                peak_frame * sample.sampling_interval_s
            )
            metrics["sampling_interval_s"] = sample.sampling_interval_s
        else:
            limitations.append("No sampling_interval_s; times reported as frame index only.")

        findings: list[Finding] = []
        if n_undefined:
            findings.append(
                Finding(
                    code="undefined_lag1",
                    severity="info",
                    message=f"{n_undefined} spatial series have undefined lag-1 correlation.",
                    evidence_refs=[],
                )
            )
        if (
            max_diff_ratio is not None
            and max_diff_ratio > params.spike_ratio_heuristic
        ):
            findings.append(
                Finding(
                    code="temporal_spike",
                    severity="flag",
                    message=(
                        f"Descriptive adjacent-frame jump (ratio={max_diff_ratio:.3g}) "
                        f"exceeds {params.threshold_source} threshold "
                        f"{params.spike_ratio_heuristic}."
                    ),
                    evidence_refs=[],
                )
            )

        return make_result(
            self.spec,
            status="success",
            findings=findings,
            metrics=metrics,
            coverage=Coverage(
                description="temporal structure summary",
                axes=["time"],
                n_time=t_len,
                n_space=v_len,
            ),
            limitations=limitations,
            signal_provenance={"signal_mode": "raw", "metric_definition_version": "raw_lag1_v1", "T": t_len},
            elapsed=timed() - started,
        )

    async def _targeted(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Inspect one contrast transition. Selectors are frame indices, not code."""
        started = timed()
        from react_agent.fmri.diagnostic.metrics import transition_energy
        from react_agent.fmri.tools.temporal_profile import _contrast_from_prior

        signal = args.get("signal", "contrast")
        if signal != "contrast":
            return make_result(
                self.spec,
                status="error",
                error="targeted mode accepts signal=contrast only",
                findings=[
                    Finding(
                        code="bad_selector",
                        severity="error",
                        message="targeted temporal diagnostics require contrast.",
                        decision_effect="none",
                    )
                ],
                elapsed=timed() - started,
            )
        data = _contrast_from_prior(context)
        if data is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_contrast_artifact",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="missing_contrast",
                        severity="warning",
                        message="No bound contrast artifact for targeted inspection.",
                        decision_effect="none",
                    )
                ],
                elapsed=timed() - started,
            )
        try:
            frm = int(args["from_frame"])
        except (KeyError, TypeError, ValueError):
            return make_result(
                self.spec,
                status="error",
                error="from_frame required",
                elapsed=timed() - started,
            )
        energy = transition_energy(np.asarray(data, dtype=np.float64), frm)
        return make_result(
            self.spec,
            status="success" if energy.get("status") == "defined" else "success",
            applicability="applicable" if energy.get("status") == "defined" else "partial",
            findings=[
                Finding(
                    code="transition_described",
                    severity="info",
                    message=(
                        f"contrast transition {energy.get('from_frame')}->{energy.get('to_frame')} "
                        f"status={energy.get('status')}. Descriptive only."
                    ),
                    decision_effect="none",
                )
            ],
            metrics={
                "mode": "targeted",
                "signal_mode": "contrast",
                **energy,
            },
            signal_provenance={
                "signal_mode": "contrast",
                "metric_definition_version": "ras_v1",
                "T": int(np.asarray(data).shape[0]),
            },
            limitations=["Targeted contrast change is not a defect label."],
            elapsed=timed() - started,
        )
