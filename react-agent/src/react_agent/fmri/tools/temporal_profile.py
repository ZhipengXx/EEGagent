"""Stimulus temporal profile, preferring an existing contrast artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.jsonutil import finite_or_none
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
from react_agent.fmri.tools.shared import event_windows, peak_stats, rms_curve, step_rms

PROFILE_SPEC = ToolSpec(
    name="stimulus_temporal_profile",
    version="1.1.0",
    description="When does the (contrast) RMS change, and is it a few jumps?",
    level="medium",
    issue_types=["temporal_structure"],
    preconditions=["validate_input"],
    cost_class="medium",
    scope="sample",
    question="When does the change appear, and is it dominated by a few jumps?",
    produces=["temporal_profile"],
    evidence_family="temporal_descriptive",
    valid_claims=["descriptive_peak_and_step"],
    unsupported_claims=["hrf_fit", "resting_state_band", "information_leak"],
    estimated_cpu_seconds=0.15,
    stage_hint="L1",
    answers_questions=["temporal"],
    expected_outputs=["temporal_profile"],
    cost_hint="medium",
)


def _contrast_from_prior(context: RunContext) -> np.ndarray | None:
    for row in context.prior_results or []:
        if row.get("tool_name") != "gray_control_contrast":
            continue
        if row.get("execution_status") != "success":
            continue
        if row.get("applicability") != "applicable":
            continue
        for art in row.get("artifacts") or []:
            if art.get("name") == "contrast_minus_gray":
                path = Path(art["path"])
                if path.is_file():
                    return np.load(path, allow_pickle=False)
    return None


class StimulusTemporalProfileTool:
    """Describe RMS / step curves. Contrast mode when a valid Delta exists."""

    spec = PROFILE_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need an array."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Curve summaries."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.15)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Write temporal_profile.json."""
        started = timed()
        limitations = [
            "A 5s post-stimulus peak is not required.",
            "T=12 is not used for resting-state bands, FC, or HRF fitting.",
            "Pre-stimulus contrast is not labeled information leak.",
        ]
        contrast = _contrast_from_prior(context)
        if contrast is not None:
            data = np.asarray(contrast, dtype=np.float64)
            mode = "contrast"
        else:
            data = np.asarray(load_tv(context), dtype=np.float64)
            mode = "raw_temporal_description"
            limitations.append("No verified contrast; this is a raw-output temporal description.")
        curve = rms_curve(data)
        steps = step_rms(curve)
        stats = peak_stats(curve)
        from react_agent.fmri.diagnostic.metrics import contrast_series, image16_windows

        ras = contrast_series(data) if mode == "contrast" else None
        windows16 = image16_windows(data.shape[0], sample.sampling_interval_s)
        require_contrast = bool(getattr(context.config, "diagnostic", None) and context.config.diagnostic.enabled and context.config.diagnostic.require_contrast_for_temporal)
        if require_contrast and mode != "contrast":
            return make_result(
                self.spec,
                status="success",
                applicability="partial",
                findings=[
                    Finding(
                        code="contrast_required",
                        severity="warning",
                        message="Diagnostic profile requires contrast D; raw temporal profile does not answer it.",
                        decision_effect="none",
                    )
                ],
                metrics={"signal_mode": "raw", "mode": mode, "satisfies_contrast_contract": False},
                limitations=limitations,
                signal_provenance={"signal_mode": "raw", "metric_definition_version": "ras_v1"},
                elapsed=timed() - started,
            )
        align = (sample.resources.time_alignment_status if sample.resources else None) == "verified"
        events = sample.stimulus_events or []
        onset = events[0].onset_s if events else None
        duration = events[0].duration_s if events else None
        windows = event_windows(
            data.shape[0],
            sampling_interval_s=sample.sampling_interval_s,
            onset_s=onset,
            duration_s=duration,
            alignment_ok=align,
        )
        if windows and windows["n_stimulus"] <= 1:
            limitations.append("Only one stimulus-period sample; no within-window variance test.")
        payload = {
            "mode": mode,
            "signal_mode": "contrast" if mode == "contrast" else "raw",
            "response_rms": [finite_or_none(float(x)) for x in curve],
            "step_rms": [finite_or_none(float(x)) for x in steps],
            "windows": windows,
            "image16_windows": windows16,
            "ras_v1": ras,
            "satisfies_contrast_contract": mode == "contrast" and ras is not None,
            **stats,
        }
        art_dir = Path(context.artifact_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / "temporal_profile.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return make_result(
            self.spec,
            status="success",
            applicability="applicable" if mode == "contrast" else "partial",
            findings=[
                Finding(
                    code="temporal_profile_described",
                    severity="info",
                    message=f"{mode}: peak_frame={stats.get('peak_frame')}",
                    decision_effect="none",
                )
            ],
            metrics=payload,
            coverage=Coverage(
                description=f"{mode} RMS temporal profile",
                axes=["time"],
                n_time=int(data.shape[0]),
                n_space=int(data.shape[1]),
            ),
            artifacts=[ArtifactRef(name="temporal_profile", path=str(path), kind="json")],
            limitations=limitations,
            signal_provenance={
                "signal_mode": "contrast" if mode == "contrast" else "raw",
                "metric_definition_version": "ras_v1",
                "T": int(data.shape[0]),
                "time_axis_id": "image16_half_open" if windows16 else "sample_metadata",
            },
            elapsed=timed() - started,
        )
