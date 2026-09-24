"""Cheap contrast spatial scan. Summary and targeted modes stay descriptive."""

from __future__ import annotations

from typing import Any

import numpy as np

from react_agent.fmri.diagnostic.metrics import transition_energy
from react_agent.fmri.jsonutil import finite_or_none
from react_agent.fmri.schemas import Availability, CostEstimate, Coverage, Finding, SampleSpec, ToolSpec
from react_agent.fmri.tools.base import RunContext, load_tv, make_result, timed
from react_agent.fmri.tools.temporal_profile import _contrast_from_prior

SPATIAL_SPEC = ToolSpec(
    name="surface_spatial_sanity",
    version="1.4.0",
    description="Hemisphere and vertex-energy description of a bound contrast. Not a quality flag.",
    level="medium",
    issue_types=["spatial_description"],
    preconditions=["gray_control_contrast"],
    cost_class="cheap",
    repeatable=True,
    stage_hint="L1",
    answers_questions=["coarse_spatial_description"],
    can_affect_verdict=False,
    cost_hint="cheap",
    dependencies=["gray_control_contrast"],
)


class SurfaceSpatialSanityTool:
    """CPU spatial summary. Mesh adjacency is unavailable without a real mesh."""

    spec = SPATIAL_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        return CostEstimate(cost_class="cheap", estimated_seconds=0.2)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        started = timed()
        mode = args.get("mode") or "summary"
        contrast = _contrast_from_prior(context)
        if contrast is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_contrast_artifact",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="missing_contrast",
                        severity="warning",
                        message="Spatial sanity requires a bound contrast artifact.",
                        decision_effect="none",
                    )
                ],
                elapsed=timed() - started,
            )
        data = np.asarray(contrast, dtype=np.float64)
        if mode == "targeted":
            return self._targeted(data, args, started)
        return self._summary(data, started)

    def _summary(self, data: np.ndarray, started: float):
        t_len, v_len = data.shape
        split = v_len // 2
        mask_note = "all_vertices_no_atlas_medial_exclusion"
        if v_len % 2 != 0:
            hemi = {"status": "undefined", "reason": "odd_vertex_count_lh_rh_order_unverified"}
        else:
            def _rms(block: np.ndarray) -> float | None:
                return finite_or_none(float(np.sqrt(np.mean(block**2))))

            hemi = {
                "order": "LH||RH",
                "lh_rms": _rms(data[:, :split]),
                "rh_rms": _rms(data[:, split:]),
                "mask_id": mask_note,
            }
        s = np.sqrt(np.mean(np.diff(data, axis=0) ** 2, axis=1)) if t_len > 1 else np.asarray([])
        frm = int(np.argmax(s)) if s.size else 0
        energy = transition_energy(data, frm) if t_len > 1 else {"status": "undefined"}
        metrics = {
            "mode": "summary",
            "signal_mode": "contrast",
            "hemisphere_rms": hemi,
            "vertex_temporal_rms_mean": finite_or_none(float(np.mean(np.sqrt(np.mean(data**2, axis=0))))),
            "top_transition": energy,
            "neighborhood": "unavailable",
            "neighborhood_reason": "no_mesh_adjacency_loaded",
        }
        return make_result(
            self.spec,
            status="success",
            applicability="applicable",
            findings=[
                Finding(
                    code="spatial_summary_described",
                    severity="info",
                    message="Contrast spatial summary. Top-1% energy is a set size, not a threshold.",
                    decision_effect="none",
                )
            ],
            metrics=metrics,
            coverage=Coverage(description="contrast spatial summary", axes=["space", "time"], n_time=t_len, n_space=v_len),
            signal_provenance={
                "signal_mode": "contrast",
                "metric_definition_version": "ras_v1",
                "hemisphere_order": "LH||RH",
                "vertex_mask_id": mask_note,
                "T": t_len,
            },
            limitations=[
                "No mesh was loaded; vertex-index neighbors are not surface neighbors.",
                "Medial wall was not excluded because no atlas mask was applied in summary mode.",
            ],
            elapsed=timed() - started,
        )

    def _targeted(self, data: np.ndarray, args: dict[str, Any], started: float):
        try:
            frm = int(args["from_frame"])
        except (KeyError, TypeError, ValueError):
            return make_result(
                self.spec,
                status="error",
                error="from_frame required",
                elapsed=timed() - started,
            )
        energy = transition_energy(data, frm)
        return make_result(
            self.spec,
            status="success",
            applicability="applicable" if energy.get("status") == "defined" else "partial",
            findings=[
                Finding(
                    code="spatial_targeted_described",
                    severity="info",
                    message="Targeted vertex energy. Neighborhood stays unavailable without a mesh.",
                    decision_effect="none",
                )
            ],
            metrics={"mode": "targeted", "signal_mode": "contrast", "neighborhood": "unavailable", **energy},
            signal_provenance={"signal_mode": "contrast", "metric_definition_version": "ras_v1"},
            limitations=["Index adjacency was not treated as mesh adjacency."],
            elapsed=timed() - started,
        )
