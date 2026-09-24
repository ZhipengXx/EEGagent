"""ROI summary tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.data import resolve_path
from react_agent.fmri.jsonutil import finite_or_none
from react_agent.fmri.schemas import Availability, CostEstimate, Coverage, Finding, SampleSpec, ToolSpec
from react_agent.fmri.tools.base import RunContext, load_tv, make_result, timed

ROI_SPEC = ToolSpec(
    name="roi_summary",
    version="1.0.0",
    description="Per-ROI mean/variance/time-course summary given an explicit label map.",
    level="medium",
    issue_types=["roi_summary"],
    preconditions=["validate_input", "roi_map"],
    cost_class="medium",
    stage_hint="L2",
    answers_questions=["roi"],
    requires_resources=["roi_map"],
    cost_hint="medium",
)


class RoiSummaryTool:
    """Summarize labeled spatial groups. Does not invent expected ROIs."""

    spec = ROI_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Require an explicit ROI map whose length matches V."""
        if not sample.roi_map_path:
            return Availability(status="unavailable", reason="no_roi_map")
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Medium CPU reduction."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.2)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Aggregate by integer labels aligned with the spatial axis."""
        started = timed()
        limitations = [
            "ROI means can cancel opposing vertices inside a label.",
            "Labels are not electrode positions and are not image-class activations.",
        ]
        if not sample.roi_map_path:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_roi_map",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="warning",
                        message="No roi_map_path; ROI summary not run.",
                        evidence_refs=[],
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )

        data = np.asarray(load_tv(context), dtype=np.float64)
        labels_path = resolve_path(sample.roi_map_path, Path(context.base_dir))
        labels = np.load(labels_path, allow_pickle=False)
        if labels.ndim != 1 or labels.shape[0] != data.shape[1]:
            return make_result(
                self.spec,
                status="error",
                error="roi_map length/order does not match V",
                findings=[
                    Finding(
                        code="roi_map_mismatch",
                        severity="error",
                        message=(
                            f"ROI map shape {list(labels.shape)} != V={data.shape[1]}"
                        ),
                        evidence_refs=[],
                    )
                ],
                elapsed=timed() - started,
            )

        names: dict[str, str] = {}
        if sample.roi_names_path:
            import json

            names_path = resolve_path(sample.roi_names_path, Path(context.base_dir))
            with names_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            names = {str(k): str(v) for k, v in loaded.items()}

        summaries: list[dict[str, Any]] = []
        for lab in sorted(set(int(x) for x in labels.tolist())):
            mask = labels == lab
            block = data[:, mask]
            spatial_mean = np.nanmean(block, axis=1)
            summaries.append(
                {
                    "label": lab,
                    "name": names.get(str(lab), names.get(lab, None)),  # type: ignore[arg-type]
                    "n_vertices": int(mask.sum()),
                    "mean": finite_or_none(float(np.nanmean(block))),
                    "var": finite_or_none(float(np.nanvar(block))),
                    "time_mean_peak_frame": int(np.nanargmax(np.abs(spatial_mean))),
                    "time_var": finite_or_none(float(np.nanvar(spatial_mean))),
                }
            )

        return make_result(
            self.spec,
            status="success",
            findings=[
                Finding(
                    code="roi_summary",
                    severity="info",
                    message=f"Summarized {len(summaries)} ROI labels.",
                    evidence_refs=[],
                )
            ],
            metrics={"n_rois": len(summaries), "rois": summaries},
            coverage=Coverage(
                description="labeled spatial groups",
                axes=["space"],
                n_time=data.shape[0],
                n_space=data.shape[1],
            ),
            limitations=limitations,
            elapsed=timed() - started,
        )
