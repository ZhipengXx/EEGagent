"""Cross-image output specificity against a fixed cohort panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.cohort import load_cohort_index, query_cohort
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import (
    Availability,
    CostEstimate,
    Coverage,
    Finding,
    SampleSpec,
    ToolSpec,
)
from react_agent.fmri.tools.base import RunContext, load_tv, make_result, timed
from react_agent.fmri.tools.temporal_profile import _contrast_from_prior

SPEC_SPEC = ToolSpec(
    name="cross_image_specificity",
    version="1.1.0",
    description="Compare this output to a fixed cohort panel (not a reference distribution).",
    level="medium",
    issue_types=["output_collapse", "duplicate_output"],
    preconditions=["validate_input"],
    cost_class="medium",
    scope="cohort",
    question="Do different images produce nearly the same output?",
    requires=["cohort_index"],
    produces=["similarity_summary"],
    evidence_family="cross_sample_descriptive",
    valid_claims=["output_diversity", "collapse_candidate"],
    unsupported_claims=["semantic_consistency", "wrong_image_label"],
    estimated_cpu_seconds=0.2,
    stage_hint="L2",
    answers_questions=["specificity"],
    requires_resources=["cohort_index"],
    cost_hint="medium",
)


class CrossImageSpecificityTool:
    """Query a frozen panel. Self comparisons are excluded."""

    spec = SPEC_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need a built cohort index."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.cohort_index_path() is None:
            return Availability(status="unavailable", reason="no_cohort_index")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Panel query."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.2)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Compare raw time-mean features to the panel."""
        started = timed()
        limitations = [
            "High similarity is not automatic error; images may themselves be similar.",
            "No image embedding: this is not semantic_consistency.",
            "The 8-sample smoke panel is not a healthy-brain distribution.",
        ]
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        index_path = resolver.cohort_index_path()
        if index_path is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_cohort_index",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="warning",
                        message="No cohort index path configured.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        index = load_cohort_index(index_path)
        contrast = _contrast_from_prior(context)
        if contrast is not None:
            data = np.asarray(contrast, dtype=np.float64)
            mode = "contrast_time_mean"
        else:
            data = np.asarray(load_tv(context), dtype=np.float64)
            mode = "raw_time_mean"
            limitations.append("Raw feature mode: common background can dominate similarity.")
        feature = np.nanmean(data, axis=0)
        fp = context.fingerprint
        result = query_cohort(sample, feature, fp, index)
        findings = [
            Finding(
                code="specificity_described",
                severity="info",
                message=(
                    f"{mode}: n_compared={result['n_compared']}; "
                    f"cosine_max={result['similarity_summary'].get('cosine_max')}"
                ),
                decision_effect="none",
            )
        ]
        if result["exact_duplicates"]:
            findings.append(
                Finding(
                    code="duplicate_output",
                    severity="flag",
                    message=f"Exact array hash matches {result['exact_duplicates']}.",
                    decision_effect="flag",
                )
            )
        return make_result(
            self.spec,
            status="success",
            applicability="applicable" if mode.startswith("contrast") else "partial",
            findings=findings,
            metrics={
                "mode": mode,
                **result,
                "image_id_known": bool(sample.resources and sample.resources.image_id),
            },
            coverage=Coverage(description="cohort panel similarity", axes=["space"]),
            limitations=limitations,
            elapsed=timed() - started,
        )
