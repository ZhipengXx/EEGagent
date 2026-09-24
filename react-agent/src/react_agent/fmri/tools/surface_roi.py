"""Surface ROI profile using a prepared LH||RH atlas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.assets import (
    AtlasError,
    hemisphere_for_index,
    load_atlas,
    roi_key,
    validate_atlas_against_sample,
)
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
from react_agent.fmri.tools.temporal_profile import _contrast_from_prior

ROI_SPEC = ToolSpec(
    name="surface_roi_profile",
    version="1.1.0",
    description="ROI-wise signed mean and RMS on a verified surface atlas.",
    level="medium",
    issue_types=["roi_summary", "atlas_mismatch"],
    preconditions=["validate_input"],
    cost_class="medium",
    scope="sample",
    question="Where on cortex does the change sit?",
    requires=["surface_atlas"],
    produces=["roi_profile"],
    evidence_family="surface_descriptive",
    valid_claims=["descriptive_roi_timecourse"],
    stage_hint="L2",
    answers_questions=["roi"],
    requires_resources=["surface_atlas"],
    cost_hint="medium",
    unsupported_claims=["visual_area_must_peak", "functional_network_identity"],
    estimated_cpu_seconds=0.4,
)


class SurfaceRoiProfileTool:
    """Hemisphere+label_id keys. Does not merge left/right same ids."""

    spec = ROI_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need a prepared atlas directory."""
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.atlas_dir() is None:
            return Availability(status="unavailable", reason="no_atlas")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Per-ROI reductions."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.4)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Write roi_profile.json."""
        if args.get("mode") == "targeted":
            return await self._targeted(sample, args, context)
        started = timed()
        limitations = [
            "Destrieux parcels are structural, not functional networks.",
            "V1.1 does not require visual ROIs to rank first.",
            "ROI means can cancel opposing vertices.",
        ]
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        atlas_dir = resolver.atlas_dir()
        if atlas_dir is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_atlas",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="warning",
                        message="No prepared atlas directory.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        try:
            atlas = load_atlas(atlas_dir)
            labels = np.asarray(atlas["labels"])
            validate_atlas_against_sample(labels, context.array_handle.shape_tv[1] if context.array_handle else None)
        except AtlasError as exc:
            return make_result(
                self.spec,
                status="error",
                error=str(exc),
                findings=[
                    Finding(
                        code="atlas_mismatch",
                        severity="error",
                        message=str(exc),
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )

        contrast = _contrast_from_prior(context)
        if contrast is not None:
            data = np.asarray(contrast, dtype=np.float64)
            mode = "contrast"
        else:
            data = np.asarray(load_tv(context), dtype=np.float64)
            mode = "raw"
            limitations.append("No contrast; reporting raw spatial distribution.")
        names = atlas["names"]
        excluded = {0}  # typical unknown / medial wall id
        n_excl = int(np.isin(labels, list(excluded)).sum())
        rows: list[dict[str, Any]] = []
        unique = sorted(int(x) for x in np.unique(labels) if int(x) not in excluded)
        for lab in unique:
            for hemi in ("lh", "rh"):
                hemi_idx = np.array(
                    [
                        i
                        for i in range(labels.size)
                        if int(labels[i]) == lab and hemisphere_for_index(i) == hemi
                    ],
                    dtype=int,
                )
                if hemi_idx.size == 0:
                    continue
                block = data[:, hemi_idx]
                mean_tc = np.nanmean(block, axis=1)
                rms_tc = np.sqrt(np.nanmean(block**2, axis=1))
                key = roi_key(hemi, lab)
                rows.append(
                    {
                        "roi_key": key,
                        "hemisphere": hemi,
                        "label_id": lab,
                        "name": names.get(key),
                        "n_vertices": int(hemi_idx.size),
                        "signed_mean": finite_or_none(float(np.nanmean(block))),
                        "rms": finite_or_none(float(np.sqrt(np.nanmean(block**2)))),
                        "mean_timecourse": [finite_or_none(float(x)) for x in mean_tc],
                        "rms_timecourse": [finite_or_none(float(x)) for x in rms_tc],
                    }
                )
        ranked = sorted(rows, key=lambda r: -(r["rms"] or 0.0))
        top_k = ranked[:8]
        art_dir = Path(context.artifact_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / "roi_profile.json"
        path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "atlas_id": atlas["manifest"].get("atlas_id"),
                    "excluded_vertex_fraction": n_excl / max(labels.size, 1),
                    "top_k": top_k,
                }
            ),
            encoding="utf-8",
        )
        return make_result(
            self.spec,
            status="success",
            applicability="applicable" if mode == "contrast" else "partial",
            findings=[
                Finding(
                    code="roi_profile_described",
                    severity="info",
                    message=f"{mode}: {len(rows)} ROI keys; top={top_k[0]['roi_key'] if top_k else None}",
                    decision_effect="none",
                )
            ],
            metrics={
                "mode": mode,
                "n_rois": len(rows),
                "excluded_vertex_fraction": n_excl / max(labels.size, 1),
                "top_k": top_k,
                "atlas_id": atlas["manifest"].get("atlas_id"),
                "atlas_source": atlas["manifest"].get("source"),
            },
            coverage=Coverage(
                description="hemisphere+label ROI profile",
                axes=["space", "time"],
                n_time=int(data.shape[0]),
                n_space=int(data.shape[1]),
            ),
            artifacts=[ArtifactRef(name="roi_profile", path=str(path), kind="json")],
            limitations=limitations
            + [f"atlas_hash={atlas.get('fingerprint')}"],
            resource_fingerprints={"atlas": str(atlas.get("fingerprint") or "")},
            elapsed=timed() - started,
        )

    async def _targeted(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """ROI share of transition energy. Unassigned mass is not renormalized away."""
        started = timed()
        from react_agent.fmri.diagnostic.metrics import DENOM_EPS
        from react_agent.fmri.tools.temporal_profile import _contrast_from_prior

        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        atlas_dir = resolver.atlas_dir()
        if atlas_dir is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_atlas",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="resource_blocked",
                        severity="warning",
                        message="ROI follow-up blocked: atlas is not prepared. No download was attempted.",
                        decision_effect="none",
                    )
                ],
                metrics={"mode": "targeted", "status": "blocked", "reason": "no_atlas"},
                elapsed=timed() - started,
            )
        data = _contrast_from_prior(context)
        if data is None:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_contrast_artifact",
                elapsed=timed() - started,
            )
        try:
            frm = int(args["from_frame"])
        except (KeyError, TypeError, ValueError):
            return make_result(self.spec, status="error", error="from_frame required", elapsed=timed() - started)
        arr = np.asarray(data, dtype=np.float64)
        if frm < 0 or frm + 1 >= arr.shape[0]:
            return make_result(self.spec, status="error", error="transition_out_of_range", elapsed=timed() - started)
        try:
            atlas = load_atlas(atlas_dir)
            labels = np.asarray(atlas["labels"])
        except AtlasError as exc:
            return make_result(self.spec, status="error", error=str(exc), elapsed=timed() - started)
        energy = (arr[frm + 1] - arr[frm]) ** 2
        total = float(np.sum(energy))
        if total <= DENOM_EPS:
            return make_result(
                self.spec,
                status="success",
                applicability="partial",
                metrics={"mode": "targeted", "signal_mode": "contrast", "status": "undefined", "reason": "zero_energy"},
                findings=[Finding(code="undefined_contribution", severity="info", message="Zero transition energy.", decision_effect="none")],
                elapsed=timed() - started,
            )
        known = 0.0
        rows = []
        for lab in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
            mask = labels == lab
            part = float(np.sum(energy[mask]))
            known += part
            rows.append({"label_id": lab, "contribution": part / total, "n_vertices": int(mask.sum())})
        unassigned = float(np.sum(energy[labels == 0])) / total
        rows.sort(key=lambda r: -r["contribution"])
        return make_result(
            self.spec,
            status="success",
            findings=[
                Finding(
                    code="roi_contribution_described",
                    severity="info",
                    message="ROI energy share uses the full valid-mask denominator, including unassigned vertices.",
                    decision_effect="none",
                )
            ],
            metrics={
                "mode": "targeted",
                "signal_mode": "contrast",
                "from_frame": frm,
                "to_frame": frm + 1,
                "status": "defined",
                "unassigned_contribution": unassigned,
                "known_contribution": known / total,
                "top": rows[:8],
            },
            signal_provenance={"signal_mode": "contrast", "metric_definition_version": "ras_v1"},
            elapsed=timed() - started,
        )

