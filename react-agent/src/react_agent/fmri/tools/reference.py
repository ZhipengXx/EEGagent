"""Reference distribution fit and comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.config import FmriCheckConfig, load_config
from react_agent.fmri.data import ArrayHandle, inspect_array
from react_agent.fmri.jsonutil import finite_or_none, jsonable
from react_agent.fmri.schemas import Availability, CostEstimate, Coverage, Finding, SampleSpec, ToolSpec
from react_agent.fmri.tools.base import RunContext, make_result, timed
from react_agent.fmri.tools.numeric import BasicStatisticsTool

REF_SPEC = ToolSpec(
    name="reference_distribution",
    version="1.1.0",
    description="Compare basic_statistics features to a frozen reference set.",
    level="medium",
    issue_types=["reference_deviation", "incompatible_reference", "insufficient_reference"],
    preconditions=["basic_statistics", "reference_stats"],
    cost_class="medium",
    question="Is there a calibrated distribution check for this sample?",
    scope="sample",
    evidence_family="numeric",
    valid_claims=["relative_feature_shift_vs_frozen_set"],
    unsupported_claims=["sample_quality", "biological_validity"],
    stage_hint="L3",
    answers_questions=["calibrated_reference"],
    requires_resources=["reference"],
    can_affect_verdict=False,
    cost_hint="medium",
)

FEATURE_KEYS = [
    "T",
    "V",
    "mean",
    "std",
    "finite_ratio",
    "zero_fraction",
    "near_constant_spatial_fraction",
    "temporal_change_ratio",
    "time_var_mean",
    "space_var_mean",
]


class ReferenceDistributionTool:
    """Compare against a precomputed reference. Never updates the reference."""

    spec = REF_SPEC

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        """Need a stats file on disk."""
        path = context.config.reference.stats_path
        if not path:
            return Availability(status="unavailable", reason="no_reference_stats")
        if not Path(path).exists():
            return Availability(status="unavailable", reason="reference_stats_missing")
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        """Cheap JSON compare."""
        return CostEstimate(cost_class="medium", estimated_seconds=0.05)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        """Score feature-wise robust deviations."""
        started = timed()
        params = context.config.reference
        limitations = [
            "Deviation is not an error probability.",
            "A generated reference only describes same-pipeline relative shift.",
            "Matching a reference does not prove image-conditioned real fMRI.",
        ]
        stats_path = params.stats_path
        if not stats_path or not Path(stats_path).exists():
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_reference_stats",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="warning",
                        message="Reference stats path missing.",
                        evidence_refs=[],
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )

        with Path(stats_path).open("r", encoding="utf-8") as handle:
            ref = json.load(handle)

        compat_ok, compat_reasons = _compatibility(sample, ref, params.required_compat_fields)
        n_ref = int(ref.get("n_samples", 0))
        fingerprints = set(ref.get("fingerprints", []))
        if context.fingerprint in fingerprints:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="sample_in_reference",
                findings=[
                    Finding(
                        code="reference_unassessed",
                        severity="warning",
                        message="Sample fingerprint overlaps the reference set; reference was not evaluated.",
                        evidence_refs=[],
                        decision_effect="none",
                        calibration_status="insufficient",
                    )
                ],
                metrics={
                    "overlap": True,
                    "comparable": False,
                    "reference_n": n_ref,
                    "calibration_status": "insufficient",
                    "reference_assessed": False,
                },
                applicability="not_applicable",
                calibration_status="insufficient",
                decision_effect="none",
                limitations=limitations + ["Overlap skip is not a completed reference check."],
                elapsed=timed() - started,
            )
        if n_ref < params.min_samples or not compat_ok:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="incompatible_or_too_small",
                findings=[
                    Finding(
                        code="incompatible_reference",
                        severity="warning",
                        message="; ".join(compat_reasons)
                        or f"n_ref={n_ref} < min_samples={params.min_samples}",
                        evidence_refs=[],
                    )
                ],
                metrics={
                    "n_ref": n_ref,
                    "comparable": False,
                    "compat_reasons": compat_reasons,
                    "reference_type": ref.get("reference_type"),
                },
                limitations=limitations,
                elapsed=timed() - started,
            )

        sample_features = _features_from_prior(context)
        if sample_features is None:
            return make_result(
                self.spec,
                status="error",
                error="basic_statistics features missing",
                elapsed=timed() - started,
            )

        quality_ok = bool(params.quality_verdict_enabled) and n_ref >= params.quality_min_n
        if quality_ok:
            calib = "calibrated"
        elif n_ref < params.quality_min_n:
            calib = "insufficient"
        else:
            calib = "engineering_smoke"
        limitations.append(
            f"calibration_status={calib}; min_samples is an engineering floor, not a validity proof."
        )
        if not quality_ok:
            limitations.append(
                "Differences are descriptive only and cannot change the quality verdict."
            )

        deviations: dict[str, Any] = {}
        flags: list[Finding] = []
        for key in FEATURE_KEYS:
            feat = ref.get("features", {}).get(key, {})
            median = feat.get("median")
            mad = feat.get("mad")
            x = sample_features.get(key)
            if x is None or median is None:
                deviations[key] = {"status": "missing"}
                continue
            mad_f = float(mad) if mad is not None else 0.0
            if mad_f < params.mad_eps:
                deviations[key] = {
                    "status": "undefined_mad",
                    "value": x,
                    "median": median,
                    "mad": mad,
                }
                continue
            robust_z = (float(x) - float(median)) / (mad_f * 1.4826)
            entry = {
                "status": "ok",
                "value": x,
                "median": median,
                "mad": mad,
                "robust_z": finite_or_none(robust_z),
            }
            deviations[key] = entry
            if abs(robust_z) >= params.robust_z_flag:
                flags.append(
                    Finding(
                        code="reference_deviation" if quality_ok else "insufficient_reference",
                        severity="flag" if quality_ok else "info",
                        message=(
                            f"{key} robust_z={robust_z:.3g} vs frozen reference"
                            + ("" if quality_ok else " (descriptive; insufficient calibration).")
                        ),
                        evidence_refs=[],
                        decision_effect="flag" if quality_ok else "none",
                        calibration_status=calib,  # type: ignore[arg-type]
                    )
                )

        if not flags:
            flags.append(
                Finding(
                    code="reference_comparable",
                    severity="info",
                    message="Features compared to frozen reference; no robust_z flag.",
                    evidence_refs=[],
                    decision_effect="none",
                    calibration_status=calib,  # type: ignore[arg-type]
                )
            )

        return make_result(
            self.spec,
            status="success",
            findings=flags,
            metrics={
                "comparable": True,
                "n_ref": n_ref,
                "reference_n": n_ref,
                "reference_type": ref.get("reference_type"),
                "reference_source": ref.get("reference_type"),
                "reference_version": ref.get("version"),
                "calibration_status": calib,
                "quality_verdict_enabled": quality_ok,
                "reference_assessed": quality_ok,
                "deviations": deviations,
            },
            coverage=Coverage(description="feature-wise reference compare"),
            limitations=limitations
            + [f"reference_type={ref.get('reference_type')}"],
            applicability="applicable" if quality_ok else "partial",
            calibration_status=calib,
            decision_effect="flag" if quality_ok and any(f.decision_effect == "flag" for f in flags) else "none",
            elapsed=timed() - started,
        )


def _compatibility(
    sample: SampleSpec, ref: dict[str, Any], fields: list[str]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    proto = ref.get("protocol", {})
    for field in fields:
        sample_val = getattr(sample, field, None)
        ref_val = proto.get(field)
        if sample_val != ref_val:
            reasons.append(f"{field} sample={sample_val!r} ref={ref_val!r}")
    return (len(reasons) == 0), reasons


def _features_from_prior(context: RunContext) -> dict[str, Any] | None:
    """Read basic_statistics metrics already stored on the run context."""
    for row in context.prior_results or []:
        if row.get("tool_name") == "basic_statistics" and row.get("execution_status") == "success":
            return extract_features_from_metrics(row.get("metrics") or {})
    return None


def extract_features_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Pick comparable scalar features from basic_statistics metrics."""
    return {k: metrics.get(k) for k in FEATURE_KEYS}


async def compute_basic_features(
    spec: SampleSpec, handle: ArrayHandle, cfg: FmriCheckConfig, base_dir: Path
) -> dict[str, Any]:
    """Run basic_statistics for reference fitting."""
    ctx = RunContext(
        config=cfg,
        sample=spec,
        array_handle=handle,
        base_dir=str(base_dir),
        artifact_dir=str(base_dir),
        fingerprint=handle.fingerprint,
    )
    result = await BasicStatisticsTool().run(spec, {}, ctx)
    if result.execution_status != "success":
        raise RuntimeError(f"basic_statistics failed for {spec.sample_id}")
    return extract_features_from_metrics(result.metrics)


async def fit_reference(
    manifest_path: Path,
    out_path: Path,
    *,
    config: FmriCheckConfig | None = None,
    reference_type: str = "demo_synthetic",
) -> dict[str, Any]:
    """Fit frozen median/MAD stats from an explicit reference manifest."""
    from react_agent.fmri.data import load_manifest

    cfg = config or load_config()
    rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    protocol: dict[str, Any] | None = None
    for spec, base in load_manifest(manifest_path):
        handle, info = inspect_array(spec, base)
        if handle is None:
            raise RuntimeError(f"cannot fit reference on {spec.sample_id}: {info}")
        feats = await compute_basic_features(spec, handle, cfg, base)
        rows.append(feats)
        fingerprints.append(handle.fingerprint)
        proto = {
            k: getattr(spec, k)
            for k in cfg.reference.required_compat_fields
        }
        if protocol is None:
            protocol = proto
        elif protocol != proto:
            raise RuntimeError("reference manifest has mixed compatibility fields")

    n = len(rows)
    features: dict[str, Any] = {}
    for key in FEATURE_KEYS:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            features[key] = {"median": None, "mad": None, "n": 0}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        q = np.quantile(arr, [0.25, 0.75])
        features[key] = {
            "median": median,
            "mad": mad,
            "q25": float(q[0]),
            "q75": float(q[1]),
            "n": len(vals),
        }
    payload = {
        "version": "reference_stats.v1",
        "n_samples": n,
        "features": jsonable(features),
        "fingerprints": fingerprints,
        "protocol": protocol,
        "reference_type": reference_type,
        "compat_fields": cfg.reference.required_compat_fields,
        "min_samples": cfg.reference.min_samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
