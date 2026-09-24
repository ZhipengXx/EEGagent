"""Independent image–fMRI RSA using frozen CLIP. Not TRIBE hf_cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.cohort import cosine
from react_agent.fmri.data import load_manifest
from react_agent.fmri.providers.clip_image import ClipImageProvider
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

SEMANTIC_SPEC = ToolSpec(
    name="semantic_consistency",
    version="1.2.0",
    description="Independent CLIP vs fMRI RSA on a fixed cohort. Not biological validity.",
    level="fine",
    issue_types=["image_fmri_consistency"],
    preconditions=["validate_input"],
    cost_class="expensive",
    scope="cohort",
    question="Do image and fMRI similarity structures align?",
    requires=["clip_embeddings", "cohort"],
    produces=["rsa_summary"],
    evidence_family="image_fmri_rsa_descriptive",
    valid_claims=["descriptive_rdm_alignment"],
    unsupported_claims=["biological_validity", "independent_brain_validation", "p_as_independent"],
    estimated_cpu_seconds=1.0,
    stage_hint="L3",
    answers_questions=["image_fmri_rsa"],
    requires_resources=["clip_embeddings", "cohort"],
    can_affect_verdict=False,
    cost_hint="expensive",
)


class SemanticConsistencyTool:
    """Spearman of RDM upper triangles. N is small; no independent p-values."""

    spec = SEMANTIC_SPEC

    def __init__(self, provider: ClipImageProvider | None = None) -> None:
        self.provider = provider

    def _provider(self, context: RunContext) -> ClipImageProvider:
        if self.provider is not None:
            return self.provider
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        return ClipImageProvider(
            asset_root=resolver.asset_root(),
            embedding_dir=resolver.image_embedding_dir(),
            model_id="openai/clip-vit-base-patch32"
            if context.config.resources.image_encoder_id == "clip"
            else context.config.resources.image_encoder_id,
        )

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.is_gray_control(sample) or not sample.image_path:
            return Availability(status="unavailable", reason="no_image_path")
        provider = self._provider(context)
        image_id = (sample.resources.image_id if sample.resources else None) or Path(
            sample.image_path
        ).name
        if provider.load_cached(image_id) is not None:
            return Availability(status="available")
        if Path(sample.image_path).is_file() and not provider.missing_reasons():
            return Availability(status="available")
        if provider.load_cached(image_id) is None:
            return Availability(
                status="unavailable",
                reason="no_clip_cache_and_encoder_unavailable",
            )
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        return CostEstimate(cost_class="expensive", estimated_seconds=1.0)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        started = timed()
        limitations = [
            "Visual similarity is a proxy, not a real-brain validation.",
            "TRIBE hf_cache visual features are not used.",
            "RDM upper-triangle entries are not independent samples; no p-value.",
        ]
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        if resolver.is_gray_control(sample) or not sample.image_path:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="no_image_path",
                applicability="not_applicable",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="info",
                        message="Gray control / missing jpg: RSA not applicable.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        provider = self._provider(context)
        rows = _cohort_rows(resolver, context)
        image_vecs: list[np.ndarray] = []
        fmri_vecs: list[np.ndarray] = []
        ids: list[str] = []
        for spec, fmri in rows:
            image_id = (spec.resources.image_id if spec.resources else None) or (
                Path(spec.image_path).name if spec.image_path else spec.sample_id
            )
            img = _image_vector(provider, spec, image_id)
            if img is None:
                continue
            image_vecs.append(img)
            fmri_vecs.append(fmri)
            ids.append(spec.sample_id)
        if sample.sample_id not in ids:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="query_embedding_missing",
                applicability="partial",
                findings=[
                    Finding(
                        code="unavailable",
                        severity="warning",
                        message="Query image embedding missing.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        if len(ids) < 3:
            return make_result(
                self.spec,
                status="skipped",
                skip_reason="cohort_too_small",
                applicability="partial",
                findings=[
                    Finding(
                        code="insufficient_rsa_panel",
                        severity="info",
                        message=f"Only {len(ids)} samples have paired CLIP+fMRI vectors.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )
        image_rdm = _rdm(image_vecs)
        fmri_rdm = _rdm(fmri_vecs)
        rho = _spearman_upper(image_rdm, fmri_rdm)
        return make_result(
            self.spec,
            status="success",
            applicability="partial",
            findings=[
                Finding(
                    code="rsa_described",
                    severity="info",
                    message=f"n={len(ids)}; spearman_rdm={rho}",
                    decision_effect="none",
                )
            ],
            metrics={
                "n": len(ids),
                "sample_ids": ids,
                "spearman_rdm": rho,
                "image_encoder": provider.model_id,
                "fmri_feature": "contrast_or_raw_time_mean",
                "used_tribe_hf_cache": False,
            },
            coverage=Coverage(description="independent CLIP–fMRI RSA", axes=["space"]),
            limitations=limitations + [f"n={len(ids)} is too small for a calibrated claim."],
            elapsed=timed() - started,
        )


def _cohort_rows(
    resolver: ResourceResolver, context: RunContext
) -> list[tuple[SampleSpec, np.ndarray]]:
    manifest = resolver.samples_manifest_path()
    rows: list[tuple[SampleSpec, np.ndarray]] = []
    contrast = _contrast_from_prior(context)
    query_feat = np.nanmean(
        np.asarray(contrast if contrast is not None else load_tv(context), dtype=np.float64),
        axis=0,
    )
    if manifest is None:
        rows.append((context.sample, query_feat))
        return rows
    gray = resolver.resolve_control(context.sample)
    gray_arr = None if gray is None else np.asarray(gray.array, dtype=np.float64)
    for spec, _base in load_manifest(manifest):
        if spec.resources and spec.resources.is_gray_control:
            continue
        if not spec.image_path:
            continue
        path = Path(spec.fmri_path)
        if not path.is_file():
            continue
        arr = np.load(path, allow_pickle=False).astype(np.float64)
        if spec.time_axis != 0:
            arr = np.moveaxis(arr, spec.time_axis, 0)
        if gray_arr is not None and arr.shape == gray_arr.shape:
            arr = arr - gray_arr
        rows.append((spec, np.nanmean(arr, axis=0)))
    if not any(spec.sample_id == context.sample.sample_id for spec, _ in rows):
        rows.append((context.sample, query_feat))
    return rows


def _image_vector(
    provider: ClipImageProvider, spec: SampleSpec, image_id: str
) -> np.ndarray | None:
    cached = provider.load_cached(image_id)
    if cached is not None:
        return cached
    if not spec.image_path:
        return None
    path = Path(spec.image_path)
    if not path.is_file():
        return None
    try:
        vec = provider.encode_image(path)
    except Exception:
        return None
    provider.save_cached(image_id, vec)
    return vec


def _rdm(vectors: list[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim = cosine(vectors[i], vectors[j])
            out[i, j] = 1.0 - sim if sim is not None else np.nan
    return out


def _spearman_upper(a: np.ndarray, b: np.ndarray) -> float | None:
    n = a.shape[0]
    if n < 3:
        return None
    iu = np.triu_indices(n, k=1)
    x = a[iu]
    y = b[iu]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None
    rx = _rank(x)
    ry = _rank(y)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks
