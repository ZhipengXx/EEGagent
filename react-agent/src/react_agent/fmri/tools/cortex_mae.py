"""CortexMAE-P embedding check. Descriptive only; never pads T=12."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.p2_store import CORTEXMAE_TRAINED_T
from react_agent.fmri.providers.base import EncoderError
from react_agent.fmri.providers.cortexmae_parcel import CortexMaeParcelProvider
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

CORTEX_SPEC = ToolSpec(
    name="cortex_mae",
    version="1.2.0",
    description="CortexMAE-P embedding on Schaefer-400. Not a validity probability.",
    level="fine",
    issue_types=["representation_prior"],
    preconditions=["validate_input"],
    cost_class="expensive",
    scope="sample",
    question="Where does this output sit in a learned fMRI representation?",
    requires=["schaefer400", "cortexmae_parcel"],
    produces=["cortexmae_embedding"],
    evidence_family="learned_representation_descriptive",
    valid_claims=["descriptive_embedding_distance"],
    unsupported_claims=["validity_probability", "biological_validity"],
    estimated_cpu_seconds=4.0,
    stage_hint="L3",
    answers_questions=["cortexmae_embedding"],
    requires_resources=["schaefer400", "cortexmae_parcel"],
    can_affect_verdict=False,
    cost_hint="expensive",
)


class CortexMaeTool:
    """Official parcel forward. T is never padded."""

    spec = CORTEX_SPEC

    def __init__(self, provider: CortexMaeParcelProvider | None = None) -> None:
        self.provider = provider

    def _provider(self, context: RunContext) -> CortexMaeParcelProvider:
        if self.provider is not None:
            return self.provider
        resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
        return CortexMaeParcelProvider(
            asset_root=resolver.asset_root(),
            schaefer_dir=resolver.schaefer_atlas_dir(),
            model_id=context.config.resources.cortexmae_model_id,
        )

    def availability(self, sample: SampleSpec, context: RunContext) -> Availability:
        if context.array_handle is None:
            return Availability(status="unavailable", reason="no_array")
        provider = self._provider(context)
        missing = provider.missing_reasons()
        if missing:
            return Availability(status="unavailable", reason="; ".join(missing))
        return Availability(status="available")

    def estimate_cost(self, sample: SampleSpec, context: RunContext) -> CostEstimate:
        return CostEstimate(cost_class="expensive", estimated_seconds=4.0)

    async def run(self, sample: SampleSpec, args: dict[str, Any], context: RunContext):
        started = timed()
        limitations = [
            "CortexMAE embedding is not a validity probability.",
            "No HCP reference set: distances are descriptive only.",
            f"Official training clip is T={CORTEXMAE_TRAINED_T}; input T is not padded.",
            f"TRIBE scaling_source={getattr(sample.resources, 'scaling_source', None)}.",
        ]
        provider = self._provider(context)
        mode_arg = str(args.get("input_mode") or "raw")
        if mode_arg == "contrast":
            contrast = _contrast_from_prior(context)
            if contrast is None:
                return make_result(
                    self.spec,
                    status="skipped",
                    skip_reason="contrast_requested_but_missing",
                    applicability="not_applicable",
                    findings=[
                        Finding(
                            code="contrast_unavailable",
                            severity="info",
                            message="input_mode=contrast but no prior gray-control contrast.",
                            decision_effect="none",
                        )
                    ],
                    metrics={
                        "mode": "contrast",
                        "padded": False,
                        "encoder_forward_validated": False,
                        "embedding_available": False,
                        "reference_score_assessed": False,
                    },
                    limitations=limitations,
                    elapsed=timed() - started,
                )
            data = np.asarray(contrast, dtype=np.float64)
            mode = "contrast"
            limitations.append("Experimental contrast input; default is the raw generated series.")
        else:
            data = np.asarray(load_tv(context), dtype=np.float64)
            mode = "raw"
            limitations.append("Default input is the raw generated series, not gray contrast.")
        t_in = int(data.shape[0])
        try:
            embedding, meta = provider.encode_surface(data)
        except EncoderError as exc:
            if exc.code == "temporal_length_unsupported":
                return make_result(
                    self.spec,
                    status="skipped",
                    skip_reason=str(exc),
                    applicability="not_applicable",
                    findings=[
                        Finding(
                            code="temporal_length_unsupported",
                            severity="info",
                            message=str(exc),
                            decision_effect="none",
                        )
                    ],
                    metrics={
                        "input_t": t_in,
                        "trained_t": CORTEXMAE_TRAINED_T,
                        "padded": False,
                        "encoder_forward_validated": False,
                        "embedding_available": False,
                        "reference_score_assessed": False,
                    },
                    limitations=limitations,
                    elapsed=timed() - started,
                )
            return make_result(
                self.spec,
                status="error",
                error=str(exc),
                findings=[
                    Finding(
                        code=exc.code,
                        severity="error",
                        message=str(exc),
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )

        if int(meta.get("input_t") or t_in) != t_in or meta.get("padded"):
            return make_result(
                self.spec,
                status="error",
                error="encoder mutated time length",
                findings=[
                    Finding(
                        code="padded_time",
                        severity="error",
                        message="Encoder changed T; refusing result.",
                        decision_effect="none",
                    )
                ],
                limitations=limitations,
                elapsed=timed() - started,
            )

        art_dir = Path(context.artifact_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        emb_path = art_dir / "cortexmae_embedding.npy"
        np.save(emb_path, embedding.astype(np.float32))
        meta_path = art_dir / "cortexmae_meta.json"
        meta_path.write_text(json.dumps({"mode": mode, **meta}, indent=2), encoding="utf-8")

        ref_dist = _gray_embedding_distance(context, embedding, provider)
        limitations.append(
            f"trained_T={CORTEXMAE_TRAINED_T} vs input_T={t_in}; aggregation={meta['aggregation']}"
        )
        return make_result(
            self.spec,
            status="success",
            applicability="partial" if t_in != CORTEXMAE_TRAINED_T else "applicable",
            findings=[
                Finding(
                    code="cortexmae_embedding_described",
                    severity="info",
                    message=(
                        f"{mode}: dim={embedding.size}; "
                        f"trained_T={CORTEXMAE_TRAINED_T} vs input_T={t_in}"
                    ),
                    decision_effect="none",
                )
            ],
            metrics={
                "mode": mode,
                "embedding_dim": int(embedding.size),
                "embedding_norm": float(np.linalg.norm(embedding)),
                "gray_cosine": ref_dist,
                "encoder_forward_validated": True,
                "embedding_available": True,
                "reference_score_assessed": False,
                **meta,
            },
            coverage=Coverage(
                description="CortexMAE-P parcel embedding",
                axes=["space", "time"],
                n_time=t_in,
                n_space=int(data.shape[1]),
            ),
            artifacts=[
                ArtifactRef(name="cortexmae_embedding", path=str(emb_path), kind="npy"),
                ArtifactRef(name="cortexmae_meta", path=str(meta_path), kind="json"),
            ],
            limitations=limitations,
            elapsed=timed() - started,
        )


def _gray_embedding_distance(
    context: RunContext,
    embedding: np.ndarray,
    provider: CortexMaeParcelProvider,
) -> float | None:
    resolver = ResourceResolver(context.config, base_dir=Path(context.base_dir))
    control = resolver.resolve_control(context.sample)
    if control is None:
        return None
    try:
        other, _ = provider.encode_surface(np.asarray(control.array, dtype=np.float64))
    except EncoderError:
        return None
    na = float(np.linalg.norm(embedding))
    nb = float(np.linalg.norm(other))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(embedding, other) / (na * nb))
