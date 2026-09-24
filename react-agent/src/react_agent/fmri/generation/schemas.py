"""Contracts for 16s image→fMRI generation. T=16 is never padded from T=12."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from react_agent.fmri.schemas import SampleSpec, ToolSpec

PROFILE_ID = "static_gray4_image1_gray11_tribev2_v1"
EXPECTED_T = 16
EXPECTED_V = 20484
FPS = 10
PRE_GRAY_S = 4.0
IMAGE_S = 1.0
POST_GRAY_S = 11.0
GRAY_RGB = (128, 128, 128)
CODEC = "libx264"
CHECKPOINT = "facebook/tribev2"
MISSING_AUDIO = "zero_fill_aggregate_features"
MISSING_TEXT = "zero_fill_aggregate_features"
EXPORT_RULE_ID = "segment_time_window_0_16_exclusive_v1"
ALIGNMENT = (
    "preds[k] aligns with segments[k].start as t_video. "
    "Training FmriExtractor.offset=5s is already in the model; "
    "the checker does not shift the series again."
)

GENERATE_SPEC = ToolSpec(
    name="tribev2_generate_fmri",
    version="1.2.0",
    description="Build 4s gray + 1s image + 11s gray and run TRIBE via a worker.",
    level="fine",
    issue_types=["generation"],
    cost_class="expensive",
    scope="sample",
    question="Can a 16-TR surface series be materialized from this image?",
    produces=["generated_fmri", "gray_control"],
    evidence_family="generation",
    valid_claims=["numeric_materialization"],
    unsupported_claims=["biological_validity", "real_brain_response"],
    estimated_gpu_seconds=60.0,
    kind="generator",
)


class StimulusProfile(BaseModel):
    """Verified local 16s protocol. FPS is 10, not 24."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = PROFILE_ID
    fps: int = FPS
    pre_gray_s: float = PRE_GRAY_S
    image_s: float = IMAGE_S
    post_gray_s: float = POST_GRAY_S
    gray_rgb: tuple[int, int, int] = GRAY_RGB
    codec: str = CODEC
    letterbox: bool = False
    fixation: bool = False

    @property
    def onset_s(self) -> float:
        return self.pre_gray_s

    @property
    def offset_s(self) -> float:
        return self.pre_gray_s + self.image_s

    @property
    def total_s(self) -> float:
        return self.pre_gray_s + self.image_s + self.post_gray_s

    @property
    def n_pre(self) -> int:
        return int(round(self.pre_gray_s * self.fps))

    @property
    def n_img(self) -> int:
        return int(round(self.image_s * self.fps))

    @property
    def n_post(self) -> int:
        return int(round(self.post_gray_s * self.fps))

    @property
    def n_frames(self) -> int:
        return self.n_pre + self.n_img + self.n_post

    @property
    def expected_t(self) -> int:
        return int(round(self.total_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.profile_id,
            "generation_profile_id": self.profile_id,
            "fps": self.fps,
            "pre_gray_s": self.pre_gray_s,
            "image_s": self.image_s,
            "post_gray_s": self.post_gray_s,
            "gray_rgb": list(self.gray_rgb),
            "codec": self.codec,
            "letterbox": self.letterbox,
            "fixation": self.fixation,
            "onset_s": self.onset_s,
            "offset_s": self.offset_s,
            "total_s": self.total_s,
            "n_pre": self.n_pre,
            "n_img": self.n_img,
            "n_post": self.n_post,
            "n_frames": self.n_frames,
        }


class ImageRequest(BaseModel):
    """One still image to materialize as 16s video + TRIBE preds."""

    model_config = ConfigDict(extra="forbid")

    image_path: str
    sample_id: str | None = None
    profile: StimulusProfile = Field(default_factory=StimulusProfile)
    include_control: bool = True


class FmriRequest(BaseModel):
    """Already-materialized series; skip generation."""

    model_config = ConfigDict(extra="forbid")

    sample: SampleSpec
    control_path: str | None = None


class GeneratedFmriBundle(BaseModel):
    """On-disk artifacts from one image (+ matching gray control)."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    profile_id: str = PROFILE_ID
    sample: SampleSpec
    preds_path: str
    sidecar_path: str
    video_path: str
    control_sample_id: str | None = None
    control_preds_path: str | None = None
    control_sidecar_path: str | None = None
    control_video_path: str | None = None
    preds_shape: tuple[int, int] = (EXPECTED_T, EXPECTED_V)
    temporal_padding_applied: bool = False
    cache_hit_image: bool = False
    cache_hit_control: bool = False
    new_tribe_predictions: int = 0
    synthetic: bool = False
    backend: Literal["worker", "mock"] = "worker"

    def summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "profile_id": self.profile_id,
            "preds_path": self.preds_path,
            "preds_shape": list(self.preds_shape),
            "temporal_padding_applied": self.temporal_padding_applied,
            "cache_hit_image": self.cache_hit_image,
            "cache_hit_control": self.cache_hit_control,
            "new_tribe_predictions": self.new_tribe_predictions,
            "synthetic": self.synthetic,
            "backend": self.backend,
            "control_preds_path": self.control_preds_path,
            "control_sample_id": self.control_sample_id,
        }


class TemporalContractError(ValueError):
    """Export window did not yield the contracted T points."""

    code = "temporal_contract_error"


class GenerationError(RuntimeError):
    """Video or TRIBE materialization failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
