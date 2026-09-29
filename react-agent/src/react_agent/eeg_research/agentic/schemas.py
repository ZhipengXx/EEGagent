"""Typed contracts for V1.9 campaigns. One module owns these names."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "eeg_research.v1.9"
ROLE_RESULT_VERSION = "eeg_research.role_result.v1"
PLAN_VERSION_NAME = "eeg_research.research_plan.v1"


class GoalSpec(BaseModel):
    """User-submitted research goal. Runtime validates; it does not invent budgets."""

    model_config = ConfigDict(extra="allow")

    goal_id: str
    objective: str = ""
    task_type: Literal["eeg_image_retrieval"] = "eeg_image_retrieval"
    research_scope: str = "pooled_subject_retrieval"
    primary_metric: str = "validation.fixed_gallery_top1"
    direction: Literal["maximize", "minimize"] = "maximize"
    min_practical_gain_pp: float | None = None
    final_test_enabled: bool = False
    max_candidates: int = 4
    max_training_jobs: int
    max_llm_calls: int
    max_gpu_seconds: float
    max_concurrent_training_jobs: int = 1
    max_api_usd: float | None = None
    max_repairs_per_candidate: int = 2
    confirmation_target_pairs: int = 3
    training_seeds: list[int] = Field(default_factory=list)
    allowed_changes: list[str] = Field(default_factory=list)
    frozen_components: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


class EvaluationContract(BaseModel):
    """Frozen evaluator identity. Training seeds are not part of this object."""

    model_config = ConfigDict(extra="allow")

    task_type: str = "eeg_image_retrieval"
    data_version: str = ""
    split_seed: int
    train_query_ids: list[str] = Field(default_factory=list)
    validation_query_ids: list[str] = Field(default_factory=list)
    gallery_image_ids: list[str] = Field(default_factory=list)
    positive_map: dict[str, str] = Field(default_factory=dict)
    input_geometry: dict[str, Any] = Field(default_factory=dict)
    preprocessing_version: str = ""
    image_cache_hash: str = ""
    evaluator_hash: str = ""
    fingerprint: str = ""
    comparable: bool = True
    incomparable_reason: str | None = None


class TrainingRecipe(BaseModel):
    """Approved training recipe. Does not include the evaluation split."""

    model_config = ConfigDict(extra="allow")

    optimizer: str = "adamw"
    batch_size: int
    lr: float
    weight_decay: float
    global_batch: int | None = None
    negative_sampling_policy: str = "data_parallel_local"
    early_stop: str = "single_early"
    precision: str = "fp32"
    full_epochs: int


class RunSpec(BaseModel):
    """One scheduled job. Scheduler is the authority."""

    model_config = ConfigDict(extra="allow")

    candidate_id: str
    source_hash: str = ""
    config_hash: str = ""
    evaluation_hash: str = ""
    recipe_hash: str = ""
    fidelity: Literal["pilot", "full"]
    training_seed: int
    split_seed: int
    deadline_at: float | None = None
    gpu: list[int] = Field(default_factory=list)
    data_root: str = ""
    command: list[str] = Field(default_factory=list)
    negative_sampling_policy: str = "data_parallel_local"


class CandidateSpec(BaseModel):
    """One immutable candidate identity. parent_id is lineage, not the experimental control."""

    model_config = ConfigDict(extra="allow")

    candidate_id: str
    parent_id: str = "baseline"
    parent_source_hash: str = ""
    hypothesis_id: str | None = None
    intervention: str = ""
    attempt_id: str = ""
    control_id: str | None = None


class Comparison(BaseModel):
    """Deterministic comparison. Missing matched control is not a completed comparison."""

    model_config = ConfigDict(extra="allow")

    control_run_id: str | None = None
    candidate_run_id: str
    comparable: bool
    same_seed_delta: float | None = None
    fidelity: str = "full"
    training_seed: int | None = None
    evidence_level: str = "observed"
    difference_sources: list[str] = Field(default_factory=list)
    reason: str = ""


class RoleResult(BaseModel):
    """One role attempt. Recovery requires matching task, attempt, and digest."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = ROLE_RESULT_VERSION
    task_id: str
    attempt_id: str
    input_digest: str
    prompt_hash: str = ""
    status: str
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    summary_zh: str = ""


class ExperimentSpec(BaseModel):
    """Executable experiment. Designer writes this; it is not a new scientific candidate by itself."""

    model_config = ConfigDict(extra="allow")

    parent_candidate_id: str = "baseline"
    intervention: str = ""
    initial_fidelity: Literal["pilot", "full"] = "pilot"
    required_capability_ids: list[str] = Field(default_factory=list)
    seed_policy: str = "unused_training_seed"
    confirmation_target_pairs: int | None = None


def comparison_key(protocol: dict[str, Any], fidelity: str) -> str:
    """Identity that must match before two full runs can be compared."""
    return "|".join(
        [
            str(protocol.get("fingerprint") or ""),
            str(fidelity),
            str(protocol.get("negative_sampling_policy") or "data_parallel_local"),
            str(protocol.get("primary_metric") or "validation.fixed_gallery_top1"),
        ]
    )


def protocol_is_comparable(protocol: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Historical protocols without frozen query identity cannot enter formal comparison."""
    if not isinstance(protocol, dict):
        return False, "protocol_missing"
    if not protocol.get("split_seed") and protocol.get("split_seed") != 0:
        return False, "split_seed_missing"
    if protocol.get("val_mode") != "other_subjects_test" and not protocol.get("validation_image_ids"):
        return False, "validation_identity_missing"
    if protocol.get("schema_version") != SCHEMA_VERSION and not protocol.get("train_query_ids") and not protocol.get(
        "validation_query_ids"
    ):
        return False, "historical_identity_unrebuildable"
    return True, None
