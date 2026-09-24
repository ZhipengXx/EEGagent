"""Typed records for one EEG research campaign."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "eeg_research.v1.6"
RESEARCH_SCOPE = "eeg_task_validation"


class TaskCard(BaseModel):
    """Frozen task, splits, and comparison protocol."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    task_id: str
    task_type: Literal["eeg_image_retrieval"] = "eeg_image_retrieval"
    research_scope: Literal["eeg_task_validation"] = RESEARCH_SCOPE
    dataset_id: str
    dataset_version: str
    dataset_manifest_hash: str
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str] = Field(default_factory=list)
    train_split_hash: str = ""
    validation_split_hash: str = ""
    test_split_hash: str = ""
    split_unit: Literal["image", "concept", "subject"] = "image"
    generalization_target: str = "within_subject"
    subject_scope: str = "seen_subjects"
    preprocessing_hash: str
    primary_metric: str = "top1"
    direction: Literal["higher", "lower"] = "higher"
    secondary_metrics: list[str] = Field(default_factory=list)
    metric_implementation_hash: str
    target_feature_hash: str
    candidate_bank_hash: str
    candidate_count: int
    similarity: str
    trial_aggregation: str
    query_unit: str
    tie_policy: Literal["retain_incumbent"] = "retain_incumbent"
    permitted_model_ids: list[str]
    permitted_profile_ids: list[str]
    baseline_ref: str | None = None
    training_fidelity: Literal["full"] = "full"
    seed_schedule: list[int] = Field(default_factory=lambda: [0])
    promotion_policy: str = "single_seed_provisional"
    minimum_delta: float | None = None
    uses_test_for_tuning: bool = False
    provenance_ref: str | None = None
    frozen_task_hash: str = ""


class ModelRecord(BaseModel):
    """One registered EEG encoder. Availability is not inferred from a paper name."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    adapter_id: str
    supported_tasks: list[str]
    input_contract: str
    code_revision: str
    dependency_environment: str
    checkpoint_origin: str | None = None
    approved_profiles: list[str]
    estimated_resources: dict[str, Any] | None = None
    estimate_source: str = "unknown"
    availability: Literal["available", "unavailable", "unverified"]
    reason: str
    last_probe: str | None = None


class ProfileRecord(BaseModel):
    """An approved training profile. Changes outside allowed_changes are rejected."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    fidelity: Literal["full"] = "full"
    allowed_changes: list[str] = Field(default_factory=list)
    factors: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """One testable explanation tied to evidence ids."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    observations: list[dict[str, Any]]
    explanation: str
    alternative_explanations: list[str]
    proposed_change: str
    expected_observable: str
    disconfirmation_condition: str
    comparison_ref: str | None = None
    allowed_profile_id: str | None = None
    estimated_cost: float | None = None
    cost_estimate_source: str = "unknown"
    memory_refs: list[str] = Field(default_factory=list)
    priority_reason: str
    status: Literal["proposed", "selected", "deferred", "pending"] = "proposed"


class ExperimentSpec(BaseModel):
    """One persisted trial. It does not carry a shell string."""

    model_config = ConfigDict(extra="forbid")

    id: str
    campaign_id: str
    hypothesis_id: str
    parent_trial_id: str | None = None
    model_id: str
    profile_id: str
    frozen_task_hash: str
    seed: int
    fidelity: Literal["full"] = "full"
    resolved_config_hash: str
    factor_changed: list[str] = Field(default_factory=list)
    output_dir: str
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = ""


class EvidenceBundle(BaseModel):
    """Evaluator output. Test scores are not stored here."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    execution_status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    protocol_status: Literal["valid", "invalid"] = "valid"
    primary_metric: float | None = None
    metric_name: str = "top1"
    evaluator: str
    train_loss_decreased: bool | None = None
    validation_plateau: bool | None = None
    changed_fields: list[str] = Field(default_factory=list)
    comparison_compatibility: str = "unchecked"
    missing_fields: list[str] = Field(default_factory=list)
    source_artifact_refs: list[str] = Field(default_factory=list)
    wall_seconds: float | None = None
    peak_memory_mb: float | None = None
    error: str | None = None
    checkpoint_ref: str | None = None
    cached_replay: bool = False
    candidate_bank_hash: str = ""
    target_feature_hash: str = ""
    fidelity: Literal["full"] = "full"
    seed: int = 0


class ExperimentDecision(BaseModel):
    """Four separate outcomes. One boolean cannot stand in for them."""

    model_config = ConfigDict(extra="forbid")

    execution_status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    comparison_status: Literal["baseline_established", "comparable", "incomparable", "invalid"]
    selection_status: Literal["provisional_incumbent", "retained", "rejected", "not_compared"]
    hypothesis_status: Literal[
        "supported_provisionally", "contradicted", "inconclusive", "not_tested"
    ]
    delta: float | None = None
    incumbent_id: str | None = None
    evidence_level: Literal["proposed", "observed", "replicated", "external_prior"] = "observed"
    model_selection_assessed: bool = False
    limitations: list[str] = Field(default_factory=list)
