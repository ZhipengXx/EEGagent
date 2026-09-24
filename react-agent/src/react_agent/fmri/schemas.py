"""Contracts for samples, tools, decisions, and reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "fmri_check.v1.1"
CLAIM_SCOPE = "numeric_consistency_only"

DecisionEffect = Literal["none", "flag", "block"]
CalibrationStatus = Literal[
    "none",
    "insufficient",
    "engineering_smoke",
    "inferred",
    "calibrated",
]
QuestionStatus = Literal[
    "untested",
    "described",
    "answered",
    "flagged",
    "unresolved",
    "unassessed",
    "blocked",
    "not_applicable",
]


class StimulusEvent(BaseModel):
    """Optional stimulus timing record from metadata."""

    model_config = ConfigDict(extra="forbid")

    onset_s: float | None = None
    onset_frame: int | None = None
    duration_s: float | None = None
    label: str | None = None


class Provenance(BaseModel):
    """Generator identity for a predicted fMRI sample."""

    model_config = ConfigDict(extra="allow")

    generator: str | None = None
    version: str | None = None
    config_summary: str | None = None


class SampleResources(BaseModel):
    """Optional read-only resource identity. Extra keys are preserved."""

    model_config = ConfigDict(extra="allow")

    control_sample_id: str | None = None
    control_artifact_id: str | None = None
    spatial_layout: dict[str, Any] | None = None
    mesh_asset_id: str | None = None
    atlas_asset_id: str | None = None
    time_alignment_status: Literal["verified", "inferred", "unverified"] | None = None
    t_video: list[float] | None = None
    image_id: str | None = None
    source_image_fingerprint: str | None = None
    scaling_source: str | None = None
    cohort_manifest_id: str | None = None
    reference_artifact_id: str | None = None
    is_gray_control: bool = False


class SampleSpec(BaseModel):
    """Input contract for one pseudo-fMRI sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    fmri_path: str
    array_key: str | None = None
    time_axis: Literal[0, 1]
    sampling_interval_s: float | None = None
    spatial_representation: Literal[
        "surface", "parcel", "volume_flattened", "unknown"
    ] = "unknown"
    space_name: str | None = None
    normalization: str | None = None
    generation_profile_id: str | None = None
    metadata_path: str | None = None
    image_path: str | None = None
    roi_map_path: str | None = None
    stimulus_events: list[StimulusEvent] | None = None
    output_time_origin: str | None = None
    alignment_description: str | None = None
    provenance: Provenance | None = None
    expected_shape: tuple[int, int] | None = None
    expected_n_vertices: int | None = None
    roi_names_path: str | None = None
    resources: SampleResources | None = None

    @model_validator(mode="after")
    def validate_numeric_fields(self) -> SampleSpec:
        """Reject non-positive sampling intervals."""
        if self.sampling_interval_s is not None and self.sampling_interval_s <= 0:
            raise ValueError("sampling_interval_s must be > 0 when provided")
        return self


class Finding(BaseModel):
    """One check finding attached to a tool result."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warning", "error", "flag"]
    message: str
    evidence_refs: list[str] = Field(default_factory=list)
    decision_effect: DecisionEffect = "none"
    calibration_status: CalibrationStatus | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_decision_effect(cls, data: Any) -> Any:
        """Old findings without decision_effect keep a deterministic meaning."""
        if not isinstance(data, dict) or "decision_effect" in data:
            return data
        code = data.get("code")
        severity = data.get("severity")
        if code == "malformed" or severity == "error":
            data["decision_effect"] = "block"
        elif severity == "flag":
            data["decision_effect"] = "flag"
        else:
            data["decision_effect"] = "none"
        return data


class Coverage(BaseModel):
    """What a tool actually inspected."""

    model_config = ConfigDict(extra="forbid")

    description: str
    axes: list[str] = Field(default_factory=list)
    n_time: int | None = None
    n_space: int | None = None


class ArtifactRef(BaseModel):
    """Pointer to a local artifact, never a raw array."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    kind: str = "file"


class CostEstimate(BaseModel):
    """Cheap / medium / expensive class plus optional numeric hints."""

    model_config = ConfigDict(extra="forbid")

    cost_class: Literal["cheap", "medium", "expensive"] = "cheap"
    estimated_seconds: float | None = None
    estimated_tokens: int | None = None
    estimated_usd: float | None = None


class ToolSpec(BaseModel):
    """Static description of a check tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    level: Literal["coarse", "medium", "fine"]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    issue_types: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    cost_class: Literal["cheap", "medium", "expensive"] = "cheap"
    repeatable: bool = False
    enabled: bool = True
    disabled_reason: str | None = None
    scope: Literal["sample", "pair", "cohort"] = "sample"
    question: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    evidence_family: str | None = None
    estimated_cpu_seconds: float | None = None
    estimated_gpu_seconds: float | None = None
    valid_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    allowed_args: dict[str, Any] = Field(default_factory=dict)
    kind: Literal["check", "generator"] = "check"
    stage_hint: Literal["L0", "L1", "L2", "L3"] | None = None
    answers_questions: list[str] = Field(default_factory=list)
    requires_resources: list[str] = Field(default_factory=list)
    requires_evidence: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    can_affect_verdict: bool = False
    cost_hint: Literal["cheap", "medium", "expensive"] | None = None


class Availability(BaseModel):
    """Whether a tool can run on the current sample."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable", "disabled"]
    reason: str | None = None


class ToolResult(BaseModel):
    """Outcome of one tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    result_id: str
    execution_status: Literal["success", "skipped", "error"]
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    coverage: Coverage | None = None
    limitations: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    elapsed_seconds: float | None = None
    gpu_seconds: float | None = None
    cost_estimate: CostEstimate | None = None
    error: str | None = None
    cache_hit: bool = False
    skip_reason: str | None = None
    metric_notes: dict[str, str] = Field(default_factory=dict)
    applicability: Literal["applicable", "partial", "not_applicable"] | None = None
    diagnostic_question: str | None = None
    answer_summary: str | None = None
    calibration_status: CalibrationStatus | None = None
    decision_effect: DecisionEffect | None = None
    resource_fingerprints: dict[str, str] = Field(default_factory=dict)
    signal_provenance: dict[str, Any] | None = None
    execution_key: str | None = None


class Decision(BaseModel):
    """Structured next action from rule or LM policy."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["run_tool", "stop"]
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    question_to_resolve: str | None = None
    expected_observation: str | None = None
    stop_reason: str | None = None
    profile: str | None = None
    source: Literal["rule", "hybrid", "fallback", "planned"] = "rule"

    @model_validator(mode="after")
    def validate_action_fields(self) -> Decision:
        """Require tool_name for run_tool and stop_reason for stop."""
        if self.action == "run_tool" and not self.tool_name:
            raise ValueError("run_tool requires tool_name")
        if self.action == "stop" and not self.stop_reason:
            raise ValueError("stop requires stop_reason")
        return self


class LMUsage(BaseModel):
    """Token and cost ledger row for one LM call."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    requested_model: str
    response_model: str | None = None
    profile: str | None = None
    thinking: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    api_usd: float | None = None
    elapsed_seconds: float | None = None
    success: bool = True
    error: str | None = None
    unconfirmed_charge_possible: bool = False
    call_id: str | None = None
    role: str | None = None
    attempt_index: int | None = None
    status: str | None = None
    finish_reason: str | None = None
    plan_revision: int | None = None
    fallback_used: bool = False
    context_section_sizes: dict[str, int] = Field(default_factory=dict)
    token_count_source: Literal["provider", "estimated"] | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    parent_call_id: str | None = None
    validation_outcome: str | None = None


class QuestionRecord(BaseModel):
    """One diagnostic question in the ledger."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    status: QuestionStatus
    tool_name: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    required: bool = False
    note: str | None = None
    answer_type: Literal["none", "described", "confirmed_normal"] = "none"
    required_by_profile: bool | None = None
    activated_by_trigger_id: str | None = None
    accepted_signal_modes: list[str] = Field(default_factory=list)
    required_metric_keys: list[str] = Field(default_factory=list)
    compatible_protocol: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_reason: str | None = None


class FinalReport(BaseModel):
    """Deterministic machine-readable report for one sample."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    sample_id: str
    safe_sample_id: str
    verdict: Literal[
        "invalid_input",
        "flagged",
        "passed_configured_checks",
        "inconclusive",
    ]
    coverage_status: Literal["complete", "partial"]
    claim_scope: str = CLAIM_SCOPE
    stop_reason: str
    unresolved_questions: list[str] = Field(default_factory=list)
    unavailable_checks: list[str] = Field(default_factory=list)
    evidence_confidence: float | None = None
    recommended_training_weight: float | None = None
    need_score: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    metrics_table: dict[str, Any] = Field(default_factory=dict)
    executed_tools: list[str] = Field(default_factory=list)
    skipped_tools: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    limitations: list[str] = Field(default_factory=list)
    lm_summary: dict[str, Any] | None = None
    cost: dict[str, Any] = Field(default_factory=dict)
    coverage_by_question: dict[str, Any] = Field(default_factory=dict)
    evidence_scopes: list[str] = Field(default_factory=list)
    unassessed_claims: list[str] = Field(default_factory=list)
    candidate_count: int | None = None
    selected_count: int | None = None
    executed_count: int | None = None
    skipped_count: int | None = None
    lm_decision_count: int | None = None
    fallback_count: int | None = None
    non_fallback_lm_successes: int | None = None
    check_profile: str | None = None
    screening_decision: Literal[
        "pass_configured",
        "flagged",
        "blocked",
        "abstain",
    ] | None = None
    screening_coverage: dict[str, Any] = Field(default_factory=dict)
    degraded_execution: bool = False
    degraded_reason: str | None = None
    requested_policy: str | None = None
    resolved_policy: str | None = None
    requested_backend: str | None = None
    resolved_backend: str | None = None
    analysis_goal: str | None = None
    coverage_scope: str | None = None
    coverage_profile: str | None = None
    coverage_by_dimension: dict[str, Any] = Field(default_factory=dict)
    depth_reached: str | None = None
    followups: list[dict[str, Any]] = Field(default_factory=list)
    executions: list[dict[str, Any]] = Field(default_factory=list)
    biological_validity: str | None = None
    upgrade_notes: list[str] = Field(default_factory=list)
