"""Typed contracts for V1.9 campaigns. One module owns these names."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "eeg_research.v1.9"
ROLE_RESULT_VERSION = "eeg_research.role_result.v1"

ROLE_OUTPUT_SCHEMAS = {
    "research_planner": "PlannerDecision: action, target_id, question_id, evidence_refs, decision_rationale, expected_information, required_artifact_refs, hypothesis_draft, plan_update, action_depends_on_plan_update, stop_reason, summary_zh",
    "research_librarian": "MethodEvidencePacket: question, search_scope, sources, method_cards, competing_hypotheses, applicability_limits, knowledge_gaps, local_only, summary_zh",
    "experiment_designer": "ExperimentDesignResult: status, experiment_spec, missing_inputs, required_capability_ids, confounders, required_corrections, evidence_refs, summary_zh",
    "candidate_coder": "One tool request: tool, args. The runtime builds PatchResult.",
    "candidate_reviewer": "ImplementationReview: status, intervention_coverage, requirement_coverage, issues, verified_invariants_with_refs, unverified_invariants, review_limits, summary_zh",
    "result_analyst": "ResultAnalysis: execution_assessment, hypothesis_assessment, observations, interpretations, competing_explanations, evidence_gaps, suggested_next_actions, evidence_refs, summary_zh",
    "memory_curator": "LessonProposal: proposed_lessons with statement, conditions, invalidation_conditions, supporting_episode_ids, contradicting_episode_ids, comparison_refs, observed_effect_refs, requested_evidence_level, uncertainty",
    "result_auditor": "AuditReport: verdict, audited_report_ref, claims, open_issues, required_corrections, review_limits, summary_zh",
}


def role_output_schema(role: str) -> str:
    """Generate the business response schema; tool requests keep their own protocol."""
    model = DOMAIN_OUTPUT_MODELS.get(role)
    if model is not None:
        return json.dumps(model.model_json_schema(), ensure_ascii=False)
    return ROLE_OUTPUT_SCHEMAS.get(role, "role-specific JSON object")


class DomainOutput(BaseModel):
    """No role response grants runtime execution permission."""

    model_config = ConfigDict(extra="forbid")


PlannerAction = Literal["inspect_data", "retrieve_memory", "retrieve_methods", "diagnose_results",
                        "collect_diagnostics", "design_experiment", "propose_experiment",
                        "implement_candidate", "repair_candidate", "run_pilot", "run_full",
                        "replicate", "audit_result", "revise_report", "stop"]


class ArtifactReadRequest(DomainOutput):
    artifact_id: str = Field(min_length=1)
    start: int = Field(default=0, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.end is not None and self.end <= self.start:
            raise ValueError("artifact_read_range_invalid")
        return self


class EstimatedCost(DomainOutput):
    llm_calls: int | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    training_jobs: int | None = Field(default=None, ge=0)


class PlannerOption(DomainOutput):
    option_id: str = Field(min_length=1)
    action: PlannerAction
    target_id: str | None = Field(default=None, description=(
        "Exact runtime candidate/action target ID or null. Artifact IDs belong in read_requests and "
        "required_artifact_refs, never target_id. For retrieve_memory use null unless an existing candidate filter is requested. "
        "Top-level and selected-option target_id must match exactly."))
    question_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    required_read_digests: list[str] = Field(default_factory=list, description=(
        "Optional exact request_digest page identities from artifact_read_index. Select only these completed pages; "
        "legacy required_artifact_refs selects the verified pages for those artifacts. Match selected/top-level lists."))
    required_artifact_refs: list[str] = Field(default_factory=list, description=(
        "Exact verified artifact_index artifact_id values only, never dataset paths or prospective files. "
        "Use [] if no verified registered artifacts are needed. Top-level and selected-option lists must be identical."))
    related_issue_ids: list[str] = Field(default_factory=list)
    observed_gap: str = Field(min_length=1)
    expected_information: str = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    interpretation_of_outcomes: dict[str, str] = Field(default_factory=dict)
    estimated_cost: EstimatedCost = Field(default_factory=EstimatedCost)
    cost_basis: list[str] = Field(default_factory=list, description=(
        "Exact cost_basis strings copied verbatim from action_cost_estimates for this action:target. "
        "Use [] when the runtime list is empty or missing. Never include prose, paths, translations or invented references."))
    cost_confidence: Literal["unknown", "rough", "runtime_bound"] = "unknown"
    value_level: Literal["high", "medium", "low", "unknown"] = "unknown"
    value_rationale: str = Field(min_length=1)
    executable: bool = True
    intervention: str | None = Field(default=None, description=(
        "Exact experiment_draft.intervention for an experiment design option. Use null for stop, "
        "retrieval, report revision and running an already approved candidate. Describe expected changes in value_rationale."))


class ReportClaimRevision(DomainOutput):
    claim_id: str = Field(min_length=1, description="Exact existing report_draft.claims claim_id with an open matching issue. Never invent subclaim IDs.")
    operation: Literal["withdraw", "narrow"]
    statement: str | None = None
    scope_limits: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def narrowing_needs_text(self):
        if self.operation == "narrow" and (not self.statement or not self.scope_limits):
            raise ValueError("claim_narrowing_needs_statement_and_scope")
        return self


class PlannerDecision(DomainOutput):
    action: PlannerAction
    target_id: str | None = Field(default=None, description=(
        "Exact runtime candidate/action target ID or null. Artifact IDs belong in read_requests and "
        "required_artifact_refs, never target_id. For retrieve_memory use null unless an existing candidate filter is requested. "
        "Top-level and selected-option target_id must match exactly."))
    question_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list, description=(
        "Exact evidence IDs matching the selected option evidence_refs. If evidence_ids is supplied, copy this identical list there."))
    evidence_ids: list[str] = Field(default_factory=list, description=(
        "Legacy alias of evidence_refs; use [] or copy the exact same complete list. Never a summary or subset."))
    decision_rationale: str = ""
    reason_zh: str = ""
    summary_zh: str = ""
    expected_information: Any = None
    required_read_digests: list[str] = Field(default_factory=list, description=(
        "Optional exact request_digest page identities from artifact_read_index. Select only these completed pages; "
        "legacy required_artifact_refs selects the verified pages for those artifacts. Match selected/top-level lists."))
    required_artifact_refs: list[str] = Field(default_factory=list, description=(
        "Exact verified artifact_index artifact_id values only, never dataset paths or prospective files. "
        "Use [] if no verified registered artifacts are needed. Top-level and selected-option lists must be identical."))
    hypothesis_draft: dict[str, Any] | None = None
    experiment_draft: dict[str, Any] | None = None
    plan_update: dict[str, Any] | None = None
    action_depends_on_plan_update: bool = False
    stop_reason: Literal["goal_addressed", "no_supported_next_experiment", "no_progress",
                         "budget_exhausted", "blocked", "user_cancelled"] | None = None
    scope_limits: list[str] = Field(default_factory=list)
    observed_gap: str | None = None
    options: list[PlannerOption] = Field(default_factory=list, max_length=4)
    selected_option_id: str | None = None
    selection_rationale: str = ""
    resolves_issue_ids: list[str] = Field(default_factory=list)
    read_requests: list[ArtifactReadRequest] = Field(default_factory=list)
    report_revision: list[ReportClaimRevision] = Field(default_factory=list)


class MethodEvidencePacket(DomainOutput):
    question: Any = None
    search_scope: str = "local_method_cards"
    sources: list[dict[str, Any]] = Field(default_factory=list)
    method_cards: list[dict[str, Any]] = Field(default_factory=list)
    competing_hypotheses: list[Any] = Field(default_factory=list)
    applicability_limits: list[Any] = Field(default_factory=list)
    knowledge_gaps: list[Any] = Field(default_factory=list)
    local_only: Literal[True] = True
    summary_zh: str = ""


class OrderedWindowProbe(DomainOutput):
    """Observe slots at the real representation consumer, after temporal pooling."""

    input_module_path: str = Field(min_length=1, description="Named module whose input is the temporal pool input.")
    representation_module_path: str = Field(min_length=1, description="Named downstream module whose input must still contain all ordered window slots.")
    slot_count: int = Field(ge=2, le=32)
    window_axis: Literal[-1, 2] = -1
    layout: Literal["flatten_channel_major", "flatten_window_major", "channel_window", "window_channel"] = "flatten_channel_major"


class ImplementationRequirement(DomainOutput):
    """An implementation promise, separate from an untested performance prediction."""

    requirement_id: str = Field(min_length=1)
    kind: Literal["implementation", "scientific_hypothesis"] = "implementation"
    core: bool = True
    expected_behavior: str = Field(min_length=1)
    code_locations: list[str] = Field(default_factory=list)
    verification: Literal["source_review", "ordered_window_slots"] = "source_review"
    probe: OrderedWindowProbe | None = None

    @model_validator(mode="after")
    def explicit_verification(self):
        if self.kind == "implementation" and not self.code_locations:
            raise ValueError("implementation_requirement_needs_code_locations")
        if self.verification == "ordered_window_slots":
            if self.kind != "implementation" or self.probe is None:
                raise ValueError("ordered_window_requirement_needs_typed_probe")
        elif self.probe is not None:
            raise ValueError("source_review_requirement_cannot_have_window_probe")
        return self


class RequirementCoverage(DomainOutput):
    requirement_id: str = Field(min_length=1)
    status: Literal["implemented", "contradicted", "unverified"]
    source_hash: str = ""
    source_refs: list[str] = Field(default_factory=list)
    check_ref: str = ""
    check_sha256: str = ""
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def implemented_has_evidence(self):
        if self.status in {"implemented", "contradicted"} and not all((self.source_hash, self.source_refs, self.check_ref, self.check_sha256)):
            raise ValueError("verified_requirement_status_needs_source_and_check_refs")
        return self


class DesignerExperimentSpec(BaseModel):
    """The executable fields required by the deterministic experiment gate."""

    model_config = ConfigDict(extra="allow")
    parent_candidate_id: str = Field(min_length=1)
    control_candidate_id: str = "baseline"
    initial_fidelity: Literal["pilot", "full"]
    hypothesis: Any
    intervention: str = ""
    principal_intervention: str | dict[str, Any] | None = None
    required_capability_ids: list[str] = Field(default_factory=list)
    model: dict[str, Any] = Field(default_factory=dict)
    objective: dict[str, Any] = Field(default_factory=dict)
    transform: dict[str, Any] = Field(default_factory=dict)
    implementation_requirements: list[ImplementationRequirement] = Field(default_factory=list)
    status: Literal["draft"] = "draft"

    @model_validator(mode="after")
    def executable_fields(self):
        if not (self.principal_intervention or self.intervention.strip()):
            raise ValueError("intervention_missing")
        if not self.hypothesis:
            raise ValueError("hypothesis_missing")
        ids = [requirement.requirement_id for requirement in self.implementation_requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_implementation_requirement_id")
        return self


class ExperimentDesignResult(DomainOutput):
    status: Literal["completed", "partial", "failed", "blocked", "requires_framework_extension"]
    experiment_spec: DesignerExperimentSpec | None = None
    missing_inputs: list[str] = Field(default_factory=list)
    required_capability_ids: list[str] = Field(default_factory=list)
    confounders: list[Any] = Field(default_factory=list)
    required_corrections: list[Any] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    summary_zh: str = ""

    @model_validator(mode="after")
    def completed_design_has_spec(self):
        if self.status == "completed" and self.experiment_spec is None:
            raise ValueError("completed_design_requires_executable_spec")
        return self


class ImplementationIssue(BaseModel):
    model_config = ConfigDict(extra="allow")
    severity: Literal["blocking", "non_blocking"]
    category: str = ""
    blocking: bool | None = None

    @model_validator(mode="after")
    def consistent_blocking_flag(self):
        if self.blocking is not None and self.blocking != (self.severity == "blocking"):
            raise ValueError("conflicting_blocking_severity")
        return self


class ImplementationReview(DomainOutput):
    status: Literal["ready", "needs_fix", "blocked"]
    intervention_coverage: Any = None
    requirement_coverage: list[RequirementCoverage] = Field(default_factory=list)
    issues: list[ImplementationIssue] = Field(default_factory=list)
    verified_invariants_with_refs: list[Any] = Field(default_factory=list)
    unverified_invariants: list[Any] = Field(default_factory=list)
    review_limits: Any = None
    summary_zh: str = ""

    @model_validator(mode="after")
    def status_matches_unresolved_issues(self):
        blocking = [issue for issue in self.issues if issue.severity == "blocking"]
        if self.status == "ready" and blocking:
            raise ValueError("ready_review_cannot_have_blocking_issues: resolved allegations belong in verified_invariants_with_refs, not issues")
        if self.status == "needs_fix" and not blocking:
            raise ValueError("needs_fix_requires_an_unresolved_blocking_issue_and_concrete_correction")
        ids = [coverage.requirement_id for coverage in self.requirement_coverage]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_requirement_coverage_id")
        return self


class ResultAnalysis(DomainOutput):
    execution_assessment: Any = None
    hypothesis_assessment: Literal["supported", "weakened", "inconclusive", "not_tested"] | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    prediction_checks: list[dict[str, Any]] = Field(default_factory=list)
    interpretations: list[Any] = Field(default_factory=list)
    competing_explanations: list[Any] = Field(default_factory=list)
    evidence_gaps: list[Any] = Field(default_factory=list)
    suggested_next_actions: list[Any] = Field(default_factory=list)
    scope_limits: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    summary_zh: str = ""


class LessonProposal(DomainOutput):
    proposed_lessons: list[dict[str, Any]] = Field(default_factory=list)
    summary_zh: str = ""


class AuditIssueResolution(DomainOutput):
    issue_id: str = Field(min_length=1)
    resolution_kind: Literal["new_evidence", "claim_withdrawn", "claim_narrowed"]
    claim_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class AuditReport(DomainOutput):
    verdict: Literal["PASS", "REVISE", "BLOCK"]
    audited_report_ref: str = Field(min_length=1)
    report_hash: str = Field(min_length=1)
    dependency_manifest_hash: str = Field(min_length=1)
    resolved_issues: list[AuditIssueResolution] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    open_issues: list[Any] = Field(default_factory=list)
    required_corrections: list[Any] = Field(default_factory=list)
    review_limits: Any = None
    summary_zh: str = ""


DOMAIN_OUTPUT_MODELS = {
    "research_planner": PlannerDecision,
    "research_librarian": MethodEvidencePacket,
    "experiment_designer": ExperimentDesignResult,
    "candidate_reviewer": ImplementationReview,
    "result_analyst": ResultAnalysis,
    "memory_curator": LessonProposal,
    "result_auditor": AuditReport,
}


PLAN_VERSION_NAME = "eeg_research.research_plan.v1"


class GoalSpec(BaseModel):
    """User-submitted research goal. Runtime validates; it does not invent budgets."""

    model_config = ConfigDict(extra="allow")

    goal_id: str
    objective: str = ""
    task_type: Literal["eeg_image_retrieval"] = "eeg_image_retrieval"
    research_scope: str = "pooled_subject_retrieval"
    evaluation_mode: Literal["single_target", "loso_method_search"] = "single_target"
    benchmark_feedback_enabled: bool = False
    subject_aggregation: Literal["none", "macro_mean"] = "none"
    benchmark_subjects: list[str] = Field(default_factory=list)
    required_fold_count: int = Field(default=1, ge=1)
    checkpoint_policy: Literal["legacy_min_delta", "strict_best"] = "legacy_min_delta"
    selection_min_delta: float = Field(default=0.001, ge=0, allow_inf_nan=False)
    method_suite_gpu_seconds_estimate: float | None = Field(default=None, gt=0, allow_inf_nan=False)
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
    require_audit_before_completion: bool = False
    planner_mode: Literal["single_action", "compare_options"] = "compare_options"
    allowed_training_actions: list[Literal["run_pilot", "run_full", "replicate"]] = Field(
        default_factory=lambda: ["run_pilot", "run_full", "replicate"]
    )
    confirmation_target_pairs: int = 3
    training_seeds: list[int] = Field(default_factory=list)
    allowed_changes: list[str] = Field(default_factory=list)
    frozen_components: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def benchmark_mode_contract(self):
        if self.evaluation_mode == "single_target":
            if self.benchmark_feedback_enabled or self.benchmark_subjects or self.required_fold_count != 1 or self.subject_aggregation != "none":
                raise ValueError("benchmark_feedback_requires_loso_method_search")
            return self
        subjects = [f"sub-{index:02d}" for index in range(1, 11)]
        if not self.benchmark_feedback_enabled or self.final_test_enabled:
            raise ValueError("loso_requires_explicit_benchmark_feedback_and_separate_final_protection")
        if sorted(self.benchmark_subjects) != subjects or self.required_fold_count != 10:
            raise ValueError("loso_requires_ten_unique_benchmark_subjects_and_folds")
        if self.subject_aggregation != "macro_mean" or self.primary_metric != "benchmark.loso_mean_fixed_gallery_top1":
            raise ValueError("loso_requires_subject_macro_mean_primary_metric")
        if self.checkpoint_policy != "strict_best":
            raise ValueError("loso_requires_strict_best_checkpoint_policy")
        if self.training_seeds != [0] or self.confirmation_target_pairs != 1:
            raise ValueError("loso_method_runtime_requires_one_seed0_suite_pair")
        if self.max_concurrent_training_jobs != 1 or not 0 <= self.max_candidates <= 2:
            raise ValueError("loso_method_runtime_requires_serial_jobs_and_at_most_two_methods")
        if not 0 <= self.max_repairs_per_candidate <= 2 or "replicate" in self.allowed_training_actions:
            raise ValueError("loso_method_runtime_requires_bounded_repairs_without_replication")
        return self


class EvaluationContract(BaseModel):
    """Frozen evaluator identity. Training seeds are not part of this object."""

    model_config = ConfigDict(extra="allow")

    task_type: str = "eeg_image_retrieval"
    evaluation_mode: Literal["single_target", "loso_method_search"] = "single_target"
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
    gallery = protocol.get("gallery_image_ids") or protocol.get("validation_image_ids")
    if not gallery:
        return False, "gallery_identity_missing"
    if protocol.get("val_mode") != "other_subjects_test" and not protocol.get("validation_image_ids"):
        return False, "validation_identity_missing"
    if protocol.get("schema_version") != SCHEMA_VERSION and not protocol.get("train_query_ids") and not protocol.get(
        "validation_query_ids"
    ):
        return False, "historical_identity_unrebuildable"
    return True, None
