"""Strict Plan / Guard types. Runtime fills ids, hashes, and step status."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PLAN_SCHEMA_VERSION = "plan.v1"
CLAIM_SCOPE = "numeric_consistency_only"

GuardOp = Literal[
    "and",
    "or",
    "not",
    "question_open",
    "resource_ready",
    "step_status_is",
    "finding_present",
    "metric_compare",
]

StepStatus = Literal["pending", "running", "completed", "blocked", "skipped", "failed"]


class Guard(BaseModel):
    """Finite condition DSL. No free-text Python/SQL."""

    model_config = ConfigDict(extra="forbid")

    op: GuardOp
    args: list[Guard] = Field(default_factory=list)
    question_id: str | None = None
    binding_id: str | None = None
    step_id: str | None = None
    status: str | None = None
    finding_code: str | None = None
    metric_path: str | None = None
    comparator: Literal["gt", "ge", "lt", "le", "eq"] | None = None
    threshold_ref: str | None = None


class PlanStep(BaseModel):
    """One planned tool invocation."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    question_id: str
    tool_id: str
    resource_binding_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    when: Guard | None = None
    expected_evidence_type: str | None = None
    brief_reason: str = ""
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    status: StepStatus = "pending"


class MemoryUsage(BaseModel):
    """How one retrieved memory item affected this proposal."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    effect: Literal["ordering", "cost_hint", "reminder", "excluded", "none"] = "none"
    note: str | None = None


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    screening_hint: Literal["pass_configured", "flagged", "blocked", "abstain"] | None = None


class PlanProposal(BaseModel):
    """Model-emitted plan. Runtime overwrites identity fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PLAN_SCHEMA_VERSION
    phase: Literal["create", "revise"] = "create"
    goal: str
    claim_scope: str = CLAIM_SCOPE
    steps: list[PlanStep] = Field(default_factory=list)
    next_step_id: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    resource_gaps: list[str] = Field(default_factory=list)
    stop_request: StopRequest | None = None
    memory_usage: list[MemoryUsage] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_claim_scope(self) -> PlanProposal:
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("claim_scope must stay numeric_consistency_only")
        return self


class Plan(PlanProposal):
    """Runtime-owned plan with identity and status."""

    plan_id: str
    revision: int = 0
    created_at: str
    input_hash: str
    context_hash: str
    source: Literal["deepseek", "mock"] = "mock"


class PlanRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    from_revision: int
    to_revision: int
    trigger: str
    accepted: bool
    reason: str | None = None


class PlanningContext(BaseModel):
    """Inputs assembled for the planner. Arrays are never included."""

    model_config = ConfigDict(extra="forbid")

    task: dict[str, Any]
    evidence: list[dict[str, Any]]
    tool_catalog: list[dict[str, Any]]
    resources: dict[str, Any]
    current_plan: dict[str, Any] | None = None
    memory_bundle: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    request_mode: Literal["create", "revise"] = "create"
    trigger: str | None = None
