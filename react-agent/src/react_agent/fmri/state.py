"""LangGraph-serializable inspection state. Lists are replaced, not appended."""

from __future__ import annotations

from typing import Any, TypedDict


class CheckState(TypedDict, total=False):
    """Serializable fMRI check state.

    Nodes return only changed keys. List fields are full replacements so
    LangGraph does not duplicate history.
    """

    run_id: str
    sample_spec: dict[str, Any]
    base_dir: str
    out_dir: str
    config: dict[str, Any]
    backend: str
    policy: str
    sample_id: str
    safe_sample_id: str
    fingerprint: str
    array_ref: dict[str, Any]
    inspect_info: dict[str, Any]
    tool_results: list[dict[str, Any]]
    completed_tools: list[str]
    failed_tools: list[str]
    unavailable_tools: list[str]
    skipped_tools: list[str]
    candidate_tools: list[str]
    issue_types: list[str]
    unresolved_questions: list[str]
    need_score: dict[str, Any]
    tool_call_count: int
    lm_call_count: int
    round_count: int
    budget: dict[str, Any]
    decision_history: list[dict[str, Any]]
    pending_decision: dict[str, Any] | None
    stop_reason: str
    invalid_input: bool
    fallback_used: bool
    force_rule: bool
    profile_upgraded: bool
    no_progress_count: int
    last_evidence_signature: str
    cache: dict[str, Any]
    final_report: dict[str, Any]
    lm_usages: list[dict[str, Any]]
    events_path: str
    error: str
    route: str
    question_ledger: dict[str, Any]
    required_unanswered: list[str]
    current_plan: dict[str, Any] | None
    memory_retrieval: dict[str, Any]
    planning_degraded: bool
    replan_count: int
    max_replans: int
    active_step_id: str | None
    screening_decision: str | None
