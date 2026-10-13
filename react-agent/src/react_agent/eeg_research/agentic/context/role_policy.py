"""Central role obligations and the bounded delivery contract."""
from __future__ import annotations

from dataclasses import dataclass
from .types import POLICY_VERSION

CORE_CONTRACT = """INPUT DELIVERY CONTRACT
Current native protocol, permission, legal actions, ledger and exact fact cards
are authoritative. A summary is interpretation, not scientific evidence,
execution approval, artifact content or a completed development read receipt.
FULL/FACT_CARD/PARTIAL/SUMMARY/REF_ONLY state the actual delivery fidelity.
Two-fold screening is not ten-fold confirmation; full training fidelity does
not imply ten-fold coverage. failed/partial/unknown/unverified remain distinct.
Sources are campaign-scoped and hash-bound. Request bounded native reads when
necessary; do not infer missing content or expand references into invented facts.
"""

@dataclass(frozen=True)
class RolePolicy:
    role: str
    mission: str
    required_present_fields: frozenset[str]
    full_read_fields: frozenset[str] = frozenset()
    version: str = POLICY_VERSION


# Present fields are obligations: absent fields do not become fabricated defaults.
CONTROL_FIELDS = frozenset({
    "goal", "contract", "budget", "capabilities", "available_actions", "eligible_targets",
    "blocked_actions", "legal_choice_context", "confirmation_policy", "confirmation_policy_hash",
    "benchmark_permission", "candidate_admission_protocol", "candidate_admissions", "inherited_candidate_admissions",
    "candidate_ready", "candidate_id", "candidate_lifecycle", "implementation_lifecycle", "implementation_target_id",
    "pending_candidate_check", "resource_wait", "repair_task", "execution_fingerprint", "contract_fingerprint",
    "training_target_sources", "action_cost_estimates", "audit_status", "audit_fresh", "completion_audit_ready",
    "require_audit_before_completion", "audited_report_hash", "action_planning", "evaluation_population",
    "evaluation_mode", "evaluation_schedule", "scientific_unit", "science_scope", "screening_evidence_scope",
    "full_required_fold_count", "declared_full_primary_metric", "primary_metric", "state", "authoritative_state",
    "task", "intent", "permissions", "resources", "frontier", "active_tasks", "pending_tasks", "current_feedback",
    "selected_research_task", "origin_plan_id", "selected_plan",
    "candidate_admissions_projection", "inherited_candidate_admissions_projection", "action_cost_estimates_projection",
})
FACT_FIELDS = frozenset({"evidence", "method_screenings", "method_suites", "method_suite", "run", "metrics",
    "diagnostics", "verified_training_diagnostics", "latest_comparison", "latest_diagnostics", "checks",
    "candidate_checks", "development_metrics", "scores", "result", "results", "verified_development_facts",
    "controller_state", "research_progress", "comparison", "evaluation", "facts", "exact_facts"})

MISSIONS = {
    "research_planner": "Choose only current legal actions using active frontier identities, costs, scope and comparisons.",
    "candidate_coder": "Modify the actual requested interface using applicable source regions and exact failure evidence.",
    "candidate_reviewer": "Review implementation, native checks, applicable scope and actual read obligations.",
    "result_analyst": "Explain exact evaluated results and limitations without upgrading coverage or execution state.",
    "experiment_designer": "Design an admissible experiment under the frozen protocol, permission and resource contract.",
    "memory_curator": "Extract conditional source-grounded experience; do not create new measured facts or approval.",
    "result_auditor": "Audit claims against their actual sources, scopes, coverage and read status.",
    "research_librarian": "Return bounded source-grounded retrieval relevant to the current question.",
    "context_summary": "Write bounded source-grounded interpretation patches, preserving failures and unresolved questions.",
    "memory_extraction": "Extract a closed operation episode or no_signal with explicit provenance.",
    "memory_consolidation": "Merge conditional lessons and skills using current support and counterexamples.",
}

SHARED_REQUIRED = frozenset({"goal", "contract", "budget", "permissions", "resources", "state",
    "authoritative_state", "task", "intent", "current_feedback", "candidate_id", "pending_tasks", "active_tasks",
    "execution_fingerprint", "contract_fingerprint", "evaluation_mode", "evaluation_schedule", "science_scope",
    "screening_evidence_scope", "full_required_fold_count", "benchmark_permission", "selected_research_task", "origin_plan_id"})
ROLE_REQUIRED = {
    "research_planner": CONTROL_FIELDS | frozenset({"evidence", "method_screenings", "method_suites", "latest_comparison", "checks", "research_progress", "controller_state"}),
    "candidate_coder": frozenset({"candidate_interface", "implementation_source", "source", "code", "approved_spec", "experiment", "repair_task", "checks", "candidate_checks", "implementation_lifecycle"}),
    "candidate_reviewer": frozenset({"candidate_interface", "implementation_source", "approved_spec", "experiment", "checks", "candidate_checks", "development_metrics", "evidence"}),
    "result_analyst": frozenset({"evidence", "run", "method_suite", "method_screenings", "method_suites", "metrics", "diagnostics", "verified_training_diagnostics", "comparison", "latest_comparison", "scores", "results"}),
    "experiment_designer": frozenset({"candidate_interface", "confirmation_policy", "experiment", "hypothesis", "approved_spec", "candidate_admissions", "evaluation_population"}),
    "memory_curator": frozenset({"source_cards", "closed_groups", "closed_sections", "fact_card", "episode", "run", "evidence", "method_suite"}),
    "result_auditor": frozenset({"evidence", "report", "report_draft", "method_suite", "method_suites", "method_screenings", "metrics", "comparison", "checks", "audit_feedback"}),
    "research_librarian": frozenset({"query", "current_question", "retrieved_memory", "retrieved_context", "search_results"}),
    "context_summary": frozenset({"closed_sections", "source_cards", "previous_snapshot", "facts"}),
    "memory_extraction": frozenset({"closed_sections", "source_cards", "fact_card", "operation"}),
    "memory_consolidation": frozenset({"new_episodes", "changes", "related_memories", "source_cards"}),
}


def role_policy(role: str, *, execution_version: str | None = None) -> RolePolicy:
    if role not in MISSIONS:
        raise KeyError("policy_missing:" + role)
    full = frozenset({"implementation_source", "code", "source"}) if role == "candidate_coder" else frozenset()
    required = SHARED_REQUIRED | ROLE_REQUIRED[role]
    if execution_version == "single_plan_v1" and role == "research_planner":
        from ..experiment_observations import HISTORY_FIELDS
        # Present current facts via exact native cards; full historical trees
        # remain registered/readable rather than compulsory literal payloads.
        required = (required - HISTORY_FIELDS) | frozenset({"experiment_observations", "history_navigation", "context_delivery_version"})
    mission = MISSIONS[role]
    if execution_version == "single_plan_v1" and role == "research_planner":
        mission = "Compare 2–4 bounded research tasks against current observation cards, costs and permissions; select one task to feedback."
    return RolePolicy(role, mission, required, full,
        version=POLICY_VERSION + (".single_plan_v1" if execution_version == "single_plan_v1" else ""))


def prompt_rule_map() -> dict[str, str]:
    return {
        "authority/permission/ledger/legal_action": "native validators + current CONTROL_FIELDS",
        "scientific scope/metrics/coverage": "native exact FACT_FIELDS + CORE_CONTRACT",
        "source identity/applicability/read obligations": "delivery manifest + native artifact/read gate",
        "independent candidate validity": "unchanged INDEPENDENT_OPTION_SCHEMA native per-option validator",
        "role mission": "RolePolicy.mission", "runtime schema/defaults/enums": "unchanged runtime schema; annotation-only provider compaction",
        "unknown current fields": "FULL delivery or NeedsContext; never silently omitted",
    }
