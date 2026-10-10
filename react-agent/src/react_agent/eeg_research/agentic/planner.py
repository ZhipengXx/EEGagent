"""Planner. Availability is computed here, not promised by the prompt."""

from __future__ import annotations

import json
import math

from typing import Any, Callable

ACTIONS = (
    "inspect_data",
    "retrieve_memory",
    "retrieve_methods",
    "diagnose_results",
    "collect_diagnostics",
    "design_experiment",
    "propose_experiment",
    "implement_candidate",
    "repair_candidate",
    "run_pilot",
    "run_full",
    "replicate",
    "audit_result",
    "revise_report",
    "stop",
)

STOP_REASONS = (
    "goal_addressed",
    "no_supported_next_experiment",
    "no_progress",
    "budget_exhausted",
    "blocked",
    "user_cancelled",
)

Backend = Callable[[dict[str, Any]], dict[str, Any]]

# Diagnostics annotate campaign history; memory retains its existing known
# candidate filter/annotation. These local actions consume no model/training
# resources by target. All training/implementation costs need exact bindings.
TARGET_INDEPENDENT_COST_ACTIONS = frozenset({"diagnose_results", "collect_diagnostics", "retrieve_memory"})
TARGETED_ACTIONS = frozenset({"run_pilot", "run_full", "replicate", "implement_candidate", "repair_candidate"})


def known_candidate_ids(observation: dict[str, Any]) -> set[str]:
    """Derive candidate identities, never method names or artifact IDs."""
    return {"baseline", *[row["candidate_id"] for row in observation.get("candidates") or [] if row.get("candidate_id")],
            *[observation[key] for key in ("candidate_id", "implementation_target_id") if observation.get(key)]}


def action_target_choices(
    observation: dict[str, Any], action: str, *, legacy_implicit: bool = False
) -> tuple[list[str], bool]:
    """Derive dispatch-supported candidate pairs without granting execution rights."""
    if action in TARGETED_ACTIONS:
        targets = list((observation.get("eligible_targets") or {}).get(action) or [])
        return targets, legacy_implicit and action in {
            "implement_candidate",
            "repair_candidate",
        }
    if action in TARGET_INDEPENDENT_COST_ACTIONS:
        return sorted(known_candidate_ids(observation)), True
    return [], True


def action_target_error(
    observation: dict[str, Any],
    action: str,
    target: str | None,
    *,
    check_eligible: bool = True,
    legacy_implicit: bool = False,
) -> str | None:
    """Validate the same candidate semantics used by choices and cost binding."""
    if target is not None and target not in known_candidate_ids(observation):
        return "unknown_option_target"
    targets, null_allowed = action_target_choices(
        observation, action, legacy_implicit=legacy_implicit
    )
    if action not in TARGETED_ACTIONS and target is not None and target not in targets:
        return "option_target_requires_null"
    if action in TARGETED_ACTIONS and check_eligible:
        if target is None and null_allowed:
            return None
        if target not in targets:
            return "option_target_unavailable"
    return None


def resolve_cost_estimate(observation: dict[str, Any], action: str, target: str | None) -> tuple[str | None, dict[str, Any]]:
    """Prefer an exact cost row; allow only verified target-independent fallback."""
    rows: dict[str, dict[str, Any]] = observation.get("action_cost_estimates") or {}
    key = f"{action}:{target or ''}"
    if action_target_error(observation, action, target, check_eligible=False):
        return None, {}
    if key in rows:
        return key, rows[key]
    generic = f"{action}:"
    if target is not None and action in TARGET_INDEPENDENT_COST_ACTIONS and generic in rows:
        return generic, rows[generic]
    return None, {}


def legal_choice_context(observation: dict[str, Any]) -> dict[str, Any]:
    """Project legal pairs and cost identities from this same observation."""
    choices = []
    single_action = observation.get("planner_mode", (observation.get("goal") or {}).get("planner_mode", "single_action")) == "single_action"
    for action in observation.get("available_actions") or []:
        targets, null_allowed = action_target_choices(observation, action, legacy_implicit=single_action)
        implicit_target = single_action and action in {"implement_candidate", "repair_candidate"}
        pairs: list[str | None] = [*([None] if null_allowed else []), *targets]
        bindings = [resolve_cost_estimate(observation, action, target)[0] for target in pairs]
        choices.append({"action": action, "targets": targets, "null_allowed": null_allowed,
                        "implicit_target_semantics": "existing controller selects the approved current implementation/repair target" if implicit_target else None,
                        "cost_keys": list(dict.fromkeys(key for key in bindings if key is not None))})
    return {"choices": choices, "candidate_ids": sorted(known_candidate_ids(observation)),
            "identity_rule": "target_id is an exact candidate ID; option_id selects an option; method names and artifact IDs are not candidates.",
            "cost_rule": "Copy the resolved runtime row. Exact pairs win. Only diagnose_results/collect_diagnostics/retrieve_memory may share targetless local costs for known candidates: diagnostics annotate campaign history and memory keeps its existing candidate filter/annotation. Other non-targeted actions use null only.",
            "null_rule": "null means no candidate annotation. Training and compared implementation/repair options require listed targets. Legacy single_action implementation/repair may omit the target and use the controller's approved current target.",
            "lifecycle_source": "implementation_lifecycle and approved experiment are authoritative; missing source does not mean unapproved."}


def _has(evidence: list[dict[str, Any]], candidate_id: Any, fidelity: str) -> bool:
    return any(
        row.get("candidate_id") == candidate_id and row.get("fidelity") == fidelity and row.get("evaluation_valid")
        for row in evidence
    )


_LOCAL = {
    "inspect_data",
    "retrieve_memory",
    "retrieve_methods",
    "diagnose_results",
    "collect_diagnostics",
    "design_experiment",
    "propose_experiment",
    "audit_result",
}
IMPLEMENT_CALLS = 8 + 2 + 1
"""Coder steps, reviewer with one repair, and the planner call that asks for the pilot."""


_DERIVED = {"data_audit", "learning_profile"}


def audit_completed(scope: dict[str, Any]) -> bool:
    """Fresh completion includes an audit with required revisions or blocks."""
    return bool(scope.get("audit_fresh") is True and scope.get("audited_report_hash")
                and scope.get("audit_status") in {"pass", "revise", "block"})


def completion_block_reason(scope: dict[str, Any], reason: Any) -> str | None:
    """An opted-in goal cannot claim completion with an unfinished audit."""
    if reason != "goal_addressed" or not scope.get("require_audit_before_completion"):
        return None
    if not audit_completed(scope):
        return "audit_required_before_goal_addressed"
    return None


def training_actions(scope: dict[str, Any]) -> set[str]:
    """Explicit execution scope. Empty permits no training; missing keeps defaults."""
    declared = scope.get("allowed_training_actions", ["run_pilot", "run_full", "replicate"])
    if not isinstance(declared, list) or any(name not in {"run_pilot", "run_full", "replicate"} for name in declared):
        return set()
    return set(declared)


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def training_block_reason(state: dict[str, Any], budget: dict[str, Any] | None = None) -> str | None:
    """Share canonical feasibility without treating missing values as zero."""
    budget = budget if budget is not None else state.get("budget") or {}
    jobs = _number(budget.get("training_jobs_left", state.get("training_jobs_left")))
    if jobs is None and "training_jobs_left" not in budget and "training_jobs_left" not in state:
        limit, used = _number(state.get("max_training_jobs")), _number(state.get("training_jobs"))
        if limit is not None and used is not None:
            jobs = limit - used
    if jobs is not None and jobs <= 0:
        return "training_job_budget"
    gpu = _number(budget.get("gpu_seconds_left", state.get("gpu_seconds_left")))
    if gpu is not None and gpu <= 0:
        return "gpu_budget"
    if not training_actions(state):
        return "training_scope"
    return None


def design_block_reason(state: dict[str, Any], budget: dict[str, Any] | None = None) -> str | None:
    """Preserve the offline design path for an explicitly training-free scope."""
    candidate_limit = _number(state.get("max_candidates", 4))
    if candidate_limit is not None and submitted_candidates(state) >= candidate_limit:
        return "candidate_budget"
    if state.get("allowed_training_actions") == []:
        return None
    if state.get("evaluation_mode") == "loso_method_search":
        jobs_left = (budget or state.get("budget") or {}).get("training_jobs_left",
                    state.get("max_training_jobs", 0) - state.get("training_jobs", 0))
        if jobs_left < 10:
            return "method_suite_requires_ten_job_slots"
    reason = training_block_reason(state, budget)
    if reason:
        return reason
    if not training_actions(state).intersection({"run_pilot", "run_full"}):
        return "training_scope_no_initial_run"
    return None


def _new_implementation_block_reason(state: dict[str, Any]) -> str | None:
    # Existing source/receipt or a native repair is engineering closeout, even
    # when no future training is possible. Preserve those pending tasks.
    if state.get("repair_task"):
        return None
    target = state.get("_implementation_target") or state.get("candidate_id")
    if any(row.get("candidate_id") == target and row.get("source_hash") for row in state.get("candidates") or []):
        return None
    if any(row.get("candidate_id") == target and row.get("evaluation_valid") for row in state.get("evidence") or []):
        return None
    return design_block_reason(state)


def submitted_candidates(state: dict[str, Any]) -> int:
    """Scientific candidates that produced a patch. Baseline is counted separately."""
    unwritten = {"implementation_failed", "requires_framework_extension"}
    return sum(
        1
        for row in state.get("candidates") or []
        if row.get("candidate_id") != "baseline" and (
            row.get("status") not in unwritten or row.get("source_hash")
            or isinstance(row.get("check"), dict) and row["check"].get("source_sha256")
        )
    )


def evidence_count(state: dict[str, Any]) -> int:
    """Rows that can change a decision. Re-derived summaries do not count."""
    return sum(1 for row in state.get("evidence") or [] if row.get("kind") not in _DERIVED)


def _seen_without_new_evidence(state: dict[str, Any], action: str) -> bool:
    """The same local action at the same substantive evidence count adds nothing."""
    if action in {"design_experiment", "propose_experiment"} and state.get("experiment_failed"):
        failure = state.get("design_failure") or {}
        epoch = [row.get("evidence_id") for row in state.get("evidence") or []
                 if row.get("evaluation_valid") is True and row.get("fidelity") in {"pilot", "full"}]
        # Changed concrete schema/transport feedback is a new design input. It
        # permits at most two follow-up attempts within one scientific epoch.
        if failure.get("recoverable") is True and failure.get("scientific_epoch") == epoch and int(failure.get("attempts_in_epoch") or 0) < 3:
            return False
    if action == "audit_result" and state.get("audit_status") == "stale":
        return False
    if action == "retrieve_memory" and state.get("_has_development_artifacts") and int(state.get("_read_remaining", 0)) > 0:
        return False
    count = evidence_count(state)
    for row in reversed(state.get("decisions") or []):
        if row.get("action") == action and row.get("ok") and row.get("executed") is not False:
            return row.get("evidence_count") == count
    return False


def eligible_targets(state: dict[str, Any]) -> dict[str, list[str]]:
    """Actions bound to one candidate. Readiness of c1 does not authorize c2."""
    evidence = state.get("evidence") or []
    room = training_block_reason(state) is None
    # A frozen baseline is an independent legal measurement, including cold start.
    # It must not require a candidate that in turn requires baseline diagnostics.
    names: list[str] = ["baseline"] if state.get("execution_fingerprint") else []
    if state.get("candidate_ready") and state.get("candidate_id"):
        names.append(str(state["candidate_id"]))
    for row in state.get("candidates") or []:
        if row.get("status") == "ready" and row.get("candidate_id"):
            names.append(str(row["candidate_id"]))
    if "baseline" not in names and any(row.get("candidate_id") == "baseline" for row in evidence):
        names.append("baseline")
    repair = state.get("repair_task") or {}
    implementation_target = state.get("_implementation_target") or repair.get("candidate_id") or state.get("candidate_id")
    targets: dict[str, list[str]] = {"run_pilot": [], "run_full": [], "replicate": [],
        "implement_candidate": [str(implementation_target)] if implementation_target else [],
        "repair_candidate": [str(repair["candidate_id"])] if repair.get("candidate_id") else []}
    if _new_implementation_block_reason(state):
        targets["implement_candidate"] = []
    permitted = training_actions(state)
    if not room:
        return targets
    from react_agent.eeg_research.agentic.confirmation_policy import policy_training_seeds

    policy = state.get("confirmation_policy") if isinstance(state.get("confirmation_policy"), dict) else None
    declared = policy_training_seeds(policy)
    if declared is None:
        declared = [int(item) for item in state.get("training_seeds") or []]
    for name in dict.fromkeys(names):
        if state.get("evaluation_mode") == "loso_method_search":
            if not _has(evidence, name, "full") and state.get("max_training_jobs", 0) - state.get("training_jobs", 0) >= 10:
                if "run_full" in permitted:
                    targets["run_full"].append(name)
            continue
        if not _has(evidence, name, "pilot") and "run_pilot" in permitted:
            targets["run_pilot"].append(name)
        elif _has(evidence, name, "pilot") and not _has(evidence, name, "full") and "run_full" in permitted:
            targets["run_full"].append(name)
        elif _has(evidence, name, "full") and "replicate" in permitted:
            used = {
                int(row.get("seed") or 0)
                for row in evidence
                if row.get("candidate_id") == name and row.get("fidelity") == "full" and row.get("evaluation_valid")
            }
            if declared and any(seed not in used for seed in declared):
                targets["replicate"].append(name)
    return targets


def blocked_actions(state: dict[str, Any]) -> list[dict[str, str]]:
    """Visible gaps. A missing dependency is not hidden as a model decision to stop."""
    from react_agent.eeg_research.agentic.experiment_gate import experiment_block_reason

    gaps: list[dict[str, str]] = []
    reason = experiment_block_reason(state.get("experiment") if isinstance(state.get("experiment"), dict) else None)
    if reason and not state.get("repair_task"):
        gaps.append({"action": "implement_candidate", "reason": reason})
    training_reason = training_block_reason(state)
    if training_reason:
        gaps.extend({"action": action, "reason": training_reason} for action in ("run_pilot", "run_full", "replicate"))
    design_reason = design_block_reason(state)
    if design_reason:
        gaps.extend({"action": action, "reason": design_reason} for action in ("design_experiment", "propose_experiment"))
    if state.get("experiment") and _new_implementation_block_reason(state):
        gaps.append({"action": "implement_candidate", "reason": _new_implementation_block_reason(state)})
    calls = _number(state.get("llm_calls_left"))
    if calls is not None and calls < IMPLEMENT_CALLS and state.get("experiment"):
        gaps.append({"action": "implement_candidate", "reason": "llm_budget"})
    return gaps


def available_actions(state: dict[str, Any]) -> list[str]:
    gpu, calls = _number(state.get("gpu_seconds_left")), _number(state.get("llm_calls_left"))
    if gpu is not None and calls is not None and gpu <= 0 and calls <= 0:
        return ["stop"]
    evidence = state.get("evidence") or []
    actions = ["inspect_data", "retrieve_memory", "retrieve_methods", "audit_result", "stop"]
    pending_experiment = bool(state.get("experiment")) and not state.get("candidate_ready") and not state.get("experiment_failed")
    if not pending_experiment and design_block_reason(state) is None:
        actions.append("propose_experiment")
        actions.append("design_experiment")
    if any(row.get("job_dir") or row.get("fidelity") for row in evidence):
        actions.append("diagnose_results")
        actions.append("collect_diagnostics")
        actions.append("audit_result")
    calls_left = calls if calls is not None else 1_000
    from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved

    approved = experiment_is_approved(state.get("experiment") if isinstance(state.get("experiment"), dict) else None)
    if state.get("repair_task") and int((state.get("repair_task") or {}).get("remaining") or 0) > 0:
        if calls_left >= IMPLEMENT_CALLS:
            actions.append("implement_candidate")
            actions.append("repair_candidate")
    elif (
        pending_experiment
        and approved
        and submitted_candidates(state) < int(state.get("max_candidates", 4))
        and calls_left >= IMPLEMENT_CALLS
        and _new_implementation_block_reason(state) is None
    ):
        actions.append("implement_candidate")
    if state.get("_has_open_report_issues"):
        actions.append("revise_report")
    actions = [name for name in actions if name not in _LOCAL or not _seen_without_new_evidence(state, name)]
    if state.get("_has_development_artifacts") and int(state.get("_read_remaining", 0)) <= 0:
        actions = [name for name in actions if name != "retrieve_memory"]
    if audit_completed(state):
        actions = [name for name in actions if name != "audit_result"]
    targets = eligible_targets(state)
    for name in ("run_pilot", "run_full", "replicate"):
        if targets.get(name):
            actions.append(name)
    return actions


def _hook_execution_status(observation: dict[str, Any]) -> str:
    diagnostics = observation.get("latest_diagnostics")
    items = diagnostics.get("items") if isinstance(diagnostics, dict) else None
    hook = items.get("hook_consumption") if isinstance(items, dict) else None
    if isinstance(hook, dict) and hook.get("execution_status"):
        return str(hook["execution_status"])
    payload = hook.get("payload") if isinstance(hook, dict) else None
    if isinstance(payload, dict) and payload.get("execution_status"):
        return str(payload["execution_status"])
    return "unknown"


def _mechanism_unresolved(observation: dict[str, Any]) -> bool:
    comparison = observation.get("latest_comparison")
    if not isinstance(comparison, dict) or not comparison:
        return True
    if comparison.get("comparable") is not True:
        return True
    delta = comparison.get("delta_pp")
    if delta is None:
        return True
    try:
        return abs(float(delta)) < 1e-12
    except (TypeError, ValueError):
        return True


def route_next_research_action(observation: dict[str, Any]) -> dict[str, Any]:
    """Schema-constrained offline scheduler. This is not live LM reasoning."""
    actions = list(observation.get("available_actions") or [])
    eligible = observation.get("eligible_targets") if isinstance(observation.get("eligible_targets"), dict) else {}
    hook = _hook_execution_status(observation)
    if hook == "not_applied" and "repair_candidate" in actions:
        return {
            "action": "repair_candidate",
            "reason_zh": "批准配置未到达训练 hook，先做工程修复",
            "observed_gap": "hook_not_consumed",
            "evidence_ids": [],
        }
    if hook in {"applied", "unknown"} and _mechanism_unresolved(observation):
        for name in ("replicate", "run_full", "collect_diagnostics", "diagnose_results"):
            if name not in actions:
                continue
            targets = list(eligible.get(name) or [])
            payload = {
                "action": name,
                "reason_zh": "执行有效但机制尚未区分，需要对照、重复或测量，而不是改学习率",
                "observed_gap": "mechanism_unresolved",
                "evidence_ids": [],
            }
            if name in {"replicate", "run_full", "run_pilot"}:
                if not targets:
                    continue
                payload["target_id"] = targets[0]
            return payload
    if "stop" in actions:
        return {
            "action": "stop",
            "stop_reason": "no_supported_next_experiment",
            "reason_zh": "没有值得执行的下一问",
            "evidence_ids": [],
        }
    return {"action": actions[0] if actions else "stop", "reason_zh": "fallback", "evidence_ids": [], "stop_reason": "blocked"}


def duplicate_option_feedback(reply: Any) -> dict[str, Any]:
    """Explain the duplicate executable signatures without adding seed syntax."""
    fields = ("action", "target_id", "question_id", "intervention")
    groups: dict[str, dict[str, Any]] = {}
    options = reply.get("options") if isinstance(reply, dict) else []
    for option in options if isinstance(options, list) else []:
        if not isinstance(option, dict):
            continue
        signature = {key: option.get(key) for key in fields}
        if option.get("action") not in {"design_experiment", "propose_experiment"}:
            signature["intervention"] = None
        key = json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)
        group = groups.setdefault(key, {"execution_signature": signature, "option_ids": []})
        group["option_ids"].append(option.get("option_id"))
    return {
        "duplicate_groups": [group for group in groups.values() if len(group["option_ids"]) > 1],
        "invalid_non_design_interventions": [
            {"option_id": option.get("option_id"), "action": option.get("action"),
             "intervention": option.get("intervention"), "required_intervention": None}
            for option in options if isinstance(option, dict)
            and option.get("action") not in {"design_experiment", "propose_experiment"}
            and option.get("intervention") is not None
        ] if isinstance(options, list) else [],
        "instruction": "Merge each listed group into one option. replicate selects the next unused authorized frozen seed at execution; two replicate options for the same candidate/question with null intervention are identical even if their labels say seed 1 and seed 2. Mention all remaining seeds in one rationale. Different IDs, rationale, costs, future seed labels or invented seed parameters do not create distinct executable actions. Keep the top-level selected action/target/question consistent with the merged option. One option is permitted; compare a distinct legal design or control only when useful.",
        "seed_selection": "runtime_next_unused_authorized_frozen_seed; no planner seed override",
    }


def _correction_view(reply: Any) -> dict[str, Any]:
    """Preserve the full invalid draft; transport may omit it only as a whole."""
    if not isinstance(reply, dict):
        return {"invalid_reply_type": type(reply).__name__}
    import copy
    return copy.deepcopy(reply)


def decide(observation: dict[str, Any], backend: Backend, *, repairs: int = 0) -> dict[str, Any]:
    """At most three attempts under the same immutable observation and gates."""
    allowed = observation.get("available_actions") or []
    known = set(observation.get("_known_evidence_ids") or [])
    trainable = set(observation.get("trainable_ids") or ["baseline"])
    eligible = observation.get("eligible_targets") if isinstance(observation.get("eligible_targets"), dict) else None
    from react_agent.eeg_research.agentic.planner_context import compact_planner_context, normalize_input_echoes
    observation = {**observation, "legal_choice_context": legal_choice_context(observation)}
    previous = observation.get("previous_planner_failure") or {}
    request = (_repair_request(observation, {"detail": previous["detail"]}, previous["raw"])
               if previous.get("raw") and previous.get("detail") else observation)
    request = compact_planner_context(request)
    history = []
    for attempt in range(max(0, min(int(repairs), 2)), 3):
        reply, _ = normalize_input_echoes(backend(request), observation)
        parsed = _validate_reply(reply, observation, allowed, known, trainable, eligible)
        if parsed.get("ok"):
            if attempt:
                parsed["repairs"] = attempt
            return parsed
        history.append({"attempt": attempt + 1, "detail": parsed.get("detail"),
                        "plan_update_error": parsed.get("plan_update_error")})
        if attempt == 2:
            return _blocked_reply(parsed, reply)
        request = _repair_request(observation, parsed, reply)
        request["repair_history"] = list(history)
        request = compact_planner_context(request)
    raise AssertionError("planner_attempt_bound_unreachable")


def _repair_request(observation: dict[str, Any], parsed: dict[str, Any], reply: Any) -> dict[str, Any]:
    known = set(observation.get("_known_evidence_ids") or [])
    repair_payload = {**observation, "schema_error": parsed.get("detail"), "previous": _correction_view(reply)}
    if parsed.get("detail") in {"unknown_option_cost_basis", "option_cost_without_runtime_basis",
                              "option_cost_basis_missing", "option_cost_confidence_without_runtime_basis"}:
        options = reply.get("options") if isinstance(reply, dict) else []
        details = []
        invalid_fields = []
        for option in (options if isinstance(options, list) else [])[:4]:
            if not isinstance(option, dict):
                continue
            key = f"{option.get('action')}:{option.get('target_id') or ''}"
            binding_key, estimate = resolve_cost_estimate(observation, str(option.get("action") or ""), option.get("target_id"))
            expected = estimate.get("estimated_cost") or {}
            submitted = option.get("estimated_cost") or {}
            if isinstance(submitted, dict):
                for field in ("llm_calls", "gpu_seconds", "training_jobs"):
                    value, required = submitted.get(field), expected.get(field)
                    if value is not None and (required is None or (
                            isinstance(value, (int, float)) and abs(value - required) > 1e-6)):
                        invalid_fields.append({
                            "option_id": option.get("option_id"),
                            "field": "estimated_cost." + field,
                            "submitted": value, "required": required,
                            "cost_estimate_key": key,
                            "cost_binding_key": binding_key,
                            "reason": "no_numeric_runtime_basis" if required is None else "runtime_value_mismatch",
                        })
            details.append({"option_id": option.get("option_id"), "cost_estimate_key": key,
                            "cost_binding_key": binding_key,
                            "allowed_cost_basis": estimate.get("cost_basis") or [],
                            "allowed_estimated_cost": estimate.get("estimated_cost") or {
                                "llm_calls": None, "gpu_seconds": None, "training_jobs": None},
                            "cost_confidence": estimate.get("cost_confidence") or "unknown"})
        repair_payload["schema_repair_context"] = {
            "instruction": "Correct every invalid_estimated_cost_fields entry to its required value. Copy the resolved cost_binding_key row: exact targets win; only diagnose_results/collect_diagnostics/retrieve_memory may use generic local costs for known candidates. Null-only actions must correct target_id in both copies, then copy their targetless runtime row. Unsupported numbers, including zero, become null. Copy cost_basis verbatim; unknown basis is []. Explain in value_rationale.",
            "option_costs": details,
            "invalid_estimated_cost_fields": invalid_fields,
        }
    if parsed.get("detail") in {"unknown_option_target", "selected_option_execution_mismatch", "option_target_requires_null"}:
        repair_payload["schema_repair_context"] = {
            "instruction": "Artifact IDs are read_requests/required_artifact_refs, not candidate target IDs. Use legal_choice_context targets/null_allowed. inspect_data, stop and all other null-only actions require target_id=null in BOTH copies. retrieve_memory retains known candidate filtering/annotation. Keep top-level action/target_id/question_id identical to the selected option. Fix every copy together.",
            "eligible_targets": observation.get("eligible_targets") or {},
            "implementation_target_id": observation.get("implementation_target_id"),
        }
    context = repair_payload.setdefault("schema_repair_context", {})
    context["mechanical_errors"] = mechanical_errors(reply, observation)
    context["legal_choice_context"] = legal_choice_context(observation)
    if parsed.get("detail") in {"duplicate_option_action", "option_intervention_requires_design"}:
        context["duplicate_option_feedback"] = duplicate_option_feedback(reply)
    if parsed.get("detail") == "design_hypothesis_missing":
        context["design_handoff_instruction"] = (
            "Put the selected falsifiable hypothesis in hypothesis_draft, including the tentative statement, "
            "verified motivating evidence refs, prediction and competing explanation. Prose in rationale or "
            "expected_information is not delivered as a hypothesis to Designer. Use a cold_start_prior when appropriate; "
            "do not invent effectiveness evidence. The API Designer will canonicalize the executable spec.")
    if str(parsed.get("detail") or "").startswith("premature_stop:"):
        context["continuation_instruction"] = (
            "Authorized resources remain and requested scientific work is unfinished. "
            "Empty run_pilot/run_full/replicate targets mean a fresh design is needed, not exhaustion. "
            "Choose design_experiment or propose_experiment when legal. Approval precedes implementation. "
            "Do not implement a merely eligible candidate unless implement_candidate is also available. "
            "Use the runtime research_progress and verified_development_facts to design a new falsifiable mechanism.")
        context["research_progress"] = observation.get("research_progress")
    context["option_constraints"] = {
        "unique_signature": ["action", "target_id", "question_id", "intervention"],
        "instruction": "Each option must have a different execution signature. Different option IDs or rationales do not make identical actions different. Remove redundant options; one option is valid. Preserve the intended legal selected action when repairing an unrelated field. intervention is an exact experiment_draft.intervention, never report-edit or training prose; use null outside experiment design. Do not repeat an already delivered artifact read.",
    }
    context["report_claims"] = [row.get("claim_id") for row in
        (observation.get("report_draft") or {}).get("claims") or [] if row.get("claim_id")]
    context["report_revision_instruction"] = "Revise only these exact existing claim IDs with matching open issues. Withdraw unsupported phrases within that claim's statement; never create universal_superiority or causal_mechanism subclaim IDs."
    previous_options = reply.get("options") if isinstance(reply, dict) else []
    selected_previous = next((row for row in (previous_options or []) if isinstance(row, dict) and
        row.get("option_id") == reply.get("selected_option_id")), {})
    if parsed.get("detail") == "selected_option_intervention_mismatch":
        design_action = reply.get("action") in {"design_experiment", "propose_experiment"}
        draft = reply.get("experiment_draft") if isinstance(reply.get("experiment_draft"), dict) else {}
        context["selected_intervention_binding"] = {
            "selected_option_id": reply.get("selected_option_id"),
            "action": reply.get("action"),
            "target_id": reply.get("target_id"),
            "previous_intervention": selected_previous.get("intervention"),
            "required_intervention": draft.get("intervention") if design_action else None,
            "instruction": "For run_pilot/run_full/replicate/implement_candidate/repair_candidate of an approved target, set this selected option's intervention to null. This field introduces a NEW experiment_draft intervention, not a description of the existing method. Keep the intended action, target and question unchanged; describe the approved method in value_rationale. Do not create a new experiment_draft just to match training prose. For a design action copy the draft intervention exactly.",
        }
    context["selected_reference_binding"] = {
        "previous_selected_evidence_refs": selected_previous.get("evidence_refs") or [],
        "allowed_evidence_ids": sorted(ref for ref in known if isinstance(ref, str)),
        "allowed_artifact_ids": sorted(row["artifact_id"] for row in observation.get("artifact_index") or []
            if row.get("verification_status") == "verified"),
        "instruction": "Use one identical complete list in top-level evidence_refs, top-level evidence_ids and selected-option evidence_refs. evidence_ids is an alias, never a subset or summary. Only use allowed_evidence_ids. required_artifact_refs in every option and at top level must contain only allowed_artifact_ids, never dataset/file paths or future outputs. An empty allowed_artifact_ids list means required_artifact_refs must be [] everywhere. Keep selected/top-level required_artifact_refs identical. Dataset paths may be described in prerequisites/rationale. Correct all copies in this reply.",
    }
    return repair_payload


def _blocked_reply(parsed: dict[str, Any], reply: Any) -> dict[str, Any]:
    result = {"ok": False, "status": "blocked", "detail": parsed.get("detail"), "raw": reply if isinstance(reply, dict) else None}
    if parsed.get("plan_update_error"):
        result.update(detail="plan_update_blocks_action", plan_update_error=parsed["plan_update_error"])
    return result


def _validate_reply(
    reply: Any,
    observation: dict[str, Any],
    allowed: list[str],
    known: set[str],
    trainable: set[str],
    eligible: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parse(reply, allowed, known, trainable, eligible)
    if not parsed.get("ok"):
        return parsed
    if parsed["action"] in {"design_experiment", "propose_experiment"}:
        goal = observation.get("goal") if isinstance(observation.get("goal"), dict) else {}
        reason = design_block_reason({**goal, **observation}, observation.get("budget"))
        if reason:
            return {"ok": False, "detail": "design_blocked:" + reason}
    implicit = parsed["action"] in {"implement_candidate", "repair_candidate"} and reply.get("target_id") is None and not reply.get("options")
    target_error = action_target_error(observation, parsed["action"], reply.get("target_id"),
        check_eligible=eligible is not None, legacy_implicit=implicit)
    if target_error:
        return {"ok": False, "detail": target_error}
    if parsed["action"] == "design_experiment":
        draft = reply.get("experiment_draft") if isinstance(reply.get("experiment_draft"), dict) else {}
        existing = observation.get("experiment") if isinstance(observation.get("experiment"), dict) else {}
        hypothesis = (reply.get("hypothesis_draft") or draft.get("hypothesis")
                      or observation.get("hypothesis") or existing.get("hypothesis"))
        if not hypothesis:
            return {"ok": False, "detail": "design_hypothesis_missing"}
    detail = validate_comparison_and_reads(reply, observation)
    if detail:
        return {"ok": False, "detail": detail}
    if parsed["action"] == "stop":
        from react_agent.eeg_research.agentic.research_progress import stopping_block_reason
        reason = (completion_block_reason(observation, reply.get("stop_reason"))
                  or stopping_block_reason(observation, reply.get("stop_reason")))
        if reason:
            return {"ok": False, "detail": reason}
    from react_agent.eeg_research.agentic.research_plan import PLAN_DEPENDENT_ACTIONS, PlanError, normalize_plan_update, validate_update

    if reply.get("action_depends_on_plan_update") and parsed["action"] in PLAN_DEPENDENT_ACTIONS:
        try:
            update = normalize_plan_update(reply.get("plan_update"), reply, known_evidence_ids=known)
            plan = (observation.get("research_plan") or {}).get("plan") or {}
            validate_update(plan, update, known_evidence_ids=known)
        except PlanError as exc:
            return {"ok": False, "detail": f"plan_update:{exc}", "plan_update_error": str(exc)}
    return parsed


def _parse(
    reply: Any,
    allowed: list[str],
    known: set[str],
    trainable: set[str] | None = None,
    eligible: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(reply, dict):
        return {"ok": False, "detail": "not_json"}
    action = reply.get("action")
    if not isinstance(action, str):
        return {"ok": False, "detail": "action_not_string"}
    if action not in ACTIONS:
        return {"ok": False, "detail": f"unknown_action:{action}"}
    if allowed and action not in allowed:
        return {"ok": False, "detail": f"action_unavailable:{action}"}
    evidence_ids = reply.get("evidence_ids") if isinstance(reply.get("evidence_ids"), list) and reply.get("evidence_ids") else reply.get("evidence_refs") or []
    if not isinstance(evidence_ids, list) or any(not isinstance(item, str) or item not in known for item in evidence_ids):
        return {"ok": False, "detail": "unknown_evidence"}
    if not reply.get("evidence_ids"):
        reply["evidence_ids"] = list(evidence_ids)
    if action == "propose_experiment" and not isinstance(reply.get("hypothesis_draft"), dict):
        return {"ok": False, "detail": "hypothesis_draft_missing"}
    if action == "stop":
        reason = reply.get("stop_reason")
        if reason not in (None, "") and reason not in STOP_REASONS:
            return {"ok": False, "detail": f"stop_reason_invalid:{reason}"}
    if action in {"run_pilot", "run_full", "replicate"}:
        target = reply.get("target_id")
        if eligible is not None:
            allowed_targets = set(eligible.get(action) or [])
        else:
            allowed_targets = trainable or set()
        if not isinstance(target, str) or not target:
            return {"ok": False, "detail": "target_id_missing"}
        if (eligible is not None or allowed_targets) and target not in allowed_targets:
            return {"ok": False, "detail": f"unknown_target:{target}"}
    return {"ok": True, "action": action, "reason_zh": reply.get("reason_zh") or reply.get("decision_rationale") or reply.get("summary_zh") or "", "raw": reply}


def validate_comparison_and_reads(reply: dict[str, Any], observation: dict[str, Any]) -> str | None:
    """The existing decide/schema-repair path validates every executable option."""
    from pydantic import ValidationError
    from react_agent.eeg_research.agentic.schemas import PlannerDecision
    from react_agent.eeg_research.agentic.handoffs import READ_MAX_IDS, READ_RANGE_CHARS, READ_TOTAL_CHARS, read_digest
    compare = observation.get("planner_mode", (observation.get("goal") or {}).get("planner_mode", "single_action")) == "compare_options"
    extended = compare or any(reply.get(key) for key in ("options", "read_requests", "resolves_issue_ids", "report_revision"))
    if not extended:
        return None
    try:
        parsed = PlannerDecision.model_validate(reply)
    except ValidationError as exc:
        return "planner_domain_invalid:" + str(exc.errors(include_input=False)[0].get("type"))
    feedback = observation.get("audit_feedback") or {}
    issues = {row["issue_id"]: row for row in feedback.get("issues") or []}
    if any(ref not in issues or issues[ref].get("status") != "open" for ref in parsed.resolves_issue_ids):
        return "unknown_or_closed_issue"
    artifacts = {row["artifact_id"]: row for row in observation.get("artifact_index") or [] if row.get("verification_status") == "verified"}
    if any(ref not in artifacts for ref in parsed.required_artifact_refs):
        return "unknown_artifact"
    pages = {row.get("request_digest"): row for row in observation.get("artifact_read_index") or [] if row.get("status") == "read"}
    def valid_pages(digests, refs):
        return all(digest in pages and pages[digest].get("artifact_id") in artifacts
                   and (not refs or pages[digest].get("artifact_id") in refs) for digest in digests)
    if len(set(parsed.required_read_digests)) != len(parsed.required_read_digests) or not valid_pages(parsed.required_read_digests, parsed.required_artifact_refs):
        return "unknown_or_duplicate_artifact_page"
    known = set(observation.get("_known_evidence_ids") or [])
    if parsed.report_revision:
        if parsed.action != "revise_report" or not parsed.resolves_issue_ids:
            return "report_revision_requires_existing_open_issue"
        claims = {row.get("claim_id") for row in (observation.get("report_draft") or {}).get("claims") or []}
        for revision in parsed.report_revision:
            if revision.claim_id not in claims or any(ref not in known for ref in revision.evidence_refs):
                return "report_revision_unknown_claim_or_evidence"
            if not any(issues[ref].get("claim_id") == revision.claim_id for ref in parsed.resolves_issue_ids):
                return "report_revision_issue_claim_mismatch"
    elif parsed.action == "revise_report":
        return "report_revision_missing"
    if parsed.read_requests:
        if parsed.action != "retrieve_memory":
            return "artifact_read_requires_retrieve_memory"
        if len(parsed.read_requests) > min(READ_MAX_IDS, (observation.get("artifact_read_budget") or {}).get("remaining_reads", 0)):
            return "artifact_read_limit"
        digests = []
        total = 0
        for request in parsed.read_requests:
            if request.artifact_id not in artifacts:
                return "unknown_artifact"
            amount = READ_RANGE_CHARS if request.end is None else request.end - request.start
            total += amount
            if amount > READ_RANGE_CHARS or total > READ_TOTAL_CHARS:
                return "artifact_read_range_limit"
            digest = read_digest(request.model_dump(), artifacts[request.artifact_id].get("content_hash"))
            if digest in digests or digest in (observation.get("_artifact_read_digests") or []):
                return "artifact_read_replayed"
            digests.append(digest)
    if compare and (not parsed.options or not parsed.selected_option_id or not parsed.selection_rationale.strip()):
        return "comparison_required"
    if not parsed.options:
        return None
    options = {option.option_id: option for option in parsed.options}
    if len(options) != len(parsed.options):
        return "option_id_duplicate"
    selected = options.get(parsed.selected_option_id)
    if selected is None or not selected.executable:
        return "selected_option_missing_or_nonexecutable"
    if (parsed.action, parsed.target_id, parsed.question_id) != (selected.action, selected.target_id, selected.question_id):
        return "selected_option_execution_mismatch"
    if parsed.evidence_refs and parsed.evidence_ids and set(parsed.evidence_refs) != set(parsed.evidence_ids):
        return "selected_option_evidence_alias_mismatch"
    if set(parsed.evidence_refs or parsed.evidence_ids) != set(selected.evidence_refs):
        return "selected_option_evidence_mismatch"
    if parsed.required_read_digests != selected.required_read_digests:
        return "selected_option_read_page_mismatch"
    if set(parsed.required_artifact_refs) != set(selected.required_artifact_refs):
        return "selected_option_artifact_mismatch"
    if not set(parsed.resolves_issue_ids) <= set(selected.related_issue_ids):
        return "selected_option_issue_mismatch"
    if selected.action not in {"design_experiment", "propose_experiment"} and selected.intervention is not None:
        return "selected_option_intervention_mismatch"
    if any(option.action not in {"design_experiment", "propose_experiment"}
           and option.intervention is not None for option in parsed.options):
        return "option_intervention_requires_design"
    if selected.intervention and (parsed.experiment_draft or {}).get("intervention") != selected.intervention:
        return "selected_option_intervention_mismatch"
    signatures = {(option.action, option.target_id, option.question_id, option.intervention) for option in parsed.options}
    if len(signatures) != len(parsed.options):
        return "duplicate_option_action"
    question_ids = {row.get("question_id") for row in ((observation.get("research_plan") or {}).get("plan") or {}).get("research_questions") or []}
    candidates = {"baseline", *[row.get("candidate_id") for row in observation.get("candidates") or []]}
    if observation.get("candidate_id"):
        candidates.add(observation["candidate_id"])
    if observation.get("implementation_target_id"):
        candidates.add(observation["implementation_target_id"])
    eligible = observation.get("eligible_targets") or {}
    for option in parsed.options:
        if option.question_id is not None and option.question_id not in question_ids:
            return "unknown_option_question"
        if option.target_id is not None and option.target_id not in candidates:
            return "unknown_option_target"
        target_error = action_target_error(observation, option.action, option.target_id, check_eligible=option.executable)
        if target_error:
            return target_error
        if any(ref not in known for ref in option.evidence_refs):
            return "unknown_option_evidence"
        if not valid_pages(option.required_read_digests, option.required_artifact_refs):
            return "unknown_option_artifact_page"
        if any(ref not in artifacts for ref in option.required_artifact_refs):
            return "unknown_option_artifact"
        if any(ref not in issues for ref in option.related_issue_ids):
            return "unknown_option_issue"
        if option.executable:
            if option.action not in (observation.get("available_actions") or []):
                return "option_action_unavailable:" + option.action
            if option.action in {"run_pilot", "run_full", "replicate"} and option.target_id not in eligible.get(option.action, []):
                return "option_target_unavailable"
            if option.action in {"implement_candidate", "repair_candidate"} and option.target_id not in eligible.get(option.action, []):
                return "option_target_unavailable"
        _, estimate = resolve_cost_estimate(observation, option.action, option.target_id)
        expected_cost = estimate.get("estimated_cost") or {}
        for key, value in option.estimated_cost.model_dump().items():
            if value is not None and (expected_cost.get(key) is None or abs(value - expected_cost[key]) > 1e-6):
                return "option_cost_without_runtime_basis"
        if any(ref not in (estimate.get("cost_basis") or []) for ref in option.cost_basis):
            return "unknown_option_cost_basis"
        if any(value is not None for value in option.estimated_cost.model_dump().values()):
            if not option.cost_basis:
                return "option_cost_basis_missing"
            if option.cost_confidence != estimate.get("cost_confidence"):
                return "option_cost_confidence_without_runtime_basis"
    return None


def cost_estimates(state: dict[str, Any], observation: dict[str, Any], *, camp=None) -> dict[str, Any]:
    """Cost of the next mandatory pipeline, with conditional follow-ups explicit.

    Settled same-target/fidelity/protocol records support rough GPU averages.
    Job counts come from the actual matched-baseline rule. Review/repair,
    analysis retries and later confirmation remain separately unknown; these
    estimates never reserve resources or modify the cost ledger.
    """
    import json
    import math
    from pathlib import Path
    from react_agent.eeg_research.agentic.execution_protocol import load_protocol, next_unused_training_seed
    if state.get("evaluation_mode") == "loso_method_search":
        from react_agent.eeg_research.agentic.method_suite import suite_cost_estimates
        return suite_cost_estimates(camp, state, observation)
    estimates = {}
    protocol = load_protocol(camp) if camp else {}
    protocol = protocol or {}
    policy = observation.get("confirmation_policy") or {}
    if policy.get("seeds_declared"):
        protocol = {**protocol, "training_seeds": policy.get("training_seeds") or []}
    fingerprint = state.get("execution_fingerprint")
    evidence = state.get("evidence") or []
    verified_evidence = evidence
    confirmation = {}
    if camp:
        from react_agent.eeg_research.agentic.research_progress import (
            remaining_confirmation_work,
            verified_seed_records,
        )
        verified_evidence = verified_seed_records(camp, state, protocol)
        confirmation = remaining_confirmation_work(camp, state, protocol, policy, records=verified_evidence)

    def samples_for(target, fidelity):
        samples, seen = [], set()
        expected_source = next((row.get("source_hash") for row in state.get("candidates") or []
                                if row.get("candidate_id") == target), None)
        for row in evidence:
            if row.get("candidate_id") != target or row.get("fidelity") != fidelity or row.get("evaluation_valid") is not True:
                continue
            if not fingerprint or (row.get("execution_fingerprint") or row.get("contract_fingerprint")) != fingerprint:
                continue
            if expected_source and row.get("source_hash") != expected_source:
                continue
            directory = Path(str(row.get("job_dir") or "")).resolve()
            if camp and not directory.is_relative_to(Path(camp).resolve()):
                continue
            record = None
            for name in ("run_record.json", "job.json"):
                try:
                    saved = json.loads((directory / name).read_text(encoding="utf-8"))
                    record = saved.get("result", saved)
                    if isinstance(record, dict) and record.get("gpu_seconds") is not None:
                        break
                except (OSError, ValueError, TypeError):
                    continue
            if not isinstance(record, dict):
                continue
            if record.get("candidate_id", target) != target or record.get("fidelity", fidelity) != fidelity:
                continue
            seconds = record.get("gpu_seconds")
            job_id = record.get("job_id") or row.get("job_id") or directory.name
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0 or job_id in seen:
                continue
            seen.add(job_id)
            samples.append({"evidence_id": row.get("evidence_id"), "job_id": job_id, "gpu_seconds": seconds,
                            "candidate_id": target, "fidelity": fidelity, "execution_fingerprint": fingerprint})
        return samples[-8:]

    for action in observation.get("available_actions") or []:
        candidate_targets, null_allowed = action_target_choices(observation, action,
            legacy_implicit=observation.get("planner_mode", (observation.get("goal") or {}).get("planner_mode", "single_action")) == "single_action")
        targets: list[str | None] = [*([None] if null_allowed else []), *candidate_targets]
        for target in targets:
            costs = {"llm_calls": None, "gpu_seconds": None, "training_jobs": None}
            components, basis, samples, unknown = {}, [], [], []
            if action in {"run_pilot", "run_full", "replicate"}:
                fidelity = "pilot" if action == "run_pilot" else "full"
                seed = int(protocol.get("training_seed", protocol.get("seed") or 0))
                if action == "replicate":
                    used = {int(row.get("seed") or 0) for row in evidence if row.get("candidate_id") == target
                            and row.get("fidelity") == fidelity and row.get("evaluation_valid")}
                    seed = next_unused_training_seed(protocol, used)
                matched = target == "baseline" or any(row.get("candidate_id") == "baseline" and row.get("fidelity") == fidelity
                    and row.get("evaluation_valid") is True and int(row.get("seed") or 0) == seed
                    and (row.get("execution_fingerprint") or row.get("contract_fingerprint")) == fingerprint for row in verified_evidence)
                if camp and target != "baseline":
                    from react_agent.eeg_research.agentic.comparison import (
                        compare_runs,
                        matched_control,
                    )
                    from react_agent.eeg_research.agentic.run_context import (
                        load_approved_binding,
                    )
                    binding = load_approved_binding(camp, target) or {}
                    prospective = {"candidate_id": target, "fidelity": fidelity, "seed": seed,
                        "execution_fingerprint": fingerprint, "evaluation_valid": True,
                        "negative_sampling_policy": binding.get("negative_sampling_policy") or protocol.get("negative_sampling_policy")}
                    control = matched_control(verified_evidence, prospective, protocol)
                    matched = compare_runs(candidate=prospective, control=control, protocol=protocol)["comparable"]
                required_targets = [target] + ([] if matched else ["baseline"])
                totals = []
                for name in required_targets:
                    history = samples_for(name, fidelity)
                    samples.extend(history)
                    seconds = sum(row["gpu_seconds"] for row in history) / len(history) if len(history) >= 2 else None
                    components["candidate" if name == target else "matched_baseline"] = {
                        "candidate_id": name, "fidelity": fidelity, "training_seed": seed,
                        "gpu_seconds": seconds, "training_jobs": 1 if seed is not None else None,
                        "cost_confidence": "rough" if seconds is not None else "unknown",
                        "cost_basis": [row["evidence_id"] for row in history if row.get("evidence_id")]}
                    totals.append(seconds)
                costs["gpu_seconds"] = sum(totals) if totals and all(value is not None for value in totals) else None
                costs["training_jobs"] = len(required_targets) if seed is not None else None
                basis = [row["evidence_id"] for row in samples if row.get("evidence_id")]
                basis.append("runtime:matched_training_seed_rule")
                components["review_or_repair"] = {"llm_calls": None, "condition": "only_if_source_or_hooks_fail"}
                components["analysis"] = {"llm_calls": None, "condition": "result_analysis_and_curator_with_paid_retries"}
                components["followup_confirmation"] = {
                    "llm_calls": None, "gpu_seconds": None, "training_jobs": None,
                    "target_pairs": policy.get("target_pairs"), "training_seeds": policy.get("training_seeds"),
                    "policy_hash": policy.get("policy_hash"), "condition": "depends_on_observed_result_and_remaining_pairs"}
                components["followup_confirmation"]["declared_full_seed_work"] = next(
                    (row for row in confirmation.get("candidates", []) if row["candidate_id"] == target), None)
                unknown = ["review_or_repair_calls", "analysis_retries", "conditional_followup_confirmation"]
                if costs["gpu_seconds"] is None:
                    unknown.append("comparable_gpu_history_missing")
            elif action in {"inspect_data", "retrieve_memory", "collect_diagnostics", "diagnose_results", "revise_report", "stop"}:
                costs = {"llm_calls": 0, "gpu_seconds": 0.0, "training_jobs": 0}
                basis = ["runtime:local_action:" + action]
                components["local_consumer"] = {"model_calls": False, "training_jobs": 0}
            else:
                unknown = ["model_retries", "review_or_repair", "future_experiments_and_confirmation"]
            estimates[f"{action}:{target or ''}"] = {
                "estimated_cost": costs, "cost_basis": list(dict.fromkeys(basis)),
                "cost_confidence": "rough" if costs["gpu_seconds"] and samples else "runtime_bound" if costs["gpu_seconds"] == 0 else "unknown",
                "historical_samples": samples, "components": components, "unknown_components": unknown,
                "cost_scope": "next_action_and_required_matched_baseline; conditional_followups_listed_separately",
                "full_followup_pipeline_cost": {"llm_calls": None, "gpu_seconds": None, "training_jobs": None},
                "api_usd": None, "usd_status": "unpriced", "estimates_are_execution_authority": False,
                "planner_call_already_in_budget_ledger": True,
            }
    return estimates


def mechanical_errors(reply: Any, observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Report bounded simultaneous mechanical errors under the existing schema."""
    from pydantic import ValidationError

    from react_agent.eeg_research.agentic.schemas import PlannerDecision

    try:
        decision = PlannerDecision.model_validate(reply)
    except ValidationError as exc:
        return [{"field": list(row["loc"]), "reason": row["type"], "detail": row["msg"]}
                for row in exc.errors(include_input=False)[:12]]
    errors: list[dict[str, Any]] = []

    def add(field: str, current: Any, required: Any, reason: str) -> None:
        errors.append({"field": field, "current": current, "required": required, "reason": reason})

    selected = next((option for option in decision.options if option.option_id == decision.selected_option_id), None)
    if selected is None and decision.options:
        add("selected_option_id", decision.selected_option_id, [row.option_id for row in decision.options], "select_an_existing_executable_option")
    if selected is not None:
        for key in ("action", "target_id", "question_id"):
            if getattr(decision, key) != getattr(selected, key):
                add(key, getattr(decision, key), getattr(selected, key), "synchronize_with_a_legal_selected_option")
        for key, expected in (("evidence_refs", selected.evidence_refs), ("required_artifact_refs", selected.required_artifact_refs), ("required_read_digests", selected.required_read_digests)):
            actual = getattr(decision, key)
            if key == "evidence_refs":
                actual = actual or decision.evidence_ids
            if (actual != expected if key == "required_read_digests" else set(actual) != set(expected)):
                add(key, actual, expected, "synchronize_selected_references")
        if decision.evidence_refs and decision.evidence_ids and set(decision.evidence_refs) != set(decision.evidence_ids):
            add("evidence_ids", decision.evidence_ids, decision.evidence_refs, "evidence_alias_mismatch")
        expected_intervention = (decision.experiment_draft or {}).get("intervention") if selected.action in {"design_experiment", "propose_experiment"} else None
        if selected.intervention is not None and expected_intervention != selected.intervention:
            add("options.selected.intervention", selected.intervention, expected_intervention, "copy_exact_design_intervention_or_null_for_non_design")
    top_error = action_target_error(observation, decision.action, decision.target_id,
        legacy_implicit=not decision.options and decision.target_id is None)
    if top_error:
        targets, null_allowed = action_target_choices(observation, decision.action)
        add("target_id", decision.target_id, [*([None] if null_allowed else []), *targets], top_error)
    for index, option in enumerate(decision.options):
        prefix = f"options[{index}]"
        if option.executable and option.action not in (observation.get("available_actions") or []):
            add(prefix + ".action", option.action, observation.get("available_actions") or [], "action_unavailable")
        target_error = action_target_error(observation, option.action, option.target_id, check_eligible=option.executable)
        if target_error:
            targets, null_allowed = action_target_choices(observation, option.action)
            add(prefix + ".target_id", option.target_id, [*([None] if null_allowed else []), *targets], target_error)
        binding, estimate = resolve_cost_estimate(observation, option.action, option.target_id)
        for key, value in option.estimated_cost.model_dump().items():
            expected_cost = (estimate.get("estimated_cost") or {}).get(key)
            if value is not None and (expected_cost is None or abs(value - expected_cost) > 1e-6):
                add(prefix + ".estimated_cost." + key, value, expected_cost, "cost_binding:" + str(binding))
        if any(ref not in (estimate.get("cost_basis") or []) for ref in option.cost_basis):
            add(prefix + ".cost_basis", option.cost_basis, estimate.get("cost_basis") or [], "copy_runtime_basis")
        numeric = any(value is not None for value in option.estimated_cost.model_dump().values())
        if numeric and not option.cost_basis:
            add(prefix + ".cost_basis", [], estimate.get("cost_basis") or [], "numeric_cost_requires_basis")
        if numeric and option.cost_confidence != estimate.get("cost_confidence"):
            add(prefix + ".cost_confidence", option.cost_confidence, estimate.get("cost_confidence") or "unknown", "copy_runtime_confidence")
        if option.action not in {"design_experiment", "propose_experiment"} and option.intervention is not None:
            add(prefix + ".intervention", option.intervention, None, "non_design_intervention_is_null")
    return errors[:32]
