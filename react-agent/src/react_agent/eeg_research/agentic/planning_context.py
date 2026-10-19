"""Scientific inputs for proposal generation and independent comparison.

These are delivery views, never native observations or feasibility proofs.
Selection/validation/recovery continue to use the unchanged native objects.
No summarizer call, text clipping, or second memory retrieval is performed here.
"""
from __future__ import annotations

import copy
import math
import re

VERSION = "research_planning_input.v1"
VERSION_V2 = "research_planning_input.v2"

GOAL_FIELDS = (
    "objective", "task_type", "dataset", "research_scope", "evaluation_mode",
    "evaluation_schedule", "primary_metric", "direction", "allowed_changes",
    "frozen_components", "allowed_training_actions", "required_fold_count",
    "training_seeds", "checkpoint_policy", "subject_aggregation",
    "min_practical_gain_pp", "selection_min_delta", "benchmark_subjects",
    "benchmark_feedback_enabled", "final_test_enabled",
)
CONTRACT_FIELDS = (
    "task_type", "dataset", "label", "research_scope", "evaluation_mode",
    "generalization_target", "primary_metric", "direction", "required_fold_count",
    "subject_aggregation", "val_mode", "input_shape", "embedding_dim",
    "benchmark_scope", "independent_final_test", "final_test_enabled",
    "within_batch_role", "legacy_unverified_note",
)
INTERFACE_FIELDS = (
    "input", "output", "dataset", "c_num", "channels", "timesteps", "gallery_size",
    "build_encoder", "build_training_objective", "build_training_transform",
    "fit_statistics", "not_available", "runtime_guarantees",
)
TASK_ACTIONS = frozenset({"design_experiment", "propose_experiment", "run_screening",
    "run_confirmation", "run_full", "run_pilot", "collect_diagnostics", "diagnose_results"})


def _pick(value, fields):
    return {key: copy.deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def _scientific_data(value):
    """Remove storage/receipt metadata from otherwise scientific structured data."""
    if isinstance(value, list):
        # Applicability rows encode metadata names as values, rather than keys.
        # Keep differences in scientific conditions, not byte-identity checks.
        return [_scientific_data(item) for item in value if not (isinstance(item, dict)
            and isinstance(item.get("field"), str)
            and (item["field"].endswith(("_hash", "_digest", "_fingerprint"))
                 or item["field"] == "method_revision"))]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    bookkeeping = {"schema_version", "validation_records", "check", "checks",
        "cpu_check_receipt", "approval_record", "operation_binding", "coding_recovery",
        "next_read", "next_reads", "read_capability", "read_unit", "json_pointer",
        "satisfies_development_read_receipt", "verification_artifact"}
    return {key: _scientific_data(child) for key, child in value.items()
        if key not in bookkeeping and not key.startswith("_")
        and not key.endswith(("_hash", "_sha256", "_digest", "_fingerprint"))}


def _expanded(view):
    """Legacy paid deliveries may use request-local exact deduplication pointers."""
    from .context.representations import resolve_request_pointer
    from .context.sources import pointer_escape

    def visit(value, path=""):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                target = value["$context_pointer"]
                return visit(resolve_request_pointer(view, path), target)
            return {key: visit(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, path + "/" + str(index)) for index, child in enumerate(value)]
        return copy.deepcopy(value)
    return visit(view)


def _source(value):
    result = _pick(value, ("candidate_id", "intervention", "hypothesis", "source", "source_ref",
        "source_truncated", "missing_inputs", "model", "objective", "transform", "recipe",
        "frozen_training_policy", "negative_sampling_policy"))
    if isinstance(value, dict) and isinstance(value.get("encoder_reference"), dict):
        result["encoder_reference"] = _pick(value["encoder_reference"], ("excerpt", "source_ref"))
    return result


def _observation_cards(value, *, cleaner=None, compact=False):
    cleaner = cleaner or _scientific_data
    cards = value.get("cards", []) if isinstance(value, dict) else value or []
    result = []
    for card in cards:
        body = card.get("body", card)
        if compact:
            from .experiment_observations import observation_science_view
            body = observation_science_view(body)
        row = _pick(body, ("observation_id", "observation_kind", "question", "intervention",
            "execution_status", "evaluation_scope", "native_facts", "analyst_interpretation",
            "unresolved_questions", "applicability"))
        if isinstance(row.get("analyst_interpretation"), dict):
            # Same flat shape the stored-science refill uses, so a delivered summary appears once.
            row["analyst_interpretation"] = _analysis(row["analyst_interpretation"])
        identity = _pick(body.get("identity") or {}, ("candidate_id", "dataset", "modality", "method_revision"))
        if identity:
            row["identity"] = identity
        failure = _pick(body.get("engineering_failure") or {},
            ("error", "error_category", "reason", "summary"))
        issues = (body.get("engineering_failure") or {}).get("blocking_issues") or []
        if issues:
            failure["blocking_issues"] = [summary for issue in issues
                if (summary := _pick(issue, ("issue_id", "requirement_id", "kind", "error_class",
                    "detail", "expected", "coverage_detail", "summary", "reason", "error")))]
        if failure:
            row["engineering_failure"] = failure
        for key in ("current_feedback", "fidelity", "condition_differences"):
            if key in card:
                row[key] = copy.deepcopy(card[key])
        # Keep each source identifier once for attribution, not its read protocol.
        from .context.representations import delivered_source_refs
        refs = sorted(delivered_source_refs(card))
        if refs:
            row["source_refs"] = [{"source_ref": ref} for ref in refs]
        result.append(cleaner(row))
    return result


# Run bookkeeping inside applicability rows; scientific conditions remain.
APPLICABILITY_BOOKKEEPING = frozenset({"campaign_id", "task", "evaluation_identity", "terminal_observation_id"})


def _applicability(row, geometry):
    applicability = row.get("applicability")
    if not isinstance(applicability, dict):
        return
    for key in APPLICABILITY_BOOKKEEPING:
        applicability.pop(key, None)
    if geometry is not None and applicability.get("input_geometry") == geometry:
        applicability.pop("input_geometry")
    recipe = applicability.get("training_recipe")
    if isinstance(recipe, dict):
        recipe.pop("data_root", None)
        recipe.pop("gpu", None)
        if geometry is not None and recipe.get("input_geometry") == geometry:
            recipe.pop("input_geometry")


def _analysis(value):
    points = (value or {}).get("points") if isinstance(value, dict) else None
    points = points if isinstance(points, dict) else value
    return _pick(points, ("hypothesis_assessment", "summary_zh", "summary"))


def _stored_science(body):
    """Fields the bounded observation card moves behind reads; no read tool exists here."""
    result = {}
    analysis = _analysis(body.get("analyst_interpretation"))
    if analysis:
        result["analyst_interpretation"] = analysis
    if body.get("unresolved_questions"):
        result["unresolved_questions"] = copy.deepcopy(body["unresolved_questions"])
    failure = body.get("engineering_failure")
    if isinstance(failure, dict) and failure.get("error"):
        result["engineering_failure"] = {"error": copy.deepcopy(failure["error"])}
    return result


def _history_row(body, *, cleaner=None):
    cleaner = cleaner or _scientific_data
    facts = _pick(body.get("native_facts") or {}, ("phase", "metrics", "paired_scores", "paired_delta_pp",
        "promotion", "missing", "required_fold_count", "full_required_fold_count"))
    identity = body.get("identity") or {}
    row = {"observation_id": body.get("observation_id"), "observation_kind": body.get("observation_kind"),
        "candidate_id": identity.get("candidate_id") or (body.get("applicability") or {}).get("candidate_id"),
        "execution_status": body.get("execution_status"), "question": body.get("question"),
        "intervention": body.get("intervention"), "evaluation_scope": copy.deepcopy(body.get("evaluation_scope")),
        "native_facts": facts}
    row.update(_stored_science(body))
    if isinstance(body.get("engineering_failure"), dict):
        row.setdefault("engineering_failure", {})["error_category"] = body["engineering_failure"].get("error_category")
    return cleaner({key: value for key, value in row.items() if value is not None})


def _campaign_observations(full):
    """Every terminal observation of this campaign, oldest first, from the native store."""
    from pathlib import Path
    camp = full.get("_campaign_path")
    if not camp or not (Path(camp) / "experiment_observations").is_dir():
        return []
    from .experiment_observations import ExperimentObservationStore
    return [record["body"] for record in reversed(ExperimentObservationStore(Path(camp)).records())]


def _knowledge(value):
    if not isinstance(value, dict):
        return _scientific_data(value)
    result = _pick(value, ("status",))
    result["items"] = [_scientific_data(_pick(item, ("record_id", "record_type", "body",
        "source_ref", "scope", "applicability", "condition_differences", "is_scientific_evidence")))
        for item in value.get("items") or []]
    return result


def _cost(value):
    result = _pick(value, ("known", "confidence", "basis", "runtime_allocation_bound"))
    if "predicted" in value:
        predicted = value["predicted"]
        result["predicted"] = _pick(predicted,
            ("llm_calls", "input_tokens", "output_tokens", "wall_seconds", "implementation_risk"))
        implementation = _pick(predicted.get("implementation") or {},
            ("required", "reuses_frozen_implementation", "reported_assessment"))
        if implementation:
            result["predicted"]["implementation"] = implementation
    return _scientific_data(result)


def _v2_enabled(full):
    return ((full.get("goal") or {}).get("research_execution") or {}).get("planning_contract_version") == "v2"


def _v2_science(value):
    """Drop known source shells; scientific config/metric keys stay literal."""
    if isinstance(value, list):
        return [_v2_science(child) for child in value if not (isinstance(child, dict)
            and isinstance(child.get("field"), str)
            and (child["field"].endswith(("_hash", "_digest", "_fingerprint"))
                 or child["field"] == "method_revision"))]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    preserve = {"model", "objective", "transform", "config", "scientific_parameters", "metrics",
                "scores", "paired_scores", "parameters", "hyperparameters", "training_recipe", "measurements"}
    omit = {"schema_version", "validation_records", "check", "checks", "cpu_check_receipt",
        "approval_record", "operation_binding", "coding_recovery", "next_read", "next_reads",
        "read_capability", "read_unit", "json_pointer", "verification_artifact",
        "satisfies_development_read_receipt", "campaign_id", "task_id", "attempt_id", "plan_id",
        "fingerprint", "digest", "sha256"}
    return {key: copy.deepcopy(child) if key in preserve else _v2_science(child)
        for key, child in value.items() if key not in omit and not key.startswith("_")
        and not key.endswith(("_hash", "_digest", "_fingerprint", "_sha256"))}


def _v2_observations(full, view):
    """Render only selected, complete cards; no native-store refill follows fit."""
    from .memory_delivery import project_memory_card
    rows = []
    cards = view.get("experiment_observations") or []
    cards = cards.get("cards", []) if isinstance(cards, dict) else cards
    for supplied in cards:
        if not isinstance(supplied, dict):
            continue
        card = project_memory_card(supplied, role="research_planner")
        row = copy.deepcopy(card["body"])
        row["observation_id"] = card["record_id"]
        for key in ("source_refs", "current_feedback", "condition_differences"):
            if key in card:
                row[key] = copy.deepcopy(card[key])
        if "applicability" in card and card["applicability"] != row.get("applicability"):
            row["applicability_status"] = copy.deepcopy(card["applicability"])
        rows.append(row)
    return rows


def _v2_knowledge(view):
    if view.get("memory_delivery_report"):
        # The same selected cards were measured with the optional observations.
        # No second truncation/projection can change that shared allocation.
        bundle = view.get("retrieved_knowledge") or {}
        return {"status": bundle.get("status", "no_related_knowledge"),
                "items": copy.deepcopy(bundle.get("items") or [])}
    result = {"status": "no_related_knowledge", "items": []}
    seen = set()
    for key in ("retrieved_knowledge", "retrieved_memory"):
        bundle = view.get(key)
        if not isinstance(bundle, dict):
            continue
        if bundle.get("status"):
            result["status"] = bundle["status"]
        elif bundle.get("degraded"):
            result["status"] = "retrieval_unavailable"
        for item in bundle.get("items") or []:
            if not isinstance(item, dict):
                continue
            identity = item.get("record_id")
            if identity and identity in seen:
                continue
            card = _v2_science(_pick(item, ("record_id", "record_type", "body", "source_ref", "source_refs",
                "scope", "conditions", "applicability", "condition_differences", "is_scientific_evidence",
                "scientific_evidence", "knowledge_role", "omissions")))
            original = item.get("full_record") or {}
            if "source_ref" not in card and isinstance(original, dict) and original.get("$context_source"):
                card["source_ref"] = original["$context_source"]
            result["items"].append(card)
            if identity:
                seen.add(identity)
    if result["items"] and result["status"] == "no_related_knowledge":
        result["status"] = "ready"
    return result


def _v2_cost(value):
    """A native fold/allocation estimate is not a measured candidate runtime."""
    known = value.get("known") or {}
    result = {key: copy.deepcopy(known[key]) for key in ("training_jobs",) if key in known}
    if "gpu_seconds" in known:
        result["gpu_seconds_estimate"] = copy.deepcopy(known["gpu_seconds"])
    result.update(_pick(value, ("basis", "confidence")))
    return result


def _v2_resources(full):
    budget = full.get("budget") or {}
    owner = budget.get("owner_budget") or {}
    # A Planner owner reserves API calls, not a future method's training. Only
    # global and unreserved amounts can bound new selected-task allocations.
    unreserved = owner.get("unreserved_available") or {}
    result = {}
    for field, dimension in (("gpu_seconds_left", "gpu_seconds"), ("training_jobs_left", "training_jobs")):
        known = [source[key] for source, key in ((budget, field), (unreserved, dimension))
                 if key in source and type(source[key]) in (int, float) and math.isfinite(source[key])]
        if known:
            result[field] = min(known)
        elif any(key in source and source[key] is None for source, key in ((budget, field), (unreserved, dimension))):
            result[field] = None
    entries = full.get("action_cost_estimates") or {}
    entries = entries.get("by_action_target") or entries
    costs = {}
    for action, body in entries.items():
        if not isinstance(body, dict) or action.split(":", 1)[0] not in TASK_ACTIONS:
            continue
        estimate = body.get("estimated_cost") or {}
        if not any(type(estimate.get(key)) in (int, float) for key in ("training_jobs", "gpu_seconds")):
            continue
        cost = _pick(estimate, ("training_jobs",))
        if "gpu_seconds" in estimate:
            cost["gpu_seconds_estimate"] = copy.deepcopy(estimate["gpu_seconds"])
        cost.update(_pick(body, ("cost_confidence", "cost_scope", "unknown_components")))
        costs[action] = cost
    if costs:
        result["native_cost_basis"] = costs
    return result


def _v2_task_view(task):
    scope = task.get("evaluation_scope") or {}
    result = _pick(task, ("task_ref", "task_kind", "target_candidate_ref"))
    result.update(_pick(scope, ("phase",)))
    return result


def _v2_protocol(full, view, baseline):
    goal = view.get("goal") or full.get("goal") or {}
    interface = view.get("candidate_interface") or full.get("candidate_interface") or {}
    recipe = copy.deepcopy(baseline.get("recipe") or {})
    geometry = recipe.pop("input_geometry", None)
    protocol = _pick(goal, ("task_type", "evaluation_mode", "evaluation_schedule", "primary_metric",
        "direction", "checkpoint_policy", "subject_aggregation", "training_seeds", "frozen_components"))
    protocol.update(_pick(full.get("frozen_evaluation_scopes") or {}, ("metric_unit",)))
    protocol.update(_pick(interface, ("input", "output", "gallery_size", "not_available")))
    geometry = geometry or _pick(interface, ("c_num", "channels", "timesteps"))
    if geometry:
        protocol["input_geometry"] = geometry
    if recipe:
        for key in ("training_seeds", "negative_sampling_policy"):
            if key in recipe:
                protocol[key] = recipe.pop(key)
        protocol["training_recipe"] = recipe
    if "negative_sampling_policy" not in protocol and baseline.get("negative_sampling_policy"):
        protocol["negative_sampling_policy"] = baseline["negative_sampling_policy"]
    contract = view.get("contract") or full.get("contract") or {}
    protocol.update(_pick(contract, ("generalization_target", "val_mode", "benchmark_scope", "within_batch_role")))
    scope = copy.deepcopy(view.get("science_scope") or full.get("science_scope") or {})
    if goal.get("final_test_enabled") is False:
        scope["independent_final_test"] = False
    if scope:
        protocol["evidence_scope"] = _v2_science(scope)
    for source, name in ((goal, "min_practical_gain_pp"), (goal, "selection_min_delta")):
        if name in source:
            protocol[name] = copy.deepcopy(source[name])
    screening = goal.get("screening") or {}
    if "min_mean_gain_pp" in screening:
        protocol["screening_min_mean_gain_pp"] = screening["min_mean_gain_pp"]
    confirmation = goal.get("confirmation") or {}
    protocol.update(_pick(confirmation, ("freeze_method_before_screening",)))
    # Capability behavior is concise and stage-neutral. Implementation signatures,
    # buffers, module import paths and Reviewer/check mechanics stay downstream.
    capability = view.get("capabilities") or full.get("capabilities") or {}
    hooks = capability.get("hooks") or {}
    meanings = {"build_encoder": "Replace the EEG encoder while preserving the protocol input/output.",
        "build_training_objective": "Optional training loss with frozen image targets and runtime positive relations.",
        "build_training_transform": "Optional augmentation applied to training EEG only.",
        "fit_statistics": "Optional per-channel statistics computed from training files only.",
        "evaluate_only": "Evaluate an existing candidate with the frozen evaluator."}
    available = {name: meanings[name] for name in meanings if (hooks.get(name) or {}).get("status") == "available"}
    if available:
        protocol["method_capabilities"] = available
    policies = capability.get("negative_sampling_policies") or {}
    alternatives = {name: _pick(body, ("status", "when")) for name, body in policies.items()
                    if name != protocol.get("negative_sampling_policy")}
    if alternatives:
        protocol["other_negative_sampling_policies"] = alternatives
    population = view.get("evaluation_population") or full.get("evaluation_population") or {}
    # Counts and score resolution belong to their real phase, not to a copied
    # ten-fold denominator interpreted as the current two-fold screen.
    counts = _pick(population, ("benchmark_query_count_per_fold", "uniform_gallery_chance_top1"))
    if counts:
        protocol["evaluation_population"] = counts
    scopes = copy.deepcopy((full.get("frozen_evaluation_scopes") or {}).get("phases") or {})
    for task in (full.get("_planning_contract") or {}).get("tasks", []):
        binding = task.get("evaluation_scope") or {}
        phase = binding.get("phase")
        if phase and phase not in scopes:
            scopes[phase] = copy.deepcopy(binding)
    if scopes:
        controls = [scope.get("controls") for scope in scopes.values() if scope.get("controls") is not None]
        shared_control = controls[0] if controls and all(item == controls[0] for item in controls) else None
        if shared_control is not None:
            protocol["matched_controls"] = shared_control
        phase_views = {}
        seeds = protocol.get("training_seeds")
        per_fold = population.get("benchmark_query_count_per_fold")
        for name, scope in scopes.items():
            item = _pick(scope, ("folds", "seed", "controls"))
            if seeds == [item.get("seed")]:
                item.pop("seed", None)
            if shared_control is not None:
                item.pop("controls", None)
            folds = item.get("folds")
            if (isinstance(folds, list) and folds and type(per_fold) is int and per_fold > 0
                    and protocol.get("subject_aggregation") == "macro_mean"):
                item["query_count"] = len(folds) * per_fold
                item["one_query_hit_delta_pp"] = 100.0 / (len(folds) * per_fold)
            phase_views[name] = item
        protocol["evaluation_scopes"] = phase_views
    return protocol


def _build_v2_proposer_input(full, delivered):
    if delivered.get("planning_context_version") == VERSION_V2:
        return copy.deepcopy(delivered)
    view = _expanded(delivered)
    goal = view.get("goal") or full.get("goal") or {}
    contract = full.get("_planning_contract")
    if not isinstance(contract, dict):
        raise ValueError("planning_contract_missing")
    baseline = next((value for value in (full.get("baseline_source"), full.get("parent_source"),
        full.get("control_source"), view.get("baseline_source")) if isinstance(value, dict)
        and value.get("candidate_id", "baseline") == "baseline"), {})
    result = {"planning_context_version": VERSION_V2, "planning_stage": "propose",
        "research_goal": _pick(goal, ("objective", "research_scope", "allowed_changes")),
        "protocol": _v2_protocol(full, view, baseline)}
    result["research_goal"]["target_k"] = ((goal.get("research_execution") or {}).get("target_k", 3))
    from .handoffs import planner_baseline_scientific_source
    method = _pick(baseline, ("intervention", "model", "objective", "transform"))
    method["scientific_source"] = planner_baseline_scientific_source(baseline)
    # Explicit method constraints remain literal, even when parameter names look
    # like metadata. They are not reconstructed from Python prose.
    if baseline.get("scientific_parameters"):
        method["scientific_parameters"] = copy.deepcopy(baseline["scientific_parameters"])
    for key in ("measured_parameter_counts", "parameter_count_comparisons", "verified_encoder_structural_facts"):
        value = view.get(key) or full.get(key)
        if not value:
            continue
        if key == "measured_parameter_counts" and not value.get("by_candidate"):
            continue
        if key == "parameter_count_comparisons" and not value.get("comparisons"):
            continue
        method[key] = _v2_science(value)
    source_map = full.get("training_target_sources") or {}
    source_map = source_map.get("by_candidate", source_map)
    allowed_targets = {task.get("target_candidate_ref") for task in contract.get("tasks", [])
                       if task.get("target_candidate_ref")}
    existing = {}
    for target, body in source_map.items():
        if target == "baseline" or target not in allowed_targets or not isinstance(body, dict):
            continue
        # These are actual source-bound methods for permitted existing-target
        # tasks; a generic baseline cannot stand in for their implementation.
        existing[target] = _v2_science(_pick(body, ("intervention", "hypothesis", "source", "source_ref",
            "source_truncated", "missing_inputs", "model", "objective", "transform", "scientific_parameters")))
    if existing:
        method["existing_methods"] = existing
    result["baseline_method"] = method
    tasks = contract.get("tasks") or []
    if len(tasks) == 1:
        result["fixed_task"] = _v2_task_view(tasks[0])
        result["fixed_task"].pop("task_ref", None)
    else:
        result["task_options"] = [_v2_task_view(task) for task in tasks]
    result["observations"] = _v2_observations(full, view)
    # Cards have already been scientifically projected and measured. Removing
    # duplicate conditions here would make the frozen measurement a different
    # view from the one the Planner and Critic actually receive.
    result["knowledge"] = _v2_knowledge(view)
    result["resources"] = _v2_resources(full)
    if any(task.get("task_kind") == "new_method" for task in tasks):
        result["resources"]["new_method_cost_basis"] = (
            "Candidate-fold work is additional to any missing matched baseline folds; "
            "native admission computes both, and GPU seconds are estimates.")
    instruction = view.get("current_user_instruction") or full.get("current_user_instruction")
    if instruction:
        result["research_goal"]["current_user_instruction"] = (_pick(instruction, ("request_zh", "requested_task_kind"))
            if isinstance(instruction, dict) else copy.deepcopy(instruction))
    for key in ("plan_repair_request", "critic_repair_request", "previous_response_error", "previous_response"):
        if key in view:
            result[key] = copy.deepcopy(view[key])
    return result


def build_proposer_input(full: dict, delivered: dict) -> dict:
    """Use the already retrieved science; native full remains outside the prompt."""
    if _v2_enabled(full) or delivered.get("planning_context_version") == VERSION_V2:
        return _build_v2_proposer_input(full, delivered)
    if delivered.get("planning_context_version") == VERSION:
        result = copy.deepcopy(delivered)
        for field in ("proposal_pool", "proposal_pool_digest", "plan_repair_request",
                      "critic_repair_request", "previous_response_error"):
            result.pop(field, None)
        result["planning_stage"] = "propose"
        return result
    view = _expanded(delivered)
    goal = view.get("goal") or full.get("goal") or {}
    result = {"planning_context_version": VERSION, "planning_stage": "propose",
        "goal": _pick(goal, GOAL_FIELDS)}
    for section, fields in {
        "screening": ("fold_count", "subjects", "selection", "selection_seed", "training_recipe", "min_mean_gain_pp"),
        "confirmation": ("fold_count", "freeze_method_before_screening", "subject_aggregation", "max_exploration_confirmations"),
        "research_execution": ("target_k", "max_completed_rounds"),
    }.items():
        if section in goal:
            result["goal"][section] = _pick(goal[section], fields)
    goal_view = result["goal"]
    # Contract values repeated from the goal carry no extra information.
    result["contract"] = {key: value for key, value in
        _pick(view.get("contract") or full.get("contract") or {}, CONTRACT_FIELDS).items()
        if key not in {"label", "legacy_unverified_note"} and (key not in goal_view or goal_view[key] != value)}
    repeated = {"evaluation_mode": "evaluation_mode", "evaluation_schedule": "evaluation_schedule",
        "declared_full_primary_metric": "primary_metric", "full_required_fold_count": "required_fold_count"}
    for key in ("evaluation_mode", "evaluation_schedule", "primary_metric", "declared_full_primary_metric",
                "full_required_fold_count", "scientific_unit", "science_scope", "screening_evidence_scope",
                "measured_parameter_counts", "parameter_count_comparisons", "verified_encoder_structural_facts"):
        value = view.get(key)
        if value is None or value == [] or (key in repeated and goal_view.get(repeated[key]) == value):
            continue
        if key == "parameter_count_comparisons" and isinstance(value, dict) and not value.get("comparisons"):
            continue
        result[key] = _scientific_data(value)
    if isinstance(result.get("measured_parameter_counts"), dict):
        result["measured_parameter_counts"].pop("scope", None)
    if full.get("frozen_evaluation_scopes"):
        result["frozen_evaluation_scopes"] = copy.deepcopy(full["frozen_evaluation_scopes"])
    result["candidate_interface"] = _pick(view.get("candidate_interface") or {}, INTERFACE_FIELDS)
    # A projected baseline must include the actual encoder, not just its wrapper.
    baseline = view.get("baseline_source")
    if not baseline:
        baseline = next((full.get(key) for key in ("parent_source", "control_source")
            if (full.get(key) or {}).get("candidate_id") == "baseline"), None)
    if baseline:
        result["baseline_source"] = _source(baseline)
        (result["baseline_source"].get("frozen_training_policy") or {}).pop("interpretation", None)
    capabilities = view.get("capabilities") or {}
    result["capabilities"] = {key: {name: _pick(item, ("status", "verified", "default", "when"))
        for name, item in capabilities.get(key, {}).items()}
        for key in ("hooks", "task_adapters", "negative_sampling_policies") if key in capabilities}
    result["available_actions"] = [action for action in view.get("available_actions", full.get("available_actions", []))
        if action in TASK_ACTIONS]
    result["eligible_targets"] = {key: copy.deepcopy(targets) for key, targets in
        (view.get("eligible_targets") or full.get("eligible_targets") or {}).items() if key in TASK_ACTIONS}
    result["candidates"] = [_pick(item, ("candidate_id", "status")) for item in
        view.get("candidates", full.get("candidates", []))]
    if view.get("candidate_id"):
        result["current_candidate"] = _pick(view, ("candidate_id", "candidate_ready"))
    experiment = view.get("experiment") or {}
    if experiment:
        result["previous_experiment"] = _scientific_data(_pick(experiment,
            ("candidate_id", "parent_candidate_id", "control_candidate_id", "hypothesis", "intervention",
             "scientific_intent", "principal_intervention", "evaluation_scope")))
        if result["previous_experiment"].get("principal_intervention") == result["previous_experiment"].get("intervention"):
            result["previous_experiment"].pop("principal_intervention", None)
    sources = full.get("training_target_sources") or {}
    if isinstance(sources, dict):
        sources = sources.get("by_candidate", sources)
        result["target_sources"] = {key: _source(value) for key, value in sources.items()
            if isinstance(value, dict) and value.get("candidate_id") != "baseline"}
    for key in ("evaluation_population", "confirmation_policy"):
        if key in view:
            result[key] = _scientific_data(view[key])
    budget = view.get("budget") or full.get("budget") or {}
    result["budget"] = _pick(budget, ("llm_calls_left", "gpu_seconds_left", "training_jobs_left"))
    costs = view.get("action_cost_estimates") or full.get("action_cost_estimates") or {}
    entries = costs.get("by_action_target") or costs
    # An estimate whose every cost component is unknown says nothing comparable.
    result["action_cost_estimates"] = {key: _scientific_data(_pick(value,
        ("estimated_cost", "cost_confidence", "cost_scope", "unknown_components")))
        for key, value in entries.items() if key.split(":", 1)[0] in TASK_ACTIONS and isinstance(value, dict)
        and not (isinstance(value.get("estimated_cost"), dict) and value["estimated_cost"]
                 and all(cost is None for cost in value["estimated_cost"].values()))}
    if view.get("memory_delivery_report"):
        # Scientific Memory selection also applies to the v1 planning contract.
        # Its public section names stay stable; selected science is never
        # replaced by a reloaded, larger campaign-history view.
        result["experiment_observations"] = _v2_observations(full, view)
        result["retrieved_knowledge"] = _v2_knowledge(view)
        instruction = view.get("current_user_instruction") or full.get("current_user_instruction")
        if instruction:
            result["current_user_instruction"] = _pick(instruction, ("request_zh", "requested_task_kind"))
        return result
    result["experiment_observations"] = _observation_cards(view.get("experiment_observations"))
    geometry = ((result.get("baseline_source") or {}).get("recipe") or {}).get("input_geometry")
    # Retrieval bounds which cards are delivered; the planning view still needs
    # every terminal result of this campaign to avoid repeating a tested method.
    stored = {body["observation_id"]: body for body in _campaign_observations(full)}
    for row in result["experiment_observations"]:
        _applicability(row, geometry)
        body = stored.get(row.get("observation_id"))
        if body:
            for key, value in _scientific_data(_stored_science(body)).items():
                if isinstance(value, dict) and isinstance(row.get(key), dict):
                    row[key] = {**row[key], **value}
                else:
                    row.setdefault(key, value)
    delivered_ids = {row.get("observation_id") for row in result["experiment_observations"]}
    history = [_history_row(body) for observation_id, body in stored.items() if observation_id not in delivered_ids]
    if history:
        result["experiment_history"] = history
    if "retrieved_knowledge" in view:
        result["retrieved_knowledge"] = _knowledge(view["retrieved_knowledge"])
    for key in ("retrieved_memory", "memory", "lessons"):
        if view.get(key):
            result[key] = _scientific_data(view[key])
    instruction = view.get("current_user_instruction") or full.get("current_user_instruction")
    if instruction:
        result["current_user_instruction"] = _pick(instruction, ("request_zh", "requested_task_kind"))
    return result


def delivered_observation_ids(view: dict) -> frozenset[str]:
    """Observation IDs the model actually received; basis IDs must come from here."""
    cards = view.get("observations") if view.get("planning_context_version") == VERSION_V2 else view.get("experiment_observations")
    cards = cards or []
    if isinstance(cards, dict):
        cards = [card.get("body") or card for card in cards.get("cards") or []]
    rows = [*cards, *(view.get("experiment_history") or [])]
    return frozenset(row["observation_id"] for row in rows
        if isinstance(row, dict) and isinstance(row.get("observation_id"), str))


def delivered_observation_rows(view: dict, ids) -> list[dict]:
    """The exact delivered rows for the given IDs, in delivery order."""
    wanted = set(ids)
    cards = view.get("observations") if view.get("planning_context_version") == VERSION_V2 else view.get("experiment_observations")
    cards = cards or []
    if isinstance(cards, dict):
        cards = [card.get("body") or card for card in cards.get("cards") or []]
    return [copy.deepcopy(row) for row in [*cards, *(view.get("experiment_history") or [])]
        if isinstance(row, dict) and row.get("observation_id") in wanted]


def selection_basis(camp, plan: dict) -> dict | None:
    """Critic rationale and the delivered history rows behind the selected proposal.

    Rows come from the frozen generation snapshot, i.e. what the Planner saw; no
    new retrieval. Rounds without a staged snapshot (legacy wires) return None.
    """
    import json
    from pathlib import Path
    rid = plan.get("originating_round")
    paths = (Path(camp) / "research_rounds" / (str(rid) + ".json"),
             Path(camp) / "research_round_inputs" / str(rid) / "generation.json")
    if not rid or not all(path.is_file() for path in paths):
        return None
    round_record, snapshot = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    view = snapshot.get("model_input") or {}
    if view.get("planning_context_version") not in {VERSION, VERSION_V2}:
        return None
    proposal = plan.get("proposal") or {}
    ids = list(dict.fromkeys([*(proposal.get("basis_observation_ids") or []),
                              *(round_record.get("selected_basis_observation_ids") or [])]))
    result = {"selection_rationale": round_record.get("selection_rationale"),
              "basis_observation_ids": ids, "basis_observations": delivered_observation_rows(view, ids)}
    selected_id = plan.get("originating_proposal") or proposal.get("proposal_id")
    selected = next((row for row in round_record.get("evaluations") or []
                     if row.get("proposal_id") == selected_id), None)
    if isinstance(selected, dict) and "assessment" in selected:
        result["selected_assessment"] = copy.deepcopy(selected["assessment"])
    return result


def _critic_v2_context(pool, full, critic_context):
    from .critic_contract import verify_critic_context
    context = critic_context if critic_context is not None else (full or {}).get("_critic_context")
    if not isinstance(context, dict):
        raise ValueError("critic_context_missing")
    if verify_critic_context(context, pool):
        raise ValueError("critic_context_invalid")
    return context


def _critic_v2_cost(value):
    if value is None:
        return None
    result = _pick(value, ("basis", "confidence", "unknown_components", "cost_scope"))
    if "known" in value:
        known = copy.deepcopy(value["known"])
        if isinstance(known, dict) and "gpu_seconds" in known:
            known["gpu_seconds_estimate"] = known.pop("gpu_seconds")
        result["known"] = known
    # Forecast dimensions are relevant to cost comparison when an estimate
    # exists. Implementation risk is already supplied once in each proposal;
    # implementation scope is represented by its exact task binding.
    predicted = {key: copy.deepcopy(item) for key, item in (value.get("predicted") or {}).items()
                 if key not in {"implementation_risk", "implementation"}}
    if any(item is not None for item in predicted.values()):
        result["predicted"] = predicted
    # An allocation maximum remains distinct from a cost estimate. It can
    # affect feasibility even when the candidate's runtime is unknown.
    if "runtime_allocation_bound" in value:
        result["runtime_allocation_bound"] = copy.deepcopy(value["runtime_allocation_bound"])
    return result


def _critic_v2_comparison(pool, full, critic_context):
    context = _critic_v2_context(pool, full, critic_context)
    aliases = {proposal_id: alias for alias, proposal_id in context["alias_to_id"].items()}
    rows, tasks, costs = [], [], []
    native_costs = pool.get("native_cost_by_id") or {}
    for native in pool["proposals"]:
        proposal_id = native["proposal_id"]
        rows.append({"proposal_ref": aliases[proposal_id],
            "question": copy.deepcopy(native["question"]),
            "method": copy.deepcopy(native["task_description"]),
            "rationale": copy.deepcopy(native["scientific_intent"]),
            "expected_observation": copy.deepcopy(native["expected_observation"]),
            "implementation_risk": copy.deepcopy(native.get("implementation_risk")),
            "basis_observation_ids": copy.deepcopy(native.get("basis_observation_ids") or []),
            "required_context_refs": copy.deepcopy(native.get("required_context_refs") or [])})
        # Use the admitted native task binding; a phase label alone cannot
        # represent candidate, controls, fold population or training seed.
        tasks.append({"task_kind": copy.deepcopy(native["task_kind"]),
            "target_candidate_ref": copy.deepcopy(native.get("target_candidate_ref")),
            "evaluation_scope": copy.deepcopy(native["evaluation_scope"])})
        costs.append(_critic_v2_cost(native_costs.get(proposal_id)))
    result = {"proposals": rows}
    if tasks and all(task == tasks[0] for task in tasks):
        result["shared_task"] = tasks[0]
    else:
        for row, task in zip(rows, tasks):
            row["task"] = task
    # Compare complete scientific cost views. Proposal risk prose is delivered
    # elsewhere, while every resource dimension, cost forecast, evidence basis
    # and allocation bound remains part of this exact equality.
    if costs and all(cost == costs[0] for cost in costs):
        result["shared_cost"] = costs[0]
    else:
        for row, cost in zip(rows, costs):
            row["cost"] = cost
    return result


def critic_pool_input(pool: dict, full: dict | None = None, *, contract: dict | None = None,
                      critic_context: dict | None = None) -> dict:
    """Compare every exact alternative, with costs rather than check trajectories."""
    from .critic_contract import enabled as lean_critic_enabled
    if critic_context is not None or lean_critic_enabled(full or {}):
        return _critic_v2_comparison(pool, full, critic_context)
    contract = contract or (full or {}).get("_planning_contract")
    if isinstance(contract, dict) and contract.get("version") == "eeg_research.planning_contract.v2":
        from .planning_contract import scientific_proposal_view
        context = {"goal": {"research_execution": {"planning_contract_version": "v2"}},
                   "_planning_contract": contract}
        return {"proposals": [{"proposal_id": item["proposal_id"], **scientific_proposal_view(item, context)}
                              for item in pool["proposals"]],
                "native_cost_by_id": {pid: _v2_cost(value) for pid, value in pool["native_cost_by_id"].items()}}
    return {"proposals": copy.deepcopy(pool["proposals"]),
        "native_cost_by_id": {pid: _cost(cost) for pid, cost in pool["native_cost_by_id"].items()},
        "pool_digest": pool["pool_digest"]}


def build_critic_input(full: dict, delivered: dict, native_pool: dict,
                       critic_context: dict | None = None) -> dict:
    from .critic_contract import enabled as lean_critic_enabled
    if lean_critic_enabled(full):
        # These science blocks were already delivered to the Planner. Do not
        # re-read sources, refill memory, or repeat protocol assembly here.
        required = ("research_goal", "protocol", "baseline_method", "observations", "knowledge", "resources")
        if delivered.get("planning_context_version") != VERSION_V2 or any(key not in delivered for key in required):
            raise ValueError("critic_frozen_science_input_missing")
        context = _critic_v2_context(native_pool, full, critic_context)
        if set(context["delivered_observation_ids"]) != set(delivered_observation_ids(delivered)):
            raise ValueError("critic_delivered_observations_changed")
        result = {"planning_context_version": VERSION_V2, "planning_stage": "critic"}
        result.update({key: copy.deepcopy(delivered[key]) for key in required})
        result["research_goal"].pop("target_k", None)
        result["resources"].pop("native_cost_basis", None)
        result["comparison"] = critic_pool_input(native_pool, full, critic_context=context)
        return result
    result = build_proposer_input(full, delivered)
    if result.get("planning_context_version") == VERSION_V2:
        for field in ("plan_repair_request", "critic_repair_request", "previous_response_error", "previous_response"):
            result.pop(field, None)
        result.update(planning_stage="critic", proposal_pool=critic_pool_input(native_pool, full))
        return result
    result.update(planning_stage="critic", proposal_pool=critic_pool_input(native_pool),
                  proposal_pool_digest=native_pool["pool_digest"])
    return result


def minimal_validation_feedback(records: list[dict]) -> list[dict]:
    """Only actionable errors; Pydantic's echoed input values are not feedback."""
    result = []
    for record in records:
        if not record.get("error"):
            continue
        lines = []
        for line in str(record["error"]).splitlines():
            if "errors.pydantic.dev/" in line or re.match(r"^\d+ validation errors? for ", line):
                continue
            # The diagnostic input_value can be an entire model response.
            line = re.sub(r"(\[type=[^,\]]+),.*", r"\1]", line)
            if line.strip():
                lines.append(line)
        row = {"error": "\n".join(lines)}
        row.update(_pick(record, ("proposal_id", "field", "type")))
        if row not in result:
            result.append(row)
    return result
