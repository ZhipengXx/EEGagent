"""Scientific inputs for proposal generation and independent comparison.

These are delivery views, never native observations or feasibility proofs.
Selection/validation/recovery continue to use the unchanged native objects.
No summarizer call, text clipping, or second memory retrieval is performed here.
"""
from __future__ import annotations

import copy
import re

VERSION = "research_planning_input.v1"

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


def _observation_cards(value):
    cards = value.get("cards", []) if isinstance(value, dict) else value or []
    result = []
    for card in cards:
        body = card.get("body", card)
        row = _pick(body, ("observation_id", "observation_kind", "question", "intervention",
            "execution_status", "evaluation_scope", "native_facts", "analyst_interpretation",
            "unresolved_questions", "applicability"))
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
        result.append(_scientific_data(row))
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
    return _pick(points, ("hypothesis_assessment", "summary_zh", "summary")) if isinstance(points, dict) else {}


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


def _history_row(body):
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
    return _scientific_data({key: value for key, value in row.items() if value is not None})


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


def build_proposer_input(full: dict, delivered: dict) -> dict:
    """Use the already retrieved science; native full remains outside the prompt."""
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
    cards = view.get("experiment_observations") or []
    if isinstance(cards, dict):
        cards = [card.get("body") or card for card in cards.get("cards") or []]
    rows = [*cards, *(view.get("experiment_history") or [])]
    return frozenset(row["observation_id"] for row in rows
        if isinstance(row, dict) and isinstance(row.get("observation_id"), str))


def delivered_observation_rows(view: dict, ids) -> list[dict]:
    """The exact delivered rows for the given IDs, in delivery order."""
    wanted = set(ids)
    cards = view.get("experiment_observations") or []
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
    if view.get("planning_context_version") != VERSION:
        return None
    proposal = plan.get("proposal") or {}
    ids = list(dict.fromkeys([*(proposal.get("basis_observation_ids") or []),
                              *(round_record.get("selected_basis_observation_ids") or [])]))
    return {"selection_rationale": round_record.get("selection_rationale"),
            "basis_observation_ids": ids, "basis_observations": delivered_observation_rows(view, ids)}


def critic_pool_input(pool: dict) -> dict:
    """Compare every exact alternative, with costs rather than check trajectories."""
    return {"proposals": copy.deepcopy(pool["proposals"]),
        "native_cost_by_id": {pid: _cost(cost) for pid, cost in pool["native_cost_by_id"].items()},
        "pool_digest": pool["pool_digest"]}


def build_critic_input(full: dict, delivered: dict, native_pool: dict) -> dict:
    result = build_proposer_input(full, delivered)
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
