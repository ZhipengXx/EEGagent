"""Compact research-task drafts and native feasibility/cost projections.

One selected research task owns engineering and its declared measurement. This
module never admits a candidate or creates implementation work for alternatives.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .action_planning import digest
from .schemas import ResearchExecutionConfig, ResearchPlanProposal, validate_research_task
from .research_execution import single_plan_enabled

VERSION = "single_plan_v1"
MEASUREMENT_ACTIONS = {"diagnostic": "collect_diagnostics", "screening_2fold": "run_screening",
                       "confirmation_10fold": "run_confirmation", "pilot": "run_pilot", "full": "run_full"}


def enabled(value: dict) -> bool:
    return single_plan_enabled(value.get("goal") or value)


def config(observation: dict) -> ResearchExecutionConfig:
    return ResearchExecutionConfig.model_validate(observation.get("research_execution") or
        (observation.get("goal") or {}).get("research_execution") or {})


def round_input(observation: dict) -> dict:
    """Paid draft/retry artifacts do not manufacture a different research round."""
    cards = observation.get("experiment_observations") or observation.get("observations") or []
    if isinstance(cards, dict):
        cards = [row.get("body") or row for row in cards.get("cards") or []]
    return {"observations": [{k: row.get(k) for k in ("observation_id", "content_digest")}
        for row in cards if isinstance(row, dict)],
        "terminal_observation_digest": observation.get("terminal_observation_digest"),
        "goal": observation.get("goal"), "contract_fingerprint": observation.get("contract_fingerprint"),
        "execution_fingerprint": observation.get("execution_fingerprint"),
        "eligible_targets": observation.get("eligible_targets"),
        "candidate_admissions": observation.get("candidate_admissions"),
        "verified_development_facts": observation.get("verified_development_facts"),
        "evidence": [{k: row.get(k) for k in ("evidence_id", "candidate_id", "kind", "source_hash", "spec_hash", "result_hash")}
                     for row in observation.get("evidence") or []]}


def round_id(observation: dict) -> str:
    return "research_round_" + digest(round_input(observation))[:32]


def _phase(proposal: dict) -> str:
    return str((proposal.get("evaluation_scope") or {}).get("phase") or "")


def validate_task(proposal: dict, observation: dict) -> tuple[str, str]:
    kind, phase = proposal["task_kind"], _phase(proposal)
    goal = observation.get("goal") or {}
    schedule = observation.get("evaluation_schedule") or goal.get("evaluation_schedule") or "full_only"
    validate_research_task(kind, phase, proposal.get("target_candidate_ref"), schedule)
    if phase != "diagnostic" and "allowed_training_actions" in goal:
        from .planner import training_actions
        action = MEASUREMENT_ACTIONS[phase]
        if action not in training_actions({**goal, "evaluation_schedule": schedule}):
            raise ValueError("task_training_action_outside_goal:" + action)
    return kind, phase


def measurement_action(proposal: dict, observation: dict) -> str:
    _, phase = validate_task(proposal, observation)
    return MEASUREMENT_ACTIONS[phase]


def scientific_projection(spec: dict | None) -> dict:
    """Engineering attempts may change source bytes; approved science may not."""
    spec = spec or {}
    if spec.get("design_contract_version") == "v2":
        from .design_contract import scientific_design_projection
        return scientific_design_projection(spec)
    return {key: spec.get(key) for key in ("intervention", "principal_intervention", "hypothesis", "model", "objective", "transform",
        "parent_candidate_id", "control_candidate_id", "initial_fidelity", "implementation_requirements",
        "required_capability_ids", "evaluation_scope")}


def proposal_cost(proposal: dict, observation: dict) -> dict:
    kind, phase = validate_task(proposal, observation)
    target = proposal.get("target_candidate_ref")
    result = {"known": {"training_jobs": None, "gpu_seconds": None}, "predicted": {
        "llm_calls": None, "input_tokens": None, "output_tokens": None, "wall_seconds": None},
        "basis": [], "confidence": "unknown"}
    if proposal.get("implementation_risk") is not None:
        result["predicted"]["implementation_risk"] = {"assessment": proposal["implementation_risk"],
            "authority": "model_prediction_not_execution_authority"}
    if phase == "diagnostic":
        result.update(known={"training_jobs": 0, "gpu_seconds": 0.0, "feedback_report_count": 1},
                      basis=["Diagnostic task reuses recorded results and needs no training."], confidence="runtime_bound")
        return result
    execution = observation.get("research_execution") or (observation.get("goal") or {}).get("research_execution") or {}
    task_budget = execution.get("budget") or {}
    envelope = (task_budget.get("envelopes") or {}).get(kind) or {}
    if "training_jobs" in envelope or "gpu_seconds" in envelope:
        result["runtime_allocation_bound"] = {key: envelope.get(key) for key in ("training_jobs", "gpu_seconds")}
        result["runtime_allocation_bound"].update(
            basis=envelope.get("allocation_basis") or "explicit_task_envelope",
            authority="maximum_runtime_allocation_not_cost_estimate")
    action = measurement_action(proposal, observation)
    costs = observation.get("action_cost_estimates") or {}
    entries = costs.get("by_action_target") or costs
    entry = entries.get(action + ":" + str(target)) or entries.get(action + ":baseline") or {}
    estimate = entry.get("estimated_cost") or {}
    if target and entry:
        result["known"] = {key: estimate.get(key) for key in ("training_jobs", "gpu_seconds")}
        result["basis"] = entry.get("cost_basis") or []
        result["confidence"] = entry.get("cost_confidence") or "unknown"
    camp_path = observation.get("_campaign_path")
    if camp_path and (Path(camp_path) / "method_search_manifest.json").is_file():
        from .method_suite import manifest
        from .screening import phase_cost, suites_for_target, valid_scores
        camp = Path(camp_path)
        common = manifest(camp)
        if common.get("evaluation_schedule") == "screen_then_confirm":
            controls = suites_for_target(camp, "baseline")
            b_done = [r["fold_id"] for r in valid_scores(camp, controls[-1])] if controls else []
            suites = suites_for_target(camp, target) if target else []
            c_done = [r["fold_id"] for r in valid_scores(camp, suites[-1])] if suites else []
            estimate = phase_cost(common, "confirmation_10fold" if kind == "confirmation" else "screening_2fold",
                                  c_done, b_done, target=target or "unadmitted_method")
            result.update(known={k: estimate.get(k) for k in ("training_jobs", "gpu_seconds", "feedback_report_count")},
                basis=["Frozen phase folds and verified completed candidate/control folds."], confidence="runtime_bound")
        else:
            baseline = any(r.get("candidate_id") == "baseline" and r.get("kind") == "method_suite"
                           and r.get("evaluation_valid") for r in observation.get("evidence") or [])
            jobs = 10 if target == "baseline" or baseline else 20
            result.update(known={"training_jobs": jobs, "gpu_seconds": common["gpu_seconds_estimate_per_suite"] * jobs / 10,
                               "feedback_report_count": 1},
                basis=["Frozen full suite plus required matched baseline if absent."], confidence="runtime_bound")
    # Implementation forecasts stay honest and differentiated by scope/history.
    result["predicted"]["implementation"] = {"required": kind == "new_method",
        "scientific_intent": proposal["scientific_intent"],
        "reuses_frozen_implementation": bool(target), "reported_assessment": proposal.get("cost_assessment") or {},
        "authority": "model_prediction_not_native_cost"}
    return result


def _check_basis(ids: list[str], observation: dict) -> None:
    """Basis IDs name delivered observations only; absent delivery sets (legacy replay) are not guessed."""
    if len(set(ids)) != len(ids):
        raise ValueError("basis_observation_ids_duplicate")
    delivered = observation.get("_delivered_observation_ids")
    if delivered is not None and any(item not in delivered for item in ids):
        raise ValueError("basis_observation_id_not_delivered")


def _validate_proposal(raw: Any, observation: dict) -> tuple[dict | None, dict]:
    try:
        item = ResearchPlanProposal.model_validate(raw).model_dump()
        kind, phase = validate_task(item, observation)
        target = item.get("target_candidate_ref")
        if kind == "new_method":
            if target is not None:
                raise ValueError("new_method_cannot_overwrite_existing_candidate")
            if not any(a in observation.get("available_actions", []) for a in ("design_experiment", "propose_experiment")):
                raise ValueError("new_method_admission_unavailable")
        else:
            from .planner import known_candidate_ids
            if not target or target not in known_candidate_ids(observation):
                raise ValueError("target_candidate_missing_or_unknown")
        if kind in {"confirmation", "validation"} and phase != "diagnostic":
            action = measurement_action(item, observation)
            if target not in (observation.get("eligible_targets") or {}).get(action, []):
                raise ValueError("measurement_target_not_eligible")
        refs = {r.get("artifact_id") for r in observation.get("artifact_index") or []
                if r.get("verification_status") == "verified"}
        for ref in item["required_context_refs"]:
            if ref not in (observation.get("_delivered_context_refs") or set()):
                raise ValueError("required_context_ref_not_delivered")
            if ref in refs:
                continue
            if not isinstance(ref, str) or not ref.startswith("context://") or not observation.get("_campaign_path"):
                raise ValueError("required_context_ref_unverified")
            delivered = observation.get("_delivered_context_refs")
            if delivered is not None and ref not in delivered:
                raise ValueError("required_context_ref_not_delivered")
            from .context.sources import ContextSourceRegistry
            if observation.get("_paid_replay_readonly"):
                from .paid_research_round import readonly_registry
                registry = readonly_registry(Path(observation["_campaign_path"]))
            else:
                registry = ContextSourceRegistry(Path(observation["_campaign_path"]))
            source = registry.get(ref, role="research_planner")
            registry.load(source, role="research_planner")
        _check_basis(item.get("basis_observation_ids") or [], observation)
        if phase != "diagnostic" and observation.get("evaluation_mode") == "loso_method_search":
            folds = item["evaluation_scope"].get("folds")
            expected = 2 if phase == "screening_2fold" else 10
            if not isinstance(folds, list) or len(folds) != expected or len(set(folds)) != expected:
                raise ValueError("declared_fold_scope_mismatch")
            if item["evaluation_scope"].get("seed") != 0:
                raise ValueError("method_task_requires_frozen_seed0")
            path = observation.get("_campaign_path")
            if path and (Path(path) / "method_search_manifest.json").is_file():
                from .method_suite import manifest
                common = manifest(Path(path))
                declared = ([r["fold_id"] for r in common["folds"]] if expected == 10 else
                            common["selection_manifest"]["screening_fold_ids"])
                if set(folds) != set(declared):
                    raise ValueError("declared_folds_differ_from_frozen_protocol")
        return item, {"proposal_id": item["proposal_id"], "status": "valid", "raw_digest": digest(raw),
            "normalized_digest": digest(item), "native_cost": proposal_cost(item, observation)}
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        return None, {"proposal_id": raw.get("proposal_id") if isinstance(raw, dict) else None,
            "status": "rejected", "error": str(exc), "raw_digest": digest(raw), "draft": copy.deepcopy(raw)}


def pool_summary(proposals: list[dict], evaluations: list[dict], cfg: ResearchExecutionConfig) -> dict:
    """A feasible proposal counts as compared only with one accepted evaluation."""
    ids = {row["proposal_id"] for row in proposals}
    coverage = {pid: sum(row["proposal_id"] == pid for row in evaluations) for pid in ids}
    missing = sorted(pid for pid, count in coverage.items() if count != 1)
    count = len(proposals)
    status = ("no_feasible_plan" if not count else "incomplete" if missing else
              "complete" if 2 <= count <= 4 else
              "degraded_singleton" if count == 1 and cfg.allow_degraded_singleton else "incomplete")
    return {"pool_status": status, "valid_count": count,
            "evaluation_count": sum(count == 1 for count in coverage.values()),
            "unevaluated_proposal_ids": missing}


def proposal_pool_digest(pool: dict) -> str:
    """Bind the exact alternatives and native costs shown to the Critic."""
    return digest({"round_id": pool["round_id"], "proposals": pool["proposals"],
                   "native_cost": pool["native_cost_by_id"]})


def freeze_proposal_pool(proposals: list[dict], observation: dict, *,
                         validation_records: list[dict] | None = None) -> dict:
    """Validate and freeze a proposal-only pool, including pre-Critic admission.

    This function creates no candidate, owner or implementation work. Callers
    may pass only the alternatives that survived complete-task budget preview.
    """
    cfg = config(observation)
    accepted, records, ids, signatures, costs = [], copy.deepcopy(validation_records or []), set(), set(), {}
    for raw in proposals:
        proposal, trace = _validate_proposal(raw, observation)
        if proposal is not None:
            keys = ("task_kind", "target_candidate_ref", "scientific_intent", "evaluation_scope", "task_description")
            if proposal["task_kind"] != "new_method":
                keys += ("question",)
            signature = digest({key: proposal.get(key) for key in keys})
            if proposal["proposal_id"] in ids or signature in signatures:
                trace.update(status="rejected", error="duplicate_research_task")
            elif len(accepted) >= 4:
                trace.update(status="rejected", error="research_pool_at_maximum")
            else:
                accepted.append(proposal)
                ids.add(proposal["proposal_id"])
                signatures.add(signature)
                costs[proposal["proposal_id"]] = copy.deepcopy(trace["native_cost"])
        records.append(trace)
    count = len(accepted)
    status = "complete" if 2 <= count <= 4 else "no_feasible_plan" if not count else "incomplete"
    pool = {"round_id": round_id(observation), "input_observation_digest": digest(round_input(observation)),
        "target_k": cfg.target_k, "proposals": accepted, "native_cost_by_id": costs,
        "validation_records": records, "rejected": [row for row in records if row.get("status") == "rejected"],
        "pool_status": status, "valid_count": count,
        "shortfall_reason": None if count >= cfg.target_k else "independent_rejections_or_insufficient_distinct_tasks"}
    pool["pool_digest"] = proposal_pool_digest(pool)
    return pool


def normalize_proposal_pool(raw: Any, observation: dict, *, existing: list[dict] | None = None,
                            stage: str = "generation") -> dict:
    """Planner supplies competing tasks; it cannot rank or select its own work."""
    from .planning_contract import materialize_planner_reply
    assembled = materialize_planner_reply(raw, observation, stage=stage, existing=existing)
    errors, native_errors = assembled["errors"], assembled["native_errors"]
    records = [{"status": "rejected", "error": row["msg"], **row} for row in errors]
    if native_errors:
        return {"ok": False, "detail": "planning_native_input_invalid", "raw": copy.deepcopy(raw),
                "errors": errors, "native_errors": native_errors, "valid_proposals": copy.deepcopy(existing or []),
                "validation_records": records}
    pool = freeze_proposal_pool([*copy.deepcopy(existing or []), *assembled["proposals"]], observation,
                                validation_records=records)
    for row in pool["validation_records"]:
        if row.get("error") and "loc" not in row:
            pid = str(row.get("proposal_id") or "")
            suffix = pid.rsplit("_", 1)[-1]
            index = suffix.rsplit("p", 1)[-1] if "p" in suffix else suffix[1:]
            loc = ["proposals", int(index) - 1] if index.isdigit() else ["proposals"]
            errors.append({"loc": loc, "type": "planning_proposal_invalid", "msg": row["error"]})
    ok = pool["pool_status"] == "complete" and not errors
    return {"ok": ok, "detail": None if ok else "planning_proposal_invalid" if errors else "research_pool_" + pool["pool_status"],
            "raw": copy.deepcopy(raw), "pool": pool, "valid_proposals": copy.deepcopy(pool["proposals"]),
            "validation_records": copy.deepcopy(pool["validation_records"]), "errors": errors, "native_errors": []}


def normalize_critique(raw: Any, frozen_pool: dict, observation: dict, *, critic_context: dict | None = None) -> dict:
    """Accept only a complete Critic ranking bound to the delivered pool.

    Neither missing evaluations nor an invalid preferred alternative can cause
    native substitution. The Critic must select its rank-one alternative and
    provide the rationale for that exact selection.
    """
    from .schemas import ResearchPlanCritiqueV2, ResearchPlanEvaluationV2

    def rejected(detail, reason):
        return {"ok": False, "detail": detail, "raw": copy.deepcopy(raw),
            "valid_proposals": copy.deepcopy(frozen_pool.get("proposals") or []),
            "selected_proposal": None,
            "validation_records": [{"status": "rejected", "error": reason}]}

    try:
        if (frozen_pool["round_id"] != round_id(observation)
                or frozen_pool["input_observation_digest"] != digest(round_input(observation))
                or frozen_pool["pool_digest"] != proposal_pool_digest(frozen_pool)):
            raise ValueError("frozen_proposal_pool_identity_mismatch")
        current = freeze_proposal_pool(frozen_pool["proposals"], observation)
        if (current["pool_status"] != "complete" or current["pool_digest"] != frozen_pool["pool_digest"]
                or frozen_pool.get("pool_status") != "complete"
                or frozen_pool.get("valid_count") != current["valid_count"]):
            raise ValueError("frozen_proposal_pool_no_longer_valid")
    except (KeyError, ValueError, TypeError) as exc:
        return rejected("research_proposal_pool_identity_invalid", str(exc))
    from .critic_contract import materialize_critic_reply
    assembled = materialize_critic_reply(raw, frozen_pool, observation, critic_context=critic_context)
    if assembled["errors"] or assembled["native_errors"]:
        errors = assembled["errors"]
        detail = "critic_native_input_invalid" if assembled["native_errors"] else "planning_critic_invalid"
        return {**rejected(detail, "Critic response or its frozen native input is incomplete or invalid."),
                "errors": errors, "native_errors": assembled["native_errors"],
                "validation_records": [{"status": "rejected", "error": row["msg"], **row} for row in errors]}
    try:
        critique = ResearchPlanCritiqueV2.model_validate(assembled["reply"])
    except ValidationError as exc:
        return rejected("research_critique_schema_invalid", str(exc))
    if critique.proposal_pool_digest != frozen_pool["pool_digest"]:
        return rejected("research_critique_pool_digest_mismatch", "critic_must_compare_exact_frozen_pool")
    proposals = copy.deepcopy(frozen_pool["proposals"])
    by_id = {proposal["proposal_id"]: proposal for proposal in proposals}
    evaluations = [ResearchPlanEvaluationV2.model_validate(row).model_dump() for row in critique.evaluations]
    ids = [row["proposal_id"] for row in evaluations]
    ranks = [row["rank"] for row in evaluations]
    if len(ids) != len(by_id) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
        return rejected("research_critique_evaluation_coverage_invalid", "every_frozen_proposal_requires_one_evaluation")
    if set(ranks) != set(range(1, len(by_id) + 1)) or len(set(ranks)) != len(ranks):
        return rejected("research_critique_rank_invalid", "critic_ranks_must_be_unique_and_complete")
    winner = next(row["proposal_id"] for row in evaluations if row["rank"] == 1)
    if critique.selected_proposal_id != winner:
        return rejected("research_critique_selection_invalid", "critic_selected_proposal_must_match_rank_one")
    try:
        basis_input = {**observation, "_delivered_observation_ids": critic_context["delivered_observation_ids"]}
        _check_basis(critique.selected_basis_observation_ids, basis_input)
    except ValueError as exc:
        return rejected("research_critique_basis_invalid", str(exc))
    cards = observation.get("experiment_observations") or []
    if isinstance(cards, dict):
        cards = [row.get("body") or row for row in cards.get("cards") or []]
    observation_ids = [row["observation_id"] for row in cards if row.get("observation_id")]
    terminal = observation.get("terminal_observation_id")
    if terminal and terminal not in observation_ids:
        observation_ids.append(terminal)
    cfg = config(observation)
    records = copy.deepcopy(frozen_pool["validation_records"])
    record = {"round_id": frozen_pool["round_id"], "input_observation_ids": observation_ids,
        "input_observation_digest": frozen_pool["input_observation_digest"], "target_k": cfg.target_k,
        "proposals": proposals, "evaluations": evaluations, "selected_proposal_id": winner,
        **pool_summary(proposals, evaluations, cfg), "shortfall_reason": frozen_pool["shortfall_reason"],
        "selection_rationale": critique.selection_rationale, "selection_source": "independent_critic_selection",
        "planning_mode": "propose_then_critic", "proposal_pool_digest": frozen_pool["pool_digest"],
        "native_cost_by_id": copy.deepcopy(frozen_pool["native_cost_by_id"]),
        "validation_records": records, "raw_digest": digest(raw)}
    if "selected_basis_observation_ids" in critique.model_fields_set:
        record["selected_basis_observation_ids"] = list(critique.selected_basis_observation_ids)
    record["critic_protocol_version"] = critique.protocol_version
    return {"ok": True, "detail": None, "raw": copy.deepcopy(raw), "round": record,
        "valid_proposals": copy.deepcopy(proposals), "selected_proposal": copy.deepcopy(by_id[winner]),
        "validation_records": records}
