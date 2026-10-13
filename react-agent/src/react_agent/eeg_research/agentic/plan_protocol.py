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
from .schemas import (ResearchExecutionConfig, ResearchPlanEvaluation,
                      ResearchPlanProposal, ResearchRoundDraft)
from .research_execution import single_plan_enabled

VERSION = "single_plan_v1"


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


def measurement_action(proposal: dict, observation: dict) -> str:
    phase = _phase(proposal)
    if proposal["task_kind"] == "confirmation":
        return "run_confirmation" if (observation.get("evaluation_schedule") or
            (observation.get("goal") or {}).get("evaluation_schedule")) == "screen_then_confirm" else "run_full"
    if phase == "screening_2fold":
        return "run_screening"
    return "run_pilot" if phase == "pilot" else "run_full"


def scientific_projection(spec: dict | None) -> dict:
    """Engineering attempts may change source bytes; approved science may not."""
    spec = spec or {}
    return {key: spec.get(key) for key in ("intervention", "principal_intervention", "hypothesis", "model", "objective", "transform",
        "parent_candidate_id", "control_candidate_id", "initial_fidelity", "implementation_requirements",
        "required_capability_ids", "evaluation_scope")}


def proposal_cost(proposal: dict, observation: dict) -> dict:
    kind, target = proposal["task_kind"], proposal.get("target_candidate_ref")
    result = {"known": {"training_jobs": None, "gpu_seconds": None}, "predicted": {
        "llm_calls": None, "input_tokens": None, "output_tokens": None, "wall_seconds": None},
        "basis": [], "confidence": "unknown"}
    if kind == "diagnosis" or _phase(proposal) == "diagnostic":
        result.update(known={"training_jobs": 0, "gpu_seconds": 0.0, "feedback_report_count": 1},
                      basis=["Diagnostic task reuses recorded results and needs no training."], confidence="runtime_bound")
        return result
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


def _validate_proposal(raw: Any, observation: dict) -> tuple[dict | None, dict]:
    try:
        item = ResearchPlanProposal.model_validate(raw).model_dump()
        kind, target = item["task_kind"], item.get("target_candidate_ref")
        phase = _phase(item)
        if phase not in {"screening_2fold", "confirmation_10fold", "full", "pilot", "diagnostic"}:
            raise ValueError("evaluation_phase_missing_or_unsupported")
        if kind == "new_method":
            if target is not None:
                raise ValueError("new_method_cannot_overwrite_existing_candidate")
            if not any(a in observation.get("available_actions", []) for a in ("design_experiment", "propose_experiment")):
                raise ValueError("new_method_admission_unavailable")
        else:
            from .planner import known_candidate_ids
            if not target or target not in known_candidate_ids(observation):
                raise ValueError("target_candidate_missing_or_unknown")
        if kind == "diagnosis" and phase != "diagnostic":
            raise ValueError("diagnosis_must_not_train")
        if kind == "confirmation" and phase not in {"confirmation_10fold", "full"}:
            raise ValueError("confirmation_phase_mismatch")
        schedule = observation.get("evaluation_schedule") or (observation.get("goal") or {}).get("evaluation_schedule")
        if schedule == "screen_then_confirm" and phase != "diagnostic":
            if phase != ("confirmation_10fold" if kind == "confirmation" else "screening_2fold"):
                raise ValueError("task_phase_does_not_match_frozen_schedule")
        if kind in {"confirmation", "validation"} and phase != "diagnostic":
            action = measurement_action(item, observation)
            if target not in (observation.get("eligible_targets") or {}).get(action, []):
                raise ValueError("measurement_target_not_eligible")
        refs = {r.get("artifact_id") for r in observation.get("artifact_index") or []
                if r.get("verification_status") == "verified"}
        for ref in item["required_context_refs"]:
            if ref in refs:
                continue
            if not isinstance(ref, str) or not ref.startswith("context://") or not observation.get("_campaign_path"):
                raise ValueError("required_context_ref_unverified")
            delivered = observation.get("_delivered_context_refs")
            if delivered is not None and ref not in delivered:
                raise ValueError("required_context_ref_not_delivered")
            from .context.sources import ContextSourceRegistry
            registry = ContextSourceRegistry(Path(observation["_campaign_path"]))
            source = registry.get(ref, role="research_planner")
            registry.load(source, role="research_planner")
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


def normalize_research_round(raw: Any, observation: dict, *, existing: list[dict] | None = None) -> dict:
    cfg = config(observation)
    try:
        draft = ResearchRoundDraft.model_validate(raw)
    except ValidationError as exc:
        return {"ok": False, "detail": "research_round_schema_invalid", "raw": copy.deepcopy(raw),
                "validation_records": [{"status": "rejected", "error": str(exc)}], "valid_proposals": []}
    proposals, records = copy.deepcopy(existing or []), []
    ids = {p["proposal_id"] for p in proposals}
    def signature(proposal):
        keys = ("task_kind", "target_candidate_ref", "scientific_intent", "evaluation_scope")
        if proposal["task_kind"] != "new_method":
            keys += ("question",)
        return digest({k: proposal.get(k) for k in keys})
    signatures = {signature(p) for p in proposals}
    for row in draft.proposals:
        proposal, trace = _validate_proposal(row, observation)
        if proposal is not None:
            operation_signature = signature(proposal)
            if proposal["proposal_id"] in ids or operation_signature in signatures:
                trace.update(status="rejected", error="duplicate_research_task")
            elif len(proposals) >= 4:
                trace.update(status="rejected", error="research_pool_at_maximum")
            else:
                proposals.append(proposal)
                ids.add(proposal["proposal_id"])
                signatures.add(operation_signature)
        records.append(trace)
    evaluations, evaluation_ids, ranks = [], set(), set()
    for row in draft.evaluations:
        try:
            item = ResearchPlanEvaluation.model_validate(row).model_dump()
            if item["proposal_id"] not in ids or item["proposal_id"] in evaluation_ids or item["rank"] in ranks:
                raise ValueError("evaluation_unknown_duplicate_or_conflicting_rank")
            evaluations.append(item)
            evaluation_ids.add(item["proposal_id"])
            ranks.add(item["rank"])
        except (ValidationError, ValueError) as exc:
            records.append({"status": "rejected", "error": str(exc), "evaluation": copy.deepcopy(row)})
    by_id = {p["proposal_id"]: p for p in proposals}
    ranked = sorted(evaluations, key=lambda e: e["rank"])
    preferred = draft.selected_proposal_id
    selected = preferred if preferred in by_id and preferred in evaluation_ids else (ranked[0]["proposal_id"] if ranked else None)
    # Explicit rankings, never container order, provide fallback evidence.
    count = len(proposals)
    status = "complete" if 2 <= count <= 4 else "degraded_singleton" if count == 1 and cfg.allow_degraded_singleton else "incomplete" if count else "no_feasible_plan"
    ok = bool(selected and status in {"complete", "degraded_singleton"})
    cards = observation.get("experiment_observations") or []
    if isinstance(cards, dict):
        cards = [r.get("body") or r for r in cards.get("cards") or []]
    round_record = {"round_id": round_id(observation), "input_observation_ids": [r.get("observation_id") for r in
        cards if r.get("observation_id")],
        "input_observation_digest": digest(round_input(observation)), "target_k": cfg.target_k,
        "proposals": proposals, "evaluations": evaluations, "selected_proposal_id": selected if ok else None,
        "pool_status": status, "valid_count": count, "shortfall_reason": None if count >= cfg.target_k else
        "independent_rejections_or_insufficient_distinct_tasks", "selection_rationale": draft.selection_rationale,
        "selection_source": "provider_selection_with_valid_evaluation" if selected == preferred else
            "valid_evaluation_rank_fallback" if selected else None,
        "validation_records": records, "raw_digest": digest(raw)}
    if observation.get("terminal_observation_id") and observation["terminal_observation_id"] not in round_record["input_observation_ids"]:
        round_record["input_observation_ids"].append(observation["terminal_observation_id"])
    return {"ok": ok, "detail": None if ok else "research_pool_" + status,
        "raw": copy.deepcopy(raw), "round": round_record, "valid_proposals": proposals,
        "selected_proposal": by_id.get(selected) if ok else None, "validation_records": records}


def decide_research_plan(observation: dict, backend) -> dict:
    cfg = config(observation)
    raw = backend(observation)
    history = [copy.deepcopy(raw)]
    parsed = normalize_research_round(raw, observation)
    if not parsed["ok"] and cfg.proposal_correction_calls:
        first_records = parsed.get("validation_records") or []
        request = {**observation, "plan_repair_request": {"kind": "correct_research_proposals",
            "valid_proposals": parsed.get("valid_proposals") or [],
            "rejected": parsed.get("validation_records") or [],
            "instruction": "Keep valid proposals. Correct failed tasks once and provide explicit evaluations and selection."}}
        corrected = backend(request)
        history.append(copy.deepcopy(corrected))
        parsed = normalize_research_round(corrected, observation, existing=parsed.get("valid_proposals"))
        parsed["validation_records"] = first_records + parsed.get("validation_records", [])
        if parsed.get("round"):
            parsed["round"]["validation_records"] = parsed["validation_records"]
    parsed["proposal_history"] = history
    return parsed
