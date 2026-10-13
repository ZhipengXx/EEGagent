"""Fixed two-fold development screening on one predeclared ten-fold suite.

No stage can grant a ten-fold score. All checkpoint, data, source and framework
checks remain owned by method_suite; phase receipts only authorize missing work.
"""
from __future__ import annotations

import random
import statistics
from pathlib import Path

from react_agent.eeg_research.agentic.loso_study import SUBJECTS, folds, identity, read, digest, write_once

SCREENING = "screening_2fold"
CONFIRMATION = "confirmation_10fold"
SCIENCE_SCOPE = {
    "evaluation_population": "fixed_development_gallery_subject_generalization",
    "checkpoint_selection": "source_subject_development_only",
    "held_out_eeg_fit_access": False, "benchmark_image_features_used_in_development": True,
    "independent_final_test": False, "selection_bias": "fixed_two_fold_selected_methods",
    "screening_folds_statistically_independent": False,
}


def freeze_selection(goal: dict) -> dict:
    from react_agent.eeg_research.agentic.schemas import ScreeningConfig
    config = ScreeningConfig.model_validate(goal.get("screening") or {})
    selected = list(config.subjects) if config.subjects is not None else random.Random(config.selection_seed).sample(list(SUBJECTS), 2)
    selected_ids = [row["fold_id"] for subject in selected for row in folds() if row["held_out_subject"] == subject]
    body = {"schema_version": "eeg_research.screening_selection.v1", "algorithm_version": "explicit_order.v1" if config.subjects is not None else "python_random_sample.v1",
            "selection": "explicit_subjects" if config.subjects is not None else config.selection,
            "selection_seed": config.selection_seed, "subjects": selected, "screening_fold_ids": selected_ids,
            "confirmation_fold_ids": [row["fold_id"] for row in folds() if row["fold_id"] not in selected_ids],
            "all_subjects": list(SUBJECTS), "selected_before_results": True}
    body["selection_manifest_hash"] = identity(body)
    return body


def validate_selection(body: dict) -> None:
    if body.get("selection_manifest_hash") != identity({key: value for key, value in body.items() if key != "selection_manifest_hash"}):
        raise ValueError("screening_selection_changed")
    subjects = body.get("subjects") or []
    expected = [row["fold_id"] for subject in subjects for row in folds() if row["held_out_subject"] == subject]
    if (len(subjects) != 2 or len(set(subjects)) != 2 or not set(subjects) <= set(SUBJECTS)
            or body.get("screening_fold_ids") != expected or body.get("all_subjects") != list(SUBJECTS)
            or body.get("confirmation_fold_ids") != [row["fold_id"] for row in folds() if row["fold_id"] not in expected]
            or body.get("selected_before_results") is not True):
        raise ValueError("screening_selection_invalid")


def canonical_fold_identity(common: dict, method: dict, fold: dict) -> dict:
    # Explicitly exclude only dispatch pointers. Scientific/execution recipe,
    # actual samples and all source/framework bytes remain in the identity.
    protocol = {key: value for key, value in fold["protocol"].items()
                if key not in {"fingerprint", "method_suite_id", "method_suite_ref", "method_revision", "fold_id", "benchmark_subject"}}
    body = {"schema_version": "eeg_research.canonical_fold_identity.v1", "held_out_subject": fold["held_out_subject"],
            "train_subjects": fold["train_subjects"], "seed": fold["seed"], "fidelity": "full",
            "protocol": protocol, "source_hash": method["source_hash"], "spec_hash": method["spec_hash"],
            "hook_spec": method["hook_spec"], "public_recipe": method["public_recipe"],
            "data_dependencies": common["data_dependencies"], "cache_dependencies": common["cache_dependencies"],
            "framework_dependencies": common["framework_dependencies"]}
    body["canonical_hash"] = identity(body)
    return body


def phase_fold_ids(common: dict, phase: str) -> list[str]:
    if phase == SCREENING:
        return list(common["selection_manifest"]["screening_fold_ids"])
    if phase == CONFIRMATION:
        return [row["fold_id"] for row in common["folds"]]
    raise ValueError("method_phase_unknown")


def suites_for_target(camp: Path, target: str) -> list[dict]:
    from react_agent.eeg_research.agentic import method_suite as ms
    return [ms.validate_suite(camp, path.parent.name) for path in sorted((camp / "suites").glob("*/suite_manifest.json"))
            if read(path).get("method", {}).get("candidate_id") == target]


def valid_scores(camp: Path, suite: dict, ids: list[str] | None = None) -> list[dict]:
    from react_agent.eeg_research.agentic import method_suite as ms
    # This validates real score/freezes/dependencies, not suite_state's counters.
    ms.validate_suite(camp, suite["suite_id"], verify_data=True)
    scores = []
    for fold in suite["folds"]:
        if ids is not None and fold["fold_id"] not in ids:
            continue
        path = camp / "suites" / suite["suite_id"] / fold["fold_id"] / "benchmark.json"
        if path.exists():
            score = read(path)
            ms.verify_score(camp, suite, score)
            ms.aggregate_records(suite, [score])
            scores.append(score)
    return scores


def phase_cost(common: dict, phase: str, candidate_completed: list[str], baseline_completed: list[str], *, target: str = "candidate") -> dict:
    ids = phase_fold_ids(common, phase)
    candidate_missing = [] if target == "baseline" else [fold for fold in ids if fold not in candidate_completed]
    baseline_missing = [fold for fold in ids if fold not in baseline_completed]
    jobs = len(candidate_missing) + len(baseline_missing)
    estimate = common.get("gpu_seconds_estimate_per_suite")
    per_fold = estimate / 10 if isinstance(estimate, (int, float)) else None
    return {"phase": phase, "missing_candidate_fold_ids": candidate_missing, "missing_baseline_fold_ids": baseline_missing,
            "missing_candidate_folds": len(candidate_missing), "missing_baseline_folds": len(baseline_missing),
            "candidate_reuse_count": 0 if target == "baseline" else len(ids) - len(candidate_missing),
            "baseline_reuse_count": len(ids) - len(baseline_missing), "training_jobs": jobs,
            "gpu_seconds": jobs * per_fold if per_fold is not None else None,
            "gpu_estimate_basis": "frozen_suite_estimate_per_fold" if per_fold is not None else "unknown_requires_frozen_estimate",
            "feedback_report_count": 1 + int(target != "baseline" and bool(baseline_missing)),
            "conditional_api_review_repair_cost": "unknown_within_existing_authorized_caps",
            "confirmation_additional_jobs": (20 - len(set(candidate_completed) | set(candidate_missing))
                                              - len(set(baseline_completed) | set(baseline_missing))) if target != "baseline" else 10 - len(ids),
            "estimates_are_execution_authority": False}


def dry_run(common: dict, *, candidate_completed: list[str] | None = None, baseline_completed: list[str] | None = None) -> dict:
    validate_selection(common["selection_manifest"])
    candidate_completed, baseline_completed = candidate_completed or [], baseline_completed or []
    selected = common["selection_manifest"]["screening_fold_ids"]
    return {"evaluation_schedule": "screen_then_confirm", "selection_manifest": common["selection_manifest"],
            "screening_folds": [{"fold_id": row["fold_id"], "held_out_subject": row["held_out_subject"],
                                  "train_subjects": row["train_subjects"], "epochs": 50, "seed": 0, "fidelity": "full"}
                                 for row in common["folds"] if row["fold_id"] in selected],
            "confirmation_remaining_fold_ids": common["selection_manifest"]["confirmation_fold_ids"],
            "screening_cost": phase_cost(common, SCREENING, candidate_completed, baseline_completed),
            "confirmation_cost_after_screening": phase_cost(common, CONFIRMATION,
                list(set(candidate_completed) | set(selected)), list(set(baseline_completed) | set(selected))),
            "science_scope": SCIENCE_SCOPE}


def planner_phase_targets(camp: Path, state: dict) -> dict:
    """Read-only phase feasibility; actual launch revalidates the same identities."""
    from react_agent.eeg_research.agentic import method_suite as ms
    common = ms.manifest(camp)
    choices = {"run_screening": [], "run_confirmation": []}
    if common.get("evaluation_schedule") != "screen_then_confirm" or state.get("active_suite"):
        return choices
    controls = suites_for_target(camp, "baseline")
    b_done = [row["fold_id"] for row in valid_scores(camp, controls[-1])] if controls else []
    cost = read(camp / "cost.json") if (camp / "cost.json").exists() else {}
    used_jobs = max(state.get("training_jobs", 0), cost.get("training_jobs", 0))
    calls_left = state["max_llm_calls"] - max(state.get("llm_calls", 0), cost.get("llm_calls", 0))
    feedback_calls = int((common.get("confirmation_policy") or {}).get("feedback_llm_calls_reserved", 2))
    targets = {"baseline"}
    targets.update(row["candidate_id"] for row in state.get("candidates", [])
                   if row.get("ready_for_train") and row.get("candidate_id"))
    if state.get("candidate_ready") and state.get("candidate_id"):
        targets.add(state["candidate_id"])
    for target in sorted(targets):
        if target in state.get("detached_candidates", []):
            continue
        try:
            candidates = suites_for_target(camp, target)
            c_done = [row["fold_id"] for row in valid_scores(camp, candidates[-1])] if candidates else []
            phase = SCREENING
            if target != "baseline" and candidates and (camp / "suites" / candidates[-1]["suite_id"] / "screening.json").exists():
                result = read_screening(camp, candidates[-1]["suite_id"])
                if len(c_done) == 10:
                    continue
                if result["screening_result"] != "screening_passed":
                    try:
                        validate_exploration_receipt(camp, candidates[-1])
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                phase = CONFIRMATION
            elif target == "baseline" and set(phase_fold_ids(common, SCREENING)) <= set(b_done):
                continue
            estimate = phase_cost(common, phase, c_done, b_done, target=target)
            if (used_jobs + estimate["training_jobs"] <= state["max_training_jobs"]
                    and float(cost.get("gpu_seconds_used", 0)) + estimate["gpu_seconds"] <= state["max_gpu_seconds"]
                    and calls_left >= feedback_calls * estimate["feedback_report_count"]):
                choices["run_confirmation" if phase == CONFIRMATION else "run_screening"].append(target)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return choices


def screening_result(common: dict, suite: dict, scores: list[dict], baseline_suite: dict | None, baseline_scores: list[dict]) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    ids = phase_fold_ids(common, SCREENING)
    selected = [row for row in scores if row["fold_id"] in ids]
    control = [row for row in baseline_scores if row["fold_id"] in ids]
    ms.aggregate_records(suite, selected)
    if baseline_suite is not None:
        ms.aggregate_records(baseline_suite, control)
    body = {"schema_version": "eeg_research.method_screening.v1", "scope": "two_fold_development", "phase": SCREENING,
            "coverage": len(selected), "required_fold_count": 2, "full_required_fold_count": 10,
            "screening_subjects": common["selection_manifest"]["subjects"],
            "screening_fold_ids": ids, "selection_manifest_hash": common["selection_manifest"]["selection_manifest_hash"],
            "candidate_id": suite["method"]["candidate_id"], "suite_id": suite["suite_id"], "suite_hash": suite["suite_hash"],
            "method_revision": suite["method"]["method_revision"], "seed": 0, "fidelity": "full",
            "scores": selected, "baseline_scores": control, "independent_final_test": False, "science_scope": SCIENCE_SCOPE,
            "confirmation_admission": None, "screening_result": "inconclusive", "status": "partial"}
    if len(selected) != 2 or suite["method"]["candidate_id"] != "baseline" and len(control) != 2:
        return body
    body.update(status="complete", screening_mean_top1=statistics.mean(row["metrics"]["top1"] for row in selected),
                screening_mean_top5=statistics.mean(row["metrics"]["top5"] for row in selected))
    if suite["method"]["candidate_id"] == "baseline":
        body["screening_result"] = "baseline_reference"
        return body
    if baseline_suite is None or baseline_suite["method"]["public_recipe"] != suite["method"]["public_recipe"]:
        raise ValueError("screening_matched_baseline_recipe_mismatch")
    controls = {row["held_out_subject"]: row for row in control}
    paired = []
    for row in selected:
        b = controls[row["held_out_subject"]]
        if (row["train_subjects"] != b["train_subjects"] or row["evaluator_hash"] != b["evaluator_hash"]
                or row["seed"] != b["seed"] or row["fidelity"] != b["fidelity"]):
            raise ValueError("screening_control_fold_identity_mismatch")
        paired.append({"subject": row["held_out_subject"], "candidate_top1": row["metrics"]["top1"],
                       "baseline_top1": b["metrics"]["top1"], "delta_pp": 100 * (row["metrics"]["top1"] - b["metrics"]["top1"])})
    gain = statistics.mean(row["delta_pp"] for row in paired)
    threshold = float((common.get("screening_policy") or {}).get("min_mean_gain_pp", 0.5))
    body.update(paired_scores=paired, screening_mean_delta_pp=gain, min_mean_gain_pp=threshold,
                screening_result="screening_passed" if gain >= threshold else "not_promoted",
                promotion_decision={"reason": "fixed_two_fold_paired_mean_threshold", "threshold_pp": threshold,
                                    "statistical_significance": "not_established"}, baseline_suite_hash=baseline_suite["suite_hash"])
    return body


def _prepare(camp: Path, state: dict, target: str) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    previous = state.get("active_suite")
    state["active_suite"] = None
    try:
        suite_id = ms._begin_full(camp, state, target, phase_preparation=True)
        return ms.validate_suite(camp, suite_id, verify_data=True)
    finally:
        state["active_suite"] = previous


def _awaiting_admission(camp: Path, candidates: list[dict], reason: str) -> None:
    from react_agent.eeg_research.agentic.loop import _write
    if not candidates:
        return
    path = camp / "suites" / candidates[-1]["suite_id"] / "suite_state.json"
    progress = read(path)
    progress.update(scheduling_status="awaiting_budget_or_resources", confirmation_admission=None,
                    admission_wait_reason=reason)
    _write(path, progress)


def reconcile_api_usage(camp: Path, state: dict, *, block: bool = True) -> dict:
    """Reconcile the same native API journals used by role execution.

    A paid/unknown outcome may not authorize new training. An explicit native
    disposition permits continuing while retaining its original charged call.
    Legacy counter-only fixtures/ledgers are not replaced by an empty journal.
    """
    from react_agent.eeg_research.agentic.llm import API_INTENTS, LlmUnavailable, refresh_method_api_cost
    from react_agent.eeg_research.agentic.api_policy import ATTEMPTS, unresolved_charge_ids
    from react_agent.eeg_research.agentic.loop import _write
    cost = read(camp / "cost.json") if (camp / "cost.json").exists() else {}
    if any((camp / name).exists() for name in (API_INTENTS, ATTEMPTS, "llm_calls.jsonl")) or cost.get("api_intent_accounting"):
        cost = refresh_method_api_cost(camp)
    state["llm_calls"] = max(int(state.get("llm_calls", 0)), int(cost.get("llm_calls", 0)))
    state["llm_calls_left"] = state["max_llm_calls"] - state["llm_calls"]
    pending = list(cost.get("api_unsettled_call_ids") or [])
    unknown = sorted(unresolved_charge_ids(camp))
    waiting = bool(pending or unknown)
    state["api_unsettled_call_ids"] = pending
    if waiting:
        reason = "unsettled_api_intent_requires_recovery" if pending else "unknown_api_charge_requires_disposition"
        state["awaiting_reconciliation"] = {"kind": "api_outcome", "reason": reason,
                                             "call_ids": sorted(set(pending + unknown)), "charges_retained": True}
    else:
        reason = None
        state.pop("awaiting_reconciliation", None)
    suite_id = state.get("phase_target_suite") or state.get("active_suite")
    progress_path = camp / "suites" / suite_id / "suite_state.json" if suite_id else None
    if progress_path and progress_path.exists():
        progress = read(progress_path)
        if waiting:
            if progress.get("scheduling_status") != "awaiting_reconciliation":
                progress["scheduling_status_before_reconciliation"] = progress.get("scheduling_status")
            progress.update(scheduling_status="awaiting_reconciliation", reconciliation_reason=reason)
        elif progress.get("scheduling_status") == "awaiting_reconciliation":
            progress["scheduling_status"] = progress.pop("scheduling_status_before_reconciliation", None)
            progress.pop("reconciliation_reason", None)
        _write(progress_path, progress)
    if waiting and block:
        raise LlmUnavailable(reason, diagnostics=state["awaiting_reconciliation"])
    return cost


def _latest_control(camp: Path, state: dict) -> None:
    control = read(camp / "control.json") if (camp / "control.json").exists() else {}
    if control.get("action") == "stop":
        state.update(status="cancelled", stop_reason="user_cancelled")
    elif control.get("action") == "pause":
        state.update(status="paused", pause_after_step=True)
    if state.get("status") in {"paused", "cancelled", "completed"} or state.get("pause_after_step"):
        raise ValueError("phase_launch_user_stopped")


def _phase_projection(camp: Path, common: dict, suite: dict, baseline: dict, receipt: dict,
                      candidate_scores: list[dict], baseline_scores: list[dict]) -> dict:
    """Display only verified folds completed before this admission's work list.

    Newly completed folds in the current phase are not relabelled as cache hits
    when recovering. The receipt binds the original cost estimate and missing
    work; checkpoint/source/canonical validation remains in valid_scores.
    """
    request = receipt["request"]
    ids = phase_fold_ids(common, request["phase"])
    baseline_target = suite["method"]["candidate_id"] == "baseline"
    work = request["fold_work"]["baseline" if baseline_target else "candidate"]
    own_scores = baseline_scores if baseline_target else candidate_scores
    by_fold = {row["fold_id"]: row for row in own_scores}
    reused_ids = [fold for fold in ids if fold not in work and fold in by_fold]
    baseline_by_fold = {row["fold_id"]: row for row in baseline_scores}
    baseline_reused = [fold for fold in ids if fold not in request["fold_work"]["baseline"] and fold in baseline_by_fold]
    estimate = request.get("phase_cost_estimate")
    if not isinstance(estimate, dict) or estimate.get("training_jobs") != request["training_jobs"]:
        # Older receipt representations did not carry the complete display
        # estimate. Derive a current verified estimate with an explicit label.
        estimate = phase_cost(common, request["phase"], [row["fold_id"] for row in candidate_scores],
                              [row["fold_id"] for row in baseline_scores],
                              target="baseline" if request["candidate_suite_id"] == request["baseline_suite_id"] else "candidate")
        estimate = {**estimate, "estimate_scope": "current_verified_remaining_score_work"}
    else:
        estimate = {**estimate, "estimate_scope": "frozen_admission_estimate_from_verified_fold_records"}
    records = []
    for fold_id in reused_ids:
        score = by_fold[fold_id]
        fold = next(row for row in suite["folds"] if row["fold_id"] == fold_id)
        if score.get("canonical_fold_hash") != fold["canonical_fold_identity"]["canonical_hash"]:
            raise ValueError("phase_reuse_canonical_identity_mismatch")
        score_path = camp / "suites" / suite["suite_id"] / fold_id / "benchmark.json"
        records.append({"fold_id": fold_id, "held_out_subject": score["held_out_subject"],
                        "canonical_fold_hash": score["canonical_fold_hash"], "score_ref": str(score_path),
                        "score_sha256": digest(score_path), "method_revision": suite["method"]["method_revision"],
                        "reuse_kind": "verified_completed_full_fold_before_current_admission"})
    return {"reused_fold_ids": reused_ids, "reused_fold_records": records, "baseline_reused_fold_ids": baseline_reused,
            "phase_cost": {**estimate, "phase_receipt_id": receipt["receipt_id"], "phase_receipt_hash": receipt["request_hash"],
                           "cost_scope": "paired_candidate_and_baseline_phase", "suite_missing_folds_at_admission": len(work),
                           "global_actual_cost_ledger": str(camp / "cost.json"), "estimate_is_actual_charge": False}}


def begin_phase(camp: Path, state: dict, target: str, *, phase: str | None = None) -> str:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.budget import reserve_phase
    from react_agent.eeg_research.agentic.loop import _write
    common = ms.manifest(camp)
    phase = phase or SCREENING
    _latest_control(camp, state)
    cost_ledger = reconcile_api_usage(camp, state)
    if state.get("active_suite"):
        raise ValueError("method_suite_already_active")
    if target in state.get("detached_candidates", []):
        raise ValueError("method_target_detached")
    # Refuse an unaffordable phase before spending on a preparation probe.
    prior_c = suites_for_target(camp, target)
    prior_b = suites_for_target(camp, "baseline")
    completed_c = [row["fold_id"] for row in valid_scores(camp, prior_c[-1])] if prior_c else []
    completed_b = [row["fold_id"] for row in valid_scores(camp, prior_b[-1])] if prior_b else []
    preview = phase_cost(common, phase, completed_c, completed_b, target=target)
    if max(state["training_jobs"], cost_ledger.get("training_jobs", 0)) + preview["training_jobs"] > state["max_training_jobs"]:
        if phase == CONFIRMATION:
            _awaiting_admission(camp, prior_c, "phase_training_job_reservation_unavailable")
        raise ValueError("phase_training_job_reservation_unavailable")
    if float(cost_ledger.get("gpu_seconds_used", 0)) + preview["gpu_seconds"] > state["max_gpu_seconds"]:
        if phase == CONFIRMATION:
            _awaiting_admission(camp, prior_c, "phase_gpu_reservation_unavailable")
        raise ValueError("phase_gpu_reservation_unavailable")
    feedback_per_report = int((common.get("confirmation_policy") or {}).get("feedback_llm_calls_reserved", 2))
    if max(state.get("llm_calls", 0), cost_ledger.get("llm_calls", 0)) + feedback_per_report * preview["feedback_report_count"] > state["max_llm_calls"]:
        if phase == CONFIRMATION:
            _awaiting_admission(camp, prior_c, "phase_feedback_api_reservation_unavailable")
        raise ValueError("phase_feedback_api_reservation_unavailable")
    suite = _prepare(camp, state, target)
    baseline = suite if target == "baseline" else _prepare(camp, state, "baseline")
    c_scores = valid_scores(camp, suite)
    b_scores = valid_scores(camp, baseline)
    result = None
    if phase == CONFIRMATION:
        if target == "baseline":
            raise ValueError("baseline_confirmation_requires_candidate_admission")
        result = read_screening(camp, suite["suite_id"])
        if result["screening_result"] != "screening_passed":
            validate_exploration_receipt(camp, suite)
    cost = phase_cost(common, phase, [row["fold_id"] for row in c_scores], [row["fold_id"] for row in b_scores], target=target)
    request = {"phase": phase, "candidate_suite_id": suite["suite_id"], "baseline_suite_id": baseline["suite_id"],
               "suite_hash": suite["suite_hash"], "method_revision": suite["method"]["method_revision"],
               "baseline_suite_hash": baseline["suite_hash"], "selection_manifest_hash": common["selection_manifest"]["selection_manifest_hash"],
               "training_jobs": cost["training_jobs"], "gpu_seconds": cost["gpu_seconds"],
               "llm_calls": feedback_per_report * cost["feedback_report_count"], "feedback_allocation_per_report": feedback_per_report,
               "phase_cost_estimate": cost,
               "fold_work": {"candidate": cost["missing_candidate_fold_ids"], "baseline": cost["missing_baseline_fold_ids"]}}
    if result is not None and result["screening_result"] != "screening_passed":
        request["exploration_receipt_id"] = validate_exploration_receipt(camp, suite)["receipt_id"]
    path = camp / "suites" / suite["suite_id"] / "suite_state.json"
    progress = read(path)
    try:
        receipt = reserve_phase(camp, state, request)
    except ValueError:
        if phase == CONFIRMATION:
            progress.update(screening_result=result["screening_result"], scheduling_status="awaiting_budget_or_resources",
                            confirmation_admission=None)
            _write(path, progress)
        raise
    progress.update(phase=phase, screening_fold_ids=phase_fold_ids(common, SCREENING),
                    frozen_method_revision=suite["method"]["method_revision"], status="pending",
                    phase_receipt_id=receipt["receipt_id"], phase_fold_ids=phase_fold_ids(common, phase),
                    pending_fold_ids=cost["missing_candidate_fold_ids"], baseline_suite_id=baseline["suite_id"],
                    confirmation_admission=receipt if phase == CONFIRMATION else None,
                    scheduling_status="confirmation_admitted" if phase == CONFIRMATION else "screening")
    progress.update(_phase_projection(camp, common, suite, baseline, receipt, c_scores, b_scores))
    _write(path, progress)
    state.update(phase=phase, phase_target_suite=suite["suite_id"], phase_reservation_id=receipt["receipt_id"],
                 screening_fold_ids=phase_fold_ids(common, SCREENING), status="training")
    if cost["missing_baseline_fold_ids"]:
        bpath = camp / "suites" / baseline["suite_id"] / "suite_state.json"
        bp = read(bpath)
        bp.update(phase=phase, phase_fold_ids=phase_fold_ids(common, phase), phase_receipt_id=receipt["receipt_id"],
                  phase_owner_suite=suite["suite_id"], status="pending", pending_fold_ids=cost["missing_baseline_fold_ids"])
        bp.update(_phase_projection(camp, common, baseline, baseline, receipt, c_scores, b_scores))
        _write(bpath, bp)
        state["active_suite"] = baseline["suite_id"]
    else:
        state["active_suite"] = suite["suite_id"]
    return suite["suite_id"]


def before_launch(camp: Path, state: dict, suite: dict, fold: dict) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.budget import phase_reservation
    _latest_control(camp, state)
    receipt = phase_reservation(camp, state["phase_reservation_id"])
    if receipt["status"] != "active":
        raise ValueError("phase_launch_reservation_released")
    ms.validate_suite(camp, suite["suite_id"], verify_data=True)
    req = receipt["request"]
    owner = ms.validate_suite(camp, req["candidate_suite_id"], verify_data=True)
    if owner["suite_hash"] != req["suite_hash"] or owner["method"]["method_revision"] != req["method_revision"]:
        raise ValueError("phase_launch_method_identity_changed")
    if req["phase"] == CONFIRMATION:
        result = read_screening(camp, owner["suite_id"])
        if result["screening_result"] != "screening_passed":
            if validate_exploration_receipt(camp, owner)["receipt_id"] != req.get("exploration_receipt_id"):
                raise ValueError("phase_launch_exploration_identity_changed")
    key = "baseline" if suite["method"]["candidate_id"] == "baseline" else "candidate"
    if fold["fold_id"] not in req["fold_work"][key] or suite["suite_id"] != req[key + "_suite_id"]:
        raise ValueError("phase_launch_not_admitted")
    # Identity checks can take time. Re-read control, API journals and the
    # admission allocation after verification, immediately before launch intent.
    _latest_control(camp, state)
    cost = reconcile_api_usage(camp, state)
    receipt = phase_reservation(camp, receipt["receipt_id"])
    if receipt["status"] != "active":
        raise ValueError("phase_launch_reservation_released")
    job_ids = {path.parent.name for pattern in ("*/launch_intent.json", "*/job.json")
               for path in (camp / "jobs").glob(pattern)}
    used_jobs = max(state["training_jobs"], cost.get("training_jobs", 0), len(job_ids))
    if (used_jobs + receipt["remaining_training_jobs"] > state["max_training_jobs"]
            or float(cost.get("gpu_seconds_used", 0)) + receipt["remaining_gpu_seconds"] > state["max_gpu_seconds"]
            or max(state.get("llm_calls", 0), cost.get("llm_calls", 0)) + receipt["remaining_llm_calls"] > state["max_llm_calls"]):
        from react_agent.eeg_research.agentic.loop import _write
        path = camp / "suites" / req["candidate_suite_id"] / "suite_state.json"
        progress = read(path)
        progress.update(scheduling_status="awaiting_budget_or_resources", admission_wait_reason="phase_launch_budget_changed")
        _write(path, progress)
        raise ValueError("phase_launch_budget_changed")
    return receipt


def commit_screening(camp: Path, state: dict, suite: dict) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.artifacts import register
    common = ms.manifest(camp)
    controls = suites_for_target(camp, "baseline")
    baseline = controls[-1] if controls else None
    scores = valid_scores(camp, suite, phase_fold_ids(common, SCREENING))
    b_scores = valid_scores(camp, baseline, phase_fold_ids(common, SCREENING)) if baseline is not None else []
    result = screening_result(common, suite, scores, baseline, b_scores)
    if result["status"] != "complete":
        raise ValueError("method_screening_requires_two_valid_matched_folds")
    path = camp / "suites" / suite["suite_id"] / "screening.json"
    deps = [{"path": str(camp / "suites" / s["suite_id"] / row["fold_id"] / "benchmark.json"),
             "sha256": digest(camp / "suites" / s["suite_id"] / row["fold_id"] / "benchmark.json")}
            for s, rows in ((suite, scores), (baseline, b_scores)) if s is not None for row in rows]
    wrapper = {"screening": result, "score_dependencies": list({row["path"]: row for row in deps}.values())}
    if not path.exists():
        write_once(path, wrapper)
    else:
        read_screening(camp, suite["suite_id"])
    eid = "ev_screening_" + suite["suite_id"]
    existing = next((row for row in state.get("evidence", []) if row.get("evidence_id") == eid), None)
    if existing:
        return existing
    artifact = register(camp, path, kind="method_screening", candidate_id=result["candidate_id"],
                        artifact_id="art_screening_" + suite["suite_hash"][:24])
    row = {"evidence_id": eid, "kind": "method_screening", "candidate_id": result["candidate_id"],
           "suite_id": suite["suite_id"], "suite_hash": suite["suite_hash"], "suite_ref": str(path),
           "result_hash": digest(path), "artifact_refs": [artifact["artifact_id"]], "screening": result,
           "scope": "two_fold_development", "metric_scope": "two_fold_development", "phase": SCREENING,
           "coverage": 2, "required_fold_count": 2, "full_required_fold_count": 10, "evaluation_valid": True,
           "source_hash": suite["method"]["source_hash"], "spec_hash": suite["method"]["spec_hash"],
           "method_revision": suite["method"]["method_revision"], "seed": 0, "fidelity": "full",
           "execution_fingerprint": state["execution_fingerprint"], "contract_fingerprint": state["contract_fingerprint"],
           "independent_final_test": False, "science_scope": SCIENCE_SCOPE}
    state.setdefault("evidence", []).append(row)
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode
    ep = episode(task_hash=state["goal_id"], candidate_id=result["candidate_id"], kind="exploratory_result",
                 fidelity="full", seed=0, metric=None, contract_fingerprint=state["contract_fingerprint"],
                 artifact=str(path), episode_id="ep_screening_" + suite["suite_hash"][:24], job_id=suite["suite_id"] + "::screening_2fold")
    ep.update(scope="two_fold_development", screening_mean_top1=result["screening_mean_top1"],
              screening_subjects=result["screening_subjects"], selection_manifest_hash=result["selection_manifest_hash"],
              method_revision=result["method_revision"], screening_result=result["screening_result"],
              science_scope=SCIENCE_SCOPE, training_recipe=suite["method"]["public_recipe"])
    stored = EpisodeStore(camp).persist_episode(ep)
    state["memory"] = [item for item in state.get("memory", []) if item.get("episode_id") != stored["episode_id"]] + [stored]
    return row


def read_screening(camp: Path, suite_id: str) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    suite = ms.validate_suite(camp, suite_id, verify_data=True)
    wrapper = read(camp / "suites" / suite_id / "screening.json")
    ms.verify_dependencies(wrapper["score_dependencies"])
    baseline_hash = wrapper["screening"].get("baseline_suite_hash")
    baseline = suite if suite["method"]["candidate_id"] == "baseline" else next(
        (s for s in suites_for_target(camp, "baseline") if s["suite_hash"] == baseline_hash), None)
    scores = valid_scores(camp, suite, phase_fold_ids(ms.manifest(camp), SCREENING))
    b_scores = valid_scores(camp, baseline, phase_fold_ids(ms.manifest(camp), SCREENING)) if baseline is not None else []
    actual = screening_result(ms.manifest(camp), suite, scores, baseline, b_scores)
    if actual != wrapper["screening"] or actual["status"] != "complete":
        raise ValueError("method_screening_result_changed")
    return actual


def verified_screening_records(camp: Path, state: dict) -> list[dict]:
    rows = []
    for row in state.get("evidence", []):
        if row.get("kind") != "method_screening" or not row.get("evaluation_valid"):
            continue
        try:
            if read_screening(camp, row["suite_id"]) == row["screening"] and digest(Path(row["suite_ref"])) == row["result_hash"]:
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def finish_phase(camp: Path, state: dict, suite: dict, progress: dict, services: dict) -> bool:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.budget import release_phase
    from react_agent.eeg_research.agentic.loop import _write, save_state
    owner = progress.get("phase_owner_suite")
    phase = progress["phase"]
    if suite["method"]["candidate_id"] == "baseline" and phase == CONFIRMATION:
        row = ms.commit_aggregate(camp, state, suite)
    else:
        row = commit_screening(camp, state, suite) if phase == SCREENING else ms.commit_aggregate(camp, state, suite)
    progress.update(status="screened" if phase == SCREENING else "complete", pending_fold_ids=[])
    if phase == SCREENING:
        progress["screening_result"] = row["screening"]["screening_result"]
        progress["scheduling_status"] = "screening_passed" if progress["screening_result"] == "screening_passed" else progress["screening_result"]
    _write(camp / "suites" / suite["suite_id"] / "suite_state.json", progress)
    if owner and owner != suite["suite_id"]:
        save_state(camp, state)
        ms.complete_scientific_feedback(camp, state, row, services)
        from react_agent.eeg_research.agentic.budget import settle_phase_feedback, phase_reservation
        receipt = phase_reservation(camp, progress["phase_receipt_id"])
        settle_phase_feedback(camp, receipt["receipt_id"], row["evidence_id"], receipt["request"]["feedback_allocation_per_report"])
        state.update(active_suite=owner, live_job=None, status="training")
        save_state(camp, state)
        return True
    release_phase(camp, state, progress["phase_receipt_id"], "phase_complete")
    state.update(active_suite=None, live_job=None, phase_target_suite=None, status="analyzing")
    save_state(camp, state)
    # Required analysis is bound to the native evidence. Optional curation may
    # stay pending without granting fabricated memory completion.
    ms.complete_scientific_feedback(camp, state, row, services)
    state["status"] = "created"
    save_state(camp, state)
    return True


def advance(camp: Path, state: dict, services: dict) -> bool:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.loop import save_state
    if state.get("status") in {"paused", "cancelled", "completed"} or state.get("pause_after_step"):
        return False
    if state.get("active_suite"):
        return ms._advance_full(camp, state, services)
    if not any(row.get("candidate_id") == "baseline" for row in verified_screening_records(camp, state)):
        begin_phase(camp, state, "baseline")
        save_state(camp, state)
        return True
    rows = [*verified_screening_records(camp, state), *ms.verified_suite_records(camp, state)]
    for row in rows:
        if "analysis" not in ms.scientific_feedback_records(camp, state, row):
            ms.complete_scientific_feedback(camp, state, row, services)
            state["status"] = "created"
            save_state(camp, state)
            return True
    return False


def audit_evidence(camp: Path, state: dict) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.handoffs import development_artifact, development_view, method_suite_scalar_view
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    common = ms.manifest(camp)
    screenings = verified_screening_records(camp, state)
    full = ms.verified_suite_records(camp, state)
    declared = [row for row in state.get("evidence", []) if row.get("kind") in {"method_screening", "method_suite"} and row.get("evaluation_valid")]
    if len(screenings) + len(full) != len(declared):
        raise ValueError("method_audit_has_invalid_phase_evidence")
    deps, inputs, feedbacks = {}, [], []
    methods, views = [], []
    def add(path: Path, expected: str | None = None, kind: str = "method_dependency"):
        path = path.resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("phase_audit_dependency_changed")
        deps[str(path)] = {"path": str(path), "kind": kind, "content_hash": actual,
                           "verification_status": "verified", "scope": "development", "data_use": "authorized_method_development_benchmark"}
    add(camp / ms.MANIFEST, kind="method_search_manifest")
    for dep in [*common["framework_dependencies"], *common["data_dependencies"], *common["cache_dependencies"]]:
        add(Path(dep["path"]), dep["sha256"])
    for row in [*screenings, *full]:
        suite = ms.validate_suite(camp, row["suite_id"], verify_data=True)
        path = Path(row["suite_ref"])
        wrapper = read(path)
        (views if row["kind"] == "method_screening" else methods).append(
            row if row["kind"] == "method_screening" else method_suite_scalar_view(row, score_dependencies=wrapper["score_dependencies"]))
        add(path, row["result_hash"], row["kind"])
        inputs.append(path)
        add(camp / "suites" / suite["suite_id"] / "suite_manifest.json", kind="method_suite_manifest")
        for dep in suite["method"]["dependencies"]:
            add(Path(dep["path"]), dep["sha256"])
        for dep in wrapper["score_dependencies"]:
            add(Path(dep["path"]), dep["sha256"], "method_fold_benchmark")
            score = read(Path(dep["path"]))
            freeze = read(Path(score["freeze_ref"]))
            add(Path(score["freeze_ref"]), score["freeze_sha256"], "method_fold_freeze")
            for bound in freeze["dependencies"]:
                add(Path(bound["path"]), bound["sha256"])
        feedback = ms.scientific_feedback_records(camp, state, row)
        if "analysis" not in feedback:
            raise ValueError("phase_audit_required_analysis_incomplete")
        for stage, record in feedback.items():
            verified = development_artifact(camp, record["artifact_id"])
            artifact = resolve_verified_artifact(camp, verified["artifact_id"])
            add(Path(artifact["path"]), verified["sha256"], stage)
            inputs.append(Path(artifact["path"]))
            feedbacks.append({"stage": stage, "suite_id": suite["suite_id"], "scope": row.get("scope", "method_development_benchmark"),
                              "artifact_id": verified["artifact_id"], "native_origin_verified": True,
                              "envelope": development_view(read(Path(artifact["path"])))})
    return {"methods": methods, "screenings": views, "latest": (methods or views)[-1] if methods or views else None,
            "dependency_manifest": sorted(deps.values(), key=lambda row: row["path"]), "inputs": inputs,
            "scientific_feedback": feedbacks, "scope": "staged_method_development", "science_scope": SCIENCE_SCOPE,
            "independent_seed_count": 1 if declared else 0, "replicated": False, "independent_final_test": False,
            "statistical_significance": "not_established"}


def progress_context(camp: Path, state: dict, goal: dict, budget: dict, actions: list[str]) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    screenings = verified_screening_records(camp, state)
    complete = ms.verified_suite_records(camp, state)
    paired = [row for row in complete if row.get("candidate_id") != "baseline" and (row.get("comparison") or {}).get("comparable")]
    phase_targets = planner_phase_targets(camp, state)
    incomplete = []
    for path in sorted((camp / "suites").glob("*/suite_state.json")):
        suite = ms.validate_suite(camp, path.parent.name, verify_data=True)
        if suite["method"]["candidate_id"] in state.get("detached_candidates", []):
            continue
        progress = read(path)
        scores = valid_scores(camp, suite)
        ids = {row["fold_id"] for row in scores}
        selected = set(ms.manifest(camp)["selection_manifest"]["screening_fold_ids"])
        if len(ids) < 10:
            incomplete.append({"suite_id": suite["suite_id"], "candidate_id": suite["method"]["candidate_id"],
                               "phase": progress.get("phase", "preflight"), "screening_coverage": len(ids & selected),
                               "screening_required_fold_count": 2, "full_coverage": len(ids), "full_required_fold_count": 10,
                               "screening_result": progress.get("screening_result"), "scheduling_status": progress.get("scheduling_status"),
                               "confirmation_admission": progress.get("confirmation_admission"),
                               "completed_fold_ids": sorted(ids), "pending_fold_ids": progress.get("pending_fold_ids", [])})
    minimum = min((phase_cost(ms.manifest(camp), CONFIRMATION if action == "run_confirmation" else SCREENING,
                  [row["fold_id"] for row in valid_scores(camp, suites_for_target(camp, target)[-1])] if suites_for_target(camp, target) else [],
                  [row["fold_id"] for row in valid_scores(camp, suites_for_target(camp, "baseline")[-1])] if suites_for_target(camp, "baseline") else [],
                  target=target)["training_jobs"] for action, targets in phase_targets.items() for target in targets), default=None)
    legal = [action for action in actions if action in {"design_experiment", "propose_experiment", "implement_candidate", "repair_candidate", "run_screening", "run_confirmation", "run_full", "diagnose_results"}]
    return {"evaluation_mode": "loso_method_search", "evaluation_schedule": "screen_then_confirm",
            "primary_metric": goal["primary_metric"], "unit": "staged_method_development", "screenings": screenings,
            "complete_methods": [{key: row.get(key) for key in ("candidate_id", "method_revision", "suite_id", "seed", "aggregate", "comparison", "promotion")} for row in complete],
            "active_suite": state.get("active_suite"), "phase_progress": incomplete, "independent_seed_count": 1 if complete or screenings else 0,
            "replicated": False, "candidate_suite_job_lower_bound": 2, "partial_performance_evidence": "fixed_two_fold_only",
            "required_method_pair_complete": bool(paired), "paired_complete_method_count": len(paired),
            "unfinished_requirements": [] if paired else [{"family": "complete_method_pair", "reason": "Two-fold feedback permits development actions; it does not prove ten-fold method confirmation."}],
            "budget_can_continue": bool(legal), "minimum_pair_remaining_job_lower_bound": minimum,
            "method_suite_gpu_seconds_estimate": ms.manifest(camp)["gpu_seconds_estimate_per_suite"],
            "legal_research_next_actions": legal, "phase_eligible_targets": phase_targets,
            "training_block_reason": None if any(phase_targets.values()) else "no_currently_affordable_phase_target",
            "new_design_block_reason": None, "remaining_resources": budget, "science_scope": SCIENCE_SCOPE,
            "limitations": ["fixed_development_gallery", "two_fold_selection_bias", "no_independent_final_test", "no_statistical_significance"]}


def recover(camp: Path, state: dict) -> dict:
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.budget import phase_reservation, consume_phase_job
    from react_agent.eeg_research.agentic.loop import _write
    from react_agent.eeg_research.agentic.multigpu_check import recover_probes
    recover_probes(camp, state)
    # Reconcile billing while preserving paid job/probe reconstruction. The
    # launch boundary blocks new GPU work until all unknown API outcomes settle.
    reconcile_api_usage(camp, state, block=False)
    reservations = read(camp / "phase_reservations.json") if (camp / "phase_reservations.json").exists() else {}
    for saved in (reservations.get("receipts") or {}).values():
        if saved["status"] != "active":
            continue
        receipt = phase_reservation(camp, saved["receipt_id"])
        request = receipt["request"]
        owner = ms.validate_suite(camp, request["candidate_suite_id"], verify_data=True)
        if owner["method"]["candidate_id"] in state.get("detached_candidates", []):
            continue
        for key in ("candidate", "baseline"):
            suite_id = request[key + "_suite_id"]
            suite = ms.validate_suite(camp, suite_id, verify_data=True)
            if suite_id == request["baseline_suite_id"] and not request["fold_work"]["baseline"] and suite_id != owner["suite_id"]:
                continue
            path = camp / "suites" / suite_id / "suite_state.json"
            progress = read(path)
            completed = [row["fold_id"] for row in valid_scores(camp, suite)]
            phase_ids = phase_fold_ids(ms.manifest(camp), request["phase"])
            missing = [fold for fold in phase_ids if fold not in completed]
            retry = request.get("engineering_retry") or {}
            if retry.get("suite_id") == suite_id:
                attempts = progress.setdefault("failed_attempts", {}).setdefault(retry["fold_id"], [])
                if not any(row["job_id"] == retry["failed_job_id"] for row in attempts):
                    job = camp / "jobs" / retry["failed_job_id"]
                    record = read(job / "job.json")
                    attempts.append({"job_id": job.name, "job_ref": str(job / "job.json"), "job_sha256": digest(job / "job.json"),
                                     "reason": retry["reason"], "status": record["status"], "gpu_seconds": record.get("gpu_seconds", 0)})
                if progress["jobs"].get(retry["fold_id"]) == retry["failed_job_id"]:
                    progress["jobs"].pop(retry["fold_id"])
            progress.update(phase=request["phase"], phase_receipt_id=receipt["receipt_id"], phase_fold_ids=phase_ids,
                            completed_folds=completed, pending_fold_ids=missing,
                            status="pending" if missing else "screened" if request["phase"] == SCREENING else "complete")
            baseline = ms.validate_suite(camp, request["baseline_suite_id"], verify_data=True)
            progress.update(_phase_projection(camp, ms.manifest(camp), suite, baseline, receipt,
                                              valid_scores(camp, owner), valid_scores(camp, baseline)))
            if suite_id != owner["suite_id"]:
                progress["phase_owner_suite"] = owner["suite_id"]
            elif request["phase"] == CONFIRMATION:
                progress["confirmation_admission"] = receipt
            _write(path, progress)
        # Intent precedes process publication and state counters. Reconstruct
        # its original slot; a missing process record still blocks relaunch.
        for intent_path in (camp / "jobs").glob("*/launch_intent.json"):
            intent = read(intent_path)
            if intent.get("phase_receipt_id") != receipt["receipt_id"]:
                continue
            suite = ms.validate_suite(camp, intent["suite_id"])
            fold = next(row for row in suite["folds"] if row["fold_id"] == intent["fold_id"])
            if (intent["protocol_hash"] != fold["protocol"]["fingerprint"]
                    or intent["canonical_fold_hash"] != fold["canonical_fold_identity"]["canonical_hash"]):
                raise ValueError("phase_recovery_launch_intent_identity_changed")
            path = camp / "suites" / suite["suite_id"] / "suite_state.json"
            progress = read(path)
            existing = progress["jobs"].get(fold["fold_id"])
            if existing not in {None, intent_path.parent.name}:
                raise ValueError("phase_recovery_multiple_launch_intents")
            progress["jobs"][fold["fold_id"]] = intent_path.parent.name
            _write(path, progress)
            consume_phase_job(camp, receipt["receipt_id"], intent_path.parent.name, ms.manifest(camp)["gpu_seconds_estimate_per_suite"] / 10)
    active = []
    for path in sorted((camp / "suites").glob("*/suite_state.json")):
        progress = read(path)
        suite = ms.validate_suite(camp, path.parent.name, verify_data=True)
        if suite["method"]["candidate_id"] in state.get("detached_candidates", []):
            continue
        scores = valid_scores(camp, suite)
        progress["completed_folds"] = [row["fold_id"] for row in scores]
        ids = progress.get("phase_fold_ids") or []
        progress["pending_fold_ids"] = [fold for fold in ids if fold not in progress["completed_folds"]]
        if (path.parent / "aggregate.json").exists():
            ms.read_aggregate(camp, suite["suite_id"])
            ms.commit_aggregate(camp, state, suite)
        if (path.parent / "screening.json").exists():
            read_screening(camp, suite["suite_id"])
            commit_screening(camp, state, suite)
        if progress.get("phase_receipt_id"):
            receipt = phase_reservation(camp, progress["phase_receipt_id"])
            owner_complete = (receipt["request"]["candidate_suite_id"] == suite["suite_id"]
                              and progress.get("status") in {"screened", "complete"})
            evidence_kind = "method_screening" if progress["phase"] == SCREENING else "method_suite"
            row = next((item for item in state.get("evidence", []) if item.get("suite_id") == suite["suite_id"]
                        and item.get("kind") == evidence_kind), None)
            feedback_missing = row is not None and "analysis" not in ms.scientific_feedback_records(camp, state, row)
            if receipt["status"] == "active" and (progress.get("status") in {"pending", "running"} or owner_complete or feedback_missing):
                active.append((suite, progress, receipt))
        _write(path, progress)
    owners = {receipt["request"]["candidate_suite_id"] for _, _, receipt in active}
    if len(owners) > 1:
        raise ValueError("method_multiple_active_phases")
    if active:
        owner = next(iter(owners))
        suite, progress, receipt = next((item for item in active if item[0]["suite_id"] != owner), active[0])
        state.update(active_suite=suite["suite_id"], phase_target_suite=owner, phase=progress["phase"],
                     phase_reservation_id=receipt["receipt_id"], screening_fold_ids=ms.manifest(camp)["selection_manifest"]["screening_fold_ids"],
                     gpu_seconds_reserved=receipt["remaining_gpu_seconds"], method_jobs_reserved=receipt["remaining_training_jobs"])
    else:
        state.update(active_suite=None, gpu_seconds_reserved=0.0, method_jobs_reserved=0)
    intents = {path.parent.name for path in (camp / "jobs").glob("*/launch_intent.json")}
    records = {path.parent.name for path in (camp / "jobs").glob("*/job.json")}
    cost = read(camp / "cost.json") if (camp / "cost.json").exists() else {}
    state["training_jobs"] = max(state["training_jobs"], len(intents | records), int(cost.get("training_jobs", 0)))
    state["gpu_seconds_left"] = state["max_gpu_seconds"] - float(cost.get("gpu_seconds_used", 0))
    return state


def validate_import_receipt(receipt: dict, expected_identity: dict) -> dict:
    """Fail closed: full-only/independent-per-fold-search never implies reuse.

    This validates a provenance mapping, not an import authorization. A new
    campaign must obtain matching canonical evidence before adopting any score.
    """
    required = {"schema_version", "source_campaign", "source_run", "source_fold", "source_manifest_hash",
                "source_run_hash", "canonical_fold_identity", "original_cost_ledger_ref", "source_method_revision",
                "same_method_across_all_folds", "receipt_hash"}
    if not required <= receipt.keys() or receipt["schema_version"] != "eeg_research.fold_import.v1":
        raise ValueError("fold_import_provenance_incomplete")
    if receipt["receipt_hash"] != identity({key: value for key, value in receipt.items() if key != "receipt_hash"}):
        raise ValueError("fold_import_receipt_changed")
    if receipt["same_method_across_all_folds"] is not True or receipt["canonical_fold_identity"] != expected_identity:
        raise ValueError("fold_import_training_evaluation_identity_unproven")
    for key in ("source_manifest_hash", "source_run_hash"):
        if not isinstance(receipt[key], str) or len(receipt[key]) != 64:
            raise ValueError("fold_import_hash_missing")
    return receipt


def issue_exploration_receipt(camp: Path, state: dict, suite_id: str, *, reason: str) -> dict:
    """Explicit bounded exception; scientific screening verdict is immutable."""
    import fcntl, json
    from react_agent.eeg_research.agentic import method_suite as ms
    from react_agent.eeg_research.agentic.api_policy import unresolved_charge_ids
    from react_agent.eeg_research.agentic.loop import _write
    if not reason.strip() or state.get("active_suite") or state.get("live_job") or unresolved_charge_ids(camp):
        raise ValueError("exploration_requires_quiescent_reconciled_state_and_reason")
    common = ms.manifest(camp)
    limit = int((common.get("confirmation_policy") or {}).get("max_exploration_confirmations", 0))
    suite = ms.validate_suite(camp, suite_id, verify_data=True)
    result = read_screening(camp, suite_id)
    if result["screening_result"] not in {"not_promoted", "inconclusive"}:
        raise ValueError("exploration_requires_scientific_nonpromotion_not_engineering_failure")
    with (camp / "exploration_receipts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = camp / "exploration_receipts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        existing = next((row for row in rows if row["suite_hash"] == suite["suite_hash"]), None)
        if existing:
            return existing
        if len(rows) >= limit:
            raise ValueError("exploration_policy_limit_exhausted")
        body = {"schema_version": "eeg_research.exploration_receipt.v1", "suite_id": suite_id,
                "suite_hash": suite["suite_hash"], "method_revision": suite["method"]["method_revision"],
                "selection_manifest_hash": common["selection_manifest"]["selection_manifest_hash"],
                "screening_result_hash": digest(camp / "suites" / suite_id / "screening.json"),
                "original_screening_result": result["screening_result"], "reason": reason.strip(),
                "policy_limit": limit, "ordinal": len(rows) + 1, "independent_final_test": False}
        body["receipt_id"] = "explore_" + identity(body)
        with path.open("a") as stream:
            stream.write(json.dumps(body) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        progress_path = camp / "suites" / suite_id / "suite_state.json"
        progress = read(progress_path)
        progress["exploration_receipt_id"] = body["receipt_id"]
        _write(progress_path, progress)
        return body


def validate_exploration_receipt(camp: Path, suite: dict) -> dict:
    import json
    from react_agent.eeg_research.agentic import method_suite as ms
    path = camp / "exploration_receipts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    progress = read(camp / "suites" / suite["suite_id"] / "suite_state.json")
    row = next((item for item in rows if item["receipt_id"] == progress.get("exploration_receipt_id")), None)
    if not row or row["receipt_id"] != "explore_" + identity({k: v for k, v in row.items() if k != "receipt_id"}):
        raise ValueError("confirmation_requires_screening_passed_or_explicit_exploration")
    common = ms.manifest(camp)
    limit = int((common.get("confirmation_policy") or {}).get("max_exploration_confirmations", 0))
    if (row["suite_hash"] != suite["suite_hash"] or row["method_revision"] != suite["method"]["method_revision"]
            or row["selection_manifest_hash"] != common["selection_manifest"]["selection_manifest_hash"]
            or row["screening_result_hash"] != digest(camp / "suites" / suite["suite_id"] / "screening.json")
            or row["ordinal"] > limit or row["policy_limit"] != limit):
        raise ValueError("exploration_receipt_identity_or_policy_changed")
    return row


def retry_phase_fold(camp: Path, state: dict, suite_id: str, fold_id: str, *, reason: str) -> dict:
    """Explicit bounded failed-process retry, with a new phase admission receipt."""
    from react_agent.eeg_research.agentic import method_suite as ms, jobs, budget
    from react_agent.eeg_research.agentic.loop import _write, event
    if state.get("status") not in {"blocked", "paused"} or state.get("active_suite") != suite_id or not reason.strip():
        raise ValueError("method_fold_retry_requires_paused_or_blocked_active_suite")
    recover(camp, state)
    suite = ms.validate_suite(camp, suite_id, verify_data=True)
    path = camp / "suites" / suite_id / "suite_state.json"
    progress = read(path)
    if fold_id not in progress["phase_fold_ids"] or fold_id in progress["completed_folds"]:
        raise ValueError("method_fold_retry_cannot_replace_complete_or_unadmitted_fold")
    job_id = progress["jobs"].get(fold_id)
    if not job_id or not (camp / "jobs" / job_id / "job.json").exists():
        raise ValueError("method_fold_launch_uncertain_requires_reconciliation")
    record = jobs.reconcile(camp / "jobs" / job_id)
    if jobs._alive(record) or record.get("status") == "running":
        raise ValueError("method_fold_retry_cannot_replace_live_process")
    if record.get("status") not in {"failed", "cancelled"}:
        raise ValueError("method_fold_retry_requires_failed_process_not_successful_training")
    attempts = list((progress.get("failed_attempts") or {}).get(fold_id) or [])
    if len(attempts) >= 2:
        raise ValueError("method_fold_engineering_retry_limit_reached")
    budget.charge_gpu(camp, state, record.get("gpu_seconds", 0), job_id=job_id)
    previous_receipt = budget.phase_reservation(camp, progress["phase_receipt_id"])
    budget.consume_phase_job(camp, previous_receipt["receipt_id"], job_id, ms.manifest(camp)["gpu_seconds_estimate_per_suite"] / 10)
    budget.settle_phase_job(camp, state, previous_receipt["receipt_id"], job_id)
    owner = ms.validate_suite(camp, previous_receipt["request"]["candidate_suite_id"], verify_data=True)
    baseline = ms.validate_suite(camp, previous_receipt["request"]["baseline_suite_id"], verify_data=True)
    cost = phase_cost(ms.manifest(camp), progress["phase"], [row["fold_id"] for row in valid_scores(camp, owner)],
                      [row["fold_id"] for row in valid_scores(camp, baseline)], target=owner["method"]["candidate_id"])
    if (state["training_jobs"] + cost["training_jobs"] > state["max_training_jobs"]
            or state["gpu_seconds_left"] < cost["gpu_seconds"]):
        raise ValueError("method_phase_retry_remaining_work_unaffordable")
    request = {**previous_receipt["request"], "training_jobs": cost["training_jobs"], "gpu_seconds": cost["gpu_seconds"],
               "llm_calls": previous_receipt["request"]["feedback_allocation_per_report"] * cost["feedback_report_count"],
               "phase_cost_estimate": cost,
               "fold_work": {"candidate": cost["missing_candidate_fold_ids"], "baseline": cost["missing_baseline_fold_ids"]},
               "engineering_retry": {"suite_id": suite_id, "fold_id": fold_id, "failed_job_id": job_id,
                                     "ordinal": len(attempts) + 1, "parent_receipt_id": previous_receipt["receipt_id"], "reason": reason}}
    # This native retry only queues work; pause remains in force until resume.
    admission = {**state, "status": "created", "pause_after_step": False}
    receipt = budget.reserve_phase(camp, admission, request, replace_receipt_id=previous_receipt["receipt_id"])
    attempts.append({"job_id": job_id, "job_ref": str(camp / "jobs" / job_id / "job.json"),
                     "job_sha256": digest(camp / "jobs" / job_id / "job.json"), "reason": reason,
                     "status": record["status"], "gpu_seconds": record.get("gpu_seconds", 0)})
    progress.setdefault("failed_attempts", {})[fold_id] = attempts
    progress["jobs"].pop(fold_id)
    progress.update(status="pending", phase_receipt_id=receipt["receipt_id"])
    _write(path, progress)
    for other in {owner["suite_id"], baseline["suite_id"]} - {suite_id}:
        other_path = camp / "suites" / other / "suite_state.json"
        other_progress = read(other_path)
        other_progress["phase_receipt_id"] = receipt["receipt_id"]
        _write(other_path, other_progress)
    state.update(phase_reservation_id=receipt["receipt_id"], live_job=None, status="paused",
                 method_jobs_reserved=receipt["remaining_training_jobs"], gpu_seconds_reserved=receipt["remaining_gpu_seconds"])
    event(camp, "method_phase_fold_retry_queued", suite_id=suite_id, fold_id=fold_id, failed_job_id=job_id,
          receipt_id=receipt["receipt_id"], engineering_retry=len(attempts))
    return {"status": "engineering_retry_queued", "suite_id": suite_id, "fold_id": fold_id,
            "failed_job_id": job_id, "engineering_retry": len(attempts), "resume_required": True}


def detach_phase(camp: Path, state: dict, suite_id: str, receipt_hash: str) -> dict:
    """Project a committed candidate-local detach; charges/evidence stay intact."""
    from react_agent.eeg_research.agentic import method_suite as ms, budget, jobs
    from react_agent.eeg_research.agentic.candidate_lifecycle import require_reconciled
    from react_agent.eeg_research.agentic.loop import _write
    suite = ms.validate_suite(camp, suite_id)
    target = suite["method"]["candidate_id"]
    detach = (state.get("detached_candidates") or {}).get(target) if isinstance(state.get("detached_candidates"), dict) else None
    if target == "baseline" or not detach or detach.get("receipt_hash") != receipt_hash or detach.get("suite_id") != suite_id:
        raise ValueError("phase_detach_requires_committed_nonbaseline_receipt")
    path = camp / "suites" / suite_id / "suite_state.json"
    progress = read(path)
    if progress.get("status") == "abandoned" and progress.get("detach_receipt") == receipt_hash:
        return progress
    require_reconciled(camp, state)
    receipt_id = progress.get("phase_receipt_id")
    receipt = budget.phase_reservation(camp, receipt_id) if receipt_id else None
    # A paired baseline suite may still point at this phase admission. Its
    # original completed scores remain reusable; terminate only its pending work.
    suites = {suite_id}
    if receipt:
        suites.add(receipt["request"]["baseline_suite_id"])
    for linked_id in suites:
        linked_path = camp / "suites" / linked_id / "suite_state.json"
        linked = read(linked_path)
        for job_id in linked["jobs"].values():
            job = camp / "jobs" / job_id
            if not (job / "job.json").exists():
                raise ValueError("phase_detach_unknown_process_requires_reconciliation")
            record = jobs.reconcile(job)
            if record.get("status") == "running" or jobs._alive(record):
                raise ValueError("phase_detach_live_process_requires_reconciliation")
            budget.charge_gpu(camp, state, record.get("gpu_seconds", 0), job_id=job_id)
            if receipt and job_id in receipt["consumed_job_ids"]:
                budget.settle_phase_job(camp, state, receipt_id, job_id)
        linked.update(status="abandoned", detach_receipt=receipt_hash, pending_fold_ids=[],
                      scheduling_status="abandoned" if linked_id == suite_id else "paired_phase_released")
        _write(linked_path, linked)
    if receipt and receipt["status"] == "active":
        budget.release_phase(camp, state, receipt_id, "committed_candidate_detach")
    if state.get("active_suite") in suites:
        state.update(active_suite=None, live_job=None, phase_target_suite=None)
    return read(path)
