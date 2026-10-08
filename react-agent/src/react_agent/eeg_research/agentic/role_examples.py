"""Select small illustrative inputs without changing runtime facts or outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

EXAMPLE_CHARS = 4000
EXAMPLE_COUNT = 2
_PATH = "extension/eeg_candidate.py"
_HASH = "0" * 64
_SCOPE = "Illustrative examples only. All IDs/hashes/code below belong to isolated example contexts, not this run, evidence, approvals or memory. Use the current request's real values."

# Complete CPU-executable toy parent, independent of any campaign candidate.
TOY_PARENT = '''from torch import nn
from react_agent.eeg_training.model import contrastive_loss

class EEGCandidate:
    def build_encoder(self, input_spec, model_config=None):
        if model_config:
            raise ValueError("unknown_model_config")
        n = input_spec["c_num"] * (input_spec["timesteps"][1] - input_spec["timesteps"][0])
        return nn.Sequential(nn.Flatten(), nn.Linear(n, 1024))

    def build_training_objective(self, objective_config=None):
        if objective_config:
            raise ValueError("unknown_objective_config")
        return contrastive_loss
'''
TOY_TRANSFORM = '''    def build_training_transform(self, transform_config=None):
        config = dict(transform_config or {})
        unknown = set(config) - {"status", "offset"}
        if unknown:
            raise ValueError("unknown_transform_config:" + ",".join(sorted(unknown)))
        if config.get("status") not in {"enabled", "disabled"}:
            raise ValueError("invalid_transform_status")
        offset = float(config["offset"])
        if config["status"] == "disabled":
            return None
        return lambda eeg: eeg + offset
'''
TOY_CANDIDATE = TOY_PARENT + "\n" + TOY_TRANSFORM


def _implementation_example(payload: dict[str, Any]) -> dict[str, Any]:
    ranges = payload.get("reference_ranges") or {}
    parent = ranges.get("reference/parent.py") or ranges.get("reference/baseline.py") or {}
    if parent.get("truncated"):
        return _example(
            "read_missing_parent_range", {"parent_page": {"truncated": True, "next_range": {"start": 13, "end": 22}}},
            {"missing_inputs": ["parent objective implementation"]},
            _tool("read_code", path="reference/parent.py", start=13, end=22),
            "Read only the missing parent range before preserving its hooks; a partial source is not a complete implementation.",
        )
    return _example(
        "first_train_only_implementation",
        {"parent_source": TOY_PARENT, "input_spec": {"c_num": 2, "timesteps": [0, 3]},
         "approved_config": {"model": {}, "objective": {}, "transform": {"status": "enabled", "offset": 0.125}},
         "approved_intervention": "Toy train-only offset; preserve the parent encoder and objective."},
        {"current_file": None, "parent_source_complete": True},
        _tool("apply_candidate_patch", path=_PATH, expected_base_hash="", content=TOY_CANDIDATE),
        "The toy contract defines status explicitly, not globally. Wire every approved field, reject unknown keys, preserve parent hooks and consume auto_check. Evaluation skips this transform.",
    )


def _visible_config_recovery() -> dict[str, Any]:
    old = TOY_TRANSFORM.replace('{"status", "offset"}', '{"offset"}')
    source = TOY_PARENT + "\n" + old
    return _example(
        "edit_visible_config_builder",
        {"failure_source": {"path": _PATH, "sha256": hashlib.sha256(source.encode()).hexdigest(), "text": old, "truncated": False},
         "approved_config": {"transform": {"status": "enabled", "offset": 0.125}},
         "declared_contract": "Toy status is enabled/disabled; offset is scientific. normalize_hook_config and _build_hook pass the nested section unchanged; top-level status is not forwarded."},
        {"ok": False, "detail": "unknown_transform_config:status"},
        _tool("edit_candidate_code", path=_PATH, expected_base_hash=hashlib.sha256(source.encode()).hexdigest(),
              edits=[{"old": old, "new": TOY_TRANSFORM}]),
        "The complete failing builder/config are visible. Correct the exact allowed-key check; validate declared status and implement offset. No repeated read, silent key filtering or baseline fallback. Unknown status semantics require the real contract first.",
    )


def _example(
    name: str,
    before: dict[str, Any],
    observed: dict[str, Any],
    response: dict[str, Any],
    why: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "example_input": before,
        "example_observation": observed,
        "next_response": response,
        "why": why,
    }


def _tool(name: str, **args: Any) -> dict[str, Any]:
    return {"tool": name, "args": args}


def _coder(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history") or []
    last = (history[-1].get("result") or {}) if history else {}
    error = (payload.get("recovery_context") or {}).get("error") or last.get("error")
    current = {
        "path": _PATH,
        "sha256": _HASH,
        "text": 'def label():\n    return "old"\n',
        "truncated": False,
    }
    examples: list[dict[str, Any]] = []
    if error in {"edit_not_found", "edit_not_unique", "base_hash_mismatch"}:
        observation = {"error": error, "path": _PATH, "current_sha256": _HASH}
        if error != "base_hash_mismatch":
            observation.update(
                edit_index=0, matches=0 if error == "edit_not_found" else 2
            )
        examples.append(
            _example(
                "recover_" + error,
                {"current_file": current},
                observation,
                _tool(
                    "edit_candidate_code",
                    path=_PATH,
                    expected_base_hash=_HASH,
                    edits=[
                        {
                            "old": current["text"],
                            "new": 'def label():\n    return "new"\n',
                        }
                    ],
                ),
                "The complete current helper is visible. Rebuild a unique exact edit with its current hash; the failed request wrote nothing. Do not reuse stale fragments/reference hashes. Consume auto_check after success.",
            )
        )
    elif error in {"invalid_tool_args", "invalid_tool_request"}:
        examples.append(
            _example(
                "one_json_tool",
                {"needed_range": {"start": 1, "end": 12}},
                {"error": error},
                _tool("read_code", path=_PATH, start=1, end=12),
                "Use only a catalog tool and its typed args; no Markdown, explanation or RoleResult envelope.",
            )
        )
    status = payload.get("check_status") or {}
    check = payload.get("last_check") or {}
    if status.get("passed") and status.get("fresh") is False:
        examples.append(
            _example(
                "stale_check",
                {"check_status": {"passed": True, "fresh": False}},
                {"error": "check_not_passed", "next_tool": "run_candidate_check"},
                _tool("run_candidate_check"),
                "Historical passing source/config identity is stale. Recheck before finish; repeating finish cannot refresh it.",
            )
        )
    elif check and not check.get("ok"):
        source = next((view for view in (payload.get("current_file"), payload.get("failure_source"))
                       if isinstance(view, dict) and view.get("text") and not view.get("truncated")), {})
        detail = json.dumps(check, default=str).lower()
        if "config" in detail and source.get("text") and not source.get("truncated"):
            examples.append(_visible_config_recovery())
        else:
            examples.append(_check_failure_example(check))
    elif status.get("passed") and status.get("fresh"):
        examples.append(
            _example(
                "fresh_finish",
                {
                    "requirements_verified": True,
                    "check_status": {"passed": True, "fresh": True},
                },
                {"auto_check": {"ok": True}},
                _tool(
                    "finish_patch",
                    summary="Updated the helper marker; fresh check passed.",
                ),
                "Finish only when all required work and fresh checks are satisfied; CPU execution does not prove scientific improvement or replace Reviewer.",
            )
        )
    if not examples and not check and not payload.get("current_file"):
        examples.append(_implementation_example(payload))
    return examples


def _check_failure_example(check: dict[str, Any]) -> dict[str, Any]:
    """Choose an isolated example for the actual failed check category."""
    description = json.dumps(check, ensure_ascii=False, default=str).lower()
    before = {"current_file": {"path": _PATH, "sha256": _HASH, "truncated": True}}
    observed = {"ok": False, "location": {"line": 37}}
    if "checkpoint" in description or "state_dict" in description:
        name = "locate_checkpoint_failure"
        observed.update(
            stage="checkpoint_round_trip",
            error="RuntimeError",
            detail="Saved helper buffer exists only after first use; reload construction lacks it.",
        )
        why = "Read construction and state lifecycle before editing. Rebuild the same buffer layout from approved configuration; keep strict checkpoint verification. Unknown shapes stay unknown. Consume auto_check after a source change."
    elif any(
        word in description
        for word in ("config", "unexpected keyword", "extra_forbidden")
    ):
        name = "locate_config_failure"
        before["approved_hook_config"] = {"model": {"marker": "new"}}
        observed.update(
            stage="build_encoder",
            error="TypeError",
            detail="helper() got an unexpected keyword argument 'marker'",
        )
        why = "Read the helper signature and approved configuration wiring. Implement the approved field at the proper hook; never silently drop it or replace the intervention with baseline. Recheck only after a source/config change or stale check."
    else:
        name = "locate_shape_failure"
        before["input_spec"] = {
            "caller_output_width": "declared by caller; omitted from this page"
        }
        observed.update(
            stage="train_forward",
            error="ValueError",
            detail="Caller contract differs from helper output; construction is outside the supplied page.",
        )
        why = "Read the missing construction/call site. Derive dimensions from observed inputs and approved config; do not guess a replacement width. Preserve the intervention, edit once informed and consume auto_check."
    return _example(
        name, before, observed, _tool("read_code", path=_PATH, start=25, end=45), why
    )


def _decision(
    action: str,
    target: str | None,
    cost: dict[str, Any],
    basis: list[str],
    confidence: str,
    *,
    intervention: str | None = None,
) -> dict[str, Any]:
    option = {
        "option_id": "example_option",
        "action": action,
        "target_id": target,
        "question_id": None,
        "evidence_refs": [],
        "observed_gap": "An isolated example needs the next legal step.",
        "expected_information": "Check the next permitted boundary.",
        "estimated_cost": cost,
        "cost_basis": basis,
        "cost_confidence": confidence,
        "value_rationale": "Use only the runtime-supported pair.",
        "intervention": intervention,
    }
    response: dict[str, Any] = {
        "action": action,
        "target_id": target,
        "question_id": None,
        "evidence_refs": [],
        "evidence_ids": [],
        "options": [option],
        "selected_option_id": "example_option",
        "selection_rationale": "Select this legal pair.",
    }
    if intervention is not None:
        response.update(
            hypothesis_draft={
                "statement": "An isolated helper marker change can be checked."
            },
            experiment_draft={"intervention": intervention},
        )
    return response


def _planner(payload: dict[str, Any]) -> list[dict[str, Any]]:
    error = str(payload.get("schema_error") or "")
    context = payload.get("schema_repair_context") or {}
    fields = json.dumps(
        context.get("mechanical_errors") or context.get("validation_errors") or [],
        ensure_ascii=False,
    )
    unknown = {"llm_calls": None, "gpu_seconds": None, "training_jobs": None}
    zero = {"llm_calls": 0, "gpu_seconds": 0.0, "training_jobs": 0}
    examples: list[dict[str, Any]] = []
    if "cost" in error or "estimated_cost" in fields:
        examples.append(
            _example(
                "copy_known_zero_keep_unknown_null",
                {
                    "available_actions": ["diagnose_results", "implement_candidate"],
                    "eligible_targets": {"implement_candidate": ["example_candidate"]},
                    "action_cost_estimates": {
                        "diagnose_results:": {
                            "estimated_cost": zero,
                            "cost_basis": ["runtime:local_action:diagnose_results"],
                            "cost_confidence": "runtime_bound",
                        },
                        "implement_candidate:example_candidate": {
                            "estimated_cost": unknown,
                            "cost_basis": [],
                            "cost_confidence": "unknown",
                        },
                    },
                },
                {
                    "error": "option_cost_without_runtime_basis",
                    "field": "unknown training cost submitted as 0",
                },
                _decision(
                    "diagnose_results",
                    None,
                    zero,
                    ["runtime:local_action:diagnose_results"],
                    "runtime_bound",
                ),
                "Copy an explicit runtime zero exactly. The other example pair stays all-null, [] basis and unknown confidence; never transfer diagnostic cost to training. Exact target costs outrank permitted generic bindings.",
            )
        )
    if "intervention" in error or "intervention" in fields:
        intervention = "Replace only the isolated helper marker in a new design."
        examples.append(
            _example(
                "design_intervention_binding",
                {"available_actions": ["design_experiment"]},
                {"error": "selected_option_intervention_mismatch"},
                _decision(
                    "design_experiment",
                    None,
                    unknown,
                    [],
                    "unknown",
                    intervention=intervention,
                ),
                "Copy the new draft intervention verbatim into its selected option. For approved implement/repair/run actions use intervention=null; do not invent a new draft to match training prose.",
            )
        )
    if (
        (not examples and error)
        or "target" in error
        or "action_unavailable" in error
        or "selected_option" in error
        or "extra_forbidden" in fields
    ):
        examples.append(
            _example(
                "legal_identity_and_copies",
                {
                    "available_actions": ["implement_candidate"],
                    "eligible_targets": {"implement_candidate": ["example_candidate"]},
                    "candidate_id": "example_candidate",
                    "method_name": "Example helper design",
                    "planner_mode": "compare_options",
                    "action_cost_estimates": {
                        "implement_candidate:example_candidate": {
                            "estimated_cost": unknown,
                            "cost_basis": [],
                            "cost_confidence": "unknown",
                        }
                    },
                },
                {
                    "error": "method/artifact name used as target or top-level copies disagree"
                },
                _decision(
                    "implement_candidate", "example_candidate", unknown, [], "unknown"
                ),
                "Select an available action AND its exact eligible target, not a method name or artifact. Synchronize selected/top-level action, target, question and references. Input planner_mode, eligible_targets and completion flags never become output fields.",
            )
        )
    if not examples:
        examples.extend(_scientific_choices(payload))
    return examples


def _scientific_choices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("available_actions", ["implement_candidate"])
    budget = payload.get("budget") or {}
    exhausted = budget.get("training_jobs_left") == 0
    preferred = (["diagnose_results", "collect_diagnostics", "audit_result", "revise_report", "stop"]
                 if exhausted else ["collect_diagnostics", "diagnose_results", "design_experiment", "run_full", "implement_candidate", "stop"])
    targets = payload.get("eligible_targets") or {}
    legal = [action for action in preferred if action in actions
             and (action not in {"run_full", "implement_candidate"}
                  or "available_actions" not in payload or targets.get(action))][:2]
    if not legal:
        return []
    unknown = {"llm_calls": None, "gpu_seconds": None, "training_jobs": None}
    zero = {"llm_calls": 0, "gpu_seconds": 0.0, "training_jobs": 0}
    isolated = {"available_actions": legal, "planner_mode": "compare_options",
                "candidates": [{"candidate_id": "example_candidate"}], "eligible_targets": {},
                "action_cost_estimates": {}, "_known_evidence_ids": [], "trainable_ids": ["example_candidate"]}
    response = None
    options = []
    comparison = payload.get("latest_comparison") or payload.get("comparison") or {}
    population = payload.get("evaluation_population") or {}
    one_hit = population.get("one_query_hit_delta_pp")
    delta = comparison.get("delta_pp")
    tiny = type(delta) in (int, float) and type(one_hit) in (int, float) and abs(delta) <= 2 * one_hit
    for i, action in enumerate(legal):
        targeted = action in {"run_full", "implement_candidate"}
        target = "example_candidate" if targeted else None
        if targeted:
            isolated["eligible_targets"][action] = [target]
        local = action in {"diagnose_results", "collect_diagnostics", "revise_report", "stop"}
        cost, basis, confidence = (zero, ["runtime:local_action:" + action], "runtime_bound") if local else (unknown, [], "unknown")
        intervention = "Toy train-only offset to test the measured input-centering gap." if action == "design_experiment" else None
        decision = _decision(action, target, cost, basis, confidence, intervention=intervention)
        if intervention is not None:
            decision["hypothesis_draft"] = {"statement": "Toy offset tests the observed input-centering gap."}
        option = decision["options"][0]
        option.update(option_id="example_option_" + str(i),
                      observed_gap="Toy pilot differs by only two hits; mechanism remains unresolved." if tiny else "Toy pilot is weak; optimization and concentrated retrieval remain competing explanations.",
                      expected_information="Measure prediction diversity, ties, margin and effective rank before spending a full job." if local else "Test whether the source-bound mechanism persists over the frozen full trajectory.",
                      interpretation_of_outcomes={"concentrated_predictions": "Inspect the supported implementation/optimization evidence.", "diverse_predictions": "Reconsider mechanism; a full run needs trajectory evidence and a justified job opportunity cost."},
                      value_rationale="Smallest legal discriminating step; pilot hits prove neither significance nor futility. Local cost excludes this paid Planner call.")
        isolated["action_cost_estimates"][f"{action}:{target or ''}"] = {
            "estimated_cost": cost, "cost_basis": basis, "cost_confidence": confidence}
        options.append(option)
        if response is None:
            response = decision
            response["selected_option_id"] = option["option_id"]
    response["options"] = options
    if response["action"] == "stop":
        response["stop_reason"] = "budget_exhausted" if exhausted else "no_supported_next_experiment"
    if exhausted:
        response["scope_limits"] = ["No training jobs remain. Empty best-method confirmation does not mean every candidate was confirmed."]
    return [_example("budget_closeout" if exhausted else "pilot_information_choice", isolated,
                     {"training_jobs_left": 0} if exhausted else {"pilot_interpretation": "small_delta_uncertain" if tiny else "weak_pilot_not_an_engineering_failure"},
                     response, "Compare only actual legal pairs; keep unknown costs null. Evidence -> competing mechanism -> minimal experiment -> outcome-dependent decision; no universal pilot threshold or promised gain.")]


def _designer(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parent = payload.get("parent_source") or payload.get("parent") or {}
    missing = payload.get("missing_inputs") or parent.get("missing_inputs") or parent.get("source_truncated")
    blocked = _example("missing_parent_facts", {"parent_source": {"source_truncated": True}},
                       {"missing_inputs": ["parent effective objective and complete encoder source"]},
                       {"status": "blocked", "experiment_spec": None, "missing_inputs": ["parent effective objective and complete encoder source"], "summary_zh": "缺少父实现事实，暂不设计。"},
                       "Do not invent parameter counts, supported hooks or cost. Obtain the missing source/config before completing a draft.")
    draft = {"parent_candidate_id": "example_parent", "control_candidate_id": "baseline", "initial_fidelity": "pilot",
             "status": "draft", "hypothesis": {"statement": "Toy offset may correct a measured input-centering gap."},
             "intervention": "Only the toy train-only offset changes.", "required_capability_ids": ["build_training_transform"],
             "model": {}, "objective": {}, "transform": {"status": "enabled", "offset": 0.125},
             "prediction": "Matched development measurements test the centering explanation; no gain threshold is invented.",
             "disconfirmation_conditions": ["No source-bound centering change or inconsistent matched observations."]}
    normal = _example("single_hook_draft", {"parent_source": {"candidate_id": "example_parent", "source": TOY_PARENT,
                                            "source_hash": hashlib.sha256(TOY_PARENT.encode()).hexdigest(), "source_truncated": False,
                                            "model": {}, "objective": {}, "transform": {}},
                                            "capabilities": {"hooks": {"build_training_transform": {"status": "available"}}}, "toy_status_contract": ["enabled", "disabled"]},
                      {"gap": "Verified toy input-centering diagnostic; runtime budget remains unknown."},
                      {"status": "completed", "experiment_spec": draft, "required_capability_ids": ["build_training_transform"], "summary_zh": "仅改变已声明的训练变换，保持父接口。"},
                      "Inherit unchanged effective configs. Predictions belong inside experiment_spec, never top-level predictions/status_scope_note. This is a draft, not approval.")
    return [blocked] if missing else [normal, blocked]


def _reviewer(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs = {"source_ref": "example_source", "source_hash": _HASH, "check_ref": "example_check", "check_hash": "1" * 64}
    ready = _example("disproved_review_allegation", {"approved_requirement": "Toy transform must receive offset.", "source": "return lambda eeg: eeg + config['offset']"},
                     {"dispatch": "_build_hook passes the approved transform section unchanged", "fresh_source_config_check": refs},
                     {"status": "ready", "issues": [], "intervention_coverage": "Toy offset wired; encoder/objective preserved.",
                      "verified_invariants_with_refs": [{"claim": "Old naming allegation is disproved by the exact dispatch and current builder.", **refs}],
                      "review_limits": "Synthetic CPU/source scope; unchanged-source re-review is a review correction, not a source repair.", "summary_zh": "旧指控已被当前证据推翻。"},
                     "Use current source/dispatch/check identity. Put resolved allegations in verified invariants; require no meaningless edit.")
    missing = _example("shape_pass_missing_intervention", {"approved_requirement": "Toy train-only offset", "source_location": "example_source:build_training_transform", "source": "return None"},
                       {"fresh_check": {"shape_ok": True, "source_hash": _HASH}},
                       {"status": "needs_fix", "issues": [{"severity": "blocking", "category": "intervention_not_implemented",
                          "location": "example_source:build_training_transform", "evidence": "return None omits the approved toy offset despite valid encoder shape.",
                          "required_correction": "Wire the approved offset; verify changed train output and unchanged evaluation path."}],
                        "review_limits": "Shape execution does not establish intervention coverage or benefit.", "summary_zh": "可运行但未实现批准的变换。"},
                       "Check coverage as well as CPU execution; keep scientific uncertainty separate from engineering defects.")
    error = str(payload.get("schema_error") or "")
    return [missing, ready] if "intervention_not" in error or "needs_fix_requires" in error else [ready, missing]


def _analyst(payload: dict[str, Any]) -> list[dict[str, Any]]:
    chance = _example("chance_without_mechanism", {"evaluation_population": {"development_query_count": 40, "gallery_image_count": 20}, "seed": 7, "fidelity": "pilot", "checkpoint": "example_checkpoint"},
                      {"valid_run": True, "hits": 2, "top1": 0.05, "diagnostics": None},
                      {"execution_assessment": "Valid toy execution", "hypothesis_assessment": "inconclusive",
                       "observations": [{"hits": 2, "query_count": 40, "gallery_size": 20, "top1_fraction": 0.05, "chance_reference": 0.05}],
                       "competing_explanations": ["optimization failure", "concentrated predictions", "weak but diverse representations"],
                       "evidence_gaps": ["unique_top1_gallery_predictions, ties_for_best_score_queries, margin_mean, centered_singular_entropy_effective_rank, centered_energy_fraction"],
                       "suggested_next_actions": ["Use verified diagnostics or collect_diagnostics: concentration/rank loss supports implementation or optimization inspection; diverse predictions shifts priority to the mechanism and trajectory."],
                       "scope_limits": ["One toy development seed; chance aggregate alone proves neither uniform randomness, constant prediction nor collapse."], "summary_zh": "接近机会水平，机制仍不确定。"},
                      "Consume source/checkpoint-bound measurements; missing diagnostics remain gaps, never invented collapse evidence.")
    pilot = _example("small_pilot_delta", {"evaluation_population": {"development_query_count": 80, "gallery_image_count": 16}, "group_ids": ["example_g1", "example_g2", "example_g3"], "matched_seeds": [4], "required_seeds": [4, 8]},
                     {"fidelity": "pilot", "baseline_hits": 8, "candidate_hits": 9, "same_protocol_seed_checkpoint_rule": True},
                     {"execution_assessment": "Valid matched pilot", "hypothesis_assessment": "inconclusive",
                      "observations": [{"baseline_hits": 8, "candidate_hits": 9, "query_count": 80, "delta_pp": 1.25, "observed_seeds": [4], "missing_seeds": [8]}],
                      "interpretations": ["One extra hit is neither significant improvement nor a universal rejection signal."],
                      "evidence_gaps": ["Matched learning trajectory and mechanism diagnostics; group measurements unavailable."],
                      "suggested_next_actions": ["Diagnose first if diversity/optimization can resolve the ambiguity; consider full only if trajectory/mechanism information justifies its job opportunity cost."],
                      "scope_limits": ["Pilot, supplied query denominator and one matched seed only; group IDs do not imply measured group scores."], "summary_zh": "小幅试验差异不足以确定晋升或淘汰。"},
                     "Toy numbers are not campaign thresholds. Bind population, scale, gallery, fidelity, seeds and checkpoint to the actual input.")
    negative = _example("valid_negative_full", {"fidelity": "full", "required_seeds": [3, 9], "observed_seeds": [3]},
                        {"execution_valid": True, "matched_delta_pp": -2.0, "approved_hook_consumed": True},
                        {"execution_assessment": "Successful valid full execution", "hypothesis_assessment": "weakened",
                         "observations": [{"seed": 3, "delta_pp": -2.0}], "scope_limits": ["Seed 9 remains unknown; a negative full experiment is executed family history."],
                         "suggested_next_actions": ["Check the signed followup commitment and legal budget; additional measurement needs a discriminating question, not a presumed code repair."], "summary_zh": "有效负实验削弱假设，执行成功。"},
                        "Separate scientific outcome from code failure; incomplete confirmation is not unexecuted research.")
    latest = payload.get("latest") or payload.get("result") or payload.get("latest_job") or {}
    return [negative, chance] if latest.get("fidelity") == "full" else [chance, pilot]


def _curator(payload: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = payload.get("allowed_source_refs") or {}
    if not allowed.get("tasks") or payload.get("skills_unavailable_reason"):
        return [
            _example(
                "unsupported_skill_is_empty",
                {"allowed_source_refs": {"tasks": [], "episodes": [], "artifacts": []}},
                {"missing_inputs": ["completed verified development source"]},
                {
                    "proposed_lessons": [],
                    "proposed_skills": [],
                    "summary_zh": "缺少已验证的流程支持。",
                },
                "Return no skill when support is absent. A suggestion to fix an observed failure is not a verified repair.",
            )
        ]
    skill = {
        "description_en": "Record an exact-edit failure against its verified development source.",
        "trigger_en": "A completed development task recorded a failed exact edit with no verified repair.",
        "procedure": [
            "Check the source task/artifact identity and scope.",
            "Record the observed edit error and source geometry/interface conditions.",
            "Keep the repair outcome unknown until a fresh original check verifies it.",
        ],
        "preconditions": {
            "scope": "development only",
            "modality": "as declared by the verified source",
            "geometry": "retain the source interface",
        },
        "failure_modes": [
            "The old fragment was absent; an untested repair must not be called successful."
        ],
        "invalidation_conditions": ["A later source-bound task verifies recovery."],
        "source_task_refs": ["example_completed_task"],
        "artifact_refs": ["example_verified_artifact"],
        "supporting_episode_ids": [],
    }
    return [
        _example(
            "source_bound_failure_procedure",
            {
                "allowed_source_refs": {
                    "tasks": [{"task_id": "example_completed_task"}],
                    "artifacts": [{"artifact_id": "example_verified_artifact"}],
                    "episodes": [],
                }
            },
            {"observed_failure": "edit_not_found", "repair_verified": False},
            {
                "proposed_lessons": [],
                "proposed_skills": [skill],
                "summary_zh": "仅记录有来源的失败流程，修复仍未验证。",
            },
            "Use only supplied completed source IDs. description_en is one short sentence (validator maximum 1000 chars); procedure is a list of nonempty steps, never procedure_note. Observed failure does not prove a fix.",
        )
    ]


def add_role_examples(
    payload: dict[str, Any],
    role: str,
    memory: Any = None,
    *,
    max_chars: int = EXAMPLE_CHARS,
) -> dict[str, Any]:
    """Add complete deterministic examples within an independent field budget."""
    selectors: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
        "candidate_coder": _coder,
        "research_planner": _planner,
        "experiment_designer": _designer,
        "candidate_reviewer": _reviewer,
        "result_analyst": _analyst,
    }
    if (
        role == "memory_curator"
        and memory is not None
        and memory.enabled
        and memory.skills_enabled
    ):
        selectors[role] = _curator
    select = selectors.get(role)
    if select is None:
        return payload
    field: dict[str, Any] = {"scope": _SCOPE, "examples": []}
    for example in select(payload):
        candidate = {"scope": _SCOPE, "examples": [*field["examples"], example]}
        size = len(
            json.dumps(
                {"illustrative_examples": candidate},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if size <= min(max_chars, EXAMPLE_CHARS):
            field = candidate
        if len(field["examples"]) == EXAMPLE_COUNT:
            break
    return {**payload, "illustrative_examples": field} if field["examples"] else payload
