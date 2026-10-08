"""One scientific method, ten deterministic execution folds.

This opt-in dispatcher uses the existing job launcher, reconciler, candidate
bindings and scorer. Only a complete suite becomes scientific evidence.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

from react_agent.eeg_research.agentic.loso_study import SUBJECTS, digest, folds, identity, read, write_once

MODE = "loso_method_search"
METRIC = "benchmark.loso_mean_fixed_gallery_top1"
MANIFEST = "method_search_manifest.json"


def enabled(state: dict) -> bool:
    return state.get("evaluation_mode") == MODE


def validate_manifest_rows(rows: list[dict]) -> None:
    if len(rows) != 10 or {row["held_out_subject"] for row in rows} != set(SUBJECTS):
        raise ValueError("method_suite_requires_ten_unique_subjects")
    if len({row["fold_id"] for row in rows}) != 10:
        raise ValueError("method_suite_duplicate_fold")
    for row in rows:
        expected = [subject for subject in SUBJECTS if subject != row["held_out_subject"]]
        if row["train_subjects"] != expected or row.get("seed") != 0:
            raise ValueError("method_suite_split_or_seed_mismatch")


def _framework_dependencies() -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "eeg_training").glob("*.py")),
             *[Path(__file__).parent / name for name in (
                 "baseline.py", "jobs.py", "runner.py", "execution_protocol.py", "run_context.py",
                 "method_suite.py", "method_evaluate.py", "check_entry.py", "implementation_semantics.py",
                 "multigpu_check.py", "multigpu_probe.py", "process_identity.py", "coder.py", "interface.py")]]
    return [{"path": str(path), "sha256": digest(path)} for path in paths]


def create_manifest(camp: Path, design, *, allow_shared_gpus: bool = False, estimate: float) -> dict:
    from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
    from react_agent.eeg_research.agentic.loso_study import gpu_inventory
    from react_agent.eeg_training.protocol import list_subjects, split_plan, feature_cache
    if design.evaluation_mode != MODE or design.epochs != 50 or design.stop != "single_full" or design.checkpoint_policy != "strict_best":
        raise ValueError("method_search_training_policy_mismatch")
    base = Path(design.data_root)
    if tuple(list_subjects(base, "eeg")) != SUBJECTS or len(design.gpu) < 2:
        raise ValueError("method_search_requires_ten_subjects_and_multiple_gpus")
    rows = []
    for fold in folds():
        fold_design = replace(design, subject=",".join(fold["train_subjects"]))
        plan = split_plan(base, fold_design)
        if list(plan.train_subjects) != fold["train_subjects"] or len(plan.forbidden_files) != 1:
            raise ValueError("method_search_loader_split_mismatch")
        protocol = build_execution_protocol(fold_design, base, training_seeds=[0], split_seed=0, training_seed=0)
        rows.append({**fold, "seed": 0, "protocol": protocol,
                     "train_files": list(map(str, plan.train_files)), "development_files": list(map(str, plan.val_files)),
                     "benchmark_file": str(plan.forbidden_files[0])})
    validate_manifest_rows(rows)
    data = sorted({path for row in rows for path in [*row["train_files"], *row["development_files"], row["benchmark_file"]]})
    # Bind EEG bytes, not only path/mtime labels. Samples stay in their fold's
    # declared role; reading bytes here is identity hashing, not statistics fit.
    dependencies = [{"path": path, "sha256": digest(Path(path))} for path in data]
    caches = [{"path": str(feature_cache(base, design, mode)), "sha256": digest(feature_cache(base, design, mode))}
              for mode in ("train", "test")]
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("method_suite_gpu_estimate_missing")
    body = {"schema_version": "eeg_research.method_search_manifest.v1", "evaluation_mode": MODE,
            "folds": rows, "seed": 0, "fidelity": "full", "required_fold_count": 10,
            "gpu": list(design.gpu), "gpu_inventory": gpu_inventory(design.gpu, allow_shared=allow_shared_gpus),
            "data_dependencies": dependencies, "cache_dependencies": caches,
            "framework_dependencies": _framework_dependencies(), "gpu_seconds_estimate_per_suite": estimate,
            "benchmark_permission": {"authorized": True, "data_role": "method_development_benchmark",
                                     "purpose": "complete_method_suite_feedback", "partial_feedback": False,
                                     "training_access": False, "checkpoint_selection": "source_subject_development_only"},
            "metric": METRIC, "metric_unit": "fraction", "aggregation": "macro_mean", "ddof": 0,
            "search_scope": "inter_subject_development_benchmark", "independent_final_test": False}
    body["manifest_hash"] = identity(body)
    write_once(camp / MANIFEST, body)
    return body


def manifest(camp: Path) -> dict:
    body = read(camp / MANIFEST)
    if identity({k: v for k, v in body.items() if k != "manifest_hash"}) != body["manifest_hash"]:
        raise ValueError("method_search_manifest_changed")
    validate_manifest_rows(body["folds"])
    if body["benchmark_permission"] != {"authorized": True, "data_role": "method_development_benchmark",
                                       "purpose": "complete_method_suite_feedback", "partial_feedback": False,
                                       "training_access": False, "checkpoint_selection": "source_subject_development_only"}:
        raise ValueError("method_benchmark_permission_mismatch")
    return body


def verify_dependencies(rows: list[dict]) -> None:
    for row in rows:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("method_suite_dependency_changed:" + Path(row["path"]).name)


def _method(camp: Path, target: str, state: dict) -> dict:
    from react_agent.eeg_research.agentic.jobs import baseline_manifest
    from react_agent.eeg_research.agentic.run_context import build_frozen_run_spec
    from react_agent.eeg_research.agentic.execution_protocol import load_protocol
    from react_agent.eeg_research.agentic.experiment_gate import resolve_approved_experiment
    from react_agent.eeg_research.agentic.multigpu_check import run_multigpu_check, validate_receipt, refresh_probe_budget
    protocol = load_protocol(camp)
    frozen = build_frozen_run_spec(camp, target, state, fidelity="full", seed=0, protocol=protocol)
    if target == "baseline":
        source = baseline_manifest()
        dependencies = [{"path": source["entry"], "sha256": source["entry_sha256"]}]
        probe = run_multigpu_check(camp, protocol)
        refresh_probe_budget(camp, state)
        validate_receipt(camp, protocol, probe)
    else:
        workspace = camp / "candidates" / target
        binding = frozen["approved_binding"]
        resolve_approved_experiment(camp, target_id=target, spec_ref=binding["spec_ref"],
                                    expected_hash=binding["spec_hash"], state=state, action="train")
        source = read(workspace / "source_manifest.json")
        from react_agent.eeg_research.agentic.native_patch import enforce_requirement_review
        review = enforce_requirement_review(workspace, read(workspace / "spec.json"), read(workspace / "review.json"))
        check = read(workspace / "checks.json")
        if review["status"] != "ready" or not check.get("ok"):
            raise ValueError("method_suite_requires_current_ready_review_and_check")
        probe = validate_receipt(camp, protocol, check.get("multigpu_check") or {}, workspace=workspace)
        dependencies = [{"path": str(workspace / name), "sha256": digest(workspace / name)} for name in
                        ("source_manifest.json", "approved_binding.json", "spec.json", "checks.json", "review.json")]
        dependencies.append({"path": source["entry"], "sha256": source["entry_sha256"]})
        if frozen["source_manifest_hash"] != source["entry_sha256"]:
            raise ValueError("method_suite_approved_source_mismatch")
    probe_dir = camp / "engineering_checks" / "multigpu" / probe["probe_identity"]
    dependencies.extend({"path": str(probe_dir / name), "sha256": digest(probe_dir / name)}
                        for name in ("request.json", "intent.json", "receipt.json"))
    recipe_keys = ("dataset", "exp_setting", "training_strategy", "evaluation_mode", "checkpoint_policy", "selection_min_delta",
                   "split_seed", "training_seeds", "full_epochs", "batch_size", "lr", "gpu", "weight_decay",
                   "input_geometry", "negative_sampling_policy", "fidelity_overrides", "data_root")
    body = {"candidate_id": target, "source_hash": source["entry_sha256"],
            "source_manifest": source, "public_recipe": {key: protocol[key] for key in recipe_keys},
            "hook_spec": frozen["hook_spec"], "spec_hash": frozen["spec_hash"],
            "approval_ref": frozen["approval_ref"], "approved_binding": frozen["approved_binding"],
            "multigpu_check_ref": str(probe_dir / "receipt.json"), "multigpu_probe_identity": probe["probe_identity"],
            "dependencies": dependencies, "manifest_hash": manifest(camp)["manifest_hash"]}
    body["method_revision"] = identity(body)
    return body


def validate_suite(camp: Path, suite_id: str, *, verify_data: bool = False) -> dict:
    path = camp / "suites" / suite_id / "suite_manifest.json"
    body = read(path)
    if identity({k: v for k, v in body.items() if k != "suite_hash"}) != body["suite_hash"]:
        raise ValueError("method_suite_manifest_changed")
    parent = manifest(camp)
    if body["manifest_hash"] != parent["manifest_hash"]:
        raise ValueError("method_suite_parent_mismatch")
    verify_dependencies(body["method"]["dependencies"] + parent["framework_dependencies"])
    if verify_data:
        verify_dependencies(parent["data_dependencies"] + parent["cache_dependencies"])
    validate_manifest_rows(body["folds"])
    for row in body["folds"]:
        protocol = read(Path(row["protocol_path"]))
        if protocol != row["protocol"] or protocol["method_revision"] != body["method"]["method_revision"]:
            raise ValueError("method_suite_fold_protocol_changed")
    return body


def begin(camp: Path, state: dict, target: str) -> str:
    from react_agent.eeg_research.agentic.budget import reserve_gpu
    if state.get("active_suite"):
        raise ValueError("method_suite_already_active")
    if target != "baseline" and not any(row.get("kind") == "method_suite" and row.get("candidate_id") == "baseline"
                                        and row.get("evaluation_valid") for row in state.get("evidence") or []):
        raise ValueError("complete_baseline_suite_required")
    common = manifest(camp)
    # Reject a new method before spending on its engineering probe when the
    # complete ten-fold execution cannot be reserved. Existing suites recover
    # from their immutable manifests without reserving ten fresh job slots.
    existing = any(read(path).get("method", {}).get("candidate_id") == target
                   for path in (camp / "suites").glob("*/suite_manifest.json"))
    if not existing:
        if state["max_training_jobs"] - state["training_jobs"] < 10:
            raise ValueError("method_suite_requires_ten_job_slots")
        if common["gpu_seconds_estimate_per_suite"] > float(state.get("gpu_seconds_left", 0)) - float(state.get("gpu_seconds_reserved", 0)):
            raise ValueError("method_suite_gpu_reservation_unavailable")
    method = _method(camp, target, state)
    suite_id = target + "_" + method["method_revision"][:16] + "_seed0_full"
    suite_dir = camp / "suites" / suite_id
    if (suite_dir / "suite_manifest.json").exists():
        validate_suite(camp, suite_id)
        state["active_suite"] = suite_id
        return suite_id
    if state["max_training_jobs"] - state["training_jobs"] < 10:
        raise ValueError("method_suite_requires_ten_job_slots")
    estimate = common["gpu_seconds_estimate_per_suite"]
    if estimate > float(state.get("gpu_seconds_left", 0)) - float(state.get("gpu_seconds_reserved", 0)):
        raise ValueError("method_suite_gpu_reservation_unavailable")
    verify_dependencies(method["dependencies"] + common["framework_dependencies"] +
                        common["data_dependencies"] + common["cache_dependencies"])
    rows = []
    for fold in common["folds"]:
        protocol = {**fold["protocol"], "method_revision": method["method_revision"],
                    "method_suite_id": suite_id, "method_suite_ref": str((suite_dir / "suite_manifest.json").resolve()),
                    "fold_id": fold["fold_id"], "benchmark_subject": fold["held_out_subject"]}
        protocol["fingerprint"] = identity({k: v for k, v in protocol.items() if k != "fingerprint"})
        path = suite_dir / fold["fold_id"] / "execution_protocol.json"
        if path.exists():
            if read(path) != protocol:
                raise ValueError("method_suite_preparation_protocol_conflict")
        else:
            write_once(path, protocol)
        rows.append({**fold, "protocol": protocol, "protocol_path": str(path.resolve())})
    body = {"schema_version": "eeg_research.method_suite.v1", "suite_id": suite_id, "method": method,
            "manifest_hash": common["manifest_hash"], "folds": rows, "seed": 0, "fidelity": "full"}
    body["suite_hash"] = identity(body)
    write_once(suite_dir / "suite_manifest.json", body)
    if not (suite_dir / "suite_state.json").exists():
        write_once(suite_dir / "suite_state.json", {"status": "pending", "jobs": {}, "completed_folds": []})
    if not reserve_gpu(state, estimate):
        raise ValueError("method_suite_gpu_reservation_unavailable")
    state["active_suite"] = suite_id
    state["status"] = "training"
    state["method_jobs_reserved"] = 10
    state["method_gpu_reservation"] = estimate
    return suite_id


def validate_fold_derivation(camp: Path, target: str, protocol: dict) -> dict:
    suite = validate_suite(camp, protocol["method_suite_id"])
    fold = next(row for row in suite["folds"] if row["fold_id"] == protocol["fold_id"])
    if suite["method"]["candidate_id"] != target or fold["protocol"] != protocol:
        raise ValueError("method_suite_fold_binding_mismatch")
    return {"schema_version": "eeg_research.method_fold_derivation.v1",
            "suite_hash": suite["suite_hash"], "method_revision": suite["method"]["method_revision"],
            "public_approval_ref": suite["method"]["approval_ref"], "fold_id": fold["fold_id"],
            "held_out_subject": fold["held_out_subject"], "train_subjects": fold["train_subjects"],
            "fold_protocol_hash": protocol["fingerprint"], "independent_initialization": True,
            "weight_or_optimizer_transfer": False, "seed_reset_per_process": 0}


def freeze_fold(camp: Path, suite: dict, fold: dict, job: Path) -> dict:
    from react_agent.eeg_training.checkpoint_selection import validate_training_completion
    from react_agent.eeg_research.agentic.runner import accept_job
    record = read(job / "job.json")
    if record.get("status") != "finished":
        raise ValueError("benchmark_requires_finished_training")
    ok, reason = validate_training_completion(read(job / "metrics.json"), read(job / "selected_checkpoint.json"), fold["protocol"], "full")
    if not ok:
        raise ValueError(reason)
    accepted = accept_job(job, suite["method"]["source_manifest"], "full", fold["protocol"])
    if accepted.get("evaluation_valid") is not True:
        raise ValueError("benchmark_training_binding_invalid:" + str(accepted.get("reason")))
    frozen = read(job / "frozen_run_spec.json")
    if frozen.get("fold_derivation") != validate_fold_derivation(camp, suite["method"]["candidate_id"], fold["protocol"]):
        raise ValueError("benchmark_fold_derivation_changed")
    consistency = read(job / "training_inference_consistency.json")
    if consistency.get("status") != "verified":
        raise ValueError("benchmark_reconstruction_consistency_required")
    path = camp / "suites" / suite["suite_id"] / fold["fold_id"] / "benchmark_freeze.json"
    if path.exists():
        existing = read(path)
        verify_dependencies(existing["dependencies"])
        if existing["checkpoint_hash"] != digest(job / "last.ckpt") or existing["suite_hash"] != suite["suite_hash"]:
            raise ValueError("benchmark_freeze_identity_conflict")
        return existing
    names = ("last.ckpt", "metrics.json", "selected_checkpoint.json", "source_binding.json", "hook_config.json",
             "frozen_run_spec.json", "training_inference_identity.json", "training_inference_consistency.json")
    dependencies = [{"path": str(job / name), "sha256": digest(job / name)} for name in names]
    from react_agent.eeg_research.agentic.diagnostics import write_job_bundle
    diagnostics = write_job_bundle(job)
    dependencies.append({"path": str(job / "diagnostic_summary.json"), "sha256": digest(job / "diagnostic_summary.json")})
    body = {"suite_id": suite["suite_id"], "suite_hash": suite["suite_hash"],
            "method_revision": suite["method"]["method_revision"], "candidate_id": suite["method"]["candidate_id"],
            "fold_id": fold["fold_id"], "held_out_subject": fold["held_out_subject"], "train_subjects": fold["train_subjects"],
            "seed": 0, "fidelity": "full", "job_dir": str(job), "protocol_path": fold["protocol_path"],
            "protocol_hash": fold["protocol"]["fingerprint"], "dependencies": dependencies,
            "development_metrics": read(job / "metrics.json"), "selected_checkpoint": read(job / "selected_checkpoint.json"),
            "diagnostics": diagnostics, "checkpoint_hash": digest(job / "last.ckpt"),
            "evaluator_hash": identity(manifest(camp)["framework_dependencies"]),
            "permission": manifest(camp)["benchmark_permission"]}
    body["evaluation_key"] = identity(body)
    path = camp / "suites" / suite["suite_id"] / fold["fold_id"] / "benchmark_freeze.json"
    if path.exists():
        # Diagnostics can contain write times. The original freeze wins after
        # verifying its immutable dependencies; no result-based overwrite.
        existing = read(path)
        verify_dependencies(existing["dependencies"])
        if existing["checkpoint_hash"] != body["checkpoint_hash"] or existing["suite_hash"] != body["suite_hash"]:
            raise ValueError("benchmark_freeze_identity_conflict")
        return existing
    write_once(path, body)
    return body


def aggregate_records(suite: dict, scores: list[dict]) -> dict:
    """Pure validation and equal-subject arithmetic; partial has no primary score."""
    expected = {row["fold_id"]: row for row in suite["folds"]}
    validate_manifest_rows(suite["folds"])
    seen = set()
    for score in scores:
        fold_id = score["fold_id"]
        if fold_id not in expected or fold_id in seen:
            raise ValueError("method_aggregate_duplicate_or_unknown_fold")
        seen.add(fold_id)
        fold = expected[fold_id]
        if (score["method_revision"] != suite["method"]["method_revision"] or score["suite_hash"] != suite["suite_hash"]
                or score["candidate_id"] != suite["method"]["candidate_id"] or score["seed"] != suite["seed"]
                or score["fidelity"] != "full" or score["held_out_subject"] != fold["held_out_subject"]
                or score["train_subjects"] != fold["train_subjects"] or score["protocol_hash"] != fold["protocol"]["fingerprint"]):
            raise ValueError("method_aggregate_identity_mismatch")
        metrics = score["metrics"]
        if metrics["candidate_count"] != 200 or metrics["query_count"] != 200 or score.get("status") != "valid":
            raise ValueError("method_aggregate_population_or_validity_mismatch")
        for key in ("top1", "top5"):
            value = metrics[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("method_aggregate_nonfinite_or_out_of_range")
    if len({score["evaluator_hash"] for score in scores}) > 1:
        raise ValueError("method_aggregate_mixed_evaluators")
    body = {"status": "complete" if len(seen) == 10 else "partial", "coverage": len(seen), "required_fold_count": 10,
            "candidate_id": suite["method"]["candidate_id"], "method_revision": suite["method"]["method_revision"],
            "suite_hash": suite["suite_hash"], "seed": suite["seed"], "fidelity": "full", "ddof": 0,
            "scope": "method_development_benchmark", "independent_final_test": False,
            "metric_unit": "fraction", "scores": sorted(scores, key=lambda row: row["fold_id"]),
            "independent_seed_count": 1, "statistical_significance": "not_established", "replicated": False}
    if len(seen) == 10:
        for key in ("top1", "top5"):
            values = [score["metrics"][key] for score in scores]
            body["mean_" + key] = statistics.mean(values)
            body["std_" + key] = statistics.pstdev(values)
        body[METRIC] = body["mean_top1"]
    return body


def read_aggregate(camp: Path, suite_id: str) -> dict:
    suite = validate_suite(camp, suite_id)
    target = camp / "suites" / suite_id / "aggregate.json"
    body = read(target)
    for dep in body["score_dependencies"]:
        verify_dependencies([dep])
        verify_score(camp, suite, read(Path(dep["path"])))
    computed = aggregate_records(suite, [read(Path(dep["path"])) for dep in body["score_dependencies"]])
    if body["aggregate"] != computed or computed["status"] != "complete":
        raise ValueError("method_suite_aggregate_invalid")
    return computed


def verify_score(camp: Path, suite: dict, score: dict) -> None:
    freeze_path = camp / "suites" / suite["suite_id"] / score["fold_id"] / "benchmark_freeze.json"
    freeze = read(freeze_path)
    if digest(freeze_path) != score["freeze_sha256"] or str(freeze_path) != score["freeze_ref"]:
        raise ValueError("method_score_freeze_changed")
    expected_evaluator = identity(manifest(camp)["framework_dependencies"])
    if (score["evaluation_key"] != freeze["evaluation_key"] or score["checkpoint_hash"] != freeze["checkpoint_hash"]
            or score["evaluator_hash"] != freeze["evaluator_hash"] or score["evaluator_hash"] != expected_evaluator
            or freeze["suite_hash"] != suite["suite_hash"] or freeze["method_revision"] != suite["method"]["method_revision"]):
        raise ValueError("method_score_checkpoint_or_evaluator_mismatch")
    if freeze["evaluation_key"] != identity({k: v for k, v in freeze.items() if k != "evaluation_key"}):
        raise ValueError("method_score_freeze_digest_invalid")
    verify_dependencies(freeze["dependencies"])
    # The scorer reports the inputs it actually used. File hashes alone do not
    # identify the applied query order, gallery, positives or encoder buffers.
    from react_agent.eeg_training.inference_consistency import validate_reconstruction_context
    parent = manifest(camp)
    fold = next(row for row in suite["folds"] if row["fold_id"] == score["fold_id"])
    metrics = score["metrics"]
    inference = metrics.get("inference_identity")
    if not isinstance(inference, dict) or identity(inference) != metrics.get("inference_fingerprint"):
        raise ValueError("method_score_inference_identity_missing_or_changed")
    job = Path(freeze["job_dir"])
    training_identity = validate_reconstruction_context(job, job / "last.ckpt")
    if (training_identity is None or inference.get("training_inference_identity_fingerprint") != training_identity["fingerprint"]
            or inference.get("checkpoint_sha256") != freeze["checkpoint_hash"]
            or inference.get("source_sha256") != suite["method"]["source_hash"]
            or inference.get("evaluator") != training_identity["evaluator"]
            or inference.get("tie_policy") != training_identity["tie_policy"]
            or inference.get("eval_mode") is not True or inference.get("augmentation") != "not_applied"
            or inference.get("device") != "cpu" or inference.get("dtype") != "torch.float32"):
        raise ValueError("method_score_actual_encoder_or_evaluator_mismatch")
    data = {row["path"]: row["sha256"] for row in parent["data_dependencies"]}
    expected_file = fold["benchmark_file"]
    held_out_files = inference.get("held_out_files")
    if expected_file not in data or held_out_files != [{"path": expected_file, "sha256": data[expected_file]}]:
        raise ValueError("method_score_actual_held_out_file_mismatch")
    caches = {row["path"]: row["sha256"] for row in parent["cache_dependencies"]}
    cache = inference.get("test_image_cache") or {}
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol, read_query_rows
    from react_agent.eeg_training.protocol import feature_cache
    expected_cache = str(feature_cache(Path(fold["protocol"]["data_root"]), design_from_protocol(fold["protocol"]), "test"))
    if expected_cache not in caches or cache != {"path": expected_cache, "sha256": caches[expected_cache]}:
        raise ValueError("method_score_actual_image_cache_mismatch")
    # Check bytes before deserializing. A replaced/corrupt trial file is an
    # identity error, not an uncontrolled pickle/loader exception.
    verify_dependencies([*held_out_files, cache])
    actual_inputs = inference.get("actual_inputs") or {}
    if (actual_inputs.get("query_count") != metrics.get("query_count")
            or actual_inputs.get("candidate_count") != metrics.get("candidate_count")
            or inference.get("input_geometry") != fold["protocol"]["input_geometry"]):
        raise ValueError("method_score_actual_query_gallery_or_preprocess_mismatch")
    for key in ("positive_map_sha256", "gallery_image_ids_sha256", "image_features_sha256", "validation_eeg_sha256"):
        value = actual_inputs.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("method_score_actual_positive_gallery_identity_missing")
    # Re-derive the two identities directly from the immutable subject file.
    # This reads benchmark metadata only; no model choice or fit consumes it.
    query_rows = read_query_rows((Path(expected_file),), fold["protocol"]["input_geometry"]["channels"], fold["held_out_subject"])
    expected_positives = [[query, image] for query, image in query_rows]
    expected_gallery = list(dict.fromkeys(image for _query, image in query_rows))
    if (actual_inputs["positive_map_sha256"] != identity(expected_positives)
            or actual_inputs["gallery_image_ids_sha256"] != identity(expected_gallery)
            or actual_inputs["query_count"] != len(query_rows) or actual_inputs["candidate_count"] != len(expected_gallery)):
        raise ValueError("method_score_actual_positive_map_mismatch")
    # On resume/aggregate, a changed physical input invalidates this record;
    # native recovery never silently scores a replacement cache or subject file.


def verified_suite_records(camp: Path, state: dict) -> list[dict]:
    records = []
    for row in state.get("evidence") or []:
        if row.get("kind") != "method_suite" or row.get("evaluation_valid") is not True:
            continue
        try:
            result = read_aggregate(camp, row["suite_id"])
            if (result == row["aggregate"] and result["method_revision"] == row["method_revision"]
                    and digest(Path(row["suite_ref"])) == row["result_hash"]):
                records.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def suite_cost_estimates(camp: Path, state: dict, observation: dict) -> dict:
    from react_agent.eeg_research.agentic.planner import action_target_choices
    common = manifest(camp)
    estimates = {}
    for action in observation["available_actions"]:
        targets, null_allowed = action_target_choices(observation, action,
            legacy_implicit=observation.get("planner_mode") == "single_action")
        for target in [*([None] if null_allowed else []), *targets]:
            training = action == "run_full"
            design = action in {"design_experiment", "propose_experiment", "implement_candidate"}
            local = action in {"inspect_data", "retrieve_memory", "collect_diagnostics", "diagnose_results", "revise_report", "stop"}
            estimates[f"{action}:{target or ''}"] = {
                "estimated_cost": {"llm_calls": 0 if local else None, "training_jobs": 10 if training else 0 if local else None,
                                   "gpu_seconds": common["gpu_seconds_estimate_per_suite"] if training else 0.0 if local else None},
                "cost_basis": ["runtime:complete_method_suite:10_full_10_cpu_scores", "manifest:" + common["manifest_hash"]],
                "cost_confidence": "rough" if training or design else "runtime_bound" if local else "unknown",
                "cost_scope": "complete_method_suite_pipeline", "components": {
                    "full_training": {"training_jobs": 10, "epochs_per_fold": 50, "seed": 0},
                    "benchmark_scoring": {"folds": 10, "device": "cpu", "gpu_seconds": 0.0},
                    "repair_allowance": {"maximum_repairs_per_candidate": 2, "additional_jobs": "only_if_needed_and_budgeted"}},
                "full_followup_pipeline_cost": {"training_jobs": 10 if design else 0,
                    "gpu_seconds": common["gpu_seconds_estimate_per_suite"] if design else 0.0, "llm_calls": None},
                "unknown_components": [] if local else ["role_retries", "repair_calls", "candidate_runtime_multiplier"],
                "historical_samples": [], "api_usd": None, "usd_status": "unpriced",
                "estimates_are_execution_authority": False, "planner_call_already_in_budget_ledger": True}
    return estimates


def commit_aggregate(camp: Path, state: dict, suite: dict) -> dict:
    directory = camp / "suites" / suite["suite_id"]
    paths = [directory / fold["fold_id"] / "benchmark.json" for fold in suite["folds"]]
    result = aggregate_records(suite, [read(path) for path in paths if path.exists()])
    for score in result["scores"]:
        verify_score(camp, suite, score)
    if result["status"] != "complete":
        raise ValueError("method_suite_not_complete")
    wrapper = {"aggregate": result, "score_dependencies": [{"path": str(path), "sha256": digest(path)} for path in paths]}
    path = directory / "aggregate.json"
    if not path.exists():
        write_once(path, wrapper)
    else:
        read_aggregate(camp, suite["suite_id"])
    eid = "ev_suite_" + suite["suite_id"]
    existing = next((row for row in state["evidence"] if row.get("evidence_id") == eid), None)
    if existing:
        return existing
    from react_agent.eeg_research.agentic.artifacts import register
    artifact = register(camp, path, kind="method_suite", candidate_id=result["candidate_id"], artifact_id="art_" + suite["suite_hash"][:24])
    control = next((row for row in state["evidence"] if row.get("kind") == "method_suite" and row.get("candidate_id") == "baseline"
                    and row.get("evaluation_valid") and row.get("seed") == result["seed"]), None)
    comparison = {"comparable": False, "reason": "baseline_reference"}
    if control and result["candidate_id"] != "baseline":
        baseline = read_aggregate(camp, control["suite_id"])
        if control["execution_fingerprint"] != state["execution_fingerprint"]:
            raise ValueError("method_suite_control_recipe_mismatch")
        a = {row["held_out_subject"]: row["metrics"]["top1"] for row in result["scores"]}
        b = {row["held_out_subject"]: row["metrics"]["top1"] for row in baseline["scores"]}
        deltas = {subject: 100 * (a[subject] - b[subject]) for subject in SUBJECTS}
        comparison = {"comparable": True, "metric": METRIC, "metric_unit": "fraction", "delta_pp": statistics.mean(deltas.values()),
                      "subject_deltas_pp": deltas, "control_run_id": control["evidence_id"], "paired_complete_suites": 1,
                      "independent_seed_count": 1, "replicated": False, "statistical_significance": "not_established"}
        comparison.update(required_fold_count=10, candidate_suite_hash=suite["suite_hash"], control_suite_hash=control["suite_hash"])
    goal = read(camp / "goal.json")
    gain = comparison.get("delta_pp")
    promotion = {"status": "development_improvement" if gain is not None and gain >= float(goal["min_practical_gain_pp"])
                 else "development_negative" if gain is not None else "baseline_reference", "replicated": False,
                 "independent_seed_count": 1, "paired_complete_suites": 1 if control else 0,
                 "scope": "method_development_benchmark", "statistical_significance": "not_established"}
    if control:
        from react_agent.eeg_research.agentic.promotion import promotion_decision
        promotion = promotion_decision(comparison=comparison, goal=goal, fidelity="full")
    row = {"evidence_id": eid, "kind": "method_suite", "candidate_id": result["candidate_id"], "suite_id": suite["suite_id"],
           "suite_ref": str(path), "artifact_refs": [artifact["artifact_id"]], "suite_hash": suite["suite_hash"],
           "evaluation_mode": MODE, "metric_scope": "method_development_benchmark", "primary_metric": METRIC,
           "aggregate": result, "evaluation_valid": True, "fidelity": "full", "seed": result["seed"],
           "source_hash": suite["method"]["source_hash"], "spec_hash": suite["method"]["spec_hash"],
           "config_hash": identity(suite["method"]["hook_spec"]), "method_revision": result["method_revision"],
           "execution_fingerprint": state["execution_fingerprint"], "contract_fingerprint": state["contract_fingerprint"],
           "result_hash": digest(path), "fixed_bank_top1": result["mean_top1"], "fixed_bank_top5": result["mean_top5"],
           "gallery_size": 200, "job_status": "finished", "comparison": comparison, "promotion": promotion,
           "diagnostics": {"scope": "ten_independent_folds", "folds": [read(directory / fold["fold_id"] / "benchmark_freeze.json")
                                                                      for fold in suite["folds"]]}}
    if gain is not None:
        row.update(delta_vs_control_pp=gain, control_id=control["evidence_id"])
    from react_agent.eeg_research.agentic.memory import EpisodeStore, episode
    ep = episode(task_hash=state["goal_id"], candidate_id=result["candidate_id"], kind="exploratory_result", fidelity="full",
                 seed=result["seed"], metric=result["mean_top1"], contract_fingerprint=state["contract_fingerprint"],
                 artifact=str(path), episode_id="ep_" + suite["suite_hash"][:24], job_id=suite["suite_id"])
    ep["scope"] = "method_development_benchmark"
    stored = EpisodeStore(camp).persist_episode(ep)
    state["memory"] = [item for item in state.get("memory", []) if item.get("episode_id") != stored["episode_id"]] + [stored]
    state["evidence"].append(row)
    from react_agent.eeg_research.agentic.loop import _write
    _write(camp / "comparisons" / (eid + ".json"), {"comparison": comparison, "promotion": promotion})
    state["audit_status"] = "stale"
    state["audit_fresh"] = False
    return row


def scientific_feedback_records(camp: Path, state: dict, row: dict) -> dict:
    """Find completed, source-bound native analysis and curation artifacts."""
    from react_agent.eeg_research.agentic.artifacts import REGISTRY
    from react_agent.eeg_research.agentic.handoffs import development_artifact, method_benchmark_origin
    origin = method_benchmark_origin(row)
    records = {}
    registry_path = camp / REGISTRY
    if not registry_path.is_file():
        return records
    for line in registry_path.read_text().splitlines():
        if not line.strip():
            continue
        artifact = json.loads(line)
        kind = artifact.get("kind")
        if kind not in {"analysis", "lessons"}:
            continue
        try:
            body = read(Path(artifact["path"]))
            if body.get("status") != "completed" or (body.get("payload") or {}).get("method_benchmark_origin") != origin:
                continue
            verified = development_artifact(camp, artifact["artifact_id"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        records["analysis" if kind == "analysis" else "curation"] = {
            "artifact_id": verified["artifact_id"], "sha256": verified["sha256"],
            "task_id": body["task_id"], "input_digest": body["input_digest"], "prompt_hash": body.get("prompt_hash"),
        }
    return records


def complete_scientific_feedback(camp: Path, state: dict, row: dict, services: dict) -> None:
    """Recover the method's original role chain before the next Planner call."""
    from react_agent.eeg_research.agentic.loop import _write, save_state
    records = scientific_feedback_records(camp, state, row)
    if set(records) != {"analysis", "curation"}:
        if not services.get("analyze"):
            raise ValueError("method_suite_scientific_feedback_service_missing")
        services["analyze"](camp, state)
        save_state(camp, state)
        records = scientific_feedback_records(camp, state, row)
    if set(records) != {"analysis", "curation"}:
        missing = sorted({"analysis", "curation"}.difference(records))
        raise ValueError("method_suite_scientific_feedback_incomplete:" + ",".join(missing))
    path = camp / "suites" / row["suite_id"] / "suite_state.json"
    progress = read(path)
    progress["scientific_feedback"] = {"status": "complete", "suite_hash": row["suite_hash"], **records}
    _write(path, progress)


def audit_evidence(camp: Path, state: dict) -> dict:
    """Build the report's complete-suite view and current immutable dependencies.

    Raw EEG/cache/checkpoint bytes are hash dependencies, never role payloads.
    The same strict readers used for scientific feedback authorize every suite
    and its completed native analysis/curation before an audit can consume it.
    """
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.handoffs import development_artifact, development_view, method_suite_scalar_view
    records = verified_suite_records(camp, state)
    declared = [row for row in state.get("evidence") or []
                if row.get("kind") == "method_suite" and row.get("evaluation_valid") is True]
    if len(records) != len(declared):
        raise ValueError("method_audit_has_invalid_complete_suite")
    dependencies = {}
    inputs = []
    views = []
    feedback_views = []

    def add(path: Path, *, sha256: str | None = None, kind: str = "method_dependency") -> None:
        path = path.resolve()
        actual = digest(path)
        if sha256 is not None and actual != sha256:
            raise ValueError("method_audit_dependency_changed:" + str(path))
        prior = dependencies.get(str(path))
        if prior is not None and prior["content_hash"] != actual:
            raise ValueError("method_audit_dependency_changed_during_projection")
        dependencies[str(path)] = {"path": str(path), "kind": kind, "content_hash": actual,
                                   "verification_status": "verified", "scope": "development",
                                   "data_use": "authorized_method_development_benchmark"}

    common = manifest(camp)
    add(camp / "method_search_manifest.json", kind="method_search_manifest")
    for dep in [*common["framework_dependencies"], *common["data_dependencies"], *common["cache_dependencies"]]:
        add(Path(dep["path"]), sha256=dep["sha256"])
    for row in records:
        suite = validate_suite(camp, row["suite_id"])
        directory = camp / "suites" / row["suite_id"]
        wrapper = read(Path(row["suite_ref"]))
        views.append(method_suite_scalar_view(row, score_dependencies=wrapper["score_dependencies"]))
        add(directory / "suite_manifest.json", kind="method_suite_manifest")
        add(Path(row["suite_ref"]), sha256=row["result_hash"], kind="method_suite_aggregate")
        inputs.append(Path(row["suite_ref"]))
        for fold in suite["folds"]:
            add(Path(fold["protocol_path"]), kind="method_fold_protocol")
        for dep in suite["method"]["dependencies"]:
            add(Path(dep["path"]), sha256=dep["sha256"])
        for dep in wrapper["score_dependencies"]:
            add(Path(dep["path"]), sha256=dep["sha256"], kind="method_fold_benchmark")
            score = read(Path(dep["path"]))
            freeze = read(Path(score["freeze_ref"]))
            add(Path(score["freeze_ref"]), sha256=score["freeze_sha256"], kind="method_fold_freeze")
            for bound in freeze["dependencies"]:
                add(Path(bound["path"]), sha256=bound["sha256"])
        feedback = scientific_feedback_records(camp, state, row)
        if set(feedback) != {"analysis", "curation"}:
            raise ValueError("method_audit_scientific_feedback_incomplete")
        for stage in ("analysis", "curation"):
            verified = development_artifact(camp, feedback[stage]["artifact_id"])
            artifact = resolve_verified_artifact(camp, verified["artifact_id"])
            add(Path(artifact["path"]), sha256=verified["sha256"], kind=stage)
            inputs.append(Path(artifact["path"]))
            feedback_views.append({"stage": stage, "suite_id": row["suite_id"],
                "candidate_id": row["candidate_id"], "artifact_id": verified["artifact_id"],
                "path": artifact["path"], "sha256": verified["sha256"],
                "native_origin_verified": True, "envelope": development_view(read(Path(artifact["path"])))})
    return {"methods": views, "latest": views[-1] if views else None,
            "dependency_manifest": sorted(dependencies.values(), key=lambda item: item["path"]),
            "inputs": inputs, "scientific_feedback": feedback_views,
            "independent_seed_count": 1 if views else 0,
            "scope": "method_development_benchmark", "replicated": False,
            "independent_final_test": False, "statistical_significance": "not_established"}


def advance(camp: Path, state: dict, services: dict) -> bool:
    """One execution step, with no paid scientific calls until 10/10 complete."""
    from react_agent.eeg_research.agentic import jobs
    from react_agent.eeg_research.agentic.budget import charge_gpu, release_gpu
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
    from react_agent.eeg_research.agentic.loop import _write, save_state
    from react_agent.eeg_training.protocol import torch_python
    suite_id = state.get("active_suite")
    if not suite_id:
        complete_rows = verified_suite_records(camp, state)
        if not any(row.get("candidate_id") == "baseline" for row in complete_rows):
            begin(camp, state, "baseline")
            save_state(camp, state)
            return True
        latest = complete_rows[-1]
        if set(scientific_feedback_records(camp, state, latest)) != {"analysis", "curation"}:
            complete_scientific_feedback(camp, state, latest, services)
            state["status"] = "created"
            save_state(camp, state)
            return True
        return False
    suite = validate_suite(camp, suite_id)
    directory = camp / "suites" / suite_id
    progress_path = directory / "suite_state.json"
    progress = read(progress_path)
    for fold in suite["folds"]:
        fold_id = fold["fold_id"]
        if fold_id in progress["completed_folds"]:
            continue
        job_id = progress["jobs"].get(fold_id)
        if job_id:
            job = camp / "jobs" / job_id
            if not (job / "job.json").exists():
                raise ValueError("method_fold_launch_uncertain_requires_reconciliation")
            record = jobs.reconcile(job)
            if record["status"] == "running":
                state.update(status="training", live_job=job_id)
                save_state(camp, state)
                return True
            charge_gpu(camp, state, record.get("gpu_seconds", 0), job_id=job_id)
            state["live_job"] = None
            freeze_fold(camp, suite, fold, job)
            # CPU scorer reuses the original implementation and carries no API
            # secrets. Its output is immutable and idempotent by evaluation key.
            env = jobs.research_env(None)
            env.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
            result = subprocess.run([torch_python(), "-m", "react_agent.eeg_research.agentic.method_evaluate",
                                     "--camp", str(camp), "--suite", suite_id, "--fold", fold_id],
                                    env=env, capture_output=True, text=True, timeout=300)
            if result.returncode:
                raise ValueError("method_benchmark_scoring_failed:" + result.stderr[-1800:])
            progress["completed_folds"].append(fold_id)
            progress["status"] = "running"
            _write(progress_path, progress)
            save_state(camp, state)
            return True
        if state["training_jobs"] >= state["max_training_jobs"] or state["gpu_seconds_left"] <= 0:
            raise ValueError("method_suite_budget_exhausted_partial")
        from react_agent.eeg_research.agentic.loso_study import gpu_inventory
        physical = gpu_inventory(tuple(manifest(camp)["gpu"]), allow_shared=True)
        hardware_keys = ("index", "uuid", "name", "total_memory_mib")
        if ([{key: row[key] for key in hardware_keys} for row in physical]
                != [{key: row[key] for key in hardware_keys} for row in manifest(camp)["gpu_inventory"]]):
            raise ValueError("method_fold_physical_gpu_identity_changed")
        job_id = "j" + str(state["training_jobs"] + 1) + "_" + suite_id + "_" + fold_id
        job = camp / "jobs" / job_id
        # Persist intent first. If an API/worker crash interrupts process
        # publication, uncertainty blocks a duplicate launch rather than guesses.
        write_once(job / "launch_intent.json", {"suite_id": suite_id, "fold_id": fold_id, "protocol_hash": fold["protocol"]["fingerprint"]})
        progress["jobs"][fold_id] = job_id
        progress["status"] = "running"
        _write(progress_path, progress)
        state["training_jobs"] += 1
        state["method_jobs_reserved"] = max(0, int(state.get("method_jobs_reserved", 10)) - 1)
        state["live_job"] = job_id
        save_state(camp, state)
        target = suite["method"]["candidate_id"]
        extension = None if target == "baseline" else camp / "candidates" / target / "extension"
        record = jobs.start_job(job, design_from_protocol(fold["protocol"], training_seed=0), candidate_id=target,
                                extension=extension, fidelity="full", root=Path(fold["protocol"]["data_root"]),
                                protocol_path=Path(fold["protocol_path"]), training_seed=0)
        if record["status"] != "running":
            raise ValueError("method_fold_launch_blocked:" + str(record.get("detail")))
        if target == "baseline":
            state["baseline_jobs"] += 1
        state["status"] = "training"
        save_state(camp, state)
        return True
    row = commit_aggregate(camp, state, suite)
    progress["status"] = "complete"
    _write(progress_path, progress)
    release_gpu(state, float(state.pop("method_gpu_reservation", 0)))
    state.update(active_suite=None, live_job=None, method_jobs_reserved=0, status="analyzing")
    save_state(camp, state)
    complete_scientific_feedback(camp, state, row, services)
    save_state(camp, state)
    return True


def retry_fold(camp: Path, state: dict, suite_id: str, fold_id: str, *, reason: str) -> dict:
    """Queue a bounded fresh process for a verified dead engineering attempt.

    All old jobs and cost receipts remain. Successful training only needs the
    idempotent scorer; it cannot be retrained to select another checkpoint.
    """
    from react_agent.eeg_research.agentic import jobs
    from react_agent.eeg_research.agentic.budget import charge_gpu
    from react_agent.eeg_research.agentic.loop import _write, event
    if not enabled(state) or not suite_id or not fold_id or not reason.strip():
        raise ValueError("method_fold_retry_requires_explicit_suite_fold_and_engineering_reason")
    if state.get("active_suite") != suite_id or state.get("status") not in {"blocked", "paused"}:
        raise ValueError("method_fold_retry_requires_paused_or_blocked_active_suite")
    # A terminal job or launch intent can precede the outer state checkpoint.
    # Recover the native counter/ledger before settlement or resource admission.
    recover(camp, state)
    suite = validate_suite(camp, suite_id, verify_data=True)
    if fold_id not in {row["fold_id"] for row in suite["folds"]}:
        raise ValueError("method_fold_retry_unknown_fold")
    path = camp / "suites" / suite_id / "suite_state.json"
    progress = read(path)
    if fold_id in progress["completed_folds"] or progress.get("status") == "complete":
        raise ValueError("method_fold_retry_cannot_replace_complete_evidence")
    job_id = progress["jobs"].get(fold_id)
    if not job_id or not (camp / "jobs" / job_id / "job.json").is_file():
        raise ValueError("method_fold_launch_uncertain_requires_reconciliation")
    if state.get("live_job") not in {None, job_id}:
        raise ValueError("method_fold_retry_cannot_replace_other_live_job")
    record = jobs.reconcile(camp / "jobs" / job_id)
    if jobs._alive(record) or record.get("status") == "running":
        raise ValueError("method_fold_retry_cannot_replace_live_process")
    if record.get("status") not in {"failed", "cancelled"}:
        raise ValueError("method_fold_retry_requires_failed_process_not_successful_training")
    attempts = list((progress.get("failed_attempts") or {}).get(fold_id) or [])
    charge_gpu(camp, state, record.get("gpu_seconds", 0), job_id=job_id)
    from react_agent.eeg_research.agentic.multigpu_check import refresh_probe_budget
    refresh_probe_budget(camp, state)
    if len(attempts) >= 2:
        raise ValueError("method_fold_engineering_retry_limit_reached")
    remaining_folds = 10 - len(progress["completed_folds"])
    remaining_estimate = manifest(camp)["gpu_seconds_estimate_per_suite"] * remaining_folds / 10
    if (state["max_training_jobs"] - state["training_jobs"] < remaining_folds
            or state["gpu_seconds_left"] < remaining_estimate):
        raise ValueError("method_fold_retry_cannot_reserve_remaining_full_jobs")
    attempts.append({"job_id": job_id, "job_ref": str(camp / "jobs" / job_id / "job.json"),
                     "job_sha256": digest(camp / "jobs" / job_id / "job.json"),
                     "status": record["status"], "gpu_seconds": record.get("gpu_seconds", 0),
                     "reason": reason.strip(), "retry_kind": "fresh_initialization_same_frozen_method"})
    progress.setdefault("failed_attempts", {})[fold_id] = attempts
    progress["jobs"].pop(fold_id)
    progress["status"] = "pending"
    _write(path, progress)
    state["live_job"] = None
    state["status"] = "paused"
    state["method_jobs_reserved"] = remaining_folds
    state["gpu_seconds_reserved"] = remaining_estimate
    state["method_gpu_reservation"] = remaining_estimate
    event(camp, "method_fold_engineering_retry_queued", suite_id=suite_id, fold_id=fold_id,
          failed_job_id=job_id, engineering_retry=len(attempts), reason=reason.strip())
    return {"status": "engineering_retry_queued", "suite_id": suite_id, "fold_id": fold_id,
            "failed_job_id": job_id, "engineering_retry": len(attempts), "resume_required": True}


def recover(camp: Path, state: dict) -> dict:
    """Rebuild only method evidence; never project individual fold scores."""
    from react_agent.eeg_research.agentic.multigpu_check import recover_probes
    recover_probes(camp, state)
    for path in sorted((camp / "suites").glob("*/suite_state.json")):
        progress = read(path)
        suite = validate_suite(camp, path.parent.name)
        if (path.parent / "aggregate.json").exists():
            read_aggregate(camp, path.parent.name)
            commit_aggregate(camp, state, suite)
        elif progress["status"] in {"pending", "running"}:
            if state.get("active_suite") not in {None, path.parent.name}:
                raise ValueError("method_multiple_active_suites")
            state["active_suite"] = path.parent.name
    intents = {path.parent.name for path in (camp / "jobs").glob("*/launch_intent.json")}
    records = {path.parent.name for path in (camp / "jobs").glob("*/job.json")}
    cost_path = camp / "cost.json"
    cost = read(cost_path) if cost_path.exists() else {}
    state["training_jobs"] = max(state["training_jobs"], len(intents | records), int(cost.get("training_jobs", 0)))
    used = float(cost.get("gpu_seconds_used", 0))
    state["gpu_seconds_left"] = float(state["max_gpu_seconds"]) - used
    if state.get("active_suite"):
        directory = camp / "suites" / state["active_suite"]
        progress = read(directory / "suite_state.json")
        estimate = manifest(camp)["gpu_seconds_estimate_per_suite"]
        remaining_estimate = estimate * (10 - len(progress["completed_folds"])) / 10
        state["gpu_seconds_reserved"] = remaining_estimate
        state["method_gpu_reservation"] = remaining_estimate
        state["method_jobs_reserved"] = 10 - len(progress["jobs"])
    else:
        state["gpu_seconds_reserved"] = 0.0
        state["method_jobs_reserved"] = 0
    return state
