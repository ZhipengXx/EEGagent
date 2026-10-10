"""Bound sequential native campaigns and freeze LOSO benchmark dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

SUBJECTS = tuple(f"sub-{i:02d}" for i in range(1, 11))
CAPS = {"llm_calls": 400, "training_jobs": 48, "gpu_seconds": 172800.0}
TERMINAL = {"finished", "blocked", "cancelled"}


def read(path: Path) -> dict:
    """Read required JSON without replacing missing records with zero usage."""
    from react_agent.eeg_research.agentic.verification_cache import read_json
    return read_json(path, lambda source: json.loads(source.read_text(encoding="utf-8")))


def digest(path: Path) -> str:
    """Hash a dependency without loading its data as EEG samples."""
    from react_agent.eeg_research.agentic.verification_cache import file_hash
    return file_hash(path, _digest_bytes)


def _digest_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value: dict) -> str:
    """Hash a canonical study or evaluation identity."""
    from react_agent.eeg_research.agentic.verification_cache import object_identity
    return object_identity(value, lambda body: hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest())


def write_once(path: Path, payload: dict) -> None:
    """Install a new native record; never overwrite an existing freeze or score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)  # Atomic creation; an existing record is never replaced.
    finally:
        temporary.unlink()


def folds() -> list[dict]:
    """Declare each subject once as held out, with its nine source subjects."""
    return [{"fold_id": f"fold_{i:02d}", "held_out_subject": held,
             "train_subjects": [subject for subject in SUBJECTS if subject != held]}
            for i, held in enumerate(SUBJECTS, 1)]


def gpu_inventory(gpus: tuple[int, ...], *, allow_shared: bool) -> list[dict]:
    """Bind physical GPU UUIDs and check explicit sharing authorization."""
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    devices = {}
    for line in output.splitlines():
        index, uuid, name, total, used = [part.strip() for part in line.split(",")]
        devices[int(index)] = {"index": int(index), "uuid": uuid, "name": name,
                               "total_memory_mib": int(total), "used_memory_mib": int(used)}
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    occupied = {line.split(",")[0].strip() for line in processes.splitlines()}
    selected = [devices[index] for index in gpus]
    if not allow_shared and any(row["uuid"] in occupied for row in selected):
        raise ValueError("study_gpu_occupied_without_sharing_authorization")
    return selected


def create(study: Path, template: dict, data_root: Path, gpus: tuple[int, ...], *, allow_shared_gpus: bool = False) -> dict:
    """Freeze the authorized study before any native campaign or paid call."""
    from react_agent.eeg_research.agentic.embedding import MemoryConfig
    from react_agent.eeg_research.agentic.schemas import GoalSpec
    from react_agent.eeg_training.protocol import (
        Design,
        data_dir,
        feature_cache,
        list_subjects,
        split_plan,
    )

    study, data_root = study.resolve(), data_root.resolve()
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("study_requires_multiple_distinct_gpus")
    inventory = gpu_inventory(gpus, allow_shared=allow_shared_gpus)
    GoalSpec.model_validate(template)
    required = {"training_seeds": [0], "confirmation_target_pairs": 1, "max_candidates": 1,
                "max_concurrent_training_jobs": 1, "final_test_enabled": False,
                "max_repairs_per_candidate": 2, "require_audit_before_completion": True,
                "allowed_training_actions": ["run_pilot", "run_full"]}
    if any(template.get(key) != value for key, value in required.items()):
        raise ValueError("study_goal_template_mismatch")
    memory = MemoryConfig.model_validate(template.get("memory") or {}).model_dump()
    expected_memory = {"enabled": True, "skills_enabled": True, "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                       "device": "cpu", "batch_size": 32, "top_k": 5, "min_similarity": 0.3,
                       "context_chars_budget": 6000, "allow_model_download": False, "fallback": "legacy"}
    if any(memory.get(key) != value for key, value in expected_memory.items()) or not re.fullmatch(r"[0-9a-f]{40}", memory.get("model_revision") or ""):
        raise ValueError("study_requires_frozen_minilm_configuration")
    template = {**template, "memory": memory}
    if tuple(list_subjects(data_root, "eeg")) != SUBJECTS:
        raise ValueError("study_subject_universe_mismatch")
    data = data_dir(data_root, "eeg")
    inputs = {}
    for subject in SUBJECTS:
        for name in ("train.pt", "test.pt"):
            path = data / subject / name
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError("empty_subject_data")
            # Identity metadata only; no held-out EEG is decoded at study creation.
            inputs[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    selected = []
    for row in folds():
        design = Design("eeg", "inter-subject", ",".join(row["train_subjects"]), gpu=gpus,
                        policy="agentic", training_strategy="pooled_subjects")
        plan = split_plan(data_root, design)
        if plan.train_subjects != tuple(row["train_subjects"]) or plan.forbidden_files != (data / row["held_out_subject"] / "test.pt",):
            raise ValueError("study_split_mismatch")
        selected.append({**row, "campaign_id": study.name + "_" + row["fold_id"],
                         "train_files": list(map(str, plan.train_files)),
                         "development_files": list(map(str, plan.val_files)),
                         "held_out_file": str(plan.forbidden_files[0])})
    caches = {str(feature_cache(data_root, design, mode)): digest(feature_cache(data_root, design, mode))
              for mode in ("train", "test")}
    from react_agent.eeg_training.data import load_feature_cache
    if len(load_feature_cache(feature_cache(data_root, design, "test"))) != 200:
        raise ValueError("study_requires_complete_200_image_test_cache")
    body = {"schema_version": "eeg_research.loso_study.v1", "study_id": study.name,
            "created_at": time.time(), "data_root": str(data_root), "gpu": list(gpus),
            "gpu_inventory": inventory, "cuda_visible_devices": ",".join(map(str, gpus)),
            "logical_device_mapping": {str(i): row["uuid"] for i, row in enumerate(inventory)},
            "gpu_sharing_authorized": allow_shared_gpus,
            "seed": 0, "split_seed": 0, "folds": selected, "caps": CAPS,
            "goal_template": template, "data_files": inputs, "cache_hashes": caches,
            "selection_rule": "The sole native non-baseline candidate with valid seed-0 full evidence; no held-out selection.",
            "benchmark_scope": "EEGagent workflow LOSO, independent fold search and memory",
            "aggregation": {"weighting": "equal_subject", "ddof": 1, "metric_unit": "fraction",
                            "gallery_size": 200, "partial_is_not_complete": True},
            "final_evaluation": {"authorized": True, "device": "cpu", "after_study_freeze_only": True}}
    body["manifest_hash"] = identity(body)
    write_once(study / "study_manifest.json", body)
    return body


def manifest(study: Path) -> dict:
    """Validate immutable study metadata and declared data/cache dependencies."""
    body = read(study / "study_manifest.json")
    if identity({key: value for key, value in body.items() if key != "manifest_hash"}) != body.get("manifest_hash"):
        raise ValueError("study_manifest_changed")
    for filename, expected in body["data_files"].items():
        stat = Path(filename).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError("study_data_changed:" + filename)
    for filename, expected in body["cache_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("study_cache_changed:" + filename)
    return body


def campaign_path(study: Path, fold: dict) -> Path:
    """Resolve the sole predeclared native campaign for a fold."""
    return study / "campaigns" / fold["campaign_id"]


def validate_fold_protocol(protocol: dict, fold: dict, body: dict) -> None:
    """Require the declared source split and unchanged common training recipe."""
    expected = {"subject": ",".join(fold["train_subjects"]), "gpu": body["gpu"], "training_seeds": [0],
                "training_seed": 0, "split_seed": 0, "dataset": "eeg", "exp_setting": "inter-subject",
                "data_root": body["data_root"], "training_strategy": "pooled_subjects", "val_mode": "other_subjects_test",
                "full_epochs": 50, "batch_size": 1024, "lr": 1e-5, "weight_decay": 1e-4,
                "negative_sampling_policy": "data_parallel_local", "final_test_enabled": False,
                "fidelity_overrides": {"pilot": {"epochs": 3, "stop": "single_full"}, "full": {"epochs": 50, "stop": "single_early"}}}
    if any(protocol.get(key) != value for key, value in expected.items()) or protocol.get("validation_files") != fold["development_files"]:
        raise ValueError("study_fold_protocol_or_recipe_mismatch")
    if len(protocol.get("gallery_image_ids") or []) != 200:
        raise ValueError("study_fold_development_gallery_not_200")


def usage(study: Path, body: dict) -> dict:
    """Sum original native ledgers, including live allocated GPU occupancy."""
    from react_agent.eeg_research.agentic.jobs import _alive
    totals = {"llm_calls": 0, "training_jobs": 0, "gpu_seconds": 0.0,
              "input_tokens": 0, "output_tokens": 0, "api_usd": None}
    for fold in body["folds"]:
        camp = campaign_path(study, fold)
        if not (camp / "campaign_state.json").exists():
            continue
        state = read(camp / "campaign_state.json")
        calls = camp / "llm_calls.jsonl"
        records = len(calls.read_text().splitlines()) if calls.exists() else 0
        cost_path = camp / "cost.json"
        if not cost_path.exists() and (records or state.get("llm_calls") or state.get("training_jobs")):
            raise ValueError("study_native_usage_ledger_missing")
        cost = read(cost_path) if cost_path.exists() else {}
        totals["llm_calls"] += max(int(cost.get("llm_calls", 0)), int(state.get("llm_calls", 0)), records)
        jobs = list((camp / "jobs").glob("*/job.json"))
        totals["training_jobs"] += max(int(state.get("training_jobs", 0)), int(cost.get("training_jobs", 0)), len(jobs))
        gpu = float(cost.get("gpu_seconds_used", 0))
        settled = set(cost.get("settled_job_ids") or [])
        for path in jobs:
            record = read(path)
            if record.get("job_id") in settled:
                continue
            if _alive(record) or record.get("status") == "running":
                # Until native reconciliation settles a dead child, retain its
                # occupancy conservatively rather than allocate the time twice.
                end = record.get("ended_at") or time.time()
                gpu += max(float(record.get("gpu_seconds") or 0),
                           max(0, end - record["started_at"]) * max(1, len(record.get("gpu") or [])))
            else:
                gpu += float(record.get("gpu_seconds") or 0)
        totals["gpu_seconds"] += gpu
        for field in ("input_tokens", "output_tokens"):
            totals[field] += int(cost.get(field) or 0)
    return totals


def allocation(totals: dict, remaining_folds: int, *, n_gpu: int = 1) -> dict:
    """Reserve four job slots for each future fold and distribute remaining caps."""
    left = {key: CAPS[key] - totals[key] for key in CAPS}
    if remaining_folds < 1 or left["training_jobs"] < 4 * remaining_folds or left["llm_calls"] < remaining_folds or left["gpu_seconds"] <= 0:
        raise ValueError("study_budget_cannot_reserve_remaining_core_work")
    # Leave room for the existing 30-second polling and 60-second TERM grace.
    # Final evaluation is CPU; no final GPU inference is hidden in this quota.
    gpu_allowance = left["gpu_seconds"] / remaining_folds - max(1, n_gpu) * 120
    if gpu_allowance <= 0:
        raise ValueError("study_gpu_budget_cannot_cover_shutdown_margin")
    return {"max_llm_calls": int(left["llm_calls"] // remaining_folds),
            "max_training_jobs": 4 + min(2, int(left["training_jobs"] - 4 * remaining_folds)),
            "max_gpu_seconds": gpu_allowance}


def worker_alive(camp: Path) -> bool:
    """Check the recorded worker PID against the actual process, not a stale lock."""
    from react_agent.eeg_research.agentic.jobs import _proc_start, worker_disconnected
    path = camp / "worker.json"
    if not path.exists():
        return False
    record = read(path)
    pid = int(record.get("pid") or 0)
    if worker_disconnected(pid) or record.get("proc_start") and _proc_start(pid) != record["proc_start"]:
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        return ("react_agent.eeg_research.agentic.cli" in argv and "run" in argv
                and argv[argv.index("--campaign") + 1] == camp.name
                and Path(argv[argv.index("--root") + 1]).resolve() == camp.parent.resolve())
    except (OSError, ValueError, IndexError):
        return False


def status(study: Path) -> dict:
    """Return native fold status and non-resettable group usage."""
    body = manifest(study)
    rows = []
    for fold in body["folds"]:
        camp = campaign_path(study, fold)
        state = read(camp / "campaign_state.json") if (camp / "campaign_state.json").exists() else {}
        rows.append({**fold, "status": state.get("status", "not_created"), "detail": state.get("detail"),
                     "worker_alive": worker_alive(camp), "live_job": state.get("live_job")})
    consumed = usage(study, body)
    return {"study_id": body["study_id"], "phase": "frozen" if (study / "benchmark_freeze.json").exists() else "research",
            "folds": rows, "usage": consumed, "remaining": {key: CAPS[key] - consumed[key] for key in CAPS}}


def advance(study: Path) -> dict:
    """Create/start at most one fold through existing CLI; never make role decisions."""
    import fcntl

    from react_agent.eeg_research.agentic import cli
    from react_agent.eeg_research.agentic.jobs import _alive

    body = manifest(study)
    if (study / "benchmark_freeze.json").exists():
        raise ValueError("study_research_closed_after_freeze")
    current = gpu_inventory(tuple(body["gpu"]), allow_shared=body["gpu_sharing_authorized"])
    if [(row["index"], row["uuid"]) for row in current] != [(row["index"], row["uuid"]) for row in body["gpu_inventory"]]:
        raise ValueError("study_physical_gpu_mapping_changed")
    with (study / "study.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        totals = usage(study, body)
        for i, fold in enumerate(body["folds"]):
            camp = campaign_path(study, fold)
            args = ["--root", str(study / "campaigns"), "--campaign", fold["campaign_id"]]
            if (camp / "campaign_state.json").exists():
                state = read(camp / "campaign_state.json")
                live = any(_alive(read(p)) for p in (camp / "jobs").glob("*/job.json"))
                if (state.get("status") not in TERMINAL or live or worker_alive(camp)) and (
                        any(totals[key] > CAPS[key] for key in ("llm_calls", "training_jobs"))
                        or totals["gpu_seconds"] >= CAPS["gpu_seconds"] - len(body["gpu"]) * 120):
                    cli.main(["stop", *args])
                    return {"status": "budget_exhausted", **status(study)}
                if worker_alive(camp):
                    return {"status": "running", "fold": fold["fold_id"], **status(study)}
                unsettled = any(read(p).get("status") == "running" for p in (camp / "jobs").glob("*/job.json"))
                if live or unsettled or state.get("status") not in TERMINAL | {"paused"}:
                    cli.main(["resume", *args])
                    return {"status": "resumed", "fold": fold["fold_id"], **status(study)}
                if state.get("status") in {"paused", "blocked"}:
                    return {"status": "needs_native_resume_or_maintenance", "fold": fold["fold_id"], **status(study)}
                continue
            if any(worker_alive(campaign_path(study, row)) for row in body["folds"]):
                raise ValueError("another_fold_worker_live")
            if any(_alive(read(p)) for p in (study / "campaigns").glob("*/jobs/*/job.json")):
                raise ValueError("another_fold_training_live")
            quota = allocation(totals, len(body["folds"]) - i, n_gpu=len(body["gpu"]))
            goal = {**body["goal_template"], **quota, "goal_id": fold["campaign_id"]}
            goal_path = study / "inputs" / (fold["fold_id"] + ".json")
            if goal_path.exists():
                if read(goal_path) != goal:
                    raise ValueError("existing_fold_allocation_changed")
            else:
                write_once(goal_path, goal)
            rc = cli.main(["create", *args, "--goal", str(goal_path), "--subject", ",".join(fold["train_subjects"]),
                           "--gpu", ",".join(map(str, body["gpu"])), "--data-root", body["data_root"]])
            if rc:
                raise ValueError("native_fold_create_failed")
            protocol = read(camp / "execution_protocol.json")
            validate_fold_protocol(protocol, fold, body)
            cli.main(["start", *args])
            return {"status": "started", "fold": fold["fold_id"], **status(study)}
    return {"status": "research_terminal", **status(study)}


def freeze(study: Path, *, allow_partial: bool = False) -> dict:
    """Exclude native advancement while closing research and freezing its outputs."""
    import fcntl
    with (study / "study.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _freeze(study, allow_partial=allow_partial)


def _freeze(study: Path, *, allow_partial: bool = False) -> dict:
    """Select native valid full records before opening any held-out score."""
    from react_agent.eeg_research.agentic.comparison import (
        compare_runs,
        matched_control,
    )
    from react_agent.eeg_research.agentic.identity import result_matches
    from react_agent.eeg_research.agentic.jobs import _alive
    from react_agent.eeg_research.agentic.native_patch import _check_fingerprint
    from react_agent.eeg_research.agentic.research_progress import verified_seed_records
    from react_agent.eeg_research.agentic.runner import accept_job, comparable

    body = manifest(study)
    target = study / "benchmark_freeze.json"
    if target.exists():
        return validate_freeze(study)
    models, failures, dependencies = [], [], {}
    for fold in body["folds"]:
        camp = campaign_path(study, fold)
        if worker_alive(camp) or any(_alive(read(p)) for p in (camp / "jobs").glob("*/job.json")):
            raise ValueError("study_freeze_requires_idle_workers_and_jobs")
        state_file = camp / "campaign_state.json"
        if not state_file.exists():
            failures.append({"fold": fold["fold_id"], "reason": "campaign_not_created"})
            continue
        state, protocol = read(state_file), read(camp / "execution_protocol.json")
        if state.get("status") not in TERMINAL | {"paused"}:
            raise ValueError("study_freeze_requires_research_terminal")
        accepted = verified_seed_records(camp, state, protocol)
        full = [row for row in accepted if row["fidelity"] == "full" and row["seed"] == 0]
        candidates = {row["candidate_id"] for row in full if row["candidate_id"] != "baseline"}
        controls = [row for row in full if row["candidate_id"] == "baseline"]
        if len(candidates) != 1 or not controls:
            failures.append({"fold": fold["fold_id"], "reason": "missing_valid_native_candidate_or_baseline_full"})
            continue
        candidate = next(iter(candidates))
        workspace = camp / "candidates" / candidate
        check, review, attempt = read(workspace / "checks.json"), read(workspace / "review.json"), read(workspace / "attempt.json")
        source = digest(workspace / "extension/eeg_candidate.py")
        # Preserve the actual checker interpreter used by the frozen check.
        python = check.get("python") or (check.get("runtime_context") or {}).get("python")
        if not check.get("ok") or check.get("source_sha256") != source or check.get("check_fingerprint") != _check_fingerprint(workspace, python):
            raise ValueError("study_candidate_check_not_fresh")
        if review.get("status") != "ready" or not result_matches(review, candidate_id=candidate,
                attempt_id=str(attempt["attempt_id"]), phase="review_candidate", input_hash=source):
            raise ValueError("study_candidate_review_not_fresh")
        validate_fold_protocol(protocol, fold, body)
        selected_candidate = [row for row in full if row["candidate_id"] == candidate][-1]
        control = matched_control(controls, selected_candidate, protocol)
        comparison = compare_runs(candidate=selected_candidate, control=control, protocol=protocol)
        if not comparison["comparable"] or not comparable(selected_candidate, control or {}):
            raise ValueError("study_full_pair_not_comparable")
        pair = [control, selected_candidate]
        for evidence in pair:
            job = Path(evidence["job_dir"]).resolve()
            if (camp.resolve() not in job.parents):
                raise ValueError("study_job_outside_fold")
            checkpoint = job / "last.ckpt"
            selected = read(job / "selected_checkpoint.json")
            if not checkpoint.is_file() or not selected:
                raise ValueError("study_selected_checkpoint_missing")
            record, metrics = read(job / "job.json"), read(job / "metrics.json")
            accepted_job = accept_job(job, record.get("manifest"), "full", protocol=protocol)
            if not accepted_job.get("evaluation_valid") or accepted_job["fixed_bank_top1"] != evidence.get("fixed_bank_top1"):
                raise ValueError("study_full_metrics_no_longer_accepted")
            best, last = selected.get("best_epoch"), selected.get("last_epoch")
            if (selected.get("checkpoint") != "last.ckpt" or selected.get("source") != "selected_checkpoint"
                    or type(best) is not int or type(last) is not int or not 1 <= best <= last <= protocol["full_epochs"]
                    or metrics.get("selected_checkpoint_epoch") != best or metrics.get("last_epoch") != last
                    or selected.get("best_fixed_bank_top1") != accepted_job["fixed_bank_top1"]
                    or metrics.get("last_epoch_fixed_bank_top1") != selected.get("last_epoch_fixed_bank_top1")):
                raise ValueError("study_development_checkpoint_selection_mismatch")
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("epoch") != best or not isinstance(saved.get("state_dict"), dict) or not saved["state_dict"]:
                raise ValueError("study_checkpoint_epoch_or_weights_mismatch")
            model = {"fold_id": fold["fold_id"], "held_out_subject": fold["held_out_subject"],
                     "train_subjects": fold["train_subjects"], "held_out_file": fold["held_out_file"],
                     "candidate_id": evidence["candidate_id"], "source_hash": evidence["source_hash"],
                     "spec_hash": evidence["spec_hash"], "protocol_hash": protocol["fingerprint"],
                     "job_dir": str(job), "checkpoint": str(checkpoint), "checkpoint_hash": digest(checkpoint),
                     "selected_checkpoint": selected, "comparison": comparison,
                     "seed": 0, "fidelity": "full", "gpu": body["gpu"]}
            model["evaluation_key"] = identity(model)
            models.append(model)
        for p in camp.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                dependencies[str(p.resolve())] = digest(p)
    if failures and not allow_partial:
        raise ValueError("study_not_ready_for_complete_final_evaluation:" + json.dumps(failures))
    if not models:
        raise ValueError("study_has_no_valid_full_pairs_to_freeze")
    source_root = Path(__file__).resolve().parents[2]
    for p in (source_root / "eeg_training").glob("*.py"):
        dependencies[str(p)] = digest(p)
    for name in ("baseline.py", "hook_config.py", "execution_protocol.py"):
        p = Path(__file__).resolve().parent / name
        dependencies[str(p)] = digest(p)
    frozen = {"study_manifest_hash": body["manifest_hash"], "created_at": time.time(),
              "models": models, "failed_folds": failures, "dependencies": dependencies,
              "scope": body["benchmark_scope"], "research_closed": True,
              "device": "cpu", "metric_unit": "fraction", "gallery_size": 200,
              "study_implementation_hash": digest(Path(__file__)),
              "aggregate_weighting": "equal_subject", "ddof": 1}
    frozen["freeze_hash"] = identity(frozen)
    write_once(target, frozen)
    return frozen


def validate_freeze(study: Path) -> dict:
    """Reject changed method, checkpoint, campaign or evaluator dependencies."""
    body = manifest(study)
    frozen = read(study / "benchmark_freeze.json")
    if frozen.get("study_manifest_hash") != body["manifest_hash"] or identity({k: v for k, v in frozen.items() if k != "freeze_hash"}) != frozen.get("freeze_hash"):
        raise ValueError("study_freeze_changed")
    for filename, expected in frozen["dependencies"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("study_frozen_dependency_changed:" + filename)
    return frozen


def aggregate(study: Path) -> dict:
    """Compute subject macro means only from original identity-bound final scores."""
    import statistics
    frozen = validate_freeze(study)
    scores = {}
    for model in frozen["models"]:
        target = study / "benchmark" / (model["evaluation_key"] + ".json")
        if not target.exists():
            continue
        receipt = read(target.with_suffix(".receipt.json"))
        if receipt["score_sha256"] != digest(target) or receipt["freeze_hash"] != frozen["freeze_hash"]:
            raise ValueError("study_score_receipt_changed")
        score = read(target)
        if score.get("evaluation_key") != model["evaluation_key"] or score.get("freeze_hash") != frozen["freeze_hash"]:
            raise ValueError("study_score_identity_mismatch")
        metrics = score["metrics"]
        if metrics.get("candidate_count") != 200 or not isinstance(metrics.get("query_count"), int) or metrics["query_count"] <= 0:
            raise ValueError("study_score_population_mismatch")
        for name in ("top1", "top5"):
            if type(metrics.get(name)) not in (float, int) or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1:
                raise ValueError("study_score_unit_or_value_invalid")
        key = (model["fold_id"], model["candidate_id"] == "baseline")
        if key in scores:
            raise ValueError("study_duplicate_final_score")
        scores[key] = {"model": model, "metrics": metrics, "record": str(target), "sha256": digest(target)}
    rows, missing = [], []
    for fold in folds():
        baseline, candidate = scores.get((fold["fold_id"], True)), scores.get((fold["fold_id"], False))
        if not baseline or not candidate:
            missing.append(fold["fold_id"])
            continue
        if baseline["metrics"]["query_count"] != candidate["metrics"]["query_count"]:
            raise ValueError("study_paired_query_count_mismatch")
        rows.append({**fold, "baseline": baseline, "candidate": candidate,
                     "delta_pp": 100 * (candidate["metrics"]["top1"] - baseline["metrics"]["top1"])})
    columns = {name + "_" + metric: [row[name]["metrics"][metric] for row in rows]
               for name in ("baseline", "candidate") for metric in ("top1", "top5")}
    columns["delta_pp"] = [row["delta_pp"] for row in rows]
    means = {key: statistics.mean(values) if values else None for key, values in columns.items()}
    stds = {key: statistics.stdev(values) if len(values) > 1 else None for key, values in columns.items()}
    result = {"freeze_hash": frozen["freeze_hash"], "scope": frozen["scope"], "status": "complete" if not missing else "partial",
              "seed": 0, "unit": "fraction_except_delta_pp", "ddof": 1, "coverage": len(rows), "required_subjects": 10,
              "folds": rows, "missing_folds": missing, "subject_mean": means if not missing else None,
              "partial_subject_mean": means if missing else None, "subject_std": stds,
              "interpretation": "Single-seed workflow across subjects; not cross-seed replication or statistical significance.",
              "map": "not_implemented"}
    out = study / "benchmark" / ("aggregate_" + identity(result) + ".json")
    if not out.exists():
        write_once(out, result)
    return {**result, "artifact": str(out)}


def evaluate(study: Path) -> dict:
    """Run isolated CPU held-out inference after freeze; never train or alter jobs."""
    import os

    from react_agent.eeg_training.protocol import torch_python
    frozen = validate_freeze(study)
    for i, model in enumerate(frozen["models"]):
        target = study / "benchmark" / (model["evaluation_key"] + ".json")
        if target.exists():
            continue  # Aggregate validates the receipt; never select a better rerun.
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0")
        for key in ("EEG_CANDIDATE_MODULE", "EEG_CANDIDATE_PATH", "EEG_EVALUATION_IDENTITY"):
            env.pop(key, None)
        python = torch_python()
        if not python:
            raise ValueError("study_evaluation_python_missing")
        command = [python, "-m", "react_agent.eeg_research.agentic.loso_evaluate", "--study", str(study), "--model-index", str(i)]
        log = study / "benchmark" / (model["evaluation_key"] + f".{time.time_ns()}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as handle:
            completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise ValueError("study_final_evaluation_failed:" + str(log))
    return aggregate(study)
