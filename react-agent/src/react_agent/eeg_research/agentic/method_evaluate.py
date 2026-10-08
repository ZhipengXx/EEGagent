"""Authorized CPU benchmark inference for one immutable method fold."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from react_agent.eeg_research.agentic.loso_study import digest, identity, read, write_once
from react_agent.eeg_research.agentic.method_suite import aggregate_records, manifest, validate_suite, verify_dependencies


def evaluate(camp: Path, suite_id: str, fold_id: str) -> dict:
    suite = validate_suite(camp, suite_id, verify_data=True)
    fold = next(row for row in suite["folds"] if row["fold_id"] == fold_id)
    directory = camp / "suites" / suite_id / fold_id
    freeze = read(directory / "benchmark_freeze.json")
    if (freeze["suite_hash"] != suite["suite_hash"] or freeze["method_revision"] != suite["method"]["method_revision"]
            or freeze["permission"] != manifest(camp)["benchmark_permission"]
            or freeze["fold_id"] != fold_id or freeze["protocol_hash"] != fold["protocol"]["fingerprint"]):
        raise ValueError("method_benchmark_frozen_identity_mismatch")
    if freeze["evaluation_key"] != identity({k: v for k, v in freeze.items() if k != "evaluation_key"}):
        raise ValueError("method_benchmark_freeze_changed")
    verify_dependencies(freeze["dependencies"])
    target = directory / "benchmark.json"
    if target.exists():
        score = read(target)
        from react_agent.eeg_research.agentic.method_suite import verify_score
        verify_score(camp, suite, score)
        receipt_path = directory / "benchmark.receipt.json"
        if not receipt_path.exists():
            write_once(receipt_path, {"sha256": digest(target), "evaluation_key": freeze["evaluation_key"]})
        receipt = read(receipt_path)
        if score["evaluation_key"] != freeze["evaluation_key"] or receipt["sha256"] != digest(target):
            raise ValueError("method_benchmark_existing_score_changed")
        aggregate_records(suite, [score])
        return score
    os.environ.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0")
    for name in ("EEG_CANDIDATE_MODULE", "EEG_CANDIDATE_PATH", "EEG_EVALUATION_IDENTITY"):
        os.environ.pop(name, None)
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
    from react_agent.eeg_training.protocol import split_plan
    from react_agent.eeg_training.train_entry import score_held_out
    from react_agent.eeg_training.inference_consistency import validate_reconstruction_context
    design = design_from_protocol(fold["protocol"], training_seed=0)
    base = Path(fold["protocol"]["data_root"])
    plan = split_plan(base, design)
    if (list(plan.train_subjects) != fold["train_subjects"] or list(map(str, plan.train_files)) != fold["train_files"]
            or list(map(str, plan.val_files)) != fold["development_files"]
            or tuple(map(str, plan.forbidden_files)) != (fold["benchmark_file"],)):
        raise ValueError("method_benchmark_actual_split_mismatch")
    job = Path(freeze["job_dir"])
    if read(job / "job.json").get("status") != "finished" or not validate_reconstruction_context(job, job / "last.ckpt"):
        raise ValueError("method_benchmark_requires_finished_bound_checkpoint")
    started = time.time()
    metrics = score_held_out(design, base, job)
    if metrics is None:
        raise ValueError("method_benchmark_score_missing")
    body = {"status": "valid", "suite_hash": suite["suite_hash"], "method_revision": suite["method"]["method_revision"],
            "candidate_id": suite["method"]["candidate_id"], "fold_id": fold_id,
            "held_out_subject": fold["held_out_subject"], "train_subjects": fold["train_subjects"],
            "seed": 0, "fidelity": "full", "protocol_hash": fold["protocol"]["fingerprint"],
            "evaluation_key": freeze["evaluation_key"], "evaluator_hash": freeze["evaluator_hash"],
            "checkpoint_hash": freeze["checkpoint_hash"], "metrics": metrics,
            "scope": "method_development_benchmark", "device": "cpu", "precision": "fp32",
            "gpu_seconds": 0.0, "started_at": started, "ended_at": time.time(),
            "freeze_ref": str(directory / "benchmark_freeze.json"), "freeze_sha256": digest(directory / "benchmark_freeze.json")}
    aggregate_records(suite, [body])
    from react_agent.eeg_research.agentic.method_suite import verify_score
    verify_score(camp, suite, body)
    validate_suite(camp, suite_id, verify_data=True)
    verify_dependencies(freeze["dependencies"])
    # Neither metrics nor checkpoint/state are modified. If publication crashes
    # after score creation, the missing receipt is reconstructed from its exact
    # immutable key on native recovery, never by rescoring for a better value.
    write_once(target, body)
    write_once(directory / "benchmark.receipt.json", {"sha256": digest(target), "evaluation_key": freeze["evaluation_key"]})
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camp", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--fold", required=True)
    args = parser.parse_args(argv)
    evaluate(args.camp.resolve(), args.suite, args.fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
