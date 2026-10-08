"""Explicit frozen-study held-out inference; never fit or overwrite job outputs."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from react_agent.eeg_research.agentic.loso_study import (
    digest,
    manifest,
    read,
    validate_freeze,
    write_once,
)


def evaluate_model(study: Path, model_index: int) -> dict:
    """Rebuild the bound full model and score the single frozen held-out subject."""
    body, frozen = manifest(study), validate_freeze(study)
    if not body["final_evaluation"]["authorized"] or not frozen["research_closed"]:
        raise ValueError("held_out_evaluation_not_authorized_or_frozen")
    if model_index < 0 or model_index >= len(frozen["models"]):
        raise ValueError("held_out_model_index_invalid")
    model = frozen["models"][model_index]
    target = study / "benchmark" / (model["evaluation_key"] + ".json")
    if target.exists():
        receipt = read(target.with_suffix(".receipt.json"))
        if receipt["score_sha256"] != digest(target) or receipt["freeze_hash"] != frozen["freeze_hash"]:
            raise ValueError("held_out_existing_score_invalid")
        return read(target)
    # Keep evaluation in CPU fp32 using the existing scorer and train environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["EEG_TRAIN_DEVICE"] = "cpu"
    os.environ["EEG_FINAL_TEST"] = "0"
    for name in ("EEG_CANDIDATE_MODULE", "EEG_CANDIDATE_PATH", "EEG_EVALUATION_IDENTITY"):
        os.environ.pop(name, None)
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
    from react_agent.eeg_training.protocol import split_plan
    from react_agent.eeg_training.train_entry import score_held_out

    job = Path(model["job_dir"])
    protocol = read(job.parent.parent / "execution_protocol.json")
    design = design_from_protocol(protocol, training_seed=0)
    split = split_plan(Path(body["data_root"]), design)
    if list(split.train_subjects) != model["train_subjects"] or tuple(map(str, split.forbidden_files)) != (model["held_out_file"],):
        raise ValueError("held_out_actual_loader_split_mismatch")
    if protocol["fingerprint"] != model["protocol_hash"] or protocol["gpu"] != model["gpu"] or digest(job / "last.ckpt") != model["checkpoint_hash"]:
        raise ValueError("held_out_model_dependency_mismatch")
    started = time.time()
    metrics = score_held_out(design, Path(body["data_root"]), job)
    if metrics is None:
        raise ValueError("held_out_score_missing_or_not_200_way")
    metrics = dict(metrics)
    # FixedBankTally returns counts as integer-valued floats. Normalize their
    # representation without accepting fractional, nonfinite or boolean counts.
    for name in ("query_count", "candidate_count"):
        count = metrics.get(name)
        if type(count) not in (int, float) or not math.isfinite(count) or count != int(count) or count <= 0:
            raise ValueError("held_out_score_invalid_population")
        metrics[name] = int(count)
    if metrics["candidate_count"] != 200:
        raise ValueError("held_out_score_missing_or_not_200_way")
    if any(type(metrics.get(name)) not in (int, float)
            or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1 for name in ("top1", "top5")):
        raise ValueError("held_out_score_invalid_unit_or_value")
    # Recheck every frozen dependency after scorer invocation, before publication.
    validate_freeze(study)
    result = {"evaluation_key": model["evaluation_key"], "freeze_hash": frozen["freeze_hash"],
              "model": model, "metrics": metrics, "device": "cpu", "precision": "fp32",
              "held_out_files": list(map(str, split.forbidden_files)), "metric_unit": "fraction",
              "started_at": started, "ended_at": time.time(), "gpu_seconds": 0,
              "research_feedback_prohibited": True}
    write_once(target, result)
    write_once(target.with_suffix(".receipt.json"), {"score_sha256": digest(target),
               "freeze_hash": frozen["freeze_hash"], "evaluation_key": model["evaluation_key"]})
    return result


def main(argv: list[str] | None = None) -> int:
    """Require a frozen native study for all final inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model-index", type=int, required=True)
    args = parser.parse_args(argv)
    evaluate_model(args.study.resolve(), args.model_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
