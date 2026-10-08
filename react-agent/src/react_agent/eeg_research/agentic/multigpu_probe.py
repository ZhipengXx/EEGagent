"""Synthetic engineering check using the production DataParallel retrieval step."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path


def run(request: dict) -> dict:
    """Check hooks, actual replicas, gradients and checkpoint buffers; never load EEG data."""
    import numpy as np
    import torch
    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_training.hooks import (
        apply_train_transform, is_custom_objective, objective_parameters, resolve_negative_policy, scalar_loss,
    )
    from react_agent.eeg_training.inference_consistency import persistent_buffers_digest, tensor_digest
    from react_agent.eeg_training.model import contrastive_loss
    from react_agent.eeg_training.train_entry import (
        _build_hook, _replica_device_probe, placed_retrieval, unique_trainable_parameters,
    )
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    cards = request["gpu"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(cards) or len(cards) < 2:
        raise ValueError("multigpu_probe_requires_all_frozen_visible_cards")
    module = importlib.import_module(request["module"])
    candidate = module.EEGCandidate()
    source = Path(request["source_path"])
    if file_sha256(source) != request["source_sha256"]:
        raise ValueError("multigpu_probe_source_changed_before_check")
    config, spec = request["hook_config"], request["input_geometry"]
    geom = {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}
    length = geom["timesteps"][1] - geom["timesteps"][0]
    encoder = _build_hook(candidate, "build_encoder", geom, config.get("model") or {})
    if hasattr(candidate, "fit_statistics"):
        # Only synthetic training input determines the fitted state.
        synthetic_fit = torch.randn(16, geom["c_num"], length)
        mean = synthetic_fit.mean(dim=(0, 2)); std = synthetic_fit.std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
        candidate.fit_statistics(encoder, {"mean": mean.tolist(), "std": std.tolist(),
            "values_per_channel": int(synthetic_fit.shape[0] * length), "source": "synthetic_training_only", "subject_ids": None})
    fitted_state = persistent_buffers_digest(encoder)
    device = torch.device("cuda:0")
    encoder.to(device)
    objective = _build_hook(candidate, "build_training_objective", {}, config.get("objective") or {}) if hasattr(candidate, "build_training_objective") else contrastive_loss
    if isinstance(objective, torch.nn.Module): objective.to(device)
    transform = _build_hook(candidate, "build_training_transform", {}, config.get("transform") or {}) if hasattr(candidate, "build_training_transform") else None
    if resolve_negative_policy(candidate, objective) != "data_parallel_local":
        raise ValueError("multigpu_probe_requires_frozen_local_negative_policy")
    custom = is_custom_objective(objective)
    model = placed_retrieval(encoder, device, len(cards), objective=objective if custom else None)
    optimizer = torch.optim.AdamW(unique_trainable_parameters(list(model.parameters()) + objective_parameters(objective)),
        lr=float(request["lr"]), weight_decay=float(request["weight_decay"]))
    batches, counters = [], {"transform_train_calls": 0, "transform_eval_calls": 0, "objective_train_calls": 0}
    for label, count in (("first", request["batch_size"]), ("tail", request["tail_batch_size"])):
        model.train()
        eeg = apply_train_transform(transform, torch.randn(count, geom["c_num"], length, device=device), counters)
        # Duplicate IDs share frozen synthetic features, exercising custom losses.
        codes = torch.arange(count, device=device) // 2
        targets = torch.randn(int(codes.max()) + 1, 1024, device=device)[codes]
        receipts = []
        raw = model.module
        handle = raw.register_forward_hook(_replica_device_probe(receipts))
        try:
            loss, _top1, _top5 = model(eeg, targets, codes)
        finally:
            handle.remove()
        loss = scalar_loss(loss.mean())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        torch.cuda.synchronize()
        if not receipts or sum(row["local_batch_size"] for row in receipts) != count:
            raise ValueError("multigpu_probe_replica_population_mismatch")
        devices = {row["input_device"] for row in receipts}
        if label == "first" and devices != {f"cuda:{i}" for i in range(len(cards))}:
            raise ValueError("multigpu_probe_not_all_cards_participated")
        for receipt in receipts:
            if receipt["parameter_devices"] != [receipt["input_device"]] or receipt["loss_device"] != receipt["input_device"]:
                raise ValueError("multigpu_probe_replica_parameter_device_mismatch")
        gradients = [param.grad for param in model.parameters() if param.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise ValueError("multigpu_probe_finite_gradients_required")
        batches.append({"position": label, "total_batch_size": count, "replicas": sorted(receipts, key=lambda row: row["input_device"]),
            "loss": float(loss.detach()), "gradient_tensor_count": len(gradients)})
    raw_encoder = model.module.encoder
    raw_encoder.eval()
    sample = torch.randn(8, geom["c_num"], length, device=device)
    with torch.no_grad(): expected = raw_encoder(sample)
    if list(expected.shape) != [8, 1024] or expected.dtype != torch.float32 or not bool(torch.isfinite(expected).all()):
        raise ValueError("multigpu_probe_embedding_shape_or_finiteness")
    # Actual serialization and rebuilding exercise persistent fitted/learned buffers.
    checkpoint = Path(request["probe_dir"]) / "synthetic_checkpoint.ckpt"
    torch.save({"state_dict": raw_encoder.state_dict(), "epoch": 0,
        "objective_state": objective.state_dict() if isinstance(objective, torch.nn.Module) else None}, checkpoint)
    rebuilt = _build_hook(candidate, "build_encoder", geom, config.get("model") or {})
    rebuilt.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    rebuilt.to(device).eval()
    with torch.no_grad(): actual = rebuilt(sample)
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-4, equal_nan=False):
        raise ValueError("multigpu_probe_checkpoint_embedding_mismatch")
    if persistent_buffers_digest(raw_encoder) != persistent_buffers_digest(rebuilt):
        raise ValueError("multigpu_probe_checkpoint_buffer_mismatch")
    objective_round_trip = None
    if isinstance(objective, torch.nn.Module):
        restored_objective = _build_hook(candidate, "build_training_objective", {}, config.get("objective") or {})
        if not isinstance(restored_objective, torch.nn.Module):
            raise ValueError("multigpu_probe_objective_rebuild_type_changed")
        restored_objective.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["objective_state"])
        objective_round_trip = tensor_digest(objective.state_dict()) == tensor_digest(restored_objective.state_dict())
        if not objective_round_trip: raise ValueError("multigpu_probe_objective_checkpoint_mismatch")
    if file_sha256(source) != request["source_sha256"]:
        raise ValueError("multigpu_probe_source_changed_during_check")
    return {"ok": True, "stage": "complete", "source_sha256": request["source_sha256"],
        "probe_identity": request["probe_identity"], "gpu": cards, "gpu_inventory": request["gpu_inventory"],
        "batches": batches, "custom_objective": custom, "checkpoint_round_trip": True,
        "objective_checkpoint_round_trip": objective_round_trip,
        "fitted_buffers_before_training_sha256": fitted_state,
        "checkpoint_buffers_sha256": persistent_buffers_digest(rebuilt), "checkpoint_sha256": file_sha256(checkpoint),
        "embedding_max_abs_difference": float((expected - actual).abs().max()), "eval_transform_calls": counters["transform_eval_calls"],
        "synthetic_training_inputs_only": True, "real_EEG_or_held_out_read": False,
        "python": sys.executable, "torch": torch.__version__, "file": str(source), "precision": "fp32"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        result = run(request)
    except Exception as exc:
        result = {"ok": False, "stage": "multigpu_synthetic_probe", "error": type(exc).__name__,
                  "detail": traceback.format_exc()[-3000:], "probe_identity": request["probe_identity"]}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
