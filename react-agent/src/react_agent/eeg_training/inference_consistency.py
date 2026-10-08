"""Bind a read-only reconstruction to the encoder, data and selection actually trained."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from react_agent.eeg_training.protocol import Design, SplitError, geometry, learning_rate, split_plan

IDENTITY_NAME = "training_inference_identity.json"
CONSISTENCY_NAME = "training_inference_consistency.json"
# Frozen before a run. Counts are exact; these tolerances apply only to embeddings.
EMBEDDING_TOLERANCES = {
    "torch.float64": {"atol": 1e-8, "rtol": 1e-7},
    "torch.float32": {"atol": 1e-5, "rtol": 1e-4},
    "torch.float16": {"atol": 0.002, "rtol": 0.002},
    "torch.bfloat16": {"atol": 0.02, "rtol": 0.02},
}


def file_digest(path: Path) -> str:
    """Hash file contents without materializing a trial/cache file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def tensor_digest(tensors: dict[str, Any]) -> str:
    """Include tensor name, shape, dtype and bytes; preserve fitted buffers exactly."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(object_digest({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}).encode())
        # Viewing as bytes also supports bfloat16, which numpy cannot represent.
        import torch
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def persistent_buffers_digest(encoder) -> str:
    state = encoder.state_dict()
    return tensor_digest({name: buffer for name, buffer in encoder.named_buffers() if name in state})


def validation_records_identity(records) -> dict[str, Any]:
    """Bind ordered queries, exact positives and fixed image targets, not labels alone."""
    bank: dict[str, Any] = {}
    positives = []
    query_eeg = hashlib.sha256()
    for index, row in enumerate(records):
        image = str(row["img"])
        query = str(row.get("query_id") or f"{row.get('subject', '')}::{image}")
        positives.append([query, image])
        if image not in bank:
            bank[image] = row["img_features"]
        query_eeg.update(tensor_digest({str(index): row["eeg"]}).encode())
    return {"query_count": len(positives), "candidate_count": len(bank),
            "positive_map_sha256": object_digest(positives), "gallery_image_ids_sha256": object_digest(list(bank)),
            "image_features_sha256": tensor_digest(bank), "validation_eeg_sha256": query_eeg.hexdigest()}


def training_recipe(design: Design) -> dict[str, Any]:
    """Freeze strategy and selection policy with the real argv-level training inputs."""
    return {"dataset": design.dataset, "exp_setting": design.exp_setting, "subject": design.subject,
            "seed": design.seed, "batch_size": design.batch_size, "lr": learning_rate(design),
            "weight_decay": design.weight_decay, "epochs": design.epochs, "stop": design.stop,
            "gpu": list(design.gpu), "training_strategy": design.training_strategy,
            "evaluation_mode": design.evaluation_mode, "checkpoint_policy": design.checkpoint_policy,
            "selection_min_delta": design.selection_min_delta,
            "negative_sampling_policy": "data_parallel_local", "input_geometry": geometry(design.dataset)}


def validate_frozen_loader_identity(records, frozen: dict[str, Any] | None) -> None:
    """Require the actual loader's order/positives/gallery to implement the frozen protocol."""
    if not frozen:
        raise SplitError("method_search_frozen_evaluation_identity_missing")
    actual_queries, actual_gallery = [], []
    seen = set()
    for row in records:
        image = str(row["img"])
        query = str(row.get("query_id") or f"{row.get('subject', '')}::{image}")
        actual_queries.append(query)
        if (frozen.get("positive_map") or {}).get(query) != image:
            raise SplitError("frozen_loader_positive_map_mismatch")
        if image not in seen:
            seen.add(image)
            actual_gallery.append(image)
    if actual_queries != frozen.get("validation_query_ids") or actual_gallery != frozen.get("gallery_image_ids"):
        raise SplitError("frozen_loader_query_or_gallery_mismatch")


def _optional_file(path: Path) -> str | None:
    return file_digest(path) if path.is_file() else None


def _source_and_context(context: Path, checkpoint: Path) -> dict[str, Any]:
    binding_file = context / "source_binding.json"
    if not binding_file.is_file():
        raise SplitError("reconstruction_source_binding_missing")
    binding = json.loads(binding_file.read_text(encoding="utf-8"))
    source = Path(str(binding.get("class_file") or ""))
    if not source.is_absolute():
        source = context / source
    if not source.is_file() or file_digest(source) != binding.get("file_sha256"):
        raise SplitError("reconstruction_source_hash_mismatch")
    framework = {}
    for name in ("train_entry.py", "fixed_bank.py", "model.py", "data.py", "protocol.py", "hooks.py", "inference_consistency.py", "checkpoint_selection.py"):
        path = Path(__file__).parent / name
        framework[name] = file_digest(path)
    return {"source_binding_sha256": file_digest(binding_file), "candidate_source_sha256": file_digest(source),
            "framework_sha256": framework, "hook_config_sha256": _optional_file(context / "hook_config.json"),
            "train_statistics_sha256": _optional_file(context / "train_statistics.json"),
            "evaluation_identity_sha256": _optional_file(context / "evaluation_identity.json"),
            "checkpoint_sha256": file_digest(checkpoint)}


def freeze_training_identity(design: Design, data_root: Path, context: Path, checkpoint: Path, encoder, records) -> dict[str, Any]:
    """Write once from training; later scorers only read and verify this artifact."""
    plan = split_plan(data_root, design)
    paths = [*plan.train_files, *plan.val_files, *plan.feature_caches]
    body = {"schema_version": "eeg_research.training_inference_identity.v1",
            "context": _source_and_context(context, checkpoint), "recipe": training_recipe(design),
            "data_files": [{"path": str(path), "sha256": file_digest(path)} for path in dict.fromkeys(paths)],
            "development": validation_records_identity(records),
            "persistent_buffers_sha256": persistent_buffers_digest(encoder),
            "encoder_dtype": str(next(encoder.parameters()).dtype),
            "eval_mode": True, "augmentation": "not_applied",
            "evaluator": "react_agent.eeg_training.fixed_bank.FixedBankTally",
            "tie_policy": "torch_topk_on_same_ordered_frozen_gallery_same_backend",
            "embedding_tolerances": EMBEDDING_TOLERANCES,
            "restoration_scope": "encoder_state_dict_includes_persistent_fit_buffers;objective_and_transform_are_training_only"}
    body["fingerprint"] = object_digest(body)
    target = context / IDENTITY_NAME
    if target.exists():
        raise SplitError("training_inference_identity_already_frozen")
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return body


def validate_reconstruction_context(context: Path, checkpoint: Path) -> dict[str, Any] | None:
    """Reject replaced source/config/statistics/checkpoint before importing the candidate."""
    path = context / IDENTITY_NAME
    if not path.is_file():
        return None
    identity = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in identity.items() if key != "fingerprint"}
    if identity.get("fingerprint") != object_digest(body):
        raise SplitError("training_inference_identity_digest_mismatch")
    if identity.get("context") != _source_and_context(context, checkpoint):
        raise SplitError("training_inference_context_mismatch")
    return identity


def validate_development_identity(identity: dict[str, Any], design: Design, records) -> None:
    """Fail on a changed recipe, input EEG, image features, query order or positive map."""
    if identity.get("recipe") != training_recipe(design):
        raise SplitError("reconstruction_recipe_mismatch")
    if identity.get("development") != validation_records_identity(records):
        raise SplitError("reconstruction_development_identity_mismatch")
    for row in identity.get("data_files") or []:
        path = Path(row["path"])
        if not path.is_file() or file_digest(path) != row.get("sha256"):
            raise SplitError("reconstruction_training_data_mismatch")


def compare_reconstruction(expected_scores: dict[str, Any], actual_scores: dict[str, Any], expected_embedding, actual_embedding) -> dict[str, Any]:
    """Verify fixed-gallery integer hits and a bounded preselected embedding probe."""
    import torch
    keys = ("query_count", "candidate_count", "fixed_bank_top1", "fixed_bank_top5")
    if any(not math.isfinite(float(actual_scores[key])) for key in keys):
        raise SplitError("reconstruction_score_non_finite")
    if any(float(expected_scores[key]) != float(actual_scores[key]) for key in keys):
        raise SplitError("reconstruction_fixed_gallery_hits_mismatch")
    dtype = str(expected_embedding.dtype)
    tolerance = EMBEDDING_TOLERANCES.get(dtype)
    if tolerance is None or expected_embedding.shape != actual_embedding.shape or not torch.allclose(
        expected_embedding, actual_embedding.to(expected_embedding.device), equal_nan=False, **tolerance
    ):
        raise SplitError("reconstruction_embedding_mismatch")
    n = int(actual_scores["query_count"])
    return {"status": "verified", "development_query_count": n,
            "candidate_count": int(actual_scores["candidate_count"]),
            "top1_hits": int(round(float(actual_scores["fixed_bank_top1"]) * n)),
            "top5_hits": int(round(float(actual_scores["fixed_bank_top5"]) * n)),
            "embedding_probe_count": int(expected_embedding.shape[0]), "embedding_dtype": dtype,
            "embedding_tolerances": tolerance, "embedding_max_abs_difference": float((expected_embedding - actual_embedding).abs().max()),
            "scope": "same_selected_checkpoint_and_legal_development_data", "held_out_scored": False}
