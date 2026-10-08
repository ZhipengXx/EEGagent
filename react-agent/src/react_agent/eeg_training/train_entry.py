"""Baseline retrieval. The test split is never passed to the fit loop."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from react_agent.eeg_training.protocol import (
    Design,
    SplitError,
    feature_cache,
    geometry,
    learning_rate,
    parse_gpu_list,
    holdout_image_ids,
    split_plan,
    validate_design,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.eeg_training.train_entry")
    parser.add_argument("--dataset", choices=["eeg", "meg"], required=True)
    parser.add_argument("--exp-setting", choices=["intra-subject", "inter-subject"], required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop", default="chain_early")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path, separate from the output directory")
    parser.add_argument("--negative-policy", default="data_parallel_local")
    parser.add_argument("--evaluation-mode", default="single_target", choices=["single_target", "loso_method_search"])
    parser.add_argument("--checkpoint-policy", default="legacy_min_delta", choices=["legacy_min_delta", "strict_best"])
    parser.add_argument("--selection-min-delta", type=float, default=0.001)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Training device. Production default remains CUDA.")
    return parser


def resolve_train_device(requested: str | None = None):
    """Public device resolver. Production stays on CUDA; tests may request CPU."""
    import torch

    choice = (requested or os.environ.get("EEG_TRAIN_DEVICE") or "").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise SplitError("cuda_unavailable")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if os.environ.get("EEG_ALLOW_CPU_TRAIN") == "1":
        return torch.device("cpu")
    raise SplitError("cuda_unavailable")


def _image_ids(files: tuple[Path, ...], channels: list[str] | None) -> list[str]:
    from react_agent.eeg_training.data import load_trials

    found: list[str] = []
    for path in files:
        trials = load_trials(path, channels)
        images = trials["img"]
        assert isinstance(images, list)
        found.extend(images)
    return found


def _frozen_identity(out_dir: Path | None = None) -> dict | None:
    env = os.environ.get("EEG_EVALUATION_IDENTITY")
    path = Path(env) if env else None
    if path is None and out_dir is not None:
        candidate = out_dir / "evaluation_identity.json"
        path = candidate if candidate.is_file() else None
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def build_loaders(design: Design, data_root: Path, identity: dict | None = None):
    """Create train and validation loaders. Test files are not opened as a loader."""
    import torch
    from torch.utils.data import DataLoader

    from react_agent.eeg_training.data import RetrievalTrials, collate_retrieval, collect_records, collect_validation_records, load_feature_cache

    plan = split_plan(data_root, design)
    forbidden_paths = set(plan.forbidden_files)
    if forbidden_paths & set(plan.train_files) or forbidden_paths & set(plan.val_files):
        raise SplitError("held_out_test_in_fit")
    spec = geometry(design.dataset)
    channels = spec["channels"]
    assert channels is None or isinstance(channels, list)
    timesteps = spec["timesteps"]
    assert isinstance(timesteps, list)
    identity = identity or _frozen_identity()
    if not plan.feature_caches:
        raise SplitError("validation_feature_cache_missing")
    if plan.val_mode == "other_subjects_test":
        train_features = load_feature_cache(plan.feature_caches[0])
        train_records, train_images = collect_records(plan.train_files, train_features, channels, None)
        val_records, val_images = collect_validation_records(plan, channels)
    else:
        features = load_feature_cache(plan.feature_caches[0])
        frozen_train = list((identity or {}).get("train_image_ids") or [])
        frozen_val = list((identity or {}).get("validation_image_ids") or [])
        if frozen_train and frozen_val:
            kept, validation = frozen_train, frozen_val
        else:
            train_ids = sorted(set(_image_ids(plan.train_files, channels)))
            test_ids = sorted(set(_image_ids(plan.forbidden_files, channels)))
            kept, validation = holdout_image_ids(train_ids, test_ids, design.seed)
        train_records, train_images = collect_records(plan.train_files, features, channels, set(kept))
        val_records, val_images = collect_validation_records(plan, channels, validation_image_ids=set(validation), features=features)
    if set(train_images) & set(val_images):
        raise SplitError("train_validation_overlap")
    train_set = RetrievalTrials(train_records, timesteps)
    val_set = RetrievalTrials(val_records, timesteps)
    train_loader = DataLoader(
        train_set,
        batch_size=min(design.batch_size, len(train_set)),
        shuffle=True,
        collate_fn=collate_retrieval,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(200, len(val_set)),
        shuffle=False,
        collate_fn=collate_retrieval,
    )
    return train_loader, val_loader, train_images, val_images, spec


def limit_visible_gpus(design: Design) -> None:
    """Restrict the process to the checked nvidia-smi indexes before torch is imported."""
    if not design.gpu:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in design.gpu)


def write_status(out_dir: Path, phase: str, epoch: int, epochs: int) -> None:
    """Record the current training phase. This file is not a score."""
    payload = {"phase": phase, "epoch": epoch, "epochs": epochs}
    (out_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def begin_history(out_dir: Path, epochs: int) -> None:
    """Start a run with an empty curve. Earlier points from a stopped run are removed."""
    history = out_dir / "history.jsonl"
    if history.exists():
        history.unlink()
    write_status(out_dir, "loading", 0, epochs)


def append_history(
    out_dir: Path,
    epoch: int,
    train_loss: float,
    val_top1: float,
    val_top5: float,
    *,
    fixed_bank_top1: float | None = None,
    fixed_bank_top5: float | None = None,
) -> None:
    """Append one finished epoch. val_top1 is within-batch; fixed_bank_top1 ranks the whole bank."""
    row: dict[str, float | int] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_top1": val_top1,
        "val_top5": val_top5,
    }
    if fixed_bank_top1 is not None and fixed_bank_top5 is not None:
        row["fixed_bank_top1"] = fixed_bank_top1
        row["fixed_bank_top5"] = fixed_bank_top5
    with (out_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def latest_train_status(root: Path) -> dict[str, object] | None:
    """Return the newest campaign curve. Opening the page does not start training."""
    if not root.is_dir():
        return None
    found = [
        path for path in root.iterdir()
        if path.is_dir() and ((path / "history.jsonl").is_file() or (path / "status.json").is_file())
    ]
    if not found:
        return None
    newest = max(found, key=lambda path: path.stat().st_mtime)
    return read_train_status(root, newest.name)


def read_train_status(root: Path, campaign: str) -> dict[str, object] | None:
    """Read one campaign's live curve. A missing directory is not a score."""
    if not campaign or campaign != Path(campaign).name or not campaign.replace("_", "").replace("-", "").isalnum():
        return None
    out = (root / campaign).resolve()
    if root.resolve() not in out.parents:
        return None
    if not out.is_dir():
        return None
    trial_curve = _trial_curve(out)
    curve = trial_curve or out
    status_path = curve / "status.json"
    status: dict[str, object] = {"phase": "idle", "epoch": 0, "epochs": 0}
    if status_path.is_file():
        loaded = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            status = {
                "phase": loaded.get("phase") or "idle",
                "epoch": int(loaded.get("epoch") or 0),
                "epochs": int(loaded.get("epochs") or 0),
            }
    history: list[dict[str, object]] = []
    history_path = curve / "history.jsonl"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if not {"epoch", "train_loss", "val_top1", "val_top5"} <= set(row):
                continue
            point: dict[str, object] = {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_top1": float(row["val_top1"]),
                "val_top5": float(row["val_top5"]),
            }
            if "fixed_bank_top1" in row:
                point["fixed_bank_top1"] = float(row["fixed_bank_top1"])
                point["fixed_bank_top5"] = float(row["fixed_bank_top5"])
            history.append(point)
    status["history"] = history
    from react_agent.eeg_research.trace import read_trace

    status["trials"] = [row.page_view() for row in read_trace(out)]
    status["test_result"] = _stored_test_result(curve) or _stored_test_result(out)
    research_path = out / "research_state.json"
    if research_path.is_file():
        loaded_research = json.loads(research_path.read_text(encoding="utf-8"))
        if isinstance(loaded_research, dict):
            research = {
                "phase": loaded_research.get("phase"),
                "status": loaded_research.get("status"),
                "action": loaded_research.get("action"),
                "action_label": loaded_research.get("action_label"),
                "reason": loaded_research.get("reason"),
                "detail": loaded_research.get("detail"),
                "raw": loaded_research.get("raw"),
                "legacy_fixed": loaded_research.get("legacy_fixed"),
            }
            status["research"] = research
            status["prior_curve"] = trial_curve is None and bool(history)
            if research.get("phase") == "planning" and status.get("phase") in {"idle", "planning"}:
                status["phase"] = "planning"
    chain_path = out / "chain.json"
    if chain_path.is_file():
        loaded_chain = json.loads(chain_path.read_text(encoding="utf-8"))
        if isinstance(loaded_chain, dict):
            status["chain"] = loaded_chain
    return status


def _trial_curve(campaign: Path) -> Path | None:
    """Return the highest-numbered trial curve written by this session."""
    root = campaign / "trials"
    if not root.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("t"):
            continue
        suffix = path.name[1:]
        if not suffix.isdigit():
            continue
        if (path / "status.json").is_file() or (path / "history.jsonl").is_file():
            found.append((int(suffix), path))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def _stored_test_result(out: Path) -> dict[str, float] | None:
    """Read a finished test score. Absence stays null."""
    metrics = out / "metrics.json"
    if not metrics.is_file():
        return None
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    result = payload.get("test_result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return None
    if "top1" not in result or "top5" not in result:
        return None
    return {"top1": float(result["top1"]), "top5": float(result["top5"])}


def score_held_out(design: Design, data_root: Path, out_dir: Path) -> dict[str, object] | None:
    """Score the held-out test files after fit. A missing cache is not a score."""
    plan = split_plan(data_root, design)
    cache = feature_cache(data_root, design, "test")
    checkpoint = out_dir / "last.ckpt"
    if not checkpoint.is_file() or not cache.is_file():
        return None
    if not plan.forbidden_files or any(not path.is_file() for path in plan.forbidden_files):
        return None
    from torch.utils.data import DataLoader

    from react_agent.eeg_training.data import RetrievalTrials, collate_retrieval, collect_records, load_feature_cache

    spec = geometry(design.dataset)
    channels = spec["channels"]
    assert channels is None or isinstance(channels, list)
    timesteps = spec["timesteps"]
    assert isinstance(timesteps, list)
    try:
        features = load_feature_cache(cache)
        records, _images = collect_records(plan.forbidden_files, features, channels, None)
    except SplitError:
        return None
    if not records:
        return None
    identity = None
    if design.evaluation_mode == "loso_method_search":
        from react_agent.eeg_training.inference_consistency import validate_reconstruction_context
        identity = validate_reconstruction_context(out_dir, checkpoint)
        if identity is None:
            raise SplitError("training_inference_identity_missing")
    dataset = RetrievalTrials(records, timesteps)
    loader = DataLoader(dataset, batch_size=min(200, len(dataset)), shuffle=False, collate_fn=collate_retrieval)
    encoder = _load_encoder(spec, checkpoint, out_dir)
    result = _fixed_bank_pass(encoder, loader, records)
    scored = {
        "top1": result["fixed_bank_top1"],
        "top5": result["fixed_bank_top5"],
        "metric": "fixed_bank",
        "query_count": result["query_count"],
        "candidate_count": result["candidate_count"],
    }
    if identity is not None:
        from react_agent.eeg_training.inference_consistency import file_digest, object_digest, validation_records_identity
        inference = {"schema_version": "eeg_research.held_out_inference_identity.v1",
                     "training_inference_identity_fingerprint": identity["fingerprint"],
                     "checkpoint_sha256": identity["context"]["checkpoint_sha256"],
                     "source_sha256": identity["context"]["candidate_source_sha256"],
                     "held_out_files": [{"path": str(path), "sha256": file_digest(path)} for path in plan.forbidden_files],
                     "test_image_cache": {"path": str(cache), "sha256": file_digest(cache)},
                     "actual_inputs": validation_records_identity(records),
                     "input_geometry": spec, "eval_mode": True, "augmentation": "not_applied",
                     "evaluator": identity["evaluator"], "tie_policy": identity["tie_policy"],
                     "device": str(next(encoder.parameters()).device), "dtype": str(next(encoder.parameters()).dtype)}
        scored["inference_identity"] = inference
        scored["inference_fingerprint"] = object_digest(inference)
    return scored


def rebuild_encoder(spec: dict[str, object], checkpoint: Path, out_dir: Path):
    """Rebuild the trained encoder in a new object. Candidate modules never fall back to EEGProjectLayer."""
    import importlib

    import torch
    from react_agent.eeg_training.inference_consistency import persistent_buffers_digest, validate_reconstruction_context, file_digest

    identity = validate_reconstruction_context(out_dir, checkpoint)

    binding_path = out_dir / "source_binding.json"
    binding = {}
    packed = False
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        packed = True
    module_name = str(binding.get("module") or "")
    if not packed:
        module_name = module_name or os.environ.get("EEG_CANDIDATE_MODULE", "")
        candidate_path = os.environ.get("EEG_CANDIDATE_PATH", "")
        if candidate_path and candidate_path not in sys.path:
            sys.path.insert(0, candidate_path)
    class_file = binding.get("class_file")
    loaded_from_file = None
    if class_file:
        class_path = Path(str(class_file))
        if not class_path.is_absolute():
            class_path = (Path(out_dir) / class_file).resolve()
        if not class_path.is_file():
            raise SplitError("candidate_source_missing")
        if binding.get("file_sha256") and file_digest(class_path) != binding["file_sha256"]:
            raise SplitError("candidate_source_hash_mismatch")
        parent = str(class_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        import importlib.util

        spec_loader = importlib.util.spec_from_file_location("eeg_candidate_pack", class_path)
        if spec_loader is None or spec_loader.loader is None:
            raise SplitError("candidate_reload_failed:spec")
        loaded_from_file = importlib.util.module_from_spec(spec_loader)
        spec_loader.loader.exec_module(loaded_from_file)
    baseline = module_name in {
        "",
        "react_agent.eeg_research.agentic.baseline",
        "react_agent.eeg_training.model",
    }
    geometry = {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}
    model_config = _hook_section(out_dir, "model")
    if packed and loaded_from_file is None and not baseline:
        raise SplitError("candidate_source_missing")
    if loaded_from_file is not None:
        try:
            encoder = _build_hook(loaded_from_file.EEGCandidate(), "build_encoder", geometry, model_config)
        except Exception as exc:  # noqa: BLE001
            raise SplitError(f"candidate_reload_failed:{type(exc).__name__}") from exc
    elif baseline:
        from react_agent.eeg_research.agentic.baseline import EEGCandidate

        encoder = _build_hook(EEGCandidate(), "build_encoder", geometry, model_config)
    else:
        try:
            module = importlib.import_module(module_name)
            candidate = module.EEGCandidate()
            encoder = _build_hook(candidate, "build_encoder", geometry, model_config)
        except Exception as exc:  # noqa: BLE001
            raise SplitError(f"candidate_reload_failed:{type(exc).__name__}") from exc
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["state_dict"])
    if identity is not None and persistent_buffers_digest(encoder) != identity.get("persistent_buffers_sha256"):
        raise SplitError("reconstruction_fitted_buffer_mismatch")
    encoder.eval()
    device = resolve_train_device()
    return encoder.to(device)


def _load_encoder(spec: dict[str, object], checkpoint: Path, out_dir: Path | None = None):
    directory = out_dir if out_dir is not None else checkpoint.parent
    return rebuild_encoder(spec, checkpoint, directory)


def _fixed_bank_pass(encoder, loader, records) -> dict[str, float]:
    """Rank each batch's queries against every image in this split."""
    import torch

    from react_agent.eeg_training.fixed_bank import FixedBankTally, frozen_bank

    device = next(encoder.parameters()).device
    bank, labels = frozen_bank(records)
    if not torch.isfinite(bank).all():
        raise SplitError("non_finite_evaluation_image_features")
    tally = FixedBankTally(bank.to(device), labels.to(device))
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            embedding = encoder(batch["eeg"].to(device))
            if not torch.isfinite(embedding).all():
                raise SplitError("non_finite_evaluation_embedding")
            tally.add(embedding)
    return tally.result()


def evaluate_checkpoint(design: Design, data_root: Path, out_dir: Path, checkpoint: Path | None = None, context_dir: Path | None = None) -> dict[str, object]:
    """Score an existing checkpoint on frozen validation. Final test is not computed."""
    limit_visible_gpus(design)
    import torch

    resolve_train_device()
    context = Path(context_dir) if context_dir is not None else (Path(checkpoint).parent if checkpoint is not None else out_dir)
    checkpoint = Path(checkpoint) if checkpoint is not None else context / "last.ckpt"
    if not checkpoint.is_file():
        raise SplitError("checkpoint_missing")
    _train_loader, val_loader, _train_images, _val_images, spec = build_loaders(
        design, data_root, identity=_frozen_identity(context)
    )
    from react_agent.eeg_training.inference_consistency import validate_development_identity, validate_reconstruction_context, validate_frozen_loader_identity
    identity = validate_reconstruction_context(context, checkpoint)
    if design.evaluation_mode == "loso_method_search" and identity is None:
        raise SplitError("training_inference_identity_missing")
    if design.evaluation_mode == "loso_method_search":
        validate_frozen_loader_identity(val_loader.dataset.records, _frozen_identity(context))
    if identity is not None:
        validate_development_identity(identity, design, val_loader.dataset.records)
    encoder = _load_encoder(spec, checkpoint, context)
    validation = _fixed_bank_pass(encoder, val_loader, val_loader.dataset.records)
    return {
        "fixed_bank_top1": validation["fixed_bank_top1"],
        "fixed_bank_top5": validation["fixed_bank_top5"],
        "query_count": validation["query_count"],
        "candidate_count": validation["candidate_count"],
        "test_result": None,
        "evaluate_only": True,
    }


def train_channel_statistics(train_loader) -> dict[str, object]:
    """Per-channel mean and std over every training sample and time point. Validation is not read."""
    import torch

    total = None
    squares = None
    count = 0
    for batch in train_loader:
        eeg = batch["eeg"].to(torch.float64)
        summed = eeg.sum(dim=(0, 2))
        squared = eeg.pow(2).sum(dim=(0, 2))
        total = summed if total is None else total + summed
        squares = squared if squares is None else squares + squared
        count += eeg.shape[0] * eeg.shape[2]
    if total is None or count == 0:
        raise SplitError("empty_train_loader")
    mean = total / count
    std = (squares / count - mean.pow(2)).clamp_min(1e-12).sqrt()
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "values_per_channel": int(count),
        "source": "train_files",
        "subject_ids": None,
    }


def _hook_section(out_dir: Path, name: str) -> dict:
    path = out_dir / "hook_config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    section = payload.get(name) if isinstance(payload, dict) else None
    return section if isinstance(section, dict) else {}


def _build_hook(candidate, method: str, geometry: dict, config: dict):
    """Call a hook with geometry plus approved config. Unknown config is an error."""
    from react_agent.eeg_training.hooks import HookConfigError

    builder = getattr(candidate, method)
    accepted = getattr(builder, "accepted_config_keys", None)
    if accepted is None:
        accepted = getattr(getattr(builder, "__func__", None), "accepted_config_keys", None)
    payload = dict(config or {})
    if accepted is not None:
        unknown = sorted(set(payload) - set(accepted))
        if unknown:
            raise HookConfigError(f"unknown_{method}_config:{','.join(unknown)}")
    if method == "build_encoder":
        try:
            built = builder(geometry, payload or None)
        except TypeError:
            if payload:
                raise HookConfigError("encoder_rejected_config") from None
            built = builder(geometry)
        return built
    try:
        return builder(payload or None)
    except TypeError:
        if payload:
            raise HookConfigError(f"{method}_rejected_config") from None
        return builder()


def _note_hook_consumed(out_dir: Path, name: str, config: dict) -> None:
    """Record that a hook actually received its approved section. File presence is not enough."""
    path = out_dir / "hook_consumed.json"
    payload: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            payload = loaded
    payload[name] = {"applied": True, "keys": sorted(config), "config": dict(config)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _instantiate_candidate(spec: dict[str, object], out_dir: Path, train_loader=None):
    """Build the candidate plugin and encoder. Missing hooks stay None, not invented."""
    module_name = os.environ.get("EEG_CANDIDATE_MODULE", "")
    candidate_path = os.environ.get("EEG_CANDIDATE_PATH", "")
    if candidate_path and candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)
    used_fit = False
    from react_agent.eeg_research.agentic.binding import write_binding

    if not module_name:
        from react_agent.eeg_research.agentic.baseline import EEGCandidate

        candidate = EEGCandidate()
        model_config = _hook_section(out_dir, "model")
        encoder = _build_hook(candidate, "build_encoder", {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}, model_config)
        _note_hook_consumed(out_dir, "model", model_config)
        write_binding(out_dir, candidate)
        return encoder, candidate, "react_agent.eeg_research.agentic.baseline", used_fit
    import importlib

    module = importlib.import_module(module_name)
    candidate = module.EEGCandidate()
    model_config = _hook_section(out_dir, "model")
    encoder = _build_hook(candidate, "build_encoder", {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}, model_config)
    _note_hook_consumed(out_dir, "model", model_config)
    if hasattr(candidate, "fit_statistics") and train_loader is not None:
        stats = train_channel_statistics(train_loader)
        candidate.fit_statistics(encoder, stats)
        (out_dir / "train_statistics.json").write_text(json.dumps(stats), encoding="utf-8")
        used_fit = True
    write_binding(out_dir, candidate)
    return encoder, candidate, module_name, used_fit


def _build_encoder(spec: dict[str, object], out_dir: Path, train_loader=None):
    """Build the baseline, or the candidate module named by the job environment."""
    encoder, _candidate, _module, _used = _instantiate_candidate(spec, out_dir, train_loader)
    return encoder


def _logit_scale(encoder, device):
    import torch
    from torch.nn import functional as F

    raw = encoder.module if isinstance(encoder, torch.nn.DataParallel) else encoder
    if hasattr(raw, "logit_scale") and hasattr(raw, "softplus"):
        return raw.softplus(raw.logit_scale)
    if hasattr(raw, "logit_scale"):
        return F.softplus(raw.logit_scale)
    return torch.ones([], device=device)


def _batch_image_ids(batch) -> list[str]:
    ids = batch.get("image_id") if isinstance(batch, dict) else None
    if ids is None:
        return []
    if isinstance(ids, (list, tuple)):
        return [str(item) for item in ids]
    return []


def _query_ids(batch) -> list[str]:
    raw = batch.get("query_id") if isinstance(batch, dict) else None
    if isinstance(raw, (list, tuple)) and raw and all(str(item) for item in raw):
        return [str(item) for item in raw]
    subjects = batch.get("subject") if isinstance(batch, dict) else None
    images = _batch_image_ids(batch)
    if isinstance(subjects, (list, tuple)) and subjects:
        return [f"{subjects[index]}::{images[index] if index < len(images) else index}" for index in range(len(subjects))]
    return images


def _query_rows(batch, embedding) -> list[tuple[str, list[float]]]:
    ids = _query_ids(batch)
    vectors = embedding.detach().cpu().tolist()
    if vectors and not isinstance(vectors[0], list):
        vectors = [vectors]
    rows: list[tuple[str, list[float]]] = []
    for index, vector in enumerate(vectors):
        query_id = ids[index] if index < len(ids) else f"q{index}"
        rows.append((query_id, [float(value) for value in vector]))
    return rows


class BoundedQueryReservoir:
    """Keep at most `limit` query embeddings. Does not materialize the full validation set."""

    def __init__(self, limit: int, seed: int) -> None:
        self.limit = max(1, int(limit))
        self.rng = random.Random(int(seed))
        self.items: list[tuple[str, list[float]]] = []
        self.seen = 0

    def add(self, batch, embedding) -> None:
        ids = _query_ids(batch)
        count = int(embedding.shape[0])
        for index in range(count):
            self.seen += 1
            query_id = ids[index] if index < len(ids) else f"q{self.seen}"
            vector = [float(value) for value in embedding[index].detach().cpu().tolist()]
            if len(self.items) < self.limit:
                self.items.append((query_id, vector))
                continue
            replace_at = self.rng.randrange(self.seen)
            if replace_at < self.limit:
                self.items[replace_at] = (query_id, vector)

    def rows(self) -> list[tuple[str, list[float]]]:
        return list(self.items)


def unique_trainable_parameters(params):
    """Deduplicate learnable tensors by object identity. Custom objectives must not be counted twice."""
    seen: set[int] = set()
    unique = []
    for param in params:
        if not getattr(param, "requires_grad", False):
            continue
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        unique.append(param)
    return unique


def placed_retrieval(encoder, device, n_visible: int, objective=None):
    """Move wrapper parameters with the encoder, then replicate.

    LocalRetrieval may allocate logit_scale on CPU when the encoder has none.
    DataParallel requires every parameter to already sit on cuda:0.
    """
    import torch

    from react_agent.eeg_training.model import LocalRetrieval

    model = LocalRetrieval(encoder, objective=objective).to(device)
    if n_visible > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_visible)))
    return model


def _replica_device_probe(receipts):
    """Observe actual replica inputs and losses without changing their values."""
    def observe(module, inputs, output):
        loss = output[0]
        # DataParallel replicas expose copied tensors as _former_parameters
        # rather than through parameters(); inspect both native registries.
        parameters = [value for child in module.modules()
                      for value in {**child._parameters, **getattr(child, "_former_parameters", {})}.values()
                      if value is not None]
        scale_owner = module.encoder if hasattr(module, "encoder") and hasattr(module.encoder, "logit_scale") else module
        scale = _logit_scale(scale_owner, inputs[0].device)
        receipts.append({"input_device": str(inputs[0].device), "local_batch_size": int(inputs[0].shape[0]),
                         "parameter_devices": sorted({str(value.device) for value in parameters}),
                         "loss_device": str(loss.device), "logit_scale": float(scale.detach()),
                         "temperature": 1.0 / float(scale.detach()) if float(scale.detach()) > 0 else None})
    return observe


def fit(design: Design, data_root: Path, out_dir: Path) -> dict[str, object]:
    """Train on validation top-1 and write metrics only after a completed loop."""
    import hashlib
    from react_agent.eeg_training.checkpoint_selection import CheckpointSelection

    validate_design(design)

    limit_visible_gpus(design)
    begin_history(out_dir, design.epochs)
    import torch

    from react_agent.eeg_training.fixed_bank import FixedBankTally, frozen_bank
    from react_agent.eeg_training.hooks import (
        apply_train_transform,
        capabilities_used_payload,
        compute_objective,
        is_custom_objective,
        note_eval_without_transform,
        objective_parameters,
        resolve_negative_policy,
        scalar_loss,
    )
    from react_agent.eeg_training.model import contrastive_loss, within_batch_accuracy

    device = resolve_train_device()
    random.seed(design.seed)
    if design.evaluation_mode == "loso_method_search":
        import numpy as np
        np.random.seed(design.seed)
    torch.manual_seed(design.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(design.seed)
    train_loader, val_loader, train_images, val_images, spec = build_loaders(
        design, data_root, identity=_frozen_identity(out_dir)
    )
    if design.evaluation_mode == "loso_method_search":
        from react_agent.eeg_training.inference_consistency import validate_frozen_loader_identity
        validate_frozen_loader_identity(val_loader.dataset.records, _frozen_identity(out_dir))
    encoder, candidate, module_name, used_fit = _instantiate_candidate(spec, out_dir, train_loader)
    encoder = encoder.to(device)
    transform_config = _hook_section(out_dir, "transform")
    objective_config = _hook_section(out_dir, "objective")
    transform = _build_hook(candidate, "build_training_transform", {}, transform_config) if hasattr(candidate, "build_training_transform") else None
    _note_hook_consumed(out_dir, "transform", transform_config)
    objective = _build_hook(candidate, "build_training_objective", {}, objective_config) if hasattr(candidate, "build_training_objective") else contrastive_loss
    _note_hook_consumed(out_dir, "objective", objective_config)
    if isinstance(objective, torch.nn.Module):
        objective = objective.to(device)
    counters: dict[str, int] = {
        "transform_train_calls": 0,
        "transform_eval_calls": 0,
        "objective_train_calls": 0,
    }
    try:
        objective._call_counts = counters  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    custom = is_custom_objective(objective)
    negative_policy = resolve_negative_policy(candidate, objective)
    if design.evaluation_mode == "loso_method_search" and negative_policy != "data_parallel_local":
        raise SplitError("method_search_requires_data_parallel_local_negatives")
    use_global = negative_policy == "global_batch"
    n_visible = len(design.gpu) if device.type == "cuda" else 1
    extra = objective_parameters(objective)
    model: torch.nn.Module | None = None
    if use_global:
        wrapped = encoder
        if n_visible > 1:
            wrapped = torch.nn.DataParallel(encoder, device_ids=list(range(n_visible)))
        params = unique_trainable_parameters(list(encoder.parameters()) + extra)
        optimizer = torch.optim.AdamW(params, lr=learning_rate(design), weight_decay=design.weight_decay)
    else:
        model = placed_retrieval(encoder, device, n_visible, objective=objective if custom else None)
        optimizer = torch.optim.AdamW(
            unique_trainable_parameters(list(model.parameters()) + extra),
            lr=learning_rate(design),
            weight_decay=design.weight_decay,
        )
        wrapped = None
    bank, bank_labels = frozen_bank(val_loader.dataset.records)
    if not torch.isfinite(bank).all():
        raise SplitError("non_finite_development_image_features")
    bank = bank.to(device)
    bank_labels = bank_labels.to(device)
    best = None
    selection = CheckpointSelection(design.checkpoint_policy, design.selection_min_delta)
    best_top5 = None
    best_within = None
    finished = 0
    best_epoch = None
    last_fixed = None
    selected_scores = None
    selected_probe = None
    method_search = design.evaluation_mode == "loso_method_search"
    early_stop_enabled = design.stop in {"single_early", "chain_early"}
    duplicate_batches: list[list[str]] = []
    device_receipts = []
    write_status(out_dir, "training", 0, design.epochs)
    for epoch_index in range(design.epochs):
        if use_global:
            encoder.train()
            if isinstance(objective, torch.nn.Module):
                objective.train()
        else:
            assert model is not None
            model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            eeg = apply_train_transform(transform, batch["eeg"].to(device), counters)
            img = batch["img_features"].to(device)
            ids = _batch_image_ids(batch)
            if ids and len(duplicate_batches) < 64:
                duplicate_batches.append(ids)
            if use_global:
                assert wrapped is not None
                eeg_z = wrapped(eeg)
                scale = _logit_scale(encoder, device)
                loss = scalar_loss(compute_objective(objective, eeg_z, img, scale, ids))
            else:
                assert model is not None
                probe, probe_handle = [], None
                if epoch_index == 0 and batch_index in {0, len(train_loader) - 1}:
                    underlying = model.module if isinstance(model, torch.nn.DataParallel) else model
                    probe_handle = underlying.register_forward_hook(_replica_device_probe(probe))
                if ids:
                    codes = torch.tensor([int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:8], 16) % (2**31) for item in ids], device=device, dtype=torch.long)
                    loss, _top1, _top5 = model(eeg, img, codes)
                else:
                    loss, _top1, _top5 = model(eeg, img)
                if probe_handle is not None:
                    probe_handle.remove()
                # LocalRetrieval checked each scalar objective; gather adds one
                # loss per replica while preserving the local negative batches.
                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()
                loss = scalar_loss(loss)
            losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not use_global and probe_handle is not None:
                device_receipts.append({"epoch": 1, "batch_index": batch_index,
                    "batch_position": "first" if batch_index == 0 else "tail", "global_batch_size": int(eeg.shape[0]),
                    "replicas": sorted(probe, key=lambda row: row["input_device"]),
                    "gathered_loss_device": str(loss.device),
                    "optimizer_parameter_devices": sorted({str(p.device) for group in optimizer.param_groups for p in group['params']}),
                    "optimizer_state_devices": {key: sorted({str(value.device) for state in optimizer.state.values()
                        for name, value in state.items() if name == key and isinstance(value, torch.Tensor)})
                        for key in ('step', 'exp_avg', 'exp_avg_sq')}})
                (out_dir / "training_devices.json").write_text(json.dumps({
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": list(design.gpu),
                    "negative_sampling_policy": negative_policy,
                    "objective_is_custom": custom,
                    "loss_definition": "candidate_custom_objective" if custom else "symmetric_diagonal_contrastive_cross_entropy",
                    "loss_direction_reduction": "unknown" if custom else "mean",
                    "loss_replica_reduction": "mean",
                    "loss_epoch_reduction": "arithmetic_mean_of_batch_scalars",
                    "optimizer_learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "embedding_dtype": str(next(encoder.parameters()).dtype),
                    "batches": device_receipts}, indent=2), encoding="utf-8")
        scores_top1: list[float] = []
        scores_top5: list[float] = []
        tally = FixedBankTally(bank, bank_labels)
        epoch_probe = None
        if use_global:
            raw_encoder = encoder.module if isinstance(encoder, torch.nn.DataParallel) else encoder
            encoder.eval()
            note_eval_without_transform(counters)
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch["eeg"].to(device)
                    embedding = raw_encoder(eeg)
                    if not torch.isfinite(embedding).all():
                        raise SplitError("non_finite_development_embedding")
                    if method_search and epoch_probe is None:
                        epoch_probe = embedding[:8].detach().cpu().clone()
                    tally.add(embedding)
                    top1, top5 = within_batch_accuracy(embedding, batch["img_features"].to(device))
                    scores_top1.append(float(top1))
                    scores_top5.append(float(top5))
        else:
            assert model is not None
            raw = model.module if isinstance(model, torch.nn.DataParallel) else model
            raw_encoder = raw.encoder
            raw_encoder.eval()
            note_eval_without_transform(counters)
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch["eeg"].to(device)
                    embedding = raw_encoder(eeg)
                    if not torch.isfinite(embedding).all():
                        raise SplitError("non_finite_development_embedding")
                    if method_search and epoch_probe is None:
                        epoch_probe = embedding[:8].detach().cpu().clone()
                    tally.add(embedding)
                    top1, top5 = within_batch_accuracy(embedding, batch["img_features"].to(device))
                    scores_top1.append(float(top1))
                    scores_top5.append(float(top5))
        finished = epoch_index + 1
        fixed = tally.result()
        val_top1 = sum(scores_top1) / len(scores_top1)
        val_top5 = sum(scores_top5) / len(scores_top5)
        if losses:
            append_history(
                out_dir,
                finished,
                sum(losses) / len(losses),
                val_top1,
                val_top5,
                fixed_bank_top1=fixed["fixed_bank_top1"],
                fixed_bank_top5=fixed["fixed_bank_top5"],
            )
        write_status(out_dir, "training", finished, design.epochs)
        last_fixed = fixed["fixed_bank_top1"]
        if selection.observe(fixed["fixed_bank_top1"], finished):
            best = fixed["fixed_bank_top1"]
            best_top5 = fixed["fixed_bank_top5"]
            best_within = val_top1
            best_epoch = finished
            selected_scores = dict(fixed)
            selected_probe = epoch_probe
            payload = {"state_dict": raw_encoder.state_dict(), "epoch": finished}
            if isinstance(objective, torch.nn.Module):
                payload["objective_state"] = objective.state_dict()
            torch.save(payload, out_dir / "last.ckpt")
        if early_stop_enabled and selection.stall >= 5:
            break
    write_status(out_dir, "finished", finished, design.epochs)
    if duplicate_batches:
        (out_dir / "train_batch_image_ids.jsonl").write_text(
            "\n".join(json.dumps(row) for row in duplicate_batches) + "\n",
            encoding="utf-8",
        )
        (out_dir / "duplicate_sampling.json").write_text(
            json.dumps(
                {
                    "sampling_policy": "first_64_batches",
                    "recorded_batches": len(duplicate_batches),
                    "limit": 64,
                    "seed": design.seed,
                }
            ),
            encoding="utf-8",
        )
    checkpoint = out_dir / "last.ckpt"
    (out_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint": "last.ckpt",
                "source": "selected_checkpoint",
                "best_epoch": best_epoch,
                "last_epoch": finished,
                "best_fixed_bank_top1": best,
                "last_epoch_fixed_bank_top1": last_fixed,
                "policy": design.checkpoint_policy,
                "raw_best_top1": selection.raw_best,
                "saved_top1": selection.saved_best,
                "patience_anchor_top1": selection.patience_anchor,
                "patience_min_delta": selection.min_delta,
                "stall": selection.stall,
                "stop_reason": "epochs_completed" if finished == design.epochs else "development_patience_exhausted",
                "monitor": "legal_development_fixed_bank_top1",
                "monitor_unit": "fraction_0_to_1",
                "tie_policy": "retain_earlier_epoch",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if checkpoint.is_file():
        from react_agent.eeg_training.diagnostics import write_validation_artifacts

        best_encoder = _load_encoder(spec, checkpoint, out_dir)
        if method_search:
            from react_agent.eeg_training.inference_consistency import (
                freeze_training_identity, validate_development_identity, compare_reconstruction, CONSISTENCY_NAME,
            )
            identity = freeze_training_identity(design, data_root, out_dir, checkpoint, best_encoder, val_loader.dataset.records)
            validate_development_identity(identity, design, val_loader.dataset.records)
            actual_scores = _fixed_bank_pass(best_encoder, val_loader, val_loader.dataset.records)
            with torch.no_grad():
                first_batch = next(iter(val_loader))
                actual_probe = best_encoder(first_batch["eeg"].to(device))[:8].detach().cpu()
            if selected_scores is None or selected_probe is None:
                raise SplitError("selected_development_probe_missing")
            consistency = compare_reconstruction(selected_scores, actual_scores, selected_probe, actual_probe)
            consistency["training_inference_identity_fingerprint"] = identity["fingerprint"]
            consistency["checkpoint_sha256"] = identity["context"]["checkpoint_sha256"]
            (out_dir / CONSISTENCY_NAME).write_text(json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8")
        best_encoder.eval()
        note_eval_without_transform(counters)
        limit = int(os.environ.get("EEG_DIAGNOSTIC_SAMPLE_LIMIT") or 32)
        sample_seed = int(os.environ.get("EEG_DIAGNOSTIC_SAMPLE_SEED") or design.seed)
        reservoir = BoundedQueryReservoir(limit, sample_seed)
        with torch.no_grad():
            for batch in val_loader:
                embedding = best_encoder(batch["eeg"].to(device))
                reservoir.add(batch, embedding)
        seen: dict[str, int] = {}
        for row in val_loader.dataset.records:
            image_id = str(row["img"])
            if image_id not in seen:
                seen[image_id] = len(seen)
        bank_pairs = []
        for image_id, index in seen.items():
            if index < bank.shape[0]:
                bank_pairs.append((image_id, [float(value) for value in bank[index].detach().cpu().tolist()]))
        sampled = reservoir.rows()
        positives = {}
        for query_id, _vector in sampled:
            image_id = query_id.split("::")[-1] if "::" in query_id else query_id
            positives[query_id] = {image_id, query_id}
        write_validation_artifacts(
            out_dir,
            sampled,
            bank_pairs,
            positives,
            limit=limit,
            checkpoint_id=str(checkpoint),
            sample_seed=sample_seed,
            total_query_count=reservoir.seen,
            sampling_policy="uniform_reservoir_without_replacement",
        )
    encoder_params = sum(int(item.numel()) for item in encoder.parameters())
    objective_params = sum(int(item.numel()) for item in extra)
    hook = {}
    hook_path = out_dir / "hook_config.json"
    if hook_path.is_file():
        try:
            hook = json.loads(hook_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hook = {}
    config_hash = hook.get("config_hash") or hashlib.sha256(
        json.dumps({"spec": spec, "seed": design.seed}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    used = capabilities_used_payload(
        module=module_name,
        hooks={
            "build_encoder": True,
            "fit_statistics": used_fit,
            "build_training_transform": transform is not None,
            "build_training_objective": True,
            "evaluate_only": False,
        },
        parameter_counts={"encoder": encoder_params, "objective": objective_params},
        counters=counters,
        negative_policy=negative_policy,
        config_hash=config_hash,
    )
    (out_dir / "capabilities_used.json").write_text(json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "primary_metric": best,
        "metric_name": "fixed_bank_top1",
        "fixed_bank_top1": best,
        "fixed_bank_top5": best_top5,
        "top5": best_top5,
        "within_batch_top1": best_within,
        "test_result": None,
        "train_image_count": len(set(train_images)),
        "validation_image_count": len(set(val_images)),
        "query_count": None if selected_scores is None else selected_scores["query_count"],
        "validation_image_ids": sorted(set(val_images)),
        "train_validation_overlap": False,
        "negative_sampling_policy": negative_policy,
        "selected_checkpoint_epoch": best_epoch,
        "last_epoch": finished,
        "last_epoch_fixed_bank_top1": last_fixed,
        "checkpoint_policy": design.checkpoint_policy,
        "evaluation_mode": design.evaluation_mode,
        "raw_best_top1": selection.raw_best,
        "saved_top1": selection.saved_best,
        "stop_reason": "epochs_completed" if finished == design.epochs else "development_patience_exhausted",
        "training_complete": finished == design.epochs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one design or exit before writing a score."""
    args = _parser().parse_args(argv)
    design = Design(
        args.dataset,
        args.exp_setting,
        args.subject,
        args.epochs,
        args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        gpu=parse_gpu_list(args.gpu),
        stop=args.stop,
        weight_decay=args.weight_decay,
        test_only=args.test_only,
        evaluation_mode=args.evaluation_mode,
        checkpoint_policy=args.checkpoint_policy,
        selection_min_delta=args.selection_min_delta,
    )
    if args.device:
        os.environ["EEG_TRAIN_DEVICE"] = args.device
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.json"
    if args.evaluate_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        return _evaluate_main(design, args.data_root, out_dir, checkpoint=checkpoint)
    try:
        validate_design(design)
        limit_visible_gpus(design)
        if args.test_only:
            if design.evaluation_mode == "loso_method_search":
                raise SplitError("method_search_requires_readonly_benchmark_entrypoint")
            payload = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
            payload["test_result"] = score_held_out(design, args.data_root, out_dir)
            metrics.write_text(json.dumps(payload), encoding="utf-8")
            return 0
        payload = fit(design, args.data_root, out_dir)
        image_ids = payload.pop("validation_image_ids", None)
        from react_agent.eeg_research.agentic.execution_protocol import apply_final_test_policy, validation_identity_for

        if isinstance(image_ids, list):
            payload["validation_identity"] = validation_identity_for(design, args.data_root, [str(item) for item in image_ids])
        final_enabled = os.environ.get("EEG_FINAL_TEST", "1") != "0" and design.evaluation_mode != "loso_method_search"
        payload = apply_final_test_policy(payload, final_enabled)
        if final_enabled:
            payload["test_result"] = score_held_out(design, args.data_root, out_dir)
    except SplitError as exc:
        (out_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        write_status(out_dir, "failed", 0, design.epochs)
        return 2
    (out_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return 0


def _evaluate_main(design: Design, data_root: Path, out_dir: Path, checkpoint: Path | None = None) -> int:
    """Write eval_scores.json only. status.json and metrics.json keep the finished trial."""
    target = out_dir / "eval_scores.json"
    try:
        validate_design(design)
        context = Path(checkpoint).parent if checkpoint is not None else out_dir
        if design.evaluation_mode == "loso_method_search" and (out_dir.resolve() == context.resolve() or target.exists()):
            raise SplitError("method_search_requires_new_readonly_scoring_directory")
        payload = evaluate_checkpoint(design, data_root, out_dir, checkpoint=checkpoint, context_dir=context)
    except (SplitError, OSError, RuntimeError, ValueError) as exc:
        target.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return 2
    target.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
