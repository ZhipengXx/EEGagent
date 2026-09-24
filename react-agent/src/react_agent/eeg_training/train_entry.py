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
    return parser


def _image_ids(files: tuple[Path, ...], channels: list[str] | None) -> list[str]:
    from react_agent.eeg_training.data import load_trials

    found: list[str] = []
    for path in files:
        trials = load_trials(path, channels)
        images = trials["img"]
        assert isinstance(images, list)
        found.extend(images)
    return found


def build_loaders(design: Design, data_root: Path):
    """Create train and validation loaders. Test files are not opened as a loader."""
    import torch
    from torch.utils.data import DataLoader

    from react_agent.eeg_training.data import RetrievalTrials, collect_records, load_feature_cache

    plan = split_plan(data_root, design)
    forbidden_paths = set(plan.forbidden_files)
    if forbidden_paths & set(plan.train_files) or forbidden_paths & set(plan.val_files):
        raise SplitError("held_out_test_in_fit")
    spec = geometry(design.dataset)
    channels = spec["channels"]
    assert channels is None or isinstance(channels, list)
    timesteps = spec["timesteps"]
    assert isinstance(timesteps, list)
    if plan.val_mode == "other_subjects_test":
        train_features = load_feature_cache(plan.feature_caches[0])
        val_features = load_feature_cache(plan.feature_caches[1])
        train_records, train_images = collect_records(plan.train_files, train_features, channels, None)
        val_records, val_images = collect_records(plan.val_files, val_features, channels, None)
    else:
        features = load_feature_cache(plan.feature_caches[0])
        train_ids = sorted(set(_image_ids(plan.train_files, channels)))
        test_ids = sorted(set(_image_ids(plan.forbidden_files, channels)))
        kept, validation = holdout_image_ids(train_ids, test_ids, design.seed)
        train_records, train_images = collect_records(plan.train_files, features, channels, set(kept))
        val_records, val_images = collect_records(plan.val_files, features, channels, set(validation))
    if set(train_images) & set(val_images):
        raise SplitError("train_validation_overlap")
    train_set = RetrievalTrials(train_records, timesteps)
    val_set = RetrievalTrials(val_records, timesteps)
    train_loader = DataLoader(train_set, batch_size=min(design.batch_size, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=min(200, len(val_set)), shuffle=False)
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


def score_held_out(design: Design, data_root: Path, out_dir: Path) -> dict[str, float] | None:
    """Score the held-out test files after fit. A missing cache is not a score."""
    plan = split_plan(data_root, design)
    cache = feature_cache(data_root, design, "test")
    checkpoint = out_dir / "last.ckpt"
    if not checkpoint.is_file() or not cache.is_file():
        return None
    if not plan.forbidden_files or any(not path.is_file() for path in plan.forbidden_files):
        return None
    from torch.utils.data import DataLoader

    from react_agent.eeg_training.data import RetrievalTrials, collect_records, load_feature_cache

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
    dataset = RetrievalTrials(records, timesteps)
    loader = DataLoader(dataset, batch_size=min(200, len(dataset)), shuffle=False)
    encoder = _load_encoder(spec, checkpoint)
    result = _fixed_bank_pass(encoder, loader, records)
    return {
        "top1": result["fixed_bank_top1"],
        "top5": result["fixed_bank_top5"],
        "metric": "fixed_bank",
        "query_count": result["query_count"],
        "candidate_count": result["candidate_count"],
    }


def _load_encoder(spec: dict[str, object], checkpoint: Path):
    import torch

    from react_agent.eeg_training.model import EEGProjectLayer

    encoder = EEGProjectLayer(
        z_dim=1024,
        c_num=int(spec["c_num"]),  # type: ignore[arg-type]
        timesteps=list(spec["timesteps"]),  # type: ignore[arg-type]
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["state_dict"])
    return encoder.to(torch.device("cuda:0"))


def _fixed_bank_pass(encoder, loader, records) -> dict[str, float]:
    """Rank each batch's queries against every image in this split."""
    import torch

    from react_agent.eeg_training.fixed_bank import FixedBankTally, frozen_bank

    device = torch.device("cuda:0")
    bank, labels = frozen_bank(records)
    tally = FixedBankTally(bank.to(device), labels.to(device))
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            tally.add(encoder(batch["eeg"].to(device)))
    return tally.result()


def evaluate_checkpoint(design: Design, data_root: Path, out_dir: Path) -> dict[str, object]:
    """Score an existing checkpoint. The trial's metrics, curve and checkpoint are not rewritten."""
    limit_visible_gpus(design)
    import torch

    if not torch.cuda.is_available():
        raise SplitError("cuda_unavailable")
    checkpoint = out_dir / "last.ckpt"
    if not checkpoint.is_file():
        raise SplitError("checkpoint_missing")
    _train_loader, val_loader, _train_images, _val_images, spec = build_loaders(design, data_root)
    encoder = _load_encoder(spec, checkpoint)
    validation = _fixed_bank_pass(encoder, val_loader, val_loader.dataset.records)
    return {
        "fixed_bank_top1": validation["fixed_bank_top1"],
        "fixed_bank_top5": validation["fixed_bank_top5"],
        "query_count": validation["query_count"],
        "candidate_count": validation["candidate_count"],
        "test_result": score_held_out(design, data_root, out_dir),
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


def _build_encoder(spec: dict[str, object], out_dir: Path, train_loader=None):
    """Build the baseline, or the candidate module named by the job environment."""
    module_name = os.environ.get("EEG_CANDIDATE_MODULE", "")
    candidate_path = os.environ.get("EEG_CANDIDATE_PATH", "")
    if candidate_path and candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)
    if not module_name:
        from react_agent.eeg_training.model import EEGProjectLayer

        return EEGProjectLayer(z_dim=1024, c_num=int(spec["c_num"]), timesteps=list(spec["timesteps"]))
    import importlib

    from react_agent.eeg_research.agentic.binding import write_binding

    module = importlib.import_module(module_name)
    candidate = module.EEGCandidate()
    encoder = candidate.build_encoder({"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])})
    if hasattr(candidate, "fit_statistics") and train_loader is not None:
        stats = train_channel_statistics(train_loader)
        candidate.fit_statistics(encoder, stats)
        (out_dir / "train_statistics.json").write_text(json.dumps(stats), encoding="utf-8")
    write_binding(out_dir, candidate)
    return encoder


def fit(design: Design, data_root: Path, out_dir: Path) -> dict[str, object]:
    """Train on validation top-1 and write metrics only after a completed loop."""
    limit_visible_gpus(design)
    begin_history(out_dir, design.epochs)
    import torch

    from react_agent.eeg_training.fixed_bank import FixedBankTally, frozen_bank
    from react_agent.eeg_training.model import LocalRetrieval

    if not torch.cuda.is_available():
        raise SplitError("cuda_unavailable")
    random.seed(design.seed)
    torch.manual_seed(design.seed)
    torch.cuda.manual_seed_all(design.seed)
    train_loader, val_loader, train_images, val_images, spec = build_loaders(design, data_root)
    device = torch.device("cuda:0")
    encoder = _build_encoder(spec, out_dir, train_loader).to(device)
    model: torch.nn.Module = LocalRetrieval(encoder)
    if len(design.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(design.gpu))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(design), weight_decay=design.weight_decay)
    bank, bank_labels = frozen_bank(val_loader.dataset.records)
    bank = bank.to(device)
    bank_labels = bank_labels.to(device)
    best = None
    stall = 0
    best_top5 = None
    best_within = None
    finished = 0
    early_stop_enabled = design.stop in {"single_early", "chain_early"}
    write_status(out_dir, "training", 0, design.epochs)
    for epoch_index in range(design.epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            loss, _top1, _top5 = model(batch["eeg"].to(device), batch["img_features"].to(device))
            loss = loss.mean()
            losses.append(float(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scores_top1: list[float] = []
        scores_top5: list[float] = []
        tally = FixedBankTally(bank, bank_labels)
        raw = model.module if isinstance(model, torch.nn.DataParallel) else model
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch["eeg"].to(device)
                _loss, top1, top5 = model(eeg, batch["img_features"].to(device))
                scores_top1.append(float(top1.mean()))
                scores_top5.append(float(top5.mean()))
                tally.add(raw.encoder(eeg))
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
        if best is None or fixed["fixed_bank_top1"] > best + 0.001:
            best = fixed["fixed_bank_top1"]
            best_top5 = fixed["fixed_bank_top5"]
            best_within = val_top1
            stall = 0
            torch.save({"state_dict": raw.encoder.state_dict()}, out_dir / "last.ckpt")
        else:
            stall += 1
        if early_stop_enabled and stall >= 5:
            break
    write_status(out_dir, "finished", finished, design.epochs)
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
        "train_validation_overlap": False,
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
    )
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.json"
    if args.evaluate_only:
        return _evaluate_main(design, args.data_root, out_dir)
    try:
        validate_design(design)
        limit_visible_gpus(design)
        if args.test_only:
            payload = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
            payload["test_result"] = score_held_out(design, args.data_root, out_dir)
            metrics.write_text(json.dumps(payload), encoding="utf-8")
            return 0
        payload = fit(design, args.data_root, out_dir)
        if os.environ.get("EEG_FINAL_TEST", "1") == "0":
            payload["test_result"] = None
        else:
            payload["test_result"] = score_held_out(design, args.data_root, out_dir)
    except SplitError as exc:
        (out_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        write_status(out_dir, "failed", 0, design.epochs)
        return 2
    (out_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return 0


def _evaluate_main(design: Design, data_root: Path, out_dir: Path) -> int:
    """Write eval_scores.json only. status.json and metrics.json keep the finished trial."""
    target = out_dir / "eval_scores.json"
    try:
        validate_design(design)
        payload = evaluate_checkpoint(design, data_root, out_dir)
    except (SplitError, OSError, RuntimeError, ValueError) as exc:
        target.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return 2
    target.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
