"""Rebuildable candidate packs. evaluate-only reloads the same encoder, not EEGProjectLayer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

PACK_FILES = (
    "last.ckpt",
    "source_binding.json",
    "capabilities_used.json",
    "train_statistics.json",
    "evaluation_identity.json",
    "metrics.json",
    "history.jsonl",
    "diagnostic_bundle.json",
    "diagnostic_summary.json",
    "hook_config.json",
    "frozen_run_spec.json",
    "job.json",
)


def pack_candidate(job_dir: Path, dest: Path, *, workspace: Path | None = None) -> dict[str, Any]:
    """Copy the files needed to rebuild and re-evaluate one candidate. Final test is not packed."""
    job_dir = Path(job_dir)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for name in PACK_FILES:
        src = job_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
        else:
            missing.append(name)
    source = None
    if workspace is not None:
        entry = workspace / "extension" / "eeg_candidate.py"
        if entry.is_file():
            target = dest / "extension"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target / "eeg_candidate.py")
            source = "extension/eeg_candidate.py"
            copied.append(source)
    binding_path = dest / "source_binding.json"
    entry = dest / "extension" / "eeg_candidate.py"
    if binding_path.is_file() and entry.is_file():
        binding = _read_json(binding_path)
        binding["class_file"] = "extension/eeg_candidate.py"
        binding["workspace"] = "."
        binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes: dict[str, str] = {}
    for name in copied:
        path = dest / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "eeg_research.candidate_pack.v2",
        "job_dir": str(job_dir),
        "copied": copied,
        "missing": missing,
        "content_hashes": hashes,
        "evaluate_only": True,
        "final_test": False,
        "checkpoint": "last.ckpt" if (dest / "last.ckpt").is_file() else None,
        "context_root": ".",
    }
    (dest / "pack_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_pack_hashes(dest: Path) -> tuple[bool, str]:
    dest = Path(dest)
    manifest = _read_json(dest / "pack_manifest.json")
    hashes = manifest.get("content_hashes") if isinstance(manifest, dict) else None
    if not hashes:
        return True, "ok"
    for relative, expected in hashes.items():
        path = dest / str(relative)
        if not path.is_file():
            return False, f"dependency_missing:{relative}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            return False, f"hash_mismatch:{relative}"
    return True, "ok"


def pack_is_rebuildable(dest: Path) -> tuple[bool, str]:
    """True when encoder identity and checkpoint exist for evaluate-only."""
    dest = Path(dest)
    ok, reason = verify_pack_hashes(dest)
    if not ok:
        return False, reason
    if not (dest / "last.ckpt").is_file():
        return False, "checkpoint_missing"
    binding = dest / "source_binding.json"
    entry = dest / "extension" / "eeg_candidate.py"
    if not binding.is_file() and not entry.is_file():
        return False, "source_missing"
    if not (dest / "evaluation_identity.json").is_file():
        return False, "evaluation_identity_missing"
    if not (dest / "source_binding.json").is_file() and not (dest / "hook_config.json").is_file():
        return False, "config_or_binding_missing"
    return True, "ok"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rewrite_evaluate_command(
    command: list[Any],
    *,
    out: Path,
    data_root: Path | None,
    checkpoint: Path | None = None,
) -> list[str]:
    argv = [str(item) for item in command if str(item) != "--test-only"]
    if "--out" in argv:
        index = argv.index("--out")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
            argv[index + 1] = str(out)
        else:
            argv.insert(index + 1, str(out))
    else:
        argv.extend(["--out", str(out)])
    if data_root is not None:
        if "--data-root" in argv:
            index = argv.index("--data-root")
            if index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
                argv[index + 1] = str(data_root)
            else:
                argv.insert(index + 1, str(data_root))
        else:
            argv.extend(["--data-root", str(data_root)])
    ckpt = str(checkpoint) if checkpoint is not None else str(Path(out) / "last.ckpt")
    if "--checkpoint" in argv:
        index = argv.index("--checkpoint")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
            argv[index + 1] = ckpt
        else:
            argv.insert(index + 1, ckpt)
    else:
        argv.extend(["--checkpoint", ckpt])
    if "--evaluate-only" not in argv:
        argv.append("--evaluate-only")
    return argv


def evaluate_only_argv(pack: Path, *, out: Path | None = None, data_root: Path | None = None) -> list[str]:
    """Complete train_entry argv for a rebuildable pack. Does not spawn training."""
    import sys

    from react_agent.eeg_training.protocol import data_root as default_root, torch_python

    pack = Path(pack)
    dest = Path(out) if out is not None else pack / "eval_only"
    identity = _read_json(pack / "evaluation_identity.json")
    job = _read_json(pack / "job.json")
    root = Path(data_root) if data_root is not None else default_root()
    command = job.get("command")
    if isinstance(command, list) and command:
        argv = _rewrite_evaluate_command(command, out=dest, data_root=root, checkpoint=pack / "last.ckpt")
    else:
        argv = _fallback_evaluate_argv(job, identity, dest, root, pack, torch_python, sys)
    device = os.environ.get("EEG_TRAIN_DEVICE", "").strip().lower()
    if device in {"cpu", "cuda"} and "--device" not in argv:
        argv.extend(["--device", device])
    return argv


def _fallback_evaluate_argv(job, identity, dest, root, pack, torch_python, sys):
    python = torch_python() or sys.executable
    dataset = str(job.get("dataset") or identity.get("dataset") or "eeg")
    exp_setting = str(job.get("exp_setting") or identity.get("exp_setting") or "inter-subject")
    subject = str(job.get("subject") or identity.get("subject") or "all")
    epochs = int(job.get("epochs") or identity.get("epochs") or 1)
    seed = int(job.get("training_seed") or job.get("seed") or identity.get("split_seed") or 0)
    argv = [
        python,
        "-m",
        "react_agent.eeg_training.train_entry",
        "--dataset",
        dataset,
        "--exp-setting",
        exp_setting,
        "--subject",
        subject,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--data-root",
        str(root),
        "--out",
        str(dest),
        "--evaluate-only",
        "--checkpoint",
        str(pack / "last.ckpt"),
    ]
    train_dir = str(job.get("train_dir") or identity.get("train_dir") or "")
    test_dir = str(job.get("test_dir") or identity.get("test_dir") or "")
    if train_dir:
        argv.extend(["--train-dir", train_dir, "--test-dir", test_dir])
    return argv
