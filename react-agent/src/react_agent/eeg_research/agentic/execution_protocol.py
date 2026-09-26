"""Persist the task the worker must run. The worker reads this file and does not invent another one."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from react_agent.eeg_training.protocol import (
    Design,
    holdout_image_ids,
    learning_rate,
    resolve_generalization,
    split_plan,
)

PILOT_EPOCHS = 3
PROTOCOL_NAME = "execution_protocol.json"


class ProtocolError(ValueError):
    """Raised when a protocol cannot be frozen or no longer matches a job."""


def identity_digest(items: list[str]) -> str:
    """Stable hash of a sample identity. Order does not matter."""
    encoded = json.dumps(sorted(items), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(body: dict[str, Any]) -> str:
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_image_ids(paths: tuple[Path, ...], channels: list[str] | None) -> list[str]:
    """Read image ids from JSON fixtures or real trial files. Missing files are an error."""
    found: list[str] = []
    for path in paths:
        if not path.is_file():
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        raw = path.read_bytes()
        if not raw.strip():
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        if raw.lstrip()[:1] in (b"[", b"{"):
            data = json.loads(raw.decode("utf-8"))
            rows = data.get("img") if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
            found.extend(str(item) for item in rows)
            continue
        from react_agent.eeg_training.data import load_trials

        trials = load_trials(path, channels)
        images = trials["img"]
        if not isinstance(images, list) or not images:
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        found.extend(str(item) for item in images)
    return found


def _channels(design: Design) -> list[str] | None:
    from react_agent.eeg_training.protocol import geometry

    channels = geometry(design.dataset)["channels"]
    return channels if isinstance(channels, list) else None


def build_execution_protocol(
    design: Design,
    data_root: Path,
    *,
    train_image_ids: list[str] | None = None,
    test_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Freeze one runnable task. Holds out images with the same rule the trainer uses."""
    plan = split_plan(data_root, design)
    if plan.val_mode == "other_subjects_test":
        missing = [path for path in plan.val_files if not path.is_file()]
        if missing:
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{missing[0]}")
        train_ids: list[str] = []
        validation_ids: list[str] = []
        validation_files = [str(path) for path in plan.val_files]
        identity = identity_digest(validation_files)
    else:
        if train_image_ids is None:
            channels = _channels(design)
            train_image_ids = read_image_ids(plan.train_files, channels)
            test_image_ids = read_image_ids(plan.forbidden_files, channels)
        if not train_image_ids:
            raise ProtocolError("缺少样本文件，不能冻结空的执行协议")
        train_ids, validation_ids = holdout_image_ids(list(train_image_ids), list(test_image_ids or []), design.seed)
        validation_files = []
        identity = identity_digest(validation_ids)
    body: dict[str, Any] = {
        "dataset": design.dataset,
        "exp_setting": design.exp_setting,
        "subject": design.subject,
        "data_root": str(data_root),
        "training_strategy": design.training_strategy,
        "seed": design.seed,
        "full_epochs": design.epochs,
        "batch_size": design.batch_size,
        "lr": learning_rate(design),
        "gpu": list(design.gpu),
        "weight_decay": design.weight_decay,
        "train_dir": design.train_dir,
        "test_dir": design.test_dir,
        "val_mode": plan.val_mode,
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids,
        "validation_files": validation_files,
        "validation_identity": identity,
        "final_test_enabled": False,
        "fidelity_overrides": {
            "pilot": {"epochs": PILOT_EPOCHS, "stop": "single_full"},
            "full": {"epochs": design.epochs, "stop": "single_early"},
        },
    }
    body["fingerprint"] = _fingerprint(body)
    return body


def load_protocol(camp: Path) -> dict[str, Any] | None:
    """Return the saved protocol. A missing or unreadable file is not a default task."""
    path = camp / PROTOCOL_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload.get("fingerprint"):
        return None
    return payload


def design_from_protocol(protocol: dict[str, Any]) -> Design:
    """Rebuild the design the command must use. Epochs are the user's full run."""
    lr = protocol.get("lr")
    return Design(
        dataset=str(protocol["dataset"]),
        exp_setting=str(protocol["exp_setting"]),
        subject=str(protocol["subject"]),
        epochs=int(protocol["full_epochs"]),
        seed=int(protocol["seed"]),
        batch_size=int(protocol["batch_size"]),
        lr=None if lr is None else float(lr),
        train_dir=str(protocol.get("train_dir") or ""),
        test_dir=str(protocol.get("test_dir") or ""),
        gpu=tuple(int(item) for item in protocol.get("gpu") or ()),
        data_root=str(protocol["data_root"]),
        weight_decay=float(protocol.get("weight_decay") or 1e-4),
        training_strategy=str(protocol.get("training_strategy") or "pooled_subjects"),
        policy="agentic",
    )


def resolve_worker_design(camp: Path, state: dict[str, Any]) -> Design | None:
    """Read the protocol. Missing protocol blocks the campaign instead of inventing EEG / all."""
    protocol = load_protocol(camp)
    if protocol is None:
        state["status"] = "blocked"
        state["detail"] = "execution_protocol_missing"
        return None
    return design_from_protocol(protocol)


def unsupported_agentic_reason(design: Design) -> str | None:
    """Refuse selections the single pooled trainer cannot carry. Empty overrides are allowed."""
    if design.training_strategy != "pooled_subjects":
        return "当前执行器只有一个合训过程，不会为每个被试单独训练"
    derived = resolve_generalization(replace(design, generalization_target=""))
    if design.generalization_target and design.generalization_target != derived:
        return "该泛化目标不会进入训练命令，当前执行器无法执行"
    from react_agent.eeg_training.protocol import _held_out_names

    implied = _held_out_names(replace(design, held_out_subjects=""))
    chosen = design.held_out_subjects.strip()
    if chosen and chosen != implied:
        return "该留出被试覆盖不会进入训练命令，当前执行器无法执行"
    return None


def fidelity_settings(protocol: dict[str, Any], fidelity: str) -> tuple[int, str]:
    overrides = protocol.get("fidelity_overrides") or {}
    row = overrides.get(fidelity) or {}
    epochs = int(row.get("epochs") or protocol["full_epochs"])
    stop = str(row.get("stop") or "single_early")
    return epochs, stop


def project_command(protocol: dict[str, Any], fidelity: str) -> list[str]:
    """Command for one fidelity. Only the recorded epoch and stop may differ from the full run."""
    from react_agent.eeg_training.protocol import train_command

    epochs, stop = fidelity_settings(protocol, fidelity)
    design = replace(design_from_protocol(protocol), epochs=epochs, stop=stop, policy="agentic")
    return train_command(design, Path("job"), Path(protocol["data_root"]))


def _flag(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def command_matches(command: list[str], protocol: dict[str, Any], fidelity: str) -> bool:
    """True when identity fields match. Pilot may use only its recorded epoch and stop."""
    epochs, stop = fidelity_settings(protocol, fidelity)
    expected = {
        "--dataset": str(protocol["dataset"]),
        "--exp-setting": str(protocol["exp_setting"]),
        "--subject": str(protocol["subject"]),
        "--seed": str(protocol["seed"]),
        "--epochs": str(epochs),
        "--data-root": str(Path(protocol["data_root"])),
        "--stop": stop,
    }
    for flag, value in expected.items():
        got = _flag(command, flag)
        if flag == "--data-root" and got is not None:
            if Path(got) != Path(value):
                return False
            continue
        if got != value:
            return False
    if str(protocol.get("training_strategy") or "pooled_subjects") != "pooled_subjects":
        return False
    return True


def validation_identity_for(design: Design, data_root: Path, validation_image_ids: list[str] | None) -> str:
    """Digest the trainer must write. File identity is used when validation is not an image holdout."""
    plan = split_plan(data_root, design)
    if plan.val_mode == "other_subjects_test":
        return identity_digest([str(path) for path in plan.val_files])
    if not validation_image_ids:
        raise ProtocolError("缺少验证样本身份")
    return identity_digest(validation_image_ids)


def apply_final_test_policy(payload: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Drop a final-test score when the research loop must not read that split."""
    if not enabled:
        payload["test_result"] = None
    return payload
