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


NEGATIVE_POLICY = "data_parallel_local"
PROTOCOL_SCHEMA = "eeg_research.v1.9"


def query_id(subject: str, image_id: str) -> str:
    """One EEG query is a subject plus an image, not a de-duplicated image name."""
    return f"{subject}::{image_id}"


def identity_digest(items: list[str]) -> str:
    """Stable hash of a sample identity. Order does not matter."""
    encoded = json.dumps(sorted(items), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subject_from_path(path: Path, fallback: str) -> str:
    name = path.parent.name
    if name.startswith("sub-") or (name and name != fallback and name not in {"train", "test"}):
        return name
    return fallback or "custom"


def _fingerprint(body: dict[str, Any]) -> str:
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_image_ids(paths: tuple[Path, ...], channels: list[str] | None) -> list[str]:
    """Read image ids from JSON fixtures or real trial files. Missing files are an error."""
    return [image for _query, image in read_query_rows(paths, channels, "custom")]


def read_query_rows(
    paths: tuple[Path, ...],
    channels: list[str] | None,
    subject_fallback: str,
) -> list[tuple[str, str]]:
    """Read (query_id, image_id) rows. Missing files are an error."""
    found: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        raw = path.read_bytes()
        if not raw.strip():
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        subject = _subject_from_path(path, subject_fallback)
        if raw.lstrip()[:1] in (b"[", b"{"):
            data = json.loads(raw.decode("utf-8"))
            rows = data.get("img") if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
            found.extend((query_id(subject, str(item)), str(item)) for item in rows)
            continue
        from react_agent.eeg_training.data import load_trials

        trials = load_trials(path, channels)
        images = trials["img"]
        if not isinstance(images, list) or not images:
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{path}")
        found.extend((query_id(subject, str(item)), str(item)) for item in images)
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
    split_seed: int | None = None,
    training_seed: int | None = None,
    training_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Freeze one runnable task. Holds out images with the same rule the trainer uses."""
    from react_agent.eeg_training.protocol import geometry

    plan = split_plan(data_root, design)
    split = int(design.seed if split_seed is None else split_seed)
    train_seed = int(design.seed if training_seed is None else training_seed)
    seeds = [int(item) for item in (training_seeds or [train_seed])]
    if train_seed not in seeds:
        seeds.insert(0, train_seed)
    channels = _channels(design)
    subject_fallback = design.subject or "custom"
    train_query_ids: list[str] = []
    validation_query_ids: list[str] = []
    positive_map: dict[str, str] = {}
    if plan.val_mode == "other_subjects_test":
        missing = [path for path in plan.val_files if not path.is_file()]
        if missing:
            raise ProtocolError(f"缺少样本文件，不能冻结空的执行协议：{missing[0]}")
        validation_files = [str(path) for path in plan.val_files]
        train_rows = read_query_rows(plan.train_files, channels, subject_fallback)
        val_rows = read_query_rows(plan.val_files, channels, subject_fallback)
        train_query_ids = [qid for qid, _img in train_rows]
        validation_query_ids = [qid for qid, _img in val_rows]
        positive_map = {qid: img for qid, img in [*train_rows, *val_rows]}
        train_ids = list(dict.fromkeys(img for _qid, img in train_rows))
        validation_ids = list(dict.fromkeys(img for _qid, img in val_rows))
        if not validation_ids or not validation_query_ids:
            raise ProtocolError("缺少样本文件，不能冻结空的执行协议")
        gallery_ids = set(validation_ids)
        missing_positive = [qid for qid in validation_query_ids if positive_map.get(qid) not in gallery_ids]
        if missing_positive:
            raise ProtocolError("positive_not_in_gallery")
        identity = identity_digest(validation_ids)
    else:
        if train_image_ids is None:
            train_rows = read_query_rows(plan.train_files, channels, subject_fallback)
            test_rows = read_query_rows(plan.forbidden_files, channels, subject_fallback)
            train_image_ids = [img for _qid, img in train_rows]
            test_image_ids = [img for _qid, img in test_rows]
            query_by_image: dict[str, list[str]] = {}
            for qid, img in train_rows:
                query_by_image.setdefault(img, []).append(qid)
                positive_map[qid] = img
        else:
            query_by_image = {img: [query_id(subject_fallback, img)] for img in train_image_ids}
            positive_map = {query_id(subject_fallback, img): img for img in train_image_ids}
        if not train_image_ids:
            raise ProtocolError("缺少样本文件，不能冻结空的执行协议")
        train_ids, validation_ids = holdout_image_ids(list(train_image_ids), list(test_image_ids or []), split)
        validation_files = []
        identity = identity_digest(validation_ids)
        for image in train_ids:
            train_query_ids.extend(query_by_image.get(image) or [query_id(subject_fallback, image)])
        for image in validation_ids:
            validation_query_ids.extend(query_by_image.get(image) or [query_id(subject_fallback, image)])
    geom = geometry(design.dataset)
    body: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset": design.dataset,
        "exp_setting": design.exp_setting,
        "subject": design.subject,
        "data_root": str(data_root),
        "training_strategy": design.training_strategy,
        "seed": train_seed,
        "split_seed": split,
        "training_seed": train_seed,
        "training_seeds": seeds,
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
        "train_query_ids": train_query_ids,
        "validation_query_ids": validation_query_ids,
        "gallery_image_ids": list(validation_ids),
        "positive_map": positive_map,
        "validation_files": validation_files,
        "validation_identity": identity,
        "input_geometry": {
            "c_num": geom["c_num"],
            "timesteps": list(geom["timesteps"]),
            "channels": geom["channels"],
        },
        "negative_sampling_policy": NEGATIVE_POLICY,
        "final_test_enabled": False,
        "fidelity_overrides": {
            "pilot": {"epochs": PILOT_EPOCHS, "stop": "single_full"},
            "full": {"epochs": design.epochs, "stop": "single_early"},
        },
    }
    body["fingerprint"] = _fingerprint({key: value for key, value in body.items() if key != "fingerprint"})
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


def _weight_decay(protocol: dict[str, Any]) -> float:
    raw = protocol.get("weight_decay")
    if raw is None:
        return 1e-4
    return float(raw)


def design_from_protocol(protocol: dict[str, Any], *, training_seed: int | None = None) -> Design:
    """Rebuild the design the command must use. Epochs are the user's full run."""
    lr = protocol.get("lr")
    if training_seed is not None:
        seed = int(training_seed)
    elif protocol.get("training_seed") is not None:
        seed = int(protocol["training_seed"])
    else:
        seed = int(protocol["seed"])
    return Design(
        dataset=str(protocol["dataset"]),
        exp_setting=str(protocol["exp_setting"]),
        subject=str(protocol["subject"]),
        epochs=int(protocol["full_epochs"]),
        seed=seed,
        batch_size=int(protocol["batch_size"]),
        lr=None if lr is None else float(lr),
        train_dir=str(protocol.get("train_dir") or ""),
        test_dir=str(protocol.get("test_dir") or ""),
        gpu=tuple(int(item) for item in protocol.get("gpu") or ()),
        data_root=str(protocol["data_root"]),
        weight_decay=_weight_decay(protocol),
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


def project_command(protocol: dict[str, Any], fidelity: str, *, training_seed: int | None = None) -> list[str]:
    """Command for one fidelity. Only the recorded epoch and stop may differ from the full run."""
    from react_agent.eeg_training.protocol import train_command

    epochs, stop = fidelity_settings(protocol, fidelity)
    design = replace(
        design_from_protocol(protocol, training_seed=training_seed),
        epochs=epochs,
        stop=stop,
        policy="agentic",
    )
    return train_command(design, Path("job"), Path(protocol["data_root"]))


def effective_config(protocol: dict[str, Any], fidelity: str, *, training_seed: int | None = None) -> dict[str, Any]:
    """Resolved training config that must match the child argv."""
    seed = int(protocol.get("training_seed") if training_seed is None else training_seed)
    epochs, stop = fidelity_settings(protocol, fidelity)
    return {
        "dataset": protocol["dataset"],
        "exp_setting": protocol["exp_setting"],
        "subject": protocol["subject"],
        "data_root": str(Path(protocol["data_root"])),
        "gpu": [int(item) for item in protocol.get("gpu") or ()],
        "batch_size": int(protocol["batch_size"]),
        "lr": float(protocol["lr"]),
        "weight_decay": _weight_decay(protocol),
        "split_seed": int(protocol.get("split_seed", protocol.get("seed") or 0)),
        "training_seed": seed,
        "epochs": epochs,
        "stop": stop,
        "negative_sampling_policy": str(protocol.get("negative_sampling_policy") or NEGATIVE_POLICY),
    }


def _flag(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def command_matches(
    command: list[str],
    protocol: dict[str, Any],
    fidelity: str,
    *,
    training_seed: int | None = None,
) -> bool:
    """True when identity and training fields match the frozen protocol."""
    config = effective_config(protocol, fidelity, training_seed=training_seed)
    expected = {
        "--dataset": str(config["dataset"]),
        "--exp-setting": str(config["exp_setting"]),
        "--subject": str(config["subject"]),
        "--seed": str(config["training_seed"]),
        "--epochs": str(config["epochs"]),
        "--data-root": str(Path(config["data_root"])),
        "--stop": str(config["stop"]),
        "--batch-size": str(config["batch_size"]),
        "--lr": _lr_flag(config["lr"]),
        "--weight-decay": _wd_flag(config["weight_decay"]),
        "--negative-policy": str(config["negative_sampling_policy"]),
    }
    for flag, value in expected.items():
        got = _flag(command, flag)
        if flag == "--data-root" and got is not None:
            if Path(got) != Path(value):
                return False
            continue
        if got != value:
            return False
    gpu_flag = _flag(command, "--gpu")
    gpu = [int(item) for item in config["gpu"]]
    if gpu:
        if gpu_flag != ",".join(str(item) for item in gpu):
            return False
    elif gpu_flag not in {None, ""}:
        return False
    if str(protocol.get("training_strategy") or "pooled_subjects") != "pooled_subjects":
        return False
    return True


def _lr_flag(value: float) -> str:
    return f"{float(value):.8g}"


def _wd_flag(value: float) -> str:
    return f"{float(value):.8g}"


def next_unused_training_seed(protocol: dict[str, Any], used: set[int]) -> int | None:
    """Return the next declared training seed that has not produced a valid run."""
    seeds = protocol.get("training_seeds") or [protocol.get("training_seed", protocol.get("seed"))]
    for seed in seeds:
        if int(seed) not in used:
            return int(seed)
    return None


def write_evaluation_identity(job_dir: Path, protocol: dict[str, Any]) -> Path:
    """Copy frozen split identity into the job directory so training cannot re-split."""
    payload = {
        "split_seed": protocol.get("split_seed", protocol.get("seed")),
        "train_image_ids": protocol.get("train_image_ids") or [],
        "validation_image_ids": protocol.get("validation_image_ids") or [],
        "train_query_ids": protocol.get("train_query_ids") or [],
        "validation_query_ids": protocol.get("validation_query_ids") or [],
        "gallery_image_ids": protocol.get("gallery_image_ids") or protocol.get("validation_image_ids") or [],
        "positive_map": protocol.get("positive_map") or {},
        "validation_identity": protocol.get("validation_identity"),
        "val_mode": protocol.get("val_mode"),
        "schema_version": protocol.get("schema_version"),
    }
    path = job_dir / "evaluation_identity.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
