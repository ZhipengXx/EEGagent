"""Experiment choices and split rules. This module does not import torch."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

HOLDOUT_FRACTION = 0.1
EEG_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
FULL_EEG_CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]
EEG_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 11)]
MEG_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 5)]
DATA_ROOT = Path("/home/zxuff/data/Uncertainty-aware-Blur-Prior/data")
DEFAULT_GPU_SECONDS = 28800.0
STOP_LABELS = {
    "single_full": "单次跑满 epoch",
    "single_early": "单次验证集早停",
    "chain_full": "自动两试，每次跑满 epoch",
    "chain_early": "自动两试，验证集早停",
}
BASELINE_WEIGHT_DECAY = 1e-4
CHAIN_WEIGHT_DECAY = 0.01
TORCH_PYTHON_CANDIDATES = (Path("/home/zxuff/miniconda3/envs/ubp/bin/python"),)


class SplitError(ValueError):
    """Raised when a design cannot keep validation away from test."""


@dataclass(frozen=True)
class Design:
    """One selectable retrieval experiment."""

    dataset: str
    exp_setting: str
    subject: str
    epochs: int = 50
    seed: int = 0
    batch_size: int = 1024
    lr: float | None = None
    train_dir: str = ""
    test_dir: str = ""
    gpu: tuple[int, ...] = ()
    data_root: str = ""
    gpu_seconds: float | None = None
    stop: str = "chain_early"
    weight_decay: float = 1e-4
    test_only: bool = False
    training_strategy: str = "pooled_subjects"
    generalization_target: str = ""
    held_out_subjects: str = ""
    policy: str = "legacy_fixed"
    gpu_mode: str = "explicit"
    evaluate_only: bool = False

    def campaign_id(self) -> str:
        """Stable directory name for this design."""
        setting = self.exp_setting.replace("-", "_")
        slug = (self.subject or "none").replace(",", "_")
        return f"{self.dataset}_{setting}_{slug}_s{self.seed}"


@dataclass(frozen=True)
class SplitPlan:
    """Files the trainer may read, and the held-out test files it must not fit on."""

    train_files: tuple[Path, ...]
    val_files: tuple[Path, ...]
    forbidden_files: tuple[Path, ...]
    feature_caches: tuple[Path, ...]
    train_subjects: tuple[str, ...]
    val_mode: str


def subjects_for(dataset: str) -> list[str]:
    """Return the subject ids for EEG or MEG."""
    if dataset == "eeg":
        return list(EEG_SUBJECTS)
    if dataset == "meg":
        return list(MEG_SUBJECTS)
    raise SplitError("unknown_dataset")


def parse_subject_selection(text: str) -> str:
    """Parse a subject selection. An empty string selects nothing; all stays a token."""
    raw = text.strip()
    if not raw:
        return ""
    if raw == "all":
        return "all"
    chosen: list[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item or item == "all" or item in chosen or "/" in item or "\\" in item or ".." in item:
            raise SplitError("bad_subject")
        chosen.append(item)
    return ",".join(chosen)


def list_subjects(root: Path, dataset: str) -> list[str]:
    """Return subject directories that contain train.pt. An empty scan stays empty."""
    base = data_dir(root, dataset)
    if not base.is_dir():
        return []
    found: list[str] = []
    for path in sorted(base.iterdir()):
        if not path.is_dir() or not (path / "train.pt").is_file() or "," in path.name or path.name == "all":
            continue
        try:
            if parse_subject_selection(path.name) != path.name:
                continue
        except SplitError:
            continue
        found.append(path.name)
    return found


def resolve_subjects(design: Design, root: Path) -> tuple[str, ...]:
    """Expand all against the scan. Explicit names stay as written."""
    if design.subject == "all":
        found = list_subjects(root, design.dataset)
        if not found:
            raise SplitError("no_subject")
        return tuple(found)
    if not design.subject:
        return ()
    return tuple(design.subject.split(","))


def subject_universe(root: Path, dataset: str) -> tuple[str, ...]:
    """Use scanned subjects, or the built-in list when the directory has none."""
    found = list_subjects(root, dataset)
    if found:
        return tuple(found)
    return tuple(subjects_for(dataset))


def validate_design(design: Design) -> None:
    """Refuse an unknown modality or protocol, or a subject selection that cannot be read."""
    if design.dataset not in {"eeg", "meg"}:
        raise SplitError("unknown_dataset")
    if design.exp_setting not in {"intra-subject", "inter-subject"}:
        raise SplitError("unknown_protocol")
    if parse_subject_selection(design.subject) != design.subject:
        raise SplitError("bad_subject")
    if not design.subject and not (design.train_dir and design.test_dir):
        raise SplitError("no_subject")
    if design.epochs < 1 or design.seed < 0 or design.batch_size < 1:
        raise SplitError("bad_budget")
    if design.lr is not None and design.lr <= 0:
        raise SplitError("bad_budget")
    if design.gpu_seconds is not None and design.gpu_seconds <= 0:
        raise SplitError("bad_budget")
    if bool(design.train_dir) != bool(design.test_dir):
        raise SplitError("训练集和测试集目录需要一起填写")
    if any(index < 0 for index in design.gpu):
        raise SplitError("bad_gpu")
    if design.stop not in STOP_LABELS:
        raise SplitError("bad_stop")
    if design.weight_decay < 0:
        raise SplitError("bad_budget")
    if design.training_strategy not in {"per_subject", "pooled_subjects"}:
        raise SplitError("bad_strategy")
    if design.policy not in {"legacy_fixed", "adaptive", "agentic"}:
        raise SplitError("bad_policy")
    if design.gpu_mode not in {"explicit", "auto_one"}:
        raise SplitError("bad_gpu_mode")
    target = resolve_generalization(design)
    if target not in {"seen_subject_unseen_stimulus", "held_out_subject", "custom"}:
        raise SplitError("bad_generalization")
    if target == "held_out_subject" and not _held_out_names(design):
        raise SplitError("held_out_subject_missing")


def data_dir(data_root: Path, dataset: str) -> Path:
    """Return the preprocessed directory for one modality."""
    if dataset == "eeg":
        return data_root / "things-eeg" / "Preprocessed_data_250Hz_whiten"
    return data_root / "things-meg" / "Preprocessed_data"


def resolve_generalization(design: Design) -> str:
    """Name the split. All-subject pooling is image holdout, not cross-subject."""
    if design.generalization_target:
        return design.generalization_target
    if design.exp_setting == "intra-subject" or design.subject == "all":
        return "seen_subject_unseen_stimulus"
    return "held_out_subject"


def _held_out_names(design: Design) -> str:
    if design.held_out_subjects.strip():
        return design.held_out_subjects.strip()
    if design.generalization_target == "held_out_subject":
        return ""
    if design.exp_setting == "inter-subject" and design.subject not in {"", "all"}:
        return design.subject
    return ""


def protocol_label(design: Design) -> str:
    """User-facing split name. It does not rename an old campaign."""
    target = resolve_generalization(design)
    if design.training_strategy == "per_subject":
        pooled = "每被试一个模型"
    else:
        pooled = "多被试合训"
    if target == "held_out_subject":
        return f"{pooled} · 留出被试"
    if design.subject == "all" and design.training_strategy == "pooled_subjects":
        return "多被试合训 · 图像留出验证"
    return f"{pooled} · 未见图像验证"


def split_manifest(plan: SplitPlan, design: Design) -> dict[str, object]:
    """Record file roles. A test.pt name does not by itself mean final test."""
    target = resolve_generalization(design)
    rows = []
    for path in plan.train_files:
        rows.append({"path": str(path), "role": "train"})
    for path in plan.val_files:
        rows.append({"path": str(path), "role": "selection_validation"})
    for path in plan.forbidden_files:
        role = "final_test" if target == "held_out_subject" else "held_out_unused"
        rows.append({"path": str(path), "role": role})
    return {
        "training_strategy": design.training_strategy,
        "generalization_target": target,
        "label": protocol_label(design),
        "files": rows,
    }


def early_stop(design: Design) -> bool:
    """Validation stall ends a trial only when that condition was selected."""
    return design.stop in {"single_early", "chain_early"}


def chains(design: Design) -> bool:
    """A chain starts the weight-decay trial after the baseline."""
    return design.stop in {"chain_full", "chain_early"}


def parse_stop(text: str) -> str:
    """Read the end-condition field. Blank keeps the default chain."""
    raw = text.strip()
    if not raw:
        return "chain_early"
    if raw not in STOP_LABELS:
        raise SplitError("bad_stop")
    return raw


def exp_name(design: Design) -> str:
    """Match the baseline cache name used by the source trainer."""
    return f"{design.dataset}_{design.exp_setting}_baseline_EEGProjectLayer_RN50"


def feature_cache(data_root: Path, design: Design, mode: str) -> Path:
    """Return the RN50 feature cache for train or test images."""
    root = data_dir(data_root, design.dataset)
    return root.parent / "Image_feature" / "DirectT" / f"{exp_name(design)}_{mode}.pt"


def default_lr(design: Design) -> float:
    """Use the source script's learning rate for each protocol."""
    return 1e-4 if design.exp_setting == "intra-subject" else 1e-5


def parse_gpu_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated GPU list. An empty string selects nothing."""
    raw = text.strip()
    if not raw:
        return ()
    chosen: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            raise SplitError("bad_gpu")
        try:
            index = int(item)
        except ValueError as exc:
            raise SplitError("bad_gpu") from exc
        if index < 0 or index in chosen:
            raise SplitError("bad_gpu")
        chosen.append(index)
    return tuple(chosen)


def learning_rate(design: Design) -> float:
    """Return the chosen learning rate, or the protocol default."""
    return default_lr(design) if design.lr is None else design.lr


def geometry(dataset: str) -> dict[str, object]:
    """Return channel count and time window for the baseline encoder."""
    if dataset == "eeg":
        return {"c_num": len(EEG_CHANNELS), "timesteps": [0, 250], "channels": list(EEG_CHANNELS)}
    return {"c_num": 271, "timesteps": [0, 201], "channels": None}


def holdout_image_ids(train_ids: list[str], test_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Hold out training images for validation. Test ids stay out of both sets."""
    train_set = set(train_ids)
    test_set = set(test_ids)
    if train_set & test_set:
        raise SplitError("train_test_overlap")
    unique = sorted(train_set)
    if len(unique) < 2:
        raise SplitError("holdout_too_small")
    count = max(1, int(round(len(unique) * HOLDOUT_FRACTION)))
    count = min(count, len(unique) - 1)
    order = unique[:]
    _shuffle(order, seed)
    validation = sorted(order[:count])
    kept = sorted(set(unique) - set(validation))
    if set(validation) & test_set or set(kept) & set(validation):
        raise SplitError("validation_test_overlap")
    return kept, validation


def split_plan(data_root: Path, design: Design) -> SplitPlan:
    """Pool the selected subjects into one model. Held-out test files stay out of fit."""
    validate_design(design)
    if design.train_dir and design.test_dir:
        train_file = Path(design.train_dir) / "train.pt"
        test_file = Path(design.test_dir) / "test.pt"
        return SplitPlan(
            train_files=(train_file,),
            val_files=(train_file,),
            forbidden_files=(test_file,),
            feature_caches=(feature_cache(data_root, design, "train"),),
            train_subjects=(),
            val_mode="custom_train_holdout",
        )
    selected = resolve_subjects(design, data_root)
    if not selected:
        raise SplitError("no_subject")
    root = data_dir(data_root, design.dataset)
    universe = subject_universe(data_root, design.dataset)
    held_out = tuple(name for name in universe if name not in selected)
    train_files = tuple(root / name / "train.pt" for name in selected)
    if design.exp_setting == "intra-subject" or not held_out:
        mode = "train_holdout" if design.exp_setting == "intra-subject" else "all_subjects_holdout"
        return SplitPlan(
            train_files=train_files,
            val_files=train_files,
            forbidden_files=tuple(root / name / "test.pt" for name in selected),
            feature_caches=(feature_cache(data_root, design, "train"),),
            train_subjects=selected,
            val_mode=mode,
        )
    return SplitPlan(
        train_files=train_files,
        val_files=tuple(root / name / "test.pt" for name in selected),
        forbidden_files=tuple(root / name / "test.pt" for name in held_out),
        feature_caches=(
            feature_cache(data_root, design, "train"),
            feature_cache(data_root, design, "test"),
        ),
        train_subjects=selected,
        val_mode="other_subjects_test",
    )


def data_root() -> Path:
    """Resolve the THINGS data root."""
    override = os.environ.get("EEG_DATA_ROOT")
    return Path(override) if override else DATA_ROOT


def torch_python() -> str | None:
    """Return the training interpreter. EEG_TRAIN_PYTHON overrides the UBP env."""
    chosen = os.environ.get("EEG_TRAIN_PYTHON")
    if chosen:
        return chosen if Path(chosen).is_file() else None
    for candidate in TORCH_PYTHON_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def gpu_seconds() -> float | None:
    """Return the environment cap. An empty value stays unknown."""
    raw = os.environ.get("EEG_GPU_SECONDS")
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def parse_gpu_seconds(text: str) -> float | None:
    """Parse a form cap. An empty string leaves the choice to the environment."""
    raw = text.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise SplitError("bad_budget") from exc
    if value <= 0:
        raise SplitError("bad_budget")
    return value


def resolve_gpu_seconds(chosen: float | None) -> float:
    """Prefer the form, then the environment, then eight hours."""
    if chosen is not None:
        return chosen
    env = gpu_seconds()
    return DEFAULT_GPU_SECONDS if env is None else env


def format_gpu_seconds(value: float) -> str:
    """Render a whole cap without a trailing decimal."""
    if value == int(value):
        return str(int(value))
    return f"{value:.8g}"


def describe_split(plan: SplitPlan) -> dict[str, object]:
    """Describe the automatic or handwritten split for the form."""
    notes = {
        "train_holdout": "自动划分。勾选被试的训练文件合成一个训练集，验证图像从中划出 10%，测试文件不进入训练。",
        "all_subjects_holdout": "已选全部被试，没有留出被试。验证集来自训练图像的 10%，测试文件不进入训练。",
        "other_subjects_test": "自动划分。勾选被试的训练文件合成训练集，他们的测试文件做验证。未勾选被试的测试文件不进入训练。",
        "custom_train_holdout": "使用手写目录。验证集从训练集划出，测试文件不进入训练。",
    }
    return {
        "automatic": plan.val_mode != "custom_train_holdout",
        "val_mode": plan.val_mode,
        "note": notes.get(plan.val_mode, plan.val_mode),
        "train_files": _file_rows(plan.train_files),
        "val_files": _file_rows(plan.val_files),
        "forbidden_files": _file_rows(plan.forbidden_files),
        "feature_caches": _file_rows(plan.feature_caches),
    }


def parse_nvidia_smi(text: str) -> list[dict[str, object]]:
    """Parse `nvidia-smi --query-gpu=index,name,memory.free` rows."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            free = int(parts[2])
        except ValueError:
            continue
        rows.append({"index": index, "name": parts[1], "memory_free_mb": free})
    return rows


def resolve_run_gpu(design: Design) -> Design:
    """Bind cards before a run. An empty selection never means every card."""
    if design.gpu:
        return design
    if design.gpu_mode != "auto_one":
        raise SplitError("gpu_not_selected")
    rows = list_gpus()
    if not rows:
        raise SplitError("gpu_not_found")
    best = max(rows, key=lambda row: int(row["memory_free_mb"]))
    return replace(design, gpu=(int(best["index"]),))


def list_gpus() -> list[dict[str, object]]:
    """Read visible GPUs. A failed query returns an empty list."""
    try:
        completed = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return parse_nvidia_smi(completed.stdout)


def probe_blockers(design: Design, root: Path | None = None) -> list[str]:
    """List why a real trial cannot start. An empty list means the files are present."""
    validate_design(design)
    base = root if root is not None else data_root()
    plan = split_plan(base, design)
    missing: list[str] = []
    for path in (*plan.train_files, *plan.val_files, *plan.forbidden_files, *plan.feature_caches):
        if not path.is_file():
            missing.append(f"未找到 {path}")
    if torch_python() is None:
        missing.append("未配置 EEG_TRAIN_PYTHON，无法确认 torch 解释器")
    return missing


def train_command(design: Design, out_dir: Path, root: Path | None = None) -> list[str]:
    """Build a shell-free command for one trial."""
    python = torch_python()
    if python is None:
        raise SplitError("torch_python_missing")
    base = Path(design.data_root) if design.data_root else (root if root is not None else data_root())
    command = [
        python,
        "-m",
        "react_agent.eeg_training.train_entry",
        "--dataset",
        design.dataset,
        "--exp-setting",
        design.exp_setting,
        "--subject",
        design.subject,
        "--epochs",
        str(design.epochs),
        "--seed",
        str(design.seed),
        "--batch-size",
        str(design.batch_size),
        "--lr",
        f"{learning_rate(design):.8g}",
        "--data-root",
        str(base),
        "--out",
        str(out_dir),
        "--stop",
        design.stop,
        "--weight-decay",
        f"{design.weight_decay:.8g}",
    ]
    if design.test_only:
        command.append("--test-only")
    if design.evaluate_only:
        command.append("--evaluate-only")
    if design.train_dir:
        command.extend(["--train-dir", design.train_dir, "--test-dir", design.test_dir])
    if design.gpu:
        command.extend(["--gpu", ",".join(str(index) for index in design.gpu)])
    return command


def options() -> dict[str, object]:
    """Choices the workbench renders."""
    return {
        "datasets": [
            {"id": "eeg", "label": "EEG"},
            {"id": "meg", "label": "MEG"},
        ],
        "protocols": [
            {"id": "intra-subject", "label": "被试内"},
            {"id": "inter-subject", "label": "被试间"},
        ],
        "epochs": 50,
        "seed": 0,
        "batch_size": 1024,
        "data_root": str(DATA_ROOT),
        "stops": [{"id": key, "label": label} for key, label in STOP_LABELS.items()],
    }


def _file_rows(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    return [{"path": str(path), "exists": path.is_file()} for path in paths]


def _shuffle(items: list[str], seed: int) -> None:
    state = seed & 0xFFFFFFFF
    for index in range(len(items) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        swap = state % (index + 1)
        items[index], items[swap] = items[swap], items[index]
