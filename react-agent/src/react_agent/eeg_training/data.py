"""Load preprocessed trials and an existing RN50 feature cache."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from react_agent.eeg_training.protocol import FULL_EEG_CHANNELS, SplitError, SplitPlan


def _alias_numpy_core() -> None:
    if "numpy._core" in sys.modules:
        return
    sys.modules["numpy._core"] = np.core
    for name in ("multiarray", "umath", "numerictypes", "_multiarray_umath"):
        module = getattr(np.core, name, None)
        if module is not None:
            sys.modules[f"numpy._core.{name}"] = module


def load_feature_cache(path: Path) -> dict[str, torch.Tensor]:
    """Read img_features from a legacy cache. Missing files are not rebuilt."""
    _alias_numpy_core()
    if not path.is_file():
        raise SplitError(f"feature_cache_missing:{path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    features = saved.get("img_features") if isinstance(saved, dict) else None
    if not isinstance(features, dict) or not features:
        raise SplitError(f"feature_cache_invalid:{path}")
    return features


def load_trials(path: Path, channels: list[str] | None) -> dict[str, object]:
    """Average repeats and keep the selected channels."""
    _alias_numpy_core()
    if not path.is_file():
        raise SplitError(f"trial_file_missing:{path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    eeg = loaded["eeg"]
    if not torch.is_tensor(eeg):
        eeg = torch.from_numpy(np.asarray(eeg))
    if channels:
        index = [FULL_EEG_CHANNELS.index(name) for name in channels]
        eeg = eeg[:, :, index]
    eeg = eeg.float().mean(dim=1)
    return {
        "eeg": eeg,
        "img": _column(loaded["img"]),
        "label": _column(loaded["label"]),
    }


def _column(value: object) -> list[str]:
    if torch.is_tensor(value):
        rows = value[:, 0].tolist() if value.ndim > 1 else value.tolist()
    else:
        array = np.asarray(value)
        rows = array[:, 0].tolist() if array.ndim > 1 else array.tolist()
    return [str(item) for item in rows]


class RetrievalTrials(Dataset):
    """Trials whose image id is in the allowed set."""

    def __init__(
        self,
        records: list[dict[str, torch.Tensor | str]],
        timesteps: list[int],
    ) -> None:
        self.records = records
        self.timesteps = timesteps

    def __len__(self) -> int:
        """Return the number of kept images."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Return one window, its cached image embedding, and the image id for duplicate stats.

        image_id is runtime metadata. It must not be concatenated into the encoder input.
        """
        row = self.records[index]
        start, end = self.timesteps
        eeg = row["eeg"]
        assert isinstance(eeg, torch.Tensor)
        features = row["img_features"]
        assert isinstance(features, torch.Tensor)
        image_id = str(row.get("img") or "")
        subject = str(row.get("subject") or "")
        query_id = str(row.get("query_id") or "")
        if not query_id:
            query_id = f"{subject}::{image_id}" if subject else image_id
        return {
            "eeg": eeg[:, start:end].float(),
            "img_features": features.float(),
            "image_id": image_id,
            "query_id": query_id,
            "subject": subject,
        }


def collate_retrieval(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    """Keep image_id as strings so default_collate never tensorizes them into the encoder."""
    return {
        "eeg": torch.stack([row["eeg"] for row in batch]),  # type: ignore[arg-type]
        "img_features": torch.stack([row["img_features"] for row in batch]),  # type: ignore[arg-type]
        "image_id": [str(row.get("image_id") or "") for row in batch],
        "query_id": [str(row.get("query_id") or "") for row in batch],
        "subject": [str(row.get("subject") or "") for row in batch],
    }


def _subject_from_path(path: Path) -> str:
    name = path.parent.name
    if name.startswith("sub-"):
        return name
    return name or "custom"


def collect_records(
    files: tuple[Path, ...],
    features: dict[str, torch.Tensor],
    channels: list[str] | None,
    allowed: set[str] | None,
) -> tuple[list[dict[str, torch.Tensor | str]], list[str]]:
    """Join trial files to cached features. allowed=None keeps every image in the files.

    Query identity is subject plus image. Subject is metadata for the evaluator, not model input.
    """
    records: list[dict[str, torch.Tensor | str]] = []
    image_ids: list[str] = []
    for path in files:
        trials = load_trials(path, channels)
        images = trials["img"]
        eeg = trials["eeg"]
        assert isinstance(images, list)
        assert isinstance(eeg, torch.Tensor)
        subject = _subject_from_path(path)
        for index, image_id in enumerate(images):
            if allowed is not None and image_id not in allowed:
                continue
            if image_id not in features:
                raise SplitError(f"feature_missing:{image_id}")
            records.append(
                {
                    "eeg": eeg[index],
                    "img": image_id,
                    "img_features": features[image_id],
                    "subject": subject,
                    "query_id": f"{subject}::{image_id}",
                }
            )
            image_ids.append(image_id)
    return records, image_ids


def collect_validation_records(
    plan: SplitPlan,
    channels: list[str] | None,
    *,
    validation_image_ids: set[str] | None = None,
    features: dict[str, torch.Tensor] | None = None,
) -> tuple[list[dict[str, torch.Tensor | str]], list[str]]:
    """Share the trainer's cache/file/filter choice without reading final holdout."""
    allowed: set[str] | None
    if plan.val_mode == "other_subjects_test":
        expected_caches, cache_index, allowed = 2, 1, None
    elif plan.val_mode in {
        "train_holdout",
        "all_subjects_holdout",
        "custom_train_holdout",
    }:
        expected_caches, cache_index, allowed = 1, 0, validation_image_ids
        if not allowed:
            raise SplitError("frozen_validation_image_identity_missing")
    else:
        raise SplitError("unsupported_validation_mode:" + plan.val_mode)
    if not plan.feature_caches:
        raise SplitError("validation_feature_cache_missing")
    if len(plan.feature_caches) != expected_caches:
        raise SplitError("validation_cache_mode_mismatch:" + plan.val_mode)
    if not plan.val_files:
        raise SplitError("validation_source_files_missing")
    forbidden = {path.resolve() for path in plan.forbidden_files}
    if forbidden & {path.resolve() for path in plan.val_files}:
        raise SplitError("development_holdout_overlap")
    selected = (
        features
        if features is not None
        else load_feature_cache(plan.feature_caches[cache_index])
    )
    return collect_records(plan.val_files, selected, channels, allowed)


def collect_frozen_validation_records(
    plan: SplitPlan, channels: list[str] | None, identity: dict[str, Any]
) -> tuple[list[dict[str, torch.Tensor | str]], list[str]]:
    """Verify complete frozen query/gallery identities using only validation files."""

    def required_ids(field: str) -> list[str]:
        values = identity.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise SplitError("frozen_" + field + "_missing")
        return values

    image_ids = required_ids("validation_image_ids")
    query_ids = required_ids("validation_query_ids")
    gallery_ids = required_ids("gallery_image_ids")
    if identity.get("val_mode") != plan.val_mode:
        raise SplitError("frozen_validation_mode_mismatch")
    if len(image_ids) != len(set(image_ids)) or Counter(gallery_ids) != Counter(
        set(image_ids)
    ):
        raise SplitError("frozen_validation_gallery_mismatch")
    positives = identity.get("positive_map")
    if not isinstance(positives, dict) or any(
        query not in positives for query in query_ids
    ):
        raise SplitError("frozen_validation_positive_identity_missing")
    if plan.val_mode == "other_subjects_test":
        files = identity.get("validation_files")
        if (
            not isinstance(files, list)
            or not files
            or any(not isinstance(name, str) or not name for name in files)
            or [Path(name).resolve() for name in files]
            != [path.resolve() for path in plan.val_files]
        ):
            raise SplitError("frozen_validation_source_mismatch")
    records, images = collect_validation_records(
        plan, channels, validation_image_ids=set(image_ids)
    )
    if Counter(str(row["query_id"]) for row in records) != Counter(query_ids):
        raise SplitError("frozen_validation_query_mismatch")
    if set(images) != set(image_ids):
        raise SplitError("frozen_validation_image_mismatch")
    if any(positives.get(str(row["query_id"])) != str(row["img"]) for row in records):
        raise SplitError("frozen_validation_positive_mismatch")
    # Keep the trainer's file/trial order and first-seen frozen_bank gallery order.
    return records, images
