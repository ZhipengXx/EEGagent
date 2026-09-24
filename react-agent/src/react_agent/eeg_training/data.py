"""Load preprocessed trials and an existing RN50 feature cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from react_agent.eeg_training.protocol import FULL_EEG_CHANNELS, SplitError


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one window and its cached image embedding."""
        row = self.records[index]
        start, end = self.timesteps
        eeg = row["eeg"]
        assert isinstance(eeg, torch.Tensor)
        features = row["img_features"]
        assert isinstance(features, torch.Tensor)
        return {"eeg": eeg[:, start:end].float(), "img_features": features.float()}


def collect_records(
    files: tuple[Path, ...],
    features: dict[str, torch.Tensor],
    channels: list[str] | None,
    allowed: set[str] | None,
) -> tuple[list[dict[str, torch.Tensor | str]], list[str]]:
    """Join trial files to cached features. allowed=None keeps every image in the files."""
    records: list[dict[str, torch.Tensor | str]] = []
    image_ids: list[str] = []
    for path in files:
        trials = load_trials(path, channels)
        images = trials["img"]
        eeg = trials["eeg"]
        assert isinstance(images, list)
        assert isinstance(eeg, torch.Tensor)
        for index, image_id in enumerate(images):
            if allowed is not None and image_id not in allowed:
                continue
            if image_id not in features:
                raise SplitError(f"feature_missing:{image_id}")
            records.append({"eeg": eeg[index], "img": image_id, "img_features": features[image_id]})
            image_ids.append(image_id)
    return records, image_ids
