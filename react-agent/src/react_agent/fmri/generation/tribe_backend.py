"""TRIBE backends. capabilities() never loads the model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from react_agent.fmri.generation.export import export_window
from react_agent.fmri.generation.schemas import (
    CHECKPOINT,
    EXPECTED_T,
    EXPECTED_V,
    GenerationError,
    StimulusProfile,
)
from react_agent.fmri.generation.worker import launch_predict


class TribeBackend(Protocol):
    """Predict a video; close releases any in-process model."""

    name: str

    def capabilities(self) -> dict[str, Any]: ...

    def predict(
        self,
        video_path: Path,
        profile: StimulusProfile,
        *,
        out_npy: Path,
        out_meta: Path,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WorkerBackend:
    """Independent trib ev2 Python via JSON files."""

    name = "worker"

    def __init__(
        self,
        *,
        python_executable: str,
        repo_path: str,
        checkpoint: str = CHECKPOINT,
        cache_folder: str,
        device: str = "cuda",
        timeout_s: float | None = None,
        cuda_visible_devices: str | None = None,
    ) -> None:
        self.python_executable = python_executable
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self.cache_folder = cache_folder
        self.device = device
        self.timeout_s = timeout_s
        self.cuda_visible_devices = cuda_visible_devices

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python_exists": Path(self.python_executable).is_file(),
            "repo_exists": Path(self.repo_path).is_dir(),
            "model_loaded": False,
            "in_process": False,
        }

    def predict(
        self,
        video_path: Path,
        profile: StimulusProfile,
        *,
        out_npy: Path,
        out_meta: Path,
    ) -> dict[str, Any]:
        caps = self.capabilities()
        if not caps["python_exists"]:
            raise GenerationError("tribe_python_missing", self.python_executable)
        if not caps["repo_exists"]:
            raise GenerationError("tribe_repo_missing", self.repo_path)
        raw = launch_predict(
            python_executable=self.python_executable,
            repo_path=self.repo_path,
            video_path=video_path,
            duration_s=profile.total_s,
            onset_s=profile.onset_s,
            checkpoint=self.checkpoint,
            cache_folder=Path(self.cache_folder),
            device=self.device,
            out_npy=out_npy,
            out_meta=out_meta,
            timeout_s=self.timeout_s,
            cuda_visible_devices=self.cuda_visible_devices,
        )
        preds = np.load(raw["preds_path"])
        exported, axis = export_window(preds, raw.get("segments"))
        np.save(out_npy, exported)
        raw["exported_shape"] = list(exported.shape)
        raw["export"] = axis
        raw["temporal_padding_applied"] = False
        out_meta.write_text(
            __import__("json").dumps({**raw, "export": axis}, indent=2),
            encoding="utf-8",
        )
        return raw

    def close(self) -> None:
        return None


class MockBackend:
    """Synthetic [16, 20484] with segment times 0..15. Tests only."""

    name = "mock"

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python_exists": True,
            "repo_exists": True,
            "model_loaded": False,
            "in_process": False,
            "synthetic": True,
        }

    def predict(
        self,
        video_path: Path,
        profile: StimulusProfile,
        *,
        out_npy: Path,
        out_meta: Path,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(abs(hash(str(video_path))) % (2**32))
        preds = rng.normal(0.0, 0.05, size=(EXPECTED_T, EXPECTED_V)).astype(np.float32)
        if "gray" in Path(video_path).name:
            preds *= 0.2
        segments = [{"start": float(i), "duration": 1.0} for i in range(EXPECTED_T)]
        exported, axis = export_window(preds, segments)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, exported)
        payload = {
            "status": "ok",
            "preds_path": str(out_npy),
            "shape": list(exported.shape),
            "segments": segments,
            "device": "mock",
            "synthetic": True,
            "export": axis,
            "temporal_padding_applied": False,
            "video_path": str(video_path),
        }
        out_meta.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
        return payload

    def close(self) -> None:
        return None


class InProcessBackend:
    """Reserved. This build does not load TribeModel in react-agent."""

    name = "inprocess"

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "implemented": False,
            "model_loaded": False,
            "reason": "react-agent torch does not match the NVIDIA driver",
        }

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise GenerationError(
            "inprocess_reserved",
            "in-process TribeModel is reserved; use generation.backend=worker",
        )

    def close(self) -> None:
        return None


def make_backend(name: str, **kwargs: Any) -> TribeBackend:
    if name == "mock":
        return MockBackend()
    if name == "worker":
        return WorkerBackend(**kwargs)
    if name == "inprocess":
        return InProcessBackend()
    raise GenerationError("unknown_backend", name)
