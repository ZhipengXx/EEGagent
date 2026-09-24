"""Read-only sample loading, path resolution, and fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.schemas import SampleSpec

_FINGERPRINT_CACHE: dict[str, str] = {}


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path relative to the manifest/sample file directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_sample_spec(path: Path) -> SampleSpec:
    """Load a SampleSpec JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SampleSpec.model_validate(payload)


def load_manifest(path: Path) -> list[tuple[SampleSpec, Path]]:
    """Load JSONL samples. Paths resolve against the manifest directory."""
    samples: list[tuple[SampleSpec, Path]] = []
    base = path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            spec = SampleSpec.model_validate(payload)
            samples.append((spec, base))
            _ = line_no
    return samples


def file_fingerprint(path: Path) -> str:
    """SHA256 of a file, cached per path string for the process."""
    key = str(path.resolve())
    cached = _FINGERPRINT_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FINGERPRINT_CACHE[key] = value
    return value


def safe_sample_id(sample_id: str, fingerprint: str) -> str:
    """Build a filesystem-safe identifier that survives duplicate IDs."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    slug = slug[:64]
    return f"{slug}_{fingerprint[:8]}"


@dataclass
class ArrayHandle:
    """Local array reference stored in state; data stays on disk."""

    path: str
    kind: str
    array_key: str | None
    time_axis: int
    shape_tv: tuple[int, int]
    dtype: str
    fingerprint: str

    def load(self) -> np.ndarray:
        """Load a [T, V] numeric array without pickles."""
        path = Path(self.path)
        if self.kind == "npy":
            raw = np.load(path, allow_pickle=False, mmap_mode="r")
        elif self.kind == "npz":
            if not self.array_key:
                raise ValueError("array_key is required for .npz")
            with np.load(path, allow_pickle=False) as bundle:
                if self.array_key not in bundle.files:
                    raise KeyError(
                        f"array_key {self.array_key!r} not in npz "
                        f"keys={list(bundle.files)}"
                    )
                raw = np.array(bundle[self.array_key], copy=True)
        else:
            raise ValueError(f"unsupported array kind {self.kind}")
        if self.time_axis == 1:
            data = np.asarray(raw).T
        else:
            data = np.asarray(raw)
        return data


def inspect_array(spec: SampleSpec, base_dir: Path) -> tuple[ArrayHandle | None, dict[str, Any]]:
    """Open metadata for an array without guessing axes.

    Returns (handle, error_info). handle is None when the file cannot be used.
    """
    info: dict[str, Any] = {}
    fmri_path = resolve_path(spec.fmri_path, base_dir)
    info["resolved_path"] = str(fmri_path)
    if not fmri_path.exists():
        info["error"] = "file_not_found"
        return None, info
    suffix = fmri_path.suffix.lower()
    try:
        if suffix == ".npy":
            raw = np.load(fmri_path, allow_pickle=False, mmap_mode="r")
            kind = "npy"
            key = None
        elif suffix == ".npz":
            if not spec.array_key:
                info["error"] = "missing_array_key"
                return None, info
            with np.load(fmri_path, allow_pickle=False) as bundle:
                names = list(bundle.files)
                if spec.array_key not in names:
                    info["error"] = "array_key_not_found"
                    info["npz_keys"] = names
                    return None, info
                raw = np.array(bundle[spec.array_key], copy=True)
            kind = "npz"
            key = spec.array_key
        else:
            info["error"] = "unsupported_suffix"
            info["suffix"] = suffix
            return None, info
    except (OSError, ValueError) as exc:
        info["error"] = "undecodable"
        info["exception_type"] = type(exc).__name__
        return None, info

    info["ndim"] = int(raw.ndim)
    info["shape"] = [int(s) for s in raw.shape]
    info["dtype"] = str(raw.dtype)
    if raw.ndim != 2:
        info["error"] = "not_2d"
        return None, info
    if raw.size == 0 or min(raw.shape) == 0:
        info["error"] = "empty_axis"
        return None, info
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        info["error"] = "non_numeric"
        return None, info

    if spec.time_axis == 0:
        shape_tv = (int(raw.shape[0]), int(raw.shape[1]))
    else:
        shape_tv = (int(raw.shape[1]), int(raw.shape[0]))

    if spec.expected_shape is not None and tuple(spec.expected_shape) != tuple(raw.shape):
        info["error"] = "shape_contract_mismatch"
        info["expected_shape"] = list(spec.expected_shape)
        return None, info
    if spec.expected_n_vertices is not None and shape_tv[1] != spec.expected_n_vertices:
        info["error"] = "vertex_contract_mismatch"
        info["expected_n_vertices"] = spec.expected_n_vertices
        info["n_vertices"] = shape_tv[1]
        return None, info

    fingerprint = file_fingerprint(fmri_path)
    handle = ArrayHandle(
        path=str(fmri_path),
        kind=kind,
        array_key=key,
        time_axis=spec.time_axis,
        shape_tv=shape_tv,
        dtype=str(raw.dtype),
        fingerprint=fingerprint,
    )
    return handle, info


def load_optional_json(path_str: str | None, base_dir: Path) -> dict[str, Any] | None:
    """Load sidecar metadata if present."""
    if not path_str:
        return None
    path = resolve_path(path_str, base_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {"_value": payload}
