"""Fixed cohort panel index, separate from calibrated reference stats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.data import inspect_array, load_manifest
from react_agent.fmri.jsonutil import jsonable
from react_agent.fmri.schemas import SampleSpec


def _vector_feature(array: np.ndarray) -> np.ndarray:
    """Mean over time, finite-safe. Explicit raw feature, not contrast."""
    data = np.asarray(array, dtype=np.float64)
    return np.nanmean(data, axis=0)


def build_cohort_index(manifest_path: Path, out_path: Path) -> dict[str, Any]:
    """Index a fixed comparison panel. Cost is setup, not per-query."""
    rows: list[dict[str, Any]] = []
    for spec, base in load_manifest(manifest_path):
        handle, info = inspect_array(spec, base)
        if handle is None:
            raise RuntimeError(f"cannot index {spec.sample_id}: {info}")
        data = handle.load()
        feat = _vector_feature(data)
        image_id = None
        if spec.resources and spec.resources.image_id:
            image_id = spec.resources.image_id
        rows.append(
            {
                "sample_id": spec.sample_id,
                "fingerprint": handle.fingerprint,
                "image_id": image_id,
                "is_gray_control": bool(spec.resources and spec.resources.is_gray_control),
                "generation_profile_id": spec.generation_profile_id,
                "feature_mean": float(np.nanmean(feat)),
                "feature_std": float(np.nanstd(feat)),
                "feature": feat.astype(np.float32).tolist(),
            }
        )
    payload = {
        "version": "cohort_index.v1",
        "manifest_id": manifest_path.name,
        "n_samples": len(rows),
        "representation_id": "raw_time_mean_vertices",
        "note": "Descriptive panel, not a healthy-brain distribution.",
        "members": [
            {k: v for k, v in row.items() if k != "feature"} for row in rows
        ],
        "features": {row["sample_id"]: row["feature"] for row in rows},
        "fingerprints": [row["fingerprint"] for row in rows],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle)
    return {"n_samples": len(rows), "path": str(out_path), "representation_id": payload["representation_id"]}


def load_cohort_index(path: Path) -> dict[str, Any]:
    """Read a previously built panel."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or float(np.nanstd(a)) < 1e-12 or float(np.nanstd(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def query_cohort(
    spec: SampleSpec,
    feature: np.ndarray,
    fingerprint: str,
    index: dict[str, Any],
) -> dict[str, Any]:
    """Compare one query to the fixed panel, excluding self."""
    features: dict[str, Any] = index.get("features") or {}
    members = {m["sample_id"]: m for m in index.get("members") or []}
    neighbors: list[dict[str, Any]] = []
    n_compared = 0
    exact_dupes: list[str] = []
    for other_id, raw in features.items():
        if other_id == spec.sample_id:
            continue
        other = np.asarray(raw, dtype=np.float64)
        member = members.get(other_id) or {}
        if member.get("fingerprint") == fingerprint:
            exact_dupes.append(other_id)
        n_compared += 1
        neighbors.append(
            {
                "sample_id": other_id,
                "image_id": member.get("image_id"),
                "cosine": cosine(feature, other),
                "pearson": pearson(feature, other),
                "l2": float(np.linalg.norm(feature - other)),
                "same_image_id": bool(
                    spec.resources
                    and spec.resources.image_id
                    and spec.resources.image_id == member.get("image_id")
                ),
            }
        )
    neighbors.sort(key=lambda row: (-(row["cosine"] or -1.0), row["l2"]))
    sims = [n["cosine"] for n in neighbors if n["cosine"] is not None]
    return {
        "n_compared": n_compared,
        "representation_id": index.get("representation_id"),
        "nearest_neighbors": neighbors[:5],
        "similarity_summary": {
            "cosine_mean": float(np.mean(sims)) if sims else None,
            "cosine_max": float(np.max(sims)) if sims else None,
            "cosine_min": float(np.min(sims)) if sims else None,
        },
        "exact_duplicates": exact_dupes,
        "self_excluded": True,
    }


def feature_fingerprint(feature: np.ndarray) -> str:
    blob = np.asarray(feature, dtype=np.float32).tobytes()
    return hashlib.sha256(blob).hexdigest()
