"""CBIG Schaefer-400 on fsaverage5 as LH||RH labels. No volume projection."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.p2_store import (
    N_HEMI,
    N_SCHAEFER,
    N_TOTAL,
    SCHAEFER_ATLAS_ID,
    asset_paths,
    ensure_layout,
    write_marker,
)

LH_ANNOT_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label/"
    "lh.Schaefer2018_400Parcels_7Networks_order.annot"
)
RH_ANNOT_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage5/label/"
    "rh.Schaefer2018_400Parcels_7Networks_order.annot"
)


class SchaeferError(RuntimeError):
    """Schaefer atlas missing or wrong identity."""


def schaefer_paths(atlas_dir: Path) -> dict[str, Path]:
    root = Path(atlas_dir)
    return {
        "manifest": root / "manifest.json",
        "labels": root / "labels_lh_rh.npy",
        "names": root / "label_names.json",
    }


def load_schaefer(atlas_dir: Path) -> dict[str, Any]:
    """Load a prepared LH||RH Schaefer-400 map."""
    paths = schaefer_paths(atlas_dir)
    if not paths["manifest"].is_file() or not paths["labels"].is_file():
        raise SchaeferError(f"Schaefer-400 atlas not prepared under {atlas_dir}")
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = np.load(paths["labels"], allow_pickle=False)
    if labels.shape != (N_TOTAL,):
        raise SchaeferError(f"Schaefer length {labels.shape} != {N_TOTAL}")
    if manifest.get("atlas_id") != SCHAEFER_ATLAS_ID:
        raise SchaeferError(f"unexpected atlas_id {manifest.get('atlas_id')}")
    names: dict[str, str] = {}
    if paths["names"].is_file():
        with paths["names"].open("r", encoding="utf-8") as handle:
            names = {str(k): str(v) for k, v in json.load(handle).items()}
    return {"manifest": manifest, "labels": labels.astype(np.int32), "names": names}


def surface_to_schaefer400(tv: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Average vertices into 400 parcels. Time length is unchanged (no pad)."""
    data = np.asarray(tv, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != N_TOTAL:
        raise SchaeferError(f"expected [T,{N_TOTAL}], got {data.shape}")
    if labels.shape != (N_TOTAL,):
        raise SchaeferError(f"label length {labels.shape} != {N_TOTAL}")
    out = np.full((data.shape[0], N_SCHAEFER), np.nan, dtype=np.float64)
    for col in range(N_SCHAEFER):
        mask = labels == (col + 1)
        if np.any(mask):
            out[:, col] = np.nanmean(data[:, mask], axis=1)
    return out


def cortexmae_normalize(parcels_tb: np.ndarray) -> np.ndarray:
    """Coordinate (time) then frame (space) z-score, as in CortexMAE."""
    tb = np.asarray(parcels_tb, dtype=np.float64)
    mu = np.nanmean(tb, axis=0, keepdims=True)
    sd = np.nanstd(tb, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    tb = (tb - mu) / sd
    mu_t = np.nanmean(tb, axis=1, keepdims=True)
    sd_t = np.nanstd(tb, axis=1, keepdims=True)
    sd_t = np.where(sd_t < 1e-8, 1.0, sd_t)
    return (tb - mu_t) / sd_t


def write_synthetic_schaefer(atlas_dir: Path) -> dict[str, Any]:
    """Deterministic 400-parcel LH||RH map for unit tests."""
    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(N_TOTAL, dtype=np.int32)
    names: dict[str, str] = {}
    per_hemi = N_SCHAEFER // 2
    chunk = N_HEMI // per_hemi
    for hemi_i, hemi in enumerate(("lh", "rh")):
        start = hemi_i * N_HEMI
        offset = hemi_i * per_hemi
        for lab in range(per_hemi):
            a = start + lab * chunk
            b = start + (lab + 1) * chunk if lab < per_hemi - 1 else start + N_HEMI
            pid = offset + lab + 1
            labels[a:b] = pid
            names[str(pid)] = f"synth_{hemi}_{lab + 1}"
    return _write_atlas(atlas_dir, labels, names, source="synthetic_test")


def prepare_schaefer400(
    atlas_dir: Path | None = None,
    *,
    asset_root: Path | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Prepare CBIG fsaverage5 Schaefer-400. Download only if allow_download."""
    paths = ensure_layout(asset_root)
    dest = Path(atlas_dir) if atlas_dir else paths["schaefer"]
    dest.mkdir(parents=True, exist_ok=True)
    files = schaefer_paths(dest)
    if files["manifest"].is_file() and files["labels"].is_file():
        with files["manifest"].open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("source") != "synthetic_test":
            return cached
    if not allow_download:
        raise SchaeferError(
            "Schaefer-400 missing; run `python -m react_agent.fmri.cli prepare-p2 "
            "--kind schaefer --download`"
        )
    raw_dir = dest / "annot"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lh_path = raw_dir / "lh.Schaefer2018_400Parcels_7Networks_order.annot"
    rh_path = raw_dir / "rh.Schaefer2018_400Parcels_7Networks_order.annot"
    _fetch(LH_ANNOT_URL, lh_path)
    _fetch(RH_ANNOT_URL, rh_path)
    lh, lh_names = _read_annot(lh_path)
    rh, rh_names = _read_annot(rh_path)
    if lh.size != N_HEMI or rh.size != N_HEMI:
        raise SchaeferError(f"annot lengths {lh.size}/{rh.size}, expected {N_HEMI}")
    lh, rh = _harmonize_ids(lh, rh)
    labels = np.concatenate([lh, rh]).astype(np.int32)
    unique = {int(x) for x in np.unique(labels) if int(x) > 0}
    if len(unique) != N_SCHAEFER:
        raise SchaeferError(f"expected {N_SCHAEFER} parcels, got {len(unique)}")
    names = _merge_names(lh_names, rh_names)
    manifest = _write_atlas(
        dest,
        labels,
        names,
        source="cbig.Schaefer2018_400Parcels_7Networks_fsaverage5",
    )
    write_marker(
        paths,
        "schaefer400",
        {"atlas_id": SCHAEFER_ATLAS_ID, "dir": str(dest)},
    )
    return manifest


def _harmonize_ids(lh: np.ndarray, rh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lh_u = {int(x) for x in np.unique(lh) if int(x) > 0}
    rh_u = {int(x) for x in np.unique(rh) if int(x) > 0}
    if lh_u and rh_u and lh_u.isdisjoint(rh_u) and max(lh_u | rh_u) == N_SCHAEFER:
        return lh.astype(np.int32), rh.astype(np.int32)
    # Each hemisphere numbered 1..200 locally → offset RH by 200.
    rh_out = rh.astype(np.int32).copy()
    rh_out[rh_out > 0] += N_SCHAEFER // 2
    return lh.astype(np.int32), rh_out


def _merge_names(lh_names: list[str], rh_names: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for idx, title in enumerate(lh_names):
        if idx == 0:
            continue
        names[str(idx)] = title
    offset = N_SCHAEFER // 2
    for idx, title in enumerate(rh_names):
        if idx == 0:
            continue
        key = str(idx + offset) if idx <= offset else str(idx)
        names.setdefault(key, title)
    return names


def _read_annot(path: Path) -> tuple[np.ndarray, list[str]]:
    from nibabel.freesurfer.io import read_annot

    labels, _ctab, names = read_annot(str(path))
    decoded: list[str] = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(name))
    return np.asarray(labels, dtype=np.int32), decoded


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as handle:
        handle.write(resp.read())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_atlas(
    atlas_dir: Path,
    labels: np.ndarray,
    names: dict[str, str],
    *,
    source: str,
) -> dict[str, Any]:
    files = schaefer_paths(atlas_dir)
    np.save(files["labels"], labels.astype(np.int32))
    with files["names"].open("w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    manifest = {
        "atlas_id": SCHAEFER_ATLAS_ID,
        "source": source,
        "space": "fsaverage5",
        "hemisphere_order": ["lh", "rh"],
        "n_vertices_per_hemisphere": N_HEMI,
        "n_vertices": N_TOTAL,
        "n_parcels": N_SCHAEFER,
        "labels_sha256": _sha(files["labels"]),
        "note": "Structural Schaefer-400 for CortexMAE-P. Not a functional network.",
    }
    with files["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
