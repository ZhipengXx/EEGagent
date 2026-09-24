"""Surface atlas assets. Download only from an explicit CLI command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ATLAS_ID = "destrieux_fsaverage5_lh_rh"
N_HEMI = 10242
N_TOTAL = 20484
MANIFEST_NAME = "manifest.json"
LABELS_NAME = "labels_lh_rh.npy"
NAMES_NAME = "label_names.json"


class AtlasError(RuntimeError):
    """Atlas missing, wrong identity, or wrong length."""


def atlas_paths(atlas_dir: Path) -> dict[str, Path]:
    """Return expected filenames under an atlas directory."""
    root = Path(atlas_dir)
    return {
        "manifest": root / MANIFEST_NAME,
        "labels": root / LABELS_NAME,
        "names": root / NAMES_NAME,
    }


def load_atlas(atlas_dir: Path) -> dict[str, Any]:
    """Load a previously prepared LH||RH Destrieux map."""
    paths = atlas_paths(atlas_dir)
    if not paths["manifest"].is_file() or not paths["labels"].is_file():
        raise AtlasError(f"atlas not prepared under {atlas_dir}")
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = np.load(paths["labels"], allow_pickle=False)
    if labels.ndim != 1 or labels.shape[0] != N_TOTAL:
        raise AtlasError(
            f"atlas length {getattr(labels, 'shape', None)} != {N_TOTAL}"
        )
    if manifest.get("atlas_id") != ATLAS_ID:
        raise AtlasError(f"unexpected atlas_id {manifest.get('atlas_id')}")
    if manifest.get("hemisphere_order") != ["lh", "rh"]:
        raise AtlasError("atlas hemisphere order is not lh||rh")
    names: dict[str, str] = {}
    if paths["names"].is_file():
        with paths["names"].open("r", encoding="utf-8") as handle:
            names = {str(k): str(v) for k, v in json.load(handle).items()}
    return {
        "manifest": manifest,
        "labels": labels,
        "names": names,
        "fingerprint": manifest.get("labels_sha256") or _sha(paths["labels"]),
    }


def validate_atlas_against_sample(labels: np.ndarray, expected_v: int | None) -> None:
    """Reject same-length but wrong-identity maps when V does not match."""
    if expected_v is not None and int(labels.shape[0]) != int(expected_v):
        raise AtlasError(f"atlas V={labels.shape[0]} != sample V={expected_v}")


def roi_key(hemisphere: str, label_id: int) -> str:
    """Primary key: hemisphere + numeric label, never merge L/R same ids."""
    return f"{hemisphere}:{int(label_id)}"


def hemisphere_for_index(index: int, n_hemi: int = N_HEMI) -> str:
    """TRIBE local layout: LH then RH."""
    return "lh" if index < n_hemi else "rh"


def write_synthetic_atlas(atlas_dir: Path, *, n_labels: int = 8) -> dict[str, Any]:
    """Deterministic LH||RH atlas for unit tests. Not a real parcellation."""
    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(N_TOTAL, dtype=np.int32)
    names: dict[str, str] = {}
    for hemi_i, hemi in enumerate(("lh", "rh")):
        start = hemi_i * N_HEMI
        chunk = N_HEMI // n_labels
        for lab in range(n_labels):
            a = start + lab * chunk
            b = start + (lab + 1) * chunk if lab < n_labels - 1 else start + N_HEMI
            labels[a:b] = lab + 1
            names[roi_key(hemi, lab + 1)] = f"synth_{hemi}_{lab + 1}"
    paths = atlas_paths(atlas_dir)
    np.save(paths["labels"], labels)
    with paths["names"].open("w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    manifest = {
        "atlas_id": ATLAS_ID,
        "source": "synthetic_test",
        "space": "fsaverage5",
        "hemisphere_order": ["lh", "rh"],
        "n_vertices_per_hemisphere": N_HEMI,
        "n_vertices": N_TOTAL,
        "labels_sha256": _sha(paths["labels"]),
        "note": "Synthetic labels for tests. Not a biological atlas.",
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _cached_manifest(atlas_dir: Path) -> dict[str, Any] | None:
    """Return a prepared manifest, or None if missing / synthetic-only."""
    paths = atlas_paths(atlas_dir)
    if not paths["manifest"].is_file() or not paths["labels"].is_file():
        return None
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("source") == "synthetic_test":
        return None
    return manifest


def prepare_destrieux_atlas(atlas_dir: Path, *, allow_download: bool = False) -> dict[str, Any]:
    """Prepare fsaverage5 Destrieux maps as LH||RH labels.

    Download happens only when allow_download is True (CLI prepare-assets).
    A synthetic_test cache is never treated as a real atlas.
    """
    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    paths = atlas_paths(atlas_dir)
    cached = _cached_manifest(atlas_dir)
    if cached is not None:
        return cached
    if not allow_download:
        raise AtlasError(
            "atlas missing; run `python -m react_agent.fmri.cli prepare-assets` "
            "or pass --from-dir with local npy maps"
        )
    try:
        from nilearn.datasets import fetch_atlas_surf_destrieux
    except ImportError as exc:
        raise AtlasError(
            "nilearn is not installed; prepare assets from local files "
            "or `uv add nilearn` then re-run prepare-assets"
        ) from exc
    fetched = fetch_atlas_surf_destrieux()
    left = np.asarray(fetched["map_left"]).astype(np.int32).reshape(-1)
    right = np.asarray(fetched["map_right"]).astype(np.int32).reshape(-1)
    if left.size != N_HEMI or right.size != N_HEMI:
        raise AtlasError(
            f"fetched Destrieux maps have shapes {left.shape}/{right.shape}, "
            f"expected {N_HEMI} per hemisphere"
        )
    labels = np.concatenate([left, right])
    raw_labels = list(fetched.get("labels") or [])
    names: dict[str, str] = {}
    for hemi, hemi_map in (("lh", left), ("rh", right)):
        for lab in np.unique(hemi_map):
            idx = int(lab)
            title = str(raw_labels[idx]) if 0 <= idx < len(raw_labels) else f"label_{idx}"
            names[roi_key(hemi, idx)] = title
    np.save(paths["labels"], labels)
    with paths["names"].open("w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    manifest = {
        "atlas_id": ATLAS_ID,
        "source": "nilearn.fetch_atlas_surf_destrieux",
        "space": "fsaverage5",
        "hemisphere_order": ["lh", "rh"],
        "n_vertices_per_hemisphere": N_HEMI,
        "n_vertices": N_TOTAL,
        "labels_sha256": _sha(paths["labels"]),
        "note": "Structural Destrieux parcels. Not a functional network.",
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def import_local_maps(
    atlas_dir: Path, *, left: Path, right: Path, labels_txt: Path | None = None
) -> dict[str, Any]:
    """Import already-downloaded LH/RH maps without fetching."""
    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)
    lh = np.asarray(np.load(left, allow_pickle=False)).astype(np.int32).reshape(-1)
    rh = np.asarray(np.load(right, allow_pickle=False)).astype(np.int32).reshape(-1)
    if lh.size != N_HEMI or rh.size != N_HEMI:
        raise AtlasError(f"local maps must be length {N_HEMI} each, got {lh.size}/{rh.size}")
    labels = np.concatenate([lh, rh])
    names: dict[str, str] = {}
    raw: list[str] = []
    if labels_txt and Path(labels_txt).is_file():
        raw = Path(labels_txt).read_text(encoding="utf-8").splitlines()
    for hemi, hemi_map in (("lh", lh), ("rh", rh)):
        for lab in np.unique(hemi_map):
            idx = int(lab)
            title = raw[idx] if 0 <= idx < len(raw) else f"label_{idx}"
            names[roi_key(hemi, idx)] = title
    paths = atlas_paths(atlas_dir)
    np.save(paths["labels"], labels)
    with paths["names"].open("w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    manifest = {
        "atlas_id": ATLAS_ID,
        "source": "local_import",
        "space": "fsaverage5",
        "hemisphere_order": ["lh", "rh"],
        "n_vertices_per_hemisphere": N_HEMI,
        "n_vertices": N_TOTAL,
        "labels_sha256": _sha(paths["labels"]),
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
