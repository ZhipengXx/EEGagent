"""Deterministic synthetic demo fixtures marked demo_synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from react_agent.fmri.schemas import SampleSpec

SEED = 42
PROFILE = "static_image_profile_v1"


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _spec(**kwargs) -> dict:
    base = {
        "time_axis": 0,
        "sampling_interval_s": 1.0,
        "spatial_representation": "surface",
        "space_name": "demo_surface",
        "normalization": "demo_z",
        "generation_profile_id": PROFILE,
    }
    base.update(kwargs)
    return base


def make_demo(out: Path) -> None:
    """Create tiny synthetic arrays, manifests, and an ROI map."""
    rng = np.random.default_rng(SEED)
    arrays = out / "arrays"
    meta = out / "metadata"
    arrays.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    t, v = 12, 16
    normal = rng.normal(0, 1, size=(t, v)).astype(np.float32)
    # slow temporal structure
    normal += np.linspace(-0.2, 0.2, t)[:, None]

    nan_arr = normal.copy()
    nan_arr[2, 3] = np.nan
    nan_arr[4, 1] = np.inf

    constant = np.ones((t, v), dtype=np.float32) * 0.5

    spike = rng.normal(0, 0.05, size=(t, v)).astype(np.float32)
    spike[6] += 10.0

    missing_tr = normal.copy()

    # time_axis=1 variant of a healthy array
    transposed = np.ascontiguousarray(normal.T)

    roi = np.repeat(np.arange(4), v // 4).astype(np.int32)
    np.save(out / "roi_labels.npy", roi)
    names = {str(i): f"roi_{i}" for i in range(4)}
    (out / "roi_names.json").write_text(json.dumps(names), encoding="utf-8")

    samples = [
        ("ok_numeric", normal, _spec(roi_map_path="roi_labels.npy", roi_names_path="roi_names.json")),
        ("nan_inf", nan_arr, _spec()),
        ("constant", constant, _spec()),
        ("temporal_spike", spike, _spec()),
        (
            "missing_tr",
            missing_tr,
            _spec(sampling_interval_s=None, spatial_representation="unknown", space_name=None),
        ),
        (
            "time_axis1",
            transposed,
            _spec(time_axis=1, fmri_note="axis1"),
        ),
    ]

    sample_rows = []
    ref_rows = []
    for name, array, fields in samples:
        npy = arrays / f"{name}.npy"
        _write_npy(npy, array)
        spec_dict = _spec()
        spec_dict.update(fields)
        spec_dict.pop("fmri_note", None)
        spec_dict.update(
            {
                "sample_id": name,
                "fmri_path": f"arrays/{name}.npy",
                "metadata_path": f"metadata/{name}.json",
            }
        )
        SampleSpec.model_validate(spec_dict)
        json.dump(spec_dict, (meta / f"{name}.json").open("w", encoding="utf-8"), indent=2)
        json.dump(spec_dict, (out / f"{name}.json").open("w", encoding="utf-8"), indent=2)
        sample_rows.append(spec_dict)
        if name in {"ok_numeric", "time_axis1"}:
            ref_rows.append(spec_dict)

    # incompatible reference: different generation profile
    incompat = rng.normal(0, 1, size=(t, v)).astype(np.float32)
    _write_npy(arrays / "incompat.npy", incompat)
    incompat_spec = _spec(
        sample_id="incompat_ref",
        fmri_path="arrays/incompat.npy",
        generation_profile_id="other_profile",
        normalization="raw",
    )
    json.dump(incompat_spec, (out / "incompat_ref.json").open("w", encoding="utf-8"), indent=2)

    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row) + "\n")
    with (out / "reference.jsonl").open("w", encoding="utf-8") as handle:
        for row in ref_rows:
            handle.write(json.dumps(row) + "\n")
    with (out / "reference_incompat.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(incompat_spec) + "\n")
        handle.write(json.dumps(incompat_spec | {"sample_id": "incompat_ref_2", "fmri_path": "arrays/incompat.npy"}) + "\n")
        handle.write(json.dumps(incompat_spec | {"sample_id": "incompat_ref_3", "fmri_path": "arrays/incompat.npy"}) + "\n")

    npz_path = arrays / "ok.npz"
    np.savez(npz_path, fmri=normal, extra=np.zeros(3))
    npz_spec = _spec(
        sample_id="ok_npz",
        fmri_path="arrays/ok.npz",
        array_key="fmri",
    )
    json.dump(npz_spec, (out / "ok_npz.json").open("w", encoding="utf-8"), indent=2)

    (out / "README.txt").write_text(
        "demo_synthetic fixtures. Not real fMRI. Seed=42.\n",
        encoding="utf-8",
    )
