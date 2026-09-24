"""Convert TRIBE v2 sidecars to SampleSpec without copying prediction arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from react_agent.fmri.schemas import Provenance, SampleResources, SampleSpec, StimulusEvent

GENERATION_PROFILE_ID = "static_1s_7s_tribev2"
EXPECTED_SHAPE = (12, 20484)
EXPECTED_N_VERTICES = 20484
DEFAULT_TIME_AXIS = 0
DEFAULT_TR_S = 1.0
DEFAULT_NORMALIZATION = "none_raw_signed"
DEFAULT_MESH = "fsaverage5"
DEFAULT_GENERATOR = "facebook/tribev2"

DERIVED_MARKERS = ("_stim_plus5", "_stim_4to6_mean", "_minus_gray")

_DEFAULT_ALIGNMENT = (
    "preds[k] aligns with t_video[k]. Training FmriExtractor.offset=5s is already "
    "in the model; the checker records this mapping and does not shift the series again. "
    "A 5s post-stimulus peak is not a hard rule."
)


def is_derived_pred_id(sample_id: str) -> bool:
    """True for sliced TRIBE products, not the full 12×V series."""
    return any(marker in sample_id for marker in DERIVED_MARKERS)


def _load_sidecar(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar is not an object: {path}")
    return payload


def _pred_npy_path(sidecar_path: Path, sidecar: dict[str, Any]) -> Path:
    raw = sidecar.get("pred_path")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return sidecar_path.with_suffix(".npy")


def _sampling_interval_s(sidecar: dict[str, Any]) -> float:
    t_video = sidecar.get("t_video")
    if isinstance(t_video, list) and len(t_video) >= 2:
        try:
            step = float(t_video[1]) - float(t_video[0])
        except (TypeError, ValueError):
            step = DEFAULT_TR_S
        if step > 0:
            return step
    raw = sidecar.get("sampling_interval_s")
    if isinstance(raw, (int, float)) and float(raw) > 0:
        return float(raw)
    return DEFAULT_TR_S


def _expected_shape(sidecar: dict[str, Any]) -> tuple[int, int]:
    for key in ("preds_shape",):
        shape = sidecar.get(key)
        if isinstance(shape, list) and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    stats = sidecar.get("stats")
    if isinstance(stats, dict):
        shape = stats.get("shape")
        if isinstance(shape, list) and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return EXPECTED_SHAPE


def _image_path(sidecar: dict[str, Any]) -> str | None:
    raw = sidecar.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    suffix = Path(raw).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return raw
    return None


def _stimulus_events(sidecar: dict[str, Any], sample_id: str) -> list[StimulusEvent]:
    protocol = sidecar.get("protocol") or {}
    onset = protocol.get("onset_s")
    duration = protocol.get("image_s")
    label = sidecar.get("filename") or sidecar.get("category") or sample_id
    if sidecar.get("kind") == "gray_control":
        label = "gray_control"
        duration = protocol.get("total_s", duration)
    return [
        StimulusEvent(
            onset_s=float(onset) if isinstance(onset, (int, float)) else 4.0,
            duration_s=float(duration) if isinstance(duration, (int, float)) else 1.0,
            label=str(label),
        )
    ]


def sidecar_to_sample_spec(
    sidecar_path: Path,
    *,
    pred_path: Path | None = None,
    control_sample_id: str | None = None,
    cohort_manifest_id: str | None = None,
) -> SampleSpec:
    """Map one TRIBE sidecar + raw npy path to a SampleSpec."""
    sidecar_path = sidecar_path.resolve()
    sidecar = _load_sidecar(sidecar_path)
    sample_id = sidecar_path.stem
    if is_derived_pred_id(sample_id):
        raise ValueError(
            f"{sample_id} is a derived TRIBE slice; use the raw [12, 20484] pred"
        )

    npy = Path(pred_path) if pred_path is not None else _pred_npy_path(sidecar_path, sidecar)
    npy = npy.resolve()
    if not npy.is_file():
        raise FileNotFoundError(f"prediction npy missing: {npy}")

    shape = _expected_shape(sidecar)
    mesh = sidecar.get("mesh") or DEFAULT_MESH
    normalization = sidecar.get("normalization") or DEFAULT_NORMALIZATION
    generator = sidecar.get("checkpoint") or DEFAULT_GENERATOR
    protocol_name = (sidecar.get("protocol") or {}).get("protocol") or "static_1s_7s"
    mapping = sidecar.get("mapping")
    alignment = mapping if isinstance(mapping, str) and mapping else _DEFAULT_ALIGNMENT
    is_control = sidecar.get("kind") == "gray_control"
    unverified = sidecar.get("time_axis_unverified")
    if unverified is True:
        time_status: str = "unverified"
    elif unverified is False and sidecar.get("t_video"):
        time_status = "verified"
    else:
        time_status = "inferred"
    resources = SampleResources(
        control_sample_id=None if is_control else control_sample_id,
        control_artifact_id=None if is_control else control_sample_id,
        is_gray_control=is_control,
        spatial_layout={
            "hemisphere_order": ["lh", "rh"],
            "n_vertices_per_hemisphere": 10242,
            "n_vertices_total": int(sidecar.get("n_vertices_expected") or shape[1] or EXPECTED_N_VERTICES),
            "status": "inferred_from_local_tribev2_utils_fmri",
            "source": "tribev2/utils_fmri.py SurfaceProjector LH||RH",
        },
        mesh_asset_id=f"mesh:{mesh}",
        time_alignment_status=time_status,  # type: ignore[arg-type]
        t_video=[float(x) for x in sidecar["t_video"]] if isinstance(sidecar.get("t_video"), list) else None,
        image_id=str(sidecar.get("filename") or sample_id),
        scaling_source="unknown",
        cohort_manifest_id=cohort_manifest_id,
    )

    return SampleSpec(
        sample_id=sample_id,
        fmri_path=str(npy),
        time_axis=DEFAULT_TIME_AXIS,
        sampling_interval_s=_sampling_interval_s(sidecar),
        spatial_representation="surface",
        space_name=str(mesh),
        normalization=str(normalization),
        generation_profile_id=str(
            sidecar.get("generation_profile_id")
            or (sidecar.get("protocol") or {}).get("generation_profile_id")
            or GENERATION_PROFILE_ID
        ),
        metadata_path=str(sidecar_path),
        image_path=_image_path(sidecar),
        stimulus_events=_stimulus_events(sidecar, sample_id),
        output_time_origin="t_video",
        alignment_description=alignment,
        provenance=Provenance(
            generator=str(generator),
            config_summary=f"{protocol_name} mesh={mesh} shape={list(shape)}",
        ),
        expected_shape=shape,
        expected_n_vertices=int(sidecar.get("n_vertices_expected") or shape[1] or EXPECTED_N_VERTICES),
        resources=resources,
    )


def write_samples_jsonl(specs: Iterable[SampleSpec], out_path: Path) -> Path:
    """Write SampleSpec rows; paths stay absolute and point at original npy files."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [spec.model_dump(mode="json") for spec in specs]
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def convert_ids(
    preds_dir: Path,
    sample_ids: Iterable[str],
    *,
    control_sample_id: str | None = None,
    cohort_manifest_id: str | None = None,
) -> list[SampleSpec]:
    """Convert an explicit list of raw TRIBE pred ids."""
    specs: list[SampleSpec] = []
    for sample_id in sample_ids:
        if is_derived_pred_id(sample_id):
            raise ValueError(f"refusing derived pred id: {sample_id}")
        sidecar = preds_dir / f"{sample_id}.json"
        if not sidecar.is_file():
            raise FileNotFoundError(f"sidecar missing: {sidecar}")
        specs.append(
            sidecar_to_sample_spec(
                sidecar,
                control_sample_id=control_sample_id,
                cohort_manifest_id=cohort_manifest_id,
            )
        )
    return specs
