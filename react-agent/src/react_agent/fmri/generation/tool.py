"""tribev2_generate_fmri: materialize 16s preds + matching gray control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.fmri.budget import BudgetLedger
from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.generation.export import export_window
from react_agent.fmri.generation.schemas import (
    ALIGNMENT,
    CHECKPOINT,
    EXPECTED_T,
    EXPECTED_V,
    EXPORT_RULE_ID,
    GENERATE_SPEC,
    MISSING_AUDIO,
    MISSING_TEXT,
    PROFILE_ID,
    GeneratedFmriBundle,
    GenerationError,
    ImageRequest,
    StimulusProfile,
    TemporalContractError,
)
from react_agent.fmri.generation.tribe_backend import TribeBackend, make_backend
from react_agent.fmri.generation.video import (
    control_cache_key,
    load_rgb,
    write_gray_video,
    write_image_video,
)
from react_agent.fmri.progress import emit
from react_agent.fmri.schemas import Provenance, SampleResources, SampleSpec, StimulusEvent


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(str(Path(path).stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def image_cache_key(
    image_path: Path,
    profile: StimulusProfile,
    *,
    checkpoint: str,
    missing_audio: str,
    missing_text: str,
    export_rule_id: str,
) -> str:
    payload = {
        "image_sha256": file_fingerprint(image_path),
        "profile_id": profile.profile_id,
        "fps": profile.fps,
        "pre_gray_s": profile.pre_gray_s,
        "image_s": profile.image_s,
        "post_gray_s": profile.post_gray_s,
        "gray_rgb": list(profile.gray_rgb),
        "codec": profile.codec,
        "checkpoint": checkpoint,
        "missing_audio": missing_audio,
        "missing_text": missing_text,
        "export_rule_id": export_rule_id,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def default_sample_id(image_path: Path) -> str:
    return Path(image_path).stem


class TribeGenerateTool:
    """Generator kind. Registered beside the check registry, never a check candidate."""

    spec = GENERATE_SPEC

    def __init__(
        self,
        config: FmriCheckConfig,
        *,
        backend: TribeBackend | None = None,
        budget: BudgetLedger | None = None,
    ) -> None:
        self.config = config
        self.budget = budget
        gen = config.generation
        self.backend = backend or make_backend(
            gen.backend,
            python_executable=gen.python_executable,
            repo_path=gen.repo_path,
            checkpoint=gen.checkpoint,
            cache_folder=gen.cache_folder,
            device=gen.device,
            timeout_s=config.budgets.generation_walltime_s,
            cuda_visible_devices=gen.cuda_visible_devices,
        )

    def capabilities(self) -> dict[str, Any]:
        return self.backend.capabilities()

    def close(self) -> None:
        self.backend.close()

    def run(self, request: ImageRequest) -> GeneratedFmriBundle:
        gen = self.config.generation
        image_path = Path(request.image_path).resolve()
        if not image_path.is_file():
            raise GenerationError("image_missing", str(image_path))
        profile = request.profile
        if profile.post_gray_s != 11.0 or profile.fps != 10 or int(round(profile.total_s)) != 16:
            raise GenerationError(
                "profile_mismatch",
                f"expected 4+1+11 at fps=10; got post={profile.post_gray_s} fps={profile.fps}",
            )
        sample_id = request.sample_id or default_sample_id(image_path)
        root = Path(gen.artifact_root)
        pred_key = image_cache_key(
            image_path,
            profile,
            checkpoint=gen.checkpoint,
            missing_audio=gen.missing_audio,
            missing_text=gen.missing_text,
            export_rule_id=gen.export_rule_id,
        )
        image = load_rgb(image_path)
        size_hw = (int(image.shape[0]), int(image.shape[1]))
        ctrl_key = control_cache_key(
            size_hw,
            profile,
            checkpoint=gen.checkpoint,
            missing_audio=gen.missing_audio,
            missing_text=gen.missing_text,
            export_rule_id=gen.export_rule_id,
        )
        img_npy = root / "preds" / f"{sample_id}_{pred_key}.npy"
        img_meta = root / "preds" / f"{sample_id}_{pred_key}.json"
        img_video = root / "videos" / f"{sample_id}_{pred_key}.mp4"
        ctrl_id = f"gray_{size_hw[0]}x{size_hw[1]}_{ctrl_key}"
        ctrl_npy = root / "control_preds" / f"{ctrl_id}.npy"
        ctrl_meta = root / "control_preds" / f"{ctrl_id}.json"
        ctrl_video = root / "control_videos" / f"{ctrl_id}.mp4"

        emit({"type": "generate", "message": f"start {sample_id}"})
        hit_image = img_npy.is_file() and img_meta.is_file()
        hit_control = (not request.include_control) or (ctrl_npy.is_file() and ctrl_meta.is_file())
        needed = (0 if hit_image else 1) + (0 if hit_control else 1)
        if self.budget is not None:
            self.budget.preflight_generation(needed)

        new_preds = 0
        mock = self.backend.name == "mock"
        if not hit_image:
            emit({"type": "generate", "message": "writing image video"})
            _ensure_video(
                img_video,
                image_path=image_path,
                size_hw=size_hw,
                profile=profile,
                gray_only=False,
                mock=mock,
                python_executable=gen.python_executable,
                repo_path=gen.repo_path,
            )
            emit({"type": "generate", "message": "TRIBE worker running (image)"})
            raw = self.backend.predict(img_video, profile, out_npy=img_npy, out_meta=img_meta)
            emit(
                {
                    "type": "generate",
                    "message": f"TRIBE worker done (image) shape={list(raw.get('exported_shape') or raw.get('shape') or [])}",
                }
            )
            _write_sidecar(
                img_meta,
                sample_id=sample_id,
                image_path=image_path,
                video_path=img_video,
                npy_path=img_npy,
                profile=profile,
                size_hw=size_hw,
                raw=raw,
                kind="image",
                control_video=str(ctrl_video) if request.include_control else None,
                checkpoint=gen.checkpoint,
                missing_audio=gen.missing_audio,
                missing_text=gen.missing_text,
            )
            new_preds += 1
        else:
            emit({"type": "generate", "message": "image cache hit"})
            assert_cached_contract(img_npy, img_meta)
        if request.include_control and not hit_control:
            emit({"type": "generate", "message": "writing control video"})
            _ensure_video(
                ctrl_video,
                image_path=None,
                size_hw=size_hw,
                profile=profile,
                gray_only=True,
                mock=mock,
                python_executable=gen.python_executable,
                repo_path=gen.repo_path,
            )
            emit({"type": "generate", "message": "TRIBE worker running (control)"})
            raw = self.backend.predict(ctrl_video, profile, out_npy=ctrl_npy, out_meta=ctrl_meta)
            emit(
                {
                    "type": "generate",
                    "message": f"TRIBE worker done (control) shape={list(raw.get('exported_shape') or raw.get('shape') or [])}",
                }
            )
            _write_sidecar(
                ctrl_meta,
                sample_id=ctrl_id,
                image_path=None,
                video_path=ctrl_video,
                npy_path=ctrl_npy,
                profile=profile,
                size_hw=size_hw,
                raw=raw,
                kind="gray_control",
                control_video=str(ctrl_video),
                checkpoint=gen.checkpoint,
                missing_audio=gen.missing_audio,
                missing_text=gen.missing_text,
            )
            new_preds += 1
        elif request.include_control:
            emit({"type": "generate", "message": "control cache hit"})
            assert_cached_contract(ctrl_npy, ctrl_meta)
        if self.budget is not None:
            self.budget.record_generation(new_preds)
        emit(
            {
                "type": "generate",
                "message": (
                    f"done {sample_id} new_predictions={new_preds} "
                    f"cache_image={hit_image} cache_control={hit_control}"
                ),
            }
        )

        sidecar = json.loads(img_meta.read_text(encoding="utf-8"))
        if sidecar.get("temporal_padding_applied"):
            raise GenerationError("padded_export", "cached sidecar marked temporal padding")
        spec = bundle_sample_spec(
            sample_id=sample_id,
            npy_path=img_npy,
            sidecar_path=img_meta,
            image_path=image_path,
            sidecar=sidecar,
            control_sample_id=ctrl_id if request.include_control else None,
        )
        return GeneratedFmriBundle(
            sample_id=sample_id,
            profile_id=profile.profile_id,
            sample=spec,
            preds_path=str(img_npy),
            sidecar_path=str(img_meta),
            video_path=str(img_video if img_video.is_file() else sidecar.get("video_path") or img_video),
            control_sample_id=ctrl_id if request.include_control else None,
            control_preds_path=str(ctrl_npy) if request.include_control else None,
            control_sidecar_path=str(ctrl_meta) if request.include_control else None,
            control_video_path=str(ctrl_video) if request.include_control else None,
            preds_shape=tuple(np_shape(img_npy)),
            temporal_padding_applied=False,
            cache_hit_image=hit_image,
            cache_hit_control=hit_control,
            new_tribe_predictions=new_preds,
            synthetic=self.backend.name == "mock",
            backend=self.backend.name if self.backend.name in {"worker", "mock"} else "worker",
        )


def _ensure_video(
    video_path: Path,
    *,
    image_path: Path | None,
    size_hw: tuple[int, int],
    profile: StimulusProfile,
    gray_only: bool,
    mock: bool,
    python_executable: str,
    repo_path: str,
) -> None:
    if video_path.is_file():
        return
    if mock:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"mock-video")
        return
    if gray_only:
        write_gray_video(
            size_hw,
            video_path,
            profile,
            python_executable=python_executable,
            repo_path=repo_path,
        )
        return
    if image_path is None:
        raise GenerationError("image_missing", "image video requested without image_path")
    write_image_video(
        image_path,
        video_path,
        profile,
        python_executable=python_executable,
        repo_path=repo_path,
    )


def np_shape(path: Path) -> tuple[int, int]:
    import numpy as np

    array = np.load(path, mmap_mode="r")
    return int(array.shape[0]), int(array.shape[1])


def bundle_sample_spec(
    *,
    sample_id: str,
    npy_path: Path,
    sidecar_path: Path,
    image_path: Path | None,
    sidecar: dict[str, Any],
    control_sample_id: str | None,
) -> SampleSpec:
    t_video = sidecar.get("t_video") or (sidecar.get("export") or {}).get("t_video")
    return SampleSpec(
        sample_id=sample_id,
        fmri_path=str(npy_path.resolve()),
        time_axis=0,
        sampling_interval_s=1.0,
        spatial_representation="surface",
        space_name="fsaverage5",
        normalization="none_raw_signed",
        generation_profile_id=PROFILE_ID,
        metadata_path=str(sidecar_path.resolve()),
        image_path=str(image_path) if image_path else None,
        stimulus_events=[
            StimulusEvent(
                onset_s=4.0,
                duration_s=1.0 if sidecar.get("kind") != "gray_control" else 16.0,
                label=sample_id if sidecar.get("kind") != "gray_control" else "gray_control",
            )
        ],
        output_time_origin="t_video",
        alignment_description=ALIGNMENT,
        provenance=Provenance(
            generator=str(sidecar.get("checkpoint") or CHECKPOINT),
            config_summary=f"{PROFILE_ID} shape={[EXPECTED_T, EXPECTED_V]}",
        ),
        expected_shape=(EXPECTED_T, EXPECTED_V),
        expected_n_vertices=EXPECTED_V,
        resources=SampleResources(
            control_sample_id=control_sample_id,
            control_artifact_id=control_sample_id,
            is_gray_control=sidecar.get("kind") == "gray_control",
            spatial_layout={
                "hemisphere_order": ["lh", "rh"],
                "n_vertices_per_hemisphere": 10242,
                "n_vertices_total": EXPECTED_V,
            },
            mesh_asset_id="mesh:fsaverage5",
            time_alignment_status="verified" if t_video else "inferred",
            t_video=[float(x) for x in t_video] if isinstance(t_video, list) else None,
            image_id=Path(image_path).name if image_path else sample_id,
            scaling_source="unknown",
        ),
    )


def _write_sidecar(
    path: Path,
    *,
    sample_id: str,
    image_path: Path | None,
    video_path: Path,
    npy_path: Path,
    profile: StimulusProfile,
    size_hw: tuple[int, int],
    raw: dict[str, Any],
    kind: str,
    control_video: str | None,
    checkpoint: str,
    missing_audio: str,
    missing_text: str,
) -> None:
    export = raw.get("export") or {}
    t_video = export.get("t_video") or [float(s["start"]) for s in raw.get("segments") or []]
    payload = {
        "sample_id": sample_id,
        "kind": kind,
        "filename": image_path.name if image_path else sample_id,
        "path": str(image_path) if image_path else None,
        "generation_profile_id": profile.profile_id,
        "protocol": profile.to_dict(),
        "video_path": str(video_path),
        "pred_path": str(npy_path),
        "checkpoint": checkpoint,
        "mesh": "fsaverage5",
        "n_vertices_expected": EXPECTED_V,
        "missing_modalities": {"audio": missing_audio, "text": missing_text},
        "normalization": "none_raw_signed",
        "image_size_hw": list(size_hw),
        "preds_shape": list(raw.get("exported_shape") or raw.get("shape") or [EXPECTED_T, EXPECTED_V]),
        "t_video": t_video,
        "time_axis_unverified": False,
        "mapping": ALIGNMENT,
        "temporal_padding_applied": False,
        "export_rule_id": EXPORT_RULE_ID,
        "gray_control_video": control_video,
        "segments": raw.get("segments"),
        "device": raw.get("device"),
        "synthetic": bool(raw.get("synthetic")),
        "backend": raw.get("device") and ("mock" if raw.get("synthetic") else "worker"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_cached_contract(npy_path: Path, sidecar_path: Path) -> None:
    """Re-check a cache hit; refuse a 12-point array masquerading as 16."""
    import numpy as np

    preds = np.load(npy_path)
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    segs = sidecar.get("segments")
    if segs:
        export_window(preds, segs)
    elif tuple(preds.shape) != (EXPECTED_T, EXPECTED_V):
        raise TemporalContractError(
            f"cached preds {list(preds.shape)} != [{EXPECTED_T},{EXPECTED_V}]"
        )
    if sidecar.get("temporal_padding_applied"):
        raise TemporalContractError("cached artifact used temporal padding")
