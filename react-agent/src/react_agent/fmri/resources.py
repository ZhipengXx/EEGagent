"""Resolve registered control / atlas / cohort resources. No free paths from the LM."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.schemas import SampleSpec


@dataclass
class ResolvedControl:
    """A paired gray-control array plus identity."""

    sample_id: str
    path: Path
    array: np.ndarray
    fingerprint: str
    generation_profile_id: str | None
    space_name: str | None
    normalization: str | None
    time_axis: int


class ResourceError(RuntimeError):
    """Resource is missing or incompatible."""


class ResourceResolver:
    """Look up registered IDs from config and SampleSpec.resources."""

    def __init__(self, config: FmriCheckConfig, *, base_dir: Path) -> None:
        self.config = config
        self.base_dir = Path(base_dir)

    def control_id(self, sample: SampleSpec) -> str | None:
        if sample.resources and sample.resources.control_sample_id:
            return sample.resources.control_sample_id
        return self.config.resources.gray_control_id

    def is_gray_control(self, sample: SampleSpec) -> bool:
        if sample.resources and sample.resources.is_gray_control:
            return True
        cid = self.config.resources.gray_control_id
        return bool(cid) and sample.sample_id == cid

    def resolve_control(self, sample: SampleSpec) -> ResolvedControl | None:
        """Load the explicitly bound gray control. Does not guess from names."""
        if self.is_gray_control(sample):
            return None
        path_s = self.config.resources.gray_control_path
        if not path_s:
            return None
        path = Path(path_s)
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        if not path.is_file():
            return None
        array = np.load(path, allow_pickle=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        profile = sample.generation_profile_id
        space = sample.space_name
        norm = sample.normalization
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            import json

            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            proto = meta.get("protocol") or {}
            profile = meta.get("generation_profile_id") or profile
            if (
                proto.get("protocol") == "static_1s_7s"
                and (sample.generation_profile_id or "").endswith("static_1s_7s_tribev2")
            ):
                profile = sample.generation_profile_id
            if proto.get("protocol") == sample.generation_profile_id:
                profile = sample.generation_profile_id
            space = meta.get("mesh") or space
            norm = meta.get("normalization") or norm
            if meta.get("checkpoint") and sample.provenance:
                if sample.provenance.generator and meta.get("checkpoint") != sample.provenance.generator:
                    profile = f"checkpoint_mismatch:{meta.get('checkpoint')}"
        return ResolvedControl(
            sample_id=self.control_id(sample) or path.stem,
            path=path,
            array=np.asarray(array),
            fingerprint=digest,
            generation_profile_id=profile,
            space_name=space,
            normalization=norm,
            time_axis=int(sample.time_axis),
        )

    def atlas_dir(self) -> Path | None:
        raw = self.config.resources.atlas_dir
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_dir() else None

    def cohort_index_path(self) -> Path | None:
        raw = self.config.resources.cohort_index_path
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def asset_root(self) -> Path:
        from react_agent.fmri.p2_store import asset_root

        return asset_root(self.config.resources.asset_root)

    def schaefer_atlas_dir(self) -> Path | None:
        from react_agent.fmri.p2_store import asset_paths

        raw = self.config.resources.schaefer_atlas_dir
        path = Path(raw) if raw else asset_paths(self.asset_root())["schaefer"]
        return path if path.is_dir() else None

    def image_embedding_dir(self) -> Path | None:
        from react_agent.fmri.p2_store import asset_paths

        raw = self.config.resources.image_embedding_dir
        path = Path(raw) if raw else asset_paths(self.asset_root())["embeddings"]
        return path if path.is_dir() else None

    def samples_manifest_path(self) -> Path | None:
        raw = self.config.resources.samples_manifest_path
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_file() else None

    def compatibility_report(
        self, sample: SampleSpec, control: ResolvedControl
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if sample.generation_profile_id != control.generation_profile_id:
            reasons.append("generation_profile_id mismatch")
        if sample.space_name != control.space_name:
            reasons.append("space_name mismatch")
        if sample.normalization != control.normalization:
            reasons.append("normalization mismatch")
        if int(sample.time_axis) != control.time_axis:
            reasons.append("time_axis mismatch")
        if sample.expected_shape and tuple(control.array.shape) != tuple(sample.expected_shape):
            reasons.append(
                f"shape sample={list(sample.expected_shape)} control={list(control.array.shape)}"
            )
        if Path(sample.fmri_path).resolve() == control.path.resolve():
            reasons.append("control_is_self")
        if sample.sample_id == control.sample_id:
            reasons.append("control_is_self")
        return (len(reasons) == 0), reasons


def registered_resource_ids(config: FmriCheckConfig) -> set[str]:
    """IDs the LM may mention in tool_args."""
    ids = {"default"}
    if config.resources.gray_control_id:
        ids.add(config.resources.gray_control_id)
    if config.resources.atlas_id:
        ids.add(config.resources.atlas_id)
    if config.resources.cohort_manifest_id:
        ids.add(config.resources.cohort_manifest_id)
    return ids


def reject_free_path_args(args: dict[str, Any]) -> str | None:
    """Refuse shell / URL / filesystem args from the LM."""
    for key, value in args.items():
        if key in {"code", "shell", "url", "path"}:
            return "illegal_tool_args"
        if isinstance(value, str) and (
            value.startswith("http") or "/" in value or "import " in value
        ):
            return "illegal_tool_args"
    return None
