"""Build a 16s SampleSpec manifest from already-cached TRIBE preds. No new generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.fmri.generation.tool import bundle_sample_spec

KNOWN = {
    "apple_01b": "preds/apple_01b_0318065ab5f8b68aa446443c.json",
    "aardvark_01b": "preds/aardvark_01b_ae915c607e5ece6106eec2d5.json",
    "aloe_02s": "preds/aloe_02s_ce47c330a1aaa7201ff23054.json",
}
CONTROL = "control_preds/gray_500x500_80abe8b0e5ca13c43e86422b.json"


def build_image16_panel(
    artifact_root: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write samples.jsonl pointing at existing npy files."""
    root = Path(artifact_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    specs = []
    control_json = root / CONTROL
    control_npy = control_json.with_suffix(".npy")
    control_id = None
    if control_json.is_file() and control_npy.is_file():
        sidecar = json.loads(control_json.read_text(encoding="utf-8"))
        spec = bundle_sample_spec(
            sample_id="gray_500x500",
            npy_path=control_npy,
            sidecar_path=control_json,
            image_path=None,
            sidecar=sidecar,
            control_sample_id=None,
        )
        specs.append(spec)
        control_id = spec.sample_id
    else:
        missing.append(str(control_npy))
    for sample_id, rel in KNOWN.items():
        sidecar_path = root / rel
        npy_path = sidecar_path.with_suffix(".npy")
        if not sidecar_path.is_file() or not npy_path.is_file():
            missing.append(str(npy_path))
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image = Path(sidecar["path"]) if sidecar.get("path") else None
        spec = bundle_sample_spec(
            sample_id=sample_id,
            npy_path=npy_path,
            sidecar_path=sidecar_path,
            image_path=image,
            sidecar=sidecar,
            control_sample_id=control_id,
        )
        specs.append(spec)
    manifest = out / "samples.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(spec.model_dump_json() + "\n")
    return {
        "n_samples": len(specs),
        "manifest": str(manifest),
        "missing": missing,
        "sample_ids": [s.sample_id for s in specs],
        "quality_verdict_enabled": False,
        "note": "N<=3 is engineering_smoke, not a healthy-brain reference.",
    }
