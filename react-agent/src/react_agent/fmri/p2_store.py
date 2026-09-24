"""P2 asset layout under the EEGagent ``assets`` directory.

Download happens only from explicit CLI. Import / availability never fetch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets"
SCHAEFER_ATLAS_ID = "schaefer400_fsaverage5_lh_rh"
N_HEMI = 10242
N_TOTAL = 20484
N_SCHAEFER = 400
CORTEXMAE_TRAINED_T = 16
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CORTEXMAE_MODEL_ID = "cortex_mae_parcel"

README_TEXT = """# EEGagent P2 assets

Weights, atlases, and frozen embeddings live here — not inside react-agent.

## Layout

- `hf/` — Hugging Face cache (`HF_HOME`) for CortexMAE-P and CLIP
- `atlas/schaefer400_fsaverage5/` — CBIG fsaverage5 Schaefer-400 as LH||RH npy
- `image_embeddings/clip/` — frozen CLIP vectors keyed by image_id

## Commands (from react-agent, `uv run`)

```bash
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
uv run python -m react_agent.fmri.cli cache-embeddings \\
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl
```

## Rules

- Do not silently pad T=12 to CortexMAE's trained T=16.
- Do not use TRIBE `hf_cache_*` as an independent image encoder.
- Embeddings are descriptive. They are not a validity probability.
"""


def asset_root(raw: str | Path | None = None) -> Path:
    """Resolve the P2 asset root."""
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_ASSET_ROOT


def asset_paths(root: str | Path | None = None) -> dict[str, Path]:
    """Return canonical subdirectories."""
    base = asset_root(root)
    return {
        "root": base,
        "hf": base / "hf",
        "schaefer": base / "atlas" / "schaefer400_fsaverage5",
        "embeddings": base / "image_embeddings" / "clip",
        "readme": base / "README.md",
        "markers": base / "markers",
    }


def ensure_layout(root: str | Path | None = None) -> dict[str, Path]:
    """Create empty directories and the README. Does not download."""
    paths = asset_paths(root)
    for key in ("root", "hf", "schaefer", "embeddings", "markers"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if not paths["readme"].is_file():
        paths["readme"].write_text(README_TEXT, encoding="utf-8")
    return paths


def write_marker(paths: dict[str, Path], name: str, payload: dict[str, Any]) -> None:
    """Record that an explicit download finished."""
    paths["markers"].mkdir(parents=True, exist_ok=True)
    target = paths["markers"] / f"{name}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def marker_exists(paths: dict[str, Path], name: str) -> bool:
    return (paths["markers"] / f"{name}.json").is_file()


def set_hf_env(hf_dir: Path) -> None:
    """Point Hugging Face caches at the EEGagent asset root."""
    import os

    hf_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_dir / "transformers"))
