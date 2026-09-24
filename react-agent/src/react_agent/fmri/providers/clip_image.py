"""Independent frozen CLIP. Not TRIBE's visual backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.p2_store import (
    CLIP_MODEL_ID,
    asset_paths,
    marker_exists,
    set_hf_env,
    write_marker,
)
from react_agent.fmri.providers.base import EncoderError


class ClipImageProvider:
    """openai/clip-vit-base-patch32 cached under EEGagent/assets."""

    name = "clip"

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        embedding_dir: Path | None = None,
        model_id: str = CLIP_MODEL_ID,
    ) -> None:
        self.paths = asset_paths(asset_root)
        self.embedding_dir = Path(embedding_dir) if embedding_dir else self.paths["embeddings"]
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None

    def missing_reasons(self) -> list[str]:
        reasons: list[str] = []
        try:
            import transformers  # noqa: F401
        except ImportError:
            reasons.append("transformers_not_installed")
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            reasons.append("pillow_not_installed")
        if not marker_exists(self.paths, "clip") and not _clip_snapshot_present(self.paths["hf"]):
            reasons.append("clip_weights_missing")
        return reasons

    def cache_path(self, image_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in image_id)
        return self.embedding_dir / f"{safe}.npy"

    def load_cached(self, image_id: str) -> np.ndarray | None:
        path = self.cache_path(image_id)
        if not path.is_file():
            return None
        return np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)

    def save_cached(self, image_id: str, vector: np.ndarray) -> Path:
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(image_id)
        np.save(path, np.asarray(vector, dtype=np.float32))
        return path

    def load(self) -> None:
        missing = [r for r in self.missing_reasons() if r != "clip_weights_missing"]
        if missing:
            raise EncoderError("unavailable", "; ".join(missing))
        set_hf_env(self.paths["hf"])
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model.eval()

    def encode_image(self, image_path: Path) -> np.ndarray:
        if self._model is None or self._processor is None:
            self.load()
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vec = self._model.get_image_features(**inputs)
        if hasattr(vec, "image_embeds"):
            vec = vec.image_embeds
        elif hasattr(vec, "pooler_output"):
            vec = vec.pooler_output
        elif hasattr(vec, "last_hidden_state"):
            vec = vec.last_hidden_state.mean(dim=1)
        arr = vec.detach().cpu().float().numpy().reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        return arr.astype(np.float64)

    def encode(self, payload: Any) -> np.ndarray:
        return self.encode_image(Path(payload))


def download_clip(*, asset_root: Path | None = None) -> dict[str, Any]:
    """Explicit CLIP weight fetch into EEGagent/assets/hf."""
    from react_agent.fmri.p2_store import ensure_layout

    paths = ensure_layout(asset_root)
    set_hf_env(paths["hf"])
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise EncoderError(
            "transformers_not_installed",
            "install optional extra `p2` then re-run prepare-p2 --kind clip",
        ) from exc
    CLIPModel.from_pretrained(CLIP_MODEL_ID)
    CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    payload = {"model_id": CLIP_MODEL_ID, "hf_home": str(paths["hf"])}
    write_marker(paths, "clip", payload)
    return payload


def cache_manifest_embeddings(
    manifest_path: Path,
    *,
    asset_root: Path | None = None,
    embedding_dir: Path | None = None,
) -> dict[str, Any]:
    """Encode THINGS jpgs listed in a SampleSpec JSONL. Skip rows without images."""
    from react_agent.fmri.data import load_manifest

    provider = ClipImageProvider(asset_root=asset_root, embedding_dir=embedding_dir)
    provider.embedding_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    skipped = 0
    for spec, _base in load_manifest(manifest_path):
        image_id = None
        if spec.resources and spec.resources.image_id:
            image_id = spec.resources.image_id
        if not spec.image_path or not image_id:
            skipped += 1
            continue
        path = Path(spec.image_path)
        if not path.is_file():
            skipped += 1
            continue
        if provider.cache_path(image_id).is_file():
            continue
        vec = provider.encode_image(path)
        provider.save_cached(image_id, vec)
        wrote += 1
    return {
        "wrote": wrote,
        "skipped": skipped,
        "dir": str(provider.embedding_dir),
        "model_id": provider.model_id,
        "note": "Independent CLIP cache. Not TRIBE hf_cache.",
    }


def _clip_snapshot_present(hf_dir: Path) -> bool:
    hub = Path(hf_dir) / "hub"
    if not hub.is_dir():
        return False
    return any("clip" in p.name.lower() for p in hub.iterdir())
