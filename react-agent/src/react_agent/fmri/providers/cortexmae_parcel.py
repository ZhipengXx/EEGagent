"""CortexMAE-P (Schaefer-400) wrapper. Never pads T=12 to T=16."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.p2_store import (
    CORTEXMAE_MODEL_ID,
    CORTEXMAE_TRAINED_T,
    N_SCHAEFER,
    asset_paths,
    marker_exists,
    set_hf_env,
    write_marker,
)
from react_agent.fmri.providers.base import EncoderError
from react_agent.fmri.providers.schaefer import (
    SchaeferError,
    cortexmae_normalize,
    load_schaefer,
    surface_to_schaefer400,
)

_MODEL_CACHE: dict[str, Any] = {}


class CortexMaeParcelProvider:
    """Official parcel model. Flat/volume are not wired."""

    name = "cortex_mae_parcel"
    trained_t = CORTEXMAE_TRAINED_T

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        schaefer_dir: Path | None = None,
        model_id: str = CORTEXMAE_MODEL_ID,
    ) -> None:
        self.paths = asset_paths(asset_root)
        self.schaefer_dir = Path(schaefer_dir) if schaefer_dir else self.paths["schaefer"]
        self.model_id = model_id
        self._model: Any = None
        self._labels: np.ndarray | None = None
        self.aggregation = "unknown"

    def missing_reasons(self) -> list[str]:
        """Local inventory. Does not download."""
        reasons: list[str] = []
        try:
            load_schaefer(self.schaefer_dir)
        except SchaeferError as exc:
            reasons.append(f"schaefer_atlas:{exc}")
        try:
            import torch  # noqa: F401
        except ImportError:
            reasons.append("torch_not_installed")
        if not _cortexmae_importable():
            reasons.append("cortex_mae_package_missing")
        if not marker_exists(self.paths, "cortexmae_parcel") and not _hf_snapshot_present(
            self.paths["hf"]
        ):
            reasons.append("cortexmae_weights_missing")
        return reasons

    def load(self) -> None:
        """Load atlas + official weights from the asset root."""
        missing = self.missing_reasons()
        if missing:
            raise EncoderError("unavailable", "; ".join(missing))
        atlas = load_schaefer(self.schaefer_dir)
        self._labels = atlas["labels"]
        set_hf_env(self.paths["hf"])
        cache_key = f"{self.paths['hf']}:{self.model_id}"
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
        else:
            self._model = _load_official_model(self.model_id)
            _MODEL_CACHE[cache_key] = self._model

    def encode_surface(self, tv: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Convert [T,20484] → parcel embedding. Time is not padded."""
        if self._labels is None or self._model is None:
            self.load()
        assert self._labels is not None
        data = np.asarray(tv, dtype=np.float64)
        t_in = int(data.shape[0])
        if t_in != data.shape[0]:
            raise EncoderError("time_mutated", "time axis changed before encode")
        parcels = surface_to_schaefer400(data, self._labels)
        if parcels.shape[0] != t_in:
            raise EncoderError("padded_time", "adapter must not change T")
        if parcels.shape[1] != N_SCHAEFER:
            raise EncoderError("parcel_width", f"got {parcels.shape[1]} parcels")
        normed = cortexmae_normalize(parcels)
        if normed.shape[0] != t_in:
            raise EncoderError("padded_time", "normalize must not change T")
        try:
            embedding, aggregation = _official_embed(self._model, normed)
        except Exception as exc:  # noqa: BLE001
            if _looks_like_length_error(exc, t_in):
                raise EncoderError(
                    "temporal_length_unsupported",
                    f"official forward rejected T={t_in} (trained_T={self.trained_t}): {exc}",
                ) from exc
            raise EncoderError("forward_failed", str(exc)) from exc
        self.aggregation = aggregation
        meta = {
            "input_t": t_in,
            "trained_t": self.trained_t,
            "n_parcels": N_SCHAEFER,
            "aggregation": aggregation,
            "padded": False,
            "model_id": self.model_id,
        }
        return np.asarray(embedding, dtype=np.float64).reshape(-1), meta

    def encode(self, payload: Any) -> np.ndarray:
        vec, _meta = self.encode_surface(payload)
        return vec


def download_cortexmae(*, asset_root: Path | None = None) -> dict[str, Any]:
    """Explicit weight fetch into EEGagent/assets/hf."""
    from react_agent.fmri.p2_store import ensure_layout

    paths = ensure_layout(asset_root)
    set_hf_env(paths["hf"])
    if not _cortexmae_importable():
        raise EncoderError(
            "cortex_mae_package_missing",
            "install optional extra `p2` (cortex_mae + torch) then re-run prepare-p2",
        )
    model = _load_official_model(CORTEXMAE_MODEL_ID)
    _ = model
    payload = {
        "model_id": CORTEXMAE_MODEL_ID,
        "hf_home": str(paths["hf"]),
        "trained_t": CORTEXMAE_TRAINED_T,
    }
    write_marker(paths, "cortexmae_parcel", payload)
    return payload


def _cortexmae_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("cortex_mae") is not None


def _import_cortexmae_cls() -> Any:
    try:
        from cortex_mae import CortexMAE  # type: ignore[import-not-found]

        return CortexMAE
    except Exception:
        from cortex_mae.model import CortexMAE  # type: ignore[import-not-found]

        return CortexMAE


def _load_official_model(model_id: str) -> Any:
    cls = _import_cortexmae_cls()
    if hasattr(cls, "from_pretrained"):
        model = cls.from_pretrained(model_id)
    else:
        raise EncoderError("no_from_pretrained", "CortexMAE class has no from_pretrained")
    if hasattr(model, "eval"):
        model.eval()
    return model


def _hf_snapshot_present(hf_dir: Path) -> bool:
    hub = Path(hf_dir) / "hub"
    if not hub.is_dir():
        return False
    return any("cortex" in p.name.lower() or "mae" in p.name.lower() for p in hub.iterdir())


def _looks_like_length_error(exc: BaseException, t_in: int) -> bool:
    text = str(exc).lower()
    tokens = ("size", "shape", "length", "time", "temporal", "position", "16", str(t_in))
    return any(tok in text for tok in tokens)


def _official_embed(model: Any, tb: np.ndarray) -> tuple[np.ndarray, str]:
    """Call official encoder.forward_embedding. Do not use run_embedding (it pads)."""
    import torch

    data = np.asarray(tb, dtype=np.float32)
    t_in, n_parcels = data.shape
    # Parcel layout after official Unmask+unpack: [N, C, T, H, W] = [1, 1, T, 400, 1]
    bold = torch.from_numpy(data).view(1, 1, t_in, n_parcels, 1)
    mask = torch.ones_like(bold)
    encoder = model.model.encoder if hasattr(model, "model") else model
    device = next(encoder.parameters()).device
    with torch.no_grad():
        cls_e, _reg_e, patch_e = encoder.forward_embedding(bold.to(device), mask.to(device))
    if cls_e is not None:
        return _as_vector(cls_e), "official_cls"
    if patch_e is None:
        raise EncoderError("no_embedding", "encoder returned no cls or patch tokens")
    return _as_vector(patch_e), "official_patch_mean"


def _as_vector(out: Any) -> np.ndarray:
    import torch

    if isinstance(out, torch.Tensor):
        arr = out.detach().cpu().float().numpy()
    else:
        arr = np.asarray(out, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    # [B, D] or [B, T, D] → mean extra axes except last
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr
