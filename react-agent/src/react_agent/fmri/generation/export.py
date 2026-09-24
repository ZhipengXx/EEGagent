"""Select [0, 16) rows by segment time. Never pad or slice preds[:16] without time."""

from __future__ import annotations

from typing import Any

import numpy as np

from react_agent.fmri.generation.schemas import (
    EXPECTED_T,
    EXPECTED_V,
    TemporalContractError,
)


def _segment_start(segment: Any) -> float:
    if isinstance(segment, dict):
        return float(segment["start"])
    return float(getattr(segment, "start"))


def export_window(
    preds: np.ndarray,
    segments: list[Any] | None,
    *,
    window_start: float = 0.0,
    window_end: float = 16.0,
    expected_t: int = EXPECTED_T,
    expected_v: int = EXPECTED_V,
    atol: float = 0.051,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep rows whose segment.start is in [window_start, window_end)."""
    array = np.asarray(preds)
    if array.ndim != 2:
        raise TemporalContractError(f"preds must be 2D, got {array.shape}")
    if segments is None:
        raise TemporalContractError(
            "segments required; refusing preds[:16] without a time basis"
        )
    if len(segments) != int(array.shape[0]):
        raise TemporalContractError(
            f"segment count {len(segments)} != preds rows {array.shape[0]}"
        )
    starts = [_segment_start(item) for item in segments]
    selected = [
        i
        for i, start in enumerate(starts)
        if (start + atol) >= window_start and start < window_end
    ]
    if len(selected) != expected_t:
        raise TemporalContractError(
            f"window [{window_start}, {window_end}) selected {len(selected)} "
            f"rows; contracted T={expected_t}. No pad/interpolate."
        )
    selected_starts = [starts[i] for i in selected]
    expected = list(np.arange(expected_t, dtype=float) + window_start)
    if not np.allclose(selected_starts, expected, atol=atol):
        raise TemporalContractError(
            f"segment starts {selected_starts} != expected {expected}"
        )
    out = np.asarray(array[selected], dtype=np.float32)
    if out.shape != (expected_t, expected_v):
        raise TemporalContractError(
            f"exported shape {list(out.shape)} != [{expected_t}, {expected_v}]"
        )
    return out, {
        "t_video": [float(x) for x in selected_starts],
        "n_selected": int(expected_t),
        "window": [window_start, window_end],
        "temporal_padding_applied": False,
        "export_rule_id": "segment_time_window_0_16_exclusive_v1",
    }
