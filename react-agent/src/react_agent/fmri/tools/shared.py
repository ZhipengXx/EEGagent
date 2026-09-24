"""Shared array summaries used by V1.1 domain tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from react_agent.fmri.jsonutil import finite_or_none


def vertex_mask(n_v: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask over vertices; default all True."""
    if valid is None:
        return np.ones(n_v, dtype=bool)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.size != n_v:
        raise ValueError(f"mask length {mask.size} != V={n_v}")
    return mask


def rms_curve(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-frame RMS over equally weighted vertices."""
    arr = np.asarray(data, dtype=np.float64)
    keep = vertex_mask(arr.shape[1], mask)
    sl = arr[:, keep]
    return np.sqrt(np.nanmean(sl**2, axis=1))


def step_rms(curve: np.ndarray) -> np.ndarray:
    """Adjacent-frame absolute differences of a 1-D curve."""
    c = np.asarray(curve, dtype=np.float64)
    if c.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.abs(np.diff(c))


def peak_stats(curve: np.ndarray) -> dict[str, Any]:
    """Peak frame / relative position and step ratios."""
    c = np.asarray(curve, dtype=np.float64)
    steps = step_rms(c)
    peak = int(np.nanargmax(c)) if c.size else None
    med = float(np.median(steps)) if steps.size else 0.0
    mx = float(np.max(steps)) if steps.size else 0.0
    ratio = finite_or_none(mx / med) if med > 1e-12 else None
    return {
        "peak_frame": peak,
        "peak_relative_position": finite_or_none(peak / max(len(c) - 1, 1)) if peak is not None else None,
        "max_step": finite_or_none(mx) if steps.size else None,
        "median_step": finite_or_none(med) if steps.size else None,
        "max_step_over_median": ratio,
        "n_frames": int(c.size),
    }


def event_windows(
    t_len: int,
    *,
    sampling_interval_s: float | None,
    onset_s: float | None,
    duration_s: float | None,
    alignment_ok: bool,
) -> dict[str, Any] | None:
    """Index windows from metadata. Not a global 4/1/7 hardcode."""
    if not alignment_ok or sampling_interval_s is None or onset_s is None:
        return None
    tr = float(sampling_interval_s)
    if tr <= 0:
        return None
    onset_i = int(round(float(onset_s) / tr))
    dur_i = max(1, int(round(float(duration_s or tr) / tr)))
    pre = list(range(0, max(0, min(onset_i, t_len))))
    stim = list(range(max(0, onset_i), max(0, min(onset_i + dur_i, t_len))))
    post = list(range(min(t_len, onset_i + dur_i), t_len))
    return {
        "pre": pre,
        "stimulus": stim,
        "post": post,
        "n_pre": len(pre),
        "n_stimulus": len(stim),
        "n_post": len(post),
    }
