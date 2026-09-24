"""R / A / S definitions. Old max_step fields are not replaced."""

from __future__ import annotations

from typing import Any

import numpy as np

from react_agent.fmri.jsonutil import finite_or_none

METRIC_VERSION = "ras_v1"
DENOM_EPS = 1e-8


def _finite_series(values: np.ndarray) -> list[float | None]:
    return [finite_or_none(float(x)) for x in np.asarray(values, dtype=np.float64)]


def _peak_transition(curve: np.ndarray) -> dict[str, Any]:
    """Peak of a transition series. Index i means frame i -> i+1."""
    c = np.asarray(curve, dtype=np.float64)
    if c.size == 0 or not np.isfinite(c).any():
        return {"status": "undefined", "reason": "empty_or_nonfinite"}
    idx = int(np.nanargmax(c))
    value = finite_or_none(float(c[idx]))
    if value is None:
        return {"status": "undefined", "reason": "nonfinite_peak"}
    return {
        "status": "defined",
        "from_frame": idx,
        "to_frame": idx + 1,
        "value": value,
    }


def contrast_series(delta: np.ndarray) -> dict[str, Any]:
    """Describe D = Y - G without changing the legacy RMS step fields.

    R[t] = sqrt(mean_v D[t]^2)
    A[t] = |R[t+1] - R[t]|
    S[t] = sqrt(mean_v (D[t+1] - D[t])^2)
    """
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        return {
            "metric_definition_version": METRIC_VERSION,
            "status": "undefined",
            "reason": "nonfinite_or_bad_shape",
        }
    r_curve = np.sqrt(np.mean(arr**2, axis=1))
    if r_curve.size < 2:
        return {
            "metric_definition_version": METRIC_VERSION,
            "status": "undefined",
            "reason": "too_short",
            "R": _finite_series(r_curve),
        }
    a_curve = np.abs(np.diff(r_curve))
    s_curve = np.sqrt(np.mean(np.diff(arr, axis=0) ** 2, axis=1))
    med_a = float(np.median(a_curve)) if a_curve.size else 0.0
    max_a = float(np.max(a_curve)) if a_curve.size else 0.0
    if med_a <= DENOM_EPS:
        ratio_status = "denominator_unstable" if med_a > 0 else "undefined"
        ratio = None
    else:
        ratio_status = "defined"
        ratio = finite_or_none(max_a / med_a)
    return {
        "metric_definition_version": METRIC_VERSION,
        "status": "defined",
        "signal_mode": "contrast",
        "aggregation": "equal_vertex_mean",
        "R": _finite_series(r_curve),
        "A": _finite_series(a_curve),
        "S": _finite_series(s_curve),
        "max_R": _peak_frame(r_curve),
        "max_A": {**_peak_transition(a_curve), "amplitude": finite_or_none(max_a), "median": finite_or_none(med_a)},
        "max_S": _peak_transition(s_curve),
        "max_A_over_median": ratio,
        "max_A_over_median_status": ratio_status,
        "max_A_amplitude": finite_or_none(max_a),
        "median_A": finite_or_none(med_a),
    }


def _peak_frame(curve: np.ndarray) -> dict[str, Any]:
    c = np.asarray(curve, dtype=np.float64)
    if c.size == 0 or not np.isfinite(c).any():
        return {"status": "undefined", "reason": "empty_or_nonfinite"}
    idx = int(np.nanargmax(c))
    return {"status": "defined", "frame": idx, "value": finite_or_none(float(c[idx]))}


def image16_windows(n_time: int, sampling_interval_s: float | None) -> dict[str, Any] | None:
    """Half-open image16 windows when the axis is 1s frames over [0, 16)."""
    if n_time != 16 or sampling_interval_s is None:
        return None
    if abs(float(sampling_interval_s) - 1.0) > 1e-6:
        return None
    return {
        "pre": [0.0, 4.0],
        "stimulus": [4.0, 5.0],
        "post": [5.0, 16.0],
        "pre_frames": list(range(0, 4)),
        "stimulus_frames": [4],
        "post_frames": list(range(5, 16)),
        "interval": "half_open",
    }


def transition_energy(delta: np.ndarray, from_frame: int) -> dict[str, Any]:
    """Per-vertex squared change E[v] = (D[k+1,v] - D[k,v])^2."""
    arr = np.asarray(delta, dtype=np.float64)
    if from_frame < 0 or from_frame + 1 >= arr.shape[0]:
        return {"status": "undefined", "reason": "transition_out_of_range"}
    diff = arr[from_frame + 1] - arr[from_frame]
    energy = diff**2
    total = float(np.sum(energy))
    if not np.isfinite(total):
        return {"status": "undefined", "reason": "nonfinite"}
    if total <= DENOM_EPS:
        return {
            "status": "undefined" if total == 0 else "denominator_unstable",
            "reason": "zero_or_tiny_energy",
            "from_frame": from_frame,
            "to_frame": from_frame + 1,
            "total_energy": finite_or_none(total),
        }
    order = np.argsort(energy)[::-1]
    k = max(1, int(np.ceil(0.01 * energy.size)))
    top = order[:k]
    return {
        "status": "defined",
        "from_frame": from_frame,
        "to_frame": from_frame + 1,
        "total_energy": finite_or_none(total),
        "top_fraction": 0.01,
        "top_count": int(k),
        "top_energy_fraction": finite_or_none(float(np.sum(energy[top]) / total)),
        "top_vertices": [int(i) for i in top[:32]],
        "abs_delta_quantiles": {
            "p50": finite_or_none(float(np.quantile(np.abs(diff), 0.5))),
            "p95": finite_or_none(float(np.quantile(np.abs(diff), 0.95))),
            "p99": finite_or_none(float(np.quantile(np.abs(diff), 0.99))),
        },
        "signed_mean_delta": finite_or_none(float(np.mean(diff))),
        "rms_delta": finite_or_none(float(np.sqrt(np.mean(diff**2)))),
    }
