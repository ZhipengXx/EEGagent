"""JSON helpers that never emit NaN/Inf as numbers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite_or_none(value: Any) -> float | int | None:
    """Convert a numeric value to JSON-safe form, else None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    return None


def jsonable(value: Any) -> Any:
    """Recursively convert values into JSON-serializable Python objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.dtype):
        return str(value)
    return str(value)
