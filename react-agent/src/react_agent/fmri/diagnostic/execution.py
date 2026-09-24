"""Execution identity so raw/contrast and summary/targeted do not collapse."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def execution_key(
    *,
    tool_id: str,
    tool_version: str,
    signal_mode: str | None,
    input_hash: str | None,
    control_hash: str | None,
    params: dict[str, Any] | None,
    metric_definition_version: str,
) -> str:
    blob = json.dumps(
        {
            "tool_id": tool_id,
            "tool_version": tool_version,
            "signal_mode": signal_mode,
            "input_hash": input_hash,
            "control_hash": control_hash,
            "params": params or {},
            "metric_definition_version": metric_definition_version,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()
