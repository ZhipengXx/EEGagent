"""Attempt bound for a review operation recorded before its current limit."""
from __future__ import annotations

import json
from pathlib import Path

from .api_policy import ATTEMPTS


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def operation_attempt_limit(camp: Path, operation: str, proposed: int = 2) -> int:
    """An old one-attempt parent cannot silently become a two-attempt parent."""
    bounds = [row["max_operation_attempts"] for row in _rows(Path(camp) / ATTEMPTS)
              if row.get("operation_id") == operation]
    return min([proposed, *bounds])
