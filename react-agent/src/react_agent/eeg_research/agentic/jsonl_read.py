"""Bounded JSONL reads for the workbench. Incomplete tails are not scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_JSONL_BYTES = 2_000_000
MAX_JSONL_LINES = 4000


def read_jsonl(
    path: Path,
    *,
    max_bytes: int = MAX_JSONL_BYTES,
    max_lines: int = MAX_JSONL_LINES,
) -> dict[str, Any]:
    """Return dict rows plus diagnostics. Missing files are empty, not errors."""
    if not path.is_file():
        return {"rows": [], "diagnostics": [], "empty": True, "truncated": False, "source": str(path.name)}
    size = path.stat().st_size
    truncated = size > max_bytes
    with path.open("rb") as handle:
        raw = handle.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    diagnostics: list[dict[str, Any]] = []
    if truncated and lines:
        lines = lines[:-1]
        diagnostics.append({"kind": "truncated_read", "max_bytes": max_bytes, "file_bytes": size})
    elif size and not text.endswith("\n") and lines:
        incomplete = lines.pop()
        if incomplete.strip():
            diagnostics.append({"kind": "incomplete_tail", "excerpt": incomplete[:200]})
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines[:max_lines]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append({"kind": "bad_line", "index": index, "excerpt": line[:200]})
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            diagnostics.append({"kind": "bad_line", "index": index, "excerpt": line[:200]})
    if len(lines) > max_lines:
        diagnostics.append({"kind": "truncated_lines", "max_lines": max_lines})
        truncated = True
    return {
        "rows": rows,
        "diagnostics": diagnostics,
        "empty": not rows,
        "truncated": truncated,
        "source": path.name,
    }
