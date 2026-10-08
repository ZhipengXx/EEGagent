"""Standard-library Linux process identity shared by jobs and GPU watchdogs."""
from __future__ import annotations

from pathlib import Path


def _stat_fields(pid: int) -> list[str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return text.rsplit(")", 1)[1].split()


def _proc_start(pid: int) -> str | None:
    fields = _stat_fields(pid)
    if fields is None or len(fields) <= 19:
        return None
    return fields[19]


def _process_state(pid: int) -> str | None:
    fields = _stat_fields(pid)
    if not fields:
        return None
    return fields[0]
