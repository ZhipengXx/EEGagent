"""Identity for one RSI attempt. Cost totals never prove that a phase finished."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.binding import file_sha256


def source_hash(workspace: Path) -> str:
    entry = workspace / "extension" / "eeg_candidate.py"
    if not entry.is_file():
        return ""
    return file_sha256(entry)


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_attempt(workspace: Path, candidate_id: str) -> dict[str, Any]:
    """Keep the same attempt when the worker restarts. A mismatched file starts a new one."""
    path = workspace / "attempt.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict) and data.get("candidate_id") == candidate_id and data.get("attempt_id"):
            data.setdefault("operations", {})
            return data
    data = {"attempt_id": uuid.uuid4().hex[:12], "candidate_id": candidate_id, "operations": {}}
    _atomic(path, data)
    return data


def operation_id(workspace: Path, candidate_id: str, phase: str) -> str:
    """Stable id for one phase inside the current attempt."""
    data = ensure_attempt(workspace, candidate_id)
    operations = data.setdefault("operations", {})
    if not operations.get(phase):
        operations[phase] = uuid.uuid4().hex[:12]
        _atomic(workspace / "attempt.json", data)
    return str(operations[phase])


def result_matches(payload: Any, *, candidate_id: str, attempt_id: str, phase: str, input_hash: str) -> bool:
    """Legacy files without identity are not a finished phase."""
    if not isinstance(payload, dict) or not input_hash:
        return False
    return (
        payload.get("candidate_id") == candidate_id
        and payload.get("attempt_id") == attempt_id
        and payload.get("phase") == phase
        and payload.get("input_hash") == input_hash
    )


def new_call_row(**fields: Any) -> dict[str, Any]:
    """One ledger row. Each invocation gets its own call id."""
    row = {"call_id": uuid.uuid4().hex[:12]}
    row.update(fields)
    return row
