"""Parent source snapshots. parent_id is code lineage, not the experimental control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.binding import file_sha256

BASELINE_ENTRY = Path(__file__).resolve().parent / "baseline.py"


class LineageError(ValueError):
    """Raised when a parent snapshot cannot be materialized."""


def parent_entry(camp: Path, parent_id: str | None) -> tuple[Path, str]:
    """Resolve the parent source file. Missing parents are an error."""
    name = parent_id or "baseline"
    if name == "baseline":
        return BASELINE_ENTRY, "baseline"
    entry = camp / "candidates" / name / "extension" / "eeg_candidate.py"
    if not entry.is_file():
        raise LineageError(f"parent_source_missing:{name}")
    return entry, name


def materialize(camp: Path, workspace: Path, experiment: dict[str, Any] | None) -> dict[str, Any]:
    """Copy the parent snapshot into reference/parent.py and record lineage.json."""
    experiment = experiment or {}
    source, parent_id = parent_entry(camp, experiment.get("parent_candidate_id"))
    digest = file_sha256(source)
    reference = workspace / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "parent.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "parent_candidate_id": parent_id,
        "parent_source_hash": digest,
        "control_id": experiment.get("control_id"),
        "hypothesis_id": experiment.get("hypothesis_id") or experiment.get("id"),
        "parent_is_not_control": True,
    }
    (workspace / "lineage.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
