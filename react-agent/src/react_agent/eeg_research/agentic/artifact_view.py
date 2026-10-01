"""Small, verified record previews for the research detail panel."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.artifacts import lookup, verify
from react_agent.eeg_research.agentic.handoffs import development_view
from react_agent.eeg_research.agentic.paths import resolve_campaign, safe_name


def _preview(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value[:3000] + (" … [preview shortened]" if len(value) > 3000 else "")
    if depth > 5:
        return "[nested record]"
    if isinstance(value, list):
        rows = [_preview(row, depth + 1) for row in value[:12]]
        return rows + ([f"[{len(value) - 12} more items]"] if len(value) > 12 else [])
    if isinstance(value, dict):
        return {key: _preview(item, depth + 1) for key, item in list(value.items())[:30]}
    return value


def artifact_view(root: Path, campaign: str, artifact_id: str) -> dict[str, Any]:
    camp = resolve_campaign(root, campaign)
    if camp is None or not safe_name(artifact_id):
        return {"ok": False, "error": "artifact_missing"}
    row = lookup(camp, artifact_id)
    if not row:
        return {"ok": False, "error": "artifact_missing"}
    path = Path(row["path"]).resolve()
    if not path.is_relative_to(camp.resolve()) or path.suffix != ".json" or not path.is_file():
        return {"ok": False, "error": "artifact_not_previewable"}
    valid, reason = verify(camp, artifact_id)
    result = {"ok": True, "artifact_id": artifact_id, "kind": row.get("kind"),
              "filename": path.name, "sha256": row.get("sha256"),
              "verification_status": "verified" if valid else "invalid", "reason": None if valid else reason,
              "producer_task_id": row.get("producer_task_id"), "body": None}
    if not valid:
        return result
    with path.open("rb") as handle:
        raw = handle.read(150_001)
    if len(raw) > 150_000:
        result["reason"] = "record_too_large_for_preview"
        return result
    try:
        result["body"] = _preview(development_view(json.loads(raw)))
    except (ValueError, UnicodeDecodeError):
        result["reason"] = "record_unreadable"
    return result
