"""Campaign list order for the workbench. No API or training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from react_agent.eeg_research.agentic.loop import create_campaign
from react_agent.eeg_research.agentic.view import agentic_status, campaign_view


def _open(root: Path, goal_id: str) -> Path:
    create_campaign(
        root,
        goal={"goal_id": goal_id, "max_training_jobs": 1, "max_llm_calls": 1, "max_gpu_seconds": 1},
        contract={"fingerprint": "fp", "research_scope": "pooled_subject_retrieval"},
        request_id=goal_id,
    )
    return root / goal_id


def test_recently_updated_campaign_is_listed_first(tmp_path: Path) -> None:
    older = _open(tmp_path, "older")
    newer = _open(tmp_path, "newer")
    os.utime(older / "campaign_state.json", (1_700_000_000, 1_700_000_000))
    os.utime(newer / "campaign_state.json", (1_800_000_000, 1_800_000_000))

    rows = agentic_status(tmp_path)["campaigns"]
    assert [row["campaign_id"] for row in rows] == ["newer", "older"]
    created = json.loads((newer / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rows[0]["created_at"] == created["at"]
    assert rows[0]["updated_at"] == (newer / "campaign_state.json").stat().st_mtime

    (older / "events.jsonl").unlink()
    fallback = campaign_view(older)
    assert fallback["created_at"] == (older / "campaign_state.json").stat().st_mtime
    assert fallback["updated_at"] == fallback["created_at"]
