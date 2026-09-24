"""Deterministic episode write plus optional curator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.llm.deepseek import DeepSeekParseError
from react_agent.fmri.memory.repository import SqliteMemoryRepository
from react_agent.fmri.memory.retrieval import retrieve_bundle
from react_agent.fmri.planning.service import complete_json, curator_prompt
from react_agent.fmri.schemas import LMUsage


def open_memory(config: FmriCheckConfig) -> SqliteMemoryRepository:
    return SqliteMemoryRepository(config.memory.path, namespace=config.memory.namespace)


def should_curate(episode: dict[str, Any]) -> bool:
    """Skip curator on ordinary repeat successes."""
    if episode.get("degraded_execution"):
        return True
    if episode.get("screening_decision") in {"flagged", "blocked", "abstain"}:
        return True
    if episode.get("resource_gaps"):
        return True
    if episode.get("plan_failed"):
        return True
    return False


async def maybe_curate(
    backend: Any,
    episode: dict[str, Any],
    *,
    config: FmriCheckConfig,
) -> tuple[list[dict[str, Any]], LMUsage | None]:
    if not config.memory.curator_enabled or backend is None:
        return [], None
    if not should_curate(episode):
        return [], None
    schema = {
        "type": "object",
        "properties": {
            "candidates": {"type": "array"},
            "conflicts": {"type": "array"},
            "no_new_lesson_reason": {"type": "string"},
        },
    }
    user = json.dumps(
        {"episode_record": episode, "LessonCandidate_schema": schema},
        default=str,
    )
    try:
        payload, usage = await complete_json(
            backend,
            system=curator_prompt(config.memory.prompt_version),
            user=user,
            profile="fast",
            role="curator",
        )
    except (DeepSeekParseError, Exception):
        return [], None
    candidates = []
    for raw in payload.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        if not raw.get("supporting_evidence_ids"):
            continue
        raw["verification_basis"] = raw.get("verification_basis") or "unverified"
        candidates.append(raw)
    return candidates, usage


def retrieve_for_sample(
    repo: SqliteMemoryRepository,
    *,
    query: dict[str, Any],
    max_items: int,
    exclude_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    items = repo.retrieve({"namespace": repo.namespace})
    return retrieve_bundle(
        items,
        query=query,
        max_items=max_items,
        exclude_sample_ids=exclude_sample_ids,
    )


def write_episode(
    repo: SqliteMemoryRepository,
    episode: dict[str, Any],
    *,
    out_dir: Path | None = None,
) -> str:
    eid = repo.append_episode(episode)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "episode.json").write_text(
            json.dumps({**episode, "episode_id": eid}, indent=2, default=str),
            encoding="utf-8",
        )
    return eid
