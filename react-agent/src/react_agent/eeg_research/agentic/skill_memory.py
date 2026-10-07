"""Procedural hints derived only from completed, verified campaign development tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from react_agent.eeg_research.agentic.embedding import canonical, content_hash
from react_agent.eeg_research.agentic.schemas import LessonProposal


class SkillProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    description_en: str = Field(min_length=1, max_length=1000)
    trigger_en: str = Field(min_length=1, max_length=1000)
    procedure: list[str] = Field(min_length=1, max_length=30)
    preconditions: dict[str, Any] = Field(default_factory=dict)
    failure_modes: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    source_task_refs: list[str] = Field(default_factory=list)
    supporting_episode_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    summary_zh: str = ""


class SkillLessonProposal(LessonProposal):
    proposed_skills: list[SkillProposal] = Field(default_factory=list)


def artifact_source(camp: Path, artifact_id: str) -> dict:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.semantic_memory import read_json, unsafe_source, unsafe_source_path

    row = resolve_verified_artifact(camp, artifact_id)
    path = Path(row["path"])
    if (not path.resolve().is_relative_to(camp.resolve()) or unsafe_source(read_json(path))
            or unsafe_source_path(path)):
        raise ValueError("artifact_not_campaign_development")
    return {"artifact_id": artifact_id, "sha256": row["sha256"],
            "producer_task_id": row.get("producer_task_id"), "kind": row.get("kind")}


def task_source(camp: Path, task_id: str) -> dict:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.task_ledger import latest, reusable
    from react_agent.eeg_research.agentic.semantic_memory import read_json, unsafe_source

    task = latest(camp, task_id)
    if not task or task.get("status") != "completed" or task.get("role") not in {"result_analyst", "experiment_designer", "candidate_coder"}:
        raise ValueError("source_task_not_completed_development")
    artifact = artifact_source(camp, task["artifact_id"])
    registered = resolve_verified_artifact(camp, task["artifact_id"])
    payload = read_json(Path(registered["path"]))
    if (payload.get("status") != "completed" or unsafe_source(payload)
            or artifact["producer_task_id"] != task_id
            or not reusable(payload, task_id=task_id, attempt_id=task["attempt_id"], input_digest=task["input_digest"], candidate_id=task.get("candidate_id"))):
        raise ValueError("source_task_identity_mismatch")
    return {"task_id": task_id, "attempt_id": task["attempt_id"], "input_digest": task["input_digest"],
            "role": task["role"], "status": "completed", "output_artifact": artifact}


def curation_extension(camp: Path, completed_task: dict, episodes: list[dict], *, state: dict) -> dict:
    from react_agent.eeg_research.agentic.semantic_memory import episode_source, role_memory, standalone_delivery

    task = task_source(camp, completed_task["task_id"])
    verified_episodes = []
    for episode in episodes:
        try:
            verified_episodes.append(episode_source(camp, episode))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    allowed = {"origin_campaign_id": camp.name, "tasks": [task],
               "episodes": verified_episodes, "artifacts": [task["output_artifact"]]}
    bundle = role_memory(camp, state=state)
    related = standalone_delivery(bundle) if bundle else None
    if related:
        related["items"] = [item for item in related["items"] if item["record_type"] == "skill"]
    return {"completed_subtask": task, "allowed_source_refs": allowed, "related_skills": related}


def extend_curation_context(camp: Path, context: dict, completed_task: dict, *, state: dict) -> dict:
    from react_agent.eeg_research.agentic.embedding import memory_config
    from react_agent.eeg_research.agentic.semantic_memory import candidates, episode_source, unsafe_source

    allowed_episode_ids = set()
    for row in context["episodes"]:
        try:
            episode_source(camp, row)
            allowed_episode_ids.add(row["episode_id"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    valid_lesson_ids = {row["record_id"] for row in candidates(camp, memory_config(camp).model_copy(update={"skills_enabled": False}), {})}
    cleaned = {**context,
            "episodes": [row for row in context["episodes"] if row.get("episode_id") in allowed_episode_ids],
            # Preserve the original recent list's ordering; only exclude invalid sources.
            "related_lessons": [row for row in context["related_lessons"] if row.get("lesson_id") in valid_lesson_ids]}
    try:
        if not isinstance(context.get("run"), dict) or unsafe_source(context["run"]):
            raise ValueError("completed_development_run_required")
        cleaned.update(curation_extension(camp, completed_task, cleaned["episodes"], state=state))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cleaned["skills_unavailable_reason"] = str(exc)
    return cleaned


def validated_sources(camp: Path, proposal: SkillProposal, allowed: dict) -> dict:
    from react_agent.eeg_research.agentic.memory import EpisodeStore
    from react_agent.eeg_research.agentic.semantic_memory import episode_source, unsafe_source

    if allowed.get("origin_campaign_id") != camp.name or unsafe_source(proposal.model_dump()) or unsafe_source(allowed):
        raise ValueError("skill_not_campaign_development")
    if not all(step.strip() for step in proposal.procedure):
        raise ValueError("empty_procedure_step")
    task_allow = {row["task_id"]: row for row in allowed.get("tasks") or []}
    artifact_allow = {row["artifact_id"]: row for row in allowed.get("artifacts") or []}
    episode_allow = {row["episode_id"]: row for row in allowed.get("episodes") or []}
    tasks = []; artifacts = []; episodes = []
    for ref in sorted(set(proposal.source_task_refs)):
        if ref not in task_allow:
            raise ValueError("task_ref_not_supplied")
        source = task_source(camp, ref)
        if source != task_allow[ref]:
            raise ValueError("source_task_changed")
        tasks.append(source)
    for ref in sorted(set(proposal.artifact_refs)):
        if ref not in artifact_allow:
            raise ValueError("artifact_ref_not_supplied")
        source = artifact_source(camp, ref)
        if source != artifact_allow[ref]:
            raise ValueError("source_artifact_changed")
        artifacts.append(source)
        task_id = source.get("producer_task_id")
        if task_id in task_allow and task_id not in {item["task_id"] for item in tasks}:
            task = task_source(camp, task_id)
            if task != task_allow[task_id]:
                raise ValueError("source_task_changed")
            tasks.append(task)
    if not tasks:
        raise ValueError("completed_subtask_source_required")
    stored_episodes = {row["episode_id"]: row for row in EpisodeStore(camp).list_episodes()}
    for ref in sorted(set(proposal.supporting_episode_ids)):
        if ref not in episode_allow or ref not in stored_episodes:
            raise ValueError("episode_ref_not_supplied")
        source = episode_source(camp, stored_episodes[ref])
        if source != episode_allow[ref]:
            raise ValueError("source_episode_changed")
        episodes.append(source)
    return {"origin_campaign_id": camp.name, "tasks": sorted(tasks, key=lambda row: row["task_id"]),
            "episodes": episodes, "artifacts": artifacts}


def accept_skills(camp: Path, proposed: list, allowed: dict) -> dict:
    from react_agent.eeg_research.agentic.semantic_memory import SidecarStore

    accepted = []; rejected = []
    # Invalid proposals do not even create a new database.
    for raw in proposed:
        try:
            proposal = SkillProposal.model_validate(raw)
            source = validated_sources(camp, proposal, allowed)
            payload = proposal.model_dump()
            identity = {k: payload[k] for k in ("description_en", "trigger_en", "procedure", "preconditions")}
            skill_id = "skill_" + content_hash(identity)[:24]
            digest = content_hash({"payload": payload, "source_manifest": source})
            store = SidecarStore(camp)
            with store.connect() as conn:
                existing = conn.execute("SELECT * FROM skills WHERE namespace=? AND skill_id=? AND content_hash=?",
                                        (camp.name, skill_id, digest)).fetchone()
                if existing:
                    version = existing["version"]
                else:
                    version = 1 + conn.execute("SELECT COALESCE(MAX(version),0) FROM skills WHERE namespace=? AND skill_id=?", (camp.name, skill_id)).fetchone()[0]
                    conn.execute("UPDATE skills SET is_current=0 WHERE namespace=? AND skill_id=?", (camp.name, skill_id))
                    conn.execute("INSERT INTO skills VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                 (camp.name, skill_id, version, digest, camp.name, canonical(payload), canonical(source),
                                  "source_validated_development_hint", 1, "pending", time.time()))
            accepted.append({"skill_id": skill_id, "version": version, "content_hash": digest})
        except Exception as exc:
            rejected.append({"reason": str(exc) if isinstance(exc, ValueError) else "skill_save_failed:" + type(exc).__name__})
    return {"accepted": accepted, "rejected": rejected}


def current_skills(camp: Path) -> list[dict]:
    from react_agent.eeg_research.agentic.semantic_memory import SidecarStore, unsafe_source

    if not (camp / "semantic_memory.sqlite").is_file():
        return []
    store = SidecarStore(camp)
    with store.connect() as conn:
        rows = conn.execute("SELECT * FROM skills WHERE namespace=? AND origin_campaign_id=? AND is_current=1 AND validation_status='source_validated_development_hint' ORDER BY skill_id,version", (camp.name, camp.name)).fetchall()
    valid = []
    for row in rows:
        row = dict(row)
        try:
            payload = json.loads(row["payload"]); source = json.loads(row["source_manifest"])
            if unsafe_source(payload) or content_hash({"payload": payload, "source_manifest": source}) != row["content_hash"]:
                continue
            if validated_sources(camp, SkillProposal.model_validate(payload), source) != source:
                continue
            row.update(payload=payload, source_manifest=source)
            valid.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return valid


def skill_applicability(payload: dict, context: dict) -> tuple[str, dict]:
    conditions = payload.get("preconditions") or {}
    missing = [key for key in conditions if key not in context or context[key] in (None, "")]
    mismatches = [key for key in conditions if key in context and context[key] not in (None, "") and context[key] != conditions[key]]
    label = "preconditions_unknown" if missing else "preconditions_mismatch" if mismatches else "preconditions_met"
    return label, {"missing_preconditions": missing, "mismatched_preconditions": mismatches}
