"""Research memory. Episodes are written by the runtime. The curator cannot promote them."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

LEVEL_ORDER = (
    "implementation_failure",
    "exploratory_result",
    "paired_seed_result",
    "confirmed_result",
)


def evidence_level(kind: str) -> str:
    if kind == "implementation_failure":
        return "implementation_failure"
    if kind == "paired_seed_result":
        return "paired_seed_result"
    if kind == "confirmed_result":
        return "confirmed_result"
    if kind == "exploratory_result":
        return "exploratory_result"
    return "exploratory_result"


def _rank(level: str | None) -> int:
    if level in LEVEL_ORDER:
        return LEVEL_ORDER.index(level)
    return LEVEL_ORDER.index("exploratory_result")


def episode(
    *,
    task_hash: str,
    candidate_id: str,
    kind: str,
    fidelity: str,
    seed: int,
    metric: float | None,
    contract_fingerprint: str,
    artifact: str,
    episode_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """One deterministic episode. Test metrics are not accepted."""
    return {
        "episode_id": episode_id or f"ep_{uuid.uuid4().hex[:12]}",
        "task_hash": task_hash,
        "candidate_id": candidate_id,
        "kind": kind,
        "evidence_level": evidence_level(kind),
        "fidelity": fidelity,
        "seed": seed,
        "primary_metric": metric,
        "contract_fingerprint": contract_fingerprint,
        "artifact": artifact,
        "job_id": job_id,
        "scope": "development",
    }


def retrieve(entries: list[dict[str, Any]], *, task_hash: str, fingerprint: str) -> list[dict[str, Any]]:
    """Compatible rows stay comparable. Other protocols are analogy only."""
    rows = []
    seen_jobs: set[str] = set()
    for entry in entries:
        if "test" in json_keys(entry):
            continue
        copied = dict(entry)
        job = str(entry.get("job_id") or entry.get("artifact") or entry.get("episode_id") or "")
        if job and job in seen_jobs:
            copied["retrieval"] = "duplicate_same_job"
            copied["new_evidence"] = False
        elif entry.get("task_hash") != task_hash or entry.get("contract_fingerprint") != fingerprint:
            copied["retrieval"] = "analogy_only"
        else:
            copied["retrieval"] = "comparable"
        if job:
            seen_jobs.add(job)
        rows.append(copied)
    return rows


def json_keys(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(key) + " " + json_keys(item) for key, item in value.items())
    if isinstance(value, list):
        return " ".join(json_keys(item) for item in value)
    return ""


def confirmation(pairs: list[float], *, target: int = 3, margin_pp: float = 0.0) -> str:
    """Paired seed rule. Fewer than the target stays provisional."""
    if len(pairs) < target:
        return "provisional_improvement" if pairs and sum(pairs) / len(pairs) > margin_pp / 100 else "no_confirmed_improvement"
    positive = sum(1 for value in pairs if value > 0)
    mean = sum(pairs) / len(pairs)
    if mean > margin_pp / 100 and 3 * positive >= 2 * len(pairs):
        return "replicated_improvement"
    return "no_confirmed_improvement"


def open_store(camp: Path) -> sqlite3.Connection:
    """Return the campaign SQLite memory. Tables are created if missing."""
    camp.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(camp / "memory.sqlite")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            task_hash TEXT,
            candidate_id TEXT,
            kind TEXT,
            evidence_level TEXT,
            fidelity TEXT,
            seed INTEGER,
            primary_metric REAL,
            contract_fingerprint TEXT,
            artifact TEXT,
            job_id TEXT,
            payload TEXT NOT NULL,
            created_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS lessons (
            lesson_id TEXT PRIMARY KEY,
            evidence_level TEXT,
            payload TEXT NOT NULL,
            created_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS curation (
            key TEXT PRIMARY KEY,
            status TEXT,
            detail TEXT,
            updated_at REAL
        )"""
    )
    conn.commit()
    return conn


class EpisodeStore:
    """Append-only episodes. The curator may only insert validated lessons."""

    def __init__(self, camp: Path) -> None:
        self.camp = Path(camp)

    def persist_episode(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload.setdefault("episode_id", f"ep_{uuid.uuid4().hex[:12]}")
        conn = open_store(self.camp)
        try:
            existing = conn.execute("SELECT episode_id FROM episodes WHERE episode_id = ?", (payload["episode_id"],)).fetchone()
            if existing:
                return payload
            conn.execute(
                """INSERT INTO episodes (
                    episode_id, task_hash, candidate_id, kind, evidence_level, fidelity, seed,
                    primary_metric, contract_fingerprint, artifact, job_id, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["episode_id"],
                    payload.get("task_hash"),
                    payload.get("candidate_id"),
                    payload.get("kind"),
                    payload.get("evidence_level"),
                    payload.get("fidelity"),
                    payload.get("seed"),
                    payload.get("primary_metric"),
                    payload.get("contract_fingerprint"),
                    payload.get("artifact"),
                    payload.get("job_id"),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return payload

    def update_episode(self, _episode_id: str, **_fields: Any) -> None:
        """Episodes are immutable. Curator and callers must not rewrite them."""
        raise PermissionError("episode_immutable")

    def list_episodes(self) -> list[dict[str, Any]]:
        conn = open_store(self.camp)
        try:
            rows = conn.execute("SELECT payload FROM episodes ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [json.loads(item[0]) for item in rows]

    def mark_pending_curation(self, detail: str = "curator_unavailable") -> None:
        conn = open_store(self.camp)
        try:
            conn.execute(
                "INSERT INTO curation(key, status, detail, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at",
                ("pending", "pending", detail, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def pending_curation(self) -> bool:
        conn = open_store(self.camp)
        try:
            row = conn.execute("SELECT status FROM curation WHERE key = ?", ("pending",)).fetchone()
        finally:
            conn.close()
        return bool(row and row[0] == "pending")

    def accept_lessons(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Validate curator lessons. Evidence level cannot exceed cited episodes. Episodes are not edited."""
        episodes = {str(item.get("episode_id")): item for item in self.list_episodes()}
        lessons = proposal.get("proposed_lessons") or []
        if isinstance(lessons, dict):
            lessons = [lessons]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        conn = open_store(self.camp)
        try:
            for lesson in lessons:
                if not isinstance(lesson, dict):
                    rejected.append({"reason": "lesson_not_object"})
                    continue
                supporting = [str(item) for item in lesson.get("supporting_episode_ids") or proposal.get("supporting_episode_ids") or []]
                missing = [item for item in supporting if item not in episodes]
                if missing:
                    rejected.append({"reason": "episode_missing", "ids": missing})
                    continue
                cited = [episodes[item] for item in supporting]
                if any("test" in json_keys(item) for item in cited):
                    rejected.append({"reason": "final_test_derived"})
                    continue
                max_level = max((_rank(item.get("evidence_level")) for item in cited), default=0)
                requested = str(lesson.get("evidence_level") or proposal.get("evidence_level_requested") or "exploratory_result")
                if _rank(requested) > max_level:
                    rejected.append({"reason": "evidence_level_exceeds_runs", "requested": requested})
                    continue
                job_ids = [str(item.get("job_id") or item.get("artifact")) for item in cited]
                if job_ids and len(set(job_ids)) < len(job_ids):
                    rejected.append({"reason": "same_job_not_independent"})
                    continue
                stored = {
                    "lesson_id": f"les_{uuid.uuid4().hex[:12]}",
                    "evidence_level": requested if _rank(requested) <= max_level else LEVEL_ORDER[max_level],
                    "supporting_episode_ids": supporting,
                    "contradicting_episode_ids": list(lesson.get("contradicting_episode_ids") or []),
                    "conditions": lesson.get("conditions") or proposal.get("conditions"),
                    "invalidation_conditions": lesson.get("invalidation_conditions") or proposal.get("invalidation_conditions"),
                    "observed_effect": lesson.get("observed_effect"),
                    "summary_zh": lesson.get("summary_zh") or proposal.get("summary_zh") or "",
                }
                conn.execute(
                    "INSERT INTO lessons(lesson_id, evidence_level, payload, created_at) VALUES (?, ?, ?, ?)",
                    (stored["lesson_id"], stored["evidence_level"], json.dumps(stored, ensure_ascii=False), time.time()),
                )
                accepted.append(stored)
            conn.commit()
        finally:
            conn.close()
        return {"accepted": accepted, "rejected": rejected}
