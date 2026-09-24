"""SQLite memory store. Artifacts stay on disk; DB holds summaries and hashes."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    run_id TEXT,
    sample_id TEXT,
    input_hash TEXT,
    artifact_hash TEXT,
    generation_profile_id TEXT,
    checkpoint_hash TEXT,
    space_name TEXT,
    t_len INTEGER,
    normalization TEXT,
    execution_domain TEXT,
    screening_decision TEXT,
    verification_status TEXT,
    verification_scope TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    idempotency_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    verification_scope TEXT,
    source TEXT,
    evidence TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lesson_candidates (
    candidate_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    lesson_type TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operational_stats (
    stat_id TEXT PRIMARY KEY,
    tool_name TEXT,
    tool_version TEXT,
    device TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class MemoryRepository(Protocol):
    def retrieve(self, query: dict[str, Any]) -> list[dict[str, Any]]: ...
    def append_episode(self, episode: dict[str, Any]) -> str: ...
    def append_candidate(self, candidate: dict[str, Any]) -> str: ...
    def record_verification(self, verification: dict[str, Any]) -> str: ...
    def inspect(self, limit: int = 20) -> dict[str, Any]: ...
    def export_jsonl(self, path: Path) -> Path: ...


class SqliteMemoryRepository:
    """Transactional SQLite implementation."""

    def __init__(self, path: str | Path, *, namespace: str = "default") -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(SCHEMA)
        cur = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', 'memory.v1')"
            )
            self._conn.commit()

    def append_episode(self, episode: dict[str, Any]) -> str:
        key = episode.get("idempotency_key") or (
            f"{self.namespace}:{episode.get('run_id')}:{episode.get('sample_id')}:"
            f"{episode.get('input_hash')}"
        )
        existing = self._conn.execute(
            "SELECT episode_id FROM episodes WHERE idempotency_key=?", (key,)
        ).fetchone()
        if existing:
            return str(existing["episode_id"])
        episode_id = episode.get("episode_id") or uuid4().hex[:16]
        episode = {**episode, "episode_id": episode_id}
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO episodes(
                episode_id, namespace, run_id, sample_id, input_hash, artifact_hash,
                generation_profile_id, checkpoint_hash, space_name, t_len, normalization,
                execution_domain, screening_decision, verification_status, verification_scope,
                payload, created_at, idempotency_key
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                episode_id,
                episode.get("namespace") or self.namespace,
                episode.get("run_id"),
                episode.get("sample_id"),
                episode.get("input_hash"),
                episode.get("artifact_hash"),
                episode.get("generation_profile_id"),
                episode.get("checkpoint_hash"),
                episode.get("space_name"),
                episode.get("t_len"),
                episode.get("normalization"),
                episode.get("execution_domain") or "real",
                episode.get("screening_decision"),
                episode.get("verification_status") or "unverified",
                episode.get("verification_scope") or "numeric_consistency_only",
                json.dumps(episode, default=str),
                now,
                key,
            ),
        )
        self._conn.commit()
        return episode_id

    def append_candidate(self, candidate: dict[str, Any]) -> str:
        cid = candidate.get("candidate_id") or uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO lesson_candidates(candidate_id, namespace, lesson_type, payload, created_at)
            VALUES (?,?,?,?,?)
            """,
            (
                cid,
                candidate.get("namespace") or self.namespace,
                candidate.get("lesson_type"),
                json.dumps(candidate, default=str),
                now,
            ),
        )
        self._conn.commit()
        return cid

    def record_verification(self, verification: dict[str, Any]) -> str:
        vid = verification.get("verification_id") or uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO verifications(
                verification_id, episode_id, verification_status, verification_scope,
                source, evidence, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                vid,
                verification["episode_id"],
                verification["verification_status"],
                verification.get("verification_scope") or "numeric_consistency_only",
                verification.get("source") or "human",
                verification.get("evidence") or "",
                now,
            ),
        )
        self._conn.execute(
            "UPDATE episodes SET verification_status=? WHERE episode_id=?",
            (verification["verification_status"], verification["episode_id"]),
        )
        self._conn.commit()
        return vid

    def retrieve(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        namespace = query.get("namespace") or self.namespace
        rows = self._conn.execute(
            "SELECT payload, verification_status FROM episodes WHERE namespace=? ORDER BY created_at DESC",
            (namespace,),
        ).fetchall()
        items = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["verification_status"] = row["verification_status"]
            items.append(payload)
        return items

    def inspect(self, limit: int = 20) -> dict[str, Any]:
        episodes = self._conn.execute(
            "SELECT episode_id, sample_id, screening_decision, verification_status, created_at "
            "FROM episodes WHERE namespace=? ORDER BY created_at DESC LIMIT ?",
            (self.namespace, limit),
        ).fetchall()
        cands = self._conn.execute(
            "SELECT candidate_id, lesson_type, created_at FROM lesson_candidates "
            "WHERE namespace=? ORDER BY created_at DESC LIMIT ?",
            (self.namespace, limit),
        ).fetchall()
        return {
            "namespace": self.namespace,
            "path": str(self.path),
            "episodes": [dict(r) for r in episodes],
            "candidates": [dict(r) for r in cands],
        }

    def export_jsonl(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._conn.execute(
            "SELECT payload FROM episodes WHERE namespace=?", (self.namespace,)
        ).fetchall()
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row["payload"] + "\n")
        return path
