"""Campaign memory in its own SQLite file."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ResearchMemory:
    """Store support and non-improvement in a namespace that is not the fMRI ledger."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                task_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                fingerprint TEXT,
                payload TEXT NOT NULL,
                evidence_level TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add(
        self,
        *,
        task_hash: str,
        kind: str,
        fingerprint: str | None,
        payload: dict[str, Any],
        evidence_level: str,
    ) -> int:
        """Insert one episode. Cached replays stay at the supplied level."""
        cur = self._conn.execute(
            """
            INSERT INTO episodes (namespace, task_hash, kind, fingerprint, payload, evidence_level)
            VALUES ('eeg_task_validation', ?, ?, ?, ?, ?)
            """,
            (task_hash, kind, fingerprint, json.dumps(payload), evidence_level),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def matching(self, task_hash: str) -> list[dict[str, Any]]:
        """Return episodes for this frozen task only."""
        rows = self._conn.execute(
            """
            SELECT id, kind, fingerprint, payload, evidence_level
            FROM episodes
            WHERE namespace = 'eeg_task_validation' AND task_hash = ?
            ORDER BY id
            """,
            (task_hash,),
        ).fetchall()
        found = []
        for row in rows:
            item = json.loads(row["payload"])
            item["memory_id"] = row["id"]
            item["kind"] = row["kind"]
            item["evidence_level"] = row["evidence_level"]
            item["fingerprint"] = row["fingerprint"]
            found.append(item)
        return found

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
