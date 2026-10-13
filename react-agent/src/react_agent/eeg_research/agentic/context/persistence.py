"""Durable working-memory state in the campaign's existing SQLite database.

Immutable files precede the one SQLite publish transaction.  A file left by a
crash is a draft, never evidence that coverage advanced.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

from .sources import atomic_immutable
from .types import canonical, digest


SCHEMA = "eeg.context.persistence.v1"


class ContextPersistence:
    def __init__(self, camp: Path | str, *, clock=time.time, campaign_id: str | None = None):
        self.camp = Path(camp).resolve()
        self.camp.mkdir(parents=True, exist_ok=True)
        self.db_path = self.camp / "memory.sqlite"
        self.artifact_root = self.camp / "context_memory"
        self.clock = clock
        self.campaign_id = campaign_id or self.camp.name
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS context_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    campaign_id TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    snapshot_ref TEXT, authority_revision TEXT NOT NULL DEFAULT '',
                    source_epoch INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS context_closed_sections (
                    section_ref TEXT PRIMARY KEY, group_id TEXT NOT NULL,
                    block_id TEXT NOT NULL, source_ref TEXT NOT NULL,
                    source_hash TEXT NOT NULL, section_id TEXT NOT NULL,
                    section_json TEXT NOT NULL, status TEXT NOT NULL,
                    captured_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS context_sections_group
                    ON context_closed_sections(group_id,section_ref);
                CREATE TABLE IF NOT EXISTS context_coverage (
                    section_ref TEXT PRIMARY KEY, block_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL, section_id TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    abstracted INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL, valid INTEGER NOT NULL DEFAULT 1);
                CREATE INDEX IF NOT EXISTS context_coverage_block
                    ON context_coverage(block_id,source_hash,valid,abstracted);
                CREATE TABLE IF NOT EXISTS context_invalid_sources (
                    source_ref TEXT PRIMARY KEY, reason TEXT NOT NULL,
                    invalidated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS context_mirror_sections (
                    mirror_block_id TEXT NOT NULL, mirror_source_ref TEXT NOT NULL,
                    mirror_source_hash TEXT NOT NULL, mirror_section_id TEXT NOT NULL,
                    original_block_id TEXT NOT NULL, original_source_ref TEXT NOT NULL,
                    original_source_hash TEXT NOT NULL, original_section_id TEXT NOT NULL,
                    binding_json TEXT NOT NULL, full_mirror_section INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(mirror_block_id,mirror_source_hash,mirror_section_id));
                CREATE TABLE IF NOT EXISTS context_snapshot_items (
                    item_id TEXT PRIMARY KEY, item_kind TEXT NOT NULL,
                    item_json TEXT NOT NULL, revision INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1, valid INTEGER NOT NULL DEFAULT 1);
                CREATE INDEX IF NOT EXISTS context_items_active
                    ON context_snapshot_items(active,valid,item_kind,item_id);
                CREATE TABLE IF NOT EXISTS context_snapshot_revisions (
                    revision INTEGER PRIMARY KEY, base_revision INTEGER NOT NULL,
                    artifact_ref TEXT NOT NULL, dependency_digest TEXT NOT NULL,
                    published_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS context_proposals (
                    proposal_id TEXT PRIMARY KEY, artifact_ref TEXT NOT NULL,
                    status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS context_delivery_scopes (
                    receipt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                    role TEXT NOT NULL, scope_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL, source_hash TEXT NOT NULL,
                    scope_json TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS context_maintenance_jobs (
                    input_digest TEXT PRIMARY KEY, status TEXT NOT NULL,
                    lease_owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_turn_id TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '', result_ref TEXT,
                    input_chars INTEGER NOT NULL DEFAULT 0,
                    output_chars INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS context_maintenance_usage (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    calls INTEGER NOT NULL DEFAULT 0, cost REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS context_maintenance_turns (
                    turn_id TEXT PRIMARY KEY, calls INTEGER NOT NULL DEFAULT 0,
                    cost REAL NOT NULL DEFAULT 0);
            """)
            db.execute("INSERT OR IGNORE INTO context_state(singleton,campaign_id) VALUES(1,?)",
                       (self.campaign_id,))
            db.execute("INSERT OR IGNORE INTO context_maintenance_usage(singleton) VALUES(1)")
            campaign = db.execute("SELECT campaign_id FROM context_state WHERE singleton=1").fetchone()[0]
            if campaign != self.campaign_id:
                raise ValueError("context database belongs to another campaign")

    @contextlib.contextmanager
    def db(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            if immediate:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def immutable(self, kind: str, value: Any) -> str:
        path = self.artifact_root / kind / (digest(value) + ".json")
        atomic_immutable(path, canonical(value))
        return str(path.relative_to(self.camp))

    def read(self, artifact_ref: str) -> Any:
        path = (self.camp / artifact_ref).resolve()
        if not path.is_relative_to(self.artifact_root):
            raise PermissionError("context artifact escapes campaign")
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.stem != digest(value):
            raise ValueError("immutable context artifact hash mismatch")
        return value

    def state(self) -> dict[str, Any]:
        with self.db() as db:
            return dict(db.execute("SELECT * FROM context_state WHERE singleton=1").fetchone())

    def coverage_page(self, *, offset: int = 0, limit: int = 32) -> list[dict[str, Any]]:
        if offset < 0 or not 0 < limit <= 64:
            raise ValueError("coverage page must be bounded")
        with self.db() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM context_coverage ORDER BY section_ref LIMIT ? OFFSET ?", (limit, offset))]

    def record_scope(self, *, role: str, task_id: str, source_ref: str,
                     source_hash: str, scope: dict[str, Any], kind: str = "delivered") -> str:
        if kind not in {"delivered", "read"}:
            raise ValueError("delivery scope must be delivered or read")
        if len(canonical(scope)) > 4_000:
            raise ValueError("delivery scope is too large")
        record = {"task_id": task_id, "role": role, "source_ref": source_ref,
                  "source_hash": source_hash, "scope": scope, "kind": kind}
        receipt = digest(record)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO context_delivery_scopes VALUES(?,?,?,?,?,?,?,?)",
                       (receipt, task_id, role, kind, source_ref, source_hash,
                        canonical(scope), self.clock()))
        return receipt

    def claim(self, input_digest: str, *, owner: str, turn_id: str, lease_seconds: float,
              max_attempts: int, input_chars: int, max_concurrent: int = 1) -> str:
        """Claim one epoch per digest/turn; expired workers can recover next turn."""
        now = self.clock()
        if max_concurrent < 1 or lease_seconds <= 0 or max_attempts < 1:
            raise ValueError("invalid maintenance lease policy")
        with self.db(immediate=True) as db:
            row = db.execute("SELECT * FROM context_maintenance_jobs WHERE input_digest=?", (input_digest,)).fetchone()
            if row:
                if row["status"] in {"published", "no_signal", "no_op"}:
                    return "cached_" + row["status"]
                if row["lease_until"] > now:
                    return "leased"
                if row["last_turn_id"] == turn_id:
                    return "already_attempted_this_turn"
                if row["retry_count"] >= max_attempts:
                    return "attempts_exhausted"
                if row["next_attempt_at"] > now:
                    return "backoff"
                live = db.execute("SELECT COUNT(*) FROM context_maintenance_jobs WHERE status='leased' AND lease_until>?", (now,)).fetchone()[0]
                if live >= max_concurrent:
                    return "concurrency_limit"
                db.execute("UPDATE context_maintenance_jobs SET status='leased',lease_owner=?,lease_until=?,retry_count=retry_count+1,last_turn_id=?,input_chars=? WHERE input_digest=?",
                           (owner, now + lease_seconds, turn_id, input_chars, input_digest))
            else:
                live = db.execute("SELECT COUNT(*) FROM context_maintenance_jobs WHERE status='leased' AND lease_until>?", (now,)).fetchone()[0]
                if live >= max_concurrent:
                    return "concurrency_limit"
                db.execute("INSERT INTO context_maintenance_jobs(input_digest,status,lease_owner,lease_until,retry_count,last_turn_id,input_chars) VALUES(?,'leased',?,?,1,?,?)",
                           (input_digest, owner, now + lease_seconds, turn_id, input_chars))
        return "claimed"

    def charge(self, *, turn_id: str, max_calls_turn: int, max_calls_campaign: int,
               estimated_cost: float, max_cost_turn: float, max_cost_campaign: float) -> bool:
        """Reserve local maintenance limits; the injected caller uses native costs."""
        with self.db(immediate=True) as db:
            db.execute("INSERT OR IGNORE INTO context_maintenance_turns(turn_id) VALUES(?)", (turn_id,))
            turn = db.execute("SELECT * FROM context_maintenance_turns WHERE turn_id=?", (turn_id,)).fetchone()
            usage = db.execute("SELECT * FROM context_maintenance_usage WHERE singleton=1").fetchone()
            if (turn["calls"] >= max_calls_turn or usage["calls"] >= max_calls_campaign
                    or turn["cost"] + estimated_cost > max_cost_turn
                    or usage["cost"] + estimated_cost > max_cost_campaign):
                return False
            db.execute("UPDATE context_maintenance_turns SET calls=calls+1,cost=cost+? WHERE turn_id=?", (estimated_cost, turn_id))
            db.execute("UPDATE context_maintenance_usage SET calls=calls+1,cost=cost+? WHERE singleton=1", (estimated_cost,))
        return True

    def settle_cost(self, *, turn_id: str, reserved: float, actual: float) -> None:
        if actual < 0 or reserved < 0:
            raise ValueError("invalid maintenance cost")
        with self.db(immediate=True) as db:
            difference = actual - reserved
            db.execute("UPDATE context_maintenance_usage SET cost=cost+? WHERE singleton=1", (difference,))
            db.execute("UPDATE context_maintenance_turns SET cost=cost+? WHERE turn_id=?", (difference, turn_id))

    def finish_job(self, input_digest: str, *, owner: str, status: str,
                   error: str = "", result_ref: str = "", output_chars: int = 0,
                   backoff_seconds: float = 0) -> bool:
        if status not in {"published", "no_signal", "no_op", "failed", "stale", "budget_exhausted"}:
            raise ValueError("unknown maintenance job outcome")
        with self.db(immediate=True) as db:
            row = db.execute("SELECT lease_owner,status FROM context_maintenance_jobs WHERE input_digest=?", (input_digest,)).fetchone()
            if not row or row["lease_owner"] != owner or row["status"] != "leased":
                return False
            db.execute("UPDATE context_maintenance_jobs SET status=?,lease_owner=NULL,lease_until=0,last_error=?,result_ref=?,output_chars=?,next_attempt_at=? WHERE input_digest=?",
                       (status, error[:2_000], result_ref, output_chars,
                        self.clock() + max(0, backoff_seconds), input_digest))
        return True
