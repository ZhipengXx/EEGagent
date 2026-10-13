"""Campaign-local, leased episode extraction and incremental consolidation.

Only injected, ledger-bound maintenance callers may generate model output.
Native episodes/lessons remain the readable memory format; pipeline tables
hold immutable source versions, extraction jobs, CAS baselines and provenance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.embedding import canonical, content_hash
from react_agent.eeg_research.agentic.memory import open_store
from react_agent.eeg_research.agentic.semantic_memory import require_eeg_campaign, unsafe_source, unsafe_source_path

VERSION = "eeg_research.memory_pipeline.v1"
CLOSED = {"complete", "completed", "failed", "cancelled", "canceled"}
VALIDATION_LEVELS = {"proposal", "procedural_hint", "preconditions_unknown", "validated"}


def valid_source_ref(camp: Path, ref: dict) -> bool:
    """Read-only source validity check used before embedding or maintenance."""
    camp = Path(camp).resolve()
    if unsafe_source(ref) or ref.get("campaign_id", camp.name) != camp.name:
        return False
    if ref.get("status") in {"revoked", "deleted", "stale", "invalid"}:
        return False
    source_id = ref.get("source_id") or ref.get("source_ref") or ref.get("artifact_id")
    digest = ref.get("source_hash") or ref.get("sha256")
    if not source_id or not digest:
        return False
    if isinstance(ref.get("payload"), dict):
        return not unsafe_source(ref["payload"]) and content_hash(ref["payload"]) == digest
    if str(source_id).startswith("context://"):
        db_path = camp / "context_sources.sqlite"
        if not db_path.is_file():
            return False
        try:
            conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error:
            return False
        try:
            row = conn.execute("SELECT s.source_hash,s.payload_ref,s.block_json,s.revoked,h.source_ref "
                "FROM context_sources s LEFT JOIN context_source_heads h ON s.campaign_id=h.campaign_id "
                "AND s.block_id=h.block_id WHERE s.source_ref=? AND s.campaign_id=?",
                (source_id, camp.name)).fetchone()
            if not row or row[3] or row[0] != digest or row[4] != source_id:
                return False
            block = json.loads(row[2]); visibility = block.get("visibility") or []
            if "campaign" not in visibility and "memory_curator" not in visibility:
                return False
            path = (camp / row[1]).resolve()
            if not path.is_relative_to(camp / "context_sources") or not path.is_file():
                return False
            body = json.loads(path.read_text(encoding="utf-8"))
            return not unsafe_source(body) and content_hash(body) == digest
        except (ValueError, OSError, sqlite3.Error, KeyError):
            return False
        finally:
            conn.close()
    path = ref.get("path") or ref.get("source_path")
    if str(source_id).startswith("art_") and not path:
        from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
        try:
            row = resolve_verified_artifact(camp, source_id)
            if row["sha256"] != digest:
                return False
            path = row["path"]
        except (ValueError, OSError, KeyError, TypeError):
            return False
    if not path:
        # Internally captured inline sources remain in the same campaign DB.
        # A compact delivered reference can be verified against that original
        # without duplicating its body into every episode/lesson delivery.
        db_path = camp / "memory.sqlite"
        if not db_path.is_file():
            return False
        try:
            with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT payload FROM memory_pipeline_sources WHERE source_id=? "
                    "AND source_hash=? AND status='valid' AND current=1", (source_id, digest)).fetchone()
            original = json.loads(row[0]) if row else {}
            body = original.get("payload")
            return isinstance(body, dict) and not unsafe_source(body) and content_hash(body) == digest
        except (ValueError, OSError, sqlite3.Error):
            return False
    path = Path(path)
    if not path.is_absolute():
        path = camp / path
    if not path.resolve().is_relative_to(camp) or unsafe_source_path(path) or not path.is_file():
        return False
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return False
    if path.suffix == ".json":
        try:
            return not unsafe_source(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            return False
    return True


def pipeline_record_sources(camp: Path, kind: str, record_id: str) -> list[dict]:
    """Return current valid support without migrating or altering old DBs."""
    path = Path(camp) / "memory.sqlite"
    if not path.is_file():
        return []
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute("SELECT s.payload FROM memory_pipeline_record_sources r JOIN memory_pipeline_sources s "
            "ON r.source_id=s.source_id AND r.source_hash=s.source_hash WHERE r.record_type=? AND r.record_id=? "
            "AND r.relation='support' AND s.current=1 AND s.status='valid' LIMIT 16", (kind, record_id)).fetchall()
        return [{key: value for key, value in json.loads(row[0]).items() if key != "payload"}
                for row in rows if valid_source_ref(camp, json.loads(row[0]))]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


@dataclass(frozen=True)
class PipelineConfig:
    scan_limit: int = 64
    extraction_batch: int = 1
    consolidation_batch: int = 8
    related_limit: int = 8
    lease_seconds: float = 300
    extraction_concurrency: int = 2
    max_attempts_per_digest: int = 3
    backoff_seconds: float = 30
    max_calls_per_turn: int = 2
    max_calls_per_campaign: int = 100
    input_chars: int = 16000
    output_chars: int = 8000
    excerpt_chars: int = 4000
    max_sources_per_group: int = 16

    def __post_init__(self):
        for name in ("scan_limit", "extraction_batch", "consolidation_batch", "related_limit",
                     "extraction_concurrency", "max_attempts_per_digest", "max_calls_per_turn",
                     "max_calls_per_campaign", "input_chars", "output_chars", "excerpt_chars",
                     "max_sources_per_group"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError("invalid_pipeline_config:" + name)
        if self.input_chars > 24000 or self.output_chars > 12000 or self.scan_limit > 256:
            raise ValueError("pipeline_budget_exceeds_protocol")


class LongMemoryPipeline:
    """Two-phase, bounded maintenance for a single verified EEG campaign."""

    def __init__(self, camp: Path, config: PipelineConfig | None = None, *,
                 clock: Callable[[], float] = time.time,
                 source_validator: Callable[[dict], bool] | None = None):
        self.camp = Path(camp).resolve()
        require_eeg_campaign(self.camp)
        self.config = config or PipelineConfig()
        self.clock = clock
        self.source_validator = source_validator
        conn = open_store(self.camp)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_pipeline_sources(
                    source_id TEXT NOT NULL, source_hash TEXT NOT NULL, revision TEXT NOT NULL,
                    status TEXT NOT NULL, visibility TEXT NOT NULL, payload TEXT NOT NULL,
                    current INTEGER NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(source_id,source_hash));
                CREATE INDEX IF NOT EXISTS memory_pipeline_current_sources
                    ON memory_pipeline_sources(source_id,current,status);
                CREATE TABLE IF NOT EXISTS memory_pipeline_jobs(
                    digest TEXT PRIMARY KEY, operation_id TEXT NOT NULL, source_digest TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
                    abstracted INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_until REAL,
                    attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
                    last_turn TEXT, error TEXT, episode_id TEXT, created_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS memory_pipeline_runnable
                    ON memory_pipeline_jobs(status,next_attempt,created_at);
                CREATE TABLE IF NOT EXISTS memory_pipeline_changes(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL, record_hash TEXT NOT NULL, change_kind TEXT NOT NULL,
                    payload TEXT NOT NULL, UNIQUE(record_type,record_id,record_hash,change_kind));
                CREATE TABLE IF NOT EXISTS memory_pipeline_writer(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL,
                    baseline INTEGER NOT NULL, lease_owner TEXT, lease_until REAL,
                    input_digest TEXT, last_turn TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, error TEXT);
                INSERT OR IGNORE INTO memory_pipeline_writer(singleton,revision,baseline)
                    VALUES(1,0,0);
                CREATE TABLE IF NOT EXISTS memory_pipeline_record_sources(
                    record_type TEXT NOT NULL, record_id TEXT NOT NULL, source_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL, relation TEXT NOT NULL,
                    PRIMARY KEY(record_type,record_id,source_id,source_hash,relation));
                CREATE TABLE IF NOT EXISTS memory_pipeline_calls(
                    call_id TEXT PRIMARY KEY, turn_id TEXT NOT NULL, digest TEXT NOT NULL,
                    phase TEXT NOT NULL, epoch INTEGER NOT NULL, correction INTEGER NOT NULL,
                    status TEXT NOT NULL, usage TEXT, created_at REAL NOT NULL,
                    UNIQUE(turn_id,digest,phase,correction));
                CREATE INDEX IF NOT EXISTS memory_pipeline_call_turn
                    ON memory_pipeline_calls(turn_id);
                CREATE TABLE IF NOT EXISTS memory_pipeline_migrations(
                    record_type TEXT NOT NULL, record_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                    PRIMARY KEY(record_type,record_id,content_hash));
                CREATE TABLE IF NOT EXISTS memory_pipeline_cursors(
                    name TEXT PRIMARY KEY, position INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_pipeline_record_versions(
                    record_type TEXT NOT NULL, record_id TEXT NOT NULL, version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(record_type,record_id,version), UNIQUE(record_type,record_id,content_hash));
            """)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def connect(self, *, write=False):
        conn = sqlite3.connect(self.camp / "memory.sqlite", timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ref(self, value: dict) -> dict:
        if not isinstance(value, dict) or unsafe_source(value):
            raise ValueError("memory_source_forbidden")
        ref = copy.deepcopy(value)
        source_id = ref.get("source_id") or ref.get("source_ref") or ref.get("artifact_id")
        digest = ref.get("source_hash") or ref.get("sha256")
        if (not isinstance(source_id, str) or not source_id or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("memory_source_identity_missing")
        if ref.get("campaign_id", self.camp.name) != self.camp.name:
            raise ValueError("memory_cross_campaign_source")
        visibility = ref.get("visibility") or ref.get("visibility_scope") or "development"
        if isinstance(visibility, (list, tuple)):
            if "campaign" not in visibility and "memory_curator" not in visibility:
                raise ValueError("memory_source_visibility_forbidden")
            visibility = "campaign"
        if visibility not in {"development", "engineering", "campaign", "all_roles"}:
            raise ValueError("memory_source_visibility_forbidden")
        return {**ref, "source_id": source_id, "source_hash": digest,
                "campaign_id": self.camp.name, "visibility": visibility}

    def _externally_valid(self, ref: dict) -> bool:
        if self.source_validator is not None:
            try:
                return bool(self.source_validator(ref))
            except (ValueError, OSError, KeyError, TypeError):
                return False
        return valid_source_ref(self.camp, ref)

    def _valid(self, conn, ref: dict) -> bool:
        row = conn.execute("SELECT status FROM memory_pipeline_sources WHERE source_id=? "
                           "AND source_hash=? AND current=1", (ref["source_id"], ref["source_hash"])).fetchone()
        return bool(row and row["status"] == "valid" and self._externally_valid(ref))

    def capture_closed(self, groups: list[dict]) -> dict:
        """Capture source versions before adding an idempotent eligible job.

        Full raw operations remain in SQLite; only bounded cards/excerpts reach
        maintenance. A partial section must have its own operation/section ID.
        """
        captured = []; skipped = []
        for group in groups[:self.config.scan_limit]:
            try:
                if (not isinstance(group, dict) or unsafe_source(group)
                        or group.get("campaign_id", self.camp.name) != self.camp.name):
                    raise ValueError("operation_scope_forbidden")
                if group.get("status") not in CLOSED:
                    skipped.append({"operation_id": group.get("operation_id"), "reason": "operation_not_closed"})
                    continue
                if not group.get("operation_id"):
                    raise ValueError("operation_identity_missing")
                raw_refs = group.get("source_refs") or []
                if not 1 <= len(raw_refs) <= self.config.max_sources_per_group:
                    raise ValueError("operation_source_count_invalid")
                refs = [self._ref(ref) for ref in raw_refs]
                if not all(self._externally_valid(ref) for ref in refs):
                    raise ValueError("operation_source_invalid")
                raw = {**copy.deepcopy(group), "source_refs": refs, "campaign_id": self.camp.name}
                digest = content_hash({"version": VERSION, "operation": raw})
                source_digest = content_hash([(ref["source_id"], ref["source_hash"]) for ref in refs])
                with self.connect(write=True) as conn:
                    for ref in refs:
                        previous = conn.execute("SELECT source_hash,payload FROM memory_pipeline_sources "
                            "WHERE source_id=? AND current=1", (ref["source_id"],)).fetchone()
                        conn.execute("UPDATE memory_pipeline_sources SET current=0 WHERE source_id=? "
                                     "AND source_hash<>?", (ref["source_id"], ref["source_hash"]))
                        conn.execute("INSERT INTO memory_pipeline_sources VALUES(?,?,?,?,?,?,?,?) "
                            "ON CONFLICT(source_id,source_hash) DO UPDATE SET current=1,status='valid'",
                            (ref["source_id"], ref["source_hash"], str(ref.get("source_revision", "1")),
                             "valid", ref["visibility"], canonical(ref), 1, self.clock()))
                        if previous and previous["source_hash"] != ref["source_hash"]:
                            self._change(conn, "source", ref["source_id"], previous["source_hash"],
                                         "revoked", json.loads(previous["payload"]))
                    conn.execute("INSERT OR IGNORE INTO memory_pipeline_jobs "
                        "(digest,operation_id,source_digest,payload,status,created_at) VALUES(?,?,?,?,?,?)",
                        (digest, raw["operation_id"], source_digest, canonical(raw), "pending", self.clock()))
                captured.append(digest)
            except (ValueError, OSError, TypeError) as exc:
                skipped.append({"operation_id": group.get("operation_id") if isinstance(group, dict) else None,
                                "reason": str(exc)})
        return {"status": "captured", "digests": captured, "skipped": skipped,
                "scan_truncated": len(groups) > self.config.scan_limit}

    def _change(self, conn, kind, record_id, digest, change_kind, payload):
        conn.execute("INSERT OR IGNORE INTO memory_pipeline_changes "
                     "(record_type,record_id,record_hash,change_kind,payload) VALUES(?,?,?,?,?)",
                     (kind, record_id, digest, change_kind, canonical(payload)))

    def revoke_source(self, source_id: str, *, source_hash: str | None = None) -> dict:
        with self.connect(write=True) as conn:
            rows = conn.execute("SELECT * FROM memory_pipeline_sources WHERE source_id=? "
                                "AND current=1", (source_id,)).fetchall()
            count = 0
            for row in rows:
                if source_hash is not None and row["source_hash"] != source_hash:
                    continue
                conn.execute("UPDATE memory_pipeline_sources SET status='revoked',current=0 "
                             "WHERE source_id=? AND source_hash=?", (source_id, row["source_hash"]))
                self._change(conn, "source", source_id, row["source_hash"], "revoked", json.loads(row["payload"]))
                count += 1
        return {"status": "revoked", "count": count}

    def synchronize_sources(self) -> dict:
        """Bounded rotating scan discovers deletion/revision changes on restart."""
        with self.connect() as conn:
            row = conn.execute("SELECT position FROM memory_pipeline_cursors WHERE name='sources'").fetchone()
            position = row[0] if row else 0
            rows = conn.execute("SELECT rowid,payload FROM memory_pipeline_sources WHERE current=1 AND status='valid' "
                "AND rowid>? ORDER BY rowid LIMIT ?", (position, self.config.scan_limit)).fetchall()
        revoked = []
        for row in rows:
            ref = json.loads(row[1])
            if not self._externally_valid(ref):
                self.revoke_source(ref["source_id"], source_hash=ref["source_hash"])
                revoked.append(ref["source_id"])
        with self.connect(write=True) as conn:
            conn.execute("INSERT INTO memory_pipeline_cursors VALUES('sources',?) ON CONFLICT(name) "
                "DO UPDATE SET position=excluded.position", (rows[-1][0] if len(rows) == self.config.scan_limit else 0,))
        return {"scanned": len(rows), "revoked": revoked}

    def _calls_available(self, conn, turn_id):
        total = conn.execute("SELECT COUNT(*) FROM memory_pipeline_calls").fetchone()[0]
        per_turn = conn.execute("SELECT COUNT(*) FROM memory_pipeline_calls WHERE turn_id=?", (turn_id,)).fetchone()[0]
        return total < self.config.max_calls_per_campaign and per_turn < self.config.max_calls_per_turn

    def _request(self, phase: str, digest: str, turn_id: str, payload: dict) -> dict:
        request = {"schema_version": VERSION, "campaign_id": self.camp.name,
                   "mode": phase, "purpose": "long_memory_" + phase,
                   "input_digest": digest, "turn_id": turn_id,
                   "allow_semantic_compaction": False, "allow_sync_memory_maintenance": False,
                   "max_output_chars": self.config.output_chars, "payload": payload}
        if len(canonical(request)) > self.config.input_chars:
            raise ValueError("maintenance_input_too_large")
        return request

    def claim_extraction(self, turn_id: str) -> dict:
        with self.connect(write=True) as conn:
            now = self.clock()
            if not self._calls_available(conn, turn_id):
                return {"status": "budget_exhausted"}
            active = conn.execute("SELECT COUNT(*) FROM memory_pipeline_jobs WHERE status='leased' "
                                  "AND lease_until>?", (now,)).fetchone()[0]
            if active >= self.config.extraction_concurrency:
                return {"status": "leased_elsewhere"}
            rows = conn.execute("SELECT * FROM memory_pipeline_jobs WHERE processed=0 AND "
                "status IN ('pending','failed','leased') AND next_attempt<=? "
                "AND (lease_until IS NULL OR lease_until<=?) AND attempts<? "
                "AND (last_turn IS NULL OR last_turn<>?) ORDER BY created_at,digest LIMIT ?",
                (now, now, self.config.max_attempts_per_digest, turn_id, self.config.scan_limit)).fetchall()
            for row in rows:
                raw = json.loads(row["payload"])
                if not all(self._valid(conn, ref) for ref in raw["source_refs"]):
                    conn.execute("UPDATE memory_pipeline_jobs SET status='stale',error='source_invalid' WHERE digest=?", (row["digest"],))
                    continue
                card = {key: raw[key] for key in ("operation_id", "status", "action", "outcome",
                        "engineering_failure", "scientific_scope", "fact_card", "section_ids") if key in raw}
                # Exact cards are never cut silently: a producer must split a
                # large closed group into real sections before retrying.
                excerpts = raw.get("excerpts") or []
                if isinstance(excerpts, str):
                    excerpts = [excerpts]
                selected = []; remaining = self.config.excerpt_chars
                for excerpt in excerpts[:4]:
                    if not isinstance(excerpt, str) or not remaining:
                        continue
                    selected.append({"text": excerpt[:remaining], "selected_char_range": [0, min(len(excerpt), remaining)],
                                     "original_chars": len(excerpt), "complete": len(excerpt) <= remaining})
                    remaining -= len(selected[-1]["text"])
                refs = [{key: ref[key] for key in ("source_id", "source_hash", "source_revision", "visibility", "campaign_id")
                         if key in ref} for ref in raw["source_refs"]]
                try:
                    request = self._request("extraction", row["digest"], turn_id,
                                            {"operation": card, "source_refs": refs, "excerpts": selected})
                except ValueError:
                    conn.execute("UPDATE memory_pipeline_jobs SET status='needs_sections',error='maintenance_input_too_large' WHERE digest=?", (row["digest"],))
                    continue
                owner = uuid.uuid4().hex
                conn.execute("UPDATE memory_pipeline_jobs SET status='leased',lease_owner=?,lease_until=?,"
                    "attempts=attempts+1,last_turn=? WHERE digest=?",
                    (owner, now + self.config.lease_seconds, turn_id, row["digest"]))
                return {"status": "claimed", "phase": "extraction", "digest": row["digest"],
                        "owner": owner, "epoch": row["attempts"] + 1, "request": request}
            return {"status": "noop", "reason": "no_eligible_closed_operation"}

    def _begin_call(self, claim: dict, *, correction=False) -> str:
        with self.connect(write=True) as conn:
            turn_id = claim["request"]["turn_id"]
            if not self._calls_available(conn, turn_id):
                raise ValueError("maintenance_budget_exhausted")
            if claim["phase"] == "extraction":
                row = conn.execute("SELECT lease_owner,lease_until FROM memory_pipeline_jobs WHERE digest=?", (claim["digest"],)).fetchone()
            else:
                row = conn.execute("SELECT lease_owner,lease_until FROM memory_pipeline_writer WHERE singleton=1").fetchone()
            if not row or row["lease_owner"] != claim["owner"] or row["lease_until"] <= self.clock():
                raise ValueError("maintenance_lease_lost")
            call_id = "memcall_" + uuid.uuid4().hex
            conn.execute("INSERT INTO memory_pipeline_calls VALUES(?,?,?,?,?,?,?,?,?)",
                (call_id, turn_id, claim["digest"], claim["phase"], claim["epoch"], int(correction),
                 "started", None, self.clock()))
        return call_id

    def _finish_call(self, call_id, status, response):
        usage = response.get("usage") if isinstance(response, dict) else None
        with self.connect(write=True) as conn:
            conn.execute("UPDATE memory_pipeline_calls SET status=?,usage=? WHERE call_id=?",
                         (status, canonical(usage), call_id))

    def _fail(self, claim, reason):
        delay = self.config.backoff_seconds * 2 ** max(0, claim.get("epoch", 1) - 1)
        with self.connect(write=True) as conn:
            if claim["phase"] == "extraction":
                conn.execute("UPDATE memory_pipeline_jobs SET status='failed',error=?,next_attempt=?,"
                    "lease_owner=NULL,lease_until=NULL WHERE digest=? AND lease_owner=?",
                    (str(reason)[:1000], self.clock() + delay, claim["digest"], claim["owner"]))
            else:
                conn.execute("UPDATE memory_pipeline_writer SET error=?,next_attempt=?,lease_owner=NULL,"
                    "lease_until=NULL WHERE singleton=1 AND lease_owner=?",
                    (str(reason)[:1000], self.clock() + delay, claim["owner"]))
        return {"status": "failed", "reason": str(reason), "digest": claim["digest"]}

    def _response(self, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, dict) or unsafe_source(value):
            raise ValueError("maintenance_response_invalid")
        if len(canonical(value)) > self.config.output_chars:
            raise ValueError("maintenance_response_too_large")
        return value.get("output", value)

    def extract(self, claim: dict, callback: Callable[[dict], dict]) -> dict:
        if claim.get("status") != "claimed" or claim.get("phase") != "extraction":
            return claim
        try:
            call_id = self._begin_call(claim)
            try:
                response = callback(copy.deepcopy(claim["request"]))
                self._finish_call(call_id, "completed", response)
            except Exception:
                self._finish_call(call_id, "failed", {})
                raise
            return self.publish_extraction(claim, self._response(response))
        except Exception as exc:
            return self._fail(claim, str(exc))

    def publish_extraction(self, claim: dict, response: dict) -> dict:
        response = self._response(response)
        with self.connect(write=True) as conn:
            row = conn.execute("SELECT * FROM memory_pipeline_jobs WHERE digest=?", (claim["digest"],)).fetchone()
            if row and row["processed"]:
                return {"status": row["status"], "episode_id": row["episode_id"], "duplicate": True}
            if (not row or row["lease_owner"] != claim["owner"] or row["lease_until"] <= self.clock()):
                return {"status": "stale", "reason": "lease_lost"}
            raw = json.loads(row["payload"])
            if not all(self._valid(conn, ref) for ref in raw["source_refs"]):
                conn.execute("UPDATE memory_pipeline_jobs SET status='stale',lease_owner=NULL,lease_until=NULL,error='source_changed' WHERE digest=?", (claim["digest"],))
                return {"status": "stale", "reason": "source_changed"}
            if response.get("status") == "no_signal":
                # Explicitly processed, never semantically abstracted.
                conn.execute("UPDATE memory_pipeline_jobs SET status='no_signal',processed=1,abstracted=0,"
                    "lease_owner=NULL,lease_until=NULL WHERE digest=?", (claim["digest"],))
                return {"status": "no_signal", "processed": True, "abstracted": False}
            if response.get("status") not in {"extracted", "success", "ok"}:
                raise ValueError("extraction_status_invalid")
            extracted = response.get("episode")
            if not isinstance(extracted, dict) or not str(extracted.get("future_behavior") or "").strip():
                raise ValueError("extraction_episode_missing_signal")
            allowed = {ref["source_id"] for ref in raw["source_refs"]}
            cited = extracted.get("source_refs") or []
            if (not cited or any((item.get("source_id") if isinstance(item, dict) else item) not in allowed for item in cited)):
                raise ValueError("extraction_unknown_source_ref")
            # Outcome/scope and fact values are programmatic, not model rewrites.
            for key in ("action", "outcome", "engineering_failure", "scientific_scope", "fact_card"):
                if key in extracted and extracted[key] != raw.get(key):
                    raise ValueError("extraction_exact_fact_changed:" + key)
            scope = raw.get("scientific_scope") or {}
            if isinstance(scope, str):
                scope = {"scope": scope}
            engineering = bool(raw.get("engineering_failure")) or scope.get("scope") == "engineering_repair"
            episode_id = "ep_mem_" + claim["digest"][:24]
            payload = {"schema_version": VERSION + ".episode", "episode_id": episode_id,
                "origin_campaign_id": self.camp.name, "operation_id": raw["operation_id"],
                "task_hash": claim["digest"], "candidate_id": scope.get("candidate_id"),
                "kind": "implementation_failure" if engineering else "memory_episode",
                "evidence_level": "implementation_failure" if engineering else "exploratory_result",
                "scope": "engineering" if engineering else scope.get("scope", "development"),
                "fidelity": scope.get("fidelity"), "seed": scope.get("seed"),
                "contract_fingerprint": scope.get("evaluation_identity"),
                "action": raw.get("action"), "outcome": raw.get("outcome"),
                "engineering_failure": raw.get("engineering_failure"), "scientific_scope": scope,
                "fact_card": raw.get("fact_card"), "source_refs": [{key: value for key, value in ref.items() if key != "payload"}
                                                                  for ref in raw["source_refs"]],
                "future_behavior": str(extracted["future_behavior"]),
                "interpretation": str(extracted.get("interpretation") or ""),
                "conditions": extracted.get("conditions") or {}, "status": "extracted",
                "validation_status": "source_validated_interpretation", "processed": True,
                "abstracted": False, "source_digest": row["source_digest"]}
            self._insert_episode(conn, payload)
            self._record_sources(conn, "episode", episode_id, raw["source_refs"], "support")
            self._change(conn, "episode", episode_id, content_hash(payload), "added", payload)
            conn.execute("UPDATE memory_pipeline_jobs SET status='processed',processed=1,abstracted=0,episode_id=?,"
                "lease_owner=NULL,lease_until=NULL WHERE digest=?", (episode_id, claim["digest"]))
        return {"status": "processed", "episode_id": episode_id, "processed": True, "abstracted": False}

    def _insert_episode(self, conn, payload):
        conn.execute("INSERT OR IGNORE INTO episodes (episode_id,task_hash,candidate_id,kind,evidence_level,"
            "fidelity,seed,contract_fingerprint,payload,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (payload["episode_id"], payload.get("task_hash"), payload.get("candidate_id"), payload.get("kind"),
             payload.get("evidence_level"), payload.get("fidelity"), payload.get("seed"),
             payload.get("contract_fingerprint"), canonical(payload), self.clock()))

    def _record_sources(self, conn, kind, record_id, refs, relation):
        for ref in refs:
            conn.execute("INSERT OR IGNORE INTO memory_pipeline_record_sources VALUES(?,?,?,?,?)",
                         (kind, record_id, ref["source_id"], ref["source_hash"], relation))

    def _retain_version(self, conn, kind, record_id, payload_text):
        digest = content_hash(json.loads(payload_text))
        known = conn.execute("SELECT 1 FROM memory_pipeline_record_versions WHERE record_type=? "
                             "AND record_id=? AND content_hash=?", (kind, record_id, digest)).fetchone()
        if not known:
            version = 1 + conn.execute("SELECT COALESCE(MAX(version),0) FROM memory_pipeline_record_versions "
                                       "WHERE record_type=? AND record_id=?", (kind, record_id)).fetchone()[0]
            conn.execute("INSERT INTO memory_pipeline_record_versions VALUES(?,?,?,?,?,?)",
                         (kind, record_id, version, digest, payload_text, self.clock()))

    def migrate_legacy(self, *, limit: int | None = None) -> dict:
        """Idempotently register bounded legacy rows without rewriting payloads."""
        from react_agent.eeg_research.agentic.semantic_memory import episode_source
        cap = min(limit or self.config.scan_limit, self.config.scan_limit)
        migrated = 0; skipped = 0; lessons_migrated = 0
        with self.connect() as conn:
            cursor = conn.execute("SELECT position FROM memory_pipeline_cursors WHERE name='legacy_episodes'").fetchone()
            position = cursor[0] if cursor else 0
            rows = conn.execute("SELECT rowid,episode_id,payload FROM episodes WHERE episode_id NOT IN "
                "(SELECT record_id FROM memory_pipeline_migrations WHERE record_type='episode') "
                "AND rowid>? ORDER BY rowid LIMIT ?", (position, cap)).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            if payload.get("schema_version", "").startswith(VERSION) or unsafe_source(payload):
                skipped += 1; continue
            try:
                source = episode_source(self.camp, payload)
                ref = self._ref({"source_id": "legacy_episode:" + row["episode_id"],
                    "source_hash": source["artifact_sha256"], "path": source["artifact_path"],
                    "campaign_id": self.camp.name, "visibility": "development"})
                if not self._externally_valid(ref):
                    raise ValueError("legacy_source_invalid")
                digest = content_hash(payload)
                with self.connect(write=True) as conn:
                    conn.execute("INSERT OR IGNORE INTO memory_pipeline_sources VALUES(?,?,?,?,?,?,?,?)",
                        (ref["source_id"], ref["source_hash"], "legacy", "valid", "development", canonical(ref), 1, self.clock()))
                    conn.execute("INSERT OR IGNORE INTO memory_pipeline_migrations VALUES(?,?,?)",
                                 ("episode", row["episode_id"], digest))
                    self._record_sources(conn, "episode", row["episode_id"], [ref], "support")
                    self._change(conn, "episode", row["episode_id"], digest, "added", payload)
                migrated += 1
            except (OSError, ValueError, KeyError, TypeError):
                skipped += 1
        with self.connect(write=True) as conn:
            conn.execute("INSERT INTO memory_pipeline_cursors VALUES('legacy_episodes',?) ON CONFLICT(name) "
                "DO UPDATE SET position=excluded.position", (rows[-1]["rowid"] if len(rows) == cap else 0,))
            cursor = conn.execute("SELECT position FROM memory_pipeline_cursors WHERE name='legacy_lessons'").fetchone()
            position = cursor[0] if cursor else 0
            lesson_rows = conn.execute("SELECT rowid,lesson_id,payload FROM lessons WHERE lesson_id NOT IN "
                "(SELECT record_id FROM memory_pipeline_migrations WHERE record_type='lesson') "
                "AND rowid>? ORDER BY rowid LIMIT ?", (position, cap)).fetchall()
            for row in lesson_rows:
                payload = json.loads(row["payload"])
                if (payload.get("pipeline_schema_version") or unsafe_source(payload)
                        or payload.get("origin_campaign_id", self.camp.name) != self.camp.name):
                    continue
                supports = payload.get("supporting_episode_ids") or []
                counters = payload.get("contradicting_episode_ids") or []
                if not supports or len(supports) + len(counters) > 16:
                    continue
                support_refs = [ref for ep in supports for ref in self._episode_refs(conn, ep)]
                if not support_refs:
                    continue
                self._record_sources(conn, "lesson", row["lesson_id"], support_refs, "support")
                self._retain_version(conn, "lesson", row["lesson_id"], row["payload"])
                counter_refs = [ref for ep in counters for ref in self._episode_refs(conn, ep)]
                self._record_sources(conn, "lesson", row["lesson_id"], counter_refs, "counterexample")
                conn.execute("INSERT OR IGNORE INTO memory_pipeline_migrations VALUES(?,?,?)",
                             ("lesson", row["lesson_id"], content_hash(payload)))
                lessons_migrated += 1
            conn.execute("INSERT INTO memory_pipeline_cursors VALUES('legacy_lessons',?) ON CONFLICT(name) "
                "DO UPDATE SET position=excluded.position", (lesson_rows[-1]["rowid"] if len(lesson_rows) == cap else 0,))
        skills_migrated = self._migrate_skills(cap)
        return {"status": "migrated", "migrated": migrated, "lessons_migrated": lessons_migrated,
                "skills_migrated": skills_migrated, "skipped": skipped}

    def _migrate_skills(self, cap):
        from react_agent.eeg_research.agentic.skill_memory import current_skills
        migrated = 0
        if not (self.camp / "semantic_memory.sqlite").is_file():
            return migrated
        for row in current_skills(self.camp, limit=cap):
            source = row["source_manifest"]
            if source.get("pipeline_schema_version"):
                continue
            with self.connect() as conn:
                known = conn.execute("SELECT 1 FROM memory_pipeline_migrations WHERE record_type='skill' "
                                     "AND record_id=? AND content_hash=?", (row["skill_id"], row["content_hash"])).fetchone()
            if known:
                continue
            artifacts = list(source.get("artifacts") or [])
            artifacts.extend(task["output_artifact"] for task in source.get("tasks") or [] if task.get("output_artifact"))
            refs = []
            for artifact in artifacts[:16]:
                try:
                    ref = self._ref({"source_id": artifact["artifact_id"], "source_hash": artifact["sha256"],
                                     "campaign_id": self.camp.name, "visibility": "development"})
                    if self._externally_valid(ref):
                        refs.append(ref)
                except (ValueError, KeyError, TypeError):
                    continue
            if not refs:
                continue
            with self.connect(write=True) as conn:
                for ref in refs:
                    conn.execute("INSERT OR IGNORE INTO memory_pipeline_sources VALUES(?,?,?,?,?,?,?,?)",
                        (ref["source_id"], ref["source_hash"], "legacy", "valid", "development", canonical(ref), 1, self.clock()))
                self._record_sources(conn, "skill", row["skill_id"], refs, "support")
                conn.execute("INSERT OR IGNORE INTO memory_pipeline_migrations VALUES(?,?,?)",
                             ("skill", row["skill_id"], row["content_hash"]))
            migrated += 1
        return migrated

    def _bounded_record(self, kind, payload):
        keys = (("episode_id", "operation_id", "kind", "scope", "evidence_level", "action", "outcome",
                 "engineering_failure", "scientific_scope", "future_behavior", "conditions", "status")
                if kind == "episode" else ("lesson_id", "statement", "conditions", "invalidation_conditions",
                 "supporting_episode_ids", "contradicting_episode_ids", "validation_status", "status"))
        view = {key: payload[key] for key in keys if key in payload}
        # Large exact fields require a narrower native section, not clipping.
        if len(canonical(view)) > 4000:
            return {"record_id": payload.get("episode_id") or payload.get("lesson_id"),
                    "record_hash": content_hash(payload), "representation": "ref_only", "requires_read": True}
        return view

    def _related(self, conn, changes):
        episode_ids = {item["record_id"] for item in changes if item["record_type"] == "episode"}
        source_ids = {item["record_id"] for item in changes if item["record_type"] == "source"}
        contexts = []
        for item in changes:
            if item["record_type"] == "episode":
                body = json.loads(item["payload"])
                contexts.append(body.get("scientific_scope") or body.get("conditions") or {"scope": body.get("scope")})
        rows = conn.execute("SELECT payload FROM lessons ORDER BY created_at DESC LIMIT ?",
                            (self.config.scan_limit,)).fetchall()
        related = []
        for row in rows:
            payload = json.loads(row["payload"])
            if unsafe_source(payload) or payload.get("status") in {"retired", "disabled", "superseded"}:
                continue
            refs = set(payload.get("supporting_episode_ids") or []) | set(payload.get("contradicting_episode_ids") or [])
            supports = conn.execute("SELECT source_id FROM memory_pipeline_record_sources WHERE record_type='lesson' "
                                   "AND record_id=?", (payload["lesson_id"],)).fetchall()
            conditions = payload.get("conditions") or {}
            overlaps = [context for context in contexts if isinstance(context, dict) and any(
                conditions.get(key) is not None and conditions.get(key) == context.get(key)
                for key in ("scope", "dataset", "intervention", "method_revision", "error_type"))]
            if refs & episode_ids or {ref[0] for ref in supports} & source_ids or overlaps:
                view = self._bounded_record("lesson", payload)
                if overlaps:
                    exact_keys = ("scope", "seed", "method_revision", "evaluation_identity", "selection_manifest_hash")
                    view["task_applicability"] = "compatible_conditions" if any(all(
                        key not in conditions or conditions[key] == context.get(key) for key in exact_keys)
                        for context in overlaps) else "analogy_only"
                related.append(view)
            if len(related) >= self.config.related_limit:
                break
        return related

    def claim_consolidation(self, turn_id: str) -> dict:
        with self.connect(write=True) as conn:
            writer = conn.execute("SELECT * FROM memory_pipeline_writer WHERE singleton=1").fetchone()
            now = self.clock()
            if writer["lease_until"] and writer["lease_until"] > now:
                return {"status": "leased_elsewhere"}
            changes = [dict(row) for row in conn.execute("SELECT * FROM memory_pipeline_changes "
                "WHERE sequence>? ORDER BY sequence LIMIT ?", (writer["baseline"], self.config.consolidation_batch)).fetchall()]
            if not changes:
                return {"status": "noop", "reason": "no_new_changed_or_revoked_source"}
            digest = content_hash({"version": VERSION, "base_revision": writer["revision"],
                                   "changes": [(item["sequence"], item["record_hash"], item["change_kind"]) for item in changes]})
            same = writer["input_digest"] == digest
            if same and (writer["last_turn"] == turn_id or writer["next_attempt"] > now
                         or writer["attempts"] >= self.config.max_attempts_per_digest):
                return {"status": "backoff", "digest": digest}
            if not self._calls_available(conn, turn_id):
                return {"status": "budget_exhausted"}
            prepared_changes = []
            dependency_refs = []
            for item in changes:
                payload = json.loads(item["payload"])
                if unsafe_source(payload):
                    return {"status": "blocked", "reason": "consolidation_source_forbidden"}
                view = self._bounded_record(item["record_type"], payload) if item["record_type"] != "source" else {
                    "source_id": payload.get("source_id"), "source_hash": payload.get("source_hash"), "status": "revoked"}
                prepared_changes.append({**{k: item[k] for k in ("sequence", "record_type", "record_id", "record_hash", "change_kind")},
                                         "record": view})
                if item["record_type"] == "episode":
                    refs = conn.execute("SELECT s.payload FROM memory_pipeline_record_sources r JOIN memory_pipeline_sources s "
                        "ON r.source_id=s.source_id AND r.source_hash=s.source_hash WHERE r.record_type='episode' AND r.record_id=?",
                        (item["record_id"],)).fetchall()
                    for ref in refs:
                        current_ref = json.loads(ref[0])
                        if self._valid(conn, current_ref):
                            dependency_refs.append(current_ref)
            related = self._related(conn, changes)
            try:
                request = self._request("consolidation", digest, turn_id,
                    {"base_revision": writer["revision"], "baseline": writer["baseline"],
                     "changes": prepared_changes, "related_lessons": related})
            except ValueError:
                # Defer bulky batches; never hand the whole archive to Curator.
                return {"status": "needs_sections", "reason": "maintenance_input_too_large"}
            owner = uuid.uuid4().hex
            epoch = writer["attempts"] + 1 if same else 1
            conn.execute("UPDATE memory_pipeline_writer SET lease_owner=?,lease_until=?,input_digest=?,last_turn=?,"
                "attempts=?,next_attempt=0,error=NULL WHERE singleton=1",
                (owner, now + self.config.lease_seconds, digest, turn_id, epoch))
            return {"status": "claimed", "phase": "consolidation", "digest": digest, "owner": owner,
                "epoch": epoch, "base_revision": writer["revision"], "baseline": writer["baseline"],
                "through_sequence": changes[-1]["sequence"], "dependency_refs": dependency_refs, "request": request}

    def consolidate(self, claim: dict, callback: Callable[[dict], dict]) -> dict:
        if claim.get("status") != "claimed":
            return claim
        try:
            call_id = self._begin_call(claim)
            try:
                response = callback(copy.deepcopy(claim["request"]))
                self._finish_call(call_id, "completed", response)
            except Exception:
                self._finish_call(call_id, "failed", {})
                raise
            return self.publish(claim, self._response(response))
        except Exception as exc:
            return self._fail(claim, str(exc))

    def _episode_refs(self, conn, episode_id):
        rows = conn.execute("SELECT s.payload FROM memory_pipeline_record_sources r JOIN memory_pipeline_sources s "
            "ON r.source_id=s.source_id AND r.source_hash=s.source_hash WHERE r.record_type='episode' AND r.record_id=? "
            "AND s.current=1 AND s.status='valid'", (episode_id,)).fetchall()
        return [{key: value for key, value in json.loads(row[0]).items() if key != "payload"}
                for row in rows if self._valid(conn, json.loads(row[0]))]

    def _proposals(self, conn, claim, response):
        if response.get("status") not in {"consolidated", "success", "ok", "no_signal"}:
            raise ValueError("consolidation_status_invalid")
        lessons = response.get("lessons", response.get("proposed_lessons", [])) or []
        skills = response.get("skills", response.get("proposed_skills", [])) or []
        if not isinstance(lessons, list) or not isinstance(skills, list) or len(lessons) + len(skills) > 16:
            raise ValueError("consolidation_proposal_count_invalid")
        available = {item["record_id"] for item in claim["request"]["payload"]["changes"] if item["record_type"] == "episode"}
        related = claim["request"]["payload"].get("related_lessons", [])
        available.update(ref for item in related for ref in (item.get("supporting_episode_ids") or []) + (item.get("contradicting_episode_ids") or []))
        prepared = []
        for kind, proposals in (("lesson", lessons), ("skill", skills)):
            for proposed in proposals:
                if not isinstance(proposed, dict) or unsafe_source(proposed):
                    raise ValueError("memory_proposal_invalid")
                text = proposed.get("statement") if kind == "lesson" else proposed.get("description_en")
                conditions = proposed.get("conditions") if kind == "lesson" else proposed.get("preconditions")
                supports = proposed.get("supporting_episode_ids") or []
                counters = proposed.get("contradicting_episode_ids") or []
                if (not isinstance(text, str) or not text.strip() or not isinstance(conditions, dict)
                        or not conditions or not isinstance(supports, list) or not supports):
                    raise ValueError("memory_proposal_conditions_or_support_missing")
                if not set(supports + counters).issubset(available):
                    raise ValueError("consolidation_unknown_episode_ref")
                support_refs = []; counter_refs = []; support_payloads = []
                for refs, target in ((supports, support_refs), (counters, counter_refs)):
                    for episode_id in refs:
                        row = conn.execute("SELECT payload FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
                        current_refs = self._episode_refs(conn, episode_id)
                        if not row or not current_refs:
                            raise ValueError("memory_proposal_support_stale")
                        target.extend(current_refs)
                        if refs is supports:
                            support_payloads.append(json.loads(row[0]))
                engineering = any(item.get("scope") == "engineering" or item.get("kind") == "implementation_failure" for item in support_payloads)
                if engineering and conditions.get("scope") != "engineering_repair":
                    raise ValueError("engineering_memory_scope_mismatch")
                if engineering:
                    import re
                    if re.search(r"(?i)top[_ -]?[15]|accuracy|\b\d+(?:\.\d+)?\s*(?:pp|percentage points)|准确率", text):
                        raise ValueError("engineering_memory_not_scientific_evidence")
                for ep in support_payloads:
                    scope = ep.get("scientific_scope") or {}
                    if scope.get("scope") == "two_fold_development":
                        for key in ("scope", "seed", "method_revision", "selection_manifest_hash", "screening_subjects"):
                            if scope.get(key) is not None and conditions.get(key) != scope[key]:
                                raise ValueError("screening_lesson_scope_mismatch:" + key)
                status = proposed.get("validation_status", "procedural_hint" if kind == "skill" else "proposal")
                if status not in VALIDATION_LEVELS:
                    raise ValueError("memory_validation_status_invalid")
                # Model output alone cannot promote a procedural hint to a
                # validated procedure; source must supply that native state.
                if status == "validated" and not all(ep.get("validation_status") == "validated" for ep in support_payloads):
                    raise ValueError("memory_validation_promotion_forbidden")
                if kind == "skill" and (not isinstance(proposed.get("procedure"), list)
                        or not proposed["procedure"] or any(not isinstance(step, str) or not step.strip() for step in proposed["procedure"])):
                    raise ValueError("skill_procedure_missing")
                record_id = proposed.get(kind + "_id") or kind + "_mem_" + content_hash({"text": text, "conditions": conditions})[:24]
                if proposed.get(kind + "_id"):
                    related_ids = {item.get("lesson_id") for item in related}
                    if kind != "lesson" or record_id not in related_ids:
                        raise ValueError("memory_record_not_supplied_for_update")
                payload = {**copy.deepcopy(proposed), kind + "_id": record_id, "origin_campaign_id": self.camp.name,
                           "status": "active", "validation_status": status, "is_current": True,
                           "pipeline_schema_version": VERSION, "source_refs": support_refs,
                           "counterexample_source_refs": counter_refs,
                           "evidence_level": "implementation_failure" if engineering else "exploratory_result"}
                prepared.append((kind, record_id, payload, support_refs, counter_refs))
        return prepared

    def _retire_unsupported(self, conn, changes):
        affected = set()
        for change in changes:
            if change["record_type"] == "source" and change["change_kind"] == "revoked":
                affected.update((row[0], row[1]) for row in conn.execute("SELECT DISTINCT record_type,record_id "
                    "FROM memory_pipeline_record_sources WHERE source_id=? AND source_hash=? AND relation='support'",
                    (change["record_id"], change["record_hash"])).fetchall())
        retired = []
        for kind, record_id in affected:
            if kind not in {"lesson", "skill"}:
                continue
            remaining = conn.execute("SELECT s.payload FROM memory_pipeline_record_sources r JOIN memory_pipeline_sources s "
                "ON r.source_id=s.source_id AND r.source_hash=s.source_hash WHERE r.record_type=? AND r.record_id=? "
                "AND r.relation='support' AND s.current=1 AND s.status='valid'", (kind, record_id)).fetchall()
            refs = [{key: value for key, value in json.loads(row[0]).items() if key != "payload"}
                    for row in remaining if self._valid(conn, json.loads(row[0]))]
            if kind == "lesson":
                row = conn.execute("SELECT payload FROM lessons WHERE lesson_id=?", (record_id,)).fetchone()
                if not row:
                    continue
                payload = json.loads(row[0]); payload["source_refs"] = refs
                self._retain_version(conn, "lesson", record_id, row[0])
                # Keep a valid claim supported by any other valid source;
                # remove revoked support instead of discarding its whole row.
                valid_episodes = [ep for ep in payload.get("supporting_episode_ids", []) if self._episode_refs(conn, ep)]
                payload["supporting_episode_ids"] = valid_episodes
                if not refs:
                    payload.update(status="retired", is_current=False, retirement_reason="all_support_revoked")
                    retired.append(record_id)
                conn.execute("UPDATE lessons SET payload=? WHERE lesson_id=?", (canonical(payload), record_id))
                self._retain_version(conn, "lesson", record_id, canonical(payload))
            else:
                # Skills share semantic_memory.sqlite; retirement is exposed
                # through pipeline validity before vector recall. The source
                # manifest remains immutable for every historical version.
                if not refs:
                    retired.append(record_id)
        return retired

    def publish(self, claim: dict, response: dict) -> dict:
        response = self._response(response)
        with self.connect(write=True) as conn:
            writer = conn.execute("SELECT * FROM memory_pipeline_writer WHERE singleton=1").fetchone()
            if (writer["revision"] != claim["base_revision"] or writer["baseline"] != claim["baseline"]
                    or writer["lease_owner"] != claim["owner"] or writer["lease_until"] <= self.clock()
                    or writer["input_digest"] != claim["digest"]):
                return {"status": "stale", "reason": "consolidation_cas_conflict"}
            if not all(self._valid(conn, ref) for ref in claim.get("dependency_refs", [])):
                conn.execute("UPDATE memory_pipeline_writer SET lease_owner=NULL,lease_until=NULL,error='source_changed' WHERE singleton=1")
                return {"status": "stale", "reason": "source_changed"}
            prepared = self._proposals(conn, claim, response)
            changes = [dict(row) for row in conn.execute("SELECT * FROM memory_pipeline_changes WHERE sequence>? "
                "AND sequence<=? ORDER BY sequence", (claim["baseline"], claim["through_sequence"])).fetchall()]
            retired = self._retire_unsupported(conn, changes)
            accepted = []
            for kind, record_id, payload, supports, counters in prepared:
                if kind == "lesson":
                    row = conn.execute("SELECT payload FROM lessons WHERE lesson_id=?", (record_id,)).fetchone()
                    if row:
                        self._retain_version(conn, "lesson", record_id, row[0])
                    if not row or canonical(json.loads(row[0])) != canonical(payload):
                        conn.execute("INSERT INTO lessons VALUES(?,?,?,?) ON CONFLICT(lesson_id) "
                            "DO UPDATE SET evidence_level=excluded.evidence_level,payload=excluded.payload",
                            (record_id, payload["evidence_level"], canonical(payload), self.clock()))
                        self._retain_version(conn, "lesson", record_id, canonical(payload))
                else:
                    self._publish_skill(conn, record_id, payload, supports, counters)
                conn.execute("DELETE FROM memory_pipeline_record_sources WHERE record_type=? AND record_id=?", (kind, record_id))
                self._record_sources(conn, kind, record_id, supports, "support")
                self._record_sources(conn, kind, record_id, counters, "counterexample")
                accepted.append({"record_type": kind, "record_id": record_id})
            conn.execute("UPDATE memory_pipeline_writer SET revision=revision+1,baseline=?,lease_owner=NULL,lease_until=NULL,"
                "error=NULL,next_attempt=0 WHERE singleton=1 AND revision=? AND lease_owner=?",
                (claim["through_sequence"], claim["base_revision"], claim["owner"]))
        return {"status": "published", "revision": claim["base_revision"] + 1,
                "baseline": claim["through_sequence"], "accepted": accepted, "retired": retired}

    def _publish_skill(self, conn, record_id, payload, supports, counters):
        # Keep the skill rows in the existing sidecar format. ATTACH makes
        # lesson/baseline/skill publication one SQLite transaction, rather than
        # separately advancing a baseline before a failed skill write.
        sidecar = self.camp / "semantic_memory.sqlite"
        if not any(row[1] == "skill_memory_pipeline" for row in conn.execute("PRAGMA database_list")):
            conn.execute("ATTACH DATABASE ? AS skill_memory_pipeline", (str(sidecar),))
        # SQLite's super-journal guarantees the two-file commit only in
        # rollback-journal mode; reject a pre-existing WAL configuration rather
        # than claim an atomic baseline/skill publication under that mode.
        if (conn.execute("PRAGMA main.journal_mode").fetchone()[0].lower() not in {"delete", "truncate", "persist"}
                or conn.execute("PRAGMA skill_memory_pipeline.journal_mode").fetchone()[0].lower() not in {"delete", "truncate", "persist"}):
            raise ValueError("memory_skill_publication_requires_atomic_rollback_journals")
        conn.execute("CREATE TABLE IF NOT EXISTS skill_memory_pipeline.skills (namespace TEXT NOT NULL,skill_id TEXT NOT NULL,"
            "version INTEGER NOT NULL,content_hash TEXT NOT NULL,origin_campaign_id TEXT NOT NULL,payload TEXT NOT NULL,"
            "source_manifest TEXT NOT NULL,validation_status TEXT NOT NULL,is_current INTEGER NOT NULL,index_status TEXT NOT NULL,"
            "created_at REAL NOT NULL,PRIMARY KEY(namespace,skill_id,version))")
        source = {"origin_campaign_id": self.camp.name, "pipeline_schema_version": VERSION,
                  "source_refs": supports, "counterexample_source_refs": counters}
        digest = content_hash({"payload": payload, "source_manifest": source})
        row = conn.execute("SELECT version FROM skill_memory_pipeline.skills WHERE namespace=? AND skill_id=? AND content_hash=?",
                           (self.camp.name, record_id, digest)).fetchone()
        if row:
            return
        version = 1 + conn.execute("SELECT COALESCE(MAX(version),0) FROM skill_memory_pipeline.skills WHERE namespace=? AND skill_id=?",
                                   (self.camp.name, record_id)).fetchone()[0]
        conn.execute("UPDATE skill_memory_pipeline.skills SET is_current=0 WHERE namespace=? AND skill_id=?", (self.camp.name, record_id))
        conn.execute("INSERT INTO skill_memory_pipeline.skills VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (self.camp.name, record_id, version, digest, self.camp.name, canonical(payload), canonical(source),
             "source_validated_development_hint", 1, "pending", self.clock()))

    def validate_record(self, record_type: str, record_id: str) -> bool:
        with self.connect() as conn:
            rows = conn.execute("SELECT s.payload FROM memory_pipeline_record_sources r JOIN memory_pipeline_sources s "
                "ON r.source_id=s.source_id AND r.source_hash=s.source_hash WHERE r.record_type=? AND r.record_id=? "
                "AND r.relation='support' AND s.current=1 AND s.status='valid'", (record_type, record_id)).fetchall()
            return any(self._valid(conn, json.loads(row[0])) for row in rows)

    def run_turn(self, groups: list[dict], *, turn_id: str, callback: Callable[[dict], dict]) -> dict:
        self.synchronize_sources()
        capture = self.capture_closed(groups)
        extracted = []
        for _ in range(self.config.extraction_batch):
            claim = self.claim_extraction(turn_id)
            if claim["status"] != "claimed":
                extracted.append(claim); break
            extracted.append(self.extract(claim, callback))
        claim = self.claim_consolidation(turn_id)
        consolidated = self.consolidate(claim, callback) if claim["status"] == "claimed" else claim
        return {"capture": capture, "extraction": extracted, "consolidation": consolidated}
