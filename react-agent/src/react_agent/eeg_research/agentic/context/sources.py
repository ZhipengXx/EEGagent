"""Campaign-local immutable delivery sources, independent of development evidence."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .types import SourceBlock, ReadRequest, ReadResult, canonical, digest

PRIVATE_SCOPES = frozenset({"final_test", "final_holdout", "held_out_unused", "secret"})
PRIVATE_KEYS = frozenset({"secret", "secrets", "apikey", "deepseekapikey", "openaiapikey", "anthropicapikey",
    "password", "credentials", "accesstoken", "refreshtoken", "finaltest", "finaltestresult", "finaltestresults",
    "finalholdout", "finalholdoutresult", "finalholdoutresults", "unusedheldoutresult"})


def _assert_visible_payload(payload: Any, role: str) -> None:
    if isinstance(payload, dict):
        if payload.get("scope") in PRIVATE_SCOPES:
            raise PermissionError("private/final-test source forbids context capture")
        visibility = payload.get("visibility")
        if isinstance(visibility, str):
            visibility = (visibility,)
        if isinstance(visibility, (list, tuple)):
            if any(scope in PRIVATE_SCOPES for scope in visibility) or ("campaign" not in visibility and "development" not in visibility and role not in visibility):
                raise PermissionError("source visibility forbids capture")
        for key, child in payload.items():
            normalized = "".join(char for char in str(key).lower() if char.isalnum())
            if normalized in PRIVATE_KEYS and child is not None:
                raise PermissionError("private credential/final-test field forbids context capture")
            _assert_visible_payload(child, role)
    elif isinstance(payload, list):
        for child in payload:
            _assert_visible_payload(child, role)


assert_visible_payload = _assert_visible_payload


def pointer_get(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("invalid JSON pointer")
    for token in pointer[1:].split("/"):
        if "~" in token.replace("~1", "").replace("~0", ""):
            raise ValueError("invalid JSON pointer escape")
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not key.isdigit() or (len(key) > 1 and key[0] == "0"):
                raise KeyError(key)
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(key)
    return value


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def atomic_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("immutable delivery source collision")
        return
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ContextSourceRegistry:
    """A registered delivery handle grants bounded delivery reads only.

    It cannot manufacture native artifact IDs, science evidence or read receipts.
    The logical block identity has one current revision; older blobs remain intact.
    """
    def __init__(self, camp: Path | str, campaign_id: str | None = None):
        self.camp = Path(camp).resolve()
        self.campaign_id = campaign_id or self.camp.name
        self.root = self.camp / "context_sources"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.camp / "context_sources.sqlite"
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS context_sources (source_ref TEXT PRIMARY KEY, block_id TEXT NOT NULL, campaign_id TEXT NOT NULL, source_hash TEXT NOT NULL, payload_ref TEXT NOT NULL, block_json TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS context_source_heads (campaign_id TEXT NOT NULL, block_id TEXT NOT NULL, source_ref TEXT NOT NULL, PRIMARY KEY(campaign_id,block_id))")
            db.execute("CREATE INDEX IF NOT EXISTS context_sources_block ON context_sources(campaign_id,block_id)")
            db.execute("CREATE TABLE IF NOT EXISTS context_delivery_reads (read_id TEXT PRIMARY KEY, source_ref TEXT NOT NULL, scope_json TEXT NOT NULL, result_hash TEXT NOT NULL)")

    def _db(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def capture(self, payload: Any, *, block_id: str, kind: str = "unclassified", revision: str = "",
                role: str = "", visibility: tuple[str, ...] = ("campaign",),
                identity: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None,
                group_id: str = "", event_ids: tuple[str, ...] = (), status: str = "captured",
                dependency_refs: tuple[str, ...] = ()) -> SourceBlock:
        if not isinstance(visibility, (tuple, list)) or any(scope in PRIVATE_SCOPES for scope in visibility) or ("campaign" not in visibility and role not in visibility):
            raise PermissionError("source visibility forbids capture")
        _assert_visible_payload(payload, role)
        _assert_visible_payload(identity or {}, role)
        _assert_visible_payload(metadata or {}, role)
        text = canonical(payload)
        hash_value = digest(payload)
        source_revision = revision or hash_value
        source_ref = "context://" + quote(self.campaign_id, safe="") + "/" + quote(block_id, safe="") + "@" + quote(source_revision, safe="") + "#" + hash_value
        path = self.root / "raw" / (hash_value + ".json")
        # Body must be durable before registry publish. A crash leaves only an
        # unreferenced immutable blob, never a registered missing artifact.
        atomic_immutable(path, text)
        bounded_metadata = dict(metadata or {})
        if len(canonical(bounded_metadata)) > 4_000:
            raise ValueError("source metadata exceeds bounded registry card")
        block = SourceBlock(block_id, self.campaign_id, kind, source_ref, source_revision,
                            hash_value, str(path.relative_to(self.camp)), group_id=group_id,
                            event_ids=event_ids, identity=dict(identity or {}), status=status,
                            visibility=visibility, dependency_refs=dependency_refs,
                            metadata=bounded_metadata)
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO context_sources(source_ref,block_id,campaign_id,source_hash,payload_ref,block_json) VALUES(?,?,?,?,?,?)",
                       (source_ref, block_id, self.campaign_id, hash_value, block.payload_ref, canonical(asdict(block))))
            db.execute("INSERT INTO context_source_heads(campaign_id,block_id,source_ref) VALUES(?,?,?) ON CONFLICT(campaign_id,block_id) DO UPDATE SET source_ref=excluded.source_ref",
                       (self.campaign_id, block_id, source_ref))
        return block

    def get(self, source_ref: str, *, role: str = "", require_current: bool = True) -> SourceBlock:
        with self._db() as db:
            row = db.execute("SELECT block_json, revoked FROM context_sources WHERE source_ref=? AND campaign_id=?", (source_ref, self.campaign_id)).fetchone()
            if row is None:
                raise KeyError("unregistered_source")
            body = json.loads(row[0])
            head = db.execute("SELECT source_ref FROM context_source_heads WHERE campaign_id=? AND block_id=?", (self.campaign_id, body["block_id"])).fetchone()
        if row[1]:
            raise ValueError("revoked_source")
        if require_current and (not head or head[0] != source_ref):
            raise ValueError("stale_source")
        for key in ("event_ids", "visibility", "dependency_refs"):
            body[key] = tuple(body.get(key, ()))
        block = SourceBlock(**body)
        if any(scope in PRIVATE_SCOPES for scope in block.visibility) or ("campaign" not in block.visibility and role not in block.visibility):
            raise PermissionError("source visibility forbids read")
        return block

    def load(self, block: SourceBlock, *, role: str = "", require_current: bool = True) -> Any:
        current = self.get(block.source_ref, role=role, require_current=require_current)
        if current.source_hash != block.source_hash:
            raise ValueError("source_hash_mismatch")
        path = (self.camp / current.payload_ref).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise PermissionError("delivery source path escapes campaign registry")
        body = json.loads(path.read_text(encoding="utf-8"))
        if digest(body) != current.source_hash:
            raise ValueError("source_hash_mismatch")
        return body

    def revoke(self, source_ref: str) -> None:
        with self._db() as db:
            db.execute("UPDATE context_sources SET revoked=1 WHERE source_ref=? AND campaign_id=?", (source_ref, self.campaign_id))

    def read(self, request: ReadRequest, *, role: str = "", hard_max_chars: int = 8_000) -> ReadResult:
        if request.max_chars <= 0 or request.max_chars > hard_max_chars:
            return ReadResult("blocked", reason="read_page_budget_exceeded")
        try:
            block = self.get(request.source_ref, role=role)
            if block.source_hash != request.source_hash:
                return ReadResult("stale", reason="requested_hash_does_not_match")
            selected = pointer_get(self.load(block, role=role), request.json_pointer)
            if request.unit in {"chars", "lines"}:
                if not isinstance(selected, str):
                    raise ValueError("text_range_requires_string")
                size = len(selected) if request.unit == "chars" else len(selected.splitlines(keepends=True))
                start, end = request.start or 0, request.end if request.end is not None else size
                if start < 0 or end < start or end > size:
                    raise ValueError("invalid_read_range")
                selected = selected[start:end] if request.unit == "chars" else "".join(selected.splitlines(keepends=True)[start:end])
            elif request.unit == "items":
                if not isinstance(selected, list):
                    raise ValueError("item_range_requires_list")
                start, end = request.start or 0, request.end if request.end is not None else len(selected)
                if start < 0 or end < start or end > len(selected):
                    raise ValueError("invalid_read_range")
                selected = selected[start:end]
            elif request.unit != "json":
                raise ValueError("unsupported_read_unit")
            if len(canonical(selected)) > request.max_chars:
                return ReadResult("needs_smaller_page", reason="selected_content_exceeds_read_budget")
            scope = {"json_pointer": request.json_pointer, "unit": request.unit,
                     "start": request.start, "end": request.end, "max_chars": request.max_chars,
                     "source_hash": block.source_hash, "fidelity": "PARTIAL"}
            read_id = digest({"source": block.source_ref, "scope": scope})
            output = self.capture(selected, block_id="read:" + read_id, kind="delivery_read",
                                  role=role, identity=block.identity, metadata={"read_scope": scope},
                                  dependency_refs=(block.source_ref,))
            with self._db() as db:
                db.execute("INSERT OR IGNORE INTO context_delivery_reads VALUES(?,?,?,?)", (read_id, block.source_ref, canonical(scope), output.source_hash))
            return ReadResult("ready", selected, output, scope)
        except PermissionError as exc:
            return ReadResult("forbidden", reason=str(exc))
        except (KeyError, ValueError, OSError, json.JSONDecodeError, IndexError) as exc:
            text = str(exc)
            return ReadResult("stale" if "stale" in text or "hash" in text or "revoked" in text else "unreadable", reason=text)

    def index(self, *, limit: int = 32, offset: int = 0, role: str = "") -> list[dict[str, Any]]:
        if limit < 0 or limit > 64 or offset < 0:
            raise ValueError("source index must be bounded")
        with self._db() as db:
            refs = db.execute("SELECT source_ref FROM context_source_heads WHERE campaign_id=? ORDER BY block_id LIMIT ? OFFSET ?", (self.campaign_id, limit, offset)).fetchall()
        result = []
        for (ref,) in refs:
            try:
                block = self.get(ref, role=role)
            except (PermissionError, ValueError):
                continue
            result.append({"block_id": block.block_id, "source_ref": block.source_ref,
                           "source_hash": block.source_hash, "kind": block.kind, "status": block.status})
        return result
