"""Artifact registry. Role handoff uses files and hashes, not in-memory dicts."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

REGISTRY = "artifact_registry.jsonl"


def file_digest(path: Path) -> str:
    """Return the sha256 of a file. Missing files are an error."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


VOLATILE_REQUEST_KEYS = {
    "task_id",
    "attempt_id",
    "request_id",
    "started_at",
    "created_at",
    "ended_at",
    "at",
    "nonce",
}


def input_digest(paths: list[Path]) -> str:
    """Stable digest of input artifacts. Order is normalized by path."""
    rows = []
    for path in sorted(paths, key=lambda item: str(item)):
        rows.append({"path": str(path), "sha256": file_digest(path) if path.is_file() else None, "missing": not path.is_file()})
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_request(value: Any) -> Any:
    """Drop cyclic metadata so a digest covers business content only."""
    if isinstance(value, dict):
        return {key: stable_request(item) for key, item in value.items() if key not in VOLATILE_REQUEST_KEYS}
    if isinstance(value, list):
        return [stable_request(item) for item in value]
    return value


def request_digest(*, request: dict[str, Any] | None = None, artifacts: list[dict[str, Any]] | None = None, paths: list[Path] | None = None) -> str:
    """Hash the normalized request plus {id, kind, content hash, schema} refs."""
    manifest: list[dict[str, Any]] = []
    for row in artifacts or []:
        if not isinstance(row, dict):
            continue
        manifest.append(
            {
                "artifact_id": row.get("artifact_id"),
                "kind": row.get("kind"),
                "content_sha256": row.get("content_sha256") or row.get("sha256"),
                "schema_version": row.get("schema_version"),
            }
        )
    for path in paths or []:
        manifest.append(
            {
                "artifact_id": None,
                "kind": "file",
                "content_sha256": file_digest(path) if path.is_file() else None,
                "schema_version": None,
                "path": str(path),
            }
        )
    body = {
        "request": stable_request(request or {}),
        "artifacts": sorted(manifest, key=lambda row: (str(row.get("artifact_id") or ""), str(row.get("path") or ""), str(row.get("content_sha256") or ""))),
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def allocate_artifact_id() -> str:
    """Reserve an id before the bytes are written. Registration must not rewrite them."""
    return f"art_{uuid.uuid4().hex[:12]}"


def register(
    camp: Path,
    path: Path,
    *,
    kind: str,
    producer_task_id: str | None = None,
    candidate_id: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Record one on-disk artifact. The hash is taken from the file at registration time."""
    digest = file_digest(path)
    row = {
        "artifact_id": artifact_id or allocate_artifact_id(),
        "kind": kind,
        "path": str(path.resolve()),
        "relative": str(path.resolve().relative_to(camp.resolve())) if _under(path, camp) else None,
        "sha256": digest,
        "producer_task_id": producer_task_id,
        "candidate_id": candidate_id,
        "at": time.time(),
    }
    with (camp / REGISTRY).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def lookup(camp: Path, artifact_id: str) -> dict[str, Any] | None:
    """Return the registered row. A missing registry is empty, not an error."""
    path = camp / REGISTRY
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("artifact_id") == artifact_id:
            return row
    return None


def verify(camp: Path, artifact_id: str) -> tuple[bool, str]:
    """True when the file still matches the registered hash."""
    row = lookup(camp, artifact_id)
    if row is None:
        return False, "artifact_unknown"
    path = Path(str(row["path"]))
    if not path.is_file():
        return False, "artifact_missing"
    if file_digest(path) != row.get("sha256"):
        return False, "artifact_hash_mismatch"
    return True, "ok"


def resolve_verified_artifact(camp: Path, artifact_id: str) -> dict[str, Any]:
    """Return a registry row only after the bytes still match. A path alone is not a read."""
    ok, reason = verify(camp, artifact_id)
    if not ok:
        raise FileNotFoundError(reason)
    row = lookup(camp, artifact_id)
    if row is None:
        raise FileNotFoundError("artifact_unknown")
    return row


def read_verified_range(camp: Path, artifact_id: str, *, start: int = 0, end: int | None = None,
                        json_view: Callable[[Any], Any] | None = None,
                        max_chars: int | None = None) -> dict[str, Any]:
    """Read a bounded slice of one registered artifact. Unknown ids are refused."""
    row = resolve_verified_artifact(camp, artifact_id)
    text = Path(str(row["path"])).read_text(encoding="utf-8")
    view_hash = None
    if json_view is not None:
        value = json_view(json.loads(text))
        if value is None:
            raise ValueError("artifact_scope_forbidden")
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        view_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if start < 0 or start > len(text) or (end is not None and end < start):
            raise ValueError("artifact_read_range_invalid")
        # Callers know the read budget, not the length of the filtered view.
        # Return the available prefix and bind the actual range in the receipt.
        end = min(len(text), len(text) if end is None else end)
        if max_chars is not None:
            end = min(end, start + max_chars)
    stop = len(text) if end is None else max(start, end)
    chunk = text[start:stop]
    return {
        "artifact_id": artifact_id,
        "sha256": row.get("sha256"),
        "view_sha256": view_hash,
        "range_scope": "development_view" if json_view is not None else "original_artifact",
        "start": start,
        "end": start + len(chunk),
        "truncated": start + len(chunk) < len(text),
        "next_range": None if start + len(chunk) >= len(text) else [start + len(chunk), len(text)],
        "text": chunk,
    }


def _under(path: Path, camp: Path) -> bool:
    try:
        path.resolve().relative_to(camp.resolve())
    except ValueError:
        return False
    return True
