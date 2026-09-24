"""Source identity. A score is comparable only when the loaded file matches the manifest."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest_for(workspace: Path, module: str, entry: Path) -> dict[str, Any]:
    """Hash the candidate entry the trainer must import."""
    return {
        "module": module,
        "entry": str(entry.resolve()),
        "entry_sha256": file_sha256(entry),
        "workspace": str(workspace.resolve()),
    }


def write_binding(out_dir: Path, candidate: Any) -> dict[str, Any]:
    """Record the class file the running process actually imported."""
    path = Path(inspect.getfile(type(candidate))).resolve()
    payload = {
        "module": type(candidate).__module__,
        "class_file": str(path),
        "file_sha256": file_sha256(path),
        "candidate_id": getattr(candidate, "candidate_id", ""),
    }
    (out_dir / "source_binding.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def binding_accepts(job_dir: Path, manifest: dict[str, Any]) -> tuple[bool, str]:
    """Refuse a score when training imported a different file than the reviewed candidate."""
    path = job_dir / "source_binding.json"
    if not path.is_file():
        return False, "binding_missing"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded.get("module") != manifest.get("module"):
        return False, "candidate_not_loaded"
    if loaded.get("file_sha256") != manifest.get("entry_sha256"):
        return False, "source_hash_mismatch"
    return True, "evaluation_valid"
