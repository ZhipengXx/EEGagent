"""Readonly source aliases and small source views for native Coder errors.

The training dependency in ``coder.py`` remains unchanged. Absolute traceback
paths are accepted only when they name an explicitly registered source; source
lookup never turns a traceback into general filesystem access.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Any

from . import coder


_PROBE_REFERENCES = {
    "reference/multigpu_probe.py": Path(__file__).resolve().with_name("multigpu_probe.py"),
    "reference/multigpu_check.py": Path(__file__).resolve().with_name("multigpu_check.py"),
}
_CANDIDATE = "extension/eeg_candidate.py"
_FRAME = re.compile(r'File ["\'](?P<file>[^"\'\n]+)["\'], line (?P<line>\d+)(?:, in (?P<function>[^\n]+))?')


def _parent_source(workspace: Path) -> Path | None:
    target = workspace.resolve() / "reference" / "parent.py"
    # Preserve the native reader's already registered parent alias, including
    # parent snapshots represented by a symlink.
    return target if target.is_file() else None


def reference_paths(workspace: Path) -> dict[str, Path]:
    """Return a fresh alias registry; never modify the frozen reference map."""
    references = {**coder._REFERENCES, **_PROBE_REFERENCES}
    parent = _parent_source(workspace)
    if parent is not None:
        references["reference/parent.py"] = parent
    return references


def canonical_read_path(workspace: Path, path: str) -> str | None:
    """Map a registered absolute source to its alias, rejecting other absolutes.

    Relative extension paths retain the native reader's existing restrictions.
    In particular, resolving an unregistered symlink to a registered file does
    not register the symlink itself.
    """
    if not isinstance(path, str) or not path:
        return None
    references = reference_paths(workspace)
    if path in references:
        return path
    supplied = Path(path)
    if supplied.is_absolute():
        # Do not lexically collapse ``symlink/..``: that can denote a different
        # file on disk. The exact registered spelling is the absolute alias.
        absolute = supplied
        for alias, target in references.items():
            if absolute == target.absolute():
                return alias
        candidate = workspace.resolve() / _CANDIDATE
        if absolute == candidate and candidate.is_file() and coder._allowed(workspace, candidate):
            return _CANDIDATE
        return None
    target = (workspace / path).resolve()
    if coder._allowed(workspace, target) and target.is_file():
        return str(target.relative_to(workspace.resolve()))
    return None


def source_target(workspace: Path, path: str) -> Path | None:
    """Resolve a readable source for hashing or cache freshness checks."""
    canonical = canonical_read_path(workspace, path)
    if canonical is None:
        return None
    target = reference_paths(workspace).get(canonical) or (workspace / canonical)
    return target if target.is_file() else None


def source_snapshot(workspace: Path, path: str) -> dict[str, str] | None:
    """Bind an allowed source's text and revision to the same bytes."""
    canonical = canonical_read_path(workspace, path)
    target = source_target(workspace, path)
    if canonical is None or target is None:
        return None
    content = target.read_bytes()
    return {"path": canonical, "source_sha256": hashlib.sha256(content).hexdigest(),
            "text": content.decode("utf-8")}


def read_code(workspace: Path, path: str, start: int = 1, end: int = 200) -> dict[str, Any]:
    """Read the same complete-line pages as the original native source reader."""
    snapshot = source_snapshot(workspace, path)
    if snapshot is None:
        return {"ok": False, "error": "read_refused"}
    lines = snapshot["text"].splitlines()
    return {"ok": True, "path": snapshot["path"], "source_sha256": snapshot["source_sha256"],
            **coder.source_page(lines, max(1, int(start)), int(end))}


def _diagnostic_frames(check: dict[str, Any]):
    """Traverse native CPU and nested supervisor/probe diagnostic payloads."""
    pending: list[tuple[tuple[Any, ...], Any]] = [((), check)]
    while pending:
        location, value = pending.pop()
        if isinstance(value, dict):
            if isinstance(value.get("file"), str) and type(value.get("line")) is int:
                yield location, value["file"], value["line"], value.get("function")
            pending.extend((location + (key,), child) for key, child in reversed(list(value.items())))
        elif isinstance(value, list):
            pending.extend((location + (index,), child) for index, child in reversed(list(enumerate(value))))
        elif isinstance(value, str):
            for match in _FRAME.finditer(value):
                yield location, match["file"], int(match["line"]), match["function"]


def traceback_sources(workspace: Path, check: dict[str, Any] | None, *, max_files: int = 3,
                      char_limit: int = 8000) -> dict[str, Any]:
    """Return deepest relevant frame per file and bounded readonly source pages.

    Raw check results and receipts stay untouched. A successful check adds no
    context, and third-party/unregistered paths add no source views.
    """
    if not isinstance(check, dict) or check.get("ok") is not False:
        return {}
    frames: list[dict[str, Any]] = []
    for diagnostic_path, original, line, function in _diagnostic_frames(check):
        canonical = canonical_read_path(workspace, original)
        if canonical is not None and line > 0:
            frame = {"path": canonical, "original_path": original, "line": line,
                     "diagnostic_path": list(diagnostic_path)}
            if function:
                frame["function"] = function
            frames.append(frame)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame in reversed(frames):
        if frame["path"] not in seen:
            selected.append(frame)
            seen.add(frame["path"])
        if len(selected) >= max(0, max_files):
            break
    if not selected or max_files <= 0:
        return {}
    pages: list[dict[str, Any]] = []
    used = 0
    for frame in selected:
        line = frame["line"]
        page = read_code(workspace, frame["path"], max(1, line - 5), line + 10)
        if not page.get("ok") or not page["start"] <= line <= page["end"]:
            continue
        numbered = "\n".join(f"{page['start'] + index}: {text}" for index, text in enumerate(page["text"].splitlines()))
        # The native pager retains whole lines, including an oversized first
        # line. Keep that behavior for reads, while bounding injected context.
        size = len(page["text"]) + len(numbered)
        if used + size > max(0, char_limit):
            continue
        pages.append({**page, "numbered_text": numbered, "error_line": line})
        used += size
    return {"frames": selected, "path_aliases": {frame["original_path"]: frame["path"] for frame in selected},
            "pages": pages}
