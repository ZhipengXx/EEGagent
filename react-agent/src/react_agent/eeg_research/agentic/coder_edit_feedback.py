"""Readonly, version-bound source evidence after native Coder edits.

The native editor remains the authority for exact matching and atomic writes.
Snapshots are supplied by its caller; this module does not read or edit files.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from typing import Any

from .coder import source_page


_VERSION = "eeg_research.coder_edit_feedback.v1"
_PATH = "extension/eeg_candidate.py"
_PAGE_CHARS = 2200
_MATCH_ERRORS = {"edit_not_found", "edit_not_unique"}


def _snapshot(value: Any, path: str) -> dict[str, str] | None:
    if not isinstance(value, dict) or value.get("path") != path or not isinstance(value.get("text"), str):
        return None
    digest = hashlib.sha256(value["text"].encode("utf-8")).hexdigest()
    if value.get("source_sha256") != digest:
        return None
    return {"path": path, "text": value["text"], "source_sha256": digest}


def _page(snapshot: dict[str, str], start: int = 1, end: int | None = None) -> dict[str, Any]:
    lines = snapshot["text"].splitlines()
    start = max(1, min(start, max(1, len(lines))))
    return {"path": snapshot["path"], "source_sha256": snapshot["source_sha256"],
            **source_page(lines, start, end or start + 23, char_limit=_PAGE_CHARS)}


def _focused_page(snapshot: dict[str, str], focus: int) -> dict[str, Any]:
    page = _page(snapshot, max(1, focus - 3))
    # Context lines must not crowd the edited/matched line out of the payload.
    return _page(snapshot, focus) if page["end"] < focus and page["total_lines"] else page


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _exact_hits(text: str, old: str) -> tuple[list[dict[str, int]], bool]:
    """Use native str.count's non-overlapping semantics, showing at most 8 hits."""
    hits = []
    offset = 0
    while len(hits) < 8:
        found = text.find(old, offset)
        if found < 0:
            return hits, False
        hits.append({"start": _line(text, found),
                     "end": _line(text, found + max(0, len(old.rstrip("\r\n")) - 1))})
        offset = found + len(old)
    return hits, text.find(old, offset) >= 0


def _nearest(text: str, old: str) -> dict[str, Any] | None:
    """Find one diagnostic window; never use approximate text for a write.

    Large fragments receive an ordinary source page instead of an expensive
    fuzzy search. No whitespace normalization or staged batch replay is used.
    """
    wanted = old.splitlines()
    lines = text.splitlines()
    if not wanted or not lines or len(wanted) > 40 or len(old) > 4000:
        return None
    anchors = [(len(line.strip()), index, line) for index, line in enumerate(wanted)
               if line.strip() and len(line) <= 1000]
    if not anchors:
        return None
    _, anchor_index, anchor = max(anchors)
    available = sorted({line for line in lines if len(line) <= 1000})
    close = set(difflib.get_close_matches(anchor, available, n=3, cutoff=0.5))
    starts = {max(0, index - anchor_index) for index, line in enumerate(lines) if line in close}
    best: tuple[float, int, int, str] | None = None
    for start in sorted(starts):
        end = min(len(lines), start + len(wanted))
        chunk = "\n".join(lines[start:end])
        if len(chunk) > 6000:
            continue
        score = difflib.SequenceMatcher(None, old, chunk, autojunk=True).ratio()
        if score >= 0.5 and (best is None or score > best[0]):
            best = score, start, end, chunk
    if best is None:
        return None
    score, start, end, chunk = best
    # Drop the optional diff as a whole when it is too large; never slice code.
    diff_lines = list(difflib.unified_diff(wanted, chunk.splitlines(),
                                         fromfile="requested_old", tofile="observed_source", n=2,
                                         lineterm=""))
    difference = "\n".join(diff_lines)
    return {"approximate": True, "start": start + 1, "end": end,
            "similarity": round(score, 3),
            **({"diff": difference} if len(difference) <= 1000 else {})}


def build_edit_feedback(tool: str, args: dict[str, Any], result: dict[str, Any],
                        before_snapshot: dict[str, Any] | None,
                        after_snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return bounded top-level log fields for a real native edit result.

    Snapshot text must preserve the original UTF-8 bytes (read bytes then decode,
    rather than newline-translating ``read_text``). Unknown or inconsistent
    evidence produces no feedback; the original native result is never changed.
    """
    if tool not in {"edit_candidate_code", "apply_candidate_patch"} or args.get("path") != _PATH:
        return {}
    before = _snapshot(before_snapshot, _PATH)
    after = _snapshot(after_snapshot, _PATH)
    if after is None:
        return {}
    base = {"schema_version": _VERSION, "path": _PATH, "source_sha256": after["source_sha256"]}
    if result.get("ok") is True:
        if (result.get("already_applied") or result.get("changed") is False
                or result.get("path") != _PATH
                or result.get("sha256") != after["source_sha256"]
                or (before is not None and args.get("expected_base_hash") != before["source_sha256"])
                or (before is not None and before["source_sha256"] == after["source_sha256"])):
            return {}
        if before is None and tool != "apply_candidate_patch":
            return {}
        previous = before["text"].splitlines() if before is not None else []
        current = after["text"].splitlines()
        prefix = 0
        while prefix < min(len(previous), len(current)) and previous[prefix] == current[prefix]:
            prefix += 1
        suffix = 0
        while (suffix < min(len(previous), len(current)) - prefix
               and previous[len(previous) - suffix - 1] == current[len(current) - suffix - 1]):
            suffix += 1
        first = min(prefix + 1, max(1, len(current)))
        last = max(first, len(current) - suffix)
        context = {**base, "changed": True,
                   "changed_span": {"start": first, "end": last},
                   "source_page": _focused_page(after, first)}
        if before is not None:
            context["base_sha256"] = before["source_sha256"]
        return {"edit_success_context": context}

    error = result.get("error")
    if result.get("ok") is not False or error not in _MATCH_ERRORS | {"base_hash_mismatch"}:
        return {}
    # All supported native failures are atomic. A disk change during execution
    # invalidates matching evidence; do not project stale locations onto it.
    if before is None or before["source_sha256"] != after["source_sha256"]:
        return {}
    if result.get("current_sha256") != after["source_sha256"]:
        return {}
    context = {**base, "error": error, "committed_edits": 0}
    if error == "base_hash_mismatch":
        context["requested_base_sha256"] = args.get("expected_base_hash")
        context["source_page"] = _page(after)
        return {"edit_failure_context": context}
    index = result.get("edit_index")
    edits = args.get("edits")
    if type(index) is not int or index < 0 or not isinstance(edits, list) or index >= len(edits):
        return {}
    edit = edits[index]
    if not isinstance(edit, dict) or not isinstance(edit.get("old"), str) or not edit["old"]:
        return {}
    context["failed_edit_index"] = index
    matches = result.get("matches")
    if type(matches) is not int or matches < 0:
        return {}
    context["native_matches"] = matches
    # Native edit_candidate_code reads with Path.read_text's universal-newline
    # behavior. Match diagnostics use that same working text, while the source
    # hash remains bound to the original disk bytes. The requested old string
    # and all other whitespace are left unchanged.
    native_text = after["text"].replace("\r\n", "\n").replace("\r", "\n")
    if index > 0:
        context["match_scope"] = "native_staged_buffer"
        context["source_page_scope"] = "unchanged_disk_source"
        context["detail"] = "Earlier edits existed only in memory; the whole batch committed zero edits. The page is unchanged disk source, not the staged failure location."
        context["source_page"] = _page(after)
    elif error == "edit_not_unique":
        if native_text.count(edit["old"]) != matches or matches <= 1:
            return {}
        hits, more = _exact_hits(native_text, edit["old"])
        context.update({"match_scope": "unchanged_disk_source", "exact_hits": hits,
                        "more_hits": more,
                        "source_page": _focused_page(after, hits[0]["start"]) if hits else _page(after)})
    else:
        if result.get("matches") != 0 or edit["old"] in native_text:
            return {}
        nearest = _nearest(native_text, edit["old"])
        context["match_scope"] = "unchanged_disk_source"
        if nearest is not None:
            context["candidate"] = nearest
        context["source_page"] = _focused_page(after, nearest["start"]) if nearest else _page(after)
        if (nearest is not None and "diff" in nearest
                and len(json.dumps(context, ensure_ascii=False)) > 3200):
            nearest.pop("diff")
    return {"edit_failure_context": context}


def valid_feedback(value: Any) -> bool:
    """Recognize our logged evidence; no inference from display-shortened rows."""
    return (isinstance(value, dict) and value.get("schema_version") == _VERSION
            and value.get("path") == _PATH
            and isinstance(value.get("source_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]) is not None
            and isinstance(value.get("source_page"), dict)
            and value["source_page"].get("path") == _PATH
            and value["source_page"].get("source_sha256") == value["source_sha256"])
