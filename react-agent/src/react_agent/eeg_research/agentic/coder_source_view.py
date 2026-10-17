"""Exact native source pages in the Coder's final, request-local view.

These helpers inspect delivery only. They do not authorize a read, execute a
tool, or turn a context summary into evidence that source was delivered.
"""
from __future__ import annotations

from typing import Any

from .context.representations import resolve_request_pointer
from .context.sources import pointer_escape


def source_page_lines(page: Any) -> list[str] | None:
    """Recognize a bounded native page, including the current_file hash alias."""
    if not isinstance(page, dict) or page.get("ok") is False:
        return None
    if (not isinstance(page.get("path"), str) or not page["path"]
            or not isinstance(page.get("source_sha256", page.get("sha256")), str)
            or not page.get("source_sha256", page.get("sha256"))
            or type(page.get("start")) is not int or type(page.get("end")) is not int
            or page["start"] < 1 or page["end"] < page["start"] - 1
            or not isinstance(page.get("text"), str)):
        return None
    count = page["end"] - page["start"] + 1
    lines = [] if count == 0 and page["text"] == "" else page["text"].split("\n")
    return lines if len(lines) == count else None


def source_page_visible(payload: Any, page: dict[str, Any]) -> bool:
    """Whether a same-version exact page/range is in the final wire payload.

    The caller supplies a canonical native path and current cached page. A
    containing page also proves delivery of the requested range, but an old
    hash, abbreviated text, numbered-only text or registry handle does not.
    Request-local shared pointers are followed with their normal hash checks.
    """
    wanted = source_page_lines(page)
    if wanted is None or not isinstance(payload, dict):
        return False
    expected_sha = page.get("source_sha256", page.get("sha256"))
    pending = [""]
    visited: set[str] = set()
    while pending:
        pointer = pending.pop()
        if pointer in visited:
            continue
        visited.add(pointer)
        try:
            value = resolve_request_pointer(payload, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            actual = source_page_lines(value)
            if (actual is not None and value["path"] == page["path"]
                    and value.get("source_sha256", value.get("sha256")) == expected_sha
                    and value["start"] <= page["start"] and value["end"] >= page["end"]):
                offset = page["start"] - value["start"]
                if actual[offset:offset + len(wanted)] == wanted:
                    return True
            # A registry handle is not delivered source. Shared pointers were
            # resolved above; descend through their logical request location.
            if "$context_source" not in value:
                pending.extend(pointer + "/" + pointer_escape(key) for key, child in value.items()
                               if isinstance(child, (dict, list)))
        elif isinstance(value, list):
            pending.extend(pointer + "/" + str(index) for index, child in enumerate(value)
                           if isinstance(child, (dict, list)))
    return False
