"""Mechanical review evidence checks; scientific judgment remains with Reviewer."""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Any


def _symbol_line(source: str, symbol: str) -> int | None:
    """Resolve definitions and direct module/class assignment-name leaves.

    Assignment targets must be simple Names; destructuring, attribute targets
    and function-local bindings are not exposed as module/class symbols.
    """
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parts = [part for part in re.split(r"::|\.", symbol) if part]
    scope: list[ast.stmt] = list(tree.body)
    assignment_scope = True
    line = None
    for index, part in enumerate(parts):
        node = next((item for item in scope if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and item.name == part), None)
        if node is None:
            if assignment_scope and index == len(parts) - 1:
                for item in scope:
                    targets = (item.targets if isinstance(item, ast.Assign) else
                               [item.target] if isinstance(item, ast.AnnAssign) else [])
                    if any(isinstance(target, ast.Name) and target.id == part for target in targets):
                        return item.lineno
            return None
        line, scope = node.lineno, list(node.body)
        assignment_scope = isinstance(node, ast.ClassDef)
    return line


def _symbol_lines(source: str, symbol: str) -> tuple[int, ...] | None:
    """Resolve a symbol or a shared-scope shorthand such as Class.init/forward."""
    members = symbol.split("/")
    symbols = [symbol]
    if len(members) > 1:
        first = re.split(r"::|\.", members[0])
        if len(first) < 2 or not all(part.isidentifier() for part in first):
            return None
        if not all(member.isidentifier() for member in members[1:]):
            return None
        prefix = ".".join(first[:-1])
        symbols = [members[0], *(prefix + "." + member for member in members[1:])]
    lines = tuple(_symbol_line(source, item) for item in symbols)
    return None if any(line is None for line in lines) else lines


def valid_source_reference(reference: Any, allowed: set, source: str | None) -> bool:
    """`path`, `path:N`, `path#LN`, or `path::Class::method` resolving in the current source."""
    if not isinstance(reference, str):
        return False
    source_lines = len(source.splitlines()) if isinstance(source, str) else 0
    path, sep, symbol = reference.partition("::")
    if sep:
        return path in allowed and isinstance(source, str) and _symbol_lines(source, symbol) is not None
    match = re.fullmatch(r"(.+?)(?::([0-9]+)|#L([0-9]+))?", reference)
    if not match or match.group(1) not in allowed or source_lines == 0:
        return False
    line = match.group(2) or match.group(3)
    return line is None or 1 <= int(line) <= source_lines


def _request_value(request: dict, key: str):
    from .context.representations import resolve_request_pointer
    from .context.sources import pointer_escape
    def expand(value, pointer):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(request, pointer), pointer)
            return {name: expand(child, pointer + "/" + pointer_escape(name)) for name, child in value.items()}
        if isinstance(value, list):
            return [expand(child, pointer + "/" + str(index)) for index, child in enumerate(value)]
        return value
    try:
        return expand(resolve_request_pointer(request, "/" + key), "/" + key)
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _delivered_file_reference(reference: str, row: dict) -> bool:
    """A file hash verifies identity; only the delivered text verifies scope."""
    source_ref = row.get("source_ref")
    text = row.get("source") if "source" in row else row.get("excerpt")
    if not isinstance(source_ref, str) or not isinstance(text, str) or not text:
        return False
    path, separator, symbol = reference.partition("::")
    match = re.fullmatch(r"(.+?)(?::([0-9]+)|#L([0-9]+))?", path)
    if not match or match.group(1) != source_ref:
        return False
    try:
        original = Path(source_ref).read_bytes()
        if hashlib.sha256(original).hexdigest() != row.get("source_hash"):
            return False
        current = original.decode("utf-8")
    except (OSError, UnicodeError, ValueError):
        return False
    ranges, cursor = [], 0
    # Runtime excerpts concatenate selected source sections. Match each literal
    # section in order, without granting access to the intervening omitted code.
    sections = [text] if "source" in row else text.split("\n\n")
    for section in sections:
        if not section.strip():
            continue
        start = current.find(section, cursor)
        if start < 0:
            return False
        cursor = start + len(section)
        ranges.append((start, cursor))
    if not ranges:
        return False
    if separator:
        symbol_lines = _symbol_lines(current, symbol)
        if symbol_lines is None:
            return False
        lines = current.splitlines(keepends=True)
        for line in symbol_lines:
            start = sum(len(value) for value in lines[:line - 1])
            end = start + len(lines[line - 1].rstrip("\r\n"))
            if not any(left <= start and end <= right for left, right in ranges):
                return False
        return True
    line = match.group(2) or match.group(3)
    if line is None:
        return True  # This cites the supplied material, not a full-file read.
    number = int(line)
    lines = current.splitlines(keepends=True)
    if not 1 <= number <= len(lines):
        return False
    start = sum(len(value) for value in lines[:number - 1])
    end = start + len(lines[number - 1].rstrip("\r\n"))
    return any(current[max(start, left):min(end, right)].strip()
               for left, right in ranges if max(start, left) < min(end, right))


def valid_review_reference(reference: Any, request: dict) -> bool:
    """Resolve candidate/check and actually delivered parent/runtime references."""
    if not isinstance(reference, str):
        return False
    identity = _request_value(request, "requirement_evidence_identity") or {}
    source = _request_value(request, "candidate_source")
    if valid_source_reference(reference, {identity.get("source_ref"), "extension/eeg_candidate.py"}, source):
        return True
    if reference in {identity.get("check_ref"), "checks.json"}:
        try:
            return hashlib.sha256(Path(identity["check_ref"]).read_bytes()).hexdigest() == identity.get("check_sha256")
        except (OSError, KeyError, TypeError, ValueError):
            return False
    parent = _request_value(request, "parent_source")
    runtime = _request_value(request, "training_runtime_source")
    rows = ([parent] if isinstance(parent, dict) else []) + (runtime if isinstance(runtime, list) else [])
    return any(_delivered_file_reference(reference, row) for row in rows if isinstance(row, dict))


def review_reference_errors(reply: dict[str, Any], request: dict[str, Any]) -> list[dict]:
    """Check only injected evidence metadata, without changing scientific claims."""
    identity = _request_value(request, "requirement_evidence_identity")
    if not isinstance(identity, dict):
        return []  # Legacy reviews without a bound evidence contract.
    source = _request_value(request, "candidate_source")
    source_lines = len(source.splitlines()) if isinstance(source, str) else 0
    allowed_checks = {identity.get("check_ref"), "checks.json"}
    errors = []

    def error(path, message, expected):
        errors.append({"type": "review_evidence_reference_invalid", "loc": path,
                       "msg": message, "ctx": {"expected": expected}})

    for index, row in enumerate(reply.get("requirement_coverage") or []):
        if not isinstance(row, dict) or row.get("status") not in {"implemented", "contradicted"}:
            continue
        base = ["requirement_coverage", index]
        for field in ("source_hash", "check_sha256"):
            if row.get(field) != identity.get(field):
                error(base + [field], "Copy the exact injected current identity.", identity.get(field))
        if row.get("check_ref") not in allowed_checks:
            error(base + ["check_ref"], "Use the injected current check path.", identity.get("check_ref"))
        refs = row.get("source_refs") or []
        if not refs:
            error(base + ["source_refs"], "A current source reference is required for this claim.", identity.get("source_ref"))
        for position, reference in enumerate(refs):
            if not valid_review_reference(reference, request):
                error(base + ["source_refs", position],
                      "Use a current bound candidate/check or a delivered parent/runtime source; line and symbol references must lie in delivered material.",
                      {"source_ref": identity.get("source_ref"), "line_count": source_lines})
    return errors
