"""Mechanical review evidence checks; scientific judgment remains with Reviewer."""
from __future__ import annotations

import re
from typing import Any


def _symbol_line(source: str, symbol: str) -> int | None:
    """Line of `Class::method`, `Class.method` or `func` in the current source."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parts = [part for part in re.split(r"::|\.", symbol) if part]
    scope: list[ast.stmt] = list(tree.body)
    line = None
    for part in parts:
        node = next((item for item in scope if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and item.name == part), None)
        if node is None:
            return None
        line, scope = node.lineno, list(node.body)
    return line


def valid_source_reference(reference: Any, allowed: set, source: str | None) -> bool:
    """`path`, `path:N`, `path#LN`, or `path::Class::method` resolving in the current source."""
    if not isinstance(reference, str):
        return False
    source_lines = len(source.splitlines()) if isinstance(source, str) else 0
    path, sep, symbol = reference.partition("::")
    if sep:
        return path in allowed and isinstance(source, str) and _symbol_line(source, symbol) is not None
    match = re.fullmatch(r"(.+?)(?::([0-9]+)|#L([0-9]+))?", reference)
    if not match or match.group(1) not in allowed or source_lines == 0:
        return False
    line = match.group(2) or match.group(3)
    return line is None or 1 <= int(line) <= source_lines


def review_reference_errors(reply: dict[str, Any], request: dict[str, Any]) -> list[dict]:
    """Check only injected evidence metadata, without changing scientific claims."""
    identity = request.get("requirement_evidence_identity")
    if not isinstance(identity, dict):
        return []  # Legacy reviews without a bound evidence contract.
    source = request.get("candidate_source")
    source_lines = len(source.splitlines()) if isinstance(source, str) else 0
    allowed_sources = {identity.get("source_ref"), "extension/eeg_candidate.py"}
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
            if not valid_source_reference(reference, allowed_sources, source):
                error(base + ["source_refs", position],
                      "Use a current file path, optionally :line, #Lline or ::Class::method that exists in the injected source.",
                      {"source_ref": identity.get("source_ref"), "line_count": source_lines})
    return errors
