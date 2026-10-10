"""Mechanical review evidence checks; scientific judgment remains with Reviewer."""
from __future__ import annotations

import re
from typing import Any


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
            match = re.fullmatch(r"(.+?)(?::([0-9]+)|#L([0-9]+))?", reference) if isinstance(reference, str) else None
            line = (match.group(2) or match.group(3)) if match else None
            if (not match or match.group(1) not in allowed_sources
                    or source_lines == 0 or line is not None and not 1 <= int(line) <= source_lines):
                error(base + ["source_refs", position],
                      "Use a current file path, optionally :line or #Lline within the injected source. ::Class.method is a code-location descriptor, not an evidence reference.",
                      {"source_ref": identity.get("source_ref"), "line_count": source_lines})
    return errors
