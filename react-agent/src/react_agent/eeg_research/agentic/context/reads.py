"""Consume complete field reads using the registry's actual immutable proof."""
from __future__ import annotations

from typing import Iterable

from .sources import ContextSourceRegistry, pointer_get
from .types import SourceBlock


def complete_field_reads(registry: ContextSourceRegistry, supplied: Iterable[SourceBlock], *, role: str):
    coverage: dict[tuple[str, str], set[str]] = {}
    proofs: list[SourceBlock] = []
    for claimed in supplied:
        registered = registry.get(claimed.source_ref, role=role)
        if registered.source_hash != claimed.source_hash or registered.kind != "delivery_read":
            continue
        scope = registered.metadata.get("read_scope") or {}
        if scope.get("unit") != "chars" or len(registered.dependency_refs) != 1:
            continue
        original = registry.get(registered.dependency_refs[0], role=role)
        if original.source_hash != scope.get("source_hash"):
            continue
        pointer = scope.get("json_pointer")
        if not isinstance(pointer, str):
            continue
        value = pointer_get(registry.load(original, role=role), pointer)
        if (not isinstance(value, str) or scope.get("start") != 0
                or scope.get("end") != len(value)
                or registry.load(registered, role=role) != value):
            continue
        coverage.setdefault((original.source_ref, original.source_hash), set()).add(pointer)
        proofs.append(registered)
    return coverage, proofs
