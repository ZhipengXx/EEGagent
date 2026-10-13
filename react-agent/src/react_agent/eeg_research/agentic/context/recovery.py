"""Finite digest-bound delivery reads; never an experiment or approval action."""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Any, Callable

from .sources import ContextSourceRegistry
from .types import Blocked, CompileRequest, CompileResult, NeedsContext, ReadResult, canonical, digest


def declared_function_targets(request: CompileRequest) -> set[str]:
    """Only explicit native task targets license code-subtask decomposition."""
    result = set()
    containers = [request.identity]
    for key in ("task", "repair_task", "context_requirements"):
        if isinstance(request.payload.get(key), dict):
            containers.append(request.payload[key])
    for item in containers:
        for key in ("target_functions", "target_methods"):
            values = item.get(key, ())
            if isinstance(values, list):
                result.update(value for value in values if isinstance(value, str))
        if isinstance(item.get("target_function"), str):
            result.add(item["target_function"])
    return result


def incorporate_partial_reads(request: CompileRequest, reads: tuple[ReadResult, ...], *, registry: ContextSourceRegistry) -> CompileRequest:
    """Replace a known, explicitly scoped code subtask with its actual read page.

    Unknown control/scientific fields remain required and cannot be repaired by
    inserting a generic page. The delivery page never grants a native read receipt.
    """
    from .sources import pointer_get
    payload = copy.deepcopy(request.payload)
    targets = declared_function_targets(request)
    delivered = []
    changed = False
    for reply in reads:
        if reply.status != "ready" or reply.source_block is None or not targets:
            continue
        dependencies = reply.source_block.dependency_refs
        if len(dependencies) != 1:
            continue
        source = registry.get(dependencies[0], role=request.role)
        if source.kind != "code" or reply.delivered_scope.get("unit") != "lines":
            continue
        original = pointer_get(registry.load(source, role=request.role), reply.delivered_scope["json_pointer"])
        from .representations import code_outline
        exact = [row for row in code_outline(original) if row["name"] in targets
                 and row["line_start"] == reply.delivered_scope.get("start")
                 and row["line_end"] == reply.delivered_scope.get("end")]
        if len(exact) != 1:
            continue
        pointer = source.metadata.get("payload_pointer", "") + reply.delivered_scope["json_pointer"]
        if not pointer.startswith("/"):
            continue
        tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
        parent: Any = payload
        for token in tokens[:-1]:
            parent = parent[int(token)] if isinstance(parent, list) else parent[token]
        current = parent[int(tokens[-1])] if isinstance(parent, list) else parent[tokens[-1]]
        prior_pages = current.get("pages", []) if isinstance(current, dict) and current.get("fidelity") == "PARTIAL" else []
        pages = prior_pages + [{"function": exact[0]["name"], "content": reply.content, "actual_read_scope": reply.delivered_scope}]
        partial = {"fidelity": "PARTIAL", "pages": pages, "source_ref": source.source_ref,
                   "source_hash": source.source_hash, "actual_read_scope": reply.delivered_scope,
                   "task_functions_delivered": sorted({page["function"] for page in pages}), "whole_file_delivered": False,
                   "satisfies_development_read_receipt": False}
        if isinstance(parent, list):
            parent[int(tokens[-1])] = partial
        else:
            parent[tokens[-1]] = partial
        delivered.append(exact[0]["name"])
        changed = True
    if changed:
        payload["context_subtask"] = {"target_functions": sorted(targets), "delivered_functions": sorted(set(delivered)),
            "undeclared_file_content_is_not_read": True, "native_read_receipts_satisfied": False}
    return replace(request, payload=payload, maintenance_callback=None,
                   allow_semantic_compaction=False, allow_sync_memory_maintenance=False)


@dataclass(frozen=True)
class RecoveryResult:
    status: str
    result: CompileResult
    reads: tuple[ReadResult, ...] = ()
    read_chars: int = 0


def recover_context(compiler, request: CompileRequest, result: NeedsContext,
                    *, incorporate: Callable[[CompileRequest, tuple[ReadResult, ...]], CompileRequest] | None = None) -> RecoveryResult:
    """The controller supplies a legal task-specific incorporation/decomposition.

    Reads alone never assert obligations satisfied. Without an applicable native
    incorporation path, return an explicit Blocked with the available pages.
    """
    registry = ContextSourceRegistry(compiler.camp, request.campaign_id)
    results = []
    used = 0
    seen = set()
    for read in result.reads[:request.budget.max_reads]:
        key = digest(read.__dict__)
        if key in seen:
            continue
        seen.add(key)
        remaining = request.budget.max_read_chars - used
        if remaining <= 0:
            break
        bounded = replace(read, max_chars=min(read.max_chars, remaining, request.budget.excerpt_chars))
        reply = registry.read(bounded, role=request.role, hard_max_chars=request.budget.excerpt_chars)
        results.append(reply)
        if reply.status == "ready":
            used += len(canonical(reply.content))
    successful = tuple(reply for reply in results if reply.status == "ready")
    if not successful:
        blocked = Blocked("context_reads_unavailable", {"read_statuses": [reply.status for reply in results]},
                          result.budget_report, ("registered readable source or smaller page",))
        return RecoveryResult("blocked", blocked, tuple(results), used)
    if incorporate is None:
        next_request = incorporate_partial_reads(request, successful, registry=registry)
        if digest(next_request.payload) == digest(request.payload):
            blocked = Blocked("context_decomposition_required", {"pages_available": len(successful),
                              "native_read_receipts_satisfied": False}, result.budget_report,
                              ("native task-specific bounded read/decomposition adapter",))
            return RecoveryResult("blocked", blocked, tuple(results), used)
    else:
        next_request = incorporate(request, successful)
    # One recompile per recovery invocation; caller must not retry unchanged digest.
    if digest(next_request.payload) == digest(request.payload):
        return RecoveryResult("blocked", Blocked("context_recovery_no_progress", {}, result.budget_report), tuple(results), used)
    compiled = compiler.compile(next_request)
    if isinstance(compiled, NeedsContext):
        return RecoveryResult("needs_decomposition", compiled, tuple(results), used)
    return RecoveryResult(compiled.status, compiled, tuple(results), used)
