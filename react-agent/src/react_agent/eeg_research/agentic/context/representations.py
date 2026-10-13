"""Typed, exact delivery views and closed request-local references."""
from __future__ import annotations

import ast
import copy
from typing import Any

from .sources import pointer_get, pointer_escape
from .types import Representation, SourceBlock, canonical, digest

RAW_DATA_FIELDS = frozenset({"history", "curves", "train_image_ids", "validation_image_ids",
    "image_ids", "predictions", "embeddings", "logits", "raw_eeg", "raw_values", "tensor", "tensors"})
NARRATIVE_FIELDS = frozenset({"narrative", "explanation", "rationale", "reasoning", "summary_text",
    "description", "observations", "interpretations", "competing_explanations", "suggested_next_actions",
    "recent_arc", "analysis", "analysis_text", "report_text", "summary_excerpt", "text", "content", "body", "note"})
CODE_FIELDS = frozenset({"source", "code", "implementation", "source_text", "code_text", "diff", "patch"})
LOG_FIELDS = frozenset({"log", "logs", "stdout", "stderr", "traceback", "trace", "training_log"})
FAILURE_WORDS = ("error", "fail", "conflict", "unknown", "partial", "unverified", "missing")


def source_kind(key: str, payload: Any) -> str:
    if key in {"experiment_observations", "evidence", "metrics", "method_screenings", "method_suites", "method_suite", "run", "scores", "diagnostics", "facts", "results", "evaluation"}:
        return "scientific"
    if "check" in key or "receipt" in key:
        return "checks"
    if "source" in key or key in CODE_FIELDS:
        return "code"
    if key in LOG_FIELDS:
        return "log"
    if key in {"analyses", "analysis", "controller_state", "research_progress"}:
        return "analysis"
    if "memory" in key or key in {"lessons", "skills", "snapshot"}:
        return "memory"
    return "unclassified"


def handle(block: SourceBlock, pointer: str = "", *, kind: str = "REF_ONLY", unit: str = "json", count: int | None = None) -> dict[str, Any]:
    result = {"$context_source": block.source_ref, "source_hash": block.source_hash,
              "json_pointer": pointer, "fidelity": kind, "read_unit": unit,
              "read_capability": "registered_bounded_delivery_read",
              "satisfies_development_read_receipt": False}
    if count is not None:
        result["item_count"] = count
    return result


def code_outline(text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    lines = text.splitlines()
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Exact line, not a reconstructed or truncated signature.
            result.append({"name": node.name, "line_start": node.lineno - 1,
                           "line_end": getattr(node, "end_lineno", node.lineno),
                           "declaration": lines[node.lineno - 1]})
    return sorted(result, key=lambda row: (row["line_start"], row["name"]))


def representations_for(block: SourceBlock, payload: Any, *, role: str, string_cap: int = 1_200,
                        excerpt_cap: int = 8_000, abstracted=None,
                        complete_read_paths: frozenset[str] = frozenset()) -> tuple[list[Representation], list[dict[str, Any]]]:
    """Preserve every control/scientific scalar; classify only bulky prose/data.

    The adapter deliberately traverses new fields. Unclassified bulky content and
    failure diagnostics produce explicit obligations rather than disappearing.
    """
    omitted: list[str] = []
    exact: list[str] = []
    unknown: list[dict[str, Any]] = []
    selected_ranges: list[dict[str, Any]] = []
    covered_sections: list[str] = []

    def visit(value: Any, path: str, key: str = "") -> Any:
        if path in complete_read_paths:
            # A verified full-field read must remain literally delivered after
            # packing. Reading the bytes does not grant execution permission.
            exact.append(path)
            return copy.deepcopy(value)
        if isinstance(value, dict):
            if value.get("fidelity") == "PARTIAL" and ("actual_read_scope" in value or "actual_read_scopes" in value):
                # Exact code pages already bounded by a registry read cannot be
                # replaced by another narrative reference during recompile.
                return copy.deepcopy(value)
            # Existing request-local pointer is a reference, not prose to rewrite.
            if "$context_pointer" in value:
                return copy.deepcopy(value)
            return {name: visit(child, path + "/" + pointer_escape(name), name) for name, child in value.items()}
        if isinstance(value, list):
            if key in RAW_DATA_FIELDS and len(canonical(value)) > excerpt_cap:
                omitted.append(path)
                return handle(block, path, count=len(value), unit="items")
            return [visit(child, path + "/" + str(index), key) for index, child in enumerate(value)]
        if isinstance(value, str) and key in NARRATIVE_FIELDS and abstracted is not None and abstracted(path):
            omitted.append(path)
            covered_sections.append(path)
            item = handle(block, path, unit="chars")
            item["abstracted_by_valid_snapshot"] = True
            return item
        if not isinstance(value, str) or len(value) <= string_cap:
            exact.append(path)
            return value
        failure = any(word in key.lower() for word in FAILURE_WORDS)
        if key in CODE_FIELDS or (block.kind == "code" and path == ""):
            omitted.append(path)
            outline = code_outline(value)
            # A huge outline is itself captured behind the same handle; no clipping.
            if len(canonical(outline)) > excerpt_cap:
                outline = []
            item = handle(block, path, unit="lines")
            item["declarations"] = outline
            item["line_count"] = len(value.splitlines())
            if role in {"candidate_coder", "candidate_reviewer"}:
                unknown.append({"path": path, "type": "code", "status": "requires_targeted_source_read",
                                "source_ref": block.source_ref, "source_hash": block.source_hash,
                                "line_count": len(value.splitlines()),
                                "reason": "full source has not been delivered to this role"})
            return item
        if key in LOG_FIELDS or block.kind == "log":
            lines = value.splitlines(keepends=True)
            diagnostic = [{"line": index, "text": line} for index, line in enumerate(lines)
                          if any(word in line.lower() for word in ("error", "exception", "failed", "traceback"))]
            item = handle(block, path, kind="PARTIAL", unit="lines")
            omitted.append(path)
            if len(canonical(diagnostic)) <= excerpt_cap:
                item["diagnostic_lines"] = diagnostic
                selected_ranges.extend({"path": path, "unit": "lines", "start": row["line"], "end": row["line"] + 1} for row in diagnostic)
            else:
                unknown.append({"path": path, "type": "log", "status": "diagnostic_pages_required",
                                "source_ref": block.source_ref, "source_hash": block.source_hash,
                                "line_count": len(lines)})
            return item
        if key in NARRATIVE_FIELDS and not failure:
            omitted.append(path)
            return handle(block, path, unit="chars")
        # Unknown fields cannot silently become an allegedly complete fact card.
        unknown.append({"path": path, "type": "string", "status": "unclassified",
                        "source_ref": block.source_ref, "source_hash": block.source_hash,
                        "chars": len(value), "failure_diagnostic": failure})
        omitted.append(path)
        return handle(block, path, unit="chars")

    full = Representation(digest({"source": block.source_hash, "kind": "FULL"}), block.block_id,
                          "FULL", copy.deepcopy(payload), (block.source_ref,), (block.source_hash,),
                          digest(payload), budget_chars=len(canonical(payload)), applicability=block.task_applicability)
    content = visit(payload, "")
    card = Representation(digest({"source": block.source_hash, "kind": "FACT_CARD", "content": content}),
                          block.block_id, "FACT_CARD", content, (block.source_ref,), (block.source_hash,),
                          digest(content), exact_fact_refs=tuple(exact), selected_ranges=tuple(selected_ranges),
                          omitted_sections=tuple(omitted), completeness="exact_facts_with_registered_omissions" if omitted else "complete",
                          applicability=block.task_applicability, budget_chars=len(canonical(content)))
    reference = handle(block)
    ref = Representation(digest({"source": block.source_hash, "kind": "REF_ONLY"}), block.block_id,
                         "REF_ONLY", reference, (block.source_ref,), (block.source_hash,), digest(reference),
                         completeness="reference_only", applicability=block.task_applicability,
                         budget_chars=len(canonical(reference)))
    # Downgrades must actually reduce effective cost, not add larger cards.
    # A valid published abstraction replaces the old prose even if the raw block
    # would fit. Reintroducing it would turn the snapshot into additive history.
    choices = [] if covered_sections else [full]
    if card.budget_chars < full.budget_chars:
        choices.append(card)
    if not choices:
        choices.append(card)
    if ref.budget_chars < choices[-1].budget_chars:
        choices.append(ref)
    return choices, unknown


def deduplicate(payload: Any, *, minimum_chars: int = 300) -> Any:
    """Factor identical final representations, rebuilding refs after packing."""
    seen: dict[str, tuple[str, str]] = {}

    def visit(value: Any, path: str) -> Any:
        if not isinstance(value, (dict, list)):
            return value
        if isinstance(value, dict) and ("$context_pointer" in value or "$context_source" in value):
            return copy.deepcopy(value)
        original_hash = digest(value)
        size = len(canonical(value))
        if size >= minimum_chars and original_hash in seen:
            target, target_hash = seen[original_hash]
            return {"$context_pointer": target, "target_hash": target_hash,
                    "fidelity": "shared_exact_representation"}
        if isinstance(value, dict):
            result = {key: visit(child, path + "/" + pointer_escape(key)) for key, child in value.items()}
        else:
            result = [visit(child, path + "/" + str(index)) for index, child in enumerate(value)]
        if size >= minimum_chars:
            seen[original_hash] = (path, digest(result))
        return result
    return visit(payload, "")


def validate_pointer_closure(payload: Any) -> tuple[bool, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    def resolve(path: str, ancestors: set[str]) -> Any:
        if path in ancestors:
            raise ValueError("pointer_cycle")
        value = pointer_get(payload, path)
        if isinstance(value, dict) and "$context_pointer" in value:
            target = value["$context_pointer"]
            resolved = resolve(target, ancestors | {path})
            if digest(pointer_get(payload, target)) != value.get("target_hash"):
                raise ValueError("pointer_target_hash_mismatch")
            return resolved
        return value
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if "$context_pointer" in value:
                try:
                    resolve(path, set())
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    errors.append({"path": path, "reason": str(exc)})
                return
            for key, child in value.items():
                visit(child, path + "/" + pointer_escape(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + "/" + str(index))
    visit(payload, "")
    return not errors, errors


def resolve_request_pointer(payload: Any, pointer: str, *, max_depth: int = 16) -> Any:
    def resolve(path: str, seen: set[str], depth: int) -> Any:
        if depth > max_depth:
            raise ValueError("pointer_depth_exceeded")
        if path in seen:
            raise ValueError("pointer_cycle")
        value = payload
        tokens = [] if path == "" else path[1:].split("/")
        if path and not path.startswith("/"):
            raise ValueError("invalid JSON pointer")
        traversed = ""
        for index in range(len(tokens) + 1):
            if isinstance(value, dict) and "$context_pointer" in value:
                target = value["$context_pointer"]
                if digest(pointer_get(payload, target)) != value.get("target_hash"):
                    raise ValueError("pointer_target_hash_mismatch")
                rest = "/".join(tokens[index:])
                combined = target + (("/" + rest) if rest else "")
                return resolve(combined, seen | {path}, depth + 1)
            if index == len(tokens):
                return value
            token = tokens[index]
            key = token.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
            traversed += "/" + token
        return value
    return resolve(pointer, set(), 0)
