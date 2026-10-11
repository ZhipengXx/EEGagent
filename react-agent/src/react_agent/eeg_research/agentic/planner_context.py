"""Provider-only Planner projection; authoritative observations stay intact."""

from __future__ import annotations

import copy
import json
from typing import Any

REVISION = "planner_working_set_v1"
INPUT_ONLY_ECHO_FIELDS = frozenset({
    "completion_audit_ready", "audit_status", "audit_fresh", "audited_report_hash",
    "require_audit_before_completion",
})


def normalize_input_echoes(reply: Any, observation: dict[str, Any]) -> tuple[Any, list[str]]:
    """Remove only identical read-only audit echoes, never actions or permissions."""
    if not isinstance(reply, dict):
        return reply, []
    removed = sorted(key for key in INPUT_ONLY_ECHO_FIELDS
                     if key in reply and key in observation
                     and type(reply[key]) is type(observation[key])
                     and reply[key] == observation[key])
    return {key: value for key, value in reply.items() if key not in removed}, removed


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in fields if key in row and row[key] is not None}
            for row in rows]


def _limits(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _limits(child) for key, child in value.items() if key != "truncated_fields"}
        if isinstance(value.get("truncated_fields"), list):
            result["truncated_field_count"] = len(value["truncated_fields"])
        return result
    if isinstance(value, list):
        return [_limits(child) for child in value]
    return value


def compact_planner_context(observation: dict[str, Any], *, target_chars: int = 230_000) -> dict[str, Any]:
    """Keep gates, signed sources, numerical facts and read pages verbatim.

    Drop duplicated narratives and index-only metadata before optional history.
    A soft target never licenses dropping an executable constraint or evidence
    fact. Content indexed by an artifact ID is explicitly not a completed read.
    """
    if (observation.get("planner_context_projection") or {}).get("revision") == REVISION:
        return copy.deepcopy(observation)
    result = copy.deepcopy(observation)
    reductions: dict[str, Any] = {}

    def replace(key: str, value: Any, reason: str) -> None:
        reductions[key] = {"before_chars": _size(result.get(key)), "after_chars": _size(value),
                           "reason": reason}
        result[key] = value

    replace("artifact_index", _rows(result.get("artifact_index") or [], (
        "artifact_id", "kind", "candidate_id", "question_id", "verification_status",
        "completion_status", "scope", "missing_inputs",
    )), "Index identities and validity retained; hashes, producer metadata and full content remain in the registry and verified reads.")
    defaults = {"verification_status": "verified", "completion_status": "completed",
                "scope": "development", "missing_inputs": []}
    result["artifact_index_defaults"] = defaults
    result["artifact_index"] = [{key: value for key, value in row.items()
                                if key not in defaults or value != defaults[key]}
                               for row in result["artifact_index"]]
    reductions["artifact_index"]["after_chars"] = _size(result["artifact_index"])
    reductions["artifact_index"]["reason"] += " Rows inherit artifact_index_defaults; exceptional validity/scope values remain explicit."
    replace("evidence_index", _rows(result.get("evidence_index") or [], (
        "evidence_id", "candidate_id", "kind", "fidelity", "evaluation_valid",
    )), "Omit null index cells; exact known evidence IDs remain authoritative.")
    replace("recent_decisions", _rows(result.get("recent_decisions") or [], (
        "decision_id", "action", "target_id", "question_id", "ok", "executed", "detail",
        "evidence_ids", "required_artifact_refs", "selected_option_id", "resolves_issue_ids",
    )), "Historical executable outcomes retained; old unselected options are not current actions.")
    replace("context_limits", _limits(result.get("context_limits") or {}),
            "Count omitted prose paths instead of retransmitting their long diagnostic list.")
    controller = copy.deepcopy(result.get("controller_state") or {})
    summaries = []
    for original in controller.get("hypothesis_evidence") or []:
        row = {key: value for key, value in original.items()
               if key not in {"competing_explanations", "suggested_next_actions", "context_limits"}}
        summaries.append(row)
    controller["hypothesis_evidence"] = summaries
    replace("controller_state", controller,
            "Retain assessments, prediction checks, gaps and identities; full Analyst interpretations remain available through their artifact IDs.")
    # memory_index duplicates identities already carried by memory/lesson views
    # and the complete artifact/evidence indexes. It is not a control input.
    if "memory_index" in result:
        replace("memory_index", _rows(result.get("memory_index") or [], (
            "episode_id", "candidate_id", "artifact",
        )), "Compatible memory content is already supplied; episode and artifact identities retained.")

    # Prioritize source/config-bound facts over historical prose if still large.
    for key in ("memory", "lessons", "analyses", "evidence"):
        rows = result.get(key)
        if not isinstance(rows, list):
            continue
        original_count = len(rows)
        kept = list(rows)
        while _size(result) > target_chars and len(kept) > 2:
            kept.pop(0)
            result[key] = kept
        if len(kept) != original_count:
            reductions[key] = {"total_rows": original_count, "shown_rows": len(kept),
                               "reason": "Older optional history omitted; exact IDs remain in the indexes. Request a verified artifact page when required."}
    result["planner_context_projection"] = {
        "revision": REVISION, "original_chars": _size(observation), "target_chars": target_chars,
        "reductions": reductions,
        "artifact_index_is_content_read": False,
        "source_and_numerical_facts_preserved": True,
        "full_observation_used_for_execution_validation": True,
        "omission_scope": "Provider view only; no on-disk evidence, source, result or gate changed.",
    }
    result["planner_context_projection"]["projected_chars_before_size_field"] = _size(result)
    result["planner_context_projection"]["soft_target_exceeded"] = _size(result) > target_chars
    return result


# Provider-view reductions applied only when a Planner request does not fit its
# role budget. Each stage is cumulative and recorded in the delivered payload.
SERIES_KEEP = 12
SOURCE_KEYS = frozenset({"implementation_source", "parent_source", "control_source"})
SOURCE_VERBATIM_CHARS = 6000


def _slim_scores(value: Any) -> Any:
    """Per-fold score provenance (refs/hashes) stays bound via suite_hash and score_dependencies."""
    if isinstance(value, dict):
        out = {key: _slim_scores(child) for key, child in value.items()}
        scores = out.get("scores")
        if isinstance(scores, list) and scores and all(isinstance(row, dict) and "fold_id" in row for row in scores):
            out["scores"] = [{key: row[key] for key in ("fold_id", "held_out_subject", "status") if key in row}
                             | {"metrics": {k: v for k, v in (row.get("metrics") or {}).items()
                                            if not isinstance(v, (dict, list))}}
                             for row in scores]
            out["scores_provenance"] = "per-fold refs/hashes omitted from this view; bound by suite_hash"
        return out
    if isinstance(value, list):
        return [_slim_scores(child) for child in value]
    return value


def _summarize_series(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _summarize_series(child) for key, child in value.items()}
    if isinstance(value, list):
        if len(value) > SERIES_KEEP and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            step = max(1, len(value) // 10)
            return {"series_summary": {
                "len": len(value), "first": value[0], "last": value[-1],
                "min": min(value), "max": max(value),
                "argmin": value.index(min(value)), "argmax": value.index(max(value)),
                "every_%d" % step: value[::step]}}
        return [_summarize_series(child) for child in value]
    return value


def code_outline(source: str) -> str | None:
    """Class/function signatures with first docstring lines; None if unparsable."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines()
    out: list[str] = []

    def visit(nodes, depth):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                header = lines[node.lineno - 1].strip()
                doc = ast.get_docstring(node)
                out.append("    " * depth + f"L{node.lineno}: {header}"
                           + (f"  # {doc.splitlines()[0][:160]}" if doc else ""))
                visit(node.body, depth + 1)
    visit(tree.body, 0)
    return "\n".join(out)


def unified_diff(old: str, new: str, context: int = 3) -> str:
    """Unified diff without difflib's autojunk, which turns long files with
    frequent lines (blank lines, padding) into whole-file replacements."""
    import difflib
    a, b = old.splitlines(True), new.splitlines(True)
    out = ["--- parent/eeg_candidate.py\n", "+++ candidate/eeg_candidate.py\n"]
    for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(context):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        out.append(f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@\n")
        for tag, a1, a2, b1, b2 in group:
            if tag == "equal":
                out.extend(" " + line for line in a[a1:a2])
                continue
            out.extend("-" + line for line in a[a1:a2])
            out.extend("+" + line for line in b[b1:b2])
    return "".join(line if line.endswith("\n") else line + "\n" for line in out)


def _source_view(record: dict[str, Any], parent: dict[str, Any] | None) -> dict[str, Any]:
    source = record.get("source")
    if not isinstance(source, str) or len(source) <= SOURCE_VERBATIM_CHARS:
        return record
    view = {key: value for key, value in record.items() if key != "source"}
    outline = code_outline(source)
    parent_source = (parent or {}).get("source")
    if isinstance(parent_source, str) and parent_source != source:
        diff = unified_diff(parent_source, source)
        if len(diff) < len(source):
            view["source_diff_vs_parent"] = diff
    if outline is not None:
        view["source_outline"] = outline
    if "source_diff_vs_parent" not in view and outline is None:
        return record  # Nothing faithful to replace it with.
    view["source_view"] = {"full_chars": len(source), "full_source": "Read source_ref / the candidate artifact for full code.",
                           "shown": [key for key in ("source_diff_vs_parent", "source_outline") if key in view]}
    return view


def _code_views(value: Any) -> Any:
    """Changed code verbatim (diff vs parent), the rest as a signature outline."""
    if not isinstance(value, dict):
        return value
    out = dict(value)
    parent = value.get("parent_source") if isinstance(value.get("parent_source"), dict) else None
    for key in SOURCE_KEYS:
        if isinstance(value.get(key), dict):
            out[key] = _source_view(value[key], parent if key == "implementation_source" else None)
    return out


def _is_provenance_key(key: str) -> bool:
    return (key.endswith(("_sha256", "_fingerprint", "_ref", "_path")) or key in {"checkpoint_id", "source_sha256"})


def _strip_provenance(value: Any, inside: bool = False) -> Any:
    """Inside diagnostics subtrees, drop hash/path provenance; IDs elsewhere are untouched."""
    if isinstance(value, dict):
        return {key: _strip_provenance(child, inside or key == "diagnostics")
                for key, child in value.items() if not (inside and _is_provenance_key(key))}
    if isinstance(value, list):
        return [_strip_provenance(child, inside) for child in value]
    return value


def _leaves(value: Any, prefix: tuple = ()) -> dict[tuple, Any]:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(_leaves(child, prefix + (key,)))
        return out
    return {prefix: value} if not isinstance(value, list) else {}


def _drop_path(value: dict, path: tuple) -> None:
    for key in path[:-1]:
        value = value.get(key)
        if not isinstance(value, dict):
            return
    value.pop(path[-1], None)


def _hoist_fold_constants(value: Any) -> Any:
    """Report a per-fold leaf once when all folds share the identical value."""
    if isinstance(value, dict):
        out = {key: _hoist_fold_constants(child) for key, child in value.items()}
        folds = out.get("folds")
        if isinstance(folds, list) and len(folds) > 2 and all(isinstance(f, dict) for f in folds):
            leaves = [_leaves({k: v for k, v in f.items() if k != "fold_id"}) for f in folds]
            shared = {path: val for path, val in leaves[0].items()
                      if all(path in other and other[path] == val for other in leaves[1:])}
            shared = {path: val for path, val in shared.items() if len(json.dumps(val, default=str)) > 8}
            if shared:
                folds = copy.deepcopy(folds)
                for fold in folds:
                    for path in shared:
                        _drop_path(fold, path)
                out["folds"] = folds
                out["shared_fold_constants"] = {".".join(map(str, path)): val for path, val in shared.items()}
        return out
    if isinstance(value, list):
        return [_hoist_fold_constants(child) for child in value]
    return value


# Values the Planner must copy back verbatim; its reply is checked against them exactly.
EXACT_ECHO_KEYS = frozenset({"action_cost_estimates", "legal_choice_context", "budget", "eligible_targets"})


def _round_floats(value: Any) -> Any:
    """4 significant digits (0.01pp for top-1 fractions); gates use the full observation."""
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {key: child if key in EXACT_ECHO_KEYS else _round_floats(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_round_floats(child) for child in value]
    return value


def _older_suites_scalars(value: Any) -> Any:
    """All but the newest complete suite keep per-fold headline metrics only."""
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return value
    rows = value["evidence"]
    suites = [i for i, row in enumerate(rows) if isinstance(row, dict) and row.get("kind") == "method_suite"]
    out = list(rows)
    for i in suites[:-1]:
        row = copy.deepcopy(rows[i])
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("folds"), list):
            row["diagnostics"] = {
                "scope": diagnostics.get("scope"),
                "detail": "older_suite_headline_metrics_only",
                "folds": [{"fold_id": fold.get("fold_id"),
                           "development_metrics": {k: v for k, v in (fold.get("development_metrics") or {}).items()
                                                   if k in {"primary_metric", "fixed_bank_top1", "fixed_bank_top5"}},
                           "best_epoch": (fold.get("selected_checkpoint") or {}).get("best_epoch")}
                          for fold in diagnostics["folds"] if isinstance(fold, dict)]}
        out[i] = row
    return {**value, "evidence": out}


BUDGET_STAGES = (
    ("round_floats_4_significant", _round_floats),
    ("strip_diagnostic_provenance", _strip_provenance),
    ("hoist_fold_constants", _hoist_fold_constants),
    ("slim_fold_score_provenance", _slim_scores),
    ("summarize_numeric_series", _summarize_series),
    ("code_diff_and_outline", _code_views),
    ("older_suites_headline_only", _older_suites_scalars),
)


def budget_reductions(payload: dict[str, Any], *, skip: tuple[str, ...] = ()):
    """Yield progressively smaller provider views of an over-budget role payload."""
    current = payload
    applied: list[dict[str, Any]] = []
    for name, stage in BUDGET_STAGES:
        if name in skip:
            continue
        before = _size(current)
        current = stage(current)
        applied.append({"stage": name, "before_chars": before, "after_chars": _size(current)})
        yield {**current, "planner_budget_reductions": {
            "stages": list(applied),
            "scope": "Provider view only; on-disk evidence, sources and gates unchanged. Request an artifact read for full detail."}}
