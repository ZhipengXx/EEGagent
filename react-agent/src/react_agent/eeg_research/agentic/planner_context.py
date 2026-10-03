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
