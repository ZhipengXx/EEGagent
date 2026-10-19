"""Pure, role-scoped scientific Memory delivery and content accounting.

The application content allowance is deliberately distinct from wire size.
Fixed schema names are free here; names in open scientific maps are data.
Neither this module nor its projection reads a store, clips prose, or calls an LLM.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

PROJECTION_VERSION = "scientific_memory_delivery.v1"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


# These are schemas at specific paths, never a global whitelist of key names.
_OBSERVATION = frozenset({"observation_kind", "question", "intervention", "execution_status", "summary_zh", "summary",
    "identity", "evaluation_scope", "native_facts", "analyst_interpretation", "engineering_failure",
    "unresolved_questions", "applicability", "applicability_label", "applicability_status",
    "condition_differences", "current_feedback", "fidelity"})
_COMMON_BODY = frozenset({"title", "name", "method", "statement", "mechanism", "description",
    "summary", "summary_zh", "idea", "conditions", "preconditions", "scope", "applicability",
    "condition_differences", "applicability_label", "applicability_status", "is_scientific_evidence",
    "limitations", "evidence_updates"})
_LESSON = _COMMON_BODY | frozenset({"confidence", "supporting_episode_ids", "contradicting_episode_ids",
    "mismatch", "missing_conditions", "invalidation_conditions", "failure_modes"})
_SKILL = _COMMON_BODY | frozenset({"description_en", "trigger_en", "procedure", "failure_modes",
    "invalidation_conditions"})
_NATIVE = frozenset({"phase", "scope", "fidelity", "seed", "coverage", "required_fold_count",
    "full_required_fold_count", "independently_replicated", "screening_fold_ids", "screening_subjects",
    "promotion", "missing", "metrics", "scores", "paired_scores", "paired_delta_pp", "measurement_kind",
    "summary", "native_diagnostics", "diagnostic_selection", "training_recipe", "input_geometry",
    "learning_profile", "folds", "fold_id", "held_out_subject", "baseline_metrics", "candidate_metrics"})
_CONDITIONS = frozenset({"dataset", "modality", "task_type", "regime", "scope", "seed", "fold_id",
    "fold_count", "fold_ids", "screening_subjects", "input_geometry", "training_recipe", "feature_target",
    "intervention", "fidelity", "negative_sampling_policy", "evaluation_mode", "generalization_target"})
_SCHEMAS = {
    ("identity",): frozenset({"dataset", "modality", "binding_level", "unknown_top_level_fields",
                             "single_scientific_revision_claimed"}),
    ("native_facts",): _NATIVE,
    ("evaluation_scope",): frozenset({"phase", "scope", "folds", "fold_ids", "seed", "seeds",
        "fold_count", "required_fold_count", "subjects", "subject_ids", "held_out_subject", "fidelity"}),
    ("analyst_interpretation",): frozenset({"hypothesis_assessment", "summary_zh", "summary", "limitations",
        "findings", "next_steps", "points"}),
    ("analyst_interpretation", "points"): frozenset({"hypothesis_assessment", "summary_zh", "summary",
        "limitations", "findings", "next_steps"}),
    ("engineering_failure",): frozenset({"error", "error_category", "reason", "summary", "blocking_issues"}),
    ("evidence_updates", "*"): frozenset({"scope", "verified_fold_coverage", "relation",
        "scientific_strength", "summary", "limitations"}),
    ("native_facts", "native_diagnostics", "*"): frozenset({"fold_id", "held_out_subject", "seed",
        "diagnostics", "learning_profile", "scope", "fidelity"}),
    ("applicability",): _CONDITIONS,
    ("conditions",): _CONDITIONS,
    ("preconditions",): _CONDITIONS,
    ("input_geometry",): frozenset({"channels", "timesteps", "c_num", "shape", "sampling_rate", "units"}),
}
_ROOT_METADATA = frozenset({"schema_version", "origin_campaign_id", "content_digest", "content_hash",
    "observation_id", "plan_id", "candidate_id", "record_id", "record_type", "lesson_id", "skill_id",
    "version", "record_version", "source_refs", "source_ref", "artifact_id", "read_ref", "source_hash",
    "spec_hash", "scientific_intent_digest", "method_revision", "created_at", "updated_at",
    "source_task_refs", "artifact_refs", "supporting_episode_ids", "contradicting_episode_ids", "retrieval"})
_CONDITION_METADATA = frozenset({"campaign_id", "task", "evaluation_identity", "terminal_observation_id",
    "candidate_id", "source_hash", "spec_hash", "method_revision", "scientific_intent_digest",
    "selection_manifest_hash"})
_FACT_METADATA = frozenset({"candidate_id", "method_revision", "source_hash", "spec_hash",
    "selection_manifest_hash", "baseline_suite_hash"})
_IDENTITY_METADATA = frozenset({"candidate_id", "source_hash", "spec_hash", "scientific_intent_digest",
                              "method_revision"})
_ANALYSIS_METADATA = frozenset({"artifact_id", "read_ref", "interpretation_only", "record_version",
                               "candidate_id", "source_hash", "spec_hash"})
_DIFFERENCE_METADATA = _CONDITION_METADATA | frozenset({"content_hash", "fingerprint"})
_DELIVERY_FLAGS = ("knowledge_role", "scientific_evidence", "execution_authority")


def _metadata(path):
    if not path:
        return _ROOT_METADATA
    if path in {("conditions",), ("preconditions",), ("applicability",)}:
        return _CONDITION_METADATA
    if path == ("identity",):
        return _IDENTITY_METADATA
    if path == ("native_facts",):
        return _FACT_METADATA
    if path == ("native_facts", "native_diagnostics", "*"):
        return _IDENTITY_METADATA
    if path == ("analyst_interpretation",):
        return _ANALYSIS_METADATA
    if path == ("evidence_updates", "*"):
        return frozenset({"source", "episode_id"})
    return frozenset()


def _short_refs(value):
    """Only native citation identifiers, not storage paths or read protocols."""
    if isinstance(value, str):
        return [{"source_ref": value}]
    if isinstance(value, dict):
        direct = {key: copy.deepcopy(value[key]) for key in ("artifact_id", "source_ref", "episode_id", "task_id")
                  if value.get(key)}
        if direct:
            return [direct]
        # A skill source manifest groups native artifacts/tasks/episodes.
        return [ref for key in ("artifacts", "tasks", "episodes")
                for ref in _short_refs(value.get(key, []))]
    if isinstance(value, list):
        result = []
        seen = set()
        for entry in value:
            for ref in _short_refs(entry):
                identity = _canonical(ref)
                if identity not in seen:
                    seen.add(identity)
                    result.append(ref)
        return result
    return []


def _differences(value, *, field_names=False):
    if isinstance(value, list):
        return [_differences(row) for row in value
                if not (isinstance(row, dict) and row.get("field") in _DIFFERENCE_METADATA)
                and not (field_names and isinstance(row, str) and row in _DIFFERENCE_METADATA)]
    if isinstance(value, dict):
        if "field" in value:
            # expected/actual can themselves be open scientific configurations.
            return copy.deepcopy(value)
        return {key: _differences(child, field_names=key in {"unknown", "mismatch", "missing_conditions"})
                for key, child in value.items() if key not in _DIFFERENCE_METADATA}
    return copy.deepcopy(value)


def _clean_body(value, path=()):
    if isinstance(value, dict):
        ignored = _metadata(path)
        # Short attribution identities remain on the final scientific row; the
        # counter independently exempts their values from the science quota.
        if not path:
            ignored = ignored - {"observation_id", "lesson_id", "skill_id", "candidate_id", "plan_id"}
        elif path in {("identity",), ("native_facts", "native_diagnostics", "*")}:
            ignored = ignored - {"candidate_id"}
        elif path == ("analyst_interpretation",):
            ignored = ignored - {"interpretation_only"}
        elif path == ("evidence_updates", "*"):
            ignored = ignored - {"episode_id"}
        result = {}
        for key, child in value.items():
            if key in ignored:
                continue
            if key == "condition_differences" and path in {(), ("applicability",)}:
                result[key] = _differences(child)
            elif not path and key in {"mismatch", "missing_conditions"}:
                result[key] = _differences(child, field_names=True)
            elif path in {("conditions", "training_recipe"), ("preconditions", "training_recipe"),
                          ("applicability", "training_recipe")} and key in {"data_root", "gpu"}:
                continue
            else:
                result[key] = _clean_body(child, path + (key,))
        return result
    if isinstance(value, list):
        return [_clean_body(child, path + ("*",)) for child in value]
    return copy.deepcopy(value)


def _kind(record, body):
    kind = record.get("record_type")
    if kind:
        return str(kind)
    if "observation_id" in body or "native_facts" in body:
        return "experiment_observation"
    if "lesson_id" in body:
        return "lesson"
    if "skill_id" in body or "procedure" in body:
        return "skill"
    return "knowledge"


def project_memory_card(record, *, role="research_planner", context=None):
    """Project a native record without mutating it or changing applicability.

    ``role`` and ``context`` are explicit delivery inputs; type filtering and
    scientific applicability remain the caller's native retrieval decisions.
    """
    original = record.get("body", record)
    if not isinstance(original, dict):
        raise TypeError("memory_body_must_be_mapping")
    kind = _kind(record, original)
    if kind == "experiment_observation":
        # This pure helper reuses the already-paid analysis, with no registry IO.
        from .experiment_observations import observation_science_view
        body = observation_science_view(original)
    else:
        body = copy.deepcopy(original)
    refs = _short_refs(record.get("source_refs") or original.get("source_refs") or [])
    analysis = original.get("analyst_interpretation")
    if isinstance(analysis, dict) and analysis.get("artifact_id"):
        refs += [{"artifact_id": analysis["artifact_id"]}]
    if record.get("record_source_ref"):
        refs += _short_refs(record["record_source_ref"])
    record_id = (record.get("record_id") or original.get("observation_id")
                 or original.get("lesson_id") or original.get("skill_id") or original.get("id"))
    if record_id is None:
        raise ValueError("memory_record_id_required")
    result = {"record_type": kind, "record_id": str(record_id), "body": _clean_body(body),
              "source_refs": _short_refs(refs)}
    applicability = (record.get("applicability", original.get("applicability"))
                     if original is not record else body.get("applicability"))
    if applicability is not None:
        if isinstance(applicability, dict):
            applicability = _clean_body(applicability, ("applicability",))
        result["applicability"] = copy.deepcopy(applicability)
    if "condition_differences" in record:
        result["condition_differences"] = _differences(record["condition_differences"])
    if "current_feedback" in record:
        result["current_feedback"] = bool(record["current_feedback"])
    for key in (*_DELIVERY_FLAGS, "precondition_status"):
        if key in record:
            result[key] = copy.deepcopy(record[key])
    return result


def _schema(path, record_type):
    if not path:
        return _OBSERVATION if record_type == "experiment_observation" else (
            _SKILL if record_type == "skill" else _LESSON if record_type == "lesson" else _COMMON_BODY)
    if path in _SCHEMAS:
        return _SCHEMAS[path]
    if path[-1:] == ("input_geometry",) and path[:-1] in {
            ("native_facts",), ("applicability",), ("conditions",), ("preconditions",)}:
        return _SCHEMAS[("input_geometry",)]
    if path == ("native_facts", "paired_scores", "*"):
        return frozenset({"fold_id", "held_out_subject", "seed", "baseline_top1", "candidate_top1",
                          "baseline_top5", "candidate_top5", "delta_pp"})
    if path == ("condition_differences", "*"):
        return frozenset({"field", "expected", "actual", "current", "historical", "reason", "status"})
    return frozenset()


def _count(value, path, record_type):
    if isinstance(value, dict):
        ignored = _metadata(path)
        fixed = _schema(path, record_type)
        return sum((0 if key in fixed else len(key)) + _count(child, path + (key,), record_type)
                   for key, child in value.items() if key not in ignored)
    if isinstance(value, list):
        return sum(_count(child, path + ("*",), record_type) for child in value)
    if isinstance(value, str):
        return len(value)
    return len(_canonical(value))


def memory_content_chars(card, *, record_type=None, role=None):
    """Count science, excluding fixed structural labels and known citation IDs.

    An explicit ``record_type`` also supports the final role row without a body
    envelope. Open maps remain open even if a custom key matches a schema label.
    """
    body = card.get("body", card)
    kind = record_type or _kind(card, body)
    result = _count(body, (), kind)
    if body is not card:
        # An envelope may add applicability labels or differences. Count those
        # once, without charging the duplicated body-level scientific condition.
        for key in ("applicability", "condition_differences", "precondition_status"):
            if key in card and (key not in body or card[key] != body[key]):
                result += _count(card[key], (key,), kind)
        if "current_feedback" in card and "current_feedback" not in body:
            result += _count(card["current_feedback"], ("current_feedback",), kind)
    return result


@dataclass(frozen=True)
class Selection:
    cards: list[dict]
    report: dict


def select_memory_cards(cards, *, content_chars_budget: int, mandatory_ids=(), dedup_keys=None):
    """Select complete preordered cards; a large optional card does not stop us.

    ``dedup_keys`` maps (record_type, record_id) to caller-verified scientific
    support identities. It stays local and only selected cards claim support.
    """
    if isinstance(content_chars_budget, bool) or not isinstance(content_chars_budget, int) or content_chars_budget < 0:
        raise ValueError("memory_content_chars_budget_must_be_nonnegative_integer")
    cards = list(cards)
    mandatory = {str(value) for value in mandatory_ids}
    mandatory.update(str(card["record_id"]) for card in cards if card.get("current_feedback") is True)
    current_cards = {(card.get("record_type"), str(card["record_id"])): card for card in cards
                     if card.get("current_feedback") is True}
    result = []
    seen = set()
    seen_sources = set()
    omitted = []
    used = required = 0
    for card in cards:
        record_id = str(card["record_id"])
        identity = (card.get("record_type"), record_id)
        # The same record can enter through recent and current paths. Preserve
        # the actual current card even if a recent duplicate was listed first.
        card = current_cards.get(identity, card)
        cost = memory_content_chars(card)
        if identity in seen:
            omitted.append({"record_id": record_id, "record_type": identity[0], "reason": "duplicate_record"})
            continue
        seen.add(identity)
        is_mandatory = record_id in mandatory or card.get("current_feedback") is True
        source_identity = (dedup_keys or {}).get(identity)
        if not is_mandatory and source_identity is not None and source_identity in seen_sources:
            omitted.append({"record_id": record_id, "record_type": identity[0], "reason": "duplicate_source"})
            continue
        if not is_mandatory and used + cost > content_chars_budget:
            omitted.append({"record_id": record_id, "record_type": identity[0],
                            "reason": "content_chars_budget", "content_chars": cost})
            continue
        result.append(copy.deepcopy(card))
        if source_identity is not None:
            seen_sources.add(source_identity)
        if is_mandatory:
            required += cost
        else:
            used += cost
    return Selection(result, {"projection_version": PROJECTION_VERSION,
        "content_chars_budget": content_chars_budget, "content_chars_used": used,
        "mandatory_content_chars": required, "selected_ids": [card["record_id"] for card in result],
        "omitted": omitted, "packet_chars": len(_canonical(result))})
