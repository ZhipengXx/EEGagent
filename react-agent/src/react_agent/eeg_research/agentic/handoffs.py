"""Resolve the actual candidate and bounded development facts for role handoffs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.artifacts import file_digest
from react_agent.eeg_research.agentic.execution_protocol import load_protocol
from react_agent.eeg_research.agentic.run_context import BASELINE_HOOK_SPEC, load_approved_binding

SOURCE_CHARS = 16000
_PRIVATE_SCOPES = {"final_test", "final_holdout", "held_out_unused", "secret"}
# Limits apply to provider views, never to the authoritative artifacts or ledger.
READ_MAX_IDS = 3
READ_RANGE_CHARS = 4000
READ_TOTAL_CHARS = 8000
READS_PER_EVIDENCE_EPOCH = 4
READ_FILE_BYTES = 1_000_000
HISTORY_ROWS = 20
HISTORY_CHARS = 16000
HISTORY_ITEM_CHARS = 2000
CONTROL_TEXT_CHARS = 16000
READ_JOURNAL = "artifact_reads.jsonl"

# Application characters, including the system/output-schema string. These are
# hard role totals, not a token/latency claim. Essential evidence is never cut to
# meet them: an over-budget projection must be paged or fail before transport.
ROLE_CONTEXT_BUDGETS = {
    "research_planner": 160_000, "candidate_coder": 160_000,
    "candidate_reviewer": 160_000, "result_analyst": 120_000,
    "experiment_designer": 120_000, "memory_curator": 100_000,
    "result_auditor": 200_000, "research_librarian": 80_000,
}
ROLE_CONTEXT_REVISION = "deterministic_role_projection_v1"


def _context_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":"))


def method_suite_role_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Select scalar training facts from complete suite fold-freeze deliveries.

    Authorization/verification belongs to the native suite reader. This projection
    never upgrades a partial suite. Full raw curve/image-ID arrays remain immutable
    and are identified by their exact payload digest plus supplied freeze refs.
    Selected checkpoint, metric definitions, diagnostic coverage and all benchmark
    scores remain literal. It performs no IO, inference, or scientific aggregation.
    """
    import copy

    def fold_view(fold: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(fold)
        metrics = result.get("development_metrics")
        omitted = []
        if isinstance(metrics, dict):
            for key in ("validation_image_ids", "train_image_ids", "history", "curves"):
                value = metrics.get(key)
                if not isinstance(value, (list, dict)):
                    continue
                metrics.pop(key)
                entry = {"field": "development_metrics." + key,
                         "sha256": hashlib.sha256(_context_json(value).encode("utf-8")).hexdigest(),
                         "item_count": len(value)}
                if key == "history" and isinstance(value, list) and value:
                    best_epoch = metrics.get("selected_checkpoint_epoch", metrics.get("best_epoch"))
                    points = [value[0], value[-1]]
                    points.extend(row for row in value if isinstance(row, dict) and row.get("epoch") == best_epoch)
                    entry["exact_boundary_and_selected_epoch_points"] = points
                omitted.append(entry)
        # The native reader already verified these complete-fold dependencies.
        # Keep the exact freeze ref/hash rather than repeating its full path and
        # permission/dependency manifests in every role delivery. Source,
        # checkpoint, protocol and evaluator identities remain literal below.
        for key in ("dependencies", "permission", "protocol_path"):
            if key not in result:
                continue
            detail = result.pop(key)
            omitted.append({"field": key,
                            "sha256": hashlib.sha256(_context_json(detail).encode()).hexdigest()})
        if omitted:
            result["training_detail_projection"] = {
                "revision": ROLE_CONTEXT_REVISION, "omitted": omitted,
                "freeze_ref": score.get("freeze_ref"), "freeze_sha256": score.get("freeze_sha256"),
                "original_fold_view_sha256": hashlib.sha256(_context_json(fold).encode("utf-8")).hexdigest(),
                "scope": "Provider scalar/boundary view. Omitted raw detail remains in the immutable native freeze. Refs are provenance, not delivered artifact pages."}
        return result

    def complete(aggregate: Any) -> bool:
        return (isinstance(aggregate, dict) and aggregate.get("status") == "complete"
                and aggregate.get("coverage") == 10)

    def diagnostics_view(diagnostics: Any, aggregate: dict[str, Any]) -> Any:
        if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("folds"), list):
            return diagnostics
        result = copy.deepcopy(diagnostics)
        scores = {row.get("fold_id"): row for row in aggregate.get("scores") or [] if isinstance(row, dict)}
        result["folds"] = [fold_view(fold, scores.get(fold.get("fold_id"), {}))
                           if isinstance(fold, dict) else fold for fold in diagnostics["folds"]]
        if aggregate.get("scope") == "method_development_benchmark":
            _share_representation_columns(result)
            _share_training_diagnostic_columns(result)
        return result

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(child) for key, child in value.items()}
        aggregate = value.get("aggregate")
        if value.get("kind") == "method_suite" and complete(aggregate) and "diagnostics" in value:
            result["diagnostics"] = diagnostics_view(value.get("diagnostics"), aggregate)
            if aggregate.get("scope") == "method_development_benchmark":
                _share_score_identity(result["aggregate"])
        # worker._analyze_run delivers the native verified aggregate directly,
        # while its full fold freezes are sibling diagnostic views. Preserve
        # that consumer shape rather than requiring an invented wrapper.
        native = value.get("method_suite")
        curation_run = value.get("run")
        if (isinstance(curation_run, dict) and curation_run.get("kind") == "method_suite"
                and complete(curation_run.get("aggregate"))):
            native = curation_run["aggregate"]
        if (complete(native) and native.get("scope") == "method_development_benchmark"
                and (value.get("evaluation_mode") == "loso_method_search"
                     or isinstance(curation_run, dict) and curation_run.get("kind") == "method_suite")):
            for key in ("diagnostics", "verified_training_diagnostics"):
                if key in value:
                    result[key] = diagnostics_view(value[key], native)
            if isinstance(curation_run, dict) and curation_run.get("kind") == "method_suite":
                _share_score_identity(result["run"]["aggregate"])
            else:
                _share_score_identity(result["method_suite"])
        return result
    return visit(payload)


def _shared_scalar_fields(records: list[dict[str, Any]], *, excluded: set[str] | None = None) -> dict[str, Any]:
    """Factor literal equal columns in a derived copy, restored by dict merge."""
    import copy
    if len(records) < 2:
        return {}
    excluded = excluded or set()
    common = {key: copy.deepcopy(value) for key, value in records[0].items()
              if key not in excluded and all(key in other and _context_json(other[key]) == _context_json(value)
                                             for other in records[1:])}
    if len(_context_json(common)) < 200:
        return {}
    for record in records:
        for key in common:
            record.pop(key)
    return common


def _share_score_identity(aggregate: dict[str, Any]) -> None:
    """Factor only exactly equal native identity columns in a derived aggregate."""
    scores = aggregate.get("scores")
    if not isinstance(scores, list) or not all(isinstance(score, dict) for score in scores):
        return
    common = _shared_scalar_fields(scores, excluded={"fold_id", "held_out_subject", "train_subjects", "metrics",
                                                   "freeze_ref", "freeze_sha256", "score_ref", "score_sha256"})
    if common:
        aggregate.setdefault("shared_score_identity", {}).update(common)
    if aggregate.get("shared_score_identity"):
        aggregate["score_identity_merge_rule"] = (
            "Each scalar score is shared_score_identity merged with its per-fold fields; "
            "all shared columns were exactly equal in the native scores. This changes delivery layout only.")


def _share_representation_columns(diagnostics: dict[str, Any]) -> None:
    """Share exact existing representation defaults without creating missing items."""
    representations = [fold["diagnostics"]["items"]["representation"]
                       for fold in diagnostics.get("folds", [])
                       if isinstance(fold, dict) and isinstance(fold.get("diagnostics"), dict)
                       and isinstance(fold["diagnostics"].get("items"), dict)
                       and isinstance(fold["diagnostics"]["items"].get("representation"), dict)]
    coverage_records = [item["coverage"] for item in representations if isinstance(item.get("coverage"), dict)]
    representation_common = _shared_scalar_fields(representations, excluded={"coverage"})
    coverage_common = _shared_scalar_fields(coverage_records)
    if representation_common:
        diagnostics.setdefault("shared_representation_fields", {}).update(representation_common)
    if coverage_common:
        diagnostics.setdefault("shared_representation_coverage", {}).update(coverage_common)
    if diagnostics.get("shared_representation_fields") or diagnostics.get("shared_representation_coverage"):
        diagnostics.setdefault("shared_representation_fields", {})
        diagnostics.setdefault("shared_representation_coverage", {})
        diagnostics["representation_merge_rule"] = (
            "For an existing per-fold representation, merge shared_representation_fields then its fields; "
            "merge shared_representation_coverage then its existing coverage. Shared values are exact equal "
            "native columns. Do not create a missing item or coverage.")


def _share_training_diagnostic_columns(diagnostics: dict[str, Any]) -> None:
    """Share exact observed training columns without filling missing records."""
    folds = diagnostics.get("folds")
    if not isinstance(folds, list) or not all(isinstance(fold, dict) for fold in folds):
        return
    identity = _shared_scalar_fields(folds, excluded={
        "fold_id", "held_out_subject", "train_subjects", "diagnostics",
        "development_metrics", "selected_checkpoint", "training_detail_projection"})
    if identity:
        diagnostics.setdefault("shared_fold_identity", {}).update(identity)
    for field, shared_name in (("development_metrics", "shared_development_metrics"),
                               ("selected_checkpoint", "shared_selected_checkpoint")):
        records = [fold[field] for fold in folds if isinstance(fold.get(field), dict)]
        common = _shared_scalar_fields(records)
        if common:
            diagnostics.setdefault(shared_name, {}).update(common)
    executions = []
    for fold in folds:
        bundle = fold.get("diagnostics")
        items = bundle.get("items") if isinstance(bundle, dict) else None
        dynamics = items.get("training_dynamics") if isinstance(items, dict) else None
        execution = dynamics.get("training_execution") if isinstance(dynamics, dict) else None
        payload = execution.get("payload") if isinstance(execution, dict) else None
        if isinstance(payload, dict):
            executions.append(payload)
    common = _shared_scalar_fields(executions, excluded={"batches"})
    if common:
        diagnostics.setdefault("shared_training_execution_payload", {}).update(common)
    by_position = {}
    for payload in executions:
        for batch in payload.get("batches") or []:
            if isinstance(batch, dict) and isinstance(batch.get("batch_position"), str):
                by_position.setdefault(batch["batch_position"], []).append(batch)
    for position, batches in by_position.items():
        common = _shared_scalar_fields(batches, excluded={"batch_position"})
        if common:
            diagnostics.setdefault("shared_training_batch_fields", {}).setdefault(position, {}).update(common)
    names = ("shared_fold_identity", "shared_development_metrics", "shared_selected_checkpoint",
             "shared_training_execution_payload", "shared_training_batch_fields")
    if any(diagnostics.get(name) for name in names):
        diagnostics["training_scalar_merge_rule"] = (
            "For each existing fold, merge shared_fold_identity then its literal fields. "
            "Only for existing development_metrics/selected_checkpoint dicts, merge the corresponding "
            "shared_development_metrics/shared_selected_checkpoint then local fields. Only for an existing "
            "diagnostics.items.training_dynamics.training_execution.payload, merge shared_training_execution_payload "
            "then local fields. For each existing batches item, use shared_training_batch_fields at its literal "
            "batch_position then local fields. All shared values were exactly equal JSON columns. "
            "Do not create a missing fold, dict, execution payload, batch, observation or coverage; "
            "retain every different scientific value and provenance identity.")


def _share_diagnostic_columns(diagnostics: dict[str, Any]) -> None:
    """Factor exact equal dictionary columns, preserving missing nodes and values.

    Dictionary paths are relative to each existing native fold. Reconstruction
    merges shared fields before local fields, only at dictionaries that already
    exist. This is delivery layout only, without aggregation or interpretation.
    """
    import copy
    folds = diagnostics.get("folds")
    if not isinstance(folds, list) or not all(isinstance(item, dict) for item in folds):
        return
    shared = copy.deepcopy(diagnostics.get("shared_diagnostic_columns") or [])

    def visit(records, path):
        if len(records) < 2:
            return
        common = {key: copy.deepcopy(value) for key, value in records[0].items()
                  if key not in {"fold_id", "held_out_subject", "train_subjects"}
                  and all(key in other and _context_json(other[key]) == _context_json(value)
                          for other in records[1:])}
        # A small column repeated ten times is useful even below the generic
        # shared-field threshold. Account for its path and delivery overhead.
        if common and (len(records) - 1) * len(_context_json(common)) > len(_context_json(path)) + 120:
            previous = next((item for item in shared if item["path"] == path), None)
            if previous is None:
                shared.append({"path": list(path), "fields": common})
            else:
                previous["fields"].update(common)
            for record in records:
                for key in common:
                    record.pop(key)
        keys = set().union(*(record.keys() for record in records))
        for key in sorted(keys):
            visit([record[key] for record in records if isinstance(record.get(key), dict)], [*path, key])

    visit(folds, [])
    if shared:
        diagnostics["shared_diagnostic_columns"] = shared
        diagnostics["diagnostic_columns_merge_rule"] = (
            "For each existing fold dictionary at a listed path, merge fields then its local fields. "
            "Do not create missing dictionaries, fields in missing dictionaries, or observations; "
            "all shared columns were exactly equal native fields. Preserve every differing value and identity.")


def method_suite_scalar_view(row: dict[str, Any], *,
                             score_dependencies: list[dict[str, Any]] | None = None,
                             historical_diagnostics: bool = False) -> dict[str, Any]:
    """Provider view of a native-reader-verified complete suite, never authority.

    Preserve exact macro metrics and all ten subject scalars rather than a JSON
    prefix. Native score/freeze paths and digests identify omitted raw detail;
    this selection computes no metric and does not claim an artifact page read.
    The caller must first verify the suite with ``verified_suite_records``.
    """
    import copy
    aggregate = row.get("aggregate") or {}
    if (row.get("kind") != "method_suite" or row.get("evaluation_valid") is not True
            or aggregate.get("status") != "complete" or aggregate.get("coverage") != 10
            or aggregate.get("scope") != "method_development_benchmark"):
        raise ValueError("method_scalar_view_requires_verified_complete_suite")
    result = method_suite_role_view(row)
    dependencies = {json.loads(Path(item["path"]).read_text(encoding="utf-8"))["fold_id"]: item
                    for item in score_dependencies or []}
    fields = ("status", "suite_hash", "method_revision", "candidate_id", "fold_id", "held_out_subject",
              "train_subjects", "seed", "fidelity", "protocol_hash", "evaluation_key", "evaluator_hash",
              "checkpoint_hash", "scope", "device", "precision", "gpu_seconds", "freeze_ref", "freeze_sha256")
    scores = []
    for score in aggregate["scores"]:
        selected = {key: copy.deepcopy(score[key]) for key in fields if key in score}
        selected["metrics"] = {key: score["metrics"][key] for key in
                               ("top1", "top5", "query_count", "candidate_count")}
        dependency = dependencies.get(score["fold_id"])
        if dependency is not None:
            selected.update(score_ref=dependency["path"], score_sha256=dependency["sha256"])
        scores.append(selected)
    result["aggregate"]["scores"] = scores
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("folds"), list):
        by_fold = {score["fold_id"]: score for score in scores}
        retained = {"fold_id", "development_metrics", "selected_checkpoint", "diagnostics"}
        folds = []
        for original in diagnostics["folds"]:
            score = by_fold[original["fold_id"]]
            # These exact identities are already literal in aggregate.scores.
            # Bind by native fold_id instead of repeating them, their freeze
            # refs and two omission manifests in every diagnostic delivery.
            for key in ("held_out_subject", "train_subjects", "seed", "fidelity", "method_revision",
                        "protocol_hash", "checkpoint_hash", "evaluator_hash", "evaluation_key"):
                if key in original and key in score and original[key] != score[key]:
                    raise ValueError("method_scalar_fold_score_identity_mismatch")
            fold = {key: value for key, value in original.items() if key in retained}
            if historical_diagnostics:
                metrics = fold.get("development_metrics")
                if isinstance(metrics, dict):
                    fold["development_metrics"] = {key: value for key, value in metrics.items()
                        if not isinstance(value, (list, dict)) or key in {"coverage", "sampling", "definitions"}}
                bundle = fold.get("diagnostics")
                if isinstance(bundle, dict) and isinstance(bundle.get("items"), dict):
                    mechanisms = {"representation", "training_dynamics", "retrieval_errors",
                                  "intervention_probe", "group_results"}
                    items = bundle["items"]
                    fold["diagnostics"] = {key: value for key, value in bundle.items() if key != "items"}
                    fold["diagnostics"].update(
                        items={key: value for key, value in items.items()
                               if key in mechanisms and isinstance(value, dict)
                               and (value.get("status") != "unavailable" or "training_execution" in value)},
                        unavailable={key: value.get("reason") for key, value in items.items()
                                     if isinstance(value, dict) and value.get("status") == "unavailable"},
                        omitted_nonmechanism_items=[key for key, value in items.items()
                            if key not in mechanisms and isinstance(value, dict) and value.get("status") != "unavailable"])
            folds.append(fold)
        diagnostics["folds"] = folds
        diagnostics["fold_identity_binding"] = {
            "basis": "Each fold_id binds to the exact same-row aggregate.scores identity, including native freeze ref/hash.",
            "source_hash": row.get("source_hash"), "detail": "historical_mechanism_scalars" if historical_diagnostics else "current_full_diagnostics",
            "raw_freeze_pages_delivered": False,
            "scope": "Refs identify immutable provenance, not delivered pages or permission. Use registered aggregate/analysis IDs for native reads."}
        _share_representation_columns(diagnostics)
    _share_score_identity(result["aggregate"])
    result["scientific_view"] = {"revision": ROLE_CONTEXT_REVISION,
        "scope": "Exact complete-suite scalar provider view. Raw score/freeze refs identify immutable provenance, not delivered pages or read permission; registered aggregate/analysis IDs are native read entry points.",
        "aggregate_ref": row.get("suite_ref"), "aggregate_sha256": row.get("result_hash"),
        "independent_seed_count": aggregate["independent_seed_count"], "raw_score_metadata_omitted": True}
    return result


def method_benchmark_origin(row: dict[str, Any]) -> dict[str, Any]:
    """Runtime origin for a derived role artifact, never a model permission."""
    aggregate = row.get("aggregate") or {}
    if (row.get("kind") != "method_suite" or row.get("evaluation_valid") is not True
            or aggregate.get("status") != "complete" or aggregate.get("coverage") != 10
            or aggregate.get("scope") != "method_development_benchmark"):
        raise ValueError("method_origin_requires_complete_native_suite")
    return {"run_evidence_id": row["evidence_id"],
            **{key: row[key] for key in ("candidate_id", "suite_id", "suite_ref", "suite_hash", "result_hash",
                "method_revision", "source_hash", "spec_hash", "config_hash", "contract_fingerprint", "seed", "fidelity")},
            "aggregate_artifact_id": row["artifact_refs"][0], "scope": "method_development_benchmark",
            "coverage": 10, "independent_seed_count": 1, "replicated": False,
            "independent_final_test": False}


def _audit_dependency_role_view(value: Any) -> Any:
    """Share only exactly equal manifest fields, preserving every missing field."""
    import copy
    if isinstance(value, list):
        return [_audit_dependency_role_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _audit_dependency_role_view(item) for key, item in value.items()}
    rows = result.get("dependency_manifest")
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, dict) for row in rows):
        return result
    original = copy.deepcopy(rows)
    shared = {}
    for field in ("scope", "verification_status", "data_use"):
        present = [row[field] for row in rows if field in row]
        if len(present) < 2 or any(_context_json(item) != _context_json(present[0]) for item in present[1:]):
            continue
        absent = [index for index, row in enumerate(rows) if field not in row]
        shared[field] = {"value": copy.deepcopy(present[0]), "absent_indices": absent}
        for row in rows:
            row.pop(field, None)
    if shared:
        result["dependency_manifest_shared_fields"] = shared
        result["dependency_manifest_projection"] = {
            "revision": "exact_audit_dependency_fields.v1",
            "original_manifest_sha256": hashlib.sha256(_context_json(original).encode()).hexdigest(),
            "merge_rule": "For each dependency_manifest row index, restore each dependency_manifest_shared_fields value unless the index is in its absent_indices. Missing fields remain absent; paths, content_hash, kind, payload and differing fields are literal. The reconstructed manifest SHA-256 must match original_manifest_sha256. Raw registry/report/hash and verification authority are unchanged."}
    return result


def compact_role_context(payload: dict[str, Any], role: str, *,
                         system_chars: int = 0, target_chars: int | None = None) -> dict[str, Any]:
    """Return a deterministic derived view with one copy of large identical facts.

    Canonical references are JSON pointers into this same request, not evidence
    IDs or claimed artifact reads. No source, metric, gate or receipt is sliced.
    Optional *older* history may be omitted only with its exact digest and IDs;
    current/full source, all legal actions, approval and schema feedback survive.
    The caller checks ``fits_budget`` before a paid API call.
    """
    import copy
    original = copy.deepcopy(payload)
    original.pop("role_context_projection", None)
    working = method_suite_role_view(original) if role in {
        "research_planner", "result_analyst", "memory_curator", "experiment_designer", "result_auditor",
    } else original
    if role == "research_planner" and original.get("evaluation_mode") == "loso_method_search":
        # Keep this additional layout entirely at the final Planner delivery
        # boundary. Analyst/Curator construction and native scalar consumers
        # keep their established field layouts.
        def planner_diagnostics(value):
            if isinstance(value, list):
                return [planner_diagnostics(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = {key: planner_diagnostics(child) for key, child in value.items()}
            aggregate = result.get("aggregate") or {}
            if (result.get("kind") == "method_suite" and aggregate.get("status") == "complete"
                    and aggregate.get("coverage") == 10 and isinstance(result.get("diagnostics"), dict)):
                _share_diagnostic_columns(result["diagnostics"])
            if result.get("evaluation_mode") == "loso_method_search" and isinstance(result.get("latest_diagnostics"), dict):
                _share_diagnostic_columns(result["latest_diagnostics"])
            return result
        working = planner_diagnostics(working)
        rows = working.get("artifact_index")
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            shared = []
            for field in ("kind", "verification_status", "completion_status", "scope", "missing_inputs"):
                groups = {}
                for index, row in enumerate(rows):
                    if field in row:
                        groups.setdefault(_context_json(row[field]), []).append(index)
                for encoded, indices in groups.items():
                    value = json.loads(encoded)
                    entry = {"field": field, "row_indices": indices, "value": value}
                    if (len(indices) > 1 and (len(indices) - 1) * (len(encoded) + len(field) + 4)
                            > len(_context_json(entry)) + 30):
                        shared.append(entry)
                        for index in indices:
                            rows[index].pop(field)
            if shared:
                working["artifact_index_shared_columns"] = shared
                working["artifact_index_column_merge_rule"] = (
                    "For each row index, restore applicable artifact_index_shared_columns fields, then its literal fields; "
                    "then apply artifact_index_defaults only to fields still missing. This preserves every exception "
                    "and original status; it is not content read or permission.")
    if role == "result_auditor" and original.get("evaluation_mode") == "loso_method_search":
        working = _audit_dependency_role_view(working)
    budget = target_chars if target_chars is not None else ROLE_CONTEXT_BUDGETS.get(role, 160_000)
    if budget <= 0 or system_chars < 0:
        raise ValueError("invalid_role_context_budget")
    canonical: dict[str, tuple[str, str, str]] = {}
    duplicates: list[dict[str, Any]] = []
    method_context = original.get("evaluation_mode") == "loso_method_search"
    canonical_min_chars = (180 if method_context and role == "research_planner" else
                           400 if method_context else 1200)

    def pointer(path: str, key: Any) -> str:
        return path + "/" + str(key).replace("~", "~0").replace("/", "~1")

    def project(value: Any, path: str) -> Any:
        encoded = _context_json(value)
        # Retain small control/identity values literally; canonicalize sizeable
        # data only. Hash equality is checked against the exact encoded bytes.
        digest = None
        if isinstance(value, (dict, list, str)) and len(encoded) >= canonical_min_chars:
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            prior = canonical.get(digest)
            if prior is not None and prior[1] == encoded:
                reference = {"canonical_context_ref": prior[0], "sha256": prior[2]}
                if not method_context:
                    reference["scope"] = "Exact duplicate already present in this request; follow the JSON pointer. Not an artifact read."
                if len(_context_json(reference)) >= len(encoded):
                    return value  # No space saving; retain the whole literal.
                duplicates.append({"path": path, "canonical_pointer": prior[0],
                                   "sha256": prior[2], "original_chars": len(encoded)})
                return reference
        if isinstance(value, dict):
            projected = {key: project(child, pointer(path, key)) for key, child in value.items()}
        elif isinstance(value, list):
            projected = [project(child, pointer(path, index)) for index, child in enumerate(value)]
        else:
            projected = value
        if digest is not None:
            canonical[digest] = (path, encoded, hashlib.sha256(_context_json(projected).encode("utf-8")).hexdigest())
        return projected

    result = project(working, "")
    omissions = []
    # A context budget does not justify truncating source or authoritative facts.
    # Historical prose is already indexed by immutable IDs; preserve at least
    # the newest two rows and give the omitted exact identities/digest.
    for key in ("memory", "lessons", "analyses", "recent_decisions"):
        rows = result.get(key)
        if not isinstance(rows, list) or len(rows) <= 2:
            continue
        if len(_context_json(result)) + system_chars <= budget:
            break
        omitted = rows[:-2]
        # Never omit a node targeted by a canonical duplicate elsewhere.
        if any(row["canonical_pointer"].startswith(f"/{key}/") for row in duplicates):
            continue
        result[key] = rows[-2:]
        omissions.append({"field": key, "rows": len(omitted),
                         "sha256": hashlib.sha256(_context_json(omitted).encode("utf-8")).hexdigest(),
                         "identities": [{name: row[name] for name in
                             ("evidence_id", "artifact_id", "analysis_artifact_id", "episode_id", "decision_id", "candidate_id")
                             if isinstance(row, dict) and name in row} for row in omitted],
                         "reason": "Older optional history omitted from provider view; authoritative registry/history unchanged. Request a verified page if needed."})
    duplicate_report = duplicates
    if method_context:
        grouped = {}
        for row in duplicates:
            key = (row["canonical_pointer"], row["sha256"], row["original_chars"])
            if key not in grouped:
                grouped[key] = {**row, "duplicate_paths": []}
            else:
                grouped[key]["duplicate_paths"].append(row["path"])
        duplicate_report = list(grouped.values())
    report = {"revision": ROLE_CONTEXT_REVISION, "role": role,
              "budget_chars": budget, "system_chars": system_chars,
              "original_user_chars": len(_context_json(original)),
              "duplicate_views": duplicate_report, "duplicate_view_count": len(duplicates), "omissions": omissions,
              "source_metrics_gates_and_schema_feedback_preserved": True,
              "canonical_reference_instruction": "Resolve canonical_context_ref as a JSON pointer within this request. Its sha256 covers the exact canonical value; it is not an artifact/evidence ID.",
              "scope": "Deterministic provider view only; no raw artifact/history/source changed and no paid summarizer used."}
    result["role_context_projection"] = report
    total = len(_context_json(result)) + system_chars
    report["projected_total_chars"] = total
    report["fits_budget"] = total + 100 <= budget  # account for these two fields
    if not report["fits_budget"]:
        report["missing_inputs"] = ["essential_context_exceeds_role_budget: obtain permitted digest-bound pages or narrow the task; do not infer omitted code"]
    return result


def training_semantics(*, config_only: bool = False) -> list[dict[str, Any]]:
    """Source evidence for how the trainer calls configs and owns logit scaling."""
    import ast
    root = Path(__file__).resolve().parents[2] / "eeg_training"
    views = []
    selections = (("model.py", {"LocalRetrieval", "contrastive_loss"}),
                           ("hooks.py", {"compute_objective", "scalar_loss", "is_custom_objective"}), ("train_entry.py", {
        "_build_hook", "_logit_scale", "_instantiate_candidate", "train_channel_statistics",
        "unique_trainable_parameters", "placed_retrieval", "rebuild_encoder", "fit",
    }))
    if config_only:
        selections = (("hooks.py", {"call_configured"}), ("train_entry.py", {"_build_hook"}),
                      ("../eeg_research/agentic/hook_config.py", {"normalize_hook_config"}))
    for name, selected in selections:
        path = root / name
        source = path.read_text(encoding="utf-8")
        nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        excerpts = []
        for node in nodes:
            if node.name == "fit":
                # Include the actual construction and optimizer branches, before
                # gallery evaluation; avoid sending the entire training loop.
                stop = next(item.lineno for item in node.body if isinstance(item, ast.Assign)
                            and any(isinstance(target, ast.Tuple) and any(
                                isinstance(part, ast.Name) and part.id == "bank" for part in target.elts
                            ) for target in item.targets))
                excerpts.append("\n".join(source.splitlines()[node.lineno - 1:stop - 1]))
            else:
                excerpts.append(ast.get_source_segment(source, node))
        views.append({"source_ref": str(path), "source_hash": file_digest(path),
                      "excerpt": "\n\n".join(excerpts)})
    return views


def development_view(value: Any) -> Any:
    """Filter scope recursively before sending data to a provider."""
    if isinstance(value, dict):
        if any(isinstance(value.get(key), str) and value[key] in _PRIVATE_SCOPES
               for key in ("scope", "role", "data_role", "partition")):
            return None
        kept = {}
        for key, item in value.items():
            if key in {"test_result", "final_test_result", "forbidden_contents", "api_key", "DEEPSEEK_API_KEY"} or (
                str(key).startswith(("test_result", "final_holdout", "final_test_")) and key != "final_test_enabled"
            ):
                continue
            clean = development_view(item)
            if clean is not None or item is None:
                kept[key] = clean
        return kept
    if isinstance(value, list):
        return [clean for item in value if (clean := development_view(item)) is not None]
    return value


def candidate_cpu_check(camp: Path, target: str, binding: dict[str, Any],
                        current_source_hash: str | None) -> dict[str, Any]:
    """Expose actual CPU receipts only for the source and config they checked."""
    if target == "baseline":
        return {"status":"runtime_frozen_baseline_reference","checks":None}
    workspace = camp / "candidates" / target
    path = workspace / "checks.json"
    view = {"status":"unavailable","check_ref":str(path),"checks":None,"missing_inputs":[]}
    try:
        checks = json.loads(path.read_text(encoding="utf-8"))
        assigned = json.loads((workspace / "spec.json").read_text(encoding="utf-8"))
        experiment = assigned.get("experiment") or assigned
    except (OSError, ValueError) as exc:
        view["missing_inputs"] = ["cpu_check_or_assigned_spec_unreadable:"+type(exc).__name__]
        return view
    if not isinstance(checks,dict) or not isinstance(experiment,dict):
        view["missing_inputs"] = ["cpu_check_or_spec_not_object"]
        return view
    expected = {key:binding.get(key) or {} for key in ("model","objective","transform")}
    identity_ok = (current_source_hash and current_source_hash == binding.get("source_hash")
                   and checks.get("source_sha256") == current_source_hash
                   and experiment.get("spec_hash") == binding.get("spec_hash")
                   and checks.get("approved_hook_config") == expected)
    if not identity_ok:
        view["status"] = "identity_mismatch"
        view["missing_inputs"] = ["cpu_receipt_does_not_match_current_approved_source_and_config"]
        return view
    from react_agent.eeg_research.agentic.native_patch import _check_fingerprint
    try:
        current_fingerprint = _check_fingerprint(workspace,None)
    except (OSError, ValueError, KeyError):
        current_fingerprint = None
    view.update(status="verified_source_config_bound_cpu_receipt",source_hash=current_source_hash,
                spec_hash=binding.get("spec_hash"),check_sha256=file_digest(path),
                check_fingerprint=checks.get("check_fingerprint"),
                current_checker_fingerprint=current_fingerprint,
                fresh_for_current_runtime=bool(current_fingerprint and checks.get("check_fingerprint")==current_fingerprint),
                checks=development_view(checks),
                scope="Actual saved synthetic CPU interface/gradient/formula/off/checkpoint probes for this source/config. Freshness is explicit. This is not an original GPU optimizer-step receipt, training score, benefit or runtime authorization.")
    return view


def candidate_context(camp: Path, target: str) -> dict[str, Any]:
    """Use target-bound source/config; baseline has concrete behavior too."""
    protocol = load_protocol(camp) or {}
    if target == "baseline":
        entry = Path(__file__).with_name("baseline.py")
        binding = dict(BASELINE_HOOK_SPEC)
        binding["model"] = {"z_dim": 1024, "drop_proj": 0.3}
        approval_ref = "runtime:frozen_baseline"
    else:
        entry = camp / "candidates" / target / "extension" / "eeg_candidate.py"
        binding = load_approved_binding(camp, target) or {}
        approval_ref = binding.get("spec_ref")
    missing = []
    if not entry.is_file():
        missing.append("source")
    if not binding:
        missing.append("approved_binding")
    source = entry.read_text(encoding="utf-8") if entry.is_file() else None
    current_hash = file_digest(entry) if entry.is_file() else None
    expected_hash = binding.get("source_hash")
    if expected_hash and current_hash != expected_hash:
        missing.append("source_hash_mismatch")
        source = None
    encoder_reference = None
    if target == "baseline":
        model_entry = Path(__file__).resolve().parents[2] / "eeg_training" / "model.py"
        model_source = model_entry.read_text(encoding="utf-8")
        encoder_reference = {"source_ref": str(model_entry), "source_hash": file_digest(model_entry),
                             "excerpt": model_source.split("class LocalRetrieval", 1)[0]}
    return development_view({
        "candidate_id": target,
        "source_hash": current_hash,
        "source_ref": str(entry),
        # Essential candidate functions cannot be safely selected by a prefix.
        # The role-level budget either carries the complete source or reports a
        # pre-transport paging/task gap; it never licenses a partial review.
        "source": source,
        "source_truncated": False,
        "source_range": None if source is None else [0, len(source)],
        "encoder_reference": encoder_reference,
        "cpu_check_receipt": candidate_cpu_check(camp,target,binding,current_hash),
        "approval_ref": approval_ref,
        "spec_hash": binding.get("spec_hash"),
        "model": binding.get("model") or {},
        "objective": binding.get("objective") or {},
        "transform": binding.get("transform") or {},
        "intervention": binding.get("intervention") or ("frozen EEGProjectLayer baseline" if target == "baseline" else None),
        "recipe": {key: protocol.get(key) for key in ("full_epochs", "batch_size", "lr", "weight_decay",
                   "negative_sampling_policy", "training_seeds", "input_geometry")},
        "negative_sampling_policy": binding.get("negative_sampling_policy") or protocol.get("negative_sampling_policy"),
        "frozen_training_policy": {"full_epochs": protocol.get("full_epochs"),
            "fidelity_overrides": protocol.get("fidelity_overrides"),
            "training_seeds": protocol.get("training_seeds"),
            "interpretation": "Maximum epochs and early-stop rules are frozen runtime invariants. A build_encoder hook cannot change them. Fewer realized epochs does not prove unequal budgets."},
        "missing_inputs": missing,
    })


def implementation_lifecycle_context(camp: Path, state: dict[str, Any], target: str,
                                     available: list[str]) -> dict[str, Any]:
    """Explain the existing implementation gate without materializing a candidate."""
    from react_agent.eeg_research.agentic.experiment_gate import (
        ExperimentResolutionError, resolve_approved_experiment,
    )
    repair = state.get('repair_task') if isinstance(state.get('repair_task'), dict) else {}
    repairing = bool(repair.get('candidate_id') == target and int(repair.get('remaining') or 0) > 0)
    view = {'target_id': target, 'available': 'implement_candidate' in available,
            'stage': 'repair_existing_candidate' if repairing else 'implement_registered_design',
            'authority': 'Read-only explanation of the existing resolver. Execution revalidates all gates; this view grants no approval.'}
    if not view['available']:
        return {**view, 'status': 'implementation_not_currently_available'}
    assigned = state.get('experiment') if isinstance(state.get('experiment'), dict) else {}
    try:
        resolved = resolve_approved_experiment(camp, spec_ref=state.get('experiment_ref'),
            expected_hash=assigned.get('spec_hash') if not repairing else None,
            target_id=target, attempt_id=repair.get('attempt_id'), state=state,
            action='repair' if repairing else 'implement')
    except ExperimentResolutionError as exc:
        return {**view, 'status': 'design_resolution_failed', 'reason': exc.reason}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**view, 'status': 'design_resolution_failed', 'reason': type(exc).__name__}
    workspace = camp / 'candidates' / target
    source_exists = (workspace / 'extension/eeg_candidate.py').is_file()
    binding_exists = load_approved_binding(camp, target) is not None
    expected_absence = []
    if not repairing:
        if not source_exists: expected_absence.append('source_not_yet_generated_by_implement_candidate')
        if not binding_exists: expected_absence.append('candidate_binding_materialized_by_worker_from_resolved_registered_design')
    return development_view({**view, 'status': 'registered_design_verified_for_existing_implementation_gate',
        'spec_hash': resolved.get('spec_hash'), 'experiment_ref': state.get('experiment_ref'),
        'approval_record': resolved.get('approval_record'),
        'source_exists': source_exists, 'candidate_binding_exists': binding_exists,
        'expected_preimplementation_absence': expected_absence,
        'parent_candidate_id': resolved.get('parent_candidate_id'),
        'control_candidate_id': resolved.get('control_candidate_id'),
        'workflow': ['The registered approved design authorizes the implementation gate, not training.',
            'The worker creates the candidate workspace and binding, materializes parent source, and invokes the real Coder.',
            'CPU checks and Reviewer must pass before the candidate becomes eligible for training.',
            'A new candidate source cannot be required before the action whose purpose is to create that source.'],
        'limits': ['An available implementation action can compete with other useful actions; this view does not force a selection.',
            'Existing repair/training source binding, registered spec hashes, review, capability and budget gates remain unchanged.']})


def training_target_source_context(camp: Path, state: dict[str, Any], eligible: dict[str, list[str]]) -> dict[str, Any]:
    """Expose actual sources of current training targets, separately from proposals."""
    names = sorted({str(target) for action in ('run_pilot', 'run_full', 'replicate')
                    for target in eligible.get(action, [])})
    current = str(state.get('candidate_id') or '')
    if current in names:
        names.remove(current); names.insert(0, current)
    selected = names[:4]
    return {'by_candidate': {name: candidate_context(camp, name) for name in selected},
            'omitted_target_ids': names[4:], 'source_limit': 4,
            'scope': 'Current source/spec-bound candidate code for eligible training targets. A research proposal or pending design is not this source. Missing/truncated source remains explicit.'}


def verified_encoder_structural_facts(camp: Path, state: dict[str, Any], *, target_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Deliver registered source/checkpoint-bound synthetic encoder facts."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    facts = []; seen = set(); protocol = load_protocol(camp) or {}
    for evidence in reversed(state.get('evidence') or []):
        if evidence.get('kind') != 'diagnostic_bundle' or evidence.get('status') != 'verified': continue
        for ref in evidence.get('artifact_refs') or []:
            if ref in seen: continue
            try:
                record = resolve_verified_artifact(camp, ref)
                report = json.loads(Path(record['path']).read_text())
                if report.get('schema_version') != 'eeg_research.c8_actual_pooling_factual_audit.v1': continue
                target = report.get('candidate_id')
                if target_ids is not None and target not in target_ids: continue
                binding = load_approved_binding(camp, str(target)) or {}
                if (report.get('final_test_accessed') is not False or report.get('gpu_seconds') != 0
                        or report.get('execution_fingerprint') != protocol.get('fingerprint')
                        or report.get('sample_count_per_position') != 32
                        or binding.get('source_hash') != report.get('source_hash')
                        or binding.get('spec_hash') != report.get('spec_hash')): continue
                current_source = camp / 'candidates' / str(target) / 'extension/eeg_candidate.py'
                if file_digest(current_source) != report['source_hash']: continue
                for path, digest in report['artifact_hashes'].items():
                    p = Path(path)
                    if not p.resolve().is_relative_to(camp.resolve()) or file_digest(p) != digest:
                        raise ValueError('encoder_structural_artifact_binding_changed')
                run = next(e for e in state.get('evidence') or [] if e.get('job_id') == report.get('job_id')
                           and e.get('evaluation_valid') is True and e.get('source_hash') == report['source_hash']
                           and e.get('spec_hash') == report['spec_hash'] and e.get('seed') == report['seed'])
                seen.add(ref)
                facts.append(development_view({'evidence_id': evidence['evidence_id'], 'artifact_id': ref,
                    'artifact_sha256': record['sha256'], 'run_evidence_id': run['evidence_id'],
                    **{key: report[key] for key in ('candidate_id', 'source_hash', 'spec_hash', 'job_id', 'seed',
                        'scope', 'sample_count_per_position', 'positions', 'measurements', 'actual_encoder_parameters',
                        'source_pooling_expression', 'corrections', 'limitations', 'provenance')},
                    'authority': 'Verified external synthetic measurement/correction; no new EEG score or training authorization.'}))
            except (OSError, ValueError, KeyError, TypeError, StopIteration): continue
    return facts


def matched_diagnostics(state: dict[str, Any], target: str) -> dict[str, Any] | None:
    """Never borrow another target's latest checkpoint diagnostics."""
    for row in reversed(state.get("evidence") or []):
        if row.get("candidate_id") != target or row.get("evaluation_valid") is not True:
            continue
        if row.get("diagnostics") or row.get("diagnostic_ref"):
            return development_view({"summary": row.get("diagnostics"), "ref": row.get("diagnostic_ref"),
                "evidence_id": row.get("evidence_id"), "candidate_id": target,
                "checkpoint_id": row.get("checkpoint_id"), "evaluation_hash": row.get("contract_fingerprint")})
    return None


def load_analysis_views(camp: Path, state: dict[str, Any], *, target: str | None = None) -> list[dict[str, Any]]:
    """Only registered bytes belonging to an analysis pointer are full handoffs."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact

    views = []
    latest_completed = {}
    superseded = {}
    for row in state.get("evidence") or []:
        if row.get("kind") != "analysis" or (target and row.get("candidate_id") != target):
            continue
        key = row.get("run_evidence_id")
        if key and row.get("status") == "completed" and row.get("analysis_artifact_id"):
            try:
                registered = resolve_verified_artifact(camp, row["analysis_artifact_id"])
                if state.get("evaluation_mode") == "loso_method_search":
                    registered = development_artifact(camp, row["analysis_artifact_id"])
                body = json.loads(Path(registered["path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if registered.get("kind") == "analysis" and body.get("status") == "completed":
                previous = latest_completed.get(key)
                if previous:superseded.setdefault(key, []).append(previous.get("evidence_id"))
                latest_completed[key] = row
    for row in state.get("evidence") or []:
        if row.get("kind") != "analysis" or (target and row.get("candidate_id") != target):
            continue
        key = row.get("run_evidence_id")
        current = latest_completed.get(key)
        if row.get("status") == "completed" and current and current.get("evidence_id") != row.get("evidence_id"):
            continue
        ref = row.get("analysis_artifact_id")
        if not ref:
            continue
        try:
            registered = resolve_verified_artifact(camp, ref)
            if state.get("evaluation_mode") == "loso_method_search":
                registered = development_artifact(camp, ref)
            body = json.loads(Path(registered["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            views.append({"evidence_id": row.get("evidence_id"), "artifact_id": ref,
                          "verification_status": "invalid", "missing_inputs": ["analysis_hash_or_file"]})
            continue
        if registered.get("kind") != "analysis":
            continue
        payload = body.get("payload") or body.get("reply") or body
        views.append(development_view({"evidence_id": row.get("evidence_id"), "candidate_id": row.get("candidate_id"),
            "artifact_id": ref, "content_hash": registered["sha256"], "verification_status": "verified",
            "completion_status": body.get("status"), "run_evidence_id": key,
            "superseded_analysis_evidence_ids": superseded.get(key, []) if current is row else [],
            "authority": "llm_interpretation", "input_digest": body.get("input_digest"), "payload": payload}))
    return views[-8:]


def _registry_rows(camp: Path) -> list[dict[str, Any]]:
    from react_agent.eeg_research.agentic.artifacts import REGISTRY
    path = camp / REGISTRY
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def development_artifact(camp: Path, artifact_id: str) -> dict[str, Any]:
    """Memory reading has a stricter boundary than approved external source consumers."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    row = resolve_verified_artifact(camp, artifact_id)
    if row.get("kind") in {"handoff_compaction_input", "handoff_compaction"}:
        raise ValueError("delivery_artifact_not_development_evidence")
    path = Path(str(row["path"])).resolve()
    if not path.is_relative_to(camp.resolve()):
        raise ValueError("artifact_outside_campaign")
    if path.suffix != ".json":
        raise ValueError("artifact_requires_structured_development_view")
    if path.stat().st_size > READ_FILE_BYTES:
        raise ValueError("artifact_exceeds_read_file_limit")
    body = json.loads(path.read_text(encoding="utf-8"))
    def contains_benchmark_scope(value: Any) -> bool:
        if isinstance(value, dict):
            return (any(value.get(key) == "method_development_benchmark" for key in
                        ("scope", "data_role", "partition", "metric_scope"))
                    or any(contains_benchmark_scope(child) for child in value.values()))
        return isinstance(value, list) and any(contains_benchmark_scope(child) for child in value)
    if row.get("kind") != "method_suite" and contains_benchmark_scope(body):
        payload = body.get("payload")
        origin = payload.get("method_benchmark_origin") if isinstance(payload, dict) else None
        if row.get("kind") not in {"analysis", "lessons"} or not isinstance(origin, dict):
            raise ValueError("benchmark_artifact_requires_native_complete_suite_kind")
        return _derived_method_artifact(camp, row, body, origin)
    if row.get("kind") == "method_suite":
        # An artifact kind/scope label never grants benchmark permission. Bind
        # the opt-in goal, frozen manifest, exact native path and all ten score
        # dependencies, then recompute the aggregate before permitting a read.
        from react_agent.eeg_research.agentic.method_suite import MODE, manifest, read_aggregate
        goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
        if goal.get("evaluation_mode") != MODE:
            raise ValueError("method_benchmark_read_requires_explicit_opt_in")
        permission = manifest(camp)["benchmark_permission"]
        suite_id = path.parent.name
        expected = (camp / "suites" / suite_id / "aggregate.json").resolve()
        if path != expected or not permission.get("authorized") or permission.get("partial_feedback") is not False:
            raise ValueError("method_benchmark_read_identity_or_permission_mismatch")
        computed = read_aggregate(camp, suite_id)
        if (body.get("aggregate") != computed or computed.get("status") != "complete"
                or computed.get("coverage") != 10 or computed.get("required_fold_count") != 10
                or computed.get("scope") != "method_development_benchmark"
                or computed.get("independent_final_test") is not False):
            raise ValueError("method_benchmark_read_requires_verified_complete_suite")
        return {**row, "development_scope": "method_development_benchmark",
                "native_completion_verified": True}
    if development_view(body) is None:
        raise ValueError("artifact_scope_forbidden")
    return row


def _derived_method_artifact(camp: Path, row: dict[str, Any], body: dict[str, Any],
                             origin: dict[str, Any]) -> dict[str, Any]:
    """Allow only a completed native role receipt bound to a current full suite.

    Registered bytes, producer/task identity, aggregate input dependency, current
    suite/source and explicit benchmark permission must all agree. A scope label
    or a model-provided origin alone cannot grant this derived read permission.
    """
    from react_agent.eeg_research.agentic.task_ledger import latest
    from react_agent.eeg_research.agentic.roles import validate_role_result
    from react_agent.eeg_research.agentic.method_suite import verified_suite_records
    from react_agent.eeg_research.agentic.semantic_memory import unsafe_source
    if unsafe_source(body):
        raise ValueError("method_derived_artifact_forbidden_scope")
    producer = row.get("producer_task_id")
    task = latest(camp, str(producer)) if producer else None
    expected_role = {"analysis": "result_analyst", "lessons": "memory_curator"}[row["kind"]]
    if (not task or task.get("status") != "completed" or task.get("role") != expected_role
            or task.get("artifact_id") != row["artifact_id"] or body.get("status") != "completed"):
        raise ValueError("method_derived_artifact_requires_completed_producer")
    envelope = validate_role_result(body, task)
    state = json.loads((camp / "campaign_state.json").read_text(encoding="utf-8"))
    source = next((item for item in verified_suite_records(camp, state)
                   if item["evidence_id"] == origin.get("run_evidence_id")), None)
    if source is None or origin != method_benchmark_origin(source):
        raise ValueError("method_derived_artifact_native_origin_mismatch")
    native = development_artifact(camp, origin["aggregate_artifact_id"])
    if (native["path"] != origin["suite_ref"] or native["sha256"] != origin["result_hash"]
            or not any(Path(ref).resolve() == Path(native["path"]).resolve()
                       for ref in task.get("input_artifact_refs") or [])):
        raise ValueError("method_derived_artifact_aggregate_input_unbound")
    if row["kind"] == "analysis" and (task.get("candidate_id") != source["candidate_id"]
                                      or row.get("candidate_id") != source["candidate_id"]):
        raise ValueError("method_derived_artifact_candidate_mismatch")
    pointers = [item for item in state.get("evidence") or []
                if item.get("analysis_artifact_id") == row["artifact_id"]]
    if any(item.get("run_evidence_id") != source["evidence_id"] or item.get("status") != "completed"
           for item in pointers):
        raise ValueError("method_derived_artifact_analysis_pointer_mismatch")
    if envelope["input_digest"] != task["input_digest"]:
        raise ValueError("method_derived_artifact_task_input_mismatch")
    return {**row, "development_scope": "method_development_benchmark",
            "native_completion_verified": True, "method_benchmark_origin": origin}


def artifact_index(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Identifiers and hashes are an index; they do not claim the model read the bytes."""
    rows = []
    evidence = state.get("evidence") or []
    for record in _registry_rows(camp):
        ref = record.get("artifact_id")
        if not ref:
            continue
        missing = []
        validated = None
        try:
            validated = development_artifact(camp, str(ref))
        except (OSError, ValueError, KeyError) as exc:
            missing = [str(exc)]
        linked = [row for row in evidence if ref in (row.get("artifact_refs") or []) or
                  ref in (row.get("analysis_artifact_id"), row.get("audit_artifact_id"))]
        completion = "unknown"
        completion_verified = False
        if validated and validated.get("native_completion_verified"):
            completion = "completed"
            completion_verified = True
        if not missing and record.get("producer_task_id"):
            from react_agent.eeg_research.agentic.task_ledger import latest
            from react_agent.eeg_research.agentic.roles import validate_role_result, RoleResultError
            task = latest(camp, str(record["producer_task_id"]))
            try:
                body = json.loads(Path(str(record["path"])).read_text(encoding="utf-8"))
                envelope = validate_role_result(body, task) if task else None
                if envelope:
                    completion = envelope["status"]
                    completion_verified = True
            except (OSError, ValueError, KeyError, RoleResultError):
                pass
        rows.append({"artifact_id": ref, "kind": record.get("kind"),
            "candidate_id": record.get("candidate_id"),
            "question_id": next((row.get("question_id") for row in linked if row.get("question_id")), None),
            "title": Path(str(record.get("path") or "")).name,
            "content_hash": record.get("sha256"), "producer_task_id": record.get("producer_task_id"),
            "evidence_refs": [row["evidence_id"] for row in linked if row.get("evidence_id")],
            "verification_status": "invalid" if missing else "verified",
            "completion_status": completion, "completion_identity_verified": completion_verified,
            "scope": (validated or {}).get("development_scope", "development") if not missing else "unavailable",
            "missing_inputs": missing, "content_injected": False})
    return rows


def read_epoch(state: dict[str, Any]) -> str:
    from react_agent.eeg_research.agentic.planner import evidence_count
    return str(evidence_count(state))


def _read_journal_records(camp: Path) -> list[dict[str, Any]]:
    path = camp / READ_JOURNAL
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"artifact_read_journal_invalid:{number}") from exc
        if not isinstance(row, dict) or row.get("record_type", "read") not in {"read", "delivery"}:
            raise ValueError(f"artifact_read_journal_invalid:{number}")
        if row.get("record_type") == "delivery":
            body = {key: value for key, value in row.items() if key != "delivery_id"}
            expected = "delivery_" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if (row.get("delivery_id") != expected or not isinstance(row.get("reads"), list)
                    or not row.get("consumer") or not row.get("consumer_request_id") or row.get("receipt_id")):
                raise ValueError(f"artifact_delivery_journal_invalid:{number}")
        records.append(row)
    return records


def read_history(camp: Path) -> list[dict[str, Any]]:
    return [row for row in _read_journal_records(camp) if row.get("record_type", "read") == "read"]


def read_digest(request: dict[str, Any], content_hash: str | None) -> str:
    from react_agent.eeg_research.agentic.artifacts import request_digest
    return request_digest(request={"artifact_id": request.get("artifact_id"),
                                  "start": request.get("start", 0), "end": request.get("end"),
                                  "content_hash": content_hash, "scope": "development_view"})


def _receipt_id(row: dict[str, Any]) -> str:
    keys = ("artifact_id", "request_digest", "request", "decision_id", "epoch", "status", "sha256",
            "view_sha256", "range_scope", "start", "end", "missing_inputs")
    identity = {key: row.get(key) for key in keys}
    identity["text_sha256"] = hashlib.sha256((row.get("text") or "").encode("utf-8")).hexdigest()
    return "read_" + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _append_read_record(camp: Path, row: dict[str, Any]) -> None:
    import os
    with (camp / READ_JOURNAL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_read_receipt(camp: Path, saved: dict[str, Any]) -> dict[str, Any]:
    """Recompute source, request, filtered view, range and actual bytes before reuse."""
    from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range
    from react_agent.eeg_research.agentic.schemas import ArtifactReadRequest
    if saved.get("receipt_id") and saved["receipt_id"] != _receipt_id(saved):
        raise ValueError("artifact_read_receipt_identity_changed")
    registered = lookup(camp, str(saved.get("artifact_id") or ""))
    source_hash = saved.get("sha256") if saved.get("status") == "read" else (registered or {}).get("sha256")
    request = saved.get("request")
    if request is None:
        path = camp / "decisions" / f"{saved.get('decision_id')}.json"
        raw = json.loads(path.read_text()).get("raw") or {} if path.is_file() else {}
        candidates = list(raw.get("read_requests") or [])
        candidates += [{"artifact_id": saved.get("artifact_id"), "start": saved.get("start", 0), "end": end}
                       for end in (None, saved.get("end"), int(saved.get("start") or 0) + READ_RANGE_CHARS)]
        request = next((row for row in candidates if isinstance(row, dict)
                        and read_digest(row, source_hash) == saved.get("request_digest")), None)
    if not isinstance(request, dict) or read_digest(request, source_hash) != saved.get("request_digest"):
        raise ValueError("artifact_read_request_identity_changed")
    parsed = ArtifactReadRequest.model_validate(request)
    if parsed.artifact_id != saved.get("artifact_id") or (parsed.end is not None and parsed.end - parsed.start > READ_RANGE_CHARS):
        raise ValueError("artifact_read_request_identity_changed")
    if saved.get("status") != "read":
        # Failed receipts authorize no text. Their original request still binds
        # replay identity and preserves the existing charged-attempt policy.
        if saved.get("text") is not None:
            raise ValueError("artifact_read_failed_receipt_has_text")
        return {**saved, "content_injected": False}
    current = development_artifact(camp, parsed.artifact_id)
    if current["sha256"] != saved.get("sha256"):
        raise ValueError("artifact_read_hash_changed")
    start, end = saved.get("start"), saved.get("end")
    if (not isinstance(start, int) or not isinstance(end, int) or start != parsed.start
            or end < start or end - start > READ_RANGE_CHARS or (parsed.end is not None and end > parsed.end)
            or saved.get("range_scope") != "development_view"):
        raise ValueError("artifact_read_range_identity_changed")
    view = read_verified_range(camp, parsed.artifact_id, start=start, end=end,
                               json_view=development_view, page_chars=READ_RANGE_CHARS)
    if view["view_sha256"] != saved.get("view_sha256") or view["text"] != saved.get("text") or view["end"] != end:
        raise ValueError("artifact_read_view_changed")
    return {**saved, **view, "content_injected": False, "delivery_status": "prepared"}


def rebuild_read_projection(camp: Path, state: dict[str, Any]) -> None:
    history = read_history(camp)
    state["artifact_read_history"] = [{key: row.get(key) for key in
        ("artifact_id", "request_digest", "receipt_id", "decision_id", "epoch", "status")} for row in history]
    if not history:
        return
    latest_id = history[-1].get("decision_id")
    if latest_id is None:
        # Optional direct callers have no persisted batch identity. Preserve a
        # still-verifiable latest projection, rather than guess a history-wide
        # default batch. Explicit artifact/page selection remains available.
        saved = state.get("artifact_read_results") or []
        rows = [row for item in saved for row in history if row.get("decision_id") is None
                and row.get("request_digest") == item.get("request_digest")]
        if not rows:
            state["artifact_read_results"] = [{"status": "missing_input", "text": None,
                "missing_inputs": ["artifact_read_batch_identity_missing"], "content_injected": False}]
            return
    else:
        rows = [row for row in history if row.get("decision_id") == latest_id]
    results = []
    for row in rows:
        try:
            results.append(verify_read_receipt(camp, row))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({"artifact_id": row.get("artifact_id"), "request_digest": row.get("request_digest"),
                            "status": "missing_input", "missing_inputs": [str(exc)], "text": None})
    state["artifact_read_results"] = results


def consume_artifact_reads(camp: Path, state: dict[str, Any], requests: list[dict[str, Any]],
                           decision_id: str | None) -> dict[str, Any]:
    """Journal bounded attempts once; verified replay of the same decision is free."""
    from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range
    from react_agent.eeg_research.agentic.schemas import ArtifactReadRequest
    record = next((row for row in state.get("decisions") or [] if row.get("decision_id") == decision_id), {})
    epoch = str(record["evidence_count"]) if "evidence_count" in record else read_epoch(state)
    history = read_history(camp)
    results = []
    used = sum(row.get("epoch") == epoch and not row.get("replayed") for row in history)
    remaining = READ_TOTAL_CHARS
    if len(requests) > READ_MAX_IDS:
        return {"status": "limited", "missing_inputs": ["artifact_read_request_limit"], "results": []}
    for raw in requests:
        ref = raw.get("artifact_id") if isinstance(raw, dict) else None
        registered = lookup(camp, str(ref)) if ref else None
        digest = read_digest(raw, None if registered is None else registered.get("sha256"))
        prior = [row for row in history if row.get("request_digest") == digest]
        previous = next((row for row in prior if row.get("decision_id") == decision_id), None)
        if previous:
            try:
                view = verify_read_receipt(camp, previous)
                if len(view.get("text") or "") > remaining:
                    raise ValueError("artifact_read_total_limit")
                remaining -= len(view.get("text") or "")
                results.append({**view, "replayed": True})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results.append({"artifact_id": ref, "status": "missing_input", "text": None,
                                "missing_inputs": [str(exc)], "replayed": True})
            continue
        if prior:
            results.append({"artifact_id": ref, "status": "missing_input", "text": None,
                            "missing_inputs": ["artifact_read_replayed"]})
            continue
        result = {"artifact_id": ref, "request_digest": digest, "request": raw,
                  "decision_id": decision_id, "record_type": "read", "epoch": epoch,
                  "status": "missing_input", "missing_inputs": [], "text": None, "content_injected": False}
        if used >= READS_PER_EVIDENCE_EPOCH or remaining <= 0:
            result.update(status="limited", missing_inputs=["artifact_read_epoch_or_context_limit"])
            results.append(result)
            continue
        try:
            request = ArtifactReadRequest.model_validate(raw)
            if request.end is not None and request.end - request.start > READ_RANGE_CHARS:
                raise ValueError("artifact_read_range_limit")
            if request.end is not None and request.end - request.start > remaining:
                raise ValueError("artifact_read_total_limit")
            development_artifact(camp, request.artifact_id)
            view = read_verified_range(camp, request.artifact_id, start=request.start, end=request.end,
                                       json_view=development_view, max_chars=min(READ_RANGE_CHARS, remaining),
                                       page_chars=READ_RANGE_CHARS)
            result.update(view, status="read")
            remaining -= len(view["text"])
        except (OSError, ValueError, KeyError) as exc:
            result["missing_inputs"] = [str(exc)]
        result["receipt_id"] = _receipt_id(result)
        _append_read_record(camp, result)
        history.append(result)
        used += 1
        results.append(result)
    state["artifact_read_history"] = [{key: row.get(key) for key in
        ("artifact_id", "request_digest", "receipt_id", "decision_id", "epoch", "status")} for row in history]
    state["artifact_read_results"] = results
    return {"status": "completed", "results": results, "remaining_reads": max(0, READS_PER_EVIDENCE_EPOCH - used)}


def verified_read_context(camp: Path, state: dict[str, Any], *, artifact_refs: list[str] | None = None,
                          request_digests: list[str] | None = None) -> list[dict[str, Any]]:
    """One verifier serves default latest-batch and explicitly selected history."""
    from react_agent.eeg_research.agentic.artifacts import read_verified_range
    history = read_history(camp)
    explicit = artifact_refs is not None or request_digests is not None
    if explicit:
        refs, digests = artifact_refs or [], request_digests or []
        if digests:
            selected = [next((row for row in history if row.get("request_digest") == digest
                             and (not refs or row.get("artifact_id") in refs)),
                            {"request_digest": digest, "status": "missing_input", "missing_inputs": ["artifact_page_not_read"]})
                        for digest in digests]
        else:
            selected = [row for row in history if row.get("artifact_id") in refs]
        missing = [ref for ref in refs if not any(row.get("artifact_id") == ref for row in selected)]
        selected += [{"artifact_id": ref, "status": "missing_input", "missing_inputs": ["artifact_not_read"]} for ref in missing]
    else:
        selected = state.get("artifact_read_results") or []
        if not selected and history:
            latest_id = history[-1].get("decision_id")
            selected = ([row for row in history if row.get("decision_id") == latest_id] if latest_id is not None else
                        [{"status": "missing_input", "missing_inputs": ["artifact_read_batch_identity_missing"]}])
    rows, unique, chars, fragments = [], set(), READ_TOTAL_CHARS, 0
    for saved in selected:
        try:
            matching = [row for row in history if row.get("request_digest") == saved.get("request_digest")
                        and row.get("decision_id") == saved.get("decision_id")]
            if not matching or any(row != matching[0] for row in matching[1:]):
                raise ValueError("artifact_read_journal_binding_missing")
            recorded = matching[0]
            if any(recorded.get(key) != saved.get(key) for key in
                   ("artifact_id", "sha256", "start", "end", "view_sha256", "receipt_id", "text")):
                raise ValueError("artifact_read_journal_binding_missing")
            view = verify_read_receipt(camp, recorded)
            if view.get("status") == "read":
                if len(unique | {view["artifact_id"]}) > READ_MAX_IDS:
                    raise ValueError("artifact_read_unique_artifact_limit")
                if fragments >= READ_MAX_IDS:
                    raise ValueError("artifact_read_fragment_limit")
                if chars <= 0:
                    raise ValueError("artifact_read_context_limit")
                unique.add(view["artifact_id"])
                fragments += 1
                if len(view["text"]) > chars:
                    partial = read_verified_range(camp, view["artifact_id"], start=view["start"], end=view["start"] + chars,
                                                  json_view=development_view, page_chars=READ_RANGE_CHARS)
                    view = {**view, **partial, "receipt_range": [view["start"], view["end"]],
                            "delivery_complete": False, "missing_inputs": ["artifact_read_context_truncated"]}
                else:
                    view["delivery_complete"] = True
                chars -= len(view["text"])
            rows.append(view)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({"artifact_id": saved.get("artifact_id"), "request_digest": saved.get("request_digest"),
                         "receipt_id": saved.get("receipt_id"), "status": "missing_input", "text": None,
                         "content_injected": False, "delivery_complete": False,
                         "missing_inputs": saved.get("missing_inputs") or [str(exc)]})
    return rows


def prepare_read_delivery(rows: list[dict[str, Any]]) -> None:
    """Mark the prepared request before its task input digest is calculated."""
    for row in rows:
        if row.get("status") == "read":
            row["content_injected"] = True
            row["delivery_status"] = "request_constructed"


def record_read_delivery(camp: Path, rows: list[dict[str, Any]], *, consumer: str,
                         request_id: str, input_digest: str | None = None) -> None:
    """Record constructed consumer requests, never infer past delivery from a receipt."""
    prepare_read_delivery(rows)
    delivered = []
    for row in rows:
        if row.get("status") != "read":
            continue
        row["content_injected"] = True
        row["delivery_status"] = "request_constructed"
        delivered.append({key: row.get(key) for key in ("artifact_id", "request_digest", "receipt_id", "sha256",
                         "view_sha256", "range_scope", "start", "end", "receipt_range", "delivery_complete")})
    if not delivered:
        return
    body = {"record_type": "delivery", "consumer": consumer, "consumer_request_id": request_id,
            "consumer_input_digest": input_digest, "delivery_phase": "request_constructed", "reads": delivered}
    body["delivery_id"] = "delivery_" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if not any(row.get("delivery_id") == body["delivery_id"] for row in _read_journal_records(camp)):
        _append_read_record(camp, body)


def bounded_control_view(value: Any, *, budget: int = CONTROL_TEXT_CHARS) -> tuple[Any, dict[str, Any]]:
    """Bound prose without losing control identities, candidates or open issues.

    The projection preserves JSON shape. Explicit paths locate omitted prose in
    the registered artifact/plan; it never overwrites authoritative content.
    """
    remaining = max(0, budget)
    truncated = []
    exact = {"action", "status", "kind", "fidelity", "verdict", "severity", "category", "scope", "completion_status",
             "verification_status", "model_audit_status", "resolution_kind", "operation", "revision_operation",
             "hypothesis_assessment", "authority", "interpretation_authority",
             "report_ref", "audited_report_ref", "request_digest", "input_digest", "resource_status"}
    def project(item: Any, key: str = "", path: str = "$") -> Any:
        nonlocal remaining
        if isinstance(item, dict):
            return {name: project(child, name, path + "." + name) for name, child in item.items()}
        if isinstance(item, list):
            return [project(child, key, path + f"[{index}]") for index, child in enumerate(item)]
        if isinstance(item, str):
            if key in exact or key.endswith(("_id", "_ids", "_ref", "_refs", "_hash", "_digest")):
                return item
            amount = min(len(item), HISTORY_ITEM_CHARS, remaining)
            remaining -= amount
            if amount < len(item):
                truncated.append({"path": path, "shown_characters": amount, "total_characters": len(item),
                                  "next_range": [amount, min(len(item), amount + READ_RANGE_CHARS)]})
                return item[:amount]
        return item
    clean = project(development_view(value))
    return clean, {"truncated": bool(truncated), "text_character_limit": budget,
                   "shown_text_characters": budget - remaining, "truncated_fields": truncated,
                   "identity_and_issue_rows_retained": True, "omitted_content": "registered artifact or versioned plan"}


def bounded_history(value: Any, *, budget: int = HISTORY_CHARS) -> tuple[Any, dict[str, Any]]:
    """Keep recent history bounded, with an explicit index of omitted rows."""
    if isinstance(value, list):
        rows, used = [], 0
        for original in reversed(value[-HISTORY_ROWS:]):
            item = development_view(original)
            encoded = json.dumps(item, ensure_ascii=False, default=str)
            identity = {key: item.get(key) for key in ("evidence_id", "episode_id", "lesson_id", "artifact_id", "candidate_id", "kind", "fidelity", "diagnostic_ref",
                        "verification_status", "completion_status", "run_evidence_id", "content_hash", "input_digest", "authority", "status")
                        if isinstance(item, dict) and item.get(key) is not None}
            if len(encoded) > HISTORY_ITEM_CHARS:
                item = {**identity, "summary_excerpt": encoded[:HISTORY_ITEM_CHARS], "truncated": True,
                        "next_range": [HISTORY_ITEM_CHARS, min(len(encoded), HISTORY_ITEM_CHARS + READ_RANGE_CHARS)]}
            size = len(json.dumps(item, ensure_ascii=False, default=str))
            if used + size > budget:
                break
            rows.append(item)
            used += size
        rows.reverse()
        return rows, {"total_rows": len(value), "shown_rows": len(rows), "omitted_row_range": [0, max(0, len(value) - len(rows))],
                      "truncated": len(rows) < len(value) or any(isinstance(row, dict) and row.get("truncated") for row in rows),
                      "character_limit": budget}
    return bounded_control_view(value, budget=budget)


def selected_read_context(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """A worker receives only explicitly cited artifacts/pages, with visible gaps."""
    return verified_read_context(camp, state, artifact_refs=list(state.get("active_artifact_refs") or []),
                                 request_digests=list(state.get("active_read_digests") or []))


def analysis_experiment_binding(camp: Path, latest: dict[str, Any]) -> dict[str, Any]:
    """Bind analysis to the approved spec and source of this exact accepted run."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.experiment_gate import _spec_hash, experiment_is_approved
    target=str(latest.get("candidate_id") or "")
    if target=="baseline":return {"status":"frozen_baseline", "missing_inputs":[]}
    binding=load_approved_binding(camp,target) or {}
    try:
        if not target or not binding:raise ValueError("approved_binding_missing")
        registry=resolve_verified_artifact(camp,str(binding.get("spec_ref") or ""))
        artifact=json.loads(Path(registry["path"]).read_text(encoding="utf-8"))
        spec=artifact.get("experiment_spec") or artifact.get("experiment") or artifact
        if not isinstance(spec,dict):raise ValueError("approved_spec_missing")
        if not experiment_is_approved(spec):raise ValueError("spec_not_approved")
        expected=latest.get("spec_hash")
        if not expected or _spec_hash(spec)!=expected or binding.get("spec_hash")!=expected:
            raise ValueError("run_spec_hash_mismatch")
        if binding.get("target_id")!=target or binding.get("source_hash")!=latest.get("source_hash"):
            raise ValueError("run_source_binding_mismatch")
        method_identity = {}
        if latest.get("kind") == "method_suite":
            # A complete method is ten native folds, not a legacy single job.
            # Its immutable suite and source/review dependencies carry the
            # exact approved intent; never fabricate a job_dir or hypothesis.
            from react_agent.eeg_research.agentic.method_suite import read_aggregate, validate_suite
            from react_agent.eeg_research.agentic.native_patch import enforce_requirement_review
            suite_id = str(latest.get("suite_id") or "")
            if not suite_id or Path(suite_id).name != suite_id or suite_id in {".", ".."}:
                raise ValueError("method_suite_identity_missing")
            suite = validate_suite(camp, suite_id)
            suite_path = camp / "suites" / suite_id / "suite_manifest.json"
            aggregate_path = suite_path.with_name("aggregate.json")
            aggregate = read_aggregate(camp, suite_id)
            if (latest.get("evaluation_valid") is not True or aggregate != latest.get("aggregate")
                    or latest.get("result_hash") != file_digest(aggregate_path)
                    or Path(str(latest.get("suite_ref") or "")).resolve() != aggregate_path.resolve()
                    or latest.get("suite_hash") != suite["suite_hash"]
                    or latest.get("method_revision") != suite["method"]["method_revision"]):
                raise ValueError("analysis_method_suite_identity_mismatch")
            method = suite["method"]
            if (method["candidate_id"] != target or method["spec_hash"] != expected
                    or method["source_hash"] != latest.get("source_hash")
                    or method["approved_binding"] != binding
                    or method["approval_ref"] != binding.get("spec_ref")):
                raise ValueError("analysis_method_binding_mismatch")
            workspace = camp / "candidates" / target
            source = json.loads((workspace / "source_manifest.json").read_text())
            path = Path(str(source.get("entry") or ""))
            if (source != method["source_manifest"] or source.get("entry_sha256") != method["source_hash"]
                    or not path.resolve().is_relative_to(workspace.resolve())
                    or file_digest(path) != method["source_hash"]):
                raise ValueError("loaded_source_changed")
            checks = json.loads((workspace / "checks.json").read_text())
            review = enforce_requirement_review(workspace, {"experiment": spec},
                json.loads((workspace / "review.json").read_text()), checks)
            if review.get("status") != "ready" or not checks.get("ok"):
                raise ValueError("analysis_method_requires_current_ready_review_and_check")
            method_identity = {"suite_id": suite_id, "suite_ref": str(aggregate_path.resolve()),
                               "suite_manifest_ref": str(suite_path.resolve()),
                               "result_hash": latest["result_hash"],
                               "suite_hash": suite["suite_hash"], "method_revision": method["method_revision"]}
        else:
            job=Path(str(latest.get("job_dir") or ""))
            if not job.resolve().is_relative_to(camp.resolve()):raise ValueError("run_outside_campaign")
            frozen=json.loads((job/"frozen_run_spec.json").read_text())
            source=json.loads((job/"source_binding.json").read_text())
            if frozen.get("spec_hash")!=expected or frozen.get("candidate_id", frozen.get("target_id"))!=target:
                raise ValueError("frozen_run_spec_mismatch")
            path=Path(str(source.get("class_file") or ""));path=path if path.is_absolute() else job/path
            if source.get("file_sha256")!=latest.get("source_hash") or file_digest(path)!=latest.get("source_hash"):
                raise ValueError("loaded_source_changed")
        hypothesis=spec.get("hypothesis")
        if not hypothesis:raise ValueError("approved_hypothesis_missing")
        return {"status":"verified", "hypothesis":hypothesis, "experiment":spec,
                "spec_hash":expected,"source_hash":latest.get("source_hash"),
                "artifact_ref":registry["artifact_id"],"artifact_sha256":registry["sha256"],
                **method_identity, "missing_inputs":[]}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {"status":"unavailable","missing_inputs":[str(exc)]}
