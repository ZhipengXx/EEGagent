"""One bounded, deterministic capture/pack/render boundary for every EEG role."""
from __future__ import annotations

import contextvars
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .budget import compact_schema, measure
from .representations import deduplicate, representations_for, source_kind, validate_pointer_closure
from .role_policy import CORE_CONTRACT, CONTROL_FIELDS, FACT_FIELDS, role_policy
from .sources import ContextSourceRegistry, atomic_immutable, pointer_escape, assert_visible_payload
from .types import (Blocked, CompileRequest, CompileResult, NeedsContext, PreparedRequest,
                    Ready, ReadRequest, RENDER_VERSION, Representation, SourceBlock, canonical, digest)

_IN_MAINTENANCE = contextvars.ContextVar("eeg_context_in_maintenance", default=False)
MAINTENANCE_MODES = frozenset({"summary", "extraction", "consolidation"})
WRAPPERS = frozenset({"observation", "input", "research_context", "context", "payload"})
OPTIONAL_FIELDS = frozenset({"analyses", "memory", "lessons", "retrieved_memory", "retrieved_knowledge", "recent_decisions",
    "artifact_index", "evidence_index", "memory_index", "context_limits", "planner_context_projection",
    "role_context_projection", "planner_budget_reductions", "optional_context_delivery", "report_draft"})
CLOSED_COLLECTIONS = frozenset({"analyses", "evidence", "method_screenings", "method_suites", "run_artifacts"})
SCIENCE_IDENTITY = ("candidate_id", "method_revision", "source_hash", "config_hash", "protocol_hash",
    "evaluation_key", "evaluator_hash", "evaluator", "seed", "scope", "science_scope", "fold_id",
    "held_out_subject", "selection_manifest_hash", "fidelity", "evidence_id", "suite_id", "job_id", "run_id", "artifact_id")


def is_research_planning_input(role: str, payload: dict[str, Any]) -> bool:
    """Identify the explicit science-only staged Planner/Critic projection."""
    return role in {"research_planner", "research_critic"} and payload.get(
        "planning_context_version") in {"research_planning_input.v1", "research_planning_input.v2"}


def is_designer_scientific_input(role: str, payload: dict[str, Any]) -> bool:
    from ..designer_context import is_designer_input
    return is_designer_input(role, payload)


def is_engineering_input(role: str, payload: dict[str, Any]) -> bool:
    """Explicit compact views own their source pages and exact check facts."""
    return ((role == "candidate_reviewer" and payload.get("reviewer_context_version") == "reviewer_implementation_input.v2")
            or (role == "candidate_coder" and payload.get("coder_context_version") == "coder_implementation_input.v2"))


def is_analysis_scientific_input(role: str, payload: dict[str, Any]) -> bool:
    return (role == "result_analyst" and payload.get("review_stage") == "analysis"
            and payload.get("analysis_context_version") == "research_analysis_input.v1")


def _analysis_diagnostic_view(value: Any) -> Any:
    """Remove known export bookkeeping only inside native diagnostic nodes."""
    if isinstance(value, list):
        return [_analysis_diagnostic_view(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    local = {"schema_version", "source_sha256", "source_hash", "spec_hash", "config_hash",
        "checkpoint_hash", "protocol_hash", "evaluator_hash", "evaluation_key", "fingerprint", "artifact_sha256",
        "validation_identity", "training_inference_identity_fingerprint", "dependencies",
        "source_bindings", "created_at", "recorded_at"}
    return {key: Path(item).name if key in {"checkpoint", "checkpoint_id"} and isinstance(item, str)
            else _analysis_diagnostic_view(item) for key, item in value.items() if key not in local}


def _analysis_training_run(run: dict[str, Any]) -> dict[str, Any]:
    result = {key: copy.deepcopy(value) for key, value in run.items() if key not in {
        "source_hash", "spec_hash", "config_hash", "method_revision", "suite_id", "suite_hash", "job_id"}}
    config = result.get("config")
    if isinstance(config, dict) and isinstance(config.get("recipe"), dict):
        # Model/objective/transform keys are scientific configuration and are
        # intentionally untouched, even if a custom parameter is called hash.
        config["recipe"] = {key: value for key, value in config["recipe"].items()
                            if key not in {"data_root", "gpu"}}
    for key in ("training_execution", "development_diagnostics", "checkpoint_selection", "development_metrics"):
        if key in result:
            result[key] = _analysis_diagnostic_view(result[key])
    diagnostics = result.get("development_diagnostics")
    items = diagnostics.get("items") if isinstance(diagnostics, dict) else None
    dynamics = items.get("training_dynamics") if isinstance(items, dict) else None
    if (isinstance(dynamics, dict) and "training_execution" in result
            and dynamics.get("training_execution") == result["training_execution"]):
        # Native exports repeat this record inside training dynamics. Deliver
        # the identical scientific facts once; keep any differing observation.
        dynamics.pop("training_execution", None)
    learning = result.get("development_learning")
    if isinstance(learning, dict) and isinstance(learning.get("points"), list) and learning["points"]:
        # Keep every epoch and metric without repeating the same JSON keys.
        points = learning.pop("points")
        columns = list(dict.fromkeys(key for row in points for key in row))
        # A column recorded twice at different float precision is one curve.
        equivalent = {}
        for index, name in enumerate(columns):
            same = next((kept for kept in columns[:index] if kept not in equivalent
                         and _same_curve([row.get(kept) for row in points], [row.get(name) for row in points])), None)
            if same is not None:
                equivalent[name] = same
        columns = [name for name in columns if name not in equivalent]
        learning["columns"] = columns
        learning["rows"] = [[_curve_value(row.get(key)) for key in columns] for row in points]
        if equivalent:
            learning["equivalent_columns"] = equivalent
    return result


def _curve_value(value: Any) -> Any:
    # Four significant digits resolve every epoch-to-epoch change in these curves;
    # the complete-precision rows remain readable through the cited local source.
    return float(f"{value:.4g}") if isinstance(value, float) else value


def _same_curve(left: list[Any], right: list[Any]) -> bool:
    if not left or len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)):
            return False
    return True


def project_analysis_scientific_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Scientific delivery only; native identities and provenance stay local."""
    fields = ("analysis_context_version", "review_stage", "research_context", "analysis_requirements")
    result = {key: copy.deepcopy(payload[key]) for key in fields if key in payload}
    facts = payload.get("analysis_facts") or {}
    result["analysis_facts"] = {key: copy.deepcopy(facts[key]) for key in (
        "measurement", "coverage", "evaluation_context", "training_runs", "scientific_diagnostics",
        "source_catalog", "allowed_evidence_refs"
    ) if key in facts}
    if isinstance(result["analysis_facts"].get("training_runs"), list):
        result["analysis_facts"]["training_runs"] = [_analysis_training_run(run)
            for run in result["analysis_facts"]["training_runs"]]
    if "scientific_diagnostics" in result["analysis_facts"]:
        result["analysis_facts"]["scientific_diagnostics"] = _analysis_diagnostic_view(
            result["analysis_facts"]["scientific_diagnostics"])
    for key in ("previous_response_error", "previous_response"):
        if key in payload:
            result[key] = copy.deepcopy(payload[key])
    return result


def native_row_identity(row: dict[str, Any]) -> dict[str, Any]:
    identity = {key: row[key] for key in SCIENCE_IDENTITY if key in row}
    scientific = row.get("screening") or row.get("aggregate")
    if isinstance(scientific, dict):
        for key in SCIENCE_IDENTITY:
            if key in scientific:
                if key in identity and canonical(identity[key]) != canonical(scientific[key]):
                    raise ValueError("closed_source_identity_conflict:" + key)
                identity[key] = scientific[key]
    return identity


def row_source_id(field: str, row: dict[str, Any]) -> str:
    key = next((str(row[name]) for name in ("evidence_id", "suite_id", "job_id", "run_id", "artifact_id", "task_id") if row.get(name)), digest(row))
    return "closed:" + field + ":" + key


def _compact_system_schema(system: str) -> str:
    # No generic line/string deletion: only JSON-schema annotation removal.
    matches = list(re.finditer(r"(?m)^(?:OUTPUT_SCHEMA|OUTPUT SCHEMA|INDEPENDENT_OPTION_SCHEMA|INDEPENDENT_PROPOSAL_SCHEMA):?\s*\n", system))
    for match in reversed(matches):
        start = match.end()
        tail = system[start:]
        leading = len(tail) - len(tail.lstrip())
        try:
            schema, consumed = json.JSONDecoder().raw_decode(tail.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(schema, dict):
            system = system[:start] + canonical(compact_schema(schema)) + tail[leading + consumed:]
    return system


def render_role_system(role: str, system: str, *, instructions: Any = None, schema: Any = None, execution_version: str | None = None, planning_mode: str | None = None, compact_engineering: bool = False, compact_analysis: bool = False) -> str:
    """Freeze policy/schema text before native task and cache identity binding."""
    policy = role_policy(role, execution_version=execution_version, planning_mode=planning_mode)
    system = _compact_system_schema(system)
    if instructions is not None:
        system += "\nCURRENT_INSTRUCTIONS\n" + canonical(instructions)
    if schema is not None:
        system += "\nOUTPUT_SCHEMA\n" + canonical(compact_schema(schema))
    independent_planning = (role in {"research_planner", "research_critic"}
                            and execution_version == "single_plan_v1"
                            and planning_mode == "propose_then_critic")
    if role == "experiment_designer" or compact_engineering or compact_analysis:
        # The explicit Designer view contains scientific constraints, not a
        # ledger/delivery protocol or tool for following context read handles.
        return system
    if not independent_planning and "\nINPUT DELIVERY CONTRACT\n" not in "\n" + system:
        system += "\n" + CORE_CONTRACT + "ROLE_MISSION\n" + policy.mission
    return system


def _sections(payload: dict[str, Any]) -> list[tuple[tuple[Any, ...], Any]]:
    result = []
    def visit(value: dict[str, Any], path: tuple[str, ...]):
        for key, child in value.items():
            if key in WRAPPERS and isinstance(child, dict):
                visit(child, path + (key,))
            elif key in CLOSED_COLLECTIONS and isinstance(child, list) and child:
                result.extend((path + (key, index), row) for index, row in enumerate(child))
            else:
                result.append((path + (key,), child))
    visit(payload, ())
    return result


def _put(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    current: Any = payload
    for index, key in enumerate(path):
        final = index == len(path) - 1
        if isinstance(current, list):
            while len(current) <= int(key):
                current.append(None)
            if final:
                current[int(key)] = value
            else:
                if current[int(key)] is None:
                    current[int(key)] = [] if isinstance(path[index + 1], int) else {}
                current = current[int(key)]
        elif final:
            current[key] = value
        else:
            current = current.setdefault(key, [] if isinstance(path[index + 1], int) else {})


def _active_candidate_ids(request: CompileRequest) -> set[str]:
    active = {str(request.identity[key]) for key in ("candidate_id", "target_id") if request.identity.get(key)}
    for path, value in _sections(request.payload):
        if path[-1] in {"candidate_id", "implementation_target_id"} and isinstance(value, str):
            active.add(value)
        if path[-1] in {"active_tasks", "pending_tasks"}:
            rows = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
            for row in rows:
                if isinstance(row, dict):
                    active.update(str(row[key]) for key in ("candidate_id", "target_id") if row.get(key))
        if path[-1] == "trainable_ids" and isinstance(value, list) and request.role == "research_planner":
            active.update(item for item in value if isinstance(item, str))
        if path[-1] == "eligible_targets" and isinstance(value, dict) and request.role == "research_planner":
            for targets in value.values():
                if isinstance(targets, list):
                    active.update(item for item in targets if isinstance(item, str))
        if path[-1] in {"frontier", "legal_choice_context"} and isinstance(value, dict):
            for field in ("candidate_ids", "target_ids", "selected_target_ids"):
                if isinstance(value.get(field), list):
                    active.update(item for item in value[field] if isinstance(item, str))
    return active


def _required_evidence_ids(request: CompileRequest) -> set[str]:
    """Read explicit native task dependencies, never collect IDs from history."""
    result: set[str] = set()
    dependency_keys = {"input_evidence_ids", "evidence_ids", "evidence_refs", "required_evidence_ids",
                       "required_artifact_refs", "input_artifact_refs", "selected_evidence_ids"}
    def collect_refs(value):
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, list):
            for child in value:
                collect_refs(child)
        elif isinstance(value, dict):
            for key in ("evidence_id", "artifact_id", "suite_id", "job_id", "run_id"):
                if isinstance(value.get(key), str):
                    result.add(value[key])
    def collect(container, depth=0):
        if not isinstance(container, dict) or depth > 4:
            return
        for key, value in container.items():
            if key in dependency_keys:
                collect_refs(value)
            elif key in {"task", "current_task", "intent", "repair_task", "frontier", "legal_choice_context", "dependencies"}:
                collect(value, depth + 1)
    collect(request.identity)
    collect(request.payload)
    for key in WRAPPERS:
        collect(request.payload.get(key))
    return result


def _scientific_frontier(block: SourceBlock, original: Any, active: set[str]) -> Representation | None:
    if block.kind != "scientific" or not isinstance(original, list) or not active or len(original) < 8:
        return None
    selected, indices = [], []
    for index, row in enumerate(original):
        if not isinstance(row, dict):
            selected.append(row); indices.append(index); continue
        candidate = row.get("candidate_id")
        status = str(row.get("status", row.get("completion_status", "")))
        # Unclassified identity or incomplete/conflicting operations remain live.
        keep = candidate is None or str(candidate) in active or str(candidate) == "baseline" or status in {
            "pending", "running", "active", "failed", "partial", "unknown", "unverified", "conflict"}
        if keep:
            selected.append(copy.deepcopy(row)); indices.append(index)
    if len(selected) == len(original):
        return None
    return Representation(digest({"source": block.source_hash, "kind": "PARTIAL", "indices": indices}),
        block.block_id, "PARTIAL", selected, (block.source_ref,), (block.source_hash,), digest(selected),
        selected_ranges=tuple({"unit": "items", "start": index, "end": index + 1} for index in indices),
        omitted_sections=("inactive completed candidate records; full registered source retained",),
        completeness="current_active_frontier", applicability="native_current_candidate_frontier",
        budget_chars=len(canonical(selected)))


def _bounded_optional(block: SourceBlock, original: Any, cap: int) -> Representation | None:
    if len(canonical(original)) <= cap:
        return None
    from .representations import handle
    if isinstance(original, list):
        rows, prefix, suffix = original, None, None
    elif isinstance(original, dict) and isinstance(original.get("items"), list):
        rows = original["items"]
        prefix = {key: value for key, value in original.items() if key != "items"}
        suffix = "items"
    else:
        content = handle(block)
        return Representation(digest({"source": block.source_hash, "kind": "REF_ONLY"}), block.block_id,
            "REF_ONLY", content, (block.source_ref,), (block.source_hash,), digest(content),
            completeness="bounded_optional_reference", budget_chars=len(canonical(content)))
    selected, indices = [], []
    for index, row in enumerate(rows):
        next_rows = selected + [row]
        candidate = {**prefix, suffix: next_rows} if prefix is not None else next_rows
        if len(canonical(candidate)) > cap - 500:
            continue
        selected.append(copy.deepcopy(row)); indices.append(index)
    content = {**prefix, suffix: selected} if prefix is not None else selected
    if len(canonical(content)) > cap:
        content = handle(block)
    return Representation(digest({"source": block.source_hash, "kind": "PARTIAL", "indices": indices}),
        block.block_id, "PARTIAL", content, (block.source_ref,), (block.source_hash,), digest(content),
        selected_ranges=tuple({"unit": "items", "start": index, "end": index + 1} for index in indices),
        omitted_sections=("optional delivery cap; complete registered source retained",),
        completeness="bounded_optional_selection", budget_chars=len(canonical(content)))


def _archive_card(block: SourceBlock, row: dict[str, Any]) -> Representation:
    from .representations import handle
    content = {key: row[key] for key in ("evidence_id", "artifact_id", "suite_id", "job_id", "candidate_id", "status",
        "completion_status", "seed", "method_revision", "protocol_hash", "evaluation_key", "evaluator_hash", "evaluator",
        "scope", "coverage", "evaluation_valid", "question_id", "task_id") if key in row}
    content["context_source"] = handle(block)
    content["body_delivery"] = "REF_ONLY"
    content["task_applicability"] = "inactive_for_current_native_frontier"
    content["omission_is_not_completion"] = True
    for key, value in row.items():
        if any(word in key.lower() for word in ("error", "fail", "conflict", "unknown", "missing")):
            content[key] = copy.deepcopy(value) if len(canonical(value)) <= 1_000 else handle(block, "/" + pointer_escape(key))
    return Representation(digest({"source": block.source_hash, "kind": "REF_ONLY", "content": content}),
        block.block_id, "REF_ONLY", content, (block.source_ref,), (block.source_hash,), digest(content),
        completeness="inactive_source_navigation", applicability="inactive_for_current_native_frontier",
        budget_chars=len(canonical(content)))


class ContextCompiler:
    def __init__(self, camp: Path | str):
        self.camp = Path(camp).resolve()
        self.registry: ContextSourceRegistry | None = None
        self.snapshot_store = None

    def compile(self, request: CompileRequest) -> CompileResult:
        if not isinstance(request.payload, dict):
            return Blocked("invalid_payload", {"expected": "object"})
        try:
            from ..research_execution import single_plan_enabled
            policy = role_policy(request.role, execution_version="single_plan_v1" if single_plan_enabled(self.camp) else None)
        except KeyError:
            return Blocked("policy_missing", {"role": request.role})
        # Compiler cannot enter its own paid maintenance from a maintenance call.
        maintenance = request.mode in MAINTENANCE_MODES or _IN_MAINTENANCE.get()
        scientific_planning = is_research_planning_input(request.role, request.payload)
        scientific_designer = is_designer_scientific_input(request.role, request.payload)
        scientific_analysis = is_analysis_scientific_input(request.role, request.payload)
        exact_science = scientific_planning or scientific_designer or scientific_analysis or is_engineering_input(request.role, request.payload)
        self.registry = ContextSourceRegistry(self.camp, request.campaign_id)
        self.snapshot_store = None
        # The staged builder already selects the required scientific context.
        # Working-memory control records remain local to the source manifest.
        snapshot_delivery = None if exact_science else self._snapshot_delivery(request)
        active_candidates = _active_candidate_ids(request)
        required_evidence = _required_evidence_ids(request)
        merged_input = (project_analysis_scientific_input(request.payload) if scientific_analysis
                        else copy.deepcopy(request.payload))
        # Memory diagnostics belong to the local selection manifest, including
        # when a native controller uses the compact view before role projection.
        for section in (merged_input, merged_input.get("observation")):
            if isinstance(section, dict):
                for key in ("memory_retrieval_health", "memory_delivery_report", "memory_delivery_report_ref"):
                    section.pop(key, None)
        if request.examples and not scientific_analysis:
            merged_input["current_examples"] = copy.deepcopy(list(request.examples))
        if request.feedback is not None and not scientific_analysis:
            merged_input["current_feedback"] = copy.deepcopy(request.feedback)
        if request.retrieved and not scientific_analysis:
            merged_input["retrieved_context"] = copy.deepcopy(list(request.retrieved))
        if snapshot_delivery:
            merged_input["working_snapshot"] = snapshot_delivery
        source_blocks: list[SourceBlock] = []
        entries: list[dict[str, Any]] = []
        try:
            # Filter before any body hashing or persistence, including a sensitive
            # field whose producer forgot to attach a scope/visibility attribute.
            assert_visible_payload(merged_input, request.role)
            if scientific_analysis:
                # Full native task/source binding is immutable local evidence,
                # not a second copy of training facts in the model request.
                native_payload = {key: value for key, value in request.payload.items()
                                  if key not in {"previous_response_error", "previous_response"}}
                native = self.registry.capture(native_payload,
                    block_id="analyst-native:" + digest(native_payload), kind="analysis_native_input",
                    role=request.role, identity=dict(request.identity),
                    metadata={"delivery_scope": "local_only_native_analysis"})
                source_blocks.append(native)
            for supplied in request.source_blocks:
                if supplied.campaign_id != request.campaign_id:
                    return Blocked("cross_campaign_source", {"block_id": supplied.block_id})
                registered = self.registry.get(supplied.source_ref, role=request.role)
                if registered.source_hash != supplied.source_hash:
                    return Blocked("stale_source", {"block_id": supplied.block_id})
            from .reads import complete_field_reads
            read_paths, read_proofs = complete_field_reads(self.registry, request.source_blocks, role=request.role)
            source_blocks.extend(read_proofs)
            # Capture complete producer output before producing any model view.
            for path, payload in _sections(merged_input):
                row = isinstance(path[-1], int) and path[-2] in CLOSED_COLLECTIONS and isinstance(payload, dict)
                key = path[-2] if row else path[-1]
                pointer = "/" + "/".join(pointer_escape(str(part)) for part in path)
                block_id = row_source_id(key, payload) if row else "input:" + pointer
                partial_dependencies = []
                def partial_sources(value, depth=0):
                    if not isinstance(value, dict) or depth > 4:
                        return
                    if value.get("fidelity") == "PARTIAL" and "actual_read_scope" in value and isinstance(value.get("source_ref"), str):
                        original = self.registry.get(value["source_ref"], role=request.role)
                        if original.source_hash != value.get("source_hash"):
                            raise ValueError("partial_read_source_hash_mismatch")
                        partial_dependencies.append(original.source_ref)
                    for child in value.values():
                        if isinstance(child, dict):
                            partial_sources(child, depth + 1)
                partial_sources(payload)
                if partial_dependencies:
                    # A read view is a derived delivery, not a new producer
                    # revision that makes its original source immediately stale.
                    block_id = "readview:" + block_id + ":" + digest(payload)
                native_identity = native_row_identity(payload) if row else dict(request.identity)
                if row:
                    native_identity["operation_id"] = block_id
                block = self.registry.capture(payload, block_id=block_id,
                    kind=source_kind(key, payload), role=request.role,
                    metadata={"payload_pointer": pointer, "top_key": key},
                    identity=native_identity, revision=digest(payload),
                    group_id=block_id if row else "", status=str(payload.get("status", "complete")) if row else "captured",
                    dependency_refs=tuple(partial_dependencies))
                source_blocks.append(block)
                abstracted = None
                if self.snapshot_store is not None and snapshot_delivery:
                    abstracted = lambda section, b=block: self.snapshot_store.is_abstracted(b.block_id, b.source_hash, section)
                choices, unknown = representations_for(block, payload, role=request.role,
                                                        excerpt_cap=request.budget.excerpt_chars, abstracted=abstracted,
                                                        complete_read_paths=frozenset(read_paths.get((block.source_ref, block.source_hash), ())))
                if exact_science:
                    # Do not change selected science after the native input
                    # is frozen. Oversized essential input must
                    # fail its ordinary budget check, not become opaque handles.
                    choices = [next(choice for choice in choices if choice.kind == "FULL")]
                    unknown = []
                frontier = None if exact_science else _scientific_frontier(block, payload, active_candidates)
                if frontier is not None:
                    # Native active identities, not recency/FIFO, select this set.
                    choices = [frontier]
                    # Selected rows are literal FULL records; their unfamiliar
                    # fields are delivered, not removed by a partial adapter.
                    unknown = []
                cap = (request.budget.retrieval_chars if key in {"retrieved_memory", "retrieved_knowledge", "retrieved_context", "memory", "lessons", "skills"}
                       else request.budget.snapshot_max_chars if key == "working_snapshot"
                       else request.budget.recent_chars if key == "recent_decisions" else None)
                bounded = _bounded_optional(block, payload, cap) if cap and not exact_science else None
                if bounded is not None:
                    choices, unknown = [bounded], []
                entries.append({"path": path, "key": key, "block": block, "choices": choices,
                                "choice": 0, "unknown": unknown, "row": row, "raw": payload,
                                "required": key in policy.required_present_fields or key not in (OPTIONAL_FIELDS | {"working_snapshot", "retrieved_context"})})
        except PermissionError as exc:
            return Blocked("unreadable_source", {"reason": str(exc)}, recoverable_when=("authorized visible source available",))
        except (OSError, ValueError, KeyError) as exc:
            return Blocked("source_capture_failed", {"reason": str(exc)})

        # Role obligations concern the current native problem. Complete producer
        # bodies remain captured; inactive failures and pending operations become
        # bounded navigation, never scientific negatives or claimed completion.
        archives: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        current_identity = {key: request.identity[key] for key in ("seed", "protocol_hash", "evaluation_key", "evaluator_hash") if key in request.identity}
        for entry in entries:
            if entry["row"] and entry["raw"].get("candidate_id") in active_candidates:
                current_identity.update({key: entry["raw"][key] for key in ("seed", "protocol_hash", "evaluation_key", "evaluator_hash") if key in entry["raw"]})
        for entry in entries:
            if exact_science or not entry["row"] or not active_candidates:
                continue
            raw = entry["raw"]; candidate = raw.get("candidate_id")
            row_refs = {str(raw[key]) for key in ("evidence_id", "artifact_id", "suite_id", "job_id", "run_id") if raw.get(key)}
            for ref in raw.get("artifact_refs", ()) if isinstance(raw.get("artifact_refs"), list) else ():
                if isinstance(ref, str):
                    row_refs.add(ref)
                elif isinstance(ref, dict) and isinstance(ref.get("artifact_id"), str):
                    row_refs.add(ref["artifact_id"])
            if row_refs.intersection(required_evidence):
                entry["required"] = True
                continue
            if candidate is None or str(candidate) in active_candidates:
                continue
            compatible_baseline = str(candidate) == "baseline" and not any(
                key in raw and key in current_identity and canonical(raw[key]) != canonical(current_identity[key])
                for key in ("seed", "protocol_hash", "evaluator_hash"))
            if compatible_baseline:
                continue
            card = _archive_card(entry["block"], raw)
            entry.update(choices=[card], choice=0, unknown=[], required=False)
            archives.setdefault(entry["path"][:-1], []).append(entry)
        archived_manifest = []
        for collection, rows in archives.items():
            # Full source IDs/status/version navigation is pageable in one real
            # immutable source; it never grows linearly inside the model request.
            navigation = [entry["choices"][0].content for entry in rows]
            block = self.registry.capture(navigation, block_id="navigation:/" + "/".join(str(x) for x in collection),
                kind="navigation", role=request.role, metadata={"body_delivery": "REF_ONLY", "omission_is_not_completion": True})
            source_blocks.append(block)
            ranked = sorted(rows, key=lambda entry: (0 if str(entry["raw"].get("status")) in {"pending", "running", "active", "failed", "partial", "unknown", "unverified"} else 1,
                                                      entry["block"].block_id))
            selected = {id(entry) for entry in ranked[:8]}
            for entry in rows:
                entry["omit_delivery"] = id(entry) not in selected
            archived_manifest.append({"field": "/".join(str(x) for x in collection), "source_ref": block.source_ref,
                "source_hash": block.source_hash, "total_count": len(rows), "delivered_count": len(selected),
                "body_delivery": "REF_ONLY", "omission_is_not_completion": True})

        system = render_role_system(request.role, request.system, instructions=request.instructions, schema=request.schema,
            execution_version="single_plan_v1" if scientific_planning else None,
            planning_mode="propose_then_critic" if scientific_planning else None,
            compact_engineering=is_engineering_input(request.role, request.payload),
            compact_analysis=request.role == "result_analyst" and request.payload.get("review_stage") == "analysis")
        # These are assembled before measuring. Sending code must reuse them.
        transport_fields: dict[str, Any] = copy.deepcopy(request.transport_options)
        transport_fields.setdefault("model", request.model)
        transport_fields.setdefault("max_tokens", request.max_output_tokens)
        if transport_fields.get("model") != request.model or transport_fields.get("max_tokens") != request.max_output_tokens:
            return Blocked("transport_policy_mismatch", {"reason": "rendered model/output options differ from budget policy"})
        if request.tools:
            transport_fields["tools"] = [compact_schema(tool) for tool in request.tools]
        base_extras: dict[str, Any] = {}

        dependency_rows = [{"block_id": block.block_id, "source_ref": block.source_ref,
                            "source_hash": block.source_hash, "revision": block.source_revision,
                            "visibility": list(block.visibility)} for block in source_blocks]
        dependency_digest = digest(dependency_rows)

        def render() -> tuple[dict[str, Any], str, dict[str, Any]]:
            payload: dict[str, Any] = {}
            for entry in entries:
                if entry.get("omit_delivery"):
                    continue
                rep = entry["choices"][entry["choice"]]
                path = entry["path"]
                if entry["row"]:
                    # Array position is presentation, never source identity.
                    parent_path = path[:-1]
                    parent: Any = payload
                    for part in parent_path:
                        parent = parent.setdefault(part, [] if part == parent_path[-1] else {})
                    parent.append(copy.deepcopy(rep.content))
                else:
                    _put(payload, path, copy.deepcopy(rep.content))
            payload.update(copy.deepcopy(base_extras))
            if not exact_science:
                payload = deduplicate(payload)
            # Manifest growth is confined to a sidecar, never replayed wholesale.
            delivery = {
                "protocol": "eeg.context.v1", "source_count": len(source_blocks),
                "source_dependency_digest": dependency_digest, "policy_version": request.policy_version,
                "source_registry": "campaign_delivery_only", "native_read_receipts_satisfied": False,
                "representation_counts": {kind: sum(entry["choices"][entry["choice"]].kind == kind for entry in entries if not entry.get("omit_delivery"))
                                          for kind in ("FULL", "FACT_CARD", "PARTIAL", "SUMMARY", "REF_ONLY")},
                "active_frontier_only": bool(archived_manifest) or any(entry["choices"][entry["choice"]].completeness == "current_active_frontier" for entry in entries),
                "archived_navigation": archived_manifest[:16],
                "navigation": [{"field": "/".join(str(part) for part in entry["path"]), "source_ref": entry["block"].source_ref,
                                "source_hash": entry["block"].source_hash, "fidelity": entry["choices"][entry["choice"]].kind}
                               for entry in entries if not entry.get("omit_delivery") and not entry["row"] and entry["choices"][entry["choice"]].kind in {"PARTIAL", "REF_ONLY"}][:min(request.budget.max_manifest_entries, 16)],
            }
            if not exact_science:
                payload["context_delivery"] = delivery
            user = canonical(payload)
            return payload, user, measure(system, user, request, transport_fields)

        dependency_rows = [{"block_id": block.block_id, "source_ref": block.source_ref,
                            "source_hash": block.source_hash, "revision": block.source_revision,
                            "visibility": list(block.visibility)} for block in source_blocks]
        dependency_digest = digest(dependency_rows)
        payload, user, report = render()
        before_report = dict(report)
        downgrades = 0
        max_downgrades = sum(len(entry["choices"]) - 1 for entry in entries)
        # Each move consumes a finite representation choice. There is no upgrade
        # or replay of FULL during a compile, so termination is structural.
        while (report["chars"] > request.budget.target_chars or not report["provider_budget_ok"]):
            candidates = []
            for index, entry in enumerate(entries):
                current = entry["choice"]
                next_index = current + 1
                if next_index >= len(entry["choices"]):
                    continue
                candidate = entry["choices"][next_index]
                if candidate.kind == "REF_ONLY" and entry["required"]:
                    continue
                if candidate.kind == "REF_ONLY" and entry["unknown"]:
                    continue
                saving = entry["choices"][current].budget_chars - candidate.budget_chars
                candidates.append((-saving, "/".join(str(part) for part in entry["path"]), index))
            if not candidates or downgrades >= max_downgrades:
                break
            # Nominal savings can be hidden by exact dedup. Only keep a move if
            # the actual rendered request cost decreases; never claim fake gain.
            moved = False
            for _, _, index in sorted(candidates):
                entries[index]["choice"] += 1
                next_payload, next_user, next_report = render()
                if next_report["chars"] < report["chars"] and next_report["safe_input_tokens"] <= report["safe_input_tokens"]:
                    payload, user, report = next_payload, next_user, next_report
                    downgrades += 1
                    moved = True
                    break
                entries[index]["choice"] -= 1
            if not moved:
                # A shared giant string may occur in several producer blocks.
                # Replacing one copy alone can expose another raw copy, hiding
                # the saving behind dedup. Downgrade the connected group once;
                # only its actual final rendered reduction is accepted.
                def bulky_hashes(value):
                    if isinstance(value, str):
                        return {digest(value)} if len(value) > 1200 else set()
                    if isinstance(value, dict):
                        return set().union(*(bulky_hashes(child) for child in value.values())) if value else set()
                    if isinstance(value, list):
                        return set().union(*(bulky_hashes(child) for child in value)) if value else set()
                    return set()
                eligible = [index for _, _, index in sorted(candidates)]
                shared = {index: bulky_hashes(entries[index]["choices"][entries[index]["choice"]].content) for index in eligible}
                tried = set()
                for first in eligible:
                    group = {first}
                    hashes = set(shared[first])
                    for _ in range(len(eligible)):
                        added = {index for index in eligible if index not in group and hashes.intersection(shared[index])}
                        if not added:
                            break
                        group.update(added)
                        hashes.update(set().union(*(shared[index] for index in added)))
                    key = tuple(sorted(group))
                    if len(group) < 2 or key in tried:
                        continue
                    tried.add(key)
                    for index in key:
                        entries[index]["choice"] += 1
                    next_payload, next_user, next_report = render()
                    if next_report["chars"] < report["chars"] and next_report["safe_input_tokens"] <= report["safe_input_tokens"]:
                        payload, user, report = next_payload, next_user, next_report
                        downgrades += len(key)
                        moved = True
                        break
                    for index in key:
                        entries[index]["choice"] -= 1
                if not moved:
                    break

        # Optional retrieved material has its own cap within the common total.
        for key, cap in (("retrieved_context", request.budget.retrieval_chars),
                         ("working_snapshot", request.budget.snapshot_max_chars)):
            if key in base_extras and len(canonical(base_extras[key])) > cap:
                original = base_extras[key]
                block = self.registry.capture(original, block_id="extra:" + key, kind="memory", role=request.role)
                source_blocks.append(block)
                if isinstance(original, list):
                    selected = []
                    for item in original:
                        if len(canonical(selected + [item])) > cap - 600:
                            continue
                        selected.append(item)
                    base_extras[key] = {"items": selected, "total_count": len(original), "source": block.source_ref,
                                        "source_hash": block.source_hash, "fidelity": "PARTIAL"}
                else:
                    base_extras[key] = {"source": block.source_ref, "source_hash": block.source_hash,
                                        "fidelity": "REF_ONLY", "reason": "bounded_delivery_cap"}
        dependency_rows = [{"block_id": block.block_id, "source_ref": block.source_ref,
                            "source_hash": block.source_hash, "revision": block.source_revision,
                            "visibility": list(block.visibility)} for block in source_blocks]
        dependency_digest = digest(dependency_rows)
        payload, user, report = render()

        maintenance_outcome = "not_requested"
        if (before_report["chars"] >= request.budget.high_chars or not before_report["provider_budget_ok"]) and not maintenance and not exact_science:
            if request.allow_semantic_compaction and request.allow_sync_memory_maintenance and request.maintenance_callback and request.remaining_api_calls != 0:
                token = _IN_MAINTENANCE.set(True)
                try:
                    result = request.maintenance_callback(request=request, blocks=tuple(source_blocks),
                        effective_payload=payload, budget_report=report)
                    maintenance_outcome = str(result.get("status", "invalid")) if isinstance(result, dict) else "invalid"
                    if (isinstance(result, dict) and result.get("status") == "published" and result.get("validated") is True
                            and result.get("source_dependency_digest") == dependency_digest
                            and len(canonical(result.get("delivery", {}))) <= request.budget.snapshot_max_chars):
                        base_extras["working_snapshot"] = result["delivery"]
                        # Publication may replace only sections whose native
                        # coverage CAS succeeded. It never trusts model covered IDs.
                        if self.snapshot_store is not None:
                            for entry in entries:
                                if entry["choices"][entry["choice"]].completeness == "current_active_frontier":
                                    continue
                                block = entry["block"]
                                original = self.registry.load(block, role=request.role)
                                old_kind = entry["choices"][entry["choice"]].kind
                                new_choices, new_unknown = representations_for(block, original, role=request.role,
                                    excerpt_cap=request.budget.excerpt_chars,
                                    abstracted=lambda section, b=block: self.snapshot_store.is_abstracted(b.block_id, b.source_hash, section),
                                    complete_read_paths=frozenset(read_paths.get((block.source_ref, block.source_hash), ())))
                                entry["choices"] = new_choices
                                entry["choice"] = next((index for index, rep in enumerate(new_choices) if rep.kind == old_kind), 0)
                                entry["unknown"] = new_unknown
                        payload, user, report = render()
                except Exception as exc:
                    maintenance_outcome = "failed:" + type(exc).__name__
                finally:
                    _IN_MAINTENANCE.reset(token)

        missing = []
        for entry in entries:
            chosen = entry["choices"][entry["choice"]]
            if chosen.kind != "FULL":
                for unknown in entry["unknown"]:
                    missing.append({**unknown, "payload_pointer": "/" + "/".join(pointer_escape(str(part)) for part in entry["path"])})
        valid, pointer_errors = validate_pointer_closure(payload)
        if not valid:
            return Blocked("invalid_reference_closure", {"errors": pointer_errors}, report)
        report.update({"before_packing_chars": before_report["chars"], "packing_downgrades": downgrades,
                       "maintenance_outcome": maintenance_outcome,
                       "target_reached": report["chars"] <= request.budget.target_chars,
                       "fallback_reason": "required_exact_material" if report["chars"] > request.budget.target_chars else None,
                       "maintenance_reentry_guard": maintenance})
        if missing:
            reads = self._recovery_reads(request, missing)
            return NeedsContext(tuple(missing), reads, "unclassified_or_unread_required_field", report,
                                {"max_reads": request.budget.max_reads, "max_read_chars": request.budget.max_read_chars})
        if not report["ready_budget_ok"]:
            if exact_science:
                return Blocked("engineering_input_exceeds_budget" if is_engineering_input(request.role, request.payload)
                    else "designer_input_exceeds_budget" if scientific_designer
                    else "analysis_input_exceeds_budget" if scientific_analysis else "research_planning_input_exceeds_budget",
                    {"role": request.role, "minimum_necessary_chars": report["chars"]}, report,
                    ("a smaller native scientific projection or larger authorized provider window",))
            kernel_report = measure(system, canonical({}), request, transport_fields)
            if not kernel_report["ready_budget_ok"]:
                return Blocked("protected_input_too_large", {"affected_task": request.task_id,
                    "minimum_necessary_chars": kernel_report["chars"]}, report,
                    ("decompose instructions/schema into a bounded legal frontier", "larger authorized provider window"))
            required = [entry for entry in entries if entry["required"]]
            reads = tuple(ReadRequest(entry["block"].source_ref, entry["block"].source_hash,
                        max_chars=request.budget.excerpt_chars, question="Decompose current obligation " + entry["key"])
                          for entry in sorted(required, key=lambda entry: (-entry["choices"][entry["choice"]].budget_chars, entry["key"]))[:request.budget.max_reads])
            return NeedsContext(({"status": "required_kernel_exceeds_budget", "chars": report["chars"],
                                  "provider_budget_ok": report["provider_budget_ok"]},), reads,
                                "required_input_too_large", report,
                                {"max_reads": request.budget.max_reads, "max_read_chars": request.budget.max_read_chars})

        # A background update, rewind or revocation during packing/maintenance
        # invalidates the frozen dependency set. Do not publish stale Ready.
        try:
            for block in source_blocks:
                self.registry.load(block, role=request.role)
        except (OSError, ValueError, KeyError, PermissionError) as exc:
            return Blocked("stale_or_unreadable_dependency", {"reason": str(exc)}, report,
                           ("recompile from the current authorized source revisions",))

        manifest = {"version": "eeg.context.delivery.v1", "campaign_id": request.campaign_id,
            "role": request.role, "task_id": request.task_id, "intent_id": request.intent_id,
            "authority_revision": request.authority_revision,
            "source_dependency_digest": dependency_digest,
            "sources": [asdict(block) for block in source_blocks],
            "representations": [asdict(entry["choices"][entry["choice"]]) | {"payload_path": list(entry["path"]), "actually_delivered": not entry.get("omit_delivery", False)} for entry in entries],
            "read_scope": "delivery_only", "native_read_receipts_satisfied": False,
            "unknown_fields": missing, "requirements": {"present_native_fields_preserved": True,
                "source_hashes_valid": True, "request_pointers_closed": True, "full_source_is_fact_source": True}}
        request_digest = digest({"system": system, "user": user, "transport_fields": transport_fields})
        # Same frozen transport can have a different no-op/cache/maintenance
        # report. Version sidecars independently without altering transport ID.
        sidecar = self.camp / "context_deliveries" / request_digest / digest({"budget": report, "manifest": manifest})
        atomic_immutable(sidecar.with_suffix(".budget.json"), canonical(report))
        atomic_immutable(sidecar.with_suffix(".manifest.json"), canonical(manifest))
        prepared = PreparedRequest(system, user, ({"role": "system", "content": system},
            {"role": "user", "content": user}), request_digest, dependency_digest,
            request.policy_version, RENDER_VERSION, report["tokenizer_version"], report, manifest,
            tuple(dependency_rows), {"budget": str(sidecar.with_suffix(".budget.json")),
                                     "manifest": str(sidecar.with_suffix(".manifest.json"))}, transport_fields)
        return Ready(prepared, manifest["requirements"])

    def _recovery_reads(self, request: CompileRequest, missing: list[dict[str, Any]]) -> tuple[ReadRequest, ...]:
        from .recovery import declared_function_targets
        from .representations import code_outline
        from .sources import pointer_get
        targets = declared_function_targets(request)
        reads = []
        for item in missing:
            if item.get("type") == "code" and targets:
                source = self.registry.get(item["source_ref"], role=request.role)
                raw = pointer_get(self.registry.load(source, role=request.role), item["path"])
                outline = code_outline(raw)
                prior_count = len(reads)
                for target in sorted(targets):
                    matching = [row for row in outline if row["name"] == target]
                    if len(matching) == 1:
                        row = matching[0]
                        reads.append(ReadRequest(item["source_ref"], item["source_hash"], item["path"],
                            row["line_start"], row["line_end"], "lines", request.budget.excerpt_chars,
                            "Deliver native task function " + target))
                if len(reads) > prior_count:
                    continue
            reads.append(ReadRequest(item["source_ref"], item["source_hash"], item["path"], 0,
                min(item.get("line_count", 100), 100) if item.get("type") in {"code", "log"} else min(item.get("chars", request.budget.excerpt_chars), request.budget.excerpt_chars - 100),
                "lines" if item.get("type") in {"code", "log"} else "chars", request.budget.excerpt_chars,
                "Classify and deliver required field " + item["path"]))
        return tuple(reads[:request.budget.max_reads])

    def _snapshot_delivery(self, request: CompileRequest) -> dict[str, Any] | None:
        if request.snapshot is not None:
            if hasattr(request.snapshot, "snapshot_delivery"):
                self.snapshot_store = request.snapshot
                return request.snapshot.snapshot_delivery(max_chars=request.budget.snapshot_target_chars,
                                                           hard_max_chars=request.budget.snapshot_max_chars)
            return copy.deepcopy(request.snapshot)
        # Bootstrapping old campaigns is optional; absence never means FULL bypass.
        try:
            from .snapshot import SnapshotStore
            store = SnapshotStore(self.camp, source_registry=self.registry)
            self.snapshot_store = store
            return store.snapshot_delivery(max_chars=request.budget.snapshot_target_chars,
                                           hard_max_chars=request.budget.snapshot_max_chars)
        except (ImportError, OSError, ValueError):
            return None
