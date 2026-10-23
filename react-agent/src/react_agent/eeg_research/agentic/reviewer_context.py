"""Literal Implementation Reviewer input and its native-only evidence binding.

An alias is navigation, not evidence.  Native identities are captured before
projection; citation permissions are bound only after the actual SDK payload
has been compiled.  No model call, source summarization, or artifact write is
performed here.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .approved_design_context import project_approved_design
from .context.representations import without_local_paths
from .context.types import canonical, digest
from .designer_context import _expanded

VERSION = "reviewer_implementation_input.v2"
CONTRACT_VERSION = "eeg_research.reviewer_contract.v2"
SCIENCE_FIELDS = ("approved_design", "runtime_contract", "review_targets", "source_materials",
                  "check_facts", "parameter_facts", "review_scope", "relevant_context")
_CORRECTION_FIELDS = {"previous_response", "previous_response_error"}


def is_reviewer_input(role: str, payload: dict[str, Any]) -> bool:
    return role == "candidate_reviewer" and payload.get("reviewer_context_version") == VERSION


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _symbols(source: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result
    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                result[name] = [first, node.end_lineno or node.lineno]
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")
            elif not prefix or prefix.endswith("."):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                for target in targets:
                    if isinstance(target, ast.Name):
                        result[prefix + target.id] = [node.lineno, node.end_lineno or node.lineno]
    visit(tree.body)
    return result


def _segments(source: str, supplied: str, *, excerpt: bool) -> list[dict]:
    """Locate literal delivered pieces; never grant omitted intervening lines."""
    parts = supplied.split("\n\n") if excerpt else [supplied]
    cursor, result = 0, []
    lines = source.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    for part in parts:
        if not part.strip():
            continue
        left = source.find(part, cursor)
        if left < 0:
            raise ValueError("review_source_delivery_not_literal")
        right = left + len(part)
        start_line = source.count("\n", 0, left) + 1
        end_line = source.count("\n", 0, right - 1) + 1
        if result and left == result[-1]["end_offset"] + 2 and source[result[-1]["end_offset"]:left] == "\n\n":
            previous = result.pop()
            left, start_line = previous["start_offset"], previous["start_line"]
            part = source[left:right]
        complete = [index + 1 for index, (start, line) in enumerate(zip(offsets, lines))
                    if left <= start and start + len(line.rstrip("\r\n")) <= right]
        result.append({"start_line": start_line, "end_line": end_line, "text": part,
            "start_offset": left, "end_offset": right,
            "complete_line_ranges": [[complete[0], complete[-1]]] if complete else []})
        cursor = right
    if not result:
        raise ValueError("review_source_delivery_empty")
    return result


def _freeze_source(alias: str, supplied: dict, *, fallback_path: str = "",
                   expected_hash: str = "") -> tuple[dict, dict]:
    text = supplied.get("source") if "source" in supplied else supplied.get("excerpt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("review_source_missing:" + alias)
    native_path = str(supplied.get("source_ref") or fallback_path)
    expected_hash = str(supplied.get("source_hash") or expected_hash or "")
    path = Path(native_path) if native_path else None
    if path is not None and path.is_file():
        original = path.read_bytes()
        actual_hash = hashlib.sha256(original).hexdigest()
        source = original.decode("utf-8")
    elif native_path or "excerpt" in supplied or supplied.get("source_truncated"):
        raise ValueError("review_source_original_unavailable:" + alias)
    else:
        source, actual_hash = text, _sha(text)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("review_source_hash_changed:" + alias)
    if supplied.get("source_truncated") is False and "source" in supplied and text != source:
        raise ValueError("review_source_full_delivery_incomplete:" + alias)
    segments = _segments(source, text, excerpt="excerpt" in supplied)
    ranges = [pair for item in segments for pair in item["complete_line_ranges"]]
    full = text == source
    wire = {"source_ref": alias, "file": Path(native_path).name if native_path else alias,
        "delivery": "full" if full else "selected_source",
        "segments": [{key: item[key] for key in ("start_line", "end_line", "text")} for item in segments]}
    native = {"kind": "source", "native_path": native_path, "sha256": actual_hash,
        "source": text, "full_source": source, "segments": segments,
        "symbol_spans": _symbols(source), "delivered_ranges": ranges, "full": full}
    # Navigation uses the same literal delivery boundary as the validator. A
    # visible function signature does not make its omitted body available.
    from .reviewer_contract import resolve_evidence_reference
    wire["symbol_refs"] = [alias + "::" + symbol for symbol in native["symbol_spans"]
        if resolve_evidence_reference(alias + "::" + symbol, {"evidence": {alias: native}})]
    wire["line_ranges"] = copy.deepcopy(ranges)
    return wire, native


def _check_refs(facts: dict) -> list[str]:
    """Short navigation for common checks; keep their values in check_facts."""
    from .reviewer_contract import _resolve_pointer
    pointers = ("/ok", "/shape", "/output_shape", "/finite", "/finite_gradients",
        "/params_with_grad", "/params_trainable", "/checkpoint_round_trip",
        "/training_objective_finite", "/implementation_semantics/requirements",
        "/multigpu_check/ok", "/multigpu_check/result/custom_objective",
        "/multigpu_check/result/eval_transform_calls", "/multigpu_check/result/batches")
    result = []
    for pointer in pointers:
        try:
            _resolve_pointer(facts, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        result.append("checks:" + pointer)
    return result


def _clean_checks(checks: dict) -> dict:
    """Remove only explicit identity/process paths, never scientific key names."""
    result = copy.deepcopy(checks)
    for name in ("file", "source_sha256", "check_fingerprint", "loaded_from_workspace"):
        result.pop(name, None)
    semantics = result.get("implementation_semantics")
    if isinstance(semantics, dict):
        semantics.pop("schema_version", None)
        semantics.pop("requirements_hash", None)
    multigpu = result.get("multigpu_check")
    if isinstance(multigpu, dict):
        for name in ("pid", "child_pid", "proc_start", "child_proc_start", "started_at", "ended_at",
                     "cost_ledger", "gpu_seconds", "probe_identity", "request_sha256"):
            multigpu.pop(name, None)
        data = multigpu.get("result")
        if isinstance(data, dict):
            for name in ("file", "source_sha256", "probe_identity"):
                data.pop(name, None)
    return result


def _runtime_contract(payload: dict, experiment: dict) -> dict:
    interface = copy.deepcopy(payload.get("candidate_interface") or {})
    for name in ("schema_version", "design_contract_version", "implementation_requirements"):
        interface.pop(name, None)
    multigpu = interface.get("multigpu_check")
    if isinstance(multigpu, dict):
        for name in ("execution_fingerprint", "selection_manifest_hash", "protocol_fingerprint"):
            multigpu.pop(name, None)
    guarantees = interface.pop("runtime_guarantees", None)
    explicit = payload.get("runtime_guarantees")
    if guarantees is not None and explicit is not None and guarantees != explicit:
        raise ValueError("review_runtime_guarantees_conflict")
    result = {"candidate_interface": interface,
        "runtime_guarantees": copy.deepcopy(guarantees if guarantees is not None else explicit or []),
        "hook_applicability": copy.deepcopy(payload.get("hook_applicability") or {}),
        "evaluation_scope": copy.deepcopy(experiment.get("evaluation_scope") or {})}
    contract = payload.get("evaluation_contract") or {}
    result["evaluation_contract"] = {name: copy.deepcopy(value) for name, value in contract.items()
                                     if name not in {"fingerprint", "contract_fingerprint"}}
    goal = payload.get("goal") or (payload.get("context_authority") or {}).get("goal") or {}
    for key in ("allowed_changes", "frozen_components", "checkpoint_policy", "training_seeds",
                "task_type", "research_scope", "evaluation_mode", "primary_metric", "subject_aggregation",
                "training_recipe", "negative_sampling_policy", "precision", "checkpoint_format"):
        if key in goal:
            result[key] = copy.deepcopy(goal[key])
    return result


def _issues(payload: dict, spec: dict) -> list[dict]:
    focus = payload.get("review_focus") or spec.get("review_focus") or {}
    candidates = list(spec.get("repair_issues") or [])
    if isinstance(focus, dict):
        candidates.extend(focus.get("issues") or [])
    result, seen = [], {}
    for issue in candidates:
        if not isinstance(issue, dict):
            raise ValueError("review_previous_issue_not_object")
        if issue.get("status") in {"resolved", "closed", "superseded_by_current_checks"}:
            continue
        if issue.get("severity") == "non_blocking" or issue.get("blocking") is False:
            continue
        key = issue.get("issue_id") or digest(issue)
        if key in seen:
            if seen[key] != issue:
                raise ValueError("review_previous_issue_identity_conflict")
            continue
        seen[key] = issue
        result.append(copy.deepcopy(issue))
    return result


def _relevant_context(payload: dict, spec: dict) -> dict:
    items = []
    memory = spec.get("retrieved_memory") or payload.get("retrieved_memory") or {}
    if isinstance(memory, dict):
        # Real lesson rows stay literal. Retrieval machinery is not lesson text.
        items.extend(copy.deepcopy(memory.get("items") or []))
    elif isinstance(memory, list):
        items.extend(copy.deepcopy(memory))
    for name in ("audit_issues", "current_user_instruction"):
        value = spec.get(name)
        if value:
            items.append({"kind": name, "content": copy.deepcopy(value)})
    hints = payload.get("relevant_implementation_hints")
    if hints:
        items.append({"kind": "implementation_hints", "content": copy.deepcopy(hints)})
    snapshot = payload.get("working_snapshot") or {}
    if isinstance(snapshot, dict):
        for name in ("engineering_failures", "open_questions"):
            if snapshot.get(name):
                items.append({"kind": name, "content": copy.deepcopy(snapshot[name])})
    return {"items": items, "status": "available" if items else "no_related_context"}


def prepare_review_input(native_payload: dict[str, Any]) -> tuple[dict, dict]:
    """Capture current native identities, then build the fixed scientific wire.

    This is the explicit new-contract entry point; callers select v2 for new
    research.  The result is not citation-authorized until bind_review_delivery.
    """
    if not isinstance(native_payload, dict):
        raise ValueError("review_native_payload_not_object")
    payload = _expanded(native_payload)
    spec = payload.get("experiment_spec") or {}
    experiment = spec.get("experiment") if isinstance(spec.get("experiment"), dict) else spec
    approved = project_approved_design(spec, selected_task=payload.get("selected_research_task"),
        interface=payload.get("candidate_interface"), effective_hook_config=payload.get("effective_hook_config"))
    original_requirements = experiment.get("implementation_requirements")
    if not isinstance(original_requirements, list):
        original_requirements = (experiment.get("design_brief") or {}).get("implementation_requirements")
    if not isinstance(original_requirements, list):
        raise ValueError("review_implementation_requirements_missing")
    declarations = approved.pop("implementation_requirements")
    principal = approved.get("principal_intervention")
    if not isinstance(principal, str) or not principal.strip():
        raise ValueError("review_principal_intervention_missing")
    identity = copy.deepcopy(payload.get("requirement_evidence_identity") or {})
    sources, evidence = [], {}
    candidate, native_candidate = _freeze_source("candidate", {"source": payload.get("candidate_source")},
        fallback_path=identity.get("source_ref", ""), expected_hash=identity.get("source_hash", ""))
    if not native_candidate["full"]:
        raise ValueError("review_candidate_source_not_full")
    sources.append(candidate)
    evidence["candidate"] = native_candidate
    identity["source_ref"] = native_candidate["native_path"]
    identity["source_hash"] = native_candidate["sha256"]
    parent = payload.get("parent_source")
    if isinstance(parent, dict) and parent:
        source, frozen = _freeze_source("parent", parent)
        sources.append(source)
        evidence["parent"] = frozen
    for index, row in enumerate(payload.get("training_runtime_source") or [], 1):
        if not isinstance(row, dict):
            raise ValueError("review_runtime_source_not_object")
        stem = Path(str(row.get("source_ref") or "")).stem
        alias = {"model": "runtime_model", "hooks": "runtime_hooks", "train_entry": "runtime_train"}.get(stem, "runtime_" + str(index))
        if alias in evidence:
            raise ValueError("review_runtime_source_alias_collision")
        source, frozen = _freeze_source(alias, row)
        sources.append(source)
        evidence[alias] = frozen

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("review_checks_missing")
    check_path = str(identity.get("check_ref") or "")
    if check_path and Path(check_path).is_file():
        raw_check = Path(check_path).read_bytes()
        check_hash = hashlib.sha256(raw_check).hexdigest()
        if json.loads(raw_check) != checks:
            raise ValueError("review_checks_delivery_changed")
    elif check_path:
        raise ValueError("review_checks_file_missing")
    else:
        check_hash = digest(checks)
    if identity.get("check_sha256") and identity["check_sha256"] != check_hash:
        raise ValueError("review_checks_hash_changed")
    if checks.get("source_sha256") and checks["source_sha256"] != identity["source_hash"]:
        raise ValueError("review_checks_source_changed")
    multigpu_source = ((checks.get("multigpu_check") or {}).get("result") or {}).get("source_sha256")
    if multigpu_source and multigpu_source != identity["source_hash"]:
        raise ValueError("review_multigpu_source_changed")
    identity["check_ref"], identity["check_sha256"] = check_path, check_hash
    identity["approved_spec_hash"] = experiment.get("spec_hash", "")
    identity.pop("scope", None)
    native_bindings = {}
    source_path = Path(identity["source_ref"]) if identity.get("source_ref") else None
    if source_path:
        spec_path = source_path.parent.parent / "spec.json"
        if spec_path.is_file():
            raw_spec = spec_path.read_bytes()
            recorded = json.loads(raw_spec)
            recorded_experiment = recorded.get("experiment") or recorded
            if identity["approved_spec_hash"] and recorded_experiment.get("spec_hash") != identity["approved_spec_hash"]:
                raise ValueError("review_approved_spec_changed")
            native_bindings = {"spec_ref": str(spec_path), "spec_sha256": hashlib.sha256(raw_spec).hexdigest()}
    facts = _clean_checks(checks)
    if "approved_hook_config" in facts:
        expected = {name: hook["effective_config"] for name, hook in approved["hooks"].items()}
        if facts["approved_hook_config"] != expected:
            raise ValueError("review_checked_hook_config_changed")
        facts.pop("approved_hook_config")
    evidence["checks"] = {"kind": "checks", "native_path": check_path, "sha256": check_hash, "facts": facts}
    mapping = payload.get("implementation_map") or {}
    if mapping.get("source_sha256") and mapping["source_sha256"] != identity["source_hash"]:
        raise ValueError("review_implementation_map_source_changed")
    if mapping.get("approved_spec_hash") and mapping["approved_spec_hash"] != identity["approved_spec_hash"]:
        raise ValueError("review_implementation_map_spec_changed")
    locations = mapping.get("requirements") or {}
    roster = {"r0": {"kind": "intervention", "required": True, "expected_behavior": principal}}
    targets = {"r0": {"kind": "intervention"}}
    for index, (requirement, projected) in enumerate(zip(original_requirements, declarations), 1):
        alias = "r" + str(index)
        identifier = projected["requirement_id"]
        roster[alias] = {"kind": "requirement", "required": bool(requirement.get("core", True)),
                         "requirement": copy.deepcopy(requirement)}
        row = {key: copy.deepcopy(value) for key, value in projected.items() if key != "requirement_id"}
        row["kind"] = "requirement"
        if identifier in locations:
            row["implementation_locations"] = _alias_locations(locations[identifier], evidence)
        targets[alias] = row
    previous_issues = {}
    for index, issue in enumerate(_issues(payload, spec), 1):
        issue_ref, target_ref = "i" + str(index), "f" + str(index)
        previous_issues[issue_ref] = copy.deepcopy(issue)
        roster[target_ref] = {"kind": "issue_resolution", "required": True,
            "issue_ref": issue_ref, "issue": copy.deepcopy(issue)}
        targets[target_ref] = {"kind": "issue_resolution", "previous_issue_ref": issue_ref,
            **{key: copy.deepcopy(issue[key])
            for key in ("title", "detail", "reason", "smallest_correction", "validation_expectation",
                        "requirement_id", "requirement_ids", "code_locations", "source_refs", "evidence_refs") if key in issue}}
        targets[target_ref] = _alias_locations(targets[target_ref], evidence)
    focus = payload.get("review_focus") or {}
    wire = {"reviewer_context_version": VERSION, "approved_design": approved,
        "runtime_contract": _runtime_contract(payload, experiment), "review_targets": targets,
        "source_materials": sources, "check_facts": without_local_paths(facts),
        "parameter_facts": copy.deepcopy(payload.get("parameter_census") or {"status": "unavailable"}),
        "review_scope": {"mode": payload.get("review_mode") or "full",
                         "check_refs": _check_refs(facts),
                         "affected_requirements": copy.deepcopy(focus.get("affected_requirements") or []) if isinstance(focus, dict) else []},
        "relevant_context": _relevant_context(payload, spec)}
    evidence["parameters"] = {"kind": "parameters", "facts": copy.deepcopy(wire["parameter_facts"]),
                              "source_hash": identity["source_hash"]}
    context = {"contract_version": CONTRACT_VERSION, "input_version": VERSION, "roster": roster,
        "evidence": evidence, "identity": identity, "native_bindings": native_bindings,
        "previous_issues": previous_issues, "approved_design": copy.deepcopy(approved),
        "native_checks": copy.deepcopy(checks), "native_parameter_census": copy.deepcopy(payload.get("parameter_census")),
        "wire_digest": digest(wire), "wire_snapshot": copy.deepcopy(wire), "delivery": {"bound": False}}
    return wire, context


def _alias_locations(value: Any, evidence: dict) -> Any:
    """Shorten only known citation slots, never rewrite scientific prose."""
    paths = {row["native_path"]: alias for alias, row in evidence.items() if row.get("native_path")}
    paths["extension/eeg_candidate.py"] = "candidate"
    paths["checks.json"] = "checks"
    def reference(item):
        if not isinstance(item, str):
            return copy.deepcopy(item)
        for path, alias in sorted(paths.items(), key=lambda entry: -len(entry[0])):
            if item == path or item.startswith(path + "::") or item.startswith(path + ":") or item.startswith(path + "#L"):
                return alias + item[len(path):]
        return item
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = copy.deepcopy(value)
    for key in ("source_refs", "evidence_refs", "code_locations"):
        if isinstance(result.get(key), list):
            result[key] = [reference(item) for item in result[key]]
    return result


def bind_review_delivery(prepared_payload: dict, native_context: dict, *, system_sha256: str = "",
                         user_sha256: str = "", request_digest: str = "") -> dict:
    """Bind the exact final delivery; feedback cannot introduce new evidence.

    The eight science blocks must remain identical. Only previous_response and
    previous_response_error may be added by a bounded format/field correction.
    The transport persists the returned context beside the actual request.
    """
    if native_context.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("review_native_context_version_invalid")
    if not is_reviewer_input("candidate_reviewer", prepared_payload):
        raise ValueError("review_delivered_input_version_invalid")
    if set(prepared_payload) - ({"reviewer_context_version", *SCIENCE_FIELDS} | _CORRECTION_FIELDS):
        raise ValueError("review_delivery_unexpected_fields")
    if "previous_response" in prepared_payload and not isinstance(prepared_payload["previous_response"], str):
        raise ValueError("review_previous_response_must_be_text")
    actual = {key: copy.deepcopy(prepared_payload[key]) for key in ("reviewer_context_version", *SCIENCE_FIELDS)
              if key in prepared_payload}
    if digest(actual) != native_context.get("wire_digest") or actual != native_context.get("wire_snapshot"):
        raise ValueError("review_delivery_science_changed")
    context = copy.deepcopy(native_context)
    if (context["evidence"].get("parameters") or {}).get("facts") != actual["parameter_facts"]:
        raise ValueError("review_delivery_parameter_facts_changed")
    # Compute permissions from the final supplied segments, not from registration.
    for row in actual["source_materials"]:
        alias = row["source_ref"]
        frozen = context["evidence"][alias]
        if row["segments"] != [{key: segment[key] for key in ("start_line", "end_line", "text")}
                               for segment in frozen["segments"]]:
            raise ValueError("review_delivery_source_changed")
        frozen["delivered_ranges"] = [pair for segment in frozen["segments"]
                                       for pair in segment["complete_line_ranges"]]
    canonical_user_hash = _sha(canonical(prepared_payload))
    if user_sha256 and user_sha256 != canonical_user_hash:
        raise ValueError("review_delivery_user_digest_changed")
    context["delivery"] = {"bound": True, "system_sha256": system_sha256,
        "user_sha256": user_sha256 or canonical_user_hash, "request_digest": request_digest,
        "delivered_wire_digest": digest(actual)}
    return context


def canonical_review_evidence_request(context: dict) -> dict:
    """Adapt only actually delivered evidence for existing native reference gates."""
    if not (context.get("delivery") or {}).get("bound"):
        raise ValueError("review_delivery_not_bound")
    evidence = context["evidence"]
    result = {"requirement_evidence_identity": copy.deepcopy(context["identity"]),
              "candidate_source": evidence["candidate"]["source"],
              "checks": copy.deepcopy(evidence["checks"]["facts"]), "training_runtime_source": []}
    for alias, row in evidence.items():
        if row.get("kind") != "source" or alias == "candidate":
            continue
        material = {"source_ref": row["native_path"], "source_hash": row["sha256"]}
        material["source" if alias == "parent" else "excerpt"] = row["source"]
        if alias == "parent":
            result["parent_source"] = material
        else:
            result["training_runtime_source"].append(material)
    return result
