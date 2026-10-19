"""Reviewer v2: small model judgments, frozen evidence and native assembly.

The model never supplies source/check identities or an overall runtime verdict.
Every judgment is interpreted against the same roster and delivered evidence,
including correction and adoption of a previously paid response.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from .context.types import digest

VERSION = "eeg_research.reviewer_contract.v2"
INPUT_VERSION = "reviewer_implementation_input.v2"
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReviewAssessment(_Wire):
    status: Literal["implemented", "contradicted", "unverified"]
    detail: Text
    evidence_refs: list[Text]


class ReviewIssue(_Wire):
    severity: Literal["blocking", "non_blocking"]
    kind: Literal["code_error", "design_conflict", "framework_issue", "scientific_replan"]
    target_refs: list[Text]
    detail: Text
    evidence_refs: list[Text]
    smallest_correction: Text | None
    validation_expectation: Text | None


class ReviewIssueWithHistory(ReviewIssue):
    previous_issue_ref: Text | None


class ReviewerContractError(ValueError):
    def __init__(self, *, errors=None, native_errors=None):
        self.errors = errors or []
        self.native_errors = native_errors or []
        super().__init__((self.native_errors or self.errors or [{"type": "reviewer_contract_invalid"}])[0]["type"])


def _error(kind: str, loc, message: str) -> dict:
    return {"type": kind, "loc": list(loc), "msg": message}


def enabled(full_or_goal: dict | Path | None) -> bool:
    if isinstance(full_or_goal, Path):
        try:
            full_or_goal = json.loads((full_or_goal / "goal.json").read_text())
        except (OSError, ValueError):
            return False
    full = full_or_goal or {}
    if full.get("contract_version") == VERSION or full.get("reviewer_context_version") == INPUT_VERSION:
        return True
    goal = full.get("goal") if isinstance(full.get("goal"), dict) else full
    from .research_execution import single_plan_enabled
    return single_plan_enabled(goal)


def reviewer_wire_schema(context: dict) -> dict:
    """One schema from the frozen roster; Reviewer has no ranking protocol."""
    row_schema = ReviewAssessment.model_json_schema()
    issue_schema = (ReviewIssueWithHistory if context.get("previous_issues") else ReviewIssue).model_json_schema()
    issue_schema["properties"]["target_refs"]["items"]["enum"] = list(context.get("roster") or {})
    if context.get("previous_issues"):
        issue_schema["properties"]["previous_issue_ref"] = {
            "anyOf": [{"type": "string", "enum": list(context["previous_issues"])}, {"type": "null"}]}
    roster_schema = {"type": "object", "properties": {
        ref: {"$ref": "#/$defs/Assessment"} for ref in context.get("roster") or {}},
        "required": list(context.get("roster") or {}), "additionalProperties": False}
    return {"type": "object", "additionalProperties": False,
        "properties": {"assessments": roster_schema,
                       "issues": {"type": "array", "items": {"$ref": "#/$defs/Issue"}},
                       "summary_zh": {"type": "string", "minLength": 1}},
        "required": ["assessments", "issues", "summary_zh"],
        "$defs": {"Assessment": row_schema, "Issue": issue_schema}}


def _context_errors(context: Any, *, require_delivery: bool) -> list[dict]:
    if not isinstance(context, dict) or context.get("contract_version") != VERSION:
        return [_error("reviewer_native_context_invalid", [], "The original frozen Reviewer v2 context is required.")]
    errors = []
    roster, evidence, identity = (context.get(key) for key in ("roster", "evidence", "identity"))
    if not isinstance(roster, dict) or not roster:
        errors.append(_error("reviewer_native_roster_invalid", ["roster"], "A nonempty frozen roster is required."))
    else:
        requirements, issues = [], []
        for ref, target in roster.items():
            loc = ["roster", ref]
            if (not isinstance(ref, str) or not ref.strip() or not isinstance(target, dict)
                    or target.get("kind") not in {"intervention", "requirement", "issue_resolution"}):
                errors.append(_error("reviewer_native_target_invalid", loc, "The frozen review target is invalid."))
                continue
            if target["kind"] == "requirement":
                req = target.get("requirement")
                if not isinstance(req, dict) or not req.get("requirement_id"):
                    errors.append(_error("reviewer_native_requirement_invalid", loc, "The native requirement binding is missing."))
                else:
                    requirements.append(req["requirement_id"])
            elif target["kind"] == "issue_resolution":
                prior_ref = target.get("issue_ref")
                issue = (context.get("previous_issues") or {}).get(prior_ref)
                if not isinstance(issue, dict) or issue != target.get("issue"):
                    errors.append(_error("reviewer_native_issue_binding_invalid", loc, "The prior issue binding is missing or changed."))
                issues.append(prior_ref)
        if len(set(requirements)) != len(requirements) or len(set(issues)) != len(issues):
            errors.append(_error("reviewer_native_target_binding_duplicate", ["roster"], "Each native requirement and prior issue has one target."))
        if set(issues) != set(context.get("previous_issues") or {}):
            errors.append(_error("reviewer_native_prior_issue_coverage_invalid", ["roster"], "Each active prior issue needs an explicit resolution target."))
    if not isinstance(evidence, dict) or not isinstance(evidence.get("candidate"), dict):
        errors.append(_error("reviewer_native_evidence_invalid", ["evidence"], "The candidate evidence binding is missing."))
    if not isinstance(identity, dict) or not all(identity.get(key) for key in ("source_hash", "source_ref", "check_ref", "check_sha256")):
        errors.append(_error("reviewer_native_identity_invalid", ["identity"], "The native source/check identities are required."))
    elif isinstance(evidence, dict):
        for alias, path_key, hash_key in (("candidate", "source_ref", "source_hash"), ("checks", "check_ref", "check_sha256")):
            row = evidence.get(alias) or {}
            if row.get("native_path") != identity[path_key] or row.get("sha256") != identity[hash_key]:
                errors.append(_error("reviewer_native_evidence_identity_mismatch", ["evidence", alias], "Evidence must bind the original source/check identity."))
        parameters = evidence.get("parameters")
        if isinstance(parameters, dict) and parameters.get("source_hash") != identity["source_hash"]:
            errors.append(_error("reviewer_native_parameter_source_mismatch", ["evidence", "parameters"], "The native parameter snapshot must bind the reviewed source."))
    if require_delivery and (context.get("delivery") or {}).get("bound") is not True:
        errors.append(_error("reviewer_native_delivery_unbound", ["delivery"], "The actual delivered request must be bound before validating a reply."))
    return errors


def validate_current_bindings(context: dict) -> list[dict]:
    """Local freshness checks; never ask a model to recreate missing identities."""
    errors = _context_errors(context, require_delivery=False)
    if errors:
        return errors
    identity = context["identity"]
    files = {identity["source_ref"]: identity["source_hash"], identity["check_ref"]: identity["check_sha256"]}
    for row in context["evidence"].values():
        if isinstance(row, dict) and row.get("native_path") and row.get("sha256"):
            files[row["native_path"]] = row["sha256"]
    bindings = context.get("native_bindings") or {}
    if bindings.get("spec_ref") and bindings.get("spec_sha256"):
        files[bindings["spec_ref"]] = bindings["spec_sha256"]
    for path, expected in files.items():
        try:
            valid = hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            errors.append(_error("reviewer_native_binding_changed", ["native_bindings", str(path)], "A frozen source, specification or check file is missing or changed."))
    checks = context.get("native_checks") or {}
    if checks.get("source_sha256") and checks["source_sha256"] != identity["source_hash"]:
        errors.append(_error("reviewer_native_check_source_mismatch", ["native_checks", "source_sha256"], "The check receipt belongs to a different source."))
    if checks.get("check_fingerprint"):
        from .native_patch import _check_fingerprint
        try:
            matches = checks["check_fingerprint"] == _check_fingerprint(Path(identity["source_ref"]).parent.parent, None)
        except (OSError, ValueError, KeyError, TypeError):
            matches = False
        if not matches:
            errors.append(_error("reviewer_native_check_fingerprint_changed", ["native_checks", "check_fingerprint"], "The frozen check no longer matches the current checker and approved configuration."))
    declared = [row["requirement"] for row in context["roster"].values() if row["kind"] == "requirement"]
    semantics = checks.get("implementation_semantics") or {}
    if semantics.get("requirements_hash"):
        from .implementation_semantics import requirements_hash
        if semantics["requirements_hash"] != requirements_hash(declared):
            errors.append(_error("reviewer_native_check_requirements_mismatch", ["native_checks", "implementation_semantics"], "The check receipt belongs to different approved requirements."))
    if bindings.get("spec_ref") and identity.get("approved_spec_hash"):
        try:
            stored = json.loads(Path(bindings["spec_ref"]).read_text())
            experiment = stored.get("experiment") or stored
            matches = experiment.get("spec_hash") == identity["approved_spec_hash"]
        except (OSError, ValueError, TypeError):
            matches = False
        if not matches:
            errors.append(_error("reviewer_native_spec_identity_mismatch", ["native_bindings", "spec_ref"], "The approved specification identity changed."))
    return errors


def _resolve_pointer(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            raise KeyError(pointer)
    return value


def _source_line_delivered(row: dict, number: int) -> bool:
    if "delivered_ranges" in row:
        return any(left <= number <= right for left, right in row["delivered_ranges"])
    segments = row.get("segments")
    if isinstance(segments, list):
        return any(isinstance(segment, dict) and int(segment.get("start_line", 0)) <= number <= int(segment.get("end_line", -1))
                   and isinstance(segment.get("text"), str) and segment["text"].strip() for segment in segments)
    source = row.get("source")
    return isinstance(source, str) and 1 <= number <= len(source.splitlines())


def resolve_evidence_reference(reference: str, context: dict) -> dict | None:
    """Resolve only a delivered source line/symbol or exact delivered check node."""
    if not isinstance(reference, str):
        return None
    evidence = context.get("evidence") or {}
    if "::" in reference:
        alias, symbol = reference.split("::", 1)
        row = evidence.get(alias)
        if not isinstance(row, dict) or row.get("kind") != "source":
            return None
        span = (row.get("symbol_spans") or {}).get(symbol)
        if span is None and isinstance(row.get("source"), str):
            from .review_context import _symbol_line
            line = _symbol_line(row["source"], symbol)
        else:
            line = span[0] if isinstance(span, (list, tuple)) and len(span) >= 1 else None
        if not isinstance(line, int) or not _source_line_delivered(row, line):
            return None
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            # A prefix containing the definition line does not grant a whole
            # method citation. Literal omitted blank separators carry no code.
            original = row.get("full_source", row.get("source", "")).splitlines()
            if any(not _source_line_delivered(row, number) for number in range(span[0], span[1] + 1)
                   if 1 <= number <= len(original) and original[number - 1].strip()):
                return None
        return {"kind": "source", "alias": alias, "native_ref": str(row["native_path"]) + "::" + symbol,
                "symbol": symbol, "line": line}
    match = re.fullmatch(r"([^:]+):([1-9][0-9]*)", reference)
    if match:
        alias, number = match.group(1), int(match.group(2))
        row = evidence.get(alias)
        if isinstance(row, dict) and row.get("kind") == "source" and _source_line_delivered(row, number):
            return {"kind": "source", "alias": alias, "native_ref": str(row["native_path"]) + ":" + str(number), "line": number}
        return None
    if ":/" in reference:
        alias, pointer = reference.split(":", 1)
        row = evidence.get(alias)
        if not isinstance(row, dict) or row.get("kind") not in {"checks", "parameters"}:
            return None
        try:
            fact = _resolve_pointer(row.get("facts"), pointer)
        except (KeyError, IndexError, ValueError, TypeError):
            return None
        kind = "parameter" if row["kind"] == "parameters" else "check"
        native_ref = "parameter_facts:" + pointer if kind == "parameter" else str(row["native_path"]) + "#" + pointer
        return {"kind": kind, "alias": alias, "native_ref": native_ref,
                "pointer": pointer, "fact": fact}
    return None


def _target_probe(target: dict, context: dict) -> dict | None:
    requirement = target.get("requirement") or {}
    if requirement.get("verification") != "ordered_window_slots":
        return None
    checks = context.get("native_checks") or {}
    return next((row for row in (checks.get("implementation_semantics") or {}).get("requirements") or []
                 if row.get("requirement_id") == requirement.get("requirement_id")
                 and row.get("verification") == "ordered_window_slots"), None)


def _direct_counterexample(target: dict, resolved: list[dict], context: dict) -> bool:
    probe = _target_probe(target, context)
    if not probe or probe.get("status") != "contradicted":
        return False
    def contains(value):
        if isinstance(value, dict):
            if (value.get("requirement_id") == probe.get("requirement_id")
                    and value.get("verification") == "ordered_window_slots" and value.get("status") == "contradicted"):
                return True
            return any(contains(child) for child in value.values())
        return isinstance(value, list) and any(contains(child) for child in value)
    for row in resolved:
        if row["kind"] != "check":
            continue
        if contains(row["fact"]):
            return True
        # Citing the exact status/actual leaf of the matching probe is also
        # valid; an unrelated /ok leaf never gains access via the root object.
        pointer = row["pointer"]
        while "/" in pointer[1:]:
            pointer = pointer.rsplit("/", 1)[0]
            if contains(_resolve_pointer(context["evidence"][row["alias"]]["facts"], pointer)):
                return True
    return False


def validate_reviewer_reply(raw: Any, context: dict, delivered: dict | None = None) -> dict:
    """The same complete validator serves live replies, correction and recovery."""
    result = {"errors": [], "native_errors": _context_errors(context, require_delivery=True), "normalized": None,
              "context_digest": digest(context)}
    if result["native_errors"]:
        return result
    if delivered is not None:
        from .reviewer_context import SCIENCE_FIELDS
        science = {key: delivered[key] for key in ("reviewer_context_version", *SCIENCE_FIELDS) if key in delivered}
    else:
        science = None
    if science is not None and (context["delivery"].get("delivered_wire_digest")
                               and digest(science) != context["delivery"]["delivered_wire_digest"]):
        result["native_errors"].append(_error("reviewer_native_delivery_changed", ["delivery"], "Use the original delivered request and roster."))
        return result
    result["native_errors"] = validate_current_bindings(context)
    if result["native_errors"]:
        return result
    errors = result["errors"]
    if not isinstance(raw, dict):
        errors.append(_error("dict_type", [], "Return the complete three-key JSON object."))
        return result
    for key in {"assessments", "issues", "summary_zh"} - raw.keys():
        errors.append(_error("missing", [key], "missing " + key))
    for key in raw.keys() - {"assessments", "issues", "summary_zh"}:
        errors.append(_error("extra_forbidden", [key], "This field is native-owned or outside the Reviewer contract."))
    assessments, issues = raw.get("assessments"), raw.get("issues")
    if not isinstance(assessments, dict):
        errors.append(_error("dict_type", ["assessments"], "Return the fixed-key assessments object."))
        assessments = {}
    for ref in context["roster"].keys() - assessments.keys():
        errors.append(_error("missing", ["assessments", ref], "missing " + ref))
    for ref in assessments.keys() - context["roster"].keys():
        errors.append(_error("extra_forbidden", ["assessments", ref], "Use only the frozen assessment keys."))
    parsed_rows = {}
    for ref, row in assessments.items():
        if ref not in context["roster"]:
            continue
        try:
            parsed_rows[ref] = ReviewAssessment.model_validate(row).model_dump()
        except ValidationError as exc:
            errors.extend(_error(error["type"], ["assessments", ref, *error["loc"]], error["msg"]) for error in exc.errors())
    parsed_issues = []
    if not isinstance(issues, list):
        errors.append(_error("list_type", ["issues"], "Return an issues array, using [] when no new finding is needed."))
        issues = []
    model = ReviewIssueWithHistory if context.get("previous_issues") else ReviewIssue
    for index, issue in enumerate(issues):
        try:
            parsed_issues.append((index, model.model_validate(issue).model_dump()))
        except ValidationError as exc:
            errors.extend(_error(error["type"], ["issues", index, *error["loc"]], error["msg"]) for error in exc.errors())
    try:
        from pydantic import TypeAdapter
        summary_zh = TypeAdapter(Text).validate_python(raw.get("summary_zh"))
    except ValidationError as exc:
        errors.extend(_error(error["type"], ["summary_zh", *error["loc"]], error["msg"]) for error in exc.errors())
        summary_zh = ""
    resolved_rows = {}
    for ref, row in parsed_rows.items():
        refs = []
        for index, reference in enumerate(row["evidence_refs"]):
            resolved = resolve_evidence_reference(reference, context)
            if resolved is None:
                errors.append(_error("review_evidence_reference_invalid", ["assessments", ref, "evidence_refs", index], "Cite a delivered source symbol/line, checks:/pointer or parameters:/pointer fact."))
            else:
                refs.append(resolved)
        resolved_rows[ref] = refs
        target = context["roster"][ref]
        code_evidence = any(item["kind"] == "source" and item["alias"] == "candidate" for item in refs)
        if row["status"] in {"implemented", "contradicted"} and not refs:
            errors.append(_error("review_evidence_missing", ["assessments", ref, "evidence_refs"], "A definite judgment requires delivered evidence; unverified may use []."))
        needs_candidate = (target["kind"] in {"intervention", "requirement"}
                           or target["kind"] == "issue_resolution" and (target.get("issue") or {}).get("kind") in {"code_error", "design_conflict"})
        if (needs_candidate and row["status"] in {"implemented", "contradicted"}
                and not code_evidence and not (row["status"] == "contradicted" and _direct_counterexample(target, refs, context))):
            errors.append(_error("review_code_evidence_missing", ["assessments", ref, "evidence_refs"], "Cite the actual candidate implementation; a passing check alone does not establish code semantics."))
    previous_updates = set()
    for index, issue in parsed_issues:
        loc = ["issues", index]
        refs = issue["target_refs"]
        if len(refs) != len(set(refs)) or any(ref not in context["roster"] for ref in refs):
            errors.append(_error("review_issue_target_invalid", [*loc, "target_refs"], "Select unique supplied target keys, or [] for an additional finding."))
        prior = issue.get("previous_issue_ref")
        if prior is not None:
            if prior not in (context.get("previous_issues") or {}) or prior in previous_updates:
                errors.append(_error("review_previous_issue_invalid", [*loc, "previous_issue_ref"], "Update a supplied prior issue at most once."))
            previous_updates.add(prior)
            resolution = next((key for key, target in context["roster"].items()
                               if target["kind"] == "issue_resolution" and target.get("issue_ref") == prior), None)
            if resolution and (parsed_rows.get(resolution) or {}).get("status") == "implemented":
                errors.append(_error("review_issue_resolution_conflict", [*loc, "previous_issue_ref"], "A resolved prior issue cannot also be returned as an open update."))
        for position, reference in enumerate(issue["evidence_refs"]):
            if resolve_evidence_reference(reference, context) is None:
                errors.append(_error("review_evidence_reference_invalid", [*loc, "evidence_refs", position], "Use an actually delivered evidence reference."))
        if issue["severity"] == "blocking":
            if not issue["evidence_refs"]:
                errors.append(_error("review_issue_evidence_missing", [*loc, "evidence_refs"], "A blocking finding needs delivered evidence."))
            if issue["kind"] != "framework_issue":
                for key in ("smallest_correction", "validation_expectation"):
                    if not issue[key]:
                        errors.append(_error("review_issue_correction_missing", [*loc, key], "An actionable blocker needs a concrete correction and validation expectation."))
            for ref in refs:
                if (parsed_rows.get(ref) or {}).get("status") == "implemented":
                    errors.append(_error("review_issue_assessment_conflict", [*loc, "target_refs"], "A directly contradicted target cannot simultaneously be implemented; use [] for a separate defect."))
    for ref, row in parsed_rows.items():
        target = context["roster"][ref]
        if (target["kind"] == "issue_resolution" or not target.get("required", True)
                or row["status"] != "contradicted" or _target_probe(target, context) and _target_probe(target, context).get("status") == "contradicted"):
            continue
        existing = any(prior.get("severity") == "blocking" and prior_ref not in previous_updates
                       and (parsed_rows.get(fref) or {}).get("status") != "implemented"
                       and _prior_issue_covers_target(prior, ref, target)
                       for fref, previous in context["roster"].items() if previous["kind"] == "issue_resolution"
                       for prior_ref, prior in [(previous["issue_ref"], previous["issue"])])
        if not existing and not any(issue["severity"] == "blocking" and ref in issue["target_refs"] for _, issue in parsed_issues):
            errors.append(_error("review_contradiction_finding_missing", ["assessments", ref], "A contradicted required target needs a concrete blocking finding identifying its correction."))
    if not errors:
        result["normalized"] = {"assessments": parsed_rows, "issues": [issue for _, issue in parsed_issues], "summary_zh": summary_zh}
    return result


def _native_refs(refs: list[str], context: dict, *, code_only=False) -> list[str]:
    resolved = [resolve_evidence_reference(reference, context) for reference in refs]
    return list(dict.fromkeys(row["native_ref"] for row in resolved if row and (not code_only or row["kind"] == "source")))


def _issue_id(issue: dict, context: dict) -> str:
    return "implementation_issue_" + digest({"identity": context["identity"], "finding": issue})[:24]


def _prior_issue_id(issue: dict) -> str:
    # An imported native finding can lack an ID before its first model review.
    # Its stable content, rather than the later repaired source, names it.
    return issue.get("issue_id") or "implementation_issue_" + digest(issue)[:24]


def _prior_issue_covers_target(issue: dict, ref: str, target: dict) -> bool:
    ids = [*(issue.get("requirement_ids") or []), issue.get("requirement_id")]
    if target["kind"] == "requirement" and any(ids):
        return target["requirement"]["requirement_id"] in ids
    return ref in (issue.get("target_refs") or [])


def materialize_reviewer_reply(raw: Any, context: dict, delivered: dict | None = None) -> dict:
    """Attach local identities/status; never change the raw paid model response."""
    checked = validate_reviewer_reply(raw, context, delivered)
    return materialize_checked_reviewer_reply(checked, context)


def materialize_checked_reviewer_reply(checked: dict, context: dict) -> dict:
    """Assemble a just-validated reply without reading the same evidence twice."""
    if checked["errors"] or checked["native_errors"]:
        raise ReviewerContractError(errors=checked["errors"], native_errors=checked["native_errors"])
    if checked.get("context_digest") != digest(context) or not isinstance(checked.get("normalized"), dict):
        raise ReviewerContractError(native_errors=[_error("reviewer_validated_context_changed", [], "Assemble the reply against the same validated frozen context.")])
    wire = checked["normalized"]
    identity, roster = context["identity"], context["roster"]
    rows, intervention, issues, resolutions = [], None, [], []
    unverified = []
    updates = {row["previous_issue_ref"]: row for row in wire["issues"] if row.get("previous_issue_ref") is not None}
    for ref, assessment in wire["assessments"].items():
        target = roster[ref]
        status, detail = assessment["status"], assessment["detail"]
        probe = _target_probe(target, context)
        if probe and probe.get("status") == "contradicted":
            status = "contradicted"
            detail = "Current native ordered-window counterexample contradicts the approved requirement. " + detail
        elif probe and probe.get("status") != "implemented" and target.get("required", True):
            status = "unverified"
            detail = "The approved ordered-window probe has no successful current receipt. " + detail
        elif (target.get("requirement") or {}).get("verification") == "ordered_window_slots" and probe is None:
            status = "unverified"
            detail = "The approved ordered-window probe receipt is missing. " + detail
        refs = _native_refs(assessment["evidence_refs"], context)
        source_refs = _native_refs(assessment["evidence_refs"], context, code_only=True)
        if target.get("required", True) and status == "unverified":
            unverified.append(ref)
        if target["kind"] == "requirement":
            req = target["requirement"]
            rows.append({"requirement_id": req["requirement_id"], "status": status, "detail": detail,
                         "source_refs": source_refs, **{key: identity[key] for key in ("source_hash", "check_ref", "check_sha256")}})
            if probe and probe.get("status") == "contradicted":
                native_issue = {"severity": "blocking" if target.get("required", True) else "non_blocking", "kind": "code_error",
                    "target_refs": [ref], "requirement_id": req["requirement_id"],
                    "detail": "The current native ordered-window probe demonstrates an approved implementation violation.",
                    "evidence_refs": [identity["check_ref"]], "native_counterexample": copy.deepcopy(probe),
                    "smallest_correction": "Implement the approved requirement: " + req["expected_behavior"],
                    "validation_expectation": "The ordered_window_slots probe for " + req["requirement_id"] + " passes on the repaired source."}
                native_issue["issue_id"] = _issue_id(native_issue, context)
                issues.append(native_issue)
        elif target["kind"] == "intervention":
            intervention = {"status": status, "detail": detail, "source_refs": source_refs,
                            "evidence_refs": refs, **{key: identity[key] for key in ("source_hash", "check_ref", "check_sha256")}}
        else:
            prior_ref = target["issue_ref"]
            prior = copy.deepcopy(context["previous_issues"][prior_ref])
            old_id = _prior_issue_id(prior)
            resolution = "resolved" if status == "implemented" else "open" if status == "contradicted" else "unverified"
            resolutions.append({"issue_id": old_id, "previous_issue_ref": prior_ref, "status": resolution,
                                "detail": detail, "evidence_refs": refs, "source_hash": identity["source_hash"],
                                "check_sha256": identity["check_sha256"]})
            if resolution != "resolved":
                # The old finding stays open even when the API returns issues=[].
                # A later explicit model update can change its cause/owner below.
                if prior_ref not in updates:
                    prior.update(issue_id=old_id, status=resolution, resolution_detail=detail,
                                 current_evidence_refs=refs)
                    issues.append(prior)
    for issue in wire["issues"]:
        native = copy.deepcopy(issue)
        prior_ref = native.pop("previous_issue_ref", None)
        prior = copy.deepcopy((context.get("previous_issues") or {}).get(prior_ref) or {})
        native["evidence_refs"] = _native_refs(issue["evidence_refs"], context)
        native["requirement_ids"] = [roster[ref]["requirement"]["requirement_id"] for ref in issue["target_refs"]
                                     if roster[ref]["kind"] == "requirement"]
        native["issue_id"] = _prior_issue_id(prior) if prior else _issue_id(native, context)
        if prior:
            native["previous_issue_ref"] = prior_ref
            native["origin_finding"] = prior.get("origin_finding") or prior
        native.update(source_hash=identity["source_hash"], check_ref=identity["check_ref"],
                      check_sha256=identity["check_sha256"], status="open")
        issues.append(native)
    blocking = [row for row in issues if row.get("severity") == "blocking"]
    native_ok = (context.get("native_checks") or {}).get("ok") is True
    required_failed = [ref for ref, assessment in wire["assessments"].items()
                       if roster[ref].get("required", True) and assessment["status"] != "implemented"]
    required_failed = list(dict.fromkeys([*required_failed, *unverified]))
    # Kind is an explicit model judgment (or an exact native counterexample),
    # never a prose/keyword classifier.
    actionable = any(row.get("kind") in {"code_error", "design_conflict", "scientific_replan"} for row in blocking)
    framework = any(row.get("kind") == "framework_issue" for row in blocking)
    status = ("blocked" if framework else "needs_fix" if actionable
              else "blocked" if blocking or required_failed or not native_ok else "ready")
    from .implementation_semantics import requirements_hash
    return {"reviewer_contract_version": "v2", "status": status, "summary_zh": wire["summary_zh"],
        "assessments": copy.deepcopy(wire["assessments"]), "requirement_coverage": rows,
        "intervention_coverage": intervention, "issues": issues, "issue_resolutions": resolutions,
        "unverified_target_refs": unverified, "unresolved_target_refs": required_failed,
        "blocking_remaining": len(blocking), "requirement_evidence_current": native_ok,
        "requirements_hash": requirements_hash([target["requirement"] for target in roster.values() if target["kind"] == "requirement"]),
        "review_binding": copy.deepcopy(identity), "review_native_bindings": copy.deepcopy(context.get("native_bindings") or {}),
        "review_context_digest": digest(context), "review_stage": "implementation",
        "format_failed": False, "input_hash": identity["source_hash"]}
