"""Verified audit feedback and reconstructable ResearchPlan issue projection.

No new store or executor: immutable role artifacts and the task ledger are the
source; plan.json is a projection. The same reader serves planning and API views.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
from react_agent.eeg_research.agentic.handoffs import _registry_rows, development_view
from react_agent.eeg_research.agentic.roles import RoleResultError, validate_role_result
from react_agent.eeg_research.agentic.task_ledger import latest

AUDIT_SCHEMA = "eeg_research.audit_feedback.v2"
IDENTITY_KEYS = ("audited_report_ref", "report_hash", "dependency_manifest_hash")


def audit_views(camp: Path) -> list[dict[str, Any]]:
    """Select by registered audit ID and envelope/ledger binding, never directory mtime."""
    views = []
    for row in _registry_rows(camp):
        if row.get("kind") != "audit":
            continue
        view = {"artifact_id": row.get("artifact_id"), "content_hash": row.get("sha256"),
                "completion_status": "unknown", "verification_status": "invalid",
                "identity_verified": False, "missing_inputs": [], "payload": None}
        try:
            registered = resolve_verified_artifact(camp, str(row["artifact_id"]))
            path = Path(str(registered["path"])).resolve()
            if not path.is_relative_to(camp.resolve()):
                raise ValueError("audit_outside_campaign")
            body = json.loads(path.read_text(encoding="utf-8"))
            task = latest(camp, str(registered.get("producer_task_id") or ""))
            if not task or task.get("role") != "result_auditor":
                raise ValueError("audit_task_missing")
            envelope = validate_role_result(body, task)
            payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else envelope
            clean = development_view(payload)
            if clean is None:
                raise ValueError("audit_scope_forbidden")
            view.update(verification_status="verified", completion_status=envelope.get("status"),
                        task_id=task["task_id"], attempt_id=task.get("attempt_id"),
                        input_digest=task.get("input_digest"), payload=clean)
            expected = {key: payload.get(key) for key in IDENTITY_KEYS}
            reported = payload.get("model_identity")
            valid = payload.get("audit_schema_version") == AUDIT_SCHEMA and all(expected.values()) and (
                isinstance(reported, dict) and reported == expected and payload.get("identity_verified") is True
            ) and payload.get("model_audit_status") == "completed" and envelope.get("status") == "completed"
            view["identity_verified"] = bool(valid)
            if not valid:
                view["missing_inputs"] = payload.get("model_audit_missing_inputs") or ["legacy_or_incomplete_model_audit"]
        except (OSError, ValueError, KeyError, RoleResultError) as exc:
            view["missing_inputs"] = [str(exc)]
        views.append(view)
    return views


def _issue(view: dict[str, Any], finding: Any, correction: Any, index: int, *, known: dict[str, set[str]]) -> dict[str, Any]:
    payload = view.get("payload") or {}
    raw = finding if isinstance(finding, dict) else {"problem": str(finding)}
    identity = json.dumps([view["artifact_id"], index, finding], sort_keys=True, ensure_ascii=False, default=str)
    severity = raw.get("severity") if isinstance(raw.get("severity"), str) and raw["severity"] in {"blocking", "warning", "non_blocking"} else "unknown"
    missing = []
    def link(key):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or value not in known[key]):
            missing.append("unknown_" + key)
            return None
        return value
    def refs(key):
        values = raw.get(key) or []
        if not isinstance(values, list):
            missing.append("invalid_" + key)
            return []
        if any(not isinstance(value, str) or value not in known[key] for value in values):
            missing.append("unknown_" + key)
        return [value for value in values if isinstance(value, str) and value in known[key]]
    claim_id, question_id, candidate_id = (link(key) for key in ("claim_id", "question_id", "candidate_id"))
    evidence_refs, artifact_refs = refs("evidence_refs"), refs("artifact_refs")
    return {"issue_id": "issue_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "source_audit_artifact_id": view["artifact_id"],
        "source_report_identity": {key: payload.get(key) for key in IDENTITY_KEYS},
        "claim_id": claim_id, "question_id": question_id, "candidate_id": candidate_id,
        "source_evidence_ids": payload.get("source_evidence_ids") or [],
        "evidence_refs": evidence_refs, "artifact_refs": artifact_refs, "missing_links": missing,
        "severity": severity, "blocking": severity in {"blocking", "unknown"},
        "category": raw.get("category") or "unknown", "problem": raw.get("problem") or raw.get("description") or str(finding),
        "required_correction": raw.get("required_correction") or correction,
        "verification_condition": raw.get("verification_condition") or "new_evidence_or_revised_claim_and_bound_reaudit",
        "status": "open", "resolution_evidence_refs": [], "resolution_audit_artifact_id": None}


def _resolution_evidence_valid(camp: Path, row: dict[str, Any], issue: dict[str, Any]) -> bool:
    if development_view(row) is None or row.get("kind") == "audit":
        return False
    if issue.get("candidate_id") and row.get("candidate_id") != issue["candidate_id"]:
        return False
    if issue.get("question_id") and row.get("question_id") not in {None, issue["question_id"]}:
        return False
    if row.get("kind") == "analysis":
        try:
            ref = row.get("analysis_artifact_id")
            registered = resolve_verified_artifact(camp, str(ref or ""))
            path = Path(str(registered["path"])).resolve()
            if registered.get("kind") != "analysis" or not path.is_relative_to(camp.resolve()):
                return False
            task = latest(camp, str(registered.get("producer_task_id") or ""))
            body = validate_role_result(json.loads(path.read_text()), task) if task else None
            return bool(body and body.get("status") == "completed" and task.get("role") == "result_analyst")
        except (OSError, ValueError, KeyError, RoleResultError):
            return False
    # Only runtime-valid experimental evidence qualifies directly. Diagnostic
    # and learning-profile summaries alone need a verified analysis artifact.
    return row.get("evaluation_valid") is True


def project_issues(camp: Path, state: dict[str, Any], views: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Only covered, evidence-backed resolutions close their corresponding issues."""
    views = audit_views(camp) if views is None else views
    from react_agent.eeg_research.agentic.research_plan import load_plan
    issues: dict[str, dict[str, Any]] = {row["issue_id"]: {**row, "status": "open",
        "resolution_evidence_refs": [], "resolution_audit_artifact_id": None}
        for row in (load_plan(camp).get("audit_issues") or []) if isinstance(row, dict) and row.get("issue_id")}
    evidence = {row.get("evidence_id"): row for row in state.get("evidence") or [] if row.get("evidence_id")}
    revisions = state.get("report_claim_revisions") or {}
    for view in views:
        payload = view.get("payload")
        if view.get("verification_status") != "verified" or not isinstance(payload, dict):
            continue
        findings = payload.get("open_issues") or []
        corrections = payload.get("required_corrections") or []
        if not isinstance(findings, list):
            findings = [findings]
        if not isinstance(corrections, list):
            corrections = [corrections]
        if not findings and corrections:
            findings = [{"problem": item, "required_correction": item} for item in corrections]
        for index, finding in enumerate(findings):
            correction = corrections[index] if index < len(corrections) else None
            known = {
                "claim_id": {row.get("claim_id") for row in payload.get("report_claims") or [] if isinstance(row, dict)},
                "question_id": {row.get("question_id") for row in load_plan(camp).get("research_questions") or []},
                "candidate_id": {"baseline", *[row.get("candidate_id") for row in state.get("candidates") or []],
                                 *[row.get("candidate_id") for row in evidence.values()]},
                "evidence_refs": set(evidence), "artifact_refs": {row.get("artifact_id") for row in _registry_rows(camp)},
            }
            item = _issue(view, finding, correction, index, known=known)
            issues.setdefault(item["issue_id"], item)
        # A complete bound audit can resolve one issue while other findings
        # keep the overall report at REVISE/BLOCK. Coverage is checked below.
        if not view.get("identity_verified"):
            continue
        report_claims = {row.get("claim_id"): row for row in (payload.get("report_claims") or []) if isinstance(row, dict)}
        for resolution in payload.get("resolved_issues") or []:
            if not isinstance(resolution, dict):
                continue
            item = issues.get(resolution.get("issue_id"))
            if not item or item["status"] != "open" or item["source_audit_artifact_id"] == view["artifact_id"]:
                continue
            refs = resolution.get("evidence_refs") or []
            if not isinstance(refs, list) or any(ref not in evidence for ref in refs):
                continue
            kind = resolution.get("resolution_kind")
            covered = False
            if kind == "new_evidence":
                new_refs = [ref for ref in refs if ref not in item["source_evidence_ids"]]
                claim_id = resolution.get("claim_id")
                claim_covered = not item.get("claim_id") or (claim_id == item["claim_id"] and claim_id in report_claims)
                covered = claim_covered and bool(new_refs) and all(ref in (payload.get("source_evidence_ids") or [])
                    and _resolution_evidence_valid(camp, evidence[ref], item) for ref in new_refs)
            elif kind in {"claim_withdrawn", "claim_narrowed"}:
                claim_id = resolution.get("claim_id")
                revision = revisions.get(claim_id) or {}
                claim = report_claims.get(claim_id) or {}
                expected = "withdraw" if kind == "claim_withdrawn" else "narrow"
                covered = bool(claim_id and claim_id == item.get("claim_id") and revision.get("operation") == expected
                               and claim.get("revision_operation") == expected
                               and all(claim.get(key) == revision.get(key) for key in ("statement", "scope_limits", "evidence_refs"))
                               and payload.get("report_hash") != item["source_report_identity"].get("report_hash"))
            if covered and resolution.get("rationale"):
                item.update(status="resolved", resolution_evidence_refs=list(refs),
                            resolution_audit_artifact_id=view["artifact_id"], resolution_kind=kind,
                            resolution_rationale=resolution["rationale"])
    return list(issues.values())


def sync_issues(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    from react_agent.eeg_research.agentic.research_plan import load_plan, _persist
    issues = project_issues(camp, state)
    plan = load_plan(camp)
    if plan and plan.get("audit_issues", []) != issues:
        plan["audit_issues"] = issues
        plan["plan_version"] = int(plan["plan_version"]) + 1
        _persist(camp, plan)
    state["audit_issues"] = issues
    return issues


def audit_feedback(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.loop import report_dependency_hash
    views = audit_views(camp)
    issues = project_issues(camp, state, views)
    selected = views[-3:]
    complete = next((row for row in reversed(views) if row.get("identity_verified")), None)
    if complete and complete not in selected:
        selected.insert(0, complete)
    current = state.get("report_draft") or {}
    snapshot = report_dependency_hash(state)
    for view in selected:
        payload = view.get("payload") or {}
        view["fresh"] = bool(view.get("identity_verified") and all(payload.get(key) == current.get(key)
            for key in ("report_hash", "dependency_manifest_hash")) and payload.get("audited_report_ref") == current.get("report_ref")
            and payload.get("audited_dependency_hash") == snapshot)
    latest_view = next((row for row in reversed(selected) if row.get("artifact_id") == state.get("latest_audit_artifact_id")), None)
    if latest_view is None and selected:
        latest_view = selected[-1]
    status = "unverified"
    if latest_view and latest_view.get("fresh"):
        verdict = (latest_view.get("payload") or {}).get("verdict")
        status = "blocked" if verdict == "BLOCK" else "needs_revision" if verdict == "REVISE" or any(
            item["status"] == "open" and item["blocking"] for item in issues) else "supported" if verdict == "PASS" else "unverified"
    return {"audits": selected, "latest_complete_audit_artifact_id": None if complete is None else complete["artifact_id"],
            "issues": issues, "unresolved_issue_ids": [item["issue_id"] for item in issues if item["status"] == "open"],
            "report_support_status": status,
            "missing_inputs": [] if views else ["registered_audit_missing"],
            "supported_is_confirmed_improvement": False}


def reusable_audit(camp: Path, *, report_hash: str, manifest_hash: str) -> dict[str, Any] | None:
    for view in reversed(audit_views(camp)):
        payload = view.get("payload") or {}
        if view.get("identity_verified") and payload.get("report_hash") == report_hash and payload.get("dependency_manifest_hash") == manifest_hash:
            return view
    return None
