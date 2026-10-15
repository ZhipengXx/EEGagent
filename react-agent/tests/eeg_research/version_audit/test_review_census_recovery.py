"""Paid reviewer diagnostics can be revalidated without replacing their evidence."""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError, create_model

from react_agent.eeg_research.agentic import llm, owner_budget
from react_agent.eeg_research.agentic.schemas import DomainOutput, ImplementationReview
from test_coder_format_feedback import sdk  # Shared offline SDK transport; real sockets are forbidden.


def campaign(path):
    goal = {"max_llm_calls": 8, "memory": {"enabled": False},
        "research_execution": {"version": "single_plan_v1", "implementation_review_mode": "single_pass"},
        "role_models": {"candidate_reviewer": {"model": "synthetic-format", "thinking": False,
                                               "max_tokens": 8192}}}
    for name, value in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                       ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (path / name).write_text(json.dumps(value))
    source, check = path / "eeg_candidate.py", path / "checks.json"
    source.write_text("class EEGCandidate:\n    def build_encoder(self):\n        return 1\n")
    check.write_text('{"ok":true}')
    sha = lambda target: hashlib.sha256(target.read_bytes()).hexdigest()
    evidence = {"source_ref": str(source), "source_hash": sha(source),
                "check_ref": str(check), "check_sha256": sha(check)}
    return {"candidate_source": source.read_text(), "checks": {"ok": True},
            "requirement_evidence_identity": evidence}


def ready_reply(payload):
    evidence = payload["requirement_evidence_identity"]
    return {"status": "ready", "issues": [], "summary_zh": "Synthetic implementation checked.",
        "parameter_census": {"candidate_id": "synthetic", "encoder_scalar_parameters": 5335969},
        "requirement_coverage": [{"requirement_id": f"requirement-{index}", "status": "implemented",
            "source_hash": evidence["source_hash"], "source_refs": [evidence["source_ref"]],
            "check_ref": evidence["check_ref"], "check_sha256": evidence["check_sha256"],
            "detail": "The supplied implementation and current checks support this requirement."}
            for index in range(5)]}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paid_schema_rejection(path, sdk, monkeypatch, *, mutate=None):
    wire, responses = sdk
    payload = campaign(path)
    reply = ready_reply(payload)
    if mutate:
        mutate(reply)
    # Reproduce a paid schema rejection under the pre-census contract, rather
    # than rewriting a successful trace into a synthetic failure.
    old_review = create_model("BeforeCensusImplementationReview", __base__=DomainOutput,
        **{name: (field.annotation, field) for name, field in ImplementationReview.model_fields.items()
           if name != "parameter_census"})
    original = llm.domain_model_for
    state = {"old": True}
    def select_model(role, *args, **kwargs):
        return old_review if role == "candidate_reviewer" and state["old"] else original(role, *args, **kwargs)
    monkeypatch.setattr(llm, "domain_model_for", select_model)
    responses.append(json.dumps(reply))
    backend = llm.role_backend(path, "candidate_reviewer")
    with pytest.raises(llm.LlmUnavailable, match="domain_schema_invalid"):
        backend(payload)
    assert len(wire) == 1
    call = rows(path / "llm_calls.jsonl")[0]
    response = path / "llm_responses" / (call["call_id"] + ".json")
    assert json.loads(response.read_text())["validation"] == "domain_schema_invalid"
    assert call["success"] is True and call["transport_outcome"] == "confirmed_response"
    state["old"] = False
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
    return backend, payload, reply, response


def test_optional_parameter_census_is_diagnostic_and_excluded_from_verdict_dump(tmp_path):
    reply = ready_reply(campaign(tmp_path))
    validated = ImplementationReview.model_validate(reply)
    assert validated.status == "ready" and len(validated.requirement_coverage) == 5
    assert validated.parameter_census == reply["parameter_census"]
    assert "parameter_census" not in validated.model_dump()
    assert "parameter_census" not in json.loads(validated.model_dump_json())
    assert "parameter_census" not in ImplementationReview.model_validate({"status": "ready"}).model_dump()


@pytest.mark.parametrize("reply", [
    {"status": "ready", "issues": [{"severity": "blocking", "category": "code_error"}]},
    {"status": "needs_fix", "issues": []},
    {"status": "ready", "unexpected_verdict_field": True},
    {"status": "ready", "requirement_coverage": [{"requirement_id": "missing-refs",
        "status": "implemented", "detail": "Unsupported implementation claim."}]},
])
def test_census_diagnostic_does_not_relax_existing_review_validation(reply):
    with pytest.raises(ValidationError):
        ImplementationReview.model_validate({**reply, "parameter_census": {"note": "diagnostic only"}})


@pytest.mark.parametrize("status,issues", [
    ("needs_fix", [{"severity": "blocking", "category": "code_error"}]), ("blocked", []),
])
def test_census_diagnostic_preserves_unresolved_verdict(status, issues):
    validated = ImplementationReview.model_validate({"status": status, "issues": issues,
                                                    "parameter_census": {"note": "diagnostic only"}})
    assert validated.model_dump()["status"] == status


def test_paid_census_schema_failure_recovers_with_current_schema_and_refs_at_zero_api(tmp_path, sdk, monkeypatch):
    backend, payload, reply, response = paid_schema_rejection(tmp_path, sdk, monkeypatch)
    preserved = {path: path.read_bytes() for path in [response, tmp_path / "llm_calls.jsonl",
        tmp_path / "api_operation_attempts.jsonl", tmp_path / "cost.json",
        *list((tmp_path / "llm_requests").glob("*.json"))]}
    assert backend(payload) == reply
    assert backend(payload) == reply  # Native recovery is idempotent on another entry.
    assert len(sdk[0]) == 1
    assert all(path.read_bytes() == value for path, value in preserved.items())
    receipts = rows(tmp_path / "role_response_recovery.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["api_calls_added"] == 0 and receipts[0]["cost_preserved"] is True
    assert receipts[0]["original_validation"] == "domain_schema_invalid"
    assert receipts[0]["revalidated_by"] == "current_review_schema_and_source_references"
    assert owner_budget.snapshot(tmp_path)["global_spent_and_occupied"]["llm_calls"] == 1


@pytest.mark.parametrize("problem", ["unknown_extra", "blocking_ready", "stale_source_ref"])
def test_paid_schema_failure_still_rejects_invalid_verdict_or_source_refs(tmp_path, sdk, monkeypatch, problem):
    def mutate(reply):
        if problem == "unknown_extra":
            reply["unexpected_verdict_field"] = True
        elif problem == "blocking_ready":
            reply["issues"] = [{"severity": "blocking", "category": "code_error"}]
        else:
            reply["requirement_coverage"][0]["source_hash"] = "stale-source"
    backend, payload, _, response = paid_schema_rejection(tmp_path, sdk, monkeypatch, mutate=mutate)
    original = response.read_bytes()
    with pytest.raises(llm.LlmUnavailable, match="role_recovery_|recovered_review_evidence_reference_invalid"):
        backend(payload)
    assert len(sdk[0]) == 1 and response.read_bytes() == original
    assert not (tmp_path / "role_response_recovery.jsonl").exists()


def test_other_role_does_not_adopt_domain_schema_invalid_reply(tmp_path, sdk, monkeypatch):
    campaign(tmp_path)
    payload = {"candidate_source_presence": "missing", "current_file_delivery": "absent", "current_file": None}
    reply = {"tool": "read_code", "args": {"path": "reference/parent.py"}}
    sdk[1].append(json.dumps(reply))
    backend = llm.role_backend(tmp_path, "candidate_coder")
    assert backend(payload) == reply
    call = rows(tmp_path / "llm_calls.jsonl")[0]
    response = tmp_path / "llm_responses" / (call["call_id"] + ".json")
    trace = json.loads(response.read_text())
    trace["validation"] = "domain_schema_invalid"
    response.write_text(json.dumps(trace))
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
    original = response.read_bytes()
    with pytest.raises(llm.LlmUnavailable, match="role_recovery_reply_invalid"):
        backend(payload)
    assert len(sdk[0]) == 1 and response.read_bytes() == original
