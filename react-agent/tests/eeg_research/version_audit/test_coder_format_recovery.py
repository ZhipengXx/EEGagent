"""A feedback revision permits one fixed child, without resetting old calls."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from react_agent.eeg_research.agentic.api_policy import operation_id
from react_agent.eeg_research.agentic.coder_format_recovery import coder_format_recovery
from react_agent.eeg_research.agentic.context.types import digest

REVISION = "eeg_research.coder_json_feedback.v2"
PARENT = "native_coder:coder_step:3"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def journal(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def failed(tmp_path):
    payload = {"step": 3, "experiment_spec": {"spec_hash": "approved-science"},
               "current_file": None, "history": [{"tool": "list_project_files"}, {"tool": "search_code"}]}
    bound = {"candidate_id": "c1", "attempt_id": "same-attempt", "input_hash": "",
             "input_digest": "approved-input", "operation_id": "native_coder"}
    semantic = operation_id("candidate_coder", payload,
                            {key: value for key, value in bound.items() if key != "operation_id"}, "")
    attempts, calls = [], []
    for number in (1, 2):
        call_id = f"failed-{number}"
        system, user = "Original frozen prompt", json.dumps(payload)
        system_sha, user_sha = (hashlib.sha256(value.encode()).hexdigest() for value in (system, user))
        hashes = {"request_system_sha256": system_sha, "request_user_sha256": user_sha}
        identity = {**{key: value for key, value in bound.items() if key != "operation_id"},
                    "call_id": call_id, "role": "candidate_coder", "prompt_hash": system_sha[:16]}
        attempt = {**identity, **hashes, "operation_id": PARENT, "operation_attempt": number,
                   "max_operation_attempts": 2, "budget_owner": "selected-plan:coder"}
        request = {**identity, "system": system, "user": user,
                   "system_sha256": system_sha, "user_sha256": user_sha,
                   "transport_fields": {"model": "synthetic"}, "role_config_hash": "same-model"}
        raw = '   {"path":"reference/check_entry.py"}</DSML parameter></DSML invoke>'
        response = {**identity, **hashes, "reply": raw, "validation": "format_error"}
        call = {**attempt, "role_config_hash": "same-model", "success": False,
                "error": "DeepSeekParseError", "failure_kind": "format_error", "finish_reason": "stop",
                "transport_outcome": "confirmed_response", "unconfirmed_charge_possible": False,
                "role_request_semantics_digest": semantic, "raw_excerpt": raw,
                "context_request_digest": digest({"system": system, "user": user,
                                                   "transport_fields": request["transport_fields"]})}
        write(tmp_path / "llm_requests" / f"{call_id}.json", request)
        write(tmp_path / "llm_responses" / f"{call_id}.json", response)
        attempts.append(attempt)
        calls.append(call)
    journal(tmp_path / "api_operation_attempts.jsonl", attempts)
    journal(tmp_path / "llm_calls.jsonl", calls)
    workspace = tmp_path / "candidates/c1"
    write(workspace / "coding_recovery.json", {"candidate_id": "c1", "steps_used": 2,
                                               "operation_binding": bound})
    journal(workspace / "coder_log.jsonl", [{"step": 1}, {"step": 2}])
    return tmp_path, payload, bound, calls, attempts


def recover(failed, **kwargs):
    camp, payload, bound, _, _ = failed
    return coder_format_recovery(camp, kwargs.pop("operation", PARENT),
                                 payload=kwargs.pop("payload", payload), bound=kwargs.pop("bound", bound),
                                 revision=kwargs.pop("revision", REVISION))


def test_one_deterministic_child_retains_parent_calls_and_tool_budget(failed):
    camp, _, _, calls, attempts = failed
    original = {path.relative_to(camp): path.read_bytes() for path in camp.rglob("*") if path.is_file()}
    first = recover(failed)
    assert first == recover(failed)
    assert first["operation_id"].startswith(PARENT + ":format_feedback:")
    assert first["parent_operation_id"] == PARENT
    assert first["parent_call_ids"] == ["failed-1", "failed-2"]
    assert first["max_attempts"] == 2 and first["budget_owner"] == "selected-plan:coder"
    assert first["rejected_format"] == calls[-1]["raw_excerpt"]
    assert all(len(item["request_sha256"]) == len(item["response_sha256"]) == 64
               for item in first["provenance"])
    assert original == {path.relative_to(camp): path.read_bytes() for path in camp.rglob("*") if path.is_file()}

    # A restart still selects this same operation for paid-reply adoption. Child
    # failure/success is handled by the normal recovery/attempt counter.
    journal(camp / "llm_calls.jsonl", calls + [{"call_id": "child", "operation_id": first["operation_id"],
                                               "success": True, "transport_outcome": "confirmed_response"}])
    journal(camp / "api_operation_attempts.jsonl", attempts + [{"call_id": "child", "operation_id": first["operation_id"]}])
    assert recover(failed) == first
    assert recover(failed, operation=first["operation_id"]) is None


@pytest.mark.parametrize("change", [
    {"success": True}, {"transport_outcome": "outcome_unknown"}, {"unconfirmed_charge_possible": True},
    {"failure_kind": "output_truncated"}, {"failure_kind": "domain_schema_invalid"},
    {"finish_reason": "length"}, {"error": "TimeoutError"}, {"coder_format_feedback_revision": REVISION},
    {"candidate_id": "different"}, {"attempt_id": "different"}, {"input_digest": "different"},
    {"budget_owner": "another-owner"}, {"role_request_semantics_digest": "different"},
])
def test_only_conclusive_previous_revision_format_failures_qualify(failed, change):
    camp, _, _, calls, _ = failed
    calls[-1].update(change)
    journal(camp / "llm_calls.jsonl", calls)
    assert recover(failed) is None


@pytest.mark.parametrize("change", [
    {"candidate_id": "c2"}, {"candidate_id": "../c1"}, {"attempt_id": "new-attempt"},
    {"input_digest": "new-input"}, {"operation_id": "new-operation"},
])
def test_different_native_binding_cannot_reuse_old_failure(failed, change):
    bound = {**failed[2], **change}
    assert recover(failed, bound=bound) is None


@pytest.mark.parametrize("key,value", [("step", 4), ("current_file", {"sha256": "new-source"}),
                                      ("experiment_spec", {"spec_hash": "changed-science"})])
def test_source_spec_and_step_are_original_business_request(failed, key, value):
    payload = {**failed[1], key: value}
    assert recover(failed, payload=payload) is None


@pytest.mark.parametrize("trace,change", [
    ("llm_requests", {"user": '{"step":3,"experiment_spec":"tampered"}'}),
    ("llm_requests", {"system": "tampered prompt"}),
    ("llm_requests", {"role_config_hash": "other-model"}),
    ("llm_responses", {"reply": '{"tool":"read_code","args":{}}'}),
    ("llm_responses", {"validation": "valid"}),
    ("llm_responses", {"candidate_id": "c2"}),
])
def test_trace_identity_hash_and_rejected_raw_must_match(failed, trace, change):
    path = failed[0] / trace / "failed-2.json"
    value = json.loads(path.read_text())
    write(path, {**value, **change})
    assert recover(failed) is None


@pytest.mark.parametrize("kind", ["missing_response", "unsettled", "one_attempt", "three_attempts", "executed_tool", "counter_changed", "extra_settlement", "extra_intent"])
def test_incomplete_or_already_executed_parent_is_not_recovered(failed, kind):
    camp, _, _, calls, attempts = failed
    if kind == "missing_response":
        (camp / "llm_responses/failed-2.json").unlink()
    elif kind == "unsettled":
        journal(camp / "llm_calls.jsonl", calls[:1])
    elif kind in {"one_attempt", "three_attempts"}:
        changed = attempts[:1] if kind == "one_attempt" else attempts + [copy.deepcopy(attempts[-1])]
        journal(camp / "api_operation_attempts.jsonl", changed)
    elif kind == "executed_tool":
        journal(camp / "candidates/c1/coder_log.jsonl", [{"step": 1}, {"step": 2}, {"step": 3}])
    elif kind == "extra_settlement":
        journal(camp / "llm_calls.jsonl", calls + [{"call_id": "unknown", "operation_id": PARENT,
                                                  "transport_outcome": "outcome_unknown"}])
    elif kind == "extra_intent":
        journal(camp / "llm_api_intents.jsonl", [{"call_id": "unknown", "operation_id": PARENT}])
    else:
        path = camp / "candidates/c1/coding_recovery.json"
        state = json.loads(path.read_text())
        write(path, {**state, "steps_used": 3})
    assert recover(failed) is None
