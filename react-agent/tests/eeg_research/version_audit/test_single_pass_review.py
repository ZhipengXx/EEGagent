"""Single-pass review uses one native SDK request; coder gets its actual source state."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import coder as coder_tools, llm, native_patch, owner_budget
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


@pytest.fixture
def sdk(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPSEEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def no_network(*args, **kwargs):
        pytest.fail("Single-pass regression attempted real network access")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    wire, clients = [], []
    behavior = {"failure": None}
    def handle(request):
        body = json.loads(request.content)
        wire.append(body)
        payload = json.loads(body["messages"][1]["content"])
        evidence = resolve_request_pointer(payload, "/requirement_evidence_identity")
        review = {"status": "ready", "issues": [], "summary_zh": "当前源码已检查。",
            "requirement_coverage": [{"requirement_id": "encoder", "status": "implemented",
                "source_hash": evidence["source_hash"], "source_refs": [evidence["source_ref"]],
                "check_ref": evidence["check_ref"], "check_sha256": evidence["check_sha256"],
                "detail": "The supplied candidate defines its encoder."}]}
        failure = behavior["failure"] if len(wire) == 1 else None
        finish = "stop"
        if failure == "schema":
            review["status"] = "needs_fix"  # No concrete blocking issue: genuinely invalid.
        elif failure == "reference":
            review["requirement_coverage"][0]["source_hash"] = "wrong-source"
        elif failure == "length":
            finish = "length"
        return HTTP.Response(200, json={"id": "offline-review", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": json.dumps(review)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 512, "total_tokens": 612}})
    class MockBackend(DeepSeekBackend):
        def __init__(self, config):
            super().__init__(config)
            hooks = {key: list(value) for key, value in self._client._client.event_hooks.items()}
            asyncio.run(self._client.close())
            self._client = AsyncOpenAI(api_key="offline-unused", base_url="https://offline.invalid", max_retries=0,
                http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(handle), trust_env=False, event_hooks=hooks))
            self._client._platform = "Linux"
            clients.append(self)
    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", MockBackend)
    yield wire, behavior
    for client in clients:
        asyncio.run(client.aclose())


def campaign(tmp_path, mode):
    execution = {"version": "single_plan_v1"}
    if mode is not None:
        execution["implementation_review_mode"] = mode
    goal = {"max_llm_calls": 8, "research_execution": execution, "memory": {"enabled": False},
        "role_models": {"candidate_reviewer": {"model": "synthetic-review", "thinking": False,
                                              "max_tokens": 8192}}}
    for name, value in (("goal.json", goal), ("campaign_state.json", {"status": "planning"}),
                        ("cost.json", {"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0})):
        (tmp_path / name).write_text(json.dumps(value))
    candidate = tmp_path / "eeg_candidate.py"
    candidate.write_text("class EEGCandidate:\n    def build_encoder(self):\n        return 1\n")
    checks = tmp_path / "checks.json"
    checks.write_text('{"ok": true}')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {"candidate_source": candidate.read_text(), "checks": {"ok": True},
        "requirement_evidence_identity": {"source_ref": str(candidate), "source_hash": sha(candidate),
            "check_ref": str(checks), "check_sha256": sha(checks)}}


@pytest.mark.parametrize("failure", ["schema", "reference", "length"])
def test_single_pass_reviewer_retains_failure_after_one_actual_sdk_request(tmp_path, sdk, failure):
    wire, behavior = sdk
    behavior["failure"] = failure
    payload = campaign(tmp_path, "single_pass")
    backend = llm.role_backend(tmp_path, "candidate_reviewer")
    with pytest.raises(llm.LlmUnavailable, match="output_truncated" if failure == "length" else "domain_schema_invalid"):
        backend(payload)
    calls = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    attempts = [json.loads(line) for line in (tmp_path / "api_operation_attempts.jsonl").read_text().splitlines()]
    assert len(wire) == len(calls) == len(attempts) == 1
    assert attempts[0]["max_operation_attempts"] == 1
    assert calls[0]["application_attempt"] == 1
    assert wire[0]["max_tokens"] == 8192
    assert owner_budget.snapshot(tmp_path)["global_spent_and_occupied"]["llm_calls"] == 1
    assert owner_budget.snapshot(tmp_path)["unknown"] == []
    response = json.loads((tmp_path / "llm_responses" / (calls[0]["call_id"] + ".json")).read_text())
    assert response["validation"] == {"schema": "domain_schema_invalid",
        "reference": "review_evidence_reference_invalid", "length": "output_truncated"}[failure]


def test_single_pass_valid_review_and_native_paid_reuse_require_no_second_request(tmp_path, sdk):
    wire, _ = sdk
    payload = campaign(tmp_path, "single_pass")
    backend = llm.role_backend(tmp_path, "candidate_reviewer")
    first = backend(payload)
    assert first["status"] == "ready"
    assert backend(payload) == first
    assert len(wire) == 1
    assert owner_budget.snapshot(tmp_path)["global_spent_and_occupied"]["llm_calls"] == 1


@pytest.mark.parametrize("mode", [None, "legacy"])
def test_legacy_reviewer_keeps_existing_two_attempt_schema_correction(tmp_path, sdk, mode):
    wire, behavior = sdk
    behavior["failure"] = "schema"
    payload = campaign(tmp_path, mode)
    backend = llm.role_backend(tmp_path, "candidate_reviewer")
    assert backend(payload)["status"] == "ready"
    attempts = [json.loads(line) for line in (tmp_path / "api_operation_attempts.jsonl").read_text().splitlines()]
    assert len(wire) == len(attempts) == 2
    assert all(row["max_operation_attempts"] == 2 for row in attempts)


@pytest.mark.parametrize("source_state", ["missing", "complete", "partial"])
def test_native_coder_request_delivers_actual_initial_source_presence(tmp_path, source_state):
    (tmp_path / "goal.json").write_text(json.dumps({"research_execution": {"version": "single_plan_v1"},
                                                  "memory": {"enabled": False}}))
    workspace = tmp_path / "candidate"
    (workspace / "extension").mkdir(parents=True)
    entry = workspace / "extension/eeg_candidate.py"
    if source_state != "missing":
        source = "class EEGCandidate:\n    pass\n"
        if source_state == "partial":
            source += "# actual source, deliberately requiring pagination\n" * 400
        patch = coder_tools.apply_candidate_patch(workspace, "extension/eeg_candidate.py", source, "")
        assert patch["ok"]
        (workspace / "coder_log.jsonl").write_text(json.dumps({"step": 1,
            "tool": "apply_candidate_patch", "result": patch}) + "\n")
    observed = []
    def coder(payload):
        observed.append(payload)
        return {"tool": "requires_framework_extension", "args": {"reason": "Synthetic fixture observes delivery only."}}
    outcome = native_patch.implement(workspace, {}, coder, max_steps=1 if source_state == "missing" else 2)
    assert outcome["status"] == "implementation_failed"
    request = observed[0]
    assert request["candidate_source_presence"] == ("missing" if source_state == "missing" else "present")
    assert request["current_file_delivery"] == ("absent" if source_state == "missing" else source_state)
    if source_state == "missing":
        assert request["current_file"] is None and not entry.exists()
    else:
        assert request["current_file"]["sha256"] == hashlib.sha256(entry.read_bytes()).hexdigest()
        assert request["current_file"]["truncated"] == (source_state == "partial")
    prepared = ContextService(tmp_path).compile(CompileRequest(tmp_path.name, "candidate_coder", request,
        system=llm.system_prompt("candidate_coder", camp=tmp_path), model="synthetic-coder",
        allow_semantic_compaction=False))
    assert isinstance(prepared, Ready) and prepared.prepared.verify()
    system = prepared.prepared.system
    assert 'candidate_source_presence="missing"' in system
    assert 'current_file_delivery="complete": use inline source/hash without rereading' in system
    from react_agent.eeg_research.agentic.runtime_prompts import runtime_mission
    assert len(runtime_mission("candidate_coder")) < 2500
    wire = json.loads(prepared.prepared.user)
    assert resolve_request_pointer(wire, "/candidate_source_presence") == request["candidate_source_presence"]
    assert resolve_request_pointer(wire, "/current_file_delivery") == request["current_file_delivery"]
