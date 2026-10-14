"""Native selected experiment with actual SDK MockTransport and actual CPU jobs.

Provider replies alone are synthetic. Approval, patches, checks, source reviews,
matched training, settlement, exact-card capture and the next Planner are native.
No external API/GPU/real EEG data is used or claimed.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import socket

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import closed_loop_evaluation as evaluation
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.fmri.llm.deepseek import DeepSeekBackend

SDK_HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2
SOURCE = '''from react_agent.eeg_research.agentic.baseline import EEGCandidate as BaselineCandidate
from torch import nn

class TemporalMeanEncoder(nn.Module):
    def __init__(self, channels, output_dim):
        super().__init__()
        self.projection = nn.Linear(channels, output_dim)
    def forward(self, eeg):
        return self.projection(eeg.mean(dim=-1))

class EEGCandidate(BaselineCandidate):
    def build_encoder(self, input_spec, model_config=None):
        config = model_config or {}
        return TemporalMeanEncoder(int(input_spec['c_num']), int(config.get('z_dim', 1024)))
'''


def resolve(value, root):
    if isinstance(value, dict):
        if "$context_pointer" in value:
            return resolve(resolve_request_pointer(root, value["$context_pointer"]), root)
        return {key: resolve(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve(child, root) for child in value]
    return value


def proposal(pid, *, kind="new_method", target=None, phase="pilot"):
    return {"proposal_id": pid, "task_kind": kind, "target_candidate_ref": target,
        "question": "Compare synthetic temporal aggregation " + pid,
        "task_description": "Test the declared small synthetic mechanism " + pid,
        "scientific_intent": "temporal_mean" if pid == "mean" else "temporal_max",
        "evaluation_scope": {"phase": phase, "seed": 0}, "expected_observation": "Exact native synthetic feedback",
        "completion_condition": "One bounded observation", "required_context_refs": [], "value_rationale": "Infrastructure coverage"}


def response(role, payload):
    if role == "research_planner":
        cards = payload.get("experiment_observations", {}).get("cards", [])
        if cards:
            target = cards[0]["body"]["identity"]["candidate_id"]
            proposals = [proposal(pid, kind="diagnosis", target=target, phase="diagnostic") for pid in ("compare", "coverage")]
        else:
            proposals = [proposal("mean"), proposal("max")]
        return {"protocol_version": "research_plan_draft.v1", "proposals": proposals,
            "evaluations": [{"proposal_id": item["proposal_id"], "rank": i, "value_rationale": "Synthetic information",
                             "implementation_risk": "bounded"} for i, item in enumerate(proposals, 1)],
            "selected_proposal_id": proposals[0]["proposal_id"], "selection_rationale": "Compare distinct task purposes"}
    if role == "experiment_designer":
        draft = payload["draft"]
        return {"status": "completed", "experiment_spec": {**draft, "status": "draft", "intervention": "temporal_mean",
            "hypothesis": {"statement": "Synthetic temporal means give an executable controlled encoder."},
            "required_capability_ids": ["build_encoder"], "implementation_requirements": [{"requirement_id": "temporal_mean",
                "kind": "implementation", "core": True, "expected_behavior": "Average the temporal axis before linear projection.",
                "code_locations": ["EEGCandidate.build_encoder", "TemporalMeanEncoder.forward"], "verification": "source_review"}]}}
    if role == "candidate_coder":
        if (payload.get("last_check") or {}).get("ok") or any(row.get("tool") == "apply_candidate_patch" and row.get("result", {}).get("ok")
                                                       for row in payload.get("history", [])):
            return {"tool": "finish_patch", "args": {"summary": "Temporal mean before projection; current native check passed."}}
        return {"tool": "apply_candidate_patch", "args": {"path": "extension/eeg_candidate.py", "content": SOURCE,
                "expected_base_hash": (payload.get("current_file") or {}).get("sha256", "")}}
    if role == "candidate_reviewer":
        identity = payload["requirement_evidence_identity"]
        return {"status": "ready", "issues": [], "requirement_coverage": [{"requirement_id": "temporal_mean",
            "status": "implemented", "source_hash": identity["source_hash"], "source_refs": [identity["source_ref"]],
            "check_ref": identity["check_ref"], "check_sha256": identity["check_sha256"],
            "detail": "The current forward uses temporal mean followed by Linear; current check identity matches."}],
            "intervention_coverage": {"principal": {"status": "implemented", "detail": "Temporal mean executes before projection.",
                "source_refs": [identity["source_ref"]], "check_ref": identity["check_ref"]}},
            "review_limits": "Synthetic infrastructure case; no efficacy claim.", "summary_zh": "当前源码与原生检查一致。"}
    if role == "result_analyst":
        return {"hypothesis_assessment": "inconclusive", "scope_limits": ["Random synthetic inputs; not EEG efficacy."],
                "observations": [{"observation": "The native pilot produced a matched controlled measurement."}],
                "summary_zh": "仅验证合成数据闭环，不代表 EEG 效果。"}
    pytest.fail("Unexpected paid maintenance/role: " + role)


@pytest.fixture
def sdk_transport(monkeypatch):
    clients, prepared, wire = [], [], []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-synthetic-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def forbidden_network(*args, **kwargs):
        pytest.fail("Offline closed-loop fixture forbids socket connections and DNS")
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_network)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    def handle(request):
        body = json.loads(request.content)
        wire.append(body)
        current = prepared[-1]
        payload = resolve(current.payload, current.payload)
        reply = response(current.delivery_manifest["role"], payload)
        return SDK_HTTP.Response(200, json={"id": "synthetic-local-response", "object": "chat.completion", "created": 0,
            "model": "synthetic-sdk-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}})
    class MockBackend(DeepSeekBackend):
        closed_loop_sdk_mock = True
        def __init__(self, config):
            super().__init__(config)
            hooks = {key: list(value) for key, value in self._client._client.event_hooks.items()}
            assert self._budget_http_boundary in hooks["request"]
            asyncio.run(self._client.close())
            self._client = AsyncOpenAI(api_key="offline-synthetic-unused", base_url="https://offline.invalid", max_retries=0,
                http_client=SDK_HTTP.AsyncClient(transport=SDK_HTTP.MockTransport(handle), trust_env=False, event_hooks=hooks))
            self._client._platform = "Linux"
            clients.append(self)
        async def complete_prepared_json(self, request, **kwargs):
            prepared.append(request)
            return await super().complete_prepared_json(request, **kwargs)
    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", MockBackend)
    yield prepared, wire
    for client in clients:
        asyncio.run(client._client.close())


def test_entry_requires_explicit_real_api_authorization(tmp_path):
    with pytest.raises(ValueError, match="explicit_real_api_flag"):
        evaluation.run_evaluation(tmp_path / "no-network", real_api=False)
    with pytest.raises(ValueError, match="between_1_and_18"):
        evaluation.run_evaluation(tmp_path / "no-network", real_api=True, max_api_calls=19)
    assert not (tmp_path / "no-network").exists()


@pytest.mark.parametrize("max_api_calls", [16, 18])
def test_real_sdk_mock_and_cpu_selected_experiment_closes_and_hands_back(tmp_path, sdk_transport, max_api_calls):
    import torch
    torch.set_num_threads(1)
    prepared, wire = sdk_transport
    report = evaluation.run_evaluation(tmp_path / "closed-loop", real_api=False, _sdk_mock=True,
        max_api_calls=max_api_calls, max_ticks=2000, wall_timeout=180, poll_seconds=.1)
    assert report["ok"], report["failure"]
    assert report["actual_external_API_attempts"] == report["physical_GPU_jobs"] == report["physical_GPU_seconds"] == 0
    assert 1 <= report["native_API_attempts"] == len(prepared) == len(wire) <= max_api_calls
    assert report["paid_maintenance_API_attempts"] == 0
    assert report["first_plan"]["proposal"]["task_kind"] == "new_method"
    assert report["terminal_plan"]["status"] == "observation_ready" and report["terminal_plan"]["repair_count"] <= 1
    assert report["terminal_original_readback_verified"] and report["terminal_delivered_to_next_Planner"]
    assert report["next_selected_plan"]["current_step"] == "prepare" and not report["second_task_dispatched"]
    assert len(report["jobs"]) == 2 and {row["candidate_id"] for row in report["jobs"]} == {"baseline", report["terminal_plan"]["candidate_id"]}
    assert report["native_slot_seconds"] > 0
    for request, body in zip(prepared, wire):
        assert request.verify() and request.budget_report["ready_budget_ok"]
        assert body["messages"] == list(request.messages)
        assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    from react_agent.eeg_research.agentic import experiment_observations as observations, loop
    camp = Path(report["campaign"])
    state = loop.load_state(camp)
    plan = report["terminal_plan"]
    entry = next(row for row in state["evidence"] if row.get("evidence_id") == plan["measurement_evidence_id"])
    directory = Path(entry["job_dir"])
    raw_metrics = json.loads((directory / "metrics.json").read_text())
    facts = report["terminal_observation"]["native_facts"]
    assert facts["metrics"]["fixed_bank_top1"] == raw_metrics["fixed_bank_top1"]
    assert facts["metrics"]["fixed_bank_top5"] == raw_metrics["fixed_bank_top5"]
    assert facts["paired_delta_pp"] == entry["comparison"]["delta_pp"]
    assert facts["seed"] == 0 and facts["coverage"] == facts["required_fold_count"] == 1
    for key in ("candidate_id", "source_hash", "spec_hash", "execution_fingerprint"):
        assert facts[key] == entry[key]
    store = observations.ExperimentObservationStore(camp)
    saved = store.records(limit=1, observation_id=plan["terminal_observation_id"])[0]
    assert store.registry.load(store.registry.get(saved["source_refs"][0]["source_ref"], role="research_planner"),
                               role="research_planner") == report["terminal_observation"]
    # Fault injections occur only after the successful synthetic run. They
    # prove rejection; they are never used to construct its successful report.
    with evaluation._cpu_environment():
        verified, _refs = observations._single_target_facts(camp, state, plan, entry)
        assert verified == facts
        for key, changed in (("seed", 42), ("source_hash", "f" * 64), ("spec_hash", "e" * 64)):
            forged = {**entry, key: changed}
            with pytest.raises(ValueError):
                observations._single_target_facts(camp, state, plan, forged)
        mutated = [(directory / "metrics.json", {**raw_metrics, "fixed_bank_top1": .125}),
            (camp / "execution_protocol.json", {**json.loads((camp / "execution_protocol.json").read_text()), "seed": 42}),
            (directory / "evaluation_identity.json", {**json.loads((directory / "evaluation_identity.json").read_text()), "validation_image_ids": []}),
            (camp / "candidates" / entry["candidate_id"] / "checks.json", {"ok": False}),
            (camp / "candidates" / entry["candidate_id"] / "review.json", {"status": "blocked"})]
        for path, invalid in mutated:
            original = path.read_bytes()
            try:
                path.write_text(json.dumps(invalid))
                with pytest.raises(ValueError):
                    observations._single_target_facts(camp, state, plan, entry)
                assert store.records(limit=1, observation_id=plan["terminal_observation_id"]) == []
            finally:
                path.write_bytes(original)
        for path in (camp / "candidates" / entry["candidate_id"] / "extension" / "eeg_candidate.py", directory / "last.ckpt"):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b" ")
                with pytest.raises(ValueError):
                    observations._single_target_facts(camp, state, plan, entry)
                assert store.records(limit=1, observation_id=plan["terminal_observation_id"]) == []
            finally:
                path.write_bytes(original)
    assert store.records(limit=1, observation_id=plan["terminal_observation_id"])[0]["body"] == report["terminal_observation"]
