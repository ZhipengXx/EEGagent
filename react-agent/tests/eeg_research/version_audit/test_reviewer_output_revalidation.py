"""Two actual SDK length failures, then a scoped native 32k Reviewer retry."""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sqlite3
from pathlib import Path

import pytest
from openai import AsyncOpenAI, _base_client

from react_agent.eeg_research.agentic import cli, llm, loop, native_patch, owner_budget, selected_plan, worker
from react_agent.eeg_research.agentic.binding import file_sha256
from react_agent.eeg_research.agentic.context.representations import resolve_request_pointer
from react_agent.eeg_research.agentic.experiment_gate import install_approved_experiment
from react_agent.eeg_research.agentic.identity import ensure_attempt
from react_agent.eeg_research.agentic.interface import candidate_interface
from react_agent.eeg_research.agentic.implementation_semantics import requirements_hash
from react_agent.eeg_research.agentic.loso_study import folds, identity, CAPS
from react_agent.eeg_research.agentic.plan_protocol import scientific_projection
from react_agent.eeg_research.agentic.schemas import SelectedPlan
from react_agent.fmri.llm.deepseek import DeepSeekBackend

HTTP = getattr(_base_client, "httpx", None) or _base_client.httpx2


def resolve(value, root):
    if isinstance(value, dict):
        if "$context_pointer" in value:
            return resolve(resolve_request_pointer(root, value["$context_pointer"]), root)
        return {key: resolve(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve(child, root) for child in value]
    return value


@pytest.fixture
def sdk(monkeypatch, request):
    wire, clients = [], []
    mode = getattr(request, "param", None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-unused")
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "0")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    def forbidden_network(*args, **kwargs):
        pytest.fail("Reviewer recovery regression forbids external network")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden_network)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(native_patch, "_parameter_census", lambda *_: {"scope": "synthetic fixture"})
    def transport(request):
        body = json.loads(request.content)
        wire.append(body)
        completed_review = bool(mode and mode.startswith("semantic_"))
        truncated = not completed_review and len(wire) <= 2
        if truncated:
            assert body["max_tokens"] == 16384
            content = '{"status":"ready","unfinished":"' + "source-reviewed " * 300
        else:
            assert body["max_tokens"] == (8192 if completed_review else 32768)
            payload = json.loads(body["messages"][1]["content"])
            payload = resolve(payload, payload)
            evidence = payload["requirement_evidence_identity"]
            supplied_source_hash = "wrong_current_source" if mode == "bad_reference" else evidence["source_hash"]
            supplied_reference = payload["training_runtime_source"][0]["source_ref"] if mode == "runtime_reference" else evidence["source_ref"]
            verdict = {"status": "ready", "issues": [], "review_limits": "Synthetic fixture only",
                "summary_zh": "原生身份与当前源码检查一致。", "intervention_coverage": {"principal": {"status": "implemented",
                    "detail": "Current checked source implements the declared encoder.", "source_refs": [evidence["source_ref"]],
                    "check_ref": evidence["check_ref"]}}, "requirement_coverage": [{"requirement_id": "encoder",
                    "status": "implemented", "source_hash": supplied_source_hash, "source_refs": [supplied_reference],
                    "check_ref": evidence["check_ref"], "check_sha256": evidence["check_sha256"], "detail": "Declared source path exists."}]}
            if completed_review and (len(wire) == 1 or mode == "semantic_blocked"):
                verdict.update(status="needs_fix" if len(wire) == 1 else "blocked",
                    issues=[{"kind": "framework_issue", "severity": "blocking",
                             "detail": "Synthetic independent review still needs evidence."}])
            content = json.dumps(verdict)
        return HTTP.Response(200, json={"id": "offline-review", "object": "chat.completion", "created": 0,
            "model": "synthetic-model", "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop",
                "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 100,
                "completion_tokens": 16384 if truncated else 512, "total_tokens": 16484 if truncated else 612,
                "completion_tokens_details": {"reasoning_tokens": 14000 if truncated else 128}}})
    class MockBackend(DeepSeekBackend):
        def __init__(self, config):
            super().__init__(config)
            hooks = {key: list(value) for key, value in self._client._client.event_hooks.items()}
            asyncio.run(self._client.close())
            self._client = AsyncOpenAI(api_key="offline-unused", base_url="https://offline.invalid", max_retries=0,
                http_client=HTTP.AsyncClient(transport=HTTP.MockTransport(transport), trust_env=False, event_hooks=hooks))
            self._client._platform = "Linux"
            clients.append(self)
    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", MockBackend)
    yield wire
    for client in clients:
        asyncio.run(client._client.close())


def fixture_campaign(tmp_path, *, requirement=None, completed_review=False):
    camp = tmp_path / "review_output"
    camp.mkdir()
    goal = {"goal_id": camp.name, "evaluation_mode": "loso_method_search", "evaluation_schedule": "full_only",
        "max_llm_calls": 18, "max_training_jobs": 20, "max_gpu_seconds": 100, "max_candidates": 2,
        "allowed_training_actions": ["run_full"], "coding_recovery": {}, "require_audit_before_completion": False,
        "research_execution": {"version": "single_plan_v1", "budget": {"envelopes": {"new_method": {
            "role_calls": {"experiment_designer": 2, "candidate_coder": 6, "candidate_reviewer": 6, "result_analyst": 2},
            "training_jobs": 2, "gpu_seconds": 20, "engineering_gpu_seconds": 0}}}}}
    if completed_review:
        goal["research_execution"].update(design_contract_version="v2", implementation_review_mode="single_pass")
    loop._write(camp / "goal.json", goal)
    loop._write(camp / "evaluation_contract.json", {"fingerprint": "synthetic_contract", "task_type": "eeg_image_retrieval"})
    loop._write(camp / "cost.json", {"llm_calls": 0, "training_jobs": 0})
    history = tmp_path / "isolated_history"
    loop._write(history / "study_manifest.json", {"synthetic_fixture": True})
    inheritance = {"schema_version": "eeg_research.budget_inheritance.v1", "source_study": str(history),
        "authorization_caps": CAPS, "historical_usage": {"llm_calls": 0, "training_jobs": 0, "gpu_seconds": 0},
        "remaining": CAPS, "dependencies": [{"path": str(history / "study_manifest.json"),
            "sha256": file_sha256(history / "study_manifest.json")}], "accounting": "synthetic_native_history"}
    from react_agent.eeg_research.agentic.budget_inheritance import _legacy_scientific_admission_scope
    inheritance["scientific_admission_history"] = _legacy_scientific_admission_scope(history)
    inheritance["receipt_hash"] = identity(inheritance)
    loop._write(camp / "budget_inheritance.json", inheritance)
    goal.update(inherited_scientific_admission_receipt_hash=inheritance["receipt_hash"], inherited_scientific_admissions=[])
    loop._write(camp / "goal.json", goal)
    state = {"goal_id": camp.name, "status": "planning", "evaluation_mode": "loso_method_search", "evaluation_schedule": "full_only",
        "contract_fingerprint": "synthetic_contract", "execution_fingerprint": None, "training_jobs": 0, "llm_calls": 0,
        "max_llm_calls": 18, "max_training_jobs": 20, "max_gpu_seconds": 100, "max_candidates": 2,
        "max_repairs_per_candidate": 2, "candidate_ready": False, "candidate_id": None,
        "evidence": [], "memory": [], "candidates": [],
        "candidate_admissions": [], "decisions": [], "allowed_training_actions": ["run_full"]}
    spec = install_approved_experiment(camp, state, {**({"design_contract_version": "v2"} if completed_review else {}),
        "intervention": "test_encoder", "hypothesis": "Fixture mechanism",
        "parent_candidate_id": "baseline", "control_candidate_id": "baseline", "initial_fidelity": "full",
        "model": {}, "required_capability_ids": ["build_encoder"], "implementation_requirements": [requirement or {"requirement_id": "encoder", "kind": "implementation", "core": True,
            "expected_behavior": "Declared encoder exists.", "code_locations": ["EEGCandidate.build_encoder"], "verification": "source_review"}]}, target_id="c1")
    assert spec["status"] == "approved"
    from react_agent.eeg_research.agentic.candidate_lifecycle import admit
    admit(camp, state, "c1", spec)
    state["candidates"] = [{"candidate_id": "c1", "status": "ready_for_review"}]
    workspace = camp / "candidates/c1"
    (workspace / "extension").mkdir()
    (workspace / "extension/eeg_candidate.py").write_text("from torch import nn\nfrom react_agent.eeg_research.agentic.baseline import EEGCandidate as Base\n\nclass Encoder(nn.Module):\n    def __init__(self, channels):\n        super().__init__()\n        self.project = nn.Linear(channels, 1024)\n    def forward(self, eeg):\n        return self.project(eeg.mean(dim=-1))\n\nclass EEGCandidate(Base):\n    def build_encoder(self, input_spec, model_config=None):\n        return Encoder(input_spec['c_num'])\n")
    loop._write(workspace / "spec.json", {"experiment": spec, "hypothesis": spec["hypothesis"]})
    loop._write(workspace / "input_spec.json", candidate_interface(None))
    if completed_review:
        loop._write(workspace / "implementation_map.json", {"synthetic_fixture": True})
    attempt = ensure_attempt(workspace, "c1")
    attempt["operations"]["review_candidate"] = "original_review_operation"
    loop._write(workspace / "attempt.json", attempt)
    sha = file_sha256(workspace / "extension/eeg_candidate.py")
    loop._write(workspace / "checks.json", {"ok": True, "source_sha256": sha,
        "check_fingerprint": native_patch._check_fingerprint(workspace, None), "synthetic_fixture": True,
        "implementation_semantics": {"requirements_hash": requirements_hash(spec["implementation_requirements"])}})
    loop._write(workspace / "implementation.json", {"candidate_id": "c1", "attempt_id": attempt["attempt_id"],
        "input_hash": sha, "status": "ready_for_review", "steps": 4})
    owner = owner_budget.allocate_selected_plan(camp, plan_id="plan_" + "a" * 32, task_kind="new_method",
        known_cost={"training_jobs": 2, "gpu_seconds": 20}, recovery_limits={"max_engineering_repairs": 2})
    proposal = {"proposal_id": "encoder", "task_kind": "new_method", "question": "Test fixture mechanism",
        "task_description": "Implement the declared fixture", "scientific_intent": "test_encoder",
        "evaluation_scope": {"phase": "full", "seed": 0}, "expected_observation": "One bounded observation",
        "completion_condition": "Publish observation", "required_context_refs": [], "value_rationale": "Fixture"}
    plan = SelectedPlan(plan_id=owner["binding"]["plan_id"], originating_round="fixture_round", originating_proposal="encoder",
        proposal=proposal, scientific_intent_digest=identity("test_encoder"), candidate_id="c1", budget_owner=owner["owner_id"],
        recovery_limits={"max_engineering_repairs": 2}, current_step="implement", status="implementing",
        spec_hash=spec["spec_hash"], approved_scientific_digest=identity(scientific_projection(spec))).model_dump()
    selected_plan.save_plan(camp, state, plan)
    manifest = {"folds": [{**row, "seed": 0} for row in folds()], "gpu_seconds_estimate_per_suite": 20,
        "benchmark_permission": {"authorized": True, "data_role": "method_development_benchmark",
            "purpose": "complete_method_suite_feedback", "partial_feedback": False, "training_access": False,
            "checkpoint_selection": "source_subject_development_only"}}
    manifest["manifest_hash"] = identity(manifest)
    loop._write(camp / "method_search_manifest.json", manifest)
    reviewer = llm.role_backend(camp, "candidate_reviewer")
    with owner_budget.budget_owner(camp, plan["budget_owner"]):
        verdict = worker._review_or_reuse(camp, workspace, {"experiment": spec}, {}, reviewer, state)
    if completed_review:
        assert verdict["status"] == "needs_fix" and not verdict["format_failed"]
        state["candidates"][0].update(status="review_needs_fix", review="needs_fix")
        worker._route_v2_repairs(camp, state, "c1", verdict)
    else:
        assert verdict is None
    loop.save_state(camp, state)
    selected_plan.reflect_control(camp, state)
    return camp


def records(camp, name):
    return [json.loads(line) for line in (camp / name).read_text().splitlines() if line]


@pytest.mark.parametrize("sdk", ["semantic_ready", "semantic_blocked"], indirect=True)
def test_completed_v2_review_gets_one_independent_native_review(tmp_path, sdk, monkeypatch):
    camp = fixture_campaign(tmp_path, completed_review=True)
    workspace = camp / "candidates/c1"
    # Record the actual failed selected-task step before the explicit re-review,
    # including its immutable decision publication. A blocked action remains
    # unexecuted; revalidation must preserve that historical pending decision.
    state = loop.load_state(camp)
    state.update(status="created")
    for key in ("failure", "detail", "framework_issue_task"):
        state.pop(key, None)
    state["candidates"][0].update(status="ready_for_review")
    plan = selected_plan.load_plan(camp, state)
    plan["status"] = "implementing"
    selected_plan.save_plan(camp, state, plan)
    def observation(_):
        current = loop._read(camp / "campaign_state.json")
        return {"goal": loop._read(camp / "goal.json"), "available_actions": ["implement_candidate"],
            "candidate_ready": current.get("candidate_ready"), "candidate_id": "c1",
            "candidates": current["candidates"], "contract_fingerprint": "synthetic_contract",
            "experiment": current["experiment"], "implementation_target_id": "c1"}
    monkeypatch.setattr(loop, "observation", observation)
    def initial_review(camp_dir, current):
        verdict = worker._review_or_reuse(camp_dir, workspace, loop._read(workspace / "spec.json"), {},
                                         llm.role_backend(camp_dir, "candidate_reviewer"), current)
        current["candidates"][0].update(status="review_needs_fix", review=verdict["status"])
        worker._route_v2_repairs(camp_dir, current, "c1", verdict)
    assert not selected_plan._native_step(camp, state, plan, "implement_candidate", {"implement": initial_review})
    loop.save_state(camp, state)
    selected_plan.reflect_control(camp, state)
    old_decisions = {p: p.read_bytes() for p in (camp / "decisions").glob("*.json")}
    assert old_decisions and state["decisions"][-1]["executed"] is False
    from react_agent.eeg_research.agentic.decision_receipts import LEDGER
    if (camp / LEDGER).is_file():
        old_decisions[camp / LEDGER] = (camp / LEDGER).read_bytes()
    previous = (workspace / "review.json").read_bytes()
    previous_sha = file_sha256(workspace / "review.json")
    old_calls = records(camp, "llm_calls.jsonl")
    frozen = {p: p.read_bytes() for p in [camp / "goal.json", workspace / "spec.json", workspace / "checks.json",
        workspace / "attempt.json", workspace / "implementation.json", workspace / "implementation_map.json",
        workspace / "extension/eeg_candidate.py"]}
    def forbidden(*args, **kwargs):
        pytest.fail("Independent re-review cannot run Coder, checks, training or rematerialize source")
    native_materialize = worker.materialize
    monkeypatch.setattr(worker, "implement", forbidden)
    monkeypatch.setattr(worker, "materialize", forbidden)
    monkeypatch.setattr(native_patch, "run_candidate_check", forbidden)
    monkeypatch.setattr(worker.jobs, "start_job", forbidden)
    args = ["revalidate-review", "--root", str(tmp_path), "--campaign", camp.name, "--candidate", "c1",
            "--retry-reason", "Independently reconsider the completed semantic review after a prompt fix."]
    assert cli.main(args) == 0
    calls = records(camp, "llm_calls.jsonl")
    assert len(calls) == len(sdk) == 2 and calls[:1] == old_calls
    assert calls[-1]["operation_id"] != old_calls[0]["operation_id"]
    prompt_sha = hashlib.sha256(sdk[-1]["messages"][0]["content"].encode()).hexdigest()
    assert calls[-1]["operation_id"] == "review_revalidation_" + identity({
        "prior_review_sha256": previous_sha, "prompt_sha256": prompt_sha})[:32]
    assert calls[-1]["budget_owner"] == old_calls[0]["budget_owner"]
    assert calls[-1]["application_attempt"] == 1
    assert owner_budget.snapshot(camp)["global_spent_and_occupied"]["llm_calls"] == 2
    assert records(camp, "api_operation_attempts.jsonl")[-1]["max_operation_attempts"] == 1
    assert (workspace / "review_history" / (previous_sha + ".json")).read_bytes() == previous
    state = loop.load_state(camp)
    verdict = loop._read(workspace / "review.json")
    assert state["candidate_ready"] is (verdict["status"] == "ready")
    assert state["training_jobs"] == 0
    if verdict["status"] == "ready":
        assert "framework_issue_task" not in state
    else:
        assert state["status"] == "blocked" and state["framework_issue_task"]["candidate_id"] == "c1"
    assert cli.main(args) == 0
    assert len(sdk) == 2 and records(camp, "llm_calls.jsonl") == calls
    for path, body in old_decisions.items():
        assert path.read_bytes() == body
    for path, body in frozen.items():
        assert path.read_bytes() == body, path
    if verdict["status"] == "ready":
        receipts = selected_plan.load_plan(camp)["step_receipts"]
        assert not any(x["kind"] == "step_completed" for x in receipts)
        monkeypatch.setattr(worker, "materialize", native_materialize)
        resumed = loop.tick(camp, forbidden, services=worker.build_services(camp))
        adopted = selected_plan.load_plan(camp, resumed)
        assert adopted["current_step"] == "measure" and adopted["source_hash"] == file_sha256(workspace / "extension/eeg_candidate.py")
        assert adopted["repair_count"] == 0 and adopted["attempt"] == 1
        assert not resumed.get("framework_issue_task") and len(sdk) == 2
        assert loop._read(workspace / "spec.json")["experiment"]["spec_hash"] == verdict["approved_spec_hash"]
        assert (workspace / "extension/eeg_candidate.py").read_bytes() == frozen[workspace / "extension/eeg_candidate.py"]
        assert (workspace / "checks.json").read_bytes() == frozen[workspace / "checks.json"]


@pytest.mark.parametrize("relative_root", [False, True])
def test_cli_revalidation_raises_only_reviewer_output_and_preserves_original_scope(tmp_path, sdk, monkeypatch, relative_root):
    camp = fixture_campaign(tmp_path)
    state = loop.load_state(camp)
    assert state["status"] == "blocked" and state["failure"]["error_type"] == "output_truncated"
    old_calls = records(camp, "llm_calls.jsonl")
    assert len(old_calls) == 2 and all(row["transport_outcome"] == "confirmed_response" for row in old_calls)
    keep = [camp / "goal.json", camp / "method_search_manifest.json", camp / "candidates/c1/spec.json",
        camp / "candidates/c1/checks.json", camp / "candidates/c1/attempt.json", camp / "candidates/c1/extension/eeg_candidate.py",
        *list((camp / "llm_responses").glob("*.json")), *list((camp / "llm_requests").glob("*.json"))]
    originals = {path: path.read_bytes() for path in keep}
    original_plan = selected_plan.load_plan(camp)
    if relative_root:
        monkeypatch.chdir(tmp_path)
    args = ["revalidate-review", "--root", "." if relative_root else str(tmp_path), "--campaign", camp.name, "--candidate", "c1",
        "--retry-reason", "Two confirmed Reviewer responses exhausted 16k output.", "--role-override", "candidate_reviewer.max_tokens=32768"]
    assert cli.main(args) == 0
    new_calls = records(camp, "llm_calls.jsonl")
    assert new_calls[:2] == old_calls and len(new_calls) == len(sdk) == 3
    assert new_calls[-1]["operation_id"] != old_calls[0]["operation_id"]
    assert new_calls[-1]["budget_owner"] == old_calls[0]["budget_owner"]
    assert new_calls[-1]["role_model_config"]["max_tokens"] == sdk[-1]["max_tokens"] == 32768
    state = loop.load_state(camp)
    assert state["candidate_ready"] and state["candidate_id"] == "c1" and state["training_jobs"] == 0
    assert selected_plan.load_plan(camp)["plan_id"] == original_plan["plan_id"]
    assert selected_plan.load_plan(camp)["repair_count"] == original_plan["repair_count"] == 0
    for path, body in originals.items():
        assert path.read_bytes() == body, path
    # Ordinary Goal routing can reuse only the newly authorized checked review.
    reviewer = llm.role_backend(camp, "candidate_reviewer")
    workspace = camp / "candidates/c1"
    verdict = worker._review_or_reuse(camp, workspace, loop._read(workspace / "spec.json"), {}, reviewer, state)
    assert verdict["status"] == "ready" and len(sdk) == 3
    # The official tick completes the selected implementation without buying
    # another review, resetting its repair count or dispatching measurement.
    monkeypatch.setattr(loop, "observation", lambda _: {"goal": loop._read(camp / "goal.json"),
        "available_actions": ["implement_candidate"], "candidate_ready": True, "candidate_id": "c1",
        "candidates": state["candidates"], "contract_fingerprint": "synthetic_contract",
        "experiment": state["experiment"], "implementation_target_id": "c1"})
    def forbid(*args, **kwargs):
        pytest.fail("Resume cannot request Planner or launch measurement in this tick")
    resumed = loop.tick(camp, forbid, services=worker.build_services(camp))
    plan = selected_plan.load_plan(camp, resumed)
    assert plan["plan_id"] == original_plan["plan_id"] and plan["budget_owner"] == original_plan["budget_owner"]
    assert plan["current_step"] == "measure" and plan["repair_count"] == 0
    assert len(sdk) == 3 and resumed["training_jobs"] == 0
    for path in (camp / "goal.json", workspace / "checks.json", workspace / "extension/eeg_candidate.py",
            *list((camp / "llm_responses").glob(old_calls[0]["call_id"] + ".json")),
            *list((camp / "llm_responses").glob(old_calls[1]["call_id"] + ".json"))):
        assert path.read_bytes() == originals[path]


@pytest.mark.parametrize("value", ["candidate_reviewer.model=another", "candidate_reviewer.thinking=false",
    "candidate_reviewer.max_tokens=65536", "candidate_reviewer.max_tokens=32768.0", "candidate_coder.max_tokens=32768"])
def test_cli_refuses_every_override_except_scoped_32k_before_campaign_access(tmp_path, value):
    assert cli.main(["revalidate-review", "--root", str(tmp_path), "--campaign", "missing", "--role-override", value]) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tamper", ["failure_label", "request", "reply", "source", "check", "settlement", "claim_role", "paid_count", "owner"])
def test_failed_label_or_changed_native_proof_cannot_buy_revalidation(tmp_path, sdk, tamper):
    camp = fixture_campaign(tmp_path)
    if tamper == "failure_label":
        state = loop.load_state(camp)
        state["failure"]["phase"] = "experiment_designer"
        loop.save_state(camp, state)
    elif tamper in {"request", "reply"}:
        row = records(camp, "llm_calls.jsonl")[0]
        path = camp / ("llm_requests" if tamper == "request" else "llm_responses") / (row["call_id"] + ".json")
        data = loop._read(path)
        data["user" if tamper == "request" else "reply"] += "changed" if tamper == "request" else ""
        if tamper == "reply":
            data["reply"] = "changed paid prefix"
        loop._write(path, data)
    elif tamper in {"source", "check"}:
        path = camp / "candidates/c1" / ("extension/eeg_candidate.py" if tamper == "source" else "checks.json")
        path.write_text(path.read_text() + "\n ")
    else:
        row = records(camp, "llm_calls.jsonl")[0]
        with sqlite3.connect(camp / "resource_owners.sqlite") as db:
            table, key = ("claims", row["call_id"]) if tamper != "owner" else ("owners", row["budget_owner"])
            data = json.loads(db.execute("SELECT payload FROM " + table + " WHERE id=?", (key,)).fetchone()[0])
            if tamper == "settlement":
                data["settlement_usage"]["transport_outcome"] = "unknown"
            elif tamper == "claim_role":
                data["role"] = "candidate_coder"
            elif tamper == "paid_count":
                data["actual"]["llm_calls"] = 0
            else:
                data["parent_id"] = "other_plan"
            db.execute("UPDATE " + table + " SET payload=? WHERE id=?", (json.dumps(data), key))
    assert cli.main(["revalidate-review", "--root", str(tmp_path), "--campaign", camp.name, "--candidate", "c1",
        "--retry-reason", "Only verified confirmed output permits this retry", "--role-override", "candidate_reviewer.max_tokens=32768"]) == 2
    assert len(sdk) == 2 and len(records(camp, "llm_calls.jsonl")) == 2
    assert not list((camp / "review_revalidations").glob("*.json"))
    assert not (camp / "candidates/c1/review.json").exists()


@pytest.mark.parametrize("sdk", ["bad_reference"], indirect=True)
def test_32k_retry_keeps_the_native_source_reference_validator(tmp_path, sdk):
    camp = fixture_campaign(tmp_path)
    original_source = (camp / "candidates/c1/extension/eeg_candidate.py").read_bytes()
    assert cli.main(["revalidate-review", "--root", str(tmp_path), "--campaign", camp.name, "--candidate", "c1",
        "--retry-reason", "Confirmed truncated review needs bounded output", "--role-override",
        "candidate_reviewer.max_tokens=32768"]) == 2
    assert len(sdk) == 4  # Exactly two original and two bounded new attempts.
    state = loop.load_state(camp)
    assert not state["candidate_ready"] and state["training_jobs"] == 0
    assert not (camp / "candidates/c1/review.json").exists()
    assert (camp / "candidates/c1/extension/eeg_candidate.py").read_bytes() == original_source
    assert selected_plan.load_plan(camp)["repair_count"] == 0


def failed_paid_runtime_review(tmp_path, monkeypatch):
    """Actual SDK replies cite delivered runtime code rejected by the old gate."""
    from react_agent.eeg_research.agentic import review_context
    camp = fixture_campaign(tmp_path)
    args = ["revalidate-review", "--root", str(tmp_path), "--campaign", camp.name, "--candidate", "c1",
        "--retry-reason", "Replay a native authorized review without another request.",
        "--role-override", "candidate_reviewer.max_tokens=32768"]
    actual = review_context.valid_review_reference
    def candidate_only(reference, request):
        identity = review_context._request_value(request, "requirement_evidence_identity")
        return review_context.valid_source_reference(reference,
            {identity.get("source_ref"), "extension/eeg_candidate.py"},
            review_context._request_value(request, "candidate_source"))
    monkeypatch.setattr(review_context, "valid_review_reference", candidate_only)
    assert cli.main(args) == 2
    monkeypatch.setattr(review_context, "valid_review_reference", actual)
    state = loop.load_state(camp)
    assert state["failure"]["error_type"] == "domain_schema_invalid"
    assert not (camp / "candidates/c1/review.json").exists()
    calls = records(camp, "llm_calls.jsonl")
    assert len(calls) == 4 and all(row["finish_reason"] == "stop" for row in calls[2:])
    assert all(loop._read(camp / "llm_responses" / (row["call_id"] + ".json"))["validation"]
        == "review_evidence_reference_invalid" for row in calls[2:])
    return camp, args


@pytest.mark.parametrize("sdk", ["runtime_reference"], indirect=True)
@pytest.mark.parametrize("crash", ["none", "receipt", "review"])
def test_paid_runtime_reply_replays_original_authorized_operation_and_native_resume(tmp_path, sdk, monkeypatch, crash):
    from react_agent.eeg_research.agentic import review_revalidation
    camp, args = failed_paid_runtime_review(tmp_path, monkeypatch)
    calls = records(camp, "llm_calls.jsonl")
    original_plan = selected_plan.load_plan(camp)
    mutable = {"campaign_state.json", "events.jsonl", "ui_events.jsonl", "worker.lock", "approved_binding.json"}
    frozen = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file() and path.name not in mutable}
    proof = review_revalidation.paid_review_proof(camp, loop.load_state(camp), "c1")
    assert proof["receipt"]["call_id"] == calls[-1]["call_id"] and proof["reply"]["status"] == "ready"
    # The proof reader cannot touch the delivery database or any original file.
    assert not [str(path.relative_to(camp)) for path, value in frozen.items() if path.read_bytes() != value]
    monkeypatch.setenv("EEG_EXTERNAL_API_DISABLED", "1")
    backend = worker.role_backend
    def forbidden(*args, **kwargs):
        pytest.fail("Paid review adoption cannot construct a provider or allocate another request")
    monkeypatch.setattr(worker, "role_backend", forbidden)
    if crash != "none":
        name = "record_paid_review" if crash == "receipt" else "review"
        module = review_revalidation if crash == "receipt" else worker
        native = getattr(module, name)
        def stop_after_publication(*args, **kwargs):
            native(*args, **kwargs)
            raise OSError("synthetic crash after " + crash)
        monkeypatch.setattr(module, name, stop_after_publication)
        assert cli.main(args) == 2
        monkeypatch.setattr(module, name, native)
    assert cli.main(args) == 0
    assert cli.main(args) == 0  # Command restart reuses the native publication.
    state = loop.load_state(camp)
    workspace = camp / "candidates/c1"
    verdict = loop._read(workspace / "review.json")
    assert verdict["status"] == "ready" and verdict["operation_id"] == calls[-1]["operation_id"]
    assert verdict["role_config_hash"] == calls[-1]["role_config_hash"]
    assert verdict["requirement_coverage"] == proof["reply"]["requirement_coverage"]
    assert state["candidate_ready"] and state["training_jobs"] == 0 and len(sdk) == 4
    assert records(camp, "llm_calls.jsonl") == calls
    assert len(list((camp / "review_revalidations").glob("review_output_*.json"))) == 1
    assert not [str(path.relative_to(camp)) for path, value in frozen.items() if path.read_bytes() != value]
    monkeypatch.setattr(worker, "role_backend", backend)
    monkeypatch.setattr(loop, "observation", lambda _: {"goal": loop._read(camp / "goal.json"),
        "available_actions": ["implement_candidate"], "candidate_ready": True, "candidate_id": "c1",
        "candidates": state["candidates"], "contract_fingerprint": "synthetic_contract",
        "experiment": state["experiment"], "implementation_target_id": "c1"})
    resumed = loop.tick(camp, forbidden, services=worker.build_services(camp))
    plan = selected_plan.load_plan(camp, resumed)
    assert plan["plan_id"] == original_plan["plan_id"] and plan["budget_owner"] == original_plan["budget_owner"]
    assert plan["current_step"] == "measure" and plan["repair_count"] == 0
    assert records(camp, "llm_calls.jsonl") == calls and len(sdk) == 4 and resumed["training_jobs"] == 0
    for path in (camp / "goal.json", workspace / "checks.json", workspace / "extension/eeg_candidate.py",
            camp / "resource_owners.sqlite", camp / "cost.json", *list((camp / "llm_responses").glob("*.json")),
            *list((camp / "llm_requests").glob("*.json"))):
        assert path.read_bytes() == frozen[path]


@pytest.mark.parametrize("sdk", ["runtime_reference"], indirect=True)
@pytest.mark.parametrize("tamper", ["reply", "request", "settlement", "operation", "authorization"])
def test_paid_reference_failure_requires_original_authorized_request_reply_and_settlement(tmp_path, sdk, monkeypatch, tamper):
    camp, args = failed_paid_runtime_review(tmp_path, monkeypatch)
    calls = records(camp, "llm_calls.jsonl")
    row = calls[-1]
    if tamper in {"request", "reply"}:
        path = camp / ("llm_requests" if tamper == "request" else "llm_responses") / (row["call_id"] + ".json")
        data = loop._read(path)
        if tamper == "request":
            data["user"] += " "
        else:
            data["reply"]["summary_zh"] += "改"
        loop._write(path, data)
    elif tamper == "settlement":
        with sqlite3.connect(camp / "resource_owners.sqlite") as db:
            data = json.loads(db.execute("SELECT payload FROM claims WHERE id=?", (row["call_id"],)).fetchone()[0])
            data["settlement_usage"]["transport_outcome"] = "unknown"
            db.execute("UPDATE claims SET payload=? WHERE id=?", (json.dumps(data), row["call_id"]))
    elif tamper == "operation":
        calls[-1]["operation_id"] = "different_operation"
        (camp / "llm_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls))
    else:
        path = next((camp / "review_revalidations").glob("review_output_*.json"))
        data = loop._read(path)
        data["plan_id"] = "another_plan"
        loop._write(path, data)
    assert cli.main(args) == 2
    assert len(sdk) == 4 and not (camp / "candidates/c1/review.json").exists()
    assert not list((camp / "review_revalidations").glob("*.recovered.json"))
