"""P1-4: real campaign publication/restart with local role transports only."""
from __future__ import annotations

import json
import sqlite3

import pytest

from react_agent.eeg_research.agentic import loop, selected_plan


class Crash(BaseException):
    """A process exit bypasses the native recoverable-error handler."""


def _proposal(pid):
    return {
        "proposal_id": pid, "task_kind": "new_method", "question": "Test mechanism " + pid,
        "task_description": "Design and measure mechanism " + pid,
        "scientific_intent": "Mechanism " + pid, "evaluation_scope": {"phase": "full", "seed": 0},
        "expected_observation": "Paired measurement or native engineering failure",
        "completion_condition": "Publish measured result or native failure", "required_context_refs": [],
        "cost_assessment": {"implementation_risk": "unknown"}, "value_rationale": "Resolve mechanism uncertainty",
    }


def _campaign(tmp_path):
    goal = {
        "goal_id": "publication", "max_training_jobs": 10, "max_llm_calls": 100,
        "max_gpu_seconds": 100, "max_candidates": 4, "allowed_training_actions": ["run_pilot", "run_full"],
        "research_execution": {"version": "single_plan_v1", "target_k": 2, "budget": {"envelopes": {
            "new_method": {"role_calls": {"experiment_designer": 2, "candidate_coder": 6,
                "candidate_reviewer": 2, "result_analyst": 2}, "training_jobs": 2, "gpu_seconds": 40}}}},
    }
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "local_contract",
        "task_type": "eeg_image_retrieval"}, request_id="publication")
    camp = tmp_path / "publication"
    planner_calls, designer_calls = [], []
    def planner(request):
        planner_calls.append(request)
        return {"protocol_version": "research_plan_draft.v1", "proposals": [_proposal("a"), _proposal("b")],
            "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Useful information",
                "implementation_risk": "Unknown"} for rank, pid in enumerate(("a", "b"), 1)],
            "selected_proposal_id": "a", "selection_rationale": "Most useful feasible measurement"}
    def designer(request):
        designer_calls.append(request)
        return {"status": "blocked", "experiment_spec": request["draft"], "missing_inputs": ["Local fixture"]}
    services = {"designer": designer, "implement": lambda *_: pytest.fail("unexpected implementation"),
        "launch": lambda *_: pytest.fail("unexpected GPU job")}
    loop.tick(camp, planner, services=services)
    loop.tick(camp, planner, services=services)
    assert selected_plan.load_plan(camp)["current_step"] == "design"
    return camp, planner, services, planner_calls, designer_calls


def test_decision_before_intent_crash_recovers_through_native_tick(tmp_path, monkeypatch):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    original_plan = selected_plan.load_plan(camp)
    original_write = loop._write
    def crash_after_decision(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if path.parent == camp / "decisions":
            raise Crash("decision persisted before intent")
    with monkeypatch.context() as injected:
        injected.setattr(loop, "_write", crash_after_decision)
        with pytest.raises(Crash):
            loop.tick(camp, planner, services=services)
    decision_path = camp / "decisions" / "d1.json"
    original_bytes = decision_path.read_bytes()
    assert designer_calls == []
    state = loop.tick(camp, planner, services=services)
    assert (state.get("failure") or {}).get("phase") != "recovery", state
    assert len(planner_calls) == len(designer_calls) == 1
    assert decision_path.read_bytes() == original_bytes
    plan = selected_plan.load_plan(camp)
    assert plan["plan_id"] == original_plan["plan_id"]
    assert plan["budget_owner"] == original_plan["budget_owner"]
    assert plan["repair_count"] == original_plan["repair_count"] == 0
    assert len(list((camp / "decisions").glob("d*.json"))) == 1


PUBLICATION_WINDOWS = ["prepare", "decision", "intent", "commit", "plan_projection", "state_projection"]
COMPLETION_WINDOWS = ["native_receipt", "native_state", "completed_receipt", "completed_plan", "completed_state"]


def _inject_crash(camp, monkeypatch, window):
    from react_agent.eeg_research.agentic import decision_receipts
    original_write, original_append = loop._write, selected_plan._append_plan_receipt
    original_native_receipt = decision_receipts.append_receipt
    state_writes = []
    def write(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        is_plan = path.parent == camp / "selected_plans" and path.name.startswith("plan_")
        if path == camp / "campaign_state.json":
            state_writes.append(path)
        if ((window == "prepare" and path.name.endswith(".prepare.json"))
                or (window == "decision" and path.parent == camp / "decisions")
                or (window == "commit" and path.name.endswith(".commit.json"))
                or (window == "plan_projection" and is_plan)
                or (window == "state_projection" and len(state_writes) == 1)
                or (window == "native_state" and path == camp / "campaign_state.json" and len(state_writes) == 2)
                or (window == "completed_plan" and is_plan and any(
                    row["kind"] == "step_completed" for row in payload["step_receipts"]))
                or (window == "completed_state" and path == camp / "campaign_state.json" and len(state_writes) == 3)):
            raise Crash(window)
    def append(_camp, plan, receipt):
        original_append(_camp, plan, receipt)
        if (window == "intent" and receipt["kind"] == "step_intent") or (
                window == "completed_receipt" and receipt["kind"] == "step_completed"):
            raise Crash(window)
    def native_receipt(*args, **kwargs):
        receipt = original_native_receipt(*args, **kwargs)
        if window == "native_receipt" and kwargs.get("source") == "native_step":
            raise Crash(window)
        return receipt
    monkeypatch.setattr(loop, "_write", write)
    monkeypatch.setattr(selected_plan, "_append_plan_receipt", append)
    monkeypatch.setattr(decision_receipts, "append_receipt", native_receipt)


def _owners(camp):
    with sqlite3.connect(camp / "resource_owners.sqlite") as connection:
        return connection.execute("SELECT id, payload FROM owners ORDER BY id").fetchall()


@pytest.mark.parametrize("window", PUBLICATION_WINDOWS + COMPLETION_WINDOWS)
def test_every_persistent_boundary_restarts_same_publication(tmp_path, monkeypatch, window):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    original = selected_plan.load_plan(camp)
    original.update(attempt=2, repair_count=1)
    selected_plan.save_plan(camp, loop.load_state(camp), original)
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, window)
        with pytest.raises(Crash, match=window):
            loop.tick(camp, planner, services=services)
    frozen_files = {path: path.read_bytes() for path in (camp / "selected_plans" / "publications").rglob("*.json")}
    decision_path = camp / "decisions" / "d1.json"
    if decision_path.exists():
        frozen_files[decision_path] = decision_path.read_bytes()
    owners = _owners(camp)
    before_calls = list(designer_calls)
    if window in PUBLICATION_WINDOWS:
        assert designer_calls == []
    # The native read-only recovery entry publishes the frozen intent but runs no action.
    recovered = loop.align_interrupt(camp)
    assert recovered.get("status") != "blocked", recovered
    assert designer_calls == before_calls and len(planner_calls) == 1
    assert _owners(camp) == owners
    plan = selected_plan.load_plan(camp)
    for key in ("plan_id", "budget_owner", "proposal", "repair_count", "attempt", "current_step"):
        assert plan[key] == original[key]
    assert len([row for row in plan["step_receipts"] if row["kind"] == "step_intent"]) == 1
    assert len(list((camp / "selected_plans" / "publications").rglob("*.prepare.json"))) == 1
    assert len(list((camp / "selected_plans" / "publications").rglob("*.commit.json"))) == 1
    loop.align_interrupt(camp)
    assert _owners(camp) == owners
    state = loop.tick(camp, planner, services=services)
    assert (state.get("failure") or {}).get("phase") != "recovery", state
    assert len(planner_calls) == len(designer_calls) == 1
    assert state.get("experiment_ref") and state["experiment"]["role_status"] == "blocked"
    assert len(list((camp / "decisions").glob("d*.json"))) == 1
    assert all(path.read_bytes() == body for path, body in frozen_files.items())
    plan = selected_plan.load_plan(camp)
    assert plan["plan_id"] == original["plan_id"] and plan["budget_owner"] == original["budget_owner"]
    assert len([row for row in plan["step_receipts"] if row["kind"] == "step_completed"]) == 1


def test_uncommitted_intent_cannot_dispatch_native_action(tmp_path, monkeypatch):
    camp, planner, services, _, designer_calls = _campaign(tmp_path)
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, "intent")
        with pytest.raises(Crash):
            loop.tick(camp, planner, services=services)
    body = json.loads((camp / "decisions" / "d1.json").read_text())
    decision = {**body["decision"], "raw": body["raw"], "execution": body["execution"]}
    with pytest.raises(ValueError, match="publication_not_committed"):
        loop._apply_action(camp, loop.load_state(camp), decision["action"], decision, None, services, loop.observation(camp))
    assert designer_calls == []


@pytest.mark.parametrize("changed", ["prepare", "decision", "intent", "commit", "owner", "proposal"])
def test_changed_publication_never_reauthorizes_or_refreshes_digest(tmp_path, monkeypatch, changed):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, "state_projection")
        with pytest.raises(Crash):
            loop.tick(camp, planner, services=services)
    plan = selected_plan.load_plan(camp)
    publication_root = camp / "selected_plans" / "publications" / plan["plan_id"]
    path = {"prepare": publication_root / "d1.prepare.json", "decision": camp / "decisions" / "d1.json",
        "commit": publication_root / "d1.commit.json", "owner": camp / "selected_plans" / (plan["plan_id"] + ".json"),
        "proposal": camp / "selected_plans" / (plan["plan_id"] + ".json")}.get(changed)
    if changed == "intent":
        path = camp / "selected_plans" / (plan["plan_id"] + ".receipts.jsonl")
        receipts = [json.loads(line) for line in path.read_text().splitlines()]
        receipts[-1]["execution_digest"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    else:
        row = json.loads(path.read_text())
        if changed == "prepare": row["binding"]["attempt"] += 1
        elif changed == "decision": row["execution"]["decision_rationale"] = "Changed payload"
        elif changed == "commit": row["decision_sha256"] = "0" * 64
        elif changed == "owner": row["budget_owner"] = "different_owner"
        else: row["proposal"]["scientific_intent"] = "Different science"
        path.write_text(json.dumps(row))
    frozen = path.read_bytes()
    state = loop.tick(camp, planner, services=services)
    assert state["status"] == "blocked" and state["failure"]["phase"] == "recovery"
    assert path.read_bytes() == frozen
    assert designer_calls == [] and len(planner_calls) == 1


@pytest.mark.parametrize("window", ["decision", "intent", "commit", "plan_projection", "state_projection"])
def test_recovery_itself_can_restart_at_each_remaining_write(tmp_path, monkeypatch, window):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, "prepare")
        with pytest.raises(Crash):
            loop.tick(camp, planner, services=services)
    prepared = next((camp / "selected_plans" / "publications").rglob("*.prepare.json"))
    original = prepared.read_bytes()
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, window)
        with pytest.raises(Crash, match=window):
            loop.align_interrupt(camp)
    owners = _owners(camp)
    state = loop.align_interrupt(camp)
    assert state["status"] != "blocked", state
    assert _owners(camp) == owners and prepared.read_bytes() == original
    assert designer_calls == [] and len(planner_calls) == 1
    state = loop.tick(camp, planner, services=services)
    assert state["status"] == "created" and len(planner_calls) == len(designer_calls) == 1


def _legacy_decision(camp, *, intent):
    from react_agent.eeg_research.agentic.artifacts import request_digest
    plan, state = selected_plan.load_plan(camp), loop.load_state(camp)
    payload = {"action": "design_experiment", "target_id": None,
        "decision_rationale": "Legacy internal selected step", "required_artifact_refs": [],
        "hypothesis_draft": {"mechanism": plan["proposal"]["scientific_intent"], "question": plan["proposal"]["question"]},
        "experiment_draft": {"intervention": plan["proposal"]["scientific_intent"], "parent_candidate_id": "baseline",
            "control_candidate_id": "baseline", "initial_fidelity": "full"}}
    record = {"decision_id": "d1", "origin_plan_id": plan["plan_id"], "campaign_id": state["goal_id"],
        "action": "design_experiment", "ok": True, "executed": False,
        "decision_schema_version": "eeg_research.decision.plan_step.v1",
        "raw_digest": request_digest(request=payload), "execution_digest": request_digest(request=payload)}
    path = camp / "decisions" / "d1.json"
    loop._write(path, {"decision": record, "raw": payload, "execution": payload}, exclusive=True)
    if intent:
        selected_plan._receipt(camp, plan, "step_intent", step="design", attempt=plan["attempt"],
            action=record["action"], decision_id="d1", execution_digest=record["execution_digest"])
        state["decisions"].append(record)
        selected_plan.save_plan(camp, state, plan)
    return path


def test_legacy_committed_intent_keeps_original_bytes_and_execution(tmp_path):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    path = _legacy_decision(camp, intent=True)
    original = path.read_bytes()
    state = loop.tick(camp, planner, services=services)
    assert state["status"] == "created" and len(planner_calls) == len(designer_calls) == 1
    assert path.read_bytes() == original
    assert not (camp / "selected_plans" / "publications").exists()


def test_legacy_orphan_has_no_durable_authority_to_invent_an_intent(tmp_path):
    camp, planner, services, planner_calls, designer_calls = _campaign(tmp_path)
    path = _legacy_decision(camp, intent=False)
    original = path.read_bytes()
    state = loop.tick(camp, planner, services=services)
    assert state["status"] == "blocked" and state["failure"]["phase"] == "recovery"
    assert state["detail"] == "selected_plan_internal_history_without_intent"
    assert path.read_bytes() == original and designer_calls == [] and len(planner_calls) == 1
    assert not (camp / "selected_plans" / "publications").exists()


def test_closed_cancelled_legacy_campaign_tick_keeps_frozen_files(tmp_path):
    camp, planner, services, _, _ = _campaign(tmp_path)
    _legacy_decision(camp, intent=True)
    loop.tick(camp, planner, services=services)
    loop.request_control(camp, "stop")
    state = loop.tick(camp, planner, services=services)
    assert state["status"] == "cancelled"
    frozen = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"}}
    state = loop.tick(camp, lambda *_: pytest.fail("cancelled Planner"), services={})
    assert state["status"] == "cancelled"
    assert all(path.read_bytes() == body for path, body in frozen.items())


def test_executed_successful_design_receipt_restores_approval_without_repeating_action(tmp_path, monkeypatch):
    camp, planner, services, planner_calls, _ = _campaign(tmp_path)
    designs = []
    def designer(request):
        designs.append(request)
        return {"status": "completed", "experiment_spec": {**request["draft"], "required_capability_ids": ["build_encoder"]}}
    services["designer"] = designer
    with monkeypatch.context() as injected:
        _inject_crash(camp, injected, "native_receipt")
        with pytest.raises(Crash):
            loop.tick(camp, planner, services=services)
    assert not loop.load_state(camp).get("experiment_ref")
    owners = _owners(camp)
    state = loop.tick(camp, planner, services=services)
    assert len(planner_calls) == len(designs) == 1
    assert _owners(camp) == owners
    assert state["experiment"]["approval_record"]["status"] == "approved"
    plan = selected_plan.load_plan(camp)
    assert plan["current_step"] == "implement" and plan["approved_scientific_digest"]
    assert plan["candidate_id"] and len(state["candidate_admissions"]) == 1
