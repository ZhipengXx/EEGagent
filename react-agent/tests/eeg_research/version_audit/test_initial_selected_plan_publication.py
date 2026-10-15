"""Initial SelectedPlan receipt/projection crash recovery at the native tick."""
from __future__ import annotations

import json
import sqlite3

import pytest

from react_agent.eeg_research.agentic import loop, selected_plan


class Crash(BaseException):
    pass


def campaign(tmp_path):
    goal = {"goal_id": "initial_selection", "max_training_jobs": 2, "max_llm_calls": 18,
        "max_gpu_seconds": 100, "max_candidates": 2, "allowed_training_actions": ["run_full"],
        "research_execution": {"version": "single_plan_v1", "target_k": 2}}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "initial_contract",
        "task_type": "eeg_image_retrieval"}, request_id="initial_selection")
    camp, calls = tmp_path / "initial_selection", []
    def planner(request):
        calls.append(request)
        proposals = [{"proposal_id": pid, "task_kind": "diagnosis", "target_candidate_ref": "baseline",
            "question": "Inspect existing comparison " + pid, "task_description": "Inspect recorded facts " + pid,
            "scientific_intent": "Explain current observation " + pid, "evaluation_scope": {"phase": "diagnostic"},
            "expected_observation": "Recorded diagnostic facts", "completion_condition": "One diagnostic observation",
            "required_context_refs": [], "value_rationale": "Reduce uncertainty"} for pid in ("a", "b")]
        return {"protocol_version": "research_plan_draft.v1", "proposals": proposals,
            "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Useful information",
                "implementation_risk": "No new training"} for rank, pid in enumerate(("a", "b"), 1)],
            "selected_proposal_id": "a", "selection_rationale": "Best bounded explanation"}
    return camp, planner, calls


def forbid(*args, **kwargs):
    pytest.fail("Recovery cannot request API, allocate authority or dispatch a job")


def owners(camp):
    with sqlite3.connect((camp / "resource_owners.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        return db.execute("SELECT id,payload FROM owners ORDER BY id").fetchall()


def inject(camp, monkeypatch, window):
    write, append = loop._write, selected_plan._append_plan_receipt
    def boundary(path, value, **kwargs):
        write(path, value, **kwargs)
        if ((window == "prepare" and path.parent == camp / "selected_plans" / "selections"
                and path.name.endswith(".prepare.json"))
                or (window == "plan" and path.parent == camp / "selected_plans" and path.name.endswith(".json"))
                or (window == "state" and path == camp / "campaign_state.json")
                or (window == "commit" and path.parent == camp / "selected_plans" / "selections"
                    and path.name.endswith(".commit.json"))):
            raise Crash(window)
    def receipt(_camp, plan, row):
        append(_camp, plan, row)
        if window == "receipt" and row["kind"] == "selected":
            raise Crash(window)
    monkeypatch.setattr(loop, "_write", boundary)
    monkeypatch.setattr(selected_plan, "_append_plan_receipt", receipt)


def frozen(camp):
    return {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()
        and path.name not in {"campaign_state.json", "events.jsonl"}
        and path.parent != camp / "selected_plans"}


def disable_execution(monkeypatch):
    monkeypatch.setattr(loop, "observation", forbid)
    monkeypatch.setattr("react_agent.eeg_research.agentic.owner_budget.allocate_selected_plan", forbid)


def assert_selection(camp, state, prepared, original_owners, calls):
    plan = selected_plan.load_plan(camp, state)
    source = prepared["plan"]
    assert {key: plan[key] for key in ("originating_round", "originating_proposal", "plan_id", "budget_owner")}
    assert all(plan[key] == source[key] for key in ("originating_round", "originating_proposal", "plan_id", "budget_owner"))
    assert plan["step_receipts"] == [prepared["receipt"]] and plan["repair_count"] == plan["attempt"] == 0
    assert owners(camp) == original_owners and len(calls) == 1
    assert state["llm_calls"] == state["training_jobs"] == 0
    assert not list((camp / "jobs").glob("*/launch_intent.json"))
    assert not (camp / "llm_calls.jsonl").exists() and not (camp / "llm_api_intents.jsonl").exists()
    return plan


def test_receipt_before_projection_exit_keeps_original_receipt(tmp_path, monkeypatch):
    camp, planner, calls = campaign(tmp_path)
    original = selected_plan._append_plan_receipt
    def crash_after_selected(_camp, plan, row):
        original(_camp, plan, row)
        if row["kind"] == "selected":
            raise Crash("selected receipt persisted")
    with monkeypatch.context() as failure:
        failure.setattr(selected_plan, "_append_plan_receipt", crash_after_selected)
        with pytest.raises(Crash):
            loop.tick(camp, planner)
    journal = next((camp / "selected_plans").glob("*.receipts.jsonl"))
    original_bytes, original_owners = journal.read_bytes(), owners(camp)
    receipt = json.loads(original_bytes)
    assert not (camp / "selected_plans" / (receipt["plan_id"] + ".json")).exists()
    recovered = loop.tick(camp, forbid, services={"launch": forbid, "designer": forbid})
    plan = selected_plan.load_plan(camp, recovered)
    assert plan["plan_id"] == receipt["plan_id"] and plan["budget_owner"] == receipt["budget_owner"]
    assert plan["step_receipts"] == [receipt] and plan["current_step"] == "prepare"
    assert journal.read_bytes() == original_bytes and owners(camp) == original_owners
    assert len(calls) == 1 and recovered["llm_calls"] == recovered["training_jobs"] == 0


@pytest.mark.parametrize("window", ["prepare", "receipt", "plan", "state", "commit"])
def test_each_initial_submission_boundary_recovers_same_selection(tmp_path, monkeypatch, window):
    camp, planner, calls = campaign(tmp_path)
    with monkeypatch.context() as failure:
        inject(camp, failure, window)
        with pytest.raises(Crash, match=window):
            loop.tick(camp, planner)
    prepared = loop._read(next((camp / "selected_plans" / "selections").glob("*.prepare.json")))
    originals, original_owners = frozen(camp), owners(camp)
    journal = camp / "selected_plans" / (prepared["plan"]["plan_id"] + ".receipts.jsonl")
    original_receipt = journal.read_bytes() if journal.exists() else None
    disable_execution(monkeypatch)
    state = loop.tick(camp, forbid, services={"launch": forbid, "designer": forbid})
    plan = assert_selection(camp, state, prepared, original_owners, calls)
    assert plan["current_step"] == ("diagnose" if window == "commit" else "prepare")
    if original_receipt is not None:
        assert journal.read_bytes() == original_receipt
    for path, content in originals.items():
        assert path.read_bytes() == content, path


@pytest.mark.parametrize("first_exit", ["prepare", "receipt", "plan", "state"])
@pytest.mark.parametrize("recovery_exit", ["receipt", "plan", "state", "commit"])
def test_recovery_itself_can_exit_and_resume_without_repeating_selection(tmp_path, monkeypatch, first_exit, recovery_exit):
    camp, planner, calls = campaign(tmp_path)
    with monkeypatch.context() as failure:
        inject(camp, failure, first_exit)
        with pytest.raises(Crash):
            loop.tick(camp, planner)
    prepared = loop._read(next((camp / "selected_plans" / "selections").glob("*.prepare.json")))
    original_owners = owners(camp)
    disable_execution(monkeypatch)
    with monkeypatch.context() as failure:
        inject(camp, failure, recovery_exit)
        with pytest.raises(Crash, match=recovery_exit):
            loop.tick(camp, forbid)
    originals = frozen(camp)
    journal = camp / "selected_plans" / (prepared["plan"]["plan_id"] + ".receipts.jsonl")
    original_receipt = journal.read_bytes()
    state = loop.tick(camp, forbid, services={"launch": forbid, "designer": forbid})
    plan = assert_selection(camp, state, prepared, original_owners, calls)
    assert plan["current_step"] == ("diagnose" if recovery_exit == "commit" else "prepare")
    assert journal.read_bytes() == original_receipt
    for path, content in originals.items():
        assert path.read_bytes() == content, path


def test_legacy_selected_receipt_without_prepare_is_adopted_byte_exact(tmp_path, monkeypatch):
    camp, planner, calls = campaign(tmp_path)
    with monkeypatch.context() as failure:
        inject(camp, failure, "receipt")
        with pytest.raises(Crash):
            loop.tick(camp, planner)
    path = next((camp / "selected_plans" / "selections").glob("*.prepare.json"))
    prepared = loop._read(path)
    path.unlink()  # Isolated fixture represents the 0694fa36 legacy window.
    original_owners = owners(camp)
    journal = camp / "selected_plans" / (prepared["plan"]["plan_id"] + ".receipts.jsonl")
    original_receipt = journal.read_bytes()
    state = loop.tick(camp, forbid, services={"launch": forbid, "designer": forbid})
    assert assert_selection(camp, state, prepared, original_owners, calls)["current_step"] == "prepare"
    assert journal.read_bytes() == original_receipt


def test_write_failure_cannot_add_control_receipts_to_uncommitted_selection(tmp_path, monkeypatch):
    camp, planner, calls = campaign(tmp_path)
    write, fired = loop._write, []
    def fail_after_projection(path, value, **kwargs):
        write(path, value, **kwargs)
        if path.parent == camp / "selected_plans" and path.name.endswith(".json") and not fired:
            fired.append(True)
            raise OSError("Initial projection write interrupted")
    with monkeypatch.context() as failure:
        failure.setattr(loop, "_write", fail_after_projection)
        state = loop.tick(camp, planner)
    assert state["status"] == "blocked"
    prepared = loop._read(next((camp / "selected_plans" / "selections").glob("*.prepare.json")))
    journal = camp / "selected_plans" / (prepared["plan"]["plan_id"] + ".receipts.jsonl")
    assert [json.loads(row) for row in journal.read_text().splitlines()] == [prepared["receipt"]]
    original_owners, original_receipt = owners(camp), journal.read_bytes()
    state["status"] = "created"  # Native resume keeps the same scientific task.
    loop.save_state(camp, state)
    disable_execution(monkeypatch)
    recovered = loop.tick(camp, forbid)
    assert assert_selection(camp, recovered, prepared, original_owners, calls)["current_step"] == "prepare"
    assert journal.read_bytes() == original_receipt
