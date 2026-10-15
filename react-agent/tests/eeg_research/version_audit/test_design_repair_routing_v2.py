"""Native v2 design handoff and repair ownership, without API or training."""
from __future__ import annotations

import copy
import json
import socket

import pytest

from react_agent.eeg_research.agentic import loop, selected_plan, worker
from react_agent.eeg_research.agentic.action_planning import digest
from react_agent.eeg_research.agentic.experiment_gate import resolve_approved_experiment
from react_agent.eeg_research.agentic.identity import ensure_attempt, source_hash
from react_agent.eeg_research.agentic.llm import LlmUnavailable
from react_agent.eeg_research.agentic.plan_protocol import scientific_projection
from react_agent.eeg_research.agentic.run_context import load_approved_binding, persist_approved_binding
from react_agent.eeg_research.agentic.schemas import SelectedPlan


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Repair regression attempted network access")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def campaign(tmp_path, monkeypatch, version="v2", limit=2):
    execution = {"version": "single_plan_v1", "paid_maintenance": "off"}
    if version == "v2":
        execution["design_contract_version"] = "v2"
    goal = {"goal_id": "repair_contract", "max_llm_calls": 200, "max_candidates": 1,
            "max_training_jobs": 2, "max_gpu_seconds": 100, "memory": {"enabled": False},
            "max_repairs_per_candidate": limit, "research_execution": execution}
    loop.create_campaign(tmp_path, goal=goal, contract={"fingerprint": "repair_contract",
                         "task_type": "eeg_image_retrieval"}, request_id="repair_contract")
    camp = tmp_path / "repair_contract"
    (camp / "cost.json").write_text(json.dumps({"llm_calls": 0, "training_jobs": 0, "gpu_seconds_used": 0}))
    task = {"question": "Does temporal attention improve retrieval?", "task_description": "Add attention before pooling",
            "scientific_intent": "Attention before temporal pooling", "evaluation_scope": {"phase": "pilot"},
            "expected_observation": "Retrieval accuracy", "completion_condition": "One pilot result"}
    monkeypatch.setattr(loop, "_design_context", lambda *args: {"selected_research_task": task})
    return camp, loop.load_state(camp), task


def brief():
    return {"hypothesis": "Temporal attention improves retrieval", "principal_intervention": "Attention before pooling",
            "scientific_parameters": {"d_model": 8},
            "hook_changes": {"model": {"mode": "replace", "config": {"d_model": 8}},
                             "objective": {"mode": "inherit"}, "transform": {"mode": "inherit"}},
            "implementation_requirements": [
                {"requirement_id": "attention", "kind": "implementation", "core": True,
                 "expected_behavior": "Apply attention before pooling", "scope": "scientific"},
                {"requirement_id": "config_handling", "kind": "implementation", "core": True,
                 "expected_behavior": "Use the supplied hook config", "scope": "engineering"}]}


def design(camp, state, payload, *, parent="baseline", revision=None):
    draft = ({"design_revision": revision} if revision else
             {"parent_candidate_id": parent, "control_candidate_id": "baseline", "initial_fidelity": "pilot"})
    seen = []
    def designer(request):
        seen.append(request)
        return {"status": "completed", "design_brief": payload, "summary_zh": "设计完成"}
    loop._design_experiment(camp, state, {}, {"experiment_draft": draft}, {"designer": designer})
    return seen


def finding(kind="design_conflict"):
    return {"issue_id": "config_conflict", "kind": kind, "severity": "blocking",
            "requirement_ids": ["config_handling"], "evidence_refs": ["reference:hook_caller"],
            "smallest_correction": "Clarify how the frozen inherited config is consumed",
            "validation_expectation": "The native hook check passes without changing the intervention"}


def source(camp):
    workspace = camp / "candidates/c1"
    entry = workspace / "extension/eeg_candidate.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("class EEGCandidate:\n    pass\n")
    attempt = ensure_attempt(workspace, "c1")
    persist_approved_binding(camp, "c1", loop.load_state(camp).get("experiment") or {},
                             attempt_id=attempt["attempt_id"], source_hash=source_hash(workspace))
    return workspace, entry


def test_designer_receives_effective_parent_configs_and_native_inheritance(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    parent = {"target_id": "parent", "model": {"d_model": 4}, "objective": {"temperature": 0.2},
              "transform": {"noise": 0.03}, "spec_hash": "parent_science"}
    path = camp / "candidates/parent/approved_binding.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(parent))
    calls = design(camp, state, brief(), parent="parent")
    assert calls[0]["design_contract_version"] == "v2"
    assert calls[0]["parent_hook_config"] == parent
    assert state["experiment"]["objective"] == parent["objective"]
    assert state["experiment"]["transform"] == parent["transform"]
    assert state["experiment"]["model"] == {"d_model": 8}
    assert state["experiment"]["implementation_requirements"][0]["code_locations"] == []
    assert len(state["candidate_admissions"]) == 1


def test_engineering_revision_preserves_candidate_source_admission_and_cost(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    loop.save_state(camp, state)
    workspace, entry = source(camp)
    old_attempt = ensure_attempt(workspace, "c1")["attempt_id"]
    old_binding = (workspace / "approved_binding.json").read_bytes()
    old_spec = copy.deepcopy(state["experiment"])
    original_artifact = (camp / "experiments").glob("spec_task_*.json")
    original_artifact = next(original_artifact)
    original_bytes = original_artifact.read_bytes()
    admissions = (camp / "candidate_admissions.jsonl").read_bytes()
    cost = (camp / "cost.json").read_bytes()
    assert worker._route_v2_repairs(camp, state, "c1", {"issues": [finding()]}) == "designer"
    revised = brief()
    revised["implementation_requirements"][1]["expected_behavior"] = "Consume the inherited config exactly as the runtime caller supplies it"
    calls = design(camp, state, revised, revision=state["design_revision_task"])
    assert calls[0]["prior_design_brief"] == old_spec["design_brief"]
    assert state["experiment"]["spec_hash"] != old_spec["spec_hash"]
    assert scientific_projection(state["experiment"]) == scientific_projection(old_spec)
    assert len(state["candidate_admissions"]) == 1
    assert (camp / "candidate_admissions.jsonl").read_bytes() == admissions
    assert (camp / "cost.json").read_bytes() == cost
    assert original_artifact.read_bytes() == original_bytes
    assert entry.read_text() == "class EEGCandidate:\n    pass\n"
    assert ensure_attempt(workspace, "c1")["attempt_id"] != old_attempt
    assert any(path.read_bytes() == old_binding for path in (workspace / "approved_binding_history").glob("*.json"))
    assert state["repair_task"]["candidate_id"] == "c1"
    assert "design_revision_task" not in state
    assigned = resolve_approved_experiment(camp, target_id="c1", action="repair", state=state,
                                           attempt_id=state["repair_task"]["attempt_id"])
    assert assigned["spec_hash"] == state["experiment"]["spec_hash"]


def test_scientific_change_is_replan_and_keeps_current_approved_binding(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    loop.save_state(camp, state)
    workspace, _ = source(camp)
    original = copy.deepcopy(state["experiment"])
    binding = (workspace / "approved_binding.json").read_bytes()
    admissions = (camp / "candidate_admissions.jsonl").read_bytes()
    worker._route_v2_repairs(camp, state, "c1", {"issues": [finding()]})
    revised = brief()
    revised["scientific_parameters"]["d_model"] = 16
    revised["hook_changes"]["model"]["config"]["d_model"] = 16
    design(camp, state, revised, revision=state["design_revision_task"])
    assert state["needs_replan"]["reason"] == "design_revision_changes_scientific_intent"
    assert state["experiment"] == original
    assert (workspace / "approved_binding.json").read_bytes() == binding
    assert (camp / "candidate_admissions.jsonl").read_bytes() == admissions


def test_designer_budget_failure_keeps_revision_pending_and_original_approval(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    original = copy.deepcopy(state["experiment"])
    worker._route_v2_repairs(camp, state, "c1", {"issues": [finding()]})
    revision = copy.deepcopy(state["design_revision_task"])
    def unavailable(*args):
        raise LlmUnavailable("owner_request_unavailable:llm_calls")
    loop._design_experiment(camp, state, {}, {"experiment_draft": {"design_revision": revision}},
                            {"designer": unavailable})
    assert state["experiment"] == original
    assert state["design_revision_task"] == revision
    assert state["design_revision_failure"]["error_type"] == "owner_request_unavailable:llm_calls"
    assert not state["candidate_ready"]


def test_design_revision_after_measurement_does_not_invoke_designer(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    worker._route_v2_repairs(camp, state, "c1", {"issues": [finding()]})
    state["evidence"].append({"candidate_id": "c1", "fidelity": "pilot"})
    with pytest.raises(ValueError, match="design_revision_requires_unmeasured_candidate"):
        loop._design_experiment(camp, state, {}, {"experiment_draft": {"design_revision": state["design_revision_task"]}},
                               {"designer": lambda *args: pytest.fail("Measured candidate cannot be redesigned")})


@pytest.mark.parametrize("kind,owner", [("code_error", "coder"), ("design_conflict", "designer"),
                                        ("framework_issue", "native"), ("scientific_replan", "planner")])
def test_findings_route_without_spending_api_or_resetting_resources(tmp_path, monkeypatch, kind, owner):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    cost = (camp / "cost.json").read_bytes()
    assert worker._route_v2_repairs(camp, state, "c1", {"issues": [finding(kind)]}) == owner
    assert not state["candidate_ready"]
    assert (camp / "cost.json").read_bytes() == cost
    assert not (camp / "llm_api_intents.jsonl").exists()
    if owner == "native":
        assert state["status"] == "blocked"
        assert state["repair_task"] is None
        assert state["framework_issue_task"]["issues"][0]["kind"] == kind


def test_design_conflict_precedes_code_repairs_and_keeps_both_open(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    issues = [finding("code_error"), dict(finding(), issue_id="design_config")]
    assert worker._route_v2_repairs(camp, state, "c1", {"issues": issues}) == "designer"
    assert state["design_revision_task"]["issues"] == issues


def test_unactionable_review_returns_to_native_instead_of_rewriting_code(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    issue = finding("code_error")
    issue.pop("smallest_correction")
    assert worker._route_v2_repairs(camp, state, "c1", {"issues": [issue]}) == "native"
    assert not state.get("repair_task")


def test_native_semantic_counterexample_remains_a_coder_repair(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    native = {"severity": "blocking", "kind": "core_implementation_contradicted",
              "requirement_id": "attention", "expected": "Apply attention before pooling",
              "actual": {"source_refs": ["extension/eeg_candidate.py::forward"]},
              "source_hash": "current_source", "check_ref": "checks.json", "check_sha256": "current_check"}
    assert worker._route_v2_repairs(camp, state, "c1", {"issues": [finding("code_error"), native]}) == "coder"
    assert len(state["repair_task"]["issues"]) == 2
    assert all(issue["smallest_correction"] for issue in state["repair_task"]["issues"])
    assert "framework_issue_task" not in state


def test_worker_adopts_coder_design_request_without_calling_reviewer(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    loop.save_state(camp, state)
    source(camp)
    class Backend:
        model = "offline"
        def bind(self, **fields):
            self.identity = fields
    monkeypatch.setattr(worker, "role_backend", lambda *args: Backend())
    observed = []
    def implement(workspace, spec, coder, **kwargs):
        observed.append(spec)
        return {"status": "needs_design_revision", "detail": finding(), "steps": 1}
    monkeypatch.setattr(worker, "implement", implement)
    monkeypatch.setattr(worker, "review", lambda *args, **kwargs: pytest.fail("A design request must precede review"))
    worker.build_services(camp)["implement"](camp, state)
    assert observed[0]["design_contract_version"] == "v2"
    interface = json.loads((camp / "candidates/c1/input_spec.json").read_text())
    assert interface["design_contract_version"] == "v2"
    assert state["design_revision_task"]["candidate_id"] == "c1"
    assert not state["experiment_failed"] and not state["candidate_ready"]


def test_worker_targeted_review_uses_independent_reviewer_after_repair(tmp_path, monkeypatch):
    camp, state, _ = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    loop.save_state(camp, state)
    source(camp)
    worker._route_v2_repairs(camp, state, "c1", {"status": "needs_fix", "issues": [finding("code_error")]})
    roles = []
    class Backend:
        model = "offline"
        def __init__(self, role):
            self.role = role
        def bind(self, **fields):
            self.identity = fields
    monkeypatch.setattr(worker, "role_backend", lambda _camp, role: Backend(role))
    monkeypatch.setattr(worker, "implement", lambda *args, **kwargs: {"status": "ready_for_review", "steps": 1,
                        "check": {"ok": True}})
    def review(workspace, spec, summary, backend, model, *, identity):
        roles.append(backend.role)
        assert spec["review_mode"] == "targeted"
        assert spec["review_focus"]["previous_issue_ids"] == ["config_conflict"]
        assert identity["approved_spec_hash"] == state["experiment"]["spec_hash"]
        return {"status": "needs_fix", "issues": [finding("code_error")], "summary_zh": "Still needs correction"}
    monkeypatch.setattr(worker, "review", review)
    worker.build_services(camp)["implement"](camp, state)
    assert roles == ["candidate_reviewer"]
    assert not state["candidate_ready"]
    assert state["repair_task"]["issues"][0]["issue_id"] == "config_conflict"


@pytest.mark.parametrize("field", ["approved_spec_hash", "intervention_config_hash", "implementation_map_hash"])
def test_review_cache_requires_current_spec_config_and_map(tmp_path, field):
    workspace = tmp_path / "c1"
    workspace.mkdir()
    review = {"status": "ready", "issues": [], "candidate_id": "c1", "attempt_id": "attempt",
              "phase": "review_candidate", "input_hash": "source", "approved_spec_hash": "spec",
              "intervention_config_hash": "config", "implementation_map_hash": "map"}
    (workspace / "review.json").write_text(json.dumps(review))
    keys = {"candidate_id": "c1", "attempt_id": "attempt", "input_hash": "source",
            "approved_spec_hash": "spec", "intervention_config_hash": "config", "implementation_map_hash": "map"}
    assert worker._stored_review(workspace, **keys)["status"] == "ready"
    assert worker._stored_review(workspace, **{**keys, field: "changed"}) is None
    # Existing v1 receipts do not acquire new mandatory fields.
    assert worker._stored_review(workspace, candidate_id="c1", attempt_id="attempt", input_hash="source")


def test_selected_plan_revision_and_repair_count_once_with_limit_one(tmp_path, monkeypatch):
    camp, state, task = campaign(tmp_path, monkeypatch, limit=1)
    design(camp, state, brief())
    proposal = {"proposal_id": "attention", "task_kind": "new_method", "target_candidate_ref": None,
                **task, "required_context_refs": [], "value_rationale": "Test attention"}
    plan = SelectedPlan(plan_id="plan_" + "a" * 32, originating_round="round", originating_proposal="attention",
                        proposal=proposal, scientific_intent_digest=digest(task["scientific_intent"]),
                        candidate_id="c1", spec_hash=state["experiment"]["spec_hash"],
                        approved_scientific_digest=digest(scientific_projection(state["experiment"])),
                        budget_owner="owner-original", recovery_limits={"max_engineering_repairs": 1},
                        current_step="implement", status="implementing").model_dump()
    selected_plan.save_plan(camp, state, plan)
    workspace, _ = source(camp)
    before_admissions = (camp / "candidate_admissions.jsonl").read_bytes()
    steps = []
    def native_step(_camp, current, current_plan, action, services, runner=None, *, raw=None):
        steps.append(action)
        if action == "implement_candidate":
            worker._route_v2_repairs(camp, current, "c1", {"issues": [finding()]})
        elif action == "design_experiment":
            revised = brief()
            revised["implementation_requirements"][1]["expected_behavior"] = "Read the supplied inherited config"
            design(camp, current, revised, revision=raw["experiment_draft"]["design_revision"])
        elif action == "repair_candidate":
            assert current["repair_task"]["candidate_id"] == "c1"
            current.update(candidate_ready=True, candidate_id="c1", repair_task=None)
        else:
            pytest.fail("Unexpected native step " + action)
        return True
    monkeypatch.setattr(selected_plan, "_native_step", native_step)
    forbid = lambda *args: pytest.fail("Repair may not rerun outer Planner")
    selected_plan.tick_selected_plan(camp, state, forbid, {})
    assert selected_plan.load_plan(camp)["current_step"] == "design_revision"
    selected_plan.tick_selected_plan(camp, state, forbid, {})
    interim = selected_plan.load_plan(camp)
    assert interim["current_step"] == "repair" and interim["repair_count"] == 0
    selected_plan.tick_selected_plan(camp, state, forbid, {})
    final = selected_plan.load_plan(camp)
    assert final["current_step"] == "measure" and final["repair_count"] == 1
    assert final["budget_owner"] == "owner-original" and final["candidate_id"] == "c1"
    assert steps == ["implement_candidate", "design_experiment", "repair_candidate"]
    assert (camp / "candidate_admissions.jsonl").read_bytes() == before_admissions
    assert {row["kind"] for row in final["step_receipts"]} >= {"design_revision_requested", "design_revision_approved"}
    assert final["spec_hash"] == load_approved_binding(camp, "c1")["spec_hash"]


def test_terminal_scientific_replan_releases_old_branch_without_refund(tmp_path, monkeypatch):
    camp, state, task = campaign(tmp_path, monkeypatch)
    design(camp, state, brief())
    state["candidates"] = [{"candidate_id": "c1", "status": "needs_design_revision"}]
    proposal = {"proposal_id": "attention", "task_kind": "new_method", "target_candidate_ref": None,
                **task, "required_context_refs": [], "value_rationale": "Test attention"}
    plan = SelectedPlan(plan_id="plan_" + "b" * 32, originating_round="round", originating_proposal="attention",
                        proposal=proposal, scientific_intent_digest=digest(task["scientific_intent"]),
                        candidate_id="c1", spec_hash=state["experiment"]["spec_hash"],
                        budget_owner="owner-original", recovery_limits={"max_engineering_repairs": 1}).model_dump()
    selected_plan.save_plan(camp, state, plan)
    admissions = (camp / "candidate_admissions.jsonl").read_bytes()
    worker._route_v2_repairs(camp, state, "c1", {"issues": [finding("scientific_replan")]})
    monkeypatch.setattr("react_agent.eeg_research.agentic.owner_budget.finish_selected_plan",
                        lambda *args, **kwargs: {"owner_id": "continuation"})
    selected_plan._finish(camp, state, plan, "needs_replan", detail="scientific intervention needs a new plan")
    assert "needs_replan" not in state
    assert "c1" in state["detached_candidates"]
    assert (camp / "candidate_admissions.jsonl").read_bytes() == admissions
    assert len(state["candidate_admissions"]) == 1
    assert loop.candidate_implementation_target(camp, state) == "c2"
    assert selected_plan.load_plan(camp)["terminal_observation_id"]


def test_real_native_design_revision_dispatch_is_legal_at_candidate_quota(tmp_path, monkeypatch):
    camp, state, task = campaign(tmp_path, monkeypatch, limit=1)
    design(camp, state, brief())
    state["candidates"] = [{"candidate_id": "c1", "status": "needs_design_revision"}]
    worker._route_v2_repairs(camp, state, "c1", {"issues": [finding()]})
    proposal = {"proposal_id": "attention", "task_kind": "new_method", "target_candidate_ref": None,
                **task, "required_context_refs": [], "value_rationale": "Test attention"}
    plan = SelectedPlan(plan_id="plan_" + "c" * 32, originating_round="round", originating_proposal="attention",
                        proposal=proposal, scientific_intent_digest=digest(task["scientific_intent"]),
                        candidate_id="c1", spec_hash=state["experiment"]["spec_hash"],
                        approved_scientific_digest=digest(scientific_projection(state["experiment"])),
                        budget_owner="owner-original", recovery_limits={"max_engineering_repairs": 1},
                        current_step="design_revision", status="preparing").model_dump()
    selected_plan.save_plan(camp, state, plan)
    source(camp)
    original = (camp / "candidate_admissions.jsonl").read_bytes()
    revised = brief()
    revised["implementation_requirements"][1]["expected_behavior"] = "Use the actual inherited config"
    calls = []
    def designer(payload):
        calls.append(payload)
        return {"status": "completed", "design_brief": revised, "summary_zh": "工程条款已修订"}
    forbidden = lambda *args: pytest.fail("Design revision may not request a new Planner")
    selected_plan.tick_selected_plan(camp, state, forbidden, {"designer": designer})
    final = selected_plan.load_plan(camp)
    assert len(calls) == 1
    assert final["current_step"] == "repair" and final["candidate_id"] == "c1"
    assert final["repair_count"] == 0 and final["budget_owner"] == "owner-original"
    assert (camp / "candidate_admissions.jsonl").read_bytes() == original
    assert not (camp / "llm_api_intents.jsonl").exists()

