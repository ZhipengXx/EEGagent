"""Persistent selected-task executor inside the native campaign tick.

The outer Planner runs once per observation. Existing native design, approval,
implementation, repair, suite, job and Analyst services implement internal steps.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

from .action_planning import digest
from .plan_protocol import (config, decide_research_plan, measurement_action,
                            pool_summary, proposal_cost, round_id, round_input,
                            scientific_projection, validate_task)
from .research_execution import single_plan_enabled
from .schemas import SelectedPlan

TERMINAL = {"observation_ready", "failed", "cancelled", "needs_replan"}


def _path(camp: Path, plan_id: str) -> Path:
    if not plan_id.startswith("plan_") or any(c not in "0123456789abcdef" for c in plan_id[5:]):
        raise ValueError("selected_plan_identity_invalid")
    return camp / "selected_plans" / (plan_id + ".json")


def load_plan(camp: Path, state: dict | None = None) -> dict | None:
    from .loop import _read, load_state
    state = state if state is not None else load_state(camp)
    pointer = state.get("selected_plan") or {}
    plan_id = pointer.get("plan_id") if isinstance(pointer, dict) else pointer
    if not plan_id:
        active = [_read(path) for path in (camp / "selected_plans").glob("plan_*.json")]
        active = [row for row in active if row.get("status") not in TERMINAL]
        if len(active) > 1:
            raise ValueError("multiple_active_selected_plans")
        if not active:
            return None
        plan_id = active[0]["plan_id"]
    row = _read(_path(camp, str(plan_id)))
    if not row:
        raise ValueError("selected_plan_projection_missing")
    projected = SelectedPlan.model_validate(row).model_dump()
    validate_task(projected["proposal"], {"goal": _read(camp / "goal.json")})
    journal = camp / "selected_plans" / (str(plan_id) + ".receipts.jsonl")
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            receipt = json.loads(line)
            sequence = receipt.get("sequence")
            if receipt.get("plan_id") != plan_id or type(sequence) is not int:
                raise ValueError("selected_plan_receipt_identity_invalid")
            if sequence < len(projected["step_receipts"]):
                if projected["step_receipts"][sequence] != receipt:
                    raise ValueError("selected_plan_receipt_projection_conflict")
            elif sequence == len(projected["step_receipts"]):
                projected["step_receipts"].append(receipt)
            else:
                raise ValueError("selected_plan_receipt_sequence_gap")
    return projected


def save_plan(camp: Path, state: dict, plan: dict) -> None:
    from .loop import _write, save_state
    plan = SelectedPlan.model_validate(plan).model_dump()
    _write(_path(camp, plan["plan_id"]), plan)
    state["selected_plan"] = {key: plan.get(key) for key in (
        "plan_id", "budget_owner", "candidate_id", "status", "current_step", "repair_count",
        "terminal_observation_id", "terminal_observation_digest", "continuation_owner")}
    save_state(camp, state)


def _receipt_row(plan: dict, kind: str, **fields: Any) -> dict:
    return {"receipt_id": "plan_receipt_" + digest({"plan": plan["plan_id"], "sequence": len(plan["step_receipts"]),
        "kind": kind, "fields": fields})[:24], "plan_id": plan["plan_id"], "kind": kind,
        "sequence": len(plan["step_receipts"]), "time": time.time(), **fields}


def _append_plan_receipt(camp: Path, plan: dict, row: dict) -> None:
    sequence = row["sequence"]
    if sequence < len(plan["step_receipts"]):
        if plan["step_receipts"][sequence] != row:
            raise ValueError("selected_plan_receipt_projection_conflict")
        return
    if sequence != len(plan["step_receipts"]):
        raise ValueError("selected_plan_receipt_sequence_gap")
    path = camp / "selected_plans" / (plan["plan_id"] + ".receipts.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    plan["step_receipts"].append(row)


def _receipt(camp: Path, plan: dict, kind: str, **fields: Any) -> None:
    _append_plan_receipt(camp, plan, _receipt_row(plan, kind, **fields))


def _initial_selection_path(camp: Path, plan_id: str, phase: str) -> Path:
    _path(camp, plan_id)
    return camp / "selected_plans" / "selections" / (plan_id + "." + phase + ".json")


def _selected_receipt(camp: Path, plan: dict, pool_status: str) -> dict:
    """Adopt the exact legacy receipt when its projection was never published."""
    expected = _receipt_row(plan, "selected", proposal_id=plan["originating_proposal"],
        budget_owner=plan["budget_owner"], pool_status=pool_status)
    path = camp / "selected_plans" / (plan["plan_id"] + ".receipts.jsonl")
    if not path.is_file():
        return expected
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(rows) != 1 or {key: value for key, value in rows[0].items() if key != "time"}
            != {key: value for key, value in expected.items() if key != "time"}):
        raise ValueError("selected_plan_initial_receipt_conflict")
    return rows[0]


def _validate_initial_selection(camp: Path, prepared: dict) -> tuple[dict, dict]:
    from .loop import _read
    from .artifacts import file_digest
    unsigned = {key: value for key, value in prepared.items() if key != "prepare_digest"}
    if (prepared.get("schema_version") != "eeg_research.initial_selection.v1"
            or prepared.get("prepare_digest") != digest(unsigned)):
        raise ValueError("selected_plan_initial_prepare_changed")
    plan = SelectedPlan.model_validate(prepared["plan"]).model_dump()
    validate_task(plan["proposal"], {"goal": _read(camp / "goal.json")})
    source = camp / "research_rounds" / (plan["originating_round"] + ".json")
    record = _read(source)
    proposal = next((row for row in record.get("proposals") or []
        if row.get("proposal_id") == plan["originating_proposal"]), None)
    if (prepared.get("round_sha256") != file_digest(source)
            or plan["plan_id"] != "plan_" + digest({"round": plan["originating_round"], "proposal": plan["proposal"]})[:32]
            or proposal != plan["proposal"] or record.get("selected_proposal_id") != plan["originating_proposal"]
            or plan["scientific_intent_digest"] != digest(plan["proposal"]["scientific_intent"])
            or plan["step_receipts"] or plan["current_step"] != "prepare" or plan["status"] != "selected"):
        raise ValueError("selected_plan_initial_identity_changed")
    receipt = prepared["receipt"]
    expected = _receipt_row(plan, "selected", proposal_id=plan["originating_proposal"],
        budget_owner=plan["budget_owner"], pool_status=record["pool_status"])
    if ({key: value for key, value in receipt.items() if key != "time"}
            != {key: value for key, value in expected.items() if key != "time"}):
        raise ValueError("selected_plan_initial_receipt_changed")
    return plan, receipt


def _publish_initial_selection(camp: Path, state: dict, prepared: dict) -> None:
    """Replay only the frozen first submission; never allocate or dispatch."""
    from .loop import _read, _write, _sync_ledger
    from .artifacts import file_digest
    plan, receipt = _validate_initial_selection(camp, prepared)
    plan_id = plan["plan_id"]
    commit_path = _initial_selection_path(camp, plan_id, "commit")
    journal = camp / "selected_plans" / (plan_id + ".receipts.jsonl")
    if journal.is_file():
        rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        if rows != [receipt]:
            raise ValueError("selected_plan_initial_receipt_conflict")
        plan["step_receipts"] = rows
    _append_plan_receipt(camp, plan, receipt)
    existing = _read(_path(camp, plan_id))
    if existing and existing != plan:
        raise ValueError("selected_plan_initial_projection_conflict")
    pointer = state.get("selected_plan")
    if pointer != prepared.get("prior_selected_plan") and (pointer or {}).get("plan_id") != plan_id:
        raise ValueError("selected_plan_initial_state_conflict")
    state["latest_research_round"] = copy.deepcopy(prepared["round"])
    _sync_ledger(camp, state)
    save_plan(camp, state, plan)
    committed = {"schema_version": "eeg_research.initial_selection_commit.v1",
        "plan_id": plan_id, "prepare_digest": prepared["prepare_digest"],
        "receipt_id": receipt["receipt_id"], "receipt_sha256": file_digest(journal)}
    if commit_path.is_file():
        if _read(commit_path) != committed:
            raise ValueError("selected_plan_initial_commit_changed")
    else:
        _write(commit_path, committed, exclusive=True)


def _prepare_initial_selection(camp: Path, state: dict, plan: dict, record: dict) -> dict:
    from .loop import _read, _write
    from .artifacts import file_digest
    path = _initial_selection_path(camp, plan["plan_id"], "prepare")
    if path.is_file():
        prepared = _read(path)
        if prepared.get("plan") != plan:
            raise ValueError("selected_plan_initial_identity_changed")
        return prepared
    prepared = {"schema_version": "eeg_research.initial_selection.v1", "plan": copy.deepcopy(plan),
        "receipt": _selected_receipt(camp, plan, record["pool_status"]), "round": copy.deepcopy(record),
        "round_sha256": file_digest(camp / "research_rounds" / (plan["originating_round"] + ".json")),
        "prior_selected_plan": copy.deepcopy(state.get("selected_plan"))}
    prepared["prepare_digest"] = digest(prepared)
    _write(path, prepared, exclusive=True)
    return prepared


def _pending_initial_selections(camp: Path) -> list[Path]:
    return [path for path in (camp / "selected_plans" / "selections").glob("plan_*.prepare.json")
        if not path.with_name(path.name.replace(".prepare.json", ".commit.json")).is_file()]


def recover_initial_selection(camp: Path, state: dict) -> bool:
    from .loop import _read
    pending = _pending_initial_selections(camp)
    if len(pending) > 1:
        raise ValueError("multiple_uncommitted_selected_plans")
    if not pending:
        return False
    prepared = _read(pending[0])
    if pending[0] != _initial_selection_path(camp, prepared["plan"]["plan_id"], "prepare"):
        raise ValueError("selected_plan_initial_path_changed")
    _publish_initial_selection(camp, state, prepared)
    return True


def _publication_path(camp: Path, plan_id: str, decision_id: str, phase: str) -> Path:
    _path(camp, plan_id)
    if not decision_id.startswith("d") or not decision_id[1:].isdigit() or int(decision_id[1:]) < 1:
        raise ValueError("selected_plan_publication_identity_invalid")
    return camp / "selected_plans" / "publications" / plan_id / (decision_id + "." + phase + ".json")


def _step_binding(plan: dict) -> dict:
    return {**{key: plan.get(key) for key in ("plan_id", "originating_round", "originating_proposal",
        "scientific_intent_digest", "budget_owner", "current_step", "attempt", "repair_count",
        "candidate_id", "source_hash", "spec_hash", "approved_scientific_digest")},
        "proposal_digest": digest(plan["proposal"])}


def _validate_publication(camp: Path, plan: dict, prepared: dict) -> tuple[dict, dict]:
    from .artifacts import request_digest
    from .schemas import PlannerActionPayload
    unsigned = {key: value for key, value in prepared.items() if key != "prepare_digest"}
    if (prepared.get("schema_version") != "eeg_research.plan_step_publication.v1"
            or prepared.get("prepare_digest") != digest(unsigned)):
        raise ValueError("selected_plan_publication_prepare_changed")
    binding, body, intent = prepared["binding"], prepared["body"], prepared["intent"]
    record, payload = body["decision"], body["execution"]
    expected = request_digest(request=payload)
    publication_id = "plan_step_" + digest({"binding": binding, "decision_id": record["decision_id"],
        "execution_digest": expected})[:32]
    if (prepared["publication_id"] != publication_id or record.get("step_publication_id") != publication_id
            or record.get("origin_plan_id") != plan["plan_id"] or record.get("action") != payload.get("action")
            or record.get("raw_digest") != expected or record.get("execution_digest") != expected
            or body.get("raw") != payload or intent.get("decision_id") != record["decision_id"]
            or intent.get("execution_digest") != expected or intent.get("publication_id") != publication_id
            or intent.get("kind") != "step_intent" or intent.get("plan_id") != plan["plan_id"]
            or intent.get("step") != binding.get("current_step") or intent.get("attempt") != binding.get("attempt")
            or intent.get("action") != record.get("action") or payload.get("target_id") != binding.get("candidate_id")):
        raise ValueError("selected_plan_publication_identity_changed")
    frozen = _step_binding(plan)
    if any(binding.get(key) != frozen[key] for key in ("plan_id", "originating_round", "originating_proposal",
            "scientific_intent_digest", "budget_owner", "proposal_digest")):
        raise ValueError("selected_plan_publication_plan_changed")
    PlannerActionPayload.model_validate(payload)
    return record, intent


def _verify_publication_commit(camp: Path, plan: dict, prepared: dict) -> None:
    from .loop import _read
    from .artifacts import file_digest
    record, intent = _validate_publication(camp, plan, prepared)
    decision_path = camp / "decisions" / (record["decision_id"] + ".json")
    committed = _read(_publication_path(camp, plan["plan_id"], record["decision_id"], "commit"))
    if (not committed or committed != {"schema_version": "eeg_research.plan_step_commit.v1",
            "publication_id": prepared["publication_id"], "prepare_digest": prepared["prepare_digest"],
            "decision_sha256": file_digest(decision_path), "intent_receipt_id": intent["receipt_id"]}
            or _read(decision_path) != prepared["body"] or intent not in plan["step_receipts"]):
        raise ValueError("selected_plan_internal_publication_not_committed")


def _publish_step(camp: Path, state: dict, plan: dict, prepared: dict) -> None:
    """Finish the same frozen publication; no service or budget allocation here."""
    from .loop import _read, _write
    from .artifacts import file_digest
    record, intent = _validate_publication(camp, plan, prepared)
    decision_path = camp / "decisions" / (record["decision_id"] + ".json")
    commit_path = _publication_path(camp, plan["plan_id"], record["decision_id"], "commit")
    if commit_path.is_file():
        _verify_publication_commit(camp, plan, prepared)
    else:
        if prepared["binding"] != _step_binding(plan):
            raise ValueError("selected_plan_publication_step_changed")
        if decision_path.is_file():
            if _read(decision_path) != prepared["body"]:
                raise ValueError("selected_plan_publication_decision_changed")
        else:
            _write(decision_path, prepared["body"], exclusive=True)
        _append_plan_receipt(camp, plan, intent)
        _write(commit_path, {"schema_version": "eeg_research.plan_step_commit.v1",
            "publication_id": prepared["publication_id"], "prepare_digest": prepared["prepare_digest"],
            "decision_sha256": file_digest(decision_path), "intent_receipt_id": intent["receipt_id"]}, exclusive=True)
    # A committed journal is authoritative when either mutable projection lags.
    pointer = state.get("selected_plan") or {}
    if isinstance(pointer, dict) and pointer.get("plan_id") == plan["plan_id"]:
        missing = not any(row.get("decision_id") == record["decision_id"] for row in state.get("decisions") or [])
        if missing:
            state.setdefault("decisions", []).append(copy.deepcopy(record))
        if missing or _read(_path(camp, plan["plan_id"])).get("step_receipts") != plan["step_receipts"]:
            save_plan(camp, state, plan)


def recover_step_publications(camp: Path, state: dict) -> None:
    """Called before ordinary decision validation, including prepare-only exits."""
    from .loop import _read
    for path in sorted((camp / "selected_plans" / "publications").glob("plan_*/*.prepare.json")):
        prepared = _read(path)
        plan_id = prepared.get("binding", {}).get("plan_id")
        decision_id = prepared.get("body", {}).get("decision", {}).get("decision_id")
        if path != _publication_path(camp, str(plan_id or ""), str(decision_id or ""), "prepare"):
            raise ValueError("selected_plan_publication_path_changed")
        plan = load_plan(camp, {"selected_plan": {"plan_id": plan_id}})
        _publish_step(camp, state, plan, prepared)


def _native_outcome(before: dict, after: dict) -> dict:
    # The append-only owner/cost journals remain the authority for resources;
    # an execution receipt only repairs the action's mutable output projection.
    from .owner_budget import CAPS, DIMENSIONS
    protected = set(CAPS.values()) | set(DIMENSIONS) | {key + "_left" for key in DIMENSIONS}
    protected.update(key for key in before.keys() | after.keys() if key.endswith("_reserved"))
    protected.update({"goal_id", "decisions", "selected_plan", "latest_research_round", "budget",
        "budget_owner", "pause_after_step", "last_control_id", "stop_reason", "report_support_status"})
    return {key: {"before_present": key in before, "before": copy.deepcopy(before.get(key)),
                  "after_present": key in after, "after": copy.deepcopy(after.get(key))}
            for key in before.keys() | after.keys() if key not in protected
            and ((key in before) != (key in after) or before.get(key) != after.get(key))}


def _restore_native_outcome(camp: Path, state: dict, record: dict) -> None:
    from .decision_receipts import LEDGER, _rows
    receipt = next((row for row in _rows(camp / LEDGER)
                    if row["receipt_hash"] == record.get("decision_receipt")), None)
    if not receipt or receipt.get("source") != "native_step":
        return
    changes = (receipt.get("result") or {}).get("selected_step_state_delta") or {}
    pending = []
    for key, value in changes.items():
        present, current = key in state, state.get(key)
        if present == value["after_present"] and current == value["after"]:
            continue
        if present != value["before_present"] or current != value["before"]:
            raise ValueError("selected_plan_native_outcome_projection_conflict:" + key)
        pending.append((key, value))
    for key, value in pending:
        if value["after_present"]:
            state[key] = copy.deepcopy(value["after"])
        else:
            state.pop(key, None)


def reflect_control(camp: Path, state: dict) -> None:
    # A write error may block the campaign while the first submission still
    # lacks its commit. Keep that frozen receipt intact for native resume.
    if _pending_initial_selections(camp):
        return
    plan = load_plan(camp, state)
    if not plan or plan["status"] in TERMINAL:
        return
    status = state.get("status")
    if status in {"paused", "cancelled", "blocked"} and plan["status"] != status:
        _receipt(camp, plan, "control", status=status, current_step=plan["current_step"],
                 detail=state.get("detail"))
        plan["status"] = status
        save_plan(camp, state, plan)
        if status == "cancelled" and not state.get("live_job") and not state.get("active_suite"):
            try:
                _finish(camp, state, plan, "cancelled", detail="user_cancelled")
            except ValueError as exc:
                # Cancellation is final authority; unknown paid/process claims
                # stay occupied until native reconciliation can release them.
                _receipt(camp, plan, "cancellation_release_deferred", detail=str(exc))
                save_plan(camp, state, plan)


def _analysis_ref(state: dict, evidence_id: str | None) -> str | None:
    return next((row.get("analysis_artifact_id") for row in reversed(state.get("evidence") or [])
                 if row.get("kind") == "analysis" and row.get("status") == "completed"
                 and row.get("run_evidence_id") == evidence_id), None)


def _measurement(state: dict, plan: dict) -> dict | None:
    proposal = plan["proposal"]
    phase = proposal["evaluation_scope"].get("phase")
    kind = "method_screening" if phase == "screening_2fold" else "method_suite" if (
        state.get("evaluation_mode") == "loso_method_search") else None
    for row in reversed(state.get("evidence") or []):
        if row.get("candidate_id") != plan["candidate_id"] or row.get("evaluation_valid") is not True:
            continue
        if kind and row.get("kind") != kind:
            continue
        if not kind and row.get("fidelity") != ("pilot" if phase == "pilot" else "full"):
            continue
        if plan.get("source_hash") and row.get("source_hash") != plan["source_hash"]:
            continue
        if plan.get("spec_hash") and row.get("spec_hash") != plan["spec_hash"]:
            continue
        declared_seed = proposal["evaluation_scope"].get("seed")
        if declared_seed is not None and row.get("seed") != declared_seed:
            continue
        return row
    return None


def _finish(camp: Path, state: dict, plan: dict, status: str, *, detail: str | None = None) -> dict:
    from .experiment_observations import record_experiment_observation
    from .owner_budget import finish_selected_plan
    if not plan.get("terminal_observation_id"):
        _receipt(camp, plan, "terminal", status=status, detail=detail,
                 repair_count=plan["repair_count"], candidate_id=plan["candidate_id"])
        plan["status"] = status
        if status in {"failed", "needs_replan"}:
            plan["engineering_failure"] = {"error": detail or "implementation_failed",
                "error_fingerprint": digest({"detail": detail, "source_hash": plan.get("source_hash"), "spec_hash": plan.get("spec_hash")}),
                "repairs_used": plan["repair_count"]}
        card = record_experiment_observation(camp, state, plan, status,
            analysis_ref=plan.get("analysis_artifact_ref") or _analysis_ref(state, plan.get("measurement_evidence_id")))
        plan["terminal_observation_id"] = card["observation_id"]
        plan["terminal_observation_digest"] = card["content_digest"]
        # Publication precedes release/handoff. Restart reuses this exact card.
        save_plan(camp, state, plan)
    if not plan.get("continuation_owner"):
        continuation = finish_selected_plan(camp, plan_id=plan["plan_id"],
            observation_digest=plan["terminal_observation_digest"])
        plan["continuation_owner"] = continuation.get("owner_id")
    from .design_contract import design_contract_version
    from .loop import _read
    routed_replan = (status == "needs_replan" and bool(state.get("needs_replan"))
                     and design_contract_version(_read(camp / "goal.json")) == "v2")
    if routed_replan:
        state.pop("needs_replan", None)
        for key in ("repair_task", "design_revision_task", "design_revision_failure", "framework_issue_task"):
            if (state.get(key) or {}).get("candidate_id") == plan.get("candidate_id"):
                state.pop(key, None)
        for candidate in state.get("candidates") or []:
            if candidate.get("candidate_id") == plan.get("candidate_id"):
                candidate["status"] = "abandoned"
                candidate["terminal_reason"] = detail or "scientific_replan"
    if (status == "failed" or routed_replan) and plan.get("candidate_id") and plan["candidate_id"] != "baseline":
        from .candidate_lifecycle import detach_failed_candidate
        try:
            detach_failed_candidate(camp, state, plan["candidate_id"], reason=detail or "selected_task_engineering_failed")
        except ValueError as exc:
            # A pending check or unknown process is a recoverable environment
            # state. It does not acquire detach/refund authority from a label.
            _receipt(camp, plan, "failed_branch_retained", reason=str(exc))
    from .decision_receipts import append_receipt, project_decisions
    for record in state.get("decisions") or []:
        if (record.get("origin_plan_id") == plan["plan_id"] and record.get("executed") is False
                and not record.get("consumed") and status in {"failed", "needs_replan", "cancelled"}):
            append_receipt(camp, record, status="executed", source="selected_task_terminal",
                reason=detail or status, result={"origin_plan_id": plan["plan_id"], "task_status": status,
                    "observation_id": plan["terminal_observation_id"], "scientific_success": False})
    project_decisions(camp, state)
    state["terminal_observation_digest"] = plan["terminal_observation_digest"]
    state["terminal_observation_id"] = plan["terminal_observation_id"]
    state.setdefault("completed_plan_ids", [])
    if plan["plan_id"] not in state["completed_plan_ids"]:
        state["completed_plan_ids"].append(plan["plan_id"])
    if state.get("status") not in {"paused", "cancelled"}:
        state.update(status="created")
        state.pop("failure", None)
        state.pop("detail", None)
    save_plan(camp, state, plan)
    return state


def _blocked(camp: Path, state: dict, plan: dict, detail: str) -> dict:
    from .loop import persist_failure
    plan["status"] = "blocked"
    persist_failure(camp, state, phase="selected_plan", error_type=detail, detail=detail, recoverable=True)
    _receipt(camp, plan, "blocked", current_step=plan["current_step"], detail=detail)
    save_plan(camp, state, plan)
    return state


def engineering_budget_failure(detail: Any) -> bool:
    text = str(detail or "")
    return any(marker in text for marker in ("owner_request_unavailable:", "coding_lineage_budget_exhausted",
        "coding_no_progress_exhausted", "engineering_repair_limit", "coding_attempt_limit"))


def finish_bounded_engineering_failure(camp: Path, state: dict, error: Any) -> dict | None:
    plan = load_plan(camp, state)
    if (not plan or plan["current_step"] not in {"design", "implement", "check", "repair", "design_revision"}
            or not engineering_budget_failure(error)):
        return None
    return _finish(camp, state, plan, "failed", detail=str(error))


def _native_step(camp: Path, state: dict, plan: dict, action: str, services: dict, runner=None, *, raw=None) -> bool:
    """Commit an ordinary native decision and its frozen selected-task intent."""
    from .loop import (_apply_action, _finish_step, _next_decision_id, _read, _write, observation)
    from .artifacts import request_digest
    from .owner_budget import budget_owner
    pending = next((r for r in reversed(plan["step_receipts"]) if r.get("kind") == "step_intent"
                   and r.get("step") == plan["current_step"] and r.get("attempt") == plan["attempt"]), None)
    complete = any(r.get("kind") == "step_completed" and pending and
                   r.get("intent_receipt_id") == pending["receipt_id"] for r in plan["step_receipts"])
    if complete:
        return True
    if pending:
        from .decision_receipts import read_projected_decision
        body = read_projected_decision(camp, pending["decision_id"])
        record = body.get("decision") or {}
        payload = body.get("execution") or body.get("raw") or {}
        if not record or request_digest(request=payload) != record.get("execution_digest"):
            raise ValueError("selected_plan_internal_decision_identity_changed")
        # Native decision receipts and service caches recover completed work.
        projected = next((r for r in state.get("decisions") or [] if r.get("decision_id") == record["decision_id"]), None)
        if projected is None:
            state.setdefault("decisions", []).append(copy.deepcopy(record))
    else:
        payload = {"action": action, "target_id": plan.get("candidate_id"),
            "decision_rationale": "Internal step of the selected research task.",
            "required_artifact_refs": [ref for ref in plan["proposal"].get("required_context_refs") or []
                if ref.startswith("art_")], **(raw or {})}
        decision_id = _next_decision_id(camp, state)
        record = {"decision_id": decision_id, "origin_plan_id": plan["plan_id"],
            "campaign_id": state["goal_id"], "action": action, "ok": True, "executed": False,
            "decision_schema_version": "eeg_research.decision.plan_step.v1",
            "raw_digest": request_digest(request=payload), "execution_digest": request_digest(request=payload)}
        binding = _step_binding(plan)
        publication_id = "plan_step_" + digest({"binding": binding, "decision_id": decision_id,
            "execution_digest": record["execution_digest"]})[:32]
        record["step_publication_id"] = publication_id
        pending = _receipt_row(plan, "step_intent", step=plan["current_step"], attempt=plan["attempt"],
            action=action, decision_id=decision_id, execution_digest=record["execution_digest"],
            publication_id=publication_id)
        prepared = {"schema_version": "eeg_research.plan_step_publication.v1", "publication_id": publication_id,
            "binding": binding, "body": {"decision": record, "raw": payload, "execution": payload}, "intent": pending}
        prepared["prepare_digest"] = digest(prepared)
        _write(_publication_path(camp, plan["plan_id"], decision_id, "prepare"), prepared, exclusive=True)
        _publish_step(camp, state, plan, prepared)
    decision = {"decision_id": record["decision_id"], "origin_plan_id": plan["plan_id"],
                "action": action, "execution": payload, "raw": payload, "ok": True}
    if not record.get("executed") and not record.get("consumed"):
        before = copy.deepcopy(state)
        with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
            _apply_action(camp, state, action, decision, runner, services, observation(camp))
        _finish_step(camp, state, selected_step_outcome=_native_outcome(before, state))
    elif record.get("executed"):
        _restore_native_outcome(camp, state, record)
    if state.get("status") in {"blocked", "paused", "cancelled"}:
        return False
    workspace = camp / "candidates" / str(plan.get("candidate_id") or "")
    from .identity import source_hash
    attempt_identity = _read(workspace / "attempt.json") if plan.get("candidate_id") else {}
    _receipt(camp, plan, "step_completed", step=plan["current_step"], attempt=plan["attempt"],
             intent_receipt_id=pending["receipt_id"], action=action,
             candidate_id=plan.get("candidate_id"), active_suite=state.get("active_suite"),
             live_job=state.get("live_job"), source_hash=source_hash(workspace) if plan.get("candidate_id") else None,
             attempt_id=attempt_identity.get("attempt_id"), spec_hash=(state.get("experiment") or {}).get("spec_hash"),
             scientific_intent_digest=plan["scientific_intent_digest"])
    save_plan(camp, state, plan)
    return True


def verify_internal_decision(camp: Path, state: dict, decision: dict) -> dict | None:
    """Internal dispatch bypasses outer K comparison, never native science gates."""
    plan_id = decision.get("origin_plan_id")
    if not plan_id:
        return None
    plan = load_plan(camp, state)
    if not plan or plan["plan_id"] != plan_id or plan["status"] in TERMINAL:
        raise ValueError("selected_plan_internal_dispatch_not_active")
    body = decision.get("execution") or decision.get("raw") or {}
    matching = next((r for r in plan["step_receipts"] if r.get("kind") == "step_intent"
                     and r.get("decision_id") == decision.get("decision_id")), None)
    from .artifacts import request_digest
    if not matching or matching.get("execution_digest") != request_digest(request=body):
        raise ValueError("selected_plan_internal_dispatch_without_intent")
    original = _read_internal_publication(camp, plan, str(decision.get("decision_id") or ""))
    if original is not None:
        _verify_publication_commit(camp, plan, original)
    if body.get("target_id") != plan.get("candidate_id"):
        raise ValueError("selected_plan_internal_target_changed")
    if body.get("action") == "run_confirmation" and plan["proposal"]["task_kind"] != "confirmation":
        raise ValueError("confirmation_requires_explicit_selected_plan")
    return plan


def verify_step_history(camp: Path, record: dict, payload: Any) -> None:
    """Read-only historical intent proof, including closed selected tasks."""
    from .loop import _read
    from .artifacts import request_digest
    from .schemas import PlannerActionPayload
    if not isinstance(payload, dict):
        raise ValueError("selected_plan_internal_execution_missing")
    plan = load_plan(camp, {"selected_plan": {"plan_id": str(record.get("origin_plan_id") or "")}})
    if plan is None:
        raise ValueError("selected_plan_internal_history_missing")
    expected = request_digest(request=payload)
    if expected != record.get("execution_digest"):
        raise ValueError("selected_plan_internal_execution_changed")
    if not any(row.get("kind") == "step_intent" and row.get("decision_id") == record.get("decision_id")
               and row.get("execution_digest") == expected for row in plan["step_receipts"]):
        raise ValueError("selected_plan_internal_history_without_intent")
    if record.get("step_publication_id"):
        prepared = _read_internal_publication(camp, plan, str(record.get("decision_id") or ""))
        if prepared is None or prepared.get("publication_id") != record["step_publication_id"]:
            raise ValueError("selected_plan_internal_publication_prepare_missing")
        _verify_publication_commit(camp, plan, prepared)
    PlannerActionPayload.model_validate(payload)


def _read_internal_publication(camp: Path, plan: dict, decision_id: str) -> dict | None:
    from .loop import _read
    record = _read(camp / "decisions" / (decision_id + ".json")).get("decision") or {}
    if not record.get("step_publication_id"):
        return None  # Previously published v1 intents keep their original authority.
    prepared = _read(_publication_path(camp, plan["plan_id"], decision_id, "prepare"))
    if not prepared:
        raise ValueError("selected_plan_internal_publication_prepare_missing")
    return prepared


def _select(camp: Path, state: dict, backend, critic=None) -> dict:
    from .loop import _read, _sync_ledger, _write, event, observation, persist_failure, save_state
    from .owner_budget import allocate_selected_plan, planner_owner, finish_planner_round
    from .api_policy import planner_api_round
    from .paid_research_round import restore_paid, publish_raw, revalidate_history
    # Paid recovery must precede observation/compaction: both may capture new
    # delivery sources even though no provider request is eventually sent.
    from .research_execution import independent_critic_enabled, planning_call_budget
    staged_mode = independent_critic_enabled(camp)
    if staged_mode:
        from .staged_planning_recovery import restore_staged_paid
        recovered = restore_staged_paid(camp.resolve(), state)
    else:
        recovered = restore_paid(camp.resolve(), state)
    full = recovered[0] if recovered else observation(camp)
    if staged_mode and not recovered and (camp / "method_search_manifest.json").is_file():
        # Supply actual frozen fold identifiers even on the first round, when
        # neither a previous experiment nor an observation card exists yet.
        from .method_suite import manifest
        common = manifest(camp)
        phase = "confirmation_10fold" if common.get("evaluation_schedule") == "screen_then_confirm" else "full"
        scopes = {phase: {"phase": phase, "folds": [row["fold_id"] for row in common["folds"]],
                          "seed": common["seed"], "controls": ["baseline"]}}
        if phase == "confirmation_10fold":
            scopes["screening_2fold"] = {"phase": "screening_2fold",
                "folds": common["selection_manifest"]["screening_fold_ids"],
                "seed": common["seed"], "controls": ["baseline"]}
        full["frozen_evaluation_scopes"] = {"phases": scopes,
            **{key: common[key] for key in ("metric", "metric_unit", "aggregation") if key in common}}
    if staged_mode:
        from .planning_contract import enabled, freeze_planning_contract
        if enabled(full) and not isinstance(full.get("_planning_contract"), dict):
            if "execution_protocol" not in full and (camp / "execution_protocol.json").is_file():
                from .execution_protocol import load_protocol
                full["execution_protocol"] = load_protocol(camp) or {}
            full["_planning_contract"] = freeze_planning_contract(full)
        native_errors = (full.get("_planning_contract") or {}).get("native_errors") or []
        if native_errors:
            persist_failure(camp, state, phase="planner", error_type="planning_native_input_invalid",
                detail="Frozen native task scope is unavailable", recoverable=True,
                diagnostics={"native_errors": native_errors})
            save_state(camp, state)
            return state
    cfg = config(full)
    input_digest = state.get("terminal_observation_digest") or digest(round_input(full))
    research_round_id = round_id(full)
    from .experiment_observations import compact_planner_view, ObservationDeliveryRequired
    try:
        if recovered and staged_mode:
            delivered = recovered[3].get("planner_delivery") or compact_planner_view(camp, full)
        else:
            delivered = full if recovered else compact_planner_view(camp, full)
    except ObservationDeliveryRequired as exc:
        from dataclasses import asdict
        # Persist the actual typed obligation before planning owner/API admission.
        # The same observation, round and consumed resources survive resume.
        state["context_requirement"] = {"schema_version": "eeg_research.planner_context_requirement.v1",
            "status": "needs_context", "round_id": research_round_id,
            "input_observation_digest": input_digest, "result": asdict(exc.result)}
        persist_failure(camp, state, phase="planner", error_type="context_needs_context",
            detail=exc.result.reason, recoverable=True,
            diagnostics={"reason": exc.result.reason, "budget_report": exc.result.budget_report})
        save_state(camp, state)
        return state
    requirement = state.get("context_requirement")
    if isinstance(requirement, dict) and requirement.get("status") == "needs_context":
        requirement.update(status="resolved", resolved_round_id=research_round_id,
            resolved_input_observation_digest=input_digest)
        if (state.get("failure") or {}).get("error_type") == "context_needs_context":
            state.pop("failure", None)
            state.pop("detail", None)
        event(camp, "planner_context_resolved", round_id=research_round_id,
              input_observation_digest=input_digest)
    if staged_mode and not recovered:
        from .planning_context import build_proposer_input
        delivered = build_proposer_input(full, delivered)
    from .context.representations import delivered_source_refs
    context_refs = delivered_source_refs(delivered)
    full["_delivered_context_refs"] = sorted(set(full.get("_delivered_context_refs") or []) | {ref for ref in context_refs if ref})
    def request(payload):
        bind = getattr(backend, "bind", None)
        if callable(bind):
            bind(operation_id=research_round_id + (":correction" if payload.get("plan_repair_request") else ":generation"))
        raw = backend(payload)
        publish_raw(camp, state, full, planning_owner, raw,
                    correction=bool(payload.get("plan_repair_request")))
        return raw
    round_path = camp / "research_rounds" / (research_round_id + ".json")
    draft_path = camp / "research_round_drafts" / (research_round_id + ".json")
    saved = _read(round_path)
    limits = {"max_engineering_repairs": min(cfg.max_engineering_repairs,
            int(state.get("max_repairs_per_candidate", cfg.max_engineering_repairs))), "coder_steps": 8}
    limits["max_repairs"] = limits["max_engineering_repairs"]
    if saved:
        if saved.get("input_observation_digest") != digest(round_input(full)):
            raise ValueError("research_round_input_identity_conflict")
        for item in saved.get("proposals") or []:
            validate_task(item, full)
        saved.update(pool_summary(saved.get("proposals") or [], saved.get("evaluations") or [], cfg))
        if saved["unevaluated_proposal_ids"]:
            saved.update(selected_proposal_id=None, shortfall_reason="missing_or_nonunique_proposal_evaluations")
        chosen = next((p for p in saved.get("proposals") or [] if p["proposal_id"] == saved.get("selected_proposal_id")), None)
        parsed = {"ok": chosen is not None and saved.get("pool_status") in {"complete", "degraded_singleton"},
            "round": {k: v for k, v in saved.items() if k not in {"proposal_history", "planning_owner"}},
            "selected_proposal": chosen, "proposal_history": saved.get("proposal_history") or [],
            "detail": "research_pool_" + str(saved.get("pool_status"))}
    else:
        paid_draft = _read(draft_path)
        if recovered and not staged_mode:
            _, planning_owner, history, proof = recovered
            parsed = revalidate_history(full, history)
            if not paid_draft:
                _write(draft_path, {"input_observation_digest": digest(round_input(full)),
                    "planning_owner": planning_owner, "parsed": parsed, "paid_recovery_proof": proof}, exclusive=True)
            elif paid_draft.get("input_observation_digest") != digest(round_input(full)) or paid_draft.get("planning_owner") != planning_owner:
                raise ValueError("research_round_draft_identity_conflict")
        elif paid_draft:
            if paid_draft["input_observation_digest"] != digest(round_input(full)):
                raise ValueError("research_round_draft_identity_conflict")
            parsed = paid_draft["parsed"]
            planning_owner = paid_draft["planning_owner"]
        else:
            if staged_mode and critic is None:
                raise ValueError("independent_research_critic_service_missing")
            with planner_owner(camp, observation_digest=input_digest, round_id=research_round_id) as planning_owner:
                round_attempts = planning_call_budget(camp) if staged_mode else cfg.max_planner_api_attempts
                with planner_api_round(camp, research_round_id, max_attempts=round_attempts):
                    if staged_mode:
                        from .staged_planning import decide_staged_plan
                        parsed = decide_staged_plan(camp, state, full, delivered, backend, critic,
                            planning_owner, limits, cfg,
                            recovered_stages=recovered[2] if recovered else None,
                            recovery_proof=recovered[3] if recovered else None)
                    else:
                        parsed = decide_research_plan(full, lambda req: request({**delivered,
                            **({"plan_repair_request": req["plan_repair_request"]} if req.get("plan_repair_request") else {})}))
            # Persist paid drafts before releasing the continuation, so a crash
            # after comparison never buys or reuses another planning allowance.
            _write(draft_path, {"input_observation_digest": digest(round_input(full)),
                "planning_owner": planning_owner, "parsed": parsed}, exclusive=True)
        finish_planner_round(camp, planning_owner)
        if parsed.get("correction_unavailable"):
            _sync_ledger(camp, state)
            persist_failure(camp, state, phase="planner", error_type=parsed["detail"],
                            detail=parsed["correction_unavailable"]["reason"], recoverable=True)
            save_state(camp, state)
            return state
        if parsed.get("round"):
            from .owner_budget import preview_selected_plan
            native_round = parsed["round"]
            if not staged_mode:
                feasible = []
                for item in native_round["proposals"]:
                    preview = preview_selected_plan(camp, task_kind=item["task_kind"],
                        known_cost=proposal_cost(item, full)["known"], recovery_limits=limits)
                    if preview.get("ok"):
                        feasible.append(item)
                    else:
                        native_round["validation_records"].append({"proposal_id": item["proposal_id"],
                            "status": "rejected", "error": preview.get("reason"), "stage": "complete_task_budget"})
                by_id = {item["proposal_id"]: item for item in feasible}
                evaluations = [item for item in native_round["evaluations"] if item["proposal_id"] in by_id]
                ranking = sorted(evaluations, key=lambda item: item["rank"])
                prior_selected = native_round["selected_proposal_id"]
                selected_id = prior_selected if prior_selected in by_id else next((item["proposal_id"]
                    for item in ranking if item["proposal_id"] in by_id), None)
                native_round.update(proposals=feasible, evaluations=evaluations,
                                    **pool_summary(feasible, evaluations, cfg))
                native_round["shortfall_reason"] = ("missing_or_nonunique_proposal_evaluations"
                    if native_round["unevaluated_proposal_ids"] else
                    "independent_validation_or_complete_task_budget_rejections"
                    if native_round["valid_count"] < cfg.target_k else None)
                ok = bool(selected_id and native_round["pool_status"] in {"complete", "degraded_singleton"})
                native_round["selected_proposal_id"] = selected_id if ok else None
                if ok and selected_id != prior_selected:
                    native_round["selection_source"] = "valid_evaluation_rank_budget_fallback"
                    native_round["selection_rationale"] += " Native complete-task admission selected the highest-ranked feasible evaluation."
                parsed.update(ok=ok, selected_proposal=by_id.get(selected_id) if ok else None,
                    detail=None if ok else "research_pool_" + native_round["pool_status"])
            _write(round_path, {**native_round, "proposal_history": parsed["proposal_history"],
                "planning_owner": planning_owner}, exclusive=True)
        else:
            _write(round_path, {"round_id": research_round_id, "pool_status": "no_feasible_plan", "valid_count": 0,
                "input_observation_digest": digest(round_input(full)), "proposal_history": parsed["proposal_history"],
                "planning_owner": planning_owner, "validation_records": parsed.get("validation_records")}, exclusive=True)
    _sync_ledger(camp, state)
    state["latest_research_round"] = parsed.get("round") or {"round_id": research_round_id}
    if not parsed.get("ok"):
        persist_failure(camp, state, phase="planner", error_type=parsed.get("detail") or "no_feasible_plan",
                        detail=parsed.get("detail") or "no_feasible_plan", recoverable=True)
        save_state(camp, state)
        return state
    proposal = parsed["selected_proposal"]
    plan_id = "plan_" + digest({"round": research_round_id, "proposal": proposal})[:32]
    owner = allocate_selected_plan(camp, plan_id=plan_id, task_kind=proposal["task_kind"],
        known_cost=proposal_cost(proposal, full)["known"], recovery_limits=limits)
    plan = SelectedPlan(plan_id=plan_id, originating_round=research_round_id,
        originating_proposal=proposal["proposal_id"], proposal=proposal,
        scientific_intent_digest=digest(proposal["scientific_intent"]), candidate_id=proposal.get("target_candidate_ref"),
        budget_owner=owner["owner_id"], recovery_limits=limits,
        input_evidence_ids=[row.get("evidence_id") for row in state.get("evidence") or [] if row.get("evidence_id")]).model_dump()
    if plan.get("candidate_id"):
        from .action_planning import target_binding
        binding = target_binding(full, measurement_action(proposal, full), plan["candidate_id"])
        plan.update(source_hash=binding.get("source_hash"), spec_hash=binding.get("spec_hash"))
        plan["method_revision"] = next((row.get("method_revision") for row in reversed(full.get("evidence") or [])
            if row.get("candidate_id") == plan["candidate_id"] and row.get("method_revision")
            and (not plan.get("source_hash") or row.get("source_hash") == plan["source_hash"])), None)
    prepared = _prepare_initial_selection(camp, state, plan, parsed["round"])
    _publish_initial_selection(camp, state, prepared)
    event(camp, "research_plan_selected", plan_id=plan_id, proposal_id=proposal["proposal_id"],
          round_id=research_round_id, task_kind=proposal["task_kind"])
    return state


def tick_selected_plan(camp: Path, state: dict, backend, services: dict, runner=None) -> dict:
    from .loop import (_launch, _record_job, _sync_ledger, candidate_implementation_target,
                       load_state, observation, save_state)
    from .owner_budget import budget_owner
    from .design_contract import design_contract_version
    from .loop import _read
    design_v2 = design_contract_version(_read(camp / "goal.json")) == "v2"
    if recover_initial_selection(camp, state):
        return state
    from .iteration_control import maybe_complete_research
    if maybe_complete_research(camp, state):
        return state
    plan = load_plan(camp, state)
    if plan is None or plan["status"] in TERMINAL:
        if plan and plan.get("terminal_observation_id") and not plan.get("continuation_owner"):
            return _finish(camp, state, plan, plan["status"])
        return _select(camp, state, backend, critic=services.get("critic"))
    # Resume continues the same identity and accumulated attempts.
    if plan["status"] in {"paused", "blocked"}:
        plan["status"] = "preparing"
        save_plan(camp, state, plan)
    if plan.get("approved_scientific_digest"):
        if plan["proposal"]["task_kind"] == "new_method" and digest(scientific_projection(state.get("experiment"))) != plan["approved_scientific_digest"]:
            return _finish(camp, state, plan, "needs_replan", detail="approved_scientific_intent_changed")
    # The native suite owns fold restart/checkpoint/source verification and
    # matched-control scheduling. Only its active selected task can advance.
    if state.get("active_suite"):
        from .method_suite import advance
        from .resource_guard import advance_needs_gpus, require_idle_gpus
        if advance_needs_gpus(camp, state):
            require_idle_gpus(camp, state)
        with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
            advance(camp, state, services)
        state = load_state(camp)
        save_plan(camp, state, plan)
        return state
    if state.get("live_job"):
        settle = services.get("settle")
        if settle is None:
            return _blocked(camp, state, plan, "selected_plan_settle_service_missing")
        record = settle(camp, state["live_job"])
        if record.get("status") == "running":
            state.update(status="training", heartbeat=record.get("heartbeat"))
            save_plan(camp, state, plan)
            return state
        _record_job(camp, state, record)
        state.update(live_job=None, status="created")
        save_plan(camp, state, plan)
        return state
    if state.get("queued"):
        item = state["queued"].pop(0)
        if item[0] != plan.get("candidate_id"):
            raise ValueError("selected_plan_queue_target_mismatch")
        with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
            _launch(camp, state, services, item[0], item[1], training_seed=item[2] if len(item) > 2 else None)
        save_plan(camp, state, plan)
        return state
    kind = plan["proposal"]["task_kind"]
    if plan["current_step"] == "prepare":
        if kind == "new_method":
            plan.update(status="preparing", current_step="design")
        elif kind == "diagnosis" or plan["proposal"]["evaluation_scope"].get("phase") == "diagnostic":
            plan.update(status="collecting_observation", current_step="diagnose")
        else:
            plan.update(status="executing_measurement", current_step="measure")
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "design":
        from .planning_contract import enabled as planning_contract_enabled
        from .research_execution import research_goal
        intervention = (plan["proposal"]["task_description"] if planning_contract_enabled(research_goal(camp))
                        else plan["proposal"]["scientific_intent"])
        draft = {"intervention": intervention, "parent_candidate_id": "baseline",
                 "control_candidate_id": "baseline", "initial_fidelity": "pilot" if
                    plan["proposal"]["evaluation_scope"].get("phase") == "pilot" else "full"}
        if not _native_step(camp, state, plan, "design_experiment", services, runner, raw={
            "hypothesis_draft": {"mechanism": plan["proposal"]["scientific_intent"], "question": plan["proposal"]["question"]},
            "experiment_draft": draft}):
            bounded = finish_bounded_engineering_failure(camp, state, state.get("detail"))
            if bounded is not None:
                return bounded
            reflect_control(camp, state)
            return state
        if state.get("experiment_failed") or not (state.get("experiment") or {}).get("approval_record"):
            return _finish(camp, state, plan, "failed", detail="design_not_approved")
        plan["candidate_id"] = candidate_implementation_target(camp, state)
        plan["spec_hash"] = state["experiment"].get("spec_hash")
        plan["approved_scientific_digest"] = digest(scientific_projection(state["experiment"]))
        plan.update(status="implementing", current_step="implement")
        save_plan(camp, state, plan)
        return state
    if design_v2 and plan["current_step"] == "design_revision":
        revision = state.get("design_revision_task")
        if not isinstance(revision, dict) or revision.get("candidate_id") != plan["candidate_id"]:
            return _blocked(camp, state, plan, "design_revision_task_missing")
        if plan["repair_count"] >= plan["recovery_limits"]["max_engineering_repairs"]:
            return _finish(camp, state, plan, "failed", detail="engineering_repair_limit")
        if not _native_step(camp, state, plan, "design_experiment", services, runner,
                            raw={"experiment_draft": {"design_revision": revision}}):
            reflect_control(camp, state)
            return state
        plan["attempt"] += 1
        if state.get("needs_replan"):
            return _finish(camp, state, plan, "needs_replan", detail=str(state["needs_replan"]))
        if state.get("design_revision_failure"):
            return _blocked(camp, state, plan, str(state["design_revision_failure"]["error_type"]))
        if state.get("design_revision_task") or not state.get("repair_task"):
            return _blocked(camp, state, plan, "design_revision_has_no_approved_outcome")
        previous_hash = plan.get("spec_hash")
        plan["spec_hash"] = state["experiment"].get("spec_hash")
        plan["approved_scientific_digest"] = digest(scientific_projection(state["experiment"]))
        _receipt(camp, plan, "design_revision_approved", candidate_id=plan["candidate_id"],
                 prior_spec_hash=previous_hash, spec_hash=plan["spec_hash"],
                 experiment_ref=state.get("experiment_ref"), budget_owner=plan["budget_owner"])
        plan.update(status="repairing", current_step="repair")
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] in {"implement", "repair"}:
        repair = plan["current_step"] == "repair"
        if repair and plan["repair_count"] >= plan["recovery_limits"]["max_engineering_repairs"]:
            return _finish(camp, state, plan, "failed", detail="engineering_repair_limit")
        if not _native_step(camp, state, plan, "repair_candidate" if repair else "implement_candidate", services, runner):
            bounded = finish_bounded_engineering_failure(camp, state, state.get("detail"))
            if bounded is not None:
                return bounded
            # The native step has not completed. An existing repair task does
            # not turn a recoverable API/resource wait into a consumed repair.
            reflect_control(camp, state)
            return state
        if repair:
            plan["repair_count"] += 1
        plan["attempt"] += 1
        if state.get("needs_replan"):
            return _finish(camp, state, plan, "needs_replan", detail=str(state["needs_replan"]))
        if design_v2 and state.get("design_revision_task"):
            revision = state["design_revision_task"]
            if revision.get("candidate_id") != plan["candidate_id"]:
                raise ValueError("selected_plan_design_revision_candidate_changed")
            _receipt(camp, plan, "design_revision_requested", candidate_id=plan["candidate_id"],
                     prior_spec_hash=plan.get("spec_hash"), issues=revision.get("issues") or [])
            plan.update(status="preparing", current_step="design_revision")
        elif state.get("pending_candidate_check"):
            plan.update(status="checking", current_step="check")
        elif state.get("candidate_ready") and state.get("candidate_id") == plan["candidate_id"]:
            from .handoffs import candidate_context
            source = candidate_context(camp, plan["candidate_id"])
            plan.update(source_hash=source.get("source_hash"), status="executing_measurement", current_step="measure")
        elif state.get("repair_task"):
            if state["repair_task"].get("candidate_id") != plan["candidate_id"]:
                raise ValueError("selected_plan_repair_candidate_changed")
            plan.update(status="repairing", current_step="repair")
            state.update(status="created")
            state.pop("failure", None)
        elif state.get("experiment_failed"):
            return _finish(camp, state, plan, "failed", detail="engineering_implementation_failed")
        else:
            return _blocked(camp, state, plan, "implementation_has_no_verified_outcome")
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "check":
        check = services.get("resume_pending_check")
        if check is None:
            return _blocked(camp, state, plan, "selected_plan_check_service_missing")
        with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
            check(camp, state)
        if state.get("pending_candidate_check"):
            reflect_control(camp, state)
            return state
        plan["current_step"] = "implement"
        plan["status"] = "checking"
        # The native implement service adopts the finished check and review.
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "measure":
        if plan.get("source_hash") and plan["candidate_id"] != "baseline":
            from .handoffs import candidate_context
            source = candidate_context(camp, plan["candidate_id"])
            if source.get("source_hash") != plan["source_hash"] or (plan.get("spec_hash")
                    and source.get("spec_hash") != plan["spec_hash"]):
                return _finish(camp, state, plan, "needs_replan", detail="selected_source_or_spec_changed")
        row = _measurement(state, plan)
        if row:
            plan.update(measurement_evidence_id=row["evidence_id"], status="collecting_observation", current_step="analyze")
            save_plan(camp, state, plan)
            return state
        failed_measurement = next((r for r in reversed(state.get("evidence") or [])
            if r.get("candidate_id") == plan["candidate_id"] and r.get("fidelity")
            and r.get("evidence_id") not in plan["input_evidence_ids"]
            and (r.get("job_status") in {"failed", "cancelled", "invalid"} or r.get("evaluation_valid") is False)), None)
        if failed_measurement:
            return _finish(camp, state, plan, "failed", detail="measurement_engineering_failed:" +
                str(failed_measurement.get("reason") or failed_measurement.get("job_status")))
        action = measurement_action(plan["proposal"], observation(camp))
        if state.get("evaluation_mode") == "loso_method_search" and action == "run_full" and plan["candidate_id"] != "baseline":
            baseline = any(r.get("candidate_id") == "baseline" and r.get("kind") == "method_suite"
                and r.get("evaluation_valid") for r in state.get("evidence") or [])
            if not baseline:
                from .method_suite import begin
                with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
                    begin(camp, state, "baseline")
                save_plan(camp, state, plan)
                return state
        plan["status"] = "executing_measurement"
        if not _native_step(camp, state, plan, action, services, runner):
            reflect_control(camp, state)
            return state
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "diagnose":
        if not _native_step(camp, state, plan, "collect_diagnostics", services, runner):
            reflect_control(camp, state)
            return state
        diagnostic = state.get("last_local_result") or {}
        evidence_id = diagnostic.get("evidence_id") or diagnostic.get("same_as")
        if not evidence_id:
            evidence_id = next((row.get("evidence_id") for row in reversed(state.get("evidence") or [])
                if row.get("kind") == "diagnostic_bundle"), None)
        plan["measurement_evidence_id"] = evidence_id
        plan["analysis_artifact_ref"] = next((row.get("analysis_artifact_id")
            for row in reversed(state.get("evidence") or []) if row.get("kind") == "analysis"
            and row.get("status") == "completed" and row.get("candidate_id") == plan["candidate_id"]), None)
        plan.update(status="collecting_observation", current_step="observation")
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "analyze":
        row = _measurement(state, plan)
        if not row:
            return _blocked(camp, state, plan, "selected_plan_verified_measurement_missing")
        if not _analysis_ref(state, row["evidence_id"]):
            analyst = services.get("analyze")
            if analyst is None:
                return _blocked(camp, state, plan, "selected_plan_analyst_missing")
            with budget_owner(camp, plan["budget_owner"], purpose="mainline"):
                analyst(camp, state)
            if not _analysis_ref(state, row["evidence_id"]):
                return _blocked(camp, state, plan, "selected_plan_analysis_incomplete")
        plan["current_step"] = "observation"
        save_plan(camp, state, plan)
        return state
    if plan["current_step"] == "observation":
        return _finish(camp, state, plan, "observation_ready")
    raise ValueError("selected_plan_step_unknown:" + str(plan["current_step"]))
