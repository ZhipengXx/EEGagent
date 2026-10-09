"""Background worker. One process per campaign; training end triggers the next decision."""

from __future__ import annotations

import subprocess

import fcntl
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic import jobs
from react_agent.eeg_research.agentic.contract import public_contract
from react_agent.eeg_research.agentic.execution_protocol import load_protocol, resolve_worker_design
from react_agent.eeg_research.agentic.identity import ensure_attempt, operation_id, result_matches, rotate_attempt, source_hash
from react_agent.eeg_research.agentic.interface import candidate_interface
from react_agent.eeg_research.agentic.lineage import LineageError, materialize
from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
from react_agent.eeg_research.agentic.loop import (
    _sync_ledger,
    align_interrupt,
    event,
    candidate_implementation_target,
    load_state,
    persist_failure,
    save_state,
    tick,
)
from react_agent.eeg_research.agentic.native_patch import RecoveryBlocked, implement, review
from react_agent.eeg_training.protocol import data_root

_TERMINAL = {"paused", "finished", "blocked", "cancelled"}
CODER_STEPS = 8
IMPLEMENT_ATTEMPTS = 3
_METHOD_ROLE_HARD_BLOCKS = {
    "role_no_progress_retry_limit", "unsettled_api_intent_requires_recovery",
    "handoff_compaction_failed", "handoff_compaction_forbidden_input",
    "handoff_compaction_already_compacted", "handoff_compaction_output_budget_exceeded",
    "handoff_compaction_protected_budget_exceeded", "budget_exhausted",
}


def _lesson_proposal(reply: Any) -> dict[str, Any]:
    """Read lessons from a RoleResult envelope or a bare curator object."""
    if not isinstance(reply, dict):
        return {}
    inner = reply.get("payload")
    body = inner if isinstance(inner, dict) else reply
    # Completed native curation stores the original model proposal beside its
    # acceptance/index receipts. Task recovery returns that runtime envelope.
    proposal = body.get("proposal")
    if isinstance(proposal, dict) and ("proposed_lessons" in proposal or "proposed_skills" in proposal):
        return proposal
    if isinstance(inner, dict) and ("proposed_lessons" in inner or "proposed_skills" in inner or "summary_zh" in inner):
        return inner
    return reply


def worker_lock_held(camp: Path) -> bool:
    """True when another worker still owns the campaign lock."""
    lock_path = camp / "worker.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _block_phase(
    camp_dir: Path,
    state: dict[str, Any],
    *,
    phase: str,
    error_type: str,
    detail: str,
    recoverable: bool,
) -> None:
    persist_failure(
        camp_dir,
        state,
        phase=phase,
        error_type=error_type,
        detail=detail,
        recoverable=recoverable,
    )


def _attach(backend: Any, **fields: Any) -> None:
    bind = getattr(backend, "bind", None)
    if callable(bind):
        bind(**fields)
    try:
        backend.identity = dict(fields)
    except Exception:
        return


def _stored_review(workspace: Path, *, candidate_id: str, attempt_id: str, input_hash: str) -> dict[str, Any] | None:
    """Return a review only when it belongs to this attempt and this source hash."""
    path = workspace / "review.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecoveryBlocked("review_result_unreadable") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {"ready", "needs_fix", "blocked"}:
        raise RecoveryBlocked("review_result_unreadable")
    if not result_matches(
        payload,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        phase="review_candidate",
        input_hash=input_hash,
    ):
        return None
    from react_agent.eeg_research.agentic.native_patch import apply_review_filter, satisfied_invariants
    return apply_review_filter(payload, satisfied=satisfied_invariants(workspace))


def _review_or_reuse(
    camp_dir: Path,
    workspace: Path,
    spec: dict[str, Any],
    summary: dict[str, Any],
    reviewer: Any,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Use a stored review, or call the reviewer once. None means the campaign is already blocked."""
    attempt = ensure_attempt(workspace, str(workspace.name))
    digest = source_hash(workspace)
    try:
        stored = _stored_review(
            workspace,
            candidate_id=str(workspace.name),
            attempt_id=str(attempt["attempt_id"]),
            input_hash=digest,
        )
    except RecoveryBlocked as exc:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type="RecoveryBlocked",
            detail=str(exc),
            recoverable=False,
        )
        return None
    if stored is not None:
        from react_agent.eeg_research.agentic.native_patch import enforce_requirement_review
        return enforce_requirement_review(workspace, spec, stored)
    cost_path = camp_dir / "cost.json"
    used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
    if int(state.get("max_llm_calls", 100)) - int(used) <= 0:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type="budget_exhausted",
            detail="budget_exhausted",
            recoverable=True,
        )
        return None
    identity = {
        "candidate_id": workspace.name,
        "attempt_id": attempt["attempt_id"],
        "phase": "review_candidate",
        "operation_id": operation_id(workspace, workspace.name, "review_candidate"),
        "input_hash": digest,
    }
    _attach(reviewer, **identity)
    try:
        return review(workspace, spec, summary, reviewer, getattr(reviewer, "model", ""), identity=identity)
    except LlmUnavailable as exc:
        _block_phase(camp_dir, state, phase="review_candidate", error_type=str(exc), detail=str(exc), recoverable=True)
        return None
    except Exception as exc:
        _block_phase(
            camp_dir,
            state,
            phase="review_candidate",
            error_type=type(exc).__name__,
            detail=f"{type(exc).__name__}: {exc}",
            recoverable=False,
        )
        return None


def build_services(camp: Path) -> dict[str, Any]:
    # Launch/settlement remain usable without constructing a paid role client.
    # Each role is resolved only when its service actually invokes it.
    def lazy_backend(role: str):
        client = None
        def call(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal client
            if client is None:
                client = role_backend(camp, role)
            return client(payload)
        return call

    analyst = lazy_backend("result_analyst")
    contract = public_contract(json.loads((camp / "evaluation_contract.json").read_text(encoding="utf-8")))
    summary = {key: contract.get(key) for key in ("research_scope", "primary_metric", "val_mode", "fingerprint")}

    def launch(camp_dir: Path, state: dict[str, Any], job_id: str, candidate_id: str, fidelity: str) -> dict[str, Any]:
        design = resolve_worker_design(camp_dir, state)
        if design is None:
            save_state(camp_dir, state)
            return {"job_id": job_id, "status": "blocked", "detail": state.get("detail")}
        extension = None if candidate_id == "baseline" else camp_dir / "candidates" / candidate_id / "extension"
        root = Path(design.data_root) if design.data_root else data_root()
        return jobs.start_job(
            camp_dir / "jobs" / job_id,
            design,
            candidate_id=candidate_id,
            extension=extension,
            fidelity=fidelity,
            root=root,
            protocol_path=camp_dir / "execution_protocol.json",
            training_seed=state.get("pending_training_seed"),
        )

    def settle(camp_dir: Path, job_id: str) -> dict[str, Any]:
        return jobs.reconcile(camp_dir / "jobs" / job_id)

    def do_implement(camp_dir: Path, state: dict[str, Any]) -> None:
        if state.get("status") in {"paused", "cancelled"}:
            return
        repair = state.get("repair_task") if isinstance(state.get("repair_task"), dict) else None
        repairing = bool(repair and int(repair.get("remaining") or 0) > 0 and repair.get("candidate_id"))
        candidate_id = candidate_implementation_target(camp_dir, state)
        selected_target = state.get("active_implementation_target_id")
        if selected_target and selected_target != candidate_id:
            persist_failure(camp_dir, state, phase="implement_candidate", error_type="implementation_target_changed",
                            detail="implementation_target_changed", recoverable=True)
            return
        workspace = camp_dir / "candidates" / candidate_id
        # A repair cache from another candidate cannot authorize this target.
        from react_agent.eeg_research.agentic.experiment_gate import ExperimentResolutionError, resolve_approved_experiment
        try:
            assigned = resolve_approved_experiment(
                camp_dir, target_id=candidate_id,
                spec_ref=state.get("experiment_ref") if not repairing else None,
                expected_hash=(state.get("experiment") or {}).get("spec_hash") if not repairing else None,
                attempt_id=(repair or {}).get("attempt_id"), state=state,
                action="repair" if repairing else "implement",
            )
        except ExperimentResolutionError as exc:
            _block_phase(camp_dir, state, phase="implement_candidate",
                         error_type=exc.reason, detail=exc.reason, recoverable=True)
            return
        coder = role_backend(camp, "candidate_coder")
        reviewer = role_backend(camp, "candidate_reviewer")
        spec = {"hypothesis": assigned.get("hypothesis"), "experiment": assigned}
        from react_agent.eeg_research.agentic.semantic_memory import read_json
        previous_spec = read_json(workspace / "spec.json")
        previous_attempt_id = read_json(workspace / "attempt.json").get("attempt_id")
        from react_agent.eeg_research.agentic.audit_context import audit_feedback
        from react_agent.eeg_research.agentic.handoffs import selected_read_context
        spec["audit_issues"] = [row for row in audit_feedback(camp_dir, state)["issues"]
                                if row["issue_id"] in (state.get("active_issue_ids") or [])]
        spec["artifact_reads"] = selected_read_context(camp_dir, state)
        if repairing:
            spec["repair_issues"] = repair.get("issues") or []
            spec["repair_intervention_coverage"] = repair.get("intervention_coverage")
            spec["repair_unverified_invariants"] = repair.get("unverified_invariants") or []
        workspace.mkdir(parents=True, exist_ok=True)
        from react_agent.eeg_research.agentic.run_context import load_approved_binding, persist_approved_binding
        from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved
        if load_approved_binding(camp_dir, candidate_id) is None and experiment_is_approved(assigned):
            persist_approved_binding(camp_dir, candidate_id, assigned,
                spec_ref=assigned.get("experiment_ref") or state.get("experiment_ref"))
        (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        protocol = load_protocol(camp_dir)
        (workspace / "input_spec.json").write_text(
            json.dumps(candidate_interface(protocol), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            materialize(camp_dir, workspace, assigned)
        except LineageError as exc:
            _block_phase(
                camp_dir,
                state,
                phase="implement_candidate",
                error_type="LineageError",
                detail=str(exc),
                recoverable=False,
            )
            return
        evidence_id = f"ev_impl_{candidate_id}"
        if not repairing and any(row.get("evidence_id") == evidence_id for row in state.get("evidence") or []):
            return
        max_calls = int(state.get("max_llm_calls", 100))

        def calls_left() -> int:
            cost_path = camp_dir / "cost.json"
            used = json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls", 0) if cost_path.is_file() else 0
            return max_calls - int(used)

        impl_path = workspace / "implementation.json"
        if repairing:
            if repair.get("attempt_id"):
                attempt = ensure_attempt(workspace, candidate_id)
                if attempt["attempt_id"] != repair["attempt_id"]:
                    _block_phase(camp_dir, state, phase="implement_candidate", error_type="RecoveryBlocked",
                                 detail="repair_attempt_mismatch", recoverable=False)
                    return
            else:
                attempt = rotate_attempt(workspace, candidate_id)
                repair["attempt_id"] = attempt["attempt_id"]
                save_state(camp_dir, state)
        else:
            attempt = ensure_attempt(workspace, candidate_id)
        from react_agent.eeg_research.agentic.semantic_memory import coder_delivery
        try:
            memory = coder_delivery(camp_dir, workspace, attempt["attempt_id"], spec, state=state,
                                    previous_spec=previous_spec, previous_attempt_id=previous_attempt_id)
        except (OSError, ValueError, sqlite3.Error) as exc:
            _block_phase(camp_dir, state, phase="implement_candidate", error_type="MemorySnapshotError",
                         detail=f"{type(exc).__name__}: {exc}", recoverable=True)
            return
        if memory is not None:
            spec["retrieved_memory"] = memory
            (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        from react_agent.eeg_research.agentic.artifacts import request_digest
        _attach(
            coder,
            candidate_id=candidate_id,
            attempt_id=attempt["attempt_id"],
            phase="implement_candidate",
            operation_id=operation_id(workspace, candidate_id, "implement_candidate"),
            input_hash=source_hash(workspace),
            **({"input_digest": request_digest(request=spec)}
               if spec.get("artifact_reads") or spec.get("audit_issues") or spec.get("retrieved_memory") else {}),
        )
        outcome: dict[str, Any] | None = None
        if impl_path.is_file() and not repairing:
            outcome = json.loads(impl_path.read_text(encoding="utf-8"))
        else:
            from react_agent.eeg_research.agentic.handoffs import record_read_delivery, prepare_read_delivery
            prepare_read_delivery(spec.get("artifact_reads") or [])
            digest = request_digest(request=spec)
            if spec.get("artifact_reads"):
                _attach(coder, candidate_id=candidate_id, attempt_id=attempt["attempt_id"],
                        phase="implement_candidate", operation_id=operation_id(workspace, candidate_id, "implement_candidate"),
                        input_hash=source_hash(workspace), input_digest=digest)
                record_read_delivery(camp_dir, spec["artifact_reads"], consumer="candidate_coder",
                                     request_id=attempt["attempt_id"], input_digest=digest)
            last_llm: LlmUnavailable | None = None
            for _attempt in range(IMPLEMENT_ATTEMPTS):
                try:
                    outcome = implement(
                        workspace,
                        spec,
                        coder,
                        max_steps=CODER_STEPS,
                        calls_left=calls_left,
                        reserve=2,
                        resume_repair=repairing,
                    )
                    break
                except RecoveryBlocked as exc:
                    _block_phase(
                        camp_dir,
                        state,
                        phase="implement_candidate",
                        error_type="RecoveryBlocked",
                        detail=str(exc),
                        recoverable=False,
                    )
                    return
                except LlmUnavailable as exc:
                    last_llm = exc
                except Exception as exc:
                    _block_phase(
                        camp_dir,
                        state,
                        phase="implement_candidate",
                        error_type=type(exc).__name__,
                        detail=f"{type(exc).__name__}: {exc}",
                        recoverable=False,
                    )
                    return
            if outcome is None:
                error_type = str(last_llm) if last_llm is not None else "LlmUnavailable"
                _block_phase(camp_dir, state, phase="implement_candidate", error_type=error_type, detail=error_type, recoverable=True)
                return
            if outcome.get("status") == "ready_for_review":
                attempt = ensure_attempt(workspace, candidate_id)
                impl_path.write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "attempt_id": attempt["attempt_id"],
                            "phase": "implement_candidate",
                            "operation_id": operation_id(workspace, candidate_id, "implement_candidate"),
                            "input_hash": source_hash(workspace),
                            "status": outcome["status"],
                            "steps": outcome.get("steps"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        row = {"candidate_id": candidate_id, "status": outcome["status"], "steps": outcome.get("steps"), "detail": outcome.get("detail")}
        check = outcome.get("check") if isinstance(outcome.get("check"), dict) else {}
        row["check"] = {key: check.get(key) for key in ("ok", "error", "stage", "location", "failures", "source_sha256", "check_fingerprint")}
        row["check"]["detail"] = str(check.get("detail") or "")[-2000:]
        if (state.get("failure") or {}).get("phase") in {"implement_candidate", "repair_candidate", "review_candidate"}:
            state.pop("failure", None)
            state.pop("detail", None)
        if row["status"] == "implementation_failed":
            state["detail"] = row["detail"]
        if outcome["status"] == "implementation_failed":
            used = int((repair or {}).get("used") or 0)
            limit = int(state.get("max_repairs_per_candidate", 2))
            # A failed preflight is concrete engineering feedback, even before
            # the first review. Keep the same approved candidate and bounded
            # repair policy rather than making repair an unavailable action.
            if (used < limit and outcome.get("detail") != "budget_exhausted"
                    and (check.get("error") or check.get("failures"))
                    and (workspace / "extension/eeg_candidate.py").is_file()):
                state["repair_task"] = {"candidate_id": candidate_id, "remaining": 1, "used": used + 1,
                    "issues": [{"category": "candidate_check_failure", **row["check"],
                                "correction_scope": "Repair the exact failing hook and approved config; preserve the scientific intervention."}],
                    "intervention_coverage": None, "unverified_invariants": []}
                state["experiment_failed"] = False
            else:
                state["repair_task"] = None
                state["experiment_failed"] = True
        elif repairing and outcome["status"] != "ready_for_review":
            state["repair_task"] = None
            state["experiment_failed"] = True
        if outcome["status"] == "ready_for_review":
            verdict = _review_or_reuse(camp_dir, workspace, spec, summary, reviewer, state)
            if verdict is None:
                return
            row["review"] = verdict["status"]
            row["review_format_failed"] = verdict.get("format_failed")
            if verdict["status"] == "ready":
                state["candidate_ready"] = True
                state["candidate_id"] = candidate_id
                state["repair_task"] = None
                row["status"] = "ready"
                bound = assigned
                if isinstance(bound, dict):
                    from react_agent.eeg_research.agentic.run_context import persist_approved_binding

                    persist_approved_binding(
                        camp_dir,
                        candidate_id,
                        bound,
                        spec_ref=bound.get("experiment_ref") or state.get("experiment_ref"),
                        attempt_id=attempt.get("attempt_id"),
                        source_hash=source_hash(workspace),
                    )
            elif verdict["status"] == "needs_fix" or (
                verdict["status"] == "blocked"
                and (blocking := [issue for issue in verdict.get("issues") or [] if issue.get("severity") == "blocking"])
                and all(issue.get("category") == "candidate_defect" and issue.get("smallest_correction") for issue in blocking)
            ):
                used = int((repair or {}).get("used") or 0)
                limit = int(state.get("max_repairs_per_candidate", 2))
                if used < limit:
                    state["repair_task"] = {
                        "candidate_id": candidate_id,
                        "issues": verdict.get("issues") or [],
                        "intervention_coverage": verdict.get("intervention_coverage"),
                        "unverified_invariants": verdict.get("unverified_invariants") or [],
                        "remaining": 1,
                        "used": used + 1,
                    }
                else:
                    state["repair_task"] = None
                    state["experiment_failed"] = True
                row["status"] = "review_" + verdict["status"]
            else:
                state["repair_task"] = None
                row["status"] = f"review_{verdict['status']}"
        if row["status"] != "ready":
            repairing_now = bool(
                isinstance(state.get("repair_task"), dict)
                and int((state.get("repair_task") or {}).get("remaining") or 0) > 0
            )
            if not repairing_now:
                state["experiment_failed"] = True
        existing = next(
            (index for index, item in enumerate(state.get("candidates") or []) if item.get("candidate_id") == candidate_id),
            None,
        )
        if existing is None:
            state.setdefault("candidates", []).append(row)
        else:
            state["candidates"][existing] = row
        evidence_row = {
            "evidence_id": evidence_id,
            "kind": "implementation",
            "candidate_id": candidate_id,
            "summary": row,
        }
        replaced = False
        for index, item in enumerate(state.get("evidence") or []):
            if item.get("evidence_id") == evidence_id:
                state["evidence"][index] = evidence_row
                replaced = True
                break
        if not replaced:
            state["evidence"].append(evidence_row)
        event(camp_dir, "implemented", **row)

    def _analyze_run(camp_dir: Path, state: dict[str, Any], latest: dict[str, Any], *, trigger: dict[str, Any]) -> None:
        comparison = latest.get("comparison")
        diagnostics = latest.get("diagnostics")
        if not latest.get("evaluation_valid"):
            return
        from react_agent.eeg_research.agentic.development_feedback import ensure_development_feedback
        method_mode = state.get("evaluation_mode") == "loso_method_search"
        scientific_rows = state.get("evidence") or []
        if method_mode:
            from react_agent.eeg_research.agentic.method_suite import read_aggregate, verified_suite_records
            scientific_rows = verified_suite_records(camp_dir, state)
            # Raw fold/partial rows cannot grant benchmark interpretation or
            # become controls, even if they carry a truthy evaluation label.
            if latest not in scientific_rows:
                return
            suite_result = read_aggregate(camp_dir, latest["suite_id"])
            feedback = {"status": "verified_complete_method_suite", "scope": "method_development_benchmark",
                        "suite_ref": latest["suite_ref"], "suite_hash": latest["suite_hash"],
                        "complete_fold_count": 10, "required_fold_count": 10, "independent_seed_count": 1}
        else:
            feedback = ensure_development_feedback(camp_dir, state, latest)
        bound_hypothesis = latest.get("hypothesis")
        bound_experiment = latest.get("experiment")
        if bound_hypothesis is None:
            for row in state.get("candidates") or []:
                if row.get("candidate_id") == latest.get("candidate_id") and row.get("hypothesis"):
                    bound_hypothesis = row.get("hypothesis")
                    bound_experiment = row.get("experiment")
        payload = {
            "analysis_trigger": trigger,
            "latest": {key: latest.get(key) for key in ("evidence_id", "job_id", "candidate_id", "fidelity", "fixed_bank_top1", "gallery_size", "delta_vs_control_pp", "control_id", "seed",
                "evaluation_valid", "reason", "execution_succeeded", "implementation_failure", "job_status", "checkpoint_id", "source_hash", "config_hash", "spec_hash", "contract_fingerprint")},
            "comparison": comparison,
            "diagnostics": diagnostics,
            "hypothesis": bound_hypothesis,
            "experiment": bound_experiment,
            "hypothesis_binding_missing": bound_hypothesis is None,
            "development_feedback": feedback,
            "controls": [
                {key: row.get(key) for key in ("evidence_id", "candidate_id", "fidelity", "fixed_bank_top1", "seed")}
                for row in scientific_rows
                if row.get("candidate_id") == "baseline" and row.get("evaluation_valid")
            ],
        }
        from react_agent.eeg_research.agentic.artifacts import request_digest, resolve_verified_artifact
        from react_agent.eeg_research.agentic.handoffs import development_view, load_analysis_views
        from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task
        from react_agent.eeg_research.agentic.schemas import ResultAnalysis

        payload["run_identity"] = {key: latest.get(key) for key in (
            "job_id", "source_hash", "config_hash", "spec_hash", "checkpoint_id", "contract_fingerprint", "result_hash")}
        from react_agent.eeg_research.agentic.handoffs import candidate_context
        payload["candidate_context"] = candidate_context(camp_dir, str(latest["candidate_id"]))
        from react_agent.eeg_research.agentic.measurement_context import evaluation_population_context
        payload["evaluation_population"] = evaluation_population_context(load_protocol(camp_dir))
        from react_agent.eeg_research.agentic.handoffs import verified_encoder_structural_facts
        payload["verified_encoder_structural_facts"] = verified_encoder_structural_facts(camp_dir, state,
            target_ids={str(latest["candidate_id"])})
        from react_agent.eeg_research.agentic.research_progress import verified_training_diagnostic_facts
        payload["verified_training_diagnostics"] = verified_training_diagnostic_facts(camp_dir, state,
            str(latest["candidate_id"]), str(latest.get("source_hash")), str(latest.get("spec_hash")))
        from react_agent.eeg_research.agentic.research_progress import objective_effectiveness_findings
        payload["objective_effectiveness_findings"] = objective_effectiveness_findings(camp_dir,state,
            str(latest["candidate_id"]),str(latest.get("source_hash")),str(latest.get("spec_hash")))
        from react_agent.eeg_research.agentic.research_progress import objective_semantic_findings
        payload["objective_semantic_findings"] = objective_semantic_findings(camp_dir,state,
            str(latest["candidate_id"]),str(latest.get("source_hash")),str(latest.get("spec_hash")))
        payload["is_frozen_baseline"] = latest["candidate_id"] == "baseline"
        from react_agent.eeg_research.agentic.handoffs import analysis_experiment_binding
        from react_agent.eeg_research.agentic.research_progress import verified_diagnostic_facts
        binding = analysis_experiment_binding(camp_dir, latest)
        payload["hypothesis_binding"] = binding
        if not payload["is_frozen_baseline"]:
            payload["hypothesis"] = binding.get("hypothesis")
            payload["experiment"] = binding.get("experiment")
            payload["hypothesis_binding_missing"] = binding.get("status") != "verified"
        same_revision = [row for row in scientific_rows
            if row.get("evaluation_valid") is True and row.get("candidate_id") == latest["candidate_id"]
            and row.get("fidelity") == latest["fidelity"] and row.get("source_hash") == latest.get("source_hash")
            and row.get("spec_hash") == latest.get("spec_hash")
            and row.get("execution_fingerprint") == latest.get("execution_fingerprint")]
        # Repeated runs of the same actual seed are not independent seeds.
        same_revision = list({row.get("seed"):row for row in same_revision}.values())
        control_ids = {row.get("control_id") for row in same_revision if row.get("control_id")}
        parent_experiment = binding.get("experiment") or {}
        parent_id = parent_experiment.get("ablation_of_candidate_id") or parent_experiment.get("parent_candidate_id")
        if parent_id == "baseline":
            parent_id = None
        parent_binding = None
        if parent_id:
            from react_agent.eeg_research.agentic.run_context import load_approved_binding
            parent_binding = load_approved_binding(camp_dir, str(parent_id)) or {}
        paired_seeds = {row.get("seed") for row in same_revision}
        matched = list(same_revision)
        for row in scientific_rows:
            if row.get("evaluation_valid") is not True or row.get("fidelity") != latest["fidelity"]:
                continue
            if row.get("execution_fingerprint") != latest.get("execution_fingerprint"):
                continue
            if row.get("evidence_id") in control_ids:
                matched.append(row)
            elif (parent_binding and row.get("candidate_id") == parent_id and row.get("seed") in paired_seeds
                  and row.get("source_hash") == parent_binding.get("source_hash")
                  and row.get("spec_hash") == parent_binding.get("spec_hash")):
                matched.append(row)
        from react_agent.eeg_research.agentic.development_feedback import _ensure_report
        payload["matched_run_history"] = []
        for row in matched:
            if latest["fidelity"] == "full" and not method_mode:
                try:
                    _ensure_report(camp_dir, state, row)
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                    pass  # An unavailable diagnostic is explicit in supplied coverage.
            payload["matched_run_history"].append({key:row.get(key) for key in (
                "evidence_id", "job_id", "candidate_id", "fidelity", "seed", "source_hash", "spec_hash", "checkpoint_id",
                "execution_fingerprint", "evaluation_valid", "fixed_bank_top1", "control_id", "comparison", "promotion")})
        declared = [int(seed) for seed in (state.get("training_seeds") or json.loads((camp_dir / "goal.json").read_text()).get("training_seeds") or [])]
        observed = sorted(seed for seed in paired_seeds if isinstance(seed,int))
        payload["candidate_seed_coverage"] = {"fidelity":latest["fidelity"], "source_hash":latest.get("source_hash"),
            "spec_hash":latest.get("spec_hash"), "observed_training_seeds":observed,
            "declared_training_seeds":declared,"missing_training_seeds":[seed for seed in declared if seed not in observed]}
        # Fixed-gallery top1 is a bounded fraction. Missing seeds have not been
        # measured; a negative observed seed is not a bound on their outcomes.
        import math
        required = sorted(set(declared))
        values = {row["seed"]: float(row["fixed_bank_top1"]) for row in same_revision
                  if row.get("seed") in required and isinstance(row.get("fixed_bank_top1"), (int, float))
                  and math.isfinite(row["fixed_bank_top1"]) and 0 <= row["fixed_bank_top1"] <= 1}
        absent = [seed for seed in required if seed not in values]
        payload["candidate_seed_coverage"]["mean_arithmetic_bounds"] = {
            "metric": "fixed_bank_top1", "metric_range": [0.0, 1.0],
            "required_seed_count": len(required), "measured_scores_by_seed": values,
            "unmeasured_seed_ids": absent, "unmeasured_count": len(absent),
            "current_observed_mean": sum(values.values()) / len(values) if values else None,
            "minimum_possible_complete_mean": sum(values.values()) / len(required) if required else None,
            "maximum_possible_complete_mean": (sum(values.values()) + len(absent)) / len(required) if required else None,
            "complete_mean_measured": bool(required) and not absent,
            "scope": "Read-only arithmetic bounds from measured same-source/spec/fidelity seeds and top1 range. Missing-seed performance is unknown; bounds are not forecasts, measurements, promotion or authorization."
        }
        payload["verified_development_facts"] = verified_diagnostic_facts(camp_dir, state,
            job_ids={str(row["job_id"]) for row in matched if row.get("job_id")},max_facts=12)
        payload["development_feedback_history"] = [] if method_mode else [
            {"job_id":row.get("job_id"), "training_seed":row.get("seed"),
             **ensure_development_feedback(camp_dir, state, row)} for row in same_revision]
        from react_agent.eeg_research.agentic.analysis_coverage import coverage_summary
        payload["verified_run_coverage"] = coverage_summary(same_revision, matched,
            payload["verified_development_facts"], payload["development_feedback_history"])
        payload["run_artifacts"] = []
        job_dir = Path(str(latest.get("job_dir") or ""))
        if latest.get("job_dir") and job_dir.resolve().is_relative_to(camp_dir.resolve()):
            from react_agent.eeg_research.agentic.artifacts import file_digest
            for name in ("run_record.json", "metrics.json", "selected_checkpoint.json", "hook_consumed.json", "capabilities_used.json", "source_binding.json", "frozen_run_spec.json"):
                path = job_dir / name
                if path.is_file():
                    content = path.read_text(encoding="utf-8")
                    payload["run_artifacts"].append({"ref": str(path), "content_hash": file_digest(path),
                        "payload": json.loads(content) if len(content) <= 30000 else None,
                        "missing_inputs": [] if len(content) <= 30000 else ["artifact_exceeds_context_limit"]})
        if method_mode:
            payload.update(evaluation_mode="loso_method_search", primary_metric="benchmark.loso_mean_fixed_gallery_top1",
                           method_suite=suite_result, benchmark_permission={"authorized": True, "partial_feedback": False},
                           scientific_unit="one_complete_method_suite_seed0", independent_seed_count=1,
                           verified_training_diagnostics=latest.get("diagnostics"),
                           evaluation_population={"unit": "ten_subjects_equal_weight", "query_count_per_subject": 200,
                                                  "candidate_count_per_subject": 200, "required_fold_count": 10},
                           limitations=["method_development_benchmark_used_for_search", "no_independent_final_test",
                                        "one_seed_not_replicated_or_statistically_significant"])
            # A suite has ten original jobs, rather than the single job_dir
            # expected by legacy coverage_summary. Report its actual native
            # coverage without inventing missing job IDs or ten seed records.
            payload["verified_run_coverage"] = {
                "authority": "native_verified_complete_method_suite",
                "scope": "method_development_benchmark", "suite_id": latest["suite_id"],
                "suite_ref": latest["suite_ref"], "suite_hash": latest["suite_hash"],
                "complete_fold_count": suite_result["coverage"], "required_fold_count": 10,
                "independent_seed_count": 1, "replicated": False,
                "matched_complete_baseline_pair": bool((comparison or {}).get("comparable")),
                "folds": [{key: score[key] for key in (
                    "fold_id", "held_out_subject", "train_subjects", "seed", "fidelity",
                    "checkpoint_hash", "freeze_ref", "freeze_sha256")}
                    for score in suite_result["scores"]],
                "diagnostic_availability": "See the exact per-fold diagnostics and their reported sampling coverage; full-query rank is not implied.",
            }
            payload["latest"].update(metric_scope="method_development_benchmark", suite_id=latest["suite_id"],
                                     suite_hash=latest["suite_hash"], method_revision=latest["method_revision"])
        payload = development_view(payload)
        digest = request_digest(request=payload)
        for row in state.get("evidence") or []:
            if row.get("kind") == "analysis" and row.get("run_evidence_id") == latest["evidence_id"] and row.get("analysis_input_digest") == digest:
                try:
                    artifact = resolve_verified_artifact(camp_dir, row["analysis_artifact_id"])
                    stored = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
                except (OSError, ValueError, KeyError):
                    continue
                if stored.get("status") == "completed":
                    if not method_mode:
                        return
                    from react_agent.eeg_research.agentic.method_suite import scientific_feedback_records
                    if "curation" in scientific_feedback_records(camp_dir, state, latest):
                        return
                    # Native begin-task reuses this completed Analyst envelope.
                    # Continue its missing curator stage without new transport.
        analysis_inputs = [Path(latest["suite_ref"])] if method_mode else []
        task = begin_role_task(camp_dir, role="result_analyst", inputs=analysis_inputs,
                               candidate_id=latest.get("candidate_id"), request=payload)
        task_payload = {**payload, "task_id": task["task_id"], "attempt_id": task["attempt_id"], "input_digest": task["input_digest"]}
        billed = 0
        try:
            raw_reply = analyst(task_payload)
            billed += 1
            body = raw_reply.get("payload") if isinstance(raw_reply.get("payload"), dict) else raw_reply
            body = {key: value for key, value in body.items() if key not in {
                "schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "status", "artifact_refs",
                "method_benchmark_origin"}}
            reply = ResultAnalysis.model_validate(body).model_dump()
            role_status = str(raw_reply.get("status") or "completed")
            if role_status not in {"completed", "partial", "failed", "blocked"}:
                role_status = "partial"
        except LlmUnavailable as exc:
            if method_mode and str(exc) in _METHOD_ROLE_HARD_BLOCKS:
                _sync_ledger(camp_dir, state)
                raise
            billed += 1
            reply = {"status": "failed", "role_failed": True, "hypothesis_assessment": None, "summary_zh": f"分析未完成：{exc}"}
            role_status = "failed"
        except ValueError as exc:
            reply = {"hypothesis_assessment": None, "summary_zh": f"分析格式未通过运行时验证：{type(exc).__name__}"}
            role_status = "failed"
        target = camp_dir / "analyses" / f"{latest['evidence_id']}_{task['task_id']}.json"
        benchmark_origin = {}
        if method_mode:
            from react_agent.eeg_research.agentic.handoffs import method_benchmark_origin
            benchmark_origin = {"method_benchmark_origin": method_benchmark_origin(latest)}
        envelope = finish_role_task(camp_dir, task, {**reply, "status": role_status,
            "prompt_hash": raw_reply.get("prompt_hash", "") if "raw_reply" in locals() else "", **benchmark_origin},
            kind="analysis", path=target, candidate_id=latest.get("candidate_id"))
        analysis_row = {
                "evidence_id": f"ev_analysis_{task['task_id']}",
                "kind": "analysis",
                "candidate_id": latest.get("candidate_id"),
                "run_evidence_id": latest["evidence_id"],
                "analysis_input_digest": digest,
                "analysis_artifact_id": envelope["artifact_refs"][-1],
                "status": role_status,
                "summary": {key: reply.get(key) for key in ("hypothesis_assessment", "summary_zh", "suggested_next_actions")},
            }
        previous_analysis = next((index for index, item in enumerate(state["evidence"])
                                  if item.get("evidence_id") == analysis_row["evidence_id"]), None)
        if previous_analysis is None:
            state["evidence"].append(analysis_row)
        else:
            state["evidence"][previous_analysis] = analysis_row
        if method_mode and role_status != "completed":
            # An incomplete Analyst cannot authorize a method-level lesson.
            # Recovery retries that role with its actual feedback before curation.
            _sync_ledger(camp_dir, state)
            return
        from react_agent.eeg_research.agentic.memory import EpisodeStore

        store = EpisodeStore(camp_dir)
        try:
            curator = role_backend(camp, "memory_curator")
            from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy

            episodes = [item for item in store.list_episodes()
                        if item.get("job_id") == latest.get("job_id") or item.get("candidate_id") == latest.get("candidate_id")][-8:]
            curation_context = development_view({
                "episodes": episodes,
                "design": bound_experiment,
                "run": latest,
                "comparison": comparison,
                "diagnostics": diagnostics,
                "analyses": load_analysis_views(camp_dir, state, target=latest.get("candidate_id")),
                "confirmation": latest.get("promotion", {}).get("confirmation") if isinstance(latest.get("promotion"), dict) else None,
                "confirmation_policy": load_confirmation_policy(camp_dir),
                "missing_inputs": ["confirmation"] if not (latest.get("promotion") or {}).get("confirmation") else [],
                "related_lessons": store.list_lessons()[-8:],
            })
            if method_mode:
                from react_agent.eeg_research.agentic.method_suite import manifest
                curation_context.update(
                    evaluation_mode="loso_method_search",
                    primary_metric="benchmark.loso_mean_fixed_gallery_top1",
                    scientific_unit="one_complete_method_suite_seed0",
                    contract_fingerprint=state["contract_fingerprint"],
                    benchmark_permission=manifest(camp_dir)["benchmark_permission"],
                    lesson_conditions_required={"fidelity": "full", "evaluation_identity": state["contract_fingerprint"]},
                    method_suite_identity={"suite_id": latest["suite_id"], "suite_hash": latest["suite_hash"],
                                           "method_revision": latest["method_revision"], "coverage": 10, "seed": 0},
                    evidence_level="exploratory_result", replicated=False,
                    missing_inputs=[],
                    limitations=["benchmark_used_for_method_development", "one_seed_not_replicated_or_statistically_significant"],
                )
            from react_agent.eeg_research.agentic.embedding import memory_config
            memory = memory_config(camp_dir)
            if memory.enabled and memory.skills_enabled and role_status == "completed":
                from react_agent.eeg_research.agentic.skill_memory import extend_curation_context
                try:
                    curation_context = extend_curation_context(camp_dir, curation_context, envelope, state=state)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    curation_context["skills_unavailable_reason"] = type(exc).__name__
            curation_inputs = [camp_dir / "goal.json"]
            if method_mode:
                curation_inputs.extend([Path(latest["suite_ref"]), target])
            compaction_artifacts = []
            prepare = getattr(curator, "prepare_context", None)
            if method_mode and callable(prepare):
                curation_context = prepare(curation_context, inputs=curation_inputs)
                receipt = curation_context.get("context_compaction")
                if receipt:
                    # Original files remain dependencies. Also pin the exact
                    # registered model summary actually delivered to curator.
                    compaction_artifacts.append(
                        {"artifact_id": receipt["artifact_id"], "sha256": receipt["sha256"]})
            task = begin_role_task(camp_dir, role="memory_curator", inputs=curation_inputs,
                                   artifacts=compaction_artifacts, request=curation_context)
            try:
                proposal = curator(
                    {
                        **curation_context,
                        "task_id": task["task_id"],
                        "attempt_id": task["attempt_id"],
                        "input_digest": task["input_digest"],
                    }
                )
                billed += 1
            except LlmUnavailable as exc:
                billed += 1
                store.mark_pending_curation("curator_unavailable")
                finish_role_task(camp_dir, task, {"status": "failed", "summary_zh": f"经验提议未完成：{exc}"},
                                 kind="lessons", path=camp_dir / "memory" / f"lessons_{task['task_id']}.json")
                if method_mode and str(exc) in _METHOD_ROLE_HARD_BLOCKS:
                    raise
            else:
                proposal_status = str(proposal.get("status") or "completed")
                if proposal_status not in {"completed", "partial", "failed", "blocked"}:
                    proposal_status = "partial"
                accepted = store.accept_lessons(_lesson_proposal(proposal)) if proposal_status == "completed" else {"accepted": [], "rejected": []}
                skills_metadata = {}
                if memory.enabled and memory.skills_enabled:
                    skills_result = {"accepted": [], "rejected": []}
                    body = _lesson_proposal(proposal)
                    if (proposal_status == "completed" and role_status == "completed"
                            and body.get("status", "completed") == "completed"):
                        from react_agent.eeg_research.agentic.skill_memory import accept_skills
                        skills_result = accept_skills(camp_dir, body.get("proposed_skills") or [],
                                                     curation_context.get("allowed_source_refs") or {})
                    skills_metadata["skills_result"] = skills_result
                if memory.enabled and proposal_status == "completed":
                    # Index failures are siblings of successful curation, never a reason to rerun it.
                    from react_agent.eeg_research.agentic.semantic_memory import retrieve
                    try:
                        indexed = retrieve(camp_dir, "", state=state)
                        skills_metadata["memory_indexing"] = {k: indexed.get(k) for k in ("backend", "degraded", "reason", "indexing")}
                    except Exception as exc:
                        skills_metadata["memory_indexing"] = {"status": "pending", "reason": type(exc).__name__}
                if proposal_status != "completed":
                    store.mark_pending_curation("curator_incomplete")
                finish_role_task(
                    camp_dir,
                    task,
                    {"status": proposal_status, "summary_zh": "已校验条件化经验提议", "proposal": _lesson_proposal(proposal), "accepted": accepted,
                     **skills_metadata, **benchmark_origin},
                    kind="lessons",
                    path=camp_dir / "memory" / f"lessons_{task['task_id']}.json",
                )
        except LlmUnavailable as exc:
            store.mark_pending_curation("curator_unavailable")
            if method_mode and str(exc) in _METHOD_ROLE_HARD_BLOCKS:
                _sync_ledger(camp_dir, state)
                raise
        except Exception:
            store.mark_pending_curation("curator_failed")
        _sync_ledger(camp_dir, state)
        if not (camp_dir / "cost.json").is_file():
            state["llm_calls"] = int(state.get("llm_calls", 0)) + billed
            state["llm_calls_left"] = int(state.get("llm_calls_left", 0)) - billed

    def analyze(camp_dir: Path, state: dict[str, Any]) -> None:
        scientific_rows = state.get("evidence") or []
        if state.get("evaluation_mode") == "loso_method_search":
            from react_agent.eeg_research.agentic.method_suite import verified_suite_records
            scientific_rows = verified_suite_records(camp_dir, state)
        latest = next((row for row in reversed(scientific_rows)
                       if row.get("fidelity") and row.get("candidate_id") and "evaluation_valid" in row), None)
        if latest is None:
            return
        from react_agent.eeg_research.agentic.analysis_dependencies import analysis_targets
        for row in analysis_targets(camp_dir, {**state, "evidence": scientific_rows}, latest):
            trigger = {"kind":"settled_run" if row["evidence_id"] == latest["evidence_id"] else "new_parent_full_result",
                       "upstream_evidence_id":latest["evidence_id"], "upstream_job_id":latest.get("job_id")}
            _analyze_run(camp_dir, state, row, trigger=trigger)

    def _lazy(role: str):
        def call(payload: dict[str, Any]) -> dict[str, Any]:
            return role_backend(camp, role)(payload)

        return call

    return {
        "launch": launch,
        "settle": settle,
        "implement": do_implement,
        "analyze": analyze,
        "librarian": _lazy("research_librarian"),
        "designer": _lazy("experiment_designer"),
        "auditor": _lazy("result_auditor"),
    }


def run_worker(camp: Path, *, poll_seconds: float = 30.0, max_ticks: int = 200) -> dict[str, Any]:
    """Advance until a terminal state. A second worker for the same campaign exits at once."""
    lock_path = camp / "worker.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"status": "worker_already_running"}
    (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
    state = load_state(camp)
    if resolve_worker_design(camp, state) is None:
        if state.get("status") not in {"paused", "cancelled"}:
            save_state(camp, state)
        return state
    align_interrupt(camp)
    state = load_state(camp)
    if state.get("status") in _TERMINAL:
        return state
    try:
        planner = role_backend(camp, "research_planner")
        services = build_services(camp)
    except LlmUnavailable as exc:
        state = load_state(camp)
        if state.get("status") not in {"paused", "cancelled"}:
            state["status"] = "blocked"
            state["detail"] = str(exc)
            save_state(camp, state)
        return state
    for tick_index in range(max_ticks):
        state = tick(camp, planner, services=services)
        (camp / "worker.json").write_text(
            json.dumps({"pid": os.getpid(), "heartbeat": time.time(), "tick": tick_index, "resumable": bool(state.get("live_job"))}),
            encoding="utf-8",
        )
        if state.get("status") in _TERMINAL:
            break
        if state.get("status") == "training":
            time.sleep(poll_seconds)
    else:
        state = load_state(camp)
        if state.get("live_job"):
            (camp / "worker.json").write_text(
                json.dumps({"pid": os.getpid(), "heartbeat": time.time(), "resumable": True, "detail": "worker_tick_limit"}),
                encoding="utf-8",
            )
            event(camp, "worker_suspended", reason="tick_limit", live_job=state.get("live_job"))
    return state
