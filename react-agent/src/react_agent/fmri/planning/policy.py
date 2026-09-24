"""PlannedPolicy: persist a Plan, execute one ready step, replan only on triggers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from react_agent.fmri.llm.deepseek import DeepSeekCallError, DeepSeekParseError
from react_agent.fmri.planning.catalog import tool_catalog
from react_agent.fmri.planning.schemas import Plan, PlanProposal, PlanningContext
from react_agent.fmri.planning.service import Planner
from react_agent.fmri.planning.validator import PlanValidationError, ready_steps, validate_proposal
from react_agent.fmri.progress import emit
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import Decision, SampleSpec, ToolResult
from react_agent.fmri.tools.base import RunContext


class PlannedPolicy:
    """Translate a validated plan into the existing Decision action."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.planner = Planner(runtime.backend, runtime.config)

    async def next_action(
        self,
        *,
        state: dict[str, Any],
        sample: SampleSpec,
        context: RunContext,
        results: list[ToolResult],
        candidates: list[str],
        completed: set[str],
        events: Path,
        memory_bundle: dict[str, Any],
    ) -> tuple[Decision, dict[str, Any]]:
        extras: dict[str, Any] = {}
        catalog = tool_catalog(
            self.runtime.registry,
            sample=sample,
            context=context,
            results=results,
            enabled=self.runtime.config.enabled_tools,
            completed=completed,
        )
        resources = _resource_map(self.runtime.config, sample, Path(state["base_dir"]))
        ledger = state.get("question_ledger") or {}
        required = list(self.runtime.config.check_profile.required_questions or [])
        if not self.runtime.config.diagnostic.enabled:
            if self.runtime.config.planning.require_gray_question_for_non_control:
                if "gray_contrast" not in required and not (
                    sample.resources and sample.resources.is_gray_control
                ):
                    required.append("gray_contrast")
            if self.runtime.config.planning.require_temporal_description_for_non_control:
                if "temporal" not in required and not (
                    sample.resources and sample.resources.is_gray_control
                ):
                    required.append("temporal")
        findings = {
            f.code
            for r in results
            if r.execution_status == "success"
            for f in r.findings
        }
        metrics = {r.tool_name: r.metrics for r in results if r.execution_status == "success"}
        thresholds = {"spike_ratio_heuristic": self.runtime.config.temporal_diagnostics.spike_ratio_heuristic}
        evidence_ids = {
            f"{r.tool_name}:{r.result_id}"
            for r in results
            if r.execution_status == "success"
        }
        has_ref = bool(
            (metrics.get("cortex_mae") or {}).get("reference_score_assessed")
            or (self.runtime.config.reference.stats_path and "reference_distribution" in candidates)
        )

        plan = _load_plan(state)
        trigger = _replan_trigger(state, plan, results, required, ledger, catalog)
        if plan is None:
            trigger = "initial"
        if trigger:
            try:
                plan, extras = await self._make_plan(
                    state=state,
                    sample=sample,
                    results=results,
                    catalog=catalog,
                    resources=resources,
                    memory_bundle=memory_bundle,
                    required=required,
                    evidence_ids=evidence_ids,
                    has_ref=has_ref,
                    events=events,
                    trigger=trigger,
                    current=plan,
                    completed=completed,
                    ledger=ledger,
                )
            except PlanValidationError as exc:
                return _validation_failure(exc, events), extras
            except Exception as exc:  # noqa: BLE001
                return _api_failure(self.runtime, state, exc, events), extras

        assert plan is not None
        extras["current_plan"] = plan.model_dump()
        ready = ready_steps(
            plan,
            ledger=ledger,
            resources=resources,
            findings=findings,
            metrics=metrics,
            thresholds=thresholds,
        )
        ready = [s for s in ready if s.tool_id in candidates and s.tool_id not in completed]
        if ready:
            step = ready[0]
            decision = Decision(
                action="run_tool",
                tool_name=step.tool_id,
                reason=step.brief_reason or f"plan step {step.step_id}",
                question_to_resolve=step.question_id,
                expected_observation=step.expected_evidence_type,
                evidence_refs=step.supporting_evidence_ids,
                source="planned",
            )
            extras["active_step_id"] = step.step_id
            emit(
                {
                    "type": "plan.step_started",
                    "step_id": step.step_id,
                    "tool_id": step.tool_id,
                },
                path=events,
            )
            return decision, extras

        open_required = [
            qid for qid in required if (ledger.get(qid) or {}).get("status") in {"untested", "unassessed"}
        ]
        if (
            getattr(self.runtime.config, "diagnostic", None)
            and self.runtime.config.diagnostic.enabled
            and ledger
        ):
            from react_agent.fmri.diagnostic.routing import derive_tickets
            from react_agent.fmri.diagnostic.stop_gate import evaluate_stop
            from react_agent.fmri.schemas import QuestionRecord

            tickets = derive_tickets(results, self.runtime.config)
            records = [
                QuestionRecord.model_validate(row) if isinstance(row, dict) else row
                for row in ledger.values()
            ]
            gate = evaluate_stop(
                records,
                tickets,
                budget_ok=self.runtime.budget.data["tool_calls"] < self.runtime.config.budgets.max_tool_calls,
                executable_actions=candidates,
                planning_failed=bool(extras.get("revise_rejected")),
            )
            extras["followups"] = tickets
            extras["stop_gate"] = gate
            if not gate.get("allow_stop"):
                extras["stop_rejected"] = True
                return (
                    Decision(
                        action="stop",
                        stop_reason="abstain",
                        reason="stop_rejected:" + ",".join(str(x) for x in (gate.get("missing_question_ids") or [])),
                        source="planned",
                    ),
                    extras,
                )
            return (
                Decision(
                    action="stop",
                    stop_reason=gate["stop_reason"],
                    reason=gate["stop_reason"],
                    source="planned",
                ),
                extras,
            )
        if open_required:
            return (
                Decision(
                    action="stop",
                    stop_reason="abstain",
                    reason="Required questions remain but no ready planned step.",
                    source="planned",
                ),
                extras,
            )
        reason = "Plan has no remaining ready step."
        if plan.stop_request and plan.stop_request.reason:
            reason = plan.stop_request.reason
        return (
            Decision(
                action="stop",
                stop_reason="configured_checks_complete",
                reason=reason,
                source="planned",
            ),
            extras,
        )

    async def _make_plan(
        self,
        *,
        state: dict[str, Any],
        sample: SampleSpec,
        results: list[ToolResult],
        catalog: list[dict[str, Any]],
        resources: dict[str, Any],
        memory_bundle: dict[str, Any],
        required: list[str],
        evidence_ids: set[str],
        has_ref: bool,
        events: Path,
        trigger: str,
        current: Plan | None,
        completed: set[str] | None = None,
        ledger: dict[str, Any] | None = None,
    ) -> tuple[Plan, dict[str, Any]]:
        completed = completed or set()
        ledger = ledger or {}
        self._last_repair = False
        phase = "revise" if current is not None else "create"
        ctx = PlanningContext(
            task={
                "goal": self.runtime.config.planning.run_goal,
                "claim_scope": "numeric_consistency_only",
                "required_questions": required,
                "generation_profile_id": sample.generation_profile_id,
                "sample_id": sample.sample_id,
            },
            evidence=[
                {
                    "evidence_id": f"{r.tool_name}:{r.result_id}",
                    "tool_name": r.tool_name,
                    "status": r.execution_status,
                    "findings": [f.model_dump() for f in r.findings],
                    "metrics": {
                        k: v
                        for k, v in (r.metrics or {}).items()
                        if not isinstance(v, list) or len(v) < 8
                    },
                }
                for r in results
            ],
            tool_catalog=catalog,
            resources=resources,
            current_plan=current.model_dump() if current else None,
            memory_bundle=memory_bundle,
            budget=self.runtime.budget.snapshot(),
            request_mode=phase,
            trigger=trigger,
        )
        emit({"type": "api.requested", "role": "planner" if phase == "create" else "replanner"}, path=events)
        self.runtime.budget.preflight_lm()
        try:
            if phase == "create":
                proposal, usage = await self.planner.create_plan(ctx)
            else:
                proposal, usage = await self.planner.revise_plan(ctx)
            self.runtime.budget.record_lm(usage)
            emit(
                {
                    "type": "api.succeeded",
                    "role": usage.role,
                    "call_id": usage.call_id,
                    "requested_model": usage.requested_model,
                    "response_model": usage.response_model,
                },
                path=events,
            )
        except DeepSeekParseError as exc:
            self.runtime.budget.record_lm(exc.usage, repair=True)
            if self.runtime.config.budgets.max_json_repairs >= 1:
                self.runtime.budget.preflight_lm()
                try:
                    if phase == "create":
                        proposal, usage = await self.planner.create_plan(ctx)
                    else:
                        proposal, usage = await self.planner.revise_plan(ctx)
                    self.runtime.budget.record_lm(usage, repair=True)
                except (DeepSeekParseError, DeepSeekCallError):
                    emit({"type": "api.failed", "role": phase}, path=events)
                    raise
            else:
                emit({"type": "api.failed", "role": phase}, path=events)
                raise
        except DeepSeekCallError:
            emit({"type": "api.failed", "role": phase}, path=events)
            raise

        try:
            validate_proposal(
                proposal,
                catalog=catalog,
                required_questions=required,
                remaining_lm_calls=self.runtime.config.budgets.max_lm_calls
                - self.runtime.budget.data["lm_calls"],
                remaining_tool_calls=self.runtime.config.budgets.max_tool_calls
                - self.runtime.budget.data["tool_calls"],
                evidence_ids=evidence_ids,
                resources=resources,
                run_goal=self.runtime.config.planning.run_goal,
                has_reference_score=has_ref,
                completed_tools=completed,
                ledger=ledger,
            )
        except PlanValidationError as exc:
            emit({"type": "plan.rejected", "error": str(exc)}, path=events)
            if self.runtime.config.budgets.max_json_repairs < 1:
                raise
            ctx = ctx.model_copy(update={"trigger": f"validator:{exc}"})
            self.runtime.budget.preflight_lm()
            if phase == "create":
                proposal, usage = await self.planner.create_plan(ctx)
            else:
                proposal, usage = await self.planner.revise_plan(ctx)
            self.runtime.budget.record_lm(usage, repair=True)
            self._last_repair = True
            try:
                validate_proposal(
                    proposal,
                    catalog=catalog,
                    required_questions=required,
                    remaining_lm_calls=self.runtime.config.budgets.max_lm_calls
                    - self.runtime.budget.data["lm_calls"],
                    remaining_tool_calls=self.runtime.config.budgets.max_tool_calls
                    - self.runtime.budget.data["tool_calls"],
                    evidence_ids=evidence_ids,
                    resources=resources,
                    run_goal=self.runtime.config.planning.run_goal,
                    has_reference_score=has_ref,
                    completed_tools=completed,
                    ledger=ledger,
                )
            except PlanValidationError as repair_exc:
                if current is not None:
                    emit({"type": "plan.rejected", "error": "revise_kept_current"}, path=events)
                    return current, {
                        "current_plan": current.model_dump(),
                        "replan_trigger": trigger,
                        "revise_rejected": True,
                    }
                emit({"type": "plan.rejected", "error": str(repair_exc)}, path=events)
                raise
        if current is not None:
            for old in current.steps:
                if old.status != "completed":
                    continue
                for new in proposal.steps:
                    if new.step_id == old.step_id:
                        new.status = "completed"
        plan = _runtime_plan(proposal, sample, current)
        self.runtime.budget.record_accepted_plan(
            phase=phase,
            repaired=bool(getattr(self, "_last_repair", False)),
        )
        emit(
            {
                "type": "plan.created" if phase == "create" else "plan.revised",
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "trigger": trigger,
            },
            path=events,
        )
        emit({"type": "plan.validated", "plan_id": plan.plan_id}, path=events)
        extras = {"current_plan": plan.model_dump(), "replan_trigger": trigger}
        _write_plan_artifacts(Path(state["out_dir"]) / state.get("safe_sample_id", "sample"), plan)
        return plan, extras


def update_plan_after_tool(plan: dict[str, Any] | None, tool_name: str, status: str) -> dict[str, Any] | None:
    if not plan:
        return plan
    mapped = {"success": "completed", "skipped": "skipped", "error": "failed"}
    step_status = mapped.get(status, status)
    for step in plan.get("steps") or []:
        if step.get("tool_id") == tool_name and step.get("status") in {"pending", "running"}:
            step["status"] = step_status
            break
    return plan


def _runtime_plan(proposal: PlanProposal, sample: SampleSpec, current: Plan | None) -> Plan:
    now = datetime.now(timezone.utc).isoformat()
    blob = json.dumps(
        {
            "sample": sample.sample_id,
            "profile": sample.generation_profile_id,
            "steps": [s.step_id for s in proposal.steps],
        },
        sort_keys=True,
    )
    return Plan(
        **proposal.model_dump(),
        plan_id=current.plan_id if current else uuid4().hex[:12],
        revision=(current.revision + 1) if current else 0,
        created_at=now,
        input_hash=hashlib.sha256(blob.encode()).hexdigest()[:16],
        context_hash=hashlib.sha256(blob.encode()).hexdigest()[16:32],
        source="deepseek" if current is None or current.source == "deepseek" else current.source,
    )


def _load_plan(state: dict[str, Any]) -> Plan | None:
    raw = state.get("current_plan")
    if not raw:
        return None
    return Plan.model_validate(raw)


def _replan_trigger(
    state: dict[str, Any],
    plan: Plan | None,
    results: list[ToolResult],
    required: list[str],
    ledger: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> str | None:
    if plan is None:
        return "initial"
    if int(state.get("replan_count") or 0) >= state.get("max_replans", 2):
        return None
    last = results[-1] if results else None
    if last and last.execution_status == "error":
        return "tool_failed"
    if last and any(f.severity in {"flag", "error"} and f.decision_effect in {"flag", "block"} for f in last.findings):
        return "new_flag"
    ready_ids = [row["tool_id"] for row in catalog if row.get("ready")]
    open_required = [q for q in required if (ledger.get(q) or {}).get("status") == "untested"]
    pending = [s for s in plan.steps if s.status == "pending" and s.tool_id in ready_ids]
    if open_required and not pending:
        return "no_ready_step"
    return None


def _validation_failure(exc: PlanValidationError, events: Path) -> Decision:
    """Create-path validation/repair failed: not an API outage."""
    emit({"type": "plan.rejected", "error": str(exc)}, path=events)
    return Decision(
        action="stop",
        stop_reason="degraded_execution",
        reason=f"Plan failed validation ({exc}); L0 evidence kept.",
        source="planned",
    )


def _api_failure(runtime: Any, state: dict[str, Any], exc: Exception, events: Path) -> Decision:
    if isinstance(exc, PlanValidationError):
        return _validation_failure(exc, events)
    mode = runtime.config.planning.api_failure_mode
    emit(
        {
            "type": "api.failed",
            "error": type(exc).__name__,
            "mode": mode,
        },
        path=events,
    )
    if mode == "degrade_to_rule":
        state["planning_degraded"] = True
        return Decision(
            action="stop",
            stop_reason="degraded_execution",
            reason=f"Planner failed ({type(exc).__name__}); degraded.",
            source="planned",
        )
    return Decision(
        action="stop",
        stop_reason="degraded_execution",
        reason=f"Planner failed ({type(exc).__name__}); refusing silent pass.",
        source="planned",
    )


def _resource_map(config: Any, sample: SampleSpec, base_dir: Path) -> dict[str, Any]:
    resolver = ResourceResolver(config, base_dir=base_dir)
    return {
        "gray_control": bool(config.resources.gray_control_path),
        "atlas": resolver.atlas_dir() is not None,
        "cohort": resolver.cohort_index_path() is not None,
        "reference": bool(config.reference.stats_path),
        "clip": bool(config.resources.image_embedding_dir),
        "cortexmae": bool(config.resources.cortexmae_model_id),
        "surface_atlas": bool(config.resources.atlas_dir),
    }


def _write_plan_artifacts(sample_dir: Path, plan: Plan) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    with (sample_dir / "plan_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(plan.model_dump_json() + "\n")
