"""Sample inspection loop used by LangGraph nodes and the CLI."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from react_agent.fmri.budget import BudgetError, BudgetLedger
from react_agent.fmri.config import FmriCheckConfig, load_config, redacted_config
from react_agent.fmri.data import ArrayHandle, inspect_array, safe_sample_id
from react_agent.fmri.llm.deepseek import (
    DeepSeekBackend,
    DeepSeekCallError,
    DeepSeekConfigError,
    DeepSeekParseError,
)
from react_agent.fmri.llm.mock import MockBackend
from react_agent.fmri.ledger import (
    unanswered_required,
    build_question_ledger,
    candidate_briefs,
    ledger_as_dict,
)
from react_agent.fmri.policy import (
    choose_profile,
    filter_candidates,
    rule_decision,
    validate_decision,
)
from react_agent.fmri.prompts import build_observation
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.progress import emit
from react_agent.fmri.reporting import build_final_report, write_report_json, write_report_md
from react_agent.fmri.schemas import Decision, SampleSpec, ToolResult
from react_agent.fmri.scoring import compute_need_score
from react_agent.fmri.state import CheckState
from react_agent.fmri.tools.base import RunContext
from react_agent.fmri.tools.registry import ToolRegistry, build_registry


def parse_results(rows: list[dict[str, Any]]) -> list[ToolResult]:
    """Rehydrate tool results from state."""
    return [ToolResult.model_validate(row) for row in rows]


class LoopRuntime:
    """Holds objects that should not live in LangGraph state."""

    def __init__(
        self,
        config: FmriCheckConfig,
        *,
        backend_name: str,
        policy_name: str,
        mock: MockBackend | None = None,
        batch_ledger: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.backend_name = backend_name
        self.policy_name = policy_name
        self.registry: ToolRegistry = build_registry(config)
        self.budget = BudgetLedger(config.budgets, batch=batch_ledger)
        self.mock = mock
        self.backend: Any = None
        if backend_name == "deepseek":
            self.backend = DeepSeekBackend(config)
        elif backend_name == "mock":
            self.backend = mock or MockBackend()
        elif backend_name in {"none", "rule"}:
            self.backend = None
        else:
            raise ValueError(f"unknown backend {backend_name}")
        if policy_name == "hybrid" and backend_name == "deepseek" and self.backend is None:
            raise DeepSeekConfigError("hybrid+deepseek requires a backend")
        if policy_name == "hybrid" and backend_name == "deepseek":
            pass
        if policy_name == "hybrid" and backend_name not in {"mock", "deepseek"}:
            raise ValueError("hybrid policy requires mock or deepseek backend")
        if policy_name == "planned" and backend_name not in {"mock", "deepseek"}:
            raise ValueError("planned policy requires mock or deepseek backend")
        if policy_name == "planned" and not self.config.planning.enabled:
            self.config = config.model_copy(deep=True)
            self.config.planning.enabled = True


def cache_key(
    tool_name: str,
    fingerprint: str,
    args: dict[str, Any],
    version: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Hash identical tool invocations including bound resources."""
    blob = json.dumps(
        {
            "tool": tool_name,
            "fp": fingerprint,
            "args": args,
            "ver": version,
            "extra": extra or {},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def make_run_context(state: CheckState, runtime: LoopRuntime) -> RunContext:
    """Rebuild a tool RunContext from serializable state."""
    spec = SampleSpec.model_validate(state["sample_spec"])
    handle = None
    ref = state.get("array_ref") or {}
    if ref.get("path"):
        handle = ArrayHandle(
            path=ref["path"],
            kind=ref["kind"],
            array_key=ref.get("array_key"),
            time_axis=int(ref["time_axis"]),
            shape_tv=tuple(ref["shape_tv"]),  # type: ignore[arg-type]
            dtype=ref["dtype"],
            fingerprint=ref["fingerprint"],
        )
    return RunContext(
        config=runtime.config,
        sample=spec,
        array_handle=handle,
        base_dir=state["base_dir"],
        artifact_dir=str(Path(state["out_dir"]) / state.get("safe_sample_id", "sample")),
        fingerprint=state.get("fingerprint", ""),
        prior_results=list(state.get("tool_results") or []),
    )


async def ingest(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Parse the sample and create local references."""
    spec = SampleSpec.model_validate(state["sample_spec"])
    handle, info = inspect_array(spec, Path(state["base_dir"]))
    fingerprint = handle.fingerprint if handle else hashlib.sha256(spec.sample_id.encode()).hexdigest()
    safe = safe_sample_id(spec.sample_id, fingerprint)
    array_ref = handle.__dict__ if handle else {}
    return {
        "sample_id": spec.sample_id,
        "safe_sample_id": safe,
        "fingerprint": fingerprint,
        "array_ref": array_ref,
        "inspect_info": info,
        "tool_results": [],
        "completed_tools": [],
        "failed_tools": [],
        "unavailable_tools": [],
        "skipped_tools": [],
        "candidate_tools": [],
        "issue_types": [],
        "unresolved_questions": [],
        "decision_history": [],
        "lm_usages": [],
        "cache": {},
        "tool_call_count": 0,
        "lm_call_count": 0,
        "round_count": 0,
        "invalid_input": False,
        "fallback_used": False,
        "force_rule": False,
        "profile_upgraded": False,
        "no_progress_count": 0,
        "last_evidence_signature": "",
        "pending_decision": None,
        "current_plan": None,
        "memory_retrieval": {},
        "planning_degraded": False,
        "replan_count": 0,
        "max_replans": runtime.config.planning.max_replans,
        "active_step_id": None,
        "route": "required",
    }


async def _run_named_tool(
    name: str,
    state: CheckState,
    runtime: LoopRuntime,
    args: dict[str, Any],
) -> ToolResult:
    ctx = make_run_context(state, runtime)
    tool = runtime.registry.get(name)
    avail = tool.availability(ctx.sample, ctx)
    if avail.status != "available":
        result = ToolResult(
            tool_name=name,
            tool_version=tool.spec.version,
            result_id=f"{name}_skipped",
            execution_status="skipped",
            skip_reason=avail.reason,
        )
        return result
    extra = {
        "control": (runtime.config.resources.gray_control_path or ""),
        "atlas": (runtime.config.resources.atlas_dir or ""),
        "cohort": (runtime.config.resources.cohort_index_path or ""),
        "signal_mode": args.get("signal") or args.get("mode"),
        "metric_definition_version": (
            runtime.config.diagnostic.metric_definition_version
            if runtime.config.diagnostic.enabled
            else "v1"
        ),
    }
    key = cache_key(name, state.get("fingerprint", ""), args, tool.spec.version, extra)
    cache = dict(state.get("cache") or {})
    if runtime.config.cache_enabled and key in cache:
        cached = ToolResult.model_validate(cache[key])
        cached.cache_hit = True
        return cached
    runtime.budget.preflight_tool()
    result = await tool.run(ctx.sample, args, ctx)
    if result.execution_status != "skipped":
        runtime.budget.record_tool()
    cache[key] = result.model_dump()
    state["cache"] = cache  # mutated for subsequent tools in the same node
    # If validate_input recovered a handle, persist it.
    if ctx.array_handle is not None and not state.get("array_ref"):
        state["array_ref"] = ctx.array_handle.__dict__
        state["fingerprint"] = ctx.array_handle.fingerprint
    return result


async def run_required_checks(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Always run validate_input then basic_statistics when possible."""
    results = parse_results(list(state.get("tool_results") or []))
    completed = set(state.get("completed_tools") or [])
    skipped = list(state.get("skipped_tools") or [])
    failed = list(state.get("failed_tools") or [])
    events = Path(state["events_path"])
    invalid = False
    working = dict(state)

    for name in runtime.config.required_checks:
        if name in completed:
            continue
        result = await _run_named_tool(name, working, runtime, {})
        results.append(result)
        emit(
            {
                "type": "tool",
                "tool_name": name,
                "status": result.execution_status,
                "result_id": result.result_id,
                "cache_hit": result.cache_hit,
            },
            path=events,
        )
        if result.execution_status == "success":
            completed.add(name)
            if name == "validate_input" and any(
                f.code == "malformed" and f.severity == "error" for f in result.findings
            ):
                invalid = True
                break
        elif result.execution_status == "skipped":
            skipped.append(name)
        else:
            failed.append(name)
            invalid = True
            break
        if name == "validate_input" and working.get("array_ref") and not state.get("array_ref"):
            pass

    return {
        "tool_results": [r.model_dump() for r in results],
        "completed_tools": sorted(completed),
        "skipped_tools": skipped,
        "failed_tools": failed,
        "invalid_input": invalid,
        "tool_call_count": runtime.budget.data["tool_calls"],
        "budget": runtime.budget.snapshot(),
        "cache": working.get("cache") or {},
        "array_ref": working.get("array_ref") or state.get("array_ref") or {},
        "fingerprint": working.get("fingerprint") or state.get("fingerprint") or "",
        "route": "finalize" if invalid else "evidence",
    }


def _signature(results: list[ToolResult]) -> str:
    blob = [(r.tool_name, r.execution_status, r.result_id) for r in results]
    return json.dumps(blob)


def update_evidence(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Refresh scores, issues, and unresolved questions."""
    results = parse_results(list(state.get("tool_results") or []))
    completed = set(state.get("completed_tools") or [])
    score = compute_need_score(results, runtime.config, completed=completed)
    flags = score.get("flag_codes") or []
    questions: list[str] = []
    if "temporal_spike" in flags or "low_temporal_change" in flags:
        if "temporal_diagnostics" not in completed:
            questions.append("Need temporal localization of the cheap pre-screen flag.")
    ctx = make_run_context(state, runtime)
    spec = ctx.sample
    resolver = ResourceResolver(runtime.config, base_dir=Path(state["base_dir"]))
    ledger = build_question_ledger(
        sample=spec, results=results, config=runtime.config, resolver=resolver
    )
    unavailable = []
    for name in [
        "roi_summary",
        "reference_distribution",
        "gray_control_contrast",
        "stimulus_temporal_profile",
        "surface_roi_profile",
        "cross_image_specificity",
        "cortex_mae",
        "semantic_consistency",
    ]:
        try:
            tool = runtime.registry.get(name)
        except KeyError:
            continue
        avail = tool.availability(spec, ctx)
        if avail.status != "available":
            unavailable.append(f"{name}:{avail.reason or avail.status}")
    sig = _signature(results)
    no_progress = state.get("no_progress_count") or 0
    if sig == state.get("last_evidence_signature"):
        no_progress += 1
    else:
        no_progress = 0
    open_required = unanswered_required(ledger)
    questions.extend([f"required_question:{qid}" for qid in open_required])
    return {
        "need_score": score,
        "issue_types": flags,
        "unresolved_questions": questions,
        "unavailable_tools": unavailable,
        "last_evidence_signature": sig,
        "no_progress_count": no_progress,
        "question_ledger": ledger_as_dict(ledger),
        "required_unanswered": open_required,
        "route": "select",
    }


async def select_action(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Choose the next action via rule or hybrid policy."""
    results = parse_results(list(state.get("tool_results") or []))
    completed = set(state.get("completed_tools") or [])
    blocked = completed | set(state.get("skipped_tools") or []) | set(state.get("failed_tools") or [])
    ctx = make_run_context(state, runtime)
    candidates = filter_candidates(
        registry=runtime.registry,
        sample=ctx.sample,
        context=ctx,
        config=runtime.config,
        results=results,
        completed=blocked,
    )
    # Required tools still missing should be candidates for rule mode.
    for name in runtime.config.required_checks:
        if name not in completed and name not in candidates:
            candidates = [name, *candidates]

    events = Path(state["events_path"])
    budgets = runtime.config.budgets
    qmap = {
        "input_mapping": "validate_input",
        "gray_contrast": "gray_control_contrast",
    }
    required_open = [
        q
        for q in list(state.get("required_unanswered") or [])
        if qmap.get(q, q) in candidates or qmap.get(q, q) in runtime.config.required_checks
    ]
    briefs = candidate_briefs(candidates, runtime.registry)
    stop: Decision | None = None
    if state.get("invalid_input"):
        stop = Decision(action="stop", stop_reason="invalid_input", reason="Malformed input.", source="rule")
    elif (state.get("round_count") or 0) >= budgets.max_rounds:
        stop = Decision(action="stop", stop_reason="max_rounds", reason="Round cap reached.", source="rule")
    elif runtime.budget.data["tool_calls"] >= budgets.max_tool_calls and not [
        c for c in candidates if c in runtime.config.required_checks and c not in completed
    ]:
        stop = Decision(
            action="stop",
            stop_reason="max_tool_calls",
            reason="Tool-call cap reached.",
            source="rule",
        )
    elif (state.get("no_progress_count") or 0) >= budgets.no_progress_limit:
        stop = Decision(
            action="stop",
            stop_reason="no_progress",
            reason="Two steps added no new evidence.",
            source="rule",
        )
    elif not candidates and runtime.policy_name != "planned":
        stop = Decision(
            action="stop",
            stop_reason="no_candidates",
            reason="No remaining executable candidate.",
            source="rule",
        )

    if stop is not None:
        emit({"type": "decision", "decision": stop.model_dump()}, path=events)
        history = list(state.get("decision_history") or []) + [stop.model_dump()]
        return {
            "candidate_tools": candidates,
            "pending_decision": stop.model_dump(),
            "decision_history": history,
            "route": "finalize",
            "stop_reason": stop.stop_reason or "",
        }

    use_planned = runtime.policy_name == "planned" and not state.get("planning_degraded")
    use_rule = (
        runtime.policy_name == "rule"
        or (bool(state.get("force_rule")) and not use_planned)
        or (runtime.backend is None and not use_planned)
    )
    fallback_used = bool(state.get("fallback_used"))
    profile = "fast"
    decision: Decision
    plan_extras: dict[str, Any] = {}
    if use_planned:
        from react_agent.fmri.planning.policy import PlannedPolicy

        memory_bundle = await _retrieve_memory(state, runtime, ctx.sample)
        policy = PlannedPolicy(runtime)
        decision, plan_extras = await policy.next_action(
            state=state,
            sample=ctx.sample,
            context=ctx,
            results=results,
            candidates=candidates,
            completed=completed,
            events=events,
            memory_bundle=memory_bundle,
        )
        source_event = "planned"
        if plan_extras.get("replan_trigger"):
            state["replan_count"] = int(state.get("replan_count") or 0) + 1
    elif use_rule:
        decision = rule_decision(
            sample=ctx.sample,
            results=results,
            candidates=candidates,
            required=runtime.config.required_checks,
            completed=completed,
        )
        source_event = "rule"
    else:
        profile = choose_profile(candidates, runtime.config, bool(state.get("profile_upgraded")))
        upgraded = bool(state.get("profile_upgraded")) or profile == "reasoning"
        observation = build_observation(
            sample=state["sample_spec"],
            results=[r.model_dump() for r in results],
            candidates=candidates,
            need_score=state.get("need_score") or {},
            budget=runtime.budget.snapshot(),
            unresolved=list(state.get("unresolved_questions") or []),
            candidate_briefs=briefs,
            question_ledger=state.get("question_ledger") or {},
            required_unanswered=required_open,
        )
        decision, fallback_used, profile = await _hybrid_decide(
            runtime,
            ctx.sample,
            observation,
            candidates,
            profile,
            results,
            completed,
            events,
        )
        source_event = "hybrid"
        state_upgraded = upgraded
    history = list(state.get("decision_history") or []) + [decision.model_dump()]
    emit(
        {
            "type": "decision",
            "decision": decision.model_dump(),
            "via": source_event,
            "candidates": candidates,
        },
        path=events,
    )
    route = "finalize" if decision.action == "stop" else "validate"
    out: dict[str, Any] = {
        "candidate_tools": candidates,
        "pending_decision": decision.model_dump(),
        "decision_history": history,
        "route": route,
        "fallback_used": fallback_used,
        "lm_call_count": runtime.budget.data["lm_calls"],
        "budget": runtime.budget.snapshot(),
        "round_count": int(state.get("round_count") or 0) + 1,
        "stop_reason": decision.stop_reason or state.get("stop_reason") or "",
    }
    if not use_rule and not use_planned:
        out["profile_upgraded"] = bool(state.get("profile_upgraded")) or profile == "reasoning"
    if plan_extras.get("current_plan"):
        out["current_plan"] = plan_extras["current_plan"]
    if plan_extras.get("active_step_id"):
        out["active_step_id"] = plan_extras["active_step_id"]
    if state.get("memory_retrieval"):
        out["memory_retrieval"] = state.get("memory_retrieval")
    return out


async def _hybrid_decide(
    runtime: LoopRuntime,
    sample: SampleSpec,
    observation: dict[str, Any],
    candidates: list[str],
    profile: str,
    results: list[ToolResult],
    completed: set[str],
    events: Path,
) -> tuple[Decision, bool, str]:
    fallback = False
    backend = runtime.backend
    assert backend is not None

    async def _once(repair: bool) -> Decision:
        runtime.budget.preflight_lm()
        try:
            decision, usage = await backend.decide(observation, candidates, profile)
            runtime.budget.record_lm(usage, repair=repair)
            emit({"type": "lm", "kind": "decide"}, path=events)
            return decision
        except DeepSeekParseError as exc:
            runtime.budget.record_lm(exc.usage, repair=repair)
            emit({"type": "lm", "kind": "decide_parse_error"}, path=events)
            raise
        except DeepSeekCallError as exc:
            runtime.budget.record_lm(exc.usage, repair=repair)
            emit({"type": "lm", "kind": "decide_error"}, path=events)
            raise

    try:
        decision = await _once(False)
        ok, err = validate_decision(
            decision,
            candidates=candidates,
            results=results,
            completed=completed,
            required_unanswered=list(observation.get("required_unanswered") or []),
        )
        if ok:
            return decision, False, profile
        observation = {**observation, "validator_error": err, "repair": True}
        decision = await _once(True)
        ok, err = validate_decision(
            decision,
            candidates=candidates,
            results=results,
            completed=completed,
            required_unanswered=list(observation.get("required_unanswered") or []),
        )
        if ok:
            return decision, False, profile
        fallback = True
    except (DeepSeekParseError, DeepSeekCallError, BudgetError, ValueError):
        fallback = True
    emit({"type": "fallback", "to": "rule"}, path=events)
    decision = rule_decision(
        sample=sample,
        results=results,
        candidates=candidates,
        required=runtime.config.required_checks,
        completed=completed,
    )
    decision.source = "fallback"
    return decision, fallback, profile


async def validate_action(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Check the pending decision before execution."""
    raw = state.get("pending_decision")
    if not raw:
        return {"route": "finalize", "stop_reason": "no_pending_decision"}
    decision = Decision.model_validate(raw)
    results = parse_results(list(state.get("tool_results") or []))
    completed = set(state.get("completed_tools") or [])
    candidates = list(state.get("candidate_tools") or [])
    ok, err = validate_decision(
        decision,
        candidates=candidates,
        results=results,
        completed=completed,
        required_unanswered=list(state.get("required_unanswered") or []),
        allow_abstain=runtime.policy_name == "planned",
    )
    if decision.action == "stop":
        return {"route": "finalize", "stop_reason": decision.stop_reason or "stop"}
    if not ok:
        emit(
            {"type": "invalid_action", "error": err},
            path=Path(state["events_path"]),
        )
        if runtime.policy_name == "planned":
            return {
                "route": "finalize",
                "stop_reason": "degraded_execution",
                "planning_degraded": True,
                "fallback_used": True,
            }
        return {
            "route": "select",
            "force_rule": True,
            "fallback_used": True,
        }
    return {"route": "execute"}


async def execute_tool(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Run exactly one tool."""
    decision = Decision.model_validate(state["pending_decision"])
    name = decision.tool_name or ""
    working = dict(state)
    result = await _run_named_tool(name, working, runtime, decision.tool_args or {})
    results = parse_results(list(state.get("tool_results") or [])) + [result]
    completed = set(state.get("completed_tools") or [])
    skipped = list(state.get("skipped_tools") or [])
    failed = list(state.get("failed_tools") or [])
    if result.execution_status == "success":
        completed.add(name)
    elif result.execution_status == "skipped":
        skipped.append(name)
    else:
        failed.append(name)
    emit(
        {
            "type": "tool",
            "tool_name": name,
            "status": result.execution_status,
            "result_id": result.result_id,
            "cache_hit": result.cache_hit,
        },
        path=Path(state["events_path"]),
    )
    return {
        "tool_results": [r.model_dump() for r in results],
        "completed_tools": sorted(completed),
        "skipped_tools": skipped,
        "failed_tools": failed,
        "tool_call_count": runtime.budget.data["tool_calls"],
        "budget": runtime.budget.snapshot(),
        "cache": working.get("cache") or state.get("cache") or {},
        "array_ref": working.get("array_ref") or state.get("array_ref") or {},
        "pending_decision": None,
        "route": "evidence",
        "current_plan": _touch_plan(state.get("current_plan"), name, result.execution_status),
    }


_DEPTH_ORDER = {"L0": 1, "L1": 1, "L2": 2, "L3": 3}


def reached_depth(results: list[ToolResult], specs: list[Any]) -> str:
    """Deepest stage among tools that actually succeeded. L0 counts as L1."""
    stage = {spec.name: spec.stage_hint for spec in specs}
    best = 1
    for result in results:
        if result.execution_status != "success":
            continue
        best = max(best, _DEPTH_ORDER.get(stage.get(result.tool_name) or "L1", 1))
    return f"L{best}"


async def finalize(state: CheckState, runtime: LoopRuntime) -> dict[str, Any]:
    """Deterministic verdict. Optional LM summary cannot change it."""
    results = parse_results(list(state.get("tool_results") or []))
    completed = set(state.get("completed_tools") or [])
    score = state.get("need_score") or compute_need_score(
        results, runtime.config, completed=completed
    )
    history = list(state.get("decision_history") or [])
    lm_decisions = [d for d in history if d.get("source") == "hybrid"]
    fallbacks = [d for d in history if d.get("source") == "fallback"]
    report = build_final_report(
        sample_id=state.get("sample_id") or "unknown",
        safe_sample_id=state.get("safe_sample_id") or "unknown",
        results=results,
        required=runtime.config.required_checks,
        invalid_input=bool(state.get("invalid_input")),
        unresolved=list(state.get("unresolved_questions") or []),
        unavailable=list(state.get("unavailable_tools") or []),
        need_score=score,
        stop_reason=state.get("stop_reason") or "finalize",
        fallback_used=bool(state.get("fallback_used")),
        cost=runtime.budget.snapshot(),
        coverage_by_question=state.get("question_ledger") or {},
        candidate_count=len(state.get("candidate_tools") or []),
        selected_count=len(history),
        lm_decision_count=len(lm_decisions),
        fallback_count=len(fallbacks),
        non_fallback_lm_successes=len(lm_decisions),
        check_profile=runtime.config.check_profile.name,
        required_questions_unanswered=bool(state.get("required_unanswered")),
        degraded_execution=bool(state.get("planning_degraded"))
        or (state.get("stop_reason") == "degraded_execution"),
        degraded_reason=state.get("stop_reason") if state.get("planning_degraded") else None,
        requested_policy=runtime.policy_name,
        resolved_policy=runtime.policy_name,
        requested_backend=runtime.backend_name,
        resolved_backend=runtime.backend_name,
        analysis_goal=runtime.config.diagnostic.analysis_goal if runtime.config.diagnostic.enabled else None,
        coverage_profile=runtime.config.diagnostic.coverage_profile if runtime.config.diagnostic.enabled else None,
        coverage_by_dimension={
            qid: {"required": row.get("required"), "status": row.get("status"), "evidence_ids": row.get("evidence_ids")}
            for qid, row in (state.get("question_ledger") or {}).items()
        } if runtime.config.diagnostic.enabled else None,
        biological_validity="not_assessed" if runtime.config.diagnostic.enabled else None,
        depth_reached=reached_depth(results, runtime.registry.specs()) if runtime.config.diagnostic.enabled else None,
    )
    payload = report.model_dump()
    lm_summary = None
    if runtime.policy_name == "hybrid" and runtime.backend is not None and not state.get("invalid_input"):
        try:
            runtime.budget.preflight_lm()
            summary, usage = await runtime.backend.summarize(payload, "fast")
            runtime.budget.record_lm(usage)
            emit({"type": "lm", "kind": "summarize"}, path=Path(state["events_path"]))
            refs_ok = True
            for ref in summary.get("evidence_refs") or []:
                allowed = {f"{r.tool_name}:{r.result_id}" for r in results if r.execution_status == "success"}
                if ":".join(str(ref).split(":")[:2]) not in allowed:
                    refs_ok = False
            lm_summary = summary if refs_ok else {
                "findings": f"verdict={report.verdict}",
                "limitations": "Template summary; LM evidence_refs failed validation.",
                "next_steps": "See metrics_table in JSON.",
            }
        except (BudgetError, DeepSeekCallError, DeepSeekParseError, Exception):
            lm_summary = {
                "findings": f"verdict={report.verdict}",
                "limitations": "Template summary; LM summarize unavailable.",
                "next_steps": "See report.json.",
            }
    if lm_summary is not None:
        payload["lm_summary"] = lm_summary
        payload["cost"] = runtime.budget.snapshot()
        report = FinalReport_from_payload(payload)
    sample_dir = Path(state["out_dir"]) / (state.get("safe_sample_id") or "sample")
    write_report_json(sample_dir / "report.json", report)
    write_report_md(sample_dir / "report.md", report)
    await _persist_planned_artifacts(state, runtime, report, sample_dir)
    dumped = report.model_dump()
    dumped["workflow"] = _attach_workflow(sample_dir, dumped, state, runtime)
    emit(
        {"type": "stop", "verdict": report.verdict, "stop_reason": report.stop_reason},
        path=Path(state["events_path"]),
    )
    return {
        "final_report": dumped,
        "budget": runtime.budget.snapshot(),
        "lm_call_count": runtime.budget.data["lm_calls"],
        "route": "done",
    }


def _attach_workflow(
    sample_dir: Path,
    report: dict[str, Any],
    state: CheckState,
    runtime: LoopRuntime,
) -> dict[str, Any]:
    from react_agent.fmri.workflow import publish_workflow

    gray = getattr(runtime.config.resources, "gray_control_path", None)
    try:
        page = publish_workflow(
            sample_dir,
            report,
            sample_spec=state.get("sample_spec") or {},
            gray_path=gray,
        )
    except Exception as exc:  # noqa: BLE001
        page = {
            "html_path": None,
            "brain_tstrip": None,
            "metric_scores": {},
            "plot_note": f"workflow page failed: {exc}",
            "image_source": "missing",
        }
    html_path = page.get("html_path")
    if html_path:
        print(f"[workflow] {html_path}", flush=True)
    return page


def _touch_plan(plan: dict[str, Any] | None, tool_name: str, status: str) -> dict[str, Any] | None:
    from react_agent.fmri.planning.policy import update_plan_after_tool

    return update_plan_after_tool(plan, tool_name, status)


async def _retrieve_memory(state: CheckState, runtime: LoopRuntime, sample: SampleSpec) -> dict[str, Any]:
    if not runtime.config.memory.enabled:
        return {"items": [], "effects": [], "hit_count": 0}
    from react_agent.fmri.memory.service import open_memory, retrieve_for_sample

    repo = open_memory(runtime.config)
    query = {
        "generation_profile_id": sample.generation_profile_id,
        "space_name": sample.space_name,
        "normalization": sample.normalization,
        "t_len": (sample.expected_shape or (None, None))[0],
        "checkpoint_hash": (sample.provenance.generator if sample.provenance else None),
        "execution_domain": "real",
        "namespace": runtime.config.memory.namespace,
    }
    bundle = retrieve_for_sample(
        repo,
        query=query,
        max_items=runtime.config.memory.max_retrieved_items,
        exclude_sample_ids={sample.sample_id},
    )
    state["memory_retrieval"] = bundle
    emit(
        {
            "type": "memory.retrieved",
            "message": f"hits={bundle.get('hit_count', 0)}",
        },
        path=Path(state["events_path"]),
    )
    sample_dir = Path(state["out_dir"]) / (state.get("safe_sample_id") or "sample")
    if runtime.config.reporting.emit_memory_retrieval:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "memory_retrieval.json").write_text(
            json.dumps(bundle, indent=2, default=str), encoding="utf-8"
        )
    return bundle


async def _persist_planned_artifacts(
    state: CheckState,
    runtime: LoopRuntime,
    report,
    sample_dir: Path,
) -> None:
    usages = list(state.get("lm_usages") or []) or list(runtime.budget.calls)
    if runtime.config.reporting.emit_llm_usage:
        (sample_dir / "llm_usage.json").write_text(
            json.dumps(
                {
                    "calls": usages,
                    "ledger": runtime.budget.snapshot(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    if not runtime.config.memory.enabled:
        return
    from react_agent.fmri.memory.service import maybe_curate, open_memory, write_episode

    spec = SampleSpec.model_validate(state["sample_spec"])
    episode = {
        "run_id": state.get("run_id"),
        "sample_id": spec.sample_id,
        "input_hash": state.get("fingerprint"),
        "artifact_hash": state.get("fingerprint"),
        "generation_profile_id": spec.generation_profile_id,
        "checkpoint_hash": spec.provenance.generator if spec.provenance else None,
        "space_name": spec.space_name,
        "t_len": (spec.expected_shape or (None, None))[0],
        "normalization": spec.normalization,
        "execution_domain": "mock" if runtime.backend_name == "mock" else "real",
        "screening_decision": report.screening_decision,
        "verification_status": "unverified",
        "verification_scope": "numeric_consistency_only",
        "plan_sequence": [
            s.get("tool_id") for s in (state.get("current_plan") or {}).get("steps") or []
        ],
        "key_metrics": {
            name: {
                k: v
                for k, v in (metrics or {}).items()
                if not isinstance(v, list)
            }
            for name, metrics in (report.metrics_table or {}).items()
        },
        "findings": [f.code for f in report.findings],
        "cost": report.cost,
        "degraded_execution": report.degraded_execution,
        "fallback_used": report.fallback_used,
        "namespace": runtime.config.memory.namespace,
        "routing_profile_version": (
            runtime.config.diagnostic.routing.profile if runtime.config.diagnostic.enabled else None
        ),
        "coverage_profile_version": (
            runtime.config.diagnostic.coverage_profile_version if runtime.config.diagnostic.enabled else None
        ),
        "question_resolved": False,
        "defect_confirmed": False,
    }
    repo = open_memory(runtime.config)
    eid = write_episode(repo, episode, out_dir=sample_dir)
    emit({"type": "memory.episode_written", "episode_id": eid}, path=Path(state["events_path"]))
    if runtime.config.memory.mode != "read_write":
        return
    try:
        candidates, usage = await maybe_curate(runtime.backend, episode, config=runtime.config)
    except Exception:  # noqa: BLE001
        return
    if usage is not None:
        runtime.budget.record_lm(usage)
    if candidates:
        (sample_dir / "memory_candidates.json").write_text(
            json.dumps(candidates, indent=2, default=str), encoding="utf-8"
        )
        for cand in candidates:
            cid = repo.append_candidate(cand)
            emit(
                {"type": "memory.candidate_written", "candidate_id": cid},
                path=Path(state["events_path"]),
            )


def FinalReport_from_payload(payload: dict[str, Any]):
    from react_agent.fmri.schemas import FinalReport

    return FinalReport.model_validate(payload)


async def run_sample(
    spec: SampleSpec,
    *,
    base_dir: Path,
    out_dir: Path,
    config: FmriCheckConfig,
    backend: str,
    policy: str,
    mock: MockBackend | None = None,
    batch_ledger: dict[str, int] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full loop without requiring LangGraph invoke."""
    run_id = run_id or uuid4().hex[:10]
    sample_events = out_dir / "events.jsonl"
    runtime = LoopRuntime(
        config, backend_name=backend, policy_name=policy, mock=mock, batch_ledger=batch_ledger
    )
    state: CheckState = {
        "run_id": run_id,
        "sample_spec": spec.model_dump(),
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "config": redacted_config(config),
        "backend": backend,
        "policy": policy,
        "events_path": str(sample_events),
    }
    started = time.perf_counter()
    state.update(await ingest(state, runtime))
    state.update(await run_required_checks(state, runtime))
    if not state.get("invalid_input"):
        while True:
            state.update(update_evidence(state, runtime))
            state.update(await select_action(state, runtime))
            if state.get("route") == "finalize":
                break
            state.update(await validate_action(state, runtime))
            if state.get("route") == "finalize":
                break
            if state.get("route") == "select":
                continue
            state.update(await execute_tool(state, runtime))
            if int(state.get("round_count") or 0) >= config.budgets.max_rounds:
                state["stop_reason"] = "max_rounds"
                break
    else:
        state["stop_reason"] = "invalid_input"
    state.update(await finalize(state, runtime))
    report = state.get("final_report") or {}
    report_elapsed = time.perf_counter() - started
    report["cost"] = {**(report.get("cost") or {}), "elapsed_seconds": report_elapsed}
    return report
