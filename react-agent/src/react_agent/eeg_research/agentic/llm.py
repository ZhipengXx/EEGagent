"""Role calls through the existing DeepSeek client. Every call is written to the cost ledger."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.identity import new_call_row
from react_agent.eeg_research.agentic.roles import ENVELOPE_ROLES
from react_agent.eeg_research.agentic.ui_events import append_ui_event

PROMPTS = Path(__file__).resolve().parent / "prompts"
PARSE_ATTEMPTS = 2
CODER_MAX_TOKENS = 16384
FAST_MAX_TOKENS = 4096
LONG_JSON_ROLES = {
    "candidate_coder",
    "candidate_reviewer",
    "result_analyst",
    "research_planner",
    "research_librarian",
    "experiment_designer",
    "memory_curator",
    "result_auditor",
}
RAW_EXCERPT_LIMIT = 2048
API_INTENTS = "llm_api_intents.jsonl"


class LlmUnavailable(RuntimeError):
    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def context_budget_trace(camp: Path, role: str, *, system: str,
                        payload: dict, projected: dict) -> dict:
    """Preserve a complete local failed preflight without reserving an API call."""
    preflight_id = "ctx_" + uuid.uuid4().hex[:12]
    target = camp / "context_diagnostics" / (preflight_id + ".json")
    record = {"preflight_id": preflight_id, "role": role, "recorded_at": time.time(),
        "system": system, "original_payload": payload, "projected_payload": projected,
        "scope": "Local context budget failure only; not a paid request, role result, artifact read, or scientific evidence."}
    encoded = json.dumps(record, ensure_ascii=False, default=str, indent=2).encode("utf-8")
    try:
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(encoded)
        return {"preflight_id": preflight_id, "path": str(target),
                "sha256": hashlib.sha256(encoded).hexdigest()}
    except OSError as exc:
        return {"preflight_id": preflight_id, "trace_write_failed": type(exc).__name__}


def project_correction_context(payload: dict[str, Any], role: str, *, system_chars: int,
                               target_chars: int | None = None) -> dict:
    """Keep authoritative inputs and complete feedback within the same hard cap.

    Planner examples are optional on initial calls as well as corrections. A
    failed draft is an optional correction aid, not source evidence. Omit whole
    aids when needed; never slice them or scientific input.
    The native response journal retains the complete original model reply.
    """
    import copy
    from react_agent.eeg_research.agentic.handoffs import compact_role_context

    def referenced_subtree(value, pointer):
        if isinstance(value, dict):
            reference = value.get("canonical_context_ref")
            if isinstance(reference, str) and (reference == pointer or reference.startswith(pointer + "/")):
                return True
            return any(referenced_subtree(child, pointer) for child in value.values())
        if isinstance(value, list):
            return any(referenced_subtree(child, pointer) for child in value)
        return False

    def project(value):
        result = compact_role_context(value, role, system_chars=system_chars, target_chars=target_chars)
        report = result["role_context_projection"]
        if role != "research_planner" or report["fits_budget"]:
            return result
        # These are optional layout/debug reports, not original observations,
        # scientific data or constraints. Every canonical ref/hash remains in
        # the actual request, so removing the duplicate log changes no fact.
        omitted = []

        for parent, key, path in ((result, "planner_context_projection", "planner_context_projection"),
                                  (report, "duplicate_views", "role_context_projection.duplicate_views")):
            if key in parent and len(json.dumps(parent[key], ensure_ascii=False, sort_keys=True,
                                                 default=str, separators=(",", ":"))) >= 400 and not referenced_subtree(result, "/" + path.replace(".", "/")):
                detail = parent.pop(key)
                omitted.append({"field": path, "sha256": hashlib.sha256(json.dumps(detail,
                    ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()})
        if omitted:
            report["omitted_optional_diagnostic_reports"] = omitted
            report["diagnostic_omission_scope"] = "Whole delivery debug reports only; every scientific/control value and canonical_context_ref/sha256 remains in this request. Full original input is retained for local failed preflight diagnostics."
            report.pop("projected_total_chars", None)
            report.pop("fits_budget", None)
            report.pop("missing_inputs", None)
            total = len(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str,
                                   separators=(",", ":"))) + system_chars
            report["projected_total_chars"] = total
            report["fits_budget"] = total + 100 <= report["budget_chars"]
            if not report["fits_budget"]:
                report["missing_inputs"] = ["essential_context_exceeds_role_budget: obtain permitted digest-bound pages or narrow the task; do not infer omitted code"]
        if not report["fits_budget"]:
            # Keep the budget contract, pointer instruction and actual history
            # omissions; compact the remaining optional delivery debug report.
            retained = {"revision", "role", "budget_chars", "system_chars", "original_user_chars",
                        "canonical_reference_instruction", "omissions",
                        "source_metrics_gates_and_schema_feedback_preserved"}
            detail = {key: report.pop(key) for key in list(report) if key not in retained}
            report["optional_delivery_metadata_sha256"] = hashlib.sha256(json.dumps(detail,
                ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
            total = len(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str,
                                   separators=(",", ":"))) + system_chars
            report["projected_total_chars"] = total
            report["fits_budget"] = total + 100 <= report["budget_chars"]
            if not report["fits_budget"]:
                report["missing_inputs"] = ["essential_context_exceeds_role_budget: obtain permitted digest-bound pages or narrow the task; do not infer omitted code"]
        return result

    view = project(payload)
    if view["role_context_projection"]["fits_budget"] or role != "research_planner":
        return view
    correcting = bool(payload.get("schema_error"))
    if not correcting and "illustrative_examples" not in payload:
        return view
    reduced = copy.deepcopy(payload)
    notice = {"scope": "Optional delivery aids only; original evidence, gates and full schema feedback retained.",
              "omitted_optional_blocks": []}
    reduced["correction_delivery" if correcting else "optional_context_delivery"] = notice
    if "illustrative_examples" in reduced:
        reduced.pop("illustrative_examples")
        notice["omitted_optional_blocks"].append("illustrative_examples")
        view = project(reduced)
        if view["role_context_projection"]["fits_budget"]:
            return view
    if not correcting or "previous" not in reduced:
        return view
    previous = reduced.pop("previous")
    notice["omitted_optional_blocks"].append("previous_full_invalid_draft")
    notice["previous_reply_sha256"] = hashlib.sha256(json.dumps(previous, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    notice["instruction"] = "Return a complete corrected response. Previous proposals are unapproved model output, not facts. Resolve every mechanical error against the current original observation. The full invalid reply remains in the native response journal."
    if isinstance(previous, dict):
        core = {key: copy.deepcopy(previous[key]) for key in (
            "action", "target_id", "question_id", "selected_option_id", "hypothesis_draft", "experiment_draft")
            if key in previous}
        reduced["previous_execution_fields"] = core
        view = project(reduced)
        if view["role_context_projection"]["fits_budget"]:
            return view
        # Even a very large hypothesis is an optional failed proposal. Omit
        # this entire block instead of truncating its meaning or source facts.
        reduced.pop("previous_execution_fields")
        notice["omitted_optional_blocks"].append("previous_execution_fields")
    view = project(reduced)
    if not view["role_context_projection"]["fits_budget"]:
        # Last correction fallback: compact only optional formatting/debug
        # metadata. Complete validation errors and every scientific/control
        # field remain. OUTPUT_SCHEMA already supplies the allowed field names.
        removed = {}
        feedback = view.get("schema_repair_context")
        if (isinstance(feedback, dict) and "allowed_top_level_fields" in feedback
                and not referenced_subtree(view, "/schema_repair_context/allowed_top_level_fields")):
            removed["allowed_top_level_fields"] = feedback.pop("allowed_top_level_fields")
        if "correction_instruction" in view and not referenced_subtree(view, "/correction_instruction"):
            removed["correction_instruction"] = view.pop("correction_instruction")
        delivery = view.get("correction_delivery") or {}
        for key in ("scope", "instruction"):
            if key in delivery and not referenced_subtree(view, "/correction_delivery/" + key):
                removed["correction_delivery." + key] = delivery.pop(key)
        report = view["role_context_projection"]
        retained = {"revision", "role", "budget_chars", "system_chars", "original_user_chars",
                    "canonical_reference_instruction", "omissions",
                    "source_metrics_gates_and_schema_feedback_preserved"}
        for key in list(report):
            if key not in retained:
                removed["role_context_projection." + key] = report.pop(key)
        report["optional_correction_metadata_sha256"] = hashlib.sha256(json.dumps(removed,
            ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
        total = len(json.dumps(view, ensure_ascii=False, sort_keys=True, default=str,
                               separators=(",", ":"))) + system_chars
        report["projected_total_chars"] = total
        report["fits_budget"] = total + 100 <= report["budget_chars"]
        if not report["fits_budget"]:
            report["missing_inputs"] = ["essential_context_exceeds_role_budget: obtain permitted digest-bound pages or narrow the task; do not infer omitted code"]
    return view


def domain_model_for(role: str, memory=None):
    from react_agent.eeg_research.agentic.schemas import DOMAIN_OUTPUT_MODELS

    if role == "research_planner":
        from react_agent.eeg_research.agentic.schemas import PlannerProposal
        return PlannerProposal

    if role == "memory_curator" and memory is not None and memory.enabled and memory.skills_enabled:
        from react_agent.eeg_research.agentic.skill_memory import SkillLessonProposal
        return SkillLessonProposal
    return DOMAIN_OUTPUT_MODELS.get(role)


def system_prompt(role: str, *, memory=None) -> str:
    from react_agent.eeg_research.agentic.schemas import role_output_schema

    shared = (PROMPTS / "shared_contract.txt").read_text(encoding="utf-8")
    mission = (PROMPTS / f"{role}.txt").read_text(encoding="utf-8")
    schema = role_output_schema(role)
    if role == "research_planner":
        from react_agent.eeg_research.agentic.schemas import PlannerOption
        schema = json.dumps(domain_model_for(role, memory).model_json_schema(), ensure_ascii=False, separators=(",", ":"))
        # The outer proposal intentionally accepts raw options so one malformed
        # draft cannot discard valid siblings. Still show the model the exact
        # independently enforced item/payload schema before it drafts them.
        mission += "\n\nINDEPENDENT_OPTION_SCHEMA\n" + json.dumps(
            PlannerOption.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    if memory is not None and memory.enabled:
        extension = {"research_planner": "retrieved_memory", "experiment_designer": "retrieved_procedural_hints",
                     "candidate_coder": "retrieved_procedural_hints"}.get(role)
        if role == "memory_curator" and memory.skills_enabled:
            extension = "procedural_skill_extension"
            schema = json.dumps(domain_model_for(role, memory).model_json_schema(), ensure_ascii=False)
        if extension:
            mission += "\n\n" + (PROMPTS / (extension + ".txt")).read_text(encoding="utf-8")
    from react_agent.eeg_research.agentic.context.compiler import render_role_system
    return render_role_system(role, shared + "\n\n" + mission + "\n\nOUTPUT_SCHEMA\n" + schema + "\nReturn one JSON object only.\n")


def _method_api_accounting(camp: Path) -> bool:
    path = camp / "goal.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("evaluation_mode") == "loso_method_search"


def transport_options(profile, *, model=None, max_tokens=None) -> dict:
    """Freeze model-facing fields; transport credentials are never captured."""
    value = {"model": model or profile.model,
             "max_tokens": max_tokens if max_tokens is not None else profile.max_tokens,
             "stream": False}
    if profile.supports_json_object:
        value["response_format"] = {"type": "json_object"}
    if profile.supports_temperature and profile.temperature is not None:
        value["temperature"] = profile.temperature
    if profile.supports_thinking:
        value["extra_body"] = {"thinking": {"type": "enabled" if profile.thinking else "disabled"}}
        if profile.thinking and profile.reasoning_effort:
            value["reasoning_effort"] = profile.reasoning_effort
    return value


def _remaining_api_calls(camp: Path) -> int | None:
    goal_path = camp / "goal.json"
    if not goal_path.is_file():
        return None
    maximum = json.loads(goal_path.read_text()).get("max_llm_calls")
    if maximum is None:
        return None
    cost_path = camp / "cost.json"
    used = json.loads(cost_path.read_text()).get("llm_calls", 0) if cost_path.is_file() else 0
    return max(0, int(maximum) - int(used))


async def _send_compiled(client, prepared, *, role: str, profile: str):
    from react_agent.eeg_research.agentic.context.service import send_prepared
    return await send_prepared(client, prepared, role=role, profile=profile)


def _ledgered_memory_maintenance(camp: Path, request: dict, **kwargs) -> dict:
    """Generate one proposal through native task/API ledgers and the compiler.

    The coordinator owns its single mechanical correction. This adapter makes
    one application attempt, not another nested two/four-attempt repair loop.
    """
    if external_api_disabled():
        raise LlmUnavailable("external_api_disabled_by_operator")
    from react_agent.eeg_research.agentic.context.compiler import render_role_system
    from react_agent.eeg_research.agentic.context.maintenance import research_delta_prompt
    from react_agent.eeg_research.agentic.context.sources import ContextSourceRegistry
    from react_agent.eeg_research.agentic.role_config import resolve_role_config
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task, role_recovery_identity
    from react_agent.fmri.config import FmriCheckConfig
    from react_agent.fmri.llm.deepseek import DeepSeekBackend
    mode = kwargs.get("mode") or request.get("mode") or "summary"
    if mode not in {"summary", "extraction", "consolidation"}:
        raise LlmUnavailable("invalid_memory_maintenance_mode")
    payload = dict(request.get("payload", request))
    correction = request.get("maintenance_correction") or kwargs.get("correction_context")
    if correction:
        payload["maintenance_correction"] = dict(correction)
    operation_ref = kwargs.get("operation_id")
    if operation_ref is None and request.get("input_digest"):
        from .context.correction import child_operation_id
        operation_ref = (child_operation_id(str(request["input_digest"]), request)
                         if correction else str(request["input_digest"]) + ":generate")
    if operation_ref:
        payload["maintenance_operation_id"] = str(operation_ref)
    if request.get("input_digest"):
        payload["maintenance_input_digest"] = request["input_digest"]
    payload.update(allow_semantic_compaction=False, allow_sync_memory_maintenance=False)
    turn = str(request.get("turn_id") or kwargs.get("turn_id") or payload.get("turn_id") or kwargs.get("operation_id", ""))
    operation_metadata = {key: kwargs[key] for key in (
        "original_maintenance_turn_id", "maintenance_authorization_ref") if kwargs.get(key)}
    if operation_metadata:
        from .native_authorization import current_authorization
        authorization = current_authorization(camp)
        if (not authorization or operation_metadata.get("maintenance_authorization_ref") != authorization["authorization_hash"]
                or operation_metadata.get("original_maintenance_turn_id") != authorization["parent_memory_turn_id"]
                or turn != authorization["memory_recovery_turn"]):
            raise LlmUnavailable("maintenance_authorization_metadata_invalid")
    # Scheduling epochs are routing metadata, not a new source version. A
    # restart in a later turn must be able to adopt an already paid proposal.
    payload.pop("turn_id", None)
    rows = []
    intents = camp / "llm_api_intents.jsonl"
    if intents.is_file():
        rows = [json.loads(line) for line in intents.read_text().splitlines() if line.strip()]
    upkeep = [r for r in rows if str(r.get("operation", "")).startswith("context_maintenance:")]
    role = "context_summary" if mode == "summary" else "memory_" + mode
    if mode == "summary":
        system = research_delta_prompt()
    elif mode == "extraction":
        system = (PROMPTS / "context_episode_extraction.txt").read_text()
    else:
        system = (PROMPTS / "context_memory_consolidation.txt").read_text()
    system = render_role_system(role, system)
    source = ContextSourceRegistry(camp).capture(payload,
        block_id="maintenance:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False,
            sort_keys=True, default=str).encode()).hexdigest(), kind="memory", role="memory_curator")
    resolved = resolve_role_config(camp, "memory_curator")
    recovery = role_recovery_identity(camp, "memory_curator", None, payload, source.source_hash)
    recovery.update(action="context_maintenance:" + mode,
        prompt_hash=hashlib.sha256(system.encode()).hexdigest()[:16],
        context_policy="eeg.context.v1",
        source_dependency_digest=source.source_hash)
    original_path = Path(source.payload_ref)
    if not original_path.is_absolute():
        original_path = camp / original_path
    from react_agent.eeg_research.agentic.context.native_recovery import recover_maintenance
    recovered = recover_maintenance(camp, request=payload, recovery_identity=recovery,
        inputs=[original_path], resolved=resolved, system=system,
        operation="context_maintenance:" + mode)
    if recovered is not None:
        from .task_ledger import task_snapshots
        recovered_task = next((row for row in reversed(task_snapshots(camp))
            if (row.get("recovery_identity") or {}).get("source_dependency_digest") == source.source_hash), {})
        return {"output": recovered, "_maintenance_transport": {"api_calls": 0,
            "cache_adopted": True, "task_ref": recovered_task.get("task_id", "")}}
    if len(upkeep) >= 100 or sum(r.get("maintenance_turn_id") == turn for r in upkeep) >= 2:
        raise LlmUnavailable("memory_maintenance_budget_exhausted")
    task = begin_role_task(camp, role="memory_curator", inputs=[original_path],
                           request=payload, recovery_identity=recovery)
    calls_path = camp / "llm_calls.jsonl"
    prior_calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()] if calls_path.is_file() else []
    prior_call = next((row for row in reversed(prior_calls) if row.get("task_id") == task["task_id"]), None)
    if (prior_call and prior_call.get("success") is True
            and prior_call.get("transport_outcome") == "confirmed_response"
            and not task.get("_cached_role_result")):
        # A paid object that failed proof checks is not permission to resend
        # the original generation. A semantic correction has its own input.
        raise LlmUnavailable("maintenance_paid_response_requires_recovery")
    cached = task.get("_cached_role_result")
    if cached:
        from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies
        from react_agent.eeg_research.agentic.task_ledger import record_task_reuse
        assert_role_task_dependencies(camp, task)
        record_task_reuse(camp, cached)
        return {"output": cached["envelope"]["payload"]["proposal"],
                "_maintenance_transport": {"api_calls": 0, "cache_adopted": True, "task_ref": task["task_id"]}}
    payload["turn_id"] = turn
    payload.update({k:task[k] for k in ("task_id", "attempt_id", "input_digest")})
    config = FmriCheckConfig(deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
                            deepseek_base_url=resolved.base_url)
    config.fast = resolved.profile()
    client, event_loop = DeepSeekBackend(config), asyncio.new_event_loop()
    try:
        response = _invoke_compaction(camp, client, event_loop, model=resolved.model,
            max_tokens=resolved.max_tokens, system=system, payload=payload, task=task,
            validate=lambda output: output if isinstance(output, dict) else (_ for _ in ()).throw(ValueError("proposal_not_object")),
            context_budget_chars=100_000, operation_name="context_maintenance:" + mode,
            mode=mode, attempts=1,
            **({"operation_metadata": operation_metadata} if operation_metadata else {}))
        finish_role_task(camp, task, {"status":"completed", "proposal":response,
            "publication_status":"proposal_only", "scientific_evidence":False,
            "execution_authority":False}, kind="handoff_compaction",
            path=camp / "context_proposals" / (task["task_id"] + ".json"))
        return {"output": response, "_maintenance_transport": {"api_calls": 1,
            "cache_adopted": False, "task_ref": task["task_id"]}}
    except Exception as exc:
        finish_role_task(camp, task, {"status":"failed", "error":str(exc),
            "publication_status":"not_published", "scientific_evidence":False},
            kind="handoff_compaction", path=camp / "context_proposals" / (task["task_id"] + ".json"))
        raise
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            try:
                event_loop.run_until_complete(close())
            except (OSError, RuntimeError):
                pass  # Cleanup does not change the recorded request outcome.
        event_loop.close()


def _write_cost(camp: Path, cost: dict[str, Any]) -> None:
    target = camp / "cost.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def refresh_method_api_cost(camp: Path) -> dict[str, Any]:
    """Reconcile native intent reservations and settled rows after a crash.

    One reserved application attempt always consumes a call. If transport may
    have occurred before persistence, the unmatched call remains uncertain and
    is never relabeled zero/exactly-once. Tokens remain unknown until settled.
    This opt-in reconciliation leaves GPU/external-budget fields intact.
    """
    cost_path = camp / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.is_file() else {}
    if not _method_api_accounting(camp):
        return cost
    intents = {}
    path = camp / API_INTENTS
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("call_id") or row.get("status") != "reserved_before_transport":
                    raise LlmUnavailable("api_intent_ledger_invalid")
                intents[str(row["call_id"])] = row
    settled = {}
    calls = camp / "llm_calls.jsonl"
    if calls.is_file():
        for index, line in enumerate(calls.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                row = json.loads(line)
                key = str(row.get("call_id") or ("legacy_row_" + str(index)))
                if key in settled and settled[key] != row:
                    raise LlmUnavailable("call_id_ledger_conflict")
                settled[key] = row
    from react_agent.eeg_research.agentic.api_policy import ATTEMPTS
    operation_path = camp / ATTEMPTS
    operation_rows = [json.loads(line) for line in operation_path.read_text().splitlines() if line.strip()] if operation_path.is_file() else []
    # Reservations precede the main intent journal. A crash in that narrow gap
    # still retains the call charge and requires explicit reconciliation.
    for item in operation_rows:
        intents.setdefault(item["call_id"], item)
    pending = sorted(set(intents).difference(settled))
    abandoned = sorted(key for key, row in settled.items()
                       if row.get("transport_outcome") in {"unknown_after_process_exit", "outcome_unknown"}
                       or row.get("unconfirmed_charge_possible"))
    cost["llm_calls"] = len(set(settled).union(intents))
    cost["llm_failures"] = sum(row.get("success") is not True for row in settled.values())
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        known = [row[key] for row in settled.values() if isinstance(row.get(key), int)]
        if known or key in cost:
            cost[key] = sum(known)
        cost[key + "_known_call_count"] = len(known)
    cost.update(api_unsettled_call_ids=pending, api_abandoned_call_ids=abandoned,
                api_uncertain_calls=len(pending) + len(abandoned),
                api_intent_accounting="application_attempt_reserved_before_transport_v1",
                api_usd=None, pricing_source=None,
                cost_status="uncertain" if pending or abandoned else "unpriced")
    known_usd = [row["api_usd"] for row in settled.values() if isinstance(row.get("api_usd"), (int, float))]
    cost["api_usd_known_subtotal"] = sum(known_usd) if known_usd else None
    if len(known_usd) == len(settled) and settled and not pending and not abandoned:
        cost["api_usd"], cost["cost_status"] = sum(known_usd), "priced"
        cost["pricing_source"] = sorted({str(row.get("pricing_source")) for row in settled.values()})
    reserve_by_id = {row["call_id"]: row.get("unknown_usd_reserve") for row in operation_rows}
    cost["api_unknown_usd_reserved"] = sum(float(reserve_by_id[key]) for key in set(pending + abandoned)
                                         if isinstance(reserve_by_id.get(key), (int, float)))
    cost.setdefault("gpu_seconds_used", 0.0)
    _write_cost(camp, cost)
    return cost


def _reserve_api_intent(camp: Path, row: dict[str, Any]) -> None:
    if not _method_api_accounting(camp):
        return
    camp.mkdir(parents=True, exist_ok=True)
    # Durability precedes transport. A crash between fsync and cost.json rename
    # is reconciled from the append-only intent on the next native budget read.
    intent = {name: row.get(name) for name in
              ("call_id", "role", "task_id", "attempt_id", "input_digest", "prompt_hash", "candidate_id", "input_hash",
               "request_trace_ref", "request_system_sha256", "request_user_sha256", "operation_id",
               "planner_round_id", "role_config_hash", "role_model_config", "requested_model",
               "maintenance_turn_id", "context_mode", "context_request_digest", "context_policy_version",
               "original_maintenance_turn_id", "maintenance_authorization_ref", "research_authorization_ref", "parent_planner_round_id",
               "thinking", "reasoning_effort", "unknown_usd_reserve")}
    if row.get("operation"):
        intent["operation"] = row["operation"]
    intent.update(status="reserved_before_transport", recorded_at=time.time(),
                  uncertain_if_unsettled=True)
    with (camp / API_INTENTS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(intent, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    refresh_method_api_cost(camp)


def _ledger(camp: Path, row: dict[str, Any]) -> None:
    camp.mkdir(parents=True, exist_ok=True)
    report = row.get("context_budget_report") or {}
    if report.get("model_tokens") and row.get("input_tokens") is not None:
        from react_agent.eeg_research.agentic.context.calibration import record_usage
        record_usage(camp, call_id=row["call_id"], role=row["role"],
            model=row["requested_model"], mode=row.get("context_mode", "normal"),
            estimated=report["model_tokens"], observed=row["input_tokens"])
    _price_usage(row)
    method_mode = _method_api_accounting(camp)
    ledger_path = camp / "llm_calls.jsonl"
    if method_mode and row.get("call_id") and ledger_path.is_file():
        prior = next((json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                      if line.strip() and json.loads(line).get("call_id") == row["call_id"]), None)
        if prior is not None:
            if prior != row:
                raise LlmUnavailable("call_id_ledger_conflict")
            refresh_method_api_cost(camp)
            return
    with (camp / "llm_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if method_mode:
            handle.flush()
            os.fsync(handle.fileno())
    if method_mode:
        refresh_method_api_cost(camp)
        return
    cost_path = camp / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.is_file() else {}
    cost["llm_calls"] = int(cost.get("llm_calls", 0)) + 1
    cost["llm_failures"] = int(cost.get("llm_failures", 0)) + (0 if row.get("success") else 1)
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        value = row.get(key)
        if isinstance(value, int):
            cost[key] = int(cost.get(key, 0)) + value
    cost["api_usd"] = None
    cost["pricing_source"] = None
    cost["cost_status"] = "unpriced"
    cost.setdefault("gpu_seconds_used", 0.0)
    cost_path.write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_trace(camp: Path, row: dict[str, Any], *, system: str,
                   user: str, max_tokens: int) -> dict[str, Any]:
    """Persist the exact application-level role input before its API attempt.

    Transport configuration and credentials are deliberately not serialized.
    This is the complete_json input, not a claim about raw HTTP wire bytes.
    """
    folder = camp / "llm_requests"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / (str(row["call_id"]) + ".json")
    system_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    user_sha = hashlib.sha256(user.encode("utf-8")).hexdigest()
    identity = {key: row.get(key) for key in
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_digest", "input_hash", "requested_model")}
    if row.get("operation"):
        identity["operation"] = row["operation"]
    record = {**identity, "system": system, "user": user,
              "system_sha256": system_sha, "user_sha256": user_sha,
              "profile": "fast", "max_tokens": max_tokens,
              "role_model_config": row.get("role_model_config"), "role_config_hash": row.get("role_config_hash"),
              "role_config_sources": row.get("role_config_sources"),
              "status": "prepared_before_transport", "recorded_at": time.time(),
              "scope": "Exact application-level system/user strings supplied to complete_json; no transport credentials or HTTP wire-body claim."}
    if row.get("context_transport_fields") is not None:
        record["transport_fields"] = row["context_transport_fields"]
    record["context_measurement"] = {"system_chars": len(system), "user_chars": len(user),
                                     "total_application_chars": len(system) + len(user)}
    temp = path.with_suffix(".json.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return {"request_trace_ref": str(path), "request_system_sha256": system_sha,
            "request_user_sha256": user_sha}


def _price_usage(row: dict) -> None:
    """Deployment supplied pricing only; reasoning is part of completion tokens."""
    source = os.environ.get("DEEPSEEK_PRICE_SOURCE")
    date = os.environ.get("DEEPSEEK_PRICE_DATE")
    input_rate = os.environ.get("DEEPSEEK_USD_INPUT_PER_MILLION")
    output_rate = os.environ.get("DEEPSEEK_USD_OUTPUT_PER_MILLION")
    inp, out = row.get("input_tokens"), row.get("output_tokens")
    row.setdefault("api_usd", None)
    if not all((source, date, input_rate, output_rate)) or not isinstance(inp, int) or not isinstance(out, int):
        return
    try:
        rates = [float(input_rate), float(output_rate)]
        if any(rate < 0 or not __import__("math").isfinite(rate) for rate in rates):
            return
        cached_rate = os.environ.get("DEEPSEEK_USD_CACHE_HIT_PER_MILLION")
        hit = row.get("cache_hit_tokens")
        hit = hit if cached_rate and isinstance(hit, int) and 0 <= hit <= inp else 0
        cache_rate = float(cached_rate) if cached_rate else rates[0]
        if cache_rate < 0 or not __import__("math").isfinite(cache_rate):
            return
        row["api_usd"] = ((inp - hit) * rates[0] + hit * cache_rate + out * rates[1]) / 1_000_000
        row.update(pricing_source=source, pricing_date=date,
                   cost_status="uncertain" if row.get("unconfirmed_charge_possible") else "priced")
    except ValueError:
        return


def _response_trace(camp: Path, row: dict[str, Any], reply: Any, *, validation: str,
                    schema_errors: Any = None, normalization: Any = None) -> None:
    """Keep the exact model response and validation outcome, without credentials."""
    folder = camp / "llm_responses"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (str(row["call_id"]) + ".json")
    identity = {key: row.get(key) for key in
                ("call_id", "role", "prompt_hash", "candidate_id", "attempt_id", "task_id", "input_digest", "input_hash",
                 "request_trace_ref", "request_system_sha256", "request_user_sha256")}
    if row.get("operation"):
        identity["operation"] = row["operation"]
    record = {**identity, "reply": reply, "validation": validation,
              "schema_errors": schema_errors, "recorded_at": time.time()}
    if normalization:
        record["normalization"] = normalization
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _fail_row(row: dict[str, Any], exc: BaseException, started: float) -> dict[str, Any]:
    row.update({"success": False, "error": type(exc).__name__, "cost_status": "uncertain", "elapsed_seconds": round(time.time() - started, 3)})
    usage = getattr(exc, "usage", None)
    if usage is not None:
        row.update(
            {
                "response_model": usage.response_model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "finish_reason": usage.finish_reason,
                "provider_error": usage.error,
                "failure_kind": getattr(usage, "failure_kind", None),
                "http_status": getattr(usage, "http_status", None),
                "unconfirmed_charge_possible": usage.unconfirmed_charge_possible,
                "transport_outcome": getattr(usage, "transport_outcome", None),
                "cache_hit_tokens": getattr(usage, "cache_hit_tokens", None), "cache_miss_tokens": getattr(usage, "cache_miss_tokens", None),
            }
        )
        row["cost_status"] = "uncertain" if usage.unconfirmed_charge_possible else "unpriced"
    else:
        from react_agent.fmri.llm.deepseek import DeepSeekConfigError
        if isinstance(exc, DeepSeekConfigError):
            # Frozen request/profile checks run before any SDK request exists.
            row.update(unconfirmed_charge_possible=False, transport_outcome="definitely_not_sent",
                       cost_status="unpriced", failure_kind="configuration_error")
        else:
            row.update(unconfirmed_charge_possible=True, transport_outcome="outcome_unknown")
    causes = []
    cause = exc.__cause__
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    if causes:
        row["error_causes"] = causes
    raw = getattr(exc, "raw", None)
    if isinstance(raw, str):
        row["raw_excerpt"] = raw[:RAW_EXCERPT_LIMIT]
    return row


def external_api_disabled() -> bool:
    """Operator process policy; local GPU execution remains independent."""
    return os.environ.get("EEG_EXTERNAL_API_DISABLED", "0").lower() in {"1", "true", "on"}


def _check_api_budget(camp: Path) -> None:
    """Apply the same native intent reconciliation to roles and compaction."""
    if external_api_disabled():
        raise LlmUnavailable("external_api_disabled_by_operator")
    control_path = camp / "control.json"
    if control_path.is_file() and json.loads(control_path.read_text()).get("action") in {"pause", "stop"}:
        raise LlmUnavailable("api_blocked_by_user_control")
    goal_path = camp / "goal.json"
    if not goal_path.is_file():
        return
    try:
        goal = json.loads(goal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        goal = {}
    used = 0
    cost_path = camp / "cost.json"
    if cost_path.is_file():
        try:
            used = int(json.loads(cost_path.read_text(encoding="utf-8")).get("llm_calls") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            used = 0
    if goal.get("evaluation_mode") == "loso_method_search":
        reconciled = refresh_method_api_cost(camp)
        used = int(reconciled.get("llm_calls") or 0)
        if reconciled.get("api_unsettled_call_ids"):
            raise LlmUnavailable("unsettled_api_intent_requires_recovery",
                diagnostics={"api_unsettled_call_ids": reconciled["api_unsettled_call_ids"],
                             "cost_status": "uncertain"})
        from react_agent.eeg_research.agentic.api_policy import unresolved_charge_ids
        unknown = sorted(unresolved_charge_ids(camp))
        if unknown:
            raise LlmUnavailable("unknown_api_charge_requires_disposition", diagnostics={"call_ids": unknown})
    limit = goal.get("max_llm_calls")
    if isinstance(limit, int) and used >= limit:
        raise LlmUnavailable("budget_exhausted")


# Quality targets: share of the hard role budget a request should stay under.
PLANNER_TARGET_RATIO = 0.7
CURATOR_TARGET_RATIO = 0.5


def _within_quality_target(projected: dict, ratio: float) -> bool:
    report = projected["role_context_projection"]
    return report["fits_budget"] and report["projected_total_chars"] <= ratio * report["budget_chars"]


# Compaction failures are mechanical (length/quote-pointer limits) and usually
# pass on a re-roll with the validation error attached.
COMPACTION_ATTEMPTS = 2


def _unknown_reserve(camp: Path) -> float | None:
    path = camp / "goal.json"
    goal = json.loads(path.read_text()) if path.is_file() else {}
    value = (goal.get("api_request_policy") or {}).get("unknown_usd_reserve")
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _adjust_output_budget(camp: Path, config: Any, row: dict) -> bool:
    """One adjustment shares the original operation allowance, never a new budget."""
    path = camp / "goal.json"
    goal = json.loads(path.read_text()) if path.is_file() else {}
    from react_agent.eeg_research.agentic.api_policy import api_request_policy
    policy = api_request_policy(camp).model_dump()
    if policy.get("allow_output_budget_adjustment", False) is not True:
        return False
    before = int(config.fast.max_tokens)
    ceiling = int(policy.get("max_output_tokens", before))
    after = min(before * 2, ceiling, 131072)
    if after <= before:
        return False
    config.fast.max_tokens = after
    before_timeout = config.fast.timeout_s
    if policy.get("output_retry_timeout_s") is not None:
        config.fast.timeout_s = policy["output_retry_timeout_s"]
    with (camp / "api_output_adjustments.jsonl").open("a") as stream:
        stream.write(json.dumps({"operation_id": row["operation_id"], "failed_call_id": row["call_id"],
            "before_max_tokens": before, "after_max_tokens": after, "reason": "output_truncated",
            "before_timeout_s": before_timeout, "after_timeout_s": config.fast.timeout_s,
            "additional_attempts_granted": 0, "at": time.time()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


def _invoke_compaction(camp: Path, client: Any, loop: Any, *, model: str, max_tokens: int,
                       system: str, payload: dict, task: dict, validate: Callable,
                       context_budget_chars: int, operation_name: str = "handoff_compaction",
                       mode: str = "summary", attempts: int | None = None,
                       operation_metadata: dict | None = None) -> dict:
    """One bounded delivery operation through the existing durable API ledger."""
    from react_agent.fmri.llm.deepseek import DeepSeekParseError
    from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies
    from react_agent.eeg_research.agentic.role_config import resolve_role_config
    from react_agent.eeg_research.agentic.api_policy import operation_id, reserve_operation_attempt
    resolved = resolve_role_config(camp, "memory_curator")
    logical_operation = operation_id(operation_name, payload, {}, resolved.role_config_hash)
    error = None
    previous_summary = None
    from react_agent.eeg_research.agentic.context.service import ContextService, budget_for_role, send_prepared
    from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
    service = ContextService(camp)
    from react_agent.eeg_research.agentic.context.calibration import tokenizer_for
    enable = getattr(client, "enable_context_protocol", None)
    if enable:
        enable()
    for attempt in range(COMPACTION_ATTEMPTS if attempts is None else attempts):
        assert_role_task_dependencies(camp, task)
        request = dict(payload)
        if error is not None:
            request["summary_validation_error"] = error
        if previous_summary is not None:
            request["previous_compaction_summary"] = previous_summary
            probe = json.dumps(request, ensure_ascii=False, default=str, separators=(",", ":"))
            if len(system) + len(probe) > context_budget_chars:
                # Optional invalid draft is omitted wholly; original source
                # and measured repair feedback are still delivered in full.
                request.pop("previous_compaction_summary")
        compiled = service.compile(CompileRequest(camp.name,
            "context_summary" if mode == "summary" else "memory_" + mode,
            request, system=system, model=model, max_output_tokens=max_tokens,
            task_id=task["task_id"], intent_id=logical_operation,
            purpose=operation_name, mode=mode,
            budget=budget_for_role("memory_curator"), identity=task,
            tokenizer=tokenizer_for(camp, role="memory_curator", model=model, mode=mode),
            transport_options=transport_options(resolved.profile(), model=model, max_tokens=max_tokens),
            allow_semantic_compaction=False, allow_sync_memory_maintenance=False))
        if not isinstance(compiled, Ready):
            raise LlmUnavailable("context_maintenance_" + compiled.status,
                                 diagnostics={"context_result": str(compiled)[:2000]})
        prepared = compiled.prepared
        user = prepared.user
        _check_api_budget(camp)
        started = time.time()
        row = new_call_row(role="memory_curator", operation=operation_name,
            prompt_hash=hashlib.sha256(prepared.system.encode()).hexdigest()[:16],
            requested_model=model, started_at=started,
            **{key: task[key] for key in ("task_id", "attempt_id", "input_digest")})
        row.update(application_attempt=attempt + 1,
                   operation_id=logical_operation, role_config_hash=resolved.role_config_hash,
                   role_config_sources=resolved.sources,
                   role_model_config=resolved.semantics(), thinking=resolved.thinking,
                   reasoning_effort=resolved.reasoning_effort,
                   application_retry_kind="compaction_validation" if error else None)
        row.update(context_request_digest=prepared.request_digest,
                   context_transport_fields=prepared.transport_fields,
                   context_sidecars=prepared.sidecar_refs, context_mode=mode,
                   context_policy_version=prepared.policy_version,
                   context_budget_report=prepared.budget_report,
                   maintenance_turn_id=payload.get("turn_id"))
        row.update({key: value for key, value in (operation_metadata or {}).items()
                    if key in {"original_maintenance_turn_id", "maintenance_authorization_ref"}})
        row.update(_request_trace(camp, row, system=prepared.system, user=user, max_tokens=max_tokens))
        row.update(reserve_operation_attempt(camp, operation=logical_operation, call_id=row["call_id"],
                   reserve_usd=_unknown_reserve(camp), metadata=row))
        _reserve_api_intent(camp, row)
        try:
            reply, usage = loop.run_until_complete(send_prepared(client, prepared,
                profile="fast", role="memory_curator"))
        except DeepSeekParseError as exc:
            _ledger(camp, _fail_row(row, exc, started))
            kind = getattr(exc.usage, "failure_kind", None)
            _response_trace(camp, row, getattr(exc, "raw", None), validation=kind or "invalid_json")
            if kind not in {None, "format_error", "output_truncated", "missing_final_content"}:
                raise LlmUnavailable(kind) from exc
            error = "Return one complete JSON object matching OUTPUT_SCHEMA."
            continue
        except Exception as exc:
            _ledger(camp, _fail_row(row, exc, started))
            raise LlmUnavailable(type(exc).__name__) from exc
        row.update(success=True, response_model=usage.response_model,
                   input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   reasoning_tokens=usage.reasoning_tokens, finish_reason=usage.finish_reason,
                   cache_hit_tokens=getattr(usage, "cache_hit_tokens", None), cache_miss_tokens=getattr(usage, "cache_miss_tokens", None),
                   token_count_source=getattr(usage, "token_count_source", None), transport_outcome="confirmed_response",
                   elapsed_seconds=round(time.time() - started, 3))
        row["response_reply_sha256"] = hashlib.sha256(json.dumps(reply, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        _ledger(camp, row)
        _response_trace(camp, row, reply, validation="pending")
        try:
            checked = validate(reply)
        except (ValueError, TypeError, KeyError) as exc:
            from react_agent.eeg_research.agentic.compaction import canonical, summary_budget
            schema_error = str(exc)[:3000]
            previous_summary = reply
            error = {"validation_error": schema_error, "previous_output_chars": len(canonical(reply)),
                     "hard_output_chars": payload.get("max_output_chars") or summary_budget(payload.get("stage", "map")),
                     "target_output_chars": payload.get("target_output_chars", 6000),
                     "instruction": payload.get("repair_instruction") or "Fix the reported validation errors against the original input. For source_quotes copy an exact path from analysis_quote_catalog and a verbatim substring of its text, never an evidence ID or observation object path. Rewrite the full draft with shorter synthesized notes and fewer repeated facts; retain all distinct findings, contradictions, scope limits and expected source refs. Follow every OUTPUT_SCHEMA length/item limit. Do not truncate."}
            _response_trace(camp, row, reply, validation="compaction_invalid", schema_errors=schema_error)
            continue
        _response_trace(camp, row, reply, validation="valid")
        return checked
    raise LlmUnavailable("compaction_validation_failed", diagnostics={"reason": error})


def role_backend(camp: Path, role: str, *, overrides: dict | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a callable that sends one structured input to DeepSeek for this role."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LlmUnavailable("DEEPSEEK_API_KEY missing")
    from react_agent.fmri.config import FmriCheckConfig
    from react_agent.fmri.llm.deepseek import DeepSeekBackend, DeepSeekParseError
    from react_agent.eeg_research.agentic.role_config import resolve_role_config, ResolvedRoleConfig, FIELDS
    from react_agent.eeg_research.agentic.api_policy import operation_id, reserve_operation_attempt

    resolved = resolve_role_config(camp, role, overrides)

    config = FmriCheckConfig(
        deepseek_api_key=key,
        deepseek_base_url=resolved.base_url,
        deepseek_trust_env=os.environ.get("DEEPSEEK_TRUST_ENV", "true").strip().lower() not in {"0", "false", "no"},
    )
    config.fast = resolved.profile()
    from react_agent.eeg_research.agentic.embedding import memory_config
    memory = memory_config(camp)
    system = system_prompt(role, memory=memory)
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]
    loop = asyncio.new_event_loop()
    client = DeepSeekBackend(config)
    enable_context = getattr(client, "enable_context_protocol", None)
    if enable_context:
        enable_context()
    from react_agent.eeg_research.agentic.context.service import ContextService
    context_service = ContextService(camp)
    def maintenance_callback(request, **kwargs):
        before = set()
        ledger = camp / "llm_calls.jsonl"
        if ledger.is_file():
            before = {json.loads(line)["call_id"] for line in ledger.read_text().splitlines() if line.strip()}
        try:
            return _ledgered_memory_maintenance(camp, request, **kwargs)
        finally:
            if ledger.is_file():
                for line in ledger.read_text().splitlines():
                    row = json.loads(line)
                    if row["call_id"] not in before and str(row.get("operation", "")).startswith("context_maintenance:"):
                        context_service.owned_calls.add(row["call_id"])
    context_service.callback = maintenance_callback
    bound: dict[str, Any] = {}

    def configure(changes: dict) -> None:
        nonlocal resolved
        merged = {name: getattr(resolved, name) for name in FIELDS}
        merged.update(changes)
        resolved = resolve_role_config(camp, role, merged)
        config.fast = resolved.profile()
        call.model = resolved.model
        call.role_config = resolved

    def bind(**fields: Any) -> None:
        bound.clear()
        bound.update({key: value for key, value in fields.items() if value is not None})
        call.identity = dict(bound)

    def prepare_context(payload: dict[str, Any], *, inputs: list[Path]) -> dict:
        # All roles compile at the common transport boundary. Keep the native
        # business request unchanged for its task/artifact digest and validator.
        # Working-memory upkeep is scheduled from closed producers, not from
        # another role's flat output schema.
        return payload

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal resolved
        task = None
        if role in ENVELOPE_ROLES:
            from react_agent.eeg_research.agentic.roles import task_identity_from_payload

            task = task_identity_from_payload(payload, bound)
            if task is None:
                raise LlmUnavailable("role_result_task_missing")
            from react_agent.eeg_research.agentic.task_ledger import (
                find_completed_task, latest, record_task_reuse, recovery_identity_digest,
            )
            persisted = latest(camp, str(task["task_id"]))
            recovery = (persisted or {}).get("recovery_identity")
            recovery_view = recovery if isinstance(recovery, dict) else {}
            recovery_config = recovery_view.get("config")
            recovery_config = recovery_config if isinstance(recovery_config, dict) else {}
            saved_config = recovery_config.get("role_model_config")
            saved_config = saved_config if isinstance(saved_config, dict) else None
            if persisted and persisted.get("input_digest") != task.get("input_digest"):
                raise LlmUnavailable("role_result_task_identity_mismatch")
            if persisted and persisted.get("status") in {"pending", "running", "waiting_for_evidence", "completed"}:
                if (not isinstance(saved_config, dict) or not saved_config.get("role_config_hash")
                        or any(name not in saved_config for name in (*FIELDS, "base_url"))):
                    raise LlmUnavailable("role_config_changed_requires_new_task" if persisted.get("status") == "completed"
                                         else "started_role_config_proof_missing")
                # A started role task resumes with its bound routing. Switching
                # settings must create a new tracked attempt at the caller.
                fields = {name: saved_config[name] for name in FIELDS}
                try:
                    restored = resolve_role_config(camp, role, fields, environ={"DEEPSEEK_BASE_URL": saved_config["base_url"]})
                except (ValueError, TypeError) as exc:
                    raise LlmUnavailable("started_role_config_proof_invalid") from exc
                if restored.role_config_hash != saved_config["role_config_hash"]:
                    raise LlmUnavailable("started_role_config_hash_mismatch")
                if restored.base_url != config.deepseek_base_url:
                    raise LlmUnavailable("started_role_config_base_url_changed")
                restored = restored.model_copy(update={"sources": recovery_config.get("role_config_sources") or restored.sources})
                resolved, config.fast = restored, restored.profile()
                call.role_config, call.model = restored, restored.model
            if persisted and str((recovery or {}).get("action") or "").startswith(("handoff_compaction:", "context_maintenance:")):
                raise LlmUnavailable("delivery_operation_task_not_role_task")
            if persisted and persisted.get("role") == role and persisted.get("input_digest") == task.get("input_digest"):
                from react_agent.eeg_research.agentic.roles import assert_role_task_dependencies, RoleResultError
                if persisted.get("status") == "completed" and "input_dependencies" not in (recovery or {}):
                    raise LlmUnavailable("completed_role_cache_dependency_proof_missing")
                try:
                    assert_role_task_dependencies(camp, persisted)
                except RoleResultError as exc:
                    raise LlmUnavailable(str(exc)) from exc
            # Reuse only an explicit current identity established by the native
            # begin-task consumer. A legacy empty prompt/digest is not guessed.
            if (persisted and persisted.get("role") == role and isinstance(recovery, dict)
                    and persisted.get("input_digest") == task.get("input_digest")
                    and recovery.get("prompt_hash") == prompt_hash
                    and (saved_config or {}).get("role_config_hash") == resolved.role_config_hash
                    and persisted.get("recovery_identity_digest") == recovery_identity_digest(recovery)):
                cached = find_completed_task(camp, role=role, recovery_identity=recovery,
                                             candidate_id=persisted.get("candidate_id"),
                                             input_digest=task.get("input_digest"))
                if cached and cached["task"]["task_id"] == task["task_id"]:
                    record_task_reuse(camp, cached)
                    return cached["envelope"]
            if persisted:
                task = persisted
        from .context.role_recovery import recover_role_reply, require_current_prompt
        from .roles import RoleResultError
        from .api_policy import operation_id
        logical_operation = operation_id(role, payload, bound, resolved.role_config_hash)
        request_semantics = operation_id(role, payload,
            {key: value for key, value in bound.items() if key != "operation_id"}, "")

        def validate_recovered(reply, frozen_payload, frozen_prompt_hash):
            parsed = dict(reply)
            if role == "research_planner":
                from .planner_context import normalize_input_echoes
                parsed, _ = normalize_input_echoes(parsed, frozen_payload)
            if role == "candidate_coder":
                from .coder import validate_tool_request
                _, _, error = validate_tool_request(parsed)
                if error:
                    raise RoleResultError("recovered_coder_tool_request_invalid")
            else:
                model = domain_model_for(role, memory)
                body = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else parsed
                identities = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "artifact_refs", "candidate_id"}
                if model is not None:
                    body = {key: value for key, value in body.items() if key not in identities
                            and (key != "status" or "status" in model.model_fields)}
                    model.model_validate(body)
                if role == "candidate_reviewer":
                    from .review_context import review_reference_errors
                    if review_reference_errors(body, frozen_payload):
                        raise RoleResultError("recovered_review_evidence_reference_invalid")
            if role in ENVELOPE_ROLES:
                from .roles import bind_role_output
                return bind_role_output(parsed, task=task, prompt_hash=frozen_prompt_hash)
            return parsed
        try:
            recovered = recover_role_reply(camp, role=role, resolved=resolved,
                validate_reply=validate_recovered, task=task, operation_id=logical_operation, identity=bound,
                request_semantics_digest=request_semantics)
            if recovered is not None:
                return recovered
            require_current_prompt(task, prompt_hash)
        except (RoleResultError, ValueError) as exc:
            raise LlmUnavailable(str(exc)) from exc
        compile_identity = dict(bound)
        if task is not None:
            compile_identity.update({key: task.get(key) for key in (
                "task_id", "attempt_id", "input_digest", "candidate_id")})
        compile_task_id = str(compile_identity.get("task_id") or "")
        option_repair = payload.get("option_repair_request")
        failed_options = option_repair.get("failed_options") if isinstance(option_repair, dict) else None
        external_repair = role == "research_planner" and (
            (isinstance(option_repair, dict)
             and option_repair.get("kind") in {"repair_failed_options", "replace_unsupported_options"}
             and isinstance(failed_options, list) and bool(failed_options)
             and all(isinstance(item, dict) and item.get("status") == "rejected"
                     and isinstance(item.get("error"), str) and bool(item["error"])
                     and isinstance(item.get("draft"), dict) for item in failed_options))
            or (isinstance(payload.get("schema_error"), str) and bool(payload["schema_error"])
                and isinstance(payload.get("previous"), dict) and bool(payload["previous"])))
        transport_retry_prepared = None
        last: DeepSeekParseError | None = None
        schema_error = None
        schema_error_kind = "domain_schema_invalid"
        previous_domain = None
        schema_feedback = None
        logical_operation = operation_id(role, payload, bound, resolved.role_config_hash)
        for _attempt in range(PARSE_ATTEMPTS):
            request_mode = "repair" if external_repair or last is not None or schema_error is not None else "normal"
            call_payload = payload
            if last is not None:
                failure = getattr(last.usage, "failure_kind", None)
                error_feedback = {
                    "kind": "output_truncated" if failure == "output_truncated" else "invalid_json",
                    "finish_reason": getattr(last.usage, "finish_reason", None),
                    "instruction": "Return one complete JSON object matching OUTPUT_SCHEMA. Escape Python newlines and quotes inside content. Keep the implementation concise; do not omit requested interventions.",
                }
                if failure == "output_truncated":
                    from .context.types import digest as response_digest
                    error_feedback.update(correction_kind="concise_final_json", parent_call_id=row["call_id"],
                        parent_task_ref=row.get("task_id"), invalid_response_digest=response_digest(last.raw),
                        max_output_tokens=int(config.fast.max_tokens),
                        instruction="The previous final JSON was truncated. Return a concise complete OUTPUT_SCHEMA object within the current output cap. Preserve required executable hooks, parameter values, core implementation requirements and source references; omit repeated input context, policies and optional narrative. Finish the final JSON promptly.")
                call_payload = {
                    **payload,
                    "previous_response_error": error_feedback,
                }
            if schema_error is not None:
                call_payload = {**call_payload, "schema_error": schema_error, "previous": previous_domain,
                                "schema_repair_context": schema_feedback,
                                "correction_instruction": "Return a complete corrected JSON object using only OUTPUT_SCHEMA fields. Input metadata is read-only, not output. Remove every reported forbidden field; correct every reported missing/invalid field. Keep the task, legal actions, evidence identities and business context unchanged."}
            from react_agent.eeg_research.agentic.role_examples import add_role_examples
            from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready
            from react_agent.eeg_research.agentic.context.service import budget_for_role
            from react_agent.eeg_research.agentic.context.calibration import tokenizer_for
            call_payload = add_role_examples(call_payload, role, memory)
            if transport_retry_prepared is not None:
                prepared = transport_retry_prepared
                transport_retry_prepared = None
            else:
                outcome = context_service.compile(CompileRequest(
                    camp.name, role, call_payload, system=system, model=config.fast.model,
                    max_output_tokens=int(config.fast.max_tokens), task_id=compile_task_id,
                    intent_id=logical_operation, purpose=bound.get("phase", role),
                    mode=request_mode,
                    identity=compile_identity, budget=budget_for_role(role),
                    transport_options=transport_options(config.fast),
                    tokenizer=tokenizer_for(
                        camp, role=role, model=config.fast.model,
                        mode=request_mode),
                    # Native task identities are already frozen. Upkeep happens
                    # in the Planner/controller or an independent closed-group turn.
                    allow_semantic_compaction=not bool(compile_task_id),
                    allow_sync_memory_maintenance=not bool(compile_task_id),
                    remaining_api_calls=_remaining_api_calls(camp)))
                if not isinstance(outcome, Ready):
                    raise LlmUnavailable("context_" + outcome.status,
                        diagnostics={"reason":getattr(outcome,"reason",None),
                                     "budget_report":outcome.budget_report,
                                     "context_result":str(outcome)[:2000]})
                prepared = outcome.prepared
            call_payload = prepared.payload
            projection = {"revision":prepared.policy_version,
                          "original_user_chars":len(json.dumps(payload,ensure_ascii=False,default=str)),
                          "projected_total_chars":prepared.budget_report["chars"],
                          "budget_chars":prepared.budget_report["hard_chars"]}
            _check_api_budget(camp)
            started = time.time()
            row = new_call_row(
                role=role,
                prompt_hash=prompt_hash,
                requested_model=config.fast.model,
                started_at=started,
                **bound,
            )
            if task is not None:
                row.update({name: task.get(name) for name in ("task_id", "attempt_id", "input_digest")})
                row.setdefault("input_hash", task.get("input_digest"))
            row.update(application_attempt=_attempt + 1,
                       role_request_semantics_digest=request_semantics,
                       operation_id=logical_operation, role_config_hash=resolved.role_config_hash,
                       role_config_sources=resolved.sources,
                       role_model_config=resolved.semantics(), thinking=resolved.thinking,
                       reasoning_effort=resolved.reasoning_effort,
                       application_retry_kind=("domain_schema" if schema_error is not None else
                                               "output_truncated" if last is not None and getattr(last.usage, "failure_kind", None) == "output_truncated" else
                                               "invalid_json" if last is not None else "native_planner_repair" if external_repair else None),
                       role_context_revision=projection["revision"],
                       role_context_original_user_chars=projection["original_user_chars"],
                       role_context_projected_total_chars=projection["projected_total_chars"],
                       role_context_budget_chars=projection["budget_chars"])
            append_ui_event(
                camp,
                "llm_call_started",
                role=role,
                call_id=row.get("call_id"),
                candidate_id=bound.get("candidate_id"),
                task_id=bound.get("task_id"),
                attempt_id=bound.get("attempt_id"),
                job_id=bound.get("job_id"),
                status="active",
            )
            user_text = prepared.user
            row.update(context_request_digest=prepared.request_digest,
                       context_transport_fields=prepared.transport_fields,
                       context_mode=request_mode,
                       context_sidecars=prepared.sidecar_refs, context_policy_version=prepared.policy_version,
                       context_budget_report=prepared.budget_report)
            row.update(_request_trace(camp, row, system=prepared.system, user=user_text,
                                      max_tokens=int(config.fast.max_tokens)))
            row.update(reserve_operation_attempt(camp, operation=logical_operation, call_id=row["call_id"],
                       reserve_usd=_unknown_reserve(camp), metadata=row))
            _reserve_api_intent(camp, row)
            try:
                reply, usage = loop.run_until_complete(
                    _send_compiled(client, prepared, profile="fast", role=role)
                )
            except DeepSeekParseError as exc:
                last = exc
                _ledger(camp, _fail_row(row, exc, started))
                _response_trace(camp, row, getattr(exc, "raw", None), validation=getattr(exc.usage, "failure_kind", None) or "invalid_json")
                append_ui_event(
                    camp,
                    "llm_parse_retry",
                    role=role,
                    call_id=row.get("call_id"),
                    candidate_id=bound.get("candidate_id"),
                    status="failed",
                    error=type(exc).__name__,
                )
                kind = getattr(exc.usage, "failure_kind", None)
                if kind == "output_truncated":
                    if (_attempt + 1 >= PARSE_ATTEMPTS or exc.usage.unconfirmed_charge_possible
                            or getattr(exc.usage, "transport_outcome", None) != "confirmed_response"):
                        raise LlmUnavailable("output_truncated") from exc
                    if _adjust_output_budget(camp, config, row):
                        resolved = resolve_role_config(camp, role, {name: getattr(config.fast, name) for name in FIELDS})
                    # A confirmed length failure can use the existing second
                    # mechanical attempt at the same cap when expansion is not
                    # authorized. Operation/round/global gates still reserve it.
                elif kind not in {None, "format_error", "missing_final_content"}:
                    raise LlmUnavailable(kind) from exc
                continue
            except Exception as exc:  # noqa: BLE001
                _ledger(camp, _fail_row(row, exc, started))
                append_ui_event(
                    camp,
                    "llm_call_failed",
                    role=role,
                    call_id=row.get("call_id"),
                    candidate_id=bound.get("candidate_id"),
                    status="failed",
                    error=type(exc).__name__,
                )
                usage = getattr(exc, "usage", None)
                if usage is not None and getattr(usage, "http_status", None) in {429, 500, 502, 503, 504} and _attempt + 1 < PARSE_ATTEMPTS:
                    from react_agent.eeg_research.agentic.api_policy import api_request_policy
                    transport_retry_prepared = prepared
                    loop.run_until_complete(asyncio.sleep(api_request_policy(camp).transport_backoff_s))
                    continue
                raise LlmUnavailable(type(exc).__name__) from exc
            row.update(
                {
                    "success": True,
                    "response_model": usage.response_model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "finish_reason": usage.finish_reason,
                    "cache_hit_tokens": getattr(usage, "cache_hit_tokens", None), "cache_miss_tokens": getattr(usage, "cache_miss_tokens", None),
                    "token_count_source": getattr(usage, "token_count_source", None), "transport_outcome": "confirmed_response",
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            from .context.types import digest as response_digest
            row["response_reply_sha256"] = response_digest(reply)
            _ledger(camp, row)
            _response_trace(camp, row, reply, validation="pending")
            parsed = reply if isinstance(reply, dict) else {"_not_object": reply}
            normalization = None
            if role == "candidate_coder":
                from react_agent.eeg_research.agentic.coder import validate_tool_request, tool_catalog
                _, _, request_error = validate_tool_request(parsed)
                if request_error is not None:
                    # A malformed tool call has executed no step. Repair its
                    # envelope/args inside this same persistent operation rather
                    # than advancing the native coding session's tool counter.
                    schema_error = json.dumps(request_error, ensure_ascii=False, default=str)
                    schema_error_kind = "coder_tool_request_invalid"
                    previous_domain = parsed
                    schema_feedback = {"failure_kind": "format_error", "validation_error": request_error,
                        "allowed_tool_arguments": tool_catalog(),
                        "instruction": "Correct only the tool envelope and parameter format against this catalog. Return one tool/args object for the same current step; no tool executed or source changed."}
                    last = None
                    _response_trace(camp, row, reply, validation=schema_error_kind, schema_errors=request_error)
                    append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                    status="failed", error=schema_error_kind)
                    continue
            if role == "research_planner":
                from react_agent.eeg_research.agentic.planner_context import normalize_input_echoes
                parsed, removed = normalize_input_echoes(parsed, payload)
                if removed:
                    normalization = {"kind": "identical_read_only_input_echoes_removed",
                                     "removed_fields": removed, "normalized_reply": parsed,
                                     "raw_model_reply_preserved": True}
            bound_reply = None
            if role in ENVELOPE_ROLES:
                from react_agent.eeg_research.agentic.roles import RoleResultError, bind_role_output
                try:
                    bound_reply = bind_role_output(parsed, task=task, prompt_hash=prompt_hash)
                except RoleResultError as exc:
                    _response_trace(camp, row, reply, validation="role_identity_invalid", schema_errors=str(exc))
                    append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                    status="failed", error=str(exc))
                    raise LlmUnavailable(str(exc)) from exc
            if role != "candidate_coder":
                from pydantic import ValidationError
                domain_model = domain_model_for(role, memory)
                body = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else parsed
                identity_keys = {"schema_version", "task_id", "attempt_id", "input_digest", "prompt_hash", "artifact_refs", "candidate_id"}
                if domain_model is not None:
                    body = {key: value for key, value in body.items() if key not in identity_keys
                            and (key != "status" or "status" in domain_model.model_fields)}
                    try:
                        domain_model.model_validate(body)
                    except ValidationError as exc:
                        schema_error = json.dumps(exc.errors(include_input=False), ensure_ascii=False, default=str)[:3000]
                        _response_trace(camp, row, reply, validation="domain_schema_invalid",
                                        schema_errors=exc.errors(include_input=False))
                        previous_domain = body
                        errors = exc.errors(include_input=False)
                        schema_feedback = {
                            "allowed_top_level_fields": sorted(domain_model.model_fields),
                            "forbidden_field_paths": [list(row["loc"]) for row in errors if row["type"] == "extra_forbidden"],
                            "missing_field_paths": [list(row["loc"]) for row in errors if row["type"] == "missing"],
                            "validation_errors": errors[:12],
                        }
                        last = None
                        append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                        status="failed", error="domain_schema_invalid")
                        continue
                    if role == "candidate_reviewer":
                        from react_agent.eeg_research.agentic.review_context import review_reference_errors
                        errors = review_reference_errors(body, payload)
                        if errors:
                            schema_error = json.dumps(errors, ensure_ascii=False)
                            previous_domain = body
                            schema_feedback = {
                                "validation_errors": errors,
                                "required_evidence_identity": payload.get("requirement_evidence_identity"),
                                "instruction": "Correct only review evidence metadata against the injected current source/check identities. Keep the scientific judgment and unresolved issues; a malformed reference is not a candidate source defect and does not require a code edit.",
                            }
                            _response_trace(camp, row, reply, validation="review_evidence_reference_invalid",
                                            schema_errors=errors)
                            append_ui_event(camp, "llm_call_failed", role=role, call_id=row.get("call_id"),
                                            status="failed", error="review_evidence_reference_invalid")
                            last = None
                            continue
            _response_trace(camp, row, reply, validation="valid", normalization=normalization)
            append_ui_event(camp, "llm_call_finished", role=role, call_id=row.get("call_id"),
                            candidate_id=bound.get("candidate_id"), status="completed")
            if bound_reply is not None:
                return bound_reply
            return parsed
        raise LlmUnavailable(schema_error_kind if schema_error is not None else
                             (type(last).__name__ if last is not None else "DeepSeekParseError"),
                             diagnostics={"domain_schema_errors": schema_error} if schema_error is not None else {}) from last

    call.model = config.fast.model  # type: ignore[attr-defined]
    call.bind = bind  # type: ignore[attr-defined]
    call.prepare_context = prepare_context  # type: ignore[attr-defined]
    call.context_service = context_service  # type: ignore[attr-defined]
    call.configure = configure  # type: ignore[attr-defined]
    call.role_config = resolved  # type: ignore[attr-defined]
    call.identity = dict(bound)  # type: ignore[attr-defined]
    return call
