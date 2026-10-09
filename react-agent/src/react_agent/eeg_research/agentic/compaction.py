"""Model-written curator handoffs, with immutable sources and native task recovery.

This is a delivery operation, not a scientific role, analysis, or memory lesson.
Every character of the business input is sent in bounded map requests. A reduce
request combines their summaries; it never silently drops a source or truncates
a model reply to meet the consumer's budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

REVISION = "curator_semantic_compaction.v1"
ROLE = "memory_curator"
OPERATION = "handoff_compaction"
INPUT_CHARS = 100_000
PART_CHARS = 60_000
SUMMARY_CHARS = 10_000
MAX_PARTS = 8


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=str, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class SourceFact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1)
    value: int | float


class CitedNote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)
    source_refs: list[str] = Field(min_length=1, max_length=MAX_PARTS)
    facts: list[SourceFact] = Field(default_factory=list, max_length=20)


class CompactionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_refs: list[str] = Field(min_length=1, max_length=MAX_PARTS)
    observations: list[CitedNote] = Field(max_length=20)
    interpretations: list[CitedNote] = Field(max_length=20)
    contradictions: list[CitedNote] = Field(max_length=20)
    uncertainties: list[CitedNote] = Field(max_length=20)
    applicability: list[CitedNote] = Field(max_length=20)
    open_questions: list[CitedNote] = Field(max_length=20)


NOTE_FIELDS = ("observations", "interpretations", "contradictions", "uncertainties",
               "applicability", "open_questions")


def compaction_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "handoff_compaction.txt"
    return path.read_text(encoding="utf-8") + "\nOUTPUT_SCHEMA\n" + canonical(
        CompactionSummary.model_json_schema()) + "\nReturn one JSON object only."


def source_value(payload: Any, pointer: str) -> Any:
    """Resolve a fact only in the original full input, not in an earlier summary."""
    if not pointer.startswith("/"):
        raise ValueError("compaction_fact_pointer_invalid")
    value = payload
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdecimal() or str(int(token)) != token:
                raise ValueError("compaction_fact_pointer_invalid")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("compaction_fact_pointer_invalid")
    return value


def validate_summary(reply: dict, payload: dict, expected_refs: list[str]) -> dict:
    """Check declared coverage, exact structured facts, and finite output size.

    Citation/number validation is not a proof of faithful natural-language
    interpretation. A real model summary still requires source review.
    """
    result = CompactionSummary.model_validate(reply).model_dump()
    if (len(result["source_refs"]) != len(set(result["source_refs"]))
            or set(result["source_refs"]) != set(expected_refs)):
        raise ValueError("compaction_source_coverage_mismatch")
    notes = [note for field in NOTE_FIELDS for note in result[field]]
    if not notes:
        raise ValueError("compaction_empty_summary")
    for note in notes:
        if not set(note["source_refs"]).issubset(expected_refs):
            raise ValueError("compaction_unknown_source")
        for fact in note["facts"]:
            try:
                original = source_value(payload, fact["path"])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("compaction_fact_pointer_invalid") from exc
            if (isinstance(original, bool) or not isinstance(original, (int, float))
                    or original != fact["value"]):
                raise ValueError("compaction_fact_mismatch")
    if len(canonical(result)) > SUMMARY_CHARS:
        raise ValueError("compaction_summary_budget_exceeded")
    return result


def split_input(payload: dict) -> tuple[str, list[dict]]:
    """Partition the entire canonical input; ranges cover it exactly once."""
    text = canonical(payload)
    if len(text) > PART_CHARS * MAX_PARTS:
        raise ValueError("compaction_input_budget_exceeded")
    parts = []
    start = 0
    while start < len(text):
        if len(parts) >= MAX_PARTS:
            raise ValueError("compaction_input_budget_exceeded")
        # part.text is encoded again inside the transport JSON. Account for
        # that escaping, not merely the size of the original source range.
        end = min(len(text), start + PART_CHARS)
        if len(canonical(text[start:end])) > PART_CHARS:
            low, high = start + 1, end
            while low < high:
                middle = (low + high + 1) // 2
                if len(canonical(text[start:middle])) <= PART_CHARS:
                    low = middle
                else:
                    high = middle - 1
            end = low
        chunk = text[start:end]
        parts.append({"ref": f"input_part_{len(parts) + 1}", "start": start,
                      "end": start + len(chunk), "text": chunk,
                      "sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest()})
        start = end
    return text, parts


def protected_context(payload: dict) -> dict:
    """Carry original scientific/permission records alongside the model summary.

    Narrative fields are represented by the model-written summary, after the
    model has consumed their full content. Numeric run/episode/comparison facts,
    permissions, source allowlists and evidence-level controls stay literal.
    """
    narrated = {"diagnostics", "analyses", "design", "related_lessons", "related_skills"}
    result = copy.deepcopy({key: value for key, value in payload.items() if key not in narrated})
    if isinstance(result.get("run"), dict):
        result["run"].pop("diagnostics", None)
        aggregate = result["run"].get("aggregate") or {}
        scores = aggregate.get("scores") or []
        if (aggregate.get("status") == "complete" and aggregate.get("coverage") == 10
                and aggregate.get("scope") == "method_development_benchmark"
                and len(scores) == 10):
            counts = [score["metrics"]["query_count"] for score in scores]
            result["benchmark_population"] = {
                "query_count_total": sum(counts), "query_counts_by_fold": counts,
                "source": "/run/aggregate/scores/*/metrics/query_count",
                "basis": "Sum of native held-out benchmark query counts; not source-subject development validation.",
            }
    return result


def _immutable_input(camp: Path, payload: dict) -> tuple[Path, dict]:
    from react_agent.eeg_research.agentic.artifacts import lookup, register, resolve_verified_artifact
    text = canonical(payload)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    folder = camp / "compaction"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("input_" + sha + ".json")
    artifact_id = "art_compaction_input_" + sha[:24]
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("compaction_input_snapshot_changed")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    artifact = lookup(camp, artifact_id)
    if artifact is None:
        artifact = register(camp, path, kind="handoff_compaction_input", artifact_id=artifact_id)
    artifact = resolve_verified_artifact(camp, artifact_id)
    if artifact["sha256"] != sha or Path(artifact["path"]).resolve() != path.resolve():
        raise ValueError("compaction_input_snapshot_changed")
    return path, artifact


def _recover_stage(camp: Path, request: dict, *, original: dict, refs: list[str],
                   inputs: list[Path], system: str, model: str, identity: dict) -> tuple[dict, dict] | None:
    """Adopt a durable validated reply after a crash, without another API call."""
    from uuid import uuid4
    from react_agent.eeg_research.agentic.artifacts import (
        input_dependency_manifest, verify_input_dependencies, request_digest,
        resolve_verified_artifact,
    )
    from react_agent.eeg_research.agentic.roles import validate_role_result, finish_role_task
    from react_agent.eeg_research.agentic.task_ledger import (
        task_snapshots, recovery_identity_digest, mark,
    )
    dependencies = input_dependency_manifest(camp, paths=inputs)
    if not verify_input_dependencies(camp, dependencies):
        return None
    wanted = recovery_identity_digest({**identity, "input_dependencies": dependencies})
    wanted_input = request_digest(request=request, paths=inputs)
    calls_path = camp / "llm_calls.jsonl"
    if not calls_path.is_file():
        return None
    calls = [json.loads(line) for line in calls_path.read_text().splitlines() if line.strip()]
    intents_path = camp / "llm_api_intents.jsonl"
    intents = [json.loads(line) for line in intents_path.read_text().splitlines() if line.strip()] \
        if intents_path.is_file() else []
    settled = {row.get("call_id") for row in calls}
    registry_path = camp / "artifact_registry.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text().splitlines() if line.strip()] \
        if registry_path.is_file() else []
    for task in reversed(task_snapshots(camp)):
        if (task.get("status") not in {"pending", "running"} or task.get("role") != ROLE
                or task.get("recovery_identity_digest") != wanted
                or task.get("input_digest") != wanted_input
                or recovery_identity_digest(task["recovery_identity"]) != wanted):
            continue
        if any(row.get("task_id") == task["task_id"] and row.get("call_id") not in settled
               for row in intents):
            continue
        call = next((row for row in reversed(calls) if row.get("task_id") == task["task_id"]), None)
        if (not call or call.get("success") is not True or call.get("operation") != OPERATION
                or call.get("requested_model") != model or call.get("prompt_hash") != identity["prompt_hash"]):
            continue
        try:
            traced = json.loads((camp / "llm_requests" / (call["call_id"] + ".json")).read_text())
            response = json.loads((camp / "llm_responses" / (call["call_id"] + ".json")).read_text())
            parsed = json.loads(traced["user"])
            if any(parsed.get(key) != task[key] or traced.get(key) != task[key]
                   or response.get(key) != task[key]
                   for key in ("task_id", "attempt_id", "input_digest")):
                continue
            base = {key: value for key, value in parsed.items() if key not in
                    {"task_id", "attempt_id", "input_digest", "summary_validation_error"}}
            if (base != request or traced["system"] != system or response.get("validation") != "valid"
                    or traced.get("operation") != OPERATION or response.get("operation") != OPERATION
                    or hashlib.sha256(traced["system"].encode()).hexdigest() != call.get("request_system_sha256")
                    or hashlib.sha256(traced["user"].encode()).hexdigest() != call.get("request_user_sha256")
                    or traced.get("system_sha256") != call.get("request_system_sha256")
                    or traced.get("user_sha256") != call.get("request_user_sha256")):
                continue
            summary = validate_summary(response["reply"], original, refs)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        # Crash after artifact registration but before the completed ledger row:
        # preserve its original bytes/id instead of rewriting that receipt.
        for row in reversed(registry):
            if row.get("producer_task_id") != task["task_id"] or row.get("kind") != "handoff_compaction":
                continue
            try:
                artifact = resolve_verified_artifact(camp, row["artifact_id"])
                stored = validate_role_result(json.loads(Path(artifact["path"]).read_text()), task)
                body = stored["payload"]
                if (stored["status"] != "completed" or stored["prompt_hash"] != identity["prompt_hash"]
                        or stored["artifact_refs"][-1] != row["artifact_id"]
                        or body.get("compaction_summary") != summary
                        or body.get("original_input_sha256") != request["original_input_sha256"]
                        or body.get("stage") != request["stage"] or body.get("operation") != OPERATION):
                    continue
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                continue
            mark(camp, task["task_id"], "completed", role=ROLE, attempt_id=task["attempt_id"],
                 artifact_id=artifact["artifact_id"], path=artifact["path"],
                 recovered_transport_call_id=call["call_id"])
            return summary, artifact
        # A valid transport result without a registered receipt gets a separate
        # recovery artifact; any partial original output remains unchanged.
        envelope = finish_role_task(camp, task,
            {"status": "completed", "prompt_hash": identity["prompt_hash"],
             "operation": OPERATION, "revision": REVISION,
             "original_input_sha256": request["original_input_sha256"],
             "stage": request["stage"], "compaction_summary": summary,
             "recovered_transport_call_id": call["call_id"]},
            kind="handoff_compaction",
            path=camp / "compaction" / (task["task_id"] + "_recovered_" + uuid4().hex[:12] + ".json"))
        return summary, resolve_verified_artifact(camp, envelope["artifact_refs"][-1])
    return None


def _stage(camp: Path, request: dict, *, original: dict, refs: list[str],
           inputs: list[Path], system: str, model: str,
           invoke: Callable[..., dict]) -> tuple[dict, dict]:
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    identity = {
        "role": ROLE, "action": OPERATION + ":" + request["stage"], "target_id": None,
        "source": request["original_input_sha256"],
        "config": {"revision": REVISION, "model": model, "summary_chars": SUMMARY_CHARS},
        "approval": None, "prompt_hash": hashlib.sha256(system.encode()).hexdigest()[:16],
        "schema_hash": digest(CompactionSummary.model_json_schema()), "evaluator": None,
        "evidence": refs, "memory_snapshot": None, "legal_actions": [],
        "budget_feasibility": {"bounded_delivery_operation": True},
        "request_semantics": digest(request), "feedback": None,
    }
    recovered = _recover_stage(camp, request, original=original, refs=refs,
        inputs=inputs, system=system, model=model, identity=identity)
    if recovered:
        return recovered
    task = begin_role_task(camp, role=ROLE, inputs=inputs, request=request,
                           recovery_identity=identity)
    if task.get("_cached_role_result"):
        cached = task["_cached_role_result"]
        summary = validate_summary(cached["envelope"]["payload"]["compaction_summary"], original, refs)
        envelope = finish_role_task(camp, task, {}, kind="handoff_compaction",
                                     path=Path(cached["artifact"]["path"]))
        return summary, resolve_verified_artifact(camp, envelope["artifact_refs"][-1])
    try:
        summary = invoke(
            system=system, payload={**request, **{key: task[key] for key in
                ("task_id", "attempt_id", "input_digest")}}, task=task,
            validate=lambda reply: validate_summary(reply, original, refs),
            context_budget_chars=INPUT_CHARS,
        )
        envelope = finish_role_task(camp, task,
            {"status": "completed", "prompt_hash": identity["prompt_hash"],
             "operation": OPERATION, "revision": REVISION,
             "original_input_sha256": request["original_input_sha256"],
             "stage": request["stage"], "compaction_summary": summary},
            kind="handoff_compaction", path=camp / "compaction" / (task["task_id"] + ".json"))
    except Exception as exc:
        finish_role_task(camp, task,
            {"status": "failed", "prompt_hash": identity["prompt_hash"],
             "operation": OPERATION, "revision": REVISION,
             "original_input_sha256": request["original_input_sha256"],
             "summary_zh": "交接压缩失败：" + str(exc)},
            kind="handoff_compaction", path=camp / "compaction" / (task["task_id"] + "_failed.json"))
        if isinstance(exc, LlmUnavailable) and str(exc) in {
                "unsettled_api_intent_requires_recovery", "budget_exhausted"}:
            raise
        raise LlmUnavailable("handoff_compaction_failed",
                             diagnostics={"reason": str(exc), "task_id": task["task_id"]}) from exc
    return summary, resolve_verified_artifact(camp, envelope["artifact_refs"][-1])


def compact_for_curator(camp: Path, payload: dict, *, inputs: list[Path], model: str,
                        invoke: Callable[..., dict]) -> dict:
    """Create/reuse model-written summaries while preserving native authority."""
    from react_agent.eeg_research.agentic.semantic_memory import unsafe_source
    from react_agent.eeg_research.agentic.llm import LlmUnavailable
    if unsafe_source(payload):
        raise LlmUnavailable("handoff_compaction_forbidden_input")
    if "context_compaction" in payload:
        raise LlmUnavailable("handoff_compaction_already_compacted")
    system = compaction_prompt()
    try:
        text, parts = split_input(payload)
    except ValueError as exc:
        raise LlmUnavailable("handoff_compaction_failed", diagnostics={"reason": str(exc)}) from exc
    original_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    snapshot, input_artifact = _immutable_input(camp, payload)
    dependencies = list(dict.fromkeys([*inputs, snapshot]))
    summaries = []
    artifacts = []
    for part in parts:
        request = {"operation": OPERATION, "revision": REVISION, "stage": "map",
                   "original_input_sha256": original_sha, "total_parts": len(parts),
                   "part": part, "expected_source_refs": [part["ref"]]}
        summary, artifact = _stage(camp, request, original=payload, refs=[part["ref"]],
            inputs=dependencies, system=system, model=model, invoke=invoke)
        summaries.append(summary)
        artifacts.append(artifact)
    refs = [part["ref"] for part in parts]
    if len(summaries) == 1:
        summary, artifact = summaries[0], artifacts[0]
    else:
        request = {"operation": OPERATION, "revision": REVISION, "stage": "reduce",
                   "original_input_sha256": original_sha, "summaries": summaries,
                   "expected_source_refs": refs}
        summary, artifact = _stage(camp, request, original=payload, refs=refs,
            inputs=[*dependencies, *[Path(row["path"]) for row in artifacts]],
            system=system, model=model, invoke=invoke)
    result = protected_context(payload)
    result["compacted_context"] = summary
    result["context_compaction"] = {
        "revision": REVISION, "operation": OPERATION,
        "artifact_id": artifact["artifact_id"], "sha256": artifact["sha256"],
        "input_artifact_id": input_artifact["artifact_id"],
        "original_input_sha256": original_sha, "source_refs": refs,
        "source_ranges": [{key: part[key] for key in ("ref", "start", "end", "sha256")} for part in parts],
        "scope": "Derived model-written delivery summary; not scientific evidence or a memory lesson.",
        "limitations": "Structured numbers/references checked; natural-language fidelity requires source review.",
    }
    return result
