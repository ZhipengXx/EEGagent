"""Candidate-local engineering diagnostics and persistent recovery limits."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CodingRecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repeated_error_limit: int = Field(default=2, ge=2)
    max_model_escalations_per_candidate: int = Field(default=1, ge=0, le=1)
    escalation_model: str = "deepseek-v4-pro"
    escalation_thinking: bool = True
    escalation_reasoning_effort: str = "high"
    max_tool_steps: int | None = Field(default=None, ge=1)
    max_failed_checks: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_effort(self):
        if self.escalation_reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("unsupported coding escalation reasoning effort")
        if not self.escalation_thinking:
            raise ValueError("coding escalation requires thinking")
        if not self.escalation_model.strip():
            raise ValueError("coding escalation model missing")
        return self


def policy(workspace: Path) -> CodingRecoveryConfig | None:
    path = workspace.parent.parent / "goal.json"
    goal = json.loads(path.read_text()) if path.is_file() else {}
    config = goal.get("coding_recovery")
    return CodingRecoveryConfig.model_validate(config) if isinstance(config, dict) else None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _normalize(text: str) -> str:
    # Keep dimension/dtype numbers. Remove only process/path/run noise.
    text = re.sub(r"0x[0-9a-fA-F]+", "<address>", text)
    text = re.sub(r"\b[0-9a-f]{12,64}\b", "<id>", text)
    text = re.sub(r"(?:/[^\s\"']+)+/([^/\s\"']+\.py)", r"\1", text)
    text = re.sub(r"line \d+", "line <location>", text)
    text = re.sub(r"\b(?:pid|run_id|timestamp)[=: ]+\S+", "<runtime>", text)
    return " ".join(text.split())


def classify_check(check: dict) -> str:
    error = str(check.get("error") or "")
    text = (error + " " + str(check.get("detail") or "") + " " +
            json.dumps(check.get("failures") or [])).lower()
    if check.get("ok"):
        return "passed"
    if check.get("cpu_check_ok") or error == "gpu_check_pending":
        return "resource_pending"
    if any(word in text for word in ("timeout", "torch_python_missing", "environment", "gpu_unavailable", "uncertain")):
        return "environment_error"
    if any(word in text for word in ("invalid_tool", "invalid_json", "schema_error")):
        return "format_error"
    if any(word in text for word in ("syntaxerror", "indentationerror")):
        return "syntax_error"
    if any(word in text for word in ("importerror", "modulenotfounderror")):
        return "import_error"
    if any(word in text for word in ("gradient", "backward", "non_finite", "nan", "infinity")):
        return "gradient_or_numeric_error"
    if any(word in text for word in ("shape", "mat1", "mat2", "size mismatch", "channels", "dtype", "device")):
        return "shape_dtype_device_error"
    if any(word in text for word in ("signature", "unexpected keyword", "required positional", "attributeerror")):
        return "interface_error"
    return "candidate_code_error"


def error_fingerprint(check: dict) -> str | None:
    category = classify_check(check)
    if category in {"passed", "resource_pending", "environment_error", "format_error"}:
        return None
    location = check.get("location") or {}
    def normalized(value):
        if isinstance(value, str):
            return _normalize(value)
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items() if key not in
                    {"timestamp", "recorded_at", "elapsed_seconds", "run_id", "pid", "source_sha256", "check_fingerprint"}}
        return value
    material = {"category": category, "stage": check.get("stage"),
                "hook": check.get("hook") or location.get("function"),
                "file": Path(str(location.get("file") or "")).name,
                "error": check.get("error"),
                "failures": normalized(check.get("failures") or []),
                "expected": normalized(check.get("expected")), "actual": normalized(check.get("actual")),
                "detail": _normalize(str(check.get("detail") or ""))}
    return _digest(material)


def reconcile_escalation(workspace: Path, state: dict, backend: Any) -> None:
    """Finish an upgrade intent once, including either side of attempt rotation."""
    from react_agent.eeg_research.agentic.identity import ensure_attempt, rotate_attempt
    binding = state.get("operation_binding") or dict(getattr(backend, "identity", {}) or {})
    attempt = ensure_attempt(workspace, workspace.name)
    if state.get("escalations"):
        upgrade = state["escalations"][-1]
        if not upgrade.get("attempt_id"):
            if attempt["attempt_id"] == upgrade["parent_attempt_id"]:
                attempt = rotate_attempt(workspace, workspace.name)
            elif attempt.get("supersedes_attempt_id") != upgrade["parent_attempt_id"]:
                raise ValueError("coding_escalation_attempt_lineage_mismatch")
            upgrade["attempt_id"] = attempt["attempt_id"]
            save(workspace, state)
        configure = getattr(backend, "configure", None)
        if callable(configure):
            configure(upgrade["overrides"])
    bind = getattr(backend, "bind", None)
    if callable(bind):
        bind(**{**binding, "candidate_id": workspace.name, "attempt_id": attempt["attempt_id"]})


def load(workspace: Path) -> dict:
    path = workspace / "coding_recovery.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def save(workspace: Path, state: dict) -> None:
    target = workspace / "coding_recovery.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(target)


def initialize(workspace: Path, config: CodingRecoveryConfig, *, max_steps: int,
               max_repairs: int, restored: dict, operation_binding: dict | None = None,
               resume_repair: bool = False) -> dict:
    state = load(workspace)
    binding = dict(operation_binding or {})
    if state:
        if state.get("candidate_id") != workspace.name:
            raise ValueError("coding_recovery_candidate_identity_changed")
        state["steps_used"] = max(state["steps_used"], int(restored.get("next_step", 1)) - 1)
        previous = state.get("operation_binding") or {}
        migrating = not state.get("tool_sessions")
        if migrating:
            state["tool_sessions"] = [{"kind": "initial", "first_step": 1,
                "last_step": state["tool_step_limit"], "binding": dict(previous)}]
        new_repair = (resume_repair and binding.get("operation_id") and binding.get("attempt_id")
            and binding.get("operation_id") != previous.get("operation_id")
            and binding.get("attempt_id") != previous.get("attempt_id")
            and not _is_escalation_attempt(workspace, state, binding["attempt_id"]))
        if new_repair:
            first_step = int(restored.get("next_step", 1))
            # Old recovery files have no session boundary. A completed patch
            # identifies the Reviewer repair's start, including tools already
            # executed before this bookkeeping fix was installed.
            if migrating and restored.get("last_finish_step"):
                first_step = int(restored["last_finish_step"]) + 1
            _start_tool_session(state, config, binding, "review_repair", first_step, max_steps)
        elif not previous and binding:
            state["operation_binding"] = binding
            state["tool_sessions"][-1]["binding"] = dict(binding)
        if config.max_tool_steps is not None:
            state["tool_step_limit"] = config.max_tool_steps
    else:
        state = {"schema_version": "eeg_research.coding_recovery.v1",
                 "candidate_id": workspace.name,
                 "tool_step_limit": config.max_tool_steps or max_steps,
                 "failed_check_limit": config.max_failed_checks or max_repairs + 1,
                 "steps_used": int(restored.get("next_step", 1)) - 1,
                 "failed_checks_used": 0,
                 "model_escalations_used": 0, "consecutive_error_count": 0,
                 "latest_error_fingerprint": None, "checks": [], "escalations": [], "tool_sessions": []}
        first_step = (int(restored.get("last_finish_step") or 0) + 1
                      if resume_repair and binding.get("operation_id") and binding.get("attempt_id") else 1)
        _start_tool_session(state, config, binding, "review_repair" if resume_repair else "initial",
                            first_step, max_steps)
    # A logged check remains consumed after migration; it must not look new.
    for row in _logged_checks(workspace):
        observe_check(state, row)
    save(workspace, state)
    return state


def _start_tool_session(state: dict, config: CodingRecoveryConfig, binding: dict,
                        kind: str, first_step: int, max_steps: int) -> None:
    # An explicit cap remains candidate-wide. Otherwise max_steps is the native
    # implementation/Reviewer-repair window already reserved by the plan owner.
    last_step = config.max_tool_steps if config.max_tool_steps is not None else first_step + max_steps - 1
    state["tool_step_limit"] = last_step
    state["operation_binding"] = dict(binding)
    state["tool_sessions"].append({"kind": kind, "first_step": first_step,
        "last_step": last_step, "binding": dict(binding)})


def _is_escalation_attempt(workspace: Path, state: dict, attempt_id: str) -> bool:
    for upgrade in state.get("escalations") or []:
        if upgrade.get("attempt_id") == attempt_id:
            return True
        if not upgrade.get("attempt_id"):
            path = workspace / "attempt.json"
            current = json.loads(path.read_text()) if path.is_file() else {}
            if (current.get("attempt_id") == attempt_id
                    and current.get("supersedes_attempt_id") == upgrade.get("parent_attempt_id")):
                return True
    return False


def _logged_checks(workspace: Path):
    path = workspace / "coder_log.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            result = row.get("result") or {}
            check = result.get("auto_check") or (result if row.get("tool") == "run_candidate_check" else None)
            if isinstance(check, dict):
                yield check


def observe_check(state: dict, check: dict) -> dict:
    identity = _digest({"fingerprint": check.get("check_fingerprint"), "source": check.get("source_sha256"),
                        "ok": check.get("ok"), "error": check.get("error"), "stage": check.get("stage")})
    if check.get("cached") or any(row["check_identity"] == identity for row in state["checks"]):
        return {"progress": False, "reason": "same_check_reused"}
    fingerprint = error_fingerprint(check)
    previous = state.get("latest_error_fingerprint")
    category = classify_check(check)
    if fingerprint:
        repeated = fingerprint == previous
        state["consecutive_error_count"] = state["consecutive_error_count"] + 1 if repeated else 1
        state["latest_error_fingerprint"] = fingerprint
        state["failed_checks_used"] += 1
        reason = "same_semantic_error" if repeated else "different_engineering_error"
    elif category == "passed":
        state["consecutive_error_count"] = 0
        state["latest_error_fingerprint"] = None
        reason = "previous_error_resolved"
    else:
        # API/resources/format failures cannot reset or escalate coding errors.
        reason = "non_coding_failure"
    observation = {"check_identity": identity, "source_hash": check.get("source_sha256"),
                   "error_fingerprint": fingerprint, "category": category,
                   "stage": check.get("stage"), "progress": reason in {"different_engineering_error", "previous_error_resolved"},
                   "reason": reason}
    state["checks"].append(observation)
    return observation


def repair_context(workspace: Path, spec: dict, check: dict, state: dict) -> dict:
    entry = workspace / "extension/eeg_candidate.py"
    location = check.get("location") or {}
    lines = entry.read_text().splitlines() if entry.is_file() else []
    line = location.get("line") if isinstance(location.get("line"), int) else 1
    start, end = max(1, line - 5), min(len(lines), line + 10)
    experiment = spec.get("experiment") or spec
    return {"candidate_id": workspace.name,
            "approved_spec_hash": experiment.get("spec_hash"),
            "source_hash": check.get("source_sha256"), "check_fingerprint": check.get("check_fingerprint"),
            "error_fingerprint": error_fingerprint(check), "error_category": classify_check(check),
            "stage": check.get("stage"), "hook": check.get("hook") or location.get("function"),
            "location": location, "expected": check.get("expected"), "actual": check.get("actual") or check.get("shape"),
            "traceback": check.get("detail"), "source_excerpt": {"start": start, "end": end, "text": "\n".join(lines[start-1:end])},
            "open_review_issues": spec.get("repair_issues") or [],
            "intervention": experiment.get("principal_intervention") or experiment.get("intervention"),
            "steps_left": max(0, state["tool_step_limit"] - state["steps_used"]),
            "failed_checks_left": max(0, state["failed_check_limit"] - state["failed_checks_used"]),
            "instruction": "Patch the smallest relevant range, preserving the approved intervention and unresolved review issues."}


def maybe_escalate(workspace: Path, spec: dict, check: dict, state: dict,
                   config: CodingRecoveryConfig, backend: Any, *, calls_left=None) -> str | None:
    if state["consecutive_error_count"] < config.repeated_error_limit:
        return None
    if (state["steps_used"] >= state["tool_step_limit"] or
            state["failed_checks_used"] >= state["failed_check_limit"]):
        return "coding_lineage_budget_exhausted"
    if state["model_escalations_used"] >= config.max_model_escalations_per_candidate:
        return "coding_no_progress_exhausted"
    current_route = getattr(backend, "role_config", None)
    if (getattr(current_route, "model", None) == config.escalation_model and
            getattr(current_route, "thinking", None) == config.escalation_thinking and
            getattr(current_route, "reasoning_effort", None) == config.escalation_reasoning_effort):
        return "coding_no_progress_exhausted"
    experiment = spec.get("experiment") or spec
    complete = ((workspace / "extension/eeg_candidate.py").is_file() and
                check.get("source_sha256") and check.get("check_fingerprint") and
                experiment.get("spec_hash") and (experiment.get("approval_record") or {}).get("status") == "approved")
    if not complete:
        return "coding_repair_inputs_missing"
    configure = getattr(backend, "configure", None)
    if not callable(configure):
        return "coding_escalation_unavailable"
    if calls_left is not None and calls_left() <= 2:
        return "budget_exhausted"
    from react_agent.eeg_research.agentic.identity import ensure_attempt, rotate_attempt
    parent_attempt = ensure_attempt(workspace, workspace.name)
    overrides = {"model": config.escalation_model, "thinking": config.escalation_thinking,
                 "reasoning_effort": config.escalation_reasoning_effort}
    # Persist the sole upgrade intent before changing model or issuing a request.
    state["model_escalations_used"] += 1
    state["consecutive_error_count"] = 0
    state["escalations"].append({"overrides": overrides, "error_fingerprint": error_fingerprint(check),
                                "source_hash": check["source_sha256"], "spec_hash": experiment["spec_hash"],
                                "parent_attempt_id": parent_attempt["attempt_id"],
                                "steps_used": state["steps_used"], "failed_checks_used": state["failed_checks_used"]})
    save(workspace, state)
    reconcile_escalation(workspace, state, backend)
    return "escalated"
