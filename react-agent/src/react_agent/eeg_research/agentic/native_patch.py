"""NativePatchCodeAgent: read, patch, check, repair. The server executes every tool."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.coder import (
    apply_candidate_patch,
    finish_patch,
    list_project_files,
    read_code,
    run_candidate_check,
    search_code,
)

Backend = Callable[[dict[str, Any]], dict[str, Any]]
_READ_TOOLS = {"list_project_files", "search_code", "read_code", "inspect_check_result"}


def implement(
    workspace: Path,
    spec: dict[str, Any],
    backend: Backend,
    *,
    max_steps: int = 12,
    max_repairs: int = 2,
    python: str | None = None,
    calls_left: Callable[[], int] | None = None,
    reserve: int = 2,
) -> dict[str, Any]:
    """Run the coder loop. finish_patch is refused until a check has passed."""
    (workspace / "extension").mkdir(parents=True, exist_ok=True)
    log_path = workspace / "coder_log.jsonl"
    history: list[dict[str, Any]] = []
    last_check: dict[str, Any] | None = None
    last_hash = ""
    failed_checks = 0
    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_research.agentic.coder import _REFERENCES

    from react_agent.eeg_research.agentic.interface import CANDIDATE_INTERFACE
    from react_agent.eeg_training.protocol import geometry

    shape = geometry("eeg")
    input_spec = {
        "c_num": int(shape["c_num"]),
        "timesteps": list(shape["timesteps"]),
        **CANDIDATE_INTERFACE,
    }
    references = {name: path.read_text(encoding="utf-8")[:6000] for name, path in _REFERENCES.items()}
    entry = workspace / "extension" / "eeg_candidate.py"
    writes = 0
    reads = 0
    extension_answered = False
    for step in range(1, max_steps + 1):
        if calls_left is not None and calls_left() <= reserve:
            return {"status": "implementation_failed", "detail": "budget_exhausted", "check": last_check, "steps": step - 1}
        current = None
        if entry.is_file():
            current = {"path": "extension/eeg_candidate.py", "sha256": file_sha256(entry), "text": entry.read_text(encoding="utf-8")[:6000]}
        request = {
            "experiment_spec": spec,
            "input_spec": input_spec,
            "allowed_write_paths": ["extension/eeg_candidate.py"],
            "step": step,
            "max_steps": max_steps,
            "repairs_used": max(0, failed_checks - 1) if failed_checks else 0,
            "max_repairs": max_repairs,
            "current_file": current,
            "references": references if step == 1 else "same as step 1",
            "last_check": last_check,
            "history": history[-8:],
        }
        if step >= 3 and writes == 0:
            request["runtime_note"] = "References and the current file are already supplied. Call apply_candidate_patch now."
        elif last_check is not None and not last_check.get("ok"):
            request["runtime_note"] = "The last check failed. Repair with apply_candidate_patch, then run_candidate_check."
        elif last_check is not None and last_check.get("ok"):
            request["runtime_note"] = "The check passed. Call finish_patch unless a required change is missing."
        reply = backend(request)
        tool = str(reply.get("tool") or "")
        args = reply.get("args") if isinstance(reply.get("args"), dict) else {}
        started = time.time()
        if tool in _READ_TOOLS and writes == 0 and reads >= 2:
            result = {"ok": False, "error": "write_required", "detail": "current_file and references are already in the request; call apply_candidate_patch."}
        elif tool == "requires_framework_extension" and not extension_answered:
            extension_answered = True
            result = {
                "ok": False,
                "error": "interface_already_available",
                "detail": (
                    "build_encoder may return any torch.nn.Module defined in extension/eeg_candidate.py; "
                    "it does not need to use or subclass EEGProjectLayer. Gallery, image cache, labels and split "
                    "are runtime guarantees and must not be asserted by the candidate. Write the module now, or "
                    "call requires_framework_extension again only if something outside candidate_interface is required."
                ),
                "candidate_interface": input_spec,
            }
            tool = "requires_framework_extension_answered"
        else:
            result = _execute(workspace, tool, args, last_check, python)
        if tool in _READ_TOOLS:
            reads += 1
        if tool == "apply_candidate_patch" and result.get("ok"):
            last_hash = str(result.get("sha256"))
            writes += 1
            check = run_candidate_check(workspace, python=python)
            result = {**result, "auto_check": check}
            tool_for_check = check
        else:
            tool_for_check = None
        if tool_for_check is not None:
            last_check = tool_for_check
            (workspace / "checks.json").write_text(json.dumps(tool_for_check, ensure_ascii=False, indent=2), encoding="utf-8")
            if not tool_for_check.get("ok"):
                failed_checks += 1
        if tool == "run_candidate_check":
            last_check = result
            (workspace / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if not result.get("ok"):
                failed_checks += 1
        if tool == "finish_patch" and not (last_check and last_check.get("ok")):
            result = {"ok": False, "error": "check_not_passed"}
        row = {
            "call_id": uuid.uuid4().hex[:12],
            "step": step,
            "tool": tool,
            "args": _short(args),
            "result": _short(result),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.append({"tool": tool, "result": _short(result)})
        if tool == "requires_framework_extension":
            return {"status": "requires_framework_extension", "detail": args, "steps": step}
        if tool == "finish_patch" and result.get("ok"):
            return {"status": "ready_for_review", "manifest": result["manifest"], "check": last_check, "steps": step}
        if failed_checks > max_repairs + 1:
            return {"status": "implementation_failed", "detail": "repair_limit", "check": last_check, "steps": step}
    return {"status": "implementation_failed", "detail": "step_limit", "check": last_check, "steps": max_steps}


def _execute(workspace: Path, tool: str, args: dict[str, Any], last_check: Any, python: str | None) -> dict[str, Any]:
    if tool == "list_project_files":
        return list_project_files(workspace)
    if tool == "search_code":
        return search_code(workspace, str(args.get("query") or ""))
    if tool == "read_code":
        return read_code(workspace, str(args.get("path") or ""), int(args.get("start") or 1), int(args.get("end") or 200))
    if tool == "apply_candidate_patch":
        return apply_candidate_patch(
            workspace,
            str(args.get("path") or ""),
            str(args.get("content") or ""),
            str(args.get("expected_base_hash") or ""),
        )
    if tool == "run_candidate_check":
        return run_candidate_check(workspace, python=python)
    if tool == "inspect_check_result":
        return {"ok": last_check is not None, "result": last_check}
    if tool == "finish_patch":
        return finish_patch(workspace, str(args.get("summary") or ""))
    if tool == "requires_framework_extension":
        return {"ok": True}
    return {"ok": False, "error": f"unknown_tool:{tool}"}


def _short(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= 3000:
        return value
    return text[:3000]


def _review_valid(reply: Any) -> bool:
    return (
        isinstance(reply, dict)
        and reply.get("status") in {"ready", "needs_fix", "blocked"}
        and isinstance(reply.get("summary_zh"), str)
        and isinstance(reply.get("issues", []), list)
    )


_RUNTIME_BLOCK_MARKERS = (
    "训练预算",
    "训练时长",
    "训练长度",
    "training length",
    "training budget",
    "epoch",
    "gallery",
    "图库",
    "特征缓存",
    "feature cache",
    "划分",
    "split manifest",
    "接口断言",
    "运行时保证",
    "runtime guarantee",
    "runtime_guarantees",
    "自己断言",
)


def _issue_text(issue: dict[str, Any]) -> str:
    parts = (issue.get("title"), issue.get("evidence"), issue.get("impact"), issue.get("repair"))
    return " ".join(str(part or "") for part in parts).lower()


def filter_runtime_blocking(issues: list[Any]) -> list[dict[str, Any]]:
    """Training length, gallery, cache, split and runtime assertions are not candidate defects."""
    filtered: list[dict[str, Any]] = []
    for issue in issues:
        row = dict(issue) if isinstance(issue, dict) else {"title": str(issue), "severity": "blocking"}
        text = _issue_text(row)
        if row.get("severity") == "blocking" and any(marker in text for marker in _RUNTIME_BLOCK_MARKERS):
            row["severity"] = "non_blocking"
            row["downgraded_from"] = "blocking"
            row["downgrade_reason"] = "runtime_owned"
        filtered.append(row)
    return filtered


def apply_review_filter(result: dict[str, Any]) -> dict[str, Any]:
    """Recompute review status after runtime-owned blocking issues are downgraded."""
    issues = filter_runtime_blocking(list(result.get("issues") or []))
    blocking = [row for row in issues if row.get("severity") == "blocking"]
    updated = dict(result)
    updated["issues"] = issues
    updated["model_status"] = result.get("model_status") or result.get("status")
    updated["runtime_issues_downgraded"] = sum(1 for row in issues if row.get("downgrade_reason") == "runtime_owned")
    if result.get("format_failed"):
        updated["status"] = "blocked"
    elif not blocking:
        updated["status"] = "ready"
    updated["blocking_remaining"] = len(blocking)
    return updated


def review(workspace: Path, spec: dict[str, Any], contract_summary: dict[str, Any], backend: Backend, model: str) -> dict[str, Any]:
    """Reviewer sees spec, code and checks. It returns blocking issues, not a score."""
    entry = workspace / "extension" / "eeg_candidate.py"
    from react_agent.eeg_research.agentic.interface import CANDIDATE_INTERFACE

    checks = json.loads((workspace / "checks.json").read_text(encoding="utf-8")) if (workspace / "checks.json").is_file() else None
    payload = {
        "experiment_spec": spec,
        "candidate_source": entry.read_text(encoding="utf-8") if entry.is_file() else None,
        "checks": checks,
        "evaluation_contract": contract_summary,
        "candidate_interface": CANDIDATE_INTERFACE,
        "runtime_guarantees": CANDIDATE_INTERFACE["runtime_guarantees"],
        "reviewer_model": model,
        "executor_model": model,
    }
    reply = backend(payload)
    repaired = False
    if not _review_valid(reply):
        repaired = True
        reply = backend(
            {
                **payload,
                "schema_error": "Return status (ready/needs_fix/blocked), issues (list), review_limits and summary_zh.",
                "previous": reply,
            }
        )
    format_failed = not _review_valid(reply)
    result = apply_review_filter(
        {
            "status": "blocked" if format_failed else reply["status"],
            "model_status": "blocked" if format_failed else reply["status"],
            "issues": [] if format_failed else reply.get("issues") or [],
            "review_limits": None if format_failed else reply.get("review_limits"),
            "summary_zh": "审查回复格式不合格，按受阻处理。" if format_failed else reply.get("summary_zh"),
            "format_failed": format_failed,
            "schema_repaired": repaired,
            "calls": 2 if repaired else 1,
            "reviewer_model": model,
            "same_family_as_executor": True,
        }
    )
    (workspace / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
