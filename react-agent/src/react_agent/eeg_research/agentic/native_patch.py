"""NativePatchCodeAgent: read, patch, check, repair. The server executes every tool."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from react_agent.eeg_research.agentic.coder import (
    apply_candidate_patch,
    edit_candidate_code,
    finish_patch,
    list_project_files,
    read_code,
    run_candidate_check,
    search_code,
    source_page,
    tool_catalog,
    validate_tool_request,
)

Backend = Callable[[dict[str, Any]], dict[str, Any]]
_READ_TOOLS = {"list_project_files", "search_code", "read_code", "inspect_check_result"}
_WRITE_TOOLS = {"apply_candidate_patch", "edit_candidate_code"}


def _is_check_receipt(result: dict[str, Any]) -> bool:
    """Exclude request validation refusals, while retaining real check errors/cache."""
    return result.get("error") not in {"invalid_tool_request", "invalid_tool_args"}


def _read_key(tool: str, args: dict[str, Any], revision: str) -> tuple[Any, ...]:
    return (tool, str(args.get("path") or ""), str(args.get("query") or ""),
            int(args.get("start") or 1), int(args.get("end") or 200), revision)


def _check_fingerprint(workspace: Path, python: str | None) -> str:
    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_training.protocol import torch_python

    spec = workspace / "input_spec.json"
    approved = workspace / "spec.json"
    assigned = json.loads(approved.read_text(encoding="utf-8")) if approved.is_file() else {}
    config = assigned.get("experiment") or assigned
    # check_entry consumes only these hook configs. Repair prose and artifact
    # receipts do not change the computation checked by this fingerprint.
    hook_config = {key: config.get(key) or {} for key in ("model", "objective", "transform")}
    payload = {
        "fingerprint_version": 3,
        "source": file_sha256(workspace / "extension" / "eeg_candidate.py"),
        "input_spec": json.loads(spec.read_text(encoding="utf-8")) if spec.is_file() else "eeg",
        "approved_hook_config": hook_config,
        "python": python or torch_python(),
        "checker": file_sha256(Path(__file__).with_name("check_entry.py")),
        "objective_effectiveness_probe": file_sha256(Path(__file__).with_name("objective_effectiveness.py")),
        "objective_semantic_probe": file_sha256(Path(__file__).with_name("objective_semantics.py")),
        "baseline_loss_reference": file_sha256(Path(__file__).resolve().parents[2] / "eeg_training" / "model.py"),
        "objective_dispatch_reference": file_sha256(Path(__file__).resolve().parents[2] / "eeg_training" / "hooks.py"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class RecoveryBlocked(RuntimeError):
    """Disk and the coder log disagree. The caller must not rerun tools."""


def _load_coder_rows(log_path: Path) -> list[dict[str, Any]]:
    """Return complete rows. A partial tail is not a safe resume point."""
    if not log_path.is_file():
        return []
    raw = log_path.read_text(encoding="utf-8")
    if raw == "":
        return []
    if not raw.endswith("\n"):
        raise RecoveryBlocked("coder_log_tail_incomplete")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryBlocked("coder_log_tail_incomplete") from exc
        if not isinstance(payload, dict):
            raise RecoveryBlocked("coder_log_tail_incomplete")
        rows.append(payload)
    return rows


def _restore_coder(workspace: Path, *, ignore_finish: bool = False) -> dict[str, Any]:
    """Replay complete log rows without executing their tools again."""
    from react_agent.eeg_research.agentic.binding import file_sha256

    rows = _load_coder_rows(workspace / "coder_log.jsonl")
    history: list[dict[str, Any]] = []
    last_check: dict[str, Any] | None = None
    last_patch_sha = ""
    failed_checks = 0
    failed_revisions: set[str] = set()
    writes = 0
    reads = 0
    finished = False
    last_step = 0
    check_fingerprint = ""
    extension_answered = False
    seen_reads: set[tuple[Any, ...]] = set()
    read_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        tool = str(row.get("tool") or "")
        result = row.get("result")
        if not isinstance(result, dict):
            raise RecoveryBlocked("coder_log_result_unreadable")
        history.append({"tool": tool, "result": result})
        last_step = max(last_step, int(row.get("step") or 0))
        if tool in _READ_TOOLS:
            reads += 1
            args = row.get("args") if isinstance(row.get("args"), dict) else {}
            if result.get("ok"):
                key = row.get("read_key")
                restored_key = tuple(key) if isinstance(key, list) else _read_key(tool, args, last_patch_sha)
                seen_reads.add(restored_key)
                read_cache[restored_key] = result
        if tool == "requires_framework_extension_answered":
            extension_answered = True
        if tool in _WRITE_TOOLS and result.get("ok"):
            finished = False
            writes += 1
            last_patch_sha = str(result.get("sha256") or "")
            auto_check = result.get("auto_check")
            if isinstance(auto_check, dict):
                last_check = auto_check
                check_fingerprint = str(auto_check.get("check_fingerprint") or "")
                failed_key = check_fingerprint or last_patch_sha
                if not auto_check.get("ok") and failed_key not in failed_revisions:
                    failed_checks += 1
                    failed_revisions.add(failed_key)
        if tool == "run_candidate_check" and _is_check_receipt(result):
            last_check = result
            check_fingerprint = str(result.get("check_fingerprint") or "")
            failed_key = check_fingerprint or last_patch_sha
            if not result.get("ok") and failed_key not in failed_revisions:
                failed_checks += 1
                failed_revisions.add(failed_key)
        if tool == "finish_patch" and result.get("ok"):
            finished = True
    if ignore_finish:
        finished = False
    checks_path = workspace / "checks.json"
    if checks_path.is_file():
        try:
            stored = json.loads(checks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecoveryBlocked("checks_unreadable") from exc
        if not isinstance(stored, dict):
            raise RecoveryBlocked("checks_unreadable")
        if _is_check_receipt(stored):
            last_check = stored
            check_fingerprint = str(stored.get("check_fingerprint") or check_fingerprint)
    entry = workspace / "extension" / "eeg_candidate.py"
    if entry.is_file():
        current = file_sha256(entry)
        if not last_patch_sha or current != last_patch_sha:
            raise RecoveryBlocked("unlogged_code_change")
    if (workspace / "source_manifest.json").is_file() and not finished and not ignore_finish:
        raise RecoveryBlocked("unlogged_finish")
    if finished:
        manifest_path = workspace / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        return {
            "done": {
                "status": "ready_for_review",
                "manifest": manifest,
                "check": last_check,
                "steps": last_step,
            }
        }
    return {
        "history": history,
        "last_check": last_check,
        "last_hash": last_patch_sha,
        "failed_checks": failed_checks,
        "failed_revisions": failed_revisions,
        "check_fingerprint": check_fingerprint,
        "extension_answered": extension_answered,
        "writes": writes,
        "reads": reads,
        "seen_reads": seen_reads,
        "read_cache": read_cache,
        "next_step": last_step + 1,
    }


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
    resume_repair: bool = False,
) -> dict[str, Any]:
    """Run the coder loop. finish_patch is refused until a check has passed.

    A complete coder log is replayed in memory. Tools already in that log are not executed again.
    """
    (workspace / "extension").mkdir(parents=True, exist_ok=True)
    restored = _restore_coder(workspace, ignore_finish=resume_repair)
    if restored.get("done") and not resume_repair:
        return restored["done"]
    log_path = workspace / "coder_log.jsonl"
    history: list[dict[str, Any]] = list(restored["history"])
    last_check: dict[str, Any] | None = restored["last_check"]
    last_hash = str(restored["last_hash"] or "")
    failed_checks = int(restored["failed_checks"])
    failed_revisions = set(restored["failed_revisions"])
    last_check_fingerprint = str(restored["check_fingerprint"])
    writes = int(restored["writes"])
    reads = int(restored["reads"])
    seen_reads: set[tuple[Any, ...]] = set(restored.get("seen_reads") or [])
    read_cache = dict(restored.get("read_cache") or {})
    next_step = int(restored["next_step"])
    step_limit = max_steps
    first_step = 1
    failed_at_start = 0
    if resume_repair:
        from react_agent.eeg_research.agentic.identity import ensure_attempt
        attempt = ensure_attempt(workspace, workspace.name)
        budget_path = workspace / "coder_attempt_budget.json"
        budget = json.loads(budget_path.read_text()) if budget_path.is_file() else {}
        if budget.get("attempt_id") != attempt["attempt_id"]:
            budget = {"attempt_id": attempt["attempt_id"], "first_step": next_step,
                      "last_step": next_step + max_steps - 1, "failed_checks_at_start": failed_checks}
            budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        first_step = int(budget["first_step"])
        step_limit = int(budget["last_step"])
        failed_at_start = int(budget["failed_checks_at_start"])
    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_research.agentic.coder import _REFERENCES
    from react_agent.eeg_research.agentic.interface import candidate_interface

    spec_path = workspace / "input_spec.json"
    if spec_path.is_file():
        input_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    else:
        input_spec = candidate_interface(None)
    pages = {name: source_page(path.read_text(encoding="utf-8").splitlines(), 1, 100000, char_limit=12000) for name, path in _REFERENCES.items()}
    parent = workspace / "reference" / "parent.py"
    if parent.is_file():
        pages["reference/parent.py"] = source_page(parent.read_text(encoding="utf-8").splitlines(), 1, 100000, char_limit=12000)
    references = {name: page["text"] for name, page in pages.items()}
    reference_ranges = {name: {key: value for key, value in page.items() if key != "text"} for name, page in pages.items()}
    entry = workspace / "extension" / "eeg_candidate.py"
    extension_answered = bool(restored["extension_answered"])
    for step in range(next_step, step_limit + 1):
        if calls_left is not None and calls_left() <= reserve:
            return {"status": "implementation_failed", "detail": "budget_exhausted", "check": last_check, "steps": step - 1}
        current = None
        if entry.is_file():
            current = {"path": "extension/eeg_candidate.py", "sha256": file_sha256(entry),
                       **source_page(entry.read_text(encoding="utf-8").splitlines(), 1, 100000, char_limit=16000)}
        request = {
            "experiment_spec": spec,
            "input_spec": input_spec,
            "allowed_write_paths": ["extension/eeg_candidate.py"],
            "step": step,
            "max_steps": step_limit,
            "attempt_step": step - first_step + 1,
            "attempt_max_steps": step_limit - first_step + 1,
            "repairs_used": max(0, failed_checks - failed_at_start - 1),
            "max_repairs": max_repairs,
            "current_file": current,
            "references": references,
            "reference_ranges": reference_ranges,
            "tools": tool_catalog(),
            "check_policy": {"automatic_after_patch": True, "explicit_recheck": "cached for identical source, interface and interpreter"},
            "last_check": last_check,
            "history": history[-8:],
        }
        current_fingerprint = _check_fingerprint(workspace, python) if entry.is_file() else ""
        check_fresh = bool(current and last_hash == current["sha256"] and last_check_fingerprint == current_fingerprint)
        request["check_status"] = {
            "passed": bool(last_check and last_check.get("ok")),
            "fresh": check_fresh,
            "current_fingerprint": current_fingerprint,
            "checked_fingerprint": last_check_fingerprint,
            "next_tool": "run_candidate_check" if last_check and not check_fresh else None,
        }
        location = (last_check or {}).get("location") or {}
        if entry.is_file() and isinstance(location.get("line"), int):
            line = location["line"]
            page = read_code(workspace, "extension/eeg_candidate.py", max(1, line - 5), line + 10)
            if page.get("ok"):
                request["failure_source"] = {**page, "numbered_text": "\n".join(f"{page['start'] + i}: {text}" for i, text in enumerate(page["text"].splitlines()))}
        if step >= 3 and writes == 0:
            request["runtime_note"] = "References and the current file are already supplied. Call apply_candidate_patch now."
        elif last_check is not None and not last_check.get("ok"):
            request["runtime_note"] = "The last check failed. Prefer edit_candidate_code for a local correction, or apply_candidate_patch for a full replacement; new source is automatically checked."
            recent_reads = sum(item.get("tool") == "read_code" and bool(item.get("result", {}).get("ok")) for item in history[-2:])
            if recent_reads == 2:
                request["runtime_note"] += " Two source ranges were just supplied. Use those ranges and the exact diagnostic now; do not spend another call rereading overlapping lines."
        elif last_check is not None and last_check.get("ok") and not check_fresh:
            request["runtime_note"] = "The stored check passed for an older fingerprint. Call run_candidate_check before finish_patch; rereading source or repeating finish cannot refresh the check."
        elif last_check is not None and last_check.get("ok"):
            request["runtime_note"] = "The current check passed. Call finish_patch unless a required source change is missing."
        latest_result = (history[-1].get("result") or {}) if history else {}
        if latest_result.get("error") in {"edit_not_found", "edit_not_unique", "base_hash_mismatch"}:
            request["recovery_context"] = {
                "error": latest_result["error"], "edit_index": latest_result.get("edit_index"),
                "matches": latest_result.get("matches"), "path": latest_result.get("path"),
                "current_sha256": (current or {}).get("sha256"),
                "source_view": "current_file and failure_source; read_code if the needed range is missing",
                "missing_inputs": ["needed source range"] if not current or current.get("truncated") else [],
            }
            request["runtime_note"] = "The last edit wrote nothing. Use current_file's hash and exact source; request a missing range or make a unique edit. Do not repeat the rejected fragment."
        reply = backend(request)
        tool, args, request_error = validate_tool_request(reply)
        started = time.time()
        key = None
        cached_check = False
        if request_error is not None:
            result = request_error
        elif tool == "finish_patch":
            fresh = entry.is_file() and file_sha256(entry) == last_hash
            if fresh and last_check_fingerprint:
                fresh = _check_fingerprint(workspace, python) == last_check_fingerprint
            if not (fresh and last_check and last_check.get("ok")):
                result = {"ok": False, "error": "check_not_passed", "detail": "The current source and executable config must have a fresh passing check. Call run_candidate_check before finish_patch.", "next_tool": "run_candidate_check", "check_fresh": fresh}
            else:
                result = _execute(workspace, tool, args, last_check, python)
        elif tool == "run_candidate_check" and entry.is_file() and last_check is not None and _check_fingerprint(workspace, python) == last_check_fingerprint:
            result = {**last_check, "cached": True}
            cached_check = True
        elif tool in _READ_TOOLS:
            key = _read_key(tool, args, last_hash)
            if key in seen_reads:
                cached = read_cache.get(key)
                visible = any(item.get("tool") == tool and cached is not None and
                              all((item.get("result") or {}).get(field) == cached.get(field)
                                  for field in ("text", "start", "end", "path", "source_sha256"))
                              for item in request["history"]) if tool == "read_code" else any(
                                  item.get("tool") == tool and item.get("result") == cached for item in request["history"])
                cache_current = False
                if cached is not None and tool == "read_code":
                    path = _REFERENCES.get(str(args.get("path") or "")) or (workspace / str(args.get("path")))
                    try:
                        lines = path.read_text(encoding="utf-8").splitlines()
                        observed_sha = cached.get("source_sha256") or (key[-1] if args.get("path") == "extension/eeg_candidate.py" else None)
                        cache_current = observed_sha == file_sha256(path) and cached.get("text") == "\n".join(lines[cached["start"] - 1:cached["end"]])
                    except (OSError, KeyError, TypeError):
                        cache_current = False
                if cache_current and isinstance(cached, dict) and not visible:
                    result = {**cached, "cached": True, "recovered_from": "coder_log",
                              "detail": "The same revision's original page was no longer visible; use this page or next_range."}
                elif cached is not None and tool == "read_code" and not cache_current:
                    result = _execute(workspace, tool, args, last_check, python)
                    if result.get("ok"):
                        read_cache[key] = result
                else:
                    result = {"ok": False, "error": "repeated_read",
                              "detail": "This revision's original result is visible in history; use its page or next_range."}
            else:
                result = _execute(workspace, tool, args, last_check, python)
                if result.get("ok"):
                    seen_reads.add(key)
                    read_cache[key] = result
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
        if tool in _WRITE_TOOLS and result.get("ok"):
            last_hash = str(result.get("sha256"))
            writes += 1
            fingerprint = _check_fingerprint(workspace, python)
            if last_check is not None and fingerprint == last_check_fingerprint:
                check = {**last_check, "cached": True}
                cached_check = True
            else:
                check = {**run_candidate_check(workspace, python=python), "source_sha256": last_hash, "check_fingerprint": fingerprint}
            result = {**result, "auto_check": check}
            tool_for_check = check
        else:
            tool_for_check = None
        if tool_for_check is not None:
            last_check = tool_for_check
            last_check_fingerprint = str(tool_for_check.get("check_fingerprint") or "")
            (workspace / "checks.json").write_text(json.dumps(tool_for_check, ensure_ascii=False, indent=2), encoding="utf-8")
            if not tool_for_check.get("ok") and last_check_fingerprint not in failed_revisions:
                failed_checks += 1
                failed_revisions.add(last_check_fingerprint)
        if tool == "run_candidate_check" and request_error is None and _is_check_receipt(result) and not cached_check:
            if entry.is_file():
                result = {**result, "source_sha256": file_sha256(entry), "check_fingerprint": _check_fingerprint(workspace, python)}
            last_check = result
            last_check_fingerprint = str(result.get("check_fingerprint") or "")
            (workspace / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if not result.get("ok") and last_check_fingerprint not in failed_revisions:
                failed_checks += 1
                failed_revisions.add(last_check_fingerprint)
        row = {
            "call_id": uuid.uuid4().hex[:12],
            "step": step,
            "tool": tool,
            "args": _short(args),
            "result": result if tool in _READ_TOOLS else _short(result),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if key is not None:
            row["read_key"] = list(key)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.append({"tool": tool, "result": result if tool in _READ_TOOLS else _short(result)})
        if tool == "requires_framework_extension":
            return {"status": "requires_framework_extension", "detail": args, "steps": step}
        if tool == "finish_patch" and result.get("ok"):
            return {"status": "ready_for_review", "manifest": result["manifest"], "check": last_check, "steps": step}
        if failed_checks - failed_at_start >= max_repairs + 1:
            return {"status": "implementation_failed", "detail": "repair_limit", "check": last_check, "steps": step}
    return {"status": "implementation_failed", "detail": "step_limit", "check": last_check, "steps": step_limit}


def _camp_from_workspace(workspace: Path) -> Path:
    return workspace.parent.parent


def _execute(workspace: Path, tool: str, args: dict[str, Any], last_check: Any, python: str | None) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.ui_events import append_ui_event

    camp = _camp_from_workspace(workspace)
    emit = (camp / "campaign_state.json").is_file()
    candidate_id = workspace.name
    started = None
    if emit:
        started = append_ui_event(
            camp,
            "tool_started",
            role="candidate_coder",
            candidate_id=candidate_id,
            tool=tool,
            status="active",
            summary=tool,
        )
    try:
        if tool == "list_project_files":
            result = list_project_files(workspace)
        elif tool == "search_code":
            result = search_code(workspace, str(args.get("query") or ""))
        elif tool == "read_code":
            result = read_code(workspace, str(args.get("path") or ""), int(args.get("start") or 1), int(args.get("end") or 200))
        elif tool == "apply_candidate_patch":
            result = apply_candidate_patch(
                workspace,
                str(args.get("path") or ""),
                str(args.get("content") or ""),
                str(args.get("expected_base_hash") or ""),
            )
        elif tool == "edit_candidate_code":
            result = edit_candidate_code(workspace, args["path"], args["edits"], args["expected_base_hash"])
        elif tool == "run_candidate_check":
            result = run_candidate_check(workspace, python=python)
        elif tool == "inspect_check_result":
            result = {"ok": last_check is not None, "result": last_check}
        elif tool == "finish_patch":
            result = finish_patch(workspace, str(args.get("summary") or ""))
        elif tool == "requires_framework_extension":
            result = {"ok": True}
        else:
            result = {"ok": False, "error": f"unknown_tool:{tool}"}
    except Exception as exc:  # noqa: BLE001
        if emit:
            append_ui_event(
                camp,
                "tool_failed",
                role="candidate_coder",
                candidate_id=candidate_id,
                tool=tool,
                status="failed",
                error=type(exc).__name__,
                parent_event_id=None if started is None else started.get("event_id"),
                summary=tool,
            )
        raise
    if emit:
        append_ui_event(
            camp,
            "tool_finished" if result.get("ok") else "tool_failed",
            role="candidate_coder",
            candidate_id=candidate_id,
            tool=tool,
            status="completed" if result.get("ok") else "failed",
            error=result.get("error"),
            parent_event_id=None if started is None else started.get("event_id"),
            summary=tool,
        )
    return result


_SHORT_LIMIT = 3000
_SHORT_STRING_KEYS = ("text", "content", "summary", "detail")


def _short(value: Any, *, limit: int = _SHORT_LIMIT) -> Any:
    """Keep JSON small without turning a dict result into a truncated string."""
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= limit:
        return value
    if isinstance(value, dict):
        shortened = dict(value)
        for key in _SHORT_STRING_KEYS:
            field = shortened.get(key)
            if isinstance(field, str) and len(field) > 400:
                shortened[key] = field[:400] + "…"
        if len(json.dumps(shortened, ensure_ascii=False, default=str)) <= limit:
            return shortened
        compact = {key: shortened[key] for key in ("ok", "error", "path", "sha256", "source_sha256", "check_fingerprint", "cached", "stage") if key in shortened}
        if isinstance(shortened.get("auto_check"), dict):
            compact["auto_check"] = _short(shortened["auto_check"], limit=2000)
        compact["truncated"] = True
        compact["keys"] = sorted(shortened)
        return compact
    if isinstance(value, str):
        return value[:limit]
    return {"truncated": True}


def _review_valid(reply: Any) -> bool:
    return (
        isinstance(reply, dict)
        and reply.get("status") in {"ready", "needs_fix", "blocked"}
        and isinstance(reply.get("summary_zh"), str)
        and isinstance(reply.get("issues", []), list)
    )


_RUNTIME_INVARIANTS = {
    "training_length_runtime_owned",
    "gallery_runtime_owned",
    "feature_cache_runtime_owned",
    "split_runtime_owned",
    "validation_identity_matches",
    "candidate_interface_check_passed",
}


def satisfied_invariants(workspace: Path, contract_summary: dict[str, Any] | None = None) -> set[str]:
    """Invariants the runtime has evidence for. Keyword matches are not evidence."""
    found: set[str] = set()
    checks_path = workspace / "checks.json"
    if checks_path.is_file():
        try:
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checks = {}
        if isinstance(checks, dict) and checks.get("ok"):
            found.add("candidate_interface_check_passed")
    protocol = workspace / "input_spec.json"
    if protocol.is_file() or (contract_summary or {}).get("fingerprint"):
        found.add("training_length_runtime_owned")
        found.add("gallery_runtime_owned")
        found.add("feature_cache_runtime_owned")
        found.add("split_runtime_owned")
        found.add("validation_identity_matches")
    return found


def filter_runtime_blocking(
    issues: list[Any],
    *,
    satisfied: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep blocking issues unless a named invariant is already satisfied with evidence."""
    known = set(satisfied or ())
    filtered: list[dict[str, Any]] = []
    for issue in issues:
        row = dict(issue) if isinstance(issue, dict) else {"title": str(issue), "severity": "blocking"}
        if row.get("blocking") is True:
            row["severity"] = "blocking"
        elif row.get("severity") not in {"blocking", "non_blocking"}:
            row["severity"] = "non_blocking" if row.get("blocking") is False else "blocking"
        invariant = str(row.get("invariant_id") or "")
        if row.get("severity") == "blocking" and invariant and invariant in known and invariant in _RUNTIME_INVARIANTS:
            row["severity"] = "non_blocking"
            row["already_satisfied"] = True
            row["downgrade_reason"] = "already_satisfied"
        filtered.append(row)
    return filtered


def apply_review_filter(result: dict[str, Any], *, satisfied: set[str] | None = None) -> dict[str, Any]:
    """Recompute review status after already-satisfied invariants are marked."""
    issues = filter_runtime_blocking(list(result.get("issues") or []), satisfied=satisfied)
    blocking = [row for row in issues if row.get("severity") == "blocking"]
    updated = dict(result)
    updated["issues"] = issues
    updated["model_status"] = result.get("model_status") or result.get("status")
    updated["runtime_issues_downgraded"] = sum(1 for row in issues if row.get("already_satisfied"))
    if result.get("format_failed"):
        updated["status"] = "blocked"
    elif blocking and result.get("status") == "ready":
        updated["status"] = "needs_fix"
    elif not blocking and (result.get("status") == "ready" or updated["runtime_issues_downgraded"]):
        updated["status"] = "ready"
    coverage = result.get("intervention_coverage")
    def unresolved(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("status") in {"missing", "not_implemented", "partial", "failed", "unknown"}:
                return True
            return any(unresolved(item) for item in value.values())
        if isinstance(value, list):
            return any(unresolved(item) for item in value)
        return isinstance(value, str) and value in {"missing", "not_implemented", "partial", "failed", "unknown"}
    if coverage is not None and unresolved(coverage):
        updated["status"] = "needs_fix"
        updated["coverage_unresolved"] = True
    updated["blocking_remaining"] = len(blocking)
    return updated


def review(
    workspace: Path,
    spec: dict[str, Any],
    contract_summary: dict[str, Any],
    backend: Backend,
    model: str,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reviewer sees spec, code and checks. It returns blocking issues, not a score."""
    entry = workspace / "extension" / "eeg_candidate.py"
    from react_agent.eeg_research.agentic.interface import candidate_interface

    checks = json.loads((workspace / "checks.json").read_text(encoding="utf-8")) if (workspace / "checks.json").is_file() else None
    spec_path = workspace / "input_spec.json"
    interface = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else candidate_interface(None)
    payload = {
        "experiment_spec": spec,
        "candidate_source": entry.read_text(encoding="utf-8") if entry.is_file() else None,
        "checks": checks,
        "evaluation_contract": contract_summary,
        "candidate_interface": interface,
        "runtime_guarantees": interface.get("runtime_guarantees"),
        "reviewer_model": model,
        "executor_model": model,
    }
    from react_agent.eeg_research.agentic.handoffs import training_semantics
    payload["training_runtime_source"] = training_semantics()
    parent = workspace / "reference/parent.py"
    if parent.is_file():
        from react_agent.eeg_research.agentic.artifacts import file_digest
        text = parent.read_text(encoding="utf-8")
        payload["parent_source"] = {"source_ref": str(parent), "source_hash": file_digest(parent),
                                    "source": text[:16000], "source_truncated": len(text) > 16000}
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
            "intervention_coverage": None if format_failed else reply.get("intervention_coverage"),
            "verified_invariants_with_refs": [] if format_failed else reply.get("verified_invariants_with_refs") or [],
            "unverified_invariants": [] if format_failed else reply.get("unverified_invariants") or [],
            "summary_zh": "审查回复格式不合格，按受阻处理。" if format_failed else reply.get("summary_zh"),
            "format_failed": format_failed,
            "schema_repaired": repaired,
            "calls": 2 if repaired else 1,
            "reviewer_model": model,
            "same_family_as_executor": True,
        },
        satisfied=satisfied_invariants(workspace, contract_summary),
    )
    if identity:
        for key in ("candidate_id", "attempt_id", "phase", "operation_id"):
            if identity.get(key):
                result[key] = identity[key]
    experiment = spec.get("experiment") or {}
    if (experiment.get("principal_intervention") or experiment.get("intervention")) and not result.get("intervention_coverage"):
        result["status"] = "needs_fix"
        result["coverage_unresolved"] = True
        result["issues"].append({"severity": "blocking", "kind": "intervention_coverage_missing",
                                 "detail": "Review must identify the approved intervention's implementation and supporting source/check refs."})
    from react_agent.eeg_research.agentic.identity import source_hash

    result["input_hash"] = source_hash(workspace)
    result["phase"] = result.get("phase") or "review_candidate"
    tmp = workspace / "review.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(workspace / "review.json")
    return result
