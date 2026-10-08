"""Candidate tools. The model names a tool; the server executes it."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

_SECRET_NAMES = {".env", "credentials.json", "id_rsa"}
_BLOCKED_PARTS = {"evaluator", "controller", "protocol.py", "launch.py", "final_test"}

# This catalog is both the model-facing contract and the argument validator.
TOOL_ARGUMENTS = {
    "list_project_files": {},
    "search_code": {"query": ("string", True)},
    "read_code": {"path": ("string", True), "start": ("integer", False), "end": ("integer", False)},
    "apply_candidate_patch": {
        "path": ("string", True), "content": ("string", True), "expected_base_hash": ("string", True),
    },
    "edit_candidate_code": {
        "path": ("string", True), "edits": ("array", True), "expected_base_hash": ("string", True),
    },
    "run_candidate_check": {},
    "inspect_check_result": {},
    "finish_patch": {"summary": ("string", True)},
    "requires_framework_extension": None,
}


def tool_catalog() -> dict[str, Any]:
    catalog = {
        name: {"args": {key: {"type": kind, "required": required} for key, (kind, required) in (fields or {}).items()}}
        for name, fields in TOOL_ARGUMENTS.items()
    }
    catalog["edit_candidate_code"]["args"]["edits"]["items"] = {
        "one_of": [{"old": "exact unique source string", "new": "replacement string"},
                   {"start": "1-based inclusive line", "end": "1-based inclusive line", "new": "replacement source"}],
    }
    return catalog


def validate_tool_request(reply: Any) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Turn model formatting mistakes into feedback, without executing a tool."""
    if not isinstance(reply, dict) or not isinstance(reply.get("tool"), str):
        return "", {}, {"ok": False, "error": "invalid_tool_request", "detail": "Return {tool: string, args: object} using the supplied tools."}
    if set(reply) - {"tool", "args"}:
        return reply["tool"], {}, {"ok": False, "error": "invalid_tool_request",
            "forbidden_fields": sorted(set(reply) - {"tool", "args"}),
            "detail": "Return only tool and args. Input examples, history and metadata are not output fields."}
    tool = reply["tool"]
    args = reply.get("args", {})
    if tool not in TOOL_ARGUMENTS:
        return tool, {}, {"ok": False, "error": f"unknown_tool:{tool}", "allowed_tools": list(TOOL_ARGUMENTS)}
    if not isinstance(args, dict):
        return tool, {}, {"ok": False, "error": "invalid_tool_args", "detail": "args must be an object"}
    fields = TOOL_ARGUMENTS[tool]
    argument_types = {"string": str, "integer": int, "array": list}
    issues = []
    if fields is not None:
        issues.extend("unknown argument: " + str(key) for key in args if key not in fields)
        for name, (kind, required) in fields.items():
            if name not in args:
                if required:
                    issues.append("missing argument: " + name)
            elif not isinstance(args[name], argument_types[kind]) or (kind == "integer" and isinstance(args[name], bool)):
                issues.append(name + " must be " + kind)
    if tool == "read_code" and not issues:
        start, end = args.get("start", 1), args.get("end", 200)
        if start < 1 or end < start:
            issues.append("read range must satisfy 1 <= start <= end")
    error = {"ok": False, "error": "invalid_tool_args", "detail": "; ".join(issues)} if issues else None
    return tool, args, error


def source_page(lines: list[str], start: int, end: int, *, char_limit: int = 4000) -> dict[str, Any]:
    """Return complete lines; pagination resumes at the first undelivered line."""
    total = len(lines)
    chunk: list[str] = []
    size = 0
    for line in lines[start - 1 : end]:
        extra = len(line) + (1 if chunk else 0)
        if chunk and size + extra > char_limit:
            break
        chunk.append(line)
        size += extra
    delivered_end = start + len(chunk) - 1
    truncated = delivered_end < total
    return {
        "text": "\n".join(chunk), "truncated": truncated,
        "next_range": {"start": delivered_end + 1, "end": min(total, delivered_end + max(1, end - start + 1))} if truncated else None,
        "total_lines": total, "start": start, "end": delivered_end,
    }


def _allowed(workspace: Path, target: Path) -> bool:
    root = (workspace / "extension").resolve()
    try:
        target.resolve().relative_to(root)
    except ValueError:
        return False
    name = target.name
    if name in _SECRET_NAMES or "test.pt" in name or "final_test" in str(target):
        return False
    return not any(part in _BLOCKED_PARTS for part in target.parts)


def list_project_files(workspace: Path) -> dict[str, Any]:
    root = workspace / "extension"
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*.py")):
            if _allowed(workspace, path):
                files.append(str(path.relative_to(workspace)))
    return {"ok": True, "files": files[:80]}


_REFERENCES = {
    "reference/baseline.py": Path(__file__).resolve().parent / "baseline.py",
    "reference/model.py": Path(__file__).resolve().parents[2] / "eeg_training" / "model.py",
    "reference/train_entry.py": Path(__file__).resolve().parents[2] / "eeg_training" / "train_entry.py",
    "reference/hooks.py": Path(__file__).resolve().parents[2] / "eeg_training" / "hooks.py",
    "reference/hook_config.py": Path(__file__).resolve().parent / "hook_config.py",
}


def read_code(workspace: Path, relative: str, start: int = 1, end: int = 200) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.binding import file_sha256
    if relative in _REFERENCES:
        target = _REFERENCES[relative]
    elif relative == "reference/parent.py":
        target = workspace / "reference" / "parent.py"
        if not target.is_file():
            return {"ok": False, "error": "read_refused"}
    else:
        target = (workspace / relative).resolve()
        if not _allowed(workspace, target) or not target.is_file():
            return {"ok": False, "error": "read_refused"}
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(1, int(start))
    end = int(end)
    return {"ok": True, "path": relative, "source_sha256": file_sha256(target), **source_page(lines, start, end)}


def search_code(workspace: Path, query: str) -> dict[str, Any]:
    hits = []
    for path in (workspace / "extension").rglob("*.py"):
        if not _allowed(workspace, path):
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if query and query in line:
                hits.append({"path": str(path.relative_to(workspace)), "line": index, "text": line[:180]})
            if len(hits) >= 20:
                return {"ok": True, "hits": hits}
    return {"ok": True, "hits": hits}


def apply_candidate_patch(workspace: Path, relative: str, content: str, expected_base_hash: str) -> dict[str, Any]:
    """Replace one extension file. Paths outside the workspace are refused."""
    from react_agent.eeg_research.agentic.binding import file_sha256

    target = (workspace / relative).resolve()
    if not str(relative).startswith("extension/") or not _allowed(workspace, target):
        return {"ok": False, "error": "patch_refused"}
    if target.is_file() and file_sha256(target) != expected_base_hash:
        return {"ok": False, "error": "base_hash_mismatch", "path": relative,
                "current_sha256": file_sha256(target),
                "detail": "The file revision changed. Use the current file/page and its hash, not a reference or previous revision."}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": relative, "sha256": file_sha256(target)}


def edit_candidate_code(workspace: Path, relative: str, edits: list[Any], expected_base_hash: str) -> dict[str, Any]:
    """Validate versioned text or line edits in memory before writing the candidate."""
    from react_agent.eeg_research.agentic.binding import file_sha256

    target = (workspace / relative).resolve()
    if not str(relative).startswith("extension/") or not _allowed(workspace, target) or not target.is_file():
        return {"ok": False, "error": "edit_refused"}
    if file_sha256(target) != expected_base_hash:
        return {"ok": False, "error": "base_hash_mismatch", "path": relative,
                "current_sha256": file_sha256(target),
                "detail": "The file revision changed. Rebuild the edit from the current source and hash."}
    if not edits or len(edits) > 20:
        return {"ok": False, "error": "invalid_tool_args", "detail": "edits must contain 1 to 20 exact replacements"}
    content = target.read_text(encoding="utf-8")
    for index, edit in enumerate(edits):
        if isinstance(edit, dict) and set(edit) == {"start", "end", "new"}:
            start, end = edit["start"], edit["end"]
            lines = content.splitlines(keepends=True)
            if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or not 1 <= start <= end <= len(lines) or not isinstance(edit["new"], str):
                return {"ok": False, "error": "invalid_tool_args", "detail": "Line edits need integer 1 <= start <= end <= total_lines and a new string", "edit_index": index}
            replacement = edit["new"]
            if replacement and lines[end - 1].endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            lines[start - 1 : end] = [replacement] if replacement else []
            content = "".join(lines)
            continue
        if not isinstance(edit, dict) or set(edit) != {"old", "new"} or not isinstance(edit.get("old"), str) or not isinstance(edit.get("new"), str) or not edit["old"]:
            return {"ok": False, "error": "invalid_tool_args", "detail": "Use old/new strings or start/end/new line replacements", "edit_index": index}
        count = content.count(edit["old"])
        if count != 1:
            return {"ok": False, "error": "edit_not_found" if count == 0 else "edit_not_unique", "edit_index": index, "matches": count,
                    "path": relative, "current_sha256": file_sha256(target),
                    "detail": "No exact old fragment matched; use the current source." if count == 0 else
                              "The old fragment matches more than once; include unique surrounding lines or observed line ranges."}
        content = content.replace(edit["old"], edit["new"], 1)
    return {**apply_candidate_patch(workspace, relative, content, expected_base_hash), "edit_count": len(edits)}


def run_candidate_check(workspace: Path, python: str | None = None, timeout_s: float = 300.0) -> dict[str, Any]:
    """Run check_entry in the training interpreter. The child gets no API key and no shell."""
    import subprocess

    from react_agent.eeg_research.agentic.runner import research_env
    from react_agent.eeg_training.protocol import torch_python

    interpreter = python or torch_python()
    if interpreter is None:
        return {"ok": False, "error": "torch_python_missing"}
    extension = (workspace / "extension").resolve()
    if not (extension / "eeg_candidate.py").is_file():
        return {"ok": False, "error": "entry_missing"}
    env = research_env(extension)
    env.pop("EEG_CANDIDATE_MODULE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    spec_path = workspace / "input_spec.json"
    dataset_arg = str(spec_path.resolve()) if spec_path.is_file() else "eeg"
    command = [interpreter, "-m", "react_agent.eeg_research.agentic.check_entry", dataset_arg]
    approved_spec = workspace / "spec.json"
    if approved_spec.is_file():
        command.append(str(approved_spec.resolve()))
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=str(extension),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "check_timeout"}
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {"ok": False, "error": "check_no_output", "detail": completed.stderr[-1500:]}
    try:
        payload = json.loads(lines[-1])
    except ValueError:
        return {"ok": False, "error": "check_bad_output", "detail": lines[-1][:500]}
    payload["returncode"] = completed.returncode
    loaded = str(payload.get("file") or "")
    try:
        Path(loaded).resolve().relative_to(extension)
        payload["loaded_from_workspace"] = True
    except (ValueError, OSError):
        payload["loaded_from_workspace"] = False
        payload["ok"] = False
    if completed.returncode != 0:
        payload["ok"] = False
        payload.setdefault("error", "check_nonzero_returncode")
    input_spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    if payload.get("ok") and (input_spec.get("multigpu_check") or {}).get("enabled"):
        from react_agent.eeg_research.agentic.execution_protocol import load_protocol
        from react_agent.eeg_research.agentic.multigpu_check import run_multigpu_check, validate_receipt
        camp = workspace.resolve().parent.parent
        protocol = load_protocol(camp)
        if protocol is None or protocol.get("evaluation_mode") != "loso_method_search":
            receipt = {"ok": False, "error": "multigpu_probe_frozen_protocol_missing"}
        else:
            receipt = run_multigpu_check(camp, protocol, workspace=workspace, python=interpreter, timeout_s=min(timeout_s, 120.0))
            if receipt.get("ok"):
                try:
                    validate_receipt(camp, protocol, receipt, workspace=workspace)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    receipt = {**receipt, "ok": False, "error": "multigpu_check_current_identity_invalid", "detail": str(exc)}
        if receipt.get("status") in {"blocked", "running", "uncertain_pending", "uncertain", "blocked_live_child", "blocked_live_gpu_child_after_deadline"}:
            from react_agent.eeg_research.agentic.native_patch import RecoveryBlocked
            raise RecoveryBlocked("multigpu_probe_environment_or_recovery_blocked:" + str(receipt.get("error")))
        payload["multigpu_check"] = receipt
        if not receipt.get("ok"):
            payload["ok"] = False
            payload["failures"] = [*(payload.get("failures") or []), "multigpu_runtime_check_failed"]
            payload["error"] = receipt.get("error") or "multigpu_runtime_check_failed"
            payload["detail"] = (receipt.get("result") or {}).get("detail") or receipt.get("stderr_tail") or str(receipt)
    return payload


def inspect_check_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "result": payload}


def finish_patch(workspace: Path, summary: str) -> dict[str, Any]:
    entry = workspace / "extension" / "eeg_candidate.py"
    if not entry.is_file():
        return {"ok": False, "error": "entry_missing"}
    ast.parse(entry.read_text(encoding="utf-8"))
    from react_agent.eeg_research.agentic.binding import manifest_for

    manifest = manifest_for(workspace, "eeg_candidate", entry)
    manifest["summary"] = summary[:500]
    (workspace / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "manifest": manifest}
