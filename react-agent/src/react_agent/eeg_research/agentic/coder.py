"""Candidate tools. The model names a tool; the server executes it."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

_SECRET_NAMES = {".env", "credentials.json", "id_rsa"}
_BLOCKED_PARTS = {"evaluator", "controller", "protocol.py", "launch.py", "final_test"}


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
}


def read_code(workspace: Path, relative: str, start: int = 1, end: int = 200) -> dict[str, Any]:
    if relative in _REFERENCES:
        target = _REFERENCES[relative]
    else:
        target = (workspace / relative).resolve()
        if not _allowed(workspace, target) or not target.is_file():
            return {"ok": False, "error": "read_refused"}
    lines = target.read_text(encoding="utf-8").splitlines()
    chunk = lines[max(0, start - 1) : end]
    return {"ok": True, "text": "\n".join(chunk)[:4000]}


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
        return {"ok": False, "error": "base_hash_mismatch"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": relative, "sha256": file_sha256(target)}


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
    try:
        completed = subprocess.run(  # noqa: S603
            [interpreter, "-m", "react_agent.eeg_research.agentic.check_entry", "eeg"],
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
    payload["loaded_from_workspace"] = str(payload.get("file", "")).startswith(str(extension))
    if not payload["loaded_from_workspace"]:
        payload["ok"] = False
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
