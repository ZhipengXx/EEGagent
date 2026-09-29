"""Flash JSON recovery. Fake LLM only. No API or training."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from react_agent.eeg_research.agentic.llm import LlmUnavailable, role_backend
from react_agent.eeg_research.agentic.native_patch import implement
from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.llm.deepseek import (
    DeepSeekBackend,
    DeepSeekParseError,
    message_text,
    parse_json_object,
    raw_excerpt,
)
from react_agent.fmri.schemas import LMUsage

_CANDIDATE = """
from react_agent.eeg_research.agentic.baseline import EEGCandidate as Baseline

class EEGCandidate(Baseline):
    candidate_id = "c1"
"""


def _usage() -> LMUsage:
    return LMUsage(
        provider="deepseek",
        requested_model="deepseek-flash",
        response_model="deepseek-flash",
        finish_reason="stop",
        input_tokens=4,
        output_tokens=8,
    )


def test_empty_content_uses_reasoning_json() -> None:
    text = message_text({"content": "", "reasoning_content": '{"tool":"list_project_files","args":{}}'})
    assert parse_json_object(text)["tool"] == "list_project_files"


def test_fenced_json_is_an_object() -> None:
    payload = parse_json_object('```json\n{"tool":"finish_patch","args":{"summary":"ok"}}\n```')
    assert payload["tool"] == "finish_patch"


def test_json_followed_by_dsml_keeps_the_tool_call() -> None:
    text = """{"tool": "list_project_files", "args": {}}

<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="list_project_files">
<｜｜DSML｜｜ parameter name="args" string="true">{}</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>
"""
    payload = parse_json_object(text)
    assert payload == {"tool": "list_project_files", "args": {}}


def test_last_tool_object_wins_when_prose_follows() -> None:
    text = """{"tool": "list_project_files", "args": {}}

---

{"tool": "search_code", "args": {"query": "positional"}}

Let me continue step by step.
"""
    payload = parse_json_object(text)
    assert payload["tool"] == "search_code"
    assert payload["args"]["query"] == "positional"


def test_prose_without_json_still_fails() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("Let me inspect the project files first.")


def test_complete_json_reads_reasoning_and_fences() -> None:
    backend = DeepSeekBackend(FmriCheckConfig(deepseek_api_key="test-key"))

    async def reasoning(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", reasoning_content='{"ok": true}'),
                    finish_reason="stop",
                )
            ],
            model="deepseek-flash",
            usage=None,
        )

    backend._client.chat.completions.create = reasoning  # type: ignore[method-assign]
    reply, usage = asyncio.run(backend.complete_json(system="s", user="{}", profile="fast", role="flash_probe"))
    assert reply == {"ok": True}
    assert usage.finish_reason == "stop"
    assert usage.response_model == "deepseek-flash"

    async def fenced(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='```json\n{"tool":"list_project_files","args":{}}\n```'),
                    finish_reason="stop",
                )
            ],
            model="deepseek-flash",
            usage=None,
        )

    backend._client.chat.completions.create = fenced  # type: ignore[method-assign]
    reply, _usage = asyncio.run(backend.complete_json(system="s", user="{}", profile="fast", role="flash_probe"))
    assert reply["tool"] == "list_project_files"


def test_three_empty_objects_are_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    hits = {"n": 0}

    class Fake:
        def __init__(self, config) -> None:
            assert config.fast.max_tokens >= 16384

        async def complete_json(self, **_kwargs):
            hits["n"] += 1
            raise DeepSeekParseError("", _usage())

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    backend = role_backend(tmp_path, "candidate_coder")
    with pytest.raises(LlmUnavailable, match="DeepSeekParseError"):
        backend({"step": 1})
    assert hits["n"] == 3
    rows = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all(row["success"] is False for row in rows)
    assert all(row.get("raw_excerpt") == "" for row in rows)
    assert rows[0]["response_model"] == "deepseek-flash"
    assert rows[0]["finish_reason"] == "stop"


def test_reviewer_and_analyst_use_long_max_tokens(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    seen: dict[str, int] = {}

    class Fake:
        def __init__(self, config) -> None:
            seen["max_tokens"] = int(config.fast.max_tokens)

        async def complete_json(self, **_kwargs):
            return ({"status": "ready"}, _usage())

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    for role in ("candidate_reviewer", "result_analyst", "research_planner"):
        seen.clear()
        backend = role_backend(tmp_path / role, role)
        assert backend({"ping": True})["status"] == "ready"
        assert seen["max_tokens"] >= 16384


def test_implement_retries_one_bad_json_then_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    workspace = tmp_path / "c1"
    (workspace / "extension").mkdir(parents=True)
    log = workspace / "coder_log.jsonl"
    rows = [
        {"call_id": "a", "step": 1, "tool": "search_code", "args": {"query": "EEGCandidate"}, "result": {"ok": True, "hits": []}},
        {"call_id": "b", "step": 2, "tool": "list_project_files", "args": {}, "result": {"ok": True, "files": []}},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hits = {"n": 0}

    class Fake:
        def __init__(self, _config) -> None:
            return

        async def complete_json(self, **_kwargs):
            hits["n"] += 1
            if hits["n"] == 1:
                raise DeepSeekParseError("not an object", _usage())
            if hits["n"] == 2:
                return (
                    {
                        "tool": "apply_candidate_patch",
                        "args": {"path": "extension/eeg_candidate.py", "content": _CANDIDATE, "expected_base_hash": ""},
                    },
                    _usage(),
                )
            return ({"tool": "finish_patch", "args": {"summary": "c1"}}, _usage())

    monkeypatch.setattr("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake)
    monkeypatch.setattr(
        "react_agent.eeg_research.agentic.native_patch.run_candidate_check",
        lambda *_args, **_kwargs: {"ok": True},
    )
    backend = role_backend(tmp_path, "candidate_coder")
    implement(workspace, {"hypothesis": None}, backend, max_steps=4)
    entry = workspace / "extension" / "eeg_candidate.py"
    assert entry.is_file()
    assert "class EEGCandidate" in entry.read_text(encoding="utf-8")
    assert not (tmp_path / "c2").exists()
    ledger = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[0]["success"] is False
    assert ledger[0]["raw_excerpt"] == "not an object"
    assert ledger[1]["success"] is True
    assert raw_excerpt("x" * 3000) == "x" * 2048
