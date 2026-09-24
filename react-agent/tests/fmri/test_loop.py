"""Scheduling, reporting, and mock LM behaviour."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from react_agent.fmri.config import load_config
from react_agent.fmri.demo import make_demo
from react_agent.fmri.llm.mock import MockBackend
from react_agent.fmri.loop import run_sample
from react_agent.fmri.schemas import SampleSpec
from react_agent.fmri.tools.reference import fit_reference


@pytest.fixture()
def demo_dir(tmp_path: Path) -> Path:
    make_demo(tmp_path)
    return tmp_path


def test_two_tool_paths(demo_dir: Path) -> None:
    cfg = load_config()
    spike = SampleSpec.model_validate_json((demo_dir / "temporal_spike.json").read_text())
    ok = SampleSpec.model_validate_json((demo_dir / "ok_numeric.json").read_text())
    r1 = asyncio.run(
        run_sample(
            spike,
            base_dir=demo_dir,
            out_dir=demo_dir / "p1",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    r2 = asyncio.run(
        run_sample(
            ok,
            base_dir=demo_dir,
            out_dir=demo_dir / "p2",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    assert r1["executed_tools"] != r2["executed_tools"] or "temporal_diagnostics" in r1["executed_tools"]
    assert "validate_input" in r1["executed_tools"]
    assert "roi_summary" in r2["executed_tools"]


def test_new_evidence_visible_to_next_decision(demo_dir: Path) -> None:
    cfg = load_config()
    spec = SampleSpec.model_validate_json((demo_dir / "temporal_spike.json").read_text())
    mock = MockBackend(
        script=[
            {
                "action": "run_tool",
                "tool_name": "temporal_diagnostics",
                "reason": "Need temporal localization.",
                "evidence_refs": [],
            },
            {
                "action": "stop",
                "stop_reason": "configured_checks_complete",
                "reason": "Saw temporal diagnostics.",
            },
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "hyb",
            config=cfg,
            backend="mock",
            policy="hybrid",
            mock=mock,
        )
    )
    events = (demo_dir / "hyb" / "events.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in events if line]
    tool_events = [e for e in parsed if e.get("type") == "tool"]
    decisions = [e for e in parsed if e.get("type") == "decision"]
    assert any(e.get("tool_name") == "temporal_diagnostics" for e in tool_events)
    assert len(decisions) >= 2
    assert "temporal_diagnostics" in report["executed_tools"]


def test_required_incomplete_cannot_pass(demo_dir: Path) -> None:
    cfg = load_config()
    cfg.required_checks = ["validate_input", "basic_statistics", "temporal_diagnostics"]
    spec = SampleSpec.model_validate_json((demo_dir / "ok_numeric.json").read_text())
    # Force skip temporal by raising min_length above T
    cfg.temporal_diagnostics.min_length = 10_000
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "req",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    assert report["verdict"] != "passed_configured_checks"


def test_skipped_not_success(demo_dir: Path) -> None:
    cfg = load_config()
    spec = SampleSpec.model_validate_json((demo_dir / "missing_tr.json").read_text())
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "skip",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    assert "roi_summary" not in report["executed_tools"]


def test_invalid_json_fallback(demo_dir: Path) -> None:
    cfg = load_config()
    spec = SampleSpec.model_validate_json((demo_dir / "temporal_spike.json").read_text())

    class BadMock(MockBackend):
        async def decide(self, observation, candidate_tools, profile):
            raise ValueError("not json")

    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "bad",
            config=cfg,
            backend="mock",
            policy="hybrid",
            mock=BadMock(),
        )
    )
    assert report["fallback_used"] is True


def test_repeat_and_budget_stop(demo_dir: Path) -> None:
    cfg = load_config()
    cfg.budgets.max_tool_calls = 2  # required checks fill the budget
    spec = SampleSpec.model_validate_json((demo_dir / "temporal_spike.json").read_text())
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "budget",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    assert report["stop_reason"] in {"max_tool_calls", "flagged_findings", "configured_checks_complete"}
    assert report["cost"]["tool_calls"] <= 2


def test_batch_mixed_does_not_crash(demo_dir: Path) -> None:
    from react_agent.fmri.data import load_manifest

    cfg = load_config()
    rows = []
    for spec, base in load_manifest(demo_dir / "samples.jsonl"):
        rows.append(
            asyncio.run(
                run_sample(
                    spec,
                    base_dir=base,
                    out_dir=demo_dir / "batch",
                    config=cfg,
                    backend="none",
                    policy="rule",
                )
            )
        )
    assert len(rows) == len(list(load_manifest(demo_dir / "samples.jsonl")))
    verdicts = {r["verdict"] for r in rows}
    assert "invalid_input" in verdicts
    assert len(verdicts) >= 2
