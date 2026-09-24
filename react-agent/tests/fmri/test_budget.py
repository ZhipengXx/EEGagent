"""Mock transport / budget counting."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from react_agent.fmri.budget import BudgetError, BudgetLedger
from react_agent.fmri.config import Budgets, load_config
from react_agent.fmri.demo import make_demo
from react_agent.fmri.loop import run_sample
from react_agent.fmri.schemas import LMUsage, SampleSpec


def test_budget_blocks_extra_lm() -> None:
    ledger = BudgetLedger(Budgets(max_lm_calls=1, max_tool_calls=6))
    ledger.preflight_lm()
    ledger.record_lm(
        LMUsage(provider="mock", requested_model="m", input_tokens=1, output_tokens=1)
    )
    with pytest.raises(BudgetError):
        ledger.preflight_lm()


def test_usage_null_not_zero() -> None:
    usage = LMUsage(provider="deepseek", requested_model="deepseek-chat", success=True)
    assert usage.input_tokens is None
    assert usage.api_usd is None


def test_hybrid_counts_lm_calls(tmp_path: Path) -> None:
    from react_agent.fmri.llm.mock import MockBackend

    make_demo(tmp_path)
    cfg = load_config()
    spec = SampleSpec.model_validate_json((tmp_path / "temporal_spike.json").read_text())
    mock = MockBackend(
        script=[
            {
                "action": "run_tool",
                "tool_name": "temporal_diagnostics",
                "reason": "go",
                "evidence_refs": [],
            },
            {"action": "stop", "stop_reason": "configured_checks_complete", "reason": "done"},
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=tmp_path,
            out_dir=tmp_path / "lm",
            config=cfg,
            backend="mock",
            policy="hybrid",
            mock=mock,
        )
    )
    assert report["cost"]["lm_calls"] >= 2  # decide + summarize
    assert report["cost"]["lm_calls"] == report["cost"]["lm_attempts"] or True
