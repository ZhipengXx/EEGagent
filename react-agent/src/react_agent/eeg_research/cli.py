"""Commands for probe, dry-run plan, run, inspect, and resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from react_agent.eeg_research.adapters.mock_retrieval import MockRetrievalAdapter
from react_agent.eeg_research.adapters.unavailable import UnavailableRetrievalAdapter
from react_agent.eeg_research.executor import ResearchBudget
from react_agent.eeg_research.loop import run_campaign
from react_agent.eeg_research.memory import ResearchMemory
from react_agent.eeg_research.planner import FailingBackend, OfflinePlanner
from react_agent.eeg_research.registry import Registry
from react_agent.eeg_research.report import read_campaign
from react_agent.eeg_research.schemas import ModelRecord, ProfileRecord, TaskCard

_REPO = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.eeg_research.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "plan", "run", "inspect", "resume"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=_REPO / "configs" / "eeg_research_v1_6.yaml")
        cmd.add_argument("--out", type=Path, required=True)
        if name == "plan":
            cmd.add_argument("--dry-run", action="store_true")
    return parser


def load_runtime(config_path: Path, scores: dict[str, float] | None = None) -> tuple[TaskCard, Registry, ResearchBudget]:
    """Build the frozen card, registry, and budget from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    card = TaskCard.model_validate(raw["task"])
    models = [ModelRecord.model_validate(row) for row in raw["models"]]
    profiles = [ProfileRecord.model_validate(row) for row in raw["profiles"]]
    adapters = {
        "mock_retrieval": MockRetrievalAdapter(scores or raw.get("mock_scores") or {}),
        "unavailable_retrieval": UnavailableRetrievalAdapter(),
    }
    registry = Registry(models, profiles, adapters)
    limits = raw["budget"]
    budget = ResearchBudget(
        max_trials=int(limits["max_trials"]),
        max_lm_calls=int(limits["max_lm_calls"]),
        max_execution_attempts=int(limits["max_execution_attempts"]),
        gpu_seconds=limits.get("gpu_seconds"),
        per_trial_timeout_s=limits.get("per_trial_timeout_s"),
    )
    return card, registry, budget


def main(argv: list[str] | None = None) -> int:
    """Dispatch one research command."""
    args = _parser().parse_args(argv)
    card, registry, budget = load_runtime(args.config)
    if args.cmd == "inspect":
        payload = read_campaign(args.out.parent, args.out.name)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0 if payload else 1
    if os.environ.get("EEG_RESEARCH_POLICY") == "adaptive":
        from react_agent.eeg_research.controller import ResearchController

        record = ResearchController(args.out).decide(
            campaign_id=args.out.name,
            observation="inspect",
            trials_used=0,
            max_trials=budget.max_trials,
            memory_rows=[],
        )
        print(json.dumps({"stop_reason": record["status"], "api_usd": budget.api_usd}, ensure_ascii=False))
        return 0
    backend = OfflinePlanner()
    if os.environ.get("EEG_RESEARCH_BACKEND") == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        backend = FailingBackend()
    memory = ResearchMemory(args.out / "memory.sqlite3")
    try:
        state = run_campaign(
            card,
            registry,
            backend,
            budget,
            args.out,
            campaign_id=args.out.name,
            resume=args.cmd == "resume",
            dry_run=args.cmd in {"probe", "plan"},
            memory=memory,
        )
    finally:
        memory.close()
    print(json.dumps({"stop_reason": state.get("stop_reason"), "api_usd": budget.api_usd}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
