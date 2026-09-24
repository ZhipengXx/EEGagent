"""Write the campaign directory and a short report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.executor import ResearchBudget
from react_agent.eeg_research.registry import Registry
from react_agent.eeg_research.schemas import TaskCard


def write_campaign(
    out_dir: Path,
    card: TaskCard,
    registry: Registry,
    state: dict[str, Any],
    budget: ResearchBudget,
) -> None:
    """Persist the files a later Research tab can read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "task_card.json", card.model_dump())
    _write(out_dir / "registry_snapshot.json", registry.snapshot())
    _jsonl(out_dir / "hypotheses.jsonl", state["hypotheses"])
    _jsonl(out_dir / "experiments.jsonl", state["experiments"])
    _jsonl(
        out_dir / "events.jsonl",
        [{"event": "stop", "reason": state.get("stop_reason")}],
    )
    decision = state.get("decision") or {}
    _write(
        out_dir / "comparison.json",
        {
            "decision": decision,
            "delta": decision.get("delta"),
            "test_result": None,
            "model_selection_assessed": False,
        },
    )
    _write(out_dir / "cost.json", budget.as_dict())
    (out_dir / "report.md").write_text(_markdown(card, state, budget), encoding="utf-8")


def read_campaign(root: Path, campaign_id: str) -> dict[str, Any] | None:
    """Read one campaign under the research root. Paths outside the root are refused."""
    rel = campaign_id.strip().strip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if not (candidate / "task_card.json").is_file():
        return None
    comparison = json.loads((candidate / "comparison.json").read_text(encoding="utf-8"))
    return {
        "campaign_id": rel,
        "research_scope": "eeg_task_validation",
        "comparison": comparison,
        "report_md": (candidate / "report.md").read_text(encoding="utf-8"),
    }


def _markdown(card: TaskCard, state: dict[str, Any], budget: ResearchBudget) -> str:
    decision = state.get("decision") or {}
    return "\n".join(
        [
            f"# EEG research {state['campaign_id']}",
            "",
            f"- task: `{card.task_id}`",
            f"- research_scope: `{card.research_scope}`",
            f"- protocol: {card.primary_metric} / {card.similarity} / candidates {card.candidate_count}",
            f"- stop: {state.get('stop_reason')}",
            f"- comparison: `{decision.get('comparison_status')}`",
            f"- selection: `{decision.get('selection_status')}`",
            f"- hypothesis: `{decision.get('hypothesis_status')}`",
            f"- delta: {decision.get('delta')}",
            "- test_result: null",
            "- model optimality was not assessed",
            "- single seed",
            "- auto-research efficiency was not assessed",
            "- a later hypothesis is not itself verified",
            f"- lm_calls: {budget.lm_calls}",
            f"- api_usd: {budget.api_usd}",
            "",
        ]
    )


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _jsonl(path: Path, rows: list[Any]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows if row is not None]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
