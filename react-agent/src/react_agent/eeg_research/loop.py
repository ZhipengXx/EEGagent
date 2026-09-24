"""One campaign: preflight, one admitted trial, then a pending next hypothesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.evidence import bind_protocol
from react_agent.eeg_research.executor import ExecutionRefused, ResearchBudget, run_trial
from react_agent.eeg_research.memory import ResearchMemory
from react_agent.eeg_research.planner import ResearchBackend
from react_agent.eeg_research.registry import Registry, RegistryError
from react_agent.eeg_research.report import write_campaign
from react_agent.eeg_research.schemas import (
    EvidenceBundle,
    ExperimentSpec,
    Hypothesis,
    TaskCard,
)
from react_agent.eeg_research.task_contract import freeze_task, validate_splits
from react_agent.eeg_research.validation import GateError, admit, compare


def run_campaign(
    card: TaskCard,
    registry: Registry,
    backend: ResearchBackend,
    budget: ResearchBudget,
    out_dir: Path,
    *,
    campaign_id: str,
    resume: bool = False,
    dry_run: bool = False,
    memory: ResearchMemory | None = None,
) -> dict[str, Any]:
    """Execute the bounded loop and write campaign files."""
    card = freeze_task(card)
    validate_splits(card)
    registry.probe_all()
    state = _load(out_dir) if resume and out_dir.exists() else _empty(campaign_id)
    if state["completed_ids"]:
        state["stop_reason"] = "resume_skipped_completed"
        write_campaign(out_dir, card, registry, state, budget)
        return state
    try:
        budget.charge_lm()
        planned = backend.complete(
            "planner",
            {
                "task": card,
                "evidence": state["evidence"],
                "permitted_profiles": card.permitted_profile_ids,
            },
        )
    except RuntimeError as exc:
        state["stop_reason"] = str(exc)
        write_campaign(out_dir, card, registry, state, budget)
        return state
    if "selected_hypothesis_id" not in planned and budget.repairs < 1:
        budget.charge_lm(repair=True)
        planned = backend.complete(
            "planner",
            {
                "task": card,
                "evidence": state["evidence"],
                "permitted_profiles": card.permitted_profile_ids,
            },
        )
    hypothesis = Hypothesis.model_validate(planned["hypotheses"][0])
    state["hypotheses"].append(hypothesis.model_dump())
    proposal = planned.get("experiment_proposal") or {}
    if budget.trials >= budget.max_trials or planned.get("stop_reason"):
        hypothesis.status = "pending"
        state["stop_reason"] = planned.get("stop_reason") or "trial_budget_exhausted"
        write_campaign(out_dir, card, registry, state, budget)
        return state
    spec = ExperimentSpec(
        id=f"{campaign_id}-t{budget.trials + 1}",
        campaign_id=campaign_id,
        hypothesis_id=hypothesis.id,
        model_id=str(proposal.get("model_id") or card.permitted_model_ids[0]),
        profile_id=str(proposal.get("profile_id") or hypothesis.allowed_profile_id),
        frozen_task_hash=card.frozen_task_hash,
        seed=int(proposal.get("seed") or card.seed_schedule[0]),
        fidelity=card.training_fidelity,
        resolved_config_hash=card.frozen_task_hash,
        factor_changed=list(proposal.get("factor_changed") or []),
        output_dir=str(out_dir / "trials" / f"t{budget.trials + 1}"),
        resource_limits={
            "candidate_bank_hash": card.candidate_bank_hash,
            "target_feature_hash": card.target_feature_hash,
        },
    )
    try:
        _, profile, adapter = registry.require(spec.model_id, spec.profile_id)
        spec.fingerprint = admit(
            spec,
            card,
            profile,
            seen_fingerprints=set(state["fingerprints"]),
            trials_used=budget.trials,
            max_trials=budget.max_trials,
        )
        bundle = bind_protocol(
            run_trial(adapter, spec, spec.output_dir, budget, dry_run=dry_run),
            card,
        )
    except (RegistryError, GateError, ExecutionRefused, KeyError, RuntimeError) as exc:
        state["stop_reason"] = str(exc)
        state["experiments"].append({"spec": spec.model_dump(), "error": str(exc)})
        write_campaign(out_dir, card, registry, state, budget)
        return state
    budget.trials += 1
    incumbent = state["incumbent"]
    decision = compare(card, bundle, incumbent)
    if decision.selection_status in {"not_compared", "provisional_incumbent"} and decision.comparison_status in {
        "baseline_established",
        "comparable",
    }:
        if decision.comparison_status == "baseline_established" or decision.selection_status == "provisional_incumbent":
            if decision.incumbent_id == bundle.experiment_id:
                state["incumbent"] = bundle
    elif decision.selection_status == "rejected":
        pass
    state["evidence"].append(bundle)
    state["fingerprints"].append(spec.fingerprint)
    state["experiments"].append(
        {"spec": spec.model_dump(), "evidence": bundle.model_dump(), "decision": decision.model_dump()}
    )
    state["completed_ids"].append(spec.id)
    state["decision"] = decision.model_dump()
    try:
        budget.charge_lm()
        review = backend.complete(
            "reviewer",
            {"task": card, "evidence": state["evidence"], "permitted_profiles": card.permitted_profile_ids},
        )
    except RuntimeError as exc:
        review = {"limitations": [str(exc)]}
    review["executed"] = False
    state["review"] = review
    state["hypotheses"].append(review.get("next_hypothesis_proposal"))
    if memory is not None:
        kind = "non_improvement" if decision.selection_status == "rejected" else "observation"
        memory.add(
            task_hash=card.frozen_task_hash,
            kind=kind,
            fingerprint=spec.fingerprint,
            payload={"decision": decision.model_dump(), "metric": bundle.primary_metric},
            evidence_level=decision.evidence_level,
        )
    if budget.trials >= budget.max_trials:
        state["stop_reason"] = "trial_budget_exhausted"
    else:
        state["stop_reason"] = "trial_recorded"
    write_campaign(out_dir, card, registry, state, budget)
    return state


def _empty(campaign_id: str) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "hypotheses": [],
        "experiments": [],
        "evidence": [],
        "fingerprints": [],
        "completed_ids": [],
        "incumbent": None,
        "decision": None,
        "review": None,
        "stop_reason": None,
    }


def _load(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "experiments.jsonl"
    state = _empty(out_dir.name)
    if not path.is_file():
        return state
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state["experiments"].append(row)
        if row.get("evidence"):
            bundle = EvidenceBundle.model_validate(row["evidence"])
            state["evidence"].append(bundle)
            state["completed_ids"].append(row["spec"]["id"])
            state["fingerprints"].append(row["spec"].get("fingerprint") or "")
            decision = row.get("decision") or {}
            if decision.get("incumbent_id") == bundle.experiment_id and decision.get("selection_status") != "rejected":
                state["incumbent"] = bundle
    return state
