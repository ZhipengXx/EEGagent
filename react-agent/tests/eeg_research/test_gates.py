"""Gates for splits, comparison, budget, and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.adapters.mock_retrieval import MockRetrievalAdapter
from react_agent.eeg_research.adapters.unavailable import UnavailableRetrievalAdapter
from react_agent.eeg_research.executor import ExecutionRefused, ResearchBudget, run_timeout_probe
from react_agent.eeg_research.loop import run_campaign
from react_agent.eeg_research.memory import ResearchMemory
from react_agent.eeg_research.planner import FailingBackend, OfflinePlanner
from react_agent.eeg_research.registry import Registry, RegistryError
from react_agent.eeg_research.report import read_campaign
from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec, ModelRecord, ProfileRecord, TaskCard
from react_agent.eeg_research.task_contract import ContractError, freeze_task, validate_splits
from react_agent.eeg_research.validation import GateError, admit, compare


def _card(**updates: object) -> TaskCard:
    raw = dict(
        task_id="toy",
        dataset_id="fixture",
        dataset_version="v0",
        dataset_manifest_hash="m",
        train_ids=["a", "b"],
        validation_ids=["c"],
        test_ids=["d"],
        preprocessing_hash="p",
        metric_implementation_hash="mock_fixed",
        target_feature_hash="features-v1",
        candidate_bank_hash="bank-v1",
        candidate_count=3,
        similarity="cosine",
        trial_aggregation="mean",
        query_unit="image",
        permitted_model_ids=["mock_eeg_encoder"],
        permitted_profile_ids=["profile_baseline", "profile_weight_decay"],
    )
    raw.update(updates)
    return freeze_task(TaskCard.model_validate(raw))


def _registry(adapter: MockRetrievalAdapter | None = None) -> Registry:
    models = [
        ModelRecord(
            model_id="mock_eeg_encoder",
            adapter_id="mock_retrieval",
            supported_tasks=["eeg_image_retrieval"],
            input_contract="fixture",
            code_revision="unverified",
            dependency_environment="react-agent",
            approved_profiles=["profile_baseline", "profile_weight_decay"],
            availability="unverified",
            reason="mock",
        ),
        ModelRecord(
            model_id="eeg_image_retrieval",
            adapter_id="unavailable_retrieval",
            supported_tasks=["eeg_image_retrieval"],
            input_contract="unknown",
            code_revision="missing",
            dependency_environment="unknown",
            approved_profiles=[],
            availability="unavailable",
            reason="missing training entry",
        ),
    ]
    profiles = [
        ProfileRecord(profile_id="profile_baseline", allowed_changes=[]),
        ProfileRecord(profile_id="profile_weight_decay", allowed_changes=["weight_decay"]),
    ]
    return Registry(
        models,
        profiles,
        {
            "mock_retrieval": adapter or MockRetrievalAdapter({"profile_baseline": 0.2, "profile_weight_decay": 0.1}),
            "unavailable_retrieval": UnavailableRetrievalAdapter(),
        },
    )


def _budget(**updates: object) -> ResearchBudget:
    raw = dict(max_trials=1, max_lm_calls=4, max_execution_attempts=2, gpu_seconds=None, per_trial_timeout_s=1)
    raw.update(updates)
    return ResearchBudget(**raw)  # type: ignore[arg-type]


def _bundle(experiment_id: str, metric: float | None, **updates: object) -> EvidenceBundle:
    card = _card()
    payload = dict(
        experiment_id=experiment_id,
        execution_status="succeeded",
        protocol_status="valid",
        primary_metric=metric,
        evaluator="mock_fixed",
        candidate_bank_hash=card.candidate_bank_hash,
        target_feature_hash=card.target_feature_hash,
        fidelity="full",
        seed=0,
    )
    payload.update(updates)
    return EvidenceBundle.model_validate(payload)


def test_overlap_and_test_tuning_are_refused() -> None:
    with pytest.raises(ContractError, match="missing_split"):
        validate_splits(_card(train_ids=[]))
    with pytest.raises(ContractError, match="train_validation_overlap"):
        validate_splits(_card(train_ids=["a"], validation_ids=["a"]))
    with pytest.raises(ContractError, match="test_split_overlap"):
        validate_splits(_card(validation_ids=["d"], test_ids=["d"]))
    with pytest.raises(ContractError, match="test_used_for_tuning"):
        validate_splits(_card(uses_test_for_tuning=True))


def test_missing_candidate_does_not_train(tmp_path: Path) -> None:
    card = _card(permitted_model_ids=["eeg_image_retrieval"], permitted_profile_ids=[])
    adapter = MockRetrievalAdapter({"profile_baseline": 0.2})
    state = run_campaign(
        card,
        _registry(adapter),
        OfflinePlanner(),
        _budget(),
        tmp_path,
        campaign_id="none",
    )
    assert state["stop_reason"]
    assert not list(tmp_path.glob("trials/**/metrics.json"))


def test_incomparable_protocol_has_null_delta() -> None:
    card = _card()
    incumbent = _bundle("base", 0.2)
    other = _bundle("next", 0.9, candidate_bank_hash="other-bank")
    decision = compare(card, other, incumbent)
    assert decision.comparison_status == "incomparable"
    assert decision.delta is None
    features = compare(card, _bundle("feat", 0.9, target_feature_hash="other-features"), incumbent)
    assert features.comparison_status == "incomparable"
    assert features.delta is None


def test_first_trial_is_baseline_not_improvement(tmp_path: Path) -> None:
    state = run_campaign(_card(), _registry(), OfflinePlanner(), _budget(), tmp_path, campaign_id="base")
    decision = state["decision"]
    assert decision["comparison_status"] == "baseline_established"
    assert decision["delta"] is None
    assert decision["model_selection_assessed"] is False
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "test_result: null" in report
    assert "single seed" in report
    cost = json.loads((tmp_path / "cost.json").read_text(encoding="utf-8"))
    assert cost["api_usd"] is None


def test_worse_score_keeps_incumbent(tmp_path: Path) -> None:
    card = _card()
    incumbent = _bundle("base", 0.4)
    worse = _bundle("next", 0.1)
    decision = compare(card, worse, incumbent)
    assert decision.selection_status == "rejected"
    assert decision.incumbent_id == "base"
    memory = ResearchMemory(tmp_path / "memory.sqlite3")
    memory.add(
        task_hash=card.frozen_task_hash,
        kind="non_improvement",
        fingerprint="fp",
        payload={"metric": 0.1},
        evidence_level=decision.evidence_level,
    )
    kinds = [row["kind"] for row in memory.matching(card.frozen_task_hash)]
    assert "non_improvement" in kinds
    memory.close()


def test_single_seed_stays_provisional_and_replay_does_not_upgrade() -> None:
    card = _card()
    decision = compare(card, _bundle("better", 0.5), _bundle("base", 0.2))
    assert decision.selection_status == "provisional_incumbent"
    assert decision.hypothesis_status == "supported_provisionally"
    assert decision.evidence_level == "observed"
    replay = compare(card, _bundle("replay", 0.5, cached_replay=True), _bundle("base", 0.2))
    assert replay.evidence_level == "proposed"


def test_unknown_profile_and_duplicate_are_rejected() -> None:
    card = _card()
    profile = ProfileRecord(profile_id="profile_baseline", allowed_changes=[])
    spec = ExperimentSpec(
        id="t",
        campaign_id="c",
        hypothesis_id="h",
        model_id="missing",
        profile_id="profile_baseline",
        frozen_task_hash=card.frozen_task_hash,
        seed=0,
        resolved_config_hash="x",
        output_dir="out",
    )
    with pytest.raises(GateError, match="model_not_permitted"):
        admit(spec, card, profile, seen_fingerprints=set(), trials_used=0, max_trials=1)
    spec = spec.model_copy(update={"model_id": "mock_eeg_encoder", "factor_changed": ["learning_rate"]})
    with pytest.raises(GateError, match="change_not_allowed"):
        admit(spec, card, profile, seen_fingerprints=set(), trials_used=0, max_trials=1)
    legal = spec.model_copy(update={"factor_changed": []})
    fingerprint = admit(legal, card, profile, seen_fingerprints=set(), trials_used=0, max_trials=1)
    with pytest.raises(GateError, match="duplicate_experiment"):
        admit(legal, card, profile, seen_fingerprints={fingerprint}, trials_used=0, max_trials=1)
    with pytest.raises(RegistryError):
        _registry().require("eeg_image_retrieval", "profile_baseline")


def test_budget_counts_repair_and_does_not_run_pending(tmp_path: Path) -> None:
    class RepairBackend(OfflinePlanner):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, role: str, payload: dict) -> dict:
            self.calls += 1
            if role == "planner" and self.calls == 1:
                return {"observations": []}
            return super().complete(role, payload)

    backend = RepairBackend()
    budget = _budget(max_trials=0, max_lm_calls=4)
    adapter = MockRetrievalAdapter({"profile_baseline": 0.2})
    state = run_campaign(_card(), _registry(adapter), backend, budget, tmp_path, campaign_id="budget")
    assert budget.repairs == 1
    assert budget.lm_calls >= 1
    counted = _budget(max_execution_attempts=1)
    counted.charge_execution()
    with pytest.raises(ExecutionRefused, match="execution_attempts_exhausted"):
        counted.charge_execution()
    assert state["experiments"] == []
    assert "metrics.json" not in list(p.name for p in tmp_path.rglob("*"))


def test_evidence_changes_hypothesis_identity() -> None:
    planner = OfflinePlanner()
    card = _card()
    empty = planner.complete("planner", {"task": card, "evidence": [], "permitted_profiles": card.permitted_profile_ids})
    filled = planner.complete(
        "planner",
        {
            "task": card,
            "evidence": [_bundle("e1", 0.33)],
            "permitted_profiles": card.permitted_profile_ids,
        },
    )
    assert empty["selected_hypothesis_id"] == "h-cold"
    assert filled["observations"][0]["evidence_id"] == "e1"
    assert filled["observations"][0]["primary_metric"] == 0.33
    assert filled["selected_hypothesis_id"] == "h-metric-0.33"


def test_resume_does_not_retrain_and_timeout_is_local(tmp_path: Path) -> None:
    adapter = MockRetrievalAdapter({"profile_baseline": 0.2, "profile_weight_decay": 0.1})
    registry = _registry(adapter)
    run_campaign(_card(), registry, OfflinePlanner(), _budget(), tmp_path, campaign_id="again")
    state = run_campaign(
        _card(),
        registry,
        OfflinePlanner(),
        _budget(),
        tmp_path,
        campaign_id="again",
        resume=True,
    )
    assert state["stop_reason"] == "resume_skipped_completed"
    assert adapter.runs == 1
    assert len(state["completed_ids"]) == 1
    timed = run_timeout_probe(0.05)
    assert timed.execution_status == "timed_out"
    assert json.loads((tmp_path / "cost.json").read_text(encoding="utf-8"))["api_usd"] is None


def test_campaign_reader_stays_outside_fmri_runs(tmp_path: Path) -> None:
    run_campaign(_card(), _registry(), OfflinePlanner(), _budget(), tmp_path / "camp", campaign_id="camp")
    payload = read_campaign(tmp_path, "camp")
    assert payload is not None
    assert payload["research_scope"] == "eeg_task_validation"
    assert read_campaign(tmp_path, "../secret") is None


def test_failing_backend_does_not_look_like_success(tmp_path: Path) -> None:
    state = run_campaign(_card(), _registry(), FailingBackend(), _budget(), tmp_path, campaign_id="api")
    assert "DEEPSEEK_API_KEY" in state["stop_reason"]
    assert state["experiments"] == []
