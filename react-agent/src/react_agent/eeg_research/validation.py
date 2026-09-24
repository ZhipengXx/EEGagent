"""Hard filters and a deterministic metric comparison."""

from __future__ import annotations

import hashlib
import json

from react_agent.eeg_research.schemas import (
    EvidenceBundle,
    ExperimentDecision,
    ExperimentSpec,
    ProfileRecord,
    TaskCard,
)
from react_agent.eeg_research.task_contract import protocol_fingerprint


class GateError(ValueError):
    """Raised when a proposed trial must not run."""


def experiment_fingerprint(spec: ExperimentSpec, card: TaskCard) -> str:
    """Hash the comparison identity of one trial."""
    payload = {
        "model_id": spec.model_id,
        "profile_id": spec.profile_id,
        "seed": spec.seed,
        "fidelity": spec.fidelity,
        "frozen_task_hash": spec.frozen_task_hash,
        "factor_changed": spec.factor_changed,
        "protocol": protocol_fingerprint(card),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def admit(
    spec: ExperimentSpec,
    card: TaskCard,
    profile: ProfileRecord,
    *,
    seen_fingerprints: set[str],
    trials_used: int,
    max_trials: int,
) -> str:
    """Return the fingerprint or refuse the trial."""
    if spec.frozen_task_hash != card.frozen_task_hash:
        raise GateError("task_hash_mismatch")
    if spec.model_id not in card.permitted_model_ids:
        raise GateError("model_not_permitted")
    if spec.profile_id not in card.permitted_profile_ids:
        raise GateError("profile_not_permitted")
    if spec.fidelity != card.training_fidelity or profile.fidelity != card.training_fidelity:
        raise GateError("fidelity_mismatch")
    extra = [name for name in spec.factor_changed if name not in profile.allowed_changes]
    if extra:
        raise GateError("change_not_allowed")
    if trials_used >= max_trials:
        raise GateError("trial_budget_exhausted")
    fingerprint = experiment_fingerprint(spec, card)
    if fingerprint in seen_fingerprints:
        raise GateError("duplicate_experiment")
    return fingerprint


def _compatible(card: TaskCard, bundle: EvidenceBundle) -> bool:
    return (
        bundle.protocol_status == "valid"
        and bundle.fidelity == card.training_fidelity
        and bundle.candidate_bank_hash == card.candidate_bank_hash
        and bundle.target_feature_hash == card.target_feature_hash
        and bundle.primary_metric is not None
    )


def compare(
    card: TaskCard,
    bundle: EvidenceBundle,
    incumbent: EvidenceBundle | None,
) -> ExperimentDecision:
    """Compare validation scores. The first success only establishes a baseline."""
    limits = [
        "model optimality was not assessed",
        "single seed",
        "auto-research efficiency was not assessed",
    ]
    if bundle.execution_status != "succeeded" or bundle.protocol_status != "valid":
        return ExperimentDecision(
            execution_status=bundle.execution_status,
            comparison_status="invalid" if bundle.protocol_status != "valid" else "incomparable",
            selection_status="not_compared",
            hypothesis_status="not_tested",
            incumbent_id=None if incumbent is None else incumbent.experiment_id,
            evidence_level="observed",
            model_selection_assessed=False,
            limitations=limits,
        )
    if not _compatible(card, bundle):
        return ExperimentDecision(
            execution_status="succeeded",
            comparison_status="incomparable",
            selection_status="not_compared",
            hypothesis_status="inconclusive",
            delta=None,
            incumbent_id=None if incumbent is None else incumbent.experiment_id,
            model_selection_assessed=False,
            limitations=limits + ["protocol fingerprint differs"],
        )
    if incumbent is None or not _compatible(card, incumbent):
        return ExperimentDecision(
            execution_status="succeeded",
            comparison_status="baseline_established",
            selection_status="not_compared",
            hypothesis_status="not_tested",
            delta=None,
            incumbent_id=bundle.experiment_id,
            evidence_level="proposed" if bundle.cached_replay else "observed",
            model_selection_assessed=False,
            limitations=limits + ["no baseline; delta is null"],
        )
    assert incumbent.primary_metric is not None and bundle.primary_metric is not None
    delta = bundle.primary_metric - incumbent.primary_metric
    better = delta > 0 if card.direction == "higher" else delta < 0
    if card.minimum_delta is not None and abs(delta) < card.minimum_delta:
        better = False
    if delta == 0 or not better:
        status = "retained" if delta == 0 or not better else "rejected"
        if delta < 0 and card.direction == "higher" or delta > 0 and card.direction == "lower":
            status = "rejected"
            hypothesis = "contradicted"
        else:
            status = "retained"
            hypothesis = "inconclusive"
        level = "proposed" if bundle.cached_replay else "observed"
        return ExperimentDecision(
            execution_status="succeeded",
            comparison_status="comparable",
            selection_status=status,  # type: ignore[arg-type]
            hypothesis_status=hypothesis,  # type: ignore[arg-type]
            delta=delta,
            incumbent_id=incumbent.experiment_id,
            evidence_level=level,  # type: ignore[arg-type]
            model_selection_assessed=False,
            limitations=limits,
        )
    level = "proposed" if bundle.cached_replay else "observed"
    return ExperimentDecision(
        execution_status="succeeded",
        comparison_status="comparable",
        selection_status="provisional_incumbent",
        hypothesis_status="supported_provisionally",
        delta=delta,
        incumbent_id=bundle.experiment_id,
        evidence_level=level,  # type: ignore[arg-type]
        model_selection_assessed=False,
        limitations=limits + ["single seed remains provisional"],
    )
