"""Turn evidence into one legal hypothesis. The runtime still admits the trial."""

from __future__ import annotations

from typing import Any, Protocol

from react_agent.eeg_research.schemas import EvidenceBundle, Hypothesis, TaskCard


class ResearchBackend(Protocol):
    """One structured completion. Implementations must not silently switch to rules."""

    name: str

    def complete(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return schema-shaped JSON for planner or reviewer."""


class OfflinePlanner:
    """Choose among approved profiles from the supplied evidence only."""

    name = "offline"

    def complete(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build observations that change when the metric evidence changes."""
        card: TaskCard = payload["task"]
        evidence: list[EvidenceBundle] = payload["evidence"]
        permitted: list[str] = payload["permitted_profiles"]
        observations = [
            {
                "evidence_id": row.experiment_id,
                "metric_name": row.metric_name,
                "primary_metric": row.primary_metric,
                "execution_status": row.execution_status,
            }
            for row in evidence
        ]
        if role == "reviewer":
            last = observations[-1] if observations else None
            return {
                "observations": observations,
                "hypothesis_assessment": "not_tested" if not evidence else "inconclusive",
                "alternative_explanations": ["optimization target differs from the retrieval metric"],
                "next_hypothesis_proposal": _hypothesis(
                    card,
                    observations,
                    permitted,
                    status="pending",
                ).model_dump(),
                "memory_summary": "store support and non-improvement separately",
                "limitations": ["single seed", "test_result is null"],
                "authoritative_metric": None if last is None else last["primary_metric"],
            }
        selected = _hypothesis(card, observations, permitted, status="selected")
        if selected.allowed_profile_id is None:
            return {
                "observations": observations,
                "hypotheses": [selected.model_dump()],
                "selected_hypothesis_id": selected.id,
                "experiment_proposal": {},
                "deferred_reasons": {},
                "stop_reason": "no_legal_profile",
            }
        deferred = [row for row in permitted if row != selected.allowed_profile_id]
        return {
            "observations": observations,
            "hypotheses": [selected.model_dump()],
            "selected_hypothesis_id": selected.id,
            "experiment_proposal": {
                "model_id": card.permitted_model_ids[0],
                "profile_id": selected.allowed_profile_id,
                "factor_changed": [] if not observations else ["weight_decay"],
                "seed": card.seed_schedule[0],
            },
            "deferred_reasons": {name: "kept in reserve until this comparison exists" for name in deferred},
            "stop_reason": None,
        }


def _hypothesis(
    card: TaskCard,
    observations: list[dict[str, Any]],
    permitted: list[str],
    *,
    status: str,
) -> Hypothesis:
    if not permitted:
        return Hypothesis(
            id="h-none",
            question="No approved profile is available.",
            observations=observations,
            explanation="no legal profile",
            alternative_explanations=["the task card permits nothing runnable"],
            proposed_change="none",
            expected_observable="no trial",
            disconfirmation_condition="a permitted profile appears",
            allowed_profile_id=None,
            estimated_cost=None,
            cost_estimate_source="unknown",
            priority_reason="no legal profile",
            status=status,  # type: ignore[arg-type]
        )
    if not observations:
        profile = permitted[0]
        ident = "h-cold"
        explanation = "cold-start prior; no campaign evidence yet"
        change = profile
    else:
        metric = observations[-1]["primary_metric"]
        profile = "profile_weight_decay" if "profile_weight_decay" in permitted else permitted[0]
        ident = f"h-metric-{metric}"
        explanation = f"validation top1 observed at {metric}"
        change = profile
    return Hypothesis(
        id=ident,
        question=f"Does an approved profile change {card.primary_metric}?",
        observations=observations,
        explanation=explanation,
        alternative_explanations=["the retrieval metric and the training loss need not move together"],
        proposed_change=change,
        expected_observable="validation top1 moves in the task direction",
        disconfirmation_condition="validation top1 does not improve against the incumbent",
        allowed_profile_id=profile,
        estimated_cost=None,
        cost_estimate_source="unknown",
        priority_reason="cold-start prior" if not observations else "evidence-linked comparison",
        status=status,  # type: ignore[arg-type]
    )


class FailingBackend:
    """Stand-in that raises, so a missing key cannot look like a planned success."""

    name = "deepseek-unconfigured"

    def complete(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Refuse the call."""
        raise RuntimeError("DEEPSEEK_API_KEY missing; refuse to fake an online call")
