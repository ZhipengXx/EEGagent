"""In-process retrieval stand-in. Scores are fixtures, not measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec


class MockRetrievalAdapter:
    """Return a predeclared validation score for an approved profile."""

    adapter_id = "mock_retrieval"
    availability = "unverified"
    reason = "mock fixture; not a probed training entry"
    requires_gpu_cap = False

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.runs = 0

    def probe(self) -> dict[str, Any]:
        """Report that this adapter is only a contract fixture."""
        return {"ok": True, "availability": self.availability, "reason": self.reason}

    def run(self, spec: ExperimentSpec, output_dir: str) -> EvidenceBundle:
        """Write the fixture metric. Missing profile scores fail closed."""
        if spec.profile_id not in self.scores:
            raise KeyError(f"mock score missing for {spec.profile_id}")
        self.runs += 1
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metric = float(self.scores[spec.profile_id])
        payload = {
            "primary_metric": metric,
            "metric_name": "top1",
            "evaluator": "mock_fixed",
            "split": "validation",
            "test_result": None,
        }
        (out / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return EvidenceBundle(
            experiment_id=spec.id,
            execution_status="succeeded",
            protocol_status="valid",
            primary_metric=metric,
            evaluator="mock_fixed",
            changed_fields=list(spec.factor_changed),
            source_artifact_refs=[str(out / "metrics.json")],
            wall_seconds=0.0,
            candidate_bank_hash="",
            target_feature_hash="",
            fidelity=spec.fidelity,
            seed=spec.seed,
        )
