"""Start one registered trial. Unknown GPU caps stay on dry-run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from react_agent.eeg_research.adapters.base import TrainingAdapter
from react_agent.eeg_research.schemas import EvidenceBundle, ExperimentSpec


class ExecutionRefused(RuntimeError):
    """Raised before a process starts."""


class ResearchBudget:
    """Count trials, API calls, and execution attempts. Dollars stay null."""

    def __init__(
        self,
        *,
        max_trials: int,
        max_lm_calls: int,
        max_execution_attempts: int,
        gpu_seconds: float | None,
        per_trial_timeout_s: float | None,
    ) -> None:
        self.max_trials = max_trials
        self.max_lm_calls = max_lm_calls
        self.max_execution_attempts = max_execution_attempts
        self.gpu_seconds = gpu_seconds
        self.per_trial_timeout_s = per_trial_timeout_s
        self.trials = 0
        self.lm_calls = 0
        self.repairs = 0
        self.execution_attempts = 0
        self.api_usd: float | None = None
        self.gpu_seconds_used = 0.0
        self.max_action_steps = 24
        self.action_steps = 0
        self.campaign_wall_seconds: float | None = None

    def charge_allocated_gpu(self, n_gpu: int, elapsed_s: float) -> float:
        """Charge assigned cards times elapsed time. This is not utilization."""
        if n_gpu < 1 or elapsed_s < 0:
            raise ExecutionRefused("bad_gpu_charge")
        cost = float(n_gpu) * float(elapsed_s)
        if self.gpu_seconds is not None and self.gpu_seconds_used + cost > self.gpu_seconds:
            raise ExecutionRefused("gpu_budget_exhausted")
        self.gpu_seconds_used += cost
        return cost

    def charge_lm(self, *, repair: bool = False) -> None:
        """Count one planner or reviewer call."""
        if self.lm_calls >= self.max_lm_calls:
            raise ExecutionRefused("lm_budget_exhausted")
        self.lm_calls += 1
        if repair:
            self.repairs += 1

    def charge_execution(self) -> None:
        """Count one launch, including a failed retry."""
        if self.execution_attempts >= self.max_execution_attempts:
            raise ExecutionRefused("execution_attempts_exhausted")
        self.execution_attempts += 1

    def as_dict(self) -> dict[str, object]:
        """Serialize the ledger. api_usd is null unless a price was supplied."""
        return {
            "trials": self.trials,
            "max_trials": self.max_trials,
            "lm_calls": self.lm_calls,
            "max_lm_calls": self.max_lm_calls,
            "repairs": self.repairs,
            "execution_attempts": self.execution_attempts,
            "max_execution_attempts": self.max_execution_attempts,
            "gpu_seconds_cap": self.gpu_seconds,
            "gpu_seconds_used": self.gpu_seconds_used,
            "api_usd": self.api_usd,
            "api_usd_source": "unknown" if self.api_usd is None else "supplied",
        }


def run_trial(
    adapter: TrainingAdapter,
    spec: ExperimentSpec,
    output_dir: str,
    budget: ResearchBudget,
    *,
    dry_run: bool,
) -> EvidenceBundle:
    """Run the adapter or refuse when the cap is unknown."""
    if adapter.availability == "unavailable":
        raise ExecutionRefused(adapter.reason)
    if adapter.requires_gpu_cap and budget.gpu_seconds is None and not dry_run:
        raise ExecutionRefused("gpu cap unknown; dry-run only")
    if dry_run:
        return EvidenceBundle(
            experiment_id=spec.id,
            execution_status="cancelled",
            protocol_status="valid",
            primary_metric=None,
            evaluator="dry_run",
            missing_fields=["primary_metric"],
            error="dry_run",
            seed=spec.seed,
            fidelity=spec.fidelity,
        )
    budget.charge_execution()
    bundle = adapter.run(spec, output_dir)
    bundle.candidate_bank_hash = bundle.candidate_bank_hash or spec.resource_limits.get(
        "candidate_bank_hash", ""
    )
    bundle.target_feature_hash = bundle.target_feature_hash or spec.resource_limits.get(
        "target_feature_hash", ""
    )
    return bundle


def run_timeout_probe(timeout_s: float) -> EvidenceBundle:
    """Kill only the sleep process started for this probe."""
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=timeout_s)
        status = "succeeded"
        error = None
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        status = "timed_out"
        error = "per_trial_timeout"
    return EvidenceBundle(
        experiment_id="timeout-probe",
        execution_status=status,  # type: ignore[arg-type]
        protocol_status="invalid" if status == "timed_out" else "valid",
        evaluator="timeout_probe",
        primary_metric=None,
        error=error,
        missing_fields=["primary_metric"],
    )


def prepare_output(path: Path) -> Path:
    """Create a new trial directory. Existing completed metrics are left in place."""
    path.mkdir(parents=True, exist_ok=True)
    return path
