"""Generate if needed, then the existing check loop. Graph topology is unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.fmri.budget import BudgetError, BudgetLedger
from react_agent.fmri.config import FmriCheckConfig, planned_preflight, redacted_config
from react_agent.fmri.generation.schemas import (
    FmriRequest,
    GeneratedFmriBundle,
    GenerationError,
    ImageRequest,
    TemporalContractError,
)
from react_agent.fmri.generation.tool import TribeGenerateTool
from react_agent.fmri.loop import run_sample
from react_agent.fmri.progress import Progress, bind_progress, emit
from react_agent.fmri.schemas import SampleSpec


def bind_generated_control(config: FmriCheckConfig, bundle: GeneratedFmriBundle) -> FmriCheckConfig:
    """Point gray_control at the matching 16s control, not the old 12-point file."""
    cfg = config.model_copy(deep=True)
    if bundle.control_preds_path:
        cfg.resources.gray_control_path = bundle.control_preds_path
        cfg.resources.gray_control_id = bundle.control_sample_id
    return cfg


def materialize(
    request: ImageRequest | FmriRequest,
    config: FmriCheckConfig,
    *,
    budget: BudgetLedger | None = None,
    backend_override: str | None = None,
) -> GeneratedFmriBundle:
    """Return a SampleSpec bundle. Cache hits do not count as new predictions."""
    if isinstance(request, FmriRequest):
        return GeneratedFmriBundle(
            sample_id=request.sample.sample_id,
            profile_id=request.sample.generation_profile_id or "",
            sample=request.sample,
            preds_path=request.sample.fmri_path,
            sidecar_path=request.sample.metadata_path or request.sample.fmri_path,
            video_path="",
            control_preds_path=request.control_path,
            preds_shape=tuple(request.sample.expected_shape or (16, 20484)),
            temporal_padding_applied=False,
            cache_hit_image=True,
            cache_hit_control=True,
            new_tribe_predictions=0,
            synthetic=False,
            backend="worker",
        )
    cfg = config.model_copy(deep=True)
    if backend_override:
        cfg.generation.backend = backend_override  # type: ignore[assignment]
    tool = TribeGenerateTool(cfg, budget=budget)
    try:
        return tool.run(request)
    finally:
        tool.close()


def generation_failed_report(
    sample_id: str,
    error: Exception,
    *,
    budget: BudgetLedger | None = None,
) -> dict[str, Any]:
    """Check is not run when generation fails."""
    code = getattr(error, "code", type(error).__name__)
    return {
        "sample_id": sample_id,
        "pipeline_status": "generation_failed",
        "check_verdict": None,
        "verdict": "inconclusive",
        "coverage_status": "partial",
        "stop_reason": f"generation_failed:{code}",
        "executed_tools": [],
        "fallback_used": False,
        "claim_scope": "numeric_consistency_only",
        "error": str(error),
        "cost": budget.snapshot() if budget else {},
    }


async def run_image_pipeline(
    image_path: str | Path,
    *,
    config: FmriCheckConfig,
    out_dir: Path,
    sample_id: str | None = None,
    backend: str = "none",
    policy: str = "rule",
    generation_backend: str | None = None,
    require_checks: list[str] | None = None,
    mock: Any = None,
) -> dict[str, Any]:
    """Materialize then call existing run_sample."""
    cfg = config.model_copy(deep=True)
    if require_checks:
        merged = list(cfg.required_checks)
        for name in require_checks:
            if name not in merged:
                merged.append(name)
        cfg.required_checks = merged
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if policy == "planned":
        preview = planned_preflight(cfg, backend=backend, policy=policy)
        print(
            "[preflight] "
            + json.dumps(preview, default=str),
            flush=True,
        )
    (out_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **redacted_config(cfg),
                "resolved_policy": policy,
                "resolved_backend": backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    budget = BudgetLedger(cfg.budgets)
    request = ImageRequest(image_path=str(Path(image_path).resolve()), sample_id=sample_id)
    events_path = out_dir / "events.jsonl"
    with bind_progress(Progress(events_path)):
        try:
            bundle = materialize(request, cfg, budget=budget, backend_override=generation_backend)
        except (GenerationError, TemporalContractError, BudgetError, Exception) as exc:
            report = generation_failed_report(sample_id or Path(image_path).stem, exc, budget=budget)
            emit(
                {
                    "type": "generate",
                    "message": f"failed {report.get('stop_reason')}",
                    "status": "error",
                },
                path=events_path,
            )
            (out_dir / "pipeline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        bound = bind_generated_control(cfg, bundle)
        spec: SampleSpec = bundle.sample
        check = await run_sample(
            spec,
            base_dir=out_dir,
            out_dir=out_dir,
            config=bound,
            backend=backend,
            policy=policy,
            mock=mock,
            batch_ledger=budget.batch,
        )
    wrapper = {
        "pipeline_status": "ok",
        "check_verdict": check.get("verdict"),
        "generation": bundle.summary(),
        **check,
    }
    wrapper["cost"] = {
        **(check.get("cost") or {}),
        "generation_requests": budget.data["generation_requests"],
        "new_tribe_predictions": budget.data["new_tribe_predictions"],
    }
    (out_dir / "pipeline.json").write_text(json.dumps(wrapper, indent=2, default=str), encoding="utf-8")
    return wrapper
