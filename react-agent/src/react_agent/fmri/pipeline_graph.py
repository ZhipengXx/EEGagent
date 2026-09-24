"""Thin LangGraph wrapper around pipeline + the existing check loop.

Does not change ``fmri_check`` / ``graph.py`` topology.
Studio sees apply_defaults -> materialize -> run_check, plus custom progress lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from react_agent.fmri.config import load_config, planned_preflight, resolve_policy_backend
from react_agent.fmri.generation.schemas import ImageRequest
from react_agent.fmri.loop import run_sample
from react_agent.fmri.pipeline import generation_failed_report, materialize
from react_agent.fmri.progress import Progress, bind_progress, emit
from react_agent.fmri.schemas import SampleSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_IMAGE_PATH = (
    "/home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/"
    "Image_set/training_images/00029_apple/apple_01b.jpg"
)
DEFAULT_CONFIG_PATH = str(_REPO_ROOT / "configs" / "fmri_check_tribe_image16.yaml")
DEFAULT_OUT_DIR = str(_REPO_ROOT / "runs" / "tribe_image16_studio_view")

DEFAULT_STUDIO_INPUT: dict[str, str] = {
    "image_path": DEFAULT_IMAGE_PATH,
    "sample_id": "apple_01b",
    "config_path": DEFAULT_CONFIG_PATH,
    "out_dir": DEFAULT_OUT_DIR,
    "backend": "none",
    "policy": "rule",
}


class PipelineState(TypedDict, total=False):
    """Studio-facing image pipeline state."""

    image_path: str
    sample_id: str
    config_path: str
    out_dir: str
    backend: str
    policy: str
    generation_backend: str
    sample_spec: dict[str, Any]
    generation: dict[str, Any]
    result: dict[str, Any]
    studio_summary: dict[str, Any]
    last_event: str
    pipeline_status: str


def apply_studio_defaults(state: PipelineState) -> PipelineState:
    """Fill missing Studio form fields. Does not invent scientific values."""
    explicit_policy = bool(state.get("policy"))
    explicit_backend = bool(state.get("backend"))
    explicit_config = bool(state.get("config_path"))
    merged: PipelineState = dict(DEFAULT_STUDIO_INPUT)
    for key, value in state.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value  # type: ignore[literal-required]
    if explicit_config and not explicit_policy:
        cfg = load_config(merged.get("config_path"))
        policy, backend = resolve_policy_backend(
            cfg,
            policy=None,
            backend=state.get("backend") if explicit_backend else None,
        )
        merged["policy"] = policy
        merged["backend"] = backend
    return merged


def studio_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Short overlay for Studio. Never includes the [T, V] array."""
    metrics = result.get("metrics_table") or {}
    basic = metrics.get("basic_statistics") or {}
    mae = metrics.get("cortex_mae") or {}
    gray = metrics.get("gray_control_contrast") or {}
    gen = result.get("generation") or {}
    workflow = result.get("workflow") or {}
    time_n = basic.get("T")
    space_n = basic.get("V")
    if time_n is None or space_n is None:
        shape = gen.get("preds_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            time_n, space_n = shape[0], shape[1]
    return {
        "sample_id": result.get("sample_id"),
        "pipeline_status": result.get("pipeline_status"),
        "check_verdict": result.get("check_verdict") or result.get("verdict"),
        "coverage_status": result.get("coverage_status"),
        "claim_scope": result.get("claim_scope") or "numeric_consistency_only",
        "shape_tv": [time_n, space_n],
        "overall_delta_rms": gray.get("overall_delta_rms"),
        "gray_comparable": gray.get("comparable"),
        "cortex_mae": {
            "mode": mae.get("mode"),
            "input_t": mae.get("input_t"),
            "trained_t": mae.get("trained_t"),
            "padded": mae.get("padded"),
            "encoder_forward_validated": mae.get("encoder_forward_validated"),
            "embedding_available": mae.get("embedding_available"),
            "reference_score_assessed": mae.get("reference_score_assessed"),
        },
        "generation": {
            "cache_hit_image": gen.get("cache_hit_image"),
            "cache_hit_control": gen.get("cache_hit_control"),
            "new_tribe_predictions": gen.get("new_tribe_predictions"),
            "preds_shape": gen.get("preds_shape"),
            "synthetic": gen.get("synthetic"),
            "temporal_padding_applied": gen.get("temporal_padding_applied"),
        },
        "workflow_html": workflow.get("html_path"),
        "brain_tstrip": workflow.get("brain_tstrip"),
        "metric_scores": workflow.get("metric_scores") or {},
        "studio_note": (
            "脑图和合同色条在 workflow.html。"
            "这里只保留路径和 0–1 分数，不放预测数组。"
        ),
    }


def apply_defaults_node(state: PipelineState) -> dict[str, Any]:
    """Fill the Studio form and announce the resolved sample."""
    resolved = apply_studio_defaults(state)
    line = emit(
        {
            "type": "generate",
            "message": f"studio input sample={resolved.get('sample_id')}",
        }
    )
    return {
        **{k: resolved[k] for k in DEFAULT_STUDIO_INPUT},
        "generation_backend": resolved.get("generation_backend"),
        "last_event": line,
        "pipeline_status": "ready",
    }


def materialize_node(state: PipelineState) -> dict[str, Any]:
    """Generate or cache-hit. Does not run checks."""
    out_dir = Path(state["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(state.get("config_path"))
    policy = state.get("policy") or cfg.policy_default
    backend = state.get("backend") or ("none" if policy == "rule" else cfg.backend_default)
    if policy == "planned":
        preview = planned_preflight(cfg, backend=backend, policy=policy)
        emit({"type": "generate", "message": f"preflight {preview.get('policy')} {preview.get('backend')}"})
    request = ImageRequest(
        image_path=str(Path(state["image_path"]).resolve()),
        sample_id=state.get("sample_id"),
    )
    events_path = out_dir / "events.jsonl"
    with bind_progress(Progress(events_path)):
        try:
            bundle = materialize(
                request,
                cfg,
                backend_override=state.get("generation_backend"),
            )
        except Exception as exc:  # noqa: BLE001
            report = generation_failed_report(state.get("sample_id") or "unknown", exc)
            line = emit(
                {"type": "generate", "message": f"failed {report.get('stop_reason')}"},
                path=events_path,
            )
            return {
                "pipeline_status": "generation_failed",
                "result": report,
                "studio_summary": studio_summary(report),
                "last_event": line,
            }
    gen = bundle.summary()
    return {
        "pipeline_status": "generated",
        "sample_spec": bundle.sample.model_dump(),
        "generation": gen,
        "studio_summary": {
            "sample_id": bundle.sample_id,
            "pipeline_status": "generated",
            "shape_tv": list(bundle.preds_shape),
            "generation": {
                "cache_hit_image": bundle.cache_hit_image,
                "cache_hit_control": bundle.cache_hit_control,
                "new_tribe_predictions": bundle.new_tribe_predictions,
                "preds_shape": list(bundle.preds_shape),
                "synthetic": bundle.synthetic,
            },
        },
        "last_event": f"[generate] done {bundle.sample_id}",
    }


async def run_check_node(state: PipelineState) -> dict[str, Any]:
    """Existing run_sample after materialize."""
    cfg = load_config(state.get("config_path"))
    spec = SampleSpec.model_validate(state["sample_spec"])
    gen = state.get("generation") or {}
    if gen.get("control_preds_path"):
        cfg = cfg.model_copy(deep=True)
        cfg.resources.gray_control_path = gen["control_preds_path"]
        cfg.resources.gray_control_id = gen.get("control_sample_id")
    out_dir = Path(state["out_dir"])
    with bind_progress(Progress(out_dir / "events.jsonl")):
        check = await run_sample(
            spec,
            base_dir=out_dir,
            out_dir=out_dir,
            config=cfg,
            backend=state.get("backend") or "none",
            policy=state.get("policy") or "rule",
        )
    wrapper = {
        "pipeline_status": "ok",
        "check_verdict": check.get("verdict"),
        "generation": gen,
        **check,
    }
    return {
        "pipeline_status": "ok",
        "result": wrapper,
        "studio_summary": studio_summary(wrapper),
        "last_event": f"[stop] {check.get('verdict')}",
    }


def _after_materialize(state: PipelineState) -> str:
    return END if state.get("pipeline_status") == "generation_failed" else "run_check"


builder = StateGraph(PipelineState)
builder.add_node("apply_defaults", apply_defaults_node)
builder.add_node("materialize", materialize_node)
builder.add_node("run_check", run_check_node)
builder.add_edge("__start__", "apply_defaults")
builder.add_edge("apply_defaults", "materialize")
builder.add_conditional_edges("materialize", _after_materialize)
builder.add_edge("run_check", "__end__")
graph = builder.compile(name="fMRI Image Pipeline")
