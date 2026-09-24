"""V1.4 diagnostic contracts, metrics, and tool behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from react_agent.fmri.config import load_config
from react_agent.fmri.data import ArrayHandle
from react_agent.fmri.diagnostic.execution import execution_key
from react_agent.fmri.diagnostic.metrics import contrast_series, transition_energy
from react_agent.fmri.diagnostic.routing import derive_tickets
from react_agent.fmri.diagnostic.stop_gate import evaluate_stop
from react_agent.fmri.ledger import build_question_ledger
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import QuestionRecord, SampleResources, SampleSpec, ToolResult
from react_agent.fmri.tools.base import RunContext
from react_agent.fmri.tools.control_contrast import GrayControlContrastTool
from react_agent.fmri.tools.numeric import TemporalDiagnosticsTool
from react_agent.fmri.tools.surface_roi import SurfaceRoiProfileTool
from react_agent.fmri.tools.surface_spatial import SurfaceSpatialSanityTool
from react_agent.fmri.tools.temporal_profile import StimulusTemporalProfileTool

CFG = Path("/home/zxuff/data/EEGagent/react-agent/configs/fmri_check_tribe_image16_diagnostic_v1_4.yaml")
ACC = Path("/home/zxuff/data/EEGagent/assets/generation/preds/accordion_01b_7f4dff7eeb052fbc6f3e1b99.npy")
GRAY = Path("/home/zxuff/data/EEGagent/assets/generation/control_preds/gray_500x500_80abe8b0e5ca13c43e86422b.npy")


def _spec(tmp: Path, name: str, arr: np.ndarray, **kwargs) -> SampleSpec:
    path = tmp / f"{name}.npy"
    np.save(path, arr.astype(np.float32))
    resources = kwargs.pop("resources", SampleResources(image_id=name, time_alignment_status="verified"))
    return SampleSpec(
        sample_id=name,
        fmri_path=str(path),
        time_axis=0,
        sampling_interval_s=1.0,
        spatial_representation="surface",
        space_name="fsaverage5",
        normalization="none_raw_signed",
        generation_profile_id="static_gray4_image1_gray11_tribev2_v1",
        expected_shape=tuple(arr.shape),
        expected_n_vertices=int(arr.shape[1]),
        resources=resources,
        **kwargs,
    )


def _ctx(tmp: Path, spec: SampleSpec, cfg, prior=None) -> RunContext:
    data = np.load(spec.fmri_path)
    handle = ArrayHandle(
        path=spec.fmri_path,
        kind="npy",
        array_key=None,
        time_axis=0,
        shape_tv=tuple(data.shape),
        dtype=str(data.dtype),
        fingerprint="fp-" + spec.sample_id,
    )
    return RunContext(
        config=cfg,
        sample=spec,
        array_handle=handle,
        base_dir=str(tmp),
        artifact_dir=str(tmp / spec.sample_id),
        fingerprint=handle.fingerprint,
        prior_results=prior or [],
    )


def _pair(tmp: Path):
    cfg = load_config(CFG)
    y = np.zeros((16, 8), dtype=np.float32)
    g = np.zeros_like(y)
    y[4, :] = 1.0
    gpath = tmp / "gray.npy"
    np.save(gpath, g)
    cfg.resources.gray_control_path = str(gpath)
    spec = _spec(tmp, "img", y)
    return cfg, spec


def test_config_loads_without_touching_v1_3() -> None:
    cfg = load_config(CFG)
    assert cfg.diagnostic.enabled
    assert cfg.diagnostic.routing.contrast_rms_step_ratio.threshold == 5.0
    assert cfg.diagnostic.routing.contrast_rms_step_ratio.decision_effect == "none"
    old = load_config(Path("/home/zxuff/data/EEGagent/react-agent/configs/fmri_check_tribe_image16_agentic.yaml"))
    assert old.check_profile.name == "tribe_diagnostic"
    assert not old.diagnostic.enabled


def test_raw_temporal_does_not_answer_contrast_question(tmp_path: Path) -> None:
    cfg, spec = _pair(tmp_path)
    raw = ToolResult(
        tool_name="temporal_diagnostics",
        tool_version="1.0.0",
        result_id="raw1",
        execution_status="success",
        metrics={"signal_mode": "raw", "peak_diff_frame": 14},
        signal_provenance={"signal_mode": "raw"},
    )
    basic = ToolResult(
        tool_name="basic_statistics",
        tool_version="1.0.0",
        result_id="b1",
        execution_status="success",
        metrics={"T": 16, "V": 8},
    )
    valid = ToolResult(
        tool_name="validate_input",
        tool_version="1.0.0",
        result_id="v1",
        execution_status="success",
        metrics={"T": 16, "V": 8},
    )
    ledger = build_question_ledger(
        sample=spec,
        results=[valid, basic, raw],
        config=cfg,
        resolver=ResourceResolver(cfg, base_dir=tmp_path),
    )
    by_id = {q.question_id: q for q in ledger}
    assert by_id["stimulus_temporal_description"].status == "unassessed"
    assert by_id["stimulus_temporal_description"].unresolved_reason == "raw_temporal_does_not_satisfy_contrast_contract"
    old = load_config(Path("/home/zxuff/data/EEGagent/react-agent/configs/fmri_check_tribe_image16_agentic.yaml"))
    old_ledger = build_question_ledger(
        sample=spec,
        results=[raw],
        config=old,
        resolver=ResourceResolver(old, base_dir=tmp_path),
    )
    temporal = next(q for q in old_ledger if q.question_id == "temporal")
    assert temporal.status == "described"


def test_ras_keeps_legacy_step_and_marks_zero_denominator(tmp_path: Path) -> None:
    cfg, spec = _pair(tmp_path)
    out = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert "max_step_over_median" in out.metrics
    ras = out.metrics["ras_v1"]
    assert ras["metric_definition_version"] == "ras_v1"
    assert ras["max_A"]["from_frame"] is not None
    zero = contrast_series(np.zeros((16, 4)))
    assert zero["max_A_over_median_status"] in {"undefined", "denominator_unstable"}
    assert zero["max_A_over_median"] is None


def test_mean_cancellation_still_localizes() -> None:
    delta = np.zeros((4, 200))
    delta[2, 0] = 5.0
    delta[2, 1] = -5.0
    energy = transition_energy(delta, 1)
    assert abs(float(np.mean(delta[2] - delta[1]))) < 1e-9
    assert energy["status"] == "defined"
    assert energy["rms_delta"] > 0.1
    assert energy["top_count"] >= 2
    assert set(energy["top_vertices"][:2]) == {0, 1}


def test_l1_profile_and_spatial_are_descriptive(tmp_path: Path) -> None:
    cfg, spec = _pair(tmp_path)
    contrast = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    prior = [contrast.model_dump()]
    profile = asyncio.run(
        StimulusTemporalProfileTool().run(spec, {}, _ctx(tmp_path, spec, cfg, prior))
    )
    spatial = asyncio.run(
        SurfaceSpatialSanityTool().run(spec, {"mode": "summary"}, _ctx(tmp_path, spec, cfg, prior))
    )
    assert profile.metrics["signal_mode"] == "contrast"
    assert profile.metrics["ras_v1"]["R"]
    assert profile.metrics["image16_windows"]["stimulus"] == [4.0, 5.0]
    assert spatial.metrics["mode"] == "summary"
    assert spatial.findings[0].decision_effect == "none"
    assert spatial.metrics["neighborhood"] == "unavailable"
    ledger = build_question_ledger(
        sample=spec,
        results=[
            ToolResult(tool_name="validate_input", tool_version="1", result_id="v", execution_status="success", metrics={"T": 16, "V": 8}),
            ToolResult(tool_name="basic_statistics", tool_version="1", result_id="b", execution_status="success", metrics={"T": 16, "V": 8}),
            contrast,
            profile,
            spatial,
        ],
        config=cfg,
        resolver=ResourceResolver(cfg, base_dir=tmp_path),
    )
    by_id = {q.question_id: q for q in ledger}
    assert by_id["stimulus_temporal_description"].status == "described"
    assert by_id["coarse_spatial_description"].status == "described"


def test_heuristic_trigger_does_not_flag(tmp_path: Path) -> None:
    cfg = load_config(CFG)
    contrast = ToolResult(
        tool_name="gray_control_contrast",
        tool_version="1.1.0",
        result_id="c1",
        execution_status="success",
        applicability="applicable",
        metrics={
            "comparable": True,
            "max_step_over_median": 6.69,
            "max_step": 0.2,
            "median_step": 0.03,
            "ras_v1": {"max_A": {"from_frame": 4, "to_frame": 5}},
        },
        signal_provenance={"signal_mode": "contrast"},
    )
    tickets = derive_tickets([contrast], cfg)
    assert tickets and tickets[0]["status"] == "open"
    assert tickets[0]["decision_effect"] == "none"
    assert tickets[0]["defect_confirmed"] is False
    gate = evaluate_stop(
        [QuestionRecord(question_id="triggered_followups", status="unassessed", required=True)],
        tickets,
        budget_ok=True,
        executable_actions=["temporal_diagnostics"],
    )
    assert gate["allow_stop"] is False


def test_execution_keys_differ_and_match() -> None:
    a = execution_key(
        tool_id="temporal_diagnostics",
        tool_version="1.0.0",
        signal_mode="raw",
        input_hash="h",
        control_hash=None,
        params={"mode": "summary"},
        metric_definition_version="ras_v1",
    )
    b = execution_key(
        tool_id="temporal_diagnostics",
        tool_version="1.0.0",
        signal_mode="contrast",
        input_hash="h",
        control_hash="g",
        params={"mode": "targeted", "from_frame": 4},
        metric_definition_version="ras_v1",
    )
    c = execution_key(
        tool_id="temporal_diagnostics",
        tool_version="1.0.0",
        signal_mode="raw",
        input_hash="h",
        control_hash=None,
        params={"mode": "summary"},
        metric_definition_version="ras_v1",
    )
    assert a != b
    assert a == c


def test_missing_atlas_blocks_roi_followup(tmp_path: Path) -> None:
    cfg, spec = _pair(tmp_path)
    cfg.resources.atlas_dir = None
    out = asyncio.run(
        SurfaceRoiProfileTool().run(spec, {"mode": "targeted", "from_frame": 4}, _ctx(tmp_path, spec, cfg))
    )
    assert out.skip_reason == "no_atlas"
    assert out.metrics["status"] == "blocked"


def test_planning_blocked_and_early_pass() -> None:
    done = [
        QuestionRecord(question_id="input_contract", status="answered", required=True),
        QuestionRecord(question_id="gray_comparability", status="answered", required=True),
        QuestionRecord(question_id="stimulus_temporal_description", status="described", required=True),
        QuestionRecord(question_id="coarse_spatial_description", status="described", required=True),
        QuestionRecord(question_id="triggered_followups", status="answered", required=True),
    ]
    early = evaluate_stop(done, [], budget_ok=True, executable_actions=["cortex_mae"])
    assert early["screening_decision"] == "pass_configured"
    blocked = evaluate_stop(
        [QuestionRecord(question_id="stimulus_temporal_description", status="unassessed", required=True)],
        [],
        budget_ok=False,
        executable_actions=[],
        planning_failed=True,
    )
    assert blocked["stop_reason"] == "planning_blocked"


def test_targeted_contrast_transition(tmp_path: Path) -> None:
    cfg, spec = _pair(tmp_path)
    contrast = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    targeted = asyncio.run(
        TemporalDiagnosticsTool().run(
            spec,
            {"mode": "targeted", "signal": "contrast", "from_frame": 3},
            _ctx(tmp_path, spec, cfg, [contrast.model_dump()]),
        )
    )
    assert targeted.metrics["signal_mode"] == "contrast"
    assert targeted.metrics["from_frame"] == 3
    assert targeted.metrics["to_frame"] == 4
    assert targeted.findings[0].decision_effect == "none"


def test_accordion_cache_contrast_if_present(tmp_path: Path) -> None:
    if not ACC.is_file() or not GRAY.is_file():
        return
    y = np.load(ACC)
    g = np.load(GRAY)
    if y.shape != g.shape:
        return
    cfg = load_config(CFG)
    cfg.resources.gray_control_path = str(GRAY)
    spec = _spec(tmp_path, "accordion_01b", y)
    out = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.execution_status == "success"
    if out.applicability != "applicable":
        return
    assert out.metrics.get("max_step") is not None
    assert out.findings[0].decision_effect == "none"
    prior = [out.model_dump()]
    spatial = asyncio.run(SurfaceSpatialSanityTool().run(spec, {"mode": "summary"}, _ctx(tmp_path, spec, cfg, prior)))
    assert spatial.metrics["signal_mode"] == "contrast"
    ratio = out.metrics.get("max_step_over_median")
    if ratio is not None and ratio >= 5:
        frm = (out.metrics.get("ras_v1") or {}).get("max_A", {}).get("from_frame")
        targeted = asyncio.run(
            TemporalDiagnosticsTool().run(
                spec,
                {"mode": "targeted", "signal": "contrast", "from_frame": int(frm)},
                _ctx(tmp_path, spec, cfg, prior),
            )
        )
        assert targeted.findings[0].decision_effect == "none"
        assert targeted.metrics["status"] == "defined"
