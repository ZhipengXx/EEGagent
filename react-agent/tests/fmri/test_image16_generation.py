"""V1.2 16s generation: video protocol, export contract, cache, CortexMAE T."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from react_agent.fmri.budget import BudgetLedger
from react_agent.fmri.config import load_config
from react_agent.fmri.generation.export import export_window
from react_agent.fmri.generation.schemas import (
    EXPECTED_T,
    EXPECTED_V,
    PROFILE_ID,
    ImageRequest,
    StimulusProfile,
    TemporalContractError,
)
from react_agent.fmri.generation.tool import TribeGenerateTool, assert_cached_contract
from react_agent.fmri.generation.video import build_frames, classify_frames, gray_frame
from react_agent.fmri.tools.registry import build_registry


def _tiny_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (32, 32), (210, 30, 30)).save(path)
    return path


def test_video_160_frames_and_boundaries() -> None:
    profile = StimulusProfile()
    assert profile.fps == 10
    assert profile.n_frames == 160
    assert profile.n_pre == 40
    assert profile.n_img == 10
    assert profile.n_post == 110
    image = np.full((40, 40, 3), (200, 10, 10), dtype=np.uint8)
    frames = build_frames(image, profile)
    report = classify_frames(frames, profile, expect_image=True)
    assert report["protocol_ok"]
    assert frames[39].mean() == pytest.approx(128.0)
    assert frames[40].mean() != pytest.approx(128.0)
    assert frames[49].mean() != pytest.approx(128.0)
    assert frames[50].mean() == pytest.approx(128.0)
    gray = build_frames(None, profile, size_hw=(40, 40))
    gray_report = classify_frames(gray, profile, expect_image=False)
    assert gray_report["protocol_ok"]
    assert gray_frame((40, 40)).shape == (40, 40, 3)


def test_export_rejects_12_and_15_points() -> None:
    segs12 = [{"start": float(i), "duration": 1.0} for i in range(12)]
    segs15 = [{"start": float(i), "duration": 1.0} for i in range(15)]
    segs16 = [{"start": float(i), "duration": 1.0} for i in range(16)]
    with pytest.raises(TemporalContractError, match="contracted T=16"):
        export_window(np.zeros((12, EXPECTED_V), dtype=np.float32), segs12)
    with pytest.raises(TemporalContractError, match="contracted T=16"):
        export_window(np.zeros((15, EXPECTED_V), dtype=np.float32), segs15)
    with pytest.raises(TemporalContractError, match="time basis"):
        export_window(np.zeros((16, EXPECTED_V), dtype=np.float32), None)
    out, meta = export_window(np.zeros((16, EXPECTED_V), dtype=np.float32), segs16)
    assert out.shape == (EXPECTED_T, EXPECTED_V)
    assert meta["temporal_padding_applied"] is False


def test_mock_registers_samplespec_and_cache(tmp_path: Path) -> None:
    image = _tiny_image(tmp_path / "tiny.jpg")
    cfg = load_config()
    cfg.generation.backend = "mock"
    cfg.generation.artifact_root = str(tmp_path / "gen")
    cfg.budgets.max_generation_requests = 4
    cfg.budgets.max_new_tribe_predictions = 4
    budget = BudgetLedger(cfg.budgets)
    tool = TribeGenerateTool(cfg, budget=budget)
    req = ImageRequest(image_path=str(image), sample_id="tiny")
    first = tool.run(req)
    assert first.sample.generation_profile_id == PROFILE_ID
    assert first.sample.expected_shape == (16, 20484)
    assert first.preds_shape == (16, 20484)
    assert first.new_tribe_predictions == 2
    assert first.temporal_padding_applied is False
    assert first.synthetic is True
    assert first.cache_hit_image is False
    preds = np.load(first.preds_path)
    assert preds.shape == (16, 20484)
    second_budget = BudgetLedger(cfg.budgets)
    second = TribeGenerateTool(cfg, budget=second_budget).run(req)
    assert second.cache_hit_image is True
    assert second.cache_hit_control is True
    assert second.new_tribe_predictions == 0
    assert second_budget.data["new_tribe_predictions"] == 0
    assert second_budget.data["generation_requests"] == 1


def test_cached_12_point_refused(tmp_path: Path) -> None:
    npy = tmp_path / "bad.npy"
    meta = tmp_path / "bad.json"
    np.save(npy, np.zeros((12, EXPECTED_V), dtype=np.float32))
    meta.write_text(
        '{"segments": '
        + str([{"start": float(i), "duration": 1.0} for i in range(12)]).replace("'", '"')
        + ', "temporal_padding_applied": false}',
        encoding="utf-8",
    )
    with pytest.raises(TemporalContractError):
        assert_cached_contract(npy, meta)


def test_generator_not_a_check_candidate(tmp_path: Path) -> None:
    from react_agent.fmri.schemas import SampleSpec
    from react_agent.fmri.tools.base import RunContext

    cfg = load_config()
    spec = SampleSpec(
        sample_id="x",
        fmri_path=str(tmp_path / "x.npy"),
        time_axis=0,
        generation_profile_id=PROFILE_ID,
    )
    ctx = RunContext(
        config=cfg,
        sample=spec,
        array_handle=None,
        base_dir=str(tmp_path),
        artifact_dir=str(tmp_path),
        fingerprint="x",
        prior_results=[],
    )
    names = build_registry(cfg).executable(
        spec,
        ctx,
        enabled=["tribev2_generate_fmri", "validate_input"],
        completed=set(),
    )
    assert "tribev2_generate_fmri" not in names


def test_cortexmae_t12_still_skipped(tmp_path: Path) -> None:
    from react_agent.fmri.providers.base import EncoderError
    from react_agent.fmri.schemas import SampleSpec
    from react_agent.fmri.tools.base import RunContext
    from react_agent.fmri.tools.cortex_mae import CortexMaeTool
    from react_agent.fmri.data import ArrayHandle

    class RejectT:
        def missing_reasons(self):
            return []

        def encode_surface(self, tv):
            raise EncoderError("temporal_length_unsupported", f"T={tv.shape[0]}")

    arr = np.zeros((12, 20484), dtype=np.float32)
    path = tmp_path / "t12.npy"
    np.save(path, arr)
    spec = SampleSpec(
        sample_id="t12",
        fmri_path=str(path),
        time_axis=0,
        expected_shape=(12, 20484),
    )
    cfg = load_config()
    ctx = RunContext(
        config=cfg,
        sample=spec,
        array_handle=ArrayHandle(
            path=str(path),
            kind="npy",
            array_key=None,
            time_axis=0,
            shape_tv=(12, 20484),
            dtype="float32",
            fingerprint="t12",
        ),
        base_dir=str(tmp_path),
        artifact_dir=str(tmp_path / "art"),
        fingerprint="t12",
        prior_results=[],
    )
    out = asyncio.run(CortexMaeTool(provider=RejectT()).run(spec, {}, ctx))
    assert out.execution_status == "skipped"
    assert out.metrics["padded"] is False
    assert out.metrics["encoder_forward_validated"] is False


def test_cortexmae_t16_mock_forward(tmp_path: Path) -> None:
    from react_agent.fmri.data import ArrayHandle
    from react_agent.fmri.schemas import SampleSpec
    from react_agent.fmri.tools.base import RunContext
    from react_agent.fmri.tools.cortex_mae import CortexMaeTool

    class Accept16:
        def missing_reasons(self):
            return []

        def encode_surface(self, tv):
            t = int(tv.shape[0])
            assert t == 16
            return np.ones(8), {
                "input_t": t,
                "trained_t": 16,
                "aggregation": "fake",
                "padded": False,
                "model_id": "fake",
                "n_parcels": 400,
            }

    arr = np.zeros((16, 20484), dtype=np.float32)
    path = tmp_path / "t16.npy"
    np.save(path, arr)
    spec = SampleSpec(
        sample_id="t16",
        fmri_path=str(path),
        time_axis=0,
        expected_shape=(16, 20484),
        generation_profile_id=PROFILE_ID,
    )
    cfg = load_config()
    ctx = RunContext(
        config=cfg,
        sample=spec,
        array_handle=ArrayHandle(
            path=str(path),
            kind="npy",
            array_key=None,
            time_axis=0,
            shape_tv=(16, 20484),
            dtype="float32",
            fingerprint="t16",
        ),
        base_dir=str(tmp_path),
        artifact_dir=str(tmp_path / "art"),
        fingerprint="t16",
        prior_results=[],
    )
    out = asyncio.run(CortexMaeTool(provider=Accept16()).run(spec, {}, ctx))
    assert out.execution_status == "success"
    assert out.metrics["padded"] is False
    assert out.metrics["input_t"] == 16
    assert out.metrics["encoder_forward_validated"] is True
    assert out.metrics["embedding_available"] is True
    assert out.metrics["reference_score_assessed"] is False
    assert out.metrics["mode"] == "raw"
    assert out.applicability == "applicable"


def test_studio_defaults_fill_empty_form() -> None:
    from react_agent.fmri.pipeline_graph import (
        DEFAULT_STUDIO_INPUT,
        apply_studio_defaults,
    )

    filled = apply_studio_defaults({"image_path": "", "backend": "none"})
    assert filled["image_path"] == DEFAULT_STUDIO_INPUT["image_path"]
    assert filled["config_path"].endswith("fmri_check_tribe_image16.yaml")
    assert filled["out_dir"].endswith("tribe_image16_studio_view")
    assert filled["backend"] == "none"
    assert filled["policy"] == "rule"
    custom = apply_studio_defaults({"out_dir": "/tmp/studio_view", "sample_id": "other"})
    assert custom["out_dir"] == "/tmp/studio_view"
    assert custom["sample_id"] == "other"
    assert custom["image_path"] == DEFAULT_STUDIO_INPUT["image_path"]


def test_progress_emit_cache_and_tool_no_arrays() -> None:
    from react_agent.fmri.progress import Progress, bind_progress, emit, format_progress_line

    assert format_progress_line({"type": "generate", "message": "image cache hit"}) == (
        "[generate] image cache hit"
    )
    assert format_progress_line(
        {"type": "tool", "tool_name": "cortex_mae", "status": "success"}
    ) == "[tool] cortex_mae success"
    assert format_progress_line(
        {
            "type": "decision",
            "via": "rule",
            "decision": {"action": "run_tool", "tool_name": "gray_control_contrast"},
        }
    ) == "[decision] rule -> gray_control_contrast"
    sink: list[dict] = []
    with bind_progress(Progress(sink=sink, print_cli=False)):
        emit({"type": "generate", "message": "image cache hit", "preds": [[0.0] * 8] * 4})
        emit({"type": "tool", "tool_name": "cortex_mae", "status": "success", "embedding": [1.0] * 16})
    assert [row["line"] for row in sink] == [
        "[generate] image cache hit",
        "[tool] cortex_mae success",
    ]
    assert all("preds" not in row and "embedding" not in row for row in sink)


def test_studio_summary_omits_arrays() -> None:
    from react_agent.fmri.pipeline_graph import studio_summary

    result = {
        "sample_id": "apple_01b",
        "pipeline_status": "ok",
        "check_verdict": "passed_configured_checks",
        "verdict": "passed_configured_checks",
        "coverage_status": "complete",
        "claim_scope": "numeric_consistency_only",
        "metrics_table": {
            "basic_statistics": {"T": 16, "V": 20484},
            "gray_control_contrast": {"overall_delta_rms": 0.05, "comparable": True},
            "cortex_mae": {
                "mode": "raw",
                "input_t": 16,
                "trained_t": 16,
                "padded": False,
                "encoder_forward_validated": True,
                "embedding_available": True,
                "reference_score_assessed": False,
            },
        },
        "generation": {
            "cache_hit_image": True,
            "cache_hit_control": True,
            "new_tribe_predictions": 0,
            "preds_shape": [16, 20484],
            "synthetic": False,
            "temporal_padding_applied": False,
        },
        "preds": [[0.0] * 8] * 4,
    }
    summary = studio_summary(result)
    assert summary["check_verdict"] == "passed_configured_checks"
    assert summary["shape_tv"] == [16, 20484]
    assert summary["overall_delta_rms"] == 0.05
    assert summary["cortex_mae"]["padded"] is False
    assert summary["cortex_mae"]["input_t"] == 16
    assert "preds" not in summary
    blob = str(summary)
    assert "0.0, 0.0" not in blob
