"""View model keeps statuses, zeros, and nulls distinct."""

from __future__ import annotations

import json
from pathlib import Path

from react_agent.fmri.workbench_view import build_run_view, ensure_frame

ACC = Path(
    "/home/zxuff/data/EEGagent/react-agent/runs/"
    "tribe_image16_diagnostic_v1_4_accordion/accordion_01b_f1f17445"
)


def test_status_and_nulls_stay_distinct(tmp_path: Path) -> None:
    sample = tmp_path / "demo"
    sample.mkdir()
    (sample / "report.json").write_text(
        json.dumps(
            {
                "sample_id": "toy",
                "verdict": "passed_configured_checks",
                "stop_reason": "abstain",
                "evidence_confidence": None,
                "recommended_training_weight": None,
                "coverage_by_dimension": {
                    "stimulus_temporal_description": {"status": "unassessed", "required": True},
                    "coarse_spatial_description": {"status": "described", "required": True},
                    "optional_gap": {"status": "unassessed"},
                },
                "metrics_table": {
                    "validate_input": {"finite_ratio": 1.0, "valid_for_numeric_checks": True, "T": 4, "V": 2},
                    "gray_control_contrast": {"comparable": False, "max_step_over_median": 0},
                },
                "executions": [],
            }
        ),
        encoding="utf-8",
    )
    view = build_run_view("demo", tmp_path)
    assert view is not None
    by_id = {row["id"]: row for row in view["coverage"]}
    assert by_id["stimulus_temporal_description"]["status_label"] == "未评估"
    assert by_id["stimulus_temporal_description"]["tone"] == "neutral"
    assert by_id["coarse_spatial_description"]["status_label"] == "已描述"
    assert by_id["coarse_spatial_description"]["tone"] == "info"
    assert by_id["optional_gap"]["required_label"] == "要求未记录"
    assert view["coverage_satisfied"] == 1
    assert view["coverage_required"] == 2
    metrics = {row["id"]: row for row in view["metrics"]}
    assert metrics["evidence_confidence"]["display"] == "未提供"
    assert metrics["recommended_training_weight"]["display"] == "未提供"
    assert metrics["comparable"]["display"] == "否"
    assert metrics["max_step_over_median"]["value"] == 0
    assert metrics["max_step_over_median"]["display"] == "0"
    assert view["decision"]["verdict_label"] == "配置内数值检查通过"
    assert view["decision"]["short_label"] == "配置内通过"
    assert view["decision"]["stop_label"] == "暂无法判断"
    assert view["headline"] == "协议未记录"
    assert view["windows"] == []
    assert metrics["finite_ratio"]["display"] == "100%"
    assert metrics["shape_tv"]["display"] == "4 个时间点 × 2 个顶点"
    assert view["followup_note"] == "跟进记录未提供"


def test_accordion_report_if_present() -> None:
    if not (ACC / "report.json").is_file():
        return
    root = ACC.parents[1]
    view = build_run_view(ACC.relative_to(root).as_posix(), root)
    assert view is not None
    assert view["sample_id"] == "accordion_01b"
    metrics = {row["id"]: row for row in view["metrics"]}
    assert metrics["finite_ratio"]["value"] == 1.0
    assert metrics["satisfies_contrast_contract"]["value"] is False
    assert metrics["evidence_confidence"]["value"] is None
    assert view["llm"]["accepted_plan_count"] == 1
    assert view["llm"]["api_usd"] is None
    assert view["series"]["available"] is True
    described = next(row for row in view["coverage"] if row["id"] == "coarse_spatial_description")
    pending = next(row for row in view["coverage"] if row["id"] == "stimulus_temporal_description")
    assert described["tone"] == "info"
    assert pending["tone"] == "neutral"
    assert described["label_zh"] == "皮层空间分布"
    assert metrics["finite_ratio"]["display"] == "100%"
    assert "20,484" in metrics["shape_tv"]["display"]
    assert view["headline"].startswith("16s")
    assert view["followup_note"] == "无待跟进问题"
    assert view["windows"][1]["label"] == "图像"
    assert view["sample"]["stimulus_rel"] is None


def test_cached_frame_does_not_rebuild_report(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "demo"
    sample.mkdir()
    (sample / "report.json").write_text("{}", encoding="utf-8")
    out = sample / "frames" / "frame_v2" / "raw" / "00"
    out.mkdir(parents=True)
    for name in ("left", "right", "posterior"):
        (out / f"{name}.png").write_bytes(b"png")

    def boom(*_args, **_kwargs):
        raise AssertionError("rebuilt")

    monkeypatch.setattr("react_agent.fmri.workbench_view.build_run_view", boom)
    path, note = ensure_frame("demo", "raw", 0, tmp_path, "posterior")
    assert note == ""
    assert path == out / "posterior.png"


def test_one_render_serves_all_views(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time

    from react_agent.fmri import workbench_view as view_mod

    sample = tmp_path / "demo"
    sample.mkdir()
    (sample / "report.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_ctx(_sample_dir, _run_id, _runs_root):
        return {
            "modes": {"contrast"},
            "n_frames": 1,
            "limits": {"contrast": {"vmin": -1.0, "vmax": 1.0}},
            "preds": None,
            "gray": None,
        }

    def fake_render(sample_dir, _ctx, mode, index):
        calls.append(index)
        time.sleep(0.2)
        out = sample_dir / "frames" / "frame_v2" / mode / f"{index:02d}"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("left", "right", "posterior"):
            (out / f"{name}.png").write_bytes(b"png")
        return ""

    monkeypatch.setattr(view_mod, "_frame_context", fake_ctx)
    monkeypatch.setattr(view_mod, "_render_missing_frame", fake_render)
    errors = []

    def grab(name: str) -> None:
        path, note = ensure_frame("demo", "contrast", 0, tmp_path, name)
        if note or path is None:
            errors.append(note or "missing")

    threads = [threading.Thread(target=grab, args=(name,)) for name in ("left", "right", "posterior")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert calls == [0]
