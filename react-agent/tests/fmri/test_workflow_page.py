"""Contract bars and workflow page. Plotting the mesh is covered by the accordion run."""

from __future__ import annotations

import json
from pathlib import Path

from react_agent.fmri.pipeline_graph import studio_summary
from react_agent.fmri.workflow import contract_bars, publish_workflow

ACC_REPORT = Path(
    "/home/zxuff/data/EEGagent/react-agent/runs/"
    "tribe_image16_diagnostic_v1_4_accordion/accordion_01b_f1f17445/report.json"
)


def test_contract_bars_mark_raw_temporal_red_and_finite_green() -> None:
    report = {
        "claim_scope": "numeric_consistency_only",
        "evidence_confidence": None,
        "metrics_table": {
            "validate_input": {"finite_ratio": 1.0, "valid_for_numeric_checks": True},
            "stimulus_temporal_profile": {
                "signal_mode": "raw",
                "satisfies_contrast_contract": False,
            },
            "gray_control_contrast": {"comparable": True, "max_step_over_median": 6.69},
            "surface_spatial_sanity": {"signal_mode": "contrast", "mode": "summary"},
        },
        "executions": [
            {"tool_name": "surface_spatial_sanity", "status": "success"},
        ],
        "coverage_by_dimension": {
            "input_contract": {"status": "answered"},
            "stimulus_temporal_description": {"status": "unassessed"},
        },
    }
    bars = {row["id"]: row for row in contract_bars(report)}
    assert bars["finite_ratio"]["score"] == 1.0
    assert bars["satisfies_contrast_contract"]["score"] == 0.0
    assert bars["gray_control_contrast.comparable"]["score"] == 1.0
    assert bars["surface_spatial_sanity"]["score"] == 1.0
    assert bars["coverage.input_contract"]["score"] == 1.0
    assert bars["coverage.stimulus_temporal_description"]["score"] == 0.0
    assert "max_step_over_median" not in bars


def test_workflow_page_without_mesh(tmp_path: Path) -> None:
    report = {
        "sample_id": "toy",
        "claim_scope": "numeric_consistency_only",
        "analysis_goal": "image16_numeric_diagnostic",
        "evidence_confidence": None,
        "recommended_training_weight": None,
        "metrics_table": {
            "validate_input": {
                "finite_ratio": 1.0,
                "valid_for_numeric_checks": True,
                "resolved_path": str(tmp_path / "missing.npy"),
            },
            "stimulus_temporal_profile": {"satisfies_contrast_contract": False},
        },
        "coverage_by_dimension": {},
    }
    page = publish_workflow(
        tmp_path,
        report,
        sample_spec={"sample_id": "toy", "image_path": "/tmp/toy.jpg"},
        gray_path="/tmp/gray.npy",
    )
    html = (tmp_path / "workflow.html").read_text(encoding="utf-8")
    assert page["image_source"] == "missing"
    assert "apple_signed_tstrip" not in html
    assert "/tmp/toy.jpg" in html
    assert "4 秒灰屏" in html
    assert "image16_numeric_diagnostic" in html
    assert page["evidence_confidence"] is None
    assert page["metric_scores"]["finite_ratio"] == 1.0
    assert page["metric_scores"]["satisfies_contrast_contract"] == 0.0


def test_studio_summary_keeps_scores_and_omits_arrays() -> None:
    summary = studio_summary(
        {
            "sample_id": "accordion_01b",
            "verdict": "passed_configured_checks",
            "claim_scope": "numeric_consistency_only",
            "preds": [[0.1, 0.2]],
            "workflow": {
                "html_path": "/tmp/run/workflow.html",
                "brain_tstrip": "/tmp/run/brain_tstrip.png",
                "metric_scores": {"finite_ratio": 1.0},
            },
        }
    )
    assert summary["workflow_html"] == "/tmp/run/workflow.html"
    assert summary["metric_scores"]["finite_ratio"] == 1.0
    assert "preds" not in summary
    blob = json.dumps(summary)
    assert "0.1" not in blob


def test_accordion_report_has_red_and_green_contracts() -> None:
    if not ACC_REPORT.is_file():
        return
    report = json.loads(ACC_REPORT.read_text(encoding="utf-8"))
    bars = {row["id"]: row for row in contract_bars(report)}
    assert bars["finite_ratio"]["score"] == 1.0
    assert bars["satisfies_contrast_contract"]["score"] == 0.0
    assert report.get("evidence_confidence") is None
