"""Deep screening option, real depth, and per-call LLM detail."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from react_agent.fmri import workbench
from react_agent.fmri.budget import BudgetLedger
from react_agent.fmri.config import DEEP_SCREEN_QUESTIONS, load_config
from react_agent.fmri.ledger import build_question_ledger
from react_agent.fmri.loop import reached_depth
from react_agent.fmri.resources import ResourceResolver
from react_agent.fmri.schemas import LMUsage, SampleResources, SampleSpec, ToolResult
from react_agent.fmri.tools.registry import build_registry

CFG = Path("/home/zxuff/data/EEGagent/react-agent/configs/fmri_check_tribe_image16_diagnostic_v1_4.yaml")
STANDARD_REQUIRED = [
    "input_contract",
    "gray_comparability",
    "stimulus_temporal_description",
    "coarse_spatial_description",
    "triggered_followups",
]


def _ok(tool: str, rid: str, **metrics) -> ToolResult:
    return ToolResult(tool_name=tool, tool_version="1.0.0", result_id=rid, execution_status="success", metrics=metrics)


def _sample(tmp: Path) -> SampleSpec:
    path = tmp / "img.npy"
    np.save(path, np.zeros((16, 8), dtype=np.float32))
    return SampleSpec(
        sample_id="img",
        fmri_path=str(path),
        time_axis=0,
        sampling_interval_s=1.0,
        spatial_representation="surface",
        space_name="fsaverage5",
        normalization="none_raw_signed",
        generation_profile_id="static_gray4_image1_gray11_tribev2_v1",
        expected_shape=(16, 8),
        expected_n_vertices=8,
        resources=SampleResources(image_id="img", time_alignment_status="verified"),
    )


def test_standard_screening_keeps_the_existing_required_questions() -> None:
    cfg = load_config(CFG)
    assert cfg.diagnostic.screen_depth == "standard"
    assert cfg.check_profile.required_questions == STANDARD_REQUIRED


def test_deep_screening_requires_four_more_questions() -> None:
    cfg = load_config(CFG, cli_overrides={"diagnostic": {"screen_depth": "deep"}})
    assert cfg.check_profile.required_questions == STANDARD_REQUIRED + [q for q, _tool in DEEP_SCREEN_QUESTIONS]
    for _question, tool in DEEP_SCREEN_QUESTIONS:
        assert tool in cfg.enabled_tools


def test_deep_ledger_marks_missing_resource_as_blocked(tmp_path: Path) -> None:
    cfg = load_config(CFG, cli_overrides={"diagnostic": {"screen_depth": "deep"}})
    gray = tmp_path / "gray.npy"
    np.save(gray, np.zeros((16, 8), dtype=np.float32))
    cfg.resources.gray_control_path = str(gray)
    spec = _sample(tmp_path)
    results = [
        _ok("validate_input", "v1", T=16, V=8),
        _ok("basic_statistics", "b1", T=16, V=8),
        _ok("surface_roi_profile", "r1"),
        ToolResult(
            tool_name="reference_distribution",
            tool_version="1.0.0",
            result_id="ref1",
            execution_status="skipped",
            skip_reason="reference_stats_incompatible",
        ),
    ]
    ledger = build_question_ledger(
        sample=spec, results=results, config=cfg, resolver=ResourceResolver(cfg, base_dir=tmp_path)
    )
    by_id = {q.question_id: q for q in ledger}
    assert by_id["roi"].status == "described"
    assert by_id["calibrated_reference"].status == "blocked"
    assert by_id["calibrated_reference"].unresolved_reason == "reference_stats_incompatible"
    assert by_id["specificity"].status == "unassessed"
    assert by_id["image_fmri_rsa"].required is True
    standard = load_config(CFG)
    standard.resources.gray_control_path = str(gray)
    plain = build_question_ledger(
        sample=spec, results=results, config=standard, resolver=ResourceResolver(standard, base_dir=tmp_path)
    )
    assert "roi" not in {q.question_id for q in plain}


def test_depth_comes_from_tools_that_succeeded() -> None:
    cfg = load_config(CFG)
    specs = build_registry(cfg).specs()
    first = [_ok("validate_input", "v1"), _ok("gray_control_contrast", "g1")]
    assert reached_depth(first, specs) == "L1"
    assert reached_depth(first + [_ok("surface_roi_profile", "r1")], specs) == "L2"
    assert reached_depth(first + [_ok("reference_distribution", "ref1")], specs) == "L3"
    failed = ToolResult(
        tool_name="semantic_consistency", tool_version="1.0.0", result_id="s1", execution_status="error"
    )
    assert reached_depth(first + [failed], specs) == "L1"


def test_each_lm_call_is_recorded() -> None:
    cfg = load_config(CFG)
    budget = BudgetLedger(cfg.budgets)
    budget.record_lm(
        LMUsage(provider="deepseek", requested_model="deepseek-chat", role="planner", input_tokens=10, output_tokens=2)
    )
    assert budget.data["lm_calls"] == 1
    assert budget.calls[0]["role"] == "planner"
    assert budget.calls[0]["input_tokens"] == 10


def test_workbench_rejects_bad_depth_and_passes_deep(tmp_path: Path) -> None:
    form = {"image_path": "/tmp/x.jpg", "config": CFG.name, "screen_depth": "maximum"}
    with pytest.raises(ValueError):
        workbench._validate_form(form)
    fields = workbench._validate_form({**form, "screen_depth": "deep"})
    assert fields["screen_depth"] == "deep"
    assert fields["out_dir"].name == "workbench_deep"
    cfg = workbench.load_job_config(fields)
    assert cfg.diagnostic.screen_depth == "deep"
    quick = workbench._validate_form({**form, "config": "fmri_check_tribe_image16.yaml", "screen_depth": "deep"})
    with pytest.raises(ValueError):
        workbench.load_job_config(quick)
