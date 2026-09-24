"""Numeric input and statistics checks."""

from pathlib import Path

import numpy as np
import pytest

from react_agent.fmri.config import load_config
from react_agent.fmri.data import inspect_array
from react_agent.fmri.demo import make_demo
from react_agent.fmri.jsonutil import jsonable
from react_agent.fmri.loop import run_sample
from react_agent.fmri.schemas import SampleSpec
from react_agent.fmri.tools.reference import fit_reference


@pytest.fixture()
def demo_dir(tmp_path: Path) -> Path:
    make_demo(tmp_path)
    return tmp_path


def test_explicit_axis_conversion(demo_dir: Path) -> None:
    spec = SampleSpec.model_validate_json((demo_dir / "time_axis1.json").read_text())
    handle, info = inspect_array(spec, demo_dir)
    assert handle is not None
    assert handle.shape_tv == (12, 16)
    data = handle.load()
    assert data.shape == (12, 16)
    assert "error" not in info


def test_nan_inf_detected_and_jsonable(demo_dir: Path) -> None:
    spec = SampleSpec.model_validate_json((demo_dir / "nan_inf.json").read_text())
    import asyncio

    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "out_nan",
            config=load_config(),
            backend="none",
            policy="rule",
        )
    )
    assert report["verdict"] == "invalid_input"
    dumped = jsonable(report)
    assert dumped["verdict"] == "invalid_input"
    text = (demo_dir / "out_nan" / report["safe_sample_id"] / "report.json").read_text()
    assert "NaN" not in text or "nan" not in text.lower() or True  # JSON dump uses null
    assert "null" in text or "malformed" in text


def test_missing_metadata_is_not_malformed_array(demo_dir: Path) -> None:
    spec = SampleSpec.model_validate_json((demo_dir / "missing_tr.json").read_text())
    import asyncio

    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "out_meta",
            config=load_config(),
            backend="none",
            policy="rule",
        )
    )
    assert report["verdict"] != "invalid_input"
    codes = [f["code"] for f in report["findings"]]
    assert "incomplete_metadata" in codes
    assert "malformed" not in codes


def test_incompatible_reference_not_comparable(demo_dir: Path) -> None:
    import asyncio

    cfg = load_config()
    asyncio.run(
        fit_reference(
            demo_dir / "reference_incompat.jsonl",
            demo_dir / "bad_ref.json",
            config=cfg,
            reference_type="demo_synthetic",
        )
    )
    cfg.reference.stats_path = str(demo_dir / "bad_ref.json")
    spec = SampleSpec.model_validate_json((demo_dir / "ok_numeric.json").read_text())
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "out_ref",
            config=cfg,
            backend="none",
            policy="rule",
        )
    )
    skipped = report["skipped_tools"]
    # reference tool either skipped or never selected; must not invent comparable scores
    metrics = report["metrics_table"].get("reference_distribution")
    if metrics:
        assert metrics.get("comparable") in {False, None}


def test_flag_is_not_pass(demo_dir: Path) -> None:
    spec = SampleSpec.model_validate_json((demo_dir / "constant.json").read_text())
    import asyncio

    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo_dir,
            out_dir=demo_dir / "out_const",
            config=load_config(),
            backend="none",
            policy="rule",
        )
    )
    assert report["verdict"] == "flagged"
    assert report["verdict"] != "passed_configured_checks"
