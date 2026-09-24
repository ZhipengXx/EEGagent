"""V1.1 domain tools and reference-authority tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from react_agent.fmri.assets import AtlasError, prepare_destrieux_atlas, write_synthetic_atlas
from react_agent.fmri.cohort import build_cohort_index
from react_agent.fmri.config import load_config
from react_agent.fmri.loop import run_sample
from react_agent.fmri.reporting import verdict_from_results
from react_agent.fmri.schemas import Finding, SampleResources, SampleSpec, ToolResult
from react_agent.fmri.tools.base import RunContext
from react_agent.fmri.tools.control_contrast import GrayControlContrastTool
from react_agent.fmri.tools.reference import ReferenceDistributionTool
from react_agent.fmri.tools.specificity import CrossImageSpecificityTool
from react_agent.fmri.tools.surface_roi import SurfaceRoiProfileTool
from react_agent.fmri.tools.temporal_profile import StimulusTemporalProfileTool
from react_agent.fmri.data import ArrayHandle


def _spec(tmp: Path, name: str, arr: np.ndarray, **kwargs) -> SampleSpec:
    path = tmp / f"{name}.npy"
    np.save(path, arr)
    resources = kwargs.pop("resources", SampleResources(image_id=name))
    return SampleSpec(
        sample_id=name,
        fmri_path=str(path),
        time_axis=0,
        sampling_interval_s=1.0,
        spatial_representation="surface",
        space_name="fsaverage5",
        normalization="none_raw_signed",
        generation_profile_id="static_1s_7s_tribev2",
        expected_shape=tuple(arr.shape),
        expected_n_vertices=int(arr.shape[1]),
        resources=resources,
        **kwargs,
    )


def _ctx(tmp: Path, spec: SampleSpec, cfg, prior=None):
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


def test_n3_reference_cannot_change_verdict(tmp_path: Path) -> None:
    finding = Finding(
        code="insufficient_reference",
        severity="info",
        message="descriptive",
        decision_effect="none",
        calibration_status="insufficient",
    )
    result = ToolResult(
        tool_name="reference_distribution",
        tool_version="1.1.0",
        result_id="r1",
        execution_status="success",
        findings=[finding],
        metrics={"calibration_status": "insufficient", "n_ref": 3},
    )
    required = ToolResult(
        tool_name="validate_input",
        tool_version="1",
        result_id="v",
        execution_status="success",
        findings=[],
    )
    stats = ToolResult(
        tool_name="basic_statistics",
        tool_version="1",
        result_id="s",
        execution_status="success",
        findings=[],
    )
    verdict, coverage, _ = verdict_from_results(
        [required, stats, result],
        required=["validate_input", "basic_statistics"],
        invalid_input=False,
        unresolved=[],
        unavailable=[],
        reference_unassessed=True,
    )
    assert verdict == "passed_configured_checks"
    assert coverage == "partial"


def test_y_equals_g_zero_and_gray_not_applicable(tmp_path: Path) -> None:
    y = np.ones((12, 8), dtype=np.float32)
    gpath = tmp_path / "gray_control.npy"
    np.save(gpath, y)
    cfg = load_config()
    cfg.resources.gray_control_id = "gray_control"
    cfg.resources.gray_control_path = str(gpath)
    spec = _spec(tmp_path, "img", y, resources=SampleResources(control_sample_id="gray_control"))
    ctx = _ctx(tmp_path, spec, cfg)
    out = asyncio.run(GrayControlContrastTool().run(spec, {}, ctx))
    assert out.applicability == "applicable"
    assert out.metrics["overall_delta_rms"] == 0.0
    assert out.metrics.get("normalized_delta_over_control_rms") is not None or out.metrics["normalization_denominator"] == 0
    gray = _spec(tmp_path, "gray_control", y, resources=SampleResources(is_gray_control=True))
    gout = asyncio.run(GrayControlContrastTool().run(gray, {}, _ctx(tmp_path, gray, cfg)))
    assert gout.applicability == "not_applicable"


def test_contrast_localizes_perturbation_without_mutating_source(tmp_path: Path) -> None:
    g = np.zeros((12, 16), dtype=np.float32)
    y = g.copy()
    y[7, 3:6] = 4.0
    gpath = tmp_path / "gray_control.npy"
    np.save(gpath, g)
    src = tmp_path / "img.npy"
    np.save(src, y)
    before = np.load(src).copy()
    cfg = load_config()
    cfg.resources.gray_control_id = "gray_control"
    cfg.resources.gray_control_path = str(gpath)
    spec = _spec(tmp_path, "img", y, resources=SampleResources(control_sample_id="gray_control"))
    spec.fmri_path = str(src)
    out = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.execution_status == "success"
    assert out.metrics["peak_frame"] == 7
    assert out.metrics["overall_delta_rms"] > 0
    assert np.allclose(np.load(src), before)


def test_lr_same_label_not_merged(tmp_path: Path) -> None:
    write_synthetic_atlas(tmp_path / "atlas")
    cfg = load_config()
    cfg.resources.atlas_dir = str(tmp_path / "atlas")
    arr = np.zeros((4, 20484), dtype=np.float32)
    arr[:, 0] = 3.0
    arr[:, 10242] = -3.0
    spec = _spec(tmp_path, "img", arr)
    out = asyncio.run(SurfaceRoiProfileTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    keys = [row["roi_key"] for row in out.metrics["top_k"]]
    assert any(k.startswith("lh:") for k in keys)
    assert any(k.startswith("rh:") for k in keys)
    assert all(":" in k for k in keys)


def test_synthetic_cache_does_not_satisfy_prepare(tmp_path: Path) -> None:
    from react_agent.fmri.assets import _cached_manifest

    atlas = tmp_path / "atlas"
    write_synthetic_atlas(atlas)
    assert _cached_manifest(atlas) is None
    try:
        prepare_destrieux_atlas(atlas, allow_download=False)
    except AtlasError as exc:
        assert "prepare-assets" in str(exc)
    else:
        raise AssertionError("synthetic_test cache must not count as a real atlas")


def test_schaefer_adapter_keeps_t12(tmp_path: Path) -> None:
    from react_agent.fmri.p2_store import N_SCHAEFER
    from react_agent.fmri.providers.schaefer import surface_to_schaefer400, write_synthetic_schaefer

    atlas = write_synthetic_schaefer(tmp_path / "schaefer")
    labels = np.load(tmp_path / "schaefer" / "labels_lh_rh.npy")
    tv = np.zeros((12, 20484), dtype=np.float64)
    tv[3, :100] = 2.0
    parcels = surface_to_schaefer400(tv, labels)
    assert parcels.shape == (12, N_SCHAEFER)
    assert atlas["n_parcels"] == N_SCHAEFER


def test_cortexmae_unavailable_without_weights(tmp_path: Path) -> None:
    from react_agent.fmri.tools.cortex_mae import CortexMaeTool

    cfg = load_config()
    cfg.resources.asset_root = str(tmp_path / "assets")
    cfg.resources.schaefer_atlas_dir = str(tmp_path / "missing")
    spec = _spec(tmp_path, "img", np.zeros((12, 20484), dtype=np.float32))
    avail = CortexMaeTool().availability(spec, _ctx(tmp_path, spec, cfg))
    assert avail.status == "unavailable"


def test_cortexmae_skip_when_t_rejected(tmp_path: Path) -> None:
    from react_agent.fmri.providers.base import EncoderError
    from react_agent.fmri.providers.schaefer import write_synthetic_schaefer
    from react_agent.fmri.tools.cortex_mae import CortexMaeTool

    class RejectT:
        name = "fake"
        trained_t = 16

        def missing_reasons(self):
            return []

        def encode_surface(self, tv):
            raise EncoderError(
                "temporal_length_unsupported",
                f"official forward rejected T={tv.shape[0]} (trained_T=16)",
            )

    write_synthetic_schaefer(tmp_path / "schaefer")
    cfg = load_config()
    cfg.resources.asset_root = str(tmp_path / "assets")
    spec = _spec(tmp_path, "img", np.zeros((12, 20484), dtype=np.float32))
    out = asyncio.run(CortexMaeTool(provider=RejectT()).run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.execution_status == "skipped"
    assert out.findings[0].code == "temporal_length_unsupported"
    assert out.metrics.get("padded") is False


def test_cortexmae_success_keeps_t12_limitation(tmp_path: Path) -> None:
    from react_agent.fmri.tools.cortex_mae import CortexMaeTool

    class AcceptT:
        name = "fake"
        trained_t = 16

        def missing_reasons(self):
            return []

        def encode_surface(self, tv):
            t = int(tv.shape[0])
            assert t == 12
            return np.ones(8), {
                "input_t": t,
                "trained_t": 16,
                "aggregation": "fake_mean",
                "padded": False,
                "model_id": "fake",
                "n_parcels": 400,
            }

    spec = _spec(tmp_path, "img", np.zeros((12, 20484), dtype=np.float32))
    cfg = load_config()
    out = asyncio.run(CortexMaeTool(provider=AcceptT()).run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.execution_status == "success"
    assert out.metrics["input_t"] == 12
    assert out.metrics["padded"] is False
    assert out.applicability == "partial"


def test_rsa_uses_cached_clip_not_tribe(tmp_path: Path) -> None:
    from react_agent.fmri.providers.clip_image import ClipImageProvider
    from react_agent.fmri.tools.semantic import SemanticConsistencyTool

    emb = tmp_path / "emb"
    emb.mkdir()
    provider = ClipImageProvider(asset_root=tmp_path / "assets", embedding_dir=emb)
    ids = ["a.jpg", "b.jpg", "c.jpg"]
    rng = np.random.default_rng(0)
    for i, image_id in enumerate(ids):
        vec = rng.normal(size=16).astype(np.float32)
        vec[i] += 3
        provider.save_cached(image_id, vec)
    manifest = tmp_path / "samples.jsonl"
    lines = []
    for i, image_id in enumerate(ids):
        arr = rng.normal(size=(4, 8)).astype(np.float32)
        arr[0, i] += 2
        npy = tmp_path / f"{image_id}.npy"
        np.save(npy, arr)
        spec = SampleSpec(
            sample_id=f"s{i}",
            fmri_path=str(npy),
            time_axis=0,
            sampling_interval_s=1.0,
            image_path=str(tmp_path / image_id),
            resources=SampleResources(image_id=image_id),
        )
        lines.append(spec.model_dump_json())
        (tmp_path / image_id).write_bytes(b"not-a-real-jpg")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = load_config()
    cfg.resources.asset_root = str(tmp_path / "assets")
    cfg.resources.image_embedding_dir = str(emb)
    cfg.resources.samples_manifest_path = str(manifest)
    query = SampleSpec.model_validate(json.loads(lines[0]))
    out = asyncio.run(SemanticConsistencyTool(provider=provider).run(query, {}, _ctx(tmp_path, query, cfg)))
    assert out.execution_status == "success"
    assert out.metrics["used_tribe_hf_cache"] is False
    assert out.metrics["n"] == 3
    assert out.metrics["spearman_rdm"] is not None


def test_unused_cortexmae_variants_stay_disabled(tmp_path: Path) -> None:
    from react_agent.fmri.tools.future import DisabledTool, UNUSED_FLAT_SPEC, UNUSED_VOLUME_SPEC

    cfg = load_config()
    for spec in (UNUSED_FLAT_SPEC, UNUSED_VOLUME_SPEC):
        assert spec.enabled is False
        dummy = _spec(tmp_path, spec.name, np.zeros((4, 8), dtype=np.float32))
        out = asyncio.run(DisabledTool(spec).run(dummy, {}, _ctx(tmp_path, dummy, cfg)))
        assert out.execution_status == "skipped"


def test_wrong_atlas_length_rejected(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    np.save(atlas / "labels_lh_rh.npy", np.arange(10, dtype=np.int32))
    (atlas / "manifest.json").write_text(
        json.dumps(
            {
                "atlas_id": "destrieux_fsaverage5_lh_rh",
                "hemisphere_order": ["lh", "rh"],
            }
        )
    )
    cfg = load_config()
    cfg.resources.atlas_dir = str(atlas)
    spec = _spec(tmp_path, "img", np.zeros((4, 20484), dtype=np.float32))
    out = asyncio.run(SurfaceRoiProfileTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.execution_status == "error"
    assert out.findings[0].code == "atlas_mismatch"


def test_incompatible_control_not_trusted(tmp_path: Path) -> None:
    g = np.zeros((12, 8), dtype=np.float32)
    np.save(tmp_path / "gray_control.npy", g)
    y = np.ones((12, 8), dtype=np.float32)
    cfg = load_config()
    cfg.resources.gray_control_id = "gray_control"
    cfg.resources.gray_control_path = str(tmp_path / "gray_control.npy")
    spec = _spec(
        tmp_path,
        "img",
        y,
        resources=SampleResources(control_sample_id="gray_control"),
    )
    (tmp_path / "gray_control.json").write_text(
        json.dumps({"generation_profile_id": "other_profile", "mesh": "fsaverage5", "normalization": "none_raw_signed"})
    )
    out = asyncio.run(GrayControlContrastTool().run(spec, {}, _ctx(tmp_path, spec, cfg)))
    assert out.applicability == "partial"
    assert out.metrics.get("comparable") is False


def test_duplicate_and_no_self_match(tmp_path: Path) -> None:
    a = np.ones((4, 6), dtype=np.float32)
    b = np.ones((4, 6), dtype=np.float32)
    specs = []
    lines = []
    for name, arr in (("a", a), ("b", b)):
        spec = _spec(tmp_path, name, arr, resources=SampleResources(image_id=name))
        specs.append(spec)
        lines.append(json.dumps(spec.model_dump(mode="json")))
    man = tmp_path / "panel.jsonl"
    man.write_text("\n".join(lines) + "\n")
    idx_path = tmp_path / "idx.json"
    build_cohort_index(man, idx_path)
    cfg = load_config()
    cfg.resources.cohort_index_path = str(idx_path)
    # same bytes → duplicate hash if fingerprints match; query uses file fingerprint
    ctx = _ctx(tmp_path, specs[0], cfg)
    # Force same fingerprint as index member b by rewriting index fingerprints
    payload = json.loads(idx_path.read_text())
    payload["members"][1]["fingerprint"] = ctx.fingerprint
    idx_path.write_text(json.dumps(payload))
    out = asyncio.run(CrossImageSpecificityTool().run(specs[0], {}, ctx))
    assert out.metrics["self_excluded"] is True
    assert specs[0].sample_id not in [n["sample_id"] for n in out.metrics["nearest_neighbors"]]
    assert "b" in out.metrics["exact_duplicates"]


def test_temporal_prefers_contrast(tmp_path: Path) -> None:
    y = np.zeros((8, 8), dtype=np.float32)
    spec = _spec(tmp_path, "img", y)
    cfg = load_config()
    contrast = np.zeros((8, 8), dtype=np.float32)
    contrast[2] = 5.0
    cpath = tmp_path / "contrast_minus_gray.npy"
    np.save(cpath, contrast)
    prior = [
        {
            "tool_name": "gray_control_contrast",
            "execution_status": "success",
            "applicability": "applicable",
            "artifacts": [{"name": "contrast_minus_gray", "path": str(cpath)}],
        }
    ]
    out = asyncio.run(
        StimulusTemporalProfileTool().run(spec, {}, _ctx(tmp_path, spec, cfg, prior=prior))
    )
    assert out.metrics["mode"] == "contrast"
    assert out.metrics["peak_frame"] == 2


def test_hybrid_two_candidates_and_second_sees_first(tmp_path: Path, demo_dir=None) -> None:
    from react_agent.fmri.demo import make_demo
    from react_agent.fmri.llm.mock import MockBackend

    demo = tmp_path / "demo"
    make_demo(demo)
    g = np.load(demo / "arrays" / "ok_numeric.npy")
    np.save(tmp_path / "gray_control.npy", np.zeros_like(g))
    cfg = load_config()
    cfg.enabled_tools = list(cfg.enabled_tools) + [
        "gray_control_contrast",
        "stimulus_temporal_profile",
    ]
    cfg.resources.gray_control_id = "gray_control"
    cfg.resources.gray_control_path = str(tmp_path / "gray_control.npy")
    spec = SampleSpec.model_validate_json((demo / "ok_numeric.json").read_text())
    spec = spec.model_copy(
        update={"resources": SampleResources(control_sample_id="gray_control")}
    )
    mock = MockBackend(
        script=[
            {
                "action": "run_tool",
                "tool_name": "gray_control_contrast",
                "reason": "Need gray contrast.",
            },
            {
                "action": "run_tool",
                "tool_name": "stimulus_temporal_profile",
                "reason": "Saw contrast, now time.",
            },
            {"action": "stop", "stop_reason": "configured_checks_complete", "reason": "done"},
        ]
    )
    report = asyncio.run(
        run_sample(
            spec,
            base_dir=demo,
            out_dir=tmp_path / "hyb",
            config=cfg,
            backend="mock",
            policy="hybrid",
            mock=mock,
        )
    )
    assert "gray_control_contrast" in report["executed_tools"]
    assert "stimulus_temporal_profile" in report["executed_tools"]
    events = [json.loads(l) for l in (tmp_path / "hyb" / "events.jsonl").read_text().splitlines() if l]
    decisions = [e for e in events if e.get("type") == "decision"]
    assert any(len(e.get("candidates") or []) >= 2 for e in decisions)
    assert len(decisions) >= 2
