"""CLI: make-demo, fit-reference, check, batch."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from react_agent.fmri.config import load_config, redacted_config, resolve_policy_backend
from react_agent.fmri.data import load_manifest, load_sample_spec
from react_agent.fmri.demo import make_demo
from react_agent.fmri.llm.mock import MockBackend
from react_agent.fmri.loop import run_sample
from react_agent.fmri.reporting import write_summary_csv
from react_agent.fmri.tools.reference import fit_reference
from react_agent.fmri.tribe_adapter import convert_ids, write_samples_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.fmri.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("make-demo", help="Write synthetic demo fixtures")
    demo.add_argument("--out", type=Path, required=True)

    fit = sub.add_parser("fit-reference", help="Fit frozen reference stats")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--config", type=Path, default=None)
    fit.add_argument("--reference-type", default="demo_synthetic")

    check = sub.add_parser("check", help="Inspect one sample")
    check.add_argument("--sample", type=Path, required=True)
    check.add_argument("--config", type=Path, default=None)
    check.add_argument("--backend", choices=["mock", "deepseek", "none"], default=None)
    check.add_argument("--policy", choices=["rule", "hybrid", "planned"], default=None)
    check.add_argument("--out", type=Path, required=True)

    batch = sub.add_parser("batch", help="Inspect a JSONL manifest")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--config", type=Path, default=None)
    batch.add_argument("--backend", choices=["mock", "deepseek", "none"], default=None)
    batch.add_argument("--policy", choices=["rule", "hybrid", "planned"], default=None)
    batch.add_argument("--out", type=Path, required=True)

    tribe = sub.add_parser(
        "from-tribe",
        help="Convert TRIBE sidecars to SampleSpec JSONL (raw preds only)",
    )
    tribe.add_argument("--preds-dir", type=Path, required=True)
    tribe.add_argument("--ids", nargs="+", required=True, help="Raw pred stems")
    tribe.add_argument("--out", type=Path, required=True)
    tribe.add_argument("--control-id", default=None)
    tribe.add_argument("--cohort-manifest-id", default=None)

    assets = sub.add_parser("prepare-assets", help="Prepare surface atlas (explicit download)")
    assets.add_argument("--atlas-dir", type=Path, required=True)
    assets.add_argument("--download", action="store_true")
    assets.add_argument("--synthetic", action="store_true")
    assets.add_argument("--left", type=Path, default=None)
    assets.add_argument("--right", type=Path, default=None)

    cohort = sub.add_parser("build-cohort-index", help="Build a fixed cohort panel index")
    cohort.add_argument("--manifest", type=Path, required=True)
    cohort.add_argument("--out", type=Path, required=True)

    p2 = sub.add_parser("prepare-p2", help="Download P2 assets into EEGagent/assets")
    p2.add_argument("--kind", choices=["schaefer", "cortexmae", "clip", "all"], default="all")
    p2.add_argument("--asset-root", type=Path, default=None)
    p2.add_argument("--download", action="store_true")

    cache = sub.add_parser("cache-embeddings", help="Write frozen CLIP vectors for a manifest")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--asset-root", type=Path, default=None)

    prep_vid = sub.add_parser("prepare-image-video", help="Write 16s protocol video; does not load TRIBE")
    prep_vid.add_argument("--image", type=Path, required=True)
    prep_vid.add_argument("--out", type=Path, required=True)
    prep_vid.add_argument("--config", type=Path, default=None)
    prep_vid.add_argument("--sample-id", default=None)

    gen_img = sub.add_parser("generate-from-image", help="TRIBE generate 16s preds; no check")
    gen_img.add_argument("--image", type=Path, required=True)
    gen_img.add_argument("--out", type=Path, required=True)
    gen_img.add_argument("--config", type=Path, default=None)
    gen_img.add_argument("--sample-id", default=None)
    gen_img.add_argument("--generation-backend", choices=["worker", "mock"], default=None)

    chk_img = sub.add_parser("check-image", help="Generate if needed, then run the existing checker")
    chk_img.add_argument("--image", type=Path, required=True)
    chk_img.add_argument("--out", type=Path, required=True)
    chk_img.add_argument("--config", type=Path, default=None)
    chk_img.add_argument("--sample-id", default=None)
    chk_img.add_argument("--backend", choices=["mock", "deepseek", "none"], default=None)
    chk_img.add_argument("--policy", choices=["rule", "hybrid", "planned"], default=None)
    chk_img.add_argument("--generation-backend", choices=["worker", "mock"], default=None)
    chk_img.add_argument("--require-check", action="append", default=None)

    batch_img = sub.add_parser("batch-images", help="Generate+check a JSONL of image paths")
    batch_img.add_argument("--manifest", type=Path, required=True)
    batch_img.add_argument("--out", type=Path, required=True)
    batch_img.add_argument("--config", type=Path, default=None)
    batch_img.add_argument("--backend", choices=["mock", "deepseek", "none"], default=None)
    batch_img.add_argument("--policy", choices=["rule", "hybrid", "planned"], default=None)
    batch_img.add_argument("--generation-backend", choices=["worker", "mock"], default=None)

    probe = sub.add_parser("probe-backend", help="Cheap DeepSeek JSON ping before generation")
    probe.add_argument("--config", type=Path, default=None)
    probe.add_argument("--out", type=Path, default=None)

    mem_ins = sub.add_parser("memory-inspect", help="List stored episodes and candidates")
    mem_ins.add_argument("--config", type=Path, default=None)
    mem_ins.add_argument("--namespace", default=None)
    mem_ins.add_argument("--limit", type=int, default=20)

    mem_exp = sub.add_parser("memory-export", help="Export memory JSONL")
    mem_exp.add_argument("--config", type=Path, default=None)
    mem_exp.add_argument("--out", type=Path, required=True)
    mem_exp.add_argument("--namespace", default=None)

    mem_fb = sub.add_parser("record-feedback", help="Attach verification/feedback to an episode")
    mem_fb.add_argument("--config", type=Path, default=None)
    mem_fb.add_argument("--episode-id", required=True)
    mem_fb.add_argument("--status", required=True, choices=[
        "confirmed_issue",
        "confirmed_clear",
        "overturned",
        "inconclusive",
    ])
    mem_fb.add_argument("--scope", default="numeric_consistency_only")
    mem_fb.add_argument("--evidence", default="")
    mem_fb.add_argument("--source", default="human")
    return parser


def _resolve_args_policy(args: argparse.Namespace, cfg) -> tuple[str, str]:
    try:
        return resolve_policy_backend(cfg, policy=args.policy, backend=args.backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


async def _run_one(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = load_config(args.config)
    policy, backend = _resolve_args_policy(args, cfg)

    sample_path: Path = args.sample
    spec = load_sample_spec(sample_path)
    mock = None
    if backend == "mock":
        mock = MockBackend(
            script=[
                {
                    "action": "run_tool",
                    "tool_name": "temporal_diagnostics",
                    "reason": "Need temporal localization.",
                    "evidence_refs": [],
                },
                {
                    "action": "stop",
                    "stop_reason": "configured_checks_complete",
                    "reason": "Optional temporal check finished.",
                },
            ]
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "resolved_config.json").write_text(
        json.dumps(redacted_config(cfg), indent=2), encoding="utf-8"
    )
    report = await run_sample(
        spec,
        base_dir=sample_path.parent,
        out_dir=args.out,
        config=cfg,
        backend=backend,
        policy=policy,
        mock=mock,
    )
    print(
        json.dumps(
            {
                "sample_id": report.get("sample_id"),
                "verdict": report.get("verdict"),
                "workflow_html": (report.get("workflow") or {}).get("html_path"),
            }
        )
    )


async def _run_batch(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = load_config(args.config)
    policy, backend = _resolve_args_policy(args, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "resolved_config.json").write_text(
        json.dumps(redacted_config(cfg), indent=2), encoding="utf-8"
    )
    batch_ledger = {"lm_calls": 0}
    rows = []
    for spec, base in load_manifest(args.manifest):
        mock = None
        if backend == "mock":
            mock = MockBackend(
                script=[
                    {
                        "action": "run_tool",
                        "tool_name": "temporal_diagnostics",
                        "reason": "Need temporal localization.",
                        "evidence_refs": [],
                    },
                    {
                        "action": "stop",
                        "stop_reason": "configured_checks_complete",
                        "reason": "Done.",
                    },
                ]
            )
        try:
            report = await run_sample(
                spec,
                base_dir=base,
                out_dir=args.out,
                config=cfg,
                backend=backend,
                policy=policy,
                mock=mock,
                batch_ledger=batch_ledger,
            )
        except Exception as exc:  # noqa: BLE001
            report = {
                "sample_id": spec.sample_id,
                "verdict": "inconclusive",
                "coverage_status": "partial",
                "stop_reason": f"sample_error:{type(exc).__name__}",
                "executed_tools": [],
                "cost": {},
                "fallback_used": False,
            }
        rows.append(
            {
                "sample_id": report.get("sample_id"),
                "verdict": report.get("verdict"),
                "coverage": report.get("coverage_status"),
                "workflow_html": (report.get("workflow") or {}).get("html_path"),
                "stop_reason": report.get("stop_reason"),
                "checks": "|".join(report.get("executed_tools") or []),
                "tool_calls": (report.get("cost") or {}).get("tool_calls"),
                "lm_calls": (report.get("cost") or {}).get("lm_calls"),
                "input_tokens": (report.get("cost") or {}).get("input_tokens"),
                "output_tokens": (report.get("cost") or {}).get("output_tokens"),
                "api_usd": (report.get("cost") or {}).get("api_usd"),
                "elapsed": (report.get("cost") or {}).get("elapsed_seconds"),
                "fallback_used": report.get("fallback_used"),
            }
        )
        print(f"{report.get('sample_id')} {report.get('verdict')}")
    write_summary_csv(args.out / "summary.csv", rows)


async def _prepare_image_video(args: argparse.Namespace) -> None:
    from react_agent.fmri.generation.schemas import StimulusProfile
    from react_agent.fmri.generation.video import probe_video, write_image_video

    cfg = load_config(args.config) if args.config else load_config()
    profile = StimulusProfile(
        profile_id=cfg.generation.profile_id,
        fps=cfg.generation.fps,
        pre_gray_s=cfg.generation.pre_gray_s,
        image_s=cfg.generation.image_s,
        post_gray_s=cfg.generation.post_gray_s,
        gray_rgb=tuple(cfg.generation.gray_rgb),
        codec=cfg.generation.codec,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    sample_id = args.sample_id or args.image.stem
    video_path = args.out / f"{sample_id}.mp4"
    _, size_hw, report = write_image_video(
        args.image,
        video_path,
        profile,
        python_executable=cfg.generation.python_executable,
        repo_path=cfg.generation.repo_path,
    )
    probe = probe_video(video_path, profile)
    payload = {
        "sample_id": sample_id,
        "video_path": str(video_path.resolve()),
        "size_hw": list(size_hw),
        "frames": report,
        "probe": probe,
        "profile": profile.to_dict(),
        "tribe_loaded": False,
    }
    (args.out / f"{sample_id}.video.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


async def _generate_from_image(args: argparse.Namespace) -> None:
    from react_agent.fmri.generation.schemas import ImageRequest
    from react_agent.fmri.pipeline import materialize
    from react_agent.fmri.progress import Progress, bind_progress

    load_dotenv()
    cfg = load_config(args.config) if args.config else load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    with bind_progress(Progress(args.out / "events.jsonl")):
        bundle = materialize(
            ImageRequest(image_path=str(args.image.resolve()), sample_id=args.sample_id),
            cfg,
            backend_override=args.generation_backend,
        )
    payload = bundle.summary()
    (args.out / "generation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    spec_path = args.out / f"{bundle.sample_id}.sample.json"
    spec_path.write_text(bundle.sample.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


async def _check_image(args: argparse.Namespace) -> None:
    from react_agent.fmri.pipeline import run_image_pipeline

    load_dotenv()
    cfg = load_config(args.config) if args.config else load_config()
    policy, backend = _resolve_args_policy(args, cfg)
    report = await run_image_pipeline(
        args.image,
        config=cfg,
        out_dir=args.out,
        sample_id=args.sample_id,
        backend=backend,
        policy=policy,
        generation_backend=args.generation_backend,
        require_checks=args.require_check,
    )
    print(
        json.dumps(
            {
                "sample_id": report.get("sample_id"),
                "pipeline_status": report.get("pipeline_status"),
                "check_verdict": report.get("check_verdict"),
                "verdict": report.get("verdict"),
                "workflow_html": (report.get("workflow") or {}).get("html_path"),
            }
        )
    )


async def _batch_images(args: argparse.Namespace) -> None:
    from react_agent.fmri.pipeline import run_image_pipeline

    load_dotenv()
    cfg = load_config(args.config) if args.config else load_config()
    policy, backend = _resolve_args_policy(args, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image = Path(rec["image_path"])
            sample_id = rec.get("sample_id") or image.stem
            report = await run_image_pipeline(
                image,
                config=cfg,
                out_dir=args.out / sample_id,
                sample_id=sample_id,
                backend=backend,
                policy=policy,
                generation_backend=args.generation_backend,
            )
            rows.append(
                {
                    "sample_id": report.get("sample_id"),
                    "pipeline_status": report.get("pipeline_status"),
                    "verdict": report.get("verdict"),
                    "check_verdict": report.get("check_verdict"),
                    "stop_reason": report.get("stop_reason"),
                }
            )
            print(f"{report.get('sample_id')} {report.get('pipeline_status')} {report.get('verdict')}")
    write_summary_csv(args.out / "summary.csv", rows)


async def _probe_backend(args: argparse.Namespace) -> None:
    from react_agent.fmri.llm.deepseek import DeepSeekBackend
    from react_agent.fmri.planning.service import probe_provider

    load_dotenv()
    cfg = load_config(args.config)
    if not cfg.deepseek_api_key:
        raise SystemExit("probe-backend requires DEEPSEEK_API_KEY")
    backend = DeepSeekBackend(cfg)
    payload = await probe_provider(backend, cfg)
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "probe.json").write_text(text, encoding="utf-8")
    print(text)


def _memory_inspect(args: argparse.Namespace) -> None:
    from react_agent.fmri.memory.service import open_memory

    load_dotenv()
    cfg = load_config(args.config)
    if args.namespace:
        cfg.memory.namespace = args.namespace
    repo = open_memory(cfg)
    print(json.dumps(repo.inspect(limit=args.limit), indent=2, default=str))


def _memory_export(args: argparse.Namespace) -> None:
    from react_agent.fmri.memory.service import open_memory

    load_dotenv()
    cfg = load_config(args.config)
    if args.namespace:
        cfg.memory.namespace = args.namespace
    repo = open_memory(cfg)
    path = repo.export_jsonl(args.out)
    print(json.dumps({"exported": str(path)}))


def _record_feedback(args: argparse.Namespace) -> None:
    from react_agent.fmri.memory.service import open_memory

    load_dotenv()
    cfg = load_config(args.config)
    repo = open_memory(cfg)
    vid = repo.record_verification(
        {
            "episode_id": args.episode_id,
            "verification_status": args.status,
            "verification_scope": args.scope,
            "evidence": args.evidence,
            "source": args.source,
        }
    )
    print(json.dumps({"verification_id": vid, "episode_id": args.episode_id}))


def main() -> None:
    """CLI entry."""
    args = _parser().parse_args()
    if args.cmd == "make-demo":
        make_demo(args.out)
        print(f"wrote demo to {args.out}")
        return
    if args.cmd == "from-tribe":
        specs = convert_ids(
            args.preds_dir,
            args.ids,
            control_sample_id=args.control_id,
            cohort_manifest_id=args.cohort_manifest_id,
        )
        write_samples_jsonl(specs, args.out)
        print(f"wrote {len(specs)} samples to {args.out}")
        return
    if args.cmd == "prepare-assets":
        from react_agent.fmri.assets import (
            import_local_maps,
            prepare_destrieux_atlas,
            write_synthetic_atlas,
        )

        if args.synthetic:
            manifest = write_synthetic_atlas(args.atlas_dir)
        elif args.left and args.right:
            manifest = import_local_maps(args.atlas_dir, left=args.left, right=args.right)
        else:
            manifest = prepare_destrieux_atlas(args.atlas_dir, allow_download=args.download)
        print(json.dumps(manifest, indent=2))
        return
    if args.cmd == "build-cohort-index":
        from react_agent.fmri.cohort import build_cohort_index

        info = build_cohort_index(args.manifest, args.out)
        print(json.dumps(info, indent=2))
        return
    if args.cmd == "prepare-p2":
        from react_agent.fmri.p2_store import ensure_layout
        from react_agent.fmri.providers.clip_image import download_clip
        from react_agent.fmri.providers.cortexmae_parcel import download_cortexmae
        from react_agent.fmri.providers.schaefer import prepare_schaefer400

        paths = ensure_layout(args.asset_root)
        kinds = ["schaefer", "cortexmae", "clip"] if args.kind == "all" else [args.kind]
        payload: dict[str, object] = {"asset_root": str(paths["root"]), "kinds": kinds}
        if not args.download:
            raise SystemExit("prepare-p2 requires --download (import/availability never fetch)")
        if "schaefer" in kinds:
            payload["schaefer"] = prepare_schaefer400(
                paths["schaefer"], asset_root=paths["root"], allow_download=True
            )
        if "cortexmae" in kinds:
            payload["cortexmae"] = download_cortexmae(asset_root=paths["root"])
        if "clip" in kinds:
            payload["clip"] = download_clip(asset_root=paths["root"])
        print(json.dumps(payload, indent=2, default=str))
        return
    if args.cmd == "cache-embeddings":
        from react_agent.fmri.providers.clip_image import cache_manifest_embeddings

        info = cache_manifest_embeddings(args.manifest, asset_root=args.asset_root)
        print(json.dumps(info, indent=2))
        return
    if args.cmd == "fit-reference":
        load_dotenv()
        cfg = load_config(args.config)
        asyncio.run(
            fit_reference(
                args.manifest,
                args.out,
                config=cfg,
                reference_type=args.reference_type,
            )
        )
        print(f"wrote reference stats to {args.out}")
        return
    if args.cmd == "prepare-image-video":
        asyncio.run(_prepare_image_video(args))
        return
    if args.cmd == "generate-from-image":
        asyncio.run(_generate_from_image(args))
        return
    if args.cmd == "check-image":
        asyncio.run(_check_image(args))
        return
    if args.cmd == "batch-images":
        asyncio.run(_batch_images(args))
        return
    if args.cmd == "check":
        asyncio.run(_run_one(args))
        return
    if args.cmd == "batch":
        asyncio.run(_run_batch(args))
        return
    if args.cmd == "probe-backend":
        asyncio.run(_probe_backend(args))
        return
    if args.cmd == "memory-inspect":
        _memory_inspect(args)
        return
    if args.cmd == "memory-export":
        _memory_export(args)
        return
    if args.cmd == "record-feedback":
        _record_feedback(args)


if __name__ == "__main__":
    main()
