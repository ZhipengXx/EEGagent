"""Command line for one code-level campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from react_agent.eeg_research.agentic.contract import freeze_contract, research_scope
from react_agent.eeg_research.agentic.execution_protocol import ProtocolError, build_execution_protocol
from react_agent.eeg_research.agentic.jobs import worker_disconnected
from react_agent.eeg_research.agentic.loop import (
    DEFAULT_MAX_GPU_SECONDS,
    align_interrupt,
    create_campaign,
    event,
    load_state,
    request_control,
    save_state,
)
from react_agent.eeg_research.agentic.schemas import GoalSpec
from react_agent.eeg_training.protocol import parse_subject_selection

DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "runs" / "eeg_research_v18"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.eeg_research.agentic.cli")
    parser.add_argument(
        "command",
        choices=["create", "start", "run", "status", "pause", "resume", "stop", "prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design", "authorize-gpu-sharing", "recover-api-intents", "validate-goal", "evaluate-export", "evaluate-pack",
                 "study-create", "study-status", "study-run", "study-freeze", "study-evaluate", "study-aggregate",
                 "suite-retry-fold"],
    )
    parser.add_argument("--campaign", default="eeg_retrieval_research_v1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--gpu", default="0", help="Comma-separated GPU indices for training jobs")
    parser.add_argument("--subject", type=parse_subject_selection, default="all",
                        help="Training subjects, comma-separated; LOSO selects nine source subjects")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-ticks", type=int, default=200,
                        help="Native worker step limit; run/start/resume only, positive integer")
    parser.add_argument("--reopen-reason", default="", help="Reopen a finished campaign after a framework fix; the reason is logged")
    parser.add_argument("--goal", type=Path, default=None, help="GoalSpec YAML. validate-goal prints it and does not start a worker.")
    parser.add_argument("--planner-mode", choices=["single_action", "compare_options"], default=None,
                        help="New campaign controller mode; persisted goals are authoritative on resume.")
    parser.add_argument("--pack", type=Path, default=None, help="Candidate pack directory for evaluate-export")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run evaluate-only for a pack. Default is dry-run argv.")
    parser.add_argument("--study", type=Path, default=None, help="Native sequential ten-fold LOSO study directory")
    parser.add_argument("--allow-partial", action="store_true", help="Freeze valid pairs with explicitly incomplete fold coverage")
    parser.add_argument("--allow-shared-gpus", action="store_true", help="Explicit authorization to use occupied GPUs without terminating their processes")
    parser.add_argument("--inherit-study", type=Path,
                        help="Native verified budget source: legacy study or quiescent method campaign")
    parser.add_argument("--additional-llm-calls", type=int, default=0,
                        help="Explicit additional authorization for inherited method-campaign API attempts; create only")
    parser.add_argument("--additional-training-jobs", type=int, default=0,
                        help="Explicit additional authorization for inherited actual training jobs; create only")
    parser.add_argument("--additional-gpu-seconds", type=float, default=0,
                        help="Explicit additional occupied-GPU seconds authorization; create only")
    parser.add_argument("--suite", default=None, help="Frozen method suite ID for an engineering retry")
    parser.add_argument("--fold", default=None, help="Fold ID for an engineering retry")
    parser.add_argument("--candidate", default=None, help="Existing checked candidate for review revalidation")
    parser.add_argument("--retry-reason", default="", help="Observed engineering failure requiring a fresh fold or feedback validation")
    parser.add_argument("--max-existing-gpu-memory-mib", type=int, default=1024,
                        help="Explicit authorize-gpu-sharing only: maximum memory already used on each frozen card")
    return parser


def load_goal_file(path: Path) -> dict:
    """Validate a GoalSpec YAML or JSON file. Sample 12/120/28800 values are not a run authorization."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("goal_not_object")
    spec = GoalSpec.model_validate(raw)
    return spec.model_dump()


def goal(campaign: str, design=None) -> dict:
    scope = "pooled_subject_retrieval" if design is None else research_scope(design)
    gpu = DEFAULT_MAX_GPU_SECONDS
    chosen = None if design is None else getattr(design, "gpu_seconds", None)
    if chosen is not None:
        gpu = float(chosen)
    return {
        "goal_id": campaign,
        "planner_mode": "compare_options",
        "objective": "改进指定数据与协议下的 EEG 到图像检索",
        "task_type": "eeg_image_retrieval",
        "research_scope": scope,
        "primary_metric": "validation.fixed_gallery_top1",
        "direction": "maximize",
        "min_practical_gain_pp": None,
        "final_test_enabled": False,
        "max_candidates": 4,
        "max_training_jobs": 20,
        "max_llm_calls": 300,
        "max_gpu_seconds": gpu,
        "max_concurrent_training_jobs": 1,
        "max_api_usd": None,
        "allowed_changes": ["eeg_encoder", "temporal_pooling", "projection_head", "training_loss", "training_only_augmentation"],
        "frozen_components": ["split_manifest", "query_gallery_manifest", "evaluator", "metric_definition", "image_feature_target"],
    }


def spawn_worker(root: Path, campaign: str, poll_seconds: float, *, max_ticks: int = 200) -> int:
    camp = root / campaign
    log = (camp / "worker.log").open("ab")
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "react_agent.eeg_research.agentic.cli", "run", "--campaign", campaign,
         "--root", str(root), "--poll-seconds", str(poll_seconds), "--max-ticks", str(max_ticks)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def open_agentic_run(
    design,
    payload: dict,
    *,
    request_id: str,
    root: Path,
    spawn=spawn_worker,
) -> dict:
    """Create one new v18 campaign for this click and start its worker. A repeated request id does not start a second one."""
    import hashlib

    from react_agent.eeg_research.agentic.contract import freeze_contract
    from react_agent.eeg_training.protocol import data_root

    result = dict(payload)
    if result.get("blockers") or not request_id:
        return result
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
    campaign = f"eeg_ui_{digest}"
    index = root / "request_index.json"
    known = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
    already = request_id in known
    base = Path(design.data_root) if design.data_root else data_root()
    try:
        protocol = build_execution_protocol(design, base)
    except ProtocolError as exc:
        result["ok"] = False
        result["blockers"] = list(result.get("blockers") or []) + [str(exc)]
        result["log"] = list(result.get("log") or []) + [str(exc)]
        return result
    contract = freeze_contract(design, base)
    contract["execution_fingerprint"] = protocol["fingerprint"]
    state = create_campaign(root, goal=goal(campaign, design), contract=contract, request_id=request_id, protocol=protocol)
    camp = root / campaign
    if not already:
        state["gpu"] = [int(item) for item in (design.gpu or (0,))]
        save_state(camp, state)
        spawn(root, campaign, 30.0)
    result["started"] = True
    result["campaign_written"] = False
    result["campaign_id"] = campaign
    result["agentic_campaign"] = campaign
    note = f"代码级研究已启动：{campaign}"
    result["log"] = list(result.get("log") or []) + [note]
    return result


def status_view(camp: Path) -> dict:
    state = load_state(camp)
    status = state.get("status")
    detail = state.get("detail")
    if status not in {"paused", "finished", "blocked", "cancelled"}:
        worker = {}
        worker_path = camp / "worker.json"
        if worker_path.is_file():
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
        pid = int(worker.get("pid") or 0)
        if pid > 0 and worker_disconnected(pid):
            status = "interrupted"
            detail = "worker 已退出"
    return {
        "status": status,
        "detail": detail,
        "live_job": state.get("live_job"),
        "training_jobs": state.get("training_jobs"),
        "llm_calls": state.get("llm_calls"),
        "gpu_seconds_left": state.get("gpu_seconds_left"),
        "candidates": state.get("candidates"),
        "last_decision": (state.get("decisions") or [None])[-1],
        "evidence": [
            {key: row.get(key) for key in ("evidence_id", "kind", "candidate_id", "fidelity", "evaluation_valid", "fixed_bank_top1", "delta_vs_control_pp")}
            for row in state.get("evidence") or []
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_ticks < 1:
        print(json.dumps({"error": "max_ticks_must_be_positive"}))
        return 2
    additional = {"llm_calls": args.additional_llm_calls,
                  "training_jobs": args.additional_training_jobs,
                  "gpu_seconds": args.additional_gpu_seconds}
    if any(additional.values()) and (args.command != "create" or args.inherit_study is None):
        print(json.dumps({"error": "additional_authorization_requires_new_inherited_method_campaign"}))
        return 2
    if args.command.startswith("study-"):
        from react_agent.eeg_research.agentic import loso_study
        from react_agent.eeg_training.protocol import data_root, parse_gpu_list

        if args.study is None:
            print(json.dumps({"error": "study_directory_required"}))
            return 2
        study = args.study.resolve()
        try:
            if args.command == "study-create":
                if args.goal is None:
                    raise ValueError("study_goal_template_required")
                payload = loso_study.create(study, load_goal_file(args.goal),
                    args.data_root if args.data_root is not None else data_root(), parse_gpu_list(args.gpu),
                    allow_shared_gpus=args.allow_shared_gpus)
            elif args.command == "study-freeze":
                payload = loso_study.freeze(study, allow_partial=args.allow_partial)
            else:
                operation = {"study-status": loso_study.status, "study-run": loso_study.advance,
                             "study-evaluate": loso_study.evaluate, "study-aggregate": loso_study.aggregate}[args.command]
                payload = operation(study)
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc), "command": args.command}, ensure_ascii=False))
            return 2
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-goal":
        if args.goal is None:
            print(json.dumps({"error": "goal_missing"}, ensure_ascii=False))
            return 2
        try:
            payload = load_goal_file(args.goal)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        from react_agent.eeg_research.agentic.capabilities import capability_manifest

        print(
            json.dumps(
                {
                    "ok": True,
                    "spawn_worker": False,
                    "note": "sample YAML is not a run authorization",
                    "goal": payload,
                    "capabilities": capability_manifest(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command in {"evaluate-export", "evaluate-pack"}:
        from react_agent.eeg_research.agentic.export import evaluate_only_argv, pack_is_rebuildable

        pack = args.pack
        if pack is None:
            print(json.dumps({"error": "pack_missing"}))
            return 2
        ok, reason = pack_is_rebuildable(pack)
        if not ok:
            print(json.dumps({"error": reason}, ensure_ascii=False))
            return 2
        out = args.out or (pack / "eval_only")
        argv_train = evaluate_only_argv(pack, out=out, data_root=args.data_root)
        payload = {
            "ok": True,
            "evaluate_only": True,
            "final_test": False,
            "pack": str(pack),
            "out": str(out),
            "rebuild_encoder": True,
            "argv": argv_train,
            "dry_run": not args.execute,
            "note": "dry-run prints train_entry argv; --execute runs evaluate-only without fitting or reading final holdout",
        }
        if args.execute:
            import os

            if not os.environ.get("EEG_ALLOW_EVALUATE_PACK"):
                payload["ok"] = False
                payload["error"] = "prerequisite_missing:set EEG_ALLOW_EVALUATE_PACK=1; CPU evaluate uses EEG_TRAIN_DEVICE=cpu"
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 2
            try:
                from react_agent.eeg_research.adapters.ubp_retrieval import child_env

                env = child_env()
                env.pop("EEG_CANDIDATE_MODULE", None)
                env.pop("EEG_CANDIDATE_PATH", None)
                cwd = os.environ.get("EEG_EVALUATE_PACK_CWD")
                completed = subprocess.run(  # noqa: S603
                    argv_train,
                    check=False,
                    env=env,
                    cwd=cwd if cwd else None,
                )
            except OSError as exc:
                payload["ok"] = False
                payload["error"] = f"prerequisite_missing:{exc}"
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 2
            payload["returncode"] = completed.returncode
            payload["ok"] = completed.returncode == 0
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if completed.returncode == 0 else 2
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    root = args.root.resolve()
    camp = root / args.campaign
    if args.command == "create":
        selected_gpus = tuple(int(item) for item in args.gpu.split(",") if item.strip())
        design = __import__("react_agent.eeg_training.protocol", fromlist=["Design"]).Design(
            "eeg", "inter-subject", args.subject, gpu=selected_gpus, policy="agentic", training_strategy="pooled_subjects"
        )
        from react_agent.eeg_training.protocol import data_root

        base = args.data_root if args.data_root is not None else data_root()
        submitted = goal(args.campaign, design)
        if args.goal is not None:
            submitted = {**submitted, **load_goal_file(args.goal)}
            submitted["goal_id"] = args.campaign
        if args.planner_mode is not None:
            submitted["planner_mode"] = args.planner_mode
        if submitted.get("evaluation_mode") == "loso_method_search":
            submitted["allow_shared_gpus"] = args.allow_shared_gpus
        inherited = None
        estimate_basis = []
        if submitted.get("evaluation_mode") == "loso_method_search":
            from dataclasses import replace
            from react_agent.eeg_research.agentic.budget_inheritance import verified_history, derive_limits, estimate_suite
            from react_agent.eeg_research.agentic.loso_study import SUBJECTS
            if args.inherit_study is None:
                print(json.dumps({"error": "method_search_requires_native_budget_inheritance"}))
                return 2
            try:
                inherited = verified_history(args.inherit_study, additional_authorization=additional)
            except (OSError, ValueError, KeyError) as exc:
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
                return 2
            submitted = derive_limits(submitted, inherited)
            estimate, estimate_basis = estimate_suite(inherited, len(selected_gpus))
            submitted["method_suite_gpu_seconds_estimate"] = max(estimate, submitted.get("method_suite_gpu_seconds_estimate") or 0)
            design = replace(design, subject=",".join(SUBJECTS[:-1]), epochs=50, stop="single_full", seed=0,
                             data_root=str(base.resolve()), evaluation_mode="loso_method_search",
                             checkpoint_policy=submitted["checkpoint_policy"], selection_min_delta=submitted["selection_min_delta"])
        elif args.inherit_study is not None:
            print(json.dumps({"error": "budget_inheritance_requires_method_search"}))
            return 2
        try:
            protocol = build_execution_protocol(
                design,
                base,
                training_seeds=submitted.get("training_seeds") or None,
            )
        except ProtocolError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        contract = freeze_contract(design, base)
        contract["execution_fingerprint"] = protocol["fingerprint"]
        state = create_campaign(
            root,
            goal=submitted,
            contract=contract,
            request_id=args.request_id or args.campaign,
            protocol=protocol,
        )
        state["gpu"] = list(design.gpu)
        if inherited is not None:
            from react_agent.eeg_research.agentic.budget_inheritance import claim_successor
            from react_agent.eeg_research.agentic.method_suite import create_manifest, manifest
            claim_successor(camp, inherited)
            if not (camp / "method_search_manifest.json").exists():
                create_manifest(camp, design, allow_shared_gpus=args.allow_shared_gpus,
                                estimate=submitted["method_suite_gpu_seconds_estimate"])
                from react_agent.eeg_research.agentic.loso_study import write_once
                write_once(camp / "suite_cost_basis.json", {"basis": estimate_basis, "estimate": submitted["method_suite_gpu_seconds_estimate"]})
            else:
                manifest(camp)
        save_state(camp, state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "fingerprint": state["contract_fingerprint"],
                    "spawn_worker": False,
                    "max_training_jobs": state.get("max_training_jobs"),
                    "max_llm_calls": state.get("max_llm_calls"),
                    "max_gpu_seconds": state.get("max_gpu_seconds"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not (camp / "campaign_state.json").is_file():
        print(json.dumps({"error": "campaign_missing"}))
        return 2
    if args.command in {"start", "resume", "run", "prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design", "authorize-gpu-sharing", "recover-api-intents"}:
        from react_agent.eeg_research.agentic.budget_inheritance import delegated_successor
        successor = delegated_successor(camp)
        if successor:
            print(json.dumps({"error": "campaign_budget_delegated_to_successor",
                              "successor": successor}))
            return 2
    if args.command in {"start", "resume", "run"} and (root.parent / "benchmark_freeze.json").is_file():
        manifest_path = root.parent / "study_manifest.json"
        if manifest_path.is_file():
            study = json.loads(manifest_path.read_text(encoding="utf-8"))
            if root.name == "campaigns" and any(row.get("campaign_id") == camp.name for row in study.get("folds") or []):
                print(json.dumps({"error": "study_research_closed_after_freeze"}))
                return 2
    if args.command == "run":
        from react_agent.eeg_research.agentic.worker import run_worker

        state = run_worker(camp, poll_seconds=args.poll_seconds, max_ticks=args.max_ticks)
        print(json.dumps({"status": state.get("status"), "detail": state.get("detail")}, ensure_ascii=False))
        return 0
    if args.command == "status":
        print(json.dumps(status_view(camp), ensure_ascii=False, indent=2))
        return 0
    state = load_state(camp)
    if args.command == "authorize-gpu-sharing":
        import fcntl
        from react_agent.eeg_research.agentic.resource_guard import authorize_low_memory_sharing
        with (camp / "worker.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt = authorize_low_memory_sharing(camp, state,
                    max_memory_mib=args.max_existing_gpu_memory_mib, reason=args.retry_reason)
                event(camp, "gpu_sharing_authorized", authorization_sha256=receipt["sha256"],
                    gpu=receipt["gpu"], max_existing_memory_mib=receipt["max_existing_memory_mib"],
                    reason=receipt["reason"])
            except (BlockingIOError, OSError, ValueError, KeyError) as exc:
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
                return 2
        print(json.dumps({"status": "gpu_sharing_authorized", "authorization": receipt}, ensure_ascii=False))
        return 0
    if args.command == "recover-api-intents":
        import fcntl
        from react_agent.eeg_research.agentic.api_recovery import abandon_orphaned_intents
        with (camp / "worker.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                cost = abandon_orphaned_intents(camp, reason=args.retry_reason)
            except (BlockingIOError, OSError, ValueError, RuntimeError) as exc:
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
                return 2
            state["llm_calls"] = cost["llm_calls"]
            state["llm_calls_left"] = state["max_llm_calls"] - cost["llm_calls"]
            save_state(camp, state)
            event(camp, "orphaned_api_intents_abandoned", reason=args.retry_reason,
                  call_ids=cost["api_abandoned_call_ids"], unknown_outcomes_preserved=True)
        print(json.dumps({"status": "recovered", "llm_calls": cost["llm_calls"],
                          "unknown_outcome_call_ids": cost["api_abandoned_call_ids"]}))
        return 0
    if args.command in {"prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design"}:
        import fcntl
        from react_agent.eeg_research.agentic.worker import build_services
        from react_agent.eeg_research.agentic.llm import LlmUnavailable
        from react_agent.eeg_research.agentic.budget_inheritance import validate_inheritance
        from react_agent.eeg_research.agentic.verification_cache import verification_scope
        revalidate = args.command == "revalidate-feedback"
        retry_design = args.command == "retry-design"
        revalidate_review = args.command == "revalidate-review"
        if (revalidate or retry_design or revalidate_review) and not args.retry_reason.strip():
            print(json.dumps({"error": "feedback_revalidation_reason_required"}))
            return 2
        with (camp / "worker.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"error": "worker_already_running"}))
                return 2
            try:
                state = load_state(camp)
                with verification_scope():
                    validate_inheritance(camp)
                    services = build_services(camp)
                    if retry_design:
                        result = services["retry_design"](camp, state, reason=args.retry_reason)
                    elif revalidate_review:
                        result = services["revalidate_review"](camp, state,
                            candidate_id=args.candidate or "", reason=args.retry_reason)
                    else:
                        result = (services["revalidate_feedback"](camp, state, reason=args.retry_reason)
                                  if revalidate else services["prepare_handoff"](camp, state))
                save_state(camp, state)
                event(camp, "review_revalidated" if revalidate_review else "designer_retried" if retry_design else "feedback_revalidated" if revalidate else "handoff_prepared", **result)
            except (LlmUnavailable, OSError, ValueError, KeyError) as exc:
                save_state(camp, state)
                event(camp, "review_revalidation_failed" if revalidate_review else "designer_retry_failed" if retry_design else "feedback_revalidation_failed" if revalidate else "handoff_preparation_failed", detail=str(exc))
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
                return 2
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "suite-retry-fold":
        import fcntl
        from react_agent.eeg_research.agentic.method_suite import retry_fold
        with (camp / "worker.lock").open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"error": "engineering_retry_requires_worker_step_boundary"}))
                return 2
            try:
                state = load_state(camp)
                payload = retry_fold(camp, state, args.suite, args.fold, reason=args.retry_reason)
                save_state(camp, state)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                print(json.dumps({"error": str(exc), "command": args.command}, ensure_ascii=False))
                return 2
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command in {"start", "resume"}:
        from react_agent.eeg_research.agentic.worker import worker_lock_held

        if worker_lock_held(camp):
            worker_path = camp / "worker.json"
            pid = 0
            if worker_path.is_file():
                pid = int(json.loads(worker_path.read_text(encoding="utf-8")).get("pid") or 0)
            print(json.dumps({"status": "worker_already_running", "pid": pid}, ensure_ascii=False))
            return 0
        request_control(camp, "resume")
        from react_agent.eeg_research.agentic.verification_cache import verification_scope
        with verification_scope():
            align_interrupt(camp)
        state = load_state(camp)
        failure = state.get("failure") if isinstance(state.get("failure"), dict) else {}
        if args.command == "resume" and state.get("status") == "blocked" and failure.get("recoverable") is False:
            print(json.dumps({"status": "blocked", "detail": state.get("detail")}, ensure_ascii=False))
            return 0
        if state.get("status") in {"paused", "blocked"} and args.command == "resume":
            state["status"] = "created"
        if state.get("status") == "finished" and args.command == "resume" and args.reopen_reason:
            state["status"] = "created"
            state["experiment_failed"] = True
            state.setdefault("evidence", []).append(
                {
                    "evidence_id": f"ev_framework_{len(state['evidence']) + 1}",
                    "kind": "framework_note",
                    "summary": {"reopened_because": args.reopen_reason, "earlier_failures_are_scientific_results": False},
                }
            )
            event(camp, "reopened", reason=args.reopen_reason)
        state["pause_after_step"] = False
        save_state(camp, state)
        pid = (spawn_worker(root, args.campaign, args.poll_seconds) if args.max_ticks == 200
               else spawn_worker(root, args.campaign, args.poll_seconds, max_ticks=args.max_ticks))
        event(camp, args.command, worker_pid=pid)
        print(json.dumps({"status": "worker_started", "pid": pid}, ensure_ascii=False))
        return 0
    if args.command == "pause":
        request_control(camp, "pause")
        save_state(camp, state)
        event(camp, "pause_requested")
    elif args.command == "stop":
        from react_agent.eeg_research.agentic.jobs import stop_job

        request_control(camp, "stop")
        if state.get("live_job"):
            stop_job(camp / "jobs" / str(state["live_job"]))
        save_state(camp, state)
        event(camp, "stopped")
    state = load_state(camp)
    print(json.dumps({"status": state.get("status"), "pause_after_step": state.get("pause_after_step")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
