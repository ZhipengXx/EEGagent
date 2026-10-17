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
        choices=["create", "start", "run", "status", "pause", "resume", "stop", "prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design", "retry-coder", "authorize-gpu-sharing", "recover-api-intents", "validate-goal", "evaluate-export", "evaluate-pack",
                 "study-create", "study-status", "study-run", "study-freeze", "study-evaluate", "study-aggregate",
                 "suite-retry-fold", "role-config-preview", "configure-roles", "screening-dry-run", "dispose-unknown-intent", "detach-candidate", "cancel-selected-plan", "historical-runtime-preview", "authorize-research-recovery"],
    )
    parser.add_argument("--campaign", default="eeg_retrieval_research_v1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--max-candidates", type=int, default=20,
                        help="authorize-research-recovery only: explicitly approved candidate scope")
    parser.add_argument("--planner-recovery-max-calls", type=int, default=2)
    parser.add_argument("--memory-recovery-max-calls", type=int, default=2)
    parser.add_argument("--parent-planner-round-id", default=None)
    parser.add_argument("--parent-memory-turn-id", default="native_closed:11")
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
    parser.add_argument("--execution-version", choices=["single_plan_v1", "legacy"], default=None,
                        help="New campaign protocol; old goal files keep legacy unless explicitly selected.")
    parser.add_argument("--pack", type=Path, default=None, help="Candidate pack directory for evaluate-export")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run evaluate-only for a pack. Default is dry-run argv.")
    parser.add_argument("--study", type=Path, default=None, help="Native sequential ten-fold LOSO study directory")
    parser.add_argument("--allow-partial", action="store_true", help="Freeze valid pairs with explicitly incomplete fold coverage")
    parser.add_argument("--allow-shared-gpus", action="store_true", help="Explicit authorization to use occupied GPUs without terminating their processes")
    parser.add_argument("--inherit-study", type=Path,
                        help="Native verified budget source: legacy study or quiescent method campaign")
    parser.add_argument("--fresh-budget-authorization", action="store_true",
                        help="Explicit new campaign allocation from goal limits; create only, requires --request-id")
    parser.add_argument("--budget-history-source", type=Path,
                        help="Read-only quiescent method campaign history/cost basis for a fresh allocation")
    parser.add_argument("--additional-llm-calls", type=int, default=0,
                        help="Explicit additional authorization for inherited method-campaign API attempts; create only")
    parser.add_argument("--additional-training-jobs", type=int, default=0,
                        help="Explicit additional authorization for inherited actual training jobs; create only")
    parser.add_argument("--additional-gpu-seconds", type=float, default=0,
                        help="Explicit additional occupied-GPU seconds authorization; create only")
    parser.add_argument("--suite", default=None, help="Frozen method suite ID for an engineering retry")
    parser.add_argument("--fold", default=None, help="Fold ID for an engineering retry")
    parser.add_argument("--candidate", default=None, help="Existing checked candidate for review revalidation")
    parser.add_argument("--plan-id", default=None, help="Exact selected plan for an explicit pre-measurement cancellation")
    parser.add_argument("--role-override", action="append", default=[], metavar="ROLE.FIELD=VALUE",
                        help="Explicit role settings for create/configure-roles; preview shows resolved request configuration")
    parser.add_argument("--call-id", default=None, help="Unknown API intent for explicit response abandonment")
    parser.add_argument("--runtime-checkout", type=Path, help="Pinned historical Git repository checkout; read-only compatibility preview")
    parser.add_argument("--original-project-root", type=Path, default=Path(__file__).resolve().parents[5],
                        help="Original absolute repository path recorded in historical framework dependencies")
    parser.add_argument("--allow-new-request", action="store_true", help="Permit a new request after recorded unknown response abandonment; retains old charge")
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


def role_overrides(values: list[str]) -> dict:
    from react_agent.eeg_research.agentic.role_config import ALIASES, FIELDS, ROLES
    result = {}
    for item in values:
        key, separator, value = item.partition("=")
        role, dot, field = key.partition(".")
        role = ALIASES.get(role, role)
        if not separator or not dot or role not in ROLES or field not in FIELDS:
            raise ValueError("invalid_role_override: expected ROLE.FIELD=VALUE")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        if field in result.get(role, {}):
            raise ValueError("duplicate_role_override")
        result.setdefault(role, {})[field] = value
    return result


def goal(campaign: str, design=None) -> dict:
    scope = "pooled_subject_retrieval" if design is None else research_scope(design)
    gpu = DEFAULT_MAX_GPU_SECONDS
    chosen = None if design is None else getattr(design, "gpu_seconds", None)
    if chosen is not None:
        gpu = float(chosen)
    return {
        "goal_id": campaign,
        "planner_mode": "compare_options",
        "research_execution": {"version": "single_plan_v1", "planning_mode": "propose_then_critic",
            "implementation_review_mode": "single_pass", "design_contract_version": "v2", "budget": {"envelopes": {
            kind: {"training_jobs": 2, "gpu_seconds": 2 * gpu / 20,
                "allocation_basis": "authorized_global_gpu_seconds_per_training_job",
                "authority": "maximum_runtime_allocation_not_cost_estimate"}
            for kind in ("new_method", "validation", "confirmation")}}},
        "objective": "改进指定数据与协议下的 EEG 到图像检索",
        "role_models": {
            "research_planner": {"model": "deepseek-v4-pro", "thinking": True,
                "reasoning_effort": "high", "max_tokens": 32768, "timeout_s": 240},
            "research_critic": {"model": "deepseek-v4-pro", "thinking": True,
                "reasoning_effort": "high", "max_tokens": 16384, "timeout_s": 240},
            "candidate_coder": {"model": "deepseek-flash", "thinking": True,
                "reasoning_effort": "high", "max_tokens": 262144, "timeout_s": 2400},
            "candidate_reviewer": {"model": "deepseek-flash", "thinking": True,
                "reasoning_effort": "high", "max_tokens": 262144, "timeout_s": 2400}},
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


def _execution_goal(submitted: dict, execution_version: str | None) -> dict:
    submitted = dict(submitted)
    if execution_version == "legacy":
        submitted.pop("research_execution", None)
    elif execution_version == "single_plan_v1":
        submitted["research_execution"] = {**(submitted.get("research_execution") or {}), "version": "single_plan_v1"}
    elif execution_version is not None:
        raise ValueError("execution_version_invalid")
    execution = submitted.get("research_execution") or {}
    if execution.get("version") == "single_plan_v1" and submitted.get("evaluation_mode", "single_target") == "single_target":
        from math import isfinite
        jobs, gpu_seconds = submitted.get("max_training_jobs", 20), submitted.get("max_gpu_seconds", DEFAULT_MAX_GPU_SECONDS)
        if (type(jobs) is not int or jobs <= 0 or type(gpu_seconds) not in (int, float)
                or not isfinite(gpu_seconds) or gpu_seconds <= 0):
            # A zero-training goal can still authorize diagnosis. Missing
            # measurement bounds remain subject to native admission rejection.
            return submitted
        budget = dict(execution.get("budget") or {})
        envelopes = dict(budget.get("envelopes") or {})
        for kind in ("new_method", "validation", "confirmation"):
            envelopes.setdefault(kind, {"training_jobs": 2, "gpu_seconds": 2 * gpu_seconds / jobs,
                "allocation_basis": "authorized_global_gpu_seconds_per_training_job",
                "authority": "maximum_runtime_allocation_not_cost_estimate"})
        budget["envelopes"] = envelopes
        submitted["research_execution"] = {**execution, "budget": budget}
    return submitted


def _entry_error(camp: Path) -> str | None:
    path = camp / "campaign_state.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "cancelled":
        return "campaign_cancelled"
    control = camp / "control.json"
    if control.is_file() and json.loads(control.read_text(encoding="utf-8")).get("action") == "stop":
        return "campaign_cancelled"
    if (camp / "campaign_creation.prepare.json").is_file() and not (camp / "campaign_creation.commit.json").is_file():
        return "campaign_creation_not_committed"
    return None


def _creation_request_path(root: Path, request_id: str) -> Path:
    from .action_planning import digest
    return root / "creation_requests" / (digest({"request_id": request_id}) + ".prepare.json")


def _read_creation_request(root: Path, request_id: str, config: dict) -> dict | None:
    from .action_planning import digest
    path = _creation_request_path(root, request_id)
    if not path.is_file():
        return None
    prepared = json.loads(path.read_text(encoding="utf-8"))
    if (digest({key: value for key, value in prepared.items() if key != "prepare_digest"}) != prepared.get("prepare_digest")
            or prepared.get("request_id") != request_id or prepared.get("request_config") != config):
        raise ValueError("campaign_creation_configuration_conflict")
    return prepared


def _publish_cli_creation(root: Path, prepared: dict) -> dict:
    return create_campaign(root, goal=prepared["goal"], contract=prepared["contract"],
        request_id=prepared["request_id"], protocol=prepared["protocol"], creation_context=prepared["creation_context"])


def _create_output(state: dict, camp: Path) -> dict:
    from .research_execution import paid_maintenance_mode, single_plan_enabled
    frozen_goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
    return {"status": state["status"], "fingerprint": state["contract_fingerprint"], "spawn_worker": False,
        "max_training_jobs": state.get("max_training_jobs"), "max_llm_calls": state.get("max_llm_calls"),
        "max_gpu_seconds": state.get("max_gpu_seconds"),
        "execution_version": "single_plan_v1" if single_plan_enabled(frozen_goal) else "legacy",
        "paid_maintenance": paid_maintenance_mode(frozen_goal), "selected_plan": state.get("selected_plan")}


def spawn_worker(root: Path, campaign: str, poll_seconds: float, *, max_ticks: int = 200) -> int:
    camp = root / campaign
    error = _entry_error(camp)
    if error:
        raise ValueError(error)
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
    execution_version: str = "single_plan_v1",
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
    existing_camp = Path(known[request_id]) if already else root / campaign
    if _entry_error(existing_camp) == "campaign_cancelled":
        result.update(ok=False, started=False, error="campaign_cancelled")
        result["blockers"] = list(result.get("blockers") or []) + ["campaign_cancelled"]
        return result
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
    state = create_campaign(root, goal=_execution_goal(goal(campaign, design), execution_version),
        contract=contract, request_id=request_id, protocol=protocol)
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
    from .research_execution import paid_maintenance_mode, single_plan_enabled
    state = load_state(camp)
    frozen_goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
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
        "execution_version": "single_plan_v1" if single_plan_enabled(frozen_goal) else "legacy",
        "paid_maintenance": paid_maintenance_mode(frozen_goal),
        "selected_plan": state.get("selected_plan"),
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
    try:
        overrides = role_overrides(args.role_override)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    if overrides and args.command == "revalidate-review":
        from .review_revalidation import validate_override
        try:
            validate_override(overrides)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
    elif overrides and args.command not in {"create", "configure-roles", "role-config-preview", "screening-dry-run"}:
        print(json.dumps({"error": "role_overrides_require_new_campaign_or_preview"}))
        return 2
    if args.execution_version is not None and args.command != "create":
        print(json.dumps({"error": "execution_version_requires_new_campaign"}))
        return 2
    if args.command == "historical-runtime-preview":
        from react_agent.eeg_research.agentic.historical_runtime import checkout_preview
        try:
            if args.runtime_checkout is None:
                raise ValueError("historical_runtime_checkout_required")
            result = checkout_preview(args.root.resolve() / args.campaign, args.runtime_checkout, args.original_project_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (ValueError, KeyError, OSError) as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
    if args.command in {"role-config-preview", "screening-dry-run"}:
        from react_agent.eeg_research.agentic.role_config import role_config_preview
        try:
            source = load_goal_file(args.goal) if args.goal else args.root.resolve() / args.campaign
            if args.command == "role-config-preview":
                result = role_config_preview(source, overrides)
            else:
                from react_agent.eeg_research.agentic.screening import dry_run, freeze_selection
                SCHEDULE = "screen_then_confirm"
                if isinstance(source, dict):
                    if source.get("evaluation_schedule") != SCHEDULE:
                        raise ValueError("screening_schedule_not_enabled")
                    subjects = source["benchmark_subjects"]
                    common = {"selection_manifest": freeze_selection(source), "screening_policy": source.get("screening") or {},
                              "evaluation_schedule": SCHEDULE, "subjects": source["benchmark_subjects"],
                              "folds": [{"fold_id": f"fold_{index:02d}", "held_out_subject": subject,
                                         "train_subjects": [other for other in subjects if other != subject]}
                                        for index, subject in enumerate(subjects, 1)],
                              "gpu_seconds_estimate_per_suite": source.get("method_suite_gpu_seconds_estimate")}
                else:
                    from react_agent.eeg_research.agentic.method_suite import manifest
                    common = manifest(source)
                    if common.get("evaluation_schedule") != SCHEDULE:
                        raise ValueError("screening_schedule_not_enabled")
                baseline_completed, candidate_completed = [], []
                if not isinstance(source, dict):
                    from react_agent.eeg_research.agentic.screening import suites_for_target, valid_scores
                    controls = suites_for_target(source, "baseline")
                    if controls:
                        baseline_completed = [row["fold_id"] for row in valid_scores(source, controls[-1])]
                    if args.candidate:
                        candidates = suites_for_target(source, args.candidate)
                        if candidates:
                            candidate_completed = [row["fold_id"] for row in valid_scores(source, candidates[-1])]
                result = dry_run(common, baseline_completed=baseline_completed, candidate_completed=candidate_completed)
                result["preview_only"] = True
                result["verified_baseline_fold_ids"] = baseline_completed
                result["verified_candidate_fold_ids"] = candidate_completed
                result["cache_status"] = ("No completed fold is assumed in a goal preview." if isinstance(source, dict) else
                                          "Completed folds verified from current native suite/source/checkpoint/evaluation dependencies.")
                result["role_config"] = role_config_preview(source, overrides)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (ValueError, KeyError, OSError) as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
    additional = {"llm_calls": args.additional_llm_calls,
                  "training_jobs": args.additional_training_jobs,
                  "gpu_seconds": args.additional_gpu_seconds}
    if args.fresh_budget_authorization or args.budget_history_source is not None:
        if (args.command != "create" or not args.fresh_budget_authorization
                or args.budget_history_source is None or args.inherit_study is not None
                or any(additional.values()) or args.goal is None or not args.request_id.strip()):
            print(json.dumps({"error": "fresh_authorization_requires_create_goal_history_request_id_without_inheritance"}))
            return 2
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
        submitted = _execution_goal(submitted, args.execution_version)
        if args.planner_mode is not None:
            submitted["planner_mode"] = args.planner_mode
        if submitted.get("evaluation_mode") == "loso_method_search":
            submitted["allow_shared_gpus"] = args.allow_shared_gpus
        request_id = args.request_id or args.campaign
        request_config = {"goal": submitted, "gpu": list(selected_gpus), "base": str(base.resolve()),
            "execution_version": args.execution_version, "planner_mode": args.planner_mode,
            "role_overrides": overrides, "fresh_budget_authorization": args.fresh_budget_authorization,
            "budget_history_source": None if args.budget_history_source is None else str(args.budget_history_source.resolve()),
            "inherit_study": None if args.inherit_study is None else str(args.inherit_study.resolve()),
            "additional_authorization": additional}
        try:
            index = root / "request_index.json"
            indexed = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
            if request_id in indexed and Path(indexed[request_id]).resolve() != camp.resolve():
                raise ValueError("campaign_creation_request_already_bound")
            prepared = _read_creation_request(root, request_id, request_config)
            if prepared:
                if _entry_error(camp) == "campaign_cancelled":
                    raise ValueError("campaign_cancelled")
                state = _publish_cli_creation(root, prepared)
                print(json.dumps(_create_output(state, camp), ensure_ascii=False))
                return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        inherited = None
        estimate_basis = []
        if submitted.get("evaluation_mode") == "loso_method_search":
            from dataclasses import replace
            from react_agent.eeg_research.agentic.budget_inheritance import (
                verified_history, fresh_authorization, derive_limits, estimate_suite,
            )
            from react_agent.eeg_research.agentic.loso_study import SUBJECTS
            if args.inherit_study is None and not args.fresh_budget_authorization:
                print(json.dumps({"error": "method_search_requires_native_budget_inheritance"}))
                return 2
            try:
                if args.fresh_budget_authorization:
                    if camp.exists():
                        raise ValueError("fresh_authorization_requires_new_campaign")
                    inherited = fresh_authorization(camp, submitted, args.budget_history_source,
                                                    authorization_id=args.request_id)
                    # Reject reuse of a named authorization before expensive
                    # protocol construction or any new campaign files exist.
                    from react_agent.eeg_research.agentic.loso_study import identity
                    prior = root / "fresh_budget_authorizations" / (identity({
                        "authorization_id": args.request_id}) + ".json")
                    index = root / "request_index.json"
                    if prior.exists() or (index.is_file() and args.request_id in json.loads(index.read_text())):
                        raise ValueError("fresh_budget_authorization_already_bound")
                else:
                    inherited = verified_history(args.inherit_study, additional_authorization=additional)
                submitted = derive_limits(submitted, inherited)
                estimate, estimate_basis = estimate_suite(inherited, len(selected_gpus))
            except (OSError, ValueError, KeyError) as exc:
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
                return 2
            submitted["method_suite_gpu_seconds_estimate"] = max(estimate, submitted.get("method_suite_gpu_seconds_estimate") or 0)
            design = replace(design, subject=",".join(SUBJECTS[:-1]), epochs=50, stop="single_full", seed=0,
                             data_root=str(base.resolve()), evaluation_mode="loso_method_search",
                             checkpoint_policy=submitted["checkpoint_policy"], selection_min_delta=submitted["selection_min_delta"])
        elif args.inherit_study is not None or args.fresh_budget_authorization:
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
        try:
            context = {"gpu": list(design.gpu), "inheritance": inherited, "files": {}}
            if overrides:
                from .role_config import role_config_preview
                role_config_preview(submitted, overrides)
                context["files"]["role_config_overrides.json"] = overrides
            if inherited is not None:
                from .method_suite import create_manifest
                from tempfile import TemporaryDirectory
                # Freeze the entire native manifest before any campaign or
                # budget claim is published. Restart copies these exact bytes.
                with TemporaryDirectory(prefix="eeg-campaign-prepare-") as temporary:
                    frozen_manifest = create_manifest(Path(temporary), design,
                        allow_shared_gpus=args.allow_shared_gpus,
                        estimate=submitted["method_suite_gpu_seconds_estimate"], goal=submitted)
                context["files"]["method_search_manifest.json"] = frozen_manifest
                context["files"]["suite_cost_basis.json"] = {"basis": estimate_basis,
                    "estimate": submitted["method_suite_gpu_seconds_estimate"]}
            from .action_planning import digest
            from .loop import _write
            prepared = {"schema_version": "eeg_research.cli_creation_prepare.v1", "request_id": request_id,
                "request_config": request_config, "goal": submitted, "contract": contract,
                "protocol": protocol, "creation_context": context}
            prepared["prepare_digest"] = digest(prepared)
            _write(_creation_request_path(root, request_id), prepared, exclusive=True)
            state = _publish_cli_creation(root, prepared)
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(_create_output(state, camp), ensure_ascii=False))
        return 0
    if not (camp / "campaign_state.json").is_file():
        print(json.dumps({"error": "campaign_missing"}))
        return 2
    if args.command in {"start", "resume", "run"}:
        error = _entry_error(camp)
        if error:
            print(json.dumps({"error": error}))
            return 2
    if args.command in {"start", "resume", "run", "prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design", "retry-coder", "configure-roles", "authorize-gpu-sharing", "recover-api-intents", "authorize-research-recovery", "cancel-selected-plan"}:
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
    if args.command == "authorize-research-recovery":
        from .native_authorization import authorize_research_recovery
        try:
            receipt = authorize_research_recovery(camp, authorization_id=args.request_id,
                max_candidates=args.max_candidates,
                planner_recovery_max_calls=args.planner_recovery_max_calls,
                memory_recovery_max_calls=args.memory_recovery_max_calls,
                parent_planner_round_id=args.parent_planner_round_id,
                parent_memory_turn_id=args.parent_memory_turn_id)
        except (ValueError, OSError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"status": "research_recovery_authorized", "authorization": receipt}, ensure_ascii=False))
        return 0
    state = load_state(camp)
    if args.command == "cancel-selected-plan":
        import fcntl
        from .selected_plan_control import cancel_selected_plan
        try:
            if not args.plan_id or not args.retry_reason.strip():
                raise ValueError("selected_plan_cancellation_requires_plan_id_and_reason")
            with (camp / "worker.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = cancel_selected_plan(camp, load_state(camp), args.retry_reason,
                                              expected_plan_id=args.plan_id)
            print(json.dumps({"status": result.get("status"), "selected_plan": result.get("selected_plan"),
                              "next_stage": "planner"}, ensure_ascii=False))
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
    if args.command == "configure-roles":
        import fcntl
        from .role_configuration import apply_role_configuration
        try:
            with (camp / "worker.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = apply_role_configuration(camp, load_state(camp), overrides,
                    reason=args.retry_reason)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
    if args.command == "detach-candidate":
        from react_agent.eeg_research.agentic.candidate_lifecycle import detach_failed_candidate, enabled
        import fcntl
        try:
            if not enabled(json.loads((camp / "goal.json").read_text())):
                raise ValueError("candidate_detach_requires_explicit_new_protocol")
            if not args.candidate or not args.retry_reason.strip():
                raise ValueError("candidate_detach_requires_target_and_reason")
            with (camp / "worker.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt = detach_failed_candidate(camp, state, args.candidate, args.retry_reason)
                save_state(camp, state)
                event(camp, "candidate_detached", candidate_id=args.candidate, receipt_hash=receipt["receipt_hash"])
            print(json.dumps(receipt, ensure_ascii=False))
            return 0
        except (ValueError, OSError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
    if args.command == "dispose-unknown-intent":
        from react_agent.eeg_research.agentic.api_policy import dispose_unknown_intent
        import fcntl
        try:
            with (camp / "worker.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = dispose_unknown_intent(camp, args.call_id, reason=args.retry_reason,
                                                allow_new_request=args.allow_new_request)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        except (ValueError, OSError, KeyError) as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
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
    if args.command in {"prepare-handoff", "revalidate-feedback", "revalidate-review", "retry-design", "retry-coder"}:
        import fcntl
        from react_agent.eeg_research.agentic.worker import build_services
        from react_agent.eeg_research.agentic.llm import LlmUnavailable
        from react_agent.eeg_research.agentic.budget_inheritance import validate_inheritance
        from react_agent.eeg_research.agentic.verification_cache import verification_scope
        revalidate = args.command == "revalidate-feedback"
        retry_design = args.command == "retry-design"
        retry_coder = args.command == "retry-coder"
        revalidate_review = args.command == "revalidate-review"
        if (revalidate or retry_design or retry_coder or revalidate_review) and not args.retry_reason.strip():
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
                    services = build_services(camp, role_overrides=overrides) if overrides else build_services(camp)
                    if retry_coder:
                        result = services["retry_coder"](camp, state, reason=args.retry_reason)
                    elif retry_design:
                        result = services["retry_design"](camp, state, reason=args.retry_reason)
                    elif revalidate_review:
                        result = services["revalidate_review"](camp, state,
                            candidate_id=args.candidate or "", reason=args.retry_reason)
                    else:
                        result = (services["revalidate_feedback"](camp, state, reason=args.retry_reason)
                                  if revalidate else services["prepare_handoff"](camp, state))
                save_state(camp, state)
                event(camp, "coder_retried" if retry_coder else "review_revalidated" if revalidate_review else "designer_retried" if retry_design else "feedback_revalidated" if revalidate else "handoff_prepared", **result)
            except (LlmUnavailable, OSError, ValueError, KeyError) as exc:
                save_state(camp, state)
                event(camp, "coder_retry_failed" if retry_coder else "review_revalidation_failed" if revalidate_review else "designer_retry_failed" if retry_design else "feedback_revalidation_failed" if revalidate else "handoff_preparation_failed", detail=str(exc))
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
