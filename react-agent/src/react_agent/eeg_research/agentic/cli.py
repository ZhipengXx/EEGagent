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
from react_agent.eeg_research.agentic.loop import align_interrupt, create_campaign, event, load_state, request_control, save_state

DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "runs" / "eeg_research_v18"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.eeg_research.agentic.cli")
    parser.add_argument("command", choices=["create", "start", "run", "status", "pause", "resume", "stop"])
    parser.add_argument("--campaign", default="eeg_retrieval_research_v1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--gpu", default="0", help="Comma-separated GPU indices for training jobs")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--reopen-reason", default="", help="Reopen a finished campaign after a framework fix; the reason is logged")
    return parser


def goal(campaign: str, design=None) -> dict:
    scope = "pooled_subject_retrieval" if design is None else research_scope(design)
    return {
        "goal_id": campaign,
        "objective": "改进指定数据与协议下的 EEG 到图像检索",
        "task_type": "eeg_image_retrieval",
        "research_scope": scope,
        "primary_metric": "validation.fixed_gallery_top1",
        "direction": "maximize",
        "min_practical_gain_pp": None,
        "final_test_enabled": False,
        "max_candidates": 4,
        "max_training_jobs": 10,
        "max_llm_calls": 100,
        "max_gpu_seconds": 28800,
        "max_concurrent_training_jobs": 1,
        "max_api_usd": None,
        "allowed_changes": ["eeg_encoder", "temporal_pooling", "projection_head", "training_loss", "training_only_augmentation"],
        "frozen_components": ["split_manifest", "query_gallery_manifest", "evaluator", "metric_definition", "image_feature_target"],
    }


def spawn_worker(root: Path, campaign: str, poll_seconds: float) -> int:
    camp = root / campaign
    log = (camp / "worker.log").open("ab")
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "react_agent.eeg_research.agentic.cli", "run", "--campaign", campaign, "--root", str(root), "--poll-seconds", str(poll_seconds)],
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
    root = args.root.resolve()
    camp = root / args.campaign
    if args.command == "create":
        design = __import__("react_agent.eeg_training.protocol", fromlist=["Design"]).Design(
            "eeg", "inter-subject", "all", policy="agentic", training_strategy="pooled_subjects"
        )
        from react_agent.eeg_training.protocol import data_root

        base = data_root()
        try:
            protocol = build_execution_protocol(design, base)
        except ProtocolError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return 2
        contract = freeze_contract(design, base)
        contract["execution_fingerprint"] = protocol["fingerprint"]
        state = create_campaign(
            root,
            goal=goal(args.campaign, design),
            contract=contract,
            request_id=args.request_id or args.campaign,
            protocol=protocol,
        )
        state["gpu"] = [int(item) for item in args.gpu.split(",") if item.strip()]
        save_state(camp, state)
        print(json.dumps({"status": state["status"], "fingerprint": state["contract_fingerprint"]}, ensure_ascii=False))
        return 0
    if not (camp / "campaign_state.json").is_file():
        print(json.dumps({"error": "campaign_missing"}))
        return 2
    if args.command == "run":
        from react_agent.eeg_research.agentic.worker import run_worker

        state = run_worker(camp, poll_seconds=args.poll_seconds)
        print(json.dumps({"status": state.get("status"), "detail": state.get("detail")}, ensure_ascii=False))
        return 0
    if args.command == "status":
        print(json.dumps(status_view(camp), ensure_ascii=False, indent=2))
        return 0
    state = load_state(camp)
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
        pid = spawn_worker(root, args.campaign, args.poll_seconds)
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
