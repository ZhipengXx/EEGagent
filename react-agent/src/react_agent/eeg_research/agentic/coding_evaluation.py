"""Bounded coding-only provider evaluation through existing native tools.

This entry creates no scientific approval, SelectedPlan, suite or measurement.
It is inert without the explicit real-API flag; tests replace only SDK transport.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def run_evaluation(root: Path, *, task: str, real_api: bool = False, max_api_calls: int = 8,
                   max_steps: int = 4, max_repairs: int = 1, python: str | None = None,
                   input_spec: dict | None = None) -> dict:
    if not real_api:
        raise ValueError("coding_evaluation_requires_explicit_real_api_flag")
    if not task.strip():
        raise ValueError("coding_evaluation_task_missing")
    if not 1 <= max_api_calls <= 16 or not 1 <= max_steps <= 8 or not 0 <= max_repairs <= 2:
        raise ValueError("coding_evaluation_limits_out_of_range")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("coding_evaluation_api_key_missing")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("coding_evaluation_requires_empty_directory")
    root.mkdir(parents=True, exist_ok=True)
    from . import llm, loop
    from .interface import candidate_interface
    from .identity import ensure_attempt, operation_id
    from .native_patch import implement
    from .owner_budget import auxiliary_owner, snapshot
    from .schemas import GoalSpec
    goal = GoalSpec.model_validate({"goal_id": "coding_eval", "objective": task,
        "max_training_jobs": 0, "max_gpu_seconds": 0, "max_llm_calls": max_api_calls,
        "max_candidates": 0, "allowed_training_actions": [], "training_seeds": [0], "confirmation_target_pairs": 1,
        "research_execution": {"version": "single_plan_v1"}, "paid_maintenance": "off",
        "api_request_policy": {"transport_backoff_s": 0}, "memory": {"enabled": False},
        "coding_evaluation_only": True}).model_dump()
    loop.create_campaign(root, goal=goal, contract={"task_type": "eeg_image_retrieval",
        "fingerprint": "engineering_interface_evaluation.v1", "research_scope": "coding_only_no_scientific_measurement"},
        request_id="coding_eval")
    camp = root / goal["goal_id"]
    workspace = camp / "candidates" / "engineering_eval"
    workspace.mkdir(parents=True)
    interface = dict(input_spec or candidate_interface(None))
    # Inputs cannot opt this non-training entry into a physical multi-GPU probe.
    interface["multigpu_check"] = {"enabled": False}
    (workspace / "input_spec.json").write_text(json.dumps(interface, ensure_ascii=False, indent=2))
    # A coding task contract is deliberately a draft, never an ApprovalRecord.
    spec = {"experiment": {"status": "draft", "intervention": task, "hypothesis": {"mechanism": task},
        "required_capability_ids": ["build_encoder"], "model": {}, "objective": {}, "transform": {}},
        "coding_evaluation_only": True}
    (workspace / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    attempt = ensure_attempt(workspace, workspace.name)
    backend = llm.role_backend(camp, "candidate_coder")
    identity = {"candidate_id": workspace.name, "attempt_id": attempt["attempt_id"],
        "phase": "implement_candidate", "operation_id": operation_id(workspace, workspace.name, "implement_candidate")}
    backend.bind(**identity)
    backend.identity = identity
    def calls_left():
        path = camp / "cost.json"
        spent = json.loads(path.read_text()).get("llm_calls", 0) if path.is_file() else 0
        return max_api_calls - int(spent)
    previous_environment = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "EEG_TRAIN_DEVICE", "EEG_FINAL_TEST")}
    os.environ.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0")
    started = time.time()
    try:
        with auxiliary_owner(camp, purpose="coding_evaluation"):
            result = implement(workspace, spec, backend, max_steps=max_steps, max_repairs=max_repairs,
                python=python or sys.executable, calls_left=calls_left, reserve=0)
    except (llm.LlmUnavailable, ValueError, OSError) as exc:
        result = {"status": "blocked", "detail": str(exc), "error_type": type(exc).__name__}
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    cost = json.loads((camp / "cost.json").read_text()) if (camp / "cost.json").is_file() else {}
    state = loop.load_state(camp)
    if cost.get("llm_calls", 0) > max_api_calls:
        raise RuntimeError("coding_evaluation_global_call_limit_exceeded")
    if state.get("selected_plan") or state.get("candidate_admissions") or state.get("training_jobs"):
        raise RuntimeError("coding_evaluation_scientific_dispatch_detected")
    if (camp / "method_search_manifest.json").exists() or (camp / "suites").exists() or (camp / "jobs").exists():
        raise RuntimeError("coding_evaluation_training_dispatch_detected")
    report = {"schema_version": "eeg.coding_evaluation.v1", "coding_only": True,
        "camp": str(camp), "workspace": str(workspace), "task": task, "result": result,
        "limits": {"api_calls": max_api_calls, "tool_steps": max_steps, "engineering_repairs": max_repairs,
            "training_jobs": 0, "gpu_seconds": 0}, "provider_cost_projection": cost,
        "owner_budget": snapshot(camp), "elapsed_seconds": time.time() - started,
        "scientific_approval_created": False, "scientific_measurement_claimed": False,
        "quality_scope": "Generated-source native syntax/interface checks only; no EEG training/effect claim."}
    (root / "coding_evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--max-api-calls", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--python")
    parser.add_argument("--input-spec", type=Path)
    args = parser.parse_args(argv)
    report = run_evaluation(args.root, task=args.task, real_api=args.real_api, max_api_calls=args.max_api_calls,
        max_steps=args.max_steps, max_repairs=args.max_repairs, python=args.python,
        input_spec=json.loads(args.input_spec.read_text()) if args.input_spec else None)
    print(json.dumps({"coding_only": True, "status": report["result"]["status"],
        "api_calls": report["provider_cost_projection"].get("llm_calls", 0), "report": str(args.root / "coding_evaluation_report.json")}, ensure_ascii=False))
    return 0 if report["result"].get("status") == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
