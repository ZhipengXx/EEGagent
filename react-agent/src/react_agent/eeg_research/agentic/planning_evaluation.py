"""Bounded provider evaluation of planning; this entry never dispatches work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _journal(camp: Path) -> list[dict]:
    path = camp / "llm_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def run_evaluation(root: Path, *, real_api: bool, max_api_calls: int = 6) -> dict:
    """Use real compiler, budget admission, provider and round validators.

    Scenarios are explicitly synthetic diagnostics, with no measurement claims.
    Candidate admission, implementation, training and action dispatch are absent.
    """
    if not real_api:
        raise ValueError("planning_evaluation_requires_explicit_real_api_flag")
    if not 1 <= max_api_calls <= 12:
        raise ValueError("planning_evaluation_call_limit_out_of_range")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("planning_evaluation_api_key_missing")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("planning_evaluation_requires_empty_directory")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from . import loop, llm
    from .experiment_observations import ExperimentObservationStore, compact_planner_view
    from .owner_budget import OwnerBudgetError, finish_planner_round, planner_owner, snapshot
    from .api_policy import planner_api_round
    from .plan_protocol import config, round_id
    from .planning_context import build_proposer_input
    from .planning_contract import freeze_planning_contract
    from .research_execution import planning_call_budget
    from .staged_planning import decide_staged_plan
    from .schemas import GoalSpec

    scenarios = [
        ("shape", "A synthetic encoder check returned the wrong output shape. Compare diagnostic tasks that distinguish an interface mismatch, a configuration mismatch and a preprocessing mismatch."),
        ("restart", "A synthetic restart check found inconsistent source/spec references. Compare tasks that determine whether identity capture, checkpoint restoration or configuration parsing caused the inconsistency."),
        ("scope", "A synthetic report has missing evaluation coverage. Compare diagnostic tasks that determine which source records are absent and whether any comparison can be interpreted. Missing metrics remain unknown."),
    ]
    results = []
    all_calls = []
    spent = 0
    for name, question in scenarios:
        # Each independent evaluation gets its own bootstrap owner. It does not
        # pretend to be a continuation without an executed terminal observation.
        remaining = max_api_calls - spent
        if remaining < 2:  # One Planner and one Critic request at minimum.
            results.append({"scenario": name, "ok": False, "status": "deferred",
                "error": "insufficient_budget_for_bounded_planner", "actual_API_attempts": 0})
            break
        # Planner and Critic share this scenario's remaining attempts. The goal's
        # wider llm_calls allowance only admits the complete (never dispatched)
        # task before the Critic; spent attempts stay under max_api_calls.
        stage_attempts = max(1, min(2, remaining // 2))
        goal = GoalSpec.model_validate({
            "goal_id": "planning_" + name, "objective": question,
            "max_llm_calls": 40, "max_training_jobs": 0, "max_gpu_seconds": 0,
            "max_candidates": 0, "allowed_training_actions": [], "training_seeds": [0], "confirmation_target_pairs": 1,
            "research_execution": {"version": "single_plan_v1", "target_k": 3,
                "allow_degraded_singleton": False, "max_planner_api_attempts": stage_attempts,
                "max_critic_api_attempts": stage_attempts},
            "paid_maintenance": "off", "memory": {"enabled": True},
            "role_models": {"research_planner": {"model": "deepseek-flash", "max_tokens": 16384},
                            "research_critic": {"model": "deepseek-flash", "max_tokens": 16384}},
        }).model_dump()
        loop.create_campaign(root, goal=goal, contract={"fingerprint": "synthetic_planning_only.v1",
            "task_type": "eeg_image_retrieval", "research_scope": "synthetic_diagnostic_planning"}, request_id=goal["goal_id"])
        camp = root / goal["goal_id"]
        state = loop.load_state(camp)
        state["candidates"] = [{"candidate_id": "fixture_decoder", "status": "unimplemented"}]
        loop.save_state(camp, state)
        store = ExperimentObservationStore(camp)
        card = store.record({"origin_campaign_id": camp.name, "plan_id": "fixture:" + name,
            "observation_kind": "engineering_failure", "execution_status": "failed",
            "question": question, "intervention": "Explicit synthetic diagnostic input; no EEG experiment was run.",
            "identity": {"candidate_id": "fixture_decoder"},
            "native_facts": {"metrics": None, "coverage": None, "scope": "synthetic_diagnostic",
                "missing": ["real_measurement", "scientific_comparison"]},
            "engineering_failure": {"error": name, "error_fingerprint": hashlib.sha256(name.encode()).hexdigest(),
                "repairs_used": 0}, "source_refs": [], "analyst_interpretation": None,
            "applicability": {"campaign_id": camp.name, "scope": "synthetic_diagnostic"}})
        full = {"_campaign_path": str(camp), "goal": goal, "research_execution": goal["research_execution"],
            "current_question": question, "evaluation_mode": "single_target", "evaluation_schedule": "full_only",
            "contract_fingerprint": "synthetic_planning_only.v1", "execution_fingerprint": None,
            "candidates": state["candidates"], "candidate_admissions": [],
            "available_actions": ["diagnose_results", "collect_diagnostics"], "experiment_observations": [card],
            "eligible_targets": {}, "artifact_index": [], "evidence": [],
            "action_cost_estimates": {"by_action_target": {}},
            "evaluation_guard": {"planning_only": True, "synthetic_input": True,
                "selected_work_is_not_dispatched": True, "training_authorization": False}}
        digest = hashlib.sha256(json.dumps(full, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        full["_planning_contract"] = freeze_planning_contract(full)
        started = time.time()
        planner = llm.role_backend(camp, "research_planner")
        critic = llm.role_backend(camp, "research_critic")
        wire = build_proposer_input(full, compact_planner_view(camp, full))
        cfg = config(full)
        limits = {"max_engineering_repairs": cfg.max_engineering_repairs,
                  "max_repairs": cfg.max_engineering_repairs, "coder_steps": 8}
        owner_id = None
        try:
            with planner_owner(camp, observation_digest=digest, round_id=round_id(full)) as owner_id:
                with planner_api_round(camp, round_id(full), max_attempts=planning_call_budget(camp)):
                    result = decide_staged_plan(camp, state, full, wire, planner, critic, owner_id, limits, cfg)
        except (llm.LlmUnavailable, OwnerBudgetError, ValueError, OSError) as error:
            result = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        finally:
            if owner_id is not None:
                try:
                    finish_planner_round(camp, owner_id)
                except OwnerBudgetError as error:
                    # Unknown/inflight claims retain their original allocation.
                    # Reporting an error cannot manufacture a refund.
                    result["owner_release_deferred"] = str(error)
        after = _read(camp / "cost.json") if (camp / "cost.json").is_file() else {}
        attempts = after.get("llm_calls", 0)
        spent += attempts
        if spent > max_api_calls:
            raise RuntimeError("planning_evaluation_global_cap_exceeded")
        calls = _journal(camp)
        all_calls.extend(calls)
        if state.get("training_jobs") or (camp / "selected_plans").exists() or ((camp / "jobs").exists() and any((camp / "jobs").iterdir())):
            raise RuntimeError("planning_evaluation_unexpected_dispatch_artifact")
        result_path = camp / ("planning_result_" + name + ".json")
        _write(result_path, result)
        results.append({"scenario": name, "campaign": str(camp), "synthetic_observation_id": card.get("observation_id"),
            "ok": result.get("ok"), "pool_status": (result.get("round") or {}).get("pool_status"),
            "valid_count": (result.get("round") or {}).get("valid_count"),
            "selected_proposal_id": (result.get("round") or {}).get("selected_proposal_id"),
            "actual_API_attempts": attempts, "budget": snapshot(camp),
            "elapsed_seconds": time.time() - started,
            "delivery_user_chars_before_compiler": len(json.dumps(wire, ensure_ascii=False)),
            "result_ref": str(result_path)})
    report = {"evaluation_version": "planning_provider_evaluation.v1", "root": str(root),
        "scope": "Real provider planning over explicitly synthetic diagnostic inputs; no training or scientific performance claim.",
        "scenarios": results, "actual_API_attempts": spent, "max_API_attempts": max_api_calls,
        "training_jobs": 0, "GPU_seconds": 0,
        "selected_tasks_executed": 0, "maintenance_API_attempts": sum(
            row.get("role") in {"memory_curator", "context_summary", "memory_extraction", "memory_consolidation"}
            or str(row.get("operation") or row.get("operation_id") or "").startswith("context_maintenance:") for row in all_calls),
        "requests": [{**{key: row.get(key) for key in ("call_id", "input_tokens", "output_tokens", "reasoning_tokens",
            "finish_reason", "context_mode", "context_budget_report", "budget_owner", "success", "api_usd")},
            "cost_status": row.get("cost_status") or ("unpriced" if row.get("api_usd") is None else "priced")}
            for row in all_calls]}
    _write(root / "planning_evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--max-api-calls", type=int, default=6)
    args = parser.parse_args()
    report = run_evaluation(args.root, real_api=args.real_api, max_api_calls=args.max_api_calls)
    print(json.dumps({"report": str(args.root / "planning_evaluation_report.json"),
        "API_attempts": report["actual_API_attempts"], "scenarios": report["scenarios"],
        "training_jobs": 0, "GPU_seconds": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
