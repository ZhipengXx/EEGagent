"""Optional, isolated CPU repair replay; never launches a training suite.

Default is a config/input preview. Explicit --execute uses the current native
coder tools/checker on COPIES of an identified historical failure, with one
shared user-supplied API allowance across all profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

PROFILES = {"flash_non_thinking": {"model": "deepseek-flash", "thinking": False, "reasoning_effort": None},
            "flash_high": {"model": "deepseek-flash", "thinking": True, "reasoning_effort": "high"},
            "pro_high": {"model": "deepseek-v4-pro", "thinking": True, "reasoning_effort": "high"}}


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    required = {"source_path", "source_sha256", "failure_check", "input_spec", "spec", "max_steps", "max_repairs"}
    if required.difference(case) or not isinstance(case["failure_check"], dict) or case["failure_check"].get("ok") is not False:
        raise ValueError("replay_requires_identified_failure_and_complete_native_inputs")
    source = (path.parent / case["source_path"]).resolve()
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != case["source_sha256"]:
        raise ValueError("replay_source_hash_changed")
    if case["failure_check"].get("source_sha256") not in {None, case["source_sha256"]}:
        raise ValueError("replay_failure_source_identity_mismatch")
    if not 1 <= case["max_steps"] <= 100 or not 0 <= case["max_repairs"] <= 20:
        raise ValueError("replay_invalid_repair_limits")
    # No GPU/tenfold work is hidden in the optional model comparison.
    if (case["input_spec"].get("multigpu_check") or {}).get("enabled"):
        raise ValueError("replay_requires_explicit_cpu_preflight_protocol")
    return {**case, "source_text": data.decode(), "source_path": str(source),
            "case_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def replay(case: dict, output: Path, *, max_total_calls: int, backend_factory=None,
           implement_fn=None) -> dict:
    """Consume real native ledgers. No mocked pass rate is claimed by this entry."""
    from react_agent.eeg_research.agentic.llm import role_backend, refresh_method_api_cost
    from react_agent.eeg_research.agentic.native_patch import implement
    backend_factory = backend_factory or role_backend
    implement_fn = implement_fn or implement
    if max_total_calls < 1 or output.exists():
        raise ValueError("replay_requires_positive_shared_budget_and_new_output_directory")
    output.mkdir(parents=True)
    used = 0
    rows = []
    for name, overrides in PROFILES.items():
        remaining = max_total_calls - used
        if remaining <= 0:
            rows.append({"profile": name, "status": "not_run", "reason": "shared_api_budget_exhausted"})
            continue
        camp = output / name
        workspace = camp / "candidates" / "replay_candidate"
        (workspace / "extension").mkdir(parents=True)
        (workspace / "extension" / "eeg_candidate.py").write_text(case["source_text"])
        (workspace / "input_spec.json").write_text(json.dumps(case["input_spec"]))
        (workspace / "spec.json").write_text(json.dumps(case["spec"]))
        goal = {"evaluation_mode": "loso_method_search", "max_llm_calls": remaining,
                "role_models": {"candidate_coder": overrides}, "replay_case_sha256": case["case_sha256"],
                "scope": "historical_failure_cpu_preflight_replay_only"}
        (camp / "goal.json").write_text(json.dumps(goal, indent=2))
        (camp / "replay_input.json").write_text(json.dumps({key: value for key, value in case.items() if key != "source_text"}, indent=2))
        backend = backend_factory(camp, "candidate_coder")
        started = time.monotonic()
        try:
            result = implement_fn(workspace, {**case["spec"], "historical_failure": case["failure_check"]}, backend,
                max_steps=case["max_steps"], max_repairs=case["max_repairs"],
                calls_left=lambda: remaining - int(refresh_method_api_cost(camp).get("llm_calls", 0)), reserve=0)
        except Exception as exc:
            result = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
        cost = refresh_method_api_cost(camp)
        used += int(cost.get("llm_calls", 0))
        log = workspace / "coder_log.jsonl"
        tools = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.is_file() else []
        checks = [(row.get("result") or {}).get("auto_check") for row in tools]
        checks += [row.get("result") for row in tools if row.get("tool") == "run_candidate_check"]
        checks = [check for check in checks if isinstance(check, dict) and not check.get("cached")]
        passed = result.get("status") == "ready_for_review" and bool((result.get("check") or {}).get("ok"))
        rows.append({"profile": name, "status": result.get("status"), "first_check_passed": bool(checks[0].get("ok")) if checks else None,
            "first_pass_rate": float(bool(checks[0].get("ok"))) if checks else None,
            "final_repair_pass_rate": float(passed) if checks else None,
            "final_check_passed": passed, "api_calls": cost.get("llm_calls", 0), "check_count": len(checks),
            "repeated_error_count": (result.get("coding_recovery") or {}).get("consecutive_error_count"),
            "total_elapsed_s": time.monotonic() - started, "api_usd": cost.get("api_usd"),
            "cost_per_passed_candidate_usd": cost.get("api_usd") if passed else None,
            "check_cost": {"cpu_preflight_count": len(checks), "gpu_seconds": None, "usd": None},
            "result": result})
        if cost.get("api_unsettled_call_ids") or cost.get("api_uncertain_calls"):
            # Explicit reconciliation/abandonment is required before spending
            # against another comparison profile following an unknown call.
            break
    report = {"scope": "historical_failure_cpu_preflight_replay_only", "case_sha256": case["case_sha256"],
              "max_total_calls": max_total_calls, "actual_api_calls": used, "profiles": rows,
              "scientific_gain_claim": False, "note": "One identified case, descriptive check outcomes; no tenfold training or model superiority estimate."}
    (output / "replay_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-total-calls", type=int)
    args = parser.parse_args(argv)
    case = load_case(args.case)
    if args.execute:
        if args.output is None or args.max_total_calls is None:
            parser.error("--execute requires --output and --max-total-calls")
        value = replay(case, args.output, max_total_calls=args.max_total_calls)
    else:
        from react_agent.eeg_research.agentic.role_config import resolve_role_config
        value = {"case_sha256": case["case_sha256"], "source_sha256": case["source_sha256"], "mode": "preview_no_api",
                 "profiles": {name: resolve_role_config(None, "candidate_coder", config).model_dump() for name, config in PROFILES.items()},
                 "max_steps": case["max_steps"], "max_repairs": case["max_repairs"], "check_scope": "cpu_preflight_only"}
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
