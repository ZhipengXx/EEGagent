"""Explicit bounded provider/CPU closed loop on isolated synthetic EEG-shaped data.

The native campaign, roles, candidate gates, job launcher, trainer, settlement
and observation handoff execute unchanged. Synthetic scores are infrastructure
checks; they are not EEG research outcomes or estimates of model capability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


@contextmanager
def _cpu_environment():
    changes = {"CUDA_VISIBLE_DEVICES": "", "EEG_TRAIN_DEVICE": "cpu", "EEG_ALLOW_CPU_TRAIN": "1",
        "EEG_TRAIN_PYTHON": sys.executable, "EEG_FINAL_TEST": "0", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "EEG_DIAGNOSTIC_SAMPLE_LIMIT": "8"}
    previous = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_synthetic_data(root: Path):
    """Generate only isolated data files, never copy or inspect a real dataset."""
    import torch
    from react_agent.eeg_training.protocol import Design, FULL_EEG_CHANNELS, data_dir, feature_cache
    design = Design("eeg", "intra-subject", "sub-01", epochs=3, seed=0, batch_size=8,
        data_root=str(root.resolve()), stop="single_full", policy="agentic", gpu=(),
        evaluation_mode="single_target")
    generator = torch.Generator().manual_seed(73129)
    files = []
    subject = data_dir(root, "eeg") / "sub-01"
    subject.mkdir(parents=True, exist_ok=False)
    for partition, count in (("train", 20), ("test", 4)):
        ids = ["synthetic-" + partition + "-" + str(i) for i in range(count)]
        values = torch.randn(count, 1, len(FULL_EEG_CHANNELS), 250, generator=generator)
        path = subject / (partition + ".pt")
        torch.save({"eeg": values, "img": ids, "label": ids}, path)
        cache = feature_cache(root, design, partition)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"img_features": {image: torch.randn(1024, generator=generator) for image in ids}}, cache)
        files.extend((path, cache))
    manifest = {"schema_version": "eeg.synthetic.closed_loop_data.v1", "synthetic": True,
        "seed": 73129, "subjects": ["sub-01"], "feature_dimension": 1024,
        "train_file_images": 20, "forbidden_test_file_images": 4,
        "scientific_claim": "Random EEG-shaped inputs and random feature targets; no EEG efficacy inference.",
        "files": [{"path": str(path.resolve()), "sha256": _hash(path)} for path in files]}
    _write(root.parent / "synthetic_data_manifest.json", manifest)
    return design, manifest


def _goal(max_api_calls: int) -> dict:
    from .schemas import GoalSpec
    return GoalSpec.model_validate({"goal_id": "synthetic_closed_loop", "evaluation_mode": "single_target",
        "objective": "Infrastructure acceptance on RANDOM SYNTHETIC EEG-shaped data only, CPU only. "
            "Cold start: compare exactly two distinct small implementable new_method tasks with evaluation_scope phase pilot "
            "and seed zero; choose one and answer it with the native three-epoch pilot and automatic matched baseline. "
            "Use only the eeg_encoder interface, with no objective, feature target, split, gallery or evaluator changes. "
            "Declare a concrete source_review implementation requirement. This is not an EEG improvement experiment. "
            "After its terminal observation, compare two distinct NO-TRAINING diagnosis tasks about the recorded existing "
            "candidate and choose one; the evaluator stops before dispatching that next task.",
        "max_llm_calls": max_api_calls, "max_training_jobs": 2, "max_gpu_seconds": 600,
        "max_candidates": 1, "max_concurrent_training_jobs": 1, "max_repairs_per_candidate": 1,
        "training_seeds": [0], "confirmation_target_pairs": 1, "allowed_training_actions": ["run_pilot"],
        "allowed_changes": ["eeg_encoder"], "frozen_components": ["split_manifest", "query_gallery_manifest",
            "evaluator", "metric_definition", "image_feature_target", "training_loss"],
        "research_execution": {"version": "single_plan_v1", "target_k": 2, "paid_maintenance": "off",
            # Each planning round reserves one Planner and one Critic call; with
            # the next diagnosis admission, coding keeps max_api_calls - 12.
            "max_engineering_repairs": 1, "max_planner_api_attempts": 1, "max_critic_api_attempts": 1, "budget": {"envelopes": {
                "new_method": {"role_calls": {"experiment_designer": 2, "candidate_coder": max(1, min(8, max_api_calls - 12)),
                    "candidate_reviewer": 2, "result_analyst": 2}, "training_jobs": 2, "gpu_seconds": 600}}}},
        "paid_maintenance": "off", "memory": {"enabled": False},
        "api_request_policy": {"max_attempts_per_operation": 2, "max_attempts_per_planner_round": 2, "transport_backoff_s": 0},
        "role_models": {role: {"model": "deepseek-flash", "max_tokens": 32768 if role == "candidate_coder" else 16384,
            "thinking": True, "reasoning_effort": "high"}
            for role in ("research_planner", "research_critic", "experiment_designer", "candidate_coder", "candidate_reviewer", "result_analyst")},
    }).model_dump()


def _request_proofs(camp: Path, calls: list[dict]) -> list[dict]:
    from .context.types import digest
    result = []
    for call in calls:
        trace = _read(Path(call.get("request_trace_ref") or "__missing_request__"))
        fields = trace.get("transport_fields") or {}
        verified = bool(trace and digest({"system": trace["system"], "user": trace["user"],
            "transport_fields": fields}) == call.get("context_request_digest"))
        result.append({**{key: call.get(key) for key in ("call_id", "role", "budget_owner", "success",
            "transport_outcome", "requested_model", "input_tokens", "output_tokens", "api_usd", "cost_status",
            "context_request_digest", "context_budget_report", "request_trace_ref", "context_sidecars")},
            "prepared_identity_verified": verified,
            "request_sha256": _hash(Path(call["request_trace_ref"])) if trace else None})
    return result


def _terminal_delivery_proof(camp: Path, trace: dict, record: dict) -> dict:
    """Verify exact current feedback in the actual subsequent transport input."""
    from .context.representations import resolve_request_pointer
    from .context.types import ReadRequest, canonical
    from .experiment_observations import ExperimentObservationStore
    payload = json.loads(trace["user"])
    def expand(value):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(payload, value["$context_pointer"]))
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value
    delivered = expand(payload)
    # The v2 Planner input renders each card's science body as one flat row.
    rows = delivered.get("observations") or []
    chosen = [row for row in rows if row.get("observation_id") == record["record_id"] and row.get("current_feedback") is True]
    if len(chosen) != 1:
        return {"verified": False, "reason": "current_terminal_card_missing_or_nonunique"}
    row, original = chosen[0], record["body"]
    from .memory_delivery import project_memory_card
    expected = project_memory_card({**record, "current_feedback": True}, role="research_planner")
    expected_science = {**expected["body"], "observation_id": expected["record_id"]}
    delivered_science = {key: value for key, value in row.items() if key not in
                         {"source_refs", "current_feedback", "condition_differences", "applicability_status"}}
    exact = {key: canonical(delivered_science.get(key)) == canonical(value) for key, value in expected_science.items()}
    body_exact = canonical(delivered_science) == canonical(expected_science)
    reference = record["source_refs"][0]
    reference_exact = canonical(row.get("source_refs") or []) == canonical(expected["source_refs"])
    store = ExperimentObservationStore(camp)
    source = store.registry.get(reference["source_ref"], role="research_planner")
    original_exact = store.registry.load(source, role="research_planner") == original
    read = store.registry.read(ReadRequest(reference["source_ref"], reference["source_hash"],
        "/native_facts/metrics", max_chars=2000), role="research_planner")
    metrics_exact = read.status == "ready" and read.content == original["native_facts"]["metrics"]
    return {"verified": body_exact and reference_exact and original_exact and metrics_exact,
        "delivery_view": "scientific_memory_card",
        "observation_id": original["observation_id"], "exact_fields": exact,
        "complete_science_body_exact": body_exact, "source_reference_identity_exact": reference_exact,
        "original_registry_readback_exact": original_exact,
        "bounded_metrics_readback_exact": metrics_exact, "source_ref": reference["source_ref"],
        "source_hash": reference["source_hash"], "actual_Planner_request_trace": trace["call_id"]}


def run_evaluation(root: Path, *, real_api: bool, max_api_calls: int = 18,
                   max_ticks: int = 2000, wall_timeout: float = 600, poll_seconds: float = .1,
                   _sdk_mock: bool = False) -> dict:
    """Run one native selected experiment then one next Planner without dispatch.

    _sdk_mock is reserved for the public offline transport regression. It must
    identify an installed SDK MockTransport subclass; it cannot inject roles or
    skip business/source/check/approval/measurement validators.
    """
    if not 1 <= max_api_calls <= 18:
        raise ValueError("closed_loop_api_cap_must_be_between_1_and_18")
    if not real_api and not _sdk_mock:
        raise ValueError("closed_loop_requires_explicit_real_api_flag")
    if real_api and _sdk_mock:
        raise ValueError("closed_loop_transport_modes_conflict")
    from . import jobs, llm, loop, worker
    from react_agent.fmri.llm.deepseek import DeepSeekBackend
    if _sdk_mock and not getattr(DeepSeekBackend, "closed_loop_sdk_mock", False):
        raise ValueError("closed_loop_sdk_mock_transport_proof_missing")
    if llm.external_api_disabled():
        raise ValueError("closed_loop_external_api_disabled_by_operator")
    if real_api:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("closed_loop_api_key_missing")
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("closed_loop_requires_empty_directory")
    root.mkdir(parents=True, exist_ok=True)
    from .contract import freeze_contract
    from .execution_protocol import build_execution_protocol
    from .experiment_observations import ExperimentObservationStore
    from .selected_plan import load_plan
    from .verification_cache import verification_scope
    started = time.monotonic()
    first_plan = None
    terminal = None
    next_plan = None
    failure = None
    ticks = 0
    with _cpu_environment():
        design, data_manifest = create_synthetic_data(root / "synthetic_data")
        protocol = build_execution_protocol(design, root / "synthetic_data")
        contract = freeze_contract(design, root / "synthetic_data")
        loop.create_campaign(root, goal=_goal(max_api_calls), contract=contract, protocol=protocol,
                             request_id="isolated-synthetic-closed-loop")
        camp = root / "synthetic_closed_loop"
        planner = llm.role_backend(camp, "research_planner")
        services = worker.build_services(camp)
        try:
            for ticks in range(1, max_ticks + 1):
                if time.monotonic() - started >= wall_timeout:
                    failure = "closed_loop_wall_limit"
                    break
                with verification_scope():
                    state = loop.tick(camp, planner, services=services)
                current = load_plan(camp, state)
                if current and first_plan is None:
                    first_plan = current
                    if first_plan["proposal"]["task_kind"] != "new_method":
                        failure = "first_selected_task_is_not_new_method"
                        break
                if int(_read(camp / "cost.json").get("llm_calls") or 0) > max_api_calls:
                    raise RuntimeError("closed_loop_native_api_cap_exceeded")
                if state.get("status") in {"blocked", "paused", "cancelled", "finished"}:
                    failure = state.get("detail") or "native_" + state["status"]
                    break
                if current and current["plan_id"] == first_plan["plan_id"] and current.get("terminal_observation_id"):
                    terminal = current
                    if current["status"] != "observation_ready":
                        failure = "selected_task_terminal_" + current["status"]
                        break
                if terminal and current and current["plan_id"] != terminal["plan_id"]:
                    next_plan = current
                    break  # Selection committed; no internal step of task two.
                if state.get("live_job"):
                    time.sleep(poll_seconds)
            else:
                failure = "closed_loop_tick_limit"
        except (llm.LlmUnavailable, ValueError, OSError, KeyError, RuntimeError) as exc:
            failure = type(exc).__name__ + ":" + str(exc)
        finally:
            state = loop.load_state(camp)
            if state.get("live_job"):
                record = jobs.stop_job(camp / "jobs" / state["live_job"])
                loop._record_job(camp, state, record)
                state.update(live_job=None, status="blocked", detail=failure or "closed_loop_stopped")
                loop.save_state(camp, state)
        calls = _rows(camp / "llm_calls.jsonl")
        requests = _request_proofs(camp, calls)
        jobs_report = []
        for directory in sorted((camp / "jobs").glob("*")):
            record = _read(directory / "job.json")
            paths = [directory / name for name in ("metrics.json", "source_binding.json", "evaluation_identity.json",
                "hook_config.json", "run_record.json", "settlement.json", "last.ckpt") if (directory / name).is_file()]
            jobs_report.append({"job_id": record.get("job_id"), "candidate_id": record.get("candidate_id"),
                "status": record.get("status"), "seed": record.get("seed"), "fidelity": record.get("fidelity"),
                "native_slot_seconds": record.get("gpu_seconds"), "command": record.get("command"),
                "native_accepted_result": record.get("result"),
                "artifacts": [{"path": str(path), "sha256": _hash(path)} for path in paths]})
        card = None
        readback = False
        current_delivered = False
        terminal_delivery = {"verified": False, "reason": "terminal_or_next_Planner_unavailable"}
        if terminal:
            store = ExperimentObservationStore(camp)
            records = store.records(limit=1, observation_id=terminal["terminal_observation_id"])
            if records:
                card = records[0]["body"]
                reference = records[0]["source_refs"][0]
                source = store.registry.get(reference["source_ref"], role="research_planner")
                readback = store.registry.load(source, role="research_planner") == card
                subsequent = [call for call in calls if call.get("role") == "research_planner"]
                if len(subsequent) >= 2:
                    terminal_delivery = _terminal_delivery_proof(camp,
                        _read(Path(subsequent[-1]["request_trace_ref"])), records[0])
                    current_delivered = terminal_delivery["verified"]
        cost = _read(camp / "cost.json")
        required_roles = {"research_planner", "experiment_designer", "candidate_coder", "candidate_reviewer", "result_analyst"}
        counts = dict(Counter(call.get("role") for call in calls))
        second_dispatched = bool(next_plan and any(row.get("kind") in {"step_intent", "step_completed"}
            for row in next_plan.get("step_receipts") or []))
        complete = bool(terminal and next_plan and readback and current_delivered and card
            and card.get("native_facts", {}).get("metrics") is not None
            and len(jobs_report) == 2 and all(row["status"] == "finished" for row in jobs_report)
            and required_roles <= set(counts) and all(row["prepared_identity_verified"] for row in requests)
            and not second_dispatched and next_plan.get("current_step") == "prepare" and not failure)
        if not complete and failure is None:
            failure = "closed_loop_acceptance_incomplete"
        report = {"schema_version": "eeg.synthetic.native_closed_loop.v1", "ok": complete, "failure": failure,
            "root": str(root), "campaign": str(camp), "synthetic": True,
            "scientific_scope": "Random synthetic EEG-shaped data; infrastructure acceptance only, no EEG efficacy claim.",
            "transport_mode": "real_provider" if real_api else "sdk_mock", "physical_GPU_jobs": 0, "physical_GPU_seconds": 0,
            "native_metering_note": "Native gpu_seconds retains conservative execution-slot elapsed accounting for CPU jobs; it is not physical GPU usage.",
            "native_slot_seconds": cost.get("gpu_seconds_used"), "native_max_slot_seconds": 600,
            "native_API_attempts": len(calls), "actual_external_API_attempts": len(calls) if real_api else 0,
            "sdk_mock_attempts": len(calls) if _sdk_mock else 0, "max_API_attempts": max_api_calls,
            "role_API_attempts": counts, "paid_maintenance_API_attempts": sum(counts.get(role, 0)
                for role in ("memory_curator", "context_summary", "memory_extraction", "memory_consolidation")),
            "cost": cost, "dataset_manifest": data_manifest, "first_plan": first_plan, "terminal_plan": terminal,
            "next_selected_plan": next_plan, "second_task_dispatched": second_dispatched, "terminal_observation": card,
            "terminal_original_readback_verified": readback, "terminal_delivered_to_next_Planner": current_delivered,
            "terminal_delivery_proof": terminal_delivery,
            "requests": requests, "jobs": jobs_report, "ticks": ticks, "elapsed_seconds": time.monotonic() - started}
        _write(root / "closed_loop_evaluation_report.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--max-api-calls", type=int, default=18)
    parser.add_argument("--wall-timeout", type=float, default=600)
    args = parser.parse_args()
    report = run_evaluation(args.root, real_api=args.real_api, max_api_calls=args.max_api_calls, wall_timeout=args.wall_timeout)
    print(json.dumps({"ok": report["ok"], "failure": report["failure"], "API_attempts": report["native_API_attempts"],
        "physical_GPU_jobs": 0, "native_slot_seconds": report["native_slot_seconds"],
        "report": str(args.root / "closed_loop_evaluation_report.json")}, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
