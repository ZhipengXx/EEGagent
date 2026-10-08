"""Native synthetic GPU-check process, recovery identity and shared cost settlement."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from react_agent.eeg_training.inference_consistency import file_digest, object_digest


class ProbeRecoveryBlocked(RuntimeError):
    """An original probe is still live or its billable completion is uncertain."""
    def __init__(self, result: dict):
        self.result = result
        self.recoverable = result.get("status") in {"running", "uncertain_pending", "blocked_live_child"}
        super().__init__("multigpu_probe_recovery_blocked:" + str(result.get("error")))


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); tmp.replace(path)


def refresh_probe_budget(camp: Path, state: dict) -> None:
    """Refresh the caller's in-memory quota after native probe cost settlement."""
    if state.get("evaluation_mode") == "loso_method_search":
        state["gpu_seconds_left"] = float(state["max_gpu_seconds"]) - float(_read(camp / "cost.json").get("gpu_seconds_used") or 0)


def _settle(directory: Path) -> dict:
    """Settle once in the existing cost ledger, retaining uncertainty and failed runs."""
    import fcntl
    from react_agent.eeg_research.agentic.budget import charge_gpu
    request, intent, record = _read(directory / "request.json"), _read(directory / "intent.json"), _read(directory / "process.json")
    if file_digest(directory / "request.json") != intent.get("request_sha256"):
        raise RuntimeError("multigpu_probe_request_changed_before_settlement")
    camp = Path(request["camp"])
    if record.get("status") not in {"finished", "failed", "deadline", "uncertain"}:
        raise RuntimeError("multigpu_probe_not_terminal")
    actual = (float(record["ended_at"]) - float(intent["started_at"])) * len(request["gpu"])
    if record["status"] == "uncertain":
        actual = actual if record.get("verified_child_alive_past_deadline") else float(intent["max_gpu_seconds"])
    with (camp / "multigpu_cost.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state, cost = _read(camp / "campaign_state.json"), _read(camp / "cost.json")
        accounting = {"gpu_seconds_left": float(state["max_gpu_seconds"]) - float(cost.get("gpu_seconds_used") or 0),
                      "gpu_seconds_reserved": 0, "training_jobs": int(state.get("training_jobs") or cost.get("training_jobs") or 0)}
        charge_gpu(camp, accounting, max(0.0, actual), job_id="engineering_multigpu_" + request["probe_identity"])
    result = {**record, "gpu_seconds": max(0.0, actual), "probe_identity": request["probe_identity"],
              "cost_ledger": str(camp / "cost.json"), "data_scope": "synthetic_training_inputs_only",
              "request_sha256": file_digest(directory / "request.json")}
    _write(directory / "receipt.json", result)
    return result


def validate_receipt(camp: Path, protocol: dict, receipt: dict, *, workspace: Path | None = None) -> dict:
    """Require current source/config/recipe and settled real replica/checkpoint evidence."""
    from react_agent.eeg_research.agentic.jobs import baseline_manifest
    directory = Path(camp) / "engineering_checks" / "multigpu" / str(receipt.get("probe_identity") or "")
    request, intent = _read(directory / "request.json"), _read(directory / "intent.json")
    if not request or receipt.get("status") != "finished" or receipt.get("ok") is not True:
        raise ValueError("multigpu_probe_successful_receipt_required")
    if (file_digest(directory / "request.json") != intent.get("request_sha256")
            or intent.get("request_sha256") != receipt.get("request_sha256")
            or _read(directory / "receipt.json") != receipt):
        raise ValueError("multigpu_probe_receipt_or_request_changed")
    source = workspace / "extension" / "eeg_candidate.py" if workspace is not None else Path(baseline_manifest()["entry"])
    assigned = _read(workspace / "spec.json") if workspace is not None else {}
    assigned = assigned.get("experiment") or assigned
    config = {key: assigned.get(key) or {} for key in ("model", "objective", "transform")}
    parent = _read(Path(camp) / "method_search_manifest.json")
    inventory = parent["gpu_inventory"]
    expected_inventory = [{key: row[key] for key in ("index", "uuid", "name", "total_memory_mib")} for row in inventory]
    samples = len(protocol.get("train_query_ids") or [])
    if (request.get("manifest_hash") != parent.get("manifest_hash") or parent.get("manifest_hash") != object_digest({key: value for key, value in parent.items() if key != "manifest_hash"})
            or request.get("source_sha256") != file_digest(source) or request.get("hook_config") != config
            or request.get("gpu") != protocol["gpu"] or request.get("gpu_inventory") != expected_inventory
            or request.get("input_geometry") != protocol["input_geometry"] or request.get("batch_size") != protocol["batch_size"]
            or request.get("tail_batch_size") != (samples % protocol["batch_size"] or protocol["batch_size"])
            or request.get("lr") != protocol["lr"] or request.get("weight_decay") != protocol["weight_decay"]
            or request.get("negative_sampling_policy") != protocol["negative_sampling_policy"]
            or request.get("checkpoint_policy") != protocol["checkpoint_policy"]
            or request.get("probe_source_sha256") != file_digest(Path(__file__).with_name("multigpu_probe.py"))
            or request.get("manager_source_sha256") != file_digest(Path(__file__))
            or request.get("training_entry_sha256") != file_digest(Path(__file__).resolve().parents[2] / "eeg_training" / "train_entry.py")):
        raise ValueError("multigpu_probe_current_source_config_or_recipe_mismatch")
    result = receipt.get("result") or {}
    if (result.get("ok") is not True or result.get("probe_identity") != request["probe_identity"]
            or result.get("source_sha256") != request["source_sha256"] or result.get("gpu") != request["gpu"]
            or result.get("gpu_inventory") != request["gpu_inventory"] or result.get("checkpoint_round_trip") is not True
            or result.get("synthetic_training_inputs_only") is not True or result.get("real_EEG_or_held_out_read") is not False
            or result.get("eval_transform_calls") != 0 or result.get("precision") != "fp32"):
        raise ValueError("multigpu_probe_actual_execution_evidence_mismatch")
    batches = result.get("batches") or []
    for position, count in (("first", request["batch_size"]), ("tail", request["tail_batch_size"])):
        rows = [row for row in batches if row.get("position") == position]
        if len(rows) != 1 or rows[0].get("total_batch_size") != count or rows[0].get("gradient_tensor_count", 0) < 1:
            raise ValueError("multigpu_probe_actual_batch_or_gradient_missing")
        replicas = rows[0].get("replicas") or []
        if sum(row.get("local_batch_size", 0) for row in replicas) != count:
            raise ValueError("multigpu_probe_actual_replica_population_mismatch")
        if position == "first" and {row.get("input_device") for row in replicas} != {f"cuda:{i}" for i in range(len(request["gpu"]))}:
            raise ValueError("multigpu_probe_actual_card_participation_missing")
        if any(row.get("parameter_devices") != [row.get("input_device")] or row.get("loss_device") != row.get("input_device") for row in replicas):
            raise ValueError("multigpu_probe_actual_replica_device_mismatch")
    cost = _read(Path(camp) / "cost.json")
    checkpoint = directory / "synthetic_checkpoint.ckpt"
    if not checkpoint.is_file() or file_digest(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("multigpu_probe_checkpoint_receipt_changed")
    if "engineering_multigpu_" + request["probe_identity"] not in cost.get("settled_job_ids", []):
        raise ValueError("multigpu_probe_gpu_cost_not_settled")
    return receipt


def reconcile(directory: Path) -> dict:
    """Reattach the original watchdog/PID, or return its immutable result; never relaunch."""
    from react_agent.eeg_research.agentic.process_identity import _proc_start, _process_state
    intent, record = _read(directory / "intent.json"), _read(directory / "process.json")
    if not intent:
        raise RuntimeError("multigpu_probe_launch_intent_missing")
    if record.get("status") in {"finished", "failed", "deadline", "uncertain"}:
        return _settle(directory)
    if record.get("status") == "blocked_live_gpu_child_after_deadline":
        return {**record, "ok": False, "error": "multigpu_probe_gpu_process_alive_after_kill"}
    if record and _proc_start(int(record.get("pid") or 0)) == record.get("proc_start") and _process_state(int(record["pid"])) in {"R", "S", "D"}:
        return {**record, "ok": False, "error": "multigpu_probe_running", "probe_identity": _read(directory / "request.json")["probe_identity"]}
    if time.time() < float(intent["deadline"]):
        return {"ok": False, "status": "uncertain_pending", "error": "multigpu_probe_pid_publication_uncertain", "deadline": intent["deadline"]}
    # If the supervisor died after publishing its child, do not ignore a live
    # GPU process. Only its verified start token authorizes terminating it.
    child_pid = int(record.get("child_pid") or 0)
    if child_pid and _proc_start(child_pid) == record.get("child_proc_start") and _process_state(child_pid) in {"R", "S", "D"}:
        os.killpg(child_pid, signal.SIGKILL)
        _write(directory / "process.json", {**record, "verified_child_alive_past_deadline": True})
        return {"ok": False, "status": "blocked_live_child", "error": "multigpu_probe_waiting_for_verified_child_exit"}
    _write(directory / "process.json", {**record, "ok": False, "status": "uncertain", "error": "multigpu_probe_unconfirmed_launch_or_completion",
                                       "ended_at": time.time()})
    return _settle(directory)


def recover_probes(camp: Path, state: dict) -> list[dict]:
    """Settle previous engineering work before a resumed worker makes paid calls."""
    results = []
    for directory in sorted((camp / "engineering_checks" / "multigpu").glob("*")):
        if not (directory / "intent.json").is_file(): continue
        result = reconcile(directory); results.append(result)
        if result.get("status") in {"running", "uncertain_pending", "blocked_live_child", "blocked_live_gpu_child_after_deadline", "uncertain"}:
            refresh_probe_budget(camp, state)
            raise ProbeRecoveryBlocked(result)
    refresh_probe_budget(camp, state)
    return results


def run_multigpu_check(camp: Path, protocol: dict, *, workspace: Path | None = None, python: str | None = None,
                       timeout_s: float = 120.0) -> dict:
    """Run/recover a source-bound native engineering check; baseline needs no candidate workspace."""
    from react_agent.eeg_research.agentic.runner import research_env
    from react_agent.eeg_research.agentic.jobs import BASELINE_MODULE, baseline_manifest
    from react_agent.eeg_training.protocol import torch_python
    camp = Path(camp).resolve()
    if protocol.get("evaluation_mode") != "loso_method_search":
        return {"ok": False, "error": "multigpu_probe_requires_opt_in_mode"}
    if protocol.get("negative_sampling_policy") != "data_parallel_local" or int(protocol["batch_size"]) != 1024:
        return {"ok": False, "error": "multigpu_probe_recipe_mismatch"}
    parent = _read(camp / "method_search_manifest.json")
    if parent.get("manifest_hash") != object_digest({key: value for key, value in parent.items() if key != "manifest_hash"}):
        return {"ok": False, "status": "blocked", "error": "multigpu_probe_manifest_digest_mismatch"}
    cards, inventory = list(protocol["gpu"]), parent.get("gpu_inventory") or []
    if cards != parent.get("gpu") or len(cards) < 2 or [row["index"] for row in inventory] != cards:
        return {"ok": False, "error": "multigpu_probe_frozen_cards_missing_or_mismatched"}
    if not python: python = torch_python()
    if not python: return {"ok": False, "error": "torch_python_missing"}
    assigned = _read(workspace / "spec.json") if workspace is not None else {}
    assigned = assigned.get("experiment") or assigned
    source = workspace / "extension" / "eeg_candidate.py" if workspace is not None else Path(baseline_manifest()["entry"])
    config = {key: assigned.get(key) or {} for key in ("model", "objective", "transform")}
    samples = len(protocol.get("train_query_ids") or [])
    if not samples or samples < 1024:
        return {"ok": False, "error": "multigpu_probe_real_training_population_missing"}
    identity = {"source_sha256": file_digest(source), "hook_config": config, "gpu": cards, "manifest_hash": parent["manifest_hash"],
        "gpu_inventory": [{key: row[key] for key in ("index", "uuid", "name", "total_memory_mib")} for row in inventory],
        "input_geometry": protocol["input_geometry"], "batch_size": 1024, "tail_batch_size": (samples % 1024) or 1024,
        "lr": protocol["lr"], "weight_decay": protocol["weight_decay"], "negative_sampling_policy": "data_parallel_local",
        "checkpoint_policy": protocol["checkpoint_policy"], "python": python,
        "probe_source_sha256": file_digest(Path(__file__).with_name("multigpu_probe.py")),
        "manager_source_sha256": file_digest(Path(__file__)),
        "training_entry_sha256": file_digest(Path(__file__).resolve().parents[2] / "eeg_training" / "train_entry.py")}
    probe_id = object_digest(identity)
    directory = camp / "engineering_checks" / "multigpu" / probe_id
    from react_agent.eeg_research.agentic.loso_study import gpu_inventory
    actual_inventory = gpu_inventory(tuple(cards), allow_shared=True)
    if [{key: row[key] for key in ("index", "uuid", "name", "total_memory_mib")} for row in actual_inventory] != identity["gpu_inventory"]:
        return {"ok": False, "error": "multigpu_probe_physical_gpu_identity_changed"}
    if (directory / "intent.json").exists():
        result = reconcile(directory)
        until = float(_read(directory / "intent.json")["deadline"]) + 5.0
        while result.get("error") == "multigpu_probe_running" and time.time() < until:
            time.sleep(0.25); result = reconcile(directory)
        return result
    state, cost = _read(camp / "campaign_state.json"), _read(camp / "cost.json")
    if state.get("live_job"):
        return {"ok": False, "status": "blocked", "error": "multigpu_probe_requires_idle_training_runtime"}
    available = float(state["max_gpu_seconds"]) - float(cost.get("gpu_seconds_used") or 0) - float(state.get("gpu_seconds_reserved") or 0)
    cap = float(timeout_s) * len(cards)
    if available < cap or timeout_s <= 1:
        return {"ok": False, "status": "blocked", "error": "multigpu_probe_budget_insufficient", "required_gpu_seconds": cap, "available_gpu_seconds": available}
    # Inventory is read in the actual process environment; mutable occupancy is
    # not confused with the immutable physical UUID/card binding.
    directory.mkdir(parents=True, exist_ok=False)
    request = {**identity, "probe_identity": probe_id, "camp": str(camp), "source_path": str(source.resolve()),
        "module": "eeg_candidate" if workspace is not None else BASELINE_MODULE, "probe_dir": str(directory), "timeout_s": timeout_s}
    _write(directory / "request.json", request)
    started = time.time()
    _write(directory / "intent.json", {"started_at": started, "deadline": started + timeout_s,
        "max_gpu_seconds": cap, "request_sha256": file_digest(directory / "request.json")})
    extension = workspace / "extension" if workspace is not None else None
    env = research_env(extension)
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, cards)), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    with (directory / "supervisor.log").open("a") as output:
        subprocess.Popen([python, "-m", "react_agent.eeg_research.agentic.multigpu_check", "--supervise", str(directory)],
            cwd=str(extension) if extension is not None else None, env=env, stdout=output, stderr=output, start_new_session=True)
    # The detached supervisor publishes its own PID and owns the deadline even
    # if the worker exits immediately after Popen and before this polling loop.
    while True:
        record = _read(directory / "process.json")
        if record.get("status") in {"finished", "failed", "deadline", "uncertain", "blocked_live_gpu_child_after_deadline"}:
            return reconcile(directory)
        if time.time() >= started + timeout_s + 5.0:
            return reconcile(directory)
        time.sleep(0.25)


def supervise(directory: Path) -> None:
    """Own the GPU-child deadline independently of the research worker lifecycle."""
    from react_agent.eeg_research.agentic.process_identity import _proc_start, _process_state
    request, intent = _read(directory / "request.json"), _read(directory / "intent.json")
    if file_digest(directory / "request.json") != intent["request_sha256"]:
        raise RuntimeError("multigpu_probe_request_changed")
    process = {"status": "running", "pid": os.getpid(), "proc_start": _proc_start(os.getpid()), "started_at": time.time()}
    _write(directory / "process.json", process)
    with (directory / "probe.stdout").open("w") as stdout, (directory / "probe.stderr").open("w") as stderr:
        child = subprocess.Popen([sys.executable, "-m", "react_agent.eeg_research.agentic.multigpu_probe", "--request", str(directory / "request.json")],
            stdout=stdout, stderr=stderr, start_new_session=True)
        process.update(child_pid=child.pid, child_proc_start=_proc_start(child.pid)); _write(directory / "process.json", process)
        deadline = max(time.time(), float(intent["deadline"]) - 1.0)
        while child.poll() is None and time.time() < deadline: time.sleep(0.1)
        timed_out = child.poll() is None
        if timed_out:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=0.3)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
        while child.poll() is None:
            _write(directory / "process.json", {**process, "status": "blocked_live_gpu_child_after_deadline",
                                                "error": "multigpu_probe_gpu_process_alive_after_kill"})
            time.sleep(0.5)
    lines = [line for line in (directory / "probe.stdout").read_text().splitlines() if line.strip().startswith("{")]
    try: payload = json.loads(lines[-1]) if lines else {}
    except ValueError: payload = {}
    ended = time.time()
    result = {**process, "result": payload, "ok": bool(payload.get("ok")) and child.returncode == 0 and not timed_out,
              "status": "deadline" if timed_out else "finished" if payload.get("ok") and child.returncode == 0 else "failed",
              "ended_at": ended, "returncode": child.returncode,
              "error": "multigpu_probe_deadline" if timed_out else payload.get("error") or (None if payload.get("ok") else "multigpu_probe_failed"),
              "stderr_tail": (directory / "probe.stderr").read_text()[-1500:]}
    _write(directory / "process.json", result); _settle(directory)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--supervise", type=Path, required=True)
    args = parser.parse_args(); supervise(args.supervise); return 0


if __name__ == "__main__":
    raise SystemExit(main())
