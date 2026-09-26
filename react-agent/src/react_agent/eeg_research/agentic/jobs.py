"""Real training jobs. A lock file alone does not prove a process is alive."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.binding import manifest_for
from react_agent.eeg_research.agentic.execution_protocol import command_matches, fidelity_settings, load_protocol
from react_agent.eeg_research.agentic.runner import accept_job, research_env
from react_agent.eeg_training.protocol import Design, train_command

BASELINE_MODULE = "react_agent.eeg_research.agentic.baseline"
PILOT_EPOCHS = 3


def baseline_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "baseline.py"
    return manifest_for(path.parent, BASELINE_MODULE, path)


def _stat_fields(pid: int) -> list[str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return text.rsplit(")", 1)[1].split()


def _proc_start(pid: int) -> str | None:
    fields = _stat_fields(pid)
    if fields is None or len(fields) <= 19:
        return None
    return fields[19]


def _process_state(pid: int) -> str | None:
    fields = _stat_fields(pid)
    if not fields:
        return None
    return fields[0]


_RUNNING = {"R", "S", "D"}


def worker_disconnected(pid: int) -> bool:
    """True when the worker pid is a zombie, dead, or gone.

    os.kill(pid, 0) succeeds for a zombie, so a signal check alone is not liveness.
    R, S, and D stay alive even when the heartbeat is old.
    """
    if pid <= 0:
        return True
    state = _process_state(pid)
    if state in _RUNNING:
        return False
    return True


def _reap(pid: int) -> None:
    """Collect an exited child. A non-child raises ChildProcessError and is left alone."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return


def start_job(
    job_dir: Path,
    design: Design,
    *,
    candidate_id: str,
    extension: Path | None,
    fidelity: str,
    root: Path,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Start one train_entry child. The candidate module comes from the job environment."""
    job_dir = job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    if (job_dir / "job.json").is_file():
        return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    protocol = None
    if protocol_path is not None and protocol_path.is_file():
        protocol = load_protocol(protocol_path.parent)
    if protocol is None:
        protocol = load_protocol(job_dir.parent.parent)
    if protocol is not None:
        epochs, stop = fidelity_settings(protocol, fidelity)
    else:
        epochs = PILOT_EPOCHS if fidelity == "pilot" else design.epochs
        stop = "single_full" if fidelity == "pilot" else "single_early"
    run_design = replace(design, epochs=epochs, stop=stop, policy="agentic")
    env = research_env(extension)
    env["EEG_CANDIDATE_MODULE"] = "eeg_candidate" if extension is not None else BASELINE_MODULE
    env["EEG_FINAL_TEST"] = "0"
    manifest = baseline_manifest() if extension is None else json.loads(
        (extension.parent / "source_manifest.json").read_text(encoding="utf-8")
    )
    command = train_command(run_design, job_dir, root)
    if protocol is not None and not command_matches(command, protocol, fidelity):
        record = {
            "job_id": job_dir.name,
            "candidate_id": candidate_id,
            "fidelity": fidelity,
            "status": "blocked",
            "detail": "protocol_mismatch",
            "command": command,
            "execution_fingerprint": protocol.get("fingerprint"),
        }
        (job_dir / "job.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
    log = (job_dir / "train.log").open("wb")
    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=str(job_dir),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record = {
        "job_id": job_dir.name,
        "candidate_id": candidate_id,
        "fidelity": fidelity,
        "epochs": epochs,
        "seed": run_design.seed,
        "gpu": list(run_design.gpu),
        "pid": proc.pid,
        "proc_start": _proc_start(proc.pid),
        "nonce": uuid.uuid4().hex,
        "started_at": time.time(),
        "status": "running",
        "manifest": manifest,
        "command": command,
        "execution_fingerprint": None if protocol is None else protocol.get("fingerprint"),
    }
    (job_dir / "job.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _alive(record: dict[str, Any]) -> bool:
    """True only while this same process is still scheduled.

    os.kill(pid, 0) succeeds for a zombie. Zombie and dead states are exited, and an
    exited child is reaped so the next reconcile can settle the job from its artifacts.
    """
    pid = int(record.get("pid") or 0)
    if pid <= 0:
        return False
    state = _process_state(pid)
    if state is None:
        return False
    if state in {"Z", "X"}:
        _reap(pid)
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return _proc_start(pid) == record.get("proc_start")


def reconcile(job_dir: Path) -> dict[str, Any]:
    """Settle a job from its process and artifacts. Metrics count only after the child exits."""
    path = job_dir / "job.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "running":
        return record
    if _alive(record):
        status = job_dir / "status.json"
        record["heartbeat"] = status.stat().st_mtime if status.is_file() else None
        return record
    marks = [path.stat().st_mtime for path in (job_dir / "metrics.json", job_dir / "status.json", job_dir / "error.txt") if path.is_file()]
    record["ended_at"] = record.get("ended_at") or (max(marks) if marks else time.time())
    elapsed = float(record["ended_at"]) - float(record["started_at"])
    record["gpu_seconds"] = elapsed * max(1, len(record.get("gpu") or []))
    protocol = load_protocol(job_dir.parent.parent)
    accepted = accept_job(job_dir, record.get("manifest"), str(record.get("fidelity")), protocol=protocol)
    record["result"] = accepted
    if accepted.get("evaluation_valid"):
        record["status"] = "finished"
    elif (job_dir / "metrics.json").is_file():
        record["status"] = "invalid"
    else:
        record["status"] = "failed"
        error = job_dir / "error.txt"
        record["error"] = error.read_text(encoding="utf-8")[:500] if error.is_file() else "no_metrics"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def stop_job(job_dir: Path) -> dict[str, Any]:
    """Stop the whole process group, including DataParallel workers."""
    path = job_dir / "job.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") == "running" and _alive(record):
        try:
            os.killpg(int(record["pid"]), signal.SIGTERM)
        except OSError:
            pass
        record["status"] = "cancelled"
        record["ended_at"] = time.time()
        elapsed = float(record["ended_at"]) - float(record["started_at"])
        record["gpu_seconds"] = elapsed * max(1, len(record.get("gpu") or []))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
