"""Pre-training trainable scalar counts for a candidate, as training will report them.

check_entry only records parameter tensor counts, so a requirement such as
"report trainable scalars against baseline" could never be verified before
training. This builds the encoder and objective with the trainer's own hooks
(train_entry._build_hook, hooks.objective_parameters) in a CPU-only child and
sums numel exactly like train_entry. It is a measurement for review, not part
of the frozen evaluator, so it may change without invalidating suites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _count(dataset: str, approved_spec: str | None) -> dict[str, Any]:
    import importlib

    from react_agent.eeg_research.agentic.check_entry import _spec
    from react_agent.eeg_training.hooks import objective_parameters
    from react_agent.eeg_training.train_entry import _build_hook

    assigned = json.loads(Path(approved_spec).read_text()) if approved_spec else {}
    config = assigned.get("experiment") or assigned
    spec = _spec(dataset)
    candidate = importlib.import_module("eeg_candidate").EEGCandidate()
    geometry = {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}
    encoder = _build_hook(candidate, "build_encoder", geometry, config.get("model") or {})
    objective = (_build_hook(candidate, "build_training_objective", {}, config.get("objective") or {})
                 if hasattr(candidate, "build_training_objective") else None)
    extra = objective_parameters(objective)
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    return {
        "status": "measured",
        "parameter_counts": {"encoder": sum(int(p.numel()) for p in encoder.parameters()),
                             "objective": sum(int(p.numel()) for p in extra)},
        "encoder_trainable_scalars": sum(int(p.numel()) for p in trainable),
        "encoder_frozen_scalars": sum(int(p.numel()) for p in encoder.parameters() if not p.requires_grad),
        "basis": "Same hooks and numel sum as train_entry parameter_counts (capabilities_used.json).",
    }


def measure(workspace: Path, timeout_s: float = 120.0) -> dict[str, Any]:
    """Run the census in the training interpreter; never raises."""
    import subprocess

    from react_agent.eeg_research.agentic.runner import research_env
    from react_agent.eeg_training.protocol import torch_python

    interpreter = torch_python()
    extension = (workspace / "extension").resolve()
    if interpreter is None or not (extension / "eeg_candidate.py").is_file():
        return {"status": "unavailable", "reason": "torch_python_or_entry_missing"}
    env = research_env(extension)
    env.pop("EEG_CANDIDATE_MODULE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    spec_path = workspace / "input_spec.json"
    command = [interpreter, "-m", "react_agent.eeg_research.agentic.parameter_census",
               str(spec_path.resolve()) if spec_path.is_file() else "eeg"]
    if (workspace / "spec.json").is_file():
        command.append(str((workspace / "spec.json").resolve()))
    try:
        done = subprocess.run(command, cwd=str(extension), env=env, capture_output=True,  # noqa: S603
                              text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": "census_timeout"}
    lines = [line for line in done.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {"status": "unavailable", "reason": "census_no_output", "detail": done.stderr[-600:]}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"status": "unavailable", "reason": "census_invalid_output"}


def main() -> int:
    try:
        payload = _count(sys.argv[1] if len(sys.argv) > 1 else "eeg", sys.argv[2] if len(sys.argv) > 2 else None)
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "unavailable", "reason": type(exc).__name__, "detail": str(exc)[:600]}
    print(json.dumps(payload))
    return 0 if payload.get("status") == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
