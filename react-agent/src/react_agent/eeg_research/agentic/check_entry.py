"""Candidate interface check. Runs in the training interpreter, never in the workbench."""

from __future__ import annotations

import importlib
import inspect
import io
import json
import sys
import traceback
from pathlib import Path


def run(dataset: str = "eeg") -> dict[str, object]:
    import torch

    from react_agent.eeg_training.protocol import geometry

    spec = geometry(dataset)
    module = importlib.import_module("eeg_candidate")
    candidate = module.EEGCandidate()
    encoder = candidate.build_encoder({"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])})
    length = int(spec["timesteps"][1]) - int(spec["timesteps"][0])
    uses_statistics = hasattr(candidate, "fit_statistics")
    stats_in_state = None
    if uses_statistics:
        buffers_before = {name: value.clone() for name, value in encoder.named_buffers()}
        params_before = {name for name, _ in encoder.named_parameters()}
        candidate.fit_statistics(
            encoder,
            {
                "mean": [0.5] * int(spec["c_num"]),
                "std": [2.0] * int(spec["c_num"]),
                "values_per_channel": 1000,
                "source": "synthetic_check",
                "subject_ids": None,
            },
        )
        changed = [
            name
            for name, value in encoder.named_buffers()
            if name not in buffers_before or not torch.equal(buffers_before[name], value)
        ]
        new_params = {name for name, _ in encoder.named_parameters()} - params_before
        stats_in_state = bool(changed) and not new_params
    batch = torch.randn(4, int(spec["c_num"]), length)
    encoder.train()
    out = encoder(batch)
    finite = bool(torch.isfinite(out).all())
    out.float().pow(2).mean().backward()
    trainable = [param for param in encoder.parameters() if param.requires_grad]
    with_grad = sum(1 for param in trainable if param.grad is not None and bool(torch.isfinite(param.grad).all()))
    encoder.eval()
    with torch.no_grad():
        first = encoder(batch)
    buffer = io.BytesIO()
    torch.save(encoder.state_dict(), buffer)
    buffer.seek(0)
    encoder.load_state_dict(torch.load(buffer))
    with torch.no_grad():
        second = encoder(batch)
    round_trip = bool(torch.allclose(first, second))
    loaded = str(Path(inspect.getfile(type(candidate))).resolve())
    ok = finite and out.shape == (4, 1024) and with_grad > 0 and round_trip
    if uses_statistics and not stats_in_state:
        ok = False
    return {
        "uses_train_statistics": uses_statistics,
        "statistics_stored_as_buffers": stats_in_state,
        "ok": ok,
        "shape": list(out.shape),
        "finite": finite,
        "params_with_grad": with_grad,
        "params_trainable": len(trainable),
        "checkpoint_round_trip": round_trip,
        "file": loaded,
        "python": sys.executable,
        "torch": torch.__version__,
    }


def main() -> int:
    try:
        payload = run(sys.argv[1] if len(sys.argv) > 1 else "eeg")
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": type(exc).__name__, "detail": traceback.format_exc()[-1500:]}
    print(json.dumps(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
