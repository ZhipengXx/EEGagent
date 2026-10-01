"""Candidate interface check. Runs in the training interpreter, never in the workbench."""

from __future__ import annotations

import importlib
import inspect
import io
import json
import sys
import traceback
from pathlib import Path


def _spec(dataset_or_path: str) -> dict[str, object]:
    path = Path(dataset_or_path)
    if path.suffix == ".json" and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("c_num") is not None:
            return payload
    from react_agent.eeg_training.protocol import geometry

    return geometry(dataset_or_path)


def run(dataset: str = "eeg", *, context: dict[str, object] | None = None) -> dict[str, object]:
    import torch

    context = context if context is not None else {}
    context.update({"python": sys.executable, "torch": torch.__version__,
                    "file": str((Path.cwd() / "eeg_candidate.py").resolve()), "stage": "input_spec"})
    spec = _spec(dataset)
    context["stage"] = "import_candidate"
    module = importlib.import_module("eeg_candidate")
    context["file"] = str(Path(inspect.getfile(module)).resolve())
    context["stage"] = "build_encoder"
    candidate = module.EEGCandidate()
    encoder = candidate.build_encoder({"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])})
    length = int(spec["timesteps"][1]) - int(spec["timesteps"][0])
    uses_statistics = hasattr(candidate, "fit_statistics")
    stats_in_state = None
    statistics_debug = None
    if uses_statistics:
        context["stage"] = "fit_statistics"
        buffers_before = {name: value.clone() for name, value in encoder.named_buffers()}
        pre_hooks_before = len(encoder._forward_pre_hooks)
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
        stats_in_state = bool(changed) and not new_params and all(name in encoder.state_dict() for name in changed)
        statistics_debug = {"buffers_before_fitting": list(buffers_before), "changed_buffers": changed,
                            "forward_pre_hooks_before_fitting": pre_hooks_before, "forward_pre_hooks_after_fitting": len(encoder._forward_pre_hooks)}
    batch = torch.randn(4, int(spec["c_num"]), length)
    context["stage"] = "train_forward"
    encoder.train()
    out = encoder(batch)
    finite = bool(torch.isfinite(out).all())
    context["stage"] = "backward"
    out.float().pow(2).mean().backward()
    trainable = [param for param in encoder.parameters() if param.requires_grad]
    with_grad = sum(1 for param in trainable if param.grad is not None and bool(torch.isfinite(param.grad).all()))
    finite_gradients = all(bool(torch.isfinite(param.grad).all()) for param in trainable if param.grad is not None)
    context["stage"] = "eval_forward"
    encoder.eval()
    with torch.no_grad():
        first = encoder(batch)
    buffer = io.BytesIO()
    context["stage"] = "checkpoint_round_trip"
    torch.save(encoder.state_dict(), buffer)
    buffer.seek(0)
    rebuilt = candidate.build_encoder({"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])})
    if statistics_debug is not None:
        statistics_debug["buffers_in_fresh_encoder"] = [name for name, _ in rebuilt.named_buffers()]
        statistics_debug["forward_pre_hooks_in_fresh_encoder"] = len(rebuilt._forward_pre_hooks)
    rebuilt.load_state_dict(torch.load(buffer))
    rebuilt.eval()
    with torch.no_grad():
        second = rebuilt(batch)
    round_trip = bool(torch.allclose(first, second))
    loaded = str(Path(inspect.getfile(type(candidate))).resolve())
    failures = []
    if not finite:
        failures.append("non_finite_output")
    if out.shape != (4, 1024):
        failures.append("embedding_shape_mismatch")
    if with_grad == 0:
        failures.append("no_finite_trainable_gradients")
    if not finite_gradients:
        failures.append("non_finite_gradients")
    if not round_trip:
        failures.append("checkpoint_output_mismatch")
    if uses_statistics and not stats_in_state:
        failures.append("statistics_not_persistent_buffers")
    hints = {
        "statistics_not_persistent_buffers": "Initialize persistent registered buffers in the encoder constructor; update them in fit_statistics. Do not use plain attributes or new parameters.",
        "embedding_shape_mismatch": f"Expected [4, 1024], received {list(out.shape)}. Derive projection widths from actual pooling geometry.",
        "no_finite_trainable_gradients": "Keep the encoder differentiable and create trainable modules in __init__, not forward.",
        "checkpoint_output_mismatch": "Rebuild the same encoder and buffer layout before loading state_dict; use deterministic evaluation.",
    }
    if statistics_debug is not None and statistics_debug["forward_pre_hooks_after_fitting"] != statistics_debug["forward_pre_hooks_in_fresh_encoder"]:
        hints["checkpoint_output_mismatch"] += (
            f" Fitted encoder has {statistics_debug['forward_pre_hooks_after_fitting']} forward pre-hooks, "
            f"fresh encoder has {statistics_debug['forward_pre_hooks_in_fresh_encoder']}. "
            "Do not remove the normalization hook in fit_statistics; fit updates buffers and preserves the hook installed during every build."
        )
    result = {
        "uses_train_statistics": uses_statistics,
        "statistics_stored_as_buffers": stats_in_state,
        "statistics_debug": statistics_debug,
        "ok": not failures,
        "failures": failures,
        "shape": list(out.shape),
        "finite": finite,
        "params_with_grad": with_grad,
        "params_trainable": len(trainable),
        "finite_gradients": finite_gradients,
        "checkpoint_round_trip": round_trip,
        "file": loaded,
        "python": sys.executable,
        "torch": torch.__version__,
        "stage": "complete",
    }
    if failures:
        result["error"] = "candidate_invariant_failed"
        result["detail"] = "\n".join(name + ": " + hints.get(name, "Inspect the computation for invalid numerical values.") for name in failures)
    return result


def main() -> int:
    context: dict[str, object] = {"stage": "import_torch", "python": sys.executable,
                                  "file": str((Path.cwd() / "eeg_candidate.py").resolve())}
    try:
        payload = run(sys.argv[1] if len(sys.argv) > 1 else "eeg", context=context)
    except Exception as exc:  # noqa: BLE001
        payload = {**context, "ok": False, "error": type(exc).__name__, "detail": traceback.format_exc()[-1500:]}
        if isinstance(exc, SyntaxError):
            payload["location"] = {"file": exc.filename, "line": exc.lineno, "column": exc.offset}
        else:
            frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if Path(frame.filename).resolve().parent == Path.cwd().resolve()]
            if frames:
                payload["location"] = {"file": frames[-1].filename, "line": frames[-1].lineno, "function": frames[-1].name}
    print(json.dumps(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
