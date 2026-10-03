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


def run(dataset: str = "eeg", *, context: dict[str, object] | None = None, hook_config: dict | None = None) -> dict[str, object]:
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
    from react_agent.eeg_training.train_entry import _build_hook
    config = hook_config or {}
    geom = {"c_num": int(spec["c_num"]), "timesteps": list(spec["timesteps"])}
    encoder = _build_hook(candidate, "build_encoder", geom, config.get("model") or {})
    context["stage"] = "build_training_hooks"
    transform = _build_hook(candidate, "build_training_transform", {}, config.get("transform") or {}) if hasattr(candidate, "build_training_transform") else None
    objective = _build_hook(candidate, "build_training_objective", {}, config.get("objective") or {}) if hasattr(candidate, "build_training_objective") else None
    # Probe a declared objective ablation with its actual off key. Presence of
    # an ablation_switch string is not proof that the config can reach the hook.
    objective_config = config.get("objective") or {}
    from react_agent.eeg_research.agentic.objective_effectiveness import objective_switch, objective_is_active
    ablation_switch = objective_switch(objective_config)
    objective_off = None
    ablation_probe = None
    if isinstance(ablation_switch, str) and ablation_switch:
        context["stage"] = "objective_ablation_switch_off"
        objective_off = _build_hook(candidate, "build_training_objective", {}, {**objective_config, ablation_switch: False})
        ablation_probe = {"switch_key": ablation_switch, "switch_value": False, "build_passed": True}
    length = int(spec["timesteps"][1]) - int(spec["timesteps"][0])
    statistics_hook_available = hasattr(candidate, "fit_statistics")
    uses_statistics = False
    stats_in_state = None
    statistics_debug = None
    if statistics_hook_available:
        context["stage"] = "fit_statistics"
        buffers_before = {name: value.clone() for name, value in encoder.named_buffers()}
        pre_hooks_before = len(encoder._forward_pre_hooks)
        params_before = {name: value.detach().clone() for name, value in encoder.named_parameters()}
        probe = torch.randn(4, int(spec["c_num"]), length)
        encoder.eval()
        with torch.no_grad():
            output_before = encoder(probe).clone()
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
        changed_params = [name for name, value in encoder.named_parameters()
                          if name not in params_before or not torch.equal(params_before[name], value)]
        with torch.no_grad():
            output_after = encoder(probe)
        output_changed = output_before.shape != output_after.shape or not torch.allclose(output_before, output_after)
        hooks_changed = pre_hooks_before != len(encoder._forward_pre_hooks)
        uses_statistics = bool(changed or changed_params or output_changed or hooks_changed)
        stats_in_state = (bool(changed) and not changed_params
                          and all(name in encoder.state_dict() for name in changed)) if uses_statistics else None
        statistics_debug = {"buffers_before_fitting": list(buffers_before), "changed_buffers": changed,
                            "changed_parameters": changed_params, "output_changed_by_fitting": output_changed,
                            "no_op": not uses_statistics,
                            "forward_pre_hooks_before_fitting": pre_hooks_before, "forward_pre_hooks_after_fitting": len(encoder._forward_pre_hooks)}
    batch = torch.randn(4, int(spec["c_num"]), length)
    context["stage"] = "train_forward"
    encoder.train()
    if transform is not None:
        batch = transform(batch)
    out = encoder(batch)
    finite = bool(torch.isfinite(out).all())
    context["stage"] = "backward"
    from react_agent.eeg_training.model import LocalRetrieval
    from react_agent.eeg_training.hooks import scalar_loss
    objective_finite = False
    duplicate_objective_finite = None
    if finite and tuple(out.shape) == (4, 1024):
        loss, _, _ = LocalRetrieval(encoder, objective)(batch, torch.randn(4, 1024), torch.arange(4))
        loss = scalar_loss(loss)
        objective_finite = bool(torch.isfinite(loss))
        if not objective_finite:
            raise ValueError("non_finite_training_objective")
    else:
        loss = out.float().pow(2).mean()
    loss.backward()
    if finite and tuple(out.shape) == (4, 1024):
        context["stage"] = "duplicate_image_objective"
        codes = torch.tensor([0, 0, 1, 1])
        duplicate_images = torch.randn(2, 1024)[codes]
        duplicate_loss, _, _ = LocalRetrieval(encoder, objective)(batch, duplicate_images, codes)
        duplicate_loss = scalar_loss(duplicate_loss)
        duplicate_objective_finite = bool(torch.isfinite(duplicate_loss))
        if not duplicate_objective_finite:
            raise ValueError("non_finite_duplicate_image_objective")
        duplicate_loss.backward()
        if ablation_probe is not None:
            context["stage"] = "objective_ablation_off_forward"
            off_loss, _, _ = LocalRetrieval(encoder, objective_off)(batch, duplicate_images, codes)
            off_loss = scalar_loss(off_loss)
            ablation_probe["finite_scalar_loss"] = bool(torch.isfinite(off_loss))
            if not ablation_probe["finite_scalar_loss"]:
                raise ValueError("non_finite_ablation_off_objective")
            off_loss.backward()
            ablation_probe["backward_passed"] = True
    objective_effectiveness = None
    from react_agent.eeg_training.model import contrastive_loss
    from react_agent.eeg_research.agentic.objective_effectiveness import gradient_probe
    if objective is not None and objective is not contrastive_loss:
        context["stage"] = "objective_tied_target_training_gradient"
        objective_effectiveness = gradient_probe(objective)
        if objective_is_active(objective_config) and objective_effectiveness["gradient_changed"] is not True:
            raise ValueError("objective_baseline_equivalent_on_tied_frozen_targets: an active custom objective must change a training gradient on valid same-ID/same-feature groups; calls, masks, changed constant values and finite backward alone do not prove an effective intervention")
    if ablation_probe is not None:
        context["stage"] = "objective_ablation_off_identity"
        ablation_probe["exact_baseline_callable"] = objective_off is contrastive_loss
        ablation_probe["effectiveness_probe"] = gradient_probe(objective_off)
        if not ablation_probe["exact_baseline_callable"]:
            raise ValueError("objective_off_must_return_exact_baseline_callable")
    objective_semantics = None
    if objective is not None and objective is not contrastive_loss:
        context["stage"] = "objective_probability_semantics"
        from react_agent.eeg_research.agentic.objective_semantics import semantic_probe, EXPLICIT_VARIANT
        objective_semantics = semantic_probe(objective)
        context["objective_semantics_probe"] = objective_semantics
        if (objective_is_active(objective_config) and objective_config.get("objective_variant") == EXPLICIT_VARIANT
                and objective_semantics["matches_reference"] is not True):
            raise ValueError("explicit_positive_mass_formula_mismatch: require raw EEG logits, all same-ID positives including diagonal, total local probability mass denominator and symmetric directional mean; inspect objective_semantics_probe")
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
    rebuilt = _build_hook(candidate, "build_encoder", geom, config.get("model") or {})
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
        "statistics_hook_available": statistics_hook_available,
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
        "approved_hook_config": config,
        "training_objective_finite": objective_finite,
        "duplicate_image_objective_finite": duplicate_objective_finite,
        "objective_ablation_probe": ablation_probe,
        "objective_effectiveness_probe": objective_effectiveness,
        "objective_semantics_probe": objective_semantics,
    }
    if failures:
        result["error"] = "candidate_invariant_failed"
        result["detail"] = "\n".join(name + ": " + hints.get(name, "Inspect the computation for invalid numerical values.") for name in failures)
    return result


def main() -> int:
    context: dict[str, object] = {"stage": "import_torch", "python": sys.executable,
                                  "file": str((Path.cwd() / "eeg_candidate.py").resolve())}
    try:
        assigned = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
        hook_config = assigned.get("experiment") or assigned
        payload = run(sys.argv[1] if len(sys.argv) > 1 else "eeg", context=context, hook_config={key: hook_config.get(key) or {} for key in ("model", "objective", "transform")})
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
