"""Requirement-driven CPU probes; no score prediction or universal pooling rule.

The ordered-window counterexample enters the actual pool input and observes the
representation consumed downstream. This detects a later window mean as well as
pooling that immediately erases slots. It never permutes raw EEG and assumes no
particular convolution weights, padding or initialization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROBE_VERSION = "eeg_research.implementation_semantics.v1"


def requirements_hash(requirements: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(requirements, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _layout(slots: Any, layout: str) -> Any:
    if layout == "flatten_channel_major":
        return slots.flatten(1)
    if layout == "flatten_window_major":
        return slots.transpose(1, 2).flatten(1)
    if layout == "channel_window":
        return slots
    if layout == "window_channel":
        return slots.transpose(1, 2)
    raise ValueError("ordered_window_layout_unsupported")


def ordered_window_probe(encoder: Any, geometry: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Inject ordered slots at a real module input and observe the real consumer."""
    import torch

    modules = dict(encoder.named_modules())
    input_path = str(probe.get("input_module_path") or "")
    representation_path = str(probe.get("representation_module_path") or "")
    if input_path not in modules or representation_path not in modules:
        return {"status": "unverified", "error_class": "semantic_probe_location_missing",
                "stage": "ordered_window_probe", "hook": "build_encoder",
                "location": {"pool_input": input_path, "representation_consumer": representation_path},
                "expected": "Both declared named modules are present in the actual encoder.",
                "actual": {"input_found": input_path in modules, "consumer_found": representation_path in modules}}
    count = int(probe.get("slot_count") or 0)
    if count < 2 or count > 32 or probe.get("window_axis", -1) not in {-1, 2}:
        raise ValueError("ordered_window_probe_spec_invalid")
    layout = str(probe.get("layout") or "flatten_channel_major")
    saved_mode = encoder.training
    buffers = {name: value.detach().clone() for name, value in encoder.named_buffers()}
    observations = []
    try:
        encoder.eval()
        for reverse in (False, True):
            injected = []
            captured = []
            def inject(_module, inputs):
                value = inputs[0]
                if value.ndim != 3:
                    raise ValueError("ordered_window_probe_requires_batch_channel_time_pool_input")
                channels = value.shape[1]
                channel_values = torch.arange(channels, dtype=value.dtype, device=value.device).view(1, -1, 1) * (count + 1)
                slot_values = torch.arange(1, count + 1, dtype=value.dtype, device=value.device).view(1, 1, -1)
                if reverse:
                    slot_values = slot_values.flip(-1)
                slots = (channel_values + slot_values).expand(value.shape[0], -1, -1).clone()
                injected.append(slots)
                # Four equal-width constant windows, independent of the raw EEG
                # geometry, are a controlled input at the declared pool boundary.
                return (slots.repeat_interleave(4, dim=-1), *inputs[1:])
            def observe(_module, inputs):
                captured.append(inputs[0].detach().clone())
            before = modules[input_path].register_forward_pre_hook(inject)
            after = modules[representation_path].register_forward_pre_hook(observe)
            try:
                length = int(geometry["timesteps"][1]) - int(geometry["timesteps"][0])
                parameter = next(encoder.parameters(), None)
                dtype = torch.float32 if parameter is None else parameter.dtype
                device = torch.device("cpu") if parameter is None else parameter.device
                with torch.no_grad():
                    encoder(torch.zeros(2, int(geometry["c_num"]), length, dtype=dtype, device=device))
            finally:
                before.remove()
                after.remove()
            if len(injected) != 1 or len(captured) != 1:
                return {"status": "unverified", "error_class": "semantic_probe_path_not_unique",
                        "stage": "ordered_window_probe", "hook": "build_encoder",
                        "expected": "Exactly one pool call and one downstream representation call.",
                        "actual": {"pool_calls": len(injected), "consumer_calls": len(captured)}}
            expected = _layout(injected[0], layout)
            actual = captured[0]
            tolerance = 16 * torch.finfo(actual.dtype).eps if actual.is_floating_point() else 0.0
            same_shape = tuple(actual.shape) == tuple(expected.shape)
            matches = bool(same_shape and torch.allclose(actual, expected, rtol=0.0, atol=tolerance))
            observations.append({"order": "reversed" if reverse else "original", "matches_ordered_slots": matches,
                                 "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
                                 "expected_first_values": expected[0].flatten()[:16].tolist(),
                                 "actual_first_values": actual[0].flatten()[:16].tolist(), "atol": tolerance})
    finally:
        for name, value in encoder.named_buffers():
            if name in buffers:
                value.copy_(buffers[name])
        encoder.train(saved_mode)
    passed = all(row["matches_ordered_slots"] for row in observations)
    return {"status": "implemented" if passed else "contradicted",
            "error_class": None if passed else "ordered_window_slots_erased_or_reordered",
            "stage": "ordered_window_probe", "hook": "build_encoder",
            "location": {"pool_input": input_path, "representation_consumer": representation_path},
            "expected": "The downstream consumer receives each declared window in its original ordered slot.",
            "actual": observations,
            "minimal_reproduction": {"batch": 2, "slot_count": count, "window_width": 4,
                                     "input": "channel*(slot_count+1)+window_index+1, original and reversed windows",
                                     "injection": "pool input, before the real computation", "layout": layout}}


def probe_requirements(encoder: Any, geometry: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute only properties explicitly declared in the approved design."""
    rows = []
    for requirement in requirements:
        identity = {"requirement_id": requirement.get("requirement_id"),
                    "kind": requirement.get("kind", "implementation"), "core": requirement.get("core", True),
                    "verification": requirement.get("verification", "source_review")}
        if identity["kind"] == "scientific_hypothesis":
            row = {"status": "unverified", "detail": "Scientific benefit is untested before training and does not block implementation."}
        elif identity["verification"] == "source_review":
            row = {"status": "unverified", "detail": "Reviewer must inspect the declared current code locations and source-bound check receipt."}
        elif identity["verification"] == "ordered_window_slots":
            try:
                row = ordered_window_probe(encoder, geometry, requirement.get("probe") or {})
            except (ValueError, TypeError, KeyError, RuntimeError, AttributeError, IndexError) as exc:
                row = {"status": "unverified", "error_class": type(exc).__name__, "stage": "ordered_window_probe",
                       "hook": "build_encoder", "detail": str(exc),
                       "expected": "The declared pool-input probe executes and exposes the ordered representation.",
                       "actual": "probe did not complete", "location": requirement.get("code_locations")}
        else:
            row = {"status": "unverified", "error_class": "semantic_probe_unsupported",
                   "detail": "Use a supported typed probe or explicit current-source review."}
        rows.append({**identity, **row})
    blocked = [row for row in rows if row["kind"] == "implementation" and row["core"]
               and row["verification"] != "source_review" and row["status"] != "implemented"]
    return {"schema_version": PROBE_VERSION, "requirements_hash": requirements_hash(requirements),
            "ok": not blocked, "requirements": rows,
            "scope": "Declared implementation properties on controlled CPU fixtures; no scientific performance claim."}
