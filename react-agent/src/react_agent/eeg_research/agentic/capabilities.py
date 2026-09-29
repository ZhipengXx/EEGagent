"""Declared capabilities. Only wired, tested hooks are available."""

from __future__ import annotations

from typing import Any

from react_agent.eeg_research.agentic.interface import candidate_interface

UNAVAILABLE_TASKS = ("classification", "reconstruction", "raw_preprocessing", "foundation_model")


def capability_manifest(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return hooks the runtime actually calls. Unwired adapters stay unavailable."""
    return {
        "schema_version": "eeg_research.capabilities.v1",
        "task_type": "eeg_image_retrieval",
        "hooks": {
            name: {
                "status": "available",
                "verified": False,
                "interface_version": "eeg_candidate.hooks.v1",
                "verification_artifact": None,
                "reason": "no_verification_artifact",
            }
            for name in (
                "build_encoder",
                "fit_statistics",
                "build_training_transform",
                "build_training_objective",
                "evaluate_only",
            )
        },
        "task_adapters": {name: {"status": "unavailable"} for name in UNAVAILABLE_TASKS},
        "negative_sampling_policies": {
            "data_parallel_local": {"status": "available", "default": True},
            "global_batch": {"status": "available", "when": "declared_by_experiment"},
        },
        "input_spec": candidate_interface(protocol),
    }
