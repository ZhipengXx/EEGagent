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
            "build_encoder": {"status": "available", "verified": True},
            "fit_statistics": {"status": "available", "verified": True},
            "build_training_transform": {"status": "available", "verified": True},
            "build_training_objective": {"status": "available", "verified": True},
            "evaluate_only": {"status": "available", "verified": True},
        },
        "task_adapters": {name: {"status": "unavailable"} for name in UNAVAILABLE_TASKS},
        "negative_sampling_policies": {
            "data_parallel_local": {"status": "available", "default": True},
            "global_batch": {"status": "available", "when": "custom_objective"},
        },
        "input_spec": candidate_interface(protocol),
    }
