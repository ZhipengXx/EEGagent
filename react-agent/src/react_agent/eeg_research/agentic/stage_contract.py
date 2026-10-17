"""Typed phase requirements without execution, inference or implicit migration."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from .schemas import AnalysisRequirement, MeasurementRequirement

STAGE_REQUIREMENT_FIELDS = ("measurement_requirements", "analysis_requirements")
_MODELS = {"measurement_requirements": MeasurementRequirement,
           "analysis_requirements": AnalysisRequirement}
_FIELDS = {"implementation": "implementation_requirements",
           "measurement": "measurement_requirements", "analysis": "analysis_requirements"}


def _experiment(spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    spec = spec if isinstance(spec, Mapping) else {}
    inner = spec.get("experiment")
    return inner if isinstance(inner, Mapping) else spec


def has_stage_contract(spec: Mapping[str, Any] | None) -> bool:
    experiment = _experiment(spec)
    return experiment.get("design_contract_version") == "v2" and any(
        field in experiment for field in STAGE_REQUIREMENT_FIELDS)


def requirements_for_stage(spec: Mapping[str, Any] | None,
                           stage: Literal["implementation", "measurement", "analysis"]) -> list[dict[str, Any]]:
    """Return an independent phase view; the approved experiment stays intact."""
    if stage not in _FIELDS:
        raise ValueError("requirement_stage_unknown:" + str(stage))
    return copy.deepcopy(_experiment(spec).get(_FIELDS[stage]) or [])


def normalize_stage_requirements(spec: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize only explicit v2 fields and keep IDs unique across all phases."""
    experiment = _experiment(spec)
    if not any(field in experiment for field in STAGE_REQUIREMENT_FIELDS):
        return {}
    if experiment.get("design_contract_version") != "v2":
        raise ValueError("stage_requirements_require_v2_contract")
    normalized = {}
    ids = [row.get("requirement_id") for row in experiment.get("implementation_requirements") or []
           if isinstance(row, Mapping)]
    for field, model in _MODELS.items():
        if field not in experiment:
            continue
        value = experiment[field]
        if not isinstance(value, list):
            raise ValueError(field + "_not_list")
        try:
            normalized[field] = [model.model_validate(item).model_dump() for item in value]
        except ValidationError as exc:
            raise ValueError(field + "_invalid:" + str(exc.errors()[0]["loc"])) from exc
        ids.extend(item["requirement_id"] for item in normalized[field])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_stage_requirement_id")
    return normalized


def stage_requirements_projection(spec: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Explicit phase placement and every acceptance predicate are immutable."""
    return {field: sorted(rows, key=lambda row: row["requirement_id"])
            for field, rows in normalize_stage_requirements(_experiment(spec)).items()}
