"""One literal implementation view of an approved design, shared by coding roles.

The approved artifact is never modified.  This module selects science and known
effective hook settings; it does not summarize, infer a new design, or retrieve.
"""
from __future__ import annotations

import copy
from typing import Any

from .context.types import canonical


def project_approved_design(spec: dict[str, Any], *, selected_task: dict | None = None,
                            interface: dict | None = None,
                            effective_hook_config: dict | None = None) -> dict[str, Any]:
    """Return the single implementation-science view of a native approved spec.

    Requirements keep their stable IDs.  Coder uses these IDs directly;
    Reviewer may present the same rows under its locally frozen rN aliases.
    Open scientific dictionaries are copied literally, including user-defined
    keys that happen to resemble native metadata names.
    """
    if not isinstance(spec, dict):
        raise ValueError("approved_design_spec_not_object")
    experiment = spec.get("experiment") if isinstance(spec.get("experiment"), dict) else spec
    brief = experiment.get("design_brief") or spec.get("design_brief") or {}
    if not isinstance(brief, dict):
        raise ValueError("approved_design_brief_not_object")
    selected = selected_task or {}
    result: dict[str, Any] = {}
    distinct: dict[str, dict] = {}
    for key in ("question", "hypothesis", "scientific_intent", "principal_intervention",
                "applicability", "scientific_parameters"):
        candidates = [experiment, brief, selected]
        if key == "principal_intervention":
            candidates = [experiment, brief]
        for source in candidates:
            if key in source:
                result[key] = copy.deepcopy(source[key])
                break
        if key in result:
            for label, source in (("approved_experiment", experiment), ("design_brief", brief),
                                  ("selected_task", selected)):
                if key in source and canonical(source[key]) != canonical(result[key]):
                    distinct.setdefault(label, {})[key] = copy.deepcopy(source[key])
    if "principal_intervention" not in result and "intervention" in experiment:
        result["principal_intervention"] = copy.deepcopy(experiment["intervention"])
    result.setdefault("scientific_parameters", {})
    if distinct:
        # An approved design can elaborate the original selected task. Keep
        # distinct statements visible rather than treating the shorter one as
        # a replacement or silently choosing one author's scientific intent.
        result["additional_scientific_context"] = distinct

    modes = experiment.get("hook_modes") or {}
    changes = brief.get("hook_changes") or {}
    configurations = effective_hook_config or {}
    hooks = {}
    for name in ("model", "objective", "transform"):
        change = changes.get(name) or {}
        mode = modes.get(name, change.get("mode"))
        if mode not in {"inherit", "replace"}:
            raise ValueError("approved_design_hook_mode_missing:" + name)
        if name in configurations:
            configuration = configurations[name]
        elif name in experiment:
            configuration = experiment[name]
        elif change.get("config") is not None:
            configuration = change["config"]
        else:
            # Native approved v2 records use {} for an inherited default hook.
            configuration = {}
        if not isinstance(configuration, dict):
            raise ValueError("approved_design_effective_hook_config_not_object:" + name)
        hooks[name] = {"mode": mode, "effective_config": copy.deepcopy(configuration)}
    result["hooks"] = hooks

    declarations = experiment.get("implementation_requirements")
    if declarations is None:
        declarations = brief.get("implementation_requirements")
    if not isinstance(declarations, list):
        raise ValueError("approved_design_implementation_requirements_missing")
    fields = ("requirement_id", "kind", "core", "scope", "expected_behavior", "verification", "probe")
    result["implementation_requirements"] = []
    ids = set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("approved_design_requirement_not_object")
        identifier = declaration.get("requirement_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError("approved_design_requirement_identity_invalid")
        ids.add(identifier)
        result["implementation_requirements"].append({key: copy.deepcopy(declaration[key])
            for key in fields if key in declaration})
    return result
