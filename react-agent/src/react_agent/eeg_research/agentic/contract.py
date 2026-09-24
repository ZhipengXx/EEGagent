"""Freeze an evaluation contract from the real split. The model cannot invent IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_training.protocol import Design, protocol_label, resolve_generalization, split_manifest, split_plan


def research_scope(design: Design) -> str:
    """Name the task the split actually supports."""
    target = resolve_generalization(design)
    if target == "held_out_subject" and design.subject not in {"", "all"}:
        return "subject_generalization"
    return "pooled_subject_retrieval"


def freeze_contract(design: Design, data_root: Path) -> dict[str, Any]:
    """Build the contract from split_plan. test.pt is a file name, not a role."""
    plan = split_plan(data_root, design)
    manifest = split_manifest(plan, design)
    body: dict[str, Any] = {
        "task_type": "eeg_image_retrieval",
        "research_scope": research_scope(design),
        "generalization_target": manifest["generalization_target"],
        "label": protocol_label(design),
        "primary_metric": "validation.fixed_gallery_top1",
        "direction": "maximize",
        "final_test_enabled": False,
        "val_mode": plan.val_mode,
        "files": manifest["files"],
        "feature_caches": [str(path) for path in plan.feature_caches],
        "within_batch_role": "diagnostic_only",
        "legacy_unverified_note": "Historical within-batch 0.363 is not a fixed-gallery baseline.",
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return body


def public_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Planner view. Final-test paths stay out."""
    files = [row for row in contract.get("files", []) if row.get("role") != "final_test"]
    kept = dict(contract)
    kept["files"] = files
    kept.pop("forbidden_contents", None)
    return kept
