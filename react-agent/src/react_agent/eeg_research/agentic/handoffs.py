"""Resolve the actual candidate and bounded development facts for role handoffs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.artifacts import file_digest
from react_agent.eeg_research.agentic.execution_protocol import load_protocol
from react_agent.eeg_research.agentic.run_context import BASELINE_HOOK_SPEC, load_approved_binding

SOURCE_CHARS = 16000
_PRIVATE_SCOPES = {"final_test", "final_holdout", "held_out_unused", "secret"}


def training_semantics() -> list[dict[str, Any]]:
    """Source evidence for how the trainer calls configs and owns logit scaling."""
    import ast
    root = Path(__file__).resolve().parents[2] / "eeg_training"
    views = []
    for name, selected in (("model.py", {"LocalRetrieval"}), ("train_entry.py", {
        "_build_hook", "_logit_scale", "_instantiate_candidate", "train_channel_statistics",
        "unique_trainable_parameters", "placed_retrieval", "rebuild_encoder", "fit",
    })):
        path = root / name
        source = path.read_text(encoding="utf-8")
        nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        excerpts = []
        for node in nodes:
            if node.name == "fit":
                # Include the actual construction and optimizer branches, before
                # gallery evaluation; avoid sending the entire training loop.
                stop = next(item.lineno for item in node.body if isinstance(item, ast.Assign)
                            and any(isinstance(target, ast.Tuple) and any(
                                isinstance(part, ast.Name) and part.id == "bank" for part in target.elts
                            ) for target in item.targets))
                excerpts.append("\n".join(source.splitlines()[node.lineno - 1:stop - 1]))
            else:
                excerpts.append(ast.get_source_segment(source, node))
        views.append({"source_ref": str(path), "source_hash": file_digest(path),
                      "excerpt": "\n\n".join(excerpts)})
    return views


def development_view(value: Any) -> Any:
    """Filter scope recursively before sending data to a provider."""
    if isinstance(value, dict):
        if value.get("scope") in _PRIVATE_SCOPES or value.get("role") in _PRIVATE_SCOPES:
            return None
        kept = {}
        for key, item in value.items():
            if key in {"test_result", "final_test_result", "forbidden_contents", "api_key", "DEEPSEEK_API_KEY"}:
                continue
            clean = development_view(item)
            if clean is not None or item is None:
                kept[key] = clean
        return kept
    if isinstance(value, list):
        return [clean for item in value if (clean := development_view(item)) is not None]
    return value


def candidate_context(camp: Path, target: str) -> dict[str, Any]:
    """Use target-bound source/config; baseline has concrete behavior too."""
    protocol = load_protocol(camp) or {}
    if target == "baseline":
        entry = Path(__file__).with_name("baseline.py")
        binding = dict(BASELINE_HOOK_SPEC)
        binding["model"] = {"z_dim": 1024, "drop_proj": 0.3}
        approval_ref = "runtime:frozen_baseline"
    else:
        entry = camp / "candidates" / target / "extension" / "eeg_candidate.py"
        binding = load_approved_binding(camp, target) or {}
        approval_ref = binding.get("spec_ref")
    missing = []
    if not entry.is_file():
        missing.append("source")
    if not binding:
        missing.append("approved_binding")
    source = entry.read_text(encoding="utf-8") if entry.is_file() else None
    current_hash = file_digest(entry) if entry.is_file() else None
    expected_hash = binding.get("source_hash")
    if expected_hash and current_hash != expected_hash:
        missing.append("source_hash_mismatch")
        source = None
    encoder_reference = None
    if target == "baseline":
        model_entry = Path(__file__).resolve().parents[2] / "eeg_training" / "model.py"
        model_source = model_entry.read_text(encoding="utf-8")
        encoder_reference = {"source_ref": str(model_entry), "source_hash": file_digest(model_entry),
                             "excerpt": model_source.split("class LocalRetrieval", 1)[0]}
    return development_view({
        "candidate_id": target,
        "source_hash": current_hash,
        "source_ref": str(entry),
        "source": None if source is None else source[:SOURCE_CHARS],
        "source_truncated": source is not None and len(source) > SOURCE_CHARS,
        "source_range": None if source is None else [0, min(len(source), SOURCE_CHARS)],
        "encoder_reference": encoder_reference,
        "approval_ref": approval_ref,
        "spec_hash": binding.get("spec_hash"),
        "model": binding.get("model") or {},
        "objective": binding.get("objective") or {},
        "transform": binding.get("transform") or {},
        "intervention": binding.get("intervention") or ("frozen EEGProjectLayer baseline" if target == "baseline" else None),
        "recipe": {key: protocol.get(key) for key in ("full_epochs", "batch_size", "lr", "weight_decay",
                   "negative_sampling_policy", "training_seeds", "input_geometry")},
        "negative_sampling_policy": binding.get("negative_sampling_policy") or protocol.get("negative_sampling_policy"),
        "missing_inputs": missing,
    })


def matched_diagnostics(state: dict[str, Any], target: str) -> dict[str, Any] | None:
    """Never borrow another target's latest checkpoint diagnostics."""
    for row in reversed(state.get("evidence") or []):
        if row.get("candidate_id") != target or row.get("evaluation_valid") is not True:
            continue
        if row.get("diagnostics") or row.get("diagnostic_ref"):
            return development_view({"summary": row.get("diagnostics"), "ref": row.get("diagnostic_ref"),
                "evidence_id": row.get("evidence_id"), "candidate_id": target,
                "checkpoint_id": row.get("checkpoint_id"), "evaluation_hash": row.get("contract_fingerprint")})
    return None


def load_analysis_views(camp: Path, state: dict[str, Any], *, target: str | None = None) -> list[dict[str, Any]]:
    """Only registered bytes belonging to an analysis pointer are full handoffs."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact

    views = []
    for row in state.get("evidence") or []:
        if row.get("kind") != "analysis" or (target and row.get("candidate_id") != target):
            continue
        ref = row.get("analysis_artifact_id")
        if not ref:
            continue
        try:
            registered = resolve_verified_artifact(camp, ref)
            body = json.loads(Path(registered["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            views.append({"evidence_id": row.get("evidence_id"), "artifact_id": ref,
                          "verification_status": "invalid", "missing_inputs": ["analysis_hash_or_file"]})
            continue
        if registered.get("kind") != "analysis":
            continue
        payload = body.get("payload") or body.get("reply") or body
        views.append(development_view({"evidence_id": row.get("evidence_id"), "candidate_id": row.get("candidate_id"),
            "artifact_id": ref, "content_hash": registered["sha256"], "verification_status": "verified",
            "completion_status": body.get("status"),
            "authority": "llm_interpretation", "input_digest": body.get("input_digest"), "payload": payload}))
    return views[-8:]
