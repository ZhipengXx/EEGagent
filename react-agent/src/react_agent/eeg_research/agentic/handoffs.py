"""Resolve the actual candidate and bounded development facts for role handoffs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.artifacts import file_digest
from react_agent.eeg_research.agentic.execution_protocol import load_protocol
from react_agent.eeg_research.agentic.run_context import BASELINE_HOOK_SPEC, load_approved_binding

SOURCE_CHARS = 16000
_PRIVATE_SCOPES = {"final_test", "final_holdout", "held_out_unused", "secret"}
# Limits apply to provider views, never to the authoritative artifacts or ledger.
READ_MAX_IDS = 3
READ_RANGE_CHARS = 4000
READ_TOTAL_CHARS = 8000
READS_PER_EVIDENCE_EPOCH = 4
READ_FILE_BYTES = 1_000_000
HISTORY_ROWS = 20
HISTORY_CHARS = 16000
HISTORY_ITEM_CHARS = 2000
CONTROL_TEXT_CHARS = 16000
READ_JOURNAL = "artifact_reads.jsonl"


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
        if any(isinstance(value.get(key), str) and value[key] in _PRIVATE_SCOPES
               for key in ("scope", "role", "data_role", "partition")):
            return None
        kept = {}
        for key, item in value.items():
            if key in {"test_result", "final_test_result", "forbidden_contents", "api_key", "DEEPSEEK_API_KEY"} or (
                str(key).startswith(("test_result", "final_holdout", "final_test_")) and key != "final_test_enabled"
            ):
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


def _registry_rows(camp: Path) -> list[dict[str, Any]]:
    from react_agent.eeg_research.agentic.artifacts import REGISTRY
    path = camp / REGISTRY
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def development_artifact(camp: Path, artifact_id: str) -> dict[str, Any]:
    """Memory reading has a stricter boundary than approved external source consumers."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    row = resolve_verified_artifact(camp, artifact_id)
    path = Path(str(row["path"])).resolve()
    if not path.is_relative_to(camp.resolve()):
        raise ValueError("artifact_outside_campaign")
    if path.suffix != ".json":
        raise ValueError("artifact_requires_structured_development_view")
    if path.stat().st_size > READ_FILE_BYTES:
        raise ValueError("artifact_exceeds_read_file_limit")
    body = json.loads(path.read_text(encoding="utf-8"))
    if development_view(body) is None:
        raise ValueError("artifact_scope_forbidden")
    return row


def artifact_index(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Identifiers and hashes are an index; they do not claim the model read the bytes."""
    rows = []
    evidence = state.get("evidence") or []
    for record in _registry_rows(camp):
        ref = record.get("artifact_id")
        if not ref:
            continue
        missing = []
        try:
            development_artifact(camp, str(ref))
        except (OSError, ValueError, KeyError) as exc:
            missing = [str(exc)]
        linked = [row for row in evidence if ref in (row.get("artifact_refs") or []) or
                  ref in (row.get("analysis_artifact_id"), row.get("audit_artifact_id"))]
        completion = "unknown"
        completion_verified = False
        if not missing and record.get("producer_task_id"):
            from react_agent.eeg_research.agentic.task_ledger import latest
            from react_agent.eeg_research.agentic.roles import validate_role_result, RoleResultError
            task = latest(camp, str(record["producer_task_id"]))
            try:
                body = json.loads(Path(str(record["path"])).read_text(encoding="utf-8"))
                envelope = validate_role_result(body, task) if task else None
                if envelope:
                    completion = envelope["status"]
                    completion_verified = True
            except (OSError, ValueError, KeyError, RoleResultError):
                pass
        rows.append({"artifact_id": ref, "kind": record.get("kind"),
            "candidate_id": record.get("candidate_id"),
            "question_id": next((row.get("question_id") for row in linked if row.get("question_id")), None),
            "title": Path(str(record.get("path") or "")).name,
            "content_hash": record.get("sha256"), "producer_task_id": record.get("producer_task_id"),
            "evidence_refs": [row["evidence_id"] for row in linked if row.get("evidence_id")],
            "verification_status": "invalid" if missing else "verified",
            "completion_status": completion, "completion_identity_verified": completion_verified,
            "scope": "development" if not missing else "unavailable",
            "missing_inputs": missing, "content_injected": False})
    return rows


def read_epoch(state: dict[str, Any]) -> str:
    from react_agent.eeg_research.agentic.planner import evidence_count
    return str(evidence_count(state))


def read_history(camp: Path) -> list[dict[str, Any]]:
    path = camp / READ_JOURNAL
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_digest(request: dict[str, Any], content_hash: str | None) -> str:
    from react_agent.eeg_research.agentic.artifacts import request_digest
    return request_digest(request={"artifact_id": request.get("artifact_id"),
                                  "start": request.get("start", 0), "end": request.get("end"),
                                  "content_hash": content_hash, "scope": "development_view"})


def consume_artifact_reads(camp: Path, state: dict[str, Any], requests: list[dict[str, Any]],
                           decision_id: str | None) -> dict[str, Any]:
    """Journal one bounded read, not scientific evidence or an experimental confirmation."""
    from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range
    from react_agent.eeg_research.agentic.schemas import ArtifactReadRequest
    epoch = read_epoch(state)
    history = read_history(camp)
    results = []
    used = sum(row.get("epoch") == epoch and not row.get("replayed") for row in history)
    remaining = READ_TOTAL_CHARS
    if len(requests) > READ_MAX_IDS:
        return {"status": "limited", "missing_inputs": ["artifact_read_id_limit"], "results": []}
    for raw in requests:
        ref = raw.get("artifact_id") if isinstance(raw, dict) else None
        registered = lookup(camp, str(ref)) if ref else None
        digest = read_digest(raw, None if registered is None else registered.get("sha256"))
        previous = next((row for row in history if row.get("request_digest") == digest), None)
        if previous:
            amount = len(previous.get("text") or "")
            if amount > remaining:
                results.append({"artifact_id": ref, "status": "limited", "missing_inputs": ["artifact_read_total_limit"], "replayed": True})
            else:
                remaining -= amount
                results.append({**previous, "replayed": True})
            continue
        result = {"artifact_id": ref, "request_digest": digest, "decision_id": decision_id,
                  "epoch": epoch, "status": "missing_input", "missing_inputs": [], "text": None}
        if used >= READS_PER_EVIDENCE_EPOCH or remaining <= 0:
            result.update(status="limited", missing_inputs=["artifact_read_epoch_or_context_limit"])
            results.append(result)
            continue
        try:
            request = ArtifactReadRequest.model_validate(raw)
            if request.end is not None and request.end - request.start > READ_RANGE_CHARS:
                raise ValueError("artifact_read_range_limit")
            if request.end is not None and request.end - request.start > remaining:
                raise ValueError("artifact_read_total_limit")
            development_artifact(camp, request.artifact_id)
            view = read_verified_range(camp, request.artifact_id, start=request.start, end=request.end,
                                       json_view=development_view, max_chars=min(READ_RANGE_CHARS, remaining))
            result.update(view, status="read", content_injected=False)
            remaining -= len(view["text"])
        except (OSError, ValueError, KeyError) as exc:
            result["missing_inputs"] = [str(exc)]
        with (camp / READ_JOURNAL).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        history.append(result)
        used += 1
        results.append(result)
    state["artifact_read_history"] = [{key: row.get(key) for key in ("artifact_id", "request_digest", "epoch", "status")}
                                      for row in history]
    state["artifact_read_results"] = results
    return {"status": "completed", "results": results, "remaining_reads": max(0, READS_PER_EVIDENCE_EPOCH - used)}


def verified_read_context(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Only journaled, still-current development views enter the next provider request."""
    from react_agent.eeg_research.agentic.artifacts import read_verified_range
    rows = []
    chars = READ_TOTAL_CHARS
    journal = {row.get("request_digest"): row for row in read_history(camp)}
    for saved in (state.get("artifact_read_results") or [])[:READ_MAX_IDS]:
        if saved.get("status") != "read":
            rows.append(saved)
            continue
        try:
            recorded = journal.get(saved.get("request_digest")) or {}
            if recorded.get("status") != "read" or any(recorded.get(key) != saved.get(key)
                    for key in ("artifact_id", "sha256", "start", "end", "view_sha256", "decision_id")):
                raise ValueError("artifact_read_journal_binding_missing")
            current = development_artifact(camp, saved["artifact_id"])
            if current["sha256"] != saved.get("sha256"):
                raise ValueError("artifact_read_hash_changed")
            view = read_verified_range(camp, saved["artifact_id"], start=saved["start"], end=saved["end"],
                                       json_view=development_view)
            if view.get("view_sha256") != saved.get("view_sha256"):
                raise ValueError("artifact_read_view_changed")
            if len(view["text"]) > chars:
                raise ValueError("artifact_read_context_limit")
            chars -= len(view["text"])
            rows.append({**saved, **view, "content_injected": True})
        except (OSError, ValueError, KeyError) as exc:
            rows.append({"artifact_id": saved.get("artifact_id"), "status": "missing_input", "missing_inputs": [str(exc)]})
    return rows


def bounded_control_view(value: Any, *, budget: int = CONTROL_TEXT_CHARS) -> tuple[Any, dict[str, Any]]:
    """Bound prose without losing control identities, candidates or open issues.

    The projection preserves JSON shape. Explicit paths locate omitted prose in
    the registered artifact/plan; it never overwrites authoritative content.
    """
    remaining = max(0, budget)
    truncated = []
    exact = {"action", "status", "kind", "fidelity", "verdict", "severity", "category", "scope", "completion_status",
             "verification_status", "model_audit_status", "resolution_kind", "operation", "revision_operation",
             "hypothesis_assessment", "authority", "interpretation_authority",
             "report_ref", "audited_report_ref", "request_digest", "input_digest", "resource_status"}
    def project(item: Any, key: str = "", path: str = "$") -> Any:
        nonlocal remaining
        if isinstance(item, dict):
            return {name: project(child, name, path + "." + name) for name, child in item.items()}
        if isinstance(item, list):
            return [project(child, key, path + f"[{index}]") for index, child in enumerate(item)]
        if isinstance(item, str):
            if key in exact or key.endswith(("_id", "_ids", "_ref", "_refs", "_hash", "_digest")):
                return item
            amount = min(len(item), HISTORY_ITEM_CHARS, remaining)
            remaining -= amount
            if amount < len(item):
                truncated.append({"path": path, "shown_characters": amount, "total_characters": len(item),
                                  "next_range": [amount, min(len(item), amount + READ_RANGE_CHARS)]})
                return item[:amount]
        return item
    clean = project(development_view(value))
    return clean, {"truncated": bool(truncated), "text_character_limit": budget,
                   "shown_text_characters": budget - remaining, "truncated_fields": truncated,
                   "identity_and_issue_rows_retained": True, "omitted_content": "registered artifact or versioned plan"}


def bounded_history(value: Any, *, budget: int = HISTORY_CHARS) -> tuple[Any, dict[str, Any]]:
    """Keep recent history bounded, with an explicit index of omitted rows."""
    if isinstance(value, list):
        rows, used = [], 0
        for original in reversed(value[-HISTORY_ROWS:]):
            item = development_view(original)
            encoded = json.dumps(item, ensure_ascii=False, default=str)
            identity = {key: item.get(key) for key in ("evidence_id", "episode_id", "lesson_id", "artifact_id", "candidate_id", "kind", "fidelity")
                        if isinstance(item, dict) and item.get(key) is not None}
            if len(encoded) > HISTORY_ITEM_CHARS:
                item = {**identity, "summary_excerpt": encoded[:HISTORY_ITEM_CHARS], "truncated": True,
                        "next_range": [HISTORY_ITEM_CHARS, min(len(encoded), HISTORY_ITEM_CHARS + READ_RANGE_CHARS)]}
            size = len(json.dumps(item, ensure_ascii=False, default=str))
            if used + size > budget:
                break
            rows.append(item)
            used += size
        rows.reverse()
        return rows, {"total_rows": len(value), "shown_rows": len(rows), "omitted_row_range": [0, max(0, len(value) - len(rows))],
                      "truncated": len(rows) < len(value) or any(isinstance(row, dict) and row.get("truncated") for row in rows),
                      "character_limit": budget}
    return bounded_control_view(value, budget=budget)


def selected_read_context(camp: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """A selected worker gets cited reads, not the controller's complete history."""
    refs = set(state.get("active_artifact_refs") or [])
    return [row for row in verified_read_context(camp, state) if row.get("artifact_id") in refs]
