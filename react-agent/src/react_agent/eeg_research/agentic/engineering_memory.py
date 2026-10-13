"""Verified historical code repairs, separate from scientific performance memory.

The native checker calls these hooks while the checked source is still current.
Historical snapshots remain immutable; retrieval never grants current training
authorization and never creates a performance episode.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCOPE = "engineering_repair"
KIND = "engineering_repair"
VERSION = "eeg_research.engineering_repair.v1"


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value, *, text=False):
    encoded = value if text else _json(value)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("engineering_snapshot_identity_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded)


def _approved(spec: dict) -> dict:
    from react_agent.eeg_research.agentic.experiment_gate import experiment_is_approved
    experiment = spec.get("experiment") or spec
    if not experiment_is_approved(experiment):
        raise ValueError("engineering_repair_requires_approved_native_spec")
    return experiment


def _method_semantics(spec: dict) -> dict:
    return {key: spec.get(key) for key in ("model", "objective", "transform", "negative_sampling_policy",
        "implementation_requirements", "parent_candidate_id", "control_candidate_id", "hypothesis",
        "principal_intervention", "intervention")}


def _current_check(workspace: Path, check: dict, spec: dict, python=None):
    from react_agent.eeg_research.agentic.native_patch import _check_fingerprint
    from react_agent.eeg_training.protocol import torch_python
    entry = workspace / "extension/eeg_candidate.py"
    if not entry.is_file() or check.get("source_sha256") != _sha(entry):
        raise ValueError("engineering_check_source_changed")
    if check.get("check_fingerprint") != _check_fingerprint(workspace, python):
        raise ValueError("engineering_check_config_changed")
    _approved(spec)
    if _read(workspace / "spec.json") != spec:
        raise ValueError("engineering_spec_input_changed")
    return python or torch_python()


def capture_failed_check(workspace: Path, check: dict, spec: dict, *, python=None) -> dict | None:
    """Capture before another patch; API/resource/cache failures are excluded."""
    from react_agent.eeg_research.agentic.coding_recovery import error_fingerprint, classify_check
    if check.get("ok") is not False or check.get("cached") or not error_fingerprint(check):
        return None
    interpreter = _current_check(workspace, check, spec, python)
    experiment = _approved(spec)
    identity = _hash({"candidate_id": workspace.name, "source": check["source_sha256"],
                      "spec": experiment["spec_hash"], "check": check["check_fingerprint"],
                      "raw_check_hash": _hash(check),
                      "semantic_error": error_fingerprint(check)})
    directory = workspace / "engineering_failures" / identity
    _write_once(directory / "extension/eeg_candidate.py", (workspace / "extension/eeg_candidate.py").read_text(), text=True)
    _write_once(directory / "input_spec.json", _read(workspace / "input_spec.json"))
    _write_once(directory / "spec.json", spec)
    _write_once(directory / "check.json", check)
    manifest = {"failure_id": identity, "candidate_id": workspace.name, "spec_hash": experiment["spec_hash"],
                "source_sha256": check["source_sha256"], "check_fingerprint": check["check_fingerprint"],
                "error_fingerprint": error_fingerprint(check), "error_category": classify_check(check),
                "python": interpreter, "files": {name: _sha(directory / name) for name in
                    ("extension/eeg_candidate.py", "input_spec.json", "spec.json", "check.json")}}
    _write_once(directory / "manifest.json", manifest)
    # This selector is a native operational projection; immutable snapshots and
    # repair evidence are never rewritten when another check fails.
    target = workspace / "engineering_failure_current.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(_json({"failure_id": identity, "manifest_sha256": _sha(directory / "manifest.json")}))
    temporary.replace(target)
    return manifest


def _snapshot(path: Path, manifest: dict) -> dict:
    from react_agent.eeg_research.agentic.native_patch import _check_fingerprint
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or _sha(target) != digest:
            raise ValueError("engineering_snapshot_source_changed")
    check = _read(path / "check.json")
    if check.get("source_sha256") != _sha(path / "extension/eeg_candidate.py"):
        raise ValueError("engineering_snapshot_check_source_mismatch")
    if check.get("check_fingerprint") != _check_fingerprint(path, manifest["python"]):
        raise ValueError("engineering_snapshot_check_protocol_changed")
    _approved(_read(path / "spec.json"))
    return check


def record_engineering_repair(workspace: Path, check: dict, spec: dict, *, python=None) -> dict | None:
    """Register a resolved engineering error after the real full check passed."""
    from react_agent.eeg_research.agentic.coding_recovery import error_fingerprint
    from react_agent.eeg_research.agentic.artifacts import lookup, register
    from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task, validate_role_result
    from react_agent.eeg_research.agentic.task_ledger import task_snapshots, mark
    if check.get("ok") is not True or check.get("cached"):
        return None
    pointer = workspace / "engineering_failure_current.json"
    if not pointer.is_file():
        return None
    interpreter = _current_check(workspace, check, spec, python)
    selected = _read(pointer)
    failed_dir = workspace / "engineering_failures" / selected["failure_id"]
    if _sha(failed_dir / "manifest.json") != selected["manifest_sha256"]:
        raise ValueError("engineering_failure_selector_changed")
    before_manifest = _read(failed_dir / "manifest.json")
    before = _snapshot(failed_dir, before_manifest)
    experiment = _approved(spec)
    if (before.get("ok") is not False or not error_fingerprint(before)
            or before_manifest["spec_hash"] != experiment["spec_hash"]
            or before_manifest["candidate_id"] != workspace.name
            or _method_semantics(_approved(_read(failed_dir / "spec.json"))) != _method_semantics(experiment)
            or _sha(failed_dir / "input_spec.json") != _hash(_read(workspace / "input_spec.json"))
            or before_manifest["source_sha256"] == check["source_sha256"]):
        raise ValueError("engineering_repair_requires_same_spec_real_source_change")
    repair_id = _hash({"before": selected["manifest_sha256"], "after": check, "spec": experiment["spec_hash"]})
    camp = workspace.parent.parent
    directory = camp / "engineering_repairs" / workspace.name / repair_id
    passed_dir = directory / "passed"
    _write_once(passed_dir / "extension/eeg_candidate.py", (workspace / "extension/eeg_candidate.py").read_text(), text=True)
    _write_once(passed_dir / "input_spec.json", _read(workspace / "input_spec.json"))
    _write_once(passed_dir / "spec.json", spec)
    _write_once(passed_dir / "check.json", check)
    after_manifest = {"python": interpreter, "files": {name: _sha(passed_dir / name) for name in
                      ("extension/eeg_candidate.py", "input_spec.json", "spec.json", "check.json")}}
    _write_once(passed_dir / "manifest.json", after_manifest)
    record = {"schema_version": VERSION, "status": "completed", "scope": SCOPE, "record_id": repair_id, "candidate_id": workspace.name,
              "spec_hash": experiment["spec_hash"], "intervention": experiment.get("principal_intervention") or experiment.get("intervention"),
              "before": {"directory": str(failed_dir.resolve()), "manifest_sha256": selected["manifest_sha256"]},
              "after": {"directory": str(passed_dir.resolve()), "manifest_sha256": _sha(passed_dir / "manifest.json")},
              "error_category": before_manifest["error_category"], "error_fingerprint": before_manifest["error_fingerprint"],
              "stage": before.get("stage"), "hook": before.get("hook") or (before.get("location") or {}).get("function"),
              "conditions": {"scope": SCOPE, "input_spec_sha256": _sha(passed_dir / "input_spec.json"),
                             "spec_hash": experiment["spec_hash"], "python": interpreter},
              "evidence_kind": "native_check_failure_resolved", "scientific_gain_claim": False,
              "producer_kind": "native_engineering_observer", "model_invoked": False,
              "limitation": "Historical engineering check evidence only; current source still requires its own check/review and EEG gain requires measured scientific evidence."}
    inputs = [failed_dir / "manifest.json", passed_dir / "manifest.json"]
    inputs += [failed_dir / name for name in before_manifest["files"]]
    inputs += [passed_dir / name for name in after_manifest["files"]]
    target = directory / "repair.json"
    # Recover a receipt written before registration/completion without changing
    # its bytes or choosing a new producer task/artifact identity.
    if target.is_file():
        envelope = _read(target)
        task = next((row for row in task_snapshots(camp) if row["task_id"] == envelope["task_id"]), None)
        if not task or envelope.get("payload") != record:
            raise ValueError("engineering_repair_receipt_identity_changed")
        validate_role_result(envelope, task)
        artifact_id = envelope["artifact_refs"][-1]
        if task.get("status") == "completed":
            return verify_engineering_artifact(camp, artifact_id)
        if task.get("status") not in {"pending", "running"}:
            raise ValueError("engineering_repair_producer_not_recoverable")
        if lookup(camp, artifact_id) is None:
            register(camp, target, kind=KIND, producer_task_id=task["task_id"], candidate_id=workspace.name, artifact_id=artifact_id)
        mark(camp, task["task_id"], "completed", role="candidate_coder", attempt_id=task["attempt_id"],
             candidate_id=workspace.name, artifact_id=artifact_id, path=str(target))
    else:
        task = begin_role_task(camp, role="candidate_coder", inputs=inputs, candidate_id=workspace.name,
                               request={"action": "record_engineering_repair", "record": record})
        envelope = finish_role_task(camp, task, {"status": "completed", **record}, kind=KIND,
                                    path=target, candidate_id=workspace.name)
    return verify_engineering_artifact(camp, envelope["artifact_refs"][-1])


def verify_engineering_artifact(camp: Path, artifact_id: str) -> dict:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact, verify_input_dependencies
    from react_agent.eeg_research.agentic.task_ledger import latest
    from react_agent.eeg_research.agentic.roles import validate_role_result
    row = resolve_verified_artifact(camp, artifact_id)
    if row.get("kind") != KIND or not Path(row["path"]).resolve().is_relative_to(camp.resolve()):
        raise ValueError("engineering_artifact_kind_or_location_invalid")
    envelope = _read(Path(row["path"]))
    task = latest(camp, row.get("producer_task_id"))
    if not task or task.get("status") != "completed" or task.get("role") != "candidate_coder" or task.get("artifact_id") != artifact_id:
        raise ValueError("engineering_completed_native_producer_required")
    validate_role_result(envelope, task)
    dependencies = (task.get("recovery_identity") or {}).get("input_dependencies")
    if not dependencies or not verify_input_dependencies(camp, dependencies):
        raise ValueError("engineering_artifact_dependencies_changed")
    record = envelope["payload"]
    if (record.get("schema_version") != VERSION or record.get("scope") != SCOPE
            or record.get("scientific_gain_claim") is not False or record.get("candidate_id") != row.get("candidate_id")):
        raise ValueError("engineering_artifact_scope_invalid")
    checks = []
    specs = []
    for side in ("before", "after"):
        path = Path(record[side]["directory"]).resolve()
        if not path.is_relative_to(camp.resolve()) or _sha(path / "manifest.json") != record[side]["manifest_sha256"]:
            raise ValueError("engineering_artifact_snapshot_changed")
        checks.append(_snapshot(path, _read(path / "manifest.json")))
        specs.append(_approved(_read(path / "spec.json")))
    from react_agent.eeg_research.agentic.coding_recovery import error_fingerprint
    if (checks[0].get("ok") is not False or checks[1].get("ok") is not True
            or not error_fingerprint(checks[0]) or checks[0]["source_sha256"] == checks[1]["source_sha256"]
            or any(spec["spec_hash"] != record["spec_hash"] for spec in specs)
            or _method_semantics(specs[0]) != _method_semantics(specs[1])):
        raise ValueError("engineering_artifact_check_transition_invalid")
    return {**row, "scope": SCOPE, "record": record, "before_check": checks[0], "after_check": checks[1],
            "native_completion_verified": True}


def verified_repairs(camp: Path, *, candidate_id: str | None = None, max_items: int = 3) -> list[dict]:
    from react_agent.eeg_research.agentic.artifacts import REGISTRY
    path = camp / REGISTRY
    if not path.is_file() or max_items <= 0:
        return []
    results = []
    for line in reversed(path.read_text().splitlines()):
        row = json.loads(line)
        if row.get("kind") != KIND or candidate_id and row.get("candidate_id") != candidate_id:
            continue
        try:
            results.append(verify_engineering_artifact(camp, row["artifact_id"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(results) >= max_items:
            break
    return results


def coder_hints(camp: Path, *, candidate_id: str | None = None, max_items: int = 3) -> dict:
    hints = []
    for verified in verified_repairs(camp, candidate_id=candidate_id, max_items=max_items):
        record = verified["record"]
        line = (verified["before_check"].get("location") or {}).get("line", 1)
        snippets = {}
        for side in ("before", "after"):
            lines = (Path(record[side]["directory"]) / "extension/eeg_candidate.py").read_text().splitlines()
            start, end = max(1, int(line) - 3), min(len(lines), int(line) + 7)
            snippets[side] = {"start": start, "end": end, "text": "\n".join(lines[start-1:end]),
                              "source_sha256": verified[side + "_check"]["source_sha256"]}
        hints.append({"artifact_id": verified["artifact_id"], "sha256": verified["sha256"],
                      "candidate_id": record["candidate_id"], "error_category": record["error_category"],
                      "stage": record["stage"], "hook": record["hook"], "conditions": record["conditions"],
                      "snippets": snippets, "scope": SCOPE, "scientific_gain_claim": False})
    return {"scope": SCOPE, "items": hints,
            "instruction": "Historical verified repair hints; adapt only applicable code, preserve the current approved intervention and obtain a fresh current check. No scientific gain evidence."}


def curation_context(camp: Path, *, candidate_id: str | None = None, max_items: int = 3) -> dict:
    from react_agent.eeg_research.agentic.skill_memory import artifact_source, task_source
    rows = verified_repairs(camp, candidate_id=candidate_id, max_items=max_items)
    return {"engineering_repairs": coder_hints(camp, candidate_id=candidate_id, max_items=max_items),
            "allowed_engineering_source_refs": {"origin_campaign_id": camp.name, "scope": SCOPE,
                "tasks": [task_source(camp, row["producer_task_id"]) for row in rows],
                "artifacts": [artifact_source(camp, row["artifact_id"]) for row in rows], "episodes": []}}


def extend_curation_context(camp: Path, context: dict, *, candidate_id: str | None = None) -> dict:
    """Add verified repair source refs without changing scientific source scope."""
    extension = curation_context(camp, candidate_id=candidate_id)
    result = {**context, "engineering_repairs": extension["engineering_repairs"]}
    allowed = dict(context.get("allowed_source_refs") or {"origin_campaign_id": camp.name})
    engineering = extension["allowed_engineering_source_refs"]
    if allowed.get("origin_campaign_id") != camp.name:
        raise ValueError("engineering_curation_origin_changed")
    for field, identity in (("tasks", "task_id"), ("artifacts", "artifact_id"), ("episodes", "episode_id")):
        rows = list(allowed.get(field) or [])
        existing = {row[identity] for row in rows}
        rows += [row for row in engineering[field] if row[identity] not in existing]
        allowed[field] = rows
    result["allowed_source_refs"] = allowed
    return result
