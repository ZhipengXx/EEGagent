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


def training_semantics(*, config_only: bool = False) -> list[dict[str, Any]]:
    """Source evidence for how the trainer calls configs and owns logit scaling."""
    import ast
    root = Path(__file__).resolve().parents[2] / "eeg_training"
    views = []
    selections = (("model.py", {"LocalRetrieval", "contrastive_loss"}),
                           ("hooks.py", {"compute_objective", "scalar_loss", "is_custom_objective"}), ("train_entry.py", {
        "_build_hook", "_logit_scale", "_instantiate_candidate", "train_channel_statistics",
        "unique_trainable_parameters", "placed_retrieval", "rebuild_encoder", "fit",
    }))
    if config_only:
        selections = (("hooks.py", {"call_configured"}), ("train_entry.py", {"_build_hook"}),
                      ("../eeg_research/agentic/hook_config.py", {"normalize_hook_config"}))
    for name, selected in selections:
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


def candidate_cpu_check(camp: Path, target: str, binding: dict[str, Any],
                        current_source_hash: str | None) -> dict[str, Any]:
    """Expose actual CPU receipts only for the source and config they checked."""
    if target == "baseline":
        return {"status":"runtime_frozen_baseline_reference","checks":None}
    workspace = camp / "candidates" / target
    path = workspace / "checks.json"
    view = {"status":"unavailable","check_ref":str(path),"checks":None,"missing_inputs":[]}
    try:
        checks = json.loads(path.read_text(encoding="utf-8"))
        assigned = json.loads((workspace / "spec.json").read_text(encoding="utf-8"))
        experiment = assigned.get("experiment") or assigned
    except (OSError, ValueError) as exc:
        view["missing_inputs"] = ["cpu_check_or_assigned_spec_unreadable:"+type(exc).__name__]
        return view
    if not isinstance(checks,dict) or not isinstance(experiment,dict):
        view["missing_inputs"] = ["cpu_check_or_spec_not_object"]
        return view
    expected = {key:binding.get(key) or {} for key in ("model","objective","transform")}
    identity_ok = (current_source_hash and current_source_hash == binding.get("source_hash")
                   and checks.get("source_sha256") == current_source_hash
                   and experiment.get("spec_hash") == binding.get("spec_hash")
                   and checks.get("approved_hook_config") == expected)
    if not identity_ok:
        view["status"] = "identity_mismatch"
        view["missing_inputs"] = ["cpu_receipt_does_not_match_current_approved_source_and_config"]
        return view
    from react_agent.eeg_research.agentic.native_patch import _check_fingerprint
    try:
        current_fingerprint = _check_fingerprint(workspace,None)
    except (OSError, ValueError, KeyError):
        current_fingerprint = None
    view.update(status="verified_source_config_bound_cpu_receipt",source_hash=current_source_hash,
                spec_hash=binding.get("spec_hash"),check_sha256=file_digest(path),
                check_fingerprint=checks.get("check_fingerprint"),
                current_checker_fingerprint=current_fingerprint,
                fresh_for_current_runtime=bool(current_fingerprint and checks.get("check_fingerprint")==current_fingerprint),
                checks=development_view(checks),
                scope="Actual saved synthetic CPU interface/gradient/formula/off/checkpoint probes for this source/config. Freshness is explicit. This is not an original GPU optimizer-step receipt, training score, benefit or runtime authorization.")
    return view


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
        "cpu_check_receipt": candidate_cpu_check(camp,target,binding,current_hash),
        "approval_ref": approval_ref,
        "spec_hash": binding.get("spec_hash"),
        "model": binding.get("model") or {},
        "objective": binding.get("objective") or {},
        "transform": binding.get("transform") or {},
        "intervention": binding.get("intervention") or ("frozen EEGProjectLayer baseline" if target == "baseline" else None),
        "recipe": {key: protocol.get(key) for key in ("full_epochs", "batch_size", "lr", "weight_decay",
                   "negative_sampling_policy", "training_seeds", "input_geometry")},
        "negative_sampling_policy": binding.get("negative_sampling_policy") or protocol.get("negative_sampling_policy"),
        "frozen_training_policy": {"full_epochs": protocol.get("full_epochs"),
            "fidelity_overrides": protocol.get("fidelity_overrides"),
            "training_seeds": protocol.get("training_seeds"),
            "interpretation": "Maximum epochs and early-stop rules are frozen runtime invariants. A build_encoder hook cannot change them. Fewer realized epochs does not prove unequal budgets."},
        "missing_inputs": missing,
    })


def implementation_lifecycle_context(camp: Path, state: dict[str, Any], target: str,
                                     available: list[str]) -> dict[str, Any]:
    """Explain the existing implementation gate without materializing a candidate."""
    from react_agent.eeg_research.agentic.experiment_gate import (
        ExperimentResolutionError, resolve_approved_experiment,
    )
    repair = state.get('repair_task') if isinstance(state.get('repair_task'), dict) else {}
    repairing = bool(repair.get('candidate_id') == target and int(repair.get('remaining') or 0) > 0)
    view = {'target_id': target, 'available': 'implement_candidate' in available,
            'stage': 'repair_existing_candidate' if repairing else 'implement_registered_design',
            'authority': 'Read-only explanation of the existing resolver. Execution revalidates all gates; this view grants no approval.'}
    if not view['available']:
        return {**view, 'status': 'implementation_not_currently_available'}
    assigned = state.get('experiment') if isinstance(state.get('experiment'), dict) else {}
    try:
        resolved = resolve_approved_experiment(camp, spec_ref=state.get('experiment_ref'),
            expected_hash=assigned.get('spec_hash') if not repairing else None,
            target_id=target, attempt_id=repair.get('attempt_id'), state=state,
            action='repair' if repairing else 'implement')
    except ExperimentResolutionError as exc:
        return {**view, 'status': 'design_resolution_failed', 'reason': exc.reason}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**view, 'status': 'design_resolution_failed', 'reason': type(exc).__name__}
    workspace = camp / 'candidates' / target
    source_exists = (workspace / 'extension/eeg_candidate.py').is_file()
    binding_exists = load_approved_binding(camp, target) is not None
    expected_absence = []
    if not repairing:
        if not source_exists: expected_absence.append('source_not_yet_generated_by_implement_candidate')
        if not binding_exists: expected_absence.append('candidate_binding_materialized_by_worker_from_resolved_registered_design')
    return development_view({**view, 'status': 'registered_design_verified_for_existing_implementation_gate',
        'spec_hash': resolved.get('spec_hash'), 'experiment_ref': state.get('experiment_ref'),
        'approval_record': resolved.get('approval_record'),
        'source_exists': source_exists, 'candidate_binding_exists': binding_exists,
        'expected_preimplementation_absence': expected_absence,
        'parent_candidate_id': resolved.get('parent_candidate_id'),
        'control_candidate_id': resolved.get('control_candidate_id'),
        'workflow': ['The registered approved design authorizes the implementation gate, not training.',
            'The worker creates the candidate workspace and binding, materializes parent source, and invokes the real Coder.',
            'CPU checks and Reviewer must pass before the candidate becomes eligible for training.',
            'A new candidate source cannot be required before the action whose purpose is to create that source.'],
        'limits': ['An available implementation action can compete with other useful actions; this view does not force a selection.',
            'Existing repair/training source binding, registered spec hashes, review, capability and budget gates remain unchanged.']})


def training_target_source_context(camp: Path, state: dict[str, Any], eligible: dict[str, list[str]]) -> dict[str, Any]:
    """Expose actual sources of current training targets, separately from proposals."""
    names = sorted({str(target) for action in ('run_pilot', 'run_full', 'replicate')
                    for target in eligible.get(action, [])})
    current = str(state.get('candidate_id') or '')
    if current in names:
        names.remove(current); names.insert(0, current)
    selected = names[:4]
    return {'by_candidate': {name: candidate_context(camp, name) for name in selected},
            'omitted_target_ids': names[4:], 'source_limit': 4,
            'scope': 'Current source/spec-bound candidate code for eligible training targets. A research proposal or pending design is not this source. Missing/truncated source remains explicit.'}


def verified_encoder_structural_facts(camp: Path, state: dict[str, Any], *, target_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Deliver registered source/checkpoint-bound synthetic encoder facts."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    facts = []; seen = set(); protocol = load_protocol(camp) or {}
    for evidence in reversed(state.get('evidence') or []):
        if evidence.get('kind') != 'diagnostic_bundle' or evidence.get('status') != 'verified': continue
        for ref in evidence.get('artifact_refs') or []:
            if ref in seen: continue
            try:
                record = resolve_verified_artifact(camp, ref)
                report = json.loads(Path(record['path']).read_text())
                if report.get('schema_version') != 'eeg_research.c8_actual_pooling_factual_audit.v1': continue
                target = report.get('candidate_id')
                if target_ids is not None and target not in target_ids: continue
                binding = load_approved_binding(camp, str(target)) or {}
                if (report.get('final_test_accessed') is not False or report.get('gpu_seconds') != 0
                        or report.get('execution_fingerprint') != protocol.get('fingerprint')
                        or report.get('sample_count_per_position') != 32
                        or binding.get('source_hash') != report.get('source_hash')
                        or binding.get('spec_hash') != report.get('spec_hash')): continue
                current_source = camp / 'candidates' / str(target) / 'extension/eeg_candidate.py'
                if file_digest(current_source) != report['source_hash']: continue
                for path, digest in report['artifact_hashes'].items():
                    p = Path(path)
                    if not p.resolve().is_relative_to(camp.resolve()) or file_digest(p) != digest:
                        raise ValueError('encoder_structural_artifact_binding_changed')
                run = next(e for e in state.get('evidence') or [] if e.get('job_id') == report.get('job_id')
                           and e.get('evaluation_valid') is True and e.get('source_hash') == report['source_hash']
                           and e.get('spec_hash') == report['spec_hash'] and e.get('seed') == report['seed'])
                seen.add(ref)
                facts.append(development_view({'evidence_id': evidence['evidence_id'], 'artifact_id': ref,
                    'artifact_sha256': record['sha256'], 'run_evidence_id': run['evidence_id'],
                    **{key: report[key] for key in ('candidate_id', 'source_hash', 'spec_hash', 'job_id', 'seed',
                        'scope', 'sample_count_per_position', 'positions', 'measurements', 'actual_encoder_parameters',
                        'source_pooling_expression', 'corrections', 'limitations', 'provenance')},
                    'authority': 'Verified external synthetic measurement/correction; no new EEG score or training authorization.'}))
            except (OSError, ValueError, KeyError, TypeError, StopIteration): continue
    return facts


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
    latest_completed = {}
    superseded = {}
    for row in state.get("evidence") or []:
        if row.get("kind") != "analysis" or (target and row.get("candidate_id") != target):
            continue
        key = row.get("run_evidence_id")
        if key and row.get("status") == "completed" and row.get("analysis_artifact_id"):
            try:
                registered = resolve_verified_artifact(camp, row["analysis_artifact_id"])
                body = json.loads(Path(registered["path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if registered.get("kind") == "analysis" and body.get("status") == "completed":
                previous = latest_completed.get(key)
                if previous:superseded.setdefault(key, []).append(previous.get("evidence_id"))
                latest_completed[key] = row
    for row in state.get("evidence") or []:
        if row.get("kind") != "analysis" or (target and row.get("candidate_id") != target):
            continue
        key = row.get("run_evidence_id")
        current = latest_completed.get(key)
        if row.get("status") == "completed" and current and current.get("evidence_id") != row.get("evidence_id"):
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
            "completion_status": body.get("status"), "run_evidence_id": key,
            "superseded_analysis_evidence_ids": superseded.get(key, []) if current is row else [],
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


def _read_journal_records(camp: Path) -> list[dict[str, Any]]:
    path = camp / READ_JOURNAL
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"artifact_read_journal_invalid:{number}") from exc
        if not isinstance(row, dict) or row.get("record_type", "read") not in {"read", "delivery"}:
            raise ValueError(f"artifact_read_journal_invalid:{number}")
        if row.get("record_type") == "delivery":
            body = {key: value for key, value in row.items() if key != "delivery_id"}
            expected = "delivery_" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if (row.get("delivery_id") != expected or not isinstance(row.get("reads"), list)
                    or not row.get("consumer") or not row.get("consumer_request_id") or row.get("receipt_id")):
                raise ValueError(f"artifact_delivery_journal_invalid:{number}")
        records.append(row)
    return records


def read_history(camp: Path) -> list[dict[str, Any]]:
    return [row for row in _read_journal_records(camp) if row.get("record_type", "read") == "read"]


def read_digest(request: dict[str, Any], content_hash: str | None) -> str:
    from react_agent.eeg_research.agentic.artifacts import request_digest
    return request_digest(request={"artifact_id": request.get("artifact_id"),
                                  "start": request.get("start", 0), "end": request.get("end"),
                                  "content_hash": content_hash, "scope": "development_view"})


def _receipt_id(row: dict[str, Any]) -> str:
    keys = ("artifact_id", "request_digest", "request", "decision_id", "epoch", "status", "sha256",
            "view_sha256", "range_scope", "start", "end", "missing_inputs")
    identity = {key: row.get(key) for key in keys}
    identity["text_sha256"] = hashlib.sha256((row.get("text") or "").encode("utf-8")).hexdigest()
    return "read_" + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _append_read_record(camp: Path, row: dict[str, Any]) -> None:
    import os
    with (camp / READ_JOURNAL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_read_receipt(camp: Path, saved: dict[str, Any]) -> dict[str, Any]:
    """Recompute source, request, filtered view, range and actual bytes before reuse."""
    from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range
    from react_agent.eeg_research.agentic.schemas import ArtifactReadRequest
    if saved.get("receipt_id") and saved["receipt_id"] != _receipt_id(saved):
        raise ValueError("artifact_read_receipt_identity_changed")
    registered = lookup(camp, str(saved.get("artifact_id") or ""))
    source_hash = saved.get("sha256") if saved.get("status") == "read" else (registered or {}).get("sha256")
    request = saved.get("request")
    if request is None:
        path = camp / "decisions" / f"{saved.get('decision_id')}.json"
        raw = json.loads(path.read_text()).get("raw") or {} if path.is_file() else {}
        candidates = list(raw.get("read_requests") or [])
        candidates += [{"artifact_id": saved.get("artifact_id"), "start": saved.get("start", 0), "end": end}
                       for end in (None, saved.get("end"), int(saved.get("start") or 0) + READ_RANGE_CHARS)]
        request = next((row for row in candidates if isinstance(row, dict)
                        and read_digest(row, source_hash) == saved.get("request_digest")), None)
    if not isinstance(request, dict) or read_digest(request, source_hash) != saved.get("request_digest"):
        raise ValueError("artifact_read_request_identity_changed")
    parsed = ArtifactReadRequest.model_validate(request)
    if parsed.artifact_id != saved.get("artifact_id") or (parsed.end is not None and parsed.end - parsed.start > READ_RANGE_CHARS):
        raise ValueError("artifact_read_request_identity_changed")
    if saved.get("status") != "read":
        # Failed receipts authorize no text. Their original request still binds
        # replay identity and preserves the existing charged-attempt policy.
        if saved.get("text") is not None:
            raise ValueError("artifact_read_failed_receipt_has_text")
        return {**saved, "content_injected": False}
    current = development_artifact(camp, parsed.artifact_id)
    if current["sha256"] != saved.get("sha256"):
        raise ValueError("artifact_read_hash_changed")
    start, end = saved.get("start"), saved.get("end")
    if (not isinstance(start, int) or not isinstance(end, int) or start != parsed.start
            or end < start or end - start > READ_RANGE_CHARS or (parsed.end is not None and end > parsed.end)
            or saved.get("range_scope") != "development_view"):
        raise ValueError("artifact_read_range_identity_changed")
    view = read_verified_range(camp, parsed.artifact_id, start=start, end=end,
                               json_view=development_view, page_chars=READ_RANGE_CHARS)
    if view["view_sha256"] != saved.get("view_sha256") or view["text"] != saved.get("text") or view["end"] != end:
        raise ValueError("artifact_read_view_changed")
    return {**saved, **view, "content_injected": False, "delivery_status": "prepared"}


def rebuild_read_projection(camp: Path, state: dict[str, Any]) -> None:
    history = read_history(camp)
    state["artifact_read_history"] = [{key: row.get(key) for key in
        ("artifact_id", "request_digest", "receipt_id", "decision_id", "epoch", "status")} for row in history]
    if not history:
        return
    latest_id = history[-1].get("decision_id")
    if latest_id is None:
        # Optional direct callers have no persisted batch identity. Preserve a
        # still-verifiable latest projection, rather than guess a history-wide
        # default batch. Explicit artifact/page selection remains available.
        saved = state.get("artifact_read_results") or []
        rows = [row for item in saved for row in history if row.get("decision_id") is None
                and row.get("request_digest") == item.get("request_digest")]
        if not rows:
            state["artifact_read_results"] = [{"status": "missing_input", "text": None,
                "missing_inputs": ["artifact_read_batch_identity_missing"], "content_injected": False}]
            return
    else:
        rows = [row for row in history if row.get("decision_id") == latest_id]
    results = []
    for row in rows:
        try:
            results.append(verify_read_receipt(camp, row))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({"artifact_id": row.get("artifact_id"), "request_digest": row.get("request_digest"),
                            "status": "missing_input", "missing_inputs": [str(exc)], "text": None})
    state["artifact_read_results"] = results


def consume_artifact_reads(camp: Path, state: dict[str, Any], requests: list[dict[str, Any]],
                           decision_id: str | None) -> dict[str, Any]:
    """Journal bounded attempts once; verified replay of the same decision is free."""
    from react_agent.eeg_research.agentic.artifacts import lookup, read_verified_range
    from react_agent.eeg_research.agentic.schemas import ArtifactReadRequest
    record = next((row for row in state.get("decisions") or [] if row.get("decision_id") == decision_id), {})
    epoch = str(record["evidence_count"]) if "evidence_count" in record else read_epoch(state)
    history = read_history(camp)
    results = []
    used = sum(row.get("epoch") == epoch and not row.get("replayed") for row in history)
    remaining = READ_TOTAL_CHARS
    if len(requests) > READ_MAX_IDS:
        return {"status": "limited", "missing_inputs": ["artifact_read_request_limit"], "results": []}
    for raw in requests:
        ref = raw.get("artifact_id") if isinstance(raw, dict) else None
        registered = lookup(camp, str(ref)) if ref else None
        digest = read_digest(raw, None if registered is None else registered.get("sha256"))
        prior = [row for row in history if row.get("request_digest") == digest]
        previous = next((row for row in prior if row.get("decision_id") == decision_id), None)
        if previous:
            try:
                view = verify_read_receipt(camp, previous)
                if len(view.get("text") or "") > remaining:
                    raise ValueError("artifact_read_total_limit")
                remaining -= len(view.get("text") or "")
                results.append({**view, "replayed": True})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results.append({"artifact_id": ref, "status": "missing_input", "text": None,
                                "missing_inputs": [str(exc)], "replayed": True})
            continue
        if prior:
            results.append({"artifact_id": ref, "status": "missing_input", "text": None,
                            "missing_inputs": ["artifact_read_replayed"]})
            continue
        result = {"artifact_id": ref, "request_digest": digest, "request": raw,
                  "decision_id": decision_id, "record_type": "read", "epoch": epoch,
                  "status": "missing_input", "missing_inputs": [], "text": None, "content_injected": False}
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
                                       json_view=development_view, max_chars=min(READ_RANGE_CHARS, remaining),
                                       page_chars=READ_RANGE_CHARS)
            result.update(view, status="read")
            remaining -= len(view["text"])
        except (OSError, ValueError, KeyError) as exc:
            result["missing_inputs"] = [str(exc)]
        result["receipt_id"] = _receipt_id(result)
        _append_read_record(camp, result)
        history.append(result)
        used += 1
        results.append(result)
    state["artifact_read_history"] = [{key: row.get(key) for key in
        ("artifact_id", "request_digest", "receipt_id", "decision_id", "epoch", "status")} for row in history]
    state["artifact_read_results"] = results
    return {"status": "completed", "results": results, "remaining_reads": max(0, READS_PER_EVIDENCE_EPOCH - used)}


def verified_read_context(camp: Path, state: dict[str, Any], *, artifact_refs: list[str] | None = None,
                          request_digests: list[str] | None = None) -> list[dict[str, Any]]:
    """One verifier serves default latest-batch and explicitly selected history."""
    from react_agent.eeg_research.agentic.artifacts import read_verified_range
    history = read_history(camp)
    explicit = artifact_refs is not None or request_digests is not None
    if explicit:
        refs, digests = artifact_refs or [], request_digests or []
        if digests:
            selected = [next((row for row in history if row.get("request_digest") == digest
                             and (not refs or row.get("artifact_id") in refs)),
                            {"request_digest": digest, "status": "missing_input", "missing_inputs": ["artifact_page_not_read"]})
                        for digest in digests]
        else:
            selected = [row for row in history if row.get("artifact_id") in refs]
        missing = [ref for ref in refs if not any(row.get("artifact_id") == ref for row in selected)]
        selected += [{"artifact_id": ref, "status": "missing_input", "missing_inputs": ["artifact_not_read"]} for ref in missing]
    else:
        selected = state.get("artifact_read_results") or []
        if not selected and history:
            latest_id = history[-1].get("decision_id")
            selected = ([row for row in history if row.get("decision_id") == latest_id] if latest_id is not None else
                        [{"status": "missing_input", "missing_inputs": ["artifact_read_batch_identity_missing"]}])
    rows, unique, chars, fragments = [], set(), READ_TOTAL_CHARS, 0
    for saved in selected:
        try:
            matching = [row for row in history if row.get("request_digest") == saved.get("request_digest")
                        and row.get("decision_id") == saved.get("decision_id")]
            if not matching or any(row != matching[0] for row in matching[1:]):
                raise ValueError("artifact_read_journal_binding_missing")
            recorded = matching[0]
            if any(recorded.get(key) != saved.get(key) for key in
                   ("artifact_id", "sha256", "start", "end", "view_sha256", "receipt_id", "text")):
                raise ValueError("artifact_read_journal_binding_missing")
            view = verify_read_receipt(camp, recorded)
            if view.get("status") == "read":
                if len(unique | {view["artifact_id"]}) > READ_MAX_IDS:
                    raise ValueError("artifact_read_unique_artifact_limit")
                if fragments >= READ_MAX_IDS:
                    raise ValueError("artifact_read_fragment_limit")
                if chars <= 0:
                    raise ValueError("artifact_read_context_limit")
                unique.add(view["artifact_id"])
                fragments += 1
                if len(view["text"]) > chars:
                    partial = read_verified_range(camp, view["artifact_id"], start=view["start"], end=view["start"] + chars,
                                                  json_view=development_view, page_chars=READ_RANGE_CHARS)
                    view = {**view, **partial, "receipt_range": [view["start"], view["end"]],
                            "delivery_complete": False, "missing_inputs": ["artifact_read_context_truncated"]}
                else:
                    view["delivery_complete"] = True
                chars -= len(view["text"])
            rows.append(view)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({"artifact_id": saved.get("artifact_id"), "request_digest": saved.get("request_digest"),
                         "receipt_id": saved.get("receipt_id"), "status": "missing_input", "text": None,
                         "content_injected": False, "delivery_complete": False,
                         "missing_inputs": saved.get("missing_inputs") or [str(exc)]})
    return rows


def prepare_read_delivery(rows: list[dict[str, Any]]) -> None:
    """Mark the prepared request before its task input digest is calculated."""
    for row in rows:
        if row.get("status") == "read":
            row["content_injected"] = True
            row["delivery_status"] = "request_constructed"


def record_read_delivery(camp: Path, rows: list[dict[str, Any]], *, consumer: str,
                         request_id: str, input_digest: str | None = None) -> None:
    """Record constructed consumer requests, never infer past delivery from a receipt."""
    prepare_read_delivery(rows)
    delivered = []
    for row in rows:
        if row.get("status") != "read":
            continue
        row["content_injected"] = True
        row["delivery_status"] = "request_constructed"
        delivered.append({key: row.get(key) for key in ("artifact_id", "request_digest", "receipt_id", "sha256",
                         "view_sha256", "range_scope", "start", "end", "receipt_range", "delivery_complete")})
    if not delivered:
        return
    body = {"record_type": "delivery", "consumer": consumer, "consumer_request_id": request_id,
            "consumer_input_digest": input_digest, "delivery_phase": "request_constructed", "reads": delivered}
    body["delivery_id"] = "delivery_" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if not any(row.get("delivery_id") == body["delivery_id"] for row in _read_journal_records(camp)):
        _append_read_record(camp, body)


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
    """A worker receives only explicitly cited artifacts/pages, with visible gaps."""
    return verified_read_context(camp, state, artifact_refs=list(state.get("active_artifact_refs") or []),
                                 request_digests=list(state.get("active_read_digests") or []))


def analysis_experiment_binding(camp: Path, latest: dict[str, Any]) -> dict[str, Any]:
    """Bind analysis to the approved spec and source of this exact accepted run."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    from react_agent.eeg_research.agentic.experiment_gate import _spec_hash, experiment_is_approved
    target=str(latest.get("candidate_id") or "")
    if target=="baseline":return {"status":"frozen_baseline", "missing_inputs":[]}
    binding=load_approved_binding(camp,target) or {}
    try:
        if not target or not binding:raise ValueError("approved_binding_missing")
        registry=resolve_verified_artifact(camp,str(binding.get("spec_ref") or ""))
        artifact=json.loads(Path(registry["path"]).read_text(encoding="utf-8"))
        spec=artifact.get("experiment_spec") or artifact.get("experiment") or artifact
        if not isinstance(spec,dict):raise ValueError("approved_spec_missing")
        if not experiment_is_approved(spec):raise ValueError("spec_not_approved")
        expected=latest.get("spec_hash")
        if not expected or _spec_hash(spec)!=expected or binding.get("spec_hash")!=expected:
            raise ValueError("run_spec_hash_mismatch")
        if binding.get("target_id")!=target or binding.get("source_hash")!=latest.get("source_hash"):
            raise ValueError("run_source_binding_mismatch")
        job=Path(str(latest.get("job_dir") or ""))
        if not job.resolve().is_relative_to(camp.resolve()):raise ValueError("run_outside_campaign")
        frozen=json.loads((job/"frozen_run_spec.json").read_text())
        source=json.loads((job/"source_binding.json").read_text())
        if frozen.get("spec_hash")!=expected or frozen.get("candidate_id", frozen.get("target_id"))!=target:
            raise ValueError("frozen_run_spec_mismatch")
        path=Path(str(source.get("class_file") or ""));path=path if path.is_absolute() else job/path
        if source.get("file_sha256")!=latest.get("source_hash") or file_digest(path)!=latest.get("source_hash"):
            raise ValueError("loaded_source_changed")
        hypothesis=spec.get("hypothesis")
        if not hypothesis:raise ValueError("approved_hypothesis_missing")
        return {"status":"verified", "hypothesis":hypothesis, "experiment":spec,
                "spec_hash":expected,"source_hash":latest.get("source_hash"),
                "artifact_ref":registry["artifact_id"],"artifact_sha256":registry["sha256"],
                "missing_inputs":[]}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {"status":"unavailable","missing_inputs":[str(exc)]}
