"""Reuse or adopt exact native maintenance proposals without another paid call.

Recovered proposals are not published snapshots, scientific evidence, reads or
execution permission. Phase A/B/Snapshot validators still check current sources
and CAS before any memory publication.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .types import digest


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _proposal(envelope: dict) -> dict | None:
    body = envelope.get("payload") or {}
    proposal = body.get("proposal")
    if (envelope.get("status") != "completed" or not isinstance(proposal, dict)
            or body.get("publication_status") != "proposal_only"
            or body.get("scientific_evidence") is not False
            or body.get("execution_authority") is not False):
        return None
    return proposal


def recover_maintenance(camp: Path, *, request: dict, recovery_identity: dict,
                        inputs: list[Path], resolved: Any, system: str,
                        operation: str) -> dict | None:
    """Recover an exact same role/config/input proposal with proven native cost."""
    from react_agent.eeg_research.agentic.artifacts import (
        input_dependency_manifest, request_digest, resolve_verified_artifact,
        verify_input_dependencies,
    )
    from react_agent.eeg_research.agentic.roles import (
        assert_role_task_dependencies, finish_role_task, validate_role_result,
    )
    from react_agent.eeg_research.agentic.task_ledger import (
        find_completed_task, mark, record_task_reuse, recovery_identity_digest,
        task_snapshots,
    )
    from react_agent.eeg_research.agentic.llm import transport_options

    camp = Path(camp)
    dependencies = input_dependency_manifest(camp, paths=inputs)
    if not verify_input_dependencies(camp, dependencies):
        return None
    # Mirror begin_role_task's routing and source dependency additions exactly.
    identity = {**recovery_identity, "input_dependencies": dependencies,
        "config": {**(recovery_identity.get("config") or {}),
                   "role_model_config": resolved.semantics(),
                   "role_config_sources": resolved.sources}}
    wanted = recovery_identity_digest(identity)
    submitted = {**request, "role_model_config": resolved.semantics()}
    wanted_input = request_digest(request=submitted, paths=inputs)
    cached = find_completed_task(camp, role="memory_curator", recovery_identity=identity,
                                 input_digest=wanted_input)
    if cached and cached["artifact"].get("kind") == "handoff_compaction":
        proposal = _proposal(cached["envelope"])
        if proposal is not None:
            assert_role_task_dependencies(camp, cached["task"])
            record_task_reuse(camp, cached)
            return proposal

    calls = _rows(camp / "llm_calls.jsonl")
    intents = _rows(camp / "llm_api_intents.jsonl")
    settled = {row.get("call_id") for row in calls}
    registry = _rows(camp / "artifact_registry.jsonl")
    expected_system_hash = hashlib.sha256(system.encode()).hexdigest()
    for task in reversed(task_snapshots(camp)):
        if (task.get("role") != "memory_curator" or task.get("status") not in {"pending", "running"}
                or task.get("recovery_identity_digest") != wanted or task.get("input_digest") != wanted_input):
            continue
        try:
            if recovery_identity_digest(task["recovery_identity"]) != wanted:
                continue
            assert_role_task_dependencies(camp, task)
        except (ValueError, KeyError, TypeError):
            continue
        if any(row.get("task_id") == task["task_id"] and row.get("call_id") not in settled for row in intents):
            continue
        # Never skip a newer failed/unknown attempt to adopt an older success.
        call = next((row for row in reversed(calls) if row.get("task_id") == task["task_id"]), None)
        if (not call or call.get("success") is not True or call.get("transport_outcome") != "confirmed_response"
                or call.get("operation") != operation or call.get("role") != "memory_curator"
                or call.get("requested_model") != resolved.model or call.get("role_config_hash") != resolved.role_config_hash
                or call.get("prompt_hash") != identity["prompt_hash"]):
            continue
        call_id = str(call["call_id"])
        try:
            traced = json.loads((camp / "llm_requests" / (call_id + ".json")).read_text())
            response = json.loads((camp / "llm_responses" / (call_id + ".json")).read_text())
            if any(call.get(key) != task[key] or traced.get(key) != task[key] or response.get(key) != task[key]
                   for key in ("task_id", "attempt_id", "input_digest")):
                continue
            if (traced.get("call_id") != call_id or response.get("call_id") != call_id
                    or traced.get("operation") != operation or response.get("operation") != operation
                    or traced.get("role") != "memory_curator" or response.get("role") != "memory_curator"
                    or response.get("validation") not in {"pending", "valid"}
                    or not isinstance(response.get("reply"), dict)):
                continue
            reply_hash = hashlib.sha256(json.dumps(response["reply"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            if reply_hash != call.get("response_reply_sha256"):
                continue
            system_hash = hashlib.sha256(traced["system"].encode()).hexdigest()
            user_hash = hashlib.sha256(traced["user"].encode()).hexdigest()
            if (system_hash != expected_system_hash or system_hash != call.get("request_system_sha256")
                    or system_hash != traced.get("system_sha256")
                    or user_hash != call.get("request_user_sha256") or user_hash != traced.get("user_sha256")):
                continue
            parsed = json.loads(traced["user"])
            if any(parsed.get(key) != task[key] for key in ("task_id", "attempt_id", "input_digest")):
                continue
            frozen_digest = digest({"system": traced["system"], "user": traced["user"],
                                    "transport_fields": transport_options(resolved.profile())})
            if frozen_digest != call.get("context_request_digest"):
                continue
            # Revalidate the adapter's exact rule: a complete JSON object.
            # Rich semantic/source/scope validation belongs to publication.
            proposal = response["reply"]
        except (ValueError, OSError, KeyError, TypeError):
            continue
        for registered in reversed(registry):
            if registered.get("producer_task_id") != task["task_id"] or registered.get("kind") != "handoff_compaction":
                continue
            try:
                artifact = resolve_verified_artifact(camp, registered["artifact_id"])
                envelope = validate_role_result(json.loads(Path(artifact["path"]).read_text()), task)
                if envelope.get("prompt_hash") != identity["prompt_hash"] or _proposal(envelope) != proposal:
                    continue
            except (ValueError, OSError, KeyError, TypeError):
                continue
            mark(camp, task["task_id"], "completed", role="memory_curator", attempt_id=task["attempt_id"],
                 artifact_id=artifact["artifact_id"], path=artifact["path"], recovered_transport_call_id=call_id)
            return proposal
        # Preserve any unregistered partial original file after a crash.
        finish_role_task(camp, task, {"status": "completed", "prompt_hash": identity["prompt_hash"],
            "proposal": proposal, "publication_status": "proposal_only", "scientific_evidence": False,
            "execution_authority": False, "recovered_transport_call_id": call_id}, kind="handoff_compaction",
            path=camp / "context_proposals" / (task["task_id"] + "_recovered_" + uuid.uuid4().hex[:12] + ".json"))
        return proposal
    return None
