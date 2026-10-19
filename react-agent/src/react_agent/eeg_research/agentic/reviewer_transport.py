"""Freeze local review evidence after the actual provider view is compiled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .context.types import digest


def _path(camp: Path, operation_id: str, payload: dict) -> Path:
    return camp / "reviewer_inputs" / (digest({"operation_id": operation_id, "payload": payload}) + ".json")


def freeze_review_delivery(camp: Path, prepared, native_context: dict, *,
                           operation_id: str, identity: dict) -> tuple[dict, dict]:
    from .reviewer_context import bind_review_delivery
    from .loop import _write

    context = bind_review_delivery(prepared.payload, native_context,
        system_sha256=hashlib.sha256(prepared.system.encode()).hexdigest(),
        user_sha256=hashlib.sha256(prepared.user.encode()).hexdigest(),
        request_digest=prepared.request_digest)
    record = {"schema_version": "eeg_research.reviewer_delivery.v2",
              "operation_id": operation_id, "identity": identity,
              "payload_digest": digest(prepared.payload),
              "request_digest": prepared.request_digest,
              "system_sha256": hashlib.sha256(prepared.system.encode()).hexdigest(),
              "user_sha256": hashlib.sha256(prepared.user.encode()).hexdigest(),
              "native_context": context}
    path = _path(camp, operation_id, prepared.payload)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("reviewer_delivery_identity_mismatch")
    else:
        _write(path, record, exclusive=True)
    return context, {"path": str(path.resolve()),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_review_delivery(camp: Path, operation_id: str, payload: dict) -> dict:
    record = json.loads(_path(camp, operation_id, payload).read_text(encoding="utf-8"))
    if (record.get("schema_version") != "eeg_research.reviewer_delivery.v2"
            or record.get("operation_id") != operation_id
            or record.get("payload_digest") != digest(payload)):
        raise ValueError("reviewer_delivery_identity_mismatch")
    return record["native_context"]


def review_call_metadata(camp: Path, operation_id: str) -> dict:
    """Derive counters from the durable journal, including paid recovery."""
    path = camp / "llm_calls.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    rows = [row for row in rows if row.get("operation_id") == operation_id]
    return {"calls": len(rows), "schema_repaired": any(row.get("application_retry_kind") == "domain_schema" for row in rows)}


def verify_review_delivery(camp: Path, call: dict, traced: dict, response: dict,
                           payload: dict) -> dict:
    ref = call.get("reviewer_delivery_ref")
    if (not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256")
            or any(row.get("reviewer_delivery_ref") != ref for row in (traced, response))):
        raise ValueError("reviewer_delivery_proof_missing")
    path = Path(ref["path"]).resolve()
    if path != _path(camp, call["operation_id"], payload).resolve():
        raise ValueError("reviewer_delivery_path_invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("reviewer_delivery_hash_mismatch")
    record = json.loads(raw)
    if (record.get("request_digest") != call.get("context_request_digest")
            or record.get("system_sha256") != traced.get("system_sha256")
            or record.get("user_sha256") != traced.get("user_sha256")
            or record.get("identity") != {key: call[key] for key in record.get("identity", {}) if key in call}):
        raise ValueError("reviewer_delivery_request_mismatch")
    return load_review_delivery(camp, call["operation_id"], payload)
