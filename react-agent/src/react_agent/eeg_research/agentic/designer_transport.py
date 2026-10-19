"""Local Designer assembly inputs and concise correction feedback.

The immutable native context is deliberately separate from provider text.
Raw model responses are never changed by these helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .context.types import canonical


def freeze_assembly_context(camp: Path, payload: dict, *, task: dict,
                            operation_id: str) -> tuple[dict, dict, str]:
    from .designer_context import designer_assembly_context
    from .loop import _write

    identity = {key: task.get(key) for key in ("task_id", "attempt_id", "input_digest")}
    name = hashlib.sha256(canonical(identity).encode()).hexdigest()[:32]
    path = camp / "designer_inputs" / (name + ".json")
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("schema_version") != "eeg_research.designer_assembly.v1"
                or any(record.get(key) != value for key, value in identity.items())
                or not isinstance(record.get("operation_id"), str)
                or not isinstance(record.get("assembly_context"), dict)):
            raise ValueError("designer_assembly_identity_mismatch")
    else:
        record = {"schema_version": "eeg_research.designer_assembly.v1", **identity, "operation_id": operation_id,
                  "assembly_context": designer_assembly_context(payload)}
        # Use the native atomic create path: a crash must not leave half a
        # sidecar which a restart could mistake for frozen assembly inputs.
        _write(path, record, exclusive=True)
    reference = {"path": str(path.resolve()),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return record["assembly_context"], reference, record["operation_id"]


def correction_feedback(errors: list[dict]) -> dict:
    rows = []
    for error in errors:
        location = error.get("loc") or []
        path = "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1")
                               for part in location)
        kind = error.get("type", "invalid")
        rows.append({"path": path, "type": kind,
                     "message": "missing " + path if kind == "missing" else str(error.get("msg", kind))})
    return {"kind": "designer_output_invalid", "errors": rows,
            "instruction": "Return the complete corrected JSON matching OUTPUT_SCHEMA. Fix all listed fields and preserve unaffected design choices."}


def native_designer_body(checked: dict) -> dict:
    """Return a local consumer record; the provider journal keeps the wire reply."""
    body = dict(checked["reply"])
    if body.get("status") == "completed" and body.get("missing_inputs"):
        body["status"] = "blocked"
    if checked.get("experiment_spec") is not None:
        body["experiment_spec"] = checked["experiment_spec"]
        body["design_brief"] = checked["design_brief"]
    return body
