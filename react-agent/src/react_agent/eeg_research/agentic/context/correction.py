"""Canonical, bounded child requests for mechanical memory publication repair."""
from __future__ import annotations

import copy
from typing import Any, Mapping
from pathlib import Path

from .types import canonical, digest


TRANSPORT_META = "_maintenance_transport"


def split_transport(value: Any) -> tuple[Any, dict[str, Any]]:
    """Keep local transport accounting out of the scientific proposal schema."""
    if not isinstance(value, Mapping):
        return value, {}
    proposal = dict(value)
    metadata = proposal.pop(TRANSPORT_META, {})
    if TRANSPORT_META in value and "output" in proposal:
        proposal = proposal["output"]
    return proposal, dict(metadata) if isinstance(metadata, Mapping) else {}


def build_child_request(base_payload: Mapping[str, Any], invalid_response: Any,
                        errors: list[str], *, parent_operation_id: str,
                        kind: str, parent_task_ref: str = "",
                        invalid_draft_ref: str = "", max_draft_chars: int = 3_000) -> dict[str, Any]:
    """Identity depends on the actual error and parent proposal, never a clock.

    Large drafts remain readable immutable artifacts. A short complete draft is
    included when it fits; partial JSON is not passed as if it were a proposal.
    """
    response, _ = split_transport(invalid_response)
    feedback = {
        "correction_kind": kind,
        "current_errors": [str(error)[:2_000] for error in errors[:8]],
        "invalid_response_digest": digest(response),
        "parent_operation_id": parent_operation_id,
        "parent_task_ref": parent_task_ref,
    }
    text = canonical(response)
    if len(text) <= max_draft_chars:
        feedback["bounded_invalid_draft"] = response
    else:
        feedback["invalid_draft_ref"] = invalid_draft_ref
        feedback["invalid_draft_chars"] = len(text)
    child = copy.deepcopy(dict(base_payload))
    child["maintenance_correction"] = feedback
    # Long-memory requests use an input_digest; the source/batch digest remains
    # in the parent_operation_id rather than being replaced by a retry epoch.
    if "input_digest" in child:
        child["input_digest"] = digest({"parent_input_digest": base_payload["input_digest"],
                                        "maintenance_correction": feedback})
    child["allow_semantic_compaction"] = False
    child["allow_sync_memory_maintenance"] = False
    return child


def child_operation_id(base_operation_id: str, child: Mapping[str, Any]) -> str:
    return base_operation_id + ":correction:" + digest(child["maintenance_correction"])[:24]


def register_invalid_draft(camp: Path, response: Any) -> str:
    """Make a bulky paid draft available through the existing bounded reader."""
    from .sources import ContextSourceRegistry
    proposal, _ = split_transport(response)
    source = ContextSourceRegistry(camp).capture(proposal,
        block_id="maintenance-invalid:" + digest(proposal), kind="memory",
        role="memory_curator", status="complete")
    return source.source_ref
