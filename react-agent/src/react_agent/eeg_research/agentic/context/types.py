"""Versioned delivery protocol. Delivery is never scientific evidence or approval."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

PROTOCOL_VERSION = "eeg.context.v1"
POLICY_VERSION = "eeg.context.policy.v1"
RENDER_VERSION = "eeg.context.render.v1"


class MaintenanceDeferred(RuntimeError):
    """Native admission proved the maintenance callback sent no new request."""
    api_calls_added = 0
    transport_started = False


def canonical(value: Any) -> str:
    # Matches the repository's compaction digest, including full float precision.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=str, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetConfig:
    hard_chars: int = 160_000
    target_chars: int = 112_000
    high_chars: int = 128_000
    fallback_chars: int = 144_000
    snapshot_target_chars: int = 8_000
    snapshot_max_chars: int = 12_000
    retrieval_chars: int = 8_000
    recent_chars: int = 24_000
    excerpt_chars: int = 8_000
    provider_context_tokens: int | None = None
    provider_input_tokens: int | None = None
    reasoning_reserve_tokens: int = 0
    token_safety_factor: float = 1.25
    max_passes: int = 6
    max_reads: int = 3
    max_read_chars: int = 24_000
    max_manifest_entries: int = 64

    def __post_init__(self) -> None:
        if not 0 < self.target_chars <= self.high_chars <= self.fallback_chars <= self.hard_chars:
            raise ValueError("context watermarks must satisfy 0 < low <= high <= fallback <= hard")
        if self.token_safety_factor < 1 or min(self.max_passes, self.max_reads, self.excerpt_chars) < 1:
            raise ValueError("invalid bounded context configuration")


@dataclass(frozen=True)
class SourceBlock:
    block_id: str
    campaign_id: str
    kind: str
    source_ref: str
    source_revision: str
    source_hash: str
    payload_ref: str
    group_id: str = ""
    event_ids: tuple[str, ...] = ()
    producer: str = "context_capture"
    schema_version: str = PROTOCOL_VERSION
    identity: dict[str, Any] = field(default_factory=dict)
    status: str = "captured"
    visibility: tuple[str, ...] = ("campaign",)
    dependency_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    source_validity: str = "valid"
    task_applicability: str = "unclassified"


@dataclass(frozen=True)
class Representation:
    representation_id: str
    block_id: str
    kind: Literal["FULL", "FACT_CARD", "PARTIAL", "SUMMARY", "REF_ONLY"]
    content: Any
    source_refs: tuple[str, ...]
    source_hashes: tuple[str, ...]
    representation_hash: str
    schema_version: str = PROTOCOL_VERSION
    render_version: str = RENDER_VERSION
    exact_fact_refs: tuple[str, ...] = ()
    selected_ranges: tuple[dict[str, Any], ...] = ()
    retained_sections: tuple[str, ...] = ()
    omitted_sections: tuple[str, ...] = ()
    completeness: str = "complete"
    applicability: str = "unclassified"
    budget_chars: int = 0
    readable: bool = True
    validation_status: str = "deterministic"
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadRequest:
    source_ref: str
    source_hash: str
    json_pointer: str = ""
    start: int | None = None
    end: int | None = None
    unit: str = "json"
    max_chars: int = 8_000
    question: str = ""


@dataclass(frozen=True)
class ReadResult:
    status: str
    content: Any = None
    source_block: SourceBlock | None = None
    delivered_scope: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    # This is intentionally false even when a delivery registry read succeeds.
    satisfies_development_read_receipt: bool = False


@dataclass(frozen=True)
class CompileRequest:
    campaign_id: str
    role: str
    payload: dict[str, Any]
    system: str = ""
    task_id: str = ""
    intent_id: str = ""
    purpose: str = ""
    authority_revision: str = ""
    model: str = "deepseek-flash"
    provider: str = "deepseek"
    max_output_tokens: int = 16_384
    mode: str = "normal"
    identity: dict[str, Any] = field(default_factory=dict)
    policy_version: str = POLICY_VERSION
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    source_blocks: tuple[SourceBlock, ...] = ()
    snapshot: Any = None
    retrieved: tuple[Any, ...] = ()
    feedback: Any = None
    instructions: Any = None
    schema: Any = None
    tools: tuple[Any, ...] = ()
    transport_options: dict[str, Any] = field(default_factory=dict)
    examples: tuple[Any, ...] = ()
    allow_semantic_compaction: bool = True
    allow_sync_memory_maintenance: bool = True
    remaining_api_calls: int | None = None
    maintenance_callback: Callable[..., Any] | None = field(default=None, compare=False, repr=False)
    tokenizer: Callable[[str], Any] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class PreparedRequest:
    system: str
    user: str
    messages: tuple[dict[str, str], ...]
    request_digest: str
    source_dependency_digest: str
    policy_version: str
    render_version: str
    tokenizer_version: str
    budget_report: dict[str, Any]
    delivery_manifest: dict[str, Any]
    dependencies: tuple[dict[str, Any], ...]
    sidecar_refs: dict[str, str]
    transport_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> dict[str, Any]:
        # Prevent a caller's edits from silently mutating the frozen transport.
        return json.loads(self.user)

    def verify(self) -> bool:
        return (self.messages == ({"role": "system", "content": self.system}, {"role": "user", "content": self.user})
                and self.request_digest == digest({"system": self.system, "user": self.user,
                                                  "transport_fields": self.transport_fields}))


@dataclass(frozen=True)
class Ready:
    prepared_request: PreparedRequest
    requirements: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"

    @property
    def prepared(self) -> PreparedRequest:
        return self.prepared_request


@dataclass(frozen=True)
class NeedsContext:
    missing: tuple[dict[str, Any], ...]
    reads: tuple[ReadRequest, ...]
    reason: str = "required_context_not_delivered"
    budget_report: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    status: str = "needs_context"


@dataclass(frozen=True)
class Blocked:
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    budget_report: dict[str, Any] = field(default_factory=dict)
    recoverable_when: tuple[str, ...] = ()
    status: str = "blocked"


CompileResult = Ready | NeedsContext | Blocked
