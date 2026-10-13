"""Bounded working-memory maintenance with an injected, ledgered model caller.

No backend is imported here.  The caller must use the unified compiler with
both semantic-compaction flags disabled; the reentry guard also prevents an
accidental callback cycle from consuming another API attempt.
"""
from __future__ import annotations

import contextvars
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .snapshot import GroupBatch, SnapshotStore
from .types import canonical, digest
from .correction import build_child_request, child_operation_id, register_invalid_draft, split_transport

MAINTENANCE_VERSION = "eeg.context.maintenance.v1"
_ACTIVE = contextvars.ContextVar("eeg_context_maintenance_active", default=False)


def source_family(base_revision: int, sections) -> str:
    """One retry budget for the same actual science and snapshot base.

    Authority/CAS, turn, task, model and prompt still freeze each request, but
    changing them alone cannot erase already consumed generation attempts.
    """
    sources = [{key: section[key] for key in ("section_ref", "source_hash", "section_id", "visibility")
                if key in section} for section in sections]
    return digest({"version": "eeg.context.maintenance.source_family.v1", "base_revision": base_revision,
                   "sections": sorted(sources, key=lambda section: section["section_ref"])})


def draft_source_family(draft) -> str:
    batch = draft["batch"]
    return source_family(batch["base_revision"], batch["sections"])


@dataclass(frozen=True)
class MaintenanceBudget:
    high_chars: int = 128_000
    low_chars: int = 112_000
    fallback_chars: int = 144_000
    input_chars: int = 36_000
    batch_chars: int = 24_000
    min_removable_chars: int = 2_000
    min_benefit_chars: int = 1_000
    max_calls_turn: int = 2
    max_calls_campaign: int = 32
    max_cost_turn: float = 2.0
    max_cost_campaign: float = 32.0
    reserved_cost_per_call: float = 1.0
    max_attempts_per_digest: int = 3
    lease_seconds: float = 360.0
    call_deadline_seconds: float = 600.0
    backoff_seconds: float = 60.0

    def __post_init__(self):
        if (min(self.input_chars, self.batch_chars, self.max_calls_turn, self.max_calls_campaign,
                self.max_attempts_per_digest, self.lease_seconds) <= 0
                or self.call_deadline_seconds <= 0 or not math.isfinite(self.call_deadline_seconds)
                or self.batch_chars > self.input_chars or self.max_calls_turn > 2
                or not 0 <= self.low_chars <= self.high_chars <= self.fallback_chars
                or not all(math.isfinite(value) and value >= 0 for value in
                           (self.max_cost_turn, self.max_cost_campaign, self.reserved_cost_per_call, self.backoff_seconds))):
            raise ValueError("invalid bounded maintenance budget")


@dataclass(frozen=True)
class MaintenanceResult:
    status: str
    snapshot: dict[str, Any]
    input_digest: str = ""
    reason: str = ""
    calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    effective_before_chars: int = 0
    effective_after_chars: int | None = None
    elapsed_seconds: float = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "published"


class MaintenanceCoordinator:
    def __init__(self, store: SnapshotStore, generator: Callable[..., Mapping[str, Any]] | None = None,
                 *, budget: MaintenanceBudget | None = None,
                 benefit_evaluator: Callable[[Any, GroupBatch], int] | None = None,
                 generation_identity: Mapping[str, Any] | None = None):
        self.store = store
        self.generator = generator
        self.budget = budget or MaintenanceBudget()
        # Actual compile/packing before-after evaluation belongs to the caller.
        # Absence of that evaluator permits independent snapshot maintenance,
        # but never claims a reduction in an actual role request.
        self.benefit_evaluator = benefit_evaluator
        self.generation_identity = dict(generation_identity or {})

    def maintain(self, *, turn_id: str, effective_chars: int = 0,
                 provider_high: bool = False, emergency: bool = False,
                 idle: bool = False, authoritative_revision: str = "",
                 remaining_api_calls: int | None = None,
                 removable_chars: int | None = None,
                 budget: MaintenanceBudget | None = None,
                 task_id: str = "", intent_id: str = "", role: str = "",
                 purpose: str = "", identity: Mapping[str, Any] | None = None) -> MaintenanceResult:
        policy = budget or self.budget
        start = time.monotonic()
        previous = self.store.snapshot_delivery(role=role, identity=identity)
        base_result = {"snapshot": previous, "effective_before_chars": effective_chars}
        if _ACTIVE.get():
            return MaintenanceResult("blocked", **base_result, reason="memory_maintenance_reentry")
        if authoritative_revision:
            self.store.set_authority_revision(authoritative_revision)
        if not self.generator:
            return MaintenanceResult("unavailable", **base_result, reason="ledgered_generator_not_configured")
        if remaining_api_calls is not None and remaining_api_calls <= 0:
            return MaintenanceResult("budget_exhausted", **base_result, reason="native_api_budget_exhausted")
        if not (idle or emergency or provider_high or effective_chars >= policy.high_chars):
            return MaintenanceResult("below_high_watermark", **base_result)
        batch = self.store.pending_groups(max_chars=policy.batch_chars, role=role)
        if not batch.sections:
            return MaintenanceResult("no_op", **base_result, reason="no_new_closed_source_sections")
        input_payload = batch.callback_payload(previous)
        input_size = len(canonical(input_payload))
        # The input has its own cap. Smaller deterministic batches do not invoke
        # a recursive map/reduce model to fit the compactor's context.
        if input_size > policy.input_chars:
            available = policy.input_chars - len(canonical(batch.callback_payload({}))) + batch.input_chars - len(canonical(previous)) - 256
            if available < 1_024:
                return MaintenanceResult("blocked", **base_result, reason="maintenance_kernel_too_large")
            batch = self.store.pending_groups(max_chars=min(policy.batch_chars, available), role=role)
            input_payload = batch.callback_payload(previous)
            input_size = len(canonical(input_payload))
            if not batch.sections or input_size > policy.input_chars:
                return MaintenanceResult("blocked", **base_result, reason="maintenance_input_too_large")
        if removable_chars is not None and removable_chars < policy.min_removable_chars:
            return MaintenanceResult("no_op", **base_result, input_digest=batch.input_dependency_digest,
                                     reason="insufficient_effective_request_benefit")
        job_digest = digest({"dependency": batch.input_dependency_digest,
                             "policy": MAINTENANCE_VERSION, "budget": dataclass_budget(policy),
                             "task": {"role": role, "task_id": task_id, "intent_id": intent_id,
                                      "purpose": purpose, "identity": dict(identity or {})},
                             "generation_identity": self.generation_identity,
                             "visibility": role or "campaign"})
        owner = uuid.uuid4().hex
        family = source_family(batch.base_revision, batch.sections)
        legacy = self.store.persistence.legacy_source_family(family, draft_source_family)
        claimed = self.store.persistence.claim(job_digest, owner=owner, turn_id=turn_id,
                                               lease_seconds=policy.lease_seconds,
                                               max_attempts=policy.max_attempts_per_digest,
                                               input_chars=input_size, source_family=family,
                                               legacy_family_digests=legacy)
        if claimed != "claimed":
            return MaintenanceResult(claimed, **base_result, input_digest=job_digest)
        token = _ACTIVE.set(True)
        calls, output_size, feedback, response = 0, 0, None, None
        cache_adoptions, validations, transport_meta = 0, 0, {}
        parent_operation_id = job_digest + ":generate"
        request_payload = input_payload
        last_error = ""
        try:
            # One generation and one mechanical correction. Backend network
            # retries stay inside the native caller's frozen-request policy.
            for correction in (False, True):
                if correction and not feedback:
                    break
                if correction:
                    draft_ref = self.store.persistence.immutable("maintenance_responses", response)
                    if len(canonical(response)) > 3_000:
                        draft_ref = register_invalid_draft(self.store.camp, response)
                    request_payload = build_child_request(input_payload, response, feedback["current_errors"],
                        parent_operation_id=parent_operation_id, kind="research_delta_publication",
                        parent_task_ref=str(transport_meta.get("task_ref") or ""), invalid_draft_ref=draft_ref)
                    if len(canonical(request_payload)) > policy.input_chars:
                        last_error = "maintenance_correction_input_too_large"
                        break
                if remaining_api_calls is not None and calls >= remaining_api_calls:
                    last_error = "native_api_budget_exhausted"
                    break
                if not self.store.persistence.charge(
                        turn_id=turn_id, max_calls_turn=policy.max_calls_turn,
                        max_calls_campaign=policy.max_calls_campaign,
                        estimated_cost=policy.reserved_cost_per_call,
                        max_cost_turn=policy.max_cost_turn, max_cost_campaign=policy.max_cost_campaign):
                    last_error = "maintenance_call_or_cost_budget_exhausted"
                    break
                calls += 1
                operation_id = child_operation_id(job_digest, request_payload) if correction else parent_operation_id
                invocation_started = False
                try:
                    if (self.store._authority() != batch.authority_revision
                            or not all(self.store.validate_source_ref(section, role=role) for section in batch.sections)
                            or not self.store.persistence.renew(job_digest, owner=owner,
                                lease_seconds=max(policy.lease_seconds, policy.call_deadline_seconds),
                                epoch_seconds=2 * policy.call_deadline_seconds + policy.lease_seconds,
                                in_flight=True, authority_revision=batch.authority_revision,
                                base_revision=batch.base_revision)):
                        raise ValueError("maintenance_source_or_lease_changed")
                    invocation_started = True
                    raw_response = self.generator(request_payload, operation_id=operation_id,
                                              mode="summary", turn_id=turn_id,
                                              correction=correction, feedback=feedback,
                                              correction_context=request_payload.get("maintenance_correction"),
                                              allow_semantic_compaction=False, allow_sync_memory_maintenance=False)
                    response, transport_meta = split_transport(raw_response)
                    if transport_meta.get("cache_adopted") and transport_meta.get("api_calls", 0) == 0:
                        cache_adoptions += 1
                        calls -= 1
                        self.store.persistence.release_cached_reservation(turn_id=turn_id, reserved=policy.reserved_cost_per_call)
                    if not self.store.persistence.renew(job_digest, owner=owner, lease_seconds=policy.lease_seconds,
                            epoch_seconds=2 * policy.call_deadline_seconds + policy.lease_seconds,
                            in_flight=False, authority_revision=batch.authority_revision,
                            base_revision=batch.base_revision):
                        raise ValueError("maintenance_lease_lost")
                    validations += 1
                    if not isinstance(response, Mapping):
                        raise ValueError("maintenance_response_not_mapping")
                    output_size = len(canonical(response))
                    # Optional usage is a sidecar, never part of the delta.
                    proposed = response.get("delta", response)
                    if len(canonical(proposed)) > 12_000:
                        raise ValueError("maintenance_output_budget_exceeded")
                    # A model proposal is not a billing authority. Keep the
                    # conservative local reservation; actual provider usage
                    # and price settlement come from the native API ledger.
                    proposal = self.store.propose_delta(proposed, batch,
                                                       expected_authoritative_revision=batch.authority_revision,
                                                       generation={"identity": self.generation_identity,
                                                                   "budget": dataclass_budget(policy),
                                                                   "operation_id": operation_id,
                                                                   "correction": correction})
                    if not proposed.get("no_signal") and self.benefit_evaluator is not None:
                        after = int(self.benefit_evaluator(proposal, batch))
                        if after > effective_chars - policy.min_benefit_chars:
                            self.store.persistence.finish_job(job_digest, owner=owner, status="no_op",
                                                              error="no_effective_request_benefit", output_chars=output_size)
                            return MaintenanceResult("no_op", **base_result, input_digest=job_digest,
                                                     reason="no_effective_request_benefit", calls=calls,
                                                     input_chars=input_size, output_chars=output_size,
                                                     effective_after_chars=after, elapsed_seconds=time.monotonic() - start)
                    outcome = self.store.publish(proposal, maintenance_job=(job_digest, owner))
                    if outcome.status in {"published", "no_signal"}:
                        return MaintenanceResult(outcome.status, self.store.snapshot_delivery(role=role, identity=identity), job_digest,
                                                 calls=calls, input_chars=input_size, output_chars=output_size,
                                                 effective_before_chars=effective_chars,
                                                 elapsed_seconds=time.monotonic() - start,
                                                 metrics={"processed_sections": outcome.processed_count,
                                                          "abstracted_sections": outcome.abstracted_count,
                                                          "cache_adoptions": cache_adoptions,
                                                          "local_validations": validations,
                                                          "real_request_benefit_checked": self.benefit_evaluator is not None})
                    self.store.persistence.finish_job(job_digest, owner=owner, status="stale",
                                                      error=outcome.reason, output_chars=output_size,
                                                      backoff_seconds=policy.backoff_seconds)
                    return MaintenanceResult(outcome.status, **base_result, input_digest=job_digest,
                                             reason=outcome.reason, calls=calls, input_chars=input_size,
                                             output_chars=output_size, elapsed_seconds=time.monotonic() - start)
                except (ValueError, KeyError, OSError, PermissionError) as exc:
                    if not invocation_started:
                        calls -= 1
                        self.store.persistence.release_cached_reservation(turn_id=turn_id, reserved=policy.reserved_cost_per_call)
                    last_error = str(exc)[:2_000]
                    # A mechanical error set only, never the accumulating bad
                    # response history. Full drafts have already been persisted.
                    feedback = {"current_errors": [last_error], "invalid_response_digest": digest(response) if response is not None else ""}
                    self.store.persistence.immutable("maintenance_validation_failures", {
                        "operation_id": operation_id, "correction": correction,
                        "response_digest": digest(response), "current_errors": [last_error]})
                    if any(word in last_error for word in ("budget_exceeded", "input_too_large", "source_hash", "stale_source", "revoked", "lease", "source_or_lease_changed")):
                        break
                except Exception as exc:
                    if not invocation_started:
                        calls -= 1
                        self.store.persistence.release_cached_reservation(turn_id=turn_id, reserved=policy.reserved_cost_per_call)
                    # Runtime/network failure is not schema correction.
                    last_error = type(exc).__name__ + ":" + str(exc)[:1_900]
                    break
            status = "budget_exhausted" if "budget_exhausted" in last_error else "failed"
            self.store.persistence.finish_job(job_digest, owner=owner, status=status,
                                              error=last_error, output_chars=output_size,
                                              backoff_seconds=policy.backoff_seconds)
            return MaintenanceResult(status, **base_result, input_digest=job_digest,
                                     reason=last_error, calls=calls, input_chars=input_size,
                                     output_chars=output_size, elapsed_seconds=time.monotonic() - start)
        finally:
            _ACTIVE.reset(token)


def dataclass_budget(policy: MaintenanceBudget) -> dict[str, Any]:
    # Every generation-budget tier participates in reuse identity; changing the
    # budget cannot silently reuse a longer old summary or exhausted task epoch.
    return {name: getattr(policy, name) for name in policy.__dataclass_fields__}


def research_delta_prompt() -> str:
    """Small versioned policy plus a strict provider schema for ledgered callers."""
    from .snapshot import ResearchDelta
    return """Maintain a campaign's incremental working research memory.
Return one ResearchDelta JSON object. The previous snapshot is navigation,
never new scientific evidence. Use only the frozen closed_sections supplied.
For each item cite actual section_ref values; do not declare arbitrary coverage.
Write short qualitative explanations, conditions and questions. Native source
identities, exact metrics, authorization, resources, gates and task states are
immutable. Cite every numeric statement through a facts path/value from that
exact delivered section. Do not infer fold IDs from array positions.
Two-fold development screening, engineering failure, partial/unverified output
and complete ten-fold confirmation are different scopes. Source completeness
does not promote scientific validity; full fidelity does not imply ten folds.
Keep contradictions and applicable conditions; newer does not imply truer.
Stable item IDs identify revisions. Retire/resolve only with relevant terminal
source evidence, never to remove an inconvenient open question. Pending async
work is excluded from these closed sections and remains in native authority.
Use action=update for an item_id already in previous_snapshot; add is for a new
item ID. Numeric facts must copy a path/value from that section's exact_facts
list. A partial text fragment with exact_facts=[] supports qualitative notes,
not new numeric fact citations. Omit already known findings with no real change.
Do not restate every old item. Add/update only real changes. With no useful
signal return no_signal=true and no patches/recent_arc. Empty false-success is
invalid. Never treat summary delivery as artifact read or execution approval.
No recursive summarization/curation, tools, or follow-up model calls.
OUTPUT_SCHEMA
""" + canonical(ResearchDelta.model_json_schema())
