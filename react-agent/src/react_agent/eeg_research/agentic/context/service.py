"""Connect the delivery protocol to native EEG sources and ledgered callbacks.

The service does not import a network backend. Its caller injects the one
native, billed maintenance adapter, which uses this same compiler in a mode
that prohibits recursive maintenance.
"""
from __future__ import annotations

import contextvars
import copy
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from .compiler import (ContextCompiler, is_research_planning_input, is_designer_scientific_input,
                       is_engineering_input, is_analysis_scientific_input)
from .maintenance import MaintenanceBudget, MaintenanceCoordinator
from .long_memory import LongMemoryPipeline, valid_source_ref
from .recovery import incorporate_partial_reads, recover_context
from .representations import NARRATIVE_FIELDS, handle, representations_for, source_kind
from .snapshot import SnapshotStore, _existing_items, _changed_ids, _item_rows
from .sources import ContextSourceRegistry, pointer_escape
from .types import Blocked, BudgetConfig, CompileRequest, MaintenanceDeferred, NeedsContext, Ready, canonical, digest

_TURN = contextvars.ContextVar("eeg_context_turn", default="")
_ACTIVE = contextvars.ContextVar("eeg_context_service_maintenance", default=False)
MAINTENANCE_MODES = frozenset({"summary", "extraction", "consolidation"})
WRAPPERS = frozenset({"observation", "input", "research_context", "context", "payload"})
CLOSED_COLLECTIONS = frozenset({"evidence", "method_screenings", "method_suites", "run_artifacts", "analyses"})
DECOMPOSABLE_COLLECTIONS = CLOSED_COLLECTIONS | {"candidates", "experiments", "comparisons", "completed_tasks"}
SCIENCE_IDENTITY = ("candidate_id", "method_revision", "source_hash", "config_hash", "protocol_hash",
    "evaluation_key", "evaluator_hash", "evaluator", "seed", "scope", "science_scope", "fold_id",
    "held_out_subject", "selection_manifest_hash", "fidelity", "evidence_id", "suite_id", "job_id", "run_id", "artifact_id")


def context_mode() -> str:
    value = os.environ.get("EEG_CONTEXT_PROTOCOL", "enforce").strip().lower()
    if value not in {"enforce", "shadow"}:
        raise ValueError("context_policy_missing: use enforce or shadow; no unbounded FULL fallback")
    return value


def budget_for_role(role: str) -> BudgetConfig:
    from react_agent.eeg_research.agentic.handoffs import ROLE_CONTEXT_BUDGETS
    hard = min(160_000, ROLE_CONTEXT_BUDGETS.get(role, 160_000))
    return BudgetConfig(hard_chars=hard, target_chars=hard * 7 // 10,
                        high_chars=hard * 8 // 10, fallback_chars=hard * 9 // 10)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def native_authority(camp: Path, *, owned_calls: set[str] | None = None) -> dict:
    """Use current native truth, accounting only for this upkeep's own calls."""
    state = _read(camp / "campaign_state.json")
    goal = _read(camp / "goal.json")
    if (camp / "goal.json").is_file():
        from react_agent.eeg_research.agentic.native_authorization import effective_goal, apply_authorized_scope
        goal = effective_goal(camp)
        apply_authorized_scope(camp, state)
    cost = _read(camp / "cost.json")
    controls = {key: state.get(key) for key in (
        "status", "pause_after_step", "live_job", "gpu", "max_candidates", "max_llm_calls",
        "max_training_jobs", "max_gpu_seconds", "gpu_seconds_reserved", "allowed_training_actions",
        "candidate_id", "candidate_ready", "candidate_admissions", "detached_candidates",
        "active_suite", "phase", "repair_task", "execution_fingerprint", "contract_fingerprint")}
    owned = owned_calls or set()
    ledger_path = camp / "llm_calls.jsonl"
    accounted = set()
    settled = {}
    if owned and ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("call_id"):
                if row["call_id"] in settled and settled[row["call_id"]] != row:
                    raise ValueError("context_cost_ledger_conflict")
                settled[row["call_id"]] = row
            if row.get("call_id") in owned and row.get("transport_outcome") == "confirmed_response":
                accounted.add(row["call_id"])
    dollars = cost.get("api_usd")
    known = [row["api_usd"] for row in settled.values() if type(row.get("api_usd")) in (int, float)]
    if accounted and type(dollars) in (int, float) and len(known) == len(settled) and sum(known) == dollars:
        # Recompute with the ledger's original ordering, instead of subtracting
        # floats and introducing a fictitious resource change through rounding.
        external = [row["api_usd"] for key, row in settled.items() if key not in accounted]
        dollars = sum(external) if external else None
    # The generator's own confirmed bookkeeping is an expected mutation;
    # external calls, changes to jobs/GPU/goal/control remain in the CAS stamp.
    resources = {"llm_calls": int(cost.get("llm_calls", state.get("llm_calls", 0))) - len(accounted),
                 "training_jobs": cost.get("training_jobs", state.get("training_jobs", 0)),
                 "gpu_seconds_used": cost.get("gpu_seconds_used", 0),
                 "api_unsettled_call_ids": cost.get("api_unsettled_call_ids", []),
                 "api_abandoned_call_ids": cost.get("api_abandoned_call_ids", []),
                 "api_usd": dollars}
    return {"goal": goal, "controls": controls, "resources": resources,
            "control_request": _read(camp / "control.json")}


def authoritative_revision(camp: Path, owned_calls: set[str] | None = None) -> str:
    return digest(native_authority(camp, owned_calls=owned_calls))


def _closed(row: dict) -> bool:
    if row.get("status") in {"pending", "running", "started", "training", "active", "revoked", "deleted"}:
        return False
    return (row.get("status") in {"completed", "complete", "failed", "cancelled", "finished", "valid"}
            or row.get("evaluation_valid") is True or row.get("training_complete") is True)


def _collections(payload: dict, path: tuple[str, ...] = ()):
    for key, value in payload.items():
        if key in WRAPPERS and isinstance(value, dict):
            yield from _collections(value, path + (key,))
        elif key in DECOMPOSABLE_COLLECTIONS and isinstance(value, list):
            yield path + (key,), value


def _row_id(row: dict) -> str | None:
    return next((str(row[key]) for key in ("evidence_id", "suite_id", "job_id", "run_id", "artifact_id", "task_id")
                 if row.get(key)), None)


def _identity(row: dict) -> dict:
    identity = {key: row[key] for key in SCIENCE_IDENTITY if key in row}
    scientific = row.get("screening") or row.get("aggregate")
    if isinstance(scientific, dict):
        for key in SCIENCE_IDENTITY:
            if key in scientific:
                if key in identity and canonical(identity[key]) != canonical(scientific[key]):
                    raise ValueError("closed_source_identity_conflict:" + key)
                identity[key] = scientific[key]
    return identity


class _ProposalView:
    """A read-only dry view; only real SnapshotStore publication changes coverage."""
    def __init__(self, store: SnapshotStore, proposal, batch):
        self.store, self.batch = store, batch
        draft = store.persistence.read(proposal.artifact_ref)
        delta = store.validate_delta(draft["delta"], batch)
        with store.persistence.db() as db:
            existing = _existing_items(db, _changed_ids(delta))
            items = {item["item_id"]: item for item in _item_rows(db, role=batch.role)}
        changed, self.covered = store._merge(delta, {item["section_ref"]: item for item in batch.sections},
                                            existing, batch.base_revision + 1)
        current = store.snapshot_delivery(role=batch.role)
        count = current.get("delivery", {}).get("item_count", len(items))
        count += sum(int(item.get("active", True) and item.get("valid", True))
                     - int(existing.get(item_id, {}).get("active", False) and existing.get(item_id, {}).get("valid", False))
                     for item_id, item in changed.items())
        items.update(changed)
        current.update(revision=batch.base_revision + 1, base_revision=batch.base_revision,
                       source_dependency_digest=batch.input_dependency_digest,
                       generation={"identity_digest": digest(draft["generation"]), "metadata_ref": proposal.artifact_ref},
                       recent_arc=delta["recent_arc"] or current.get("recent_arc", ""),
                       recent_arc_sources=[{"source_ref": item["source_ref"], "source_hash": item["source_hash"]}
                                           for item in batch.sections] if delta["recent_arc"] else current.get("recent_arc_sources", []),
                       visibility=["campaign"])
        self.view = store._bounded_view(current, [item for item in items.values() if item.get("active") and item.get("valid")], 8_000, total_count=count)

    def snapshot_delivery(self, **kwargs):
        return copy.deepcopy(self.view)

    def is_abstracted(self, block_id: str, source_hash: str, section: str = "") -> bool:
        if self.store.is_abstracted(block_id, source_hash, section):
            return True
        if any(item["block_id"] == block_id and item["source_hash"] == source_hash and item["section_id"] == section
               and item["section_ref"] in self.covered for item in self.batch.sections):
            return True
        with self.store.persistence.db() as db:
            alias = db.execute("SELECT original_source_ref,original_source_hash,original_section_id FROM context_mirror_sections WHERE mirror_block_id=? AND mirror_source_hash=? AND mirror_section_id=?",
                               (block_id, source_hash, section)).fetchone()
        return bool(alias and any(item["source_ref"] == alias[0] and item["source_hash"] == alias[1]
                                 and item["section_id"] == alias[2] and item["section_ref"] in self.covered
                                 for item in self.batch.sections))


class ContextService:
    """Compile every role, with finite upkeep and a real bounded read loop."""
    def __init__(self, camp: Path | str, *, maintenance_callback: Callable | None = None):
        self.camp = Path(camp).resolve()
        self.compiler = ContextCompiler(self.camp)
        self.registry = ContextSourceRegistry(self.camp)
        self.owned_calls: set[str] = set()
        self.callback = maintenance_callback
        self.snapshot = SnapshotStore(self.camp, source_registry=self.registry,
            authority_provider=lambda: authoritative_revision(self.camp, self.owned_calls))
        self.pipeline = None
        if (self.camp / "goal.json").is_file() and (self.camp / "evaluation_contract.json").is_file():
            try:
                self.pipeline = LongMemoryPipeline(self.camp, source_validator=self.source_valid)
            except ValueError as exc:
                if str(exc) != "not_a_verified_eeg_campaign":
                    raise
        with self.snapshot.persistence.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS context_service_scan (field TEXT PRIMARY KEY,last_key TEXT NOT NULL)")
        self._native_blocks: dict[str, Any] = {}

    def source_valid(self, ref: dict) -> bool:
        source_ref = str(ref.get("source_ref") or ref.get("source_id") or "")
        if source_ref.startswith("context://"):
            # No fallback after revocation or a stale current head.
            return self.snapshot.validate_source_ref({**ref, "source_ref": source_ref}, role="memory_curator")
        return valid_source_ref(self.camp, ref)

    def turn_id(self, request: CompileRequest) -> str:
        if _TURN.get():
            return _TURN.get()
        state = _read(self.camp / "campaign_state.json")
        return str(request.identity.get("turn_id") or request.task_id or
                   digest({"role": request.role, "phase": state.get("status"),
                           "decisions": len(state.get("decisions") or []),
                           "authority": request.authority_revision}))

    def capture_closed(self, payload: dict) -> list[dict]:
        """Capture bounded closed producers, never active/pending operations."""
        groups = []
        self._native_blocks = {}
        rows = []
        collections = [(path, value) for path, value in _collections(payload)
                       if path[-1] in CLOSED_COLLECTIONS and value]
        if len(collections) > 128:
            with self.snapshot.persistence.db() as db:
                cursor = db.execute("SELECT last_key FROM context_service_scan WHERE field='__collection_frontier__'").fetchone()
            after = cursor[0] if cursor else ""
            collections.sort(key=lambda pair: "/".join(pair[0]))
            collections = [pair for pair in collections if "/".join(pair[0]) > after] + [pair for pair in collections if "/".join(pair[0]) <= after]
            collections = collections[:128]
            with self.snapshot.persistence.db() as db:
                db.execute("INSERT INTO context_service_scan VALUES('__collection_frontier__',?) ON CONFLICT(field) DO UPDATE SET last_key=excluded.last_key",
                           ("/".join(collections[-1][0]),))
        # Reserve a fair bounded share before advancing any cursor. Every chosen
        # row is actually considered; a later truncation must never lose a field.
        share = min(64, max(1, 128 // max(1, len(collections))))
        for path, value in collections:
            field = path[-1]
            if field not in CLOSED_COLLECTIONS:
                continue
            ordered = sorted(((_row_id(row) or digest(row), row) for row in value if isinstance(row, dict)), key=lambda item: item[0])
            with self.snapshot.persistence.db() as db:
                cursor = db.execute("SELECT last_key FROM context_service_scan WHERE field=?", ("/".join(path),)).fetchone()
            after = cursor[0] if cursor else ""
            ordered = [item for item in ordered if item[0] > after] + [item for item in ordered if item[0] <= after]
            picked = ordered[:share]
            if picked:
                with self.snapshot.persistence.db() as db:
                    db.execute("INSERT INTO context_service_scan VALUES(?,?) ON CONFLICT(field) DO UPDATE SET last_key=excluded.last_key", ("/".join(path), picked[-1][0]))
            rows.extend((field, row) for _, row in picked)
        seen = set()
        for field, row in rows:
            if not _closed(row):
                continue
            key = _row_id(row) or digest(row)
            operation = "closed:" + field + ":" + key
            if operation in seen:
                continue
            seen.add(operation)
            try:
                identity = _identity(row)
                identity["operation_id"] = operation
                block = self.registry.capture(row, block_id=operation,
                    kind=source_kind(field, row), group_id=operation, status=row.get("status", "complete"),
                    role="memory_curator", identity=identity, revision=digest(row))
                self.snapshot.capture_closed_group(operation, [block], status="complete",
                    authoritative_revision=authoritative_revision(self.camp, self.owned_calls), role="memory_curator")
            except (ValueError, PermissionError, KeyError):
                # Revocation does not resurrect a closed source. The compiler
                # validates current required inputs independently below.
                continue
            self._native_blocks[key] = block
            choices, _ = representations_for(block, row, role="memory_curator")
            card = next((rep.content for rep in choices if rep.kind == "FACT_CARD"), row)
            scientific = row.get("screening") or row.get("aggregate") or row
            scope = {key: scientific.get(key, row.get(key)) for key in ("scope", "candidate_id",
                "method_revision", "seed", "fidelity", "coverage", "required_fold_count",
                "full_required_fold_count", "selection_manifest_hash", "screening_subjects",
                "protocol_hash", "evaluation_key", "evaluator_hash") if key in scientific or key in row}
            groups.append({"operation_id": operation, "campaign_id": self.camp.name, "status": "complete",
                "source_refs": [{"source_id": block.source_ref, "source_ref": block.source_ref,
                    "source_hash": block.source_hash, "source_revision": block.source_revision,
                    "visibility": ["campaign"], "path": block.payload_ref}],
                "action": row.get("kind", field), "outcome": row.get("status", "complete"),
                "scientific_scope": scope,
                "engineering_failure": row.get("error") or row.get("failures"),
                "fact_card": card, "excerpts": []})
        if self.pipeline is not None and groups:
            self.pipeline.capture_closed(groups)
        return groups

    def _bind_mirrors(self, payload: dict, *, role: str) -> int:
        """Link explicit same-event narrative mirrors, never chance equal values."""
        bound, scanned = 0, 0
        def leaves(value, pointer="", count=None):
            count = count if count is not None else [0]
            if count[0] >= 256:
                return
            count[0] += 1
            if isinstance(value, dict):
                for key, child in value.items():
                    path = pointer + "/" + pointer_escape(str(key))
                    if key in NARRATIVE_FIELDS and isinstance(child, str):
                        yield path, child
                    elif isinstance(child, dict):
                        yield from leaves(child, path, count)
        def visit(value, path=()):
            nonlocal bound, scanned
            if scanned >= 32 or not isinstance(value, dict):
                return
            for key, row in value.items():
                if key in WRAPPERS and isinstance(row, dict):
                    visit(row, path + (key,)); continue
                if key not in {"controller_state", "analysis", "latest_analysis", "research_progress"} or not isinstance(row, dict):
                    continue
                scanned += 1
                origin = next((str(row[field]) for field in ("evidence_id", "analysis_evidence_id", "source_evidence_id", "artifact_id") if row.get(field)), None)
                if not origin:
                    continue
                original = self._native_blocks.get(origin)
                if original is None:
                    with self.registry._db() as db:
                        refs = db.execute("SELECT source_ref FROM context_source_heads WHERE campaign_id=? AND block_id IN (" + ",".join("?" for _ in CLOSED_COLLECTIONS) + ")",
                                          (self.camp.name, *("closed:" + field + ":" + origin for field in sorted(CLOSED_COLLECTIONS)))).fetchall()
                    for (ref,) in refs:
                        try:
                            original = self.registry.get(ref, role=role); break
                        except (KeyError, ValueError, PermissionError):
                            continue
                if original is None:
                    continue
                try:
                    raw = self.registry.load(original, role=role)
                    original_leaves = list(leaves(raw))[:32]
                    identity = {**original.identity, **_identity(row)}
                    identity.setdefault("evidence_id", origin)
                    pointer = "/" + "/".join(pointer_escape(part) for part in path + (key,))
                    mirror = self.registry.capture(row, block_id="input:" + pointer,
                        kind=source_kind(key, row), role=role, identity=identity,
                        dependency_refs=(original.source_ref,), revision=digest(row))
                    with self.snapshot.persistence.db() as db:
                        sections = [(r[0], json.loads(r[1])) for r in db.execute(
                            "SELECT section_id,section_json FROM context_closed_sections WHERE source_ref=? AND source_hash=?",
                            (original.source_ref, original.source_hash))]
                    mapping = {}
                    for mirror_pointer, text in list(leaves(row))[:16]:
                        matching = [(p, t) for p, t in original_leaves if t == text]
                        if len(matching) != 1:
                            continue
                        original_pointer = matching[0][0]
                        parents = [(sid, section) for sid, section in sections if section["complete_section"] and
                                   (section["json_pointer"] == "" or original_pointer == section["json_pointer"] or original_pointer.startswith(section["json_pointer"] + "/"))]
                        if len(parents) != 1:
                            continue
                        mapping[mirror_pointer] = {"section_id": parents[0][0], "original_pointer": original_pointer}
                    if mapping:
                        bound += self.snapshot.bind_mirror(mirror, original, mapping, role=role)["mapped"]
                except (KeyError, ValueError, PermissionError, OSError):
                    # Unclassified/ambiguous/stale mirrors remain ordinary input.
                    continue
        visit(payload)
        return bound

    def _benefit(self, request, proposal, batch) -> int:
        provisional = _ProposalView(self.snapshot, proposal, batch)
        # A count-only compile uses its own ephemeral registry. It cannot move
        # current source heads or publish coverage in the real campaign.
        with tempfile.TemporaryDirectory(prefix="eeg-context-benefit-") as temporary:
            compiler = ContextCompiler(Path(temporary))
            candidate = replace(request, snapshot=provisional, source_blocks=(),
                maintenance_callback=None, allow_semantic_compaction=False,
                allow_sync_memory_maintenance=False)
            result = compiler.compile(candidate)
        if isinstance(result, Ready):
            return result.prepared.budget_report["chars"]
        return int(result.budget_report.get("chars", request.budget.hard_chars + 1))

    def _call_maintenance(self, request: dict, **kwargs) -> dict:
        if self.callback is None:
            raise RuntimeError("maintenance_model_unavailable")
        if request.get("allow_semantic_compaction") is not False or request.get("allow_sync_memory_maintenance") is not False:
            # Always explicitly set both guards at the actual model boundary.
            request = {**request, "allow_semantic_compaction": False,
                       "allow_sync_memory_maintenance": False}
        from ..research_execution import paid_maintenance_mode
        mode = paid_maintenance_mode(self.camp)
        if mode == "off":
            # The native caller can adopt an exact already paid response; it
            # must reject a cache miss before creating a transport intent.
            kwargs = {**kwargs, "recovery_only": True, "maintenance_policy": "off"}
        elif mode == "batch":
            from ..owner_budget import auxiliary_owner
            with auxiliary_owner(self.camp, purpose="memory_maintenance"):
                return self._maintenance_callback(request, **{**kwargs, "maintenance_policy": "batch"})
        elif mode == "emergency":
            from ..owner_budget import current_owner
            owner = current_owner(self.camp)
            if not owner or owner.get("kind") != "role_preparation":
                raise MaintenanceDeferred("maintenance_deferred_no_role_preparation_owner")
            kwargs = {**kwargs, "maintenance_policy": "emergency"}
        return self._maintenance_callback(request, **kwargs)

    def _maintenance_callback(self, request, **kwargs):
        """Keep a zero-transport admission rejection distinct from paid failure."""
        def journal_ids():
            ids = set()
            for name in ("llm_api_intents.jsonl", "api_operation_attempts.jsonl", "llm_calls.jsonl"):
                path = self.camp / name
                if path.is_file():
                    ids.update(json.loads(line).get("call_id") for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            return ids - {None}
        before = journal_ids()
        try:
            return self.callback(request, **kwargs)
        except Exception as exc:
            rejected = str(exc).startswith(("auxiliary_request_deferred:", "owner_request_unavailable:",
                "emergency_requires_role_preparation_owner", "batch_requires_unreserved_auxiliary",
                "paid_maintenance_off_deferred", "paid_maintenance_recovery_cache_miss_deferred"))
            if rejected and journal_ids() == before:
                raise MaintenanceDeferred(str(exc)) from exc
            raise

    def _background_maintenance(self, base_turn_id: str):
        """Apply only an explicitly authorized closed-turn recovery scope."""
        from react_agent.eeg_research.agentic.native_authorization import authorized_maintenance_turn
        authorized = authorized_maintenance_turn(self.camp, base_turn_id)
        if authorized is None:
            return base_turn_id, self._call_maintenance
        recovery_turn_id, authorization_ref = authorized

        def callback(request, **kwargs):
            from react_agent.eeg_research.agentic.api_policy import authorized_memory_api_scope
            with authorized_memory_api_scope(self.camp, base_turn_id=base_turn_id,
                    recovery_turn_id=recovery_turn_id, authorization_ref=authorization_ref):
                return self._call_maintenance(request, **{**kwargs,
                    "original_maintenance_turn_id": base_turn_id,
                    "maintenance_authorization_ref": authorization_ref})

        return recovery_turn_id, callback

    def _upkeep_api_allowance(self, request) -> int | None:
        remaining = request.remaining_api_calls
        if request.role not in {"research_planner", "research_critic"}:
            return remaining
        from react_agent.eeg_research.agentic.api_policy import ATTEMPTS, ROUND, api_request_policy
        round_ = ROUND.get()
        if round_ is None or round_[0] != str(self.camp.resolve()):
            return remaining
        policy = api_request_policy(self.camp)
        path = self.camp / ATTEMPTS
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
        used = sum(row.get("planner_round_id") == round_[1] for row in rows)
        round_remaining = max(0, min(round_[2], policy.max_attempts_per_planner_round) - used)
        # The paid Planner request and its existing one-correction allowance
        # take priority over optional inline summary work. Native reservation
        # still atomically rechecks every actual call against the same round.
        allowance = max(0, round_remaining - policy.max_attempts_per_operation)
        return allowance if remaining is None else min(remaining, allowance)

    def _upkeep(self, *, request, blocks, effective_payload, budget_report):
        if self.callback is None or _ACTIVE.get():
            return {"status": "unavailable"}
        from ..research_execution import paid_maintenance_mode
        mode = paid_maintenance_mode(self.camp)
        if mode in {"off", "batch"}:
            return {"status": "deferred", "reason": "paid_maintenance_" + mode, "calls": 0}
        if mode == "emergency" and budget_report.get("ready_budget_ok"):
            return {"status": "deferred", "reason": "necessary_context_already_deliverable", "calls": 0}
        if mode == "emergency" and budget_report.get("chars", 0) <= request.budget.high_chars and budget_report.get("provider_budget_ok"):
            return {"status": "deferred", "reason": "no_necessary_context_emergency", "calls": 0}
        token = _ACTIVE.set(True)
        try:
            coordinator = MaintenanceCoordinator(self.snapshot, generator=self._call_maintenance,
                budget=MaintenanceBudget(low_chars=request.budget.target_chars,
                                         high_chars=request.budget.high_chars,
                                         fallback_chars=request.budget.fallback_chars),
                benefit_evaluator=lambda proposal, batch: self._benefit(request, proposal, batch),
                generation_identity={"provider": request.provider, "model": request.model,
                                     "policy": request.policy_version, "schema": "ResearchDelta.v1"})
            outcome = coordinator.maintain(turn_id=self.turn_id(request),
                effective_chars=budget_report["chars"],
                provider_high=not budget_report["provider_budget_ok"],
                authoritative_revision=authoritative_revision(self.camp, self.owned_calls),
                remaining_api_calls=self._upkeep_api_allowance(request),
                role=request.role, task_id=request.task_id, intent_id=request.intent_id,
                purpose=request.purpose, identity=request.identity,
                removable_chars=max(0, budget_report["chars"] - request.budget.target_chars))
            outcome = asdict(outcome)
            if outcome.get("status") == "published":
                return {"status": "published", "validated": True,
                        "delivery": self.snapshot.snapshot_delivery(role=request.role),
                        "source_dependency_digest": digest([{"block_id": b.block_id,
                            "source_ref": b.source_ref, "source_hash": b.source_hash,
                            "revision": b.source_revision, "visibility": list(b.visibility)} for b in blocks])}
            return outcome
        finally:
            _ACTIVE.reset(token)

    def _current_kernel(self, request: CompileRequest, *, callbacks: bool) -> CompileRequest:
        if (is_research_planning_input(request.role, request.payload)
                or is_designer_scientific_input(request.role, request.payload)
                or is_engineering_input(request.role, request.payload)
                or is_analysis_scientific_input(request.role, request.payload)):
            # Native planning snapshots / Designer assembly sidecars bind full
            # state separately. Control revisions stay in the local manifest.
            return replace(request, authority_revision=authoritative_revision(self.camp, self.owned_calls),
                maintenance_callback=None, allow_semantic_compaction=False,
                allow_sync_memory_maintenance=False)
        payload = copy.deepcopy(request.payload)
        authority = native_authority(self.camp)
        from ..research_execution import single_plan_enabled
        if single_plan_enabled(self.camp) and request.role in {"research_planner", "research_critic"}:
            from ..experiment_observations import _admission_view
            controls = authority.get("controls") or {}
            admissions = controls.get("candidate_admissions")
            compact, projection = _admission_view(admissions)
            if projection:
                block = self.registry.capture(admissions, block_id="planner-history:candidate_admissions",
                    kind="checks", role=request.role)
                controls["candidate_admissions"] = compact
                controls["candidate_admissions_projection"] = {**projection, "receipt_source_ref": block.source_ref,
                    "source_hash": block.source_hash}
                authority["delivery_version"] = "single_plan.native_authority_delivery.v1"
            # CAS and execution still hash/check the complete native authority;
            # only this provider delivery view shares the compact receipt cards.
        state = _read(self.camp / "campaign_state.json")
        if state and (self.camp / "cost.json").is_file():
            from react_agent.eeg_research.agentic.budget import snapshot as native_budget
            budget = native_budget(self.camp, state)
            def update(value):
                for key, child in value.items():
                    if key in WRAPPERS and isinstance(child, dict):
                        update(child)
                    elif key == "budget" and isinstance(child, dict):
                        child.update(budget)
                    elif key in {"llm_calls_left", "gpu_seconds_left", "training_jobs_left"}:
                        value[key] = budget[key]
            update(payload)
        payload["context_authority"] = authority
        return replace(request, payload=payload, snapshot=self.snapshot,
            authority_revision=authoritative_revision(self.camp, self.owned_calls),
            maintenance_callback=self._upkeep if callbacks else None,
            allow_semantic_compaction=request.allow_semantic_compaction if callbacks else False,
            allow_sync_memory_maintenance=request.allow_sync_memory_maintenance if callbacks else False)

    def _decompose(self, request: CompileRequest) -> CompileRequest | None:
        """Choose a native current frontier, retaining all pending/unknown obligations."""
        selected_candidates = {str(request.identity[key]) for key in ("candidate_id", "target_id", "parent_candidate_id", "control_candidate_id") if request.identity.get(key)}
        selected_evidence, questions = set(), set()
        def identities(value, depth=0):
            if not isinstance(value, dict) or depth > 3:
                return
            for key in ("candidate_id", "target_id", "implementation_target_id", "parent_candidate_id", "control_candidate_id"):
                if value.get(key):
                    selected_candidates.add(str(value[key]))
            if value.get("question_id"):
                questions.add(str(value["question_id"]))
            for key in ("evidence_refs", "required_evidence_ids", "input_evidence_ids", "based_on_evidence_ids"):
                if isinstance(value.get(key), list):
                    selected_evidence.update(str(item) for item in value[key] if isinstance(item, str))
            for key in WRAPPERS | {"task", "intent", "experiment", "run", "candidate_context", "implementation_source"}:
                if isinstance(value.get(key), dict):
                    identities(value[key], depth + 1)
        identities(request.payload)
        if not selected_candidates and not selected_evidence and not questions:
            return None
        payload = copy.deepcopy(request.payload)
        navigation, changed = [], False
        for path, rows in _collections(payload):
            if len(rows) < 4:
                continue
            chosen = []
            for row in rows:
                if not isinstance(row, dict):
                    chosen.append(row); continue
                candidate = row.get("candidate_id")
                status = row.get("status", row.get("completion_status"))
                identity = _identity(row)
                compatible = not any(key in request.identity and key in identity and
                                     canonical(request.identity[key]) != canonical(identity[key])
                                     for key in ("seed", "protocol_hash", "evaluation_key", "evaluator_hash", "scope"))
                pending = status in {"pending", "running", "started", "training", "active"}
                required = (_row_id(row) in selected_evidence or row.get("question_id") in questions
                            or row.get("task_id") == request.task_id or candidate is None
                            or pending or candidate in selected_candidates or candidate == "baseline" and compatible)
                if required:
                    chosen.append(row)
            if len(chosen) == len(rows):
                continue
            original = self.registry.capture(rows, block_id="frontier-origin:/" + "/".join(path),
                kind=source_kind(path[-1], rows), role=request.role, identity=request.identity,
                revision=digest(rows), metadata={"scope": "original_frontier_collection"})
            parent = payload
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = chosen
            navigation.append({"field": "/" + "/".join(path), "source_ref": original.source_ref,
                "source_hash": original.source_hash, "total_count": len(rows), "delivered_count": len(chosen),
                "unit": "items", "max_page_chars": request.budget.excerpt_chars,
                "omission_is_not_completion": True, "native_read_receipt": False})
            changed = True
        if not changed:
            return None
        payload["context_frontier"] = {"task_id": request.task_id, "candidate_ids": sorted(selected_candidates),
            "question_ids": sorted(questions), "navigation": navigation[:16],
            "full_manifest_ref": self.snapshot.persistence.immutable("frontiers", navigation),
            "scope": "current_native_question_and_targets", "permissions_unchanged": True,
            "pending_and_unclassified_preserved": True, "native_read_receipts_satisfied": False}
        return replace(request, payload=payload, maintenance_callback=None,
                       allow_semantic_compaction=False, allow_sync_memory_maintenance=False)

    def compile(self, request: CompileRequest):
        mode = context_mode()
        from react_agent.eeg_research.agentic.handoffs import development_view
        filtered = development_view(request.payload)
        if not isinstance(filtered, dict):
            return Blocked("unreadable_source", {"reason":"private or unavailable source"})
        request = replace(request, payload=filtered)
        maintenance = request.mode in MAINTENANCE_MODES
        scientific_planning = is_research_planning_input(request.role, request.payload)
        scientific_designer = is_designer_scientific_input(request.role, request.payload)
        exact_science = (scientific_planning or scientific_designer
                         or is_analysis_scientific_input(request.role, request.payload)
                         or is_engineering_input(request.role, request.payload))
        if not maintenance and not exact_science:
            try:
                self.snapshot.invalidate_stale_sources(role=request.role)
                self.capture_closed(request.payload)
                self._bind_mirrors(request.payload, role=request.role)
            except (ValueError, KeyError, PermissionError, OSError) as exc:
                return Blocked("context_source_unavailable", {"reason": str(exc)},
                               recoverable_when=("valid applicable source or native classification",))
        from ..research_execution import single_plan_enabled, paid_maintenance_mode
        if single_plan_enabled(self.camp) and request.role == "research_planner" and not maintenance and not scientific_planning:
            from ..experiment_observations import compact_planner_view, planner_applicability_context, ObservationDeliveryRequired
            payload = copy.deepcopy(request.payload)
            try:
                # Terminal feedback is required current material. The separate
                # optional retrieval cap must not silently exclude its facts.
                delivery_cap = min(16000, request.budget.hard_chars)
                if isinstance(payload.get("observation"), dict):
                    payload["observation"] = compact_planner_view(self.camp, payload["observation"], char_cap=delivery_cap,
                        context=planner_applicability_context(self.camp, payload))
                else:
                    payload = compact_planner_view(self.camp, payload, char_cap=delivery_cap)
            except ObservationDeliveryRequired as exc:
                return exc.result
            request = replace(request, payload=payload)
        revision_before, calls_before = self.snapshot.revision, set(self.owned_calls)
        callbacks = not maintenance and not exact_science and paid_maintenance_mode(self.camp) in {"legacy", "emergency"}
        current = self._current_kernel(request, callbacks=callbacks)
        result = self.compiler.compile(current)
        if not maintenance and (self.snapshot.revision != revision_before or self.owned_calls != calls_before):
            # Own billed maintenance may change balances. Rebuild native values;
            # the old captured source remains history, never current authority.
            current = self._current_kernel(request, callbacks=False)
            result = self.compiler.compile(current)
        if isinstance(result, NeedsContext):
            decomposed = self._decompose(current) if result.reason == "required_input_too_large" else None
            if decomposed is not None:
                current = decomposed
                result = self.compiler.compile(current)
        if isinstance(result, NeedsContext):
            # This path is controller delivery only. It cannot authorize an
            # experiment or produce a scientific artifact read receipt.
            def incorporate(previous, reads):
                targeted = incorporate_partial_reads(previous, reads, registry=self.registry)
                next_payload = dict(targeted.payload)
                next_payload["context_recovery_pages"] = [{"content": r.content,
                    "delivery_scope": r.delivered_scope,
                    "native_read_receipt": False} for r in reads]
                return replace(targeted, payload=next_payload,
                               source_blocks=tuple({block.source_ref: block for block in
                                   (*targeted.source_blocks, *(read.source_block for read in reads if read.source_block is not None))}.values()),
                               allow_semantic_compaction=False,
                               allow_sync_memory_maintenance=False)
            recovery = recover_context(self.compiler, current, result, incorporate=incorporate)
            result = recovery.result
            if isinstance(result, NeedsContext):
                return Blocked("context_required_field_needs_classification_or_decomposition",
                    {"missing": list(result.missing)[:8], "role": request.role,
                     "bounded_reads_attempted": len(recovery.reads), "native_read_receipts_satisfied": False},
                    result.budget_report, ("a bounded typed current-task source adapter",))
        if isinstance(result, Ready):
            report = {**result.prepared.budget_report, "context_protocol_mode": mode}
            if mode == "shadow":
                report["shadow_comparison"] = {
                    "comparator": "privacy_filtered_effective_input_before_deterministic_packing",
                    "before_chars": report["before_packing_chars"], "after_chars": report["chars"],
                    "difference_chars": report["before_packing_chars"] - report["chars"],
                    "full_legacy_transport_sent": False, "contains_source_content": False,
                    "same_prepared_transport_sent": True}
            ref = self.snapshot.persistence.immutable("budget_reports", report)
            prepared = replace(result.prepared, budget_report=report,
                sidecar_refs={**result.prepared.sidecar_refs, "budget": str(self.camp / ref)})
            result = replace(result, prepared_request=prepared)
        return result

    def run_long_memory(self, payload: dict, *, turn_id: str) -> dict:
        from ..research_execution import paid_maintenance_mode
        if paid_maintenance_mode(self.camp) in {"off", "emergency"}:
            self.capture_closed(payload)
            return {"status": "deferred", "reason": "paid_maintenance_" + paid_maintenance_mode(self.camp), "calls": 0,
                    "recovery": self.recover_paid_publications()}
        if self.pipeline is None or self.callback is None or _ACTIVE.get():
            return {"status": "unavailable"}
        turn_id, callback = self._background_maintenance(turn_id)
        groups = self.capture_closed(payload)
        token = _ACTIVE.set(True)
        try:
            result = self.pipeline.run_turn(groups, turn_id=turn_id, callback=callback)
            if paid_maintenance_mode(self.camp) == "batch":
                statuses = {item.get("status") for item in result.get("extraction", [])} | {result.get("consolidation", {}).get("status")}
                if statuses & {"failed", "stale", "blocked", "deferred", "budget_exhausted"}:
                    result = {**result, "status": "deferred", "reason": "optional_batch_maintenance_deferred"}
            return result
        finally:
            _ACTIVE.reset(token)

    def recover_paid_publications(self) -> dict:
        """Complete verified paid drafts with the original source/CAS proof.

        No fresh maintenance claim, abstract cost reservation or model callback
        is made. A changed authority/source remains deferred for native handling.
        """
        from ..task_ledger import task_snapshots, find_completed_task
        from .snapshot import SnapshotProposal
        with self.snapshot.persistence.db() as db:
            drafts = db.execute("SELECT proposal_id,artifact_ref FROM context_proposals WHERE status='validated' ORDER BY created_at LIMIT 8").fetchall()
        tasks = [task for task in reversed(task_snapshots(self.camp)) if task.get("role") == "memory_curator" and task.get("status") == "completed"]
        ledger = self.camp / "llm_calls.jsonl"
        calls = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.is_file() else []
        recovered, deferred = [], []
        for row in drafts:
            draft = self.snapshot.persistence.read(row["artifact_ref"])
            operation_id = str((draft.get("generation") or {}).get("operation_id") or "")
            with self.snapshot.persistence.db() as db:
                no_benefit = db.execute("SELECT 1 FROM context_maintenance_jobs WHERE status='no_op' AND last_error='no_effective_request_benefit' AND ? LIKE input_digest || '%'", (operation_id,)).fetchone()
            if no_benefit:
                deferred.append({"proposal_id": row["proposal_id"], "reason": "original_no_effective_request_benefit"})
                continue
            proven = False
            for task in tasks[:64]:
                cache = find_completed_task(self.camp, role="memory_curator",
                    recovery_identity=task.get("recovery_identity") or {}, input_digest=task.get("input_digest"))
                if cache and cache["artifact"].get("kind") == "handoff_compaction":
                    envelope = cache["envelope"]
                    body = envelope.get("payload") or {}
                    dependencies = (task.get("recovery_identity") or {}).get("input_dependencies") or []
                    original_inputs = [_read(Path(dep["path"])) for dep in dependencies if isinstance(dep, dict) and dep.get("path")]
                    source_match = any(item.get("closed_sections") == draft["batch"]["sections"]
                        and item.get("base_revision") == draft["batch"]["base_revision"]
                        and item.get("maintenance_operation_id") == operation_id for item in original_inputs)
                    if (source_match and envelope.get("status") == "completed" and body.get("publication_status") == "proposal_only"
                            and body.get("scientific_evidence") is False and body.get("execution_authority") is False
                            and body.get("proposal") == draft["delta"]):
                        proven = True
                        self.owned_calls.update(call["call_id"] for call in calls if call.get("task_id") == task["task_id"]
                            and call.get("success") is True and call.get("transport_outcome") == "confirmed_response")
                        break
            if not proven:
                deferred.append({"proposal_id": row["proposal_id"], "reason": "original_paid_proposal_proof_unavailable"})
                continue
            proposal = SnapshotProposal(row["proposal_id"], row["artifact_ref"], draft["batch"]["base_revision"],
                draft["batch"]["input_dependency_digest"])
            try:
                outcome = self.snapshot.publish(proposal)
                destination = recovered if outcome.status in {"published", "no_signal", "already_published", "already_no_signal"} else deferred
                destination.append({"proposal_id": row["proposal_id"], "status": outcome.status, "reason": outcome.reason})
            except (ValueError, KeyError, OSError, PermissionError) as exc:
                deferred.append({"proposal_id": row["proposal_id"], "reason": str(exc)[:500]})
        pipeline = self._recover_paid_pipeline_publications(tasks)
        return {"api_calls": 0, "recovered": recovered, "deferred": deferred, "long_memory": pipeline}

    def _recover_paid_pipeline_publications(self, tasks) -> dict:
        if self.pipeline is None:
            return {"api_calls": 0, "recovered": [], "deferred": []}
        from ..task_ledger import find_completed_task
        import uuid
        with self.pipeline.connect() as db:
            rows = db.execute("SELECT request_digest,parent_digest,phase,response,metadata FROM memory_pipeline_proposals WHERE status='proposal_only' AND response IS NOT NULL ORDER BY created_at LIMIT 8").fetchall()
        results, deferred = [], []
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            claim = metadata.get("publication_claim")
            if not isinstance(claim, dict) or claim.get("digest") != row["parent_digest"] or claim.get("phase") != row["phase"]:
                deferred.append({"digest": row["parent_digest"], "reason": "original_publication_claim_missing"})
                continue
            response = json.loads(row["response"])
            task = next((task for task in tasks if task.get("task_id") == metadata.get("task_ref")), None)
            cache = find_completed_task(self.camp, role="memory_curator", recovery_identity=task.get("recovery_identity") or {},
                input_digest=task.get("input_digest")) if task else None
            if (not cache or cache["artifact"].get("kind") != "handoff_compaction"
                    or (cache["envelope"].get("payload") or {}).get("proposal") != response):
                deferred.append({"digest": row["parent_digest"], "reason": "original_paid_proposal_proof_unavailable"})
                continue
            dependencies = (task.get("recovery_identity") or {}).get("input_dependencies") or []
            originals = [_read(Path(dep["path"])) for dep in dependencies if isinstance(dep, dict) and dep.get("path")]
            original_request = claim.get("request") or {}
            expected_input = original_request.get("payload", original_request)
            if not any(item.get("maintenance_input_digest") == original_request.get("input_digest")
                and all(item.get(key) == value for key, value in expected_input.items()
                        if key not in {"turn_id", "mode", "maintenance_correction", "input_digest"}) for item in originals):
                deferred.append({"digest": row["parent_digest"], "reason": "original_frozen_request_mismatch"})
                continue
            # Reacquire only this original publication lease, never an API or
            # retry allocation. Live/inflight ownership is left untouched.
            now, owner = self.pipeline.clock(), uuid.uuid4().hex
            with self.pipeline.connect(write=True) as db:
                table, where = ("memory_pipeline_jobs", "digest=?") if row["phase"] == "extraction" else ("memory_pipeline_writer", "singleton=1")
                parameters = (claim["digest"],) if row["phase"] == "extraction" else ()
                head = db.execute("SELECT * FROM " + table + " WHERE " + where, parameters).fetchone()
                if (not head or (head["lease_until"] or 0) > now or head["in_flight"] and head["lease_deadline"] > now):
                    deferred.append({"digest": claim["digest"], "reason": "publication_lease_active_or_missing"})
                    continue
                if row["phase"] == "extraction" and head["processed"]:
                    results.append({"digest": claim["digest"], "status": "already_processed"})
                    continue
                if row["phase"] == "consolidation" and (head["revision"] != claim["base_revision"] or head["baseline"] != claim["baseline"] or head["input_digest"] != claim["digest"]):
                    deferred.append({"digest": claim["digest"], "reason": "original_publication_cas_changed"})
                    continue
                db.execute("UPDATE " + table + " SET lease_owner=?,lease_until=? WHERE " + where,
                    (owner, now + self.pipeline.config.lease_seconds, *parameters))
            claim = {**claim, "owner": owner}
            try:
                publisher = self.pipeline.publish_extraction if row["phase"] == "extraction" else self.pipeline.publish
                result = publisher(claim, response)
                with self.pipeline.connect(write=True) as db:
                    db.execute("UPDATE memory_pipeline_proposals SET status=? WHERE request_digest=?", (result["status"], row["request_digest"]))
                results.append({"digest": claim["digest"], **result})
            except (ValueError, KeyError, OSError) as exc:
                deferred.append({"digest": claim["digest"], "reason": str(exc)[:500]})
        return {"api_calls": 0, "recovered": results, "deferred": deferred}

    def after_task_background(self, payload: dict, *, turn_id: str, remaining_api_calls: int | None = None) -> dict:
        """Finite closed-source upkeep; no new training or execution permission."""
        from ..research_execution import paid_maintenance_mode
        if paid_maintenance_mode(self.camp) in {"off", "emergency"}:
            self.capture_closed(payload)
            return {"status": "deferred", "reason": "paid_maintenance_" + paid_maintenance_mode(self.camp), "calls": 0,
                    "recovery": self.recover_paid_publications()}
        if self.callback is None or _ACTIVE.get():
            return {"status": "unavailable"}
        if remaining_api_calls is not None and remaining_api_calls <= 0:
            return {"status": "budget_exhausted"}
        turn_id, callback = self._background_maintenance(turn_id)
        self.capture_closed(payload)
        token = _ACTIVE.set(True)
        try:
            # An idle snapshot is independent of one role's transient input size.
            outcome = MaintenanceCoordinator(self.snapshot, callback).maintain(
                turn_id=turn_id, idle=True, authoritative_revision=authoritative_revision(self.camp, self.owned_calls),
                remaining_api_calls=remaining_api_calls, purpose="closed_task_background")
            remaining = max(0, 2 - outcome.calls)
            if remaining_api_calls is not None:
                remaining = min(remaining, max(0, remaining_api_calls - outcome.calls))
            memory = {"status": "budget_exhausted" if remaining == 0 else "unavailable"}
            if self.pipeline and remaining > 0:
                previous_config = self.pipeline.config
                self.pipeline.config = replace(previous_config, max_calls_per_turn=min(remaining, previous_config.max_calls_per_turn))
                try:
                    memory = self.pipeline.run_turn([], turn_id=turn_id, callback=callback)
                finally:
                    self.pipeline.config = previous_config
            nested_statuses = {outcome.status}
            nested_statuses.update(row.get("status") for row in memory.get("extraction", []))
            nested_statuses.add(memory.get("consolidation", {}).get("status", memory.get("status")))
            if nested_statuses & {"failed", "stale", "blocked"}:
                status = "deferred" if paid_maintenance_mode(self.camp) == "batch" else "failed"
            elif nested_statuses & {"budget_exhausted", "deferred", "backoff", "leased", "attempts_exhausted", "concurrency_limit"}:
                status = "deferred"
            else:
                status = "complete"
            return {"status": status, "snapshot": asdict(outcome), "long_memory": memory,
                    "real_llm_quality_verified": False, "native_read_receipts_satisfied": False}
        finally:
            _ACTIVE.reset(token)


async def send_prepared(client, prepared, *, role: str, profile: str = "fast"):
    """Production sends prepared messages; compatible test doubles receive the same strings."""
    if not prepared.verify() or not prepared.budget_report.get("ready_budget_ok"):
        raise ValueError("unmeasured_or_mutated_context_request")
    method = getattr(client, "complete_prepared_json", None)
    if method is not None:
        return await method(prepared, profile=profile, role=role)
    return await client.complete_json(system=prepared.system, user=prepared.user,
                                      profile=profile, role=role)
