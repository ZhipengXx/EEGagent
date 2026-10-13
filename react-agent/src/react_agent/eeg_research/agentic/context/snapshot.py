"""Incremental, source-anchored working memory, separate from native authority.

Coverage is section/version scoped.  Only validated publication can replace
narrative sections; delivery and actual reads have their own receipt table.
"""
from __future__ import annotations

import copy
import contextlib
import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .persistence import ContextPersistence
from .sources import ContextSourceRegistry, pointer_escape, pointer_get
from .types import SourceBlock, canonical, digest

SNAPSHOT_VERSION = "eeg.context.working_snapshot.v1"
DELTA_VERSION = "eeg.context.research_delta.v1"
CLOSED = frozenset({"complete", "completed", "failed", "cancelled", "canceled", "revoked", "no_signal"})
FAILURE = frozenset({"failed", "partial", "unknown", "unverified"})
NATIVE_IDENTITY = ("candidate_id", "method_revision", "seed", "protocol", "protocol_hash",
                   "evaluation_scope", "science_scope", "evidence_level", "evaluator",
                   "held_out_subject", "coverage", "required", "full_required",
                   "execution_status", "validation_status", "independent_seed_count",
                   "scope", "fold_id", "evaluation_key", "evaluator_hash", "selection_manifest_hash",
                   "source_hash", "config_hash", "checkpoint_hash", "execution_fingerprint",
                   "evidence_id", "artifact_id", "suite_id", "job_id", "run_id", "operation_id")
ORIGIN_IDENTITY = ("evidence_id", "artifact_id", "suite_id", "job_id", "run_id", "operation_id",
                   "analysis_evidence_id", "source_artifact_id", "approved_artifact_id")
MIRROR_SCIENCE_KEYS = ("candidate_id", "method_revision", "seed", "protocol_hash", "evaluation_key",
                      "evaluator", "evaluator_hash", "held_out_subject", "fold_id", "science_scope",
                      "scope", "evaluation_scope", "fidelity", "selection_manifest_hash", "source_hash",
                      "config_hash", "checkpoint_hash", "execution_fingerprint", "approved_spec_hash")
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?!\w|\.\d)")


class FactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section_ref: str = Field(min_length=1, max_length=1_200)
    path: str = Field(max_length=500)
    value: int | float


class ItemPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    item_id: str = Field(pattern=r"^[\w.:/-]{1,120}$")
    action: str = Field(pattern=r"^(add|revise|retire|update|resolve|reopen)$")
    text: str = Field(default="", max_length=1_000)
    source_refs: list[str] = Field(min_length=1, max_length=12)
    facts: list[FactReference] = Field(default_factory=list, max_length=16)
    conditions: str = Field(default="", max_length=400)
    supersedes: list[str] = Field(default_factory=list, max_length=8)


class ResearchDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = DELTA_VERSION
    base_revision: int = Field(ge=0)
    input_dependency_digest: str = Field(min_length=64, max_length=64)
    findings: list[ItemPatch] = Field(default_factory=list, max_length=12)
    engineering_failures: list[ItemPatch] = Field(default_factory=list, max_length=12)
    open_questions: list[ItemPatch] = Field(default_factory=list, max_length=12)
    decisions: list[ItemPatch] = Field(default_factory=list, max_length=8)
    recent_arc: str = Field(default="", max_length=600)
    no_signal: bool = False


@dataclass(frozen=True)
class GroupBatch:
    base_revision: int
    input_dependency_digest: str
    sections: tuple[dict[str, Any], ...]
    authority_revision: str
    source_epoch: int
    groups: tuple[str, ...]
    input_chars: int
    pending_count: int = 0
    role: str = ""

    def callback_payload(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {"schema_version": DELTA_VERSION, "base_revision": self.base_revision,
                "input_dependency_digest": self.input_dependency_digest,
                "previous_snapshot": dict(snapshot), "closed_sections": list(self.sections),
                "coverage_rule": "Only cited, verified frozen sections may be abstracted.",
                "authority_rule": "Native permissions, balances, gates and exact metrics are immutable.",
                "allow_semantic_compaction": False, "allow_sync_memory_maintenance": False}


@dataclass(frozen=True)
class SnapshotProposal:
    proposal_id: str
    artifact_ref: str
    base_revision: int
    input_dependency_digest: str
    status: str = "validated"


@dataclass(frozen=True)
class PublishResult:
    status: str
    revision: int
    artifact_ref: str = ""
    reason: str = ""
    processed_count: int = 0
    abstracted_count: int = 0


def _item_rows(db, *, role: str = "", limit: int = 64) -> list[dict[str, Any]]:
    return [json.loads(row[0]) for row in db.execute("""SELECT item_json FROM context_snapshot_items
        WHERE active=1 AND valid=1 AND EXISTS (SELECT 1 FROM
        json_each(COALESCE(json_extract(item_json,'$.visibility'),'["campaign"]')) v
        WHERE v.value IN ('campaign',?))
        ORDER BY CASE item_kind WHEN 'open_questions' THEN 0 WHEN 'engineering_failures' THEN 1
        WHEN 'decisions' THEN 2 ELSE 3 END, revision DESC,item_id LIMIT ?""", (role, limit))]


def _changed_ids(delta: Mapping[str, Any]) -> set[str]:
    return {item_id for kind in ("findings", "engineering_failures", "open_questions", "decisions")
            for patch in delta.get(kind, []) for item_id in [patch["item_id"], *patch.get("supersedes", [])]}


def _existing_items(db, item_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not item_ids:
        return {}
    return {row[0]: json.loads(row[1]) for row in db.execute(
        "SELECT item_id,item_json FROM context_snapshot_items WHERE item_id IN ("
        + ",".join("?" for _ in item_ids) + ")", tuple(sorted(item_ids)))}


class SnapshotStore:
    """Maintain immutable drafts and atomically published bounded snapshots."""
    def __init__(self, camp: Path | str, *, source_registry: ContextSourceRegistry | None = None,
                 authority_provider: Callable[[], str] | None = None, clock=None):
        self.camp = Path(camp).resolve()
        kwargs = {"clock": clock} if clock is not None else {}
        self.registry = source_registry or ContextSourceRegistry(self.camp)
        self.campaign_id = self.registry.campaign_id
        self.persistence = ContextPersistence(self.camp, campaign_id=self.campaign_id, **kwargs)
        self.authority_provider = authority_provider

    @property
    def revision(self) -> int:
        return int(self.persistence.state()["revision"])

    def set_authority_revision(self, revision: str) -> None:
        with self.persistence.db(immediate=True) as db:
            db.execute("UPDATE context_state SET authority_revision=? WHERE singleton=1", (str(revision),))

    def _authority(self) -> str:
        return str(self.authority_provider()) if self.authority_provider else self.persistence.state()["authority_revision"]

    def validate_source_ref(self, ref: str | Mapping[str, Any], *, role: str = "",
                            verify_payload: bool = True) -> bool:
        source_ref = str(ref.get("source_ref", "")) if isinstance(ref, Mapping) else str(ref)
        try:
            source = self.registry.get(source_ref, role=role)
            if isinstance(ref, Mapping) and ref.get("source_hash") != source.source_hash:
                return False
            visiting, checked = set(), set()
            def valid(block: SourceBlock) -> bool:
                if block.source_ref in visiting:
                    return False
                if block.source_ref in checked:
                    return True
                if len(visiting) + len(checked) >= 64:
                    return False
                visiting.add(block.source_ref)
                if verify_payload:
                    # This filesystem may preserve mtime and ctime across writes.
                    # Only a freshly verified full content hash establishes body
                    # validity. DAG reuse is limited to this validation operation.
                    self.registry.load(block, role=role)
                for dependency_ref in block.dependency_refs:
                    if not valid(self.registry.get(dependency_ref, role=role)):
                        return False
                visiting.remove(block.source_ref)
                checked.add(block.source_ref)
                return True
            return valid(source)
        except (KeyError, ValueError, OSError, PermissionError):
            return False

    @contextlib.contextmanager
    def _publish_db(self):
        # Keep registered source heads stable through the memory publication
        # transaction. The source database is read-only within this lock; only
        # memory.sqlite changes, so a crash has one commit boundary.
        with self.registry._db() as registry_lock:
            registry_lock.execute("BEGIN IMMEDIATE")
            with self.persistence.db(immediate=True) as db:
                yield db

    def capture_closed_group(self, group_id: str, sources: list[SourceBlock] | tuple[SourceBlock, ...],
                             *, status: str = "complete", authoritative_revision: str = "",
                             section_chars: int = 4_000, role: str = "") -> dict[str, Any]:
        """Capture finite, exact sections after an operation reaches a terminal state.

        Async/pending groups are rejected rather than treated as orphan history.
        A large field is split into explicit character ranges; no head/tail loss.
        """
        if status not in CLOSED:
            return {"status": "pending", "captured": 0, "reason": "operation_not_closed"}
        if not group_id or len(group_id) > 240 or not 256 <= section_chars <= 8_000:
            raise ValueError("invalid closed group or section budget")
        if authoritative_revision:
            self.set_authority_revision(authoritative_revision)
        captured = 0
        for source in sources:
            if not isinstance(source, SourceBlock):
                source = SourceBlock(**dict(source))
            if source.campaign_id != self.campaign_id:
                raise PermissionError("working snapshot cannot cross campaigns")
            current = self.registry.get(source.source_ref, role=role)
            if (current.source_hash != source.source_hash or source.source_validity != "valid"
                    or any(scope in {"final_test", "final_holdout", "secret"} for scope in source.visibility)):
                raise PermissionError("source is not valid and visible for working memory")
            raw = self.registry.load(current, role=role)
            fields = list(raw.items()) if isinstance(raw, dict) else [("", raw)]
            for key, value in fields:
                pointer = "/" + pointer_escape(str(key)) if isinstance(raw, dict) else ""
                text = canonical(value)
                for start in range(0, max(1, len(text)), section_chars):
                    end = min(len(text), start + section_chars)
                    section_id = pointer if len(text) <= section_chars else f"{pointer}#chars={start}:{end}"
                    # Metadata binds the full identity; equal values from another
                    # seed/evaluator/protocol remain different source sections.
                    native = {key: source.identity[key] for key in NATIVE_IDENTITY if key in source.identity}
                    section_ref = digest({"source_ref": source.source_ref, "source_hash": source.source_hash,
                                          "section_id": section_id, "event_ids": source.event_ids})
                    exact = self._fact_catalog(value, pointer) if len(text) <= section_chars else []
                    section = {"section_ref": section_ref, "section_id": section_id,
                               "block_id": source.block_id, "source_ref": source.source_ref,
                               "source_hash": source.source_hash, "source_revision": source.source_revision,
                               "event_ids": list(source.event_ids), "group_id": group_id,
                               "kind": source.kind, "status": source.status if source.status != "captured" else status,
                               "visibility": list(source.visibility),
                               "identity": native, "json_pointer": pointer,
                               "selected_range": {"unit": "canonical_chars", "start": start, "end": end},
                               "complete_section": len(text) <= section_chars,
                               "text": text[start:end], "exact_facts": exact}
                    with self.persistence.db() as db:
                        changed = db.execute("INSERT OR IGNORE INTO context_closed_sections VALUES(?,?,?,?,?,?,?,?,?)",
                                             (section_ref, group_id, source.block_id, source.source_ref,
                                              source.source_hash, section_id, canonical(section),
                                              section["status"], self.persistence.clock())).rowcount
                    captured += changed
        return {"status": "captured" if captured else "unchanged", "captured": captured}

    @staticmethod
    def _fact_catalog(value: Any, pointer: str) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        def visit(item: Any, path: str):
            if len(facts) >= 64:
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, path + "/" + pointer_escape(str(key)))
            elif isinstance(item, list):
                for i, child in enumerate(item):
                    visit(child, path + "/" + str(i))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                facts.append({"path": path, "value": item})
        visit(value, pointer)
        return facts

    def pending_groups(self, *, limit: int = 12, max_chars: int = 24_000,
                       max_groups: int = 4, role: str = "") -> GroupBatch:
        if not 0 < limit <= 64 or not 1_024 <= max_chars <= 60_000 or not 0 < max_groups <= 16:
            raise ValueError("maintenance batch must be bounded")
        self.invalidate_stale_sources(role=role)
        state = self.persistence.state()
        with self.persistence.db() as db:
            rows = db.execute("""SELECT s.* FROM context_closed_sections s
                LEFT JOIN context_coverage c ON c.section_ref=s.section_ref AND c.valid=1
                WHERE COALESCE(c.processed,0)=0 AND NOT EXISTS
                    (SELECT 1 FROM context_invalid_sources i WHERE i.source_ref=s.source_ref)
                ORDER BY CASE WHEN s.section_id LIKE '/error%' OR s.section_id LIKE '/failure%'
                    OR s.section_id LIKE '/traceback%' THEN 0
                    WHEN s.section_id LIKE '/note%' OR s.section_id LIKE '/analysis%'
                    OR s.section_id LIKE '/narrative%' OR s.section_id LIKE '/explanation%'
                    OR s.section_id LIKE '/interpretation%' OR s.section_id LIKE '/log%' THEN 1 ELSE 2 END,
                    s.captured_at,s.group_id,s.section_ref
                LIMIT ?""", (limit * 4,)).fetchall()
            count = db.execute("""SELECT COUNT(*) FROM context_closed_sections s
                LEFT JOIN context_coverage c ON c.section_ref=s.section_ref AND c.valid=1
                WHERE COALESCE(c.processed,0)=0 AND NOT EXISTS
                    (SELECT 1 FROM context_invalid_sources i WHERE i.source_ref=s.source_ref)""").fetchone()[0]
        selected, groups, size = [], [], 0
        source_checks = {}
        for row in rows:
            section = json.loads(row["section_json"])
            check_key = (section["source_ref"], section["source_hash"])
            if check_key not in source_checks:
                source_checks[check_key] = self.validate_source_ref(section, role=role)
            if not source_checks[check_key]:
                continue
            if section["group_id"] not in groups and len(groups) >= max_groups:
                continue
            cost = len(canonical(section))
            if size + cost > max_chars:
                continue
            if section["group_id"] not in groups:
                groups.append(section["group_id"])
            selected.append(section)
            size += cost
            if len(selected) >= limit:
                break
        dependency = digest({"base_revision": state["revision"], "authority_revision": self._authority(),
                             "source_epoch": state["source_epoch"],
                             "sections": [{"section_ref": item["section_ref"], "source_hash": item["source_hash"]}
                                          for item in selected]})
        return GroupBatch(state["revision"], dependency, tuple(selected), self._authority(),
                          state["source_epoch"], tuple(groups), size, count, role)

    def current_snapshot(self) -> dict[str, Any]:
        state = self.persistence.state()
        if not state["snapshot_ref"]:
            return {"schema_version": SNAPSHOT_VERSION, "campaign_id": self.campaign_id,
                    "revision": 0, "base_revision": 0, "findings": [],
                    "engineering_failures": [], "open_questions": [], "decisions": [],
                    "recent_arc": "", "coverage_manifest_ref": "sqlite:context_coverage",
                    "policy_version": "eeg.context.policy.v1", "source_dependency_digest": digest([])}
        return self.persistence.read(state["snapshot_ref"])

    def items_page(self, *, kind: str | None = None, offset: int = 0,
                   limit: int = 20, include_retired: bool = False, role: str = "") -> list[dict[str, Any]]:
        if offset < 0 or not 0 < limit <= 64:
            raise ValueError("snapshot item navigation must be bounded")
        where, args = ["valid=1", "EXISTS (SELECT 1 FROM json_each(COALESCE(json_extract(item_json,'$.visibility'),'[\"campaign\"]')) v WHERE v.value IN ('campaign',?))"], [role]
        if not include_retired:
            where.append("active=1")
        if kind:
            where.append("item_kind=?")
            args.append(kind)
        with self.persistence.db() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT item_json FROM context_snapshot_items WHERE " + " AND ".join(where)
                + " ORDER BY item_id LIMIT ? OFFSET ?", (*args, limit, offset))]

    @staticmethod
    def _bounded_view(snapshot: dict[str, Any], items: list[dict[str, Any]], max_chars: int,
                      *, total_count: int | None = None) -> dict[str, Any]:
        view = {key: value for key, value in snapshot.items()
                if key not in {"findings", "engineering_failures", "open_questions", "decisions"}}
        for kind in ("findings", "engineering_failures", "open_questions", "decisions"):
            view[kind] = []
        view["delivery"] = {"status": "bounded", "item_count": len(items) if total_count is None else total_count, "delivered_count": 0,
                            "navigation": {"store": "sqlite:context_snapshot_items", "max_page": 64},
                            "omission_is_not_retirement": True}
        # Unresolved questions and engineering failures precede settled findings.
        # No item is removed from storage when a view reaches its budget.
        priority = {"open_questions": 0, "engineering_failures": 1, "decisions": 2, "findings": 3}
        ranked = sorted(items, key=lambda item: (priority.get(item["item_kind"], 4), -item.get("revision", 0), item["item_id"]))
        for item in ranked:
            if item["item_kind"] not in view:
                continue
            view[item["item_kind"]].append(item)
            view["delivery"]["delivered_count"] += 1
            if len(canonical(view)) > max_chars:
                view[item["item_kind"]].pop()
                view["delivery"]["delivered_count"] -= 1
        if len(canonical(view)) > max_chars:
            # Explicitly fail the delivery envelope rather than truncating a
            # mandatory identity/policy manifest into a plausible snapshot.
            return {"schema_version": SNAPSHOT_VERSION, "revision": snapshot["revision"],
                    "delivery": {"status": "needs_context", "reason": "snapshot_kernel_too_large",
                                 "navigation": "sqlite:context_snapshot_items"}}
        return view

    def snapshot_delivery(self, *, max_chars: int = 8_000, hard_max_chars: int = 12_000,
                          role: str = "", identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not 256 <= max_chars <= hard_max_chars <= 12_000:
            raise ValueError("snapshot delivery exceeds protocol cap")
        self.invalidate_stale_sources(role=role)
        state = self.persistence.state()
        with self.persistence.db() as db:
            rows = _item_rows(db, role=role)
            total = db.execute("""SELECT COUNT(*) FROM context_snapshot_items WHERE active=1 AND valid=1
                AND EXISTS (SELECT 1 FROM json_each(COALESCE(json_extract(item_json,'$.visibility'),'["campaign"]')) v
                WHERE v.value IN ('campaign',?))""", (role,)).fetchone()[0]
        source_checks: dict[tuple[str, str], bool] = {}
        def valid(source):
            key = (source["source_ref"], source["source_hash"])
            if key not in source_checks:
                source_checks[key] = self.validate_source_ref(source, role=role)
            return source_checks[key]
        rows = [row for row in rows if all(valid(source) for source in row.get("sources", []))]
        if identity:
            rows = [row for row in rows if not any(
                key in source["identity"] and source["identity"][key] != value
                for source in row.get("sources", []) for key, value in identity.items())]
        current = self.current_snapshot()
        # A revocation makes the persisted snapshot a historical representation;
        # delivery is rebuilt from the validated entries, never replayed blindly.
        current["revision"] = state["revision"]
        visibility = current.get("visibility", ["campaign"])
        if ("campaign" not in visibility and role not in visibility
                or any(not valid(source)
                       for source in current.get("recent_arc_sources", []))):
            current["recent_arc"] = ""
            current["recent_arc_sources"] = []
        return self._bounded_view(current, rows, max_chars, total_count=total)

    def validate_delta(self, proposed: Mapping[str, Any] | ResearchDelta,
                       batch: GroupBatch) -> dict[str, Any]:
        delta = ResearchDelta.model_validate(proposed).model_dump()
        if delta["schema_version"] != DELTA_VERSION:
            raise ValueError("research_delta_schema_version")
        if delta["base_revision"] != batch.base_revision or delta["input_dependency_digest"] != batch.input_dependency_digest:
            raise ValueError("research_delta_dependency_mismatch")
        sections = {section["section_ref"]: section for section in batch.sections}
        patches = [(kind, item) for kind in ("findings", "engineering_failures", "open_questions", "decisions")
                   for item in delta[kind]]
        if not patches and not delta["no_signal"]:
            raise ValueError("research_delta_empty_or_no_op")
        if delta["no_signal"] and (patches or delta["recent_arc"]):
            raise ValueError("research_delta_no_signal_has_claims")
        ids = [item["item_id"] for _, item in patches]
        if len(ids) != len(set(ids)):
            raise ValueError("research_delta_duplicate_item")
        with self.persistence.db() as db:
            existing = _existing_items(db, _changed_ids(delta))
        for kind, item in patches:
            allowed = {"add", "update", "resolve", "reopen", "retire"} if kind == "open_questions" else {"add", "revise", "retire"}
            if item["action"] not in allowed:
                raise ValueError("research_delta_invalid_action_for_kind")
            if item["action"] != "add" and item["item_id"] not in existing:
                raise ValueError("research_delta_unknown_item")
            if item["action"] == "add" and item["item_id"] in existing:
                raise ValueError("research_delta_existing_item_requires_revision")
            if item["item_id"] in existing and existing[item["item_id"]]["item_kind"] != kind:
                raise ValueError("research_delta_item_kind_is_stable")
            if item["action"] not in {"retire", "resolve"} and not item["text"].strip():
                raise ValueError("research_delta_empty_claim")
            if not set(item["source_refs"]).issubset(sections) or len(set(item["source_refs"])) != len(item["source_refs"]):
                raise ValueError("research_delta_unknown_source_ref")
            if not set(item["supersedes"]).issubset(existing):
                raise ValueError("research_delta_unknown_superseded_item")
            # Equal metrics from another seed/method/evaluator cannot revise or
            # retire an old claim merely because that event arrived later.
            related_old = [existing[old] for old in item["supersedes"]]
            if item["action"] != "add":
                related_old.append(existing[item["item_id"]])
            for old in related_old:
                for prior in old.get("sources", []):
                    for ref in item["source_refs"]:
                        identity = sections[ref]["identity"]
                        for key in ("candidate_id", "seed", "protocol_hash", "evaluator", "held_out_subject"):
                            if key in prior["identity"] and key in identity and prior["identity"][key] != identity[key]:
                                raise ValueError("research_delta_cross_scope_revision")
            if item["action"] == "resolve" and not any(sections[ref]["status"] in CLOSED for ref in item["source_refs"]):
                raise ValueError("question_resolution_requires_terminal_source")
            valid_numbers = []
            for fact in item["facts"]:
                if fact["section_ref"] not in item["source_refs"]:
                    raise ValueError("research_delta_fact_not_cited")
                section = sections[fact["section_ref"]]
                selected_path = section["json_pointer"]
                if not section["complete_section"] or not (fact["path"] == selected_path or fact["path"].startswith(selected_path + "/") or selected_path == ""):
                    raise ValueError("research_delta_fact_outside_delivered_section")
                block = self.registry.get(section["source_ref"], role=batch.role)
                actual = pointer_get(self.registry.load(block, role=batch.role), fact["path"])
                if isinstance(actual, bool) or not isinstance(actual, (int, float)) or type(actual) is not type(fact["value"]) or actual != fact["value"]:
                    raise ValueError("research_delta_exact_fact_mismatch")
                valid_numbers.append(actual)
            for number in NUMBER.findall(item["text"] + " " + item["conditions"]):
                if float(number) not in valid_numbers:
                    raise ValueError("research_delta_ungrounded_numeric_text")
            # Scope/state is not supplied by the model. It is copied literally
            # from original source identities when the native merge occurs.
        if NUMBER.search(delta["recent_arc"]):
            raise ValueError("research_delta_recent_arc_must_reference_items_for_numbers")
        if len(canonical(delta)) > 12_000:
            raise ValueError("research_delta_output_budget_exceeded")
        return delta

    def propose_delta(self, delta: Mapping[str, Any] | ResearchDelta, input_batch: GroupBatch,
                      *, expected_authoritative_revision: str | None = None,
                      generation: Mapping[str, Any] | None = None) -> SnapshotProposal:
        draft = {"delta": delta.model_dump() if isinstance(delta, ResearchDelta) else dict(delta),
                 "batch": dataclasses.asdict(input_batch), "authority_revision": expected_authoritative_revision if expected_authoritative_revision is not None else input_batch.authority_revision}
        draft["generation"] = dict(generation or {})
        if len(canonical(draft["generation"])) > 4_000:
            raise ValueError("working_memory_generation_metadata_too_large")
        draft_ref = self.persistence.immutable("drafts", draft)
        proposal_id = digest(draft)
        with self.persistence.db() as db:
            db.execute("INSERT OR IGNORE INTO context_proposals VALUES(?,?,'draft','',?)", (proposal_id, draft_ref, self.persistence.clock()))
        try:
            validated = self.validate_delta(delta, input_batch)
        except (ValueError, KeyError, OSError, PermissionError) as exc:
            with self.persistence.db() as db:
                db.execute("UPDATE context_proposals SET status='invalid',detail=? WHERE proposal_id=?", (str(exc)[:2_000], proposal_id))
            raise
        with self.persistence.db() as db:
            db.execute("UPDATE context_proposals SET status='validated' WHERE proposal_id=?", (proposal_id,))
        return SnapshotProposal(proposal_id, draft_ref, input_batch.base_revision,
                                input_batch.input_dependency_digest, "no_signal" if validated["no_signal"] else "validated")

    def _merge(self, delta: dict[str, Any], sections: dict[str, dict[str, Any]],
               existing: dict[str, dict[str, Any]], revision: int) -> tuple[dict[str, dict[str, Any]], set[str]]:
        covered: set[str] = set()
        result = copy.deepcopy(existing)
        for kind in ("findings", "engineering_failures", "open_questions", "decisions"):
            for patch in delta[kind]:
                sources = [{key: sections[ref][key] for key in
                            ("section_ref", "section_id", "block_id", "source_ref", "source_hash", "identity", "status", "visibility")}
                           for ref in patch["source_refs"]]
                private_scopes = [set(source["visibility"]) for source in sources if "campaign" not in source["visibility"]]
                visibility = sorted(set.intersection(*private_scopes)) if private_scopes else ["campaign"]
                item = {**patch, "item_kind": kind, "sources": sources,
                        "revision": revision, "valid": True,
                        "visibility": visibility,
                        "active": patch["action"] not in {"retire", "resolve"},
                        "representation_fidelity": "SUMMARY",
                        "scientific_evidence": False, "execution_approval": False,
                        "satisfies_artifact_read": False}
                result[patch["item_id"]] = item
                if patch["action"] not in {"retire", "resolve"}:
                    covered.update(patch["source_refs"])
                for old_id in patch["supersedes"]:
                    result[old_id]["active"] = False
                    result[old_id]["retired_by"] = patch["item_id"]
        return result, covered

    def publish(self, proposal: SnapshotProposal, *, failpoint: Callable[[str], None] | None = None,
                maintenance_job: tuple[str, str] | None = None) -> PublishResult:
        """CAS, dependencies, coverage and optional job success share one transaction."""
        draft = self.persistence.read(proposal.artifact_ref)
        batch = GroupBatch(**{**draft["batch"], "sections": tuple(draft["batch"]["sections"]),
                              "groups": tuple(draft["batch"]["groups"])})
        state = self.persistence.state()
        with self.persistence.db() as db:
            published = db.execute("SELECT status FROM context_proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()
        if published and published[0] in {"published", "no_signal"}:
            return PublishResult("already_" + published[0], state["revision"], state["snapshot_ref"] or "")
        if state["revision"] != batch.base_revision:
            return PublishResult("conflict", state["revision"], reason="base_revision_changed")
        if self._authority() != draft["authority_revision"] or state["source_epoch"] != batch.source_epoch:
            return PublishResult("stale", state["revision"], reason="native_authority_or_branch_changed")
        for section in batch.sections:
            if not self.validate_source_ref(section, role=batch.role):
                return PublishResult("stale", state["revision"], reason="source_hash_changed_or_revoked")
        delta = self.validate_delta(draft["delta"], batch)
        with self.persistence.db() as db:
            existing = _existing_items(db, _changed_ids(delta))
            working = {item["item_id"]: item for item in _item_rows(db, role=batch.role)}
            old_total = db.execute("SELECT COUNT(*) FROM context_snapshot_items WHERE active=1 AND valid=1").fetchone()[0]
        next_revision = batch.base_revision + (0 if delta["no_signal"] else 1)
        merged, abstracted = self._merge(delta, {s["section_ref"]: s for s in batch.sections}, existing, next_revision)
        current = self.current_snapshot()
        private = [set(section["visibility"]) for section in batch.sections if "campaign" not in section["visibility"]]
        snapshot_visibility = sorted(set.intersection(*private)) if private else ["campaign"]
        previous_arc_visible = "campaign" in current.get("visibility", ["campaign"]) or batch.role in current.get("visibility", [])
        previous_arc_valid = previous_arc_visible and all(self.validate_source_ref(source, role=batch.role) for source in current.get("recent_arc_sources", []))
        snapshot = {**current, "schema_version": SNAPSHOT_VERSION,
                    "revision": next_revision, "base_revision": batch.base_revision,
                    "source_dependency_digest": batch.input_dependency_digest,
                    "generation": {"identity_digest": digest(draft["generation"]),
                                   "metadata_ref": proposal.artifact_ref},
                    "visibility": snapshot_visibility,
                    "recent_arc": delta["recent_arc"] or (current.get("recent_arc", "") if previous_arc_valid else ""),
                    "recent_arc_sources": [{"source_ref": section["source_ref"], "source_hash": section["source_hash"]}
                                           for section in batch.sections] if delta["recent_arc"] else (current.get("recent_arc_sources", []) if previous_arc_valid else []),
                    "coverage_manifest_ref": "sqlite:context_coverage"}
        working.update(merged)
        new_count = old_total + sum(int(item.get("active", True) and item.get("valid", True))
                                   - int(existing.get(item_id, {}).get("active", False) and existing.get(item_id, {}).get("valid", False))
                                   for item_id, item in merged.items())
        snapshot = self._bounded_view(snapshot, [item for item in working.values() if item.get("active") and item.get("valid")],
                                     12_000, total_count=new_count)
        if snapshot.get("delivery", {}).get("status") == "needs_context":
            return PublishResult("invalid", state["revision"], reason="snapshot_kernel_too_large")
        artifact = "" if delta["no_signal"] else self.persistence.immutable("snapshots", snapshot)
        if failpoint:
            failpoint("after_artifact_before_db")
        with self._publish_db() as db:
            now_state = dict(db.execute("SELECT * FROM context_state WHERE singleton=1").fetchone())
            if now_state["revision"] != batch.base_revision:
                return PublishResult("conflict", now_state["revision"], reason="base_revision_changed")
            if self._authority() != draft["authority_revision"] or now_state["source_epoch"] != batch.source_epoch:
                return PublishResult("stale", now_state["revision"], reason="native_authority_or_branch_changed")
            # Registry current heads may change while the background model is
            # running. Recheck every frozen source at publication, not only draft.
            if any(not self.validate_source_ref(section, role=batch.role) for section in batch.sections):
                return PublishResult("stale", now_state["revision"], reason="source_hash_changed_or_revoked")
            if maintenance_job:
                job = db.execute("SELECT * FROM context_maintenance_jobs WHERE input_digest=?", (maintenance_job[0],)).fetchone()
                if not job or job["status"] != "leased" or job["lease_owner"] != maintenance_job[1] or job["lease_until"] <= self.persistence.clock():
                    return PublishResult("stale", now_state["revision"], reason="maintenance_lease_expired")
            for section in batch.sections:
                db.execute("""INSERT INTO context_coverage VALUES(?,?,?,?,1,?,?,1)
                    ON CONFLICT(section_ref) DO UPDATE SET processed=1,
                    abstracted=MAX(context_coverage.abstracted,excluded.abstracted),revision=excluded.revision,valid=1""",
                           (section["section_ref"], section["block_id"], section["source_hash"], section["section_id"],
                            int(section["section_ref"] in abstracted), next_revision))
            if not delta["no_signal"]:
                # Only patched/superseded rows change. Old detail remains in its
                # indexed table without being rewritten or concatenated.
                for item in merged.values():
                    db.execute("INSERT INTO context_snapshot_items VALUES(?,?,?,?,?,?) ON CONFLICT(item_id) DO UPDATE SET item_json=excluded.item_json,revision=excluded.revision,active=excluded.active,valid=excluded.valid",
                               (item["item_id"], item["item_kind"], canonical(item), item["revision"], int(item.get("active", True)), int(item.get("valid", True))))
                db.execute("INSERT INTO context_snapshot_revisions VALUES(?,?,?,?,?)", (next_revision, batch.base_revision, artifact, batch.input_dependency_digest, self.persistence.clock()))
                db.execute("UPDATE context_state SET revision=?,snapshot_ref=? WHERE singleton=1 AND revision=?", (next_revision, artifact, batch.base_revision))
            outcome = "no_signal" if delta["no_signal"] else "published"
            db.execute("UPDATE context_proposals SET status=? WHERE proposal_id=?", (outcome, proposal.proposal_id))
            if maintenance_job:
                db.execute("UPDATE context_maintenance_jobs SET status=?,lease_owner=NULL,lease_until=0,result_ref=?,output_chars=? WHERE input_digest=?",
                           (outcome, artifact, len(canonical(snapshot)), maintenance_job[0]))
            if failpoint:
                failpoint("inside_publish_transaction")
        return PublishResult(outcome, next_revision, artifact,
                             processed_count=len(batch.sections), abstracted_count=len(abstracted))

    def is_abstracted(self, block_id: str, source_hash: str, section: str = "") -> bool:
        with self.persistence.db() as db:
            if section:
                rows = db.execute("SELECT abstracted,valid FROM context_coverage WHERE block_id=? AND source_hash=? AND section_id=?", (block_id, source_hash, section)).fetchall()
                if not rows and "#chars=" not in section:
                    # A plain field is covered only after every actual captured
                    # partition of that field was published, not after one page.
                    rows = db.execute("""SELECT COALESCE(c.abstracted,0),COALESCE(c.valid,0)
                        FROM context_closed_sections s LEFT JOIN context_coverage c ON c.section_ref=s.section_ref
                        WHERE s.block_id=? AND s.source_hash=? AND substr(s.section_id,1,?)=?""",
                        (block_id, source_hash, len(section + "#chars="), section + "#chars=")).fetchall()
            else:
                rows = db.execute("""SELECT COALESCE(c.abstracted,0),COALESCE(c.valid,0)
                    FROM context_closed_sections s LEFT JOIN context_coverage c ON c.section_ref=s.section_ref
                    WHERE s.block_id=? AND s.source_hash=?""", (block_id, source_hash)).fetchall()
        if not rows or not all(row[0] and row[1] for row in rows):
            return self._mirror_is_abstracted(block_id, source_hash, section)
        with self.persistence.db() as db:
            ref = db.execute("SELECT source_ref FROM context_closed_sections WHERE block_id=? AND source_hash=? LIMIT 1", (block_id, source_hash)).fetchone()
        return bool(ref and self.validate_source_ref({"source_ref": ref[0], "source_hash": source_hash}))

    def _mirror_is_abstracted(self, block_id: str, source_hash: str, section: str) -> bool:
        if not section:
            # A handful of known mirrored fields never cover a whole container
            # whose other sections may still contain unresolved/new material.
            return False
        with self.persistence.db() as db:
            alias = db.execute("SELECT * FROM context_mirror_sections WHERE mirror_block_id=? AND mirror_source_hash=? AND mirror_section_id=?",
                               (block_id, source_hash, section)).fetchone()
            if not alias:
                return False
            coverage = db.execute("SELECT c.abstracted,c.valid,s.source_ref FROM context_closed_sections s JOIN context_coverage c ON c.section_ref=s.section_ref WHERE s.block_id=? AND s.source_hash=? AND s.section_id=?",
                                  (alias["original_block_id"], alias["original_source_hash"], alias["original_section_id"])).fetchall()
        if not coverage or not all(row[0] and row[1] for row in coverage):
            return False
        binding = json.loads(alias["binding_json"])
        role = binding.get("role", "")
        return (self.validate_source_ref({"source_ref": alias["mirror_source_ref"], "source_hash": source_hash}, role=role)
                and self.validate_source_ref({"source_ref": alias["original_source_ref"], "source_hash": alias["original_source_hash"]}, role=role))

    def invalidate_stale_sources(self, *, role: str = "", rewind: bool = False) -> int:
        """Invalidate derived items/coverage without deleting original records."""
        with self.persistence.db() as db:
            refs = db.execute("SELECT DISTINCT source_ref,source_hash FROM context_closed_sections").fetchall()
            items = db.execute("SELECT item_id,item_json FROM context_snapshot_items WHERE valid=1").fetchall()
            previous_invalid = {row[0] for row in db.execute("SELECT source_ref FROM context_invalid_sources")}
        invalid = set()
        for row in refs:
            if rewind:
                invalid.add(row[0])
                continue
            if row[0] in previous_invalid:
                continue
            try:
                self.registry.get(row[0], role=role)
            except PermissionError:
                # Valid but invisible to this role is a delivery decision, not
                # source revocation or global retirement of another role's item.
                continue
            except (KeyError, ValueError, OSError):
                invalid.add(row[0])
                continue
            if not self.validate_source_ref({"source_ref": row[0], "source_hash": row[1]}, role=role):
                invalid.add(row[0])
        if not invalid and not rewind:
            return 0
        with self.persistence.db(immediate=True) as db:
            db.execute("UPDATE context_state SET source_epoch=source_epoch+1 WHERE singleton=1")
            for ref in invalid:
                db.execute("INSERT OR REPLACE INTO context_invalid_sources VALUES(?,?,?)", (ref, "branch_rewind" if rewind else "source_stale_or_revoked", self.persistence.clock()))
                db.execute("UPDATE context_coverage SET valid=0 WHERE section_ref IN (SELECT section_ref FROM context_closed_sections WHERE source_ref=?)", (ref,))
            affected = 0
            for row in items:
                body = json.loads(row[1])
                if any(source["source_ref"] in invalid for source in body.get("sources", [])):
                    body["valid"] = False
                    body["invalidated_reason"] = "branch_rewind" if rewind else "source_stale_or_revoked"
                    db.execute("UPDATE context_snapshot_items SET valid=0,item_json=? WHERE item_id=?", (canonical(body), row[0]))
                    affected += 1
        return affected

    def record_delivery(self, *, task_id: str, role: str, source_ref: str,
                        source_hash: str, scope: dict[str, Any], read: bool = False) -> str:
        if not self.validate_source_ref({"source_ref": source_ref, "source_hash": source_hash}, role=role):
            raise ValueError("cannot record delivery of stale source")
        return self.persistence.record_scope(role=role, task_id=task_id, source_ref=source_ref,
                                             source_hash=source_hash, scope=scope, kind="read" if read else "delivered")

    @staticmethod
    def _mirror_identity(block: SourceBlock, raw: Any) -> dict[str, Any]:
        identity = dict(block.identity)
        # Only typed producer wrappers are inspected, never arbitrary traversal
        # of arrays or a chance metric/string match elsewhere in the payload.
        keys = set(ORIGIN_IDENTITY) | set(MIRROR_SCIENCE_KEYS)
        def inspect(value: Any, depth: int = 0):
            if not isinstance(value, dict) or depth > 4:
                return
            for key in keys:
                if key in value:
                    if key in identity and canonical(identity[key]) != canonical(value[key]):
                        raise ValueError("mirror_identity_conflicts_with_original_payload:" + key)
                    identity[key] = value[key]
            for key in ("aggregate", "screening", "source_identity", "evaluation_identity", "result"):
                if isinstance(value.get(key), dict):
                    inspect(value[key], depth + 1)
        inspect(raw)
        return identity

    @staticmethod
    def _mirror_range(value: Any, span: Mapping[str, Any] | None) -> tuple[Any, bool, str]:
        if span is None:
            return value, True, ""
        if set(span) - {"start", "end", "unit"}:
            raise ValueError("mirror_range_unknown_field")
        unit = span.get("unit", "chars")
        text = canonical(value) if unit == "canonical_chars" else value
        if not isinstance(text, str) or unit not in {"chars", "canonical_chars"}:
            raise ValueError("mirror_range_requires_text")
        start, end = span.get("start", 0), span.get("end", len(text))
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("mirror_range_invalid")
        return text[start:end], start == 0 and end == len(text), f"#{unit}={start}:{end}"

    def bind_mirror(self, mirror: SourceBlock | Mapping[str, Any],
                    original: SourceBlock | Mapping[str, Any],
                    section_map: Mapping[str, str | Mapping[str, Any]], *, role: str = "",
                    mirror_ranges: Mapping[str, Mapping[str, Any]] | None = None,
                    original_ranges: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Register proven mirrors, without granting new coverage or read authority.

        Map a mirror JSON pointer to an actual closed ``section_id``. A mapping
        object may additionally specify ``original_pointer`` inside that full
        closed section and exact ``mirror_range``/``original_range``. A partial
        mirror is indexed only under its explicit range ID, never its full field.
        """
        if not 0 < len(section_map) <= 16:
            raise ValueError("mirror_mapping_must_be_bounded")
        mirror = mirror if isinstance(mirror, SourceBlock) else SourceBlock(**dict(mirror))
        original = original if isinstance(original, SourceBlock) else SourceBlock(**dict(original))
        if mirror.campaign_id != self.campaign_id or original.campaign_id != self.campaign_id:
            raise PermissionError("mirror_cannot_cross_campaigns")
        if mirror.source_ref == original.source_ref:
            raise ValueError("mirror_cycle_or_self_alias")
        if set(mirror.visibility) != set(original.visibility):
            raise PermissionError("mirror_visibility_cannot_widen_original_source")
        if not self.validate_source_ref(dataclasses.asdict(mirror), role=role) or not self.validate_source_ref(dataclasses.asdict(original), role=role):
            raise ValueError("mirror_source_stale_or_unreadable")
        mirror_raw = self.registry.load(self.registry.get(mirror.source_ref, role=role), role=role)
        original_raw = self.registry.load(self.registry.get(original.source_ref, role=role), role=role)
        mirror_identity = self._mirror_identity(mirror, mirror_raw)
        original_identity = self._mirror_identity(original, original_raw)
        origins = [key for key in ORIGIN_IDENTITY if original_identity.get(key) not in (None, "")]
        if not origins or not any(key in mirror_identity and canonical(mirror_identity[key]) == canonical(original_identity[key]) for key in origins):
            raise ValueError("mirror_requires_matching_immutable_origin_identity")
        for key in set(ORIGIN_IDENTITY) | set(MIRROR_SCIENCE_KEYS):
            if key in original_identity and (key not in mirror_identity or canonical(original_identity[key]) != canonical(mirror_identity[key])):
                raise ValueError("mirror_source_identity_or_science_scope_mismatch:" + key)
        scientific = "seed" in original_identity or any("development" in str(original_identity.get(key, ""))
                                                        for key in ("science_scope", "scope", "evaluation_scope"))
        if scientific:
            required_sets = (("candidate_id",), ("seed",), ("method_revision",),
                             ("protocol_hash", "execution_fingerprint", "evaluation_key"),
                             ("evaluator", "evaluator_hash"), ("science_scope", "scope", "evaluation_scope"))
            if any(not any(key in original_identity and original_identity[key] not in (None, "") for key in group)
                   for group in required_sets):
                raise ValueError("mirror_scientific_identity_incomplete")
        bindings = []
        with self.persistence.db() as db:
            # Aliases target actual captured originals, never another alias.
            native_sections = {row["section_id"]: json.loads(row["section_json"])
                               for row in db.execute("SELECT section_id,section_json FROM context_closed_sections WHERE source_ref=? AND source_hash=?",
                                                     (original.source_ref, original.source_hash))}
            alias_source = db.execute("SELECT 1 FROM context_mirror_sections WHERE mirror_source_ref=? LIMIT 1", (original.source_ref,)).fetchone()
        if alias_source:
            raise ValueError("mirror_alias_chain_forbidden")
        for mirror_pointer, spec in section_map.items():
            if not isinstance(mirror_pointer, str) or len(mirror_pointer) > 500:
                raise ValueError("mirror_pointer_invalid")
            details = {"section_id": spec} if isinstance(spec, str) else dict(spec)
            if set(details) - {"section_id", "original_pointer", "mirror_range", "original_range"}:
                raise ValueError("mirror_mapping_unknown_field")
            section_id = details.get("section_id")
            section = native_sections.get(section_id)
            if not section:
                raise ValueError("mirror_target_is_not_actual_closed_section")
            original_pointer = details.get("original_pointer", section["json_pointer"])
            parent_pointer = section["json_pointer"]
            if not (original_pointer == parent_pointer or parent_pointer == "" or original_pointer.startswith(parent_pointer + "/")):
                raise ValueError("mirror_pointer_outside_original_section")
            mirror_value = pointer_get(mirror_raw, mirror_pointer)
            original_value = pointer_get(original_raw, original_pointer)
            m_span = details.get("mirror_range", (mirror_ranges or {}).get(mirror_pointer))
            o_span = details.get("original_range", (original_ranges or {}).get(mirror_pointer))
            if not section["complete_section"]:
                native_range = section["selected_range"]
                if original_pointer != parent_pointer:
                    raise ValueError("mirror_partial_section_cannot_expand_original")
                o_span = o_span or {"unit": "canonical_chars", "start": native_range["start"], "end": native_range["end"]}
                if (o_span.get("unit") != "canonical_chars" or o_span.get("start", 0) < native_range["start"]
                        or o_span.get("end", len(canonical(original_value))) > native_range["end"]):
                    raise ValueError("mirror_range_outside_frozen_original_section")
            mirror_selected, full_mirror, suffix = self._mirror_range(mirror_value, m_span)
            original_selected, _, _ = self._mirror_range(original_value, o_span)
            if canonical(mirror_selected) != canonical(original_selected):
                raise ValueError("mirror_payload_or_range_mismatch")
            alias_section = mirror_pointer if full_mirror else mirror_pointer + suffix
            binding = {"mirror_pointer": mirror_pointer, "mirror_range": m_span,
                       "original_pointer": original_pointer, "original_range": o_span,
                       "original_section_id": section_id, "selected_hash": digest(original_selected),
                       "identity_digest": digest(original_identity), "role": role,
                       "satisfies_development_read_receipt": False, "creates_coverage": False}
            bindings.append((mirror.block_id, mirror.source_ref, mirror.source_hash, alias_section,
                             original.block_id, original.source_ref, original.source_hash, section_id,
                             canonical(binding), int(full_mirror), self.persistence.clock()))
        with self._publish_db() as db:
            if not self.validate_source_ref(dataclasses.asdict(mirror), role=role) or not self.validate_source_ref(dataclasses.asdict(original), role=role):
                raise ValueError("mirror_source_changed_before_bind")
            for binding in bindings:
                existing = db.execute("SELECT binding_json,original_source_ref FROM context_mirror_sections WHERE mirror_block_id=? AND mirror_source_hash=? AND mirror_section_id=?",
                                      (binding[0], binding[2], binding[3])).fetchone()
                if existing and (existing[0] != binding[8] or existing[1] != original.source_ref):
                    raise ValueError("mirror_mapping_conflicts_with_existing_binding")
                db.execute("INSERT OR IGNORE INTO context_mirror_sections VALUES(?,?,?,?,?,?,?,?,?,?,?)", binding)
        return {"status": "bound", "mapped": len(bindings), "creates_coverage": False,
                "satisfies_development_read_receipt": False}
