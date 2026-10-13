"""Campaign-only derived memory index, bounded delivery, and explicit maintenance CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.embedding import (
    DIMENSION, TEXT_BUILDER_VERSION, EmbeddingUnavailable, MemoryConfig,
    backend_for, cached_vector, canonical, content_hash, memory_config, vector_blob,
)
from react_agent.eeg_research.agentic.context.retrieval import (
    MAX_CANDIDATES, MAX_DELIVERY_CHARS, deduplicate_hits, prepare_query,
)

GROUPS = ("compatible_evidence", "analogy_only", "applicability_unknown")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def unsafe_source(value: Any) -> bool:
    from react_agent.eeg_research.agentic.handoffs import _PRIVATE_SCOPES

    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"role", "scope", "data_role", "partition", "modality", "namespace"} and isinstance(item, str):
                if item.lower() in _PRIVATE_SCOPES | {"test", "fmri", "eeg_task_validation"}:
                    return True
            if str(key).startswith(("final_test", "final_holdout", "test_result")) and key not in {"final_test_enabled", "final_test_accessed"}:
                return True
            if key in {"final_test_accessed", "final_test_enabled"} and item is True:
                return True
            if unsafe_source(item):
                return True
    elif isinstance(value, list):
        return any(unsafe_source(item) for item in value)
    return False


def unsafe_source_path(path: Path) -> bool:
    from react_agent.eeg_research.agentic.handoffs import _PRIVATE_SCOPES

    return any(part.lower() in _PRIVATE_SCOPES | {"fmri"}
               or any(part.lower().startswith(scope + suffix)
                      for scope in _PRIVATE_SCOPES for suffix in ("_", "-", "."))
               for part in path.parts)


def require_eeg_campaign(camp: Path) -> None:
    contract = read_json(camp / "evaluation_contract.json")
    protocol = read_json(camp / "execution_protocol.json")
    if contract.get("task_type") not in {"eeg_image_retrieval", "meg_image_retrieval"} and not str(protocol.get("schema_version", "")).startswith("eeg_research.execution_protocol."):
        raise ValueError("not_a_verified_eeg_campaign")
    if contract.get("task_type") == "fmri" or protocol.get("modality") == "fMRI":
        raise ValueError("not_a_verified_eeg_campaign")


class SidecarStore:
    """Short independent transactions; no model encoding occurs under a write lock."""

    def __init__(self, camp: Path):
        self.camp = Path(camp)
        self.namespace = self.camp.name
        require_eeg_campaign(self.camp)
        self.path = self.camp / "semantic_memory.sqlite"
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS skills (
                    namespace TEXT NOT NULL, skill_id TEXT NOT NULL, version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, origin_campaign_id TEXT NOT NULL,
                    payload TEXT NOT NULL, source_manifest TEXT NOT NULL,
                    validation_status TEXT NOT NULL, is_current INTEGER NOT NULL,
                    index_status TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(namespace, skill_id, version)
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    namespace TEXT NOT NULL, record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL, record_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL, text_hash TEXT NOT NULL, search_text TEXT NOT NULL,
                    embedding_fingerprint TEXT NOT NULL, model_id TEXT NOT NULL, model_revision TEXT NOT NULL,
                    text_builder_version TEXT NOT NULL, dimension INTEGER NOT NULL,
                    dtype TEXT NOT NULL, normalized INTEGER NOT NULL, vector BLOB NOT NULL,
                    encoding_metadata TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(namespace, record_type, record_id, record_version, text_hash, embedding_fingerprint)
                );
                CREATE TABLE IF NOT EXISTS frozen_delivery (
                    namespace TEXT NOT NULL, delivery_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(namespace, delivery_key)
                );
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def cached(self, record: dict, fingerprint: str, metadata: dict | None = None):
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM embedding_cache WHERE namespace=? AND record_type=?
                AND record_id=? AND text_hash=? AND embedding_fingerprint=? ORDER BY created_at DESC LIMIT 1""",
                (self.namespace, record["record_type"], record["record_id"], record["text_hash"], fingerprint)).fetchone()
        # Vectors represent text, not the full source's scientific revision.
        # Current source/filter validation happens before this lookup.
        if row is None:
            return None
        row = dict(row)
        try:
            if metadata and (row["model_id"] != metadata["model_id"] or row["model_revision"] != metadata["model_revision"]):
                return None
            if row["text_builder_version"] != TEXT_BUILDER_VERSION or row["text_hash"] != hashlib.sha256(row["search_text"].encode()).hexdigest():
                return None
            return cached_vector(row, fingerprint)
        except (ValueError, KeyError, TypeError):
            return None

    def save_vectors(self, records: list[dict], vectors, metadata: list[dict], backend) -> None:
        # Validate all bytes before beginning the transaction.
        blobs = [vector_blob(vec) for vec in vectors]
        if len(blobs) != len(records) or len(metadata) != len(records):
            raise ValueError("encoding_batch_identity_mismatch")
        with self.connect() as conn:
            for record, blob, encoding in zip(records, blobs, metadata):
                conn.execute("""INSERT OR REPLACE INTO embedding_cache VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    self.namespace, record["record_type"], record["record_id"], str(record["version"]),
                    record["content_hash"], record["text_hash"], record["search_text"], backend.fingerprint,
                    backend.metadata["model_id"], backend.metadata["model_revision"], TEXT_BUILDER_VERSION,
                    DIMENSION, "<f4", 1, sqlite3.Binary(blob), canonical(encoding), time.time(),
                ))
                if record["record_type"] == "skill":
                    conn.execute("UPDATE skills SET index_status='indexed' WHERE namespace=? AND skill_id=? AND version=?",
                                 (self.namespace, record["record_id"], record["version"]))

    def cached_encoding(self, record: dict, fingerprint: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT encoding_metadata FROM embedding_cache WHERE namespace=? AND record_type=? "
                "AND record_id=? AND text_hash=? AND embedding_fingerprint=? ORDER BY created_at DESC LIMIT 1",
                (self.namespace, record["record_type"], record["record_id"], record["text_hash"], fingerprint)).fetchone()
        return json.loads(row[0]) if row else None

    def freeze(self, key: str, build):
        with self.connect() as conn:
            row = conn.execute("SELECT payload, content_hash FROM frozen_delivery WHERE namespace=? AND delivery_key=?",
                               (self.namespace, key)).fetchone()
        if row:
            payload = json.loads(row["payload"])
            if content_hash(payload) != row["content_hash"]:
                raise ValueError("frozen_memory_hash_mismatch")
            if unsafe_source(payload):
                raise ValueError("frozen_memory_scope_forbidden")
            return payload
        payload = build()  # Includes encoding; always outside the transaction.
        if unsafe_source(payload):
            raise ValueError("frozen_memory_scope_forbidden")
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO frozen_delivery VALUES (?,?,?,?,?)",
                         (self.namespace, key, content_hash(payload), canonical(payload), time.time()))
            row = conn.execute("SELECT payload, content_hash FROM frozen_delivery WHERE namespace=? AND delivery_key=?",
                               (self.namespace, key)).fetchone()
        saved = json.loads(row["payload"])
        if content_hash(saved) != row["content_hash"]:
            raise ValueError("frozen_memory_hash_mismatch")
        if unsafe_source(saved):
            raise ValueError("frozen_memory_scope_forbidden")
        return saved


def episode_source(camp: Path, episode: dict) -> dict:
    from react_agent.eeg_research.agentic.artifacts import file_digest, resolve_verified_artifact

    if str(episode.get("schema_version", "")).startswith("eeg_research.memory_pipeline."):
        from react_agent.eeg_research.agentic.context.long_memory import pipeline_record_sources
        if unsafe_source(episode) or episode.get("origin_campaign_id") != camp.name:
            raise ValueError("episode_not_campaign_development")
        refs = pipeline_record_sources(camp, "episode", episode["episode_id"])
        if not refs:
            raise ValueError("pipeline_episode_sources_invalid")
        return {"episode_id": episode["episode_id"], "content_hash": content_hash(episode),
                "source_refs": refs, "scope": episode.get("scope"),
                "scientific_scope": episode.get("scientific_scope"),
                "interpretation_only": True, "replicated": False}
    method = episode.get("scope") == "method_development_benchmark"
    screening = episode.get("scope") == "two_fold_development"
    if unsafe_source(episode) or (episode.get("scope") != "development" and not method and not screening):
        raise ValueError("episode_not_development")
    if screening:
        from react_agent.eeg_research.agentic.memory import verified_screening_episode
        native = verified_screening_episode(camp, episode)
        return {"episode_id": episode["episode_id"], "content_hash": content_hash(episode),
                "artifact_path": str(Path(native["path"]).resolve()), "artifact_sha256": native["sha256"],
                "scope": "two_fold_development", "screening_subjects": episode["screening_subjects"],
                "selection_manifest_hash": episode["selection_manifest_hash"], "method_revision": episode["method_revision"],
                "science_scope": episode["science_scope"], "independent_seed_count": 1, "replicated": False}
    if method:
        from react_agent.eeg_research.agentic.memory import verified_method_episode
        native = verified_method_episode(camp, episode)
        return {"episode_id": episode["episode_id"], "content_hash": content_hash(episode),
                "artifact_path": str(Path(native["path"]).resolve()), "artifact_sha256": native["sha256"],
                "scope": "method_development_benchmark", "independent_seed_count": 1, "replicated": False}
    ref = str(episode.get("artifact") or "")
    if ref.startswith("art_"):
        row = resolve_verified_artifact(camp, ref)
        path = Path(row["path"])
    else:
        path = Path(ref)
        if not path.is_absolute():
            path = camp / path
    if not path.is_file() or not path.resolve().is_relative_to(camp.resolve()):
        raise ValueError("episode_artifact_unavailable")
    if unsafe_source(read_json(path)) or unsafe_source_path(path):
        raise ValueError("episode_artifact_not_development")
    return {"episode_id": episode["episode_id"], "content_hash": content_hash(episode),
            "artifact_path": str(path.resolve()), "artifact_sha256": file_digest(path)}


def build_context(camp: Path, *, state: dict | None = None, experiment: dict | None = None) -> dict:
    goal = read_json(camp / "goal.json")
    protocol = read_json(camp / "execution_protocol.json")
    contract = read_json(camp / "evaluation_contract.json")
    state = state if state is not None else read_json(camp / "campaign_state.json")
    experiment = experiment if experiment is not None else state.get("experiment") or {}
    context = {"task": goal.get("goal_id")}
    for key in ("dataset", "input_geometry", "feature_target", "evaluation_identity", "negative_sampling_policy"):
        value = protocol.get(key, contract.get(key))
        if key == "evaluation_identity" and value is None:
            value = contract.get("fingerprint") or protocol.get("validation_identity")
        if value not in (None, ""):
            context[key] = value
    for key in ("intervention", "fidelity"):
        value = experiment.get(key)
        if key == "fidelity" and value is None:
            value = experiment.get("initial_fidelity")
        if value not in (None, ""):
            context[key] = value
    dataset = str(protocol.get("dataset") or "").lower()
    if "meg" in dataset or contract.get("task_type") == "meg_image_retrieval":
        context["modality"] = "MEG"
    elif "eeg" in dataset or contract.get("task_type") == "eeg_image_retrieval":
        context["modality"] = "EEG"
    if protocol.get("exp_setting"):
        context["regime"] = protocol["exp_setting"]
    if goal.get("evaluation_schedule") == "screen_then_confirm":
        from react_agent.eeg_research.agentic.method_suite import manifest, validate_suite
        common = manifest(camp)
        selection = common["selection_manifest"]
        context.update(scope="method_development_benchmark" if state.get("phase") == "confirmation_10fold" else "two_fold_development",
                       screening_subjects=selection["subjects"], selection_manifest_hash=selection["selection_manifest_hash"], seed=0)
        suite_id = state.get("phase_target_suite") or state.get("active_suite")
        if not suite_id:
            target = state.get("candidate_id")
            row = next((item for item in reversed(state.get("evidence", [])) if item.get("candidate_id") == target and item.get("suite_id")), None)
            suite_id = row.get("suite_id") if row else None
        if suite_id:
            suite = validate_suite(camp, suite_id)
            context.update(method_revision=suite["method"]["method_revision"], training_recipe=suite["method"]["public_recipe"])
    return context


def build_query(camp: Path, *, state: dict | None = None, experiment: dict | None = None) -> str:
    from react_agent.eeg_research.agentic.research_progress import verified_diagnostic_facts
    from react_agent.eeg_research.agentic.research_plan import load_plan

    goal = read_json(camp / "goal.json")
    state = state if state is not None else read_json(camp / "campaign_state.json")
    experiment = experiment if experiment is not None else state.get("experiment") or {}
    hypothesis = experiment.get("hypothesis") or state.get("hypothesis")
    questions = []
    try:
        plan = load_plan(camp)
        version = plan.get("plan_version")
        if (plan.get("goal_id") != goal.get("goal_id") or not goal.get("goal_id")
                or not isinstance(version, int) or isinstance(version, bool) or version <= 0):
            plan = {}
        rows = plan.get("research_questions") or []
        if isinstance(rows, list):
            questions = [row for row in rows if isinstance(row, dict)
                         and isinstance(row.get("question_id"), str)
                         and isinstance(row.get("text"), str)
                         and row.get("status") in {"open", "inconclusive"} and not unsafe_source(row)]
    except (OSError, ValueError, TypeError):
        plan = {}
    active = next((qid for qid in (experiment.get("question_id"), state.get("active_question_id"),
                                  plan.get("active_question_id"))
                   if isinstance(qid, str) and any(row["question_id"] == qid for row in questions)), None)
    selected = [row for row in questions if row["question_id"] == active] if active else questions[:3]
    question_lines = list(dict.fromkeys(
        row["text"].strip()[:250] + (" — " + row["note"][:100] if isinstance(row.get("note"), str) else "")
        for row in selected if row["text"].strip()))
    sections = []
    if question_lines:
        label = "Current plan question" if active else "Unresolved plan questions (no active selection)"
        sections.append((label + ": " + "\n".join(question_lines))[:1100])
    sections.append(("Objective: " + str(goal.get("objective") or ""))[:800])
    parts = []
    if isinstance(hypothesis, str):
        parts.append(hypothesis)
    elif isinstance(hypothesis, dict):
        parts.extend(str(hypothesis[k]) for k in ("question", "statement", "mechanism", "prediction", "intervention") if hypothesis.get(k))
    if parts:
        sections.append(("Hypothesis: " + "\n".join(dict.fromkeys(parts)))[:1100])
    targets = {str(experiment[k]) for k in ("parent_candidate_id", "control_candidate_id") if experiment.get(k)}
    facts = verified_diagnostic_facts(camp, state, target_ids=targets or None, max_facts=3)
    diagnoses = []
    for fact in facts:
        diagnostic = {k: fact[k] for k in ("representation", "mean_direction_norm", "mean_margin", "realized_training") if fact.get(k) is not None}
        if diagnostic and not unsafe_source(diagnostic):
            diagnoses.append(canonical(diagnostic)[:210])
    if diagnoses:
        sections.append(("Verified development diagnosis: " + "\n".join(dict.fromkeys(diagnoses)))[:700])
    context = build_context(camp, state=state, experiment=experiment)
    scope = {key: context[key] for key in ("modality", "regime", "dataset") if key in context}
    if scope:
        sections.append(("Verified task scope: " + canonical(scope))[:200])
    action = state.get("last_action") or state.get("action") or ""
    error = state.get("error") or state.get("block_reason") or ""
    return prepare_query(question="\n".join(dict.fromkeys(section for section in sections if section.strip())),
        action=action, error=error, method=experiment, protocol=context,
        identities=[experiment.get("candidate_id"), state.get("active_question_id")])["text"]


def _search_text(kind: str, body: dict) -> str:
    if kind == "experiment_observation":
        from .experiment_observations import observation_search_text
        return observation_search_text(body)
    scope = body.get("conditions") if kind == "lesson" else body.get("preconditions")
    scope = scope or {}
    # The old goal_id is an applicability identity, not semantic text.
    readable = {k: scope[k] for k in ("modality", "regime", "dataset", "input_geometry", "feature_target", "intervention", "fidelity",
                                    "scope", "screening_subjects", "selection_manifest_hash", "method_revision", "seed") if scope.get(k) not in (None, "")}
    description = str(body.get("statement") or "") if kind == "lesson" else str(body["description_en"]) + "\n" + str(body["trigger_en"])
    return description + ("\nApplicable scope: " + canonical(readable) if readable else "")


def candidates(camp: Path, config: MemoryConfig, context: dict | None = None, *, query: str = "") -> list[dict]:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, query_lessons

    require_eeg_campaign(camp)
    context = context if context is not None else build_context(camp)
    store = EpisodeStore(camp)
    # Index navigation is bounded separately from the immutable full records.
    lesson_rows = store.list_lessons(limit=MAX_CANDIDATES)
    needed_ids = {ref for lesson in lesson_rows for ref in (lesson.get("supporting_episode_ids") or [])[:16]}
    conn = sqlite3.connect(camp / "memory.sqlite")
    try:
        episodes = {}
        for ref in sorted(needed_ids):
            row = conn.execute("SELECT payload FROM episodes WHERE episode_id=?", (ref,)).fetchone()
            if row:
                episode = json.loads(row[0]); episodes[episode["episode_id"]] = episode
        # Full-confirmation updates have a bounded navigation window as well.
        for ep in store.list_episodes(limit=MAX_CANDIDATES):
            episodes.setdefault(ep["episode_id"], ep)
    finally:
        conn.close()
    valid = []; sources = {}
    for lesson in lesson_rows:
        if unsafe_source(lesson) or lesson.get("origin_campaign_id", camp.name) != camp.name or lesson.get("is_current") is False or lesson.get("status") in {"disabled", "superseded", "retired"}:
            continue
        try:
            refs = lesson.get("supporting_episode_ids") or []
            if not refs or len(refs) > 16:
                continue
            if lesson.get("pipeline_schema_version"):
                from react_agent.eeg_research.agentic.context.long_memory import pipeline_record_sources
                manifest = pipeline_record_sources(camp, "lesson", lesson["lesson_id"])
                if not manifest:
                    continue
            else:
                manifest = [episode_source(camp, episodes[ref]) for ref in refs]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        valid.append(lesson); sources[lesson["lesson_id"]] = manifest
    rows = []
    updates = {}
    for ep in episodes.values():
        if ep.get("scope") != "method_development_benchmark" or not ep.get("updates_episode_ids"):
            continue
        try:
            source = episode_source(camp, ep)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for updated in ep["updates_episode_ids"]:
            updates.setdefault(updated, []).append({"episode_id": ep["episode_id"], "source": source,
                "scope": "method_development_benchmark", "verified_fold_coverage": 10,
                "relation": ep.get("screening_update_relation", "additional_coverage"),
                "scientific_strength": "greater_subject_coverage_same_frozen_method; development_selection_bias_remains"})
    for label, lessons in query_lessons(context, valid).items():
        for body in lessons:
            original = next(item for item in valid if item["lesson_id"] == body["lesson_id"])
            digest = content_hash(original)
            evidence_updates = [update for ref in body.get("supporting_episode_ids", []) for update in updates.get(ref, [])]
            if evidence_updates:
                # Delivery-only relationship; retain the original lesson/episode
                # bytes and expose the stronger native source beside them.
                body = {**body, "evidence_updates": evidence_updates}
            rows.append({"record_type": "lesson", "record_id": body["lesson_id"], "version": digest,
                         "content_hash": digest, "source_refs": sources[body["lesson_id"]] + [item["source"] for item in evidence_updates],
                         "applicability": label, "condition_differences": {k: body[k] for k in ("mismatch", "missing_conditions") if k in body},
                         "body": body})
    if config.skills_enabled:
        from react_agent.eeg_research.agentic.skill_memory import current_skills, skill_applicability
        for skill in current_skills(camp, limit=MAX_CANDIDATES):
            label, differences = skill_applicability(skill["payload"], context)
            rows.append({"record_type": "skill", "record_id": skill["skill_id"], "version": skill["version"],
                         "content_hash": skill["content_hash"], "source_refs": skill["source_manifest"],
                         "applicability": "procedural_hint", "precondition_status": label,
                         "condition_differences": differences, "body": skill["payload"]})
    from .research_execution import single_plan_enabled
    if single_plan_enabled(camp):
        from .experiment_observations import ExperimentObservationStore
        # Current cards precede old lesson navigation; no paid extraction needed.
        observation_store = ExperimentObservationStore(camp)
        observations = observation_store.records(limit=MAX_CANDIDATES, query=query, context=context) if query.strip() else []
        recent = observation_store.records(limit=16 if query.strip() else MAX_CANDIDATES, context=context)
        known = {row["record_id"] for row in observations}
        observations += [row for row in recent if row["record_id"] not in known]
        rows = observations + rows
    rows = rows[:MAX_CANDIDATES]
    for row in rows:
        row["search_text"] = _search_text(row["record_type"], row["body"])
        row["text_hash"] = hashlib.sha256(row["search_text"].encode()).hexdigest()
    return rows


def index_records(store: SidecarStore, records: list[dict], backend, *, rebuild: bool = False) -> dict:
    chunks = []
    for record in records[:MAX_CANDIDATES]:
        base = {k: v for k, v in record.items() if k != "embedding_chunks"}
        if hasattr(backend, "prepare_chunks"):
            parts = backend.prepare_chunks(record["search_text"], record_id=record["record_id"])
            record["embedding_chunks"] = [{**base, "search_text": part["text"], "text_hash": part["text_hash"],
                "version": part["chunk_id"], "encoding_scope": {k: v for k, v in part.items() if k != "text"}}
                for part in parts]
        else:
            # Legacy injected backends must still explicitly report encoding
            # scope. Production MiniLM always implements the tokenizer path.
            record["embedding_chunks"] = [base]
        chunks.extend(record["embedding_chunks"])
    pending = [row for row in chunks if rebuild or store.cached(row, backend.fingerprint, backend.metadata) is None]
    stats = {"indexed": 0, "reused": len(chunks) - len(pending), "pending": len(pending), "failed": 0}
    if pending:
        try:
            vectors, metadata = backend.encode([row["search_text"] for row in pending])
            metadata = [{**encoding, "representation_scope": row.get("encoding_scope", {"representation": "legacy_backend"})}
                        for row, encoding in zip(pending, metadata)]
            if any(encoding.get("truncated") for encoding in metadata):
                raise ValueError("implicit_embedding_truncation_forbidden")
            store.save_vectors(pending, vectors, metadata, backend)
            stats.update(indexed=len(pending), pending=0)
        except Exception as exc:
            stats.update(failed=len(pending), reason="index_failed:" + type(exc).__name__)
    return stats


def _delivery_fields(bundle: dict, *, standalone: bool) -> dict:
    """Serialize the exact added request fields, including the size statistic itself."""
    # The search query is derived from the current goal/plan, not a memory record.
    # Keep its full text in retrieval diagnostics; role fields carry its identity
    # without spending the whole-record budget on a second copy of that text.
    # Cache bookkeeping stays in diagnostics; the embedding fingerprint retains
    # builder identity in role fields. Record bodies and source refs stay whole.
    metadata = {key: copy.deepcopy(value) for key, value in bundle.items()
                if key not in {"lessons", "indexing", "text_builder_version"}}
    query = str(bundle.get("query") or "")
    query_identity = {"query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest()}
    if len(canonical({"query": query})) > len(canonical(query_identity)):
        metadata.pop("query", None)
        metadata.update(query_identity)
    fields = {"retrieved_memory": metadata}
    if bundle["backend"] == "minilm":
        if standalone:
            for item in metadata["items"]:
                if item["record_type"] == "lesson":
                    item["body"] = copy.deepcopy(next(
                        row for row in bundle["lessons"][item["applicability"]]
                        if row["lesson_id"] == item["record_id"]))
        else:
            fields["lessons"] = copy.deepcopy(bundle["lessons"])
    metadata["context_chars_used"] = 0
    while True:
        size = len(canonical(fields))
        if size == metadata["context_chars_used"]:
            return fields
        metadata["context_chars_used"] = size


def _fit_delivery(bundle: dict, *, standalone: bool = False) -> tuple[dict | None, dict]:
    """One whole-record budget rule for Planner and standalone role requests."""
    budget = min(bundle["context_chars_budget"], MAX_DELIVERY_CHARS)
    if budget <= 0 or bundle.get("delivery_available") is False or bundle.get("reason") == "disabled":
        return None, {}
    source_items = bundle["items"] if bundle["backend"] == "minilm" else []
    selected = copy.deepcopy(bundle)
    selected.update(items=[], lessons={key: [] for key in GROUPS}, hit_count=0)
    omitted = int(bundle.get("omitted_count", 0))

    def measure(candidate):
        candidate["omitted_count"] = omitted + len(source_items) - len(candidate["items"])
        candidate["truncated"] = bool(bundle.get("truncated") or candidate["omitted_count"])
        fields = _delivery_fields(candidate, standalone=standalone)
        candidate["context_chars_used"] = fields["retrieved_memory"]["context_chars_used"]
        return fields

    fields = measure(selected)
    if selected["context_chars_used"] > budget:
        return None, {}
    for item in source_items:
        trial = copy.deepcopy(selected)
        trial["items"].append(copy.deepcopy(item))
        if item["record_type"] == "lesson":
            body = next(row for row in bundle["lessons"][item["applicability"]]
                        if row["lesson_id"] == item["record_id"])
            trial["lessons"][item["applicability"]].append(copy.deepcopy(body))
        trial["hit_count"] = len(trial["items"])
        trial_fields = measure(trial)
        if trial["context_chars_used"] > budget:
            if item["record_type"] == "experiment_observation":
                from .experiment_observations import minimum_observation_card
                minimum = minimum_observation_card(item)
                # Native classification was computed against the retrieval's
                # complete current context; projection cannot upgrade it.
                minimum["applicability"] = item["applicability"]
                minimum["condition_differences"] = copy.deepcopy(item["condition_differences"])
                trial["items"][-1].update(minimum)
                trial_fields = measure(trial)
            # A long entry may still be useful through an explicitly partial
            # record card.  Preserve the original body in SQLite; this view
            # does not claim a full read or semantic coverage.
            partial = copy.deepcopy(trial["items"][-1].get("body") if item["record_type"] != "lesson" else body)
            selected_ranges = {}
            for key in ("statement", "description_en", "trigger_en", "summary_zh"):
                if isinstance(partial.get(key), str) and len(partial[key]) > 1200:
                    original = partial[key]
                    partial[key] = original[:1200]
                    selected_ranges[key] = {"char_range": [0, 1200], "full_chars": len(original),
                                            "full_text_hash": hashlib.sha256(original.encode()).hexdigest()}
            if selected_ranges:
                trial["items"][-1]["representation"] = {"kind": "partial_record_card", "selected_ranges": selected_ranges,
                    "full_record_id": item["record_id"], "full_record_hash": item["content_hash"],
                    "complete": False, "actual_read": False}
                if item["record_type"] != "lesson":
                    trial["items"][-1]["body"] = partial
                else:
                    trial["lessons"][item["applicability"]][-1] = partial
                trial_fields = measure(trial)
        if trial["context_chars_used"] <= budget:
            selected, fields = trial, trial_fields
    fields = measure(selected)
    return selected, fields


def _finalize_bundle(bundle: dict) -> dict:
    fitted, _ = _fit_delivery(bundle)
    if fitted is not None:
        return fitted
    return {**bundle, "items": [], "lessons": {key: [] for key in GROUPS}, "hit_count": 0,
            "context_chars_used": 0, "delivery_available": False,
            "truncated": bool(bundle.get("truncated") or bundle["items"]),
            "omitted_count": int(bundle.get("omitted_count", 0)) + len(bundle["items"])}


def retrieve(camp: Path, query: str | None = None, *, state: dict | None = None,
             experiment: dict | None = None, config: MemoryConfig | None = None,
             require_embedding: bool = False, rebuild: bool = False,
             role: str = "research_planner", context: dict | None = None) -> dict:
    config = config or memory_config(camp)
    query = build_query(camp, state=state, experiment=experiment) if query is None else prepare_query(question=query)["text"]
    bundle = {"backend": "legacy", "degraded": False, "reason": "disabled", "query": query[:800],
              "items": [], "lessons": {k: [] for k in GROUPS}, "candidate_count": 0, "hit_count": 0,
              "context_chars_budget": min(config.context_chars_budget, MAX_DELIVERY_CHARS), "truncated": False}
    if not config.enabled:
        if require_embedding:
            raise EmbeddingUnavailable("memory_disabled")
        return bundle
    try:
        require_eeg_campaign(camp)
        context = {**build_context(camp, state=state, experiment=experiment), **(context or {})}
        records = candidates(camp, config, context, query=query)
        from .research_execution import single_plan_enabled
        if single_plan_enabled(camp) and role == "research_planner":
            # Engineering procedures belong to Coder/Reviewer, not every round.
            records = [row for row in records if row["record_type"] != "skill"]
        bundle.update(candidate_count=len(records), indexing={"indexed": 0, "reused": 0, "pending": len(records), "failed": 0})
        backend = backend_for(config)
        store = SidecarStore(camp)
        stats = index_records(store, records, backend, rebuild=rebuild)
        bundle["indexing"] = stats
        if stats["failed"]:
            raise EmbeddingUnavailable(stats["reason"])
        bundle.update(backend="minilm", degraded=False, reason=None, model=backend.metadata["model_id"],
                      revision=backend.metadata["model_revision"], fingerprint=backend.fingerprint,
                      text_builder_version=TEXT_BUILDER_VERSION, indexing=stats,
                      candidate_count=len(records), language_limit="Original lesson language; no translation. MiniLM is English-focused.")
        if not query.strip() or config.top_k <= 0 or not records:
            return _finalize_bundle(bundle)
        query_card = backend.prepare_query(query) if hasattr(backend, "prepare_query") else prepare_query(question=query)
        query = query_card["text"]
        bundle["query_representation"] = query_card["metadata"]
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        query_record = {"record_type": "query", "record_id": "query_" + query_hash[:24], "version": TEXT_BUILDER_VERSION,
                        "content_hash": query_hash, "text_hash": query_hash, "search_text": query}
        query_vector = store.cached(query_record, backend.fingerprint, backend.metadata)
        query_encoding = store.cached_encoding(query_record, backend.fingerprint) if query_vector is not None else None
        if query_vector is None or query_encoding is None:
            query_vectors, query_metadata = backend.encode([query])
            query_vector = query_vectors[0]; query_encoding = query_metadata[0]
            if query_encoding.get("truncated"):
                raise EmbeddingUnavailable("implicit_query_truncation_forbidden")
            store.save_vectors([query_record], query_vectors, query_metadata, backend)
        vector_blob(query_vector)
        bundle["query_encoding"] = {k: v for k, v in query_encoding.items() if k != "encoded_text"}
        ranked = []
        priorities = {"compatible_evidence": 0, "preconditions_met": 1, "analogy_only": 2,
                      "preconditions_mismatch": 2, "applicability_unknown": 3, "preconditions_unknown": 3}
        for row in records:
            vectors = [store.cached(chunk, backend.fingerprint, backend.metadata)
                       for chunk in row.get("embedding_chunks", [row])]
            scores = [float(vector @ query_vector) for vector in vectors if vector is not None]
            if not scores:
                continue
            score = max(scores)
            if score >= config.min_similarity:
                ranked.append((priorities[row.get("precondition_status", row["applicability"])], -score,
                               (row["record_type"], row["record_id"], str(row["version"])), row, score))
        ranked.sort(key=lambda item: item[:3])
        bundle["matched_count"] = len(ranked)
        for _, _, _, row, score in ranked[:max(0, min(config.top_k, 32))]:
            item = {k: copy.deepcopy(v) for k, v in row.items() if k not in {"search_text", "text_hash", "body", "embedding_chunks"}}
            item["cosine_score"] = score
            if row["record_type"] in {"skill", "experiment_observation"}:
                item["body"] = copy.deepcopy(row["body"])
            bundle["items"].append(item)
            if row["record_type"] == "lesson":
                bundle["lessons"][row["applicability"]].append(copy.deepcopy(row["body"]))
        selected = deduplicate_hits(bundle["items"], role=role)
        selected_ids = {item["record_id"] for item in selected}
        bundle["omitted_count"] = len(bundle["items"]) - len(selected)
        bundle["items"] = selected
        for group in GROUPS:
            bundle["lessons"][group] = [item for item in bundle["lessons"][group] if item["lesson_id"] in selected_ids]
        bundle["hit_count"] = len(bundle["items"])
        return _finalize_bundle(bundle)
    except Exception as exc:
        if require_embedding:
            raise EmbeddingUnavailable(str(exc)) from exc
        # Do not pretend structured applicability is a lexical/embedding hit.
        bundle.update(backend="legacy", degraded=True, reason=str(exc) if isinstance(exc, EmbeddingUnavailable) else type(exc).__name__,
                      items=[], lessons={key: [] for key in GROUPS}, hit_count=0)
        return _finalize_bundle(bundle)


def apply_memory(request: dict, bundle: dict) -> dict:
    result = copy.deepcopy(request)
    _, fields = _fit_delivery(bundle)
    result.update(fields)
    return result


def role_memory(camp: Path, *, state: dict | None = None, experiment: dict | None = None) -> dict | None:
    config = memory_config(camp)
    return retrieve(camp, state=state, experiment=experiment, config=config) if config.enabled else None


def standalone_delivery(bundle: dict) -> dict | None:
    """Designer/coder lack the planner's legacy lessons view: include each body once."""
    _, fields = _fit_delivery(bundle, standalone=True)
    return fields.get("retrieved_memory")


def coder_delivery(camp: Path, workspace: Path, attempt_id: str, spec: dict, *, state: dict,
                   previous_spec: dict, previous_attempt_id: str | None) -> dict | None:
    config = memory_config(camp)
    if not config.enabled:
        return None
    spec_hash = (spec.get("experiment") or {}).get("spec_hash")
    def build():
        # A pre-feature in-flight attempt retains its original lack of semantic input.
        existing = previous_attempt_id == attempt_id
        if existing and "retrieved_memory" in previous_spec:
            body = previous_spec["retrieved_memory"]
        elif existing and ((workspace / "coder_log.jsonl").is_file() or (workspace / "implementation.json").is_file()):
            body = None
        else:
            bundle = role_memory(camp, state=state, experiment=spec["experiment"])
            body = standalone_delivery(bundle) if bundle else None
        return {"spec_hash": spec_hash, "retrieved_memory": body}
    frozen = SidecarStore(camp).freeze("coder:" + workspace.name + ":" + attempt_id, build)
    if not isinstance(frozen, dict) or "spec_hash" not in frozen or "retrieved_memory" not in frozen:
        raise ValueError("frozen_memory_payload_invalid")
    if frozen["spec_hash"] != spec_hash:
        raise ValueError("frozen_memory_experiment_changed")
    body = frozen["retrieved_memory"]
    if body is not None:
        if not isinstance(body, dict):
            raise ValueError("frozen_memory_payload_invalid")
        stored_budget = body.get("context_chars_budget", config.context_chars_budget)
        if not isinstance(stored_budget, int) or isinstance(stored_budget, bool) or stored_budget < 0:
            raise ValueError("frozen_memory_budget_invalid")
        size = len(canonical({"retrieved_memory": body}))
        if size > min(config.context_chars_budget, stored_budget):
            raise ValueError(f"frozen_memory_budget_exceeded:{size}")
        if not isinstance(body.get("items"), list):
            raise ValueError("frozen_memory_payload_invalid")
    return frozen["retrieved_memory"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index", "backfill", "rebuild", "query"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--require-embedding", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--model-revision")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = read_json(args.campaign / "goal.json").get("memory") or {}
        config = MemoryConfig.model_validate({**raw, "enabled": True, "allow_model_download": args.allow_model_download})
        if args.model_revision:
            config = config.model_copy(update={"model_revision": args.model_revision})
        if args.retry:
            backend_for(config, retry=True)
        result = retrieve(args.campaign, args.text if args.command == "query" else "",
                          config=config, require_embedding=args.require_embedding, rebuild=args.command == "rebuild")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (EmbeddingUnavailable, ValueError) as exc:
        print(json.dumps({"backend": "unavailable", "error": str(exc), "require_embedding": args.require_embedding}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
