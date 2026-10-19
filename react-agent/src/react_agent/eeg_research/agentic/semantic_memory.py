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
    MiniLMBackend, backend_for, cached_vector, canonical, content_hash, memory_config, vector_blob,
)
from react_agent.eeg_research.agentic.context.retrieval import (
    MAX_CANDIDATES, prepare_query, scientific_support_key,
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


def planner_query(camp: Path, observation: dict) -> str:
    """Prioritize the current question, mechanism and latest native feedback."""
    from .context.retrieval import prepare_query
    from .experiment_observations import ExperimentObservationStore

    selected = observation.get("selected_research_task") or (observation.get("selected_plan") or {}).get("proposal") or {}
    experiment = observation.get("experiment") or {}
    hypothesis = experiment.get("hypothesis") or observation.get("hypothesis") or {}
    terminal = observation.get("terminal_observation_id") or (observation.get("selected_plan") or {}).get("terminal_observation_id")
    original = {}
    if terminal:
        # Use the same verified native source as observation delivery, rather
        # than letting a generic objective displace the just-completed question.
        records = ExperimentObservationStore(Path(camp)).records(observation_id=str(terminal), limit=1)
        original = records[0]["body"] if records else {}
    question = (observation.get("current_question") or selected.get("question")
                or original.get("question")
                or (hypothesis.get("question") if isinstance(hypothesis, dict) else None)
                or build_query(camp, state=observation, experiment=experiment))
    mechanism = (hypothesis.get("mechanism") or hypothesis.get("intended_intervention")
                 if isinstance(hypothesis, dict) else hypothesis) or selected.get("scientific_intent") or experiment.get("intervention") or original.get("intervention")
    feedback = ""
    if original:
        failure = original.get("engineering_failure") or {}
        facts = original.get("native_facts") or {}
        parts = ["status=" + str(original.get("execution_status"))]
        if failure.get("error_category"):
            parts.append("category=" + str(failure["error_category"])[:60])
        issue = next((row for row in failure.get("blocking_issues") or [] if isinstance(row, dict)), {})
        parts.extend(key + "=" + str(issue[key])[:80]
                     for key in ("requirement_id", "error_class", "kind") if issue.get(key))
        if facts.get("metrics"):
            parts.append(canonical({key: facts[key] for key in ("metrics", "paired_delta_pp", "seed", "coverage") if key in facts})[:180])
        feedback = " ".join(parts)
    # retrieve() subsequently applies a 600-character encoding view. Allocate
    # that view here so feedback and mechanism survive even a huge question.
    sections = [("Feedback", feedback, 220), ("Question", question, 180), ("Mechanism", mechanism, 140)]
    text = "\n".join(label + ": " + value.strip()[:limit]
                     for label, value, limit in sections if isinstance(value, str) and value.strip())
    return prepare_query(question=text, max_chars=600)["text"]


def _search_text(kind: str, body: dict) -> str:
    if kind == "experiment_observation":
        from .experiment_observations import observation_search_text
        return observation_search_text(body)
    if kind == "knowledge":
        return "\n".join(str(body[key]) for key in ("title", "name", "method", "statement", "mechanism", "description", "summary", "summary_zh", "idea") if body.get(key))
    scope = body.get("conditions") if kind == "lesson" else body.get("preconditions")
    scope = scope or {}
    # The old goal_id is an applicability identity, not semantic text.
    readable = {k: scope[k] for k in ("modality", "regime", "dataset", "input_geometry", "feature_target", "intervention", "fidelity",
                                    "scope", "screening_subjects", "selection_manifest_hash", "method_revision", "seed") if scope.get(k) not in (None, "")}
    description = str(body.get("statement") or "") if kind == "lesson" else str(body["description_en"]) + "\n" + str(body["trigger_en"])
    return description + ("\nApplicable scope: " + canonical(readable) if readable else "")


def registered_knowledge(camp: Path, context: dict) -> list[dict]:
    """Index existing native librarian artifacts; never invent a lesson or paper."""
    from .artifacts import REGISTRY, resolve_verified_artifact
    from .context.sources import ContextSourceRegistry
    from .experiment_observations import observation_applicability
    path = Path(camp) / REGISTRY
    if not path.is_file():
        return []
    registry = ContextSourceRegistry(camp)
    rows, seen = [], set()
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if len(rows) >= MAX_CANDIDATES:
            break
        try:
            item = json.loads(line)
            if item.get("kind") not in {"method_hits", "method_evidence_packet"} or item.get("artifact_id") in seen:
                continue
            seen.add(item["artifact_id"])
            native = resolve_verified_artifact(camp, item["artifact_id"])
            original = read_json(Path(native["path"]))
            if original.get("status") != "completed" or unsafe_source(original):
                continue
            payload = original.get("payload") if isinstance(original.get("payload"), dict) else original
            cards = payload.get("method_cards") or original.get("method_cards") or []
            for index, body in enumerate(cards[:MAX_CANDIDATES - len(rows)]):
                if not isinstance(body, dict) or unsafe_source(body) or not _search_text("knowledge", body).strip():
                    continue
                rid = "knowledge:" + item["artifact_id"] + ":" + str(body.get("id") or index)
                source = registry.capture(body, block_id="semantic-knowledge:" + rid,
                    kind="memory", role="research_planner", identity={"record_id": rid, "record_type": "knowledge"})
                label, differences = observation_applicability(body.get("conditions") or body.get("preconditions") or {}, context)
                rows.append({"record_type": "knowledge", "record_id": rid, "version": content_hash(body),
                    "content_hash": content_hash(body), "body": copy.deepcopy(body),
                    "source_refs": [{"artifact_id": native["artifact_id"], "path": native["path"], "sha256": native["sha256"]}],
                    "record_source_ref": source.source_ref, "record_source_hash": source.source_hash,
                    "applicability": label, "condition_differences": differences,
                    "source_retrieval": "native_registered_local_method_card"})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def embedding_health(bundle: dict) -> dict:
    """Keep actual encoding diagnostics separate from optional hit packing."""
    keys = ("backend", "degraded", "reason", "model", "revision", "fingerprint", "runtime_class",
            "dimension", "query", "query_sha256", "query_representation", "query_encoding", "query_status",
            "indexing", "candidate_count", "matched_count", "hit_count", "language_limit")
    return {key: copy.deepcopy(bundle.get(key)) for key in keys if key in bundle}


def _embedding_ready(bundle: dict) -> bool:
    return not bundle.get("degraded") and bool(bundle.get("fingerprint"))


def candidates(camp: Path, config: MemoryConfig, context: dict | None = None, *, query: str = "",
               record_types: frozenset[str] | None = None) -> list[dict]:
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
            from .context.sources import ContextSourceRegistry
            source = ContextSourceRegistry(camp).capture(original, block_id="semantic-lesson:" + body["lesson_id"],
                kind="memory", role="research_planner", identity={"record_id": body["lesson_id"], "record_type": "lesson"})
            evidence_updates = [update for ref in body.get("supporting_episode_ids", []) for update in updates.get(ref, [])]
            if evidence_updates:
                # Delivery-only relationship; retain the original lesson/episode
                # bytes and expose the stronger native source beside them.
                body = {**body, "evidence_updates": evidence_updates}
            rows.append({"record_type": "lesson", "record_id": body["lesson_id"], "version": digest,
                         "content_hash": digest, "source_refs": sources[body["lesson_id"]] + [item["source"] for item in evidence_updates],
                         "record_source_ref": source.source_ref, "record_source_hash": source.source_hash,
                         "applicability": label, "condition_differences": {k: body[k] for k in ("mismatch", "missing_conditions") if k in body},
                         "body": body})
    rows += registered_knowledge(camp, context)
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
    if record_types is not None:
        rows = [row for row in rows if row["record_type"] in record_types]
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


def _delivery_fields(bundle: dict, *, standalone: bool = False) -> dict:
    """Model-visible cards only; retrieval and selection diagnostics stay local."""
    return {"retrieved_memory": {
        "status": ("retrieval_unavailable" if bundle.get("degraded") else
                   "ready" if bundle.get("items") else "no_memory_selected"),
        "items": copy.deepcopy(bundle.get("items") or []),
    }}


def _fit_delivery(bundle: dict, *, standalone: bool = False) -> tuple[dict | None, dict]:
    """Project before selecting whole cards; never charge the retrieval envelope."""
    from .memory_delivery import project_memory_card, select_memory_cards

    if bundle.get("delivery_available") is False or bundle.get("reason") == "disabled":
        return None, {}
    cards = []
    if _embedding_ready(bundle):
        for item in bundle.get("items") or []:
            record = copy.deepcopy(item)
            if record.get("record_type") == "lesson" and "body" not in record:
                record["body"] = copy.deepcopy(next(row for row in
                    bundle.get("lessons", {}).get(record["applicability"], [])
                    if row["lesson_id"] == record["record_id"]))
            cards.append(project_memory_card(record, role=bundle.get("role", "research_planner")))
    dedup_keys = {(item["record_type"], item["record_id"]): scientific_support_key(item)
                  for item in bundle.get("items") or []}
    selection = select_memory_cards(cards, content_chars_budget=bundle["content_chars_budget"], dedup_keys=dedup_keys)
    selected = {**bundle, "items": selection.cards, "delivery_report": selection.report,
                "selected_count": len(selection.cards)}
    return selected, _delivery_fields(selected, standalone=standalone)


def record_delivery_report(camp: Path, role: str, report: dict, *, retrieval: dict | None = None) -> str:
    """Persist a local delivery decision without making diagnostics model input."""
    from .context.sources import atomic_immutable

    payload = {"role": role, "selection": copy.deepcopy(report)}
    if retrieval is not None:
        payload["retrieval"] = embedding_health(retrieval)
    path = Path(camp) / "memory_deliveries" / (content_hash(payload) + ".json")
    atomic_immutable(path, canonical(payload))
    return str(path)


def _finalize_bundle(bundle: dict) -> dict:
    """Finalize local ranked candidates, before any role content selection."""
    bundle["retrieval_hit_count"] = len(bundle.get("items") or [])
    return bundle


def retrieve(camp: Path, query: str | None = None, *, state: dict | None = None,
             experiment: dict | None = None, config: MemoryConfig | None = None,
             require_embedding: bool = False, rebuild: bool = False,
             role: str = "research_planner", context: dict | None = None,
             record_types: frozenset[str] | None = None) -> dict:
    config = config or memory_config(camp)
    query = build_query(camp, state=state, experiment=experiment) if query is None else prepare_query(question=query)["text"]
    bundle = {"backend": "disabled", "degraded": False, "reason": "disabled", "query": query,
              "items": [], "lessons": {k: [] for k in GROUPS}, "candidate_count": 0, "hit_count": 0,
              "model": config.model_id, "revision": config.model_revision, "fingerprint": None,
              "indexing": {"indexed": 0, "reused": 0, "pending": 0, "failed": 0}, "query_status": "not_encoded",
              "content_chars_budget": config.content_chars_budget, "role": role, "truncated": False}
    if not config.enabled:
        if require_embedding:
            raise EmbeddingUnavailable("memory_disabled")
        return bundle
    try:
        require_eeg_campaign(camp)
        context = {**build_context(camp, state=state, experiment=experiment), **(context or {})}
        records = candidates(camp, config, context, query=query, record_types=record_types)
        from .research_execution import single_plan_enabled
        if single_plan_enabled(camp) and role == "research_planner":
            # Engineering procedures belong to Coder/Reviewer, not every round.
            records = [row for row in records if row["record_type"] != "skill"]
        bundle.update(candidate_count=len(records), indexing={"indexed": 0, "reused": 0, "pending": len(records), "failed": 0})
        backend = backend_for(config)
        bundle.update(backend="minilm" if isinstance(backend, MiniLMBackend) else "synthetic_embedding",
                      runtime_class=type(backend).__module__ + "." + type(backend).__qualname__,
                      model=backend.metadata["model_id"], revision=backend.metadata["model_revision"],
                      fingerprint=backend.fingerprint, dimension=DIMENSION)
        store = SidecarStore(camp)
        stats = index_records(store, records, backend, rebuild=rebuild)
        bundle["indexing"] = stats
        if stats["failed"]:
            raise EmbeddingUnavailable(stats["reason"])
        bundle.update(degraded=False, reason=None, model=backend.metadata["model_id"],
                      revision=backend.metadata["model_revision"], fingerprint=backend.fingerprint,
                      text_builder_version=TEXT_BUILDER_VERSION, indexing=stats,
                      candidate_count=len(records), language_limit="Original lesson language; no translation. MiniLM is English-focused.")
        if not query.strip():
            bundle["query_status"] = "empty_index_only"
            return _finalize_bundle(bundle)
        query_card = backend.prepare_query(query) if hasattr(backend, "prepare_query") else prepare_query(question=query)
        query = query_card["text"]
        bundle["query"] = query
        bundle["query_sha256"] = hashlib.sha256(query.encode()).hexdigest()
        bundle["query_representation"] = query_card["metadata"]
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        query_record = {"record_type": "query", "record_id": "query_" + query_hash[:24], "version": TEXT_BUILDER_VERSION,
                        "content_hash": query_hash, "text_hash": query_hash, "search_text": query}
        query_vector = store.cached(query_record, backend.fingerprint, backend.metadata)
        query_encoding = store.cached_encoding(query_record, backend.fingerprint) if query_vector is not None else None
        bundle["query_status"] = "cache_reused"
        if query_vector is None or query_encoding is None:
            bundle["query_status"] = "encoding"
            query_vectors, query_metadata = backend.encode([query])
            query_vector = query_vectors[0]; query_encoding = query_metadata[0]
            if query_encoding.get("truncated"):
                raise EmbeddingUnavailable("implicit_query_truncation_forbidden")
            store.save_vectors([query_record], query_vectors, query_metadata, backend)
            bundle["query_status"] = "encoded"
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
            item["body"] = copy.deepcopy(row["body"])
            bundle["items"].append(item)
            if row["record_type"] == "lesson":
                bundle["lessons"][row["applicability"]].append(copy.deepcopy(row["body"]))
        # Only duplicate record identities may disappear before content fitting.
        # A large representative must not hide a smaller whole card sharing its
        # native support; support deduplication follows successful selection.
        seen_records = set()
        selected = []
        for item in bundle["items"]:
            identity = (item["record_type"], item["record_id"])
            if identity not in seen_records:
                seen_records.add(identity)
                selected.append(item)
        selected_ids = {item["record_id"] for item in selected}
        bundle["omitted_count"] = len(bundle["items"]) - len(selected)
        bundle["items"] = selected
        for group in GROUPS:
            bundle["lessons"][group] = [item for item in bundle["lessons"][group] if item["lesson_id"] in selected_ids]
        bundle["hit_count"] = len(bundle["items"])
        return _finalize_bundle(bundle)
    except Exception as exc:
        bundle.update(backend="unavailable", degraded=True,
                      reason=str(exc) if isinstance(exc, EmbeddingUnavailable) else type(exc).__name__,
                      items=[], lessons={key: [] for key in GROUPS}, hit_count=0, query_status="failed")
        if require_embedding:
            failure = EmbeddingUnavailable(str(exc))
            failure.retrieval_health = embedding_health(bundle)
            raise failure from exc
        # Do not pretend structured applicability is a lexical/embedding hit.
        return _finalize_bundle(bundle)


def planner_knowledge_delivery(camp: Path, bundle: dict, *, context: dict | None = None) -> dict:
    """Build complete prior cards; the Planner selects all history together."""
    from .context.sources import ContextSourceRegistry
    from .context.representations import handle
    from .memory_delivery import project_memory_card

    result = {"status": "no_related_knowledge", "items": [], "omitted": [], "omitted_count": 0,
              "scientific_evidence": False, "execution_authority": False,
              "content_chars_budget": int(bundle.get("content_chars_budget") or 0)}
    if not _embedding_ready(bundle):
        result["status"] = "embedding_unavailable" if bundle.get("degraded") else "disabled"
        return result
    registry = ContextSourceRegistry(camp)
    items = [item for item in bundle.get("items") or [] if item.get("record_type") in {"lesson", "knowledge"}]
    for item in items[:32]:
        try:
            source = registry.get(item["record_source_ref"], role="research_planner")
            original = registry.load(source, role="research_planner")
            if source.source_hash != item["record_source_hash"] or content_hash(original) != item["content_hash"]:
                raise ValueError("knowledge_original_identity_changed")
            text = original.get("statement") if item["record_type"] == "lesson" else _search_text("knowledge", original)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("knowledge_body_missing")
            # The candidate may carry a native coverage update beside the
            # original lesson; retain it without changing its source identity.
            record = {**item, "body": copy.deepcopy(item.get("body") or original)}
            card = project_memory_card(record, role="research_planner", context=context)
            card.update(knowledge_role="prior_or_analogy", scientific_evidence=False, execution_authority=False)
            # Native navigation stays on this local intermediate object. The
            # shared selector projects it out before constructing model input.
            card["full_record"] = handle(source)
            result["items"].append(card)
            result["status"] = "ready"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["omitted_count"] += 1
            result["omitted"].append({"record_id": item.get("record_id"), "reason": "knowledge_source_unreadable", "detail": str(exc)[:120]})
    return result


def _selected_delivery(bundle: dict) -> dict:
    cached = bundle.get("_delivery_fields")
    if cached is not None:
        return copy.deepcopy(cached)
    _, fields = _fit_delivery(bundle, standalone=True)
    return fields


def apply_memory(request: dict, bundle: dict) -> dict:
    result = copy.deepcopy(request)
    result.update(_selected_delivery(bundle))
    return result


def role_memory(camp: Path, *, state: dict | None = None, experiment: dict | None = None,
                role: str = "research_planner", record_types: frozenset[str] | None = None) -> dict | None:
    config = memory_config(camp)
    if not config.enabled:
        return None
    bundle = retrieve(camp, state=state, experiment=experiment, config=config, role=role, record_types=record_types)
    selected, fields = _fit_delivery(bundle, standalone=True)
    bundle["_delivery_fields"] = fields
    if selected is not None:
        bundle["_delivery_report"] = selected["delivery_report"]
        bundle["_delivery_report_ref"] = record_delivery_report(camp, role, selected["delivery_report"], retrieval=bundle)
    return bundle


def standalone_delivery(bundle: dict) -> dict | None:
    """Attach an already selected role view, with every body appearing once."""
    return _selected_delivery(bundle).get("retrieved_memory")


def coder_delivery(camp: Path, workspace: Path, attempt_id: str, spec: dict, *, state: dict,
                   previous_spec: dict, previous_attempt_id: str | None) -> dict | None:
    from .memory_delivery import PROJECTION_VERSION, memory_content_chars

    config = memory_config(camp)
    if not config.enabled:
        return None
    spec_hash = (spec.get("experiment") or {}).get("spec_hash")
    def build():
        bundle = role_memory(camp, state=state, experiment=spec["experiment"], role="candidate_coder")
        body = standalone_delivery(bundle) if bundle else None
        return {"spec_hash": spec_hash, "retrieved_memory": body,
                "projection_version": PROJECTION_VERSION, "content_chars_budget": config.content_chars_budget,
                "delivery_report": copy.deepcopy((bundle or {}).get("_delivery_report"))}
    frozen = SidecarStore(camp).freeze("coder:" + workspace.name + ":" + attempt_id, build)
    if not isinstance(frozen, dict) or "spec_hash" not in frozen or "retrieved_memory" not in frozen:
        raise ValueError("frozen_memory_payload_invalid")
    if frozen["spec_hash"] != spec_hash:
        raise ValueError("frozen_memory_experiment_changed")
    if frozen.get("projection_version") != PROJECTION_VERSION:
        raise ValueError("frozen_memory_projection_changed")
    body = frozen["retrieved_memory"]
    if body is not None:
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            raise ValueError("frozen_memory_payload_invalid")
        stored_budget = frozen["content_chars_budget"]
        if not isinstance(stored_budget, int) or isinstance(stored_budget, bool) or stored_budget < 0:
            raise ValueError("frozen_memory_budget_invalid")
        size = sum(memory_content_chars(card, role="candidate_coder") for card in body["items"])
        if size > stored_budget:
            raise ValueError(f"frozen_memory_budget_exceeded:{size}")
    return body


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
        print(json.dumps({"backend": "unavailable", "error": str(exc), "require_embedding": args.require_embedding,
                          "embedding_health": getattr(exc, "retrieval_health", {})}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
