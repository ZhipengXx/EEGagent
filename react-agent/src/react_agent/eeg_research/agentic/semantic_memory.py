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

GROUPS = ("compatible_evidence", "analogy_only", "applicability_unknown")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def unsafe_source(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"scope", "data_role", "partition", "modality", "namespace"} and isinstance(item, str):
                if item.lower() in {"final_test", "final_holdout", "test", "fmri", "eeg_task_validation"}:
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
                AND record_id=? AND record_version=? AND text_hash=? AND embedding_fingerprint=?""",
                (self.namespace, record["record_type"], record["record_id"], str(record["version"]), record["text_hash"], fingerprint)).fetchone()
        if row is None or row["content_hash"] != record["content_hash"]:
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

    def freeze(self, key: str, build):
        with self.connect() as conn:
            row = conn.execute("SELECT payload, content_hash FROM frozen_delivery WHERE namespace=? AND delivery_key=?",
                               (self.namespace, key)).fetchone()
        if row:
            payload = json.loads(row["payload"])
            if content_hash(payload) != row["content_hash"]:
                raise ValueError("frozen_memory_hash_mismatch")
            return payload
        payload = build()  # Includes encoding; always outside the transaction.
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO frozen_delivery VALUES (?,?,?,?,?)",
                         (self.namespace, key, content_hash(payload), canonical(payload), time.time()))
            row = conn.execute("SELECT payload, content_hash FROM frozen_delivery WHERE namespace=? AND delivery_key=?",
                               (self.namespace, key)).fetchone()
        saved = json.loads(row["payload"])
        if content_hash(saved) != row["content_hash"]:
            raise ValueError("frozen_memory_hash_mismatch")
        return saved


def episode_source(camp: Path, episode: dict) -> dict:
    from react_agent.eeg_research.agentic.artifacts import file_digest, resolve_verified_artifact

    if unsafe_source(episode) or episode.get("scope") != "development":
        raise ValueError("episode_not_development")
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
    if unsafe_source(read_json(path)) or any(part.lower() in {"fmri", "final_test", "final_holdout"} for part in path.parts):
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
    return context


def build_query(camp: Path, *, state: dict | None = None, experiment: dict | None = None) -> str:
    from react_agent.eeg_research.agentic.research_progress import verified_diagnostic_facts

    goal = read_json(camp / "goal.json")
    state = state if state is not None else read_json(camp / "campaign_state.json")
    experiment = experiment if experiment is not None else state.get("experiment") or {}
    hypothesis = experiment.get("hypothesis") or state.get("hypothesis")
    parts = [str(goal.get("objective") or "")]
    if isinstance(hypothesis, str):
        parts.append(hypothesis)
    elif isinstance(hypothesis, dict):
        parts.extend(str(hypothesis[k]) for k in ("question", "statement", "mechanism", "prediction", "intervention") if hypothesis.get(k))
    targets = {str(experiment[k]) for k in ("parent_candidate_id", "control_candidate_id") if experiment.get(k)}
    facts = verified_diagnostic_facts(camp, state, target_ids=targets or None, max_facts=3)
    for fact in facts:
        diagnostic = {k: fact[k] for k in ("representation", "mean_direction_norm", "mean_margin", "realized_training") if fact.get(k) is not None}
        if diagnostic and not unsafe_source(diagnostic):
            parts.append("Verified development diagnosis: " + canonical(diagnostic)[:800])
    return "\n".join(part for part in parts if part.strip())[:4000]


def _search_text(kind: str, body: dict) -> str:
    scope = body.get("conditions") if kind == "lesson" else body.get("preconditions")
    scope = scope or {}
    # The old goal_id is an applicability identity, not semantic text.
    readable = {k: scope[k] for k in ("modality", "regime", "dataset", "input_geometry", "feature_target", "intervention", "fidelity") if scope.get(k) not in (None, "")}
    description = str(body.get("statement") or "") if kind == "lesson" else str(body["description_en"]) + "\n" + str(body["trigger_en"])
    return description + ("\nApplicable scope: " + canonical(readable) if readable else "")


def candidates(camp: Path, config: MemoryConfig, context: dict | None = None) -> list[dict]:
    from react_agent.eeg_research.agentic.memory import EpisodeStore, query_lessons

    require_eeg_campaign(camp)
    context = context if context is not None else build_context(camp)
    store = EpisodeStore(camp)
    episodes = {ep["episode_id"]: ep for ep in store.list_episodes()}
    valid = []; sources = {}
    for lesson in store.list_lessons():
        if unsafe_source(lesson) or lesson.get("origin_campaign_id", camp.name) != camp.name or lesson.get("is_current") is False or lesson.get("status") in {"disabled", "superseded"}:
            continue
        try:
            refs = lesson.get("supporting_episode_ids") or []
            if not refs:
                continue
            manifest = [episode_source(camp, episodes[ref]) for ref in refs]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        valid.append(lesson); sources[lesson["lesson_id"]] = manifest
    rows = []
    for label, lessons in query_lessons(context, valid).items():
        for body in lessons:
            original = next(item for item in valid if item["lesson_id"] == body["lesson_id"])
            digest = content_hash(original)
            rows.append({"record_type": "lesson", "record_id": body["lesson_id"], "version": digest,
                         "content_hash": digest, "source_refs": sources[body["lesson_id"]],
                         "applicability": label, "condition_differences": {k: body[k] for k in ("mismatch", "missing_conditions") if k in body},
                         "body": body})
    if config.skills_enabled:
        from react_agent.eeg_research.agentic.skill_memory import current_skills, skill_applicability
        for skill in current_skills(camp):
            label, differences = skill_applicability(skill["payload"], context)
            rows.append({"record_type": "skill", "record_id": skill["skill_id"], "version": skill["version"],
                         "content_hash": skill["content_hash"], "source_refs": skill["source_manifest"],
                         "applicability": "procedural_hint", "precondition_status": label,
                         "condition_differences": differences, "body": skill["payload"]})
    for row in rows:
        row["search_text"] = _search_text(row["record_type"], row["body"])
        row["text_hash"] = hashlib.sha256(row["search_text"].encode()).hexdigest()
    return rows


def index_records(store: SidecarStore, records: list[dict], backend, *, rebuild: bool = False) -> dict:
    pending = [row for row in records if rebuild or store.cached(row, backend.fingerprint, backend.metadata) is None]
    stats = {"indexed": 0, "reused": len(records) - len(pending), "pending": len(pending), "failed": 0}
    if pending:
        try:
            vectors, metadata = backend.encode([row["search_text"] for row in pending])
            store.save_vectors(pending, vectors, metadata, backend)
            stats.update(indexed=len(pending), pending=0)
        except Exception as exc:
            stats.update(failed=len(pending), reason="index_failed:" + type(exc).__name__)
    return stats


def retrieve(camp: Path, query: str | None = None, *, state: dict | None = None,
             experiment: dict | None = None, config: MemoryConfig | None = None,
             require_embedding: bool = False, rebuild: bool = False) -> dict:
    config = config or memory_config(camp)
    query = build_query(camp, state=state, experiment=experiment) if query is None else query
    bundle = {"backend": "legacy", "degraded": False, "reason": "disabled", "query": query[:800],
              "items": [], "lessons": {k: [] for k in GROUPS}, "candidate_count": 0, "hit_count": 0,
              "context_chars_budget": config.context_chars_budget, "truncated": False}
    if not config.enabled:
        if require_embedding:
            raise EmbeddingUnavailable("memory_disabled")
        return bundle
    try:
        require_eeg_campaign(camp)
        context = build_context(camp, state=state, experiment=experiment)
        records = candidates(camp, config, context)
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
            bundle["context_chars_used"] = len(canonical(bundle))
            return bundle
        query_vectors, query_metadata = backend.encode([query])
        query_vector = query_vectors[0]
        vector_blob(query_vector)
        bundle["query_encoding"] = {k: v for k, v in query_metadata[0].items() if k != "encoded_text"}
        ranked = []
        priorities = {"compatible_evidence": 0, "preconditions_met": 1, "analogy_only": 2,
                      "preconditions_mismatch": 2, "applicability_unknown": 3, "preconditions_unknown": 3}
        for row in records:
            vector = store.cached(row, backend.fingerprint, backend.metadata)
            if vector is None:
                continue
            score = float(vector @ query_vector)
            if score >= config.min_similarity:
                ranked.append((priorities[row.get("precondition_status", row["applicability"])], -score,
                               (row["record_type"], row["record_id"], str(row["version"])), row, score))
        ranked.sort(key=lambda item: item[:3])
        bundle["matched_count"] = len(ranked)
        for _, _, _, row, score in ranked:
            if len(bundle["items"]) >= max(0, config.top_k):
                break
            item = {k: copy.deepcopy(v) for k, v in row.items() if k not in {"search_text", "text_hash", "body"}}
            item["cosine_score"] = score
            if row["record_type"] == "skill":
                item["body"] = copy.deepcopy(row["body"])
            trial = copy.deepcopy(bundle)
            trial["items"].append(item)
            if row["record_type"] == "lesson":
                trial["lessons"][row["applicability"]].append(copy.deepcopy(row["body"]))
            trial["hit_count"] = len(trial["items"])
            # Include all metadata and both lesson/skill bodies in the same budget.
            trial["context_chars_used"] = 0
            used = len(canonical(trial)) + 10
            if used <= config.context_chars_budget:
                trial["context_chars_used"] = used
                bundle = trial
            else:
                bundle["truncated"] = True
        bundle["hit_count"] = len(bundle["items"])
        bundle["context_chars_used"] = len(canonical(bundle)) + 10
        return bundle
    except Exception as exc:
        if require_embedding:
            raise EmbeddingUnavailable(str(exc)) from exc
        # Do not pretend structured applicability is a lexical/embedding hit.
        bundle.update(backend="legacy", degraded=True, reason=str(exc) if isinstance(exc, EmbeddingUnavailable) else type(exc).__name__, items=[], hit_count=0)
        return bundle


def apply_memory(request: dict, bundle: dict) -> dict:
    result = copy.deepcopy(request)
    metadata = {k: copy.deepcopy(v) for k, v in bundle.items() if k != "lessons"}
    if bundle["backend"] == "minilm":
        if bundle.get("context_chars_used", 0) > bundle["context_chars_budget"]:
            return result  # A budget too small for identities cannot deliver a body.
        result["lessons"] = copy.deepcopy(bundle["lessons"])
    result["retrieved_memory"] = metadata
    return result


def role_memory(camp: Path, *, state: dict | None = None, experiment: dict | None = None) -> dict | None:
    config = memory_config(camp)
    return retrieve(camp, state=state, experiment=experiment, config=config) if config.enabled else None


def standalone_delivery(bundle: dict) -> dict:
    """Designer/coder lack the planner's legacy lessons view: include each body once."""
    result = copy.deepcopy(bundle)
    result.pop("lessons", None)
    for item in result["items"]:
        if item["record_type"] == "lesson":
            item["body"] = next(row for row in bundle["lessons"][item["applicability"]] if row["lesson_id"] == item["record_id"])
    return result


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
            body = standalone_delivery(bundle) if bundle and bundle.get("context_chars_used", 0) <= config.context_chars_budget else None
        return {"spec_hash": spec_hash, "retrieved_memory": body}
    frozen = SidecarStore(camp).freeze("coder:" + workspace.name + ":" + attempt_id, build)
    if frozen["spec_hash"] != spec_hash:
        raise ValueError("frozen_memory_experiment_changed")
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
