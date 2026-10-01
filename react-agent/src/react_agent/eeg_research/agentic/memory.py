"""Research memory. Episodes are written by the runtime. The curator cannot promote them."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

LEVEL_ORDER = (
    "implementation_failure",
    "exploratory_result",
    "paired_seed_result",
    "confirmed_result",
)


def _usable_comparison(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not data:
        return False
    comparison = data.get("comparison") if isinstance(data, dict) else None
    if comparison is None and isinstance(data, dict) and data.get("comparable") is not None:
        comparison = data
    return isinstance(comparison, dict) and comparison.get("comparable") is True


def _valid_confirmation(camp: Path, ref: str, cited: list[dict[str, Any]]) -> bool:
    """Resolve confirmed support against the frozen policy and actual full runs."""
    from react_agent.eeg_research.agentic.paths import safe_name
    if safe_name(ref) is None:
        return False
    try:
        record = json.loads((camp / "confirmations" / f"{ref}.json").read_text(encoding="utf-8"))
        policy = json.loads((camp / "confirmation_policy.json").read_text(encoding="utf-8"))
        state = json.loads((camp / "campaign_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("schema_version") != "eeg_research.confirmation_record.v1":
        return False
    if record.get("status") != "confirmed" or not policy.get("policy_hash") or record.get("policy_hash") != policy["policy_hash"]:
        return False
    refs = record.get("pair_refs") or []
    required = int(policy.get("target_pairs") or 0)
    if required < 1 or len(set(refs)) < required or (record.get("aggregate") or {}).get("status") != "confirmed":
        return False
    rows = []
    for ref_id in refs:
        row = next((item for item in state.get("evidence") or []
                    if item.get("job_id") == ref_id or item.get("evidence_id") == ref_id), None)
        if row is None or row.get("fidelity") != "full" or row.get("evaluation_valid") is not True:
            return False
        if (row.get("comparison") or {}).get("comparable") is not True:
            return False
        rows.append(row)
    seeds = {row.get("seed") for row in rows}
    if len(seeds) < required or not seeds.issubset(set(policy.get("training_seeds") or [])):
        return False
    identities = {(row.get("candidate_id"), row.get("source_hash"), row.get("config_hash"), row.get("contract_fingerprint")) for row in rows}
    if len(identities) != 1 or any(part in (None, "") for part in next(iter(identities))):
        return False
    run_ids = {row.get("job_id") for row in rows}
    return all(item.get("job_id") in run_ids for item in cited)


_LESSON_ALIASES = {
    "claim": "statement",
    "text": "statement",
    "confidence": "uncertainty",
    "evidence_level_requested": "requested_evidence_level",
    "requested_level": "requested_evidence_level",
    "support_refs": "supporting_episode_ids",
    "supporting_refs": "supporting_episode_ids",
    "contradict_refs": "contradicting_episode_ids",
    "effect_refs": "observed_effect_refs",
}


def canonicalize_lesson_proposal(lesson: dict[str, Any]) -> dict[str, Any]:
    """Explicit alias migration. Requested and stored evidence levels stay distinct."""
    copied = dict(lesson)
    for old, new in _LESSON_ALIASES.items():
        if old in copied and new not in copied:
            copied[new] = copied[old]
        elif old in copied and copied[new] != copied[old]:
            raise ValueError(f"lesson_alias_conflict:{old}")
    return copied


def evidence_level(kind: str) -> str:
    if kind == "implementation_failure":
        return "implementation_failure"
    if kind == "paired_seed_result":
        return "paired_seed_result"
    if kind == "confirmed_result":
        return "confirmed_result"
    if kind == "exploratory_result":
        return "exploratory_result"
    return "exploratory_result"


def _rank(level: str | None) -> int:
    if level in LEVEL_ORDER:
        return LEVEL_ORDER.index(level)
    return LEVEL_ORDER.index("exploratory_result")


def episode(
    *,
    task_hash: str,
    candidate_id: str,
    kind: str,
    fidelity: str,
    seed: int,
    metric: float | None,
    contract_fingerprint: str,
    artifact: str,
    episode_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """One deterministic episode. Test metrics are not accepted."""
    return {
        "episode_id": episode_id or f"ep_{uuid.uuid4().hex[:12]}",
        "task_hash": task_hash,
        "candidate_id": candidate_id,
        "kind": kind,
        "evidence_level": evidence_level(kind),
        "fidelity": fidelity,
        "seed": seed,
        "primary_metric": metric,
        "contract_fingerprint": contract_fingerprint,
        "artifact": artifact,
        "job_id": job_id,
        "scope": "development",
    }


def retrieve(entries: list[dict[str, Any]], *, task_hash: str, fingerprint: str) -> list[dict[str, Any]]:
    """Compatible rows stay comparable. Other protocols are analogy only."""
    rows = []
    seen_jobs: set[str] = set()
    for entry in entries:
        if "test" in json_keys(entry):
            continue
        copied = dict(entry)
        job = str(entry.get("job_id") or entry.get("artifact") or entry.get("episode_id") or "")
        if job and job in seen_jobs:
            copied["retrieval"] = "duplicate_same_job"
            copied["new_evidence"] = False
        elif entry.get("task_hash") != task_hash or entry.get("contract_fingerprint") != fingerprint:
            copied["retrieval"] = "analogy_only"
        else:
            copied["retrieval"] = "comparable"
        if job:
            seen_jobs.add(job)
        rows.append(copied)
    return rows


def json_keys(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(key) + " " + json_keys(item) for key, item in value.items())
    if isinstance(value, list):
        return " ".join(json_keys(item) for item in value)
    return ""


def confirmation(pairs: list[float], *, target: int = 3, margin_pp: float = 0.0) -> str:
    """Paired seed rule on percentage points. Values are already pp; they are not divided again."""
    if len(pairs) < target:
        return "provisional_improvement" if pairs and sum(pairs) / len(pairs) > margin_pp else "no_confirmed_improvement"
    positive = sum(1 for value in pairs if value > 0)
    mean = sum(pairs) / len(pairs)
    if mean > margin_pp and 3 * positive >= 2 * len(pairs):
        return "replicated_improvement"
    return "no_confirmed_improvement"


def query_lessons(context: dict[str, Any], lessons: list[dict[str, Any]]) -> dict[str, Any]:
    """Structured filter. Compatible lessons may be cited; other protocols are analogy only."""
    compatible: list[dict[str, Any]] = []
    analogy: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    keys = (
        "task",
        "dataset",
        "input_geometry",
        "feature_target",
        "evaluation_identity",
        "negative_sampling_policy",
        "intervention",
        "fidelity",
    )
    for lesson in lessons:
        conditions = lesson.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            continue
        copied = dict(lesson)
        mismatches = [
            key
            for key in keys
            if context.get(key) not in (None, "")
            and conditions.get(key) not in (None, "")
            and context.get(key) != conditions.get(key)
        ]
        required_missing = [
            key
            for key in keys
            if context.get(key) not in (None, "") and conditions.get(key) in (None, "")
        ]
        if required_missing:
            copied["retrieval"] = "applicability_unknown"
            copied["missing_conditions"] = required_missing
            unknown.append(copied)
        elif mismatches:
            copied["retrieval"] = "analogy_only"
            copied["mismatch"] = mismatches
            analogy.append(copied)
        else:
            copied["retrieval"] = "compatible_evidence"
            compatible.append(copied)
    return {"compatible_evidence": compatible, "analogy_only": analogy, "applicability_unknown": unknown}


def open_store(camp: Path) -> sqlite3.Connection:
    """Return the campaign SQLite memory. Tables are created if missing."""
    camp.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(camp / "memory.sqlite")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            task_hash TEXT,
            candidate_id TEXT,
            kind TEXT,
            evidence_level TEXT,
            fidelity TEXT,
            seed INTEGER,
            primary_metric REAL,
            contract_fingerprint TEXT,
            artifact TEXT,
            job_id TEXT,
            payload TEXT NOT NULL,
            created_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS lessons (
            lesson_id TEXT PRIMARY KEY,
            evidence_level TEXT,
            payload TEXT NOT NULL,
            created_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS curation (
            key TEXT PRIMARY KEY,
            status TEXT,
            detail TEXT,
            updated_at REAL
        )"""
    )
    conn.commit()
    return conn


class EpisodeStore:
    """Append-only episodes. The curator may only insert validated lessons."""

    def __init__(self, camp: Path) -> None:
        self.camp = Path(camp)

    def persist_episode(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload.setdefault("episode_id", f"ep_{uuid.uuid4().hex[:12]}")
        conn = open_store(self.camp)
        try:
            existing = conn.execute("SELECT payload FROM episodes WHERE episode_id = ?", (payload["episode_id"],)).fetchone()
            if existing:
                return json.loads(existing[0])
            job_id = payload.get("job_id")
            if job_id:
                by_job = conn.execute("SELECT payload FROM episodes WHERE job_id = ?", (str(job_id),)).fetchone()
                if by_job:
                    return json.loads(by_job[0])
            conn.execute(
                """INSERT INTO episodes (
                    episode_id, task_hash, candidate_id, kind, evidence_level, fidelity, seed,
                    primary_metric, contract_fingerprint, artifact, job_id, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["episode_id"],
                    payload.get("task_hash"),
                    payload.get("candidate_id"),
                    payload.get("kind"),
                    payload.get("evidence_level"),
                    payload.get("fidelity"),
                    payload.get("seed"),
                    payload.get("primary_metric"),
                    payload.get("contract_fingerprint"),
                    payload.get("artifact"),
                    payload.get("job_id"),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return payload

    def update_episode(self, _episode_id: str, **_fields: Any) -> None:
        """Episodes are immutable. Curator and callers must not rewrite them."""
        raise PermissionError("episode_immutable")

    def list_episodes(self) -> list[dict[str, Any]]:
        conn = open_store(self.camp)
        try:
            rows = conn.execute("SELECT payload FROM episodes ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [json.loads(item[0]) for item in rows]

    def list_lessons(self) -> list[dict[str, Any]]:
        conn = open_store(self.camp)
        try:
            rows = conn.execute("SELECT payload FROM lessons ORDER BY created_at").fetchall()
        finally:
            conn.close()
        return [json.loads(item[0]) for item in rows]

    def mark_pending_curation(self, detail: str = "curator_unavailable") -> None:
        conn = open_store(self.camp)
        try:
            conn.execute(
                "INSERT INTO curation(key, status, detail, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at",
                ("pending", "pending", detail, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def pending_curation(self) -> bool:
        conn = open_store(self.camp)
        try:
            row = conn.execute("SELECT status FROM curation WHERE key = ?", ("pending",)).fetchone()
        finally:
            conn.close()
        return bool(row and row[0] == "pending")

    def accept_lessons(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Validate curator lessons. Evidence level cannot exceed cited episodes. Episodes are not edited."""
        episodes = {str(item.get("episode_id")): item for item in self.list_episodes()}
        lessons = proposal.get("proposed_lessons") or []
        if isinstance(lessons, dict):
            lessons = [lessons]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        conn = open_store(self.camp)
        try:
            for lesson in lessons:
                if not isinstance(lesson, dict):
                    rejected.append({"reason": "lesson_not_object"})
                    continue
                try:
                    lesson = canonicalize_lesson_proposal(lesson)
                except ValueError:
                    rejected.append({"reason": "lesson_alias_conflict"})
                    continue
                supporting = [str(item) for item in lesson.get("supporting_episode_ids") or proposal.get("supporting_episode_ids") or []]
                contradicting = [str(item) for item in lesson.get("contradicting_episode_ids") or []]
                reasons: list[str] = []
                if not supporting:
                    reasons.append("supporting_missing")
                missing = [item for item in supporting if item not in episodes]
                missing_counter = [item for item in contradicting if item not in episodes]
                if missing or missing_counter:
                    reasons.append("episode_missing")
                cited = [episodes[item] for item in supporting if item in episodes]
                if any("test" in json_keys(item) for item in cited):
                    reasons.append("final_test_derived")
                group_level = min((_rank(item.get("evidence_level")) for item in cited), default=0)
                requested = str(
                    lesson.get("requested_evidence_level")
                    or lesson.get("evidence_level")
                    or proposal.get("evidence_level_requested")
                    or "exploratory_result"
                )
                job_ids = [str(item.get("job_id") or item.get("artifact")) for item in cited]
                if job_ids and len(set(job_ids)) < len(job_ids):
                    reasons.append("same_job_not_independent")
                conditions = lesson.get("conditions") if "conditions" in lesson else proposal.get("conditions")
                if not isinstance(conditions, dict) or not conditions:
                    reasons.append("conditions_missing")
                effect = lesson.get("observed_effect")
                comparison_refs = [str(item) for item in lesson.get("comparison_refs") or lesson.get("observed_effect_refs") or []]
                confirmation_refs = [str(item) for item in lesson.get("confirmation_refs") or []]
                if effect not in (None, "") and not comparison_refs:
                    reasons.append("forged_effect")
                fake_refs = []
                from react_agent.eeg_research.agentic.paths import safe_name
                effects = []
                for item in comparison_refs:
                    if safe_name(item) is None:
                        fake_refs.append(item)
                        continue
                    path = self.camp / "comparisons" / f"{item}.json"
                    if path.is_file():
                        if not _usable_comparison(path):
                            fake_refs.append(item)
                        else:
                            raw = json.loads(path.read_text(encoding="utf-8"))
                            comp = raw.get("comparison") or raw
                            if comp.get("delta_pp") is not None:
                                effects.append({"comparison_ref": item, "delta_pp": float(comp["delta_pp"]), "unit": "percentage_points"})
                    elif item not in {row.get("evidence_id") for row in self.list_episodes()}:
                        fake_refs.append(item)
                if comparison_refs and fake_refs:
                    reasons.append("comparison_ref_missing")
                if requested in {"confirmed_result"}:
                    confirmed = any(_valid_confirmation(self.camp, item, cited) for item in confirmation_refs)
                    if not confirmed:
                        reasons.append("confirmation_missing")
                    else:
                        group_level = _rank("confirmed_result")
                if cited and _rank(requested) > group_level:
                    reasons.insert(0, "evidence_level_exceeds_runs")
                if requested in {"confirmed_result"} and not supporting:
                    reasons.append("confirmation_missing")
                if reasons:
                    rejected.append({"reason": reasons[0], "reasons": reasons, "ids": missing + missing_counter, "requested": requested})
                    continue
                stored = {
                    "statement": lesson.get("statement") or proposal.get("statement") or "",
                    "uncertainty": lesson.get("uncertainty") if "uncertainty" in lesson else proposal.get("uncertainty"),
                    "requested_evidence_level": requested,
                    "evidence_level": LEVEL_ORDER[group_level] if cited else requested,
                    "supporting_episode_ids": supporting,
                    "contradicting_episode_ids": contradicting,
                    "conditions": conditions,
                    "invalidation_conditions": lesson.get("invalidation_conditions") or proposal.get("invalidation_conditions"),
                    "observed_effect": effects or None,
                    "observed_effect_refs": comparison_refs,
                    "comparison_refs": comparison_refs,
                    "confirmation_refs": confirmation_refs,
                    "summary_zh": lesson.get("summary_zh") or proposal.get("summary_zh") or "",
                }
                canonical = {key: value for key, value in stored.items() if key not in {"summary_zh", "uncertainty"}}
                for key in ("supporting_episode_ids", "contradicting_episode_ids", "observed_effect_refs", "comparison_refs", "confirmation_refs"):
                    canonical[key] = sorted(set(canonical[key]))
                stored["lesson_id"] = "les_" + hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
                existing = conn.execute("SELECT payload FROM lessons WHERE lesson_id = ?", (stored["lesson_id"],)).fetchone()
                if existing:
                    accepted.append(json.loads(existing[0]))
                    continue
                conn.execute(
                    "INSERT INTO lessons(lesson_id, evidence_level, payload, created_at) VALUES (?, ?, ?, ?)",
                    (stored["lesson_id"], stored["evidence_level"], json.dumps(stored, ensure_ascii=False), time.time()),
                )
                accepted.append(stored)
            conn.commit()
        finally:
            conn.close()
        return {"accepted": accepted, "rejected": rejected}
