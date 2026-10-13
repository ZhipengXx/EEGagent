"""Bounded query construction and explicit MiniLM representations.

The original record is never shortened.  Only the derived embedding view is
segmented, and every segment records its original identity and token range.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

VERSION = "eeg_research.context.retrieval.v1"
MAX_DELIVERY_CHARS = 8000
MAX_CANDIDATES = 128
MAX_QUERY_CHARS = 1800
MAX_CHUNKS = 32


def _text(value: Any, limit: int = 320) -> str:
    if isinstance(value, str):
        return value.strip()[:limit]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def prepare_query(*, question: Any = "", action: Any = "", error: Any = "",
                  method: Any = "", protocol: Any = "", identities: Any = (),
                  tokenizer=None, max_tokens: int | None = None,
                  max_chars: int = MAX_QUERY_CHARS) -> dict:
    """Use the current question and a few selected fields, never an observation.

    Structured method/protocol dictionaries have an explicit small keyword
    projection.  Caller-supplied strings are treated as excerpts, with the
    omission recorded, rather than claiming their full meaning was encoded.
    """
    def keywords(value):
        if isinstance(value, dict):
            return " ".join(filter(None, (_text(value.get(k), 100) for k in
                ("method", "method_revision", "intervention", "scope", "regime",
                 "dataset", "evaluation_identity", "error_type", "code"))))[:400]
        return _text(value, 400)

    fields = [("Question", _text(question, 600)), ("Action", _text(action, 120)),
              ("Error", keywords(error)), ("Method", keywords(method)),
              ("Protocol", keywords(protocol))]
    ids = []
    if isinstance(identities, (list, tuple)):
        ids = [_text(v, 80) for v in identities[:4] if _text(v, 80)]
    fields.append(("IDs", " ".join(ids)))
    active = [(k, v) for k, v in fields if v]
    complete = active[0][1] if len(active) == 1 and active[0][0] == "Question" else "\n".join(f"{k}: {v}" for k, v in active)
    text = complete[:max(0, min(max_chars, MAX_QUERY_CHARS))]
    original_hash = hashlib.sha256(complete.encode()).hexdigest()
    input_fields = (("question", question, 600), ("action", action, 120),
                    ("error", error, 400), ("method", method, 400), ("protocol", protocol, 400))
    selected_fields = {name: {"original_chars": len(value), "selected_char_range": [0, min(len(value.strip()), limit)],
                             "complete": len(value.strip()) <= limit}
                       for name, value, limit in input_fields if isinstance(value, str) and value}
    metadata = {"builder_version": VERSION, "query_hash": original_hash,
                "representation": "bounded_query_card", "omitted": text != complete or any(
                    not field["complete"] for field in selected_fields.values()), "selected_fields": selected_fields,
                "chars": len(text)}
    if tokenizer is not None:
        if max_tokens is None or max_tokens < 4:
            raise ValueError("query_token_budget_missing")
        ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        metadata["original_token_count"] = len(ids)
        # Deterministic prefix excerpt; metadata explicitly says what it covers.
        while len(ids) > max_tokens and text:
            text = text[:max(0, len(text) * max_tokens // len(ids) - 1)]
            ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        metadata.update(encoded_token_count=len(ids), max_sequence_length=max_tokens,
                        omitted=metadata["omitted"] or len(ids) < metadata["original_token_count"],
                        selected_char_range=[0, len(text)], chars=len(text))
        if len(ids) > max_tokens:
            raise ValueError("query_special_tokens_exceed_budget")
    return {"text": text, "metadata": metadata}


def prepare_embedding_chunks(text: str, *, tokenizer, max_tokens: int,
                             record_id: str, max_chunks: int = MAX_CHUNKS) -> list[dict]:
    """Return stable, length-verified chunks bound to the untouched full text."""
    if max_tokens < 4 or not 1 <= max_chunks <= MAX_CHUNKS:
        raise ValueError("invalid_embedding_chunk_budget")
    full_hash = hashlib.sha256(text.encode()).hexdigest()
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    special = len(tokenizer("", truncation=False, add_special_tokens=True)["input_ids"])
    capacity = max_tokens - special
    if capacity < 1:
        raise ValueError("embedding_special_tokens_exceed_budget")
    chunks = []
    if not ids:
        return []
    start = 0
    while start < len(ids) and len(chunks) < max_chunks:
        end = min(len(ids), start + capacity)
        segment = text if start == 0 and end == len(ids) else tokenizer.decode(
            ids[start:end], skip_special_tokens=True)
        encoded = tokenizer(segment, truncation=False, add_special_tokens=True)["input_ids"]
        # Decode/re-encode can add tokens; shrink monotonically and validate.
        while len(encoded) > max_tokens and end > start:
            end -= 1
            segment = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            encoded = tokenizer(segment, truncation=False, add_special_tokens=True)["input_ids"]
        if end <= start:
            raise ValueError("embedding_chunk_cannot_fit")
        digest = hashlib.sha256(segment.encode()).hexdigest()
        chunks.append({"chunk_id": "emb_" + digest[:24], "record_id": record_id,
                       "text": segment, "text_hash": digest, "full_text_hash": full_hash,
                       "token_range": [start, end], "full_token_count": len(ids),
                       "encoded_token_count": len(encoded), "max_sequence_length": max_tokens,
                       "representation": "full" if start == 0 and end == len(ids) else "token_chunk",
                       "builder_version": VERSION})
        start = end
    for chunk in chunks:
        chunk.update(chunk_count=len(chunks), complete_index_coverage=start == len(ids),
                     omitted_token_range=None if start == len(ids) else [start, len(ids)])
    return chunks


def source_signature(item: dict) -> str:
    """Group shared support deterministically without conflating record scope."""
    refs = item.get("source_refs") or []
    return hashlib.sha256(json.dumps(refs, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def deduplicate_hits(items: list[dict], role: str = "research_planner") -> list[dict]:
    """Choose a role-appropriate representative only for identical support/scope."""
    prefer = "skill" if role in {"candidate_coder", "candidate_reviewer"} else "lesson"
    ordered = sorted(items, key=lambda item: (item.get("record_type") != prefer,
                     -float(item.get("cosine_score", 0)), str(item.get("record_id", ""))))
    seen = set(); result = []
    for item in ordered:
        key = (source_signature(item), item.get("applicability"),
               json.dumps(item.get("condition_differences") or {}, sort_keys=True))
        # Empty supports cannot justify conflating unrelated claims.
        if not item.get("source_refs"):
            key += (item.get("record_id"),)
        if key not in seen:
            seen.add(key); result.append(item)
    return result
