"""Optional, process-cached MiniLM encoding. Importing this module loads no model."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TEXT_BUILDER_VERSION = "eeg_campaign_memory.v2.token_bounded"
DIMENSION = 384
_BACKENDS: dict[tuple, Any] = {}
_FAILURES: dict[tuple, str] = {}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    skills_enabled: bool = False
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str | None = None
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(default=32, ge=1, le=256)
    top_k: int = 5
    min_similarity: float = Field(default=0.30, ge=-1, le=1, allow_inf_nan=False)
    content_chars_budget: int = Field(default=6000, ge=0, le=30000)
    allow_model_download: bool = False
    fallback: Literal["legacy"] = "legacy"


def memory_config(camp: Path) -> MemoryConfig:
    path = Path(camp) / "goal.json"
    if not path.is_file():
        return MemoryConfig()
    goal = json.loads(path.read_text(encoding="utf-8"))
    raw = goal.get("memory") or {}
    # An explicit opt-out must not validate or activate unused backend options.
    # The delivery allowance still controls recent scientific history when no
    # semantic backend is used, so retain that independent configuration.
    if isinstance(raw, dict) and not raw.get("enabled", False):
        return MemoryConfig(content_chars_budget=raw.get(
            "content_chars_budget", MemoryConfig.model_fields["content_chars_budget"].default))
    return MemoryConfig.model_validate(raw)


class EmbeddingUnavailable(RuntimeError):
    """Embedding failed; callers may explicitly choose legacy or strict behavior."""


class MiniLMBackend:
    def __init__(self, config: MemoryConfig):
        # These optional dependencies are intentionally confined to this boundary.
        from sentence_transformers import SentenceTransformer

        revision = config.model_revision
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            if not config.allow_model_download:
                raise EmbeddingUnavailable("immutable_model_revision_required")
            from huggingface_hub import HfApi

            revision = HfApi().model_info(config.model_id, revision=revision or "main", timeout=20).sha
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise EmbeddingUnavailable("unresolved_model_revision")
        self.model = SentenceTransformer(
            config.model_id, device="cpu", revision=revision,
            local_files_only=not config.allow_model_download,
        )
        if self.model.get_sentence_embedding_dimension() != DIMENSION:
            raise EmbeddingUnavailable("expected_384_dimensions")
        tokenizer = self.model.tokenizer
        self.config = config
        self.metadata = {
            "model_id": config.model_id, "model_revision": revision,
            "dimension": DIMENSION, "dtype": "<f4", "normalized": True,
            "text_builder_version": TEXT_BUILDER_VERSION,
            "max_sequence_length": int(self.model.max_seq_length),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "vocabulary_hash": content_hash(tokenizer.get_vocab()),
                "model_max_length": tokenizer.model_max_length,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "do_lower_case": getattr(tokenizer, "do_lower_case", None),
                "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
            },
        }
        self.fingerprint = content_hash(self.metadata)

    def prepare_chunks(self, text: str, *, record_id: str):
        from react_agent.eeg_research.agentic.context.retrieval import prepare_embedding_chunks
        return prepare_embedding_chunks(text, tokenizer=self.model.tokenizer,
            max_tokens=self.metadata["max_sequence_length"], record_id=record_id)

    def prepare_query(self, text: str):
        from react_agent.eeg_research.agentic.context.retrieval import prepare_query
        # Keep the selected method/protocol/error cards ahead of potentially
        # longer prose before applying the actual MiniLM token budget.
        lines = text.splitlines()
        priority = ("Action:", "Error:", "Method:", "Protocol:", "IDs:")
        ordered = [line for line in lines if line.startswith(priority)]
        ordered.extend(line for line in lines if not line.startswith(priority))
        return prepare_query(question="\n".join(ordered), tokenizer=self.model.tokenizer,
                             max_tokens=self.metadata["max_sequence_length"])

    def encode(self, texts: list[str]):
        import numpy as np

        metadata = []
        for text in texts:
            ids = self.model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            if len(ids) > self.metadata["max_sequence_length"]:
                raise ValueError("embedding_input_exceeds_tokenizer_limit_use_explicit_chunks")
            metadata.append({
                "input_token_count": len(ids), "encoded_token_count": len(ids),
                "max_sequence_length": self.metadata["max_sequence_length"],
                "truncated": False,
                "encoded_text": text,
            })
        vectors = self.model.encode(
            texts, batch_size=self.config.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="<f4")
        if vectors.shape != (len(texts), DIMENSION):
            raise ValueError("invalid_embedding_batch_shape")
        for vector in vectors:
            vector_blob(vector)
        return vectors, metadata


def backend_for(config: MemoryConfig, *, retry: bool = False):
    key = (config.model_id, config.model_revision, config.device, config.allow_model_download, TEXT_BUILDER_VERSION)
    if retry:
        _FAILURES.pop(key, None)
    if key in _FAILURES:
        raise EmbeddingUnavailable(_FAILURES[key])
    if key not in _BACKENDS:
        try:
            _BACKENDS[key] = MiniLMBackend(config)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, EmbeddingUnavailable) else "model_unavailable:" + type(exc).__name__
            _FAILURES[key] = reason
            raise EmbeddingUnavailable(reason) from exc
    return _BACKENDS[key]


def vector_blob(vector) -> bytes:
    import numpy as np

    vec = np.asarray(vector, dtype="<f4")
    if vec.shape != (DIMENSION,) or not np.isfinite(vec).all():
        raise ValueError("invalid_embedding")
    if not np.isclose(float(np.linalg.norm(vec)), 1.0, atol=1e-4, rtol=0):
        raise ValueError("embedding_not_normalized")
    blob = vec.tobytes(order="C")
    if len(blob) != DIMENSION * 4:
        raise ValueError("invalid_embedding_length")
    return blob


def cached_vector(row: dict[str, Any], fingerprint: str):
    import numpy as np

    if (row["embedding_fingerprint"] != fingerprint or row["dimension"] != DIMENSION
            or row["dtype"] != "<f4" or row["normalized"] != 1
            or len(row["vector"]) != DIMENSION * 4):
        raise ValueError("invalid_cached_embedding_metadata")
    vec = np.frombuffer(row["vector"], dtype="<f4").copy()
    vector_blob(vec)
    return vec
