"""Job-level diagnostic calculations. Missing inputs are unavailable, not zeros."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from react_agent.eeg_training.hooks import image_id_duplicate_stats

UNAVAILABLE = "unavailable"


def _item(status: str, payload: dict[str, Any] | None = None, reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"status": status}
    if payload is not None:
        row["payload"] = payload
    if reason:
        row["reason"] = reason
    if status == UNAVAILABLE:
        row.setdefault("reason", reason or UNAVAILABLE)
    return row


def training_dynamics(job_dir: Path) -> dict[str, Any]:
    history = job_dir / "history.jsonl"
    if not history.is_file():
        return _item(UNAVAILABLE, reason="history_missing")
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return _item(UNAVAILABLE, reason="history_empty")
    last = rows[-1]
    best = max(rows, key=lambda item: float(item.get("fixed_bank_top1") or 0.0))
    return _item(
        "observed",
        {
            "epochs": len(rows),
            "best_epoch": best.get("epoch"),
            "last_loss": last.get("train_loss") or last.get("loss"),
            "last_fixed_bank_top1": last.get("fixed_bank_top1"),
            "last_fixed_bank_top5": last.get("fixed_bank_top5"),
            "train_loss": [row.get("train_loss") or row.get("loss") for row in rows],
            "fixed_bank_top1": [row.get("fixed_bank_top1") for row in rows],
            "sample_count": len(rows),
            "definition": "epoch_series_from_history",
        },
    )


def duplicate_audit(batches: list[list[str]]) -> dict[str, Any]:
    if not batches:
        return _item(UNAVAILABLE, reason="no_train_batches")
    stats = [image_id_duplicate_stats(batch) for batch in batches]
    mean_rate = sum(row["duplicate_rate"] for row in stats) / len(stats)
    return _item(
        "observed",
        {
            "batch_count": len(stats),
            "mean_duplicate_rate": mean_rate,
            "max_same_image": max(row["max_same_image"] for row in stats),
            "batches": stats[:8],
        },
    )


def _load_duplicate_batches(job_dir: Path) -> list[list[str]]:
    path = job_dir / "train_batch_image_ids.jsonl"
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            rows.append([str(item) for item in payload])
    return rows


def retrieval_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Positive rank and margin from one frozen validation pass. Ranks are 1-based."""
    if not rows:
        return _item(UNAVAILABLE, reason="query_ranks_not_exported")
    margins = []
    hits = 0
    parsed = []
    for row in rows:
        rank = row.get("positive_rank")
        positive = row.get("positive_score")
        negative = row.get("best_negative_score")
        if rank is None or positive is None or negative is None:
            return _item(UNAVAILABLE, reason="retrieval_row_incomplete")
        margin = float(positive) - float(negative)
        margins.append(margin)
        if int(rank) <= int(row.get("k") or 1):
            hits += 1
        parsed.append(
            {
                "query_id": row.get("query_id"),
                "positive_rank": int(rank),
                "positive_score": float(positive),
                "best_negative_score": float(negative),
                "margin": margin,
            }
        )
    return _item(
        "observed",
        {
            "sample_count": len(parsed),
            "mean_margin": sum(margins) / len(margins),
            "hit_rate": hits / len(parsed),
            "rows": parsed,
            "definition": "positive_score_minus_best_negative",
            "scope": "frozen_validation",
        },
    )


def representation_from_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    """Norm, per-dimension variance, and participation-ratio effective rank."""
    if not vectors or not vectors[0]:
        return _item(UNAVAILABLE, reason="embeddings_not_exported")
    width = len(vectors[0])
    if any(len(row) != width for row in vectors):
        return _item(UNAVAILABLE, reason="embedding_width_mismatch")
    count = len(vectors)
    norms = []
    for row in vectors:
        norms.append(sum(value * value for value in row) ** 0.5)
    means = [sum(row[index] for row in vectors) / count for index in range(width)]
    variances = [
        sum((row[index] - means[index]) ** 2 for row in vectors) / count
        for index in range(width)
    ]
    centered = [[row[index] - means[index] for index in range(width)] for row in vectors]
    if count < 2 or all(all(value == 0 for value in row) for row in centered):
        return _item(
            "undefined",
            {
                "sample_count": count,
                "width": width,
                "mean_norm": sum(norms) / count,
                "mean_dimension_variance": (sum(variances) / width) if width else 0.0,
                "effective_rank": None,
                "effective_rank_definition": "participation_ratio_of_centered_covariance_eigenvalues",
                "method": "thin_svd_of_centered_samples",
                "centered": True,
                "reason": "constant_or_insufficient_samples",
            },
        )
    import math

    try:
        import numpy as np

        matrix = np.asarray(centered, dtype=float)
        singular = np.linalg.svd(matrix, compute_uv=False)
        eigenvalues = [float(value * value) / max(count - 1, 1) for value in singular]
    except Exception:  # noqa: BLE001
        gram = [[0.0] * width for _ in range(width)]
        for row in centered:
            for i in range(width):
                for j in range(width):
                    gram[i][j] += row[i] * row[j]
        scale = 1.0 / max(count - 1, 1)
        eigenvalues = [gram[i][i] * scale for i in range(width)]
    total = sum(eigenvalues)
    squares = sum(value * value for value in eigenvalues)
    if not math.isfinite(total) or squares == 0:
        return _item("undefined", {"sample_count": count, "width": width, "effective_rank": None, "reason": "non_finite_or_zero_spectrum"})
    effective = (total * total) / squares
    return _item(
        "observed",
        {
            "sample_count": count,
            "width": width,
            "mean_norm": sum(norms) / count,
            "mean_dimension_variance": (sum(variances) / width) if width else 0.0,
            "effective_rank": effective,
            "effective_rank_definition": "participation_ratio_of_centered_covariance_eigenvalues",
            "method": "thin_svd_of_centered_samples",
            "centered": True,
            "scope": "supplied_embeddings",
        },
    )


def retrieval_rows_from_scores(
    queries: list[tuple[str, list[float]]],
    bank: list[tuple[str, list[float]]],
    positives: dict[str, set[str]],
    *,
    k: int = 1,
) -> list[dict[str, Any]]:
    """One development query row. Rank is 1-based; ties count strictly higher scores first."""
    rows: list[dict[str, Any]] = []
    for query_id, vector in queries:
        allowed = positives.get(query_id) or set()
        ranked = sorted(((image_id, _cosine(vector, candidate)) for image_id, candidate in bank), key=lambda item: item[1], reverse=True)
        pos = [(image_id, score) for image_id, score in ranked if image_id in allowed]
        neg = [(image_id, score) for image_id, score in ranked if image_id not in allowed]
        if not pos:
            continue
        best_pos = max(score for _image, score in pos)
        best_neg = max((score for _image, score in neg), default=float("-inf"))
        rank = 1 + sum(1 for _image, score in ranked if score > best_pos)
        rows.append(
            {
                "query_id": query_id,
                "positive_identity": [image_id for image_id, _score in pos],
                "positive_count": len(pos),
                "positive_rank": rank,
                "positive_score": best_pos,
                "best_negative_score": best_neg if best_neg != float("-inf") else None,
                "margin": None if best_neg == float("-inf") else best_pos - best_neg,
                "top_k_hit": rank <= k,
                "k": k,
                "tie_policy": "strictly_higher_scores_precede",
            }
        )
    return rows


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_n = sum(a * a for a in left) ** 0.5
    right_n = sum(b * b for b in right) ** 0.5
    if left_n == 0 or right_n == 0:
        return 0.0
    return dot / (left_n * right_n)


def _load_jsonl(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_embedding_matrix(job_dir: Path) -> list[list[float]]:
    path = job_dir / "embeddings.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("vectors") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    matrix: list[list[float]] = []
    for row in rows:
        if isinstance(row, list):
            matrix.append([float(value) for value in row])
    return matrix


def write_validation_artifacts(
    job_dir: Path,
    queries: list[tuple[str, list[float]]],
    bank: list[tuple[str, list[float]]],
    positives: dict[str, set[str]],
    *,
    limit: int = 32,
) -> dict[str, Any]:
    """Write retrieval_queries.jsonl and a bounded embeddings.json from a real evaluator pass."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    bounded = queries[:limit]
    rows = retrieval_rows_from_scores(bounded, bank, positives)
    (job_dir / "retrieval_queries.jsonl").write_text(
        ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n") if rows else "",
        encoding="utf-8",
    )
    vectors = [vector for _query_id, vector in bounded[:16]]
    (job_dir / "embeddings.json").write_text(
        json.dumps({"vectors": vectors, "sample_count": len(vectors), "source": "validation_encoder"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"query_rows": len(rows), "embedding_sample": len(vectors)}


def compute_job_diagnostics(job_dir: Path, *, batches: list[list[str]] | None = None) -> dict[str, Any]:
    """Fill every DiagnosticBundle category. Absent files stay unavailable."""
    job_dir = Path(job_dir)
    metrics = {}
    metrics_path = job_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
    identity = job_dir / "evaluation_identity.json"
    binding = job_dir / "source_binding.json"
    capabilities = job_dir / "capabilities_used.json"
    if batches is None:
        batches = _load_duplicate_batches(job_dir)
    return {
        "schema_version": "eeg_research.diagnostic_bundle.v1",
        "data_audit": _item(UNAVAILABLE, reason="raw_eeg_not_in_job_dir") if not identity.is_file() else _item(
            "observed",
            {
                "evaluation_identity": True,
                "query_count": metrics.get("query_count"),
                "gallery_size": metrics.get("validation_image_count") or metrics.get("candidate_count"),
                "sampling_rate": UNAVAILABLE,
                "repeats_policy": "mean_over_repeats",
                "split_overlap": metrics.get("train_validation_overlap"),
            },
        ),
        "training_dynamics": training_dynamics(job_dir),
        "retrieval_errors": retrieval_from_rows([row for row in _load_jsonl(job_dir / "retrieval_queries.jsonl") if isinstance(row, dict)]),
        "representation": representation_from_vectors(_load_embedding_matrix(job_dir)),
        "group_results": _item(UNAVAILABLE, reason="subject_metadata_not_trusted")
        if not (job_dir / "subject_groups.json").is_file()
        else _item("observed", json.loads((job_dir / "subject_groups.json").read_text(encoding="utf-8"))),
        "intervention_probe": _item(UNAVAILABLE, reason="occlusion_not_requested"),
        "integrity_cost": _item(
            "observed" if binding.is_file() else UNAVAILABLE,
            {
                "source_binding": binding.is_file(),
                "checkpoint": (job_dir / "last.ckpt").is_file(),
                "capabilities_used": capabilities.is_file(),
                "gpu_seconds": None,
            },
            reason=None if binding.is_file() else "binding_missing",
        ),
        "image_id_duplicates": duplicate_audit(batches or []),
    }
