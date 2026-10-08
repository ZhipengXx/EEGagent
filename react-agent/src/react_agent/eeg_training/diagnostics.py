"""Job-level diagnostic calculations. Missing inputs are unavailable, not zeros."""

from __future__ import annotations

import json
import math
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
            "checkpoint_selection": _read_object(job_dir / "selected_checkpoint.json"),
            "training_execution": training_execution(job_dir),
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
    """One development query row. Actual hits match fixed_bank_accuracy; optimistic rank is an interval."""
    from react_agent.eeg_training.fixed_bank import score_query

    rows: list[dict[str, Any]] = []
    for query_id, vector in queries:
        allowed = positives.get(query_id) or set()
        if not allowed:
            image = query_id.split("::")[-1]
            allowed = {image, query_id}
        scored = score_query(vector, bank, allowed, k=k)
        if scored.get("skip"):
            continue
        rows.append(
            {
                "query_id": query_id,
                "positive_identity": scored["positive_identity"],
                "positive_count": scored["positive_count"],
                "positive_rank": scored["min_rank"],
                "min_rank": scored["min_rank"],
                "max_rank": scored["max_rank"],
                "rank_interval": scored["rank_interval"],
                "positive_score": scored["positive_score"],
                "best_negative_score": scored["best_negative_score"],
                "margin": scored["margin"],
                "top_k_hit": scored["top_k_hit"],
                "k": k,
                "tie_policy": "stable_bank_order_then_score",
                "all_ties": scored["all_ties"],
                "zero_norm_query": scored["zero_norm_query"],
                "non_finite": scored["non_finite"],
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


def hook_consumption(job_dir: Path) -> dict[str, Any]:
    """Compare approved hook_config to the train_entry consumption log."""
    job_dir = Path(job_dir)
    hook_path = job_dir / "hook_config.json"
    if not hook_path.is_file():
        return _item(UNAVAILABLE, reason="hook_config_missing")
    try:
        hook = json.loads(hook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _item(UNAVAILABLE, reason="hook_config_invalid")
    if not isinstance(hook, dict):
        return _item(UNAVAILABLE, reason="hook_config_invalid")
    consumed_path = job_dir / "hook_consumed.json"
    consumed: dict[str, Any] = {}
    if consumed_path.is_file():
        try:
            loaded = json.loads(consumed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            consumed = loaded
    mismatches: list[str] = []
    expected: dict[str, Any] = {}
    for name in ("model", "objective", "transform"):
        section = hook.get(name) if isinstance(hook.get(name), dict) else {}
        expected[name] = section
        if not section:
            continue
        got = consumed.get(name) if isinstance(consumed.get(name), dict) else None
        applied = (got or {}).get("applied")
        config = (got or {}).get("config") if isinstance((got or {}).get("config"), dict) else None
        if applied is not True or config is None:
            mismatches.append(f"{name}_not_consumed")
            continue
        for key, value in section.items():
            if config.get(key) != value:
                mismatches.append(f"{name}.{key}")
    return _item(
        "observed",
        {
            "expected": expected,
            "consumed": consumed,
            "mismatches": mismatches,
            "execution_status": "not_applied" if mismatches else "applied",
        },
    )


def write_validation_artifacts(
    job_dir: Path,
    queries: list[tuple[str, list[float]]],
    bank: list[tuple[str, list[float]]],
    positives: dict[str, set[str]],
    *,
    limit: int = 32,
    checkpoint_id: str | None = None,
    sample_seed: int | None = None,
    total_query_count: int | None = None,
    sampling_policy: str = "prefix_of_supplied_queries",
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
        json.dumps(
            {
                "vectors": vectors,
                "sample_count": len(vectors),
                "source": "selected_checkpoint",
                "checkpoint_id": checkpoint_id,
                "sample_seed": sample_seed,
                "total_query_count": total_query_count if total_query_count is not None else len(queries),
                "query_coverage": len(vectors) / (total_query_count or len(queries)) if (total_query_count or queries) else None,
                "sampling_policy": sampling_policy,
                "limitations": "Bounded sample variance/rank are not all-query statistics.",
                "schema_version": "eeg_research.validation_embeddings.v2",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    coverage = {"query_rows": len(rows), "embedding_sample": len(vectors), "checkpoint_id": checkpoint_id,
                "total_query_count": total_query_count if total_query_count is not None else len(queries),
                "sampling_policy": sampling_policy, "sample_seed": sample_seed,
                "limitations": "Retrieval margins/ranks use sampled queries against the full frozen gallery; embedding rank covers at most 16 sampled queries."}
    binding = _read_object(job_dir / "source_binding.json")
    coverage["source_sha256"] = binding.get("file_sha256")
    identity = _read_object(job_dir / "training_inference_identity.json")
    coverage["training_inference_identity_fingerprint"] = identity.get("fingerprint")
    (job_dir / "validation_sampling.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return coverage


def uniform_logit_reference(local_batch_sizes: list[int], *, direction_reduction: str = "mean", replica_reduction: str = "mean", objective_is_custom: bool = False) -> dict[str, Any]:
    """Derive diagonal-CE loss for actual replica sizes; custom losses are not inferred."""
    if objective_is_custom:
        return _item(UNAVAILABLE, reason="custom_loss_uniform_reference_not_defined")
    if not local_batch_sizes or any(size < 1 for size in local_batch_sizes):
        return _item(UNAVAILABLE, reason="local_batch_sizes_missing")
    if direction_reduction not in {"mean", "sum"} or replica_reduction not in {"mean", "sum"}:
        return _item(UNAVAILABLE, reason="loss_reduction_unknown")
    multiplier = 1 if direction_reduction == "mean" else 2
    local = [multiplier * math.log(size) for size in local_batch_sizes]
    value = sum(local) / len(local) if replica_reduction == "mean" else sum(local)
    return _item("derived", {"uniform_logit_loss": value, "local_batch_sizes": local_batch_sizes,
                "per_replica_reference": local, "direction_reduction": direction_reduction,
                "replica_reduction": replica_reduction,
                "definition": "diagonal_target_bidirectional_cross_entropy_at_uniform_logits",
                "limitations": "Reference for the declared diagonal CE only; repeated images remain separate diagonal targets. No collapse diagnosis follows from proximity to this value."})


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def training_execution(job_dir: Path) -> dict[str, Any]:
    """Report observed first/tail replica batches and the matching loss reference."""
    receipt = _read_object(job_dir / "training_devices.json")
    if not receipt:
        return _item(UNAVAILABLE, reason="training_devices_receipt_missing")
    batches = []
    for row in receipt.get("batches") or []:
        sizes = [int(replica["local_batch_size"]) for replica in row.get("replicas") or []]
        batches.append({**row, "uniform_logit_reference": uniform_logit_reference(sizes,
            direction_reduction=str(receipt.get("loss_direction_reduction") or "unknown"),
            replica_reduction=str(receipt.get("loss_replica_reduction") or "unknown"),
            objective_is_custom=receipt.get("objective_is_custom") is not False)})
    return _item("observed", {"loss_definition": receipt.get("loss_definition"),
                 "loss_direction_reduction": receipt.get("loss_direction_reduction"),
                 "loss_replica_reduction": receipt.get("loss_replica_reduction"),
                 "negative_sampling_policy": receipt.get("negative_sampling_policy"),
                 "objective_is_custom": receipt.get("objective_is_custom"),
                 "optimizer_learning_rates": receipt.get("optimizer_learning_rates"),
                 "embedding_dtype": receipt.get("embedding_dtype"),
                 "epoch_reduction": receipt.get("loss_epoch_reduction"), "batches": batches,
                 "coverage": "first_and_tail_training_batches_of_first_epoch",
                 "limitations": "Sampled replica reference is not the uniform reference for the complete epoch, and loss alone cannot identify a mechanism."})


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
    representation = representation_from_vectors(_load_embedding_matrix(job_dir))
    retrieval = retrieval_from_rows([row for row in _load_jsonl(job_dir / "retrieval_queries.jsonl") if isinstance(row, dict)])
    sampling = _read_object(job_dir / "validation_sampling.json")
    if representation.get("payload") is not None:
        representation["payload"]["coverage"] = sampling or _read_object(job_dir / "embeddings.json")
    if retrieval.get("payload") is not None:
        retrieval["payload"]["coverage"] = sampling
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
        "retrieval_errors": retrieval,
        "representation": representation,
        "group_results": _item(UNAVAILABLE, reason="subject_metadata_not_trusted")
        if not (job_dir / "subject_groups.json").is_file()
        else _item("observed", json.loads((job_dir / "subject_groups.json").read_text(encoding="utf-8"))),
        "intervention_probe": _item(UNAVAILABLE, reason="occlusion_not_requested"),
        "hook_consumption": hook_consumption(job_dir),
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
