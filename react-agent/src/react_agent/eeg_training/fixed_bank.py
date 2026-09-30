"""Fixed-candidate retrieval scores. Batch diagonals are not the selection metric."""

from __future__ import annotations

import math


def _cosine(left: list[float], right: list[float]) -> float:
    if any(not math.isfinite(value) for value in left) or any(not math.isfinite(value) for value in right):
        return float("nan")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def score_query(
    vector: list[float],
    bank: list[tuple[str, list[float]]],
    allowed: set[str],
    *,
    k: int = 1,
) -> dict:
    """Shared ranking used by the main evaluator and diagnostics."""
    non_finite = any(not math.isfinite(value) for value in vector) or any(
        any(not math.isfinite(value) for value in candidate) for _image, candidate in bank
    )
    zero_norm_query = _norm(vector) == 0
    scored = [(image_id, _cosine(vector, candidate)) for image_id, candidate in bank]
    finite_scores = [score for _image, score in scored if math.isfinite(score)]
    all_ties = bool(finite_scores) and max(finite_scores) == min(finite_scores)
    ranked = sorted(scored, key=lambda item: (item[1] if math.isfinite(item[1]) else float("-inf")), reverse=True)
    pos = [(image_id, score) for image_id, score in ranked if image_id in allowed]
    if not pos:
        return {"skip": True}
    best_pos = max(score for _image, score in pos)
    best_neg = max((score for image_id, score in ranked if image_id not in allowed), default=float("-inf"))
    min_rank = 1 + sum(1 for _image, score in ranked if math.isfinite(score) and math.isfinite(best_pos) and score > best_pos)
    max_rank = 1 + sum(1 for _image, score in ranked if math.isfinite(score) and math.isfinite(best_pos) and score >= best_pos) - 1
    max_rank = max(min_rank, max_rank)
    collapsed = non_finite or zero_norm_query or all_ties
    actual_top1 = bool(ranked and ranked[0][0] in allowed) and not collapsed
    actual_topk = any(image_id in allowed for image_id, _score in ranked[:k]) and not collapsed
    return {
        "skip": False,
        "positive_identity": [image_id for image_id, _score in pos],
        "positive_count": len(pos),
        "positive_score": best_pos if math.isfinite(best_pos) else None,
        "best_negative_score": None if best_neg == float("-inf") or not math.isfinite(best_neg) else best_neg,
        "margin": None if best_neg == float("-inf") or not math.isfinite(best_pos) or not math.isfinite(best_neg) else best_pos - best_neg,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "rank_interval": [min_rank, max_rank],
        "top_k_hit": actual_top1 if k == 1 else actual_topk,
        "all_ties": all_ties,
        "zero_norm_query": zero_norm_query,
        "non_finite": non_finite,
        "collapsed": collapsed,
    }


def fixed_bank_accuracy(
    queries: list[tuple[str, list[float]]],
    bank: list[tuple[str, list[float]]],
    positives: dict[str, set[str]],
    *,
    chunk_size: int = 0,
) -> dict[str, float]:
    """Score every query against the same frozen bank. Chunk size does not change the bank."""
    if not queries or not bank:
        raise ValueError("empty_bank")
    size = chunk_size if chunk_size > 0 else len(queries)
    hits1 = 0
    hits5 = 0
    for start in range(0, len(queries), size):
        for query_id, vector in queries[start : start + size]:
            allowed = positives.get(query_id) or set()
            if not allowed:
                raise ValueError(f"missing_positive:{query_id}")
            top1 = score_query(vector, bank, allowed, k=1)
            top5 = score_query(vector, bank, allowed, k=5)
            if top1.get("top_k_hit"):
                hits1 += 1
            if top5.get("top_k_hit"):
                hits5 += 1
    total = len(queries)
    return {
        "fixed_bank_top1": hits1 / total,
        "fixed_bank_top5": hits5 / total,
        "query_count": float(total),
        "candidate_count": float(len(bank)),
    }


def fixed_bank_hits(query, bank, labels) -> tuple[int, int]:
    """Top-1 and top-5 hits for one batch of tensors against the whole bank."""
    from torch.nn import functional as F

    q = F.normalize(query.float(), dim=-1)
    b = F.normalize(bank.float().to(q.device), dim=-1)
    target = labels.to(q.device).long()
    top = (q @ b.T).topk(min(5, b.shape[0]), dim=-1).indices
    hits1 = int((top[:, 0] == target).sum().item())
    hits5 = int((top == target[:, None]).any(dim=-1).sum().item())
    return hits1, hits5


class FixedBankTally:
    """Sum hits over validation batches. The bank stays the same for every batch."""

    def __init__(self, bank, labels) -> None:
        self.bank = bank
        self.labels = labels
        self.offset = 0
        self.hits1 = 0
        self.hits5 = 0

    def add(self, query) -> None:
        count = int(query.shape[0])
        batch = self.labels[self.offset : self.offset + count]
        self.offset += count
        hits1, hits5 = fixed_bank_hits(query, self.bank, batch)
        self.hits1 += hits1
        self.hits5 += hits5

    def result(self) -> dict[str, float]:
        if self.offset == 0:
            raise ValueError("empty_bank")
        return {
            "fixed_bank_top1": self.hits1 / self.offset,
            "fixed_bank_top5": self.hits5 / self.offset,
            "query_count": float(self.offset),
            "candidate_count": float(self.bank.shape[0]),
        }


def frozen_bank(records) -> tuple[object, object]:
    """One vector per image id in first-seen order, and each record's bank index."""
    import torch

    index: dict[str, int] = {}
    vectors = []
    labels = []
    for row in records:
        image_id = str(row["img"])
        if image_id not in index:
            index[image_id] = len(vectors)
            vectors.append(row["img_features"].float())
        labels.append(index[image_id])
    if not vectors:
        raise ValueError("empty_bank")
    return torch.stack(vectors), torch.tensor(labels, dtype=torch.long)
