"""Fixed-candidate retrieval scores. Batch diagonals are not the selection metric."""

from __future__ import annotations

import math


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


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
            ranked = sorted(
                ((image_id, _cosine(vector, candidate)) for image_id, candidate in bank),
                key=lambda item: item[1],
                reverse=True,
            )
            top = [image_id for image_id, _score in ranked[:5]]
            if top and top[0] in allowed:
                hits1 += 1
            if any(image_id in allowed for image_id in top):
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
