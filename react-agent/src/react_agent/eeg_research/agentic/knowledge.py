"""Local method cards. Retrieval is keyword overlap until a vector index is added."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CARDS = Path(__file__).resolve().parent / "method_cards.json"


def load_cards() -> list[dict[str, Any]]:
    """Return the bundled method cards. A missing file is an empty library."""
    if not CARDS.is_file():
        return []
    payload = json.loads(CARDS.read_text(encoding="utf-8"))
    return list(payload) if isinstance(payload, list) else []


def retrieve_methods(query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Score cards by token overlap. Empty query returns the first cards."""
    cards = load_cards()
    tokens = {part.lower() for part in query.replace(",", " ").split() if part.strip()}
    if not tokens:
        return cards[:limit]
    ranked: list[tuple[int, dict[str, Any]]] = []
    for card in cards:
        blob = " ".join(str(card.get(key) or "") for key in ("id", "title", "mechanism", "tags", "when")).lower()
        score = sum(1 for token in tokens if token in blob)
        ranked.append((score, card))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [card for score, card in ranked if score > 0][:limit]
