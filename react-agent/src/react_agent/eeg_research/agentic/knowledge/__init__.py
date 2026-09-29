"""Local method cards. Retrieval is keyword overlap until a vector index is added."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CARDS = Path(__file__).resolve().parent.parent / "method_cards.json"


def load_cards() -> list[dict[str, Any]]:
    """Return the bundled method cards. A missing file is an empty library."""
    if not CARDS.is_file():
        return []
    payload = json.loads(CARDS.read_text(encoding="utf-8"))
    return list(payload) if isinstance(payload, list) else []


def retrieve_methods(query: str, *, limit: int = 5, online: bool = False) -> dict[str, Any]:
    """Score cards by token overlap. There is no network literature backend in this build.

    online=True is ignored: the result stays local_only and never pretends a web search ran.
    """
    del online
    cards = load_cards()
    tokens = {part.lower() for part in query.replace(",", " ").split() if part.strip()}
    if not tokens:
        hits = cards[:limit]
    else:
        ranked: list[tuple[int, dict[str, Any]]] = []
        for card in cards:
            blob = " ".join(str(card.get(key) or "") for key in ("id", "title", "mechanism", "tags", "when")).lower()
            score = sum(1 for token in tokens if token in blob)
            ranked.append((score, card))
        ranked.sort(key=lambda item: item[0], reverse=True)
        hits = [card for score, card in ranked if score > 0][:limit]
    labeled = []
    for card in hits:
        row = dict(card)
        row["retrieval"] = "local_only"
        row["online"] = False
        labeled.append(row)
    return {
        "search_scope": "local_method_cards",
        "local_only": True,
        "online": False,
        "query": query,
        "hits": labeled,
        "sources": [
            {
                "id": row.get("id") or row.get("method_id"),
                "access": "local_only",
                "source_refs": row.get("source_refs") or [],
            }
            for row in labeled
        ],
    }
