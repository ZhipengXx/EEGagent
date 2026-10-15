"""Resource admission recovers frozen cards for an unhydrated campaign state."""
from __future__ import annotations

import json

from react_agent.eeg_research.agentic.resource_guard import require_idle_gpus


def test_protocol_cards_admit_checks_when_runtime_state_has_no_gpu(tmp_path, monkeypatch):
    (tmp_path / "goal.json").write_text(json.dumps({"allow_shared_gpus": True}))
    (tmp_path / "execution_protocol.json").write_text(json.dumps({"gpu": [1, 4]}))
    inventories = []
    def inventory(cards, *, allow_shared):
        inventories.append((cards, allow_shared))
        return []
    monkeypatch.setattr("react_agent.eeg_research.agentic.loso_study.gpu_inventory", inventory)
    require_idle_gpus(tmp_path, {"evaluation_mode": "loso_method_search"})
    assert inventories == [((1, 4), True)]


def test_existing_runtime_cards_preserve_resource_selection(tmp_path, monkeypatch):
    (tmp_path / "goal.json").write_text("{}")
    (tmp_path / "execution_protocol.json").write_text(json.dumps({"gpu": [1, 4]}))
    inventories = []
    def inventory(cards, *, allow_shared):
        inventories.append((cards, allow_shared))
        return []
    monkeypatch.setattr("react_agent.eeg_research.agentic.loso_study.gpu_inventory", inventory)
    require_idle_gpus(tmp_path, {"evaluation_mode": "loso_method_search", "gpu": [2, 3]})
    assert inventories == [((2, 3), False)]
