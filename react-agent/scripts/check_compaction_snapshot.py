"""Replay a captured real input through a fake SDK transport in a new directory.

This checks wiring, budgets, provenance, and recovery. It does not validate a
real model's summarization quality or create scientific lessons.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from react_agent.eeg_research.agentic.compaction import canonical, split_input, NOTE_FIELDS
from react_agent.eeg_research.agentic.llm import role_backend
from react_agent.eeg_research.agentic.roles import begin_role_task
from react_agent.fmri.schemas import LMUsage


def check(snapshot: Path, goal: Path, output: Path) -> dict:
    output = output.resolve()
    if output.is_relative_to(goal.resolve().parent):
        raise ValueError("engineering_output_must_be_outside_native_campaign")
    output.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix="isolated_engineering_", dir=output))
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    original = copy.deepcopy(payload)
    (case / "goal.json").write_bytes(goal.read_bytes())
    calls = []

    class Fake:
        def __init__(self, _config):
            pass

        async def complete_json(self, **kwargs):
            calls.append(kwargs)
            request = json.loads(kwargs["user"])
            if request.get("operation") == "handoff_compaction":
                refs = request["expected_source_refs"]
                result = {"source_refs": refs, **{field: [] for field in NOTE_FIELDS}}
                result["observations"] = [{
                    "text": "Offline interface fixture: preserve the native complete benchmark and its scope.",
                    "source_refs": refs,
                    "facts": [{"path": "/run/aggregate/mean_top1",
                               "value": payload["run"]["aggregate"]["mean_top1"]}]}]
                result["contradictions"] = [{
                    "text": "The Analysis query-population statement conflicts with the native per-fold benchmark counts.",
                    "source_refs": refs, "facts": []}]
                result["uncertainties"] = [{
                    "text": "Single-seed exploratory benchmark; sampled diagnostics do not prove full-population collapse.",
                    "source_refs": refs, "facts": []}]
            else:
                result = {"proposed_lessons": [], "proposed_skills": [],
                          "summary_zh": "Offline interface fixture; no scientific memory proposal."}
            return result, LMUsage(provider="deepseek", requested_model="offline-interface-fixture",
                response_model="offline-interface-fixture", input_tokens=0, output_tokens=0,
                success=True, finish_reason="stop")

    with patch("dotenv.load_dotenv", lambda *_a, **_k: False), \
         patch("react_agent.fmri.llm.deepseek.DeepSeekBackend", Fake), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-no-transport",
                    "DEEPSEEK_FAST_MODEL": "offline-interface-fixture",
                    "EEG_HANDOFF_COMPACTION": "1"}):
        backend = role_backend(case, "memory_curator")
        dependencies = [case / "goal.json", snapshot.resolve()]
        prepared = backend.prepare_context(payload, inputs=dependencies)
        assert payload == original
        assert prepared["run"]["aggregate"] == original["run"]["aggregate"]
        assert prepared["benchmark_population"]["query_count_total"] == sum(
            row["metrics"]["query_count"] for row in original["run"]["aggregate"]["scores"])
        map_requests = [json.loads(row["user"]) for row in calls
                        if json.loads(row["user"]).get("stage") == "map"]
        assert "".join(row["part"]["text"] for row in map_requests) == canonical(original)
        count = len(calls)
        cached = backend.prepare_context(payload, inputs=dependencies)
        assert cached == prepared and len(calls) == count
        receipt = prepared["context_compaction"]
        task = begin_role_task(case, role="memory_curator", inputs=dependencies,
            artifacts=[{"artifact_id": receipt["artifact_id"], "sha256": receipt["sha256"]}],
            request=prepared)
        response = backend({**prepared, **{key: task[key] for key in
                                          ("task_id", "attempt_id", "input_digest")}})
        assert response["payload"]["proposed_lessons"] == []
        assert not (case / "memory.sqlite").exists()
    result = {
        "scope": "Captured real source input with FAKE SDK transport; no real API, training, or scientific memory.",
        "model_summary_quality": "not_run; declared synthetic summary only",
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "full_source_coverage": True, "native_aggregate_unchanged": True,
        "compaction_mock_calls": count, "consumer_mock_calls": 1,
        "cache_replay_mock_calls": 0, "real_api_calls": 0,
        "consumer_chars": len(calls[-1]["system"]) + len(calls[-1]["user"]),
        "consumer_budget_chars": 100000,
        "benchmark_query_count_total": prepared["benchmark_population"]["query_count_total"],
        "map_request_chars": [len(row["system"]) + len(row["user"]) for row in calls[:-1]
                              if json.loads(row["user"]).get("stage") == "map"],
        "source_ranges": prepared["context_compaction"]["source_ranges"],
        "isolated_directory": str(case),
    }
    (output / "real_input_offline_replay.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.snapshot, args.goal, args.output), ensure_ascii=False, indent=2))
