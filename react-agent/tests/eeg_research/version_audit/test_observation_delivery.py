"""Public P1-5/P2 fixtures: native SQLite/readback/PreparedRequest, no API/GPU.

Numbers are synthetic assertions about exact delivery, not EEG model accuracy.
The deterministic backend verifies index plumbing; the optional pinned CPU
MiniLM test verifies only local encoding/retrieval/cache, not research quality.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from react_agent.eeg_research.agentic import experiment_observations as cards, semantic_memory as sm
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import BudgetConfig, CompileRequest, NeedsContext, ReadRequest, Ready, canonical
from react_agent.eeg_research.agentic.embedding import EmbeddingUnavailable, backend_for, memory_config


PIN = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def campaign(tmp_path, *, memory=False):
    camp = tmp_path / "synthetic-observation-delivery"
    camp.mkdir()
    (camp / "goal.json").write_text(json.dumps({"goal_id": camp.name, "objective": "Diagnose temporal alignment",
        "research_execution": {"version": "single_plan_v1", "paid_maintenance": "off"}, "max_llm_calls": 20,
        "memory": {"enabled": memory, "model_revision": PIN, "allow_model_download": False,
                   "min_similarity": -1, "top_k": 5, "context_chars_budget": 16000}}))
    (camp / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval","dataset":"EEG-synthetic"}')
    (camp / "execution_protocol.json").write_text('{"dataset":"EEG-synthetic","modality":"EEG"}')
    return camp


def observation(camp, *, i=1, diagnostics=False, analyst=False):
    store = cards.ExperimentObservationStore(camp)
    facts = {"scope": "two_fold_development", "fidelity": "full", "seed": 0, "coverage": 2,
        "required_fold_count": 2, "full_required_fold_count": 10,
        "screening_fold_ids": [0, 1], "screening_subjects": ["s01", "s02"],
        "metrics": {"synthetic_top1": 12.3456789012345}, "paired_delta_pp": -0.1234567890123,
        "promotion": "not_promoted", "missing": [], "independently_replicated": False}
    if diagnostics:
        facts.update(measurement_kind="existing_development_diagnostic", metrics=None,
            paired_delta_pp=None, native_diagnostics=[{"candidate_id": "c" + str(i), "fold_id": fold,
                "held_out_subject": "s0" + str(fold + 1), "seed": 0,
                "source_hash": "a" * 64, "spec_hash": "b" * 64,
                "diagnostics": {"status": "ready", "exact_direction_norm": 0.987654321098765,
                                "report_text": "HUGE_DIAGNOSTIC" * 10000},
                "learning_profile": {"status": "ready", "best_epoch": 17, "best_fixed_bank_top1": 12.3456789012345}}
                for fold in range(2)], summary={"native_source_count": 2, "summary_text": "HUGE_SUMMARY" * 10000})
    native = store.registry.capture(facts, block_id="synthetic-native:" + str(i), kind="scientific")
    body = {"origin_campaign_id": camp.name, "plan_id": "plan" + str(i),
        "question": "Why is temporal alignment weak?", "intervention": "synthetic mechanism",
        "observation_kind": "experiment", "execution_status": "observation_ready",
        "identity": {"candidate_id": "c" + str(i), "method_revision": "r1", "source_hash": "a" * 64,
                     "spec_hash": "b" * 64, "dataset": "EEG-synthetic", "modality": "EEG"},
        "native_facts": facts, "source_refs": [{"source_ref": native.source_ref, "source_hash": native.source_hash}],
        "analyst_interpretation": None}
    if analyst:
        body["analyst_interpretation"] = {"interpretation_only": True,
            "points": {"summary_zh": "HUGE_ANALYST_INTERPRETATION" * 10000},
            "candidate_id": "c" + str(i)}
    return store.record(body)


def current_context(saved):
    return {**cards.observation_conditions(saved), "candidate_id": saved["identity"]["candidate_id"],
            "terminal_observation_id": saved["observation_id"]}


def compile_current(camp, saved, *, budget=None):
    sent = []
    service = ContextService(camp, maintenance_callback=lambda **kwargs: sent.append(kwargs))
    context = current_context(saved)
    payload = {"current_task_context": context, "terminal_observation_id": saved["observation_id"],
               "selected_plan": {"plan_id": saved["plan_id"], "terminal_observation_id": saved["observation_id"]}}
    result = service.compile(CompileRequest(camp.name, "research_planner", payload,
        system="Compare research tasks against exact native feedback.", budget=budget or BudgetConfig()))
    assert not sent
    with service.snapshot.persistence.db() as db:
        assert db.execute("SELECT calls FROM context_maintenance_usage").fetchone()[0] == 0
    assert not (camp / "llm_calls.jsonl").exists()
    return result


def feedback(result):
    assert isinstance(result, Ready), result
    assert result.prepared.verify() and result.prepared.budget_report["ready_budget_ok"]
    card = result.prepared.payload["experiment_observations"]["cards"][0]
    assert card["current_feedback"] is True and card["fidelity"] == "FACT_CARD"
    return card


def assert_original_readable(camp, saved, card):
    store = cards.ExperimentObservationStore(camp)
    handle = card["full_record"]
    assert store.registry.load(store.registry.get(handle["$context_source"], role="research_planner"),
                               role="research_planner") == saved
    assert json.loads((camp / "experiment_observations" / (saved["observation_id"] + ".json")).read_text()) == saved
    assert store.records(limit=1, observation_id=saved["observation_id"])[0]["body"] == saved
    for request in card["next_reads"]:
        result = store.registry.read(ReadRequest(**request), role="research_planner")
        assert result.status == "ready", result
        assert result.satisfies_development_read_receipt is False


def test_30k_engineering_failure_is_current_exact_bounded_and_readable(tmp_path):
    camp = campaign(tmp_path)
    plan = {"plan_id": "failed-current", "candidate_id": "c1", "source_hash": "a" * 64, "spec_hash": "b" * 64,
        "repair_count": 2, "engineering_failure": {"error": "checkpoint_mismatch " + "X" * 30000,
          "error_category": "checkpoint_mismatch"}, "proposal": {"task_kind": "new_method",
          "question": "Implement temporal alignment", "scientific_intent": "synthetic temporal model",
          "evaluation_scope": {"phase": "screening_2fold", "scope": "two_fold_development", "seed": 0,
                               "folds": 2, "subjects": ["s01", "s02"]}}}
    saved = cards.record_experiment_observation(camp, {}, plan, "failed")
    card = feedback(compile_current(camp, saved))
    assert card["record_id"] == saved["observation_id"] and card["body"]["plan_id"] == plan["plan_id"]
    assert card["body"]["identity"] == saved["identity"]
    assert card["body"]["evaluation_scope"] == saved["evaluation_scope"]
    assert card["body"]["execution_status"] == "failed"
    assert card["body"]["native_facts"]["metrics"] is None
    assert card["body"]["engineering_failure"]["error_fingerprint"] == saved["engineering_failure"]["error_fingerprint"]
    assert card["body"]["engineering_failure"]["error_category"] == "checkpoint_mismatch"
    assert card["body"]["engineering_failure"]["repairs_used"] == 2
    assert any(row["path"] == "/engineering_failure/error" for row in card["omissions"])
    assert len(canonical(card)) < 8000
    assert_original_readable(camp, saved, card)


@pytest.mark.parametrize("diagnostics,analyst", [(True, False), (False, True), (True, True)])
def test_huge_diagnostic_and_analyst_keep_current_exact_facts(tmp_path, diagnostics, analyst):
    camp = campaign(tmp_path)
    saved = observation(camp, diagnostics=diagnostics, analyst=analyst)
    card = feedback(compile_current(camp, saved))
    assert card["body"]["identity"] == saved["identity"]
    for key in ("scope", "seed", "coverage", "required_fold_count", "full_required_fold_count",
                "metrics", "paired_delta_pp", "screening_fold_ids", "screening_subjects"):
        assert card["body"]["native_facts"][key] == saved["native_facts"][key]
    if diagnostics:
        assert card["body"]["native_facts"]["native_diagnostics"][0]["diagnostics"]["exact_direction_norm"] == 0.987654321098765
        assert card["body"]["native_facts"]["native_diagnostics"][0]["learning_profile"]["best_epoch"] == 17
    if analyst:
        assert any(row["path"] == "/analyst_interpretation" for row in card["omissions"])
    assert "HUGE_DIAGNOSTIC" not in canonical(card) and "HUGE_ANALYST_INTERPRETATION" not in canonical(card)
    assert len(canonical(card)) < 8000
    assert_original_readable(camp, saved, card)


def test_insufficient_current_minimum_is_true_needs_context(tmp_path):
    camp = campaign(tmp_path)
    saved = observation(camp, diagnostics=True, analyst=True)
    budget = BudgetConfig(hard_chars=1000, target_chars=600, high_chars=700, fallback_chars=900, retrieval_chars=1000)
    result = compile_current(camp, saved, budget=budget)
    assert isinstance(result, NeedsContext) and result.reason == "current_observation_minimum_exceeds_budget"
    assert result.missing[0]["observation_id"] == saved["observation_id"] and result.reads
    assert result.resources["paid_maintenance_calls"] == 0
    assert cards.ExperimentObservationStore(camp).registry.read(result.reads[0], role="research_planner").status == "ready"
    # A previously compacted delivery must obey a smaller actual request cap.
    view = cards.compact_planner_view(camp, {"terminal_observation_id": saved["observation_id"]})
    second = ContextService(camp).compile(CompileRequest(camp.name, "research_planner", view, budget=budget))
    assert isinstance(second, NeedsContext) and second.reason == result.reason


def test_explicit_current_id_precedes_more_recent_history_and_unreadable_is_explicit(tmp_path):
    camp = campaign(tmp_path)
    saved = observation(camp, i=1)
    observation(camp, i=2)
    card = feedback(compile_current(camp, saved))
    assert card["record_id"] == saved["observation_id"]
    store = cards.ExperimentObservationStore(camp)
    source = store.records(limit=1, observation_id=saved["observation_id"])[0]["source_refs"][0]["source_ref"]
    store.registry.revoke(source)
    result = compile_current(camp, saved)
    assert isinstance(result, NeedsContext) and result.reason == "current_observation_unreadable"


class DeterministicBackend:
    """Synthetic unit vectors; this exercises native retrieval, not MiniLM quality."""
    metadata = {"model_id": "synthetic-vector", "model_revision": "f" * 40}
    fingerprint = "public-observation-test-v1"
    def encode(self, texts):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result, [{"input_token_count": len(text.split()), "truncated": False} for text in texts]


def test_recent_semantic_compact_share_conditions_and_never_match_different_scope(tmp_path, monkeypatch):
    camp = campaign(tmp_path, memory=True)
    saved = observation(camp)
    monkeypatch.setattr(sm, "backend_for", lambda config: DeterministicBackend())
    current = {"scope": "method_development_benchmark", "dataset": "MEG-synthetic", "modality": "MEG",
        "seed": 42, "required_fold_count": 10, "subjects": ["m01", "m02"], "fold_ids": list(range(10)),
        "source_hash": "c" * 64, "spec_hash": "d" * 64, "method_revision": "r2"}
    recent = cards.recall_observations(camp, "temporal alignment", recent=1, relevant=0, context=current)["cards"][0]
    bundle = sm.retrieve(camp, "temporal alignment", context=current, require_embedding=True)
    assert bundle["backend"] == "synthetic_embedding"  # Unit vectors are not the real MiniLM model.
    semantic = next(row for row in bundle["items"] if row["record_id"] == saved["observation_id"])
    view = cards.compact_planner_view(camp, {"current_task_context": current})
    compact = view["experiment_observations"]["cards"][0]
    for card in (recent, semantic, compact):
        assert card["applicability"] == "analogy_only"
        differences = card["condition_differences"]
        assert differences["matched_comparison"] is False
        fields = {row["field"] for row in differences["mismatch"]}
        assert {"scope", "dataset", "modality", "seed", "fold_count", "subjects", "fold_ids", "source_hash", "spec_hash"} <= fields
    assert recent["condition_differences"] == semantic["condition_differences"] == compact["condition_differences"]
    matching = cards.recall_observations(camp, "temporal alignment", context=current_context(saved))["cards"][0]
    assert matching["applicability"] == "compatible_evidence" and matching["condition_differences"]["matched_comparison"]
    unknown = cards.recall_observations(camp, "temporal alignment", context={})["cards"][0]
    assert unknown["applicability"] == "applicability_unknown" and not unknown["condition_differences"]["matched_comparison"]
    with sm.SidecarStore(camp).connect() as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_cache WHERE record_type='experiment_observation'").fetchone()[0] >= 1


def test_recall_envelope_obeys_char_and_utf8_caps(tmp_path):
    camp = campaign(tmp_path)
    saved = observation(camp, analyst=True)
    observation(camp, i=2)
    context = current_context(saved)
    result = cards.recall_observations(camp, "temporal alignment", context=context, char_cap=7000, token_cap=7000)
    assert len(canonical(result)) <= 7000 and len(canonical(result).encode("utf-8")) <= 7000
    assert result["cards"][0]["record_id"] == saved["observation_id"]


def test_wrapped_observation_uses_outer_current_task_conditions(tmp_path):
    camp = campaign(tmp_path)
    saved = observation(camp)
    current = {"scope": "method_development_benchmark", "dataset": "MEG-synthetic", "modality": "MEG",
        "seed": 42, "fold_count": 10, "subjects": ["m01"], "source_hash": "c" * 64}
    result = ContextService(camp).compile(CompileRequest(camp.name, "research_planner", {
        "current_task_context": current, "terminal_observation_id": saved["observation_id"], "observation": {}}))
    assert isinstance(result, Ready), result
    card = result.prepared.payload["observation"]["experiment_observations"]["cards"][0]
    assert card["applicability"] == "analogy_only"
    assert {row["field"] for row in card["condition_differences"]["mismatch"]} >= {"scope", "dataset", "modality", "seed", "fold_count", "subjects", "source_hash"}


def test_unclassified_current_observation_fields_stay_literal_or_needs_context(tmp_path):
    camp = campaign(tmp_path)
    saved = observation(camp)
    custom = copy.deepcopy(saved)
    for key in ("observation_id", "content_digest", "schema_version"):
        custom.pop(key)
    custom["plan_id"] = "with-new-current-obligation"
    custom["new_scientific_gate"] = {"required": "unknown_current_obligation"}
    small = cards.ExperimentObservationStore(camp).record(custom)
    assert feedback(compile_current(camp, small))["body"]["new_scientific_gate"] == custom["new_scientific_gate"]
    custom["plan_id"] = "with-huge-new-current-obligation"
    custom["new_scientific_gate"]["required"] *= 3000
    huge = cards.ExperimentObservationStore(camp).record(custom)
    result = compile_current(camp, huge)
    assert isinstance(result, NeedsContext) and result.reason == "current_observation_minimum_exceeds_budget"


def test_local_pinned_minilm_retrieval_and_restart_cache(tmp_path):
    camp = campaign(tmp_path, memory=True)
    saved = observation(camp)
    try:
        backend = backend_for(memory_config(camp))
    except EmbeddingUnavailable as exc:
        pytest.skip("Pinned local MiniLM cache unavailable; no download permitted: " + str(exc))
    context = current_context(saved)
    first = sm.retrieve(camp, "temporal alignment", context=context, require_embedding=True)
    second = sm.retrieve(camp, "temporal alignment", context=context, require_embedding=True)
    assert first["backend"] == "minilm" and first["fingerprint"] == backend.fingerprint
    assert saved["observation_id"] in {row["record_id"] for row in first["items"]}
    assert second["indexing"]["indexed"] == 0 and second["indexing"]["reused"] >= 1
    with sm.SidecarStore(camp).connect() as db:
        rows = db.execute("SELECT encoding_metadata FROM embedding_cache WHERE record_type='experiment_observation'").fetchall()
    assert rows and all(json.loads(row[0])["encoded_token_count"] <= backend.metadata["max_sequence_length"] for row in rows)


def native_selected_campaign(tmp_path):
    from react_agent.eeg_research.agentic import loop
    goal = {"goal_id": "native-context", "max_llm_calls": 100, "max_training_jobs": 10,
        "max_gpu_seconds": 100, "max_candidates": 4, "allowed_training_actions": ["run_pilot", "run_full"],
        "research_execution": {"version": "single_plan_v1", "target_k": 2, "paid_maintenance": "off",
            "budget": {"envelopes": {"new_method": {"role_calls": {"experiment_designer": 2,
                "candidate_coder": 6, "candidate_reviewer": 2, "result_analyst": 2},
                "training_jobs": 2, "gpu_seconds": 40}}}}}
    loop.create_campaign(tmp_path, goal=goal,
        contract={"fingerprint": "synthetic-context-contract", "task_type": "eeg_image_retrieval"},
        request_id="native-context")
    return tmp_path / "native-context"


def bind_terminal(camp, saved):
    from react_agent.eeg_research.agentic import loop
    state = loop.load_state(camp)
    state.update(terminal_observation_id=saved["observation_id"], terminal_observation_digest=saved["content_digest"])
    loop.save_state(camp, state)


def test_native_selected_persists_needs_context_reads_before_provider_and_owner(tmp_path):
    from react_agent.eeg_research.agentic import loop
    camp = native_selected_campaign(tmp_path)
    saved = observation(camp)
    body = {key: value for key, value in saved.items() if key not in {"schema_version", "observation_id", "content_digest"}}
    body.update(plan_id="native-unclassified-current", new_scientific_gate="EXACT_CURRENT_OBLIGATION" * 3000)
    saved = cards.ExperimentObservationStore(camp).record(body)
    bind_terminal(camp, saved)
    original = (camp / "experiment_observations" / (saved["observation_id"] + ".json")).read_bytes()
    calls = []
    state = loop.tick(camp, lambda request: calls.append(request))
    assert state["status"] == "blocked" and state["failure"]["error_type"] == "context_needs_context"
    need = loop.load_state(camp)["context_requirement"]
    assert need["status"] == need["result"]["status"] == "needs_context"
    assert need["input_observation_digest"] == saved["content_digest"]
    assert need["result"]["missing"][0]["observation_id"] == saved["observation_id"]
    assert need["result"]["reads"]
    reply = cards.ExperimentObservationStore(camp).registry.read(ReadRequest(**need["result"]["reads"][0]), role="research_planner")
    assert reply.status == "ready" and reply.content.startswith("EXACT_CURRENT_OBLIGATION")
    assert not calls and not (camp / "llm_calls.jsonl").exists()
    with sqlite3.connect(camp / "resource_owners.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM owners").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    assert not list((camp / "research_rounds").glob("*.json"))
    loop.tick(camp, lambda request: calls.append(request))
    assert loop.load_state(camp)["context_requirement"] == need
    assert (camp / "experiment_observations" / (saved["observation_id"] + ".json")).read_bytes() == original


def test_native_selected_resume_after_source_restored_uses_same_feedback(tmp_path, monkeypatch):
    from react_agent.eeg_research.agentic import cli, loop, selected_plan
    camp = native_selected_campaign(tmp_path)
    saved = observation(camp)
    bind_terminal(camp, saved)
    store = cards.ExperimentObservationStore(camp)
    native_ref = saved["source_refs"][0]
    native = store.registry.get(native_ref["source_ref"], role="research_planner")
    original_native = store.registry.load(native, role="research_planner")
    store.registry.capture({"seed": 42}, block_id=native.block_id, kind="scientific")
    calls = []
    blocked = loop.tick(camp, lambda request: calls.append(request))
    assert blocked["context_requirement"]["result"]["reason"] == "current_observation_unreadable" and not calls
    requirement = copy.deepcopy(blocked["context_requirement"])
    restored = store.registry.capture(original_native, block_id=native.block_id, kind="scientific")
    assert restored.source_ref == native.source_ref and restored.source_hash == native.source_hash
    monkeypatch.setattr(cli, "spawn_worker", lambda *args, **kwargs: 12345)
    assert cli.main(["resume", "--root", str(tmp_path), "--campaign", camp.name, "--max-ticks", "1"]) == 0
    def planner(request):
        calls.append(request)
        return {"protocol_version": "research_plan_draft.v1", "proposals": [{"proposal_id": pid,
            "task_kind": "new_method", "question": "Question " + pid, "task_description": "Bounded task " + pid,
            "scientific_intent": "Independent synthetic mechanism " + pid, "evaluation_scope": {"phase": "full", "seed": 0},
            "expected_observation": "Native feedback", "completion_condition": "One observation",
            "required_context_refs": [], "value_rationale": "Resolve uncertainty"} for pid in ("a", "b")],
            "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Information", "implementation_risk": "Unknown"}
                            for rank, pid in enumerate(("a", "b"), 1)],
            "selected_proposal_id": "a", "selection_rationale": "Compared bounded tasks"}
    resumed = loop.tick(camp, planner)
    assert selected_plan.load_plan(camp)["originating_round"] == requirement["round_id"]
    assert resumed["context_requirement"]["status"] == "resolved"
    assert canonical(resumed["context_requirement"]["result"]) == canonical(requirement["result"])
    assert resumed["context_requirement"]["resolved_input_observation_digest"] == saved["content_digest"]
    assert len(calls) == 1 and calls[0]["experiment_observations"]["cards"][0]["record_id"] == saved["observation_id"]
    assert not (camp / "llm_calls.jsonl").exists()
