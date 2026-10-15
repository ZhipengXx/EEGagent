"""Native knowledge -> actual Prepared Planner input, with no provider/GPU.

All records live in isolated synthetic campaigns. Only the final local-cache
case uses real MiniLM; deterministic vectors prove plumbing and are labelled.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from react_agent.eeg_research.agentic import experiment_observations as obs, loop, semantic_memory as sm, selected_plan
from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.sources import ContextSourceRegistry
from react_agent.eeg_research.agentic.context.types import CompileRequest, NeedsContext, ReadRequest, Ready, canonical
from react_agent.eeg_research.agentic.embedding import EmbeddingUnavailable, backend_for, memory_config
from react_agent.eeg_research.agentic.llm import system_prompt
from react_agent.eeg_research.agentic.memory import EpisodeStore, episode
from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task

PIN = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def campaign(tmp_path):
    goal = {"goal_id": "synthetic-knowledge", "objective": "Investigate temporal convolution EEG encoders",
        "max_training_jobs": 4, "max_llm_calls": 100, "max_gpu_seconds": 600, "max_candidates": 4,
        "allowed_training_actions": ["run_pilot"], "research_execution": {"version": "single_plan_v1", "target_k": 2,
            "paid_maintenance": "off"}, "memory": {"enabled": True, "model_revision": PIN,
            "allow_model_download": False, "min_similarity": .3, "top_k": 5, "context_chars_budget": 6000}}
    loop.create_campaign(tmp_path, goal=goal, contract={"task_type": "eeg_image_retrieval", "fingerprint": "synthetic_contract", "dataset": "eeg"}, request_id="synthetic-knowledge")
    return tmp_path / goal["goal_id"]


def lesson(camp, text="Temporal convolution learns local EEG temporal patterns before projection.", **conditions):
    store = EpisodeStore(camp)
    index = len(store.list_episodes())
    source = camp / ("synthetic_source_" + str(index) + ".json")
    source.write_text(canonical({"scope": "development", "synthetic": True, "note": text}))
    ep = store.persist_episode(episode(task_hash=camp.name, candidate_id="baseline", kind="exploratory_result",
        fidelity="pilot", seed=0, metric=None, contract_fingerprint="synthetic_contract", artifact=str(source), episode_id="ep_" + str(index)))
    accepted = store.accept_lessons({"proposed_lessons": [{"statement": text,
        "conditions": {"task": camp.name, "evaluation_identity": "synthetic_contract", "dataset": "eeg", **conditions},
        "supporting_episode_ids": [ep["episode_id"]]}]})
    assert len(accepted["accepted"]) == 1 and not accepted["rejected"]
    return accepted["accepted"][0]


class SyntheticVectors:
    metadata = {"model_id": "synthetic-unit-vectors", "model_revision": "f" * 40}
    fingerprint = "synthetic-planner-knowledge-v1"
    def __init__(self, fail_call=None):
        self.calls, self.fail_call = [], fail_call
    def encode(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("synthetic_encoding_failure")
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for n, text in enumerate(texts):
            vectors[n, 1 if "UNRELATED" in text else 0] = 1
        return vectors, [{"input_token_count": len(text.split()), "truncated": False} for text in texts]


def compile_input(camp, payload):
    service = ContextService(camp, maintenance_callback=lambda **_: pytest.fail("No paid memory maintenance"))
    return service.compile(CompileRequest(camp.name, "research_planner", payload,
        system=system_prompt("research_planner", camp=camp)))


def expand(payload):
    def pointer(path):
        current = payload
        for part in path.lstrip("/").split("/") if path else []:
            part = part.replace("~1", "/").replace("~0", "~")
            current = current[int(part)] if isinstance(current, list) else current[part]
        return current
    def visit(value):
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return visit(pointer(value["$context_pointer"]))
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return visit(payload)


def test_old_relevant_lesson_reaches_actual_prepared_with_exact_sources_and_readback(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    relevant = lesson(camp, scope="development", modality="EEG", seed=0, fold_count=2, subjects=["e1", "e2"])
    for n in range(20):
        lesson(camp, "UNRELATED archival note " + str(n))
    assert relevant["lesson_id"] not in {x["lesson_id"] for x in EpisodeStore(camp).list_lessons(limit=5)}
    vectors = SyntheticVectors()
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: vectors)
    original = canonical(relevant)
    current = {"current_question": "Which temporal convolution mechanism explains weak EEG alignment?",
        "current_task_context": {"dataset": "meg", "modality": "MEG", "seed": 42, "fold_count": 10,
                                 "subjects": ["m1"], "scope": "method_development_benchmark"},
        "hypothesis": {"mechanism": "temporal convolution encoder"}}
    prepared = compile_input(camp, current)
    assert isinstance(prepared, Ready), prepared
    assert prepared.prepared.verify() and prepared.prepared.budget_report["ready_budget_ok"]
    actual = expand(json.loads(prepared.prepared.user))
    health = actual["memory_retrieval_health"]
    assert health["backend"] == "synthetic_embedding" and not health["degraded"]
    assert health["model"] == vectors.metadata["model_id"] and health["fingerprint"] == vectors.fingerprint
    assert health["query_sha256"] == hashlib.sha256(health["query"].encode()).hexdigest()
    assert "temporal" in health["query"] and health["query_status"] in {"encoded", "cache_reused"}
    card = next(x for x in actual["retrieved_knowledge"]["items"] if x["record_id"] == relevant["lesson_id"])
    assert card["body"]["statement"] == relevant["statement"] and card["cosine_score"] == 1.
    assert card["applicability"] == "analogy_only" and not card["scientific_evidence"] and not card["execution_authority"]
    assert {x["field"] for x in card["condition_differences"]["mismatch"]} >= {"dataset", "modality", "seed", "fold_count", "subjects", "scope"}
    assert card["full_record"]["$context_source"] in {x["$context_source"] for x in actual["history_navigation"]}
    registry = ContextSourceRegistry(camp)
    source = registry.get(card["full_record"]["$context_source"], role="research_planner")
    assert canonical(registry.load(source, role="research_planner")) == original
    read = registry.read(ReadRequest(source.source_ref, source.source_hash, "/statement", max_chars=1000), role="research_planner")
    assert read.status == "ready" and read.content == relevant["statement"]
    assert canonical(next(x for x in EpisodeStore(camp).list_lessons() if x["lesson_id"] == relevant["lesson_id"])) == original
    assert not (camp / "llm_calls.jsonl").exists()


def test_healthy_cold_start_really_encodes_query_and_reports_zero_knowledge(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    vectors = SyntheticVectors()
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: vectors)
    result = compile_input(camp, {"current_question": "Temporal EEG convolution mechanism"})
    assert isinstance(result, Ready), result
    actual = expand(result.prepared.payload)
    assert actual["retrieved_knowledge"]["status"] == "no_related_knowledge" and actual["retrieved_knowledge"]["items"] == []
    health = actual["memory_retrieval_health"]
    assert not health["degraded"] and health["candidate_count"] == 0 and health["hit_count"] == 0
    assert health["indexing"]["failed"] == 0 and health["query_status"] in {"encoded", "cache_reused"}
    assert vectors.calls and any("Temporal" in text for group in vectors.calls for text in group)


@pytest.mark.parametrize("phase", ["load", "index", "query"])
def test_embedding_failure_is_native_recoverable_before_planner_owner_or_provider(tmp_path, monkeypatch, phase):
    camp = campaign(tmp_path)
    lesson(camp)
    if phase == "load":
        def unavailable(*_a, **_k):
            raise EmbeddingUnavailable("model_unavailable_synthetic")
        monkeypatch.setattr(sm, "backend_for", unavailable)
    else:
        vectors = SyntheticVectors(fail_call=1 if phase == "index" else 2)
        monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: vectors)
    before = (camp / "cost.json").read_bytes() if (camp / "cost.json").exists() else None
    state = loop.load_state(camp)
    stopped = selected_plan._select(camp, state, lambda _: pytest.fail("Embedding failure must precede provider"))
    requirement = stopped["context_requirement"]
    assert requirement["status"] == "needs_context" and requirement["result"]["reason"] == "embedding_retrieval_unavailable"
    health = requirement["result"]["missing"][0]["embedding_health"]
    assert health["degraded"] and health["backend"] == "unavailable" and health["reason"]
    assert stopped["failure"]["recoverable"] is True
    assert not (camp / "llm_api_intents.jsonl").exists() and not (camp / "llm_calls.jsonl").exists()
    if before is not None:
        assert (camp / "cost.json").read_bytes() == before
    with sqlite3.connect(camp / "resource_owners.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM owners").fetchone()[0] == 0
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: SyntheticVectors())
    resumed = compile_input(camp, {"current_question": "Temporal EEG convolution mechanism"})
    assert isinstance(resumed, Ready), resumed
    assert not expand(resumed.prepared.payload)["memory_retrieval_health"]["degraded"]


def test_registered_native_method_card_is_similarity_knowledge_not_keyword_or_measurement(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    body = {"id": "temporal", "title": "Temporal EEG encoder", "mechanism": "Temporal convolution before EEG projection",
            "retrieval": "local_only", "online": False, "source_refs": [{"kind": "synthetic_fixture", "path": "fixture_only"}]}
    task = begin_role_task(camp, role="research_librarian", inputs=[camp / "goal.json", camp / "evaluation_contract.json"], request={"query": "Temporal EEG"})
    finish_role_task(camp, task, {"status": "completed", "local_only": True, "method_cards": [body]},
        kind="method_hits", path=camp / "knowledge/native_synthetic_method_card.json")
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: SyntheticVectors())
    result = compile_input(camp, {"current_question": "Temporal EEG convolution mechanism"})
    assert isinstance(result, Ready), result
    knowledge = expand(result.prepared.payload)["retrieved_knowledge"]["items"]
    card = next(x for x in knowledge if x["record_type"] == "knowledge")
    assert "Temporal convolution" in card["body"]["statement"] and card["cosine_score"] == 1.
    assert card["similarity_method"] == "synthetic_embedding_cosine" and card["source_retrieval"] == "native_registered_local_method_card"
    assert not card["scientific_evidence"] and not card["condition_differences"]["matched_comparison"]
    registry = ContextSourceRegistry(camp)
    assert registry.load(registry.get(card["full_record"]["$context_source"], role="research_planner"), role="research_planner") == body


def test_current_terminal_survives_knowledge_and_query_uses_failure_feedback(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    lesson(camp)
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: SyntheticVectors())
    saved = obs.record_experiment_observation(camp, {}, {"plan_id": "synthetic_failed_current", "candidate_id": "c1",
        "source_hash": "a" * 64, "spec_hash": "b" * 64, "engineering_failure": {"error": "X" * 30000, "error_category": "checkpoint_mismatch"},
        "proposal": {"question": "Temporal convolution alignment", "scientific_intent": "Synthetic mechanism",
                     "evaluation_scope": {"phase": "pilot", "seed": 0}}}, "failed")
    original = (camp / "experiment_observations" / (saved["observation_id"] + ".json")).read_bytes()
    result = compile_input(camp, {"current_question": "Why did temporal EEG alignment fail?", "terminal_observation_id": saved["observation_id"]})
    assert isinstance(result, Ready), result
    actual = expand(result.prepared.payload)
    assert "checkpoint_mismatch" in actual["memory_retrieval_health"]["query"]
    current = next(x for x in actual["experiment_observations"]["cards"] if x["current_feedback"])
    assert current["body"]["identity"] == saved["identity"] and current["body"]["execution_status"] == "failed"
    assert current["body"]["engineering_failure"]["error_fingerprint"] == saved["engineering_failure"]["error_fingerprint"]
    assert actual["retrieved_knowledge"]["items"] and (camp / "experiment_observations" / (saved["observation_id"] + ".json")).read_bytes() == original


def test_real_pinned_minilm_lesson_query_cosine_and_restart_cache(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    relevant = lesson(camp, "Temporal convolution filters learn local temporal patterns in EEG signals before projecting the encoder to image features.")
    lesson(camp, "UNRELATED penguins migrate through ocean ice and nest on polar islands.")
    try:
        backend = backend_for(memory_config(camp))
    except EmbeddingUnavailable as exc:
        pytest.skip("Pinned local MiniLM unavailable, no download: " + str(exc))
    assert type(backend.model).__module__.startswith("sentence_transformers")
    query = "Temporal convolution EEG encoder learns temporal patterns"
    result = sm.retrieve(camp, query, require_embedding=True)
    assert result["backend"] == "minilm" and result["revision"] == PIN and result["dimension"] == 384
    assert result["fingerprint"] == backend.fingerprint and result["query_encoding"]["truncated"] is False
    hit = next(x for x in result["items"] if x["record_id"] == relevant["lesson_id"])
    assert .3 <= hit["cosine_score"] <= 1.
    restarted = type(backend)(memory_config(camp))
    assert restarted.fingerprint == backend.fingerprint
    monkeypatch.setattr(restarted, "encode", lambda _: pytest.fail("Native record and query cache must survive backend restart"))
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: restarted)
    again = sm.retrieve(camp, query, require_embedding=True)
    assert again["query_status"] == "cache_reused" and again["indexing"]["indexed"] == 0
    assert again["items"] == result["items"]
    knowledge = sm.planner_knowledge_delivery(camp, again, context=sm.build_context(camp))
    assert relevant["statement"] in canonical(knowledge) and knowledge["items"][0]["similarity_method"] == "MiniLM_normalized_cosine"


def test_native_resume_same_round_can_cite_delivered_knowledge_handle(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    lesson(camp)
    def unavailable(*_a, **_k):
        raise EmbeddingUnavailable("synthetic_backend_temporarily_missing")
    monkeypatch.setattr(sm, "backend_for", unavailable)
    stopped = selected_plan._select(camp, loop.load_state(camp), lambda _: pytest.fail("Unavailable embeddings cannot buy Planner input"))
    rid = stopped["context_requirement"]["round_id"]
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: SyntheticVectors())
    delivered = []
    def planner(payload):
        delivered.append(payload)
        ref = payload["retrieved_knowledge"]["items"][0]["full_record"]["$context_source"]
        assert ref in {row["$context_source"] for row in payload["history_navigation"]}
        proposals = [{"proposal_id": pid, "task_kind": "diagnosis", "target_candidate_ref": "baseline",
            "question": "Inspect bounded recorded context " + pid, "task_description": "Read existing knowledge " + pid,
            "scientific_intent": "Explain prior applicability " + pid, "evaluation_scope": {"phase": "diagnostic"},
            "expected_observation": "A sourced limitation", "completion_condition": "One bounded feedback",
            "required_context_refs": [ref], "value_rationale": "No training before uncertainty is resolved"} for pid in ("A", "B")]
        return {"protocol_version": "research_plan_draft.v1", "proposals": proposals,
            "evaluations": [{"proposal_id": pid, "rank": rank, "value_rationale": "Read existing sources",
                "implementation_risk": "Read-only bounded diagnosis"} for rank, pid in enumerate(("A", "B"), 1)],
            "selected_proposal_id": "A", "selection_rationale": "Use the delivered prior to diagnose uncertainty"}
    resumed = selected_plan._select(camp, stopped, planner)
    assert len(delivered) == 1
    assert resumed["context_requirement"]["status"] == "resolved"
    plan = selected_plan.load_plan(camp, resumed)
    assert plan["originating_round"] == rid and plan["current_step"] == "prepare" and plan["repair_count"] == 0
    assert plan["proposal"]["required_context_refs"] == [delivered[0]["retrieved_knowledge"]["items"][0]["full_record"]["$context_source"]]
    assert not (camp / "llm_api_intents.jsonl").exists() and not (camp / "jobs").exists()


def test_revoked_knowledge_source_has_explicit_omission_and_no_delivered_body(tmp_path, monkeypatch):
    camp = campaign(tmp_path)
    relevant = lesson(camp)
    monkeypatch.setattr(sm, "backend_for", lambda *_a, **_k: SyntheticVectors())
    bundle = sm.retrieve(camp, "Temporal convolution EEG", require_embedding=True)
    hit = next(item for item in bundle["items"] if item["record_id"] == relevant["lesson_id"])
    ContextSourceRegistry(camp).revoke(hit["record_source_ref"])
    result = sm.planner_knowledge_delivery(camp, bundle, context=sm.build_context(camp))
    assert result["items"] == [] and result["omitted_count"] == 1
    assert result["omitted"][0]["record_id"] == relevant["lesson_id"]
    assert result["omitted"][0]["reason"] == "knowledge_source_unreadable"
    assert "revoked" in result["omitted"][0]["detail"]
