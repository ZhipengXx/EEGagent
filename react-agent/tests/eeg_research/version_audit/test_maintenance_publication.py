"""Native paid-publication/cache recovery with synthetic settled receipts only.

No model/provider/GPU is called. An already paid, confirmed receipt is a local
fixture; production task/artifact/dependency/CAS validators own adoption.
"""
from __future__ import annotations

import json

import pytest

from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.snapshot import DELTA_VERSION
from react_agent.eeg_research.agentic.context.types import canonical
from react_agent.eeg_research.agentic.roles import begin_role_task, finish_role_task
from react_agent.eeg_research.agentic.task_ledger import find_completed_task


def campaign(tmp_path):
    camp = tmp_path / "synthetic-paid-publication"
    camp.mkdir()
    (camp / "goal.json").write_text(json.dumps({"goal_id": camp.name, "objective": "Synthetic development feedback",
        "max_llm_calls": 20, "research_execution": {"version": "single_plan_v1", "paid_maintenance": "off"},
        "memory": {"enabled": False}}))
    (camp / "evaluation_contract.json").write_text('{"task_type":"eeg_image_retrieval"}')
    (camp / "cost.json").write_text('{"llm_calls":0,"training_jobs":0,"gpu_seconds_used":0}')
    return camp


def service(camp):
    def forbidden(**kwargs):
        pytest.fail("paid off recovery must never call a model or a provider")
    return ContextService(camp, maintenance_callback=forbidden)


def completed_paid_proposal(tmp_path):
    camp = campaign(tmp_path)
    current = service(camp)
    raw = {"note": "Original synthetic limitation.", "metric": 0.0125}
    identity = {"candidate_id": "c1", "source_hash": "a" * 64, "method_revision": "revision_a", "seed": 0,
                "science_scope": "two_fold_development", "coverage": 2, "full_required": 10}
    source = current.registry.capture(raw, block_id="synthetic-paid-source", kind="analysis", identity=identity,
        status="complete", event_ids=("synthetic-result",))
    current.snapshot.capture_closed_group("synthetic-closed", [source], status="complete")
    batch = current.snapshot.pending_groups(role="memory_curator")
    metric = next(item for item in batch.sections if item["section_id"] == "/metric")
    delta = {"schema_version": DELTA_VERSION, "base_revision": batch.base_revision,
        "input_dependency_digest": batch.input_dependency_digest,
        "findings": [{"item_id": "synthetic_finding", "action": "add", "text": "Observed synthetic value 0.0125.",
            "conditions": "Limited development scope", "source_refs": [metric["section_ref"]],
            "facts": [{"section_ref": metric["section_ref"], "path": "/metric", "value": 0.0125}]}]}
    operation = "synthetic-already-paid-operation"
    original = {"closed_sections": list(batch.sections), "base_revision": batch.base_revision,
                "maintenance_operation_id": operation}
    input_path = camp / "synthetic_frozen_maintenance_input.json"
    input_path.write_text(canonical(original))
    task = begin_role_task(camp, role="memory_curator", inputs=[input_path], request=original)
    # This is an already settled historical fixture, not an actual HTTP receipt.
    paid = {"call_id": "synthetic_confirmed_paid", "task_id": task["task_id"], "role": "memory_curator",
        "success": True, "transport_outcome": "confirmed_response", "prompt_hash": task["recovery_identity"]["prompt_hash"]}
    (camp / "llm_calls.jsonl").write_text(canonical(paid) + "\n")
    (camp / "cost.json").write_text('{"llm_calls":1,"training_jobs":0,"gpu_seconds_used":0}')
    current.owned_calls.add(paid["call_id"])
    output_path = camp / "synthetic_original_paid_reply.json"
    envelope = finish_role_task(camp, task, {"publication_status": "proposal_only",
        "scientific_evidence": False, "execution_authority": False, "proposal": delta},
        kind="handoff_compaction", path=output_path)
    proposal = current.snapshot.propose_delta(delta, batch, generation={"operation_id": operation})
    return camp, current, source, raw, identity, original, input_path, output_path, task, envelope, proposal


def test_paid_off_native_capture_and_retrieval_do_not_schedule_paid_work(tmp_path):
    camp = campaign(tmp_path)
    current = service(camp)
    payload = {"evidence": [{"evidence_id": "synthetic_closed", "candidate_id": "c1", "status": "complete",
        "scope": "two_fold_development", "seed": 0, "note": "Closed original development limitation."}]}
    first = current.after_task_background(payload, turn_id="off-capture")
    second = current.run_long_memory(payload, turn_id="off-capture")
    assert first["reason"] == second["reason"] == "paid_maintenance_off"
    with current.snapshot.persistence.db() as db:
        assert db.execute("SELECT COUNT(*) FROM context_closed_sections").fetchone()[0] > 0
        assert db.execute("SELECT calls FROM context_maintenance_usage").fetchone()[0] == 0
    assert not (camp / "llm_calls.jsonl").exists()
    assert json.loads((camp / "cost.json").read_text())["llm_calls"] == 0


def test_paid_off_adopts_completed_publication_and_cache_idempotently(tmp_path):
    camp, original_service, source, raw, identity, original, input_path, output_path, task, envelope, proposal = completed_paid_proposal(tmp_path)
    original_files = {path: path.read_bytes() for path in (input_path, output_path, camp / "cost.json", camp / "llm_calls.jsonl",
        camp / proposal.artifact_ref, camp / "artifact_registry.jsonl", camp / "task_ledger.jsonl") if path.exists()}
    fresh = service(camp)
    recovered = fresh.recover_paid_publications()
    assert recovered["api_calls"] == 0 and not recovered["deferred"]
    assert recovered["recovered"] == [{"proposal_id": proposal.proposal_id, "status": "published", "reason": ""}]
    item = fresh.snapshot.snapshot_delivery()["findings"][0]
    assert item["sources"][0]["identity"] == identity
    assert item["facts"][0]["value"] == 0.0125
    assert item["scientific_evidence"] is False and item["execution_approval"] is False
    assert fresh.registry.load(source) == raw
    revision = fresh.snapshot.revision
    second = service(camp).recover_paid_publications()
    assert second["api_calls"] == 0 and second["recovered"] == second["deferred"] == []
    assert fresh.snapshot.revision == revision == 1
    cached = begin_role_task(camp, role="memory_curator", inputs=[input_path], request=original,
        recovery_identity=task["recovery_identity"])
    assert cached["task_id"] == task["task_id"] and cached.get("_cached_role_result")
    verified = find_completed_task(camp, role="memory_curator", recovery_identity=task["recovery_identity"],
                                   input_digest=task["input_digest"])
    assert verified and verified["envelope"] == envelope
    adopted = finish_role_task(camp, cached, {"ignored_new_reply": True}, kind="handoff_compaction", path=output_path)
    assert adopted == envelope
    assert all(path.read_bytes() == contents for path, contents in original_files.items())
    with fresh.snapshot.persistence.db() as db:
        assert db.execute("SELECT calls FROM context_maintenance_usage").fetchone()[0] == 0


@pytest.mark.parametrize("invalid", ["source", "input", "reply", "authority"])
def test_paid_off_rejects_stale_or_invalid_publication_and_preserves_original(tmp_path, invalid):
    camp, original_service, source, raw, identity, original, input_path, output_path, task, envelope, proposal = completed_paid_proposal(tmp_path)
    original_cost = (camp / "cost.json").read_bytes()
    original_ledger = (camp / "llm_calls.jsonl").read_bytes()
    original_draft = (camp / proposal.artifact_ref).read_bytes()
    if invalid == "source":
        original_service.registry.capture({"metric": 0.99}, block_id=source.block_id, kind="analysis", identity=identity, status="complete")
    elif invalid == "input":
        input_path.write_text(canonical({**original, "maintenance_operation_id": "changed-operation"}))
    elif invalid == "reply":
        output_path.write_text(canonical({**envelope, "prompt_hash": "changed-original-prompt"}))
    else:
        (camp / "control.json").write_text('{"action":"pause","at":1}')
    fresh = service(camp)
    result = fresh.recover_paid_publications()
    assert result["api_calls"] == 0 and not result["recovered"] and result["deferred"]
    assert fresh.snapshot.revision == 0 and fresh.snapshot.items_page() == []
    assert (camp / "cost.json").read_bytes() == original_cost and (camp / "llm_calls.jsonl").read_bytes() == original_ledger
    assert (camp / proposal.artifact_ref).read_bytes() == original_draft
    assert fresh.registry.load(source, require_current=False) == raw
    with fresh.snapshot.persistence.db() as db:
        assert db.execute("SELECT calls FROM context_maintenance_usage").fetchone()[0] == 0
