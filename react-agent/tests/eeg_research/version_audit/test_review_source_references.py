"""Reviewer citations resolve only within its actual delivered source material."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from react_agent.eeg_research.agentic.context.service import ContextService
from react_agent.eeg_research.agentic.context.types import CompileRequest, Ready, digest
from react_agent.eeg_research.agentic.llm import system_prompt
from react_agent.eeg_research.agentic.review_context import review_reference_errors, valid_review_reference


@pytest.fixture
def prepared_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    camp = tmp_path / "review_delivery"
    camp.mkdir()
    (camp / "goal.json").write_text(json.dumps({"research_execution": {"version": "single_plan_v1"},
                                               "memory": {"enabled": False}}))
    candidate = camp / "eeg_candidate.py"
    candidate.write_text("class EEGCandidate:\n    def build_encoder(self):\n        return 1\n"
                         "    def forward(self):\n        return 1\n")
    checks = camp / "checks.json"
    checks.write_text('{"ok":true}')
    parent = camp / "parent.py"
    supplied_parent = "class Parent:\n    def visible(self):\n        return 2\n"
    parent.write_text(supplied_parent + "\n    def hidden(self):\n        return 3\n"
                      "\ndef parent_omitted():\n    return 3\n")
    runtime = camp / "train_entry.py"
    first = "def build_encoder():\n    return 4"
    omitted = "def omitted_training_branch():\n    return 5"
    final = "def fit():\n    return 6"
    runtime.write_text(first + "\n\n" + omitted + "\n\n" + final + "\n")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"candidate_source": candidate.read_text(), "checks": {"ok": True},
        "requirement_evidence_identity": {"source_ref": str(candidate), "source_hash": sha(candidate),
            "check_ref": str(checks), "check_sha256": sha(checks)},
        "parent_source": {"source_ref": str(parent), "source_hash": sha(parent), "source": supplied_parent,
                          "source_truncated": True},
        "training_runtime_source": [{"source_ref": str(runtime), "source_hash": sha(runtime),
                                    "excerpt": first + "\n\n" + final}]}
    outcome = ContextService(camp).compile(CompileRequest(camp.name, "candidate_reviewer", payload,
        system=system_prompt("candidate_reviewer", camp=camp), model="synthetic-model", allow_semantic_compaction=False))
    assert isinstance(outcome, Ready)
    assert outcome.prepared.verify()
    request = json.loads(outcome.prepared.user)
    identity = payload["requirement_evidence_identity"]
    reply = {"status": "ready", "requirement_coverage": [{"requirement_id": "encoder", "status": "implemented",
        "source_hash": identity["source_hash"], "check_sha256": identity["check_sha256"],
        "check_ref": identity["check_ref"], "source_refs": [], "detail": "Cite only delivered material."}]}
    return request, reply, candidate, checks, parent, runtime


@pytest.mark.parametrize("reference", ["candidate", "candidate_symbol", "check", "check_alias", "parent",
                                    "parent_symbol", "runtime", "runtime_line", "runtime_hashline", "runtime_symbol"])
def test_actual_prepared_candidate_check_parent_and_runtime_citations(prepared_sources, reference):
    request, reply, candidate, checks, parent, runtime = prepared_sources
    refs = {"candidate": str(candidate), "candidate_symbol": str(candidate) + "::EEGCandidate::build_encoder",
        "check": str(checks), "check_alias": "checks.json", "parent": str(parent),
        "parent_symbol": str(parent) + "::Parent::visible", "runtime": str(runtime),
        "runtime_line": str(runtime) + ":7", "runtime_hashline": str(runtime) + "#L8",
        "runtime_symbol": str(runtime) + "::fit"}
    reply["requirement_coverage"][0]["source_refs"] = [refs[reference]]
    assert review_reference_errors(reply, request) == []
    assert request["parent_source"]["source_truncated"] is True


@pytest.mark.parametrize("failure", ["not_delivered", "wrong_runtime_hash", "deleted", "unread_handle",
    "fake_excerpt", "omitted_line", "omitted_symbol", "omitted_parent_symbol", "blank_line", "candidate_hash", "check_hash"])
def test_citations_cannot_expand_delivery_or_change_candidate_check_identity(prepared_sources, failure):
    request, reply, candidate, checks, parent, runtime = prepared_sources
    reference = str(runtime)
    if failure == "not_delivered":
        fake = runtime.with_name("unseen.py")
        fake.write_text(runtime.read_text())
        reference = str(fake)
    elif failure == "wrong_runtime_hash":
        request["training_runtime_source"][0]["source_hash"] = "0" * 64
    elif failure == "deleted":
        runtime.unlink()
    elif failure == "unread_handle":
        request["training_runtime_source"][0]["excerpt"] = {"$context_source": "context://revoked-source"}
    elif failure == "fake_excerpt":
        request["training_runtime_source"][0]["excerpt"] = "def invented():\n    return 999"
    elif failure == "omitted_line":
        reference += ":4"
    elif failure == "omitted_symbol":
        reference += "::omitted_training_branch"
    elif failure == "omitted_parent_symbol":
        reference = str(parent) + "::parent_omitted"
    elif failure == "blank_line":
        request["training_runtime_source"][0]["excerpt"] += "\n\n \n\n"
        reference += ":6"
    else:
        reference = str(candidate)
        reply["requirement_coverage"][0]["source_hash" if failure == "candidate_hash" else "check_sha256"] = "wrong"
    reply["requirement_coverage"][0]["source_refs"] = [reference]
    assert review_reference_errors(reply, request)


def test_actual_wire_local_pointer_preserves_source_scope(prepared_sources):
    request, _reply, _candidate, _checks, _parent, runtime = prepared_sources
    original = copy.deepcopy(request["training_runtime_source"][0]["excerpt"])
    request["delivered_runtime_text"] = original
    request["training_runtime_source"][0]["excerpt"] = {
        "$context_pointer": "/delivered_runtime_text", "target_hash": digest(original)}
    assert valid_review_reference(str(runtime) + "::fit", request)
    assert not valid_review_reference(str(runtime) + "::omitted_training_branch", request)


def test_reply_metadata_cannot_register_another_source(prepared_sources):
    request, reply, _candidate, _checks, _parent, runtime = prepared_sources
    other = runtime.with_name("not_in_request.py")
    other.write_text(runtime.read_text())
    reply["training_runtime_source"] = [{"source_ref": str(other), "source_hash": hashlib.sha256(other.read_bytes()).hexdigest(),
                                         "excerpt": other.read_text()}]
    reply["requirement_coverage"][0]["source_refs"] = [str(other)]
    assert review_reference_errors(reply, request)


@pytest.mark.parametrize("separator", [".", "::"])
def test_paid_shared_scope_methods_resolve_without_rewriting_the_review(prepared_sources, separator):
    request, reply, candidate, _checks, _parent, _runtime = prepared_sources
    reply["status"] = "needs_fix"
    reply["issues"] = [{"severity": "blocking", "detail": "The approved hook contract still conflicts."}]
    reply["requirement_coverage"][0]["status"] = "contradicted"
    reply["requirement_coverage"][0]["source_refs"] = [str(candidate) + "::EEGCandidate" + separator + "build_encoder/forward"]
    original = copy.deepcopy(reply)
    assert review_reference_errors(reply, request) == []
    assert reply == original


@pytest.mark.parametrize("symbol", ["EEGCandidate.build_encoder/missing", "EEGCandidate.build_encoder/",
    "EEGCandidate.build_encoder//forward", "EEGCandidate.build_encoder/Other.forward",
    "EEGCandidate.build_encoder/../forward", "build_encoder/forward"])
def test_shared_scope_citation_requires_every_member_in_the_same_scope(prepared_sources, symbol):
    request, reply, candidate, _checks, _parent, _runtime = prepared_sources
    reply["requirement_coverage"][0]["source_refs"] = [str(candidate) + "::" + symbol]
    assert review_reference_errors(reply, request)


def test_shared_scope_citation_cannot_include_an_undelivered_method(prepared_sources):
    request, reply, _candidate, _checks, parent, _runtime = prepared_sources
    assert valid_review_reference(str(parent) + "::Parent.visible", request)
    reply["requirement_coverage"][0]["source_refs"] = [str(parent) + "::Parent.visible/hidden"]
    assert review_reference_errors(reply, request)


def test_shared_scope_citation_accepts_all_members_after_full_parent_delivery(prepared_sources):
    request, reply, _candidate, _checks, parent, _runtime = prepared_sources
    request["parent_source"]["source"] = parent.read_text()
    request["parent_source"]["source_truncated"] = False
    reply["requirement_coverage"][0]["source_refs"] = [str(parent) + "::Parent.visible/hidden"]
    assert review_reference_errors(reply, request) == []


def test_shared_scope_citation_accepts_disjoint_delivered_runtime_members(prepared_sources):
    request, reply, _candidate, _checks, _parent, runtime = prepared_sources
    first = "class Runtime:\n    def first(self):\n        return 1"
    omitted = "    def omitted(self):\n        return 2"
    last = "    def last(self):\n        return 3"
    runtime.write_text(first + "\n\n" + omitted + "\n\n" + last + "\n")
    request["training_runtime_source"][0] = {"source_ref": str(runtime),
        "source_hash": hashlib.sha256(runtime.read_bytes()).hexdigest(), "excerpt": first + "\n\n" + last}
    reply["requirement_coverage"][0]["source_refs"] = [str(runtime) + "::Runtime.first/last"]
    assert review_reference_errors(reply, request) == []
    reply["requirement_coverage"][0]["source_refs"] = [str(runtime) + "::Runtime.first/omitted"]
    assert review_reference_errors(reply, request)


def test_shared_scope_citation_cannot_register_another_candidate_file(prepared_sources):
    request, reply, candidate, _checks, _parent, _runtime = prepared_sources
    other = candidate.with_name("other_candidate.py")
    other.write_text(candidate.read_text())
    reply["requirement_coverage"][0]["source_refs"] = [str(other) + "::EEGCandidate.build_encoder/forward"]
    assert review_reference_errors(reply, request)
