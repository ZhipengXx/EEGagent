"""Read-only reconstruction of native curator input, before tasks or transport.

Memory retrieval is deliberately disabled, labelled in the receipt, and never
claimed to reproduce the transient retrieval metadata of an old failed request.
All current scientific input is assembled by the actual worker source prefix.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from react_agent.eeg_research.agentic import worker, method_suite, semantic_memory
from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
from react_agent.eeg_research.agentic.compaction import canonical, split_input, protected_context
from react_agent.eeg_research.agentic.embedding import memory_config, EmbeddingUnavailable
from react_agent.eeg_research.agentic.handoffs import (
    development_view, load_analysis_views, compact_role_context,
)
from react_agent.eeg_research.agentic.llm import system_prompt
from react_agent.eeg_research.agentic.role_examples import add_role_examples


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(camp: Path, output: Path) -> dict:
    camp, output = camp.resolve(), output.resolve()
    if output.is_relative_to(camp):
        raise ValueError("observation_output_must_be_outside_campaign")
    state = json.loads((camp / "campaign_state.json").read_text())
    if state.get("evaluation_mode") != "loso_method_search" or state.get("live_job"):
        raise ValueError("requires_quiescent_method_campaign")
    protected = [camp / name for name in ("campaign_state.json", "cost.json", "goal.json",
                  "task_ledger.jsonl", "artifact_registry.jsonl", "method_search_manifest.json",
                  "memory.sqlite")]
    before = {str(p): sha(p) for p in protected if p.is_file()}
    records = method_suite.verified_suite_records(camp, state)
    latest = records[-1]
    row = next(row for row in reversed(state["evidence"]) if row.get("kind") == "analysis"
               and row.get("status") == "completed" and row.get("run_evidence_id") == latest["evidence_id"])
    artifact = resolve_verified_artifact(camp, row["analysis_artifact_id"])
    envelope = json.loads(Path(artifact["path"]).read_text())

    tree = ast.parse(Path(worker.__file__).read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_analyze_run")
    block = next(node for node in function.body if isinstance(node, ast.Try)
                 and any(isinstance(item, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "curation_context"
                     for target in item.targets) for item in node.body))
    prefix = []
    for statement in block.body:
        targets = [target.id for target in statement.targets if isinstance(target, ast.Name)] \
            if isinstance(statement, ast.Assign) else []
        if "compaction_artifacts" in targets or "task" in targets:
            break
        if "curator" not in targets:
            prefix.append(copy.deepcopy(statement))
    captured = ast.parse("def read_capture(camp_dir, state, latest, envelope, store):\n    pass\n").body[0]
    captured.body = prefix + [ast.Return(value=ast.Name(id="curation_context", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[captured], type_ignores=[]))
    bound_experiment = latest.get("experiment")
    if bound_experiment is None:
        bound_experiment = next((item.get("experiment") for item in state.get("candidates", [])
                                if item.get("candidate_id") == latest.get("candidate_id")), None)
    namespace = {**worker.__dict__, "development_view": development_view,
                 "load_analysis_views": load_analysis_views, "method_mode": True,
                 "bound_experiment": bound_experiment, "role_status": "completed",
                 "comparison": latest.get("comparison"), "diagnostics": latest.get("diagnostics"),
                 "target": Path(artifact["path"])}
    exec(compile(module, "readonly_native_curator_prefix", "exec"), namespace)

    def read_store(store_camp):
        return sqlite3.connect((Path(store_camp) / "memory.sqlite").resolve().as_uri() + "?mode=ro",
                               uri=True)

    def no_embedding(_config):
        raise EmbeddingUnavailable("readonly_snapshot_embedding_not_executed")

    from react_agent.eeg_research.agentic.memory import EpisodeStore
    with patch("react_agent.eeg_research.agentic.memory.open_store", read_store), \
         patch.object(semantic_memory, "backend_for", no_embedding):
        payload = namespace["read_capture"](camp, state, latest, envelope, EpisodeStore(camp))
    memory = memory_config(camp)
    system = system_prompt("memory_curator", memory=memory)
    probe = {**payload, "task_id": "task_" + "0" * 12,
             "attempt_id": "0" * 12, "input_digest": "0" * 64}
    full = add_role_examples(probe, "memory_curator", memory)
    view = compact_role_context(full, "memory_curator", system_chars=len(system))
    text, parts = split_input(payload)
    receipt = {
        "scope": "Readonly reconstruction from native source; no task, transport, training, or memory write.",
        "historical_request_exact_match": False,
        "retrieval_boundary": "Embedding/retrieval not executed; labelled legacy fallback metadata.",
        "campaign": str(camp), "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "original_business_chars": len(text),
        "system_chars": len(system), "provider_chars": len(system) + len(canonical(view)),
        "budget_chars": view["role_context_projection"]["budget_chars"],
        "fits_budget": view["role_context_projection"]["fits_budget"],
        "field_chars": {key: len(canonical(value)) for key, value in payload.items()},
        "protected_business_chars": len(canonical(protected_context(payload))),
        "source_ranges": [{key: part[key] for key in ("ref", "start", "end", "sha256")} for part in parts],
        "native_files_sha256": before,
    }
    if any(sha(Path(p)) != expected for p, expected in before.items()):
        raise RuntimeError("native_files_changed_during_readonly_capture")
    output.mkdir(parents=True, exist_ok=True)
    (output / "curator_input_snapshot.json").write_text(canonical(payload), encoding="utf-8")
    (output / "curator_system_prompt.txt").write_text(system, encoding="utf-8")
    (output / "curator_capture_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def capture(camp: Path, output: Path) -> dict:
    """Reuse exact hashes of unchanged file versions only within this read."""
    from react_agent.eeg_research.agentic import artifacts, loso_study
    native_digest = loso_study.digest
    native_manifest = method_suite.manifest
    hashes = {}
    manifests = {}

    def signature(path):
        stat = Path(path).stat()
        return (str(Path(path).resolve()), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)

    def cached_digest(path):
        key = signature(path)
        if key not in hashes:
            value = native_digest(Path(path))
            if signature(path) != key:
                raise RuntimeError("source_changed_during_hash")
            hashes[key] = value
        return hashes[key]

    def cached_manifest(directory):
        key = signature(Path(directory) / method_suite.MANIFEST)
        if key not in manifests:
            manifests[key] = native_manifest(directory)
        return manifests[key]

    # The worker prefix and native verification used here are read-only.
    # No production cache, source, or authorization is changed by these patches.
    with ExitStack() as stack:
        stack.enter_context(patch.object(method_suite, "digest", cached_digest))
        stack.enter_context(patch.object(loso_study, "digest", cached_digest))
        stack.enter_context(patch.object(artifacts, "file_digest", cached_digest))
        stack.enter_context(patch.object(method_suite, "manifest", cached_manifest))
        receipt = _capture(camp, output)
        receipt["readonly_hash_cache"] = {
            "scope": "Exact file-version hashes reused inside this observation process only.",
            "distinct_versions_hashed": len(hashes), "distinct_manifests": len(manifests)}
        (output / "curator_capture_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.camp, args.output), ensure_ascii=False, indent=2))
