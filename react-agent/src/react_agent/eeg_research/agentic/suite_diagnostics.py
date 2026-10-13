"""Read existing development outputs into immutable, suite-scoped diagnostics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .artifacts import file_digest, register, resolve_verified_artifact, lookup
from .diagnostics import summarize_bundle
from .jsonl_read import read_jsonl

VERSION = "eeg_research.suite_diagnostics.v1"

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()

def sources(camp: Path, state: dict) -> list[dict]:
    """Use native verified scores/freezes, retaining phase rather than inventing a job."""
    camp = Path(camp).resolve()
    rows = []
    if state.get("evaluation_mode") == "loso_method_search":
        from .method_suite import verified_suite_records
        from .screening import verified_screening_records
        records = [*verified_screening_records(camp, state), *verified_suite_records(camp, state)]
        for record in records:
            scientific = record.get("screening") or record.get("aggregate") or {}
            for score in scientific.get("scores") or []:
                fold = score["fold_id"]
                path = camp / "suites" / record["suite_id"] / fold / "benchmark_freeze.json"
                freeze = _read(path)
                job = Path(freeze["job_dir"]).resolve()
                if not job.is_relative_to(camp):
                    raise ValueError("diagnostic_job_outside_campaign")
                j = _read(job / "job.json") if (job / "job.json").is_file() else {}
                if j and (j.get("status") != "finished" or j.get("candidate_id") != record["candidate_id"]
                        or j.get("seed") != score["seed"] or freeze["checkpoint_hash"] != score["checkpoint_hash"]):
                    raise ValueError("diagnostic_fold_job_identity_mismatch")
                rows.append({"candidate_id": record["candidate_id"], "suite_id": record["suite_id"],
                    "fold_id": fold, "held_out_subject": score["held_out_subject"], "seed": score["seed"],
                    "phase": scientific.get("phase", "confirmation_10fold"),
                    "scope": scientific.get("scope", "method_development_benchmark"),
                    "coverage": scientific.get("coverage", len(scientific.get("scores") or [])),
                    "source_hash": (j.get("manifest") or {}).get("entry_sha256") or record.get("source_hash"),
                    "spec_hash": record.get("spec_hash"),
                    "method_revision": record.get("method_revision"), "job_id": j.get("job_id", job.name),
                    "resolver_gap": None if j else "job_record_missing",
                    "job_dir": str(job), "freeze_ref": str(path), "freeze_sha256": file_digest(path),
                    "scientific_result_ref": record["suite_ref"], "scientific_result_hash": record["result_hash"]})
    else:
        for record in state.get("evidence") or []:
            if not record.get("job_dir"):
                continue
            job = Path(record["job_dir"]).resolve()
            if not job.is_relative_to(camp):
                raise ValueError("diagnostic_job_outside_campaign")
            rows.append({key: record[key] for key in ("candidate_id", "job_id", "seed", "fidelity", "source_hash", "spec_hash") if key in record} | {"job_dir": str(job), "scope": "source_development"})
    # A ten-fold record supersedes its two-fold navigation, not its historical scores.
    unique = {}
    for row in rows:
        unique[(row.get("suite_id"), row.get("job_id"), row.get("fold_id"))] = row
    return list(unique.values())

def _dependencies(rows):
    result = {}
    for row in rows:
        job = Path(row["job_dir"])
        paths = [job / name for name in ("history.jsonl", "diagnostic_bundle.json", "diagnostic_summary.json", "metrics.json", "source_binding.json", "selected_checkpoint.json")]
        paths.extend(Path(row[key]) for key in ("freeze_ref", "scientific_result_ref") if row.get(key))
        for path in paths:
            if path.is_file():
                result[str(path)] = {"path": str(path), "sha256": file_digest(path)}
    return list(result.values())

def input_epochs(camp: Path, state: dict) -> dict:
    rows = sources(camp, state)
    dependencies = _dependencies(rows)
    return {action: _digest({"version": VERSION, "action": action, "sources": rows,
                             "dependencies": dependencies})
            for action in ("collect_diagnostics", "diagnose_results")}

def curve(job: Path) -> dict:
    read = read_jsonl(job / "history.jsonl")
    points = read["rows"]
    measured = [row for row in points if type(row.get("fixed_bank_top1")) in (int, float)]
    best = max(measured, key=lambda row: row["fixed_bank_top1"]) if measured else {}
    return {"status": "partial" if read["diagnostics"] else "ready" if points else "unavailable",
        "reason": "history_missing_or_empty" if not points else None,
        "epochs": len(points), "epoch_ids": [row.get("epoch") for row in points],
        "best_epoch": best.get("epoch"), "best_fixed_bank_top1": best.get("fixed_bank_top1"),
        "first_train_loss": points[0].get("train_loss") if points else None,
        "last_train_loss": points[-1].get("train_loss") if points else None,
        "fixed_bank_trend": [row.get("fixed_bank_top1") for row in points],
        "read_coverage": {"valid_rows": len(points), "truncated": read["truncated"], "diagnostics": read["diagnostics"]},
        "metric_scope": "source_subject_development_curve"}

def collect(camp: Path, state: dict, *, kind="diagnostic_bundle") -> dict:
    """Campaign-history annotation; targets remain annotations as in the cost contract."""
    rows = sources(camp, state)
    dependencies = _dependencies(rows)
    source_epoch = _digest({"version": VERSION, "kind": kind, "sources": rows, "dependencies": dependencies})
    path = Path(camp) / "diagnostics" / (kind + "_" + source_epoch + ".json")
    reused = path.is_file()
    if reused:
        report = _read(path)
        if report.get("source_epoch") != source_epoch or report.get("dependencies") != dependencies:
            raise ValueError("diagnostic_cached_source_changed")
    else:
        runs = []
        for row in rows:
            job = Path(row["job_dir"])
            item = {**row, "learning_profile": curve(job)}
            if row.get("resolver_gap"):
                item["learning_profile"] = {"status": "unavailable", "reason": row["resolver_gap"], "epochs": 0}
                item["diagnostics"] = {"status": "unavailable", "reason": row["resolver_gap"]}
                runs.append(item)
                continue
            bundle = job / "diagnostic_bundle.json"
            if bundle.is_file():
                try:
                    item["diagnostics"] = summarize_bundle(_read(bundle))
                except json.JSONDecodeError:
                    item["diagnostics"] = {"status": "partial", "reason": "malformed_existing_diagnostic_bundle"}
            else:
                item["diagnostics"] = {"status": "unavailable", "reason": "diagnostic_bundle_not_exported"}
            runs.append(item)
        report = {"schema_version": VERSION, "kind": kind, "source_epoch": source_epoch,
            "diagnostic_scope": "campaign_history_development_annotation", "runs": runs,
            "dependencies": dependencies, "final_test_accessed": False,
            "status": ("partial" if any(r.get("resolver_gap") for r in runs) else "ready") if runs else "unavailable", "reason": None if runs else "no_verified_development_jobs",
            "read_scope": "existing source-subject development exports and training curves; no new inference/training",
            "summary": {"folds_read": len(runs), "missing_or_partial_history": sum(r["learning_profile"]["status"] != "ready" for r in runs)}}
        from .loso_study import write_once
        write_once(path, report)
    artifact_id = "art_diag_" + source_epoch[:24]
    artifact = lookup(camp, artifact_id)
    if artifact is None:
        artifact = register(camp, path, kind=kind, artifact_id=artifact_id)
    else:
        artifact = resolve_verified_artifact(camp, artifact_id)
    new_information = not any(row.get("evidence_id") == "ev_" + kind + "_" + source_epoch[:16]
                              for row in state.get("evidence") or [])
    return {"evidence_id": "ev_" + kind + "_" + source_epoch[:16], "kind": kind,
        "status": report["status"], "source_epoch": source_epoch, "diagnostic_ref": str(path),
        "artifact_refs": [artifact["artifact_id"]], "summary": report["summary"],
        "curve": [{**{key: r.get(key) for key in ("candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "phase", "scope")}, **r["learning_profile"]} for r in report["runs"]] if kind == "learning_profile" else [],
        "new_information": new_information, "diagnostic_scope": report["diagnostic_scope"]}

def verified_facts(camp: Path, state: dict, *, job_ids=None, target_ids=None, suite_ids=None, max_facts=12) -> list[dict]:
    facts = []
    current = None
    for row in reversed(state.get("evidence") or []):
        if row.get("kind") not in {"diagnostic_bundle", "learning_profile"}:
            continue
        for ref in row.get("artifact_refs") or []:
            try:
                artifact = resolve_verified_artifact(camp, ref)
                report = _read(artifact["path"])
                if report.get("schema_version") != VERSION or report.get("final_test_accessed") is not False:
                    continue
                if current is None:
                    current = sources(camp, state)
                if any(not Path(d["path"]).is_file() or file_digest(Path(d["path"])) != d["sha256"] for d in report["dependencies"]):
                    continue
                allowed = {(r.get("job_id"), r.get("suite_id"), r.get("source_hash")) for r in current}
                for run in report["runs"]:
                    if (run.get("job_id"), run.get("suite_id"), run.get("source_hash")) not in allowed:
                        continue
                    if target_ids is not None and run.get("candidate_id") not in target_ids:
                        continue
                    if suite_ids is not None and run.get("suite_id") not in suite_ids:
                        continue
                    if job_ids is not None and run.get("job_id") not in job_ids and suite_ids is None:
                        continue
                    facts.append({key: run.get(key) for key in ("candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "phase", "scope", "coverage", "source_hash", "spec_hash", "method_revision", "job_id", "diagnostics", "learning_profile")} | {"evidence_id": row["evidence_id"], "artifact_id": ref, "artifact_sha256": artifact["sha256"], "final_test_accessed": False})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if facts:
            break
    return facts[:max_facts]


def select_verified_facts(camp: Path, state: dict, plan: dict, artifact_ref: str, *, max_facts=12) -> dict:
    """Select current task sources before the card cap, without rewriting facts.

    Explicit native refs and scope filters precede recency. The bounded card is
    an annotation window into the complete immutable diagnostic artifact.
    """
    max_facts = max(0, min(int(max_facts), 12))
    proposal = plan.get("proposal") or {}
    scope = proposal.get("evaluation_scope") or {}
    target = plan.get("candidate_id") or proposal.get("target_candidate_ref")
    report_row = resolve_verified_artifact(camp, artifact_ref)
    report = _read(report_row["path"])
    # Existing validation resolves active source identities and dependencies;
    # defer the delivery cap until after selecting the exact requested report.
    candidates = [row for row in verified_facts(camp, state, target_ids={target} if target else None,
        max_facts=max(len(report.get("runs") or []), 12)) if row.get("artifact_id") == artifact_ref]
    preferred_jobs = list(dict.fromkeys(scope.get("job_ids") or []))
    preferred_suites = list(dict.fromkeys(scope.get("suite_ids") or []))
    for ref in proposal.get("required_context_refs") or []:
        if str(ref).startswith("art_"):
            matching = [row for row in state.get("evidence") or [] if ref == row.get("artifact_id")
                or ref == row.get("analysis_artifact_id") or ref in (row.get("artifact_refs") or [])]
            for row in matching:
                if row.get("job_id") and row["job_id"] not in preferred_jobs:
                    preferred_jobs.append(row["job_id"])
                if row.get("suite_id") and row["suite_id"] not in preferred_suites:
                    preferred_suites.append(row["suite_id"])
        elif str(ref).startswith("context://"):
            from .context.sources import ContextSourceRegistry
            registry = ContextSourceRegistry(camp)
            identity = registry.get(ref, role="research_planner").identity
            if identity.get("job_id") and identity["job_id"] not in preferred_jobs:
                preferred_jobs.append(identity["job_id"])
            if identity.get("suite_id") and identity["suite_id"] not in preferred_suites:
                preferred_suites.append(identity["suite_id"])
    # A declared job/suite scope restricts the task; refs provide precedence
    # among its matching sources. Unknown requested identities stay missing.
    scoped = [row for row in candidates if
        (not scope.get("job_ids") or row.get("job_id") in scope["job_ids"])
        and (not scope.get("suite_ids") or row.get("suite_id") in scope["suite_ids"])
        and (scope.get("seed") is None or row.get("seed") == scope["seed"])
        and (not scope.get("folds") or row.get("fold_id") in scope["folds"])
        and (not scope.get("subjects") or row.get("held_out_subject") in scope["subjects"])
        and (not scope.get("source_hash") or row.get("source_hash") == scope["source_hash"])
        and (not scope.get("spec_hash") or row.get("spec_hash") == scope["spec_hash"])]
    current_sources = sources(camp, state)
    recency = {(row.get("job_id"), row.get("suite_id"), row.get("fold_id")): index
               for index, row in enumerate(current_sources)}
    scoped.sort(key=lambda row: (
        0 if row.get("job_id") in preferred_jobs else 1 if row.get("suite_id") in preferred_suites else 2,
        -recency.get((row.get("job_id"), row.get("suite_id"), row.get("fold_id")), -1)))
    selected = scoped[:max_facts]
    for row in selected:
        row["source_ref"] = artifact_ref
        row["source_identity_missing"] = [key for key in ("source_hash", "spec_hash") if row.get(key) is None]
    delivered_jobs = {row.get("job_id") for row in selected}
    delivered_suites = {row.get("suite_id") for row in selected}
    missing = ["requested_job:" + str(job) for job in preferred_jobs if job not in delivered_jobs]
    missing += ["requested_suite:" + str(suite) for suite in preferred_suites if suite not in delivered_suites]
    missing += ["requested_subject:" + str(subject) for subject in scope.get("subjects") or []
                if subject not in {row.get("held_out_subject") for row in selected}]
    if len(scoped) > len(selected):
        missing.append("historical_sources_outside_card_window")
    if not selected:
        missing.append("verified_scoped_diagnostics")
    return {"facts": selected, "selection": {"selection_basis": "explicit_refs_and_task_scope_then_latest_related",
        "native_report_source_count": len(report.get("runs") or []), "target_source_count": len(candidates),
        "scoped_source_count": len(scoped), "delivered_source_count": len(selected),
        "omitted_scoped_source_count": max(0, len(scoped) - len(selected)),
        "outside_scope_source_count": max(0, len(candidates) - len(scoped)),
        "selected_job_ids": [row.get("job_id") for row in selected], "missing": missing,
        "complete_source_read_ref": artifact_ref, "complete_source_sha256": report_row["sha256"],
        "scope_authority": "campaign_history_development_annotation_not_scientific_measurement"}}
