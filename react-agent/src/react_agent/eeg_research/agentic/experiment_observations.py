"""Native experiment cards, immutable provenance and bounded local retrieval.

Cards are a delivery view. Native evaluators still own metric/identity gates;
interpretation is bound separately to the already paid Analyst artifact.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import time
import re
from pathlib import Path
from typing import Any

from .embedding import canonical, content_hash, memory_config
from .research_execution import single_plan_enabled
from .context.sources import ContextSourceRegistry, atomic_immutable
from .context.types import NeedsContext, ReadRequest

VERSION = "eeg_research.experiment_observation.v1"
TERMINAL = frozenset({"observation_ready", "failed", "needs_replan", "cancelled"})


class ObservationDeliveryRequired(RuntimeError):
    """A saved current feedback cannot fit the authorized delivery budget."""
    def __init__(self, result: NeedsContext):
        super().__init__(result.reason)
        self.result = result


def observation_conditions(value: dict) -> dict:
    """Normalize the same exact comparison conditions for recent and semantic hits."""
    owners = [value[key] for key in ("native_facts", "identity", "evaluation_scope", "applicability", "science_scope")
              if isinstance(value.get(key), dict)] + [value]
    aliases = {
        "scope": ("scope", "science_scope"), "dataset": ("dataset",), "modality": ("modality",),
        "seed": ("seed", "training_seed"),
        "fold_count": ("required_fold_count", "fold_count", "folds", "coverage"),
        "fold_ids": ("screening_fold_ids", "fold_ids"),
        "subjects": ("subjects", "screening_subjects", "held_out_subjects", "held_out_subject"),
        "source_hash": ("source_hash",), "spec_hash": ("spec_hash",),
        "method_revision": ("method_revision",), "selection_manifest_hash": ("selection_manifest_hash",),
        "evaluation_identity": ("evaluation_identity", "evaluation_key", "contract_fingerprint"),
    }
    result = {}
    for key, names in aliases.items():
        found = next((owner[name] for owner in owners for name in names
                      if owner.get(name) is not None and not isinstance(owner[name], dict)), None)
        if found is None:
            continue
        if key == "fold_count" and isinstance(found, list):
            found = len(found)
        if key in {"subjects", "fold_ids"}:
            found = sorted(found if isinstance(found, (list, tuple)) else [found], key=canonical)
        if key == "modality" and isinstance(found, str):
            found = found.upper()
        result[key] = copy.deepcopy(found)
    folds = (value.get("native_facts") or {}).get("folds")
    if isinstance(folds, list) and folds and all(isinstance(row, dict) for row in folds):
        result.setdefault("fold_ids", sorted([row["fold_id"] for row in folds if "fold_id" in row], key=canonical))
        result.setdefault("subjects", sorted([row["held_out_subject"] for row in folds if "held_out_subject" in row], key=canonical))
    return result


def observation_applicability(body: dict, context: dict | None) -> tuple[str, dict]:
    recorded, current = observation_conditions(body), observation_conditions(context or {})
    keys = set(recorded) | set(current) | {"scope", "dataset", "seed", "fold_count", "subjects", "source_hash"}
    mismatch = {key: {"record": recorded[key], "current": current[key]} for key in sorted(keys)
                if key in recorded and key in current and canonical(recorded[key]) != canonical(current[key])}
    unknown = {key: {"record": recorded.get(key), "current": current.get(key)} for key in sorted(keys)
               if key not in recorded or key not in current}
    label = "analogy_only" if mismatch else "applicability_unknown" if unknown else "compatible_evidence"
    return label, {"mismatch": [{"field": key, **detail} for key, detail in mismatch.items()],
                   "unknown": [{"field": key, **detail} for key, detail in unknown.items()],
                   "matched_comparison": label == "compatible_evidence"}


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class ExperimentObservationStore:
    def __init__(self, camp: Path):
        from .semantic_memory import SidecarStore
        self.camp = Path(camp).resolve()
        self.sidecar = SidecarStore(self.camp)
        self.registry = ContextSourceRegistry(self.camp)
        with self.sidecar.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiment_observations (
                namespace TEXT NOT NULL, observation_id TEXT NOT NULL,
                plan_id TEXT NOT NULL, content_hash TEXT NOT NULL, payload TEXT NOT NULL,
                source_ref TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY(namespace,observation_id))""")
            db.execute("CREATE INDEX IF NOT EXISTS observation_recent ON experiment_observations(namespace,created_at DESC)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiment_observations)")}
            if "search_text" not in columns:
                db.execute("ALTER TABLE experiment_observations ADD COLUMN search_text TEXT NOT NULL DEFAULT ''")

    def record(self, body: dict) -> dict:
        from .semantic_memory import unsafe_source
        if body.get("origin_campaign_id") != self.camp.name or unsafe_source(body):
            raise ValueError("observation_campaign_or_scope_invalid")
        if not body.get("plan_id") or body.get("execution_status") not in TERMINAL:
            raise ValueError("observation_requires_terminal_plan_feedback")
        if body.get("observation_kind") == "engineering_failure" and body.get("native_facts", {}).get("metrics"):
            raise ValueError("engineering_failure_cannot_claim_measured_outcome")
        if body.get("native_facts", {}).get("metrics") and not body.get("source_refs"):
            raise ValueError("observation_measurement_requires_verified_source")
        body = copy.deepcopy(body)
        body["schema_version"] = VERSION
        signature = content_hash(body)
        body["observation_id"] = "obs_" + signature[:24]
        body["content_digest"] = content_hash(body)
        path = self.camp / "experiment_observations" / (body["observation_id"] + ".json")
        atomic_immutable(path, canonical(body))
        block = self.registry.capture(body, block_id="experiment_observation:" + body["observation_id"],
            kind="scientific" if body["observation_kind"] == "experiment" else "checks",
            identity=body.get("identity") or {}, dependency_refs=tuple(ref.get("source_ref", "") for ref in body.get("source_refs", [])))
        with self.sidecar.connect() as db:
            db.execute("INSERT OR IGNORE INTO experiment_observations VALUES(?,?,?,?,?,?,?,?)",
                (self.camp.name, body["observation_id"], body["plan_id"], content_hash(body), canonical(body), block.source_ref, time.time(), observation_search_text(body)))
        return body

    def valid(self, body: dict, source_ref: str) -> bool:
        from .artifacts import file_digest, resolve_verified_artifact
        try:
            block = self.registry.get(source_ref, role="research_planner")
            if self.registry.load(block, role="research_planner") != body:
                return False
            for ref in body.get("source_refs", []):
                if ref.get("source_ref", "").startswith("context://"):
                    source = self.registry.get(ref["source_ref"], role="research_planner")
                    if source.source_hash != ref.get("source_hash"):
                        return False
                elif ref.get("artifact_id"):
                    row = resolve_verified_artifact(self.camp, ref["artifact_id"])
                    if row["sha256"] != ref.get("sha256"):
                        return False
                    if ref.get("kind") == "method_screening":
                        from .screening import read_screening
                        read_screening(self.camp, Path(row["path"]).parent.name)
                    elif ref.get("kind") == "method_suite":
                        from .method_suite import read_aggregate
                        read_aggregate(self.camp, Path(row["path"]).parent.name)
                    elif ref.get("kind") in {"diagnostic_bundle", "learning_profile"}:
                        report = _read(Path(row["path"]))
                        if any(file_digest(Path(dep["path"])) != dep["sha256"] for dep in report.get("dependencies", [])):
                            return False
                elif ref.get("path"):
                    path = Path(ref["path"])
                    if not path.resolve().is_relative_to(self.camp) or file_digest(path) != ref.get("sha256"):
                        return False
                else:
                    return False
            return True
        except (OSError, KeyError, ValueError, PermissionError):
            return False

    def records(self, *, limit=256, query: str = "", before: tuple[float, str] | None = None,
                observation_id: str | None = None, context: dict | None = None) -> list[dict]:
        """Bounded navigation and a query window into arbitrarily old cards."""
        parameters: list[Any] = [self.camp.name]
        where = "namespace=?"
        if observation_id is not None:
            where += " AND observation_id=?"
            parameters.append(observation_id)
        if before is not None:
            where += " AND (created_at<? OR (created_at=? AND observation_id<?))"
            parameters.extend((before[0], before[0], before[1]))
        keywords = list(dict.fromkeys(word for word in re.findall(r"[\w-]+", query.lower())
            if len(word) >= 4 and word not in {"question", "current", "scope", "method", "protocol", "identity", "with", "from", "that"}))[:8]
        if keywords:
            where += " AND (" + " OR ".join("lower(search_text) LIKE ?" for _ in keywords) + ")"
            parameters.extend("%" + word + "%" for word in keywords)
        parameters.append(min(max(int(limit), 0), 256))
        with self.sidecar.connect() as db:
            rows = db.execute("SELECT payload,source_ref,content_hash,created_at FROM experiment_observations WHERE " + where +
                " ORDER BY created_at DESC,observation_id DESC LIMIT ?", parameters).fetchall()
        results = []
        for row in rows:
            body = json.loads(row["payload"])
            if content_hash(body) == row["content_hash"] and self.valid(body, row["source_ref"]):
                label, differences = observation_applicability(body, context)
                results.append({"record_type": "experiment_observation", "record_id": body["observation_id"],
                    "version": body["content_digest"], "content_hash": row["content_hash"], "body": body,
                    "source_refs": [{"source_ref": row["source_ref"], "source_hash": row["content_hash"]}],
                    "applicability": label, "condition_differences": differences,
                    "navigation": {"created_at": row["created_at"], "observation_id": body["observation_id"]}})
        return results


def observation_search_text(body: dict) -> str:
    """Only the question/mechanism/error, never full observations or IDs."""
    parts = [str(body.get(key) or "") for key in ("question", "intervention")]
    failure = body.get("engineering_failure") or {}
    parts += [str(failure.get("error") or ""), str(failure.get("error_category") or "")]
    identity = body.get("identity") or {}
    facts = body.get("native_facts") or {}
    scope = {key: facts.get(key) for key in ("scope", "fidelity", "seed") if key in facts}
    scope.update({key: identity[key] for key in ("modality", "dataset") if key in identity})
    return "\n".join(part for part in parts if part) + "\nApplicable scope: " + canonical(scope)


def _artifact(camp, ref):
    from .artifacts import resolve_verified_artifact
    row = resolve_verified_artifact(camp, ref)
    return row, _read(Path(row["path"]))


def _single_target_job(camp: Path, entry: dict, protocol: dict) -> tuple[dict, dict, list[dict]]:
    """Read a completed native job through its original source/spec/protocol gates."""
    from .artifacts import file_digest
    from .execution_protocol import command_matches
    from .handoffs import analysis_experiment_binding
    from .runner import accept_job
    directory = Path(str(entry.get("job_dir") or "")).resolve()
    if directory.parent != (camp / "jobs").resolve() or entry.get("evaluation_valid") is not True:
        raise ValueError("observation_single_target_job_invalid")
    job = _read(directory / "job.json")
    frozen = _read(directory / "frozen_run_spec.json")
    binding = _read(directory / "source_binding.json")
    metrics = _read(directory / "metrics.json")
    seed, fidelity, target = entry.get("seed"), entry.get("fidelity"), entry.get("candidate_id")
    if (job.get("status") != "finished" or job.get("candidate_id") != target
            or seed != job.get("seed") or seed != frozen.get("seed")
            or fidelity != job.get("fidelity") or fidelity != frozen.get("fidelity")
            or frozen.get("target_id") != target or frozen.get("spec_hash") != entry.get("spec_hash")
            or frozen.get("evaluation_hash") != protocol["fingerprint"]
            or entry.get("execution_fingerprint") != protocol["fingerprint"]
            or job.get("execution_fingerprint") != protocol["fingerprint"]
            or not command_matches(job.get("command") or [], protocol, fidelity, training_seed=seed)):
        raise ValueError("observation_single_target_execution_identity_mismatch")
    source_path = Path(str(binding.get("class_file") or ""))
    source_path = source_path if source_path.is_absolute() else directory / source_path
    if (file_digest(source_path) != entry.get("source_hash")
            or binding.get("file_sha256") != entry.get("source_hash")
            or job.get("source_hash") != entry.get("source_hash")):
        raise ValueError("observation_single_target_source_changed")
    approval = analysis_experiment_binding(camp, entry)
    if target != "baseline":
        from .native_patch import enforce_requirement_review
        workspace = camp / "candidates" / str(target)
        if approval.get("status") != "verified":
            raise ValueError("observation_single_target_approval_binding_invalid")
        if _read(workspace / "source_manifest.json") != job.get("manifest"):
            raise ValueError("observation_single_target_source_manifest_changed")
        checks = _read(workspace / "checks.json")
        review = enforce_requirement_review(workspace, {"experiment": approval["experiment"]},
            _read(workspace / "review.json"), checks)
        if checks.get("ok") is not True or review.get("status") != "ready":
            raise ValueError("observation_single_target_current_check_review_invalid")
    accepted = accept_job(directory, job.get("manifest"), fidelity, protocol=protocol)
    if (accepted.get("evaluation_valid") is not True or accepted.get("fixed_bank_top1") != entry.get("fixed_bank_top1")
            or metrics.get("fixed_bank_top1") != accepted["fixed_bank_top1"]):
        raise ValueError("observation_single_target_native_result_mismatch")
    expected_identity = {key: protocol.get(key) for key in ("train_image_ids", "validation_image_ids", "train_query_ids",
        "validation_query_ids", "gallery_image_ids", "positive_map", "validation_identity", "val_mode", "schema_version")}
    actual_identity = _read(directory / "evaluation_identity.json")
    if any(canonical(actual_identity.get(key)) != canonical(value) for key, value in expected_identity.items()):
        raise ValueError("observation_single_target_evaluation_identity_mismatch")
    if file_digest(directory / "last.ckpt") != entry.get("checkpoint_id"):
        raise ValueError("observation_single_target_checkpoint_changed")
    paths = [directory / name for name in ("metrics.json", "run_record.json", "source_binding.json",
        "evaluation_identity.json", "frozen_run_spec.json", "hook_config.json", "last.ckpt")]
    if source_path.resolve().is_relative_to(camp.resolve()):
        paths.append(source_path)
    if target != "baseline":
        paths += [workspace / name for name in ("checks.json", "review.json", "source_manifest.json", "approved_binding.json")]
    sources = [{"path": str(path), "sha256": file_digest(path), "kind": "native_single_target_job"} for path in paths]
    if target != "baseline":
        sources.append({"artifact_id": approval["artifact_ref"], "sha256": approval["artifact_sha256"], "kind": "experiment_spec"})
    return copy.deepcopy(entry), metrics, sources


def _single_target_facts(camp: Path, state: dict, plan: dict, entry: dict) -> tuple[dict, list[dict]]:
    from .artifacts import file_digest
    from .comparison import compare_runs, matched_control
    from .execution_protocol import _fingerprint, load_protocol
    protocol = load_protocol(camp)
    if (not protocol or protocol.get("evaluation_mode", "single_target") != "single_target"
            or _fingerprint({key: value for key, value in protocol.items() if key != "fingerprint"}) != protocol["fingerprint"]):
        raise ValueError("observation_single_target_protocol_invalid")
    candidate, metrics, sources = _single_target_job(camp, entry, protocol)
    if any(plan.get(key) is not None and plan[key] != candidate.get(key) for key in ("candidate_id", "source_hash", "spec_hash")):
        raise ValueError("observation_frozen_plan_measurement_identity_mismatch")
    baseline = matched_control(state.get("evidence") or [], candidate, protocol)
    if baseline is not None:
        baseline, baseline_metrics, baseline_sources = _single_target_job(camp, baseline, protocol)
        sources += baseline_sources
    comparison = compare_runs(candidate=candidate, control=baseline, protocol=protocol)
    comparison_path = camp / "comparisons" / (candidate["evidence_id"] + ".json")
    published = _read(comparison_path)
    if published.get("comparison") != comparison or candidate.get("comparison") != comparison:
        raise ValueError("observation_single_target_comparison_changed")
    if candidate.get("promotion") != published.get("promotion"):
        raise ValueError("observation_single_target_promotion_changed")
    sources += [{"path": str(path), "sha256": file_digest(path), "kind": "native_single_target_identity"}
        for path in (camp / "execution_protocol.json", comparison_path)]
    facts = {key: copy.deepcopy(candidate[key]) for key in ("candidate_id", "source_hash", "spec_hash", "seed", "fidelity",
        "checkpoint_id", "execution_fingerprint", "job_id") if key in candidate}
    facts.update(scope="frozen_development", measurement_kind="native_single_target_job", metric_unit="fraction",
        metrics={key: copy.deepcopy(metrics[key]) for key in ("fixed_bank_top1", "fixed_bank_top5") if key in metrics},
        coverage=1, required_fold_count=1, subjects=[protocol["subject"]], query_count=metrics.get("query_count"),
        gallery_size=metrics.get("validation_image_count"), comparison=comparison,
        paired_delta_pp=comparison.get("delta_pp"), promotion=copy.deepcopy(published.get("promotion")),
        paired_baseline={"evidence_id": baseline["evidence_id"], "source_hash": baseline["source_hash"],
            "spec_hash": baseline.get("spec_hash"), "seed": baseline["seed"],
            "metrics": {key: copy.deepcopy(baseline_metrics[key]) for key in ("fixed_bank_top1", "fixed_bank_top5") if key in baseline_metrics}}
            if baseline else None,
        missing=[] if baseline or candidate["candidate_id"] == "baseline" else ["matched_baseline"],
        independently_replicated=False)
    return facts, list({ref.get("path") or ref.get("artifact_id"): ref for ref in sources}.values())


def _native_facts(camp: Path, state: dict, plan: dict) -> tuple[dict, list]:
    target = plan.get("candidate_id") or plan.get("target_candidate_ref") or (plan.get("proposal") or {}).get("target_candidate_ref")
    evidence_id = plan.get("measurement_evidence_id")
    evidence = [row for row in state.get("evidence", []) if isinstance(row, dict)
        and (row.get("evidence_id") == evidence_id if evidence_id else row.get("candidate_id") == target)]
    facts = {"metrics": None, "paired_delta_pp": None, "coverage": None, "scope": None,
        "promotion": None, "missing": ["verified_measurement"]}
    sources = []
    for entry in reversed(evidence):
        if entry.get("job_dir") and entry.get("evaluation_valid") is True and entry.get("kind") not in {"method_suite", "method_screening"}:
            try:
                return _single_target_facts(camp, state, plan, entry)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        ref = entry.get("artifact_id") or entry.get("artifact") or entry.get("evaluation_ref")
        if not ref and entry.get("artifact_refs"):
            ref = entry["artifact_refs"][0]
        if not ref and entry.get("suite_id"):
            from .artifacts import lookup
            registry_path = camp / "artifact_registry.jsonl"
            if registry_path.is_file():
                registered = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                ref = next((item["artifact_id"] for item in reversed(registered)
                    if item.get("kind") == entry.get("kind") and Path(item["path"]).parent.name == entry["suite_id"]), None)
        if not isinstance(ref, str) or not ref.startswith("art_"):
            continue
        try:
            row, result = _artifact(camp, ref)
            if row["kind"] == "method_screening":
                from .screening import read_screening
                result = read_screening(camp, Path(row["path"]).parent.name)
                from .method_suite import validate_suite
                method = validate_suite(camp, Path(row["path"]).parent.name)["method"]
                if any(plan.get(key) is not None and plan[key] != method.get(key) for key in ("candidate_id", "source_hash", "spec_hash", "method_revision")):
                    raise ValueError("observation_frozen_plan_measurement_identity_mismatch")
                facts = {key: copy.deepcopy(result[key]) for key in (
                    "scope", "phase", "coverage", "required_fold_count", "full_required_fold_count", "seed",
                    "fidelity", "screening_subjects", "screening_fold_ids", "selection_manifest_hash", "science_scope",
                    "candidate_id", "method_revision", "source_hash", "spec_hash", "screening_result", "comparison",
                    "paired_scores", "baseline_suite_hash", "screening_mean_delta_pp", "promotion_decision", "status") if key in result}
                facts.update(metrics={key: result[key] for key in ("screening_mean_top1", "screening_mean_top5") if key in result},
                    paired_delta_pp=result.get("screening_mean_delta_pp"), promotion=result.get("screening_result"),
                    missing=[] if result.get("coverage") == result.get("required_fold_count") else ["declared_fold_coverage"],
                    independently_replicated=False)
                facts.update({key: method[key] for key in ("source_hash", "spec_hash", "method_revision") if key in method})
            elif row["kind"] == "method_suite":
                from .method_suite import verified_suite_records
                verified = next(item for item in verified_suite_records(camp, state) if item.get("artifact_id") == ref or item.get("evidence_id") == entry.get("evidence_id"))
                aggregate = verified["aggregate"]
                if any(plan.get(key) is not None and plan[key] != verified.get(key) for key in ("candidate_id", "source_hash", "spec_hash", "method_revision")):
                    raise ValueError("observation_frozen_plan_measurement_identity_mismatch")
                facts = {key: copy.deepcopy(aggregate[key]) for key in ("scope", "seed", "candidate_id", "coverage", "required_fold_count", "metric_unit", "independent_seed_count", "science_scope") if key in aggregate}
                facts.update(metrics={key: aggregate[key] for key in ("mean_top1", "mean_top5") if key in aggregate},
                    paired_delta_pp=(verified.get("comparison") or {}).get("delta_pp"), comparison=verified.get("comparison"),
                    scope="method_development_benchmark", fidelity="full", promotion=copy.deepcopy(verified.get("promotion")),
                    missing=[] if "promotion" in verified else ["promotion_status"], independently_replicated=False,
                    folds=[{key: score[key] for key in ("fold_id", "held_out_subject", "seed") if key in score} |
                        {"metrics": {key: score["metrics"][key] for key in ("top1", "top5")}} for score in aggregate.get("scores", [])])
                facts.update({key: verified[key] for key in ("source_hash", "spec_hash", "method_revision") if key in verified})
            elif row["kind"] in {"development_score", "benchmark", "evaluation_result"}:
                from .handoffs import development_artifact
                development_artifact(camp, ref)
                facts = {key: copy.deepcopy(result[key]) for key in ("scope", "seed", "fidelity", "metrics", "coverage", "comparison") if key in result}
                facts.setdefault("metrics", None)
                facts["missing"] = [key for key in ("scope", "coverage", "seed") if key not in facts]
            elif row["kind"] in {"diagnostic_bundle", "learning_profile"}:
                from .suite_diagnostics import VERSION as DIAGNOSTIC_VERSION, select_verified_facts
                from .artifacts import file_digest
                if result.get("schema_version") != DIAGNOSTIC_VERSION or result.get("final_test_accessed") is not False:
                    raise ValueError("observation_diagnostic_source_not_native")
                if any(file_digest(Path(dep["path"])) != dep["sha256"] for dep in result.get("dependencies", [])):
                    raise ValueError("observation_diagnostic_dependencies_changed")
                selected = select_verified_facts(camp, state, plan, ref, max_facts=12)
                verified, selection = selected["facts"], copy.deepcopy(selected["selection"])
                readings = []
                missing = list(selection["missing"])
                for item in verified:
                    profile = item.get("learning_profile") or {}
                    readings.append({key: copy.deepcopy(item[key]) for key in ("candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "phase", "scope", "source_hash", "spec_hash", "method_revision", "job_id", "diagnostics", "source_ref", "source_identity_missing") if key in item} |
                        {"learning_profile": {key: copy.deepcopy(profile[key]) for key in ("status", "reason", "epochs", "best_epoch", "best_fixed_bank_top1", "first_train_loss", "last_train_loss", "read_coverage", "metric_scope") if key in profile}})
                    label = str(item.get("job_id") or item.get("suite_id") or item.get("fold_id") or "unknown_source")
                    missing.extend("source_identity:" + label + ":" + field for field in item.get("source_identity_missing", []))
                    if profile.get("status") != "ready":
                        missing.append("partial_or_missing_history:" + label)
                    if (item.get("diagnostics") or {}).get("status") in {"unavailable", "partial"}:
                        missing.append("partial_or_missing_diagnostic_export:" + label)
                selection["satisfies_development_read_receipt"] = False
                facts = {"scope": result["diagnostic_scope"], "metrics": None, "paired_delta_pp": None,
                    "measurement_kind": "existing_development_diagnostic", "source_epoch": result["source_epoch"],
                    "diagnostic_status": result["status"], "summary": copy.deepcopy(result.get("summary") or {}),
                    "summary_authority": "native_collector_summary_not_planner_read_receipt",
                    "native_diagnostics": readings, "additional_training_jobs": 0,
                    "diagnostic_selection": selection, "source_count": selection["native_report_source_count"],
                    "delivered_source_count": selection["delivered_source_count"],
                    "omitted_scoped_source_count": selection["omitted_scoped_source_count"],
                    "missing": list(dict.fromkeys(missing)),
                    "independently_replicated": False}
            else:
                continue
            sources.append({"artifact_id": ref, "sha256": row["sha256"], "kind": row["kind"]})
            break
        except (OSError, ValueError, KeyError, StopIteration):
            continue
    return facts, sources


def _current_review_failure(camp: Path, state: dict, plan: dict) -> tuple[dict, list[dict]]:
    """Carry source-bound native blocking issues into terminal failure feedback."""
    from .artifacts import file_digest
    from .identity import result_matches
    from .native_patch import _check_fingerprint
    from .implementation_semantics import requirements_hash
    from .plan_protocol import scientific_projection
    from .context.types import digest
    target = plan.get("candidate_id")
    if not target or target == "baseline" or plan.get("current_step") not in {"implement", "repair", "check"}:
        return {}, []
    workspace = camp / "candidates" / str(target)
    paths = {name: workspace / name for name in ("review.json", "checks.json", "spec.json", "attempt.json", "implementation.json")}
    entry = workspace / "extension/eeg_candidate.py"
    if not entry.is_file() or not all(path.is_file() for path in paths.values()):
        return {}, []  # Legacy/environment failures need not have a review.
    review, checks, spec, attempt, implementation = (_read(paths[name]) for name in paths)
    source_hash, check_hash = file_digest(entry), file_digest(paths["checks.json"])
    experiment = spec.get("experiment") or {}
    if (not result_matches(review, candidate_id=str(target), attempt_id=attempt.get("attempt_id"),
            phase="review_candidate", input_hash=source_hash)
            or review.get("status") not in {"blocked", "needs_fix"}
            or checks.get("source_sha256") != source_hash or checks.get("check_fingerprint") != _check_fingerprint(workspace, None)
            or (checks.get("implementation_semantics") or {}).get("requirements_hash")
                != requirements_hash(experiment.get("implementation_requirements") or [])
            or implementation.get("attempt_id") != attempt.get("attempt_id") or implementation.get("input_hash") != source_hash
            or experiment.get("spec_hash") != plan.get("spec_hash")
            or experiment.get("spec_hash") != (state.get("experiment") or {}).get("spec_hash")
            or digest(scientific_projection(experiment)) != plan.get("approved_scientific_digest")
            or (plan.get("source_hash") and plan["source_hash"] != source_hash)):
        return {"failure_feedback_status": "current_review_identity_mismatch"}, []
    blocking = [copy.deepcopy(issue) for issue in review.get("issues") or []
                if isinstance(issue, dict) and issue.get("severity") == "blocking"]
    if not blocking:
        return {}, []
    identity = {"candidate_id": target, "attempt_id": attempt["attempt_id"], "source_hash": source_hash,
        "spec_hash": experiment["spec_hash"], "check_sha256": check_hash,
        "review_sha256": file_digest(paths["review.json"])}
    # Retain only the original blocking rows behind a real bounded read; never
    # copy unrelated model context or treat a failed review as measurement.
    block = ContextSourceRegistry(camp).capture({"identity": identity, "blocking_issues": blocking},
        block_id="selected-review-failure:" + plan["plan_id"], kind="checks", role="research_planner", identity=identity)
    record = {"source_refs": [{"source_ref": block.source_ref, "source_hash": block.source_hash}]}
    projected = []
    for index, issue in enumerate(blocking[:4]):
        pointer = "/blocking_issues/" + str(index)
        row = {}
        for key in ("requirement_id", "kind", "error_class", "stage", "detail", "expected"):
            if key in issue:
                value = issue[key]
                row[key] = copy.deepcopy(value) if len(canonical(value)) <= 1200 else _feedback_handle(record, pointer + "/" + key)
        actual = issue.get("actual") or {}
        if isinstance(actual, dict):
            if "status" in actual:
                row["coverage_status"] = actual["status"]
            if "detail" in actual:
                value = actual["detail"]
                row["coverage_detail"] = value if len(canonical(value)) <= 1200 else _feedback_handle(record, pointer + "/actual/detail", unit="chars", size=len(value))
        row["source_hash"] = source_hash
        row["original_issue"] = _feedback_handle(record, pointer)
        projected.append(row)
    sources = [{"path": str(path.resolve()), "sha256": file_digest(path), "kind": "implementation_failure_input"}
               for path in [entry, *paths.values()]]
    sources.append({"source_ref": block.source_ref, "source_hash": block.source_hash, "kind": "blocking_review_issues"})
    return {"failure_stage": "review_candidate", "verified_failure_source": identity,
        "blocking_issues": projected, "blocking_issue_count": len(blocking),
        "omitted_blocking_issue_count": max(0, len(blocking) - len(projected)),
        "blocking_issues_read_ref": _feedback_handle(record, "/blocking_issues", unit="items", size=len(blocking))}, sources


def record_experiment_observation(camp: Path, state: dict, plan: dict, execution_status: str,
                                  analysis_ref: str | None = None) -> dict:
    camp = Path(camp)
    from .research_execution import research_goal
    planning_v2 = ((research_goal(camp).get("research_execution") or {}).get("planning_contract_version") == "v2")
    proposal = plan.get("proposal") or plan.get("selected_proposal") or plan
    failure = copy.deepcopy(plan.get("engineering_failure") or {})
    engineering = execution_status in {"failed", "needs_replan"}
    facts, sources = _native_facts(camp, state, plan) if not engineering else ({"metrics": None, "missing": ["measurement_not_completed"]}, [])
    measured_task = proposal.get("task_kind") in {"new_method", "confirmation", "validation"} and (
        (proposal.get("evaluation_scope") or {}).get("phase") != "diagnostic")
    if execution_status == "observation_ready" and measured_task and (facts.get("metrics") is None or not sources):
        raise ValueError("observation_verified_measurement_missing")
    interpretation = None
    if analysis_ref:
        row, analysis = _artifact(camp, analysis_ref)
        if row.get("kind") not in {"analysis", "result_analysis", "role_result", "analysis_output"}:
            raise ValueError("observation_analyst_artifact_kind_invalid")
        if plan.get("candidate_id") is not None and row.get("candidate_id") != plan["candidate_id"]:
            raise ValueError("observation_analyst_candidate_identity_mismatch")
        from .handoffs import development_artifact
        development_artifact(camp, analysis_ref)
        explanation = analysis.get("payload") or analysis
        measured = next((entry for entry in state.get("evidence", []) if entry.get("evidence_id") == plan.get("measurement_evidence_id")), None)
        if measured and measured.get("kind") in {"method_screening", "method_suite"}:
            from .handoffs import method_screening_origin, method_benchmark_origin
            screening = measured["kind"] == "method_screening"
            key = "method_screening_origin" if screening else "method_benchmark_origin"
            expected = method_screening_origin(measured) if screening else method_benchmark_origin(measured)
            if explanation.get(key) != expected:
                raise ValueError("observation_analyst_measurement_origin_mismatch")
        from .analysis_contract import analysis_memory_view, is_record
        if is_record(explanation):
            # Staged records keep the complete summary; native facts stay the Observation's own.
            view = analysis_memory_view(explanation)
            points = {key: view[key] for key in ("hypothesis_assessment", "summary_zh", "limitations", "next_steps")}
            if planning_v2:
                points["findings"] = copy.deepcopy(view["findings"])
            omitted = []
        else:
            points = {key: copy.deepcopy(explanation[key]) for key in (
                "hypothesis_assessment", "summary_zh", "evidence_gaps", "suggested_next_actions")
                if key in explanation and len(canonical(explanation[key])) <= 1200}
            omitted = [key for key in ("hypothesis_assessment", "summary_zh", "evidence_gaps", "suggested_next_actions")
                if key in explanation and len(canonical(explanation[key])) > 1200]
        interpretation = {"artifact_id": analysis_ref, "sha256": row["sha256"], "candidate_id": row.get("candidate_id"), "interpretation_only": True,
            "read_ref": analysis_ref, "points": points, "omitted_point_fields": omitted}
        if is_record(explanation):
            interpretation["record_version"] = explanation["record_version"]
        sources.append({"artifact_id": analysis_ref, "sha256": row["sha256"], "kind": row["kind"]})
    if engineering:
        feedback, failure_sources = _current_review_failure(camp, state, plan)
        failure.update(feedback)
        sources.extend(failure_sources)
        failure.setdefault("error", plan.get("error") or plan.get("block_reason") or "implementation_failed")
        failure.setdefault("error_fingerprint", content_hash({"error": failure["error"], "source_hash": plan.get("source_hash"), "spec_hash": plan.get("spec_hash")}))
        failure.setdefault("repairs_used", plan.get("repair_count", 0))
        failure.setdefault("error_category", "engineering_failure")
    conditions = planner_applicability_context(camp, {**state, "experiment": plan,
        "evaluation_scope": proposal.get("evaluation_scope") or {}})
    intervention = ((proposal.get("method") or proposal.get("task_description") or proposal.get("scientific_intent"))
                    if planning_v2 else (proposal.get("scientific_intent") or proposal.get("task_description")))
    body = {"origin_campaign_id": camp.name, "plan_id": plan["plan_id"], "observation_kind": "engineering_failure" if engineering else "experiment",
        "execution_status": execution_status, "question": proposal.get("question"),
        "intervention": intervention,
        "identity": {key: plan[key] for key in ("candidate_id", "method_revision", "source_hash", "spec_hash", "scientific_intent_digest") if key in plan} |
            {key: conditions[key] for key in ("dataset", "modality") if key in conditions},
        "evaluation_scope": copy.deepcopy(proposal.get("evaluation_scope") or {}),
        "native_facts": facts, "analyst_interpretation": interpretation,
        "engineering_failure": failure if engineering else None,
        "unresolved_questions": copy.deepcopy(plan.get("unresolved_questions") or []),
        "source_refs": sources, "applicability": {**conditions, "campaign_id": camp.name,
            "scope": facts.get("scope", conditions.get("scope")), "seed": facts.get("seed", conditions.get("seed")),
            "scientific_intent_digest": plan.get("scientific_intent_digest")}}
    if facts.get("measurement_kind") == "existing_development_diagnostic":
        body["identity"]["binding_level"] = "plan_target_with_per_source_diagnostic_identity"
        body["identity"]["unknown_top_level_fields"] = [key for key in ("source_hash", "spec_hash", "method_revision")
            if body["identity"].get(key) is None]
        body["identity"]["single_scientific_revision_claimed"] = False
    return ExperimentObservationStore(camp).record(body)


def planner_applicability_context(camp: Path, fullobs: dict) -> dict:
    """Pass task conditions, not only the query's candidate ID, to both paths."""
    from .semantic_memory import build_context
    experiment = fullobs.get("experiment") or {}
    context = build_context(camp, state=fullobs, experiment=experiment)
    for owner in (fullobs.get("contract"), fullobs.get("protocol"), experiment,
                  fullobs.get("selected_research_task"), fullobs.get("current_task_context"), fullobs):
        if isinstance(owner, dict):
            context.update(observation_conditions(owner))
    for key in ("candidate_id", "terminal_observation_id"):
        if fullobs.get(key) is not None:
            context[key] = fullobs[key]
    selected = fullobs.get("selected_plan") or {}
    context.setdefault("terminal_observation_id", selected.get("terminal_observation_id"))
    return context


def _feedback_handle(record: dict, pointer: str = "", *, unit: str = "json", size: int | None = None) -> dict:
    source = record["source_refs"][0]
    request = {"source_ref": source["source_ref"], "source_hash": source["source_hash"],
               "json_pointer": pointer, "unit": unit, "max_chars": 2000}
    if size is not None:
        request.update(start=0, end=min(size, 512 if unit == "chars" else 1))
    return {"$context_source": source["source_ref"], "source_hash": source["source_hash"],
            "json_pointer": pointer, "fidelity": "REF_ONLY", "read_unit": unit,
            "read_capability": "registered_bounded_delivery_read", "satisfies_development_read_receipt": False,
            "next_read": request}


def observation_science_view(body: dict, *, include_analysis: bool = True) -> dict:
    """A pure delivery view; immutable observations remain the complete record.

    Reuse the paid Analyst's complete summary instead of replaying every finding.
    Scope and numeric results stay exact. Unknown fields are retained so a new
    required fact cannot silently disappear through this adapter.
    """
    result = {key: copy.deepcopy(value) for key, value in body.items() if key != "analyst_interpretation"}
    # The registered original and its digest bind these identities. Repeating
    # storage and revision fields inside every scientific block adds no evidence.
    for key in ("schema_version", "origin_campaign_id", "content_digest"):
        result.pop(key, None)
    identity = result.get("identity")
    if isinstance(identity, dict) and body.get("observation_kind") != "engineering_failure":
        for key in ("source_hash", "spec_hash", "scientific_intent_digest", "method_revision"):
            identity.pop(key, None)
    facts = result.get("native_facts")
    if isinstance(facts, dict):
        for key in ("candidate_id", "method_revision", "source_hash", "spec_hash",
                    "selection_manifest_hash", "baseline_suite_hash"):
            facts.pop(key, None)
    applicability = result.get("applicability")
    if isinstance(applicability, dict):
        # Retrieval checks the original applicability before this view. Keep
        # scientific conditions, including a possibly different historical recipe.
        for key in ("campaign_id", "task", "evaluation_identity", "terminal_observation_id",
                    "candidate_id", "source_hash", "spec_hash", "method_revision",
                    "scientific_intent_digest", "selection_manifest_hash"):
            applicability.pop(key, None)
        recipe = applicability.get("training_recipe")
        if isinstance(recipe, dict):
            for key in ("data_root", "gpu"):
                recipe.pop(key, None)
            if (applicability.get("input_geometry") is not None
                    and recipe.get("input_geometry") == applicability["input_geometry"]):
                recipe.pop("input_geometry")
        # Remove only exact repetitions, not historical differences. Each
        # retained condition is already present in a scientific block of this card.
        conditions = {"dataset": (identity or {}).get("dataset"),
            "modality": (identity or {}).get("modality"),
            **{key: (facts or {}).get(name) for key, name in (
                ("scope", "scope"), ("seed", "seed"), ("fold_count", "required_fold_count"),
                ("fold_ids", "screening_fold_ids"), ("screening_subjects", "screening_subjects"))}}
        if isinstance(recipe, dict):
            conditions["negative_sampling_policy"] = recipe.get("negative_sampling_policy")
        for key, value in conditions.items():
            if value is not None and key in applicability and canonical(applicability[key]) == canonical(value):
                applicability.pop(key)
    if isinstance(result.get("source_refs"), list):
        result["source_refs"] = [{key: ref[key] for key in ("artifact_id", "source_ref") if key in ref}
                                 for ref in result["source_refs"] if isinstance(ref, dict)
                                 and (ref.get("artifact_id") or ref.get("source_ref"))]
    interpretation = body.get("analyst_interpretation")
    if isinstance(interpretation, dict) and include_analysis:
        points = interpretation.get("points")
        points = points if isinstance(points, dict) else interpretation
        summary = {key: copy.deepcopy(points[key]) for key in
                   ("hypothesis_assessment", "summary_zh", "summary", "limitations") if key in points}
        if summary:
            summary["interpretation_only"] = True
            ref = interpretation.get("artifact_id") or interpretation.get("read_ref")
            if ref:
                summary["artifact_id"] = ref
            result["analyst_interpretation"] = summary
        else:
            result.pop("analyst_interpretation", None)
    else:
        result.pop("analyst_interpretation", None)
    return result


def minimum_observation_card(record: dict, *, current: bool = False, context: dict | None = None,
                             inline_analysis: bool = True) -> dict:
    """Deterministic feedback: exact identities/results, prose behind real reads.

    Large exact measurements or identities remain exact. If this minimum is too
    large the caller must report NeedsContext; it never clips native facts.
    """
    original = record["body"]
    body = observation_science_view(original, include_analysis=inline_analysis)
    from .context.representations import NARRATIVE_FIELDS, LOG_FIELDS
    prose_fields = NARRATIVE_FIELDS | LOG_FIELDS | {"reason", "error", "warning", "summary_zh"}
    omissions = []
    next_reads = []
    identity_fields = {"candidate_id", "plan_id", "method_revision", "source_hash", "spec_hash", "scope",
        "seed", "fold_id", "held_out_subject", "suite_id", "job_id", "dataset", "modality",
        "evaluation_identity", "source_ref", "source_identity_missing", "error_category", "error_fingerprint"}
    def first_text(value, path):
        if isinstance(value, str) and len(value) > 512:
            return path, value
        if isinstance(value, dict):
            children = ((path + "/" + str(key).replace("~", "~0").replace("/", "~1"), child) for key, child in value.items())
        elif isinstance(value, list):
            children = ((path + "/" + str(i), child) for i, child in enumerate(value))
        else:
            return None
        return next((found for child_path, child in children if (found := first_text(child, child_path))), None)
    def omit(path, value):
        unit = "chars" if isinstance(value, str) else "items" if isinstance(value, list) else "json"
        omissions.append({"path": path, "reason": "bounded_current_feedback_view",
            "read_unit": unit, "original_size": len(value) if isinstance(value, (str, list, dict)) else None})
        if len(next_reads) < 3:
            text = first_text(value, path)
            read_path, read_value = text if text else (path, value)
            read_unit = "chars" if isinstance(read_value, str) else "items" if isinstance(read_value, list) else "json"
            next_reads.append(_feedback_handle(record, read_path, unit=read_unit,
                size=len(read_value) if isinstance(read_value, (str, list)) else None)["next_read"])
    def exact_diagnostics(value, path):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
                narrative = key in prose_fields or any(word in key.lower() for word in ("error", "traceback"))
                if isinstance(child, str) and len(child) > 512 and narrative and key not in identity_fields:
                    omit(child_path, child)
                else:
                    result[key] = exact_diagnostics(child, child_path)
            return result
        if isinstance(value, list):
            return [exact_diagnostics(child, path + "/" + str(i)) for i, child in enumerate(value)]
        return copy.deepcopy(value)
    # Producer fields not classified by this adapter remain literal obligations.
    # New scientific/control fields must not disappear behind a claimed minimum.
    projected = {"native_facts", "engineering_failure", "question", "intervention", "unresolved_questions", "analyst_interpretation"}
    selected = {key: copy.deepcopy(value) for key, value in body.items() if key not in projected}
    facts = copy.deepcopy(body.get("native_facts") or {})
    for key in ("summary", "native_diagnostics", "diagnostic_selection"):
        if key in facts:
            facts[key] = exact_diagnostics(facts[key], "/native_facts/" + key)
    selected["native_facts"] = facts
    failure = body.get("engineering_failure")
    if isinstance(failure, dict):
        selected["engineering_failure"] = exact_diagnostics(
            {key: value for key, value in failure.items() if key != "error"}, "/engineering_failure")
        selected["engineering_failure"].setdefault("error_category", "engineering_failure")
        selected["engineering_failure"].setdefault("error_fingerprint", content_hash({"error": failure.get("error"),
                "source_hash": (original.get("identity") or {}).get("source_hash"), "spec_hash": (original.get("identity") or {}).get("spec_hash")}))
        if "error" in failure:
            omit("/engineering_failure/error", failure["error"])
    for key in ("question", "intervention", "unresolved_questions", "analyst_interpretation"):
        value = body.get(key)
        if key == "unresolved_questions" and not value:
            continue
        if value is not None:
            if key in {"question", "intervention", "analyst_interpretation"}:
                selected[key] = copy.deepcopy(value)
            else:
                omit("/" + key, value)
    if original.get("analyst_interpretation") and not inline_analysis:
        omit("/analyst_interpretation", original["analyst_interpretation"])
    points = (original.get("analyst_interpretation") or {}).get("points") or {}
    if inline_analysis and points.get("findings"):
        # A source pointer documents that findings are saved, not available as a
        # model tool. Planner receives the existing summary, not a partial finding.
        omissions.append({"path": "/analyst_interpretation/points/findings",
                          "reason": "complete_analysis_saved_summary_delivered"})
    if not next_reads:
        # A minimum blocked by an unfamiliar long obligation still gives a
        # usable page of that exact source, without classifying it as omitted.
        # The Analyst summary was delivered, and complete findings are an
        # intentional saved detail. Do not add a redundant paging request for
        # either. Unknown required fields still provide a real recovery page.
        text = first_text({key: value for key, value in original.items()
                           if key != "analyst_interpretation"}, "")
        if text:
            next_reads.append(_feedback_handle(record, text[0], unit="chars", size=len(text[1]))["next_read"])
    label, differences = observation_applicability(original, context)
    return {"record_type": "experiment_observation", "record_id": record["record_id"], "body": selected,
        "source_refs": copy.deepcopy(record["source_refs"]), "applicability": label,
        "condition_differences": differences, "current_feedback": current, "fidelity": "FACT_CARD",
        "omissions": omissions,
        "full_record": {key: value for key, value in _feedback_handle(record).items()
                        if key in {"$context_source", "source_hash", "fidelity"}},
        "next_reads": next_reads, "minimum_feedback_version": "eeg.observation.minimum.v2"}


def _legacy_recall_observations(camp: Path, question: str, *, recent=2, relevant=4, char_cap=16000, token_cap=16000,
                               context: dict | None = None, retrieval_out: dict | None = None) -> dict:
    """Current terminal feedback is mandatory; optional history uses remaining space."""
    from .semantic_memory import retrieve, embedding_health
    from .context.retrieval import prepare_query
    store = ExperimentObservationStore(camp)
    records = store.records(context=context)
    current_id = (context or {}).get("terminal_observation_id")
    current = (store.records(limit=1, observation_id=current_id, context=context) if current_id else records[:1])
    if current_id and not current:
        raise ObservationDeliveryRequired(NeedsContext(({"observation_id": current_id,
            "status": "current_observation_unreadable"},), (), "current_observation_unreadable"))
    choices = current + records[:min(max(recent, 0), 2)]
    config = memory_config(camp)
    backend = "recent_only"
    health = {"backend": "recent_only", "degraded": False, "reason": "similarity_not_requested" if config.enabled else "disabled",
              "model": config.model_id, "revision": config.model_revision, "fingerprint": None,
              "query": question, "query_status": "not_encoded", "indexing": {"status": "not_requested"}}
    if config.enabled and relevant > 0:
        query = prepare_query(question=question, identities=[(context or {}).get("candidate_id")])["text"]
        bundle = retrieve(camp, query, config=config, role="research_planner", context=context)
        health = embedding_health(bundle)
        if bundle.get("degraded") and single_plan_enabled(camp):
            raise ObservationDeliveryRequired(NeedsContext(({
                "status": "embedding_retrieval_unavailable", "embedding_health": health,
                "required_action": "Restore the frozen embedding backend and resume the same campaign."},), (),
                "embedding_retrieval_unavailable", {}, {"paid_maintenance_calls": 0}))
        if retrieval_out is not None:
            # Current/recent observations and related priors have independent
            # top-k allocations; five recent experiment hits cannot crowd every
            # useful older lesson out of the Planner's knowledge channel.
            knowledge = retrieve(camp, query, config=config, role="research_planner", context=context,
                                 record_types=frozenset({"lesson", "knowledge"}))
            if knowledge.get("degraded"):
                raise ObservationDeliveryRequired(NeedsContext(({
                    "status": "embedding_retrieval_unavailable", "embedding_health": embedding_health(knowledge)},), (),
                    "embedding_retrieval_unavailable", {}, {"paid_maintenance_calls": 0}))
            retrieval_out.update(knowledge)
            health["knowledge_indexing"] = knowledge.get("indexing")
            health["knowledge_candidate_count"] = knowledge.get("candidate_count")
            health["knowledge_query_status"] = knowledge.get("query_status")
            health["knowledge_query_sha256"] = knowledge.get("query_sha256")
        backend = bundle.get("backend", backend)
        choices += [item for item in bundle.get("items", []) if item["record_type"] == "experiment_observation"][:min(relevant, 5)]
    result = {"status": "ready", "cards": [], "budget_chars": char_cap, "budget_tokens": token_cap,
        "backend": backend, "max_recent": min(max(recent, 0), 2), "max_relevant": min(max(relevant, 0), 5),
        "retrieval_health": health,
        "full_records_readable": True, "current_observation_id": current[0]["record_id"] if current else None,
        "omitted_history_count": 0, "delivered_chars": 0}
    def fits(candidate):
        candidate["delivered_chars"] = len(canonical(candidate["cards"]))
        text = canonical(candidate)
        return len(text) <= char_cap and len(text.encode("utf-8")) <= token_cap
    seen, seen_sources = set(), set()
    for record in choices:
        if record["record_id"] in seen:
            continue
        seen.add(record["record_id"])
        body = record.get("body") or next((row["body"] for row in records if row["record_id"] == record["record_id"]), None)
        if record.get("minimum_feedback_version"):
            original = store.records(limit=1, observation_id=record["record_id"], context=context)
            body = original[0]["body"] if original else None
        if body is None:
            continue
        record = {**record, "body": body}
        mandatory = bool(current and record["record_id"] == current[0]["record_id"])
        native_sources = tuple(sorted(ref.get("artifact_id") or ref.get("source_ref") or ref.get("path") or ""
            for ref in body.get("source_refs", []) if ref.get("kind") not in {"analysis", "result_analysis", "role_result", "analysis_output"}))
        if native_sources and native_sources in seen_sources and not mandatory:
            continue
        card = minimum_observation_card(record, current=mandatory, context=context)
        trial = {**result, "cards": result["cards"] + [card]}
        if not fits(trial) and card["body"].get("analyst_interpretation"):
            # Too large with its inline conclusion: deliver the reference and record the omission
            # instead of blocking the round or claiming the text was delivered.
            card = minimum_observation_card(record, current=mandatory, context=context, inline_analysis=False)
            trial = {**result, "cards": result["cards"] + [card]}
        if fits(trial):
            result = trial
            if native_sources:
                seen_sources.add(native_sources)
        elif mandatory:
            handles = card["next_reads"]
            if not handles:
                handles = [_feedback_handle(record)["next_read"]]
            raise ObservationDeliveryRequired(NeedsContext(({"observation_id": record["record_id"],
                "plan_id": body["plan_id"], "status": "minimum_current_feedback_exceeds_budget",
                "minimum_chars": len(canonical(trial)), "minimum_utf8_token_upper_bound": len(canonical(trial).encode("utf-8")),
                "full_record": card["full_record"]},), tuple(ReadRequest(**handle) for handle in handles[:3]),
                "current_observation_minimum_exceeds_budget", {"budget_chars": char_cap, "budget_tokens": token_cap},
                {"paid_maintenance_calls": 0}))
        else:
            result["omitted_history_count"] += 1
    fits(result)
    return result


def _scientific_memory_enabled(camp: Path) -> bool:
    """Memory delivery is independent of the Planner's output contract."""
    return single_plan_enabled(camp)


def recall_observations(camp: Path, question: str, *, recent=2, relevant=4, char_cap=16000, token_cap=16000,
                        context: dict | None = None, retrieval_out: dict | None = None) -> dict:
    """Freeze complete feedback plus one shared allowance for optional priors.

    The legacy char/token arguments remain part of the old compiler API. The
    single-plan flow applies the scientific content allowance here independently
    of the Planner's response contract;
    the final request compiler owns the complete wire and model token limits.
    """
    if not _scientific_memory_enabled(camp):
        return _legacy_recall_observations(camp, question, recent=recent, relevant=relevant,
            char_cap=char_cap, token_cap=token_cap, context=context, retrieval_out=retrieval_out)
    from .semantic_memory import retrieve, embedding_health, planner_knowledge_delivery, record_delivery_report
    from .context.retrieval import prepare_query
    from .memory_delivery import project_memory_card, select_memory_cards

    store = ExperimentObservationStore(camp)
    records = store.records(limit=max(1, min(max(recent, 0), 2)), context=context)
    current_id = (context or {}).get("terminal_observation_id")
    current = store.records(limit=1, observation_id=current_id, context=context) if current_id else records[:1]
    if current_id and not current:
        raise ObservationDeliveryRequired(NeedsContext(({"observation_id": current_id,
            "status": "current_observation_unreadable"},), (), "current_observation_unreadable"))
    config = memory_config(camp)
    health = {"backend": "recent_only", "degraded": False,
        "reason": "similarity_not_requested" if config.enabled else "disabled",
        "model": config.model_id, "revision": config.model_revision, "fingerprint": None,
        "query": question, "query_status": "not_encoded", "indexing": {"status": "not_requested"}}
    related_observations, knowledge_cards = [], []
    knowledge = {"status": "disabled", "items": []}
    rank_by_id = {}
    if config.enabled and relevant > 0:
        query = prepare_query(question=question, identities=[(context or {}).get("candidate_id")])["text"]
        # Independent type allocations prevent a run of experiment hits from
        # excluding every useful lesson before the shared content selection.
        bundle = retrieve(camp, query, config=config, role="research_planner", context=context,
                          record_types=frozenset({"experiment_observation"}))
        knowledge_bundle = retrieve(camp, query, config=config, role="research_planner", context=context,
                                    record_types=frozenset({"lesson", "knowledge"}))
        health = embedding_health(bundle)
        health.update(knowledge_indexing=knowledge_bundle.get("indexing"),
            knowledge_candidate_count=knowledge_bundle.get("candidate_count"),
            knowledge_query_status=knowledge_bundle.get("query_status"),
            knowledge_query_sha256=knowledge_bundle.get("query_sha256"))
        for candidate in (bundle, knowledge_bundle):
            if candidate.get("degraded"):
                raise ObservationDeliveryRequired(NeedsContext(({
                    "status": "embedding_retrieval_unavailable", "embedding_health": embedding_health(candidate),
                    "required_action": "Restore the frozen embedding backend and resume the same campaign."},), (),
                    "embedding_retrieval_unavailable", {}, {"paid_maintenance_calls": 0}))
            for item in candidate.get("items", []):
                rank_by_id[(item["record_type"], item["record_id"])] = item
        related_observations = bundle.get("items", [])[:min(max(relevant, 0), 5)]
        knowledge = planner_knowledge_delivery(camp, knowledge_bundle, context=context)
        knowledge_cards = knowledge.get("items", [])

    current_id = current[0]["record_id"] if current else None
    seen, dedup_keys = set(), {}

    def observation_card(record):
        identity = (record.get("record_type", "experiment_observation"), record["record_id"])
        if identity in seen:
            return None
        body = record.get("body")
        if body is None or record.get("minimum_feedback_version"):
            original = store.records(limit=1, observation_id=record["record_id"], context=context)
            body = original[0]["body"] if original else None
        if body is None:
            return None
        mandatory = record["record_id"] == current_id
        native_refs = tuple(sorted(ref.get("artifact_id") or ref.get("source_ref") or ref.get("path") or ""
            for ref in body.get("source_refs", []) if isinstance(ref, dict)
            and ref.get("kind") not in {"analysis", "result_analysis", "role_result", "analysis_output"}))
        # The same native source is duplicate evidence only within the same
        # scientific conditions and evaluation scope, never across folds/seeds.
        native_identity = (native_refs, canonical(body.get("evaluation_scope")),
                           canonical(observation_conditions(body)))
        seen.add(identity)
        label, differences = observation_applicability(body, context)
        card = project_memory_card({**record, "record_type": "experiment_observation", "body": body,
            "applicability": label, "condition_differences": differences, "current_feedback": mandatory},
            role="research_planner", context=context)
        if native_refs:
            dedup_keys[identity] = native_identity
        return card

    priority = {"compatible_evidence": 0, "preconditions_met": 1, "analogy_only": 2,
                "preconditions_mismatch": 2, "applicability_unknown": 3, "preconditions_unknown": 3}

    def related_order(card):
        row = rank_by_id.get((card["record_type"], card["record_id"]), {})
        label = row.get("precondition_status", row.get("applicability", card.get("applicability")))
        return (priority.get(label, 3) if isinstance(label, str) else 3, -float(row.get("cosine_score", 0)),
                card["record_type"], card["record_id"], str(row.get("version", "")))

    ordered = [card for record in current + records[:min(max(recent, 0), 2)]
               if (card := observation_card(record)) is not None]
    related = [card for record in related_observations if (card := observation_card(record)) is not None]
    related.extend(project_memory_card(card, role="research_planner", context=context) for card in knowledge_cards)
    ordered.extend(sorted(related, key=related_order))
    selected = select_memory_cards(ordered, content_chars_budget=config.content_chars_budget,
                                   mandatory_ids=[current_id] if current_id else (), dedup_keys=dedup_keys)
    report_ref = record_delivery_report(camp, "research_planner", selected.report, retrieval=health)
    observation_cards = [card for card in selected.cards if card["record_type"] == "experiment_observation"]
    selected_knowledge = {"status": knowledge.get("status", "no_related_knowledge"),
        "items": [card for card in selected.cards if card["record_type"] in {"lesson", "knowledge"}]}
    if selected_knowledge["items"]:
        selected_knowledge["status"] = "ready"
    elif selected_knowledge["status"] == "ready":
        selected_knowledge["status"] = "no_related_knowledge"
    if retrieval_out is not None:
        retrieval_out.update(selected_knowledge=selected_knowledge, selection_report=selected.report,
                             selection_report_ref=report_ref)
    return {"status": "ready", "cards": observation_cards,
        "current_observation_id": current_id, "retrieval_health": health,
        "omitted_history_count": sum(row.get("record_type") == "experiment_observation"
                                     for row in selected.report["omitted"]),
        "delivered_chars": len(canonical(observation_cards))}


# Fields explicitly moved behind readable references in the versioned adapter.
# Unknown fields remain delivered and therefore retain Compiler obligations.
HISTORY_FIELDS = frozenset({"evidence", "method_screenings", "method_suites", "analyses", "checks", "candidate_checks",
    "recent_decisions", "artifact_index", "evidence_index", "memory_index", "research_progress", "method_evidence_packet",
    "training_target_sources", "source", "implementation_source", "parent_source", "control_source", "history", "curves",
    "stdout", "stderr", "lessons", "retrieved_memory", "verified_development_facts", "latest_comparison", "latest_diagnostics"})


def _admission_view(rows) -> tuple[Any, dict]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return copy.deepcopy(rows), {}
    receipt_fields = {"schema_version", "campaign_id", "execution_fingerprint", "refund_on_failure"}
    common = {key: copy.deepcopy(rows[0][key]) for key in receipt_fields if rows and key in rows[0]
        and all(key in row and canonical(row[key]) == canonical(rows[0][key]) for row in rows)}
    compact = [{key: copy.deepcopy(value) for key, value in row.items() if key not in common and key != "committed_at"} for row in rows]
    return compact, {"version": "candidate_admission_delivery.v1", "common_exact_fields": common,
        "total_count": len(rows), "omitted_receipt_fields": ["committed_at"], "full_receipts_readable": True}


def _cost_view(costs) -> tuple[Any, dict]:
    if not isinstance(costs, dict):
        return copy.deepcopy(costs), {}
    rows = costs.get("by_action_target") if isinstance(costs.get("by_action_target"), dict) else costs
    archived = {"components", "historical_samples", "full_followup_pipeline_cost", "conditional_followup"}
    independent = {"diagnose_results", "collect_diagnostics", "retrieve_memory"}
    result, generic = {}, set()
    for key, row in rows.items():
        if not isinstance(key, str) or not isinstance(row, dict):
            result[key] = copy.deepcopy(row)
            continue
        action, _, target = key.partition(":")
        base = rows.get(action + ":")
        if target and action in independent and isinstance(base, dict) and canonical(base) == canonical(row):
            generic.add(action)
            continue
        # Known detailed old-action material is readable as one original table;
        # new/unknown cost/control fields remain literal and require packing.
        result[key] = {field: copy.deepcopy(value) for field, value in row.items() if field not in archived}
    view = {**copy.deepcopy(costs), "by_action_target": result} if "by_action_target" in costs else result
    return view, {"version": "native_action_cost_delivery.v1", "generic_target_independent_actions": sorted(generic),
        "total_original_rows": len(rows), "delivered_rows": len(result), "archived_detail_fields": sorted(archived),
        "cost_scope": "native action facts; selected task envelope adds necessary preparation and feedback; future confirmation requires another selected plan",
        "full_details_readable": True}


def compact_planner_view(camp: Path, fullobs: dict, *, char_cap=16000, token_cap=16000,
                         context: dict | None = None) -> dict:
    if not single_plan_enabled(camp):
        return copy.deepcopy(fullobs)
    from .semantic_memory import planner_query, planner_knowledge_delivery
    conditions = {**planner_applicability_context(camp, fullobs), **(context or {})}
    question = planner_query(camp, fullobs)
    def attach_retrieval(result):
        bundle = {}
        recalled = recall_observations(camp, question, context=conditions,
            char_cap=char_cap, token_cap=token_cap, retrieval_out=bundle)
        result["memory_retrieval_health"] = recalled.pop("retrieval_health")
        result["experiment_observations"] = recalled
        if "selected_knowledge" in bundle:
            result["retrieved_knowledge"] = bundle["selected_knowledge"]
            result["memory_delivery_report"] = bundle["selection_report"]
            result["memory_delivery_report_ref"] = bundle["selection_report_ref"]
        elif bundle:
            result["retrieved_knowledge"] = planner_knowledge_delivery(camp, bundle, context=conditions)
        else:
            result["retrieved_knowledge"] = {"status": "disabled", "items": [],
                "scientific_evidence": False, "execution_authority": False}
        navigation = [row for row in result.get("history_navigation") or []
                      if not str(row.get("field", "")).startswith("retrieved_knowledge/")]
        for card in result["retrieved_knowledge"]["items"]:
            original = card.get("full_record")
            if not isinstance(original, dict):
                continue
            navigation.append({"field": "retrieved_knowledge/" + card["record_id"],
                "$context_source": original["$context_source"], "source_hash": original["source_hash"],
                "fidelity": "REF_ONLY", "read_capability": "registered_bounded_delivery_read",
                "satisfies_development_read_receipt": False})
        result["history_navigation"] = navigation
        return result
    if fullobs.get("context_delivery_version") == "observation_card_first.v1":
        if _scientific_memory_enabled(camp) and fullobs.get("memory_delivery_report"):
            # A frozen scientific selection is already complete. The compiler
            # can measure the same science without reretrieving or dropping it.
            return copy.deepcopy(fullobs)
        # Recheck the actual request's budget without recapturing projections
        # as new authoritative heads for their original history sources.
        result = copy.deepcopy(fullobs)
        return attach_retrieval(result)
    registry = ContextSourceRegistry(camp)
    result, navigation = {}, []
    from .design_contract import design_contract_version
    design_v2 = design_contract_version(camp) == "v2"
    def register(key, value):
        block = registry.capture(value, block_id="planner-history:" + key, kind="scientific" if key in {"evidence", "method_screenings", "method_suites"} else "analysis", role="research_planner")
        navigation.append({"field": key, "$context_source": block.source_ref, "source_hash": block.source_hash,
            "fidelity": "REF_ONLY", "read_capability": "registered_bounded_delivery_read", "satisfies_development_read_receipt": False})
    for key, value in fullobs.items():
        if key in HISTORY_FIELDS:
            register(key, value)
            if (design_v2 and key in {"parent_source", "control_source"}
                    and isinstance(value, dict) and value.get("candidate_id") == "baseline"
                    and "baseline_source" not in result):
                # A new scientific proposal needs the actual baseline method.
                # Parent/control are often identical; carry its wrapper and
                # encoder excerpt once to both staged planning roles.
                result["baseline_source"] = copy.deepcopy(value)
        elif key in {"candidate_admissions", "inherited_candidate_admissions"}:
            result[key], projection = _admission_view(value)
            if projection:
                register(key, value)
                result[key + "_projection"] = projection
        elif key == "action_cost_estimates":
            result[key], projection = _cost_view(value)
            if projection:
                register(key, value)
                result["action_cost_estimates_projection"] = projection
        elif key == "controller_state" and isinstance(value, dict):
            result[key] = {}
            for field, item in value.items():
                if field in {"hypothesis_evidence", "candidate_states", "next_comparisons", "reserved_confirmation"}:
                    register("controller_state/" + field, item)
                else:
                    # New controller obligations require literal delivery or
                    # explicit Compiler classification, including nested fields.
                    result[key][field] = copy.deepcopy(item)
        else:
            result[key] = copy.deepcopy(value)
    result["history_navigation"] = navigation
    attach_retrieval(result)
    result["context_delivery_version"] = "observation_card_first.v1"
    return result
