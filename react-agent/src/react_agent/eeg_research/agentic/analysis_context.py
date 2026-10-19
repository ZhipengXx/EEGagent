"""Read accepted run evidence for the Analyst without generating new diagnostics.

Measurement owns validation of the results. This collector consumes that accepted
snapshot, checks the small files it opens against the recorded bindings, and reads
each source once per collection. No checkpoint, EEG data, model or artifact writer
is used. The returned source bindings belong to the local record, not the prompt.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


DEVELOPMENT_SCOPE = "source_subject_development"


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


class _Sources:
    def __init__(self, camp: Path):
        self.camp = Path(camp).resolve()
        self.raw: dict[Path, bytes] = {}
        self.source_catalog: dict[str, dict] = {}
        self.source_bindings: dict[str, dict] = {}

    def path(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        if not target.is_relative_to(self.camp):
            raise ValueError("analysis_source_outside_campaign")
        return target

    def read(self, path: str | Path) -> bytes:
        target = self.path(path)
        if target not in self.raw:
            try:
                self.raw[target] = target.read_bytes()
            except OSError as exc:
                raise ValueError("analysis_source_unavailable:" + target.name) from exc
        return self.raw[target]

    def document(self, path: str | Path, *, expected_sha: str | None = None) -> dict:
        raw = self.read(path)
        if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("analysis_source_binding_changed:" + Path(path).name)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("analysis_source_json_invalid:" + Path(path).name) from exc
        if not isinstance(body, dict):
            raise ValueError("analysis_source_not_object:" + Path(path).name)
        return body

    def cite(self, path: str | Path, kind: str, scope: str) -> str:
        target = self.path(path)
        sha = hashlib.sha256(self.read(target)).hexdigest()
        # Equal contents at different run paths are different sources.
        identity = hashlib.sha256((str(target) + "\0" + sha).encode()).hexdigest()
        ref = "analysis_source_" + identity[:24]
        self.source_catalog.setdefault(ref, {"source_ref": ref, "content_kind": kind,
                                              "scientific_scope": scope})
        self.source_bindings[ref] = {"path": str(target), "sha256": sha}
        return ref


def _learning(sources: _Sources, job: Path) -> dict:
    path = job / "history.jsonl"
    if not path.is_file():
        return {**_unavailable("history_not_exported"), "scope": DEVELOPMENT_SCOPE,
                "points": [], "read_coverage": {"valid_rows": 0, "complete": False}}
    raw = sources.read(path)
    points, errors = [], []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not line.endswith(b"\n"):
            errors.append({"kind": "incomplete_tail", "line": index + 1})
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            errors.append({"kind": "invalid_json_row", "line": index + 1})
            continue
        if not isinstance(row, dict):
            errors.append({"kind": "non_object_row", "line": index + 1})
            continue
        points.append(row)
    ref = sources.cite(path, "development_learning_curve", DEVELOPMENT_SCOPE)
    return {"status": "partial" if errors else "ready" if points else "unavailable",
            "scope": DEVELOPMENT_SCOPE, "points": points,
            "read_coverage": {"valid_rows": len(points), "complete": not errors and bool(points),
                              "truncated": False, "diagnostics": errors}, "source_ref": ref}


def _config(spec: dict) -> dict:
    result = {key: copy.deepcopy(spec[key]) for key in ("model", "objective", "transform", "recipe") if key in spec}
    policy = spec.get("negative_policy", (spec.get("recipe") or {}).get("negative_sampling_policy"))
    if policy is not None:
        result["negative_sampling_policy"] = policy
    return result


def _run(sources: _Sources, *, arm: str, job: Path, identity: dict, freeze: dict | None = None,
         freeze_path: Path | None = None) -> dict:
    job = sources.path(job)
    spec_path = job / "frozen_run_spec.json"
    refs = []
    expected = None
    if freeze is not None:
        dependencies = {str(sources.path(row["path"])): row["sha256"] for row in freeze.get("dependencies") or []
                        if Path(row["path"]).name == "frozen_run_spec.json"}
        expected = dependencies.get(str(spec_path))
        if not expected:
            raise ValueError("analysis_frozen_config_binding_missing")
        refs.append(sources.cite(freeze_path, "frozen_fold_development_evidence", DEVELOPMENT_SCOPE))
    spec = sources.document(spec_path, expected_sha=expected)
    for key, expected_value in (("target_id", identity.get("candidate_id")), ("seed", identity.get("seed"))):
        if key in spec and expected_value is not None and spec[key] != expected_value:
            raise ValueError("analysis_frozen_config_identity_mismatch:" + key)
    identity = copy.deepcopy(identity)
    for key, source_key in (("source_hash", "source_manifest_hash"), ("spec_hash", "spec_hash")):
        if spec.get(source_key) is not None:
            if key in identity and identity[key] != spec[source_key]:
                raise ValueError("analysis_frozen_config_identity_mismatch:" + key)
            identity[key] = spec[source_key]
    refs.append(sources.cite(spec_path, "frozen_training_config", "training_configuration"))
    if freeze is not None:
        diagnostics = copy.deepcopy(freeze.get("diagnostics") or _unavailable("diagnostics_not_exported"))
        selected = copy.deepcopy(freeze.get("selected_checkpoint") or _unavailable("checkpoint_selection_not_exported"))
        metrics = copy.deepcopy(freeze.get("development_metrics") or _unavailable("development_metrics_not_exported"))
    else:
        def optional(name, kind):
            path = job / name
            if not path.is_file():
                return _unavailable(name + "_not_exported")
            try:
                value = sources.document(path)
            except ValueError:
                return _unavailable(name + "_invalid")
            refs.append(sources.cite(path, kind, DEVELOPMENT_SCOPE))
            return copy.deepcopy(value)
        diagnostics = optional("diagnostic_summary.json", "development_diagnostics")
        selected = optional("selected_checkpoint.json", "development_checkpoint_selection")
        metrics = optional("metrics.json", "development_metrics")
    dynamics = (diagnostics.get("items") or {}).get("training_dynamics") or {}
    execution = copy.deepcopy(dynamics.get("training_execution") or _unavailable("training_execution_not_exported"))
    learning = _learning(sources, job)
    if learning.get("source_ref"):
        refs.append(learning["source_ref"])
    return {"arm": arm, **copy.deepcopy(identity), "job_id": job.name,
            "config": _config(spec), "training_execution": execution,
            "development_diagnostics": diagnostics, "checkpoint_selection": selected,
            "development_metrics": metrics, "development_learning": learning,
            "diagnostic_scope": DEVELOPMENT_SCOPE, "source_refs": list(dict.fromkeys(refs))}


def _score_runs(sources: _Sources, scores: list[dict], arm: str) -> list[dict]:
    runs = []
    seen = set()
    for score in scores:
        if not score.get("freeze_ref") or not score.get("freeze_sha256"):
            raise ValueError("analysis_score_freeze_binding_missing")
        path = sources.path(score["freeze_ref"])
        freeze = sources.document(path, expected_sha=score["freeze_sha256"])
        if arm == "baseline" and freeze.get("candidate_id") != "baseline":
            raise ValueError("analysis_matched_control_identity_mismatch")
        for key in ("candidate_id", "suite_hash", "fold_id", "seed", "checkpoint_hash"):
            if key in score and freeze.get(key) != score[key]:
                raise ValueError("analysis_score_freeze_identity_mismatch:" + key)
        identity = {key: copy.deepcopy(freeze[key]) for key in (
            "candidate_id", "suite_id", "fold_id", "held_out_subject", "seed", "fidelity", "method_revision") if key in freeze}
        key = (identity.get("suite_id"), identity.get("fold_id"), identity.get("seed"))
        if key in seen:
            raise ValueError("analysis_duplicate_fold")
        seen.add(key)
        runs.append(_run(sources, arm=arm, job=Path(freeze["job_dir"]), identity=identity,
                         freeze=freeze, freeze_path=path))
    return runs


def _confirmation_control(sources: _Sources, state: dict, latest: dict) -> dict | None:
    comparison = latest.get("comparison") or {}
    control_id = latest.get("control_id") or comparison.get("control_run_id")
    suite_hash = comparison.get("control_suite_hash")
    def checked(body, expected_hash):
        if not isinstance(body, dict) or not body.get("scores"):
            raise ValueError("analysis_matched_control_unavailable")
        if expected_hash and any(score.get("suite_hash") != expected_hash for score in body["scores"]):
            raise ValueError("analysis_control_suite_mismatch")
        return body
    rows = [row for row in state.get("evidence") or [] if row.get("candidate_id") == "baseline"
            and (row.get("evidence_id") == control_id if control_id else suite_hash and row.get("suite_hash") == suite_hash)]
    if rows:
        row = rows[0]
        if suite_hash and row.get("suite_hash") != suite_hash:
            raise ValueError("analysis_control_suite_mismatch")
        if row.get("suite_ref"):
            wrapper = sources.document(row["suite_ref"], expected_sha=row.get("result_hash"))
            return checked(wrapper.get("aggregate"), suite_hash or row.get("suite_hash"))
        if isinstance(row.get("aggregate"), dict):
            return checked(row["aggregate"], suite_hash or row.get("suite_hash"))
    if suite_hash:
        for path in sorted((sources.camp / "suites").glob("*/suite_manifest.json")):
            manifest = sources.document(path)
            if manifest.get("suite_hash") == suite_hash and (manifest.get("method") or {}).get("candidate_id") == "baseline":
                return checked(sources.document(path.with_name("aggregate.json")).get("aggregate"), suite_hash)
    if control_id or suite_hash or comparison.get("comparable"):
        raise ValueError("analysis_matched_control_unavailable")
    return None


def collect_analysis_evidence(camp: Path, state: dict, latest: dict, *, verified_result: dict | None = None) -> dict:
    """Collect facts for a newly accepted single run, screening or confirmation.

    ``verified_result`` is the existing native verifier's result. When omitted for
    a suite, invoke that verifier rather than treating arbitrary JSON as measured
    evidence. The collector does not validate another full campaign or synthesize
    missing diagnostic measurements.
    """
    sources = _Sources(camp)
    kind = latest.get("kind")
    if kind in {"method_screening", "method_suite"}:
        if verified_result is None:
            if kind == "method_screening":
                from .screening import read_screening
                verified_result = read_screening(sources.camp, latest["suite_id"])
            else:
                from .method_suite import read_aggregate
                verified_result = read_aggregate(sources.camp, latest["suite_id"])
        arm = "baseline" if latest.get("candidate_id") == "baseline" else "candidate"
        runs = _score_runs(sources, verified_result.get("scores") or [], arm)
        if not runs:
            raise ValueError("analysis_accepted_training_runs_missing")
        if kind == "method_screening":
            baseline_scores = verified_result.get("baseline_scores") or []
            if arm == "candidate":
                runs.extend(_score_runs(sources, baseline_scores, "baseline"))
        elif arm == "candidate":
            control = _confirmation_control(sources, state, latest)
            if control is not None:
                runs.extend(_score_runs(sources, control.get("scores") or [], "baseline"))
    else:
        rows = [(latest, "baseline" if latest.get("candidate_id") == "baseline" else "candidate")]
        control_id = latest.get("control_id")
        if control_id:
            control = next((row for row in state.get("evidence") or [] if row.get("evidence_id") == control_id), None)
            if control is None or control.get("candidate_id") != "baseline":
                raise ValueError("analysis_matched_control_unavailable")
            rows.append((control, "baseline"))
        runs = []
        for row, arm in rows:
            if not row.get("job_dir"):
                raise ValueError("analysis_job_binding_missing")
            identity = {key: copy.deepcopy(row[key]) for key in ("candidate_id", "seed", "fidelity", "source_hash", "spec_hash") if key in row}
            runs.append(_run(sources, arm=arm, job=Path(row["job_dir"]), identity=identity))
    return {"training_runs": runs, "source_catalog": list(sources.source_catalog.values()),
            "source_bindings": sources.source_bindings}
