"""Scientific-design handoff and CPU/GPU scheduling, without paid APIs or GPUs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic import coder, native_patch

SOURCE = "class EEGCandidate:\n    def build_encoder(self, geometry, config=None):\n        return None\n"


def workspace(tmp_path, *, version="v2", requirements=None):
    spec = {"experiment": {"design_contract_version": version, "spec_hash": "approved-design",
                           "model": {"d_model": 128}, "objective": {}, "transform": {},
                           "implementation_requirements": requirements or []}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "input_spec.json").write_text(json.dumps({"design_contract_version": version, "c_num": 17, "timesteps": [0, 250],
                                                         "multigpu_check": {"enabled": True}}))
    return spec


def patch(request, content=SOURCE):
    return {"tool": "apply_candidate_patch", "args": {"path": "extension/eeg_candidate.py", "content": content,
                                                       "expected_base_hash": (request.get("current_file") or {}).get("sha256", "")}}


def checks(monkeypatch, *, final=None):
    calls = []
    def checked(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("cpu_only"):
            return {"ok": True, "stage": "complete"}
        return dict(final or {"ok": True, "multigpu_check": {"ok": True}})
    monkeypatch.setattr(native_patch, "run_candidate_check", checked)
    return calls


def test_v2_first_handoff_contains_real_caller_and_reference_directory(tmp_path, monkeypatch):
    spec = workspace(tmp_path)
    checks(monkeypatch)
    def backend(request):
        assert request["effective_hook_config"] == {"model": {"d_model": 128}, "objective": {}, "transform": {}}
        assert "def _build_hook(" in "\n".join(row["excerpt"] for row in request["hook_config_runtime_source"])
        assert "reference/check_entry.py" in request["available_references"]
        assert "reference/train_entry.py" not in request["references"]
        return {"tool": "request_design_revision", "args": {
            "reason": "effective config conflicts with a design promise", "requirement_ids": [],
            "evidence_refs": ["reference/train_entry.py::_build_hook"],
            "smallest_correction": "clarify the config contract", "validation_expectation": "hook accepts the effective config"}}
    result = native_patch.implement(tmp_path, spec, backend, max_steps=1)
    assert result["status"] == "needs_design_revision"
    assert result["detail"]["kind"] == "design_conflict"
    assert result["detail"]["approved_spec_hash"] == "approved-design"
    assert not (tmp_path / "checks.json").exists()


def test_patch_checks_cpu_twice_finish_runs_gpu_once_without_extra_model_call(tmp_path, monkeypatch):
    spec = workspace(tmp_path)
    calls = checks(monkeypatch)
    model_calls = []
    def backend(request):
        model_calls.append(request)
        if request["step"] < 3:
            if request["step"] == 2:
                assert (tmp_path / "cpu_checks.json").exists()
                assert not (tmp_path / "checks.json").exists()
                assert request["check_status"]["ready_for_review"] is False
            return patch(request, SOURCE + f"revision = {request['step']}\n")
        return {"tool": "finish_patch", "args": {"summary": "complete"}}
    outcome = native_patch.implement(tmp_path, spec, backend, max_steps=3)
    assert outcome["status"] == "ready_for_review"
    assert len(model_calls) == 3
    assert [bool(call.get("cpu_only")) for call in calls] == [True, True, False]
    assert calls[-1]["cpu_result"]["check_scope"] == "cpu"
    assert json.loads((tmp_path / "checks.json").read_text())["check_scope"] == "complete"


def test_v1_keeps_combined_checks_after_each_patch(tmp_path, monkeypatch):
    spec = workspace(tmp_path, version="v1")
    calls = checks(monkeypatch)
    def backend(request):
        assert request["check_policy"]["automatic_after_patch"] is True
        if request["step"] < 3:
            return patch(request, SOURCE + f"revision = {request['step']}\n")
        return {"tool": "finish_patch", "args": {"summary": "legacy"}}
    assert native_patch.implement(tmp_path, spec, backend, max_steps=3)["status"] == "ready_for_review"
    assert len(calls) == 2 and all("cpu_only" not in call for call in calls)
    assert not (tmp_path / "cpu_checks.json").exists()


def test_final_gpu_failure_cannot_create_manifest(tmp_path, monkeypatch):
    spec = workspace(tmp_path)
    checks(monkeypatch, final={"ok": False, "error": "synthetic_gpu_shape_failure"})
    def backend(request):
        return patch(request) if request["step"] == 1 else {"tool": "finish_patch", "args": {"summary": "complete"}}
    outcome = native_patch.implement(tmp_path, spec, backend, max_steps=2)
    assert outcome["status"] == "implementation_failed"
    assert outcome["check"]["error"] == "synthetic_gpu_shape_failure"
    assert not (tmp_path / "source_manifest.json").exists()


def test_pending_gpu_restores_without_model_or_repeating_cpu(tmp_path, monkeypatch):
    spec = workspace(tmp_path)
    calls = checks(monkeypatch, final={"ok": False, "cpu_check_ok": True, "error": "gpu_check_pending"})
    def backend(request):
        return patch(request) if request["step"] == 1 else {"tool": "finish_patch", "args": {"summary": "complete"}}
    assert native_patch.implement(tmp_path, spec, backend, max_steps=2)["status"] == "check_pending"
    calls[:] = []
    def available(path, **kwargs):
        calls.append(kwargs)
        assert not kwargs.get("cpu_only")
        return {"ok": True, "multigpu_check": {"ok": True}}
    monkeypatch.setattr(native_patch, "run_candidate_check", available)
    monkeypatch.setattr("react_agent.eeg_research.agentic.engineering_memory.record_engineering_repair", lambda *a, **k: None)
    outcome = native_patch.implement(tmp_path, spec, lambda request: pytest.fail("resource recovery called a model"), max_steps=2)
    assert outcome["status"] == "ready_for_review"
    assert len(calls) == 1 and "cpu_result" in calls[0]


def test_implementation_map_is_separate_and_source_bound(tmp_path, monkeypatch):
    requirements = [{"requirement_id": "attention", "kind": "implementation", "core": True, "binding_stage": "design",
                     "expected_behavior": "temporal attention", "verification": "source_review"}]
    spec = workspace(tmp_path, requirements=requirements)
    original = (tmp_path / "spec.json").read_bytes()
    calls = checks(monkeypatch)
    def backend(request):
        if request["step"] == 1:
            return patch(request)
        return {"tool": "finish_patch", "args": {"summary": "mapped", "implementation_map": {
            "attention": {"source_refs": ["extension/eeg_candidate.py::EEGCandidate.build_encoder"]}}}}
    outcome = native_patch.implement(tmp_path, spec, backend, max_steps=2)
    assert outcome["status"] == "ready_for_review"
    assert (tmp_path / "spec.json").read_bytes() == original
    mapping = json.loads((tmp_path / "implementation_map.json").read_text())
    assert mapping["source_sha256"] == outcome["check"]["source_sha256"]
    assert mapping["approved_spec_hash"] == "approved-design"
    assert sum(bool(call.get("cpu_only")) for call in calls) == 1  # Source refs are review metadata, not another CPU computation.


def test_ordered_probe_binding_preserves_scientific_declaration():
    requirement = {"requirement_id": "ordered", "verification": "ordered_window_slots", "probe": {
        "slot_count": 4, "layout": "flatten_channel_major", "window_axis": -1}}
    original = copy.deepcopy(requirement)
    binding = {"source_sha256": "source", "approved_spec_hash": "spec", "requirements": {
        "ordered": {"probe": {"input_module_path": "pool", "representation_module_path": "projection"}}}}
    derived = coder.bound_implementation_requirements([requirement], binding, source_sha256="source", spec_hash="spec")
    assert requirement == original
    assert derived[0]["probe"] == {**original["probe"], **binding["requirements"]["ordered"]["probe"]}
    binding["requirements"]["ordered"]["probe"]["slot_count"] = 2
    with pytest.raises(ValueError, match="cannot_change_probe_semantics"):
        coder.bound_implementation_requirements([requirement], binding, source_sha256="source", spec_hash="spec")
    assert coder.bound_implementation_requirements([requirement], binding, source_sha256="new-source", spec_hash="spec") == [original]


def test_reference_search_and_checker_alias_are_readable(tmp_path):
    result = coder.search_code(tmp_path, "def _build_hook(")
    assert any(row["path"] == "reference/train_entry.py" for row in result["hits"])
    hit = next(row for row in coder.search_code(tmp_path, "from react_agent.eeg_training.train_entry import _build_hook")["hits"]
               if row["path"] == "reference/check_entry.py")
    result = coder.read_code(tmp_path, "reference/check_entry.py", hit["line"], hit["line"] + 8)
    assert result["ok"] and "_build_hook" in result["text"]


def test_checker_fingerprint_changes_when_actual_caller_changes(tmp_path, monkeypatch):
    workspace(tmp_path)
    (tmp_path / "extension").mkdir()
    (tmp_path / "extension/eeg_candidate.py").write_text(SOURCE)
    from react_agent.eeg_research.agentic import binding
    original = binding.file_sha256
    marker = {"hash": "caller-v1"}
    monkeypatch.setattr(binding, "file_sha256", lambda path: marker["hash"] if path.name == "train_entry.py" else original(path))
    before = native_patch._check_fingerprint(tmp_path, "/python", cpu_only=True)
    marker["hash"] = "caller-v2"
    assert native_patch._check_fingerprint(tmp_path, "/python", cpu_only=True) != before


@pytest.mark.parametrize("kind", ["design_conflict", "scientific_replan"])
def test_design_revision_tool_has_native_typed_arguments(kind):
    tool, args, error = coder.validate_tool_request({"tool": "request_design_revision", "args": {
        "kind": kind, "reason": "contract conflict", "requirement_ids": ["r"], "evidence_refs": ["checks.json"],
        "smallest_correction": "revise contract", "validation_expectation": "CPU hook check passes"}})
    assert error is None and tool == "request_design_revision" and args["kind"] == kind


@pytest.mark.parametrize("erase", [False, True])
def test_real_cpu_ordered_probe_uses_coder_mapping_and_detects_erased_slots(tmp_path, erase):
    from react_agent.eeg_training.protocol import torch_python
    from react_agent.eeg_research.agentic.implementation_semantics import requirements_hash
    interpreter = torch_python()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("training Python is unavailable")
    requirement = {"requirement_id": "ordered", "kind": "implementation", "core": True, "binding_stage": "design",
                   "expected_behavior": "four ordered window slots reach the projection", "verification": "ordered_window_slots",
                   "probe": {"slot_count": 4, "layout": "flatten_channel_major", "window_axis": -1}}
    spec = workspace(tmp_path, requirements=[requirement])
    (tmp_path / "extension").mkdir()
    candidate = f'''from torch import nn
class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.projection = nn.Linear(channels * 4, 1024)
    def forward(self, x):
        features = self.pool(x)
        if {erase!r}:
            features = features.mean(-1, keepdim=True).expand_as(features)
        return self.projection(features.flatten(1))
class EEGCandidate:
    def build_encoder(self, geometry, config=None):
        return Encoder(geometry['c_num'])
'''
    (tmp_path / "extension/eeg_candidate.py").write_text(candidate)
    stored = native_patch._store_implementation_map(tmp_path, spec, {
        "ordered": {"source_refs": ["extension/eeg_candidate.py::Encoder.forward"],
                    "probe": {"input_module_path": "pool", "representation_module_path": "projection"}}})
    assert stored["ok"]
    receipt = coder.run_candidate_check(tmp_path, python=interpreter, cpu_only=True)
    assert receipt["implementation_semantics"]["requirements_hash"] == requirements_hash([requirement])
    observed = receipt["implementation_semantics"]["requirements"][0]
    assert observed["status"] == ("contradicted" if erase else "implemented")
    assert receipt["ok"] is (not erase)
    assert "multigpu_check" not in receipt


def test_unbound_probe_does_not_consume_one_failure_budget_before_final_binding(tmp_path, monkeypatch):
    from react_agent.eeg_training.protocol import torch_python
    from react_agent.eeg_research.agentic import coding_recovery
    interpreter = torch_python()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("training Python is unavailable")
    camp = tmp_path / "camp"
    target = camp / "candidates/c1"
    target.mkdir(parents=True)
    (camp / "goal.json").write_text(json.dumps({"coding_recovery": {"max_failed_checks": 1, "max_tool_steps": 4}}))
    requirement = {"requirement_id": "ordered", "kind": "implementation", "core": True, "binding_stage": "design",
                   "expected_behavior": "ordered slots", "verification": "ordered_window_slots",
                   "probe": {"slot_count": 4, "layout": "flatten_channel_major", "window_axis": -1}}
    spec = workspace(target, requirements=[requirement])
    candidate = '''from torch import nn
class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.projection = nn.Linear(channels * 4, 1024)
    def forward(self, x):
        return self.projection(self.pool(x).flatten(1))
class EEGCandidate:
    def build_encoder(self, geometry, config=None):
        return Encoder(geometry['c_num'])
'''
    # Real CPU, fake final GPU completion; no physical device is opened.
    monkeypatch.setattr(coder, "_complete_multigpu_check", lambda workspace, python, input_spec, payload, timeout: {
        **payload, "multigpu_check": {"ok": True}})
    monkeypatch.setattr("react_agent.eeg_research.agentic.engineering_memory.record_engineering_repair", lambda *a, **k: None)
    def backend(request):
        if request["step"] < 4:
            if request["step"] > 1:
                assert request["last_check"]["ok"]
                assert request["last_check"]["probe_bindings_pending"] == ["ordered"]
                assert request["coding_recovery"]["failed_checks_used"] == 0
            return patch(request, candidate + f"revision = {request['step']}\n")
        return {"tool": "finish_patch", "args": {"summary": "bound", "implementation_map": {
            "ordered": {"source_refs": ["extension/eeg_candidate.py::Encoder.forward"],
                        "probe": {"input_module_path": "pool", "representation_module_path": "projection"}}}}}
    outcome = native_patch.implement(target, spec, backend, python=interpreter, max_steps=4)
    assert outcome["status"] == "ready_for_review"
    assert coding_recovery.load(target)["failed_checks_used"] == 0
    assert outcome["check"]["probe_bindings_pending"] == []
    assert outcome["check"]["implementation_semantics"]["requirements"][0]["status"] == "implemented"


def test_final_preflight_without_required_module_binding_is_blocked(tmp_path):
    from react_agent.eeg_training.protocol import torch_python
    interpreter = torch_python()
    if interpreter is None or not Path(interpreter).is_file():
        pytest.skip("training Python is unavailable")
    requirement = {"requirement_id": "ordered", "kind": "implementation", "core": True, "binding_stage": "design",
                   "expected_behavior": "ordered slots", "verification": "ordered_window_slots",
                   "probe": {"slot_count": 4, "layout": "flatten_channel_major", "window_axis": -1}}
    spec = workspace(tmp_path, requirements=[requirement])
    candidate = '''from torch import nn
class EEGCandidate:
    def build_encoder(self, geometry, config=None):
        return nn.Sequential(nn.Flatten(), nn.Linear(geometry['c_num'] * 250, 1024))
'''
    def backend(request):
        return patch(request, candidate) if request["step"] == 1 else {"tool": "finish_patch", "args": {"summary": "missing binding"}}
    outcome = native_patch.implement(tmp_path, spec, backend, python=interpreter, max_steps=2)
    assert outcome["status"] == "implementation_failed"
    assert outcome["check"]["implementation_semantics"]["requirements"][0]["status"] == "unverified"
    assert not (tmp_path / "source_manifest.json").exists()


@pytest.mark.parametrize("scope", ["scientific", "engineering"])
def test_native_confirmed_contradiction_has_correct_repair_owner(tmp_path, scope):
    from react_agent.eeg_research.agentic.binding import file_sha256
    from react_agent.eeg_research.agentic.implementation_semantics import requirements_hash
    requirement = {"requirement_id": "r", "kind": "implementation", "core": True, "binding_stage": "design", "scope": scope,
                   "expected_behavior": "preserve baseline hooks", "verification": "source_review"}
    spec = workspace(tmp_path, requirements=[requirement])
    (tmp_path / "extension").mkdir()
    source = tmp_path / "extension/eeg_candidate.py"
    source.write_text(SOURCE)
    receipt = {"ok": True, "source_sha256": file_sha256(source), "check_fingerprint": native_patch._check_fingerprint(tmp_path, None),
               "implementation_semantics": {"requirements_hash": requirements_hash([requirement]), "requirements": []}}
    check_path = tmp_path / "checks.json"
    check_path.write_text(json.dumps(receipt))
    coverage = {"requirement_id": "r", "status": "contradicted", "detail": "hook conflict", "source_hash": file_sha256(source),
                "source_refs": ["extension/eeg_candidate.py::EEGCandidate.build_encoder"], "check_ref": "checks.json",
                "check_sha256": file_sha256(check_path)}
    original = {"status": "needs_fix", "requirement_coverage": [coverage], "issues": []}
    if scope == "engineering":
        original["issues"].append({"kind": "design_conflict", "severity": "blocking", "requirement_id": "r"})
    result = native_patch.enforce_requirement_review(tmp_path, spec, original, checks=receipt)
    if scope == "scientific":
        issue = result["issues"][0]
        assert issue["kind"] == "code_error" and issue["evidence_refs"] and issue["smallest_correction"] and issue["validation_expectation"]
    else:
        assert len(result["issues"]) == 1 and result["issues"][0]["kind"] == "design_conflict"
