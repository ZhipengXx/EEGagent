"""Public CLI creation/control fixtures: local files, no provider or GPU process."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic import budget_inheritance as budget, cli, loop, loso_study, method_suite
from react_agent.eeg_research.agentic.loso_study import SUBJECTS
from react_agent.eeg_training.protocol import Design, data_dir, feature_cache


class Crash(BaseException):
    pass


def _local_protocol(monkeypatch):
    from react_agent.eeg_research.agentic import execution_protocol
    def build(design, base, **kwargs):
        return {"schema_version": "isolated_cli_protocol_fixture", "fingerprint": "local_" + design.subject,
                "training_seeds": kwargs.get("training_seeds") or [0], "split_seed": 0,
                "gpu": list(design.gpu), "isolated_fixture": True}
    monkeypatch.setattr(cli, "build_execution_protocol", build)
    monkeypatch.setattr(execution_protocol, "build_execution_protocol", build)
    monkeypatch.setattr(cli, "freeze_contract", lambda *_: {"fingerprint": "isolated_cli_contract"})
    monkeypatch.setattr(loso_study, "gpu_inventory", lambda cards, **_: [
        {"index": card, "uuid": "isolated_" + str(card), "name": "synthetic", "total_memory_mib": 24000}
        for card in cards])


def _history(tmp_path):
    """A native validated v1 allocation plus settled method-campaign increment."""
    old = tmp_path / "legacy_study"
    loop._write(old / "study_manifest.json", {"isolated_fixture": True})
    receipt = {"schema_version": "eeg_research.budget_inheritance.v1", "source_study": str(old),
        "authorization_caps": loso_study.CAPS, "historical_usage": {"llm_calls": 10, "training_jobs": 2, "gpu_seconds": 1000.0},
        "remaining": {"llm_calls": 390, "training_jobs": 46, "gpu_seconds": 171800.0},
        "dependencies": [{"path": str(old / "study_manifest.json"), "sha256": loso_study.digest(old / "study_manifest.json")}],
        "accounting": "historical_reference_plus_successor_incremental"}
    receipt["receipt_hash"] = loso_study.identity(receipt)
    source = tmp_path / "settled_method"
    loop._write(source / "budget_inheritance.json", receipt)
    loop._write(source / "goal.json", {"evaluation_mode": "loso_method_search", "historical_budget_receipt_hash": receipt["receipt_hash"]})
    loop._write(source / "method_search_manifest.json", {"isolated_fixture": True})
    loop._write(source / "execution_protocol.json", {"isolated_fixture": True})
    loop._write(source / "campaign_state.json", {"status": "cancelled", "live_job": None, "llm_calls": 2,
        "training_jobs": 1, "gpu_seconds_reserved": 0, "candidates": []})
    loop._write(source / "cost.json", {"llm_calls": 2, "training_jobs": 1, "gpu_seconds_used": 400,
        "settled_job_ids": ["j1"], "api_uncertain_calls": 0, "api_unsettled_call_ids": []})
    for name in ("llm_calls.jsonl", "llm_api_intents.jsonl"):
        (source / name).write_text('{"call_id":"a"}\n{"call_id":"b"}\n')
    loop._write(source / "jobs/j1/job.json", {"job_id": "j1", "status": "finished", "pid": 0,
        "fidelity": "full", "gpu": [0, 1], "gpu_seconds": 400})
    loop._write(source / "jobs/j1/metrics.json", {"last_epoch": 50})
    return source


def _fresh_args(tmp_path, monkeypatch):
    _local_protocol(monkeypatch)
    source = _history(tmp_path)
    data = tmp_path / "data"
    for subject in SUBJECTS:
        folder = data_dir(data, "eeg") / subject
        folder.mkdir(parents=True)
        for name in ("train.pt", "test.pt"):
            (folder / name).write_bytes(b"isolated identity bytes; never decoded")
    design = Design("eeg", "inter-subject", ",".join(SUBJECTS[:-1]), gpu=(2, 7), data_root=str(data))
    for mode in ("train", "test"):
        cache = feature_cache(data, design, mode)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"isolated feature identity; never decoded")
    goal = {"goal_id": "fresh", "evaluation_mode": "loso_method_search", "benchmark_feedback_enabled": True,
        "primary_metric": "benchmark.loso_mean_fixed_gallery_top1", "research_scope": "inter_subject_development_benchmark",
        "subject_aggregation": "macro_mean", "benchmark_subjects": list(SUBJECTS), "required_fold_count": 10,
        "checkpoint_policy": "strict_best", "training_seeds": [0], "confirmation_target_pairs": 1,
        "allowed_training_actions": ["run_pilot", "run_full"],
        "max_llm_calls": 30, "max_training_jobs": 40, "max_gpu_seconds": 100000.0}
    goal_path = tmp_path / "goal.json"
    goal_path.write_text(json.dumps(goal))
    root = tmp_path / "campaigns"
    args = ["create", "--root", str(root), "--campaign", "fresh", "--gpu", "2,7", "--data-root", str(data),
        "--goal", str(goal_path), "--fresh-budget-authorization", "--budget-history-source", str(source),
        "--request-id", "explicit_fresh_grant"]
    return args, root / "fresh", source


def test_default_cli_goal_really_selects_single_plan():
    assert cli.goal("new")["research_execution"]["version"] == "single_plan_v1"


def test_fresh_cli_goal_write_crash_retries_same_grant(tmp_path, monkeypatch, capsys):
    args, camp, source = _fresh_args(tmp_path, monkeypatch)
    original_write = loop._write
    def write(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if path == camp / "goal.json":
            raise Crash("goal persisted")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with monkeypatch.context() as injected:
        injected.setattr(loop, "_write", write)
        with pytest.raises(Crash):
            cli.main(args)
    assert cli.main(args) == 0, capsys.readouterr().out
    assert budget.validate_inheritance(camp)["new_allocation_caps"]["llm_calls"] == 30
    assert method_suite.manifest(camp)["required_fold_count"] == 10
    assert all(path.read_bytes() == body for path, body in before.items())


@pytest.mark.parametrize("command", ["start", "resume", "run"])
def test_cancelled_cli_entry_rejects_before_writes_or_spawn(tmp_path, monkeypatch, capsys, command):
    loop.create_campaign(tmp_path, goal={"goal_id": "cancelled"}, contract={"fingerprint": "isolated"}, request_id="cancelled")
    camp = tmp_path / "cancelled"
    state = loop.load_state(camp)
    state["status"] = "cancelled"
    loop.save_state(camp, state)
    frozen = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()}
    monkeypatch.setattr(cli, "spawn_worker", lambda *_a, **_k: pytest.fail("cancelled spawn"))
    assert cli.main([command, "--root", str(tmp_path), "--campaign", camp.name]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "campaign_cancelled"
    assert {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()} == frozen


@pytest.mark.parametrize("file_version,option,expected", [
    (None, None, "single_plan_v1"), (None, "legacy", "legacy"),
    ("old_goal", None, "legacy"), ("old_goal", "single_plan_v1", "single_plan_v1"),
    ("single_plan_v1", None, "single_plan_v1"), ("single_plan_v1", "legacy", "legacy"),
])
def test_cli_create_protocol_selection_preserves_old_goal_semantics(tmp_path, monkeypatch, capsys, file_version, option, expected):
    _local_protocol(monkeypatch)
    args = ["create", "--root", str(tmp_path / "campaigns"), "--campaign", "protocol", "--data-root", str(tmp_path)]
    if file_version:
        submitted = {"goal_id": "template", "max_llm_calls": 10, "max_training_jobs": 4,
            "max_gpu_seconds": 1000, "training_seeds": [0], "confirmation_target_pairs": 1}
        if file_version == "single_plan_v1":
            submitted["research_execution"] = {"version": "single_plan_v1"}
        path = tmp_path / "template.json"
        path.write_text(json.dumps(submitted))
        args += ["--goal", str(path)]
    if option:
        args += ["--execution-version", option]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution_version"] == expected
    view = cli.status_view(tmp_path / "campaigns/protocol")
    assert view["execution_version"] == expected
    assert view["paid_maintenance"] == ("off" if expected == "single_plan_v1" else "legacy")
    persisted = json.loads((tmp_path / "campaigns/protocol/goal.json").read_text())
    if expected == "single_plan_v1":
        configured = persisted["research_execution"]["budget"]["envelopes"]["new_method"]
        assert configured["training_jobs"] == 2
        assert configured["gpu_seconds"] == 2 * persisted["max_gpu_seconds"] / persisted["max_training_jobs"]
        assert configured["authority"] == "maximum_runtime_allocation_not_cost_estimate"


CREATE_WINDOWS = ["cli_prepare", "campaign_prepare", "goal", "resolved_goal", "contract", "policy", "state",
    "protocol", "plan", "plan_version", "global_grant", "campaign_grant", "inheritance", "role_overrides", "manifest", "cost_basis", "commit", "index", "event"]


def _crash_creation_after(camp, monkeypatch, window):
    from react_agent.eeg_research.agentic import research_plan
    original_write, original_plan_write, original_claim_write, original_event = loop._write, research_plan._write, budget.write_once, loop.event
    files = {"campaign_prepare": "campaign_creation.prepare.json", "goal": "goal.json", "resolved_goal": "resolved_goal.json",
        "contract": "evaluation_contract.json", "policy": "confirmation_policy.json", "state": "campaign_state.json",
        "protocol": "execution_protocol.json", "plan": "plan.json", "plan_version": "v1.json",
        "campaign_grant": "fresh_budget_authorization.json", "inheritance": "budget_inheritance.json",
        "manifest": "method_search_manifest.json", "cost_basis": "suite_cost_basis.json", "commit": "campaign_creation.commit.json"}
    files["role_overrides"] = "role_config_overrides.json"
    def matched(path):
        return ((window in files and path.is_relative_to(camp) and path.name == files[window])
            or (window == "cli_prepare" and path.parent == camp.parent / "creation_requests")
            or (window == "global_grant" and path.parent == camp.parent / "fresh_budget_authorizations")
            or (window == "index" and path == camp.parent / "request_index.json"))
    def write(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if matched(path): raise Crash(window)
    def plan_write(path, payload):
        original_plan_write(path, payload)
        if matched(path): raise Crash(window)
    def claim_write(path, payload):
        original_claim_write(path, payload)
        if matched(path): raise Crash(window)
    def event(_camp, kind, **fields):
        original_event(_camp, kind, **fields)
        if window == "event" and _camp == camp and kind == "created": raise Crash(window)
    monkeypatch.setattr(loop, "_write", write)
    monkeypatch.setattr(research_plan, "_write", plan_write)
    monkeypatch.setattr(budget, "write_once", claim_write)
    monkeypatch.setattr(loop, "event", event)


@pytest.mark.parametrize("window", CREATE_WINDOWS)
def test_fresh_cli_restart_at_every_persistent_publication(tmp_path, monkeypatch, capsys, window):
    args, camp, source = _fresh_args(tmp_path, monkeypatch)
    args += ["--execution-version", "single_plan_v1", "--role-override", "candidate_coder.max_tokens=128"]
    old = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with monkeypatch.context() as injected:
        _crash_creation_after(camp, injected, window)
        with pytest.raises(Crash, match=window):
            cli.main(args)
    frozen = {path: path.read_bytes() for path in camp.parent.rglob("*") if path.is_file()}
    assert cli.main(args) == 0, capsys.readouterr().out
    first = json.loads(capsys.readouterr().out)
    assert first["execution_version"] == "single_plan_v1" and first["paid_maintenance"] == "off"
    assert cli.main(args) == 0
    capsys.readouterr()
    assert all(path.read_bytes() == body for path, body in frozen.items())
    assert all(path.read_bytes() == body for path, body in old.items())
    state = loop.load_state(camp)
    assert state["llm_calls"] == state["training_jobs"] == 0 and state["candidates"] == []
    assert (state["max_llm_calls"], state["max_training_jobs"], state["max_gpu_seconds"]) == (30, 40, 100000)
    assert budget.validate_inheritance(camp)["historical_usage"] == {"llm_calls": 12, "training_jobs": 3, "gpu_seconds": 1400}
    assert len(list((camp.parent / "fresh_budget_authorizations").glob("*.json"))) == 1
    assert not (camp / "cost.json").exists() and not (camp / "jobs").exists()
    manifest = method_suite.manifest(camp)
    assert manifest["required_fold_count"] == 10 and manifest["research_execution_version"] == "single_plan_v1"
    assert json.loads((camp.parent / "request_index.json").read_text())["explicit_fresh_grant"] == str(camp)


@pytest.mark.parametrize("change", ["caps", "candidate_scope", "gpu", "goal_id", "version"])
def test_frozen_creation_request_rejects_changed_configuration_without_new_grant(tmp_path, monkeypatch, capsys, change):
    args, camp, _ = _fresh_args(tmp_path, monkeypatch)
    with monkeypatch.context() as injected:
        _crash_creation_after(camp, injected, "campaign_grant")
        with pytest.raises(Crash): cli.main(args)
    if change in {"caps", "candidate_scope"}:
        path = Path(args[args.index("--goal") + 1])
        payload = json.loads(path.read_text())
        payload["max_llm_calls" if change == "caps" else "max_candidates"] = 15
        path.write_text(json.dumps(payload))
    elif change == "gpu": args[args.index("--gpu") + 1] = "3,6"
    elif change == "goal_id": args[args.index("--campaign") + 1] = "new_identity"
    else: args += ["--execution-version", "single_plan_v1"]
    frozen = {path: path.read_bytes() for path in camp.parent.rglob("*") if path.is_file()}
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "campaign_creation_configuration_conflict"
    assert {path: path.read_bytes() for path in camp.parent.rglob("*") if path.is_file()} == frozen


def test_partial_campaign_cannot_start_before_creation_commit(tmp_path, monkeypatch, capsys):
    args, camp, _ = _fresh_args(tmp_path, monkeypatch)
    with monkeypatch.context() as injected:
        _crash_creation_after(camp, injected, "state")
        with pytest.raises(Crash): cli.main(args)
    frozen = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()}
    monkeypatch.setattr(cli, "spawn_worker", lambda *_a, **_k: pytest.fail("uncommitted spawn"))
    assert cli.main(["start", "--root", str(camp.parent), "--campaign", camp.name]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "campaign_creation_not_committed"
    assert {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()} == frozen


def test_persisted_stop_intent_cannot_be_overwritten_by_resume(tmp_path, monkeypatch, capsys):
    loop.create_campaign(tmp_path, goal={"goal_id": "stopped"}, contract={"fingerprint": "isolated"}, request_id="stopped")
    camp = tmp_path / "stopped"
    loop.request_control(camp, "stop")
    frozen = {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()}
    monkeypatch.setattr(cli, "spawn_worker", lambda *_a, **_k: pytest.fail("stop intent spawn"))
    assert cli.main(["resume", "--root", str(tmp_path), "--campaign", camp.name]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "campaign_cancelled"
    assert {path: path.read_bytes() for path in camp.rglob("*") if path.is_file()} == frozen


def test_ui_repeated_cancelled_request_and_spawn_helper_are_read_only(tmp_path, monkeypatch):
    from hashlib import sha256
    request = "closed-ui-click"
    campaign = "eeg_ui_" + sha256(request.encode()).hexdigest()[:12]
    loop.create_campaign(tmp_path, goal={"goal_id": campaign}, contract={"fingerprint": "isolated"}, request_id=request)
    camp = tmp_path / campaign
    state = loop.load_state(camp)
    state["status"] = "cancelled"
    loop.save_state(camp, state)
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    def forbid(*_a, **_k): pytest.fail("closed request protocol construction or spawn")
    monkeypatch.setattr(cli, "build_execution_protocol", forbid)
    result = cli.open_agentic_run(Design("eeg", "inter-subject", "all"), {"ok": True, "blockers": []},
        request_id=request, root=tmp_path, spawn=forbid)
    assert result["started"] is False and result["error"] == "campaign_cancelled"
    with pytest.raises(ValueError, match="campaign_cancelled"):
        cli.spawn_worker(tmp_path, campaign, 30)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == frozen


def test_explicit_new_protocol_on_zero_training_old_goal_still_admits_diagnosis(tmp_path, monkeypatch, capsys):
    from react_agent.eeg_research.agentic import owner_budget
    _local_protocol(monkeypatch)
    submitted = {"goal_id": "old_template", "max_llm_calls": 20, "max_training_jobs": 0,
        "max_gpu_seconds": 0, "training_seeds": [0], "confirmation_target_pairs": 1}
    path = tmp_path / "zero_training_goal.json"
    path.write_text(json.dumps(submitted))
    assert cli.main(["create", "--root", str(tmp_path), "--campaign", "diagnosis", "--goal", str(path),
        "--data-root", str(tmp_path), "--execution-version", "single_plan_v1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution_version"] == "single_plan_v1" and result["max_training_jobs"] == result["max_gpu_seconds"] == 0
    camp = tmp_path / "diagnosis"
    frozen_goal = json.loads((camp / "goal.json").read_text())
    assert not frozen_goal["research_execution"]["budget"].get("envelopes")
    limits = {"max_engineering_repairs": 2, "coder_steps": 8}
    preview = owner_budget.preview_selected_plan(camp, task_kind="diagnosis",
        known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits=limits)
    assert preview["ok"], preview
    owner = owner_budget.allocate_selected_plan(camp, plan_id="plan_" + "d" * 32, task_kind="diagnosis",
        known_cost={"training_jobs": 0, "gpu_seconds": 0}, recovery_limits=limits)
    assert owner["allocated"]["training_jobs"] == owner["allocated"]["gpu_seconds"] == 0
    denied = owner_budget.preview_selected_plan(camp, task_kind="new_method",
        known_cost={"training_jobs": None, "gpu_seconds": None}, recovery_limits=limits)
    assert not denied["ok"] and "measurement_cost_unknown" in denied["reason"]
