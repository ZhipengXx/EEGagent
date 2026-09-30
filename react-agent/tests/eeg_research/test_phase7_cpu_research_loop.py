"""Phase 7: CPU toy campaign through public job/worker paths.

Offline agent-loop routing is schema-constrained and is not live LM reasoning.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from react_agent.eeg_research.agentic.coder import finish_patch
from react_agent.eeg_research.agentic.contract import freeze_contract
from react_agent.eeg_research.agentic.execution_protocol import build_execution_protocol
from react_agent.eeg_research.agentic.experiment_gate import install_approved_experiment
from react_agent.eeg_research.agentic.export import pack_candidate, pack_is_rebuildable
from react_agent.eeg_research.agentic.loop import (
    _record_job,
    create_campaign,
    load_state,
    observation,
    rebuild_campaign_projection,
    save_state,
    tick,
)
from react_agent.eeg_research.agentic.planner import route_next_research_action
from react_agent.eeg_research.agentic.run_context import persist_approved_binding
from react_agent.eeg_training.protocol import FULL_EEG_CHANNELS, Design, feature_cache


CANDIDATE_SRC = '''
"""Non lr/wd intervention: drop_proj reaches the encoder; later eval collapses in memory."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class PlantedCollapseEncoder(nn.Module):
    def __init__(self, z_dim: int, drop_proj: float) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.plant_dim = 16
        self.drop = nn.Dropout(float(drop_proj))
        self.scale = nn.Parameter(torch.ones(self.plant_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self._train_epochs_seen = 0
        self._was_training = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        planted = inputs[:, 0, : self.plant_dim] * self.scale.abs()
        planted = self.drop(planted)
        pad = inputs.new_zeros(inputs.shape[0], self.z_dim - self.plant_dim)
        out = torch.cat([planted, pad], dim=-1)
        if self.training:
            if not self._was_training:
                self._train_epochs_seen += 1
            self._was_training = True
            return out
        self._was_training = False
        if self._train_epochs_seen >= 2:
            return torch.zeros_like(out)
        return out


class EEGCandidate:
    candidate_id = "c1"

    def build_encoder(self, input_spec: dict[str, Any], model_config: dict[str, Any] | None = None):
        config = model_config or {}
        return PlantedCollapseEncoder(
            z_dim=int(config.get("z_dim", 1024)),
            drop_proj=float(config.get("drop_proj", 0.0)),
        )

    build_encoder.accepted_config_keys = {"z_dim", "drop_proj"}

    def build_training_objective(self, objective_config: dict[str, Any] | None = None) -> Any:
        from react_agent.eeg_training.model import contrastive_loss

        if objective_config:
            raise ValueError("toy_objective_rejects_config")
        return contrastive_loss

    def build_training_transform(self, transform_config: dict[str, Any] | None = None) -> Any:
        if transform_config:
            raise ValueError("toy_transform_rejects_config")
        return None
'''


def _cpu_env() -> dict[str, str]:
    return {
        "EEG_TRAIN_DEVICE": "cpu",
        "EEG_ALLOW_CPU_TRAIN": "1",
        "EEG_TRAIN_PYTHON": sys.executable,
        "EEG_FINAL_TEST": "0",
        "EEG_DIAGNOSTIC_SAMPLE_LIMIT": "8",
        "EEG_DIAGNOSTIC_SAMPLE_SEED": "0",
    }


@pytest.fixture
def cpu_train_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _cpu_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EEG_CANDIDATE_MODULE", raising=False)
    monkeypatch.delenv("EEG_CANDIDATE_PATH", raising=False)


def write_toy_dataset(root: Path, *, n_train: int = 20, n_test: int = 4, z_dim: int = 1024, plant: int = 16) -> Design:
    import torch

    train_dir = root / "all"
    test_dir = root / "hold"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    n_ch = len(FULL_EEG_CHANNELS)
    p7 = FULL_EEG_CHANNELS.index("P7")

    def _trials(prefix: str, count: int) -> tuple[list[str], dict[str, torch.Tensor]]:
        images = [f"{prefix}-{index:02d}" for index in range(count)]
        eeg = torch.zeros(count, 1, n_ch, 250)
        features: dict[str, torch.Tensor] = {}
        for index, image in enumerate(images):
            vec = torch.zeros(z_dim)
            planted = torch.randn(plant)
            planted = planted / planted.norm().clamp_min(1e-6)
            vec[:plant] = planted
            eeg[index, 0, p7, :plant] = planted
            features[image] = vec
        return images, features, eeg

    train_images, train_features, train_eeg = _trials("img", n_train)
    test_images, test_features, test_eeg = _trials("hold", n_test)
    torch.save({"eeg": train_eeg, "img": train_images, "label": train_images}, train_dir / "train.pt")
    torch.save({"eeg": test_eeg, "img": test_images, "label": test_images}, test_dir / "test.pt")
    design = Design(
        "eeg",
        "inter-subject",
        "all",
        epochs=3,
        seed=0,
        batch_size=8,
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        data_root=str(root),
        policy="agentic",
        training_strategy="pooled_subjects",
        stop="single_full",
    )
    cache = feature_cache(root, design, "train")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"img_features": {**train_features, **test_features}}, cache)
    return design


def _prepare_campaign(tmp_path: Path) -> tuple[Path, Path, Design]:
    data_root = tmp_path / "data"
    design = write_toy_dataset(data_root)
    protocol = build_execution_protocol(design, data_root, training_seeds=[0])
    contract = freeze_contract(design, data_root)
    root = tmp_path / "root"
    create_campaign(
        root,
        goal={
            "goal_id": "toy",
            "max_training_jobs": 6,
            "max_llm_calls": 20,
            "max_gpu_seconds": 600,
            "max_candidates": 4,
            "training_seeds": [0],
            "confirmation_target_pairs": 1,
        },
        contract=contract,
        request_id="p7-cpu",
        protocol=protocol,
    )
    camp = root / "toy"
    state = load_state(camp)
    approved = install_approved_experiment(
        camp,
        state,
        {
            "intervention": "drop_proj_zero",
            "hypothesis": {"mechanism": "去掉投影 dropout，保留 planted 通道信息"},
            "parent_candidate_id": "baseline",
            "control_candidate_id": "baseline",
            "initial_fidelity": "full",
            "model": {"drop_proj": 0.0},
        },
        target_id="c1",
    )
    workspace = camp / "candidates" / "c1"
    (workspace / "extension").mkdir(parents=True, exist_ok=True)
    (workspace / "extension" / "eeg_candidate.py").write_text(CANDIDATE_SRC, encoding="utf-8")
    finished = finish_patch(workspace, "planted drop_proj")
    assert finished.get("ok") is True
    persist_approved_binding(
        camp,
        "c1",
        approved,
        spec_ref=approved.get("experiment_ref"),
        source_hash=finished["manifest"]["entry_sha256"],
    )
    state["candidate_ready"] = True
    state["candidate_id"] = "c1"
    state["candidates"] = [{"candidate_id": "c1", "status": "ready"}]
    state["experiment"] = approved
    state["experiment_ref"] = approved.get("experiment_ref")
    save_state(camp, state)
    return camp, data_root, design


def _backend(obs: dict) -> dict:
    if obs.get("schema_error"):
        return {"action": "stop", "stop_reason": "blocked", "reason_zh": "schema", "evidence_ids": []}
    actions = obs.get("available_actions") or []
    eligible = obs.get("eligible_targets") if isinstance(obs.get("eligible_targets"), dict) else {}
    for name in ("run_full", "run_pilot"):
        if name not in actions:
            continue
        targets = list(eligible.get(name) or [])
        return {
            "action": name,
            "target_id": targets[0] if targets else "c1",
            "reason_zh": "跑可执行保真并保留对照",
            "evidence_ids": [],
        }
    routed = route_next_research_action(obs)
    routed.setdefault("evidence_ids", [])
    return routed


def _services(camp: Path) -> dict:
    from react_agent.eeg_research.agentic.worker import build_services

    full = build_services(camp)
    return {"launch": full["launch"], "settle": full["settle"]}


def _job_rows(state: dict) -> list[dict]:
    return [row for row in (state.get("evidence") or []) if row.get("job_dir") and row.get("fidelity")]


def _spin_until_jobs(camp: Path, services: dict, count: int, timeout: float = 240.0, *, fidelity: str | None = "full") -> dict:
    deadline = time.time() + timeout
    last = load_state(camp)
    while time.time() < deadline:
        last = tick(camp, _backend, services=services)
        rows = _job_rows(last)
        settled = [
            row
            for row in rows
            if (fidelity is None or row.get("fidelity") == fidelity)
            and (row.get("evaluation_valid") is not None or row.get("job_status") in {"finished", "invalid", "failed"})
        ]
        if last.get("status") in {"finished", "blocked", "cancelled"} and len(settled) < count:
            raise TimeoutError(
                f"premature {last.get('status')} detail={last.get('detail')} decisions={last.get('decisions')} evidence={last.get('evidence')}"
            )
        if len(settled) >= count and not last.get("live_job"):
            return last
        time.sleep(0.2)
    logs = []
    for path in (camp / "jobs").glob("*/train.log"):
        logs.append(f"{path}:\n{path.read_text(encoding='utf-8')[-2000:]}")
    raise TimeoutError(
        f"status={last.get('status')} live={last.get('live_job')} evidence={last.get('evidence')} decisions={last.get('decisions')}\n"
        + "\n".join(logs)
    )


def test_diagnostic_routing_two_scenes() -> None:
    repair = route_next_research_action(
        {
            "available_actions": ["repair_candidate", "run_full", "stop"],
            "eligible_targets": {"run_full": ["c1"]},
            "latest_diagnostics": {"items": {"hook_consumption": {"execution_status": "not_applied", "mismatches": ["model.drop_proj"]}}},
            "latest_comparison": {},
        }
    )
    assert repair["action"] == "repair_candidate"
    assert repair["observed_gap"] == "hook_not_consumed"
    measure = route_next_research_action(
        {
            "available_actions": ["run_full", "replicate", "collect_diagnostics", "stop"],
            "eligible_targets": {"replicate": ["c1"], "run_full": ["c1"]},
            "latest_diagnostics": {"items": {"hook_consumption": {"execution_status": "applied", "mismatches": []}}},
            "latest_comparison": {"comparable": True, "delta_pp": 0.0},
        }
    )
    assert measure["action"] in {"replicate", "run_full", "collect_diagnostics"}
    assert measure["observed_gap"] == "mechanism_unresolved"
    assert measure["action"] != "repair_candidate"


def test_cpu_fit_best_checkpoint_not_last_epoch(tmp_path: Path, cpu_train_env: None) -> None:
    from react_agent.eeg_research.agentic.hook_config import write_hook_config as write_hooks
    from react_agent.eeg_training.train_entry import fit

    data_root = tmp_path / "data"
    design = write_toy_dataset(data_root)
    out = tmp_path / "job"
    out.mkdir()
    write_hooks(out, {"model": {"drop_proj": 0.0}, "objective": {}, "transform": {}}, spec_ref="spec", spec_hash="hash1")
    os.environ["EEG_CANDIDATE_MODULE"] = "eeg_candidate"
    os.environ["EEG_CANDIDATE_PATH"] = str(tmp_path / "ext")
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "eeg_candidate.py").write_text(CANDIDATE_SRC, encoding="utf-8")
    try:
        metrics = fit(design, data_root, out)
    finally:
        os.environ.pop("EEG_CANDIDATE_MODULE", None)
        os.environ.pop("EEG_CANDIDATE_PATH", None)
    selected = json.loads((out / "selected_checkpoint.json").read_text(encoding="utf-8"))
    assert selected["best_epoch"] != selected["last_epoch"]
    assert selected["best_fixed_bank_top1"] == metrics["fixed_bank_top1"]
    assert float(selected["best_fixed_bank_top1"]) > float(selected["last_epoch_fixed_bank_top1"] or 0.0)
    consumed = json.loads((out / "hook_consumed.json").read_text(encoding="utf-8"))
    assert consumed["model"]["applied"] is True
    assert consumed["model"]["config"]["drop_proj"] == 0.0
    embeddings = json.loads((out / "embeddings.json").read_text(encoding="utf-8"))
    assert embeddings["source"] == "selected_checkpoint"
    queries = (out / "retrieval_queries.jsonl").read_text(encoding="utf-8")
    assert "all::" in queries or "img-" in queries
    history = [json.loads(line) for line in (out / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(float(row.get("train_loss") or 0.0) != 0.0 for row in history)


def test_phase7_campaign_jobs_recovery_and_pack(tmp_path: Path, cpu_train_env: None) -> None:
    camp, data_root, _design = _prepare_campaign(tmp_path)
    services = _services(camp)
    state = _spin_until_jobs(camp, services, 2)
    rows = _job_rows(state)
    names = {row.get("candidate_id") for row in rows}
    assert "baseline" in names
    assert "c1" in names
    valid = [row for row in rows if row.get("evaluation_valid") is True]
    assert valid, f"no valid jobs: {rows}"
    candidate = next(row for row in rows if row.get("candidate_id") == "c1")
    baseline = next(row for row in rows if row.get("candidate_id") == "baseline")
    cand_dir = Path(str(candidate["job_dir"]))
    base_dir = Path(str(baseline["job_dir"]))
    assert (cand_dir / "last.ckpt").is_file()
    assert (cand_dir / "metrics.json").is_file()
    consumed = json.loads((cand_dir / "hook_consumed.json").read_text(encoding="utf-8"))
    assert consumed["model"]["config"]["drop_proj"] == 0.0
    selected = json.loads((cand_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
    assert selected["best_epoch"] != selected["last_epoch"]
    cand_metrics = json.loads((cand_dir / "metrics.json").read_text(encoding="utf-8"))
    assert cand_metrics["fixed_bank_top1"] == selected["best_fixed_bank_top1"]
    assert cand_metrics.get("test_result") in (None, {})
    assert (base_dir / "hook_consumed.json").is_file()
    base_hook = json.loads((base_dir / "hook_config.json").read_text(encoding="utf-8"))
    assert (base_hook.get("model") or {}) == {}
    assert candidate.get("comparison") or (camp / "comparisons").exists()
    jobs_before = int(state.get("training_jobs") or 0)
    episodes_before = len(state.get("memory") or [])
    cost_before = json.loads((camp / "cost.json").read_text(encoding="utf-8")) if (camp / "cost.json").is_file() else {}
    wiped = load_state(camp)
    wiped["evidence"] = []
    wiped["memory"] = []
    save_state(camp, wiped)
    fresh = load_state(camp)
    rebuild_campaign_projection(camp, fresh)
    rebuilt = _job_rows(fresh)
    assert {row.get("candidate_id") for row in rebuilt} >= {"baseline", "c1"}
    assert len(fresh.get("memory") or []) >= episodes_before
    _record_job(
        camp,
        fresh,
        {
            "job_id": Path(str(candidate["job_id"])).name if candidate.get("job_id") else cand_dir.name,
            "candidate_id": "c1",
            "seed": candidate.get("seed"),
            "status": "finished",
            "gpu_seconds": 0,
            "result": dict(candidate),
        },
    )
    assert int(fresh.get("training_jobs") or 0) >= jobs_before
    if (camp / "cost.json").is_file() and cost_before:
        cost_after = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
        assert cost_after.get("training_jobs") == cost_before.get("training_jobs")
    save_state(camp, fresh)
    obs = observation(camp)
    nxt = route_next_research_action(obs)
    assert nxt["action"] in {"repair_candidate", "replicate", "run_full", "collect_diagnostics", "diagnose_results", "stop"}
    pack = tmp_path / "pack"
    manifest = pack_candidate(cand_dir, pack, workspace=camp / "candidates" / "c1")
    assert manifest["content_hashes"]
    ok, reason = pack_is_rebuildable(pack)
    assert ok is True, reason
    original_score = float(cand_metrics["fixed_bank_top1"])
    shutil.rmtree(camp)
    assert not cand_dir.exists()
    other = tmp_path / "other_cwd"
    other.mkdir()
    env = os.environ.copy()
    env.update(_cpu_env())
    env["EEG_ALLOW_EVALUATE_PACK"] = "1"
    env["EEG_EVALUATE_PACK_CWD"] = str(other)
    env.pop("EEG_CANDIDATE_MODULE", None)
    env.pop("EEG_CANDIDATE_PATH", None)
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (src, env.get("PYTHONPATH", "")) if part)
    out = pack / "eval_only"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "react_agent.eeg_research.agentic.cli",
            "evaluate-pack",
            "--pack",
            str(pack),
            "--out",
            str(out),
            "--data-root",
            str(data_root),
            "--execute",
        ],
        check=False,
        cwd=str(other),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    scores = json.loads((out / "eval_scores.json").read_text(encoding="utf-8"))
    assert scores.get("evaluate_only") is True
    assert scores.get("test_result") is None
    assert abs(float(scores["fixed_bank_top1"]) - original_score) <= 1e-5
    (pack / "extension" / "eeg_candidate.py").unlink()
    broken = subprocess.run(
        [
            sys.executable,
            "-m",
            "react_agent.eeg_research.agentic.cli",
            "evaluate-pack",
            "--pack",
            str(pack),
            "--out",
            str(tmp_path / "broken_eval"),
            "--data-root",
            str(data_root),
            "--execute",
        ],
        check=False,
        cwd=str(other),
        env=env,
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert "baseline" not in (broken.stdout + broken.stderr).lower() or "error" in (broken.stdout + broken.stderr).lower()


def write_standard_pool(root: Path) -> None:
    import torch

    n_ch = len(FULL_EEG_CHANNELS)
    p7 = FULL_EEG_CHANNELS.index("P7")
    features: dict[str, object] = {}
    for subject, offset in (("sub-01", 0), ("sub-02", 10)):
        dest = root / "things-eeg" / "Preprocessed_data_250Hz_whiten" / subject
        dest.mkdir(parents=True, exist_ok=True)
        images = [f"img-{offset + index:02d}" for index in range(10)]
        eeg = torch.zeros(10, 1, n_ch, 250)
        for index, image in enumerate(images):
            vec = torch.zeros(1024)
            planted = torch.randn(16)
            planted = planted / planted.norm().clamp_min(1e-6)
            vec[:16] = planted
            eeg[index, 0, p7, :16] = planted
            features[image] = vec
        torch.save({"eeg": eeg, "img": images, "label": images}, dest / "train.pt")
        hold = [f"hold-{subject}-{index}" for index in range(2)]
        teeg = torch.zeros(2, 1, n_ch, 250)
        for image in hold:
            features[image] = torch.zeros(1024)
        torch.save({"eeg": teeg, "img": hold, "label": hold}, dest / "test.pt")
    design = Design("eeg", "inter-subject", "all", policy="agentic", training_strategy="pooled_subjects")
    cache = feature_cache(root, design, "train")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"img_features": features}, cache)


def test_t14_cli_create_freezes_goal_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from react_agent.eeg_research.agentic.cli import main
    from react_agent.eeg_research.agentic.confirmation_policy import load_confirmation_policy
    from react_agent.eeg_research.agentic.execution_protocol import next_unused_training_seed
    from react_agent.eeg_research.agentic.planner import eligible_targets

    data_root = tmp_path / "data"
    write_standard_pool(data_root)
    monkeypatch.setenv("EEG_DATA_ROOT", str(data_root))
    goal_path = tmp_path / "goal.yaml"
    goal_path.write_text(
        "\n".join(
            [
                "goal_id: cli_seeds",
                "max_training_jobs: 8",
                "max_llm_calls: 20",
                "max_gpu_seconds: 60",
                "confirmation_target_pairs: 3",
                "training_seeds: [0, 1, 2]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "root"
    code = main(["create", "--campaign", "cli_seeds", "--root", str(root), "--goal", str(goal_path), "--request-id", "cli-t14"])
    assert code == 0
    camp = root / "cli_seeds"
    policy = load_confirmation_policy(camp)
    assert policy is not None
    assert policy["training_seeds"] == [0, 1, 2]
    proto = json.loads((camp / "execution_protocol.json").read_text(encoding="utf-8"))
    assert proto["training_seeds"] == [0, 1, 2]
    state = load_state(camp)
    assert state["training_seeds"] == [0, 1, 2]
    state["candidate_ready"] = True
    state["candidate_id"] = "c1"
    state["candidates"] = [{"candidate_id": "c1", "status": "ready"}]
    state["evidence"] = [
        {"candidate_id": "c1", "fidelity": "pilot", "evaluation_valid": True, "seed": 0},
        {"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 0},
    ]
    state["training_jobs"] = 1
    assert "c1" in eligible_targets(state)["replicate"]
    assert next_unused_training_seed(proto, {0}) == 1
    assert next_unused_training_seed(proto, {0, 1, 2}) is None
    state["evidence"].extend(
        [
            {"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 1},
            {"candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "seed": 2},
        ]
    )
    assert eligible_targets(state)["replicate"] == []
