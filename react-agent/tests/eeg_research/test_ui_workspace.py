"""Workbench projection: ranking, timeline, job history, identity, and GET side effects."""

from __future__ import annotations

import json
import os
from pathlib import Path

from react_agent.eeg_research.agentic.job_metrics import job_metrics
from react_agent.eeg_research.agentic.jsonl_read import read_jsonl
from react_agent.eeg_research.agentic.llm import _ledger
from react_agent.eeg_research.agentic.timeline import campaign_timeline
from react_agent.eeg_research.agentic.ui_demo import DEMO_CAMPAIGN, write_demo_campaign
from react_agent.eeg_research.agentic.ui_events import append_ui_event
from react_agent.eeg_research.agentic.view import campaign_view, rank_full_rows


def _snap(camp: Path) -> dict[str, tuple[int, bytes]]:
    rows = {}
    for name in ("campaign_state.json", "cost.json", "worker.json"):
        path = camp / name
        if path.is_file():
            rows[name] = (path.stat().st_mtime_ns, path.read_bytes())
    return rows


def test_zero_delta_ranks_above_negative_full(tmp_path: Path) -> None:
    camp = write_demo_campaign(tmp_path)
    view = campaign_view(camp)
    assert view["best_full"]["evidence_id"] == "ev_zero"
    assert view["best_full"]["delta_vs_control_pp"] == 0
    assert view["best_full"]["beats_control"] is False
    assert view["best_full"]["rank_note"] == "尚未优于对照"
    assert view["best_full"]["research_success"] is False
    ranked = rank_full_rows(view["full_rows"])
    assert [row["evidence_id"] for row in ranked] == ["ev_zero", "ev_neg"]


def test_rank_filters_none_invalid_pilot_baseline_and_nonfinite() -> None:
    rows = [
        {"evidence_id": "none", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": None},
        {"evidence_id": "nan", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": float("nan")},
        {"evidence_id": "inf", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": float("inf")},
        {"evidence_id": "pilot", "candidate_id": "c1", "fidelity": "pilot", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": 9},
        {"evidence_id": "base", "candidate_id": "baseline", "fidelity": "full", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": 4},
        {"evidence_id": "bad", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": False, "comparable": True, "delta_vs_control_pp": 8},
        {"evidence_id": "incomp", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "comparable": False, "delta_vs_control_pp": 7},
        {"evidence_id": "zero", "candidate_id": "c1", "fidelity": "full", "evaluation_valid": True, "comparable": True, "delta_vs_control_pp": 0},
    ]
    ranked = rank_full_rows(rows)
    assert [row["evidence_id"] for row in ranked] == ["zero"]


def test_timeline_pages_past_eight_and_keeps_identities(tmp_path: Path) -> None:
    camp = write_demo_campaign(tmp_path)
    first = campaign_timeline(tmp_path, DEMO_CAMPAIGN, cursor="0", limit=8)
    assert first["ok"]
    assert first["has_more"] is True
    assert len(first["events"]) == 8
    second = campaign_timeline(tmp_path, DEMO_CAMPAIGN, cursor=first["next_cursor"], limit=8)
    ids = [row["event_id"] for row in first["events"] + second["events"]]
    assert len(ids) == len(set(ids))
    assert first["total"] > 8
    other = campaign_timeline(tmp_path, "other_campaign")
    assert other["ok"]
    assert all(row.get("campaign_id") != DEMO_CAMPAIGN for row in other["events"] if row.get("campaign_id"))
    assert all("secret_job" not in str(row.get("job_id") or "") for row in first["events"])


def test_history_survives_cleared_live_job_and_preserves_zero_missing_nonfinite(tmp_path: Path) -> None:
    camp = write_demo_campaign(tmp_path)
    job = camp / "jobs" / "j_pilot"
    with (job / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": 4, "train_loss": float("nan"), "val_top1": 0.1, "val_top5": 0.2}) + "\n")
        handle.write(json.dumps({"epoch": 5, "train_loss": None, "val_top1": 0.1, "val_top5": 0.2, "fixed_bank_top1": None}) + "\n")
        handle.write('{"epoch":')
    metrics = job_metrics(tmp_path, DEMO_CAMPAIGN, "j_pilot")
    assert metrics["ok"]
    assert metrics["job_id"] == "j_pilot"
    by_epoch = {row["epoch"]: row for row in metrics["history"]}
    assert by_epoch[2]["train_loss"]["status"] == "ok"
    assert by_epoch[2]["train_loss"]["value"] == 0.0
    assert by_epoch[3]["fixed_bank_top1"]["status"] == "missing"
    assert by_epoch[4]["train_loss"]["status"] == "non_finite"
    assert any(item.get("kind") == "incomplete_tail" for item in metrics["history_diagnostics"])
    stolen = job_metrics(tmp_path, DEMO_CAMPAIGN, "secret_job")
    assert stolen["ok"] is False
    traversal = job_metrics(tmp_path, DEMO_CAMPAIGN, "../other_campaign/jobs/secret_job")
    assert traversal["ok"] is False
    other = job_metrics(tmp_path, "other_campaign", "j_pilot")
    assert other["ok"] is False


def test_ui_events_do_not_charge_the_cost_ledger(tmp_path: Path) -> None:
    camp = write_demo_campaign(tmp_path)
    before = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
    append_ui_event(camp, "llm_call_started", role="research_planner", call_id="c_new", status="active")
    after_ui = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
    assert after_ui["llm_calls"] == before["llm_calls"]
    _ledger(camp, {"success": True, "call_id": "charged", "role": "research_planner"})
    charged = json.loads((camp / "cost.json").read_text(encoding="utf-8"))
    assert charged["llm_calls"] == before["llm_calls"] + 1


def test_sleeping_worker_stays_alive_and_get_is_read_only(tmp_path: Path, monkeypatch) -> None:
    camp = write_demo_campaign(tmp_path)
    (camp / "worker.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": 0}), encoding="utf-8")
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs._process_state", lambda _pid: "S")
    before = _snap(camp)
    view = campaign_view(camp)
    assert view["status"] != "interrupted"
    assert view["health"]["process_state"] == "S"
    assert view["health"]["process_liveness"] == "alive"
    campaign_timeline(tmp_path, DEMO_CAMPAIGN, limit=5)
    job_metrics(tmp_path, DEMO_CAMPAIGN, "j_pilot")
    after = _snap(camp)
    assert after == before
    monkeypatch.setattr("react_agent.eeg_research.agentic.jobs._process_state", lambda _pid: "Z")
    zombie = campaign_view(camp)
    assert zombie["status"] == "interrupted"
    assert json.loads((camp / "campaign_state.json").read_text(encoding="utf-8"))["status"] == "analyzing"


def test_jsonl_reader_keeps_bad_lines_as_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": true}\nnot-json\n{"ok": false}\n', encoding="utf-8")
    payload = read_jsonl(path)
    assert len(payload["rows"]) == 2
    assert any(item.get("kind") == "bad_line" for item in payload["diagnostics"])


def test_candidate_source_preview_flag(tmp_path: Path) -> None:
    from react_agent.eeg_research.agentic.view import candidate_source

    write_demo_campaign(tmp_path)
    preview = candidate_source(tmp_path, DEMO_CAMPAIGN, "c01", preview=True)
    full = candidate_source(tmp_path, DEMO_CAMPAIGN, "c01", preview=False)
    assert preview["ok"] and full["ok"]
    assert preview["source_chars"] == full["source_chars"]
    missing = candidate_source(tmp_path, DEMO_CAMPAIGN, "../other_campaign")
    assert missing["ok"] is False
