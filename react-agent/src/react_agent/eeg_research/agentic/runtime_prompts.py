"""Versioned, task-relevant missions for single-plan execution.

Legacy files stay intact for exact paid-request recovery. Mechanical contracts
are enforced by native validators; only active implementation cases are loaded.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).with_name("prompts") / "runtime_v1"
PROMPT_VERSION = "eeg_research.runtime_mission.single_plan.v1"


def runtime_mission(role: str, *, design_contract_version: str = "v1", review_stage: str | None = None) -> str:
    root = ROOT
    if design_contract_version == "v2" and (role in {"experiment_designer", "candidate_coder", "candidate_reviewer"}
            or role == "result_analyst" and review_stage == "analysis"):
        root = ROOT.parent / "runtime_v2"
    return (root / "shared_contract.txt").read_text(encoding="utf-8") + "\n\n" + (root / (role + ".txt")).read_text(encoding="utf-8")


def conditional_hints(role: str, payload: dict) -> dict | None:
    if role not in {"candidate_coder", "candidate_reviewer", "experiment_designer", "result_analyst"}:
        return None
    # Examine the current task/config/check, not unrelated history or RAG.
    current = {key: payload[key] for key in ("approved_spec", "experiment", "experiment_spec", "design_brief", "hypothesis", "task", "selected_research_task", "repair_task", "last_check", "check", "recovery_context", "current_feedback", "objective_semantics_probe", "objective_semantic_findings", "probe") if key in payload}
    text = json.dumps(current, ensure_ascii=False, default=str).lower()
    matchers = {
        "positive_mass_infonce": ("positive-mass", "positive_mass", "multi_positive", "logsumexp"),
        "tied_image_ce": ("tied-image", "tied_image", "shared_image_classifier"),
        "objective_disabled": ("objective-off", "objective_off", "objective_disabled"),
        "ordered_pooling": ("ordered_window_slots", "ordered temporal", "window slots"),
        "capacity": ("parameter_budget", "capacity_match", "capacity-match", "trainable_parameter"),
        "checkpoint": ("checkpoint_round_trip", "state_dict", "missing key", "unexpected key"),
    }
    selected = [name for name, words in matchers.items() if any(word in text for word in words)]
    notes = []
    for name in selected[:2]:
        body = (ROOT / "cases" / (name + ".txt")).read_text(encoding="utf-8")
        notes.append({"case_id": name, "applicability": "current_task_keyword_match_verify_against_approved_contract", "instruction": body})
    return {"version": PROMPT_VERSION, "notes": notes} if notes else None


def prompt_rule_mapping() -> dict:
    return {
        "authority, admissibility, current identity, exact cost mirrors": "native action/spec/owner validators; compact authority kernel",
        "scientific coverage, metric units, matched control, final-test scope": "native ExperimentObservation facts + shared mission",
        "candidate source, interfaces, checkpoints and actual-read obligations": "native code/check receipts + Coder/Reviewer mission",
        "independent proposals, K shortfall and selection": "ResearchRoundDraft/ResearchPlanProposal validators + Planner mission",
        "positive-mass InfoNCE, tied-image CE, objective-off": "conditional implementation notes; current approved definition remains authority",
        "ordered pooling, capacity probes, checkpoint layout": "conditional probe/check notes; native requirements validators",
        "pulse and particular tensor/config failures": "current native probe/error/source only; relevant engineering skill retrieval",
        "legacy pending paid prompts": "legacy exact system retained; new version creates new request/task identity",
    }
