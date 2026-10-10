"""Abandon orphaned API attempts without guessing their transport outcome."""
from pathlib import Path
import json
import time


def abandon_orphaned_intents(camp: Path, *, reason: str) -> dict:
    """Caller must hold worker.lock; every reserved attempt stays fully charged.

    This records a delivery failure only. It creates no response or scientific
    result, and permanently preserves unknown token/transport costs.
    """
    from react_agent.eeg_research.agentic.llm import refresh_method_api_cost, _ledger
    from react_agent.eeg_research.agentic.loso_study import worker_alive
    from react_agent.eeg_research.agentic.task_ledger import latest, mark
    if not reason.strip():
        raise ValueError("api_recovery_reason_required")
    if worker_alive(camp):
        raise ValueError("api_recovery_worker_still_alive")
    if json.loads((camp / "goal.json").read_text()).get("evaluation_mode") != "loso_method_search":
        raise ValueError("api_recovery_requires_method_campaign")
    cost = refresh_method_api_cost(camp)
    pending = set(cost["api_unsettled_call_ids"])
    intents_path = camp / "llm_api_intents.jsonl"
    intents = [json.loads(line) for line in intents_path.read_text().splitlines()
               if line.strip()] if intents_path.exists() else []
    for intent in intents:
        if intent["call_id"] not in pending:
            continue
        # A durable response needs exact validation/adoption, not abandonment.
        if (camp / "llm_responses" / (intent["call_id"] + ".json")).exists():
            raise ValueError("api_recovery_durable_response_requires_adoption")
    for intent in intents:
        if intent["call_id"] not in pending:
            continue
        row = {key: value for key, value in intent.items()
               if key not in {"status", "uncertain_if_unsettled"}}
        row.update(success=False, error="process_exited_before_reply_persisted",
                   transport_outcome="unknown_after_process_exit", recovery_reason=reason,
                   recovered_at=time.time(), input_tokens=None, output_tokens=None,
                   reasoning_tokens=None, budget_charge_calls=1)
        _ledger(camp, row)
    # Also finish a task if recovery itself previously crashed after its call row.
    cost = refresh_method_api_cost(camp)
    for intent in intents:
        if intent["call_id"] not in cost["api_abandoned_call_ids"]:
            continue
        task_id = intent.get("task_id")
        task = latest(camp, task_id) if task_id else None
        if task and task.get("status") in {"pending", "running"}:
            mark(camp, task_id, "failed", role=intent.get("role"),
                 attempt_id=intent.get("attempt_id"),
                 error="api_reply_lost_after_process_exit", recovery_reason=reason,
                 abandoned_transport_call_id=intent["call_id"])
    return cost
