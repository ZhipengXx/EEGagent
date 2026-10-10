"""Native GPU admission with explicit, scoped operational sharing controls."""
import hashlib
import json
import time
from pathlib import Path

AUTHORIZATIONS = "gpu_resource_authorizations.jsonl"


def _authorization_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def authorize_low_memory_sharing(camp: Path, state: dict, *, max_memory_mib: int, reason: str) -> dict:
    """Native operator control; append authorization without changing the goal/recipe."""
    from react_agent.eeg_research.agentic.loso_study import gpu_inventory
    if (state.get("evaluation_mode") != "loso_method_search" or state.get("live_job")
            or state.get("active_suite") or not state.get("execution_fingerprint")
            or not isinstance(state.get("gpu"), list) or len(state["gpu"]) < 2
            or any(type(card) is not int for card in state["gpu"])
            or len(set(state["gpu"])) != len(state["gpu"])
            or type(max_memory_mib) is not int or max_memory_mib < 1 or not reason.strip()):
        raise ValueError("gpu_sharing_requires_quiescent_method_campaign_limit_and_reason")
    inventory = gpu_inventory(tuple(state["gpu"]), allow_shared=True)
    if any(row["used_memory_mib"] > max_memory_mib for row in inventory):
        raise ValueError("gpu_existing_memory_exceeds_authorized_limit")
    body = {"schema_version": "eeg_research.gpu_resource_authorization.v1",
        "authority": "explicit_user_permission", "mode": "low_existing_memory_sharing",
        "campaign": str(camp.resolve()), "execution_fingerprint": state["execution_fingerprint"],
        "gpu": list(state["gpu"]), "physical_gpu_uuids": [row["uuid"] for row in inventory],
        "max_existing_memory_mib": max_memory_mib, "reason": reason, "recorded_at": time.time()}
    record = {**body, "sha256": _authorization_hash(body)}
    with (camp / AUTHORIZATIONS).open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def sharing_authorization(camp: Path, state: dict) -> dict | None:
    path = camp / AUTHORIZATIONS
    if not path.is_file():
        return None
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    record = json.loads(rows[-1])
    if not isinstance(record, dict):
        raise ValueError("gpu_resource_authorization_invalid")
    body = {key: value for key, value in record.items() if key != "sha256"}
    if (record.get("sha256") != _authorization_hash(body)
            or body.get("schema_version") != "eeg_research.gpu_resource_authorization.v1"
            or body.get("authority") != "explicit_user_permission"
            or body.get("mode") != "low_existing_memory_sharing"
            or body.get("campaign") != str(camp.resolve())
            or body.get("execution_fingerprint") != state.get("execution_fingerprint")
            or body.get("gpu") != state.get("gpu")
            or type(body.get("max_existing_memory_mib")) is not int
            or body["max_existing_memory_mib"] < 1
            or not isinstance(body.get("physical_gpu_uuids"), list)
            or len(body["physical_gpu_uuids"]) != len(state["gpu"])):
        raise ValueError("gpu_resource_authorization_invalid")
    return record


def require_idle_gpus(camp: Path, state: dict) -> None:
    if state.get("evaluation_mode") != "loso_method_search":
        return
    from react_agent.eeg_research.agentic.loso_study import read, gpu_inventory
    goal = read(camp / "goal.json")
    authorization = sharing_authorization(camp, state)
    inventory = gpu_inventory(tuple(state["gpu"]),
        allow_shared=bool(authorization) or goal.get("allow_shared_gpus") is True)
    if authorization:
        if [row["uuid"] for row in inventory] != authorization["physical_gpu_uuids"]:
            raise ValueError("gpu_resource_authorization_physical_cards_changed")
        if any(row["used_memory_mib"] > authorization["max_existing_memory_mib"] for row in inventory):
            raise ValueError("gpu_existing_memory_exceeds_authorized_limit")


def advance_needs_gpus(camp: Path, state: dict) -> bool:
    if state.get("live_job"):
        return False
    suite_id = state.get("active_suite")
    if not suite_id:
        return not any(row.get("kind") == "method_suite" and row.get("candidate_id") == "baseline"
                       and row.get("evaluation_valid") is True for row in state.get("evidence", []))
    from react_agent.eeg_research.agentic.loso_study import read
    directory = camp / "suites" / suite_id
    progress = read(directory / "suite_state.json")
    # Fixed ten-fold IDs are native execution slots; this check grants no
    # completion/score authority and does not change the frozen suite.
    for index in range(1, 11):
        fold = f"fold_{index:02d}"
        if fold not in progress["completed_folds"]:
            return fold not in progress["jobs"]
    return False
