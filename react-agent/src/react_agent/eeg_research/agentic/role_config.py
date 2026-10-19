"""EEG role routing. Configuration is resolved before request/cache identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROLES = ("research_planner", "experiment_designer", "candidate_coder", "candidate_reviewer",
         "result_analyst", "result_auditor", "research_librarian", "memory_curator", "research_critic")
FIELDS = ("model", "thinking", "reasoning_effort", "max_tokens", "timeout_s")
ALIASES = dict(zip(("planner", "designer", "coder", "reviewer", "analyst", "auditor", "librarian", "curator", "critic"), ROLES))


class RoleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1)
    thinking: bool
    reasoning_effort: Literal["low", "high", "max"] | None = None
    # Configuration accepts the provider's largest documented output size;
    # actual model/context compatibility is resolved before each request.
    max_tokens: int = Field(gt=0, le=393216)
    timeout_s: float = Field(gt=0, le=3600)

    @model_validator(mode="after")
    def consistent_effort(self):
        if self.thinking and self.reasoning_effort is None:
            raise ValueError("thinking_requires_reasoning_effort")
        if not self.thinking and self.reasoning_effort is not None:
            raise ValueError("disabled_thinking_has_no_effective_effort")
        return self


class ResolvedRoleConfig(RoleModelConfig):
    role: str
    sources: dict[str, str]
    role_config_hash: str
    base_url: str = "https://api.deepseek.com"

    def profile(self):
        from react_agent.fmri.config import LmProfile
        return LmProfile(**{name: getattr(self, name) for name in FIELDS},
                         strict_final_output=True, wall_clock_deadline=True)

    def semantics(self) -> dict:
        return {**{name: getattr(self, name) for name in FIELDS}, "base_url": self.base_url,
                "role_config_hash": self.role_config_hash,
                "request_protocol": "eeg_json_final_stop.v1"}


def defaults(role: str) -> dict:
    if role not in ROLES:
        raise ValueError("unknown_research_role:" + role)
    if role == "research_critic":
        return {"model": "deepseek-v4-pro", "thinking": True, "reasoning_effort": "high",
                "max_tokens": 16384, "timeout_s": 240}
    if role in {"candidate_coder", "candidate_reviewer"}:
        return {"model": "deepseek-flash", "thinking": True, "reasoning_effort": "high",
                "max_tokens": 262144, "timeout_s": 2400}
    return {"model": "deepseek-flash", "thinking": role != "research_librarian",
            "reasoning_effort": None if role == "research_librarian" else "low" if role == "memory_curator" else "high",
            "max_tokens": 4096 if role == "research_librarian" else 8192 if role == "memory_curator" else 16384,
            "timeout_s": 120 if role == "research_librarian" else 180}


def _parse_env(field: str, value: str):
    if field == "thinking":
        if value.lower() not in {"1", "0", "true", "false", "on", "off", "enabled", "disabled"}:
            raise ValueError("invalid_role_thinking")
        return value.lower() in {"1", "true", "on", "enabled"}
    if field == "max_tokens":
        return int(value)
    if field == "timeout_s":
        return float(value)
    return None if field == "reasoning_effort" and value.lower() in {"none", "", "null"} else value


def resolve_role_config(camp: Path | dict | None, role: str, overrides: dict | None = None,
                        *, environ: dict | None = None) -> ResolvedRoleConfig:
    """CLI > goal > role env > legacy global env > EEG defaults; no API secrets."""
    role = ALIASES.get(role, role)
    value = defaults(role)
    sources = {name: "default" for name in FIELDS}
    if environ is None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
    env = os.environ if environ is None else environ
    goal = camp if isinstance(camp, dict) else {}
    cli = {}
    if isinstance(camp, Path):
        path = camp / "goal.json"
        goal = json.loads(path.read_text()) if path.is_file() else {}
        path = camp / "role_config_overrides.json"
        cli = json.loads(path.read_text()) if path.is_file() else {}
    # Every single-plan Planner is the staged proposer, whatever mode keys the goal records.
    if role == "research_planner" and (goal.get("research_execution") or {}).get("version") == "single_plan_v1":
        value.update(model="deepseek-v4-pro", thinking=True, reasoning_effort="high",
                     max_tokens=32768, timeout_s=240)
        sources.update({name: "staged_planner_default" for name in FIELDS})
    def merge(data, label):
        if not isinstance(data, dict) or set(data).difference(FIELDS):
            raise ValueError("invalid_role_config_fields")
        for name, item in data.items():
            value[name], sources[name] = item, label
        # false disables any inherited effort; an explicitly contradictory
        # effort in the same layer is rejected by the model.
        if data.get("thinking") is False and "reasoning_effort" not in data:
            value["reasoning_effort"], sources["reasoning_effort"] = None, label
        elif data.get("thinking") is True and value["reasoning_effort"] is None and "reasoning_effort" not in data:
            value["reasoning_effort"], sources["reasoning_effort"] = "high", label
    globals_ = {name: _parse_env(name, env["DEEPSEEK_" + name.upper()]) for name in FIELDS
                if env.get("DEEPSEEK_" + name.upper()) is not None}
    legacy = env.get("DEEPSEEK_FAST_MODEL") or env.get("DEEPSEEK_REASONING_MODEL")
    if legacy and "model" not in globals_:
        globals_["model"] = legacy
    merge(globals_, "global_env_compat")
    role_env = {}
    short_role = next((key for key, item in ALIASES.items() if item == role), role)
    for name in FIELDS:
        for prefix in ("EEG_" + role.upper() + "_", "DEEPSEEK_" + role.upper() + "_",
                       "EEG_" + short_role.upper() + "_", "DEEPSEEK_" + short_role.upper() + "_"):
            if prefix + name.upper() in env:
                role_env[name] = _parse_env(name, env[prefix + name.upper()])
                break
    merge(role_env, "role_env")
    configs = goal.get("role_models") or {}
    unknown = set(configs).difference(ROLES).difference(ALIASES)
    if unknown:
        raise ValueError("unknown_role_models:" + ",".join(sorted(unknown)))
    merge(configs.get(role) or configs.get(next((key for key, item in ALIASES.items() if item == role), "")) or {}, "goal")
    merge(cli.get(role) or {}, "cli")
    merge(overrides or {}, "cli")
    config = RoleModelConfig.model_validate(value).model_dump()
    base = env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    identity = {**config, "base_url": base, "request_protocol": "eeg_json_final_stop.v1"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ResolvedRoleConfig(**config, role=role, sources=sources, role_config_hash=digest, base_url=base)


def role_config_preview(camp: Path | dict | None, overrides: dict | None = None) -> dict:
    from react_agent.eeg_research.agentic.api_policy import validate_api_request_policy
    goal = camp if isinstance(camp, dict) else {}
    if isinstance(camp, Path) and (camp / "goal.json").is_file():
        goal = json.loads((camp / "goal.json").read_text())
    recovery = goal.get("coding_recovery") or {}
    return {"roles": {role: resolve_role_config(camp, role, (overrides or {}).get(role)).model_dump() for role in ROLES},
            "escalation": {"model": recovery.get("escalation_model", "deepseek-v4-pro"),
                           "thinking": recovery.get("escalation_thinking", True),
                           "reasoning_effort": recovery.get("escalation_reasoning_effort", "high"),
                           "max_per_candidate": recovery.get("max_model_escalations_per_candidate", 1),
                           "requires": "complete coding inputs, repeated semantic code error and remaining lineage/API budget"},
            "request_policy": validate_api_request_policy(goal.get("api_request_policy") or {})}
