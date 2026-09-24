"""Planner / replanner / probe using the existing JSON-object provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from react_agent.fmri.config import FmriCheckConfig
from react_agent.fmri.llm.deepseek import DeepSeekBackend, DeepSeekCallError, DeepSeekParseError
from react_agent.fmri.planning.schemas import PlanProposal, PlanningContext
from react_agent.fmri.schemas import LMUsage

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def planner_prompt(version: str = "planner_v1") -> str:
    path = _PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        path = _PROMPT_DIR / "planner_v1.md"
    return path.read_text(encoding="utf-8")


def curator_prompt(version: str = "memory_curator_v1") -> str:
    path = _PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        path = _PROMPT_DIR / "memory_curator_v1.md"
    return path.read_text(encoding="utf-8")


def plan_schema_text() -> str:
    return json.dumps(PlanProposal.model_json_schema())


class Planner:
    """Create or revise a PlanProposal via JSON Action."""

    def __init__(self, backend: Any, config: FmriCheckConfig) -> None:
        self.backend = backend
        self.config = config

    async def create_plan(self, context: PlanningContext) -> tuple[PlanProposal, LMUsage]:
        return await self._call(context, phase="create")

    async def revise_plan(self, context: PlanningContext) -> tuple[PlanProposal, LMUsage]:
        return await self._call(context, phase="revise")

    async def _call(
        self, context: PlanningContext, *, phase: str
    ) -> tuple[PlanProposal, LMUsage]:
        payload = context.model_dump()
        payload["request_mode"] = phase
        user = (
            "PlanProposal JSON Schema:\n"
            + plan_schema_text()
            + "\nContext:\n"
            + json.dumps(payload, default=str)
        )
        system = planner_prompt(self.config.planning.prompt_version)
        section_sizes = {
            "schema": len(plan_schema_text()),
            "context": len(json.dumps(payload, default=str)),
            "system": len(system),
        }
        text, usage = await complete_json(
            self.backend,
            system=system,
            user=user,
            profile="fast",
            role="planner" if phase == "create" else "replanner",
        )
        usage.context_section_sizes = section_sizes
        usage.prompt_version = self.config.planning.prompt_version
        usage.token_count_source = "provider" if usage.input_tokens is not None else "estimated"
        usage.schema_version = "PlanProposal"
        try:
            proposal = PlanProposal.model_validate({**text, "phase": phase})
        except (ValidationError, TypeError) as exc:
            usage.success = False
            usage.error = f"invalid_plan:{type(exc).__name__}"
            raise DeepSeekParseError(json.dumps(text), usage) from exc
        return proposal, usage


async def complete_json(
    backend: Any,
    *,
    system: str,
    user: str,
    profile: str,
    role: str,
) -> tuple[dict[str, Any], LMUsage]:
    """Shared JSON object call. Backends must implement complete_json or _complete."""
    if hasattr(backend, "complete_json"):
        payload, usage = await backend.complete_json(
            system=system, user=user, profile=profile, role=role
        )
        return payload, usage
    raise TypeError(f"backend {type(backend)} has no complete_json")


async def probe_provider(backend: DeepSeekBackend, config: FmriCheckConfig) -> dict[str, Any]:
    """One billed JSON ping. Never claimed free."""
    payload, usage = await complete_json(
        backend,
        system="Return {\"ok\": true, \"role\": \"probe\"} as JSON.",
        user=json.dumps({"probe": True, "claim_scope": "numeric_consistency_only"}),
        profile="fast",
        role="probe",
    )
    return {
        "ok": bool(payload.get("ok")),
        "payload": payload,
        "usage": usage.model_dump(),
        "requested_model": usage.requested_model,
        "response_model": usage.response_model,
        "key_validated": bool(usage.success),
        "billed": True,
        "model_configured": config.fast.model,
    }


def new_call_id() -> str:
    return uuid4().hex[:12]
