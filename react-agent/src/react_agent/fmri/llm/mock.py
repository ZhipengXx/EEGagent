"""Deterministic offline LM for tests and demos."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from react_agent.fmri.schemas import Decision, LMUsage


class MockBackend:
    """Replay injected JSON decisions. Never pretends to be DeepSeek."""

    name = "mock"

    def __init__(
        self,
        script: list[dict[str, Any]] | None = None,
        plans: list[dict[str, Any]] | None = None,
        curator: list[dict[str, Any]] | None = None,
    ) -> None:
        self.script = list(script or [])
        self.plans = list(plans or [])
        self.curator = list(curator or [])
        self._i = 0
        self._plan_i = 0
        self._curator_i = 0

    async def decide(
        self,
        observation: dict[str, Any],
        candidate_tools: list[str],
        profile: str,
    ) -> tuple[Decision, LMUsage]:
        """Pop the next scripted decision or stop."""
        usage = LMUsage(
            provider="mock",
            requested_model="mock",
            response_model="mock",
            profile=profile,
            input_tokens=1,
            output_tokens=1,
            success=True,
        )
        payload: dict[str, Any]
        if self._i < len(self.script):
            payload = dict(self.script[self._i])
            self._i += 1
        else:
            payload = {
                "action": "stop",
                "stop_reason": "mock_script_exhausted",
                "reason": "No further mock decisions.",
            }
        # If script asks for a tool that is no longer a candidate, stop instead.
        if payload.get("action") == "run_tool" and payload.get("tool_name") not in candidate_tools:
            payload = {
                "action": "stop",
                "stop_reason": "mock_no_matching_candidate",
                "reason": "Scripted tool is not in the current candidate list.",
            }
        try:
            decision = Decision.model_validate({**payload, "source": "hybrid", "profile": profile})
        except ValidationError as exc:
            raise ValueError(f"mock decision invalid: {exc}") from exc
        return decision, usage

    async def summarize(
        self, report: dict[str, Any], profile: str
    ) -> tuple[dict[str, Any], LMUsage]:
        """Tiny canned explanation."""
        usage = LMUsage(
            provider="mock",
            requested_model="mock",
            response_model="mock",
            profile=profile,
            input_tokens=1,
            output_tokens=1,
            success=True,
        )
        return {
            "findings": f"verdict={report.get('verdict')}",
            "limitations": "Mock summary; not a neuroscience claim.",
            "next_steps": "Inspect report.json metrics_table.",
            "evidence_refs": [],
        }, usage

    def push_raw(self, text: str) -> None:
        """Queue a raw JSON string as the next decision payload."""
        self.script.append(json.loads(text))

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        profile: str,
        role: str,
    ) -> tuple[dict[str, Any], LMUsage]:
        """Return the next scripted plan or curator payload."""
        usage = LMUsage(
            provider="mock",
            requested_model="mock",
            response_model="mock",
            profile=profile,
            role=role,
            input_tokens=1,
            output_tokens=1,
            success=True,
            status="succeeded",
            call_id=f"mock_{role}_{self._plan_i}",
        )
        if role in {"planner", "replanner"}:
            if self._plan_i < len(self.plans):
                payload = dict(self.plans[self._plan_i])
                self._plan_i += 1
                return payload, usage
            raise ValueError("mock plan script exhausted")
        if role == "curator":
            if self._curator_i < len(self.curator):
                payload = dict(self.curator[self._curator_i])
                self._curator_i += 1
                return payload, usage
            return {"candidates": [], "conflicts": [], "no_new_lesson_reason": "mock_empty"}, usage
        if role == "probe":
            return {"ok": True, "role": "probe"}, usage
        return {"ok": True}, usage
