"""One research controller. Adaptive mode does not fall back to a fixed pair of trials."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

_PROMPT = Path(__file__).resolve().parent / "prompts" / "controller.txt"
ACTION_LABELS = {
    "inspect_task_data": "查看数据",
    "retrieve_research_memory": "检索经验",
    "analyze_learning_curve": "分析曲线",
    "analyze_retrieval_errors": "分析检索错误",
    "propose_experiment": "提出实验",
    "train_candidate": "训练",
    "evaluate_validation": "验证集评价",
    "replicate_candidate": "重复 seed",
    "stop_research": "停止",
}
TRAIN_ACTIONS = {"train_candidate", "replicate_candidate"}
_ALLOWED_CHANGES = {"learning_rate", "weight_decay"}
_LABEL_TO_ACTION = {label: key for key, label in ACTION_LABELS.items()}
_RAW_LIMIT = 2048


class StaleState(RuntimeError):
    """Raised when a writer does not hold the current state version."""


class PlanningError(RuntimeError):
    """Raised when a plan cannot be used. Callers must not switch policy."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def action_for_evidence(observation: str, *, trials_used: int, max_trials: int, memory_hit: bool) -> str:
    """Pick a legal next action from evidence. Trial count two is not a stop by itself."""
    if observation == "positive_mapping_error":
        return "analyze_retrieval_errors"
    if observation == "repeat_no_gain" and memory_hit:
        return "stop_research"
    if observation == "no_legal_question":
        return "stop_research"
    if observation == "new_evidence" and trials_used < max_trials:
        return "train_candidate"
    if trials_used >= max_trials:
        return "stop_research"
    return "inspect_task_data"


def retrieve_memory(rows: list[dict[str, Any]], *, task_hash: str, metric: str) -> dict[str, Any]:
    """Keep compatible episodes and record exclusions. Test scores are not retrieved."""
    used = []
    excluded = []
    for row in rows:
        if row.get("test_result") is not None:
            excluded.append({"id": row.get("memory_id"), "reason": "test_result"})
            continue
        if row.get("task_hash") != task_hash or row.get("metric") != metric:
            excluded.append({"id": row.get("memory_id"), "reason": "incompatible"})
            continue
        used.append(row.get("memory_id"))
    return {"memory_used": used, "excluded": excluded, "cold_start": not used}


def open_adaptive() -> str:
    """Refuse to invent a plan or to switch to the fixed two-trial policy."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return "无法继续规划：没有 DEEPSEEK_API_KEY，不会退回固定两试。"
    return "已接通自适应规划。"


def _extract_action(payload: dict[str, Any]) -> Any:
    """Prefer action. If it is missing, use decision.action, then next_action."""
    action = payload.get("action")
    if action not in (None, ""):
        return action
    decision = payload.get("decision")
    if isinstance(decision, dict) and decision.get("action") not in (None, ""):
        return decision.get("action")
    if payload.get("next_action") not in (None, ""):
        return payload.get("next_action")
    return action


def _canonicalize_action(action: Any) -> Any:
    """Map a known Chinese label onto its English id. Other values stay as sent."""
    if not isinstance(action, str):
        return action
    text = action.strip()
    if text in ACTION_LABELS:
        return text
    return _LABEL_TO_ACTION.get(text, text)


def _show_action(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, str):
        text = action.strip()
    else:
        text = json.dumps(action, ensure_ascii=False, default=str)
    return text[:200]


def clip_raw(reply: Any, limit: int = _RAW_LIMIT) -> Any:
    """Keep a rejected reply. Objects under the limit stay objects."""
    if reply is None:
        return None
    try:
        text = json.dumps(reply, ensure_ascii=False, default=str)
    except TypeError:
        text = str(reply)
    if len(text.encode("utf-8")) <= limit and isinstance(reply, dict):
        return reply
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def parse_planner_reply(payload: Any) -> dict[str, Any]:
    """Accept one legal action. Only learning rate and weight decay may change."""
    if not isinstance(payload, dict):
        raise PlanningError("无法继续规划：返回的不是 JSON 对象，不会退回固定两试。")
    seen = _extract_action(payload)
    action = _canonicalize_action(seen)
    if action not in ACTION_LABELS:
        raise PlanningError(
            f"无法继续规划：动作不在允许列表里（实际动作：{_show_action(seen)}），不会退回固定两试。"
        )
    changes = payload.get("changes") or {}
    if not isinstance(changes, dict) or any(key not in _ALLOWED_CHANGES for key in changes):
        raise PlanningError("无法继续规划：只能修改 learning rate 或 weight decay，不会退回固定两试。")
    reason = str(payload.get("reason") or payload.get("summary") or "")
    return {
        "action": action,
        "action_label": ACTION_LABELS[action],
        "reason": reason,
        "changes": {key: changes[key] for key in changes},
    }


def request_deepseek(context: dict[str, Any]) -> dict[str, Any]:
    """One JSON completion. Dollars stay unknown."""
    from react_agent.fmri.config import FmriCheckConfig
    from react_agent.fmri.llm.deepseek import DeepSeekBackend, DeepSeekCallError, DeepSeekParseError

    config = FmriCheckConfig(
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
    model = os.environ.get("DEEPSEEK_FAST_MODEL")
    if model:
        config.fast.model = model
    system = _PROMPT.read_text(encoding="utf-8")
    user = json.dumps(context, ensure_ascii=False)
    try:
        payload, _usage = asyncio.run(
            DeepSeekBackend(config).complete_json(
                system=system,
                user=user,
                profile="fast",
                role="research_controller",
            )
        )
    except (DeepSeekCallError, DeepSeekParseError, OSError, RuntimeError) as exc:
        raise PlanningError(f"无法继续规划：{type(exc).__name__}，不会退回固定两试。") from exc
    if not isinstance(payload, dict):
        raise PlanningError("无法继续规划：返回的不是 JSON 对象，不会退回固定两试。")
    return payload


def _planner_public(value: Any) -> Any:
    """Drop test scores and batch-local scores at every level."""
    if isinstance(value, dict):
        return {
            key: _planner_public(item)
            for key, item in value.items()
            if "test" not in str(key) and "within_batch" not in str(key)
        }
    if isinstance(value, list):
        return [_planner_public(item) for item in value]
    return value


def planner_context(
    *,
    campaign_id: str,
    observation: Any,
    trials_used: int,
    max_trials: int,
    memory_rows: list[dict[str, Any]],
    completed_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Build the prompt body. Test scores and batch-local scores are omitted."""
    public = _planner_public(observation) if isinstance(observation, dict) else observation
    done = set(completed_actions or [])
    return {
        "campaign_id": campaign_id,
        "observation": public,
        "trials_used": trials_used,
        "max_trials": max_trials,
        "memory": retrieve_memory(memory_rows, task_hash=campaign_id, metric="fixed_bank_top1"),
        "legal_actions": [name for name in ACTION_LABELS if name not in done],
        "allowed_changes": sorted(_ALLOWED_CHANGES),
    }


class ResearchController:
    """Persist one research state for the workbench, the CLI, and resume."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)

    def decide(
        self,
        *,
        campaign_id: str,
        observation: Any,
        trials_used: int,
        max_trials: int,
        memory_rows: list[dict[str, Any]],
        expected_version: int | None = None,
        backend: Callable[[dict[str, Any]], Any] | None = None,
        budget: Any = None,
        completed_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask for one next action and store it. This method does not start training."""
        if expected_version is not None and expected_version != self._current_version():
            raise StaleState("stale_state_version")
        context = planner_context(
            campaign_id=campaign_id,
            observation=observation,
            trials_used=trials_used,
            max_trials=max_trials,
            memory_rows=memory_rows,
            completed_actions=completed_actions,
        )
        self._commit(
            {
                "campaign_id": campaign_id,
                "phase": "planning",
                "status": "planning",
                "action": "",
                "action_label": "",
                "reason": "",
                "detail": "正在请求规划",
                "memory": context["memory"],
                "legacy_fixed": False,
                "raw": None,
            }
        )
        if budget is not None:
            from react_agent.eeg_research.executor import ExecutionRefused

            try:
                budget.charge_lm()
            except ExecutionRefused as exc:
                return self._blocked(campaign_id, context, f"无法继续规划：{exc}，不会退回固定两试。")
        reply: Any = None
        try:
            reply = self._reply(context, backend)
            parsed = parse_planner_reply(reply)
        except PlanningError as exc:
            return self._blocked(campaign_id, context, exc.detail, raw=reply)
        detail = f"下一步：{parsed['action_label']}。{parsed['reason']}".strip()
        return self._commit(
            {
                "campaign_id": campaign_id,
                "phase": "planned",
                "status": "planned",
                "action": parsed["action"],
                "action_label": parsed["action_label"],
                "reason": parsed["reason"],
                "changes": parsed["changes"],
                "detail": detail,
                "memory": context["memory"],
                "plan": {
                    "kind": "question_graph",
                    "questions": [
                        {"id": "q-data", "ask": "划分和候选集合是否冻结", "action": "inspect_task_data"},
                        {"id": "q-next", "ask": parsed["reason"] or "下一次动作", "depends_on": ["q-data"], "action": parsed["action"]},
                    ],
                },
                "legacy_fixed": False,
                "raw": reply if isinstance(reply, dict) else None,
            }
        )

    def note_repeat(self, record: dict[str, Any], note: str) -> dict[str, Any]:
        """Stop a repeated local action. The policy stays adaptive."""
        updated = dict(record)
        updated["reason"] = note
        updated["detail"] = note
        updated["legacy_fixed"] = False
        return self._commit(updated)

    def _reply(self, context: dict[str, Any], backend: Callable[[dict[str, Any]], Any] | None) -> Any:
        if backend is not None:
            return backend(context)
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise PlanningError(open_adaptive())
        return request_deepseek(context)

    def _blocked(
        self,
        campaign_id: str,
        context: dict[str, Any],
        detail: str,
        raw: Any = None,
    ) -> dict[str, Any]:
        return self._commit(
            {
                "campaign_id": campaign_id,
                "phase": "planning_blocked",
                "status": "planning_blocked",
                "action": "",
                "action_label": "",
                "reason": "",
                "detail": detail,
                "memory": context["memory"],
                "legacy_fixed": False,
                "raw": clip_raw(raw),
            }
        )

    def _current_version(self) -> int:
        path = self.out_dir / "research_state.json"
        if not path.is_file():
            return 0
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            return 0
        return int(stored.get("state_version") or 0)

    def _commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload["state_version"] = self._current_version() + 1
        path = self.out_dir / "research_state.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.out_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": payload.get("action") or payload.get("phase"), "state_version": payload["state_version"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return payload
