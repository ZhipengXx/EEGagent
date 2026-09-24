"""One-line progress for CLI and Studio. Reuses events.jsonl; never dumps arrays."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from react_agent.fmri.reporting import append_event

_DROP_KEYS = frozenset(
    {
        "preds",
        "embedding",
        "array",
        "contrast",
        "data",
        "tokens",
        "candidate_briefs",
    }
)

_active: contextvars.ContextVar["Progress | None"] = contextvars.ContextVar(
    "fmri_progress", default=None
)


def format_progress_line(event: dict[str, Any]) -> str:
    """Turn an event dict into a single CLI/Studio line."""
    kind = str(event.get("type") or "event")
    if kind == "generate":
        return f"[generate] {event.get('message') or event.get('status') or ''}".rstrip()
    if kind == "tool":
        name = event.get("tool_name") or "tool"
        status = event.get("status") or ""
        return f"[tool] {name} {status}".rstrip()
    if kind == "decision":
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        via = event.get("via") or decision.get("source") or "rule"
        action = decision.get("action")
        if action == "stop" or not decision.get("tool_name"):
            reason = decision.get("stop_reason") or event.get("stop_reason") or "stop"
            return f"[decision] {via} stop {reason}"
        return f"[decision] {via} -> {decision.get('tool_name')}"
    if kind == "stop":
        verdict = event.get("verdict") or ""
        reason = event.get("stop_reason") or ""
        return f"[stop] {verdict} {reason}".strip()
    if kind == "lm":
        return f"[lm] {event.get('kind') or 'call'}"
    if kind == "fallback":
        return f"[decision] fallback -> {event.get('to') or 'rule'}"
    if kind == "plan.rejected":
        return f"[plan] rejected {event.get('error') or ''}".rstrip()
    if kind.startswith("plan"):
        return f"[plan] {kind} {event.get('tool_id') or event.get('plan_id') or event.get('trigger') or ''}".rstrip()
    if kind.startswith("api"):
        return f"[api] {kind} {event.get('role') or ''}".rstrip()
    if kind.startswith("memory"):
        return f"[memory] {kind} {event.get('message') or event.get('episode_id') or ''}".rstrip()
    message = event.get("message") or event.get("error") or event.get("status") or ""
    return f"[{kind}] {message}".rstrip()


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in event.items():
        if key in _DROP_KEYS:
            continue
        if isinstance(value, list) and value and not isinstance(value[0], (str, int, float, bool)):
            continue
        out[key] = value
    return out


def _try_stream_writer(line: str, event: dict[str, Any]) -> None:
    writer = None
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        try:
            from langgraph.types import StreamWriter  # noqa: F401
        except Exception:  # noqa: BLE001
            return
    if writer is None:
        return
    try:
        writer({"line": line, "type": event.get("type")})
    except Exception:  # noqa: BLE001
        return


class Progress:
    """Write JSONL, print one line, optionally push a Studio custom event."""

    def __init__(
        self,
        events_path: Path | str | None = None,
        *,
        sink: list[dict[str, Any]] | None = None,
        print_cli: bool = True,
    ) -> None:
        self.events_path = Path(events_path) if events_path else None
        self.sink = sink
        self.print_cli = print_cli

    def emit(self, event: dict[str, Any]) -> str:
        safe = _safe_event(event)
        line = format_progress_line(safe)
        safe["line"] = line
        if self.events_path is not None:
            append_event(self.events_path, safe)
        if self.sink is not None:
            self.sink.append(safe)
        if self.print_cli:
            print(line, flush=True)
        _try_stream_writer(line, safe)
        return line


def emit(event: dict[str, Any], *, path: Path | str | None = None) -> str:
    """Emit on the bound progress, or write to ``path`` if nothing is bound."""
    prog = _active.get()
    if prog is None:
        prog = Progress(events_path=path, print_cli=True)
    elif prog.events_path is None and path is not None:
        prog.events_path = Path(path)
    return prog.emit(event)


@contextmanager
def bind_progress(progress: Progress) -> Iterator[Progress]:
    """Bind a Progress for generate + check in the current task."""
    token = _active.set(progress)
    try:
        yield progress
    finally:
        _active.reset(token)
