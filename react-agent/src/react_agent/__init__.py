"""React Agent.

This module defines a custom reasoning and action agent graph.
It invokes tools in a simple loop.
"""

from typing import Any

__all__ = ["graph"]


def __getattr__(name: str) -> Any:
    """Load the chat graph only when something asks for it."""
    if name == "graph":
        from react_agent.graph import graph

        return graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
