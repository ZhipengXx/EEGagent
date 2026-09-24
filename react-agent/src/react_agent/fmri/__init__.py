"""Pseudo-fMRI inspection package.

The original ReAct chat graph remains at ``react_agent.graph``. This package
adds an independent ``fmri_check`` loop that does not import Tavily tools.
"""

from react_agent.fmri.graph import graph

__all__ = ["graph"]
