"""Training adapters registered by id."""

from react_agent.eeg_research.adapters.mock_retrieval import MockRetrievalAdapter
from react_agent.eeg_research.adapters.unavailable import UnavailableRetrievalAdapter

__all__ = ["MockRetrievalAdapter", "UnavailableRetrievalAdapter"]
