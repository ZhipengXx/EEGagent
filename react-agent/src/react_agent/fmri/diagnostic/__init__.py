"""V1.4 numeric diagnostic contracts. Inactive unless diagnostic.enabled."""

from react_agent.fmri.diagnostic.contracts import build_diagnostic_ledger
from react_agent.fmri.diagnostic.metrics import METRIC_VERSION, contrast_series
from react_agent.fmri.diagnostic.routing import derive_tickets
from react_agent.fmri.diagnostic.stop_gate import evaluate_stop

__all__ = [
    "METRIC_VERSION",
    "build_diagnostic_ledger",
    "contrast_series",
    "derive_tickets",
    "evaluate_stop",
]
