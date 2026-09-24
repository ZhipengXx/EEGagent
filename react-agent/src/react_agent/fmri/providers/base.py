"""Encoder provider protocol. New backbones add a class, not a new loop."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class EncoderError(RuntimeError):
    """Provider cannot load or encode."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class EncoderProvider(Protocol):
    """Frozen encoder with an explicit missing-reason list."""

    name: str

    def missing_reasons(self) -> list[str]:
        """Local checks only. Never download."""

    def load(self) -> None:
        """Load weights from the asset root."""

    def encode(self, payload: Any) -> np.ndarray:
        """Return a 1-D embedding. Must not pad time."""
