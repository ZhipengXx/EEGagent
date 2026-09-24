"""Registered models and profiles. Unknown ids are not executable."""

from __future__ import annotations

from react_agent.eeg_research.adapters.base import TrainingAdapter
from react_agent.eeg_research.schemas import ModelRecord, ProfileRecord


class RegistryError(ValueError):
    """Raised when a model or profile is not approved."""


class Registry:
    """Lookup table for adapters. Probe results are stored on the model record."""

    def __init__(
        self,
        models: list[ModelRecord],
        profiles: list[ProfileRecord],
        adapters: dict[str, TrainingAdapter],
    ) -> None:
        self.models = {row.model_id: row for row in models}
        self.profiles = {row.profile_id: row for row in profiles}
        self.adapters = adapters

    def probe_all(self) -> list[ModelRecord]:
        """Refresh availability from the adapter. Mock stays unverified."""
        refreshed: list[ModelRecord] = []
        for row in self.models.values():
            adapter = self.adapters.get(row.adapter_id)
            if adapter is None:
                updated = row.model_copy(
                    update={"availability": "unavailable", "reason": "adapter_missing", "last_probe": "missing"}
                )
            else:
                report = adapter.probe()
                availability = report.get("availability", adapter.availability)
                if adapter.adapter_id == "mock_retrieval":
                    availability = "unverified"
                updated = row.model_copy(
                    update={
                        "availability": availability,
                        "reason": str(report.get("reason") or adapter.reason),
                        "last_probe": "probe",
                    }
                )
            self.models[row.model_id] = updated
            refreshed.append(updated)
        return refreshed

    def require(self, model_id: str, profile_id: str) -> tuple[ModelRecord, ProfileRecord, TrainingAdapter]:
        """Return a legal triple or raise."""
        model = self.models.get(model_id)
        profile = self.profiles.get(profile_id)
        if model is None or profile is None:
            raise RegistryError("unknown_model_or_profile")
        if profile_id not in model.approved_profiles:
            raise RegistryError("profile_not_approved")
        if model.availability == "unavailable":
            raise RegistryError(model.reason)
        adapter = self.adapters.get(model.adapter_id)
        if adapter is None:
            raise RegistryError("adapter_missing")
        return model, profile, adapter

    def snapshot(self) -> dict[str, object]:
        """Serialize the registry without live adapter objects."""
        return {
            "models": [row.model_dump() for row in self.models.values()],
            "profiles": [row.model_dump() for row in self.profiles.values()],
        }
