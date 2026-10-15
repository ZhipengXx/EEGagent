"""Native scalar arithmetic, without paid APIs, GPU use or campaign mutation."""
from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from react_agent.eeg_research.agentic import parameter_census


DECLARATION = {"matched_baseline_trainable_params": 5_404_673,
               "parameter_tolerance_fraction": 0.05,
               "target_parameter_band": [5_134_439, 5_674_907]}


def test_current_measured_count_is_inside_declared_band_without_changing_target():
    original = copy.deepcopy(DECLARATION)
    result = parameter_census._capacity_comparison(5_335_969, DECLARATION)
    assert result["within_declared_band"] is True
    assert result["encoder_trainable_scalars"] == 5_335_969
    assert result["declared_baseline_trainable_scalars"] == 5_404_673
    assert result["declared_band"] == [5_134_439, 5_674_907]
    assert result["delta_scalars"] == -68_704
    assert result["relative_deviation"] == pytest.approx(-68_704 / 5_404_673)
    assert result["relative_deviation_percent"] == pytest.approx(-68_704 / 5_404_673 * 100)
    assert DECLARATION == original


@pytest.mark.parametrize("count,inside", [(5_134_438, False), (5_134_439, True), (5_674_907, True), (5_674_908, False)])
def test_declared_band_has_inclusive_boundaries(count, inside):
    assert parameter_census._capacity_comparison(count, DECLARATION)["within_declared_band"] is inside


@pytest.mark.parametrize("parameters", [None, {}, {"matched_baseline_trainable_params": 5_404_673},
    {"matched_baseline_trainable_params": True, "target_parameter_band": [1, 5]},
    {"matched_baseline_trainable_params": 0, "target_parameter_band": [1, 5]},
    {"matched_baseline_trainable_params": 5_404_673, "target_parameter_band": [5, 1]},
    {"matched_baseline_trainable_params": 5_404_673, "target_parameter_band": [1, float("inf")]},
    {"matched_baseline_trainable_params": 5_404_673, "target_parameter_band": [True, 5]},
    {"matched_baseline_trainable_params": 5_404_673, "target_parameter_band": [1]},
])
def test_missing_or_invalid_capacity_target_does_not_invent_a_comparison(parameters):
    assert parameter_census._capacity_comparison(5_335_969, parameters) is None


@pytest.mark.parametrize("declared", [False, True])
def test_existing_single_build_path_adds_native_candidate_identity_and_optional_comparison(tmp_path, monkeypatch, declared):
    target = tmp_path / "c3"
    extension = target / "extension"
    extension.mkdir(parents=True)
    monkeypatch.chdir(extension)
    config = {"model": {}, "objective": {}}
    if declared:
        config["scientific_parameters"] = DECLARATION
    path = target / "spec.json"
    path.write_text(json.dumps({"experiment": config}))
    spec_before = path.read_bytes()
    parameter = SimpleNamespace(requires_grad=True, numel=lambda: 5_335_969)
    encoder = SimpleNamespace(parameters=lambda: [parameter])
    built = []
    def build(candidate, method, geometry, model_config):
        built.append(method)
        assert method == "build_encoder"
        return encoder
    monkeypatch.setitem(sys.modules, "eeg_candidate", SimpleNamespace(EEGCandidate=lambda: object()))
    monkeypatch.setitem(sys.modules, "react_agent.eeg_training.hooks", SimpleNamespace(objective_parameters=lambda _: []))
    monkeypatch.setitem(sys.modules, "react_agent.eeg_training.train_entry", SimpleNamespace(_build_hook=build))
    monkeypatch.setattr("react_agent.eeg_research.agentic.check_entry._spec", lambda _: {"c_num": 17, "timesteps": [0, 250]})
    result = parameter_census._count("synthetic", str(path))
    assert built == ["build_encoder"]
    assert result["status"] == "measured" and result["candidate_id"] == "c3"
    assert result["parameter_counts"] == {"encoder": 5_335_969, "objective": 0}
    assert result["encoder_trainable_scalars"] == 5_335_969 and result["encoder_frozen_scalars"] == 0
    assert ("capacity_comparison" in result) is declared
    if declared:
        assert result["capacity_comparison"]["within_declared_band"] is True
    assert path.read_bytes() == spec_before
