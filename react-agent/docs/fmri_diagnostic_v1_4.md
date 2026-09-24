# fMRI numeric diagnostic v1.4

V1.4 adds an `image16_numeric_diagnostic` coverage profile on top of the V1.3 planned loop. The old `tribe_diagnostic` config and its runs are unchanged.

Configured required questions are input contract, gray comparability, stimulus-temporal description of contrast, coarse spatial description, and any follow-up the exploratory router actually opens. Reference quality stays optional and `not_assessed` without a calibrated reference.

`max_step` on the gray-control tool is still the legacy adjacent difference of the RMS curve. `ras_v1` adds R, A, and S with explicit from/to frames. A heuristic ratio at 5.0 can open a localization ticket. It does not flag the sample or set a training weight.

## Acceptance

| item | status |
| --- | --- |
| diagnostic config, ledger, StopGate | implemented |
| provenance fields and execution key | implemented |
| contrast temporal profile and surface_spatial_sanity | implemented |
| exploratory routing and targeted temporal/ROI | implemented |
| planner_v2 and planning_blocked stop | implemented |
| synthetic contract tests | synthetic_passed when `tests/fmri/test_diagnostic_v1_4.py` passes |
| real DeepSeek revise_plan on accordion | not_run |
| accordion cache L1 tools | real_data_passed only for the cache tool test, not a biological claim |
