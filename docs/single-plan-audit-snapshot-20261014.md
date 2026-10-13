# Single-plan audit development snapshot — 2026-10-14

This commit saves the current implementation. It does not mark the audit of
`40f1243232dab7f2f600d76cc5a1a42a1520cc9f` complete.

Implemented changes include evaluation coverage for every valid proposal,
shared task-kind/phase validation, recoverable prepare/decision/intent/commit
publication and execution receipts, deterministic delivery of current terminal
observations, and consistent recent/semantic applicability checks. The Planner
prompt now includes its proposal schema once. Pricing reservation and settlement
use a frozen deployment price identity; that work is still incomplete.

The three public synthetic regression files in
`react-agent/tests/eeg_research/version_audit/` pass together: **71 passed**.
Changed Python files also pass compilation and `git diff --check`.
These tests use synthetic campaigns and local services, with no real API or GPU
training. They establish execution behavior, not EEG research performance.

Run the focused tests from an installed project environment with pytest:

```bash
cd react-agent
PYTHONPATH=src EEG_EXTERNAL_API_DISABLED=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/eeg_research/version_audit -q
```

The local MiniLM check uses revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; it skips if that model is absent
from the local cache and does not download it. The validation host had the cache
available, so its 71-test result includes that check.

Known remaining work:

- CLI/UI execution-protocol selection and status, fresh-creation crash recovery,
  and explicit refusal of start/resume for cancelled campaigns.
- Handling `ObservationDeliveryRequired` at the direct SelectedPlan selection
  entry point and persisting its missing-context/read requirements.
- Pricing recovery: settlement receipts currently omit `cache_hit_tokens`.
  Replaying a cached request with a distinct cache-hit price can fail with
  `settlement_pricing_identity_conflict`. This was reproduced offline.
- Broader public budget/continuation coverage and bounded real API validation.

Existing research runs, scientific artifacts, credentials, local model caches,
and unrelated `git_file/` materials are excluded from this snapshot.
