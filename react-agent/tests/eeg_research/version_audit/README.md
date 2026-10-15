# Public version-audit regressions

These fixtures generate their own campaigns, inputs, receipts and source files.
They do not require private EEG data, historical CAMs, credentials, unpublished
test helpers or the original validation host's temporary environment.

From `react-agent`, use Python 3.11 or newer and install project dependencies,
pytest, and the optional memory dependencies for CPU training/local encoding:

```bash
python -m pip install -e '.[memory]' pytest
PYTHONPATH=src EEG_EXTERNAL_API_DISABLED=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/eeg_research/version_audit -q
```

Provider transport fixtures temporarily enable native admission and install an
actual SDK `MockTransport` with synthetic credentials while refusing real socket
connections. No test authorizes external API calls or GPU training. The native
closed-loop fixture performs actual training on random synthetic EEG-shaped
inputs using the CPU; its scores test infrastructure and imply no EEG efficacy.

The MiniLM test uses revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It skips if the pinned local cache
is absent and never downloads a model. Other context fixtures use a deterministic
local embedding backend and continue without this cache.

Coverage includes proposal/evaluation admission, task-kind/phase consistency,
every internal decision publication window, current observation delivery and
readback, finite continuation, owner/native GPU accounting without a GPU
process, frozen pricing through the real SDK/HTTP boundary, paid-off publication
adoption, CLI creation/restart/cancellation, and UI protocol propagation.
`test_planner_knowledge.py` additionally verifies sourced lesson/method-card
text in actual Prepared Planner JSON, independent knowledge allocation,
explicit embedding failure before owner/provider admission, healthy cold start,
source revocation/readback and a real pinned MiniLM cosine/cache case. Its
synthetic vector cases identify their backend explicitly.
Paid Planner recovery fixtures cover original settled replies, crash boundaries,
the same frozen round and continuation, and identity/tamper refusal before any
new context capture, API claim or request.

Real provider validation is a separate explicit invocation of
`react_agent.eeg_research.agentic.closed_loop_evaluation --real-api`; it is never
started by pytest. Its root must be new and empty, its API cap is at most 18,
and its dataset remains synthetic. It preserves the native design, source,
review, training, metric and terminal-observation gates.

## Independent planning and shorter implementation review

The staged planning configuration, protocol, owner, paid recovery and dispatch
fixtures cover separate Pro thinking proposal and Pro thinking Critic SDK requests,
as well as an explicitly configured Flash proposal role, complete
pool delivery, winner-only publication and crash recovery without duplicate
paid calls. `test_single_pass_review.py` covers the one-request Reviewer policy
and native Coder source-presence delivery. These use isolated local campaigns
and SDK MockTransport; they do not claim live Pro or EEG/GPU performance.

`test_review_source_references.py` additionally covers shared-scope method
shorthand in saved replies. Every referenced method must exist and be within
the supplied source range, including disjoint runtime excerpts; missing members
or another file cannot expand delivery. The Reviewer prompt examples are checked
in the actual Prepared Request while preserving its compact mission budget.

The `test_stage_*` fixtures cover explicit v2 Implementation / native Measurement /
Analysis contracts, immutable phase acceptance identity, final SDK input isolation,
optional hook applicability, source/spec/receipt binding, idempotent measurement,
analysis completion for negative results, and exact legacy Analyst schema recovery.
They use synthetic receipts or native CPU jobs and SDK MockTransport only.
