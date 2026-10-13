# Reliable action planning and staged LOSO execution

The new schedule is enabled only by `evaluation_schedule: screen_then_confirm`.
The complete [goal example](../react-agent/configs/goals/eeg_screen_then_confirm.yaml)
includes all eight role profiles, bounded action planning, candidate coding recovery,
fixed screening selection and confirmation policy. Its resource values are example
ceilings: native creation must inherit verified remaining authorization. This change
does not start an experiment or increase an existing campaign's budget.

## Preview and create

Run these commands from `react-agent/` with the project Python environment:

```bash
python -m react_agent.eeg_research.agentic.cli validate-goal \
  --goal configs/goals/eeg_screen_then_confirm.yaml
python -m react_agent.eeg_research.agentic.cli role-config-preview \
  --goal configs/goals/eeg_screen_then_confirm.yaml
python -m react_agent.eeg_research.agentic.cli screening-dry-run \
  --goal configs/goals/eeg_screen_then_confirm.yaml
```

The dry-run shows two preselected subjects, nine training subjects in each fold,
the eight confirmation folds and both sides of the phase cost. A preview without a
campaign assumes no valid cache. GPU cost remains unknown until a native historical
cost basis exists; missing-fold job counts remain explicit. Preview does not launch
training, reserve resources or call a model.

For an existing new-schedule campaign, inspect its verified caches with
`screening-dry-run --campaign my_screening_study --candidate c1`. Completed fold IDs
come from current native source/checkpoint/evaluation verification, not filenames
or stored score labels. The UI displays stage cost and the exact reused folds.

When real execution is authorized, create a new campaign from a quiescent budget
source, specifying available physical cards and the actual data root:

```bash
python -m react_agent.eeg_research.agentic.cli create \
  --campaign my_screening_study \
  --goal configs/goals/eeg_screen_then_confirm.yaml \
  --inherit-study /absolute/path/to/quiescent_budget_source \
  --data-root /absolute/path/to/data --gpu 0,1
python -m react_agent.eeg_research.agentic.cli start --campaign my_screening_study
python -m react_agent.eeg_research.agentic.cli status --campaign my_screening_study
python -m react_agent.eeg_research.agentic.cli pause --campaign my_screening_study
python -m react_agent.eeg_research.agentic.cli resume --campaign my_screening_study
python -m react_agent.eeg_research.agentic.cli stop --campaign my_screening_study
```

Occupied cards require the existing explicit sharing controls. The same frozen
card group and recipe apply to baseline and candidates. No process belonging to
another job is terminated by admission. Pause/stop, unknown process state and
unreconciled API intent take priority over plan execution.

## Role routing and request limits

Resolution order is CLI override, goal `role_models`, role environment variables,
global compatibility variables, then EEG defaults. CLI overrides use, for example,
`--role-override candidate_coder.model=deepseek-v4-pro`; they are available for new
campaign creation and preview. Role environment names include
`EEG_CANDIDATE_CODER_MODEL`, `EEG_CANDIDATE_CODER_THINKING` and
`EEG_CANDIDATE_CODER_REASONING_EFFORT`. Credentials stay in environment variables.

Planner, Designer, Coder, Reviewer, Analyst and Auditor default to Flash thinking
high; Librarian disables thinking; Curator uses thinking low. The request carries
the resolved model, `thinking.type`, `reasoning_effort`, token budget and deadline.
Disabled thinking has no effective effort. Only nonempty final content with normal
`stop` completion may supply JSON tools. Reasoning text, truncated output and
non-stop completions remain recorded failures. An async wall-clock deadline covers
the entire request even when a server keeps the connection alive.

Each logical role request or coder tool step shares a persistent limit of two
actual attempts. A Planner round additionally shares six calls across generation,
targeted option repair, replacement and compaction. Changing an attempt or restarting
a worker does not refresh these limits. Repeated semantic coding errors may upgrade
once to Pro high only with complete approved inputs and remaining candidate/API
budget. Candidate-wide consumed tool steps and failed checks survive that upgrade.

Use `role-config-preview` to inspect actual resolved settings and their sources.
The ledger records requested/returned model, configuration hash, tokens, finish
reason and elapsed time. Missing provider price/usage is unknown, never zero.

Unknown provider outcomes retain their consumed call and uncertainty. After the
worker is quiescent, explicit disposition can permit a new bounded request:

```bash
python -m react_agent.eeg_research.agentic.cli dispose-unknown-intent \
  --campaign my_screening_study --call-id ACTUAL_CALL_ID \
  --retry-reason 'Response could not be recovered; retain unknown charge' \
  --allow-new-request
```

A durable response must first follow native adoption/reconciliation. Disposition
does not claim a refund or confirmed provider cost.

## Scientific scope and costs

Each screening fold trains a separately initialized model on the other nine
subjects. The other screened subject can be among those nine. Screening uses the
same 50-epoch full recipe, seed 0, development checkpoint selection and fixed
query/gallery as confirmation. The two overlapping training sets are not independent
replications. Per-fold held-out EEG is excluded from training and fitted statistics;
source development validation still uses the existing image-feature/gallery scope.

Screening records have `scope=two_fold_development`, coverage 2 and separate
`screening_mean_*` fields. Default promotion requires valid paired mean Top-1 gain
of at least 0.5 percentage points. A failed screening is local development evidence,
not a ten-subject rejection of the scientific mechanism. Passed screening with
insufficient resources remains passed and waits for admission.

Without baseline cache, initial screening can require four jobs; with both baseline
folds it requires two. If both methods have only their screening folds, confirmation
requires sixteen jobs; with complete baseline it requires eight. Actual admission
atomically reserves missing work and required feedback, checks current resources
again before each launch and releases unused reservations at phase termination.

One frozen method suite declares all ten folds. Valid screening folds count toward
the final ten, so a normal promotion adds eight candidate folds. Changing source,
spec, seed, recipe, checkpoint policy or relevant execution/evaluation dependencies
invalidates reuse. Partial coverage never yields a full mean. Ten-fold results are
development confirmation after two-fold selection, retain selection bias and do not
become an independent final test or multi-seed replication.

## Recovery and historical protocols

Existing persisted full-only manifests keep their meaning. Do not edit their hashes,
feedback permissions, state, scores or ledgers to adopt the new schedule. New goals
must select the schedule explicitly. In-stage reuse requires the same complete
canonical training/evaluation identity and original run/hash/cost provenance.

Framework bytes may already be frozen by historical manifests. Keep a fixed old
checkout/runtime for historical inspection; verify the exact manifest dependencies
and path identities before attempting native recovery. A newer checkout is not an
automatic substitute. An old checkout at a different path is also insufficient:
historical manifests reference absolute paths. Native recovery needs an isolated
mount/container that maps verified old source, data and campaign paths to those
original absolute locations. This implementation does not automatically mount or
resume a historical experiment. `historical-runtime-preview --campaign OLD_CAMPAIGN
--runtime-checkout /path/to/pinned/repository` compares the proposed source checkout
to each frozen framework dependency and reports the required absolute mount. A
matching preview does not authorize resume; data, runtime and process checks still
run in the historical environment. Cross-version reuse needs a controlled import receipt with
the original manifest, fold/run hashes, canonical identity, cost ledger and mapping.
Missing proof is rejected; independently searched legacy per-fold methods cannot
be assembled into a single method suite. Historical API and GPU consumption remains
charged when a new campaign inherits its remaining budget.

Failed branches can be explicitly detached through native control after process,
API and probe reconciliation. This retains source, errors, results and consumed
method slots; it only releases unused reservations and pending selectors:

```bash
python -m react_agent.eeg_research.agentic.cli detach-candidate \
  --campaign my_screening_study --candidate c1 \
  --retry-reason 'Bounded engineering recovery exhausted; retain this failed branch'
```

A pending GPU check, an unresolved provider intent or a live/unknown process cannot
be bypassed by detachment. Historical candidate admissions remain consumed when a
new protocol inherits budgets; approval without a later passing check still counts.

## Verification

Offline request-capture and native mock integration tests exercise the client,
Planner, dispatch, candidate checks, phase scheduling, recovery and consumers.
Mock replies and synthetic scores do not establish model quality or EEG gains.
Real candidate checks may use CPU-only small modules; they do not launch a ten-fold
training experiment. Check the accompanying implementation verification report for
the exact tested commands, results and remaining limitations.
