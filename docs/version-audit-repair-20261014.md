# Version audit repair evidence — 2026-10-14

Scope: the findings in `react-agent/docs/EEGagent-version-audit-40f12432.txt`,
whose reviewed source revision was `40f1243232dab7f2f600d76cc5a1a42a1520cc9f`.
This report describes working-tree repairs after development snapshot `d9baa3c`.
The earlier snapshot's remaining-work list is a historical record.

## Current validation status

The audited defects are repaired and real DeepSeek Planner/Designer/Coder/
Reviewer/Analyst calls have completed a candidate and two actual native CPU
training jobs. A terminal observation reached the next actual Planner request
with exact facts and read handles. Native recovery then used the original paid
reply to complete its three-task comparison and next task selection.

The user approved thinking mode, source/context disclosure to DeepSeek, EEG data
reads, and a cumulative **28-call cap**. **21 real calls** were made and settled;
**7 remain**. Recovery added no API call or training job. All validation inputs
were random synthetic EEG-shaped data, with zero physical GPU training. This is
execution acceptance, not EEG efficacy evidence or a guarantee against future bugs.
The original failed-run reports remain intact.

The final public suite passed **199 tests** (`public_complete.xml`), including
the newly reproduced correction-receipt crash window and strict confirmed-reply
recovery. One upstream MiniLM method-renaming warning remains. Independent
Planner/recovery review passed 61 tests, including the original failed window.

## Findings and authoritative checks

| Finding | Current behavior and public evidence |
| --- | --- |
| P1-1 protocol activation | New CLI/UI campaigns default to `single_plan_v1`; `--execution-version {single_plan_v1,legacy}` is explicit. Old goal files remain legacy unless explicitly selected for a new CAM. CLI and UI status report actual frozen protocol, maintenance mode and SelectedPlan. `test_cli_creation.py`, `test_workbench_protocol.py`. |
| P1-2 competition coverage | Every effective valid proposal needs a unique valid evaluation; incomplete coverage gets at most one correction. Budget filtering recomputes the proposal/evaluation pool together. `test_plan_admission.py`. |
| P1-3 task/phase consistency | Proposal validation, cost projection, dispatch and persisted-plan loading share the task/phase matrix and native allowed-action policy. Illegal combinations never reach Designer/Coder. `test_plan_admission.py`. |
| P1-4 decision publication | Durable prepare/decision/intent/commit binds the exact payload and frozen plan. Uncommitted decisions cannot dispatch. Completed native receipts restore output projections without buying the action again. Tests inject exits at every internal publication/completion boundary. `test_selected_step_publication.py`. |
| P1-5 current feedback | The current terminal observation has a mandatory deterministic fact card with exact identities, status, scope, metrics or error fingerprint, explicit omissions and genuine read handles. Insufficient space yields a persisted `NeedsContext` before Planner owner/provider admission. `test_observation_delivery.py`. |
| Pricing authority | Reserve, preflight, final HTTP request hook and settlement use the same frozen provider/model/price identity. Conflicts and model changes are rejected before HTTP; cache-hit usage survives durable recovery; unknown charges remain occupied; real overruns remain visible. `test_pricing_authority.py`. |
| Recent/semantic applicability | Both paths compare the same scope/dataset/modality/seed/fold/source conditions. Differences are explicit analogy/history, and unknown conditions are not matched evidence. `test_observation_delivery.py`. |
| Planner schema repetition | The provider receives one proposal schema while raw siblings retain independent native validation. `test_prompt_schema.py`. |
| Fresh creation restart | Frozen configuration and request preparation survive 19 persistent publication boundaries. Same-authorization retries complete the same creation, grant and manifest without increasing budgets or changing history. `test_cli_creation.py`. |
| Cancelled restart | Cancelled CAMs and persistent stop intents are refused before control writes or worker spawn. `test_cli_creation.py`. |
| Finite continuation and GPU dedup | The same continuation permits two bounded Planner attempts and is not renewed by restart; owner and native job receipts charge the same GPU job once. `test_budget_continuation.py`. |
| Paid-off recovery | Deterministic capture continues, newly paid maintenance stays off, and an already paid valid publication/cache can be adopted without new API calls. Stale source/input/reply or invalid authority is refused. `test_maintenance_publication.py`. |

The public tests and their synthetic fixtures are included in the worktree under
`react-agent/tests/eeg_research/version_audit/`; the root ignore rules now permit
this focused suite. Its README gives the standalone command and dependencies.
It does not import unpublished test helpers or depend on private EEG data/CAMs.

## Native execution checks added during integration

Two additional execution gaps were reproduced and repaired:

- A newly enabled single-target default goal had no complete measurement bound.
  New submissions now expose a conservative two-job allocation for the cold-start
  matched baseline plus candidate. This uses the existing authorized global
  budget and native launch deadlines. The model's unknown cost remains unknown;
  the allocation cap is reported separately. Existing task envelopes are
  preserved, and zero-training diagnosis goals still create normally.
- A finished single-target job exposed flat native metrics instead of the
  method-suite artifact format, so terminal-card publication previously failed.
  Its adapter now re-verifies the native job, source/check/review, approved spec,
  seed, protocol/argv, checkpoint, evaluation identity, metrics and matched
  comparison before publishing the exact result. Fault injection rejects changes
  to these identities and artifacts.

Real API and Prepared-request checks additionally found:

- Planner reference descriptions were removed with schema annotations. Explicit
  runtime guidance and the bounded correction now distinguish delivered
  `context://`/verified artifact references from JSON pointers such as `/goal`.
  Invalid references and extra output fields still fail native validation.
- Generated code rejected fields in the approved Designer model mapping. The
  Designer/Coder runtime contracts now distinguish constructor parameters from
  interface metadata and require the Coder to support the approved mapping and
  inspect the actual check result.
- An 8192-token Coder response was entirely consumed by thinking. New validation
  uses the existing native Coder output limit of 32768 (16384 for the other roles)
  with thinking enabled, retaining finite admission and output accounting.
- `development_view` conflated a business JSON null with a hidden private object.
  A distinct internal omission sentinel now preserves array positions, null
  shapes, mapping values, original source hashes and exact Prepared delivery;
  private objects remain filtered.

The real Coder trace also exposed an empty output schema and extra `type`
wrappers rejected by the native tool validator. Single-plan Coder now receives
the native catalog's explicit `tool`/`args` schema. The strict native validator
and legacy output paths are retained; actual Prepared-schema regressions pass.

The public closed-loop fixture runs native Planner, Designer approval, Coder
patch and interface check, Reviewer, two actual CPU `train_entry` processes,
Analyst, terminal observation, and a subsequent Planner selection. The second
selected task is not dispatched. It verifies exact feedback fields in the next
actual transport JSON, the Prepared request digest/budget, original source
readback, and a bounded exact metrics read. Its seven SDK mock calls are not
reported as real provider calls or EEG research outcomes.

The explicit `closed_loop_evaluation --real-api` command is bounded to at most
18 API attempts and always uses random synthetic EEG-shaped data. Reports
separate physical GPU usage (zero) from the native controller's conservative CPU
execution-slot accounting. No pricing constants were invented: absent effective
deployment pricing, dollar cost remains unknown.

## Related compatibility and preservation

- The relevant existing UI/observation groups pass: **37 passed**.
- Existing owner-budget tests pass: **29 passed**.
- Existing CLI/handoff/fresh-authorization groups have **56 passed / 16 failed**;
  the failing nodes exactly match the isolated `d9baa3c` source baseline. These
  are retained as failures, not removed from the reported result. Legacy fixtures
  for input propagation and same-request fresh retry were updated to exercise
  native creation and the newly required idempotent behavior.
- Changed Python files compile; the browser JavaScript passes syntax checking
  with the available modern Node runtime; `git diff --check` passes.
- The old cancelled CAM's 11 protected files and frozen runtime's original 29
  dependencies match their recorded bytes/hashes. It remains `cancelled` with
  control `stop`. No old goals, manifests, authorization records or scientific
  artifacts were rewritten to make these checks pass.

Local evidence is retained under
`react-agent/version/version_audit_40f12432_20261014/`: `public_final.xml`,
`compat_ui_observations.xml`, `public_ready_r3.xml`, `offline_closed_loop_report.json`,
`old_campaign_protection_final.json`, `public_complete.xml`, `request_disclosure_preview.json` and
`real_api_approval_rejection.json`. Original temporary reproduction/baseline
reports are preserved separately. This work does not establish EEG efficacy or
guarantee that future experiments will encounter no bugs.

## Real provider closure and immutable recovery

| Run | Actual API calls | Original outcome |
| --- | ---: | --- |
| `real_closed_loop_r1` | 2 | Planner reference/schema rejection; no method execution |
| `real_closed_loop_r2` | 8 | Native Coder/check engineering failure; no training |
| `real_closed_loop_r3` | 11 | Two native CPU jobs and terminal feedback completed; next-round schema mismatch hit the original two-attempt limit |

R3 ran `j1_baseline_pilot` and `j2_c1_pilot`, seed 0, three epochs, two validation
queries and a two-image gallery. Both Top-1 values were 0.5, paired delta 0.0 pp.
Native CPU execution-slot accounting was 5.153153419494629 seconds; physical GPU
usage was zero. These synthetic scores establish infrastructure execution only.

Terminal observation `obs_c1cba8dfcdf3cd56d2340467` reached actual next Planner
request `63acf57f0a74`. All eight exact identity/status/scope/fact/ref fields,
original source readback and bounded exact metrics read passed. Every actual
Prepared request passed its recorded identity and final budget checks. Token
precision is explicitly estimated; characters are not official token counts or
proof of indefinite scalability.

The next reply contained useful per-proposal `implementation_risk` text. It is
now an explicitly typed optional prediction (1–1000 characters), without
execution authority. Missing values do not change historical proposal/plan
serialization or publication digests; other unknown fields remain invalid.

Paid Planner replies now persist at backend return before semantic validation.
A refused bounded correction retains the incomplete draft and failure reason.
Historical confirmed replies recover through original request/reply hashes,
settlement, owner/round, terminal and scientific identities, with read-only
original context sources before any compaction. No budget is renewed.
`test_paid_round_recovery.py` provides 17 synthetic native SDK crash/tamper
cases. A correction settled just before its raw receipt write now recovers from
the verified native response tail. Published raw receipts must match original
confirmed request/reply/settlement proof; changing a local receipt and its
digest cannot replace the paid response. Original receipts are preserved.

One externally disabled native recovery tick revalidated the original paid reply
in original round `research_round_f9308eed179a7133fb1b898c3378cf90`. It compared
all three proposals and evaluations and selected
`diag_provenance_hook_fidelity_c1` as
`plan_9f524bd85c25034c0112426823f3b109`, stopped at `prepare`. The second task was
not dispatched. This added **zero API calls and zero training jobs**.

The original R3 report still has `ok=false / planner_round_attempt_limit`.
`real_closed_loop_r3_recovered_report.json` is a separate supplemental acceptance
report with the original report SHA and native recovery evidence.
`real_api_attempts.json` records the aggregate 21/28 count.

Independent checks inspected all **357 original files**: 354 immutable files
retained exact bytes. Only native state/owner projections and an event-log
append changed. Original source, spec/check/review, all 48 job artifacts and
checkpoints, costs, 12 requests, 11 responses, old plan/publications, observation
and context database remain exact. The original continuation consumed 2/2 calls
and was normally released; allocation, consumption, round and terminal binding
were unchanged. No API claim or measurement was recreated.

Evidence: `native_paid_recovery_report.json`,
`evidence/r3_pre_native_recovery/r3_original_sha256_manifest.json`,
`evidence/r3_pre_native_recovery/r3_paid_replay_identity_review.json`, and
`evidence/r3_pre_native_recovery/r3_post_native_recovery_integrity.json`.

Additional compatibility checks retain **32 passed / 5 failed** in historical
handoff/curator tests, with exactly the same failing nodes at isolated `d9baa3c`.
See `compat_handoff_comparison.json`; these failures are not claimed green.
Final old-CAM preservation is recorded in `old_campaign_protection_final.json`:
11 protected files and all 29 frozen dependencies unchanged, cancelled/stop.

This snapshot includes the repaired source, public regression suite and report.
Raw CAMs and local validation artifacts remain under the ignored evidence root.
