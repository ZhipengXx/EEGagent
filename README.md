# EEGagent

**An agent-guided workbench for fMRI screening and EEG retrieval research.**

EEGagent coordinates planning, tools, code changes and evidence across two neuroscience workflows. Screen predicted fMRI responses with explicit checks, or investigate EEG-to-image retrieval through controlled candidate experiments. Each run records what was proposed, what actually executed and what the evidence supports.

[Quick start](#quick-start) · [Agent collaboration](#agent-collaboration) · [Run a workflow](#run-a-workflow) · [Architecture and source map](docs/architecture.md)

![EEGagent system overview: a shared workbench exposes two independent research loops.](docs/figures/01-system-overview.png)

## What you can do

| Workflow | Input | Main operations | Output |
| --- | --- | --- | --- |
| **fMRI screening** | An existing surface series, or an image with a configured TRIBE backend | Generate if needed; run required checks; select further checks; update evidence; finalize | Screening report, diagnostic artifacts and a memory episode |
| **EEG / MEG retrieval research** | Preprocessed brain signals, image-feature caches and an experiment protocol | Establish controls; propose and review candidate code; train; compare; analyze; revise the plan | Candidate extensions, checkpoints, comparisons, scoped lessons and audit records |

The workflows share a local interface while keeping their state and memory separate. There is currently no automatic path that turns fMRI screening results into EEG training data. The LangGraph ReAct graph in [`graph.py`](react-agent/src/react_agent/graph.py) remains a separate conversation entry.

## Agent collaboration

### EEG retrieval: from a hypothesis to comparable evidence

The research planner chooses among actions available in the current campaign state. It can inspect data, retrieve experience and methods, design an experiment, implement or repair a candidate, run a pilot or full experiment, request replication, collect diagnostics, audit a result, or stop.

![Eight EEG LLM roles collaborate through a campaign runtime, with deterministic approval, training and evidence checks.](docs/figures/02-eeg-agent-collaboration.png)

*The figure shows representative handoffs. It is not a mandatory sequence: roles are invoked as needed, and the runtime mediates their inputs and outputs.*

| LLM role | Responsibility | Handoff |
| --- | --- | --- |
| **Research Planner** | Choose the next action and revise the plan using evidence and budget | Decision and validated plan update |
| **Research Librarian** | Interpret local method cards, source references and applicability limits | Method evidence for planning and design |
| **Experiment Designer** | Draft a falsifiable experiment with a control, predictions and required capabilities | Experiment draft for runtime validation |
| **Candidate Coder** | Implement or repair an isolated candidate extension using controlled tools | Candidate source and implementation checks |
| **Candidate Reviewer** | Identify blocking implementation and contract issues | Ready, needs-fix or blocked verdict |
| **Result Analyst** | Interpret valid comparisons against the tested hypothesis | Assessment and suggested next actions |
| **Memory Curator** | Propose conditional lessons grounded in recorded episodes | Lesson proposals checked by the runtime |
| **Result Auditor** | Check draft claims using the latest comparison and diagnostics | Audit verdict; runtime tracks evidence freshness |

These are role-specific LLM calls, not eight permanently running processes. [Prompts](react-agent/src/react_agent/eeg_research/agentic/prompts/) share a common contract; [task records](react-agent/src/react_agent/eeg_research/agentic/task_ledger.py), attempt identities and [artifact hashes](react-agent/src/react_agent/eeg_research/agentic/artifacts.py) identify the handoffs.

**The runtime controls execution.** An LLM draft requires a deterministic, context-bound approval record. The worker binds candidate source and configuration to the run, checks evaluation identity, and compares results with a matched control. Pilot results remain pilot evidence; stronger confirmation depends on full runs, declared seeds and the frozen confirmation policy. Audit PASS means a report is supported within its stated scope.

### fMRI screening: select checks, observe, replan

The workflow ingests the sample and runs required checks. A rule policy or configured LLM planner then selects additional checks. The runtime validates each proposed action, executes a registered tool and updates the evidence and question ledger before deciding whether to continue.

![fMRI screening loop: required checks, planning, validation, tools, final report and optional memory curation.](docs/figures/03-fmri-screening-loop.png)

Available checks depend on input and resources: numeric and temporal diagnostics, gray-control contrast, stimulus timing, surface and ROI profiles, cross-image specificity, reference distributions, and optional semantic or CortexMAE checks. Image generation uses an external TRIBE worker and matched gray control when configured.

Reports retain **`claim_scope=numeric_consistency_only`**. They distinguish `verdict`, `screening_decision` and `stop_reason`; diagnostic configurations can emit next-step tickets. Episode writing is deterministic, and optional LLM curation proposes evidence-linked lessons for later samples.

## Quick start

Use Python **3.11 or newer** and `uv`. From a fresh checkout:

```bash
git clone https://github.com/ZhipengXx/EEGagent.git
cd EEGagent/react-agent
uv sync --group dev
cp .env.example .env
```

### Run the offline numeric demo

This example uses the rule policy and requires neither an LLM API call nor TRIBE generation:

```bash
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo

uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok
```

### Open the workbench

```bash
uv run python -m react_agent.fmri.workbench
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) to inspect screening runs and configure retrieval experiments. The code-level research view exposes campaign state, decisions, candidate code, comparable results and run controls.

## Run a workflow

All commands below run from `react-agent/`. YAML paths are relative to that directory.

### Generate from an image and screen the prediction

Configure the external TRIBE checkout, model resources and `TRIBE_PYTHON` first:

```bash
uv run python -m react_agent.fmri.cli check-image \
  --config configs/fmri_check_tribe_image16.yaml \
  --image /path/to/image.jpg \
  --out runs/image_check
```

For planned screening, use [`fmri_check_tribe_image16_agentic.yaml`](react-agent/configs/fmri_check_tribe_image16_agentic.yaml) with a configured backend and API key. Optional CortexMAE and CLIP weights can be prepared separately:

```bash
uv sync --group dev --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
```

See [asset setup](assets/README.md) and [planned screening](react-agent/docs/fmri_agentic_v1_3.md).

### Inspect retrieval prerequisites

Set `EEG_DATA_ROOT` to your preprocessed THINGS-EEG / THINGS-MEG data and feature caches, and `EEG_TRAIN_PYTHON` to the compatible Torch interpreter. Inspect a design before training:

```bash
uv run python -m react_agent.eeg_training.cli run \
  --dataset eeg --exp-setting intra-subject --subject sub-01 \
  --out runs/eeg_training/probe --dry-run
```

The training interface supports EEG / MEG and intra-subject / inter-subject designs. The actual split and selected training strategy determine the scope of a generalization claim.

### Validate a goal without starting a campaign

```bash
uv run python -m react_agent.eeg_research.agentic.cli validate-goal \
  --goal configs/goals/eeg_retrieval_v1_9.yaml
```

The checked-in GoalSpec is an example. Validation prints its fields and the capability manifest; it does not start a worker or verify that the listed hooks have run successfully in your environment. Prepare your own goal with budgets, training seeds and any practical-gain threshold before a real campaign.

### Create, start and inspect code-level research

After configuring data, the training interpreter, DeepSeek and your goal:

```bash
uv run python -m react_agent.eeg_research.agentic.cli create \
  --campaign my_retrieval_study --goal /path/to/goal.yaml --gpu 0

uv run python -m react_agent.eeg_research.agentic.cli start \
  --campaign my_retrieval_study

uv run python -m react_agent.eeg_research.agentic.cli status \
  --campaign my_retrieval_study
```

`create` freezes the protocol and writes state; `start` launches the worker and can consume API and GPU budget. Use `pause`, `resume` or `stop` with the same campaign name to control it. A lock prevents a second worker for that campaign.

### Inspect an exported candidate pack

```bash
uv run python -m react_agent.eeg_research.agentic.cli evaluate-pack \
  --pack /path/to/candidate_pack \
  --data-root /path/to/data --out runs/pack_eval
```

The default prints an evaluation command without fitting. To execute evaluation, configure `EEG_ALLOW_EVALUATE_PACK=1` and add `--execute`. A pack's existence alone does not establish successful reconstruction or evaluation; verify reuse with an actual evaluate-only run.

## Configuration

Copy [`.env.example`](react-agent/.env.example) and keep real values local. Export machine-specific values for CLI processes as needed.

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | DeepSeek-backed planning and research role calls |
| `DEEPSEEK_FAST_MODEL` / `DEEPSEEK_REASONING_MODEL` | Model profiles; current EEG role calls use the fast profile |
| `TRIBE_PYTHON` | Interpreter for external TRIBE generation |
| `EEG_DATA_ROOT` | Preprocessed data and image-feature caches |
| `EEG_TRAIN_PYTHON` | Torch interpreter for training and candidate checks |
| `EEG_GPU_SECONDS` | Budget input for standalone training; campaigns record their own submitted goal / run budgets |
| `MODEL` | Model for the separate ReAct graph |
| `TAVILY_API_KEY` | Search on the ReAct graph; not required by the research loops |

The native code-level worker requires the configured LLM backend. Offline rule screening and dry probes are separate entry points. Sample [`eeg_research_v1_9.yaml`](react-agent/configs/eeg_research_v1_9.yaml) describes roles and policies; the role client and submitted goal determine runtime behavior.

## Current scope and evidence

| Area | Current implementation / boundary |
| --- | --- |
| Method retrieval | Local method cards with keyword overlap; no online literature backend |
| Candidate changes | Hooks for encoders, training statistics, transforms, objectives and evaluate-only reconstruction; execution requires registered available hooks and runtime approval |
| Evaluation | Frozen split, query/gallery, metric and image-feature identities; research decisions use validation evidence |
| Confirmation | Matched full-run seed pairs and an explicit policy; a short pilot does not confirm improvement |
| Memory | Episodes and conditional lessons; requested evidence levels cannot replace runtime evidence |
| Audit context | Claim draft and latest-result summary; a complete structured report/dependency manifest is not currently injected into the auditor call |
| Other tasks | Classification, image reconstruction, raw preprocessing and foundation-model adapters are currently unavailable |
| Cost | Calls and tokens are recorded; unknown API dollar cost remains unpriced |

Implementation and offline tests are separate from demonstrated real-data gains. The [canonical acceptance record](react-agent/docs/revision_acceptance_canonical.md) documents offline/CPU checks and marks fresh live-role and GPU confirmation runs as not run in that revision. It does not establish a new retrieval benchmark result.

## Repository and documentation

| Path | Purpose |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Figure semantics and implementation pointers |
| [`docs/figures/`](docs/figures/README.md) | Editable SVG sources and PNG exports |
| [`react-agent/src/react_agent/fmri/`](react-agent/src/react_agent/fmri/) | Screening, tools, generation, memory and workbench |
| [`react-agent/src/react_agent/eeg_research/agentic/`](react-agent/src/react_agent/eeg_research/agentic/) | Campaigns, eight role prompts, approval, jobs, evidence and export |
| [`react-agent/src/react_agent/eeg_training/`](react-agent/src/react_agent/eeg_training/) | Protocol, hooks, training and evaluation |
| [`react-agent/configs/`](react-agent/configs/) | Screening configs and example research policies / goals |
| [`react-agent/tests/`](react-agent/tests/) | Unit, workflow, recovery and integration checks |

Historical notes under `react-agent/docs/` describe their own revisions. Use current source and the [architecture map](docs/architecture.md) to interpret today's runtime.

## Development

From `react-agent/`, run the offline suites:

```bash
uv run python -m pytest \
  tests/unit_tests tests/fmri tests/eeg_research tests/eeg_training \
  --ignore=tests/eeg_research/test_live_roles.py
```

Live-role and integration checks require separate prerequisites. See [canonical acceptance](react-agent/docs/revision_acceptance_canonical.md) for what a recorded check establishes.

Secrets, environments, run outputs, weights, generation artifacts and memory databases stay outside git via [`.gitignore`](.gitignore). The repository contains source, tests, example configs, small atlases and frozen vectors.

## Acknowledgments

The conversation entry originated from the [LangGraph ReAct template](https://github.com/langchain-ai/react-agent). Retrieval builds on the Uncertainty-aware Blur Prior workflow; generation integrates an external TRIBE checkout. Optional screening providers include CortexMAE and CLIP, with their own resources and setup requirements.
