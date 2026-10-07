# Running EEGagent

Start with the [project README](../README.md) for the research workflows. This guide collects setup, commands, configuration and the current execution boundaries.

## Install and open the workbench

Use Python **3.11 or newer** and `uv`. From a fresh checkout:

```bash
git clone https://github.com/ZhipengXx/EEGagent.git
cd EEGagent/react-agent
uv sync --group dev
cp .env.example .env
```

### Open the research workbench

```bash
uv run python -m react_agent.fmri.workbench
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The default entry is autonomous code-level research, with process, experiment and code / artifact views. The fMRI screening and standalone comparison workflows remain available in navigation.

For an interface preview, open [the explicit platform demo](http://127.0.0.1:8765/?demo=1#research/demo_ui_workspace). It writes synthetic records under `runs/eeg_research_ui_demo/`, without an LLM call or training job. Demo values are not experimental evidence. The [platform gallery](platform.md) distinguishes the checked-in screenshots from the latest interface.

### Run the offline numeric screening demo

This fMRI example uses the rule policy and requires neither an LLM API call nor TRIBE generation:

```bash
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo

uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok
```

Inspect the report in the workbench's fMRI screening view.

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

For planned screening, use [`fmri_check_tribe_image16_agentic.yaml`](../react-agent/configs/fmri_check_tribe_image16_agentic.yaml) with a configured backend and API key. Optional CortexMAE and CLIP weights can be prepared separately:

```bash
uv sync --group dev --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
```

See [asset setup](../assets/README.md) and [screening architecture](architecture.md#fmri-screening-collaboration).

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

New goals default to `planner_mode=compare_options`: the planner normally compares 2–4 next-action options in one response and records the alternatives, selected option and rationale. One is allowed when only one is reasonable. Runtime checks precede dispatch of the selected action. Use `--planner-mode single_action` on `create` only when the legacy interface is intended. An existing campaign's persisted goal remains authoritative when resumed.

### Inspect an exported candidate pack

```bash
uv run python -m react_agent.eeg_research.agentic.cli evaluate-pack \
  --pack /path/to/candidate_pack \
  --data-root /path/to/data --out runs/pack_eval
```

The default prints an evaluation command without fitting. To execute evaluation, configure `EEG_ALLOW_EVALUATE_PACK=1` and add `--execute`. A pack's existence alone does not establish successful reconstruction or evaluation; verify reuse with an actual evaluate-only run.

## Configuration

Copy [`.env.example`](../react-agent/.env.example) and keep real values local. Export machine-specific values for CLI processes as needed.

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

The native code-level worker requires the configured LLM backend. Offline rule screening and dry probes are separate entry points. Sample [`eeg_research_v1_9.yaml`](../react-agent/configs/eeg_research_v1_9.yaml) describes roles and policies; the role client and submitted goal determine runtime behavior.

## Current scope and evidence

| Area | Current implementation / boundary |
| --- | --- |
| Method retrieval | Local method cards with keyword overlap; no online literature backend |
| Candidate changes | Hooks for encoders, training statistics, transforms, objectives and evaluate-only reconstruction; execution requires registered available hooks and runtime approval |
| Evaluation | Frozen split, query/gallery, metric and image-feature identities; research decisions use validation evidence |
| Confirmation | Matched full-run seed pairs and an explicit policy; a short pilot does not confirm improvement |
| Memory | Episodes and conditional lessons; requested evidence levels cannot replace runtime evidence |
| Planner | Single-call next-action comparisons with exact targets, evidence, prerequisites and resource estimates; runtime gates still determine executability |
| Artifact reads | Bounded development views with verified hashes, page identities and recorded consumer delivery; a read receipt is not experimental evidence |
| Audit context | Structured report draft, development dependency manifest and verified analyses; identity / dependency freshness checks govern reusable feedback and tracked corrections |
| Other tasks | Classification, image reconstruction, raw preprocessing and foundation-model adapters are currently unavailable |
| Cost | Calls and tokens are recorded; unknown API dollar cost remains unpriced |

Implementation and offline tests are separate from demonstrated real-data gains. Historical acceptance records document offline/CPU checks and identify which fresh live-role and GPU confirmation runs were not executed. Those records describe their own revisions and do not establish a new retrieval benchmark result.

## Local checks and historical validation

The current public tree does not include the earlier `tests/` suites. For a public checkout, begin with the offline numeric screening example above and validate the example retrieval goal without launching a worker:

```bash
uv run python -m react_agent.eeg_research.agentic.cli validate-goal \
  --goal configs/goals/eeg_retrieval_v1_9.yaml
```

These commands inspect configuration and numeric screening behavior; they do not replace a full regression suite, live-role validation or a GPU retrieval study. Historical acceptance documents retain the commands and outcomes for their recorded revisions. See the [architecture map](architecture.md) for the current runtime and audit context.
