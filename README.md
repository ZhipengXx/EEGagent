# EEGagent

EEG image-retrieval experiments and an fMRI quality-check workbench, built beside a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct template.

Numeric checks describe consistency. They do not certify that a predicted response is biologically correct.

## Contents

- [Layout](#layout)
- [fMRI checks](#fmri-checks)
- [EEG retrieval](#eeg-retrieval)
- [Setup](#setup)
- [Environment](#environment)
- [Commands](#commands)
- [What stays local](#what-stays-local)
- [Tests](#tests)
- [Further reading](#further-reading)

## Layout

```text
EEGagent/
├── README.md
├── assets/                         # atlases and frozen vectors (weights are not committed)
│   ├── atlas/                      # Schaefer-400 on fsaverage5
│   ├── image_embeddings/clip/      # frozen CLIP vectors keyed by image id
│   ├── markers/
│   ├── hf/                         # local Hugging Face cache (gitignored)
│   ├── generation/                 # videos and predictions (gitignored)
│   └── memory/                     # sqlite memory store (gitignored)
└── react-agent/                    # Python package, configs, examples, docs
    ├── src/react_agent/
    │   ├── graph.py                # original ReAct chat graph
    │   ├── fmri/                   # inspection loop, TRIBE generation, workbench
    │   ├── eeg_research/           # campaign loop and code-level agent
    │   └── eeg_training/           # EEG/MEG retrieval training entry
    ├── configs/
    ├── examples/
    ├── docs/
    └── tests/
```

Run package commands from `react-agent/`. Paths in the YAML configs are resolved from that directory.

## fMRI checks

`react_agent.fmri` inspects surface time series and, when configured, image-conditioned predictions. The original ReAct graph in `graph.py` is unchanged.

| Piece | Role |
| --- | --- |
| `fmri.cli` | Checks, reference stats, TRIBE image-to-fMRI, memory export |
| `fmri.workbench` | Local UI on `127.0.0.1:8765` |
| `configs/fmri_check.yaml` | Numeric demo profile |
| `configs/fmri_check_tribe_*.yaml` | TRIBE diagnostic, agentic planning, and v1.4 coverage profiles |
| `fmri.planning` | Persistent plan. A planned policy does not silently fall back to rules |
| `fmri.memory` | SQLite episodes, separate from LangGraph checkpoints |

`claim_scope` for numeric screening stays `numeric_consistency_only`. Passing a check is not evidence that an image-conditioned response matches real fMRI.

Generation talks to an external TRIBE checkout through `TRIBE_PYTHON`. The repository path in config is relative (`../../tribev2` from `react-agent`). Cache and prediction files under `assets/generation/` are not committed.

## EEG retrieval

Two layers share the same package:

**Training entry** (`react_agent.eeg_training`). Four designs: EEG or MEG, intra-subject or inter-subject. The test split is not used for fitting. Without `EEG_GPU_SECONDS`, a run stays a dry probe.

**Research campaigns**

- `react_agent.eeg_research.cli` plans and records trials against a frozen task card (`configs/eeg_research_v1_6.yaml`).
- `react_agent.eeg_research.agentic.cli` runs a code-level campaign: propose a candidate, review it, and launch a training job. Default output root is `react-agent/runs/eeg_research_v18` (gitignored).

`mock_retrieval` is for contract tests. A mock pass does not set `real_training_passed`.

## Setup

Requires Python 3.11+.

```bash
cd react-agent
uv sync --group dev
cp .env.example .env
```

Optional weights for CortexMAE and CLIP:

```bash
uv sync --group dev --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
```

Downloaded weights land in `assets/hf/` and are not part of the git tree. See `assets/README.md`.

## Environment

Copy names from `react-agent/.env.example`. Put real values only in `.env`.

| Variable | Used for |
| --- | --- |
| `DEEPSEEK_API_KEY` | Planned fMRI policies and EEG research when the backend is DeepSeek |
| `DEEPSEEK_BASE_URL` | API base, default `https://api.deepseek.com` |
| `DEEPSEEK_FAST_MODEL` / `DEEPSEEK_REASONING_MODEL` | Profile model names |
| `TRIBE_PYTHON` | Interpreter that can import the TRIBE worker |
| `EEG_DATA_ROOT` | THINGS-EEG / THINGS-MEG preprocessed data |
| `EEG_TRAIN_PYTHON` | Torch interpreter for retrieval training |
| `EEG_GPU_SECONDS` | Required before a non-dry training run |
| `MODEL` | ReAct chat model (`provider/model-name`) |
| `TAVILY_API_KEY` | Only the original search tool in the ReAct template |

fMRI numeric checks do not need Tavily, Anthropic, or LangSmith keys. A planned DeepSeek policy refuses to start when `DEEPSEEK_API_KEY` is missing.

## Commands

From `react-agent`:

```bash
# Synthetic fixtures and a rule-only check
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok

# Image to prediction, then the existing checker
uv run python -m react_agent.fmri.cli check-image \
  --config configs/fmri_check_tribe_image16.yaml \
  --image /path/to/image.jpg \
  --out runs/image_check

# Local workbench
uv run python -m react_agent.fmri.workbench

# EEG campaign (offline planner unless EEG_RESEARCH_BACKEND=deepseek)
uv run python -m react_agent.eeg_research.cli plan \
  --config configs/eeg_research_v1_6.yaml \
  --out runs/eeg_research/campaign --dry-run

# One retrieval design, dry
uv run python -m react_agent.eeg_training.cli run \
  --dataset eeg --exp-setting intra-subject --subject sub-01 \
  --out runs/eeg_training/probe --dry-run

# Code-level research campaign
uv run python -m react_agent.eeg_research.agentic.cli create --campaign eeg_retrieval_research_v1
uv run python -m react_agent.eeg_research.agentic.cli status --campaign eeg_retrieval_research_v1
```

`fmri.cli` also provides `batch`, `from-tribe`, `fit-reference`, `prepare-assets`, `build-cohort-index`, `cache-embeddings`, `prepare-image-video`, `generate-from-image`, `batch-images`, `probe-backend`, `memory-inspect`, `memory-export`, and `record-feedback`.

## What stays local

[`.gitignore`](.gitignore) keeps these out of git:

| Path | Why |
| --- | --- |
| `react-agent/.env` | API keys and `TRIBE_PYTHON` |
| `react-agent/.venv/` | Local environment |
| `react-agent/runs/` | Campaign logs, checkpoints, workbench output |
| `react-agent/.langgraph_api/` | LangGraph local checkpoints |
| `assets/hf/` | Model weight cache |
| `assets/generation/` | Videos and predicted arrays |
| `assets/memory/*.sqlite3` | Episode database |
| `react-agent/.git.upstream-backup/` | Previous upstream clone metadata |

Committed on purpose: source, tests, `configs/`, `docs/`, small atlases, frozen CLIP vectors, and `react-agent/.env.example` (placeholders only).

## Tests

```bash
cd react-agent
uv run python -m pytest tests/unit_tests tests/fmri tests/eeg_research tests/eeg_training
```

`make test` in `react-agent` runs `tests/unit_tests/` only. Integration tests that call a live model are under `tests/integration_tests/` and are separate from the numeric fMRI suite.

## Further reading

Version notes live in `react-agent/docs/`:

| Doc | Topic |
| --- | --- |
| `fmri_check.md` | Numeric check boundaries |
| `tribe_tools_v1_1.md` | TRIBE sidecar tools |
| `tribe_image16_v1_2.md` | 16-second image protocol |
| `fmri_agentic_v1_3.md` | Planning and memory |
| `fmri_diagnostic_v1_4.md` | Coverage and follow-up routing |
| `fmri_workbench_ui_v1_5.md` | Workbench UI |
| `eeg_autoresearch_v1_6.md` | EEG research loop |
| `eeg_research_v1_8.md` | Code-level retrieval campaigns |

`react-agent/README.md` is the upstream ReAct template note. This file is the project overview.
