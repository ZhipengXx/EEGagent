# EEGagent

EEGagent is a multi-agent workflow for EEG-to-image retrieval and fMRI screening. It sits on a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct template and runs two research loops: one that screens fMRI predictions, and one that proposes, reviews, and trains EEG retrieval models.

## Contents

- [Workflow](#workflow)
- [Repository](#repository)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Commands](#commands)
- [What stays local](#what-stays-local)
- [Tests](#tests)

## Workflow

Two agent loops share one package. A separate ReAct chat graph in `react-agent/src/react_agent/graph.py` remains available for tool-calling conversation. It is not part of either research loop.

### fMRI screening

The planner chooses the next check. Tools run numeric screening and, when configured, TRIBE image-to-fMRI generation. Memory stores episodes in SQLite, apart from LangGraph checkpoints. The workbench is a local UI at `127.0.0.1:8765`.

Profiles live in `react-agent/configs/`. `fmri_check.yaml` is the numeric demo. `fmri_check_tribe_*.yaml` covers TRIBE diagnostics, planned screening, and the image-16 protocol. Generation calls an external TRIBE checkout through `TRIBE_PYTHON`. From `react-agent`, that checkout is the relative path `../../tribev2`.

### EEG retrieval

The planner proposes a candidate change. The coder writes the patch. The reviewer checks it. The training worker then launches retrieval training under a frozen protocol.

Designs are EEG or MEG, intra-subject or inter-subject. The test split is held out of fitting. A run without `EEG_GPU_SECONDS` stays a dry probe. Campaign output goes to `react-agent/runs/` and is not committed.

## Repository

Commands run from `react-agent/`. YAML paths resolve from that directory.

```text
EEGagent/
├── README.md
├── assets/                         # atlases and frozen vectors
│   ├── atlas/                      # Schaefer-400 on fsaverage5
│   ├── image_embeddings/clip/      # CLIP vectors keyed by image id
│   ├── markers/
│   ├── hf/                         # weight cache, not committed
│   ├── generation/                 # videos and predictions, not committed
│   └── memory/                     # episode database, not committed
└── react-agent/
    ├── src/react_agent/
    │   ├── graph.py                # ReAct chat graph
    │   ├── fmri/                   # fMRI planner, tools, memory, workbench
    │   ├── eeg_research/           # EEG planner, coder, reviewer
    │   └── eeg_training/           # retrieval training worker
    ├── configs/
    ├── examples/
    ├── docs/
    └── tests/
```

## Quick start

Python 3.11 or newer.

```bash
cd react-agent
uv sync --group dev
cp .env.example .env
```

Fill in `.env`, then open the workbench or run a demo check:

```bash
uv run python -m react_agent.fmri.workbench
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
```

Optional CortexMAE and CLIP weights:

```bash
uv sync --group dev --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
```

Weights download into `assets/hf/` and stay out of git. See `assets/README.md`.

## Configuration

Copy names from `react-agent/.env.example`. Real values belong only in `.env`.

| Variable | Used for |
| --- | --- |
| `DEEPSEEK_API_KEY` | Planned fMRI runs and DeepSeek-backed EEG research |
| `DEEPSEEK_BASE_URL` | API base, default `https://api.deepseek.com` |
| `DEEPSEEK_FAST_MODEL` / `DEEPSEEK_REASONING_MODEL` | Model names for the two profiles |
| `TRIBE_PYTHON` | Interpreter for the TRIBE generation worker |
| `EEG_DATA_ROOT` | THINGS-EEG / THINGS-MEG preprocessed data |
| `EEG_TRAIN_PYTHON` | Torch interpreter for retrieval training |
| `EEG_GPU_SECONDS` | Budget required before a real training run |
| `MODEL` | ReAct chat model, as `provider/model-name` |
| `TAVILY_API_KEY` | Search tool on the ReAct chat graph only |

The fMRI and EEG loops do not need Tavily, Anthropic, or LangSmith keys. A planned DeepSeek run needs `DEEPSEEK_API_KEY`.

## Commands

From `react-agent`.

### fMRI

```bash
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok

uv run python -m react_agent.fmri.cli check-image \
  --config configs/fmri_check_tribe_image16.yaml \
  --image /path/to/image.jpg \
  --out runs/image_check

uv run python -m react_agent.fmri.workbench
```

The same CLI also covers `batch`, `from-tribe`, `fit-reference`, `prepare-assets`, `build-cohort-index`, `cache-embeddings`, `prepare-image-video`, `generate-from-image`, `batch-images`, `probe-backend`, `memory-inspect`, `memory-export`, and `record-feedback`.

### EEG

```bash
uv run python -m react_agent.eeg_research.cli plan \
  --config configs/eeg_research_v1_6.yaml \
  --out runs/eeg_research/campaign --dry-run

uv run python -m react_agent.eeg_training.cli run \
  --dataset eeg --exp-setting intra-subject --subject sub-01 \
  --out runs/eeg_training/probe --dry-run

uv run python -m react_agent.eeg_research.agentic.cli create --campaign eeg_retrieval_research_v1
uv run python -m react_agent.eeg_research.agentic.cli status --campaign eeg_retrieval_research_v1
```

Set `EEG_RESEARCH_BACKEND=deepseek` when the research planner should call DeepSeek. Otherwise it stays offline.

## What stays local

[`.gitignore`](.gitignore) keeps secrets, caches, and run output off the remote:

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
| `react-agent/version/` | Local version notes |
| `react-agent/src/react_agent/version/` | Local cursor prompts |

The repository does include source, tests, `configs/`, `docs/`, small atlases, frozen CLIP vectors, and `react-agent/.env.example`.

## Tests

```bash
cd react-agent
uv run python -m pytest tests/unit_tests tests/fmri tests/eeg_research tests/eeg_training
```

`make test` runs `tests/unit_tests/` only. Live-model checks sit in `tests/integration_tests/` and are separate from the fMRI and EEG suites.
