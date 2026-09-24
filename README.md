# EEGagent

_A research workflow, not another chat template. What matters is the loop: plan, check, review, then train._

> Two agent loops share one package, built beside a [LangGraph](https://github.com/langchain-ai/langgraph) ReAct graph.
>
> **fMRI screening.** The planner picks the next check. Tools run numeric screening and, when configured, TRIBE image-to-fMRI generation. Memory stores episodes. The workbench is the local UI.
>
> **EEG retrieval.** The planner proposes a change. The coder writes the patch. The reviewer checks it. The training worker launches retrieval training under a frozen protocol.
>
> The ReAct graph in `graph.py` stays a conversation entry. It is not a third research loop.

## Contents

1. [Workflows](#1-workflows)
2. [Quick start](#2-quick-start)
3. [Repository](#3-repository)
4. [Configuration](#4-configuration)
5. [What stays local](#5-what-stays-local)
6. [Tests](#6-tests)

---

## 1. Workflows

Run these from `react-agent/`. YAML paths resolve from that directory.

**fMRI screening** — inspect a surface series, or generate a prediction from an image and then screen it.

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

`fmri_check.yaml` is the numeric demo. `fmri_check_tribe_*.yaml` covers TRIBE diagnostics, planned screening, and the image-16 protocol. Generation calls an external TRIBE checkout through `TRIBE_PYTHON` (`../../tribev2` from `react-agent`). The workbench listens on `127.0.0.1:8765`.

The same CLI also covers `batch`, `from-tribe`, `fit-reference`, `prepare-assets`, `build-cohort-index`, `cache-embeddings`, `prepare-image-video`, `generate-from-image`, `batch-images`, `probe-backend`, `memory-inspect`, `memory-export`, and `record-feedback`.

**EEG retrieval** — propose a candidate, review the patch, then train EEG or MEG, intra-subject or inter-subject. The test split stays out of fitting. Without `EEG_GPU_SECONDS` the run stays a dry probe.

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

Set `EEG_RESEARCH_BACKEND=deepseek` when the planner should call DeepSeek. Otherwise it stays offline. Campaign output goes to `react-agent/runs/` and is not committed.

## 2. Quick start

Python 3.11 or newer.

```bash
cd react-agent
uv sync --group dev
cp .env.example .env
```

Fill in `.env`, then open the workbench or build the demo fixtures:

```bash
uv run python -m react_agent.fmri.workbench
uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
```

Optional CortexMAE and CLIP weights:

```bash
uv sync --group dev --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
```

Weights land in `assets/hf/` and stay out of git. See `assets/README.md`.

## 3. Repository

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
    │   ├── fmri/                   # planner, tools, memory, workbench
    │   ├── eeg_research/           # planner, coder, reviewer
    │   └── eeg_training/           # retrieval training worker
    ├── configs/
    ├── examples/
    ├── docs/
    └── tests/
```

## 4. Configuration

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

The research loops do not need Tavily, Anthropic, or LangSmith keys. A planned DeepSeek run needs `DEEPSEEK_API_KEY`.

## 5. What stays local

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

## 6. Tests

```bash
cd react-agent
uv run python -m pytest tests/unit_tests tests/fmri tests/eeg_research tests/eeg_training
```

`make test` runs `tests/unit_tests/` only. Live-model checks sit in `tests/integration_tests/` and are separate from the fMRI and EEG suites.
