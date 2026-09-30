# EEGagent runtime

This directory contains EEGagent's fMRI screening workflow, EEG / MEG retrieval research runtime, training worker and local workbench.

Start with the [project README](../README.md) for the overview, collaboration figures, setup and workflow commands. The [architecture map](../docs/architecture.md) connects the figures to the current implementation.

## Setup and offline demo

Run these commands from this directory:

```bash
uv sync --group dev
cp .env.example .env

uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok

uv run python -m react_agent.fmri.workbench
```

The workbench is available at [http://127.0.0.1:8765](http://127.0.0.1:8765).

## Entry points

| Module / graph | Purpose |
| --- | --- |
| `react_agent.fmri.cli` / `fmri_check` | Screen an existing series |
| `react_agent.fmri.cli check-image` / `fmri_pipeline` | Generate when configured, then screen |
| `react_agent.fmri.workbench` | Local interface for both workflows |
| `react_agent.eeg_research.agentic.cli` / `eeg_research` | Create, control and inspect code-level campaigns |
| `react_agent.eeg_training.cli` | Inspect or run a retrieval design |
| `react_agent.graph` / `agent` | Separate ReAct conversation entry |

[`langgraph.json`](langgraph.json) registers the Studio views. The EEG view wraps an existing campaign tick as load → step → summarize; it does not replace the campaign runtime.

## Further reading

- [Architecture and handoffs](../docs/architecture.md)
- [fMRI screening configuration](docs/fmri_check.md)
- [Planned screening](docs/fmri_agentic_v1_3.md)
- [Campaign commands and recovery](docs/eeg_research_v1_8.md) — historical revision; compare its limitations with current source
- [Canonical acceptance and evidence boundaries](docs/revision_acceptance_canonical.md)

The ReAct entry originates from the [LangGraph ReAct template](https://github.com/langchain-ai/react-agent). Its optional search and model configuration are separate from the two research workflows.
