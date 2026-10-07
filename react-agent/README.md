# EEGagent runtime

This directory contains EEGagent's fMRI screening workflow, EEG / MEG retrieval research runtime, training worker and local workbench.

Start with the [project README](../README.md) for the research story and collaboration figures. The [running guide](../docs/running.md) collects setup and workflow commands; the [architecture map](../docs/architecture.md) connects the figures to the current implementation.

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
- [Workflow commands and configuration](../docs/running.md)
- [fMRI screening configuration](docs/fmri_check.md)
- [Planned screening](docs/fmri_agentic_v1_3.md)
- [Campaign commands and recovery](docs/eeg_research_v1_8.md) — historical revision; compare its limitations with current source
- [Canonical acceptance and evidence boundaries](docs/revision_acceptance_canonical.md)

The ReAct entry originates from the [LangGraph ReAct template](https://github.com/langchain-ai/react-agent). Its optional search and model configuration are separate from the two research workflows.


## Optional campaign memory: MiniLM and procedural skills

Campaign memory is opt-in. Existing campaigns, empirical lessons, approvals,
training hashes, method cards and the fMRI memory remain on their original paths.
No embedding model is imported or loaded and no semantic database is opened when
`memory` is absent or `enabled` is false.

Add a `memory` block when creating a new campaign goal. The immutable revision
below was verified with the real model on CPU:

```json
{
  "memory": {
    "enabled": true,
    "skills_enabled": true,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "device": "cpu",
    "batch_size": 32,
    "top_k": 5,
    "min_similarity": 0.30,
    "context_chars_budget": 6000,
    "allow_model_download": false,
    "fallback": "legacy"
  }
}
```

The optional dependency is `react-agent[memory]` (`uv sync --extra memory` for a
fresh project environment). Check the dependency resolution before changing an
existing training environment; use an isolated environment if its Torch or
Transformers versions would change. The local CPU validation used
sentence-transformers 5.7.0, Transformers 4.57.6 and the unchanged project Torch
2.12.1. `PYTHONPATH=src` allows an isolated interpreter to run the current source.

From `react-agent/`, set `MEMORY_PYTHON` to the interpreter with the optional
memory dependency and `CAMPAIGN_DIR` to one EEG campaign directory. These
maintenance commands do not modify the goal, record scientific evidence or start
an LLM/training worker:

```bash
# Explicit first model preparation and indexing; weights are downloaded only here.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  index --campaign "$CAMPAIGN_DIR" --allow-model-download --require-embedding

# Subsequent commands use the local model cache. Unchanged corpus rows are reused.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  backfill --campaign "$CAMPAIGN_DIR" --require-embedding

PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  query --campaign "$CAMPAIGN_DIR" --text "Diagnose EEG and image feature alignment" \
  --require-embedding

# Explicitly replace this model fingerprint's corrupt/changed cache entries.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  rebuild --campaign "$CAMPAIGN_DIR" --require-embedding
```

`--model-revision` can supply an immutable revision for an explicit command;
`--retry` clears a process-local model failure before retrying. A fresh process
also retries. Normal runtime requests never download weights unless explicitly
configured. Missing weights/dependencies produce `backend=legacy`, `degraded=true`
and a reason; the planner retains its original lessons. Strict commands fail
instead of reporting a successful MiniLM result. Index reports contain
`indexed/reused/pending/failed`; indexing errors leave accepted source records
intact for a later backfill.

The new sidecar is `CAMPAIGN_DIR/semantic_memory.sqlite`. It contains `skills`,
`embedding_cache` and `frozen_delivery`; it does not duplicate the existing
`memory.sqlite` episodes, lessons or curation tables. Vectors are 384-dimensional,
normalized little-endian float32 BLOBs (1536 bytes). Cache identity includes the
immutable model revision, tokenizer vocabulary/configuration, actual sequence
limit, normalization and text-builder version. Input token counts and truncation
are recorded using the tokenizer. A changed body/version or fingerprint cannot
reuse stale output; explicit rebuild replaces invalid vectors.

Existing lesson applicability rules group candidates before cosine ranking.
Lessons and procedural hints share one total Top-k and one serialized context
budget. Compatible lessons have priority over analogy/unknown items; similarity
never upgrades evidence strength. Oversized bodies are omitted with truncation
metadata. An empty result is valid. Original lesson languages are retained;
MiniLM is English-focused and there is no automatic translation.

The planner receives selected lessons once plus memory identities/scores. The
designer and coder receive their bounded bodies in `retrieved_memory`. Curator
skills are extracted during the existing post-analysis call using an extended
schema and the same validator. Task/episode/artifact identities and hashes are
checked against the injected development-source manifest before preserving the
complete procedure, preconditions and limits. These are source-validated hints,
not confirmation of every proposed step or a performance improvement. Empirical
findings still pass through the original lesson acceptance rules.

Coder memory is outside the approved `experiment` object, covered by its request
digest, and frozen by candidate/attempt and approved spec identity. A restart
reuses the same ID/version/hash/body without replaying completed tools. Existing
in-flight attempts without semantic input are not retrofitted. Indexing or skill
storage failure does not rerun the paid curator or invalidate a completed
analysis.

To disable the hooks, use `"memory": {"enabled": false,"skills_enabled": false}`.
This stops new memory delivery; it does not erase stored skills or vectors. This
version searches only the current EEG/MEG campaign. Cross-campaign sharing and
semantic method-card retrieval are outside its scope. Final-test/final-holdout,
fMRI and foreign-campaign sources are excluded before indexing or skill delivery.
