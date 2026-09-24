# EEGagent P2 assets

Weights, atlases, and frozen embeddings live here — not inside react-agent.

## Layout

- `hf/` — Hugging Face cache (`HF_HOME`) for CortexMAE-P and CLIP
- `atlas/schaefer400_fsaverage5/` — CBIG fsaverage5 Schaefer-400 as LH||RH npy
- `image_embeddings/clip/` — frozen CLIP vectors keyed by image_id

## Commands (from react-agent, `uv run`)

```bash
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
uv run python -m react_agent.fmri.cli cache-embeddings \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl
```

## Rules

- Do not silently pad T=12 to CortexMAE's trained T=16.
- Do not use TRIBE `hf_cache_*` as an independent image encoder.
- Embeddings are descriptive. They are not a validity probability.
