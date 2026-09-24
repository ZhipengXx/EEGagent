"""fMRI check usage, boundaries, and extension points."""

# fMRI check agent

This package adds a **pseudo-fMRI inspection loop** next to the original ReAct chat
graph. It does **not** require Tavily, Anthropic, or LangSmith keys.

Numeric screening passing **does not** prove that an image-conditioned real fMRI
response is correct. `claim_scope` is always `numeric_consistency_only`.

## Commands (from the repository root)

```bash
uv sync --group dev
python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
python -m react_agent.fmri.cli fit-reference \
  --manifest examples/fmri_demo/reference.jsonl \
  --out examples/fmri_demo/reference_stats.json \
  --config configs/fmri_check.yaml
python -m react_agent.fmri.cli batch \
  --manifest examples/fmri_demo/samples.jsonl \
  --config configs/fmri_check.yaml \
  --backend mock --policy hybrid \
  --out runs/demo
python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok
```

TRIBE v2 `static_1s_7s` sidecars can be converted without copying `.npy` files:

```bash
python -m react_agent.fmri.cli from-tribe \
  --preds-dir /path/to/things_static_1s_7s/preds \
  --ids apple_01b gray_control \
  --out examples/tribe_1s7s_smoke/samples.jsonl
```

A checked-in 8-sample smoke uses `configs/fmri_check_tribe_1s7s.yaml` (its own
reference stats, not the synthetic demo set):

```bash
python -m react_agent.fmri.cli batch \
  --manifest examples/tribe_1s7s_smoke/samples.jsonl \
  --config configs/fmri_check_tribe_1s7s.yaml \
  --backend none --policy rule \
  --out runs/tribe_1s7s_smoke
```

Passing there is numeric-contract only (`claim_scope=numeric_consistency_only`).

Rule mode (`--policy rule`) uses no LM backend and no API key.
`--backend deepseek` with `--policy hybrid` requires `DEEPSEEK_API_KEY` and
fails loudly if the key is missing.

LangGraph Studio (`uv run langgraph dev`, see `langgraph.json`):

- `fmri_pipeline`：可直接点跑。空表单会填 `apple_01b` + `fmri_check_tribe_image16.yaml`，写出 `runs/tribe_image16_studio_view`。图拆成 `apply_defaults` → `materialize` → `run_check`。state 看 `studio_summary` / `last_event`；stream 还会推 `[generate]` / `[tool]` / `[decision]` / `[stop]` 自定义事件。Studio **不是** 曲线/脑图看板。
- CLI `check-image` / `generate-from-image` 终端同步打这些一行进度（并写入 `events.jsonl`），最后仍有一行 verdict JSON。
- `fmri_check`：仍需 CLI `bind_runtime`，Studio 里直接点跑会失败。日常检查用 CLI。
- `agent`：原聊天图，未改。

## `.env` variables (credentials only)

- `DEEPSEEK_API_KEY` (required only for live DeepSeek)
- `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`)
- `DEEPSEEK_FAST_MODEL` / `DEEPSEEK_REASONING_MODEL` (default `deepseek-chat`)

Scientific thresholds live in `configs/fmri_check.yaml`, not in env vars.

Config priority: CLI flags > credential env vars > YAML > coded defaults.

## Implemented tools

| Tool | Status |
| --- | --- |
| validate_input | real |
| basic_statistics | real |
| temporal_diagnostics | real |
| roi_summary | real (needs `roi_map_path`) |
| reference_distribution | real (needs frozen stats from `fit-reference`) |
| cortex_mae | real CortexMAE-P; T=12 官方 encoder 拒收则 skip，不补零 |
| semantic_consistency | real independent CLIP RSA; 不用 TRIBE hf_cache |

`max_tool_calls` counts **required checks as well** as optional tools.

## Adding a tool

1. Implement the `CheckTool` protocol in `src/react_agent/fmri/tools/`.
2. Register it in `build_registry()`.
3. Enable it in YAML `enabled_tools`.
4. Do not change `graph.py` topology.

## Adding an LM backend

Implement `decide` / `summarize` like `DeepSeekBackend` or `MockBackend`.
Wire it in `LoopRuntime`. Numeric tools stay unchanged.

## Replacing routing

Implement a function with the same shape as `rule_decision` / hybrid in
`policy.py`. Data loading and `FinalReport` schema stay unchanged.

## CortexMAE / CLIP RSA

Weights live under `/home/zxuff/data/EEGagent/assets`. Official `run_embedding` pads short clips; this package calls `encoder.forward_embedding` instead and skips if T≠16. CLIP RSA uses a frozen independent encoder, not TRIBE `hf_cache`. Neither result is a validity probability.
