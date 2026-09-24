# fMRI V1.3：planned 策略、Memory 与 16s 面板

实现状态文档。`claim_scope` 仍是 `numeric_consistency_only`。不改 V1.0–1.2 快照，不覆盖旧 `runs/tribe_1s7s_*` / `runs/tribe_image16_rule_smoke`。

## 怎么跑

一律在仓库根目录 `uv run`。

```bash
# 验证 DeepSeek JSON（计费，不是免费）
uv run python -m react_agent.fmri.cli probe-backend \
  --config configs/fmri_check_tribe_image16_agentic.yaml \
  --out runs/tribe_image16_planned_v1_3_probe

# planned：L0 后由 DeepSeek 写持久计划；复用 16s 缓存，不重跑 TRIBE
uv run python -m react_agent.fmri.cli check-image \
  --image /path/to.jpg \
  --config configs/fmri_check_tribe_image16_agentic.yaml \
  --policy planned --backend deepseek \
  --out runs/tribe_image16_planned_v1_3_smoke

# 旧 rule 仍强制 backend=none
uv run python -m react_agent.fmri.cli check-image \
  --image /path/to.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --policy rule --backend none \
  --out runs/tribe_image16_rule_keep
```

Studio：空表单仍是旧 image16 + rule。把 `config_path` 换成 `configs/fmri_check_tribe_image16_agentic.yaml` 后，policy/backend 跟配置（planned + deepseek），不会静默 rule。看进度行 / `studio_summary` / `plan.json`。

Memory：`memory-inspect`、`memory-export`、`record-feedback`。反馈必须带 status / evidence / source。

## Prompt 与 schema

| 文件 | 角色 |
| --- | --- |
| `src/react_agent/fmri/prompts/planner_v1.md` | planner/replanner system prompt |
| `src/react_agent/fmri/prompts/memory_curator_v1.md` | curator；不改 verdict |
| `src/react_agent/fmri/planning/schemas.py` | `PlanProposal` / `Guard` |
| `src/react_agent/fmri/memory/rules/procedural_v1.md` | 版本化规则；curator 不能写 |

运行时把真实 `PlanProposal` JSON Schema 和预算追加到 user 消息。  
发给模型的 system prompt（planner、memory curator、hybrid decide/summary、probe）均为英文。

## 验收状态

| 项 | 状态 | 证据 |
| --- | --- | --- |
| probe-backend 真实 JSON | **real_passed** | `runs/tribe_image16_planned_v1_3_probe`；requested `deepseek-chat`，response `deepseek-flash`；`billed=true` |
| planned 不被改成 none | **implemented** / 单测 `test_planned_not_rewritten_to_none` | `resolve_policy_backend` |
| 缺 key 清晰失败 | **implemented** | `ValueError: DEEPSEEK_API_KEY` |
| 真实 create_plan + 按计划执行 | **real_passed** | `runs/tribe_image16_planned_v1_3_accepted`：`accepted_plan_count=1`，`fallback_used=false`，`degraded_execution=false` |
| revise_plan | **mock_passed**（validator 单测）；live 曾因重规划校验失败 | 不写 real_passed |
| CortexMAE 无参考不强制 | **real_passed** | accepted 轨迹未跑 `cortex_mae`；rule 对比会跑 |
| 16s 面板 | **implemented** | `examples/tribe_image16_panel/`，N=3 图 + gray；`quality_verdict_enabled=false` |
| episode 写入 | **real_passed** | SQLite + `episode.json`；`verification_status=unverified` |
| 同输入第二轮 | **real_passed（检索）** | `memory2`：`prior_runs` 命中 apple 历史；`hit_count=0`（不作独立参考）；当次 create 校验失败 → 路径未改 |
| 跨样本 memory 改变路径 | **not_run** | 未为展示强迫复用异常标签 |
| 旧 rule / 12s / 16s 协议 | **implemented** | 55 个 `tests/fmri` 通过；原 image16 YAML 未改 |

## 一条完整轨迹（accepted）

样本：缓存 `apple_01b`，`[16,20484]`，0 条新 TRIBE prediction。

1. L0：`validate_input`、`basic_statistics`（必做，无 API）。
2. Memory：cold，`hits=0`。
3. DeepSeek create_plan 成功并通过 validator。
4. 执行 `temporal_diagnostics`（回答 temporal，描述性），再执行 `gray_control_contrast`（`overall_delta_rms≈0.050`，`comparable`）。
5. 必做问题已 described，停止。`screening_decision=pass_configured`，legacy `verdict=passed_configured_checks`。
6. **未升级 CortexMAE / RSA / reference**（无兼容质量参考；embedding 不能当质量证据）。
7. 费用（该次）：`lm_calls=2`（含一次 repair 计数），`planner_calls_succeeded=1`，`accepted_plan_count=1`，`input_tokens=9158`，`output_tokens=1547`，`api_usd=null`（无价格表）。

为何停：required questions 已按「已描述」完成，没有 flag/block，计划无剩余 ready step。这不是生物效度通过。

## rule vs planned（同一缓存图）

| | planned accepted | rule + agentic YAML |
| --- | --- | --- |
| 决策 | DeepSeek 计划 | 离线队列 |
| 可选工具 | temporal_diagnostics + gray | gray、temporal_profile、surface ROI、specificity、cortex_mae、semantic… 直到 `max_tool_calls` |
| CortexMAE | 不跑 | 跑了（rule 见到候选就上） |
| 停止 | configured_checks_complete | max_tool_calls |
| LM | 2 calls | 0 |

小 N 工程对比：planned 更少升级昂贵工具。命中率不是效果指标。没有独立故障标注，不算 precision/recall。

## 哪些能筛、哪些只是描述

可筛（已有确定性 finding）：畸形 shape / 非有限、全常数、`duplicate_output`（exact hash）、资源不匹配（拒绝该比较）。

仅描述：gray RMS、peak_frame、ROI top-K、RSA、CortexMAE embedding、N=3 参考的 robust z（`quality_verdict_enabled=false`）。

## 主要改动文件

- 新：`planning/*`、`memory/*`、`prompts/planner_v1.md`、`prompts/memory_curator_v1.md`、`configs/fmri_check_tribe_image16_agentic.yaml`、`examples/tribe_image16_panel/`、`tests/fmri/test_planning.py`、`test_memory.py`、`test_agentic_policy.py`
- 扩：`config.py`、`cli.py`、`loop.py`、`policy.py`、`ledger.py`、`reporting.py`、`budget.py`、`schemas.py`、`pipeline.py`、`pipeline_graph.py`、`llm/deepseek.py`、`llm/mock.py`、工具 `ToolSpec` 元数据
