# version 1.3：planned 检查器如何工作

这是仓库在 **version 1.3** 时的实现快照，不是提案。  
不替代 [`version/version_1.2/agent.md`](../version_1.2/agent.md)。详细验收见 [`docs/fmri_agentic_v1_3.md`](../../docs/fmri_agentic_v1_3.md)。

`claim_scope` 永远是 `numeric_consistency_only`。

---

## 1. 相对 1.2 多了什么

- 新策略 **`planned`**：L0（validate + basic）之后检索 memory，DeepSeek 产出持久 `Plan`，校验后一次执行一个 ready step；条件可判定则不再打 API。
- 新配置 [`configs/fmri_check_tribe_image16_agentic.yaml`](../../configs/fmri_check_tribe_image16_agentic.yaml)：必做只有 `validate_input`、`basic_statistics`。原 image16 YAML 仍把 CortexMAE 当必做。
- SQLite memory：确定性 episode；可选 curator；unverified 不能升格。
- 16s 冻结面板：`examples/tribe_image16_panel/`（3 张缓存图 + gray）。N 小，不是健康脑参考。
- `probe-backend`、`memory-inspect` / `export` / `record-feedback`。

rule / hybrid 行为保留。`policy=rule` 仍强制 `backend=none`。**planned 不会被改成 none。**  
发给模型的 system prompt（含 planner 与 memory curator）为英文。

---

## 2. 三张图（未改拓扑）

`agent` 聊天图未改。`fmri_check` 仍是 ingest → required → evidence ⇄ select → validate → execute → finalize。`fmri_pipeline` Studio 空表单默认旧 rule；`config_path` 指向 agentic YAML 时跟配置走 planned。

---

## 3. 如何执行

```bash
uv run python -m react_agent.fmri.cli probe-backend \
  --config configs/fmri_check_tribe_image16_agentic.yaml

uv run python -m react_agent.fmri.cli check-image \
  --image /home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/Image_set/training_images/00003_accordion/accordion_01b.jpg \
  --config configs/fmri_check_tribe_image16_agentic.yaml \
  --policy planned --backend deepseek \
  --out runs/tribe_image16_planned_v1_3_test
```

缺 `DEEPSEEK_API_KEY` 会退出，不静默回退 rule。已有 16s 缓存可复用，接通 planner 不强制重跑 TRIBE。

进度：`[generate]` / `[tool]` / `[api]` / `[plan]` / `[memory]` / `[decision]` / `[stop]`。

---

## 4. 检查指标（与 1.2 相同工具，调度不同）

L0 必做：`validate_input`、`basic_statistics`。  
L1 默认：`gray_control_contrast`、`stimulus_temporal_profile` / `temporal_diagnostics`。  
L2：`surface_roi_profile`、`cross_image_specificity`（需 atlas / 16s cohort）。  
L3：`cortex_mae`、`reference_distribution`、`semantic_consistency` — 无兼容参考时 **不能** 当质量证据；planner 不得因“高阶模型”强制执行。

描述性工具的 `decision_effect` 仍是 `none`。planner 不能发明拒绝阈值。

---

## 5. API / Memory

| 模式 | LLM |
| --- | --- |
| rule | 无 |
| hybrid + deepseek | 每步 JSON Action |
| planned + deepseek | planner / 有界 repair / 可选 replan / 可选 curator；同一预算账本 |

真实 accepted 烟雾（`runs/tribe_image16_planned_v1_3_accepted`）：create_plan 成功，执行 temporal_diagnostics + gray_control_contrast，**不跑 CortexMAE**，`screening_decision=pass_configured`，`fallback_used=false`。tokens 9158+1547，`api_usd=null`。

同输入第二轮会看到 `prior_runs`，但不能把同输入历史当独立参考。跨 profile 12 点案例不会命中 16s 检索。

---

## 6. 未完成限制

- 16s 参考 N=3，`engineering_smoke`，`quality_verdict_enabled=false`。
- live `revise_plan` 未标 real_passed（校验失败后保持旧计划 / 单测覆盖）。
- 无独立故障标注，不做 precision/recall 声明。
- CortexMAE 仍无 HCP 参考分。
