# TRIBE 检查工具 V1.1

增量相对 version 1.0：四个可执行领域工具、显式资源绑定、question ledger、hybrid 在多个候选间选择。  
`claim_scope` 仍是 `numeric_consistency_only`。不声称真实脑响应或图像语义正确。

不改 `graph.py` 拓扑，不覆盖 `runs/tribe_1s7s_smoke`，不改 `version/version_1.0/agent.md`。

## 验收状态

| 项 | implemented | synthetic_passed | real_data_passed |
| --- | --- | --- | --- |
| `gray_control_contrast` | 是 | 是 | 是（8 条 rule；显式 `gray_control.npy`） |
| `stimulus_temporal_profile` | 是 | 是 | 是（contrast 模式） |
| `surface_roi_profile` | 是 | 是 | **是（Nilearn Destrieux）**：`runs/tribe_1s7s_v1_1_rule_destrieux`，`source=nilearn.fetch_atlas_surf_destrieux` |
| `cross_image_specificity` | 是 | 是 | 是（8 条面板，`n_compared=7`） |
| N=3 参考不再越权 | 是 | 是 | 是（旧 flagged 三条现为 `passed` + `insufficient_reference`） |
| hybrid 多候选 + 二次决策见新证据 | 是 | 是（mock） | 是（DeepSeek 1 条，`fallback=false`） |
| `cortex_mae`（V1.1 / T=12 旧 8 条） | 是（CortexMAE-P + Schaefer-400） | 是 | **本页 not_run（T=12 被拒）**。T=16 真实 raw forward 已在 V1.2：`docs/tribe_image16_v1_2.md`，`runs/tribe_image16_rule_smoke` |
| `semantic_consistency` | 是（独立 CLIP RSA） | 是 | **是（独立 CLIP 缓存，非 TRIBE hf_cache）**：`apple_01b` spearman_rdm≈0.0065，N=7 |
| 真实 Destrieux / nilearn 下载 | 是 | — | **是**（nilearn 0.14.1，`examples/tribe_1s7s_smoke/atlas_destrieux/`） |
| P1 spatial sanity | 未实现 | — | — |

全程 fallback **不**算模型路由验证。本次 DeepSeek：1 样本、3 次 LM（2 decide + 1 decide/stop）、0 fallback。

## 修过的 V1 问题

- N=3 smoke 参考只出描述，`decision_effect=none`，不能把样本打成质量 `flagged`。
- 参考集内 fingerprint 重叠 → `reference_unassessed`，coverage=`partial`。
- Finding 增加 `decision_effect`；verdict 只认 `flag` / `block`。
- fmri_check DeepSeek 是 **JSON Action**，不是聊天图的原生 `tool_calls`。保留 `deepseek-chat`。

## 工具契约（本轮实现）

**T1 `gray_control_contrast`**  
输入：Y[T,V] + YAML 显式 `gray_control`。不按文件名猜。  
输出：`contrast_minus_gray.npy`、RMS 曲线。gray 自身 `not_applicable`。  
不能输出 SNR / p-value。

**T2 `stimulus_temporal_profile`**  
优先分析已有 contrast，否则 `raw_temporal_description`。不硬编码 5s 峰值。

**T3 `surface_roi_profile`**  
LH\|\|RH，主键 `hemisphere:label_id`。真实 atlas：`prepare-assets --download` 写入 `atlas_destrieux/`（`atlas_id` 不变）。`examples/tribe_1s7s_smoke/atlas/` 的 `source=synthetic_test` 只留测试，**不能**冒充真实 Destrieux。top-K 只描述，不要求视觉区最高。

**T4 `cross_image_specificity`**  
固定 cohort 面板。排除 self。exact hash → `duplicate_output`。不是 semantic_consistency。

**P2 `cortex_mae`**  
模型：CortexMAE-P。表面：CBIG Schaefer-400 LH||RH 顶点等权平均。禁止 pad T=12。官方 `run_embedding` 会补零，故只调 `encoder.forward_embedding`。当前 8 条 skip。

**P2 `semantic_consistency`**  
独立 CLIP 缓存 vs fMRI 时间均值 RDM Spearman。不用 TRIBE hf_cache。N 小，描述性。

## 命令（一律 `uv run`）

```bash
cd /home/zxuff/data/EEGagent/react-agent

# 合成 atlas（仅测试）。真实 Destrieux 必须 --download，且写到新目录（不覆盖 synthetic）
uv run python -m react_agent.fmri.cli prepare-assets \
  --atlas-dir examples/tribe_1s7s_smoke/atlas --synthetic
uv run python -m react_agent.fmri.cli prepare-assets \
  --atlas-dir examples/tribe_1s7s_smoke/atlas_destrieux --download

uv run python -m react_agent.fmri.cli from-tribe \
  --preds-dir /home/zxuff/data/tribev2/exp/things_static_1s_7s/preds \
  --ids apple_01b gray_control \
  --control-id gray_control \
  --out examples/tribe_1s7s_smoke/samples_v1_1.jsonl

uv run python -m react_agent.fmri.cli build-cohort-index \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl \
  --out examples/tribe_1s7s_smoke/cohort_index.json

uv run python -m react_agent.fmri.cli batch \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl \
  --config configs/fmri_check_tribe_v1_1.yaml \
  --backend none --policy rule \
  --out runs/tribe_1s7s_v1_1_rule

# 真实 Destrieux：写到新 runs，不覆盖上一目录
uv run python -m react_agent.fmri.cli batch \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl \
  --config configs/fmri_check_tribe_v1_1.yaml \
  --backend none --policy rule \
  --out runs/tribe_1s7s_v1_1_rule_destrieux

# P2 权重与冻结 CLIP（统一放 /home/zxuff/data/EEGagent/assets）
uv sync --extra p2
uv run python -m react_agent.fmri.cli prepare-p2 --kind all --download
uv run python -m react_agent.fmri.cli cache-embeddings \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl
uv run python -m react_agent.fmri.cli batch \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl \
  --config configs/fmri_check_tribe_v1_1.yaml \
  --backend none --policy rule \
  --out runs/tribe_1s7s_v1_1_rule_p2
```

Hybrid DeepSeek 须有 `DEEPSEEK_API_KEY`。v1.1 YAML：`max_lm_calls=4`，`max_batch_lm_calls=6`。

`numeric_minimal`：只做 V1 必做（`configs/fmri_check.yaml`）。  
`tribe_diagnostic`：必做仍是契约 + 统计；非 control 必须回答 gray 问题（资源不够则 partial）。

## 真实数值例子（`apple_01b` rule）

- `gray_control_contrast.overall_delta_rms = 0.05473`，`comparable=true`
- temporal / ROI / specificity 进入 contrast 模式
- specificity：`n_compared=7`，`cosine_max≈-0.127`，无 exact duplicate
- Destrieux top-K（描述，非“视觉区必须最高”）：`rh:11 G_cuneus`、`lh:11 G_cuneus`、`rh:66 S_parieto_occipital`、`rh:45 S_calcarine`
- 旧 smoke 中 apple_02s / abacus / calzone 由 `reference_deviation` flagged → 现为 `passed_configured_checks` + `insufficient_reference`

## DeepSeek 轨迹（`runs/tribe_1s7s_v1_1_hybrid_deepseek`）

1. 候选 6 个：`temporal_diagnostics`、`reference_distribution`、`gray_control_contrast`、`stimulus_temporal_profile`、`surface_roi_profile`、`cross_image_specificity`
2. 模型选 `gray_control_contrast` 并执行
3. 第二次决定看到新证据后 `stop`（仍有 5 个候选）
4. `fallback_used=false`；requested `deepseek-chat`，response `deepseek-flash`

## 如何加第 5 / 6 个工具

1. 在 `src/react_agent/fmri/tools/` 实现 CheckTool  
2. `build_registry()` 注册  
3. YAML `enabled_tools` 打开  
4. 可选：在 `ledger.py` 加 question  
**不要**改 `graph.py`。

## P2（2026-09-22）

权重与缓存：`/home/zxuff/data/EEGagent/assets/`（`hf/`、`atlas/schaefer400_fsaverage5/`、`image_embeddings/clip/`）。

- **CortexMAE-P**：CBIG fsaverage5 Schaefer-400 真实聚合 → `[T,400]`；官方 `encoder.forward_embedding`（**不用** `run_embedding`，它会 pad）。T=12 被拒（1200 vs 1600），`temporal_length_unsupported`，`padded=false`。embedding 不是效度概率。flat/volume 仍 disabled。
- **image–fMRI RSA**：独立 `openai/clip-vit-base-patch32`，7 张 THINGS jpg 已缓存。`used_tribe_hf_cache=false`。N 太小，RDM 上三角不当独立样本，不算 p。
- `claim_scope` 仍是 `numeric_consistency_only`。8 条写在 `runs/tribe_1s7s_v1_1_rule_p2`。

## 未完成（仍属 V1.1 范围）

- surface spatial sanity
- 生理合理性与 EEG 训练收益：另设评估，本轮未做

V1.2 图像 16 秒生成与 T=16 CortexMAE 见 [`docs/tribe_image16_v1_2.md`](tribe_image16_v1_2.md)，不覆盖本页旧 12s 报告。
