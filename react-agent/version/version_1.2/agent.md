# version 1.2：当前检查器如何工作

这是仓库在 **version 1.2** 时的实现快照：有哪些检查指标、怎么跑、默认会不会打 API。  
不是设计提案。不替代 [`version/version_1.0/agent.md`](../version_1.0/agent.md)（那份只描述 1.0，不再改）。

仓库根目录：`/home/zxuff/data/EEGagent/react-agent`。下文路径相对该根目录。

相对 1.0 已落地：V1.1 四个领域工具 + Destrieux/P2、V1.2 图像 16 秒生成、CLI/Studio 中间进度、`fmri_pipeline` 三节点。  
`claim_scope` 在报告里**永远**是 `numeric_consistency_only`。数值通过 ≠ 真实脑响应正确。

---

## 1. 现在有没有 API 调用？

**你现在这种检查（`check-image --backend none --policy rule`）没有决策 LLM 的 HTTP 调用。**

[`configs/fmri_check_tribe_image16.yaml`](../../configs/fmri_check_tribe_image16.yaml) 默认 `policy_default: rule`、`backend_default: none`。CLI 里 `policy=rule` 会**强制** `backend=none`（见 [`src/react_agent/fmri/cli.py`](../../src/react_agent/fmri/cli.py)）。下一步由 [`policy.py`](../../src/react_agent/fmri/policy.py) `rule_decision` 离线决定。

| 通道 | 默认 check-image / image16 rule | 何时才会联网 |
| --- | --- | --- |
| DeepSeek（选下一步 / 摘要） | **不调用** | `--policy hybrid --backend deepseek` 且有 `DEEPSEEK_API_KEY` |
| 聊天图 `agent` 的 DeepSeek / Tavily | 不走这张图 | Studio 选 `agent` |
| TRIBE 推理 | 本机 trib ev2 **worker**（子进程，`shell=False`） | 不走 DeepSeek；首次无缓存才跑 GPU |
| CortexMAE / CLIP | 读 `/home/zxuff/data/EEGagent/assets/` 本地权重与冻结向量 | 权重缺失且显式 `prepare-p2 --download` 才拉 HF |
| 终端 `HF Hub unauthenticated` / `CUDA driver too old` | react-agent 里 torch 2.12 与驱动不匹配时的**本地警告** | **不是** DeepSeek |

Hybrid 全程 fallback **不算**模型路由验证。DeepSeek 是 JSON Action，不是聊天图的原生 `tool_calls`。

---

## 2. 三张图

[`langgraph.json`](../../langgraph.json)：

| 图名 | 入口 | 用途 |
| --- | --- | --- |
| `agent` | `src/react_agent/graph.py:graph` | 聊天 ReAct（可调 DeepSeek / Tavily） |
| `fmri_check` | `src/react_agent/fmri/graph.py:graph` | 检查闭环；Studio 直接点跑会缺 `bind_runtime` |
| `fmri_pipeline` | `src/react_agent/fmri/pipeline_graph.py:graph` | Studio 可点跑：`apply_defaults` → `materialize` → `run_check` |

检查循环拓扑仍是 1.0 那张（不改 `graph.py`）：

```text
ingest → run_required_checks → update_evidence ⇄ select → validate → execute → finalize
```

CLI 与 Studio 的 `fmri_pipeline` 都调用同一套 `run_sample` / `materialize`。生成器 `tribev2_generate_fmri` 的 `kind=generator`，检查开始后**不是**候选工具。

---

## 3. 如何执行

一律从仓库根目录 `uv run`（不要用系统 Python 3.10）。

### 3.1 图像 → 16s 生成 + 检查（V1.2）

配置：[`configs/fmri_check_tribe_image16.yaml`](../../configs/fmri_check_tribe_image16.yaml)。  
**不**绑旧 12 点 cohort / reference。必做：`validate_input`、`basic_statistics`、`cortex_mae`。

```bash
# 只写 4s 灰 + 1s 图 + 11s 灰视频，不加载 TRIBE
uv run python -m react_agent.fmri.cli prepare-image-video \
  --image /path/to.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --out runs/tribe_image16_video_smoke

# 真实 TRIBE，不检查（无缓存时要空闲 GPU；worker 设 CUDA_DEVICE_ORDER=PCI_BUS_ID）
uv run python -m react_agent.fmri.cli generate-from-image \
  --image /path/to.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --out runs/tribe_image16_generate_smoke

# 生成（可 cache hit）+ 检查；无 DeepSeek
uv run python -m react_agent.fmri.cli check-image \
  --image /path/to.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --backend none --policy rule \
  --out runs/tribe_image16_rule_smoke
```

已核过的单图：`apple_01b.jpg` → `runs/tribe_image16_rule_smoke`（preds `[16,20484]`，二次检查 0 条新 prediction）。  
`batch-images` 命令已实现，本轮不对旧 8 条执行。

Studio：`uv run langgraph dev`，选 **`fmri_pipeline`**（不要把 jpg 跟在 `langgraph dev` 后面）。空表单默认 `apple_01b`，写出 `runs/tribe_image16_studio_view`。看 `studio_summary` / `last_event`。Studio **不是**脑图看板。

### 3.2 旧 12s 八条（V1.1，只读旧 npy）

```bash
uv run python -m react_agent.fmri.cli batch \
  --manifest examples/tribe_1s7s_smoke/samples_v1_1.jsonl \
  --config configs/fmri_check_tribe_v1_1.yaml \
  --backend none --policy rule \
  --out runs/tribe_1s7s_v1_1_rule
```

不要覆盖已有 `runs/tribe_1s7s_*`。T=12 上 CortexMAE **skip**（官方 encoder 拒 1200 vs 1600 patch，不补零）。

### 3.3 中间进度

终端与 `events.jsonl` 同步一行一条（不打印 `[T,V]` / preds / embedding）：

`[generate] …` / `[tool] cortex_mae success` / `[decision] rule -> gray_control_contrast` / `[stop] passed_configured_checks …`

长等待前必有 `[generate] TRIBE worker running (image|control)`。

---

## 4. 检查指标

注册表：[`src/react_agent/fmri/tools/registry.py`](../../src/react_agent/fmri/tools/registry.py)。  
Verdict 只认 finding 的 `decision_effect` 为 `flag` / `block`；多数领域工具是 `none`（描述性）。

| 工具 | 何时跑 | 关键 metrics / 产物 | 不能声称 |
| --- | --- | --- | --- |
| `validate_input` | 必做 | `shape`、`finite_ratio`、`valid_for_numeric_checks`、`degenerate_signal` | 生物真实性 |
| `basic_statistics` | 必做 | mean/std/min/max、quantiles、`temporal_change_ratio`、`max_frame_diff_ratio` | 质量分数 |
| `temporal_diagnostics` | rule 在有时序 flag 时 | `lag1`、`adjacent_diff_*`、`peak_abs_mean_frame`、`max_diff_ratio` | HRF / 泄漏 |
| `roi_summary` | 有 `roi_map_path` | 每 ROI 均值；image16 **无此资源 → unavailable** | 功能区身份 |
| `reference_distribution` | 有冻结 stats 且 profile 兼容 | robust z；N 小则 `insufficient_reference`、`decision_effect=none`。image16 **不绑**旧 12 点参考 | 样本质量判决 |
| `gray_control_contrast` | 显式 gray 绑定且形状/profile 可比 | `overall_delta_rms`、`comparable`、`peak_frame`、`response_rms.json`、`contrast_minus_gray.npy` | SNR、p 值、显著激活 |
| `stimulus_temporal_profile` | 有 contrast 或 raw | `response_rms`、`windows`（16s 为 4+1+11）、`peak_frame` | 硬编码 5s 峰 |
| `surface_roi_profile` | Destrieux atlas | top-K `hemisphere:label_id`（描述） | 视觉区必须最高 |
| `cross_image_specificity` | 固定 cohort 面板 | `n_compared`、cosine、exact hash→`duplicate_output`。image16 **未绑 cohort** | 图像语义对错 |
| `cortex_mae` | image16 为必做；T=16 raw | `mode=raw`、`input_t`/`trained_t`、`padded`、`encoder_forward_validated`、`embedding_available`、`reference_score_assessed`（本轮无 HCP → false） | 效度概率、生物效度。T=12 skip，不 pad |
| `semantic_consistency` | V1.1 YAML 打开；image16 **未 enabled** | 独立 CLIP vs 时间均值 RDM `spearman_rdm`，`used_tribe_hf_cache=false` | 独立脑验证、把上三角当独立样本算 p |
| `tribev2_generate_fmri` | pipeline 必要生成，**不是检查候选** | 产物 `[16,20484]`、`temporal_padding_applied=false`、匹配 16s gray | 真实脑响应 |

image16 一次 rule 检查实际会跑完：validate、basic、cortex_mae（必做），以及 gray / temporal / surface ROI（rule 队列）。`roi_summary`、`reference_distribution`、`cross_image_specificity` 在该配置下 unavailable。

16s 协议（继承本机 stim，不是 prompt 里的 24 fps）：FPS=10，4s 灰 + 1s 图 + 11s 灰，160 帧，灰 RGB(128,128,128)，`preds[k]` 对齐 `segments[k].start`，不再平移 5s。禁止把旧 12 点 pad/插值成 16 点。

---

## 5. 输入与产物

- **12s 旧数据**：`from-tribe` 读 sidecar，`generation_profile_id=static_1s_7s_tribev2`，`[12,20484]`。旧目录 `tribev2/exp/things_static_1s_7s/` **只读**。
- **16s 新数据**：profile `static_gray4_image1_gray11_tribev2_v1`，写 `/home/zxuff/data/EEGagent/assets/generation/` 与 `runs/tribe_image16_*`。
- 单样本：`runs/<out>/<safe_id>/report.json`、`report.md`，以及 `temporal_profile.json` / `roi_profile.json` / `response_rms.json` 等。
- 根目录：`events.jsonl`、`pipeline.json`、`resolved_config.json`。

`schema_version` 配置仍是 `fmri_check.v1.1`。

---

## 6. 相关文档

- 1.0 快照：[`../version_1.0/agent.md`](../version_1.0/agent.md)
- V1.1 工具状态：[`../../docs/tribe_tools_v1_1.md`](../../docs/tribe_tools_v1_1.md)
- V1.2 生成验收：[`../../docs/tribe_image16_v1_2.md`](../../docs/tribe_image16_v1_2.md)
- 用法：[`../../docs/fmri_check.md`](../../docs/fmri_check.md)
