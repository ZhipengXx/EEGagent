# TRIBE 图像 16 秒生成 V1.2

在 V1.1 检查循环上增量：`tribev2_generate_fmri` 把静图做成 **4s 灰 + 1s 图 + 11s 灰**（FPS=10，160 帧），经独立 TRIBE worker 推出 `[16, 20484]`，再进入现有检查。  
`claim_scope` 仍是 `numeric_consistency_only`。禁止把旧 12 点 pad/插值成 16 点。不覆盖 `runs/tribe_1s7s_*`，不改 `graph.py` 拓扑，不改 version 1.0。

单图验收：`apple_01b.jpg`（500×500）。

## 验收状态

| 项 | implemented | synthetic_passed | real_data_passed | not_run |
| --- | --- | --- | --- | --- |
| 16s 协议视频（FPS=10，40+10+110，灰 RGB 128，无 letterbox/音轨） | 是 | 是 | **是** `runs/tribe_image16_video_smoke`：duration=16、160 帧、500×500、无音轨、`protocol_ok` |  |
| `tribev2_generate_fmri` worker + mock | 是 | 是（mock `[16,20484]`；二次调用 0 条新 prediction） | **是** `runs/tribe_image16_generate_smoke`：worker、`synthetic=false`、`[16,20484]`、2 条新 prediction |  |
| 导出按 segment 时间选 `[0,16)`；12/15 点契约错误 | 是 | 是 | **是** `t_video=0..15`，`temporal_padding_applied=false`，segments 16 条 |  |
| 匹配 16s gray control（键=size/fps/gray/codec/profile，与文件名无关） | 是 | 是 | **是** `gray_500x500_*` `[16,20484]`；检查时 cache hit |  |
| pipeline → 现有 `run_sample`；失败 `pipeline_status=generation_failed` / `check_verdict=null` | 是 | 是 | **是** `runs/tribe_image16_rule_smoke`：`pipeline_status=ok`，`check_verdict=passed_configured_checks`，coverage=complete |  |
| CortexMAE-P 默认 raw 生成序列；T=12 仍拒且不 pad | 是 | 是（T=12 skip；T=16 mock `padded=false`） | **是** T=16 raw forward：`encoder_forward_validated=true`，`embedding_available=true`，`padded=false`，`reference_score_assessed=false`（无 HCP） |  |
| gray / temporal / surface ROI | 是 | 是 | **是** contrast `comparable=true`；`overall_delta_rms≈0.050`；windows 4+1+11 |  |
| `batch-images` CLI | 是 | — | — | **是**（不对旧 8 条执行） |
| Hybrid DeepSeek 16s | 是（命令已接） | — | — | **是**（本轮不写 `runs/tribe_image16_hybrid_smoke`） |

CLI 决策 LM 仍是 `backend=none|deepseek`。TRIBE 推理后端是 YAML `generation.backend=worker|mock`。

## 真实数值（apple_01b）

- 生成：`/home/zxuff/data/EEGagent/assets/generation/preds/apple_01b_0318065ab5f8b68aa446443c.npy`，shape `[16,20484]`，`missing_modalities.audio/text=zero_fill_aggregate_features`。
- Control：同目录 `control_preds/gray_500x500_80abe8b0e5ca13c43e86422b.npy`。
- 检查二次调用：`cache_hit_image=true`，`cache_hit_control=true`，`new_tribe_predictions=0`。
- `gray_control_contrast.overall_delta_rms≈0.05009`，`comparable=true`，`peak_frame=4`（onset）。
- CortexMAE-P：`mode=raw`，`input_t=16`，`trained_t=16`，`embedding_dim=768`，`aggregation=official_cls`，`gray_cosine≈0.967`（描述性，不是参考分）。
- 未绑旧 12 点 cohort/reference；`cross_image_specificity` / `reference_distribution` 标 unavailable。

Worker 必须设 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 再设 `generation.cuda_visible_devices`，否则默认 FASTEST_FIRST 会撞上占用中的旧 12s 作业。本机验收用 PCI GPU 4。

## 继承的本地事实

- FPS=**10**（不是 24）。16s = 40+10+110=160 帧。
- 灰 RGB(128,128,128)；画布跟原图；codec `libx264`；无音轨；tmp 后原子替换。
- 事件：单条 `Video` 覆盖整段。不自动 caption / ASR / TTS。
- `preds[k]` 对齐 `segments[k].start`，**不再平移 5s**。
- 旧缓存 `tribev2/exp/things_static_1s_7s/` **只读**。
- react-agent torch 与驱动不匹配；TRIBE 走 trib ev2 Python，JSON 交换，`shell=False`。生成后 worker 退出，再跑 CortexMAE。

## 命令（一律 `uv run`）

```bash
cd /home/zxuff/data/EEGagent/react-agent

uv run python -m react_agent.fmri.cli prepare-image-video \
  --image /home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/Image_set/training_images/00029_apple/apple_01b.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --sample-id apple_01b \
  --out runs/tribe_image16_video_smoke

uv run python -m react_agent.fmri.cli generate-from-image \
  --image /home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/Image_set/training_images/00029_apple/apple_01b.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --sample-id apple_01b \
  --out runs/tribe_image16_generate_smoke

uv run python -m react_agent.fmri.cli check-image \
  --image /home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/Image_set/training_images/00029_apple/apple_01b.jpg \
  --config configs/fmri_check_tribe_image16.yaml \
  --backend none --policy rule \
  --require-check cortex_mae \
  --out runs/tribe_image16_rule_smoke
```

`batch-images` 已实现，本轮不对旧 8 条执行。

## 指标含义

CortexMAE：`encoder_forward_validated` / `embedding_available` / `reference_score_assessed`。本轮无 HCP 参考，第三项为 false。默认输入是 **raw 生成序列**，不是 gray contrast。

## 目录

- 配置：`configs/fmri_check_tribe_image16.yaml`（不绑旧 12 点 cohort/reference）
- 生成包：`src/react_agent/fmri/generation/`
- 薄图：`fmri_pipeline` 三节点 `apply_defaults` → `materialize` → `run_check`。终端与 Studio 都会打 `[generate]` / `[tool]` / `[decision]` / `[stop]`，不打印 `[T,V]` 数组。空表单默认 `apple_01b`，输出 `runs/tribe_image16_studio_view`。
- 旧 12s V1.1 状态：[`docs/tribe_tools_v1_1.md`](tribe_tools_v1_1.md)
