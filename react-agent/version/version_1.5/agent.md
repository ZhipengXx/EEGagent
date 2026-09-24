# version 1.5：当前筛查做到哪，下一步分开做什么

这是仓库到 **version 1.5 / 1.5.1** 的实现快照，加上下一步要另开的子模块方向。  
不是改代码的任务说明。不替代 [`version/version_1.3/agent.md`](../version_1.3/agent.md)。  
`claim_scope` 仍是 `numeric_consistency_only`。数值通过不等于真实脑响应正确，也不等于下游 EEG 该用哪个模型。

仓库根目录：`/home/zxuff/data/EEGagent/react-agent`。下文路径相对该根目录。

---

## 1. 现在是什么

版本叠在同一条检查链上，不是互相替换。

- **1.1**：数值工具。输入、统计、时序、ROI、参考分布、CortexMAE-P、CLIP RSA。
- **1.2**：image16 生成协议。10 fps，4 秒灰 + 1 秒图 + 后续灰，导出 T=16、fsaverage5、20484 顶点。`t_stim = t_video − 4`。不再加 5 秒，不做 HRF。
- **1.3**：`planned` 策略。L0 之后检索 memory，DeepSeek 出计划，一次执行一个 ready step。rule / hybrid 仍在。
- **1.4**：诊断覆盖合同、StopGate、contrast 的 R/A/S、探索性跟进。跟进只要求再看一眼，`defect_confirmed` 保持 false。
- **1.5 / 1.5.1**：本机工作台只读展示。不重跑 TRIBE，不改判决。帧和三视图一起换。

工作台在 http://127.0.0.1:8765 。快速检查走 rule，分层诊断走 planned。

---

## 2. 筛查怎么走

一次检查的顺序是固定的。模型不能改阈值或判决。

1. 输入是服务器上的图片路径和配置。没有浏览器上传。
2. 生成或复用 TRIBE 预测和配对灰屏。缓存命中时 `new_tribe_predictions` 可以是 0。
3. **L0**：`validate_input`、`basic_statistics`。形状、有限值、近常量。
4. **L1**：`gray_control_contrast`（等顶点 RMS）、`stimulus_temporal_profile`。时序问题只接受 contrast 上的 `ras_v1`。raw 时序不能把这题标成已回答。
5. **粗空间**：`surface_spatial_sanity` 的半球 RMS。状态是「已描述」，不是通过。
6. **探索路由**：`max_step_over_median` 相对 5.0 只开 ticket。低于阈值或分母不稳就关掉。打开后要有 targeted 描述，否则 StopGate 不能当成配置内完成。
7. **可选高阶**：`surface_roi_profile`、`cross_image_specificity`、`cortex_mae`、`reference_distribution`、`semantic_consistency`。没有兼容参考时，参考质量保持未评估，不能写成通过。
8. `planned` 时 planner 只提议下一步。覆盖和 `screening_decision` 由运行时合同与 StopGate 写。
9. 报告同时留下四件事，互不替代：
   - `verdict`：配置内数值检查
   - `screening_decision`：`pass_configured` / `abstain` / `blocked` / `flagged`
   - `stop_reason`：例如 `abstain`
   - `claim_scope`：`numeric_consistency_only`
   `evidence_confidence` 和 `recommended_training_weight` 保持空。

六个问题合同在 [`src/react_agent/fmri/diagnostic/contracts.py`](../../src/react_agent/fmri/diagnostic/contracts.py)：

| 问题 | 是否必答 | 怎样才算完成 |
| --- | --- | --- |
| 输入格式 | 必答 | `validate_input` 与 `basic_statistics` 都成功 |
| 灰屏对照 | 必答（灰屏样本本身则不适用） | contrast 可对照，并有整体差异 RMS |
| 刺激时序 | 必答 | contrast 上有 `ras_v1`，记为已描述 |
| 皮层空间分布 | 必答 | contrast 的粗空间摘要，记为已描述 |
| 后续检查 | 必答 | 没有 ticket，或 ticket 已被描述；打开着则未评估 |
| 参考质量评估 | 非必答 | 当前没有兼容校准参考，保持未评估 |

完成状态只有 `answered`、`described`、`not_applicable`。`described` 不是低分，`unassessed` 也不是失败分。

---

## 3. 已实现

- 上面列出的工具可以执行。`cortex_mae_flat` 与 `cortex_mae_volume` 已注册，但不执行，避免没有投影时假装通过。
- image16 与 12 秒记录分开标协议。工作台只有 metadata 确认是 image16 时才写「16s · 4s 灰 / 1s 图 / 后续灰」。
- 诊断 StopGate：必答未完成或跟进还开着时，不能写成配置内完成；规划失败且问题仍开着时是 `planning_blocked`。
- memory 只存摘要，不能把未验证结论升格成事实。
- 工作台按 run 目录分开同名样本。范围句按这份 run 的 `claim_scope` 和覆盖情况写。三视图到齐后才和帧说明一起换。

---

## 4. 未做

这些都不能从「配置内通过」推断出来：

- 真实脑响应、生理合理性、HRF。
- 证据置信度和训练权重。16 秒参考面板 N 很小，`quality_verdict_enabled` 为 false。
- 筛查的 precision / recall。没有独立故障标注。
- accordion 上真实 DeepSeek `revise_plan` 未作为 real_passed 验收。
- 费用经常是未提供，不能写成 0。
- 对下游 EEG 训练是否有益。文档从 1.1 起就把它划在筛查之外。
- 按任务选择该用哪个 EEG 模型。

---

## 5. 下一步：第二个子模块

现有筛查整段收成 **子模块 A**。接口冻结：输入是图片和协议，输出是数值报告，不输出模型名或训练权重。

**子模块 B** 另开，面向下游 EEG 任务。做法接近 NS-Copilot 式的模型试选，而不是再加一条 fMRI 指标。它要单独回答：给定任务（例如分类、检索或重构）、数据条件和候选模型，先试哪一个、用什么协议比较、结果能否留下。

必要方向按这个顺序。仍然不改 A 的判决：

1. **任务卡**：任务名、数据划分、指标、预算。和 `claim_scope` 分开存储。
2. **候选模型登记**：只列本地已经能跑的 EEG 模型，写清输入形状和适用任务。未接线的不报成可用。
3. **试选循环**：提出一个小实验、跑、和当前保留结果比、留下或丢掉。失败不回写筛查的 `verdict`。
4. **报告分开**：A 的「配置内通过」不能出现在 B 的模型推荐里。B 的空结果写未试，不写成不适合。

第一版 B 只做「选一个已有模型、在一个已有 EEG 任务上试一次」。不做自动改模型结构，也不把 fMRI 筛查分数当成 EEG 损失权重。
