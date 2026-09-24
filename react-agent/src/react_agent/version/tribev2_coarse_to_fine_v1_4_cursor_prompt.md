# TRIBE fMRI V1.4：从基础检查通过到有证据驱动的分层诊断

这是交给 Cursor 的增量实现任务书，依据用户提供的 V1.3 快照和 accordion_01b 报告编写。未直接审阅用户本机代码；文中的新配置、字段和模块是待实现要求，不是完成声明。

仓库：/home/zxuff/data/EEGagent/react-agent。

本轮复用 V1.3 的 planned、Plan、validator、SQLite memory、DeepSeek provider、共享预算和 CLI/Studio。不要重建框架，不要再把“接通 API”作为主要交付。

## 1. 先对当前报告建立准确理解

用户的 accordion_01b 报告：

- 四个实际工具：validate_input、basic_statistics、gray_control_contrast、temporal_diagnostics。
- 没有展示 stimulus_temporal_profile 或 surface_roi_profile。
- screening_decision=pass_configured，coverage=complete。
- contrast overall_delta_rms=0.0595887，peak_frame=4，max_step_over_median=6.69012。
- temporal_diagnostics 的 peak_abs_mean_frame=12，peak_diff_frame=14，max_diff_ratio=5.26203。
- 两次 LM HTTP 成功，lm_repairs=1，accepted_plan_count=1，curator_calls=0。

正确解读：

1. API 已经接通；两次成功返回不等于两份接受的计划，也不等于发生了 revise_plan。
2. coverage=complete 可能符合旧配置，不能未经代码审阅就判为实现 bug；但旧要求可能不足以覆盖用户希望的分层诊断。
3. report 的工具章节顺序不是执行顺序。先检查 events/plan_history，确认调用顺序及 signal_source。
4. contrast RMS 的峰、raw 的绝对空间均值峰、相邻变化峰是不同量；4、12、14 不相等不构成矛盾。
5. 报告未明确每个 temporal metric 的 signal_source，不能只凭名字断言它分析了 contrast。
6. 6.69 和 5.26 目前是描述性/启发式值，不能据此宣布该 fMRI 异常。
7. curator_calls=0 不表示 memory 没写入：V1.3 episode 应已由代码确定性保存。
8. evidence_confidence、recommended_training_weight 为 null，在未校准时是合理状态，不要为了完整报表填假数值。
9. 快照里的 9158+1547 tokens 与这份报告的 20580+1386 属于需要通过 run_id 区分的运行，不能混合比较。

先读 AGENTS.md、V1.3 文档、实际配置、ledger、policy、planner/validator、工具实现、memory 和 report writer，定位上述字段来自哪里。输出简短映射，然后继续实现。

## 2. V1.4 的核心目标

在现有 planned 上实现：

便宜的基础覆盖 → 从结果生成明确的后续问题 → 针对时间段/ROI/顶点群检查 → 有依据地结束或说明阻塞。

“更深”是问题更具体、证据分辨率更细，不是必须调用更贵的 backbone，也不是所有样本都把工具跑完。

确定性代码负责最低覆盖、依赖、证据合同和停止门槛；DeepSeek 在可行动作中选择诊断路径、处理新问题和修订计划。

V1.4 保留：

- claim_scope=numeric_consistency_only。
- 原 V1.3 配置和历史 runs 可复现，不覆盖。
- 原三张 graph 的拓扑，聊天 agent 不改。
- image16：10 FPS，4s gray + 1s image + 11s gray，160 帧，RGB(128,128,128)。
- [16,20484]、LH||RH、已有 segments 时间轴和生成缓存。
- 不加额外 5s shift；不补齐旧 T=12；不换生成 checkpoint。
- N=3 panel 仍仅为 engineering_smoke，quality_verdict_enabled=false。
- 无兼容参考时不强制 CortexMAE，不声称生物真实性或训练收益。
- system prompt 继续英文。

## 3. 新增诊断配置，不修改旧配置的验收含义

新增 configs/fmri_check_tribe_image16_diagnostic_v1_4.yaml。

新增或复用 analysis_goal / coverage_profile，明确本次目标是 image16_numeric_diagnostic。不要复用原 complete 字段来暗示所有可能的质量问题都已解决。

新 profile 的最低覆盖：

| 问题 | 所需证据 | 能否被别的证据替代 |
| --- | --- | --- |
| input_contract | validate_input + basic_statistics；必要的 shape/profile/time-axis 一致性 | 仅由实际满足同一合同的工具替代 |
| gray_comparability | 显式匹配 gray，生成可追溯的 contrast | raw temporal 不能替代 |
| stimulus_temporal_description | 对 contrast 按刺激窗口做时序描述 | raw lag1/全局峰值不能替代 |
| coarse_spatial_description | 对 contrast 做顶点/半球层面的便宜空间扫描 | 全脑均值不能替代 |
| triggered_followups | 本轮有效触发的后续问题已回答或明确阻塞 | “工具执行成功”本身不能替代 |

gray_control 样本使用明确的 control profile；gray contrast 为 not_applicable，不强迫 self-contrast 或把 not_applicable 算失败。

跨图像 specificity、参考分布、CortexMAE 参考分、RSA 默认列为 optional dimensions。缺失时报告 not_assessed/资源原因，不必使仅限数值诊断的整个目标失败。若用户显式把相应问题设为 required，缺失就必须 partial/abstain。

扩展 question ledger 的证据合同：

question_id、required_by_profile、activated_by_trigger_id、
accepted_signal_modes、required_metric_keys、compatible_protocol、
answer_type、status、evidence_ids、unresolved_reason。

状态至少区分 unassessed、described、answered、blocked、not_applicable。

described 仅能满足目标本身就是“描述”的问题；不能满足“已校准质量是否正常”。

required 属性由配置和触发规则决定，LLM 不能改低目标或删掉问题来获得 complete。

## 4. 分层执行和停止门槛

L0：validate_input + basic_statistics。

- 硬契约失败直接停止不适用分析。
- 通过后读取 memory，创建初始计划。

L1：便宜的共同诊断基础。

- gray_control_contrast。
- stimulus_temporal_profile，绑定上述 contrast artifact。
- 新增 surface_spatial_sanity(mode=summary)，同样绑定 contrast。
- gray 后的时序和空间总结可独立执行；不强制无意义的串行顺序。
- temporal_diagnostics(raw) 可以回答 raw 专属问题，但不能代替 contrast 的事件窗口问题。

L2：有条件的定向检查。

- 针对可疑 transition/window，执行 temporal_diagnostics(mode=targeted, signal=contrast)。
- 针对贡献集中的 ROI/顶点群，执行 surface_roi_profile(mode=targeted) 或 surface_spatial_sanity(mode=targeted)。
- 怀疑输出重复/跨图像缺乏区分，且 cohort 可用时，执行 cross_image_specificity。
- 每个 L2 动作必须关联一个未解决问题、实际证据和预期能获得的证据类型。

L3：依赖兼容参考的分析。

- CortexMAE/reference/RSA 只有在适用且能回答当前问题时选择。
- embedding 可用不等于质量分数可用；N=3 不能因加入 planning 自动变成校准参考。

确定性的 StopGate 必须检查：

1. 当前 goal 的 required questions 是否完成；
2. 是否存在 open 的 triggered followup；
3. 相关判决 finding 是否有效；
4. 是否存在可执行且能回答未解决问题的工具；
5. 预算/资源是否允许继续。

合法结束：

- 基础覆盖和已激活问题完成，没有有效 flag/block → pass_configured。
- 有确定性、已有合同支持的问题 → flagged/blocked，保留作用范围。
- 关键问题未完成，预算或资源不足 → abstain + coverage=partial。
- 无更深入的兼容参考，但该参考问题不是当前 goal 的 required → 完成当前数值诊断，同时明确 reference_quality=not_assessed。

后续问题“这个 transition 的变化集中在哪里”可以被描述性定位回答；这不等于“已证明该变化正常”。对需要校准才能回答的问题，不能用同一个描述性结果强行关闭。

停止请求被拒绝时记录 stop_rejected 和缺少的 question_id；有可行动作则继续/修订，没有则明确 abstain，禁止无限循环。

## 5. 给每个指标绑定信号来源和计算定义

所有相关 ToolResult / evidence 新增 SignalProvenance：

signal_mode(raw/contrast)、input_artifact_id/hash、control_artifact_id/hash、
generation_profile_id、generation_checkpoint_id、normalization_id、
surface_space、hemisphere_order、vertex_mask_id、T、time_axis_id、
source_execution_id、metric_definition_version。

不能仅通过“已有 contrast 文件存在”猜测工具实际读了什么。

定义：

~~~text
Y[t,v] = image prediction
G[t,v] = matched gray prediction
D[t,v] = Y[t,v] - G[t,v]

R[t] = sqrt(mean_v(D[t,v]^2))
A[t] = abs(R[t+1] - R[t])
S[t] = sqrt(mean_v((D[t+1,v] - D[t,v])^2))
~~~

R 的变化 A 与完整顶点模式的变化 S 不是同一指标。不要把所有 max_step 都当成同一种变化。均值的相邻差也不能代替 S，因为正负局部变化会抵消。

先核对旧 gray_control_contrast.max_step 等字段的实现，补充定义；不要不兼容地偷偷换公式。新定义用新字段和版本。

每个峰值写清：

- metric_name、signal_mode；
- frame/time 或 transition；
- 对 transition 明确 from_frame/to_frame/from_time_s/to_time_s；
- 使用的 vertex mask、权重、聚合方式、分母/epsilon；
- 是否有定义、何种原因 undefined。

不要仅保留 peak_diff_frame=14 这种不知道指 13→14 还是 14→15 的索引。

## 6. 加强 stimulus_temporal_profile 与定向 temporal_diagnostics

从 sidecar/segments 读取事件窗口，保留半开区间：

pre=[0,4)，stimulus=[4,5)，post=[5,16)。

检查时间轴真的与上述 profile 一致；不只凭文件名或 shape 推断。

stimulus_temporal_profile 在新 profile 的 required 问题上必须使用 D：

- 保存完整的 16 点 R(t)，以及定义明确的 A(t)、S(t)；
- 每个窗口的描述性幅度/能量摘要；
- 最大 R 和最大 S 的位置、对应事件段；
- transition 是否跨事件边界；
- source/provenance 与具体 artifact。

stimulus 仅 1 个时间点，不能装作足以拟合 HRF 或稳定估计相关性。

TRIBE 的时间对齐沿用当前经过核对的导出约定。4s 峰本身不构成 HRF 异常；也不能仅凭刺激前变化断言未来信息泄漏，需检查模型上下文和对齐约定。

扩展 temporal_diagnostics 的 targeted 模式：

- 输入已绑定的 D、少量 transition/window selectors。
- 输出相应 transition 的 S、顶点 abs(delta) 分位数、贡献分布、raw/control/contrast 的并列变化描述。
- 能回答“变化主要来自共享的 gray 模式还是 image-gray 差异”“是广泛变化还是少量顶点贡献”。
- 不输出因果结论，不宣称这些描述足以区分神经响应与模型错误。

window/transition 由实际时间轴解析，越界必须拒绝；LLM 不得传任意切片代码。

## 7. 最小新增工具：surface_spatial_sanity

采用 summary/targeted 两种模式，复用已有 atlas/资源加载模块。两种模式均为数值描述，默认 decision_effect=none。

summary 模式：CPU 上完成，不需要 CortexMAE。

- 同一有效 vertex mask 上计算每帧的空间 RMS/分位数、每顶点时间 RMS。
- 左右半球分别描述，不设左右必须对称的健康判据。
- 对若干高变化 transition 计算逐顶点平方变化贡献。
- 保存 top fraction 的能量占比与顶点列表，例如 top 1% 的贡献占比；1% 是描述性集合大小，不是异常阈值。
- 记录均值变化与全顶点 RMS 变化，避免正负抵消造成漏看。
- medial wall/无标签顶点怎么处理必须明确，不能悄悄混合不同 mask。

targeted 模式：

- 输入明确的 transition/window/vertex group selector；输出局部数值分布。
- 有真实兼容的 fsaverage5 mesh 时，可计算相邻顶点差异、局部残差和空间连通结构。
- 使用真实 mesh/hemisphere order/hash；Destrieux label 文件本身不等于 mesh adjacency。
- 无 mesh 时，邻域分析项为 unavailable；不能把顶点索引相邻当表面邻居。
- 如果新增资源需下载，沿用显式 prepare-assets 入口，不在 planner 执行中自动下载。

ROI 定向分析复用 surface_roi_profile，增加 metric mode 和 selector 即可：

对 transition k→k+1，令 E[v]=(D[k+1,v]-D[k,v])^2。
ROI 的 contribution = sum(E[v] in ROI) / sum(E[v] over declared valid mask)。

同时保留 signed mean 和 RMS 等不同指标；贡献率不是显著激活，也不是 ROI 功能正确性的概率。分母为零则输出 undefined；不能用假 0%/100% 掩盖。

atlas 覆盖不全时输出 unassigned 的贡献以及 coverage；不默默让已知 ROI 重新归一化成 100%。

## 8. 将“触发深入检查”和“拒绝样本”分开

新增 RoutingSignal / FollowupTicket，复用已有 evidence 引用：

trigger_id、rule_id/version、question_id、source_execution_id、
metric_path、observed_value、threshold_ref、basis、
suggested_tool_capabilities、priority、decision_effect=none、
status(open/characterized/blocked/dismissed_with_evidence)。

basis 必须明确 heuristic/calibrated/exact_contract；其中 heuristic 只能触发调查，不能产生质量 flag。

为让初版确实可执行，允许一个显式命名的 exploratory_v1 路由配置：

~~~yaml
routing:
  profile: exploratory_v1
  contrast_rms_step_ratio:
    enabled: true
    threshold: 5.0
    basis: heuristic
    decision_effect: none
    question: contrast_transition_localization
  spatial_concentration:
    enabled: false
    # 未校准时先保留描述；不要凭这份 accordion 报告调出一个必触发阈值。
~~~

这里的 5.0 是“何时值得进一步查看”的初始工程启发式，不是文献支持的生理界限，也不是通过 accordion 校准出的有效阈值。

使用前要求：

- metric 的 signal/公式确认与配置一致；
- 分母有效性单独检查，数值 epsilon 只解决数值问题，不能伪装为噪声底；
- 极小分母/零差异时输出 undefined 或 denominator_unstable，并走明确的数值描述分支；不要报告无限异常。
- 同时输出实际变化幅度和分母，避免只给一个比率。
- 事件边界处的变化仍可被查看，但不能因跨事件边界就拒绝。
- 校准与评估时冻结 routing config，并做阈值敏感性分析；不要在测试集上反复调到想要的结果。

若当前 accordion 的 6.69012 经代码核对确实是该配置使用的 contrast 指标，会生成“定位该变化”的 followup；后续输出仍可能只有描述性结论。这是预期行为，不能预设 flagged。

预算不允许执行时票据 blocked，required followup 导致 partial/abstain。模型不能无证据删除票据来停止。

## 9. 修复同一工具多模式执行与证据覆盖问题

检查 V1.3 是否仅按 tool_id 去重、缓存或存放 metrics。若是，扩展为 ToolExecutionKey：

hash(tool_id, tool_version, signal/input/control hashes,
     normalized_params, atlas/mesh/reference hashes,
     metric_definition_version, relevant_config_hash)。

- 同一个 temporal_diagnostics 的 raw 和 contrast 是不同执行。
- 同一工具 summary 和 targeted 是不同执行。
- 输入、参数均不变的执行复用缓存，不重复收费。
- 预算按实际 execution 计数，不能只按不同 tool_id 数量计数。
- findings 和 evidence 引用 execution_id，避免 raw 结果覆盖 contrast。
- plan readiness 以需要的具体 execution/result 为准。
- 老报告 metrics[tool_id] 保持兼容；新增 executions 为权威细节。多个执行时明确列各实例，禁止 last-write-wins。

## 10. Planning、repair 与 replan 的改进

继续使用 V1.3 的 typed Plan 与 Guard，不换一套框架。

初始计划应覆盖 L1 并声明可能的 followup 能力。看不到 L1 结果时不得声称已知道可疑 ROI/transition。

L1 完成后：

- 现有合法条件分支可以根据真实证据执行，不必硬性多打 API；
- 出现未覆盖的新票据或需要选择定向检查时，调用 revise_plan；
- revise 请求只包含新证据、活跃问题、旧计划的必要状态、ready 候选与剩余预算。

未执行的 steps 可修订，已完成步骤及证据不得改写。

replan 失败时不能只写“保持旧计划”就算处理成功：

- 旧计划重新通过当前 readiness/coverage 校验且仍能解决问题，才可继续；
- 否则有界 repair；
- 最后无合法计划则输出 planning_blocked + abstain/partial，不执行失效动作。

区分统计：

- provider_calls_succeeded；
- create_plan_attempts / create_plan_accepted；
- repair_attempts / repaired_plan_accepted；
- revise_plan_attempts / revise_plan_accepted；
- accepted_plan_count，保留原定义并文档化；
- curator_calls。

旧 planner_calls_succeeded 若继续保留，注明它统计什么，不能用 2 次 HTTP 成功暗示 2 次 accepted plan。

修复目前需要 repair 的原因：记录 validation_error_code/field、schema/prompt version、错误类别和对应修复；不能为减少 repair 而关闭 validator。

## 11. 英文 Planner prompt 增量

新建 planner_v2.md，基于 V1.3 prompt 修改。以下内容必须覆盖，结构化字段由程序注入实际 schema；不要把下面的自然语言当可执行条件：

~~~text
You plan numerical diagnostics for TRIBE-generated fMRI.
Your claim scope is numeric_consistency_only.

A successful tool call is not automatically a satisfied diagnostic question.
Use the question contracts, signal provenance, and actual metric definitions.
Raw temporal statistics cannot satisfy a question that requires event-aligned
image-minus-gray contrast.

Complete the required inexpensive diagnostic coverage before requesting a
successful stop. This includes the configured control, temporal, and coarse
spatial questions, except where the runtime explicitly marks them not applicable.

Depth means more targeted evidence: a transition, a time window, an ROI, or a
vertex group. It does not mean executing every tool or choosing a larger model.

For each optional action, cite:
- the open question or active follow-up ticket;
- the source evidence and its signal mode;
- the evidence that this action is expected to add;
- the resources and budget needed.

Routing heuristics may request closer inspection. They do not justify rejecting
a sample, assigning biological validity, or setting a training weight.
Do not invent thresholds or change question requirements.

Do not compare peaks from different statistical quantities as if they measured
the same response. A contrast-RMS peak, an absolute raw-mean peak, and a maximum
transition can legitimately occur at different times.

Inspect only the signal and time axis explicitly bound by the runtime.
Do not add an HRF shift, pad T=12, infer mesh adjacency from vertex indices,
or replace missing calibrated references with a few remembered examples.

Do not claim CortexMAE embeddings answer quality questions without an applicable
reference or another explicitly supported evaluation contract.

Use compatible memory as routing advice, never as a quality label for the current
sample. Preserve counterexamples and unverified status.

On revision, respond to the newly observed evidence. Retain completed steps and
their actual results. Do not repeat the same execution key.

An open required question cannot be silently removed. If no valid action remains
or the budget is exhausted, request an incomplete stop and identify the blocked
questions. If all required diagnostic questions are answered, an early stop is
allowed without running optional model-based tools.

Return one JSON object matching the supplied schema.
Provide short evidence-linked reasons, not long internal reasoning.
The runtime validates execution and determines coverage and screening_decision.
~~~

## 12. Memory 只做与本轮有关的增量

沿用 SQLite、episode 和 verification，不重新设计数据库或引入向量检索。

增加或复用结构化记录：

- routing_profile_version、coverage_profile_version；
- 触发 metric/公式版本、signal_mode、trigger_id；
- followup 问题、选择的 action/参数、实际新增证据；
- ticket 是否被描述性回答、是否有独立的缺陷确认；
- 实际成本、没有解决问题的动作、误报或被推翻的解释。

question_resolved 和 defect_confirmed 是两个字段，不能把一次成功定位当“筛掉坏样本”。

旧 episode 缺新字段时仍可 inspect/export，但不能假装满足 V1.4 的可比性要求。

memory curator prompt 增补：

~~~text
Record what the follow-up actually established, not what the planner hoped
to establish. Distinguish a routing trigger, a localized numerical pattern,
and a verified defect.

A completed targeted inspection is not a successful rejection.
Preserve the signal mode, metric definition, routing configuration, and coverage
profile. Do not generalize one transition or ROI to all images or protocols.

Store unhelpful actions and corrected interpretations as well as useful ones.
Do not claim cost savings or improved precision without a valid comparison.
~~~

curator_calls=0 仍允许，只要确定性的 episode 和路径记录已落地。验证 memory 改善需要后续对照，不把“成功检索”写成“提升筛查效果”。

## 13. 控制 prompt 体积和成本

这份报告两次调用累计 input_tokens=20580、output_tokens=1386。先记录每次 payload 的组成，不能直接断言浪费来自 memory 或某个 schema。

新增每次调用的：

role、context_section_sizes、input_tokens/output_tokens、
token_count_source(provider/estimated)、prompt/schema version、
validation outcome、parent_call_id、fallback 状态。

精简上下文：

- 仅发送紧凑指标、来源、question/ticket、候选工具与预算；
- 不发送完整数组/embedding、所有历史 episode、冗长报告或重复工具文档；
- repair 发送必要任务约束、原 proposal 和字段级错误，不无条件重贴全部执行历史；
- revise 发送增量证据和必要旧状态，但保留判断依赖的原证据摘要；
- 安全/科学边界、required question 和关键 provenance 不能为省 token 被裁掉。

所有调用沿用同一预算账本。api_usd 缺实际价格表时继续 null，不能用猜测价格补齐。

不要求通过增加 LM 次数证明 autonomous；用接受的新计划、证据驱动的路径变化来验收。

## 14. 新报告：让用户看懂检查到了哪里

保留旧 verdict/metrics/cost；增加：

- analysis_goal、coverage_profile/version；
- coverage_by_dimension，包括 required/optional、status、evidence_ids；
- depth_reached 与实际执行路径；
- active/resolved/blocked followup；
- 每次 execution 的 signal_mode/selector/provenance；
- 为什么升级、为什么没有升级；
- configured numerical decision 与尚未评估的质量维度；
- API create/repair/revise 的分别计数。

示例字段是结构示意，不是当前 accordion 的预测结果：

~~~json
{
  "analysis_goal": "image16_numeric_diagnostic",
  "coverage_scope": "configured_required_numeric_questions",
  "coverage_by_dimension": {
    "input_contract": {"required": true, "status": "answered"},
    "gray_comparability": {"required": true, "status": "answered"},
    "stimulus_temporal_description": {"required": true, "status": "described"},
    "coarse_spatial_description": {"required": true, "status": "described"},
    "reference_quality": {
      "required": false,
      "status": "unassessed",
      "reason": "no_compatible_calibrated_reference"
    }
  },
  "biological_validity": "not_assessed",
  "recommended_training_weight": null
}
~~~

report.md 顶部应写明“配置要求的数值诊断完成/未完成”。即使完整，也不能只给 complete 而省略 scope。

保存可核对的 temporal curves 和局部贡献摘要；可生成一个简洁的诊断图，标注事件窗口与已选 transition。图仅显示实际数值，不额外生成生理解释。

新产物使用独立 runs/tribe_image16_diagnostic_v1_4_*，不覆盖 V1.3。

## 15. 必须验证的行为

围绕分层与停止逻辑设计少量有效测试：

1. 旧报告回放：原四个工具结果不能在新 diagnostic profile 下满足 stimulus-temporal 与 coarse-spatial 合同；旧 profile 结果不被追溯改写。
2. 来源合同：raw temporal 不能回答 contrast 问题；缺 provenance 明确 incomplete，而非按工具名字匹配成功。
3. 合理早停：受控无触发输入完成 L1 后停止；CortexMAE 不执行；无需为了展示深度强迫 L2。
4. 定向升级：带有限数值阶跃的受控输入触发路由，选择相应 transition；执行后有可追溯新增证据，不能仅靠触发自动 flag。
5. 均值抵消：两个相反方向的局部扰动使空间均值几乎不变，全顶点变化指标仍能展示差异。使用已知注入位置验证定位，而非让模型自评正确。
6. 峰值混淆：共享 gray 分量使 raw 峰在晚期而 contrast 峰在刺激窗口，系统不能仅凭峰值不同判冲突。
7. 多次执行：同工具 raw/contrast、summary/targeted 的记录均保留；完全相同 execution key 不重复算成本。
8. 资源阻塞：需要 mesh/atlas 的 followup 缺资源 → blocked/partial；不伪造邻接，不自动下载，不错误 pass。
9. 失效 replan：新证据使旧计划不适用，replan/repair 均失败后必须明示 planning_blocked，不能用旧 stop 绕过 gate。
10. 数值边界：零能量/近零分母/缺帧/非有限值，不产生伪无穷质量分数或无来源判断。

真实验收：

- 使用现有 accordion_01b 缓存，不强制重新生成 TRIBE。
- L1 的 contrast 时序与便宜空间扫描真实执行，并导出 provenance。
- 若启发式触发，完成一条真实定向检查路径并说明结论；不预设一定拒绝 accordion。
- 至少验证一次真实 DeepSeek revise_plan 被接受，且后续执行使用其合法改变；若真实数据无重规划需求，不为了验收制造异常。
- 可在受控 synthetic 数据上用真实 API 验证 replan，标为 real_api_synthetic_data；不能写 real_data_passed。
- mock、synthetic、真实 API、真实 fMRI 产物四种证据分开标注。
- 对比 V1.3/V1.4：问题覆盖、动作、repair/revise、tokens、工具耗时。深度更多不直接等于质量更好。
- 真实缺陷 precision/recall 仍需独立标注；本轮测试不产生生物学有效性声明。

## 16. 操作入口与交付

新命令示例仅在实现后可运行：

~~~bash
uv run python -m react_agent.fmri.cli check-image \
  --image /home/zxuff/data/Uncertainty-aware-Blur-Prior/data/things-eeg/Image_set/training_images/00003_accordion/accordion_01b.jpg \
  --config configs/fmri_check_tribe_image16_diagnostic_v1_4.yaml \
  --policy planned --backend deepseek \
  --out runs/tribe_image16_diagnostic_v1_4_accordion
~~~

CLI/Studio 使用相同 resolved config，不改变空表单原默认。

交付：

- 新 diagnostic YAML、ledger/StopGate、provenance、多 execution 支持；
- stimulus temporal 增强、surface_spatial_sanity、必要的 targeted 工具扩展；
- planner_v2 和 memory curator 的小幅增补；
- 分角色 API 统计与上下文大小诊断；
- docs/fmri_diagnostic_v1_4.md；
- version/version_1.4/agent.md；
- 完整验收表：implemented、synthetic_passed、real_api_passed、real_data_passed、not_run。

实施顺序：

P0：证据合同与停止门槛 → signal provenance/多 execution → L1 时序与空间 → L2 触发与定向分析 → replan 验收。

P1：上下文压缩、memory 新字段与可读报告，完成本轮适用的回归。

不在本轮引入新的大型医学模型、自动生成新刺激实验、在线更新质量阈值或多 agent 系统。

## 17. 设计依据与限制

- [TRIBE v2 官方仓库](https://github.com/facebookresearch/tribev2)：输出为皮层表面预测，官方说明包含时间偏移补偿。具体运行以本仓库已核对的 segments/sidecar 为准，不据 README 再加一次 shift。
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)：支持确定性流程与动态工具选择的组合。本文的诊断层次、路由启发式和停止规则是本项目的设计选择，不是该文档给出的 fMRI 标准。

本轮目标是“更完整且可审计的数值诊断”。要把伪标签用于筛除或赋训练权重，后续仍需独立故障标注、兼容参考或下游对照实验。
