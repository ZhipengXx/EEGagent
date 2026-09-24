# Cursor 增量实施 Prompt：TRIBE v2 检查工具包与动态选择，V1 → V1.1

把本文件交给 Cursor，在现有 react-agent 仓库继续修改。优先级是新增并接通实际可执行的检查工具，同时用真实执行轨迹验证 DeepSeek 选工具。不要重新搭建 V1 框架。

## 1. 已知现状与本轮目标

用户的 version 1.0 文档说明：
- 仓库：/home/zxuff/data/EEGagent/react-agent。
- 已有 agent 和 fmri_check 两张图，状态独立。
- fmri_check 已有 registry、SampleSpec、DeepSeek JSON Action、rule / hybrid policy、成本与报告。
- 节点在 src/react_agent/fmri/loop.py，图在 src/react_agent/fmri/graph.py。
- 已实现 validate_input、basic_statistics、temporal_diagnostics、roi_summary、reference_distribution。
- cortex_mae、semantic_consistency 尚未实现。
- TRIBE 适配器已读取 .npy + sidecar。
- static_1s_7s smoke 使用 [12, 20484]、1 秒采样、fsaverage5。
- 已跑 8 条 rule 样本，含 gray_control。
- 当前没有 ROI map，冻结参考统计只有 3 条样本。
- 当前 claim_scope=numeric_consistency_only。

以上是文档描述，必须先核对实际源码、配置和 events.jsonl，不把文档当作代码审计结果。保留用户已有改动，不修改用户的原始 TRIBE 数组，也不重新运行生成器。

本轮目标：
1. 保留现有执行骨架。
2. 封装灰色对照、时间定位、皮层 ROI、跨图像输出区分等具体工具。
3. 为每个工具提供明确的问题、前置条件、结果和解释边界。
4. 让 hybrid 策略在多个真正可用的工具间选择，根据新结果继续或停止。
5. 完成离线行为验证；条件具备时完成有明确调用上限的 DeepSeek smoke。
6. 新工具实现完成但真实资源缺失时，准确区分“实现完成”“合成验证”“真实验证未完成”。

本轮不训练脑模型，不自动下载大权重，不开发多 Agent，不重建 Web UI，不合并聊天图。继续使用 uv run 和现有 Python 环境。

## 2. 先修正现有实现中可能存在的问题

先审查并报告下列项目，确认后直接修复：

A. 三条参考：
- 当前 reference_distribution 是否用 N=3 计算 median / MAD 或分位数并直接触发 flagged？
- N=3 只能用于工程 smoke 的相对差异描述；默认不能驱动样本质量 verdict。
- 增加 calibration_status、reference_n、reference_source、reference_version。
- insufficient_reference 输出描述性差异与限制，但不产生具有质量判定权限的 finding。
- 最小样本数配置只是工程门槛，不等同于充分校准。任何更大固定数字也不能自动证明参考分布可信。
- 正式启用的阈值必须有独立、冻结的校准流程；不得由 LM 调节。
- 参考内样本因 fingerprint 重叠被 skip 时，不得在报告中声称完成了 reference 检查。
- reference 被声明为 required 时，skip 必须产生 coverage=partial；若是 optional，可通过更窄范围的检查，但明确 reference 未评估。

B. 错误分类：
- 验证工具执行 error、资源 unavailable、描述性 observation、数据异常 finding 必须分开。
- 网络失败或 atlas 缺失不等于 fMRI 异常。
- 不应仅因为 finding 文本里包含 error，就给样本 flagged。
- 在现有 schema 增加非破坏性的 decision_effect=none / flag / block 等明确字段；旧数据经迁移有确定含义。

C. DeepSeek：
- 核对 fmri_check 用的是 JSON Action 还是原生 function calling。
- JSON Action 不要求模型支持原生 tool_calls，不能沿用聊天图的“必须 tool-calling”的限制。
- 保留用户当前可用模型配置，不无条件替换 model ID。
- 对“禁止一切 reasoning 模型”的说明，改为基于具体版本、能力与适配器测试的说明。
- 原生 thinking tool calls 若以后启用，再按官方协议处理；这次不为该功能重写已能工作的 JSON Action。

D. 路由：
- 候选列表是否只剩固定规则挑出的一个工具？
- completed / failed / unavailable 是否区分清楚？
- LM 在可选工具尚未执行时 stop，是否可能跳过必需证据？
- LoopRuntime 是否存在跨样本共享可变状态或预算串用？
- 原有预算、重复动作和无进展终止机制是否仍生效？

## 3. TRIBE 的数据身份和时间轴

为当前样本添加或补齐只读资源引用，可存入已有 metadata / resources，不必推翻 SampleSpec：

- control_sample_id / control_artifact_id
- spatial_layout：左右半球顺序、每半球顶点数、是否完整顶点、valid_vertex_mask
- mesh / atlas 资源 ID、空间与版本
- 时间信息：实际输出 segments 或其可追溯 sidecar 映射
- generator checkpoint / config / modality preprocessing identity
- cohort_manifest_id：用于跨图像描述的固定样本集合
- reference_artifact_id：用于已校准分布比较，与 cohort 区分
- image_id / source_image_fingerprint：区分不同图像与重复导出

要求：
- [12, 20484] 是当前 profile 的显式契约，不写成所有 TRIBE 输出的唯一可能形状。
- fsaverage5 + 长度相同不足以证明顶点顺序一致。
- 只读查看用户本地 TRIBE 的生成、导出和 plotting 代码，确认 LH/RH 拼接、mask 与输出 segments。
- 官方 predict 返回 preds 和对应 segments，可能去掉没有事件的片段；不能仅用行号猜原视频秒数。
- 对当前“preds[k] 对齐 t_video[k]、不再加 5 秒”的说明，应记录支持它的本地实现和 sidecar。检查器不要自动增加或减去 5 秒。
- sidecar 与源码无法核对时，time_alignment_status=unverified，继续可做的数值检查，但禁用需要严格时间对应的结论。
- none_raw_signed 描述了导出形式，不足以证明模型目标未经标准化；记录 scaling_source，未知就保持未知。
- 不使用样本名称中的 apple / gray 作为科学标签或自动匹配依据；gray_control 的绑定来自显式配置和相容性检查。

继续拒绝把旧的派生切片当完整原始输入。但允许工具在独立 artifact 目录中创建有来源记录的 contrast / ROI / epoch 数据；这些派生产物不能覆盖源数组。

## 4. 工具契约：扩展当前 registry

复用已有 ToolSpec / ToolResult，增加以下可选字段及兼容处理：

ToolSpec：
- scope：sample / pair / cohort
- question：这个工具能回答的具体问题
- requires：所需资源或前置工具产物
- produces：产物类型
- issue_types / evidence_family
- cost_class、estimated_cpu_seconds、estimated_gpu_seconds
- valid_claims / unsupported_claims
- allowed_args schema

ToolResult：
- execution_status
- applicability：applicable / partial / not_applicable
- metrics、findings、evidence_refs、artifacts
- diagnostic_question、answer_summary
- calibration_status
- decision_effect
- limitations
- resource_fingerprints、工具版本、参数、时间和成本

工具依赖资源通过 Runtime 中的 ResourceResolver / ArtifactStore 获取。LM 只能选择已登记的资源 ID，不能提供自由路径、shell 或 Python 代码。

availability 检查应廉价：只读缓存或元信息。不要为了“判断工具是否可用”就先做完它的全部计算或隐式下载。

同一份 contrast、ROI feature、mesh adjacency、cohort index 只计算一次；依赖的工具读取 artifact。缓存 key 必须包含数据、control、atlas、时间选择、mask、工具版本与相关配置 fingerprint。

## 5. 本轮优先实现的四个领域工具

### T1. gray_control_contrast：图像刺激相对灰色对照有什么变化？

优先级：P0。本工具使用现有 gray_control，是当前最直接增加的信息来源。

输入：
- 样本预测 Y[T,V]。
- 显式配对的灰色对照 G[T,V]。
- checkpoint、生成协议、时间轴、空间映射、标准化及有效顶点信息。

预检：
- 确认两者除待比较的视觉刺激外，其余已知生成配置可比。
- 不允许只凭 shape 相同就认为配对成立。
- 找不到对照、对照与自身相同、关键配置冲突时给出明确状态。
- gray_control 自身输出 not_applicable，而非低响应异常。
- 原始 fMRI 内的刺激前片段不能自动替代独立的整段灰色视频对照。
- 元数据不全时可返回 partial 的描述性结果，但不能把它当已验证的配对效应。

计算：
- Delta = Y - G，原始 signed 值保持不变。
- 以有效顶点集合 M 计算每个时间点的 RMS：
  response_rms[t] = sqrt(mean(Delta[t,M] ** 2))。
- overall_delta_rms、max_response_rms、peak_frame。
- 若时间映射已核实，补充 peak_time_s。
- 可输出相对对照 RMS 的归一化量，但必须同时报告分母；分母接近 0 时 normalized 值为 null，不能制造极大 ratio。
- 空间比较明确采用按顶点等权或表面积加权，并记入报告，默认等顶点权重。

产物：
- contrast array 的本地 artifact 引用。
- 12 点或真实 T 长度的 response curve。
- 资源匹配报告。

解释：
- 只能说明模型输出对图像与灰色输入有多大差异。
- 一条确定性灰色对照不是生理噪声估计，不能输出 SNR、p-value 或“统计显著激活”。
- 响应很小只产生描述或由配置明确启用的 candidate flag，不能自动判为错误标签。

### T2. stimulus_temporal_profile：差异在什么时候出现，是否由少数跳变主导？

优先级：P0。在已有 temporal_diagnostics 上新增明确的 contrast mode，或实现一个薄封装。避免复制同一套数组运算。

输入：
- 原始数组；若已有有效 contrast，优先分析 contrast。
- 明确的 output-time mapping 与 stimulus_events。
- 若没有 contrast，结果标为 raw_temporal_description，不能混成 stimulus-specific effect。

计算：
- 每帧 RMS 曲线、相邻帧差分 RMS。
- peak_frame、peak_relative_position、max_step / median_step 等描述。
- 对近零分母明确处理。
- 时间映射 verified 时，按元数据生成 pre / stimulus / post 的索引，并报告每段实际样本数。
- 对当前协议，4 秒灰、1 秒图像、7 秒灰只是核实后的标签来源，不硬编码全局假设。
- 仅有 1 个刺激期时间点时不进行组内方差估计或显著性检验。
- 需要时标记异常集中位置，以便后续 ROI 细查。

产物：
- temporal_profile.json。
- response / step curve 小图，可由 deterministic plotting 生成。
- event-window coverage。

解释：
- 不要求固定 5 秒峰值。
- 若模型上下文非因果或窗口机制未确认，不能把 pre-stimulus 差异称作信息泄漏。
- T=12 的序列不用于可靠的静息态频带 / 功能连接 / HRF 拟合判断。
- 时间结构异常须区分描述性极值与有校准依据的告警。

### T3. surface_roi_profile：变化位于哪些真实皮层区域？

优先级：P0。扩展现有 roi_summary，首先解决真实 ROI 资产缺失的问题。

资产准备：
- 增加 prepare-assets 命令，优先使用本地缓存。
- 选择与输出一致的 fsaverage5 表面 atlas。
- 可以优先使用 Nilearn fetch_atlas_surf_destrieux 返回的 map_left、map_right 和 labels；这是结构分区，不自动等同于功能网络。
- 不把 MNI 体积 atlas 的向量直接拼到 fsaverage5 输出上。
- 如果用户已有经过核实的 Schaefer / Yeo 表面版本，也通过同一资产接口加载。
- 下载必须由显式命令或已授权的实现步骤触发，不在 import 或 availability 中触发。
- 官方公开 atlas / mesh 小型资源可为本轮准备；网络不可用时保留真实 provider 实现与本地导入方式，不能用随机 ROI 替代真实验证。

空间验证：
- 确认输出的每半球顶点顺序和 atlas 相容。
- ROI 主键使用 hemisphere + label_id，避免把左右同编号脑区错误合并。
- atlas 的 unknown / medial-wall / excluded labels 通过显式映射处理，记录被排除比例。
- 缺少 valid-mask 时标记覆盖限制，不能宣称每个顶点都有可靠解剖含义。
- 记录空间/atlas 来源、版本、hash 及核实方法。

计算：
- ROI-wise signed mean timecourse。
- ROI-wise RMS timecourse，保留正负抵消的差异。
- 在固定、预声明的窗口或全序列汇总 ROI 响应强度。
- top-K ROI 的名称、半球、有效顶点数与数值。
- 可选 surface snapshot，用固定或明确记录的色标，不用图像颜色代替数值。
- 没有 contrast 时只报告 raw output 的空间分布。

如果需要“视觉相关 ROI 比例”：
- 用固定、注明来源的 ROI 集合定义。
- 同时考虑集合大小与其余 ROI；不把大 ROI 的总能量天然当更强响应。
- 不允许 LM 根据看到 apple / face / scene 就临时编一个正常 ROI 集合。
- V1.1 默认只描述，不设“视觉区必须最高”的通过标准。

### T4. cross_image_specificity：不同图像是不是产生几乎一样的输出？

优先级：P0。它是 pair/cohort 工具，需要显式 cohort manifest，不能只读一个数组假装完成。

cohort 与 reference 分离：
- cohort 是固定的比较面板，用于输出之间的描述性差异。
- reference 是用于已校准异常分数的外部资源。
- 当前 8 条可以用于工程 smoke；样本数、类别组成和重复图像都要报告。
- 不把这 8 条当正常人脑分布。

实现：
- 对每个样本使用一致的 mask、时间表示与归一化设置。
- 优先缓存相同定义的 contrast feature 或 ROI × time feature。
- 没有对照 / ROI 时允许使用明确标注的 raw feature 模式，但解释范围更窄。
- exact hash 检查重复数组。
- 计算 query 与固定面板其余样本的 cosine / Pearson（退化向量返回 undefined）及绝对或归一化 L2 距离。
- 输出 nearest_neighbors、similarity_summary、n_compared、representation_id。
- 源图像相同但重新导出的样本单独标记，不混同不同图像输出坍缩。
- self comparison 必须排除；若无法确认 source_image_id，限制结论。
- raw 高相似可能由共同背景成分主导，必须与 contrast 模式区分。
- 不对一万条默认建立完整 N×N×T×V 数据；使用固定小面板、分块相似度或已有索引，并记录实际比较范围。
- 构建 cohort index 的费用独立记录，后续 query 缓存命中不重复计费。

解释：
- 不同图像的输出完全相同或高度接近是需要定位的问题。
- 图像本身可能非常相似，因此“高相似”不自动证明错误。
- 没有图像侧表征时不能声称完成语义一致性检查。
- 结果最多支持 output_diversity / collapse_candidate。

## 6. 下一层工具：做清楚依赖，再逐个接入

### surface_spatial_sanity（P1）

资源齐全时可在本轮实现，否则单列后续任务：
- 输入 fsaverage5 的真实 mesh faces、valid-mask、原始或 contrast 数组。
- 基于 mesh adjacency 计算相邻顶点差分 / graph roughness、孤立热点和半球差异的描述。
- 不将展开后的相邻数组索引当皮层邻居；不得跨半球连边。
- half-to-half 不对称通常不等于错误。
- roughness 对标准化、平滑和 mesh 有依赖；无匹配校准时只提供描述性 evidence。
- 合成 vertex permutation 测试用于验证敏感性，不称为真实生理错误 benchmark。

### cortex_mae_embedding_reference（P2）

不要继续仅新增一个“能返回成功”的占位函数。真正接入至少需要：
1. 明确选择 parcel / flat / volume 模型。
2. 验证 TRIBE surface 到该表示的真实转换。
3. 验证时间长度、采样率、mask 与标准化要求。
4. 真实权重和真实 forward。
5. 明确 embedding 聚合方式和参考数据。
6. 验证距离或 probe 与要检测的问题有关。

当前 12 帧不能为了通过输入检查而静默补零、循环拼接或假装已有长序列。模型是否支持当前长度须由实际代码和实验确认。

它给出 learned representation 和相对偏离证据，不直接判断图像对应关系。不假定官方 embedding 接口包含现成质量概率或重建评分接口。

本轮没有上述条件时保留 unavailable 原因，不纳入已完成工具数。

### image_fmri_rsa / semantic_consistency（P2）

可后续利用用户已有冻结 DINO / CLIP 图像表征：
- 在预声明的 cohort 上比较 image 与 fMRI 表征的相似性结构。
- 使用固定 metric、明确样本匹配和留出评估。
- RDM 上三角条目彼此不独立，显著性计算不能当独立样本。
- 视觉相似与脑表示一致只是代理证据，可能被亮度、纹理、共同生成特征影响。
- 同源视觉 backbone 的一致性不是独立的真实脑验证。
- 真正验证 stimulus-conditioned fMRI 仍需要配对真实数据或经独立评估的对应模型。

本轮不训练这个检查器，也不把 LM 看脑图后的意见叫 semantic_consistency。

## 7. 从固定顺序变成基于问题的工具选择

保留原图中的 select_action / validate_action / execute_tool 闭环。

将当前 if temporal → ROI → reference 的硬总顺序保留为 rule baseline，但 hybrid 需要：

1. 生成 question ledger：
- 输入/映射是否可信？
- 是否存在相对于 gray 的变化？
- 变化集中在何时？
- 变化位于哪些区域？
- 与其他图像输出能否区分？
- 是否有可用的、校准过的分布检查？
每项保存 untested / described / flagged / unresolved / not_applicable，并引用真实证据。

2. 生成 candidate list：
- 输入、资源、依赖、预算、去重均满足的工具都可以成为候选。
- 排除技术上不能执行的工具。
- 不因为 rule baseline 先选了一个工具，就把剩余可执行工具全部隐藏。
- 在 observation 中给每个工具的 question、预计成本、可解决问题和局限。
- 排序可以用启发式，但不能伪称已估计真实 information gain。

3. DeepSeek 选择一个动作：
- name、args、question_to_resolve、evidence_refs、简短 reason。
- 不要求长篇思维链。
- 模型不能修改参考集、阈值、资源路径或最终 claim scope。
- 参数限定为合法资源 ID、固定窗口或 ROI ID。
- 不允许反复挑峰值/子区域搜索出一个“通过”的结果。若探索性选择了窗口，报告其 exploratory 性质。

4. 执行后更新证据并重新选择：
- control contrast 发现响应集中 → temporal 或 ROI。
- ROI 内变化丰富但不同图像总体相似 → specificity。
- 同一空间位置出现孤立极值 → spatial sanity（已实现且资源可用时）。
- 无可解决的问题或预算耗尽 → 停止，保留未回答问题。

下面是示例轨迹，不是必须照抄的统一顺序：
- A：基础检查 → gray contrast → temporal profile → stop。
- B：基础检查 → gray contrast → ROI profile → specificity → stop。
- C：空间身份无法核实 → 数据契约说明 → 执行不依赖 ROI 的检查 → inconclusive / partial。

至少一个 hybrid 验证状态应有两个以上合法候选；日志保存候选全集、选择结果和执行后改变的证据。

## 8. 成本与必需检查的正确结合

- 保留现有预算器。
- 必需检查可以固定；不要把所有新工具设为每个样本必做后再声称有动态选择。
- 建议定义 numeric_minimal 与 tribe_diagnostic 两个显式 profile。
- numeric_minimal 保留现有 V1 意义。
- tribe_diagnostic 要求契约与基础检查；对于非 control 样本，把已匹配对照的检查设为需要回答的问题，并允许按资源情况给 partial。
- 时间、ROI、跨样本检查由疑点、用户需求和 budget 决定。
- atlas 预处理和 cohort index 构建作为共享 setup 成本；样本调用成本单独记录。
- 各工具 CPU / GPU / API 费用分列，不制造未经测量的 GPU 成本。
- 先用实测 smoke 时间更新粗略 cost profile，不写死“高级工具一定贵”。
- 新增工具可能耗尽原 max_tool_calls；调整示例预算并明确必要检查也计数，不能偷偷绕过上限。
- 预算包含 DeepSeek 的决策、修正、重试和摘要。

## 9. 报告范围与 schema 兼容

保留旧 report 字段和解析兼容，新增 schema minor version 或明确迁移。

报告增加：
- coverage_by_question
- evidence_scopes，例如 numeric、control_relative、temporal_descriptive、surface_descriptive、cross_sample_descriptive
- unassessed_claims，始终包含未实际验证的 stimulus-conditioned biological correctness
- 每个资源的 verified / unverified / incompatible 状态
- findings 的 decision_effect 与 calibration_status
- calibration / reference / cohort / atlas 身份
- candidate、selected、executed、skipped 数量
- 实际 LM 决策数、fallback 数和非 fallback 成功决策数
- setup 与 per-sample 成本分开

当前 claim_scope 可以为兼容继续保留 numeric_consistency_only，并用 evidence_scopes 补充描述。不要只是把字符串改为 biological_validity 来夸大能力。

保留 verdict：
- invalid_input
- flagged
- inconclusive
- passed_configured_checks

限制：
- flagged 表示具体检查的 flag，不能翻译成伪标签一定错误。
- passed_configured_checks 不能声称全部工具都已运行。
- 描述性差异和低校准证据默认 decision_effect=none。
- genuine biological confidence / training_weight 没有独立校准时保持 null。
- LM 摘要不能覆写确定性判定或添造数值。

## 10. 实际实施顺序与文件改动

先审查，再基于实际布局修改。建议新增或扩展：
- tools/control_contrast.py
- tools/temporal.py 或现有 numeric.py 的 contrast mode
- tools/spatial.py：ROI profile
- tools/specificity.py
- assets.py：surface / atlas 资源管理
- cohort.py：固定面板索引及查询
- resources.py：control / atlas / cohort 资源解析
- policy.py、loop.py：question ledger 和丰富候选集
- schemas.py、reporting.py：兼容字段
- CLI：prepare-assets、build-cohort-index、原有 check / batch
- configs/fmri_check_tribe_v1_1.yaml
- docs/tribe_tools_v1_1.md

文件名可以按已有结构调整。不要为了名字一致而复制现有功能或建立第二套 registry。正常新增工具不应要求修改 graph.py 拓扑。

P0：
1. 修正三条 reference 的判定权限。
2. 核实空间/时间身份，连接真实 control。
3. 实现 T1、T2、T3、T4 与 resource provider。
4. 补充 hybrid 的候选描述、question ledger 与日志。
5. 完成测试、demo 与条件具备时的真实 smoke。
P1/P2 仅在 P0 稳定且资源具备时推进，准确列出状态。

## 11. 验收必须包含的实质行为

数值工具：
- Y=G 时 contrast 为零，ratio 分母退化被正确处理；gray 本身不被误判。
- 已知时刻/ROI 的合成 perturbation 能被定位，原数组不变。
- 左右半球相同 label_id 不被合并。
- 相同长度但错误空间身份的 atlas 被拒绝。
- control checkpoint / profile / time mapping 不兼容时，不能输出可靠配对效应。
- 不同 image_id 的完全相同输出被 duplicate 检查识别。
- query 不与自身匹配。
- 三条参考默认不能给质量判定权限。

路由：
- 至少两种不同有效工具路径。
- 至少一个状态具有两个以上可选工具。
- 第二次决策确实收到第一次工具执行结果。
- LM 提前 stop 不能跳过 profile 的必需证据。
- 不可用工具不能被调用；缺失 atlas 不导致样本异常。
- fallback、无进展与预算耗尽都可追踪。
- 使用相同观测但不同预算，验证可执行动作集合合理改变；不要强行要求模型必须随机选不同动作。

现实数据：
- 先在文档中的 8 条上跑 rule 与 hybrid，产物分目录，不覆盖旧报告。
- 只有本地已配置 API key 且执行真实 smoke 时才做付费调用，并设置单样本和全批调用上限。
- 记录真正使用 DeepSeek 的样本数与决定数；全程 fallback 不算模型路由验证成功。
- 至少一个真实 hybrid trace 展示：候选 → DeepSeek 选择 → 工具真实执行 → 新证据 → 下一次决定。
- 若现有真实资源无法产生两个合法候选，就说明阻塞资源，先用 mock transport / 合成数据验证接口，不伪称真实多工具选择已经验证。
- 没有本地数据访问或 key，就把对应项标为 not_run，并继续完成可测试部分。

评估边界：
- 合成扰动的检测率只证明已知错误模式敏感性。
- “通过率提高”不作为优化目标。
- 8 条 smoke 不足以报告真实性准确率。
- 后续 benchmark 可加入错配 control、空间映射错误、时间乱序、NaN、复制输出等已知问题，并保留正常结构的对照；明确这是工程错误集。
- 生理合理性或对 EEG 学习是否有益，必须另设独立评估，不在此轮声称已经解决。

## 12. 最后给用户的结果

完成修改后，报告：
1. 本次真实实现了哪些工具，每个工具输入、输出、适用条件。
2. 可复用旧实现的部分以及修复的 bug / 文档错误。
3. 本地实际可执行的资源准备与检查命令，统一 uv run。
4. 一个有真实数值和证据 ID 的工具输出。
5. 两条不同执行路径，以及真实 DeepSeek 路由是否已验证。
6. 新旧 8 条报告差异，特别是三条 reference 不再越权决定质量后发生的变化。
7. 耗时、API usage、费用可用性、fallback 统计。
8. 未完成项与缺失资源，区分未实现和未验证。
9. 如何新增第 5 个或第 6 个工具而不改图结构。

请直接从实际代码审查开始，并连续完成 P0，不要仅返回设计建议，也不要把 TODO / disabled 占位算作已完成工具。

## 官方资料与事实边界

本文件的工具指标与 routing 是项目设计方案，不是 TRIBE 或 CortexMAE 作者已经验证的质量判据。

- TRIBE v2 官方仓库：https://github.com/facebookresearch/tribev2
- TRIBE predict / segments：https://github.com/facebookresearch/tribev2/blob/main/tribev2/demo_utils.py
- fsaverage5 表面 Destrieux atlas：https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_surf_destrieux.html
- CortexMAE 官方 embedding 接口：https://github.com/MedARC-AI/CortexMAE
- DeepSeek tool calls：https://api-docs.deepseek.com/guides/tool_calls/
- DeepSeek JSON output：https://api-docs.deepseek.com/guides/json_mode/

