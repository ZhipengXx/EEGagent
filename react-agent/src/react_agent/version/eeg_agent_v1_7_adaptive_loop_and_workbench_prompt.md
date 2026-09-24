# V1.7：自适应实验循环与研究工作台

这是下一版的增量开发任务，不是实现快照。依据为用户提供的 V1.6 技术记录和三张工作台截图。尚未读取实际服务器仓库，所有定位与迁移必须以现场代码为准。

目标仓库：/home/zxuff/data/EEGagent/react-agent。

## 判断与参考

V1.6 已经有训练执行器、缓存、日志和曲线。当前工作台不调用 DeepSeek，固定衔接 baseline 与 weight_decay 两个配置；它是有限的实验自动化。AutoML 和 agent 并不互斥：本轮要增加的是根据观测选择行动、修改计划和使用经验的能力，而不是靠命名或增加角色数证明 agent 能力。

预算上限可以固定，下一步与实际停止时机应由状态、证据和资源决定。把两次改为十次、每轮强制调用一次 API 或增加 memory 文本框都不构成此升级。

参考：

- Anthropic, Building effective agents：https://www.anthropic.com/engineering/building-effective-agents 。参考动态工具选择、环境反馈与有界循环。
- GOME：https://arxiv.org/html/2603.01692v1 。参考将反馈用于下一次修改、显式假设及条件化经验。
- AgentEvolver：https://arxiv.org/html/2511.10395v1 。参考经验指导和事后分析；本轮不训练策略模型，不声称实现权重自进化。
- Carbon Data table：https://carbondesignsystem.com/components/data-table/usage/ 。参考紧凑表格、展开详情、行列对齐。
- Carbon Empty states：https://carbondesignsystem.com/patterns/empty-states-pattern/ 。参考简洁、具体、带下一步操作的空状态。

以下产品与技术要求是针对本项目的设计，不是上述工作的完整复现。

---

## 给 Cursor 的任务

请在现有仓库实施 V1.7，复用已接线训练器，不另起一个只演示的 agent 或 UI。按 P0、P1、P2 顺序完成；提供实现、必要测试、截图与真实运行证据。不得只更新说明文档。

### 0. 审计与边界

先阅读 AGENTS.md、V1.6 文档和真实入口。查清 UI→HTTP handler→训练启动的调用路径，与 CLI / bounded_research / planner 的路径是否分叉。列出可复用 backend、memory、budget、event、evaluator 的真实文件位置。

尤其核对：

1. UI 中自动两试是否直接在 handler 里串联两个 subprocess。
2. validation/test 的真实来源、candidate bank、重复试次与被试聚合。
3. 多卡评价是否是 per-rank within-batch，是否丢掉末尾 batch。
4. GPU seconds 是否直接传给 subprocess timeout。
5. 最近 campaign 是否为全局变量，造成新建表单和历史结果串状态。
6. 重复表格、裁切脑图来自重复订阅/渲染、无效 HTML、溢出布局还是截图拼接；先复现，不凭截图断言根因。

输出 docs/v1_7_inventory.md，并继续实施已明确的任务。

模块 A 的数值检查合同、判决和 claim_scope=numeric_consistency_only 保持；可修展示，不重跑 TRIBE、不覆盖旧 runs、不修改旧版本快照。

模块 B 独立维护 research_scope、研究目标、划分与指标，不将验证分数写入 A 的通过卡片。B 现有 API client 可复用，不能强行借用 A 的 StopGate 充当实验比较器。

不重写前端技术栈、不默认引入向量数据库、不引入多 agent 框架、不自主安装新模型。先用现有技术实现单 controller 和持久化状态。

## P0：先修会影响结论和运行可信度的问题

### 1. 明确训练策略与泛化协议

禁止用一个 intra/inter 布尔值代表所有设置。拆开两个维度：

- training_strategy：per_subject 或 pooled_subjects。
- generalization_target：seen_subject_unseen_stimulus 或 held_out_subject；保留额外明确命名的自定义模式。

每被试独立模型与多被试合训一个模型必须分别展示。选中全部被试合训且没有 held-out subject 时，不能显示为跨被试泛化；应明确为“多被试合训 · 图像留出验证”。用户要求 held_out_subject 但未指定留出对象时，阻止启动并定位到该表单项，不静默改变模式。

保存 SplitManifest，包含训练、选择用验证、最终测试的文件来源、样本 ID、图像/概念 ID、subject ID、用途和哈希。按声明目标校验交集；重复 EEG 试次不能跨越本应互斥的图像/概念划分。

文件名 test.pt 不自动决定它的统计角色。若用作选择，它就是开发/验证数据，不能再称为独立最终测试。若保留旧协议，明确旧 split role 与其局限，不回填成新协议。新 campaign 默认只从训练池构建符合泛化目标的 validation：

- seen_subject_unseen_stimulus：按图像或概念分组留出，分组层级按任务定义。
- held_out_subject：外层测试被试完全留出；模型选择使用训练被试中的内层被试划分，或明确标记为代理验证而非跨被试验证。
- 数据不足以建立要求的划分时返回 actionable blocker，不擅自更换目标。

关闭每次 trial 后自动展示 test 的默认行为。campaign 确认停止并冻结推荐结果后，由独立 evaluator 阶段完成最终测试。test 结果不发给正在规划的 controller 或检索 memory。

可在创建任务时选择“研究完成后进行最终评估”，不要求每次实验额外批准。若用户在看过测试结果后继续调参，创建有 provenance 的新 campaign，并记录该 test 已暴露；不能继续宣称它是未见测试。

### 2. 实现稳定的检索评价

within-batch top1/top5 仅作为训练诊断，不能作为 V1.7 跨 trial 主选择指标。

新增或修正 evaluator：

- 对全部 validation query，在冻结 candidate bank 上评价；可以分块计算，但每个 query 的比较集合一致。
- candidate bank 来自 validation 协议，不固定宣称为 200-way。K 由清单决定；若采用固定 200-way 开发协议，候选 ID 与抽样规则必须先冻结。
- 保存 target_feature_hash、candidate_bank_hash、query_hash、similarity、aggregation、positive_definition、tie_policy、metric_version。
- 同一图像对应多被试/多 trial 时，固定候选图像去重规则，使用实际图像 ID 建立正例关系；不能仅按矩阵对角线假定唯一正例。
- 不静默 drop_last；跨 rank 评价按 query 数正确聚合，不能均匀平均不等大小 batch 的准确率。
- 训练 batch、evaluation chunk size 和 GPU 数变化，不改变固定 checkpoint 的候选集合或结果，允许数值容差。
- 主指标可用 fixed_bank_top1，次指标 fixed_bank_top5；其他指标仅在已有实现和清楚定义时暴露。

单 query 单正例时 Top-k 是“正确图像进入前 k 的 query 比例”；多正例需要另存分母定义。不要将 mAP 与平均倒数排名在未核对正例定义时混用。

旧报告约 0.363 的值保留原 evaluator 标签。不迁移成新固定候选指标，不和新结果放在同一排名榜。新旧协议不兼容时，旧 checkpoint 可用于重新评价，但结果是新 evaluation artifact，不覆盖历史。

### 3. 分离 GPU 预算、墙钟与训练早停

新增并分别显示：

- campaign_gpu_seconds_limit：研究任务分配 GPU 的累计时间上限。
- campaign_wall_seconds_limit：整个任务墙钟上限。
- per_trial_wall_seconds_limit：单训练进程上限。
- max_trials、max_action_steps、max_lm_calls：独立安全上限，不是计划目标。

GPU 秒数按本任务实际持有的 GPU 数与持续时间积分记账；固定分配时为 n_gpu × elapsed。注明这是分配时间，不等于 GPU 利用率积分。2 张卡 4 小时即 8 GPU 小时，不能允许每个 trial 重获全部 campaign 预算。

启动前预留预算；运行中累计；结束释放未用预留。CPU 动作不扣 GPU 时间。失败、重试、验证和重复 seed 的实际资源都入账。api_usd 未知保持 null，但不在普通首页反复显示内部字段。

没有选择 GPU 时不能仅省略 CUDA_VISIBLE_DEVICES 并继承全部显卡。界面提供“自动选择一张空闲卡”或显式选择；启动时实际解析并绑定设备，无可用资源则 queued/blocked。保留页面打开不自动扫描机器的行为，提供“刷新数据与设备”动作。不得停止其他用户任务。

epoch patience 仅控制本次训练，不决定整个研究循环结束。将 early_stop_min_delta 的单位明确为 fraction；0.001 即 0.1 pp，不是 0.001%。

### 4. 修 UI 显示缺陷与状态串扰

先复现第三张截图里的脑图裁切、密集重复 10–15 行、长页面溢出。检查 DOM 唯一性、重复 mount/subscription、HTML 闭合、position、overflow、sticky 容器和 key。不要通过删数据或隐藏全部内容掩盖问题。

- 时间表用 frame_index 作稳定键，每份数据只渲染一次。16 点输入必须恰好有 16 个唯一时刻行。
- epoch 历史按 trial_id + epoch + metric 去重，流式重连不能重复追加。
- 选择一个 run 后所有状态、图、进度和文字都绑定该 run，不读全局“最近有数据的任意任务”。
- 新任务表单与历史任务详情分开；不得新表单为空而曲线显示旧成绩、顶部已完成但日志等待操作。
- 已早停的 23/50 显示“已完成 · 提前停止，训练 23 轮”，不留下 46% 的未完成任务进度；若保留 epoch bar，明确只是训练轮数比例。
- 刷新仅重建读视图，不重复训练或额外调用 LLM。

## P1：让 memory、planning、action 进入真正的训练主路径

### 5. 单一 controller

创建或重构 ResearchController，使 CLI、工作台、resume 共用相同 engine/service。删除 UI 私有的 baseline→weight_decay 自动续跑分支；旧固定模式可保留在 legacy_fixed policy，用于复现与对照，不是新的 autonomous 模式。

建议目录按现有结构调整：controller、state、plan_store、action_registry、memory_retriever、evidence_builder、runtime_validator、budget、presentation_adapter。不要建立两套重复的 executor。

UI 启动仅提交冻结 GoalSpec 和预算；训练 worker 不决定下一个 trial。controller 在完成事件后决定下一步。

### 6. 持久化研究状态

ResearchState 至少含：

- campaign_id、goal_spec_hash、task_contract_hash、protocol_version。
- lifecycle_status、state_version、last_event_id。
- active_plan_id/version、open_questions、hypotheses。
- incumbent_ref、trial_refs、latest_evidence_refs。
- retrieved_memory_refs、remaining_budget、pending_action、active_job。
- stop_reason、resume_checkpoint、recommendation_status。

生命周期显式区分 draft / preflighting / planning / executing / reviewing / paused / blocked / completed / cancelled / failed。展示状态来自 engine，不让前端根据“有没有 metrics.json”猜测。

事件必须有 event_id、campaign_id、state_version、plan_version、action_id、timestamp、payload。状态与动作提交使用乐观锁或等价机制，拒绝过期 plan、重复点击和并发调度。

已有 active_job 时恢复先重新关联并核对 PID/worker identity，不能盲目再启动。进程提交、记录与状态更新跨崩溃窗口需可恢复的 job token、输出锁或其他幂等方案。

### 7. Plan 是可修订的问题图

Plan 不等于提前写死的 trial 列表。保存：goal、questions、hypotheses、steps、dependencies、status、evidence_needed、decision_criteria、revision_reason。

Question 有 open / investigating / answered / inconclusive / blocked 状态；Hypothesis 有 proposed / tested / supported_provisionally / contradicted / inconclusive，均引用 evidence。

只把当前 ready 的一步具体化并执行。后续步骤允许写成条件分支或待定问题。新证据到来后可以插入诊断、取消已无必要步骤、选择一个新配置、请求重复验证或停止。

动态只发生在合法任务边界内：不能改变研究指标、candidate bank、test 隔离或总预算。修改任务目标、划分等根合同应创建新 campaign。

### 8. 提供真正不同的行动

每个 action schema 包含 prerequisites、typed arguments、outputs、resource_cost_kind、idempotence、side_effects、availability、cancellation_behavior。

V1.7 至少接通这些类别，而不是都伪装成 train：

| Action | 用途 | 约束 |
| --- | --- | --- |
| inspect_task_data | 查看划分、形状、重复与匹配元数据 | 只用已授权 train/validation 范围 |
| retrieve_research_memory | 围绕具体问题检索兼容正反经验 | 记录 query 与命中条件，禁止 test 成绩 |
| analyze_learning_curve | 提取曲线、训练停止、梯度/损失异常等已有信息 | 只报告实际记录的字段 |
| analyze_retrieval_errors | 在 validation 上分析排名、分组错误、正例映射 | 没有预测产物则 unavailable |
| propose_experiment | 将一个假设实例化为合法 ExperimentSpec | 不启动训练，不能直接执行命令 |
| train_candidate | 跑登记的模型和允许的配置 | 预算、白名单、独立输出目录 |
| evaluate_validation | 固定 bank 评价已有 checkpoint | 可复用产物，禁止 test |
| replicate_candidate | 对预登记 seed 做匹配重复 | 需要可比较 baseline 的对应 seed，计完整成本 |
| stop_research | 给出预算内结论或保留待研究问题 | 返回结构化原因，不要求耗尽上限 |

暂停/取消是用户控制动作，立即进入调度器；暂停应明确“当前训练结束后暂停”，若提供立即中断则有独立按钮和 checkpoint 语义。

可扩展 read_method_note 读取带出处的已批准方法说明，形成 external_prior。本轮不强制联网文献检索、自动下载代码或安装环境；读取论文不自动授权实施其模型。

### 9. 扩展可行动空间，但保持实验可控

允许在已有训练器支持的参数范围内生成配置，例如 learning_rate、weight_decay 的离散候选或有界范围。由 YAML capability schema 明确字段、类型、边界、默认值和有效组合；模型不能编写自由命令。

不要只把 baseline / weight_decay 两个名字包装成 LLM 的二选一。即使暂时只有两个训练 profile，agent 也应能选择分析错误、重复确认、继续检索经验或停止。报告同时展示实际可用训练候选数，避免夸大搜索空间。

固定 image features、EEG backbone、数据和评价协议时，优先单因素改动。若模型 profile 包含多处变化，结果仅支持整体方案比较，不生成单因素归因。

每个提议写清：observation、evidence_refs、hypothesis、alternative_explanations、intervention、expected_observable、disconfirmation、cost_estimate/source。

不要用训练 loss 下降直接宣布过拟合；需要可比较曲线和验证证据。train within-batch 与 full-bank validation 指标不同，不直接将二者差值解释为泛化差距。

### 10. 每一轮的执行语义

循环如下：

1. 处理用户暂停/取消及硬预算；必要时终止。
2. 将新完成动作转成 EvidenceBundle，记录缺失证据。
3. 更新工作记忆；按当前问题检索少量兼容长期经验。
4. 没有计划或出现实质新证据时调用 DeepSeek 创建/修订计划。
5. 请求下一项合法 action，验证 prerequisites、fingerprint、state_version、预算和参数。
6. 执行 action，保存结果；返回第 1 步。

不在每个 epoch 调用 LLM；在训练结束、诊断结果、执行失败、重要用户干预等边界调用。运行中只推送进度，不让 controller 在同一个未完成训练上无限反思。

纯依赖步骤和已有计划中条件确定的操作可以由 runtime 前进；涉及研究选择时必须留下实际 DeepSeek decision。支持 create_plan / revise_plan / choose_action / reflect_episode 的角色记录，不要求每个角色独立 API，更不要求多 agent。

建议初始配置：adaptive policy、max_trials=6、max_action_steps=24、max_lm_calls=12、max_concurrent_trials=1。这些只是可配置上限，绝不强制执行六次。GPU/墙钟上限必须另行冻结；默认数值不代表授权消耗所有预算。

允许在第一条基线之后停止，也允许有预算且出现新问题时超过两次训练。停止原因包括目标满足、无合法且有依据的下一步、资源不足、有效重复无改进达到预登记条件、外部阻塞或用户停止。

区分 plateau 与 unknown：没有足够可比实验不能说已收敛；max_trials 用完写 budget_exhausted，不写找到最优模型。cheap actions 也有步数和 API 上限。

### 11. Memory 要实际影响行动

使用三层概念，不必部署三套存储：

- WorkingState：当前计划、未解决问题、预算、最新证据。
- EpisodeStore：每个动作/实验的配置、证据、结果、失败原因和实际成本。
- ResearchMemory：可检索的条件化经验、反证、适用范围、证据级别和策略注意事项。

经验条目至少包括 task/protocol/feature/subject/metric/fidelity 指纹、观察、干预、比较对象、结果、支持与反对证据、seed、来源、等级和失效条件。

先用确定性过滤 + 简单排序，暂不要求 embedding/RAG 服务。精确可比历史才可供数值比较；相似任务仅作为 prior。支持和相反案例同时检索。禁止使用其他 campaign 最终 test 表现指导正在进行的研究。

每次 decision 保存 memory_used：引用了什么、适用条件是否匹配、如何影响候选、哪些历史经验因不匹配被排除。没有相关记忆应显示 cold_start，不能生成虚构经验。

LLM 可提议 summary，但不能改原始指标、复制次数、证据等级或人类反馈。用户批注属于 annotation，不自动升级为实验事实。缓存重播不算新证据。

可以加入有限策略约束，如某组合反复 OOM 时暂时降低其优先级；只作用于检索/候选排序，不自动扩权、改系统 prompt 或改变不可变合同。称为经验更新，不宣称 LLM 权重自进化。

### 12. DeepSeek 与结构化输出

复用当前 DeepSeek provider；模型名可配置且记录实际 response model，不自动替换供应商。JSON Action 即可实现行动，无须为了“像 agent”强制换成原生 tool_calls。

输入为精简 EvidenceBundle，不发送大矩阵和全部 epoch 原始日志。支持按需读取摘要片段。记录 input/output token、延迟、角色、prompt version、request/response model、重试与成本来源。

adaptive 模式无 key 或 API 失败时显示 planning_blocked/暂无法继续规划；允许在预算内有界 retry/一次 repair，失败不能静默变成固定两试。固定 legacy 模式明确不使用 LLM。

英文 controller prompt，单独文件并可版本化：

~~~text
You are a bounded EEG/MEG research controller.
Use the frozen goal, protocol, available actions, structured observations,
compatible memory and remaining budget to choose the next useful action.

Do not follow a fixed baseline-then-weight-decay sequence.
Actions may inspect evidence, retrieve experience, analyze errors, train,
evaluate validation, replicate, or stop. A budget is a ceiling, not a target.

Separate observations from hypotheses. Cite evidence IDs and memory IDs.
For an experiment, specify an approved intervention, an expected observable,
an alternative explanation, and a disconfirmation condition.
For a plan revision, identify the new evidence that made revision necessary.
State briefly why the selected action is preferable to another available action.

Do not change split roles, candidate banks, metrics, budgets or test access.
Do not invent available models, results, memory, costs or biological claims.
Do not compare batch-local retrieval scores across different candidate sets.
If required evidence is unavailable, inspect it or stop with a precise blocker.
Single-seed gains are provisional. A failed process is not a poor model score.
No relevant memory means cold start. Contradictory memory must not be suppressed.

Return schema-valid JSON with a concise decision summary, not a private
chain-of-thought transcript. Runtime owns execution and objective acceptance.
~~~

ActionDecision 必含 state_version、plan_version、action_type、arguments、evidence_refs、memory_refs、expected_result、alternative_action、reason_summary、stop_reason。structured reason 是面向用户的证据摘要，不要求输出完整内部推理。

### 13. 研究效果与 agent 能力分开验收

执行成功、协议有效、验证改善、假设获得支持分别存储。单 seed 改善 provisional；未运行测试不显示 0；只有一个候选不写最优。

必须证明行为随证据变化，而不是只换文字：

- 给出正例映射错误证据时，先检查评价流程，不继续提高 weight_decay。
- 相同问题已有严格兼容的重复无收益经验时，避免相同试验或解释为何需要一次预登记复现。
- 只有预算剩余但没有合法问题时，提前停止。
- 明确新证据和预算充足时，能合法进行第三个及后续 trial。

后续研究评价使用相同动作空间、指标、资源预算，对照 legacy 两试、随机/固定顺序、score-only LLM、structured evidence LLM、加 memory。记录最佳可比 validation、冻结后的独立表现、GPU 时间、API 成本和无效重复次数。agent 真实执行轨迹不等于它优于这些对照。

## P2：面向医生和研究者的工作台

### 14. 信息层级与语言

产品定位“神经信号研究工作台”。医生视角意味着先回答结果、依据、未解决问题和下一步；不伪装为临床诊断产品，不添加虚构患者、疾病概率或脑健康评分。

普通视图分三层：

1. 结果与当前行动：短标题、主结果、关键上下文、一个主要操作。
2. 依据与解释：指标含义、研究问题、结果曲线、方法细节。
3. 技术记录：run ID、内部字段、原始 JSON、日志、目录、API token。

默认隐藏第 3 层，用统一“技术详情”抽屉或页面，而非每行一个“内部 id”。内部字段依然可复制、搜索、导出，不从数据模型删除。

禁止将整页免责声明作为首屏介绍；必要范围用一处固定短句，具体限制放对应指标解释里。LLM 参与的建议可在详情记录来源，不需要每张卡片加 AI 标记。

语言映射集中在 presentation adapter，不散落在模板：

| 当前暴露 | 普通视图 |
| --- | --- |
| numeric_consistency_only | 检查范围：生成信号的数值一致性 |
| 配置内数值检查通过 | 本次配置的数值检查通过 |
| input_contract | 数据完整性 |
| gray_comparability | 灰屏对照 |
| stimulus_temporal_description | 刺激前后变化 |
| coarse_spatial_description | 皮层分布 |
| Agent 轨迹 | 研究过程 / 检查过程（按页面） |
| 产物与记忆 | 文件与历史依据 |
| campaign | 研究任务 |
| profile_weight_decay | 加强正则化；具体数值在方法详情 |
| api_usd: null | 普通首页不展示；成本详情：费用暂不可得 |
| reference_quality unassessed | 参考比较尚未开展（附实际原因） |

“未开展”“不适用”“失败”“无法判断”四种状态不能混用。已描述不是通过，完成问题数不是质量分数。

### 15. 统一壳层与对齐规则

延续简洁浅色科研工具风格，参考 Carbon 的排版和逐层展开，使用当前技术栈实现，不强制安装 Carbon。禁止大面积渐变、玻璃效果、装饰性大图和聊天气泡堆砌。

建议设计 token，可按现有字体微调：

- app header 56–64px；sidebar 240–264px，可折叠。
- 主内容左右内边距 24–32px；统一内容左边线；最大阅读宽度约 1440px。
- 8px 间距体系：8/16/24/32；相邻卡片至少 16px，不相互贴边。
- 正文 14–16px；次要文字 13px；页标题 24–28px；数据数字 24–30px。
- 白色内容、极浅中性背景、细灰边框、单一蓝色操作强调；状态颜色低饱和。
- 按钮同组同高 36–40px；输入框 40px；label 基线一致。
- 数字用 tabular-nums；表格文字左对齐、数值右对齐；徽标 nowrap，不能“已回答”断成两行。
- Grid 子项 min-width:0；文本截断有详情；不得通过全局缩小字体掩盖溢出。

页面滚动与 sidebar 滚动各自清楚，不套多层卡片滚动。窄屏改为单列，不将三列硬挤。sticky 标题不得遮挡脑图和内容锚点。键盘焦点可见，色彩不是唯一状态提示。

### 16. 研究任务：创建页与进行页分开

创建页首先显示四组用户决策：

1. 研究目标：如“比较当前数据上可用的检索训练方案”；可编辑目标，约束从结构化字段确认。
2. 数据与评价：EEG/MEG、数据集、被试训练策略、明确留出对象。
3. 研究方式：单次实验 / 自适应研究；后者写“根据结果调整下一步，在预算内停止”。
4. 资源预算：最长时间、GPU 小时、设备选择；显示能力与可用性。

“发现数据与设备”是上下文准备按钮；“开始研究”是唯一主按钮。预检由主流程自动做并显示结果；dry-run 放次要菜单。不要把检索/预检/试运行/开始训练四个同级按钮交给普通用户理解。

epoch、seed、learning_rate、batch_size、正则化范围、目录覆盖、API 设置放高级设置。高级设置默认折叠但可检索；缺失必填项必须引导到对应字段，不把关键 protocol 隐藏进去。

点击开始前显示紧凑任务摘要：训练来源、验证方式、最终测试保留对象、模型训练策略、研究预算。已有冻结合同足够时直接启动，不每轮弹确认。

启动后进入独立 campaign URL，展示研究进行页，不把长表单放在结果上方。历史列表显示“EEG · 多被试合训”“9 月 23 日 14:30 · 已完成”，内部 ID 放详情。名称来自真实元数据，不根据文件名猜协议。

### 17. 研究进行页的布局

顶栏：任务名 + 当前阶段 + 暂停/继续/停止；每项动作对应真实 controller。

概览区：

- 目标：一句话。
- 当前状态：“分析第 1 次实验的验证结果”。
- 当前最好结果：数值 + 数据集角色 + candidate K + 协议标签；旧批内指标显示“历史批内指标”，不伪装固定 bank。
- 已进行实验数与预算用量，作为资源信息，不作为研究质量。

桌面下方为主栏约 2/3、辅助栏约 1/3：

- 主栏：验证表现曲线、候选实验比较表、训练曲线折叠区域。
- 辅助栏：当前研究问题、下一步及理由、相关历史依据（默认最多 2 条）。

研究过程用紧凑时间线，每条是“动作 / 关键结果 / 下一步原因”，不展示模型自言自语。当前计划显示问题状态和版本修改摘要，不把未来步骤画成必定会执行的固定进度条。

区分三种进度：当前训练 epoch；研究阶段；预算消耗。开放式研究没有准确总步数，不生成虚假百分比。

建议一级 tabs：概览 / 实验比较 / 研究过程 / 数据与方法。日志、完整 memory、raw JSON、token 明细在技术详情，默认无黑色终端。

### 18. 数值检查页的调整

保留侧栏按样本分组，但默认每个样本只露最近检查；历史次数可展开。样本名是显示名，运行时间是副信息，run ID 不当主标题。没有可靠中文标签时保留 antenna_01b，不由 LLM 随意翻译身份。

主卡只显示一次结论、一次范围说明和主要产物；避免侧栏、标题、卡片、段落重复四次“通过”。首屏三项即可：本次检查结论、必需检查完成情况、需关注事项。调用成本与内部 L1 移到检查过程/技术详情。

范围短句：“仅评估生成信号的数值一致性。” 点击查看解释可读“通过不代表真实脑响应已验证”。不把提示到处重复。

“问题覆盖”改为“检查项目”，每行：中文项目名、状态、一个实际证据短句，点开详细指标与来源。原始内部 key 不在行内展开。5/5 标注“必需检查已完成”，不画红绿质量评分条。

脑图区域：

- raw 显示“原始预测”，contrast 显示“相对灰屏差值”。
- 左/右/后视图完整且同一帧；给定展示容器尺寸，object-fit:contain 或等价方案，不裁掉下半脑。
- 三图、色标、帧说明、时间线标记准备好后原子切换；保留旧帧直至新帧完整，不混帧。
- 当前帧按 1-based 显示，内部 0-based 索引移入技术详情；t_stim 仅使用真实元数据。
- 同一信号模式固定跨帧色阶；raw 与 contrast 不默认可按颜色强弱比较。
- 只有明确 image16 metadata 才标“4 秒灰屏—1 秒图像—11 秒灰屏”，旧数据不套用。
- 时序曲线紧接脑图，明确 x/y 单位；逐帧数值表默认折叠，展开只有唯一 T 行。
- 缺 ROI 时简洁显示原因，没必要出现一整个空白大卡。

### 19. 指标解释系统

建立独立 MetricDefinition registry，包含 label、definition、unit、formula_ref、interpretation、limitations、source_field、format、scope 和可选 value-aware template。不要每次刷新发 API 解释指标。

分三层：表格里有短解释；点击侧栏有公式和上下文；技术详情保留 raw key 与原值。不要只放几十个“查看解释”按钮，用户在不点击时也能理解大意。

必须覆盖的例子：

| 指标 | 简短解释 | 不应暗示 |
| --- | --- | --- |
| 有限值比例 | 不含 NaN/Inf 的元素占比 | 不等于生理质量 |
| 灰屏差异 RMS | 预测信号与配对灰屏预测的均方根差异 | 不是 SNR、p 值或显著激活 |
| 峰值时间 | 当前定义下差异最大的采样时刻 | 不自动解释为 HRF 延迟 |
| fixed-bank Top-1 | 固定候选集合中正确图像排第一的 query 比例 | 不跨候选集合比较 |
| fixed-bank Top-5 | 正确图像进入前五的 query 比例，按登记正例规则 | 不是置信度 |
| 训练损失 | 当前训练目标的优化值 | 不同目标或设置未必可比 |
| 验证改善 | 相对可比基线的百分点变化 | 单 seed 不代表稳定收益 |
| GPU 小时 | 本任务分配 GPU 的累计时长 | 不是美元费用或实际利用率 |

百分比数值显示一位小数即可，hover/详情提供精度；RMS 通常保留四位有效数字。差异用 pp，统一 null 展示规则；不能把 false、0、null 混为同一缺失状态。

缺分数时说明具体原因，如“尚未完成固定候选评价”；已有有效分数则展示实际来源，不能继续写“真实训练后才会出现”。状态文案必须由状态机和数据生成。

### 20. UI 的可用性验收

使用项目现有浏览器测试能力；若已有 Playwright，则用它生成同一 fixture 的截图并实际检查，不只声称页面能启动。

- 1440×900、1920×1080 的桌面截图，以及 1280 宽和窄屏溢出检查。
- 数值概览、脑图、创建任务、运行中、规划失败、提前停止、已完成、新任务空状态分别检查。
- 切换 tab、刷新、重连、快速切换任务至少确认无重复 DOM/事件订阅和跨 run 内容串扰。
- 16 帧表只有 16 条唯一行；history 去重；脑图不裁切；长 ID 不撑开布局；状态徽标不换行。
- 新表单不默认加载历史曲线；当前任务以 URL/明确 selection 为主，不能全局最新覆盖。
- 键盘可操作 tab、展开、滑块与暂停；解释不只依赖 hover。
- 附 before/after 截图，并明确截图用的是 fixture 还是实际 campaign。UI fixture 不能写入科学报告或 memory。

## 交付与验收顺序

### M0：可信实验与稳定显示

split role、协议命名、fixed-bank evaluator、预算分离、重复渲染和状态串扰修复。冻结旧记录，不伪造迁移成绩。

### M1：可执行的自适应主路径

UI/CLI/resume 统一 ResearchController；持久化计划、actions、memory retrieval 和真实 DeepSeek 接通。旧固定模式单独标明。

### M2：解释与用户界面

实现 presentation adapter、创建/进行页分离、技术信息收纳、指标解释和浏览器验收。

### 必要自动测试

1. 测试参与优化、非法 split、无被试留出但要求跨被试时阻止启动。
2. 固定 checkpoint 下，评价 chunk/batch/device 划分不同，指标一致；重复正例按 ID 正确处理。
3. 2 GPU 的预算消耗为同时间 1 GPU 的两倍；跨 trial 累计；超时只终止本任务进程。
4. 同一 campaign 的三种观测导致不同的合法 action，分别覆盖诊断、训练/重复和提前停止。
5. 没有硬编码 trial==2 停止；在预算内能超过两次，且可在上限前结束。
6. memory 命中与不兼容排除有据可查，失败和反证不丢失，cached replay 不升级证据。
7. API key 缺失、schema repair 失败、成本耗尽不静默切 fixed policy。
8. state_version 拒绝过期 action；重复点击、刷新和 crash/resume 不重复启动 GPU job。
9. A 的必要回归通过，UI 只改显示不改判决。

### 真实验收

不要用测试规定强制烧满所有 trial。先跑一个修正协议下的真实 baseline 或重新评价严格兼容 checkpoint，再用真实 DeepSeek 根据证据选择诊断/实验/重复/停止。若缺 API、资源或数据，完成其他部分并准确标 blocked。

至少留下真实 initial plan、执行结果、基于新证据的 revise_plan 或有理由维持计划、memory_used、下一 action 和终止原因。真实是否进行了第三次训练如实记录；mock 的三轮测试不能替代 real_multi_trial_passed。

验收矩阵分开：implemented、unit_passed、browser_passed、real_api_passed、real_training_passed、real_replan_passed、real_memory_use_passed、real_multi_trial_passed、research_benefit_assessed。

最后交付：实际修改文件、可运行命令、接口变化、必要测试、API 与训练轨迹、UI 截图、未完成项，以及下一次最小验证实验。不能以多了几个页面或几个 prompt 作为完成自适应研究的证据。
