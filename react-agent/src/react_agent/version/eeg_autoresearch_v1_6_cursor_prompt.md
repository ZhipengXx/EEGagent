# V1.6：有预算约束、证据驱动的 EEG Auto-Research

用途：把本文交给 Cursor，作为 react-agent 仓库的增量开发任务。本文是规划，不是已有功能或验收结果。以用户提供的 V1.5/1.5.1 快照为起点；实施者必须先核对实际代码。

## 设计依据与范围

参考资料：

- NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis：https://arxiv.org/abs/2609.01971 。本次核实到摘要，采用其领域模型接入与任务编排方向，不据此推断具体内部选择算法。
- Reasoning as Gradient: Scaling MLE Agents Beyond Tree Search：https://arxiv.org/html/2603.01692v1 。重点借鉴结构化执行反馈驱动下一次修改、成功经验以及多路线探索的思想。这里的 gradient 是优化类比，不是对代码计算真实梯度。本版本只实现单路线，且改用确定性准入与比较器。
- AgentEvolver: Towards Efficient Self-Evolving Agent System：https://arxiv.org/html/2511.10395v1 。借鉴提出问题、经验检索和事后分析；原工作包含强化学习策略更新。本版本是固定 DeepSeek API 下的经验指导，不声称复现其训练或实现权重自进化。

以下为本项目的独立设计，不是三篇论文的原算法复现。

## 可直接执行的开发任务

你是当前仓库的实现工程师。请完成 V1.6 的模块 B，先检查代码，再实施、验证和留下真实运行记录；不要只生成设计文档。仓库预期根目录：/home/zxuff/data/EEGagent/react-agent。

目标：给定一个已有 EEG 任务、可执行模型目录和明确预算，选择一个合法实验，调用已有训练入口，收集可审计反馈，并产出下一步可检验假设。预算允许时重复此闭环。首个真实验收默认仅执行一次训练。

### 0. 先核对实际仓库

阅读 AGENTS.md 和现有配置、CLI、DeepSeek backend、预算账本、memory、运行事件、报告代码。使用 rg 搜索已有 EEG 训练入口、数据划分、指标输出及本地候选模型。阅读 V1.5/1.5.1 文档，先建立 implementation_inventory.md：

1. 当前已实现并可复用的组件及真实路径。
2. 当前可运行的 EEG 任务和模型，证据是入口、依赖与数据，而不是论文名称。
3. 数据 split 是否已经包含独立 validation；训练脚本是否在每个 epoch 读取 test。
4. 实际训练需要哪个 Python 环境、配置与 GPU。
5. 本轮拟变更文件和兼容性影响。

不要猜不存在的模块或配置参数。外部训练仓库位置不明时先完成协议、适配接口和 mock 测试，将真实 adapter 标为 unavailable 并列明缺项，不能伪造训练成绩。不要为寻找模型全盘扫描无关私有目录。

### 1. 模块边界

模块 A：现有 fMRI 生成、检查、StopGate、memory 和工作台保持语义兼容。

- 不改 A 的图拓扑、阈值、六个问题合同、verdict、screening_decision。
- A 的 claim_scope 仍为 numeric_consistency_only。
- 不把 A 的通过状态转成 EEG 模型推荐、样本训练权重或生物学效度。
- 不覆盖旧 runs 或历史版本文档，不重跑 TRIBE。

模块 B：单独的 EEG 实验任务、状态、预算、memory 命名空间和报告。

- 新增 research_scope，例如 eeg_task_validation；不覆盖 A 的 claim_scope。
- A report 最多作为可选 provenance_ref，不能作为 B 的收益目标。
- V1.6 不自动修改模型结构、不执行 LLM 任意生成代码、不训练 LLM、不加入 fMRI 辅助损失。
- 只选择注册的现有模型或已有训练配置。配置修改只能落在 adapter 显式允许的有限集合内。
- 默认 one_trial：max_trials=1。它验证执行闭环，不证明自动研究优于人工。
- 可实现 bounded_research 模式供下一轮显式启用，默认关闭，max_trials=3（包括建立 baseline 的训练）。本次真实验收不偷偷升级到三次训练。
- 只做单路线。并行研究、多路线共享、自动代码编辑留作后续版本。

### 2. 优先打通一个真实任务

若本地已有 EEG→image retrieval，优先复用它；否则选择现有可执行任务，不为凑模型数量引入新训练框架。

对于检索，冻结 image target encoder、特征缓存、候选图像集合、相似度函数、重复试次聚合、评价单位和划分。首版只能选已有 EEG encoder / 训练 profile。image foundation model、EEG encoder、优化器是不同角色，不混入一个模型名单。

只有一个可用候选也允许执行，但报告 model_selection_assessed=false，不能宣称选出了最优模型。无可比较 incumbent 时，首次有效 trial 只建立 baseline，不记录 improvement。

### 3. 建议文件组织

按实际仓库风格调整名称，职责保持分离，避免把所有代码放进 policy.py：

- src/react_agent/eeg_research/schemas.py：严格数据模型及版本。
- task_contract.py：任务、划分、指标、fingerprint 校验。
- registry.py：可用模型、adapter、已批准 profile。
- adapters/base.py：训练与评估适配协议。
- adapters/<actual_existing_task>.py：唯一首批真实接线。
- executor.py：进程、超时、取消、产物、预算、恢复。
- evidence.py：指标、曲线、错误和 provenance 摘要。
- planner.py 与 prompts/research_planner.txt：形成实验假设。
- validation.py：执行合法性、协议可比性和客观比较。
- memory.py：带条件的成功、失败和不确定经验。
- loop.py：状态机与终止。
- report.py、cli.py：独立报告和命令。
- configs/eeg_research_v1_6.yaml、docs/eeg_autoresearch_v1_6.md。

能够复用的 DeepSeek client、成本统计、事件 writer 用依赖注入复用；不要复制一套漂移的 backend，不要为了复用强行让 B 继承 A 的数值判决。

### 4. 必须有的强类型对象

TaskCard：

- task_id、task_type、research_scope、dataset_id/version。
- dataset_manifest_hash、train_split_hash、validation_split_hash、test_split_hash。
- split_unit、generalization_target、subject_scope、preprocessing_hash。
- primary_metric、direction、secondary_metrics、metric_implementation_hash。
- 对检索额外记录 target_feature_hash、candidate_bank_hash、candidate_count、similarity、trial_aggregation、query_unit、tie_policy。
- permitted_model_ids、permitted_profile_ids、baseline_ref（可空）。
- training_fidelity、seed_schedule、promotion_policy、budget。
- task_card 在第一次实验前冻结，任何修改建立新 campaign。

ModelRecord：

- model_id、adapter_id、supported_tasks、input_contract。
- code_revision/hash、dependency_environment、checkpoint_origin（如适用）。
- approved_profiles、estimated_resources 及估计来源。
- availability=available|unavailable|unverified、reason、last_probe。
- 声称 available 必须已经通过实际接口检查；mock 不算真实可用性验证。

Hypothesis：

- id、question、observations（每条引用 evidence_id）。
- explanation、alternative_explanations、proposed_change。
- expected_observable、disconfirmation_condition、comparison_ref。
- allowed_profile_id、estimated_cost、cost_estimate_source。
- memory_refs、priority_reason、status。

ExperimentSpec：

- id、campaign_id、hypothesis_id、parent_trial_id、model_id、profile_id。
- frozen_task_hash、seed、fidelity、resolved_config_hash。
- factor_changed、output_dir、resource_limits。
- 只能引用注册 adapter 和 profile，不接受任意 shell 字符串或 Python 文件。

EvidenceBundle：

- experiment_id、execution_status、protocol_status。
- 真实 validation 指标及完整 evaluator provenance。
- 可用的训练曲线摘要、训练耗时、峰值显存、异常、checkpoint。
- changed_fields、comparison_compatibility、缺失字段。
- source_artifact_refs；大数组和全量日志不发给 API。

ExperimentDecision：分开记录四个维度，不用一个 success 布尔值代替：

- execution_status：succeeded|failed|cancelled|timed_out。
- comparison_status：baseline_established|comparable|incomparable|invalid。
- selection_status：provisional_incumbent|retained|rejected|not_compared。
- hypothesis_status：supported_provisionally|contradicted|inconclusive|not_tested。
- 候选胜出不自动证明解释正确；配置包含多个差异时不能归因于其中一个差异。

### 5. 数据合同是硬门，不是 prompt 提醒

V1.6 最重要的实现风险是把 test 当 validation：

1. 训练、早停、选模型和 planner 只能使用 train/validation。
2. 若原脚本每 epoch 算 test，adapter 必须切换到训练数据内部构建的 validation，关闭 test 分支。无法做到则拒绝进入 auto-research。
3. EEG 重复试次按任务所需的 image/concept/subject 分组，禁止随机拆 trial 造成目标泄漏。具体分组规则写入 TaskCard 并检测交集。
4. 跨被试任务需要符合该泛化目标的内层 validation；最终 held-out subject 不能参与选型。
5. 标准化及需要拟合的预处理仅 fit train，再应用 validation。
6. 检索候选集合固定；候选数、候选内容、image 特征或试次聚合不同，不能直接比较 Top-1。
7. 不把开发集检索结果标成官方 test 成绩，即使两者候选数恰好相同。
8. 真实 test 评估是 campaign 冻结后的独立阶段；默认本轮不执行。test 路径、标签及结果不进入 planner/evidence/memory 检索。
9. 本地训练进程若仍有全盘读取权限，不能声称实现了 OS 级 test 隔离。报告实际保障级别；可控 adapter 的显式输入隔离与自主代码沙箱是不同保障。
10. 重复使用 validation 也会选择过拟合，因此限制 trial 数并记录所有尝试；最终泛化结论需要独立评估，不能因本地 validation 改善就宣称科学突破。

### 6. 闭环：先观察，再提出下一次实验

执行顺序：

1. 确定性 preflight：任务合同、数据、可用候选、预算。
2. 查找严格兼容的 baseline 和 memory；没有则 planner 选择初始 profile 并说明依据为 cold-start prior。
3. Planner 提出最多 3 个候选假设，选择 1 个；每个候选都有可执行 intervention 和可反驳预期。
4. Runtime 校验选择是否合法、是否重复、预算是否足够、task hash 是否一致。
5. 执行一个 trial，收集 EvidenceBundle。
6. 确定性 validator 判定能否比较、是否保留为临时 incumbent。
7. 一次 reviewer API 调用解释反馈并给出下一步假设建议。它不能修改 evaluator 指标或推翻合同。
8. 写入 memory；若预算已耗尽，建议只保存，不执行。
9. 预算允许才进入下一轮。没有新合法问题时允许主动停止，不为凑次数重复训练。

默认一次真实训练仍要留下结果后的 hypothesis update，所以至少能看到“执行后学到了什么”，而非仅有运行前 plan。下一次 campaign 可以显式恢复这条研究路线。

多轮模式可让 reviewer 同时生成下一步提议，减少无意义重复 API。创建后的每个 trial 使用完整持久化 ExperimentSpec；重试和恢复不能产生隐式新实验。

### 7. 启发式决策

不实现伪精确的 expected_accuracy=0.93 或任意加权 quality score。先硬过滤，再可解释排序：

硬过滤：任务不兼容、资源缺失、禁止改动、预算不足、相同 experiment fingerprint 已执行且非预注册重复验证 → 不可选。

候选优先级：

1. 会使全部成绩无效的协议或实现问题，优先停止并处理；不是调参机会。
2. 有当前实验观测支持、能用一次小实验区分解释的问题。
3. 严格兼容历史条件下得到支持的改进方向。
4. 冷启动或探索性方向，必须标 prior/unverified。

同层优先考虑可解释、低成本、较少混杂因素的实验。理由记录“有哪些证据”和“为何另两个暂缓”。这套排序是本项目启发式，不是 GOME 原公式。

示例（仅说明格式，不是实际观测）：

- 观察：某次训练损失下降，固定 validation 检索指标在后半程停滞。
- 解释 A：过拟合；解释 B：优化目标与检索指标不一致。不能直接宣布 A 成立。
- 若 registry 有只改变 weight_decay 的两个批准 profile，可选一个匹配预算和 seed 的比较；预期是 validation 改善而非只看 train loss。
- 若缺少这种合法 profile，记录待研究问题，不自动编辑训练代码来满足假设。
- OOM 记录资源不满足，不等于模型任务效果差。处理 OOM 若改变 effective batch size，必须作为新配置，而非无痕 retry。

首版使用固定 fidelity。少 epoch 的速度 probe 只能验证可运行性或估计成本，不将其分数与完整训练分数竞争。未来加入粗到细训练时再引入 promotion 阶段和相同 fidelity 的比较规则。

### 8. 三层验证与保留规则

第一层 execution：退出码、产物 schema、有限指标、checkpoint、真实 evaluator 运行。

第二层 comparability：相同任务与评价 fingerprint、相同 fidelity、匹配 seed、合法变量变化、无已知 split 泄漏。test 参与优化直接 invalid。

第三层 improvement：由确定性 comparator 计算 delta，方向来自冻结 TaskCard。默认严格数值改善仅可更新 provisional incumbent；不是统计显著。若预注册 minimum_delta，则使用它，LLM 不得改值。tie 保留既定 incumbent；可选成本 tie-break 必须事先登记。

后续多 seed 确认使用预先规定的全部 seed 聚合，不挑最佳 seed。未做重复验证只能 supported_provisionally，不能升格为可迁移规律。不同模型 profile 的 bundle 比较只能支持整体方案选择，不能对单个结构作因果归因。

GOME 的原设计允许 LLM 参与最终接受；本系统有意采用客观比较器，LLM 负责解释与提议。代码更整洁或 API 更少可单独记录为工程改进，不能覆盖 EEG 主指标下降。

### 9. Memory 存实验知识，不只存运行摘要

复用 SQLite 技术，但 B 与 A 的表/namespace 分开。保存所有 episodes：有效改善、未改善、无效协议、运行失败、尚未测试。

每条经验包含：条件指纹、问题、观测、干预、比较对象、指标 delta、fidelity、seed、证据路径、适用条件、反证、证据级别与失效原因。

建议证据级别：

- proposed：尚未执行。
- observed：一次有效实验。
- replicated：在预注册重复条件下复现，仍只对这些条件有效。
- external_prior：来自资料，不是本项目实证。

检索先按 task/split/generalization/target features/protocol 硬约束筛选。跨 subject 或跨数据集经验仅作为 external-like prior，不得读取其成绩作为当前可比 baseline。test evaluation memory 不提供给正在优化的 campaign。

必须同时检索支持和相反经验；不允许只记成功形成确认偏差。同一次 run 的重播不是独立证据。缓存命中不是再次复现。失败记录需区分 environment failure 与 scientific non-improvement。

memory curator 可生成摘要，但不能改 evidence level、原始数值、兼容性、重复次数和 hypothesis_status。首版默认不额外调用 curator，用确定性 episode 写入控制成本。

### 10. DeepSeek 与预算

沿用已有 DeepSeek API client 和可配置 model，不默认切换未知模型名。记录 requested_model、response_model、call_role、latency、usage、retries、repair、fallback。

- planner/reviewer 使用同一预算账本。
- 无 API key 或服务失败要明确标记；不得悄悄切成 rule 并宣称 LLM 规划验证通过。
- schema repair 最多 1 次，计入总 API 限额。
- 不要求模型输出长篇内部思考；要求简短证据引用、备选解释、实验选择理由和可检验预测。
- 默认 max_trials=1、max_concurrent_trials=1、max_lm_calls=4（包含 repair/reviewer）。
- gpu_seconds、wall_seconds、per_trial_timeout 必须在实际配置中有明确上限；按本地训练入口确定，缺失时只允许 dry-run。
- estimated_cost 不知道则 null，并写来源 unknown，不编造预计收益或耗时。
- api_usd 未取得可靠 pricing/usage 时仍 null；不能将 unknown 当 0。若没有可靠美元估算，不可声称美元硬上限得到保证，应使用调用数/token/时间硬限额。
- 训练失败、probe、retry、确认 seed 均计资源成本。max_trials 约束科研候选尝试，另设 max_execution_attempts 防止无限失败重试；模型解释不能扩预算。
- GPU 设备、可用性由配置和现场检查决定，不硬编码占用全部 GPU，不终止已有用户进程。

executor 用参数数组启动已登记入口，shell=False；记录 resolved args（脱敏）、环境版本、开始结束、退出码。超时和取消只清理本次创建的子进程。新输出目录、文件锁、写入原子性与 resume 防重执行是必要能力。

### 11. Runtime planner prompt（英文，单独文件）

~~~text
You plan bounded EEG experiments over an approved registry.
Your objective is to improve the frozen task's validation objective within the
remaining budget, while producing testable, evidence-linked hypotheses.

Inputs: frozen TaskCard, available ModelRecords and approved profiles,
incumbent metadata, structured EvidenceBundles, compatible memory, and budget.

Rules:
1. Use only supplied evidence. Cite evidence IDs for observations.
2. Distinguish observations, explanations, and untested priors.
3. Propose at most three hypotheses and select at most one executable experiment.
4. Each hypothesis must specify a permitted intervention, an expected observable,
   a disconfirmation condition, and at least one alternative explanation when known.
5. Do not change splits, metrics, evaluation candidates, budgets, or test access.
6. Do not invent models, metrics, files, cost estimates, or validation results.
7. Do not treat fMRI numeric screening as EEG performance evidence.
8. Prefer comparable experiments with fewer changed factors. A bundled model
   comparison does not identify the causal effect of an individual component.
9. Failed execution is not negative task-performance evidence.
10. If no legal action exists, return stop with a precise reason.
11. Keep reasons concise. Return schema-valid JSON only.

Output fields: observations, hypotheses, selected_hypothesis_id,
experiment_proposal, deferred_reasons, stop_reason.
Runtime owns final validation, execution, comparison, and budget enforcement.
~~~

### 12. Runtime reviewer prompt（英文，单独文件）

~~~text
You review one completed EEG experiment. The runtime decision and evaluator
metrics are authoritative inputs; you must not rewrite them.

Explain what was observed, which hypotheses remain plausible, and which next
approved experiment could distinguish them. Cite evidence and memory IDs.
Keep execution validity, performance change, and explanatory support separate.
A single-seed improvement is provisional. No improvement may be inconclusive.
Do not claim causality, biological validity, or generalization from a local score.
Never use test results to propose the next experiment.
Preserve negative results and applicability conditions in the memory proposal.
If the budget is exhausted, provide a pending hypothesis without requesting execution.
Do not upgrade memory evidence levels or fabricate a successful experiment.
Return schema-valid JSON: observations, hypothesis_assessment,
alternative_explanations, next_hypothesis_proposal, memory_summary, limitations.
~~~

### 13. 命令与产物

根据仓库实际 argparse/Typer 风格实现以下职责；不要只在文档里写命令：

- research probe：检查本地候选、task contract、数据/环境，默认不训练。
- research plan --dry-run：校验可执行方案；API 调用与离线静态 preflight 分开注明。
- research run：遵守 mode 和 max_trials 执行。
- research inspect：展示 trial、hypothesis、evidence、cost、stop reason。
- research resume：恢复现有 campaign，不重复已完成 trial。

命令使用 uv run；外部训练环境可以独立，但必须由 adapter 明确绑定，不要求重装到 react-agent 环境。

每个 campaign 至少输出 task_card.json、registry_snapshot.json、hypotheses.jsonl、experiments.jsonl、events.jsonl、comparison.json、cost.json、report.md，以及 trials/<id>/ 下的真实配置、日志、指标、checkpoint 引用。

报告必须回答：任务是什么；比较协议是什么；选了什么及依据；实际做了什么；结果是否可比；是否只是 baseline；哪个假设得到何种程度支持；下一步是什么；为什么停止；花费哪些资源。

单次训练报告必须显式写：模型最优性未评估；单 seed；自动研究效率未评估。没有 baseline 则 delta=null。没有 test 则 test_result=null。一次训练后成功生成下一步假设，也不等于下一步假设已经被验证。

工作台本轮只需保留未来独立 Research tab 的数据接口；不要顺带大改 V1.5 UI 或把 B 成绩塞进 A 的数值通过卡片。

### 14. 验证与验收

为科研正确性和进程控制编写必要测试，不写大量镜像实现的测试。

必须覆盖：

1. 无候选、缺数据、协议不匹配时不启动训练。
2. train/validation 分组交集或 test 被用来调参时拒绝实验。
3. 不同 candidate bank、target features 或 fidelity 的成绩不可比。
4. 第一次有效 trial 建 baseline，不能算 improve。
5. mock 第二次成绩下降时不覆盖 incumbent，保存失败经验。
6. 改善但只有一个 seed 时保持 provisional；cached replay 不提升证据级别。
7. 不存在的 model/profile、越界修改、重复实验被 runtime 拒绝。
8. API repair 和进程 retry 计预算；预算耗尽不执行 pending hypothesis。
9. 同一输入替换为不同有证据的结果后，planner/reviewer 的结构化观测及假设能相应变化；不能只变措辞。mock 用于契约测试，真实 API 轨迹另行标记。
10. resume 不重复训练；超时只结束本次进程；未知 cost 保留 null。
11. module A 的已有必要回归检查通过，报告语义和旧目录不变。

分开标记 implemented、mock_passed、real_api_passed、real_training_passed、multi_trial_real_passed。禁止混为一个 passed。

本次真实验收：一个已有任务、一个已接线模型、一个预先规定的训练预算；真实 DeepSeek 初始选择、真实训练、真实结果后假设更新。默认 max_trials=1。若缺路径、GPU 或 API，完整实现可验证部分并报告具体阻塞，不能捏造日志。

多轮真实验收留到显式启用 bounded_research：baseline + 最多两次候选；所有实验相同比较协议，预算总额固定。它证明能持续研究，不足以证明相对其他策略更优。

### 15. Auto-research 的后续评价计划

本轮只准备评价接口和方案，不自动跑大量 benchmark。

对比至少三种等总预算策略：固定候选顺序/简单随机选择、只看末尾分数的 LLM、结构化反馈驱动的 LLM。再比较后者加/不加兼容 memory。候选空间、数据、seed、硬件、时间和 API 成本统计一致。

主要看：预算内最佳可比 validation 指标、独立最终评估、达到预定性能所需 GPU 时间、有效实验比例、重复无效尝试比例。LLM 调用数和完成次数属于成本/可靠性指标，不是主要科研收益。

完整方法的附加 reasoning token 成本必须计入。跨任务 memory 的评估应区分见过的任务与独立新任务，不能把同任务重复运行当迁移能力。

若最终要证明 A 的筛查对 EEG 有用，应另立研究任务：无伪标签、使用全部伪标签、按预注册筛查策略选择伪标签，并加入等样本量随机选择对照；保持训练预算与其他条件一致。该实验不在 V1.6 自动执行，也不预设筛查有益。

### 16. 交付要求

提交实际变更文件清单、可运行命令、必要测试摘要、真实执行与 API 证据路径、未完成项及下一轮最小实验建议。

V1.6 的完成标准是：已建立可信任务协议，已有模型能被受限调度，结果能改变后续假设，并能持久化恢复。不要用 agent 数量、prompt 长度或产生多少篇总结充当研究能力提升。
