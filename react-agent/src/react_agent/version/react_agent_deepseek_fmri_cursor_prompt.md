# Cursor 实施 Prompt：基于 react-agent + DeepSeek 的伪 fMRI 分级检查闭环

使用方式：将本文件放入本地 react-agent 仓库，在 Cursor Agent 中要求“读取本文件，并按阶段直接实施”。本文件是完整任务说明。文中的新增文件名、CLI 名称和数据结构是本项目的设计要求，不代表上游仓库已经实现它们。

---

## 0. 你的角色、目标与执行方式

你是负责落地该项目的 Python / LangGraph 工程师。请在当前 langchain-ai/react-agent 仓库基础上，实现一个可运行、可测试、便于扩展的伪 fMRI 检查 Agent。

请直接检查代码、修改项目、运行验证，完成最初版本，不要只返回架构建议、伪代码或待办清单。先给简短实施计划，然后连续完成下述阶段。对可自行判断的工程细节采用合理默认并记录。没有真实数据或 API key 时，完成离线版本及模拟闭环，准确说明尚未验证的部分，不要伪造真实 API 测试结果。

背景：
- 我使用 TRIBE v2 将图像刺激转换为预测 fMRI，想检查这些伪标签是否存在数值、数据契约、时间或空间方面的问题。
- 后续可能把筛选后的伪标签用于 EEG 相关模型训练，但本轮不实现 EEG 训练或 TRIBE 重新生成。
- LM API 使用我已有的 DeepSeek 配置。
- 第一版先完成“检查 → 更新证据 → 选择下一工具 → 执行 → 再判断 → 停止并报告”。
- 未来会加入 CortexMAE、独立 image–fMRI 对应模型、其他 API，以及学习得到的 routing policy。

本轮优先级：
1. 实际可运行的最小闭环。
2. 数值结果与状态含义正确。
3. API、工具、决策策略之间解耦。
4. 成本和执行轨迹可追溯。
5. 扩展能力通过清楚的接口实现，避免提前建设复杂平台。

不要自动安装大模型权重、下载大规模数据集、运行 TRIBE、训练模型、发布服务或修改无关仓库。不要在报告里暴露 API key。保留当前用户已有的代码改动。

## 1. 先审查当前仓库，再决定最小改动

先读取当前目录适用的 AGENTS.md、README、pyproject.toml、锁文件、langgraph.json、tests，以及 src/react_agent 下的实际文件。

重点查明：
- 实际使用的 LangGraph / LangChain / Python 版本。
- graph.py 的 call_model、tools、条件边与导出 graph。
- state.py、context.py 或 configuration.py 的真实接口。
- tools.py、prompts.py、utils.py 中模型初始化和工具绑定方式。
- 当前 DeepSeek 配置来自哪里；只检查变量名和配置结构，不打印密钥值。
- 默认网页搜索工具是否引入 Tavily 等不必要的依赖或 key 要求。

输出简短审查结果，然后实施：
- 优先复用当前 StateGraph 和项目启动方式。
- 不照抄旧版本教程，不为了本任务无条件升级整个依赖树。
- 如果仓库版本与下方建议路径不同，保持职责划分并适配真实代码。
- 新的 fMRI 流程不能要求 Anthropic、Tavily 或 LangSmith key 才能导入和运行。
- 保留原通用 ReAct 入口，另导出 fmri_check 图并注册到 langgraph.json；若项目结构使复用原入口更合理，可以调整，但必须记录兼容性变化。
- fmri_check 必须被真正接到 LangGraph 入口和 CLI，不能只创建未使用的模块。

## 2. 明确 V1 的边界

V1 必须实现：
- .npy / .npz 二维时空数组 + JSON 元数据的输入。
- 单样本及 JSONL manifest 批处理。
- 真实可执行的基础检查与可选诊断工具。
- rule 与 hybrid 两种 routing 模式。
- DeepSeek fast / reasoning 两个配置档位，以及摘要调用。
- 没有 key 也能完整运行的规则模式、MockLM 与合成 demo。
- 有界循环、工具前置条件验证、错误处理、成本日志。
- JSON 主报告、Markdown 摘要、逐步事件日志及 batch 汇总。
- 足以验证关键行为的自动测试。
- README 中的启动步骤与新增工具说明。

V1 不实现：
- 重新训练或微调任何 LM / fMRI foundation model。
- 在线修改阈值以增加通过率。
- 自动“修复”原始 fMRI、反复重采样直到通过。
- 多 Agent 协作、RL policy 训练、数据库、Web 前端、微服务、Kubernetes。
- 将完整 fMRI 数组或大规模 embedding 填进 LM prompt。
- 用 LM 的自信程度代替神经科学证据。

CortexMAE 与 image–fMRI 一致性检查只预留扩展位置，并在工具列表显示 disabled / unavailable 原因。没有真实实现、权重或参考数据时，不允许返回伪造结果，也不能将 skipped 记为 passed。

## 3. 输入数据契约与本地数据访问

### 3.1 样本定义

用 Pydantic 或项目已采用的等效方式定义 SampleSpec。字段至少包括：
- sample_id
- fmri_path
- array_key：仅 .npz 需要，必须显式给出
- time_axis：0 或 1，必须显式给出
- sampling_interval_s：可缺失，若提供必须大于 0
- spatial_representation：surface / parcel / volume_flattened / unknown
- space_name：例如明确的 surface space 或 atlas；可缺失
- normalization：明确描述已做的标准化；可缺失
- generation_profile_id：生成协议的稳定标识；可缺失
- metadata_path、image_path、roi_map_path：可选
- stimulus_events：可选
- output_time_origin、alignment_description：可选
- provenance：generator、版本、配置摘要等

示例：

```json
{
  "sample_id": "example_0001",
  "fmri_path": "arrays/example_0001.npy",
  "time_axis": 0,
  "sampling_interval_s": 1.0,
  "spatial_representation": "surface",
  "space_name": null,
  "normalization": null,
  "generation_profile_id": "static_image_profile_v1",
  "metadata_path": "metadata/example_0001.json"
}
```

样本 schema 的版本必须写入输出。路径相对于 manifest 所在目录解析，不依赖运行时 cwd。没有 manifest 的单样本 CLI 路径按明确文档约定解析。

输入要求：
- 默认内部数组约定为 [T, V]，仅根据显式 time_axis 转换，不猜测轴、不静默 reshape。
- .npz 含多个数组而未指定 array_key 时明确报错。
- .npy 使用 allow_pickle=False，可用 mmap；避免反复读取整份数据。
- 数值 dtype、维度、空轴、NaN/Inf、全零、全常数等分别记录。
- 缺少采样间隔或空间元数据，不等同于信号错误；只限制相关工具的可用性和结论范围。
- 不因为不知道某一固定顶点数就直接拒绝输入；只有用户显式声明的契约不匹配时才判定对应错误。
- 输入文件只读。报告 NaN/Inf，不通过 nan_to_num 静默掩盖。
- 大数组放在本地 ArtifactStore / DataRepository，LangGraph state 只存引用和小型 JSON 结果。
- sample_id 不直接拼成任意输出路径；使用稳定、安全的文件标识，处理重复 ID。

### 3.2 TRIBE 相关语义约束

不要凭通用 fMRI 常识硬编码：
- “刺激后必须恰好 5 秒达到峰值”。
- “某一幅图像必须激活某个 ROI”。
- “12 秒信号必须满足静息态功能连接分布”。
- “预测 fMRI 越接近真实 fMRI 的方差越好”。

先核对输入元数据及实际生成代码中的时间轴含义。预测输出可能已有时间对齐处理；没有依据时，不要额外平移或重新定义刺激响应窗口。短序列上的频谱、相关性或峰值只能描述，不自动解释为生理有效性。

## 4. 模块职责与建议组织

采用清晰的模块边界，避免把所有逻辑塞进 graph.py。可按当前项目合并小模块，但不得混合关键职责。

| 建议位置 | 职责 |
| --- | --- |
| src/react_agent/fmri/schemas.py | SampleSpec、ToolResult、Decision、FinalReport 等契约 |
| src/react_agent/fmri/config.py | 类型化配置、环境变量、校验与默认值 |
| src/react_agent/fmri/state.py | LangGraph 可序列化状态 |
| src/react_agent/fmri/data.py | 输入适配器、元数据、数组引用与读取 |
| src/react_agent/fmri/tools/base.py | 工具协议、ToolSpec、可用性与成本接口 |
| src/react_agent/fmri/tools/registry.py | 注册、发现、候选工具筛选 |
| src/react_agent/fmri/tools/numeric.py | 输入检查、基础统计、时间诊断 |
| src/react_agent/fmri/tools/spatial.py | ROI 检查 |
| src/react_agent/fmri/tools/reference.py | 参考统计构建及对照工具 |
| src/react_agent/fmri/policy.py | 分数、候选排序、规则与 hybrid policy |
| src/react_agent/fmri/llm/base.py | LMBackend、响应及 usage 契约 |
| src/react_agent/fmri/llm/deepseek.py | DeepSeek 配置与协议适配 |
| src/react_agent/fmri/llm/mock.py | 可注入、确定性的离线 LM |
| src/react_agent/fmri/prompts.py | 决策与摘要 prompt |
| src/react_agent/fmri/budget.py | 预算预检、调用计数、费用记录 |
| src/react_agent/fmri/graph.py | 节点与边的组合 |
| src/react_agent/fmri/reporting.py | JSON / Markdown / JSONL 输出 |
| src/react_agent/fmri/cli.py | check、batch、make-demo、fit-reference |
| configs/fmri_check.yaml | 示例配置 |
| tests/ | 关键行为测试 |
| docs/fmri_check.md | 使用说明、设计边界、扩展指南 |

这是职责清单，不要求为了文件数量拆分一行类或创建泛化插件平台。简单 Python registry 足够。

依赖方向：
- 数值工具不导入具体 LM，也不直接操作图的调度。
- LM 适配器不加载 fMRI 文件。
- policy 根据 state 和 ToolSpec 决策，不直接执行工具。
- executor 负责输入验证、成本预检、执行与记录。
- graph 只编排。
- reporter 从已存在证据生成结论，不能增加未执行检查。

## 5. 统一的工具与状态契约

### 5.1 工具接口

实现等效于以下职责的 Protocol / 抽象接口：

```python
class CheckTool(Protocol):
    spec: ToolSpec

    def availability(self, sample, context) -> Availability: ...
    def estimate_cost(self, sample, context) -> CostEstimate: ...
    async def run(self, sample, args, context) -> ToolResult: ...
```

ToolSpec 至少含：
- name、version、description
- level：coarse / medium / fine
- input_schema
- issue_types：能够检查哪些问题
- preconditions、dependencies
- estimated_cost / cost_class
- repeatable：默认 false

ToolResult 至少含：
- tool_name、tool_version、result_id
- execution_status：success / skipped / error
- findings：每项含 code、severity、message、evidence_refs
- metrics：可序列化数值及单位
- coverage：实际检查到什么
- limitations：什么不能据此判断
- artifacts：本地结果引用
- elapsed_seconds、gpu_seconds（不可测则 null）
- cost_estimate、error 信息、cache_hit

重要：
- 工具成功执行与样本通过检查是不同概念。
- skipped 必须给出原因。
- NaN / Inf 不能作为 JSON 数值写出；转换为 null 并记录解释。
- evidence_refs 必须能指向真实 ToolResult 及 metric / finding 字段。
- 同一工具、样本、参数、输入 hash、工具版本与相关配置相同，默认不重复执行。
- 允许轻量缓存；缓存命中不得冒充本轮执行，也不能再次计费。
- 数组 hash / 文件 fingerprint 每样本只计算一次并缓存，避免每次工具调用全量重复扫描。
- 归一化方式、参考集版本改变，相关缓存必须失效。

### 5.2 状态

CheckState 保存：
- run_id、sample spec / sample reference
- 当前 evidence 和 tool_results
- completed / failed / unavailable 工具
- candidate_tools
- issue_types 与 unresolved_questions
- need_score 及分项、缺失掩码
- tool_call_count、lm_call_count、round_count
- budget ledger
- decision_history、stop_reason
- 最终 report

LangGraph 的 list reducer 必须与节点更新方式一致，不得在使用 append reducer 时反复返回整个历史造成重复追加。工具和 LM 的调用次数要与事件日志一致。

## 6. 第一版必须真实实现的检查工具

### 6.1 validate_input：强制，便宜，无 LM

检查文件读取、二维数值数组、轴、空数组、有限值、显式 shape 契约。

输出区分：
- malformed：不可解码、错误维数、非数值、NaN/Inf 等明确输入问题。
- incomplete_metadata：缺少 TR、空间说明、生成协议等。
- valid_for_numeric_checks：可继续做数值描述。

全零 / 全常数可作为 degenerate_signal flag；不要混同于文件损坏或疾病判断。

### 6.2 basic_statistics：强制，便宜，无 LM

计算适合该数据规模的汇总：
- T、V、dtype、finite_ratio
- mean、std、min、max、若干 quantile
- 零值比例、近常数空间序列比例
- 跨时间和跨空间方差的汇总
- 一个便宜的时间变化预筛指标，例如空间汇总后的帧间差分比例；明确归一化定义，并记录空间平均可能掩盖局部异常的限制

近常数判断的绝对 / 相对容差放在配置，记录实际使用值。已标准化数据 mean 接近 0 不构成错误。

可用于硬 flag 的默认项限制在明确退化数据；振幅异常等依赖归一化和参考分布的判断，不设置为普适生理阈值。

为 demo 提供单独的、明确标为 heuristic 的时间变化触发阈值，用于启动 temporal_diagnostics。生产配置只在用户提供或接受对应阈值时启用；没有阈值时可仅报告该指标，或按显式 required_checks 执行时间检查。预筛只给出汇总，时间诊断负责定位、补充局部统计与解释限制。

### 6.3 temporal_diagnostics：可选，中等，真实实现

输出 lag-1 相关性汇总、相邻时间点差分强度、时间方差与峰值位置的描述。

- 先定义最短长度；不足时 skipped / insufficient_length。
- 常数序列导致相关性无定义时明确计数，不把 NaN 当 0。
- 无采样间隔时只报告 frame index，不能换算成秒。
- 短序列默认不做静息态频带、功能连接和显著性检验。
- 时间突变只能称为描述性 flag，阈值来源标为 heuristic 或 reference_calibrated。
- 不自动检验固定 HRF 延迟；没有时间对齐信息时记录证据缺口。

### 6.4 roi_summary：可选，中等，真实实现

接收显式 ROI mapping（例如长度 V 的整数标签数组 + 名称表）：
- 验证索引顺序、长度与空间描述一致。
- 计算每个 ROI 的均值、方差、时间曲线摘要。
- 保留 ROI 内抵消等解释限制。
- 没有 mapping 或无法确认空间对应时 unavailable / skipped。
- 不按图像类别臆造“正常激活 ROI”，也不把电极位置直接当成 fMRI 顶点对应。

### 6.5 reference_distribution：可选，中等，真实实现

通过独立的 fit-reference 命令，从显式指定的参考 manifest 提取与 basic_statistics 相同定义的样本特征，保存版本化参考统计。

V1 可以使用少量特征的 median / MAD、经验分位数：
- 每个特征给出偏离程度，而不是未经验证的“错误概率”。
- MAD 接近 0 的特征单独处理，不能通过除以极小数制造巨大分数。
- 参考样本数不足、协议 / 标准化 / 空间 / 时间窗不兼容时，标记不可比较。
- 所需兼容字段与最小参考样本数在配置中定义并写入报告。
- 参考集类型标明 real_measured / generated / demo_synthetic。
- generated 参考集只能说明同生成流程中的相对偏离，不能证明真实脑响应正确。
- 参考集及阈值必须预先固定，不在处理测试样本时自动更新。
- 不让被测样本进入自己的参考集；通过 sample_id / fingerprint 尽量识别重叠。
- 用户明确提供的参考数据才用于校准，不能把第一批待筛样本自动当正常样本。

### 6.6 未来工具接口

CortexMAE、独立 semantic_consistency 检查仅登记能力说明 / 扩展文档：
- 需要哪些输入转换、权重、参考数据及评估。
- CortexMAE 的 embedding 不等同于合理性概率。
- 参考分布正常也不代表与当前图像匹配。
- 不借用 TRIBE 自己的预测作为独立验证证据。
- 工具未实现就不进入 executable candidates。

## 7. need_score、候选工具与 LM 档位

将“检查需要程度”与实际执行成本分开。

### 7.1 评分

保留三项：
- A：已有证据中的异常程度。
- M：当前配置要求的证据缺口比例。
- D：对同一明确命题存在冲突的证据程度。

need_score = w_A * A + w_M * M + w_D * D。

要求：
- 该分数标记为 heuristic_priority，不称为风险概率、置信度或医学判断。
- 没计算的 A / D 是 null，不是 0；保存 observed_mask。
- V1 没有可靠的冲突定义时 D 不启用，默认 w_D=0。
- 不将“总体数值正常”和“局部 ROI 有异常”自动视为矛盾。
- M 只基于当前配置声明的必要证据，不包括全部未来工具。
- 定义缺失项处理：对已启用且可观测分项重新归一化权重，同时将缺失列表单独提供给 policy；关键证据缺失仍会阻止通过。
- 分值用于排序 / 升级，不覆盖 validate_input 的硬错误。
- 默认参数注明工程启发式，不能声称已在真实脑数据上校准。

不要仅按 need_score 选择工具。先根据 issue_types、证据缺口、前置条件、已执行情况筛出候选，再考虑费用。

### 7.2 rule policy

必须可以完全无 LM 运行：
- 强制基础检查先完成。
- 明确坏输入直接结束。
- 时间问题优先 temporal_diagnostics。
- 空间疑点且 ROI mapping 可用时选 roi_summary。
- 分布疑点且参考集兼容时选 reference_distribution。
- 没有前置条件的工具不能被选中。
- 对无可用证据支持的“细查”明确 abstain / inconclusive。
- 不为了凑步数运行工具。

### 7.3 hybrid policy

规则负责硬约束、必需检查和候选筛选；DeepSeek 从候选工具中选择一个下一步动作或提出停止建议。

- 低复杂度、无明确冲突时使用 fast profile。
- 多个合理候选、边界情况或有明确定义的冲突时使用 reasoning profile。
- 哪个 profile 生效由代码和预算决定，不由 LM 任意修改。
- 同一样本的 profile 升级至多一次或按清楚规则限制；避免来回切换浪费预算。
- 模型建议仍需 validator 检查；不能跳过必需检查、引用不存在的 evidence、调用未注册工具或突破预算。
- API 失败时允许降级 rule policy，输出必须标记 fallback_used。
- 样本之间独立初始化 LM 状态；共享的只是只读参考数据、配置、缓存和 batch budget。

V1 不训练 router，也不把某样本的 LM 判断反馈为下一样本的新阈值。

## 8. DeepSeek 适配器与结构化决策

### 8.1 配置与后端

定义 LMBackend：
- decide(observation, candidate_tools, profile) -> Decision + LMUsage
- summarize(final_evidence, profile) -> Summary + LMUsage

默认 DeepSeekBackend；测试使用 MockBackend。LM 实例、HTTP client 可复用，schema 和业务状态不依赖某一厂商。

优先保留已有 DeepSeek 配置：
- DEEPSEEK_API_KEY
- DEEPSEEK_BASE_URL
- DEEPSEEK_FAST_MODEL
- DEEPSEEK_REASONING_MODEL
- fast / reasoning 的 thinking、reasoning_effort、timeout、max_tokens

两个 profile 可以使用同一模型，只切换经过验证的 thinking 配置。不要强制购买或接入第二家 API。

本任务撰写时，官方文档使用 https://api.deepseek.com，示例模型包括 deepseek-flash、deepseek-v4-pro。它们只作为 .env.example 的可修改示例；不要覆盖用户已验证可用的模型名。模型、能力、价格全部配置化。

V1 建议：
- 保留 LangGraph 进行循环编排。
- 在 DeepSeekBackend 内使用与当前 API 兼容的 SDK；使用 OpenAI 兼容客户端不代表调用 OpenAI 服务。
- 暂时不强制引入 LiteLLM；未来可新增 LiteLLMBackend，业务模块无需改变。
- 默认 stream=false，先做好稳定调用和使用量记录。
- 没有 key 时 mock / rule 模式正常工作；显式选择 deepseek 但没有 key 时给出配置错误，不能静默冒充在线调用。
- 模型不接受的参数通过能力配置排除，不能假定所有模型都支持 temperature、thinking 或相同的 structured output。

### 8.2 V1 决策协议

V1 采用 JSON Action 协议：模型返回结构化下一步动作，程序验证后调用本地工具。它仍然是“观察—决策—动作—再观察”的 ReAct 循环。

这样可先把领域状态和执行逻辑与厂商原生 tool-call 消息协议解耦。不要同时实现两套完整协议。未来原生 function calling 可放在适配器内部扩展。

模型请求使用当前 API 支持的 JSON output 方式，prompt 明确包含 JSON 要求和 schema；收到结果后仍做 Pydantic 校验。JSON 可解析不等于符合业务要求。

示例：

```json
{
  "action": "run_tool",
  "tool_name": "temporal_diagnostics",
  "tool_args": {},
  "reason": "当前尚缺时间结构检查，已有长度统计表明可以执行时间诊断。",
  "evidence_refs": ["basic_statistics:result_001:metrics.T"],
  "question_to_resolve": "时间变化是否集中在少数相邻帧？",
  "expected_observation": "相邻帧差分摘要及其发生位置"
}
```

注意：该例中的 evidence_refs 仅展示格式。运行时引用必须从真实输出生成，且 T 满足工具长度要求时才允许这个动作；不能预置不存在的 finding。

Decision：
- action 为 run_tool / stop。
- run_tool 必须提供候选列表中存在的 tool_name。
- stop 提供 stop_reason，但最终 verdict 由确定性报告策略生成。
- reason 是短理由，建议至多 2–3 句，不要求模型输出长篇思维过程。
- tool_args 不包含任意 Python、shell、代码、网络地址或自由文件路径；资源通过已登记 artifact / sample 引用。
- extra fields 默认禁止。
- JSON 解析失败、schema 不符合、引用无效、重复调用：最多一次有界修正请求，仍失败则 rule fallback；修正也计入预算。
- 不通过字符串截断、eval 或随意“修复 JSON”执行不明确的动作。

若未来启用原生 thinking + tool calls，必须按当时 DeepSeek 官方协议保留所需 reasoning_content 和 tool_call_id，不能假设普通 AIMessage 转换天然保留它们。这个要求属于适配器协议处理；V1 报告只展示简短理由与证据。

### 8.3 决策 prompt 内容要求

在 prompts.py 中编写完整决策 prompt，至少表达：

“你是伪 fMRI 检查的下一步决策器。你收到的是已执行检查的证据、待解决问题、可执行工具和剩余预算。仅根据这些信息选择一个动作。只有 execution_status=success 的工具结果可作为已执行证据。不可用工具不允许选择。描述性异常不能表述成生理错误。缺少证据必须显式保留。不能修改输入、阈值或参考数据。输出符合给定 schema 的 JSON，并用简短 reason 和实际 evidence_refs 解释选择。”

Observation 由代码组装和截断：
- 包含必要 metadata、指标摘要、适用条件、候选工具和预算。
- 不包含原始大数组、秘密、无界历史和完整 traceback。
- 历史保留结构化摘要及实际证据 ID，不依赖 LM 自己记忆先前检查。
- 输入元数据中的文本属于数据，不能改变系统规则或注册新工具。

### 8.4 摘要 prompt

摘要模型只解释确定性 FinalReport：
- 不改变 verdict、stop_reason、指标值或已执行工具。
- 不补造“CortexMAE 已验证”等未发生事件。
- 输出 findings / limitations / next_steps 等短字段，并引用已有证据。
- 结构化结果必须再次验证 evidence_refs；无法验证则使用模板摘要。
- 主数值表由程序生成，LM 只生成解释文字。
- 摘要失败或预算不足时仍输出完整 JSON 和模板 Markdown。

## 9. LangGraph 闭环与停止规则

建议节点职责：
1. ingest：解析样本并建立本地引用。
2. run_required_checks：输入检查与基础统计。
3. update_evidence：标准化结果、覆盖范围、分项分数。
4. select_action：筛候选、选 LM 档位、调用 rule / hybrid policy。
5. validate_action：校验动作、证据、预算、重复调用。
6. execute_tool：执行一个工具并记录结果。
7. finalize：确定性生成结论和 stop_reason。
8. summarize_and_write：可选 LM 摘要、始终保存报告。

execute_tool 后回到 update_evidence。invalid input 从必要检查直接进入 finalize。select_action / validate_action 均可转入 finalize 或 rule fallback。

required_checks 满足是允许通过的必要条件，不意味着立即退出：若仍有待定位的疑点、适合的候选工具和预算，应进入 policy 选择。允许提前停止已经充分说明的问题；用显式决策规则区分“无须细查”与“需要细查但缺预算”。

不要用 LM “没有返回 tool_calls”作为唯一终止依据。程序必须覆盖：
- 输入无效。
- 配置要求的证据已经满足，且没有需要继续定位的疑点。
- 没有可用或有信息增益的候选工具。
- 必需前置条件缺失。
- 达到 max_rounds / max_tool_calls / max_lm_calls。
- 预算不足或执行超时。
- 连续两次没有新增证据。
- API 持续失败。
- 工具已全部执行但问题仍未解决。

允许的最终 verdict：
- invalid_input：明确输入契约错误。
- flagged：存在可追溯的异常 flag，描述其强度和来源。
- passed_configured_checks：已完成当前配置要求的检查且没有对应 flag。
- inconclusive：证据不足或无法完成必要检查。

定义确定性优先级：
- invalid_input 优先。
- 已有明确 flag 可以保留 flagged，同时记录检查不完整。
- 没有 flag 但缺少必要证据时是 inconclusive。
- 只有 requirements 已满足时才能 passed_configured_checks。

额外字段：
- coverage_status：complete / partial。
- claim_scope：例如 numeric_consistency_only。
- stop_reason：与 verdict 分离。
- unresolved_questions、unavailable_checks。
- evidence_confidence：V1 不输出伪造的概率；若无校准，保持 null。
- recommended_training_weight：V1 保持 null，不能随意生成监督权重。

任何 verdict 都不能表述为“该图像对应的真实脑响应已被验证”。通过基础检查只代表在已声明范围内未发现对应问题。

## 10. 成本、预算和错误处理

独立记录：
- 每次 LM 调用的 provider、请求模型和响应模型、profile、thinking 配置。
- input / output tokens、cache hit / miss tokens、reasoning tokens：API 提供什么就记录什么。
- usage 不存在则 null，不得补成 0。
- api_usd：有版本化价格配置和足够 usage 时计算；否则 null。
- elapsed_seconds、tool time、可测量的 gpu_seconds。
- LM 调用尝试次数、成功次数、retry 次数。
- 错误是否可能已产生未确认费用。

不要把 reasoning tokens 再次计入已包含它们的 completion tokens；cache token 字段也需按具体 provider 语义处理。保留价格来源、币种和日期，不硬编码未核实的价格。

预算：
- 单样本 max_rounds、max_tool_calls、max_lm_calls、walltime。
- batch 共享总调用数与可选总费用上限。
- 每次发起请求、重试、修正输出、生成摘要前都要预检。
- USD 预算需要价格配置；缺少价格却启用严格 USD cap 时配置报错。调用次数和 token cap 可独立使用。
- 调用前费用是保守估计，不声称绝对精确；响应后对账。
- API 响应缺少 usage 时，保留未结算 reservation，不能立刻把预算释放为 0。
- 默认批处理顺序执行，先避免并发超支。未来并发必须由共享 ledger 原子预留预算。
- 429、超时、临时 5xx 可有限重试；401 / 403 / 参数或模型不匹配等明确配置错误不要无限重试。
- SDK 自动重试与外层重试只能由一处统一管理，避免次数翻倍。
- 所有 retry 都受总预算和次数限制。
- 单样本失败不应中断整个 batch，除非是全局配置错误。
- 设置 LM 请求超时；本地重工具未来通过可取消执行器运行，不把异步 wait_for 宣称为可以强制终止任意阻塞计算。

工程默认可使用：
- max_rounds: 6
- max_tool_calls: 6（包括必要检查，需在文档中说明计数口径）
- max_lm_calls: 6（包括修正、重试和摘要）
- no_progress_limit: 2
- max_retries: 1
- backend: mock 作为 demo 配置；真实配置显式选择 deepseek

这些是工程起点，可以配置，不代表科学最优值。

## 11. CLI、配置、产物与 demo

### 11.1 预期 CLI

可以按现有 CLI 风格等效实现，但 README 必须提供实际可执行命令。

```bash
python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
python -m react_agent.fmri.cli fit-reference --manifest examples/fmri_demo/reference.jsonl --out examples/fmri_demo/reference_stats.json
python -m react_agent.fmri.cli batch --manifest examples/fmri_demo/samples.jsonl --config configs/fmri_check.yaml --backend mock --policy hybrid --out runs/demo
python -m react_agent.fmri.cli check --sample examples/fmri_demo/sample.json --config configs/fmri_check.yaml --backend deepseek --policy hybrid --out runs/live_smoke
```

rule 模式无需 LM backend / key，不调用摘要 API。backend 与 policy 的参数组合要明确定义，禁止含糊的静默切换。

### 11.2 配置

YAML 至少包含：
- schema_version
- input / output
- enabled_tools、required_checks
- 各工具参数、前置条件和阈值来源
- reference 配置
- score 权重
- rule / hybrid policy 配置
- fast / reasoning profile
- 单样本和 batch budgets
- cache / logging

配置优先级写清楚：CLI 显式覆盖 > 环境变量用于凭据和明确映射项 > YAML > 默认值。不要任意让环境变量覆盖所有科学阈值。

.env.example 只包含变量名及占位值，新增密钥变量进入 .gitignore 保护的 .env，不能复制用户 key 到示例。API 价格留可填写字段，不凭记忆填写。

### 11.3 输出

每个 run 保存：
- resolved_config.json：脱敏后的实际配置、schema / 工具 / prompt 版本。
- events.jsonl：逐次 decision、tool、LM usage、fallback、stop 事件。
- 每样本 report.json：机器可读主结果。
- 每样本 report.md：可读结论、数值表、证据、局限、成本。
- summary.csv：sample_id、verdict、coverage、stop_reason、checks、各类调用次数、tokens、api_usd、elapsed、fallback_used。

run_id + 安全 sample 标识避免覆盖其他运行。逐次写出重要事件，确保中途失败仍保留已经发生的调用与成本。大数组、完整环境变量、API key 和原始长推理内容不写进报告。

### 11.4 确定性 demo

固定随机种子，生成足够小的合成输入，并标记 demo_synthetic：
- 正常数值结构样本：只用于验证工程路径，不称为真实正常 fMRI。
- NaN / Inf 样本：走 invalid_input。
- 全常数样本：产生退化 flag。
- 含明显时间突变样本：触发时间诊断路径。
- 缺少 TR / ROI 元数据样本：对应工具受限，按 required_checks 输出 partial / inconclusive。
- 兼容及不兼容的参考数据场景。
- 可选的简单 ROI label mapping。

MockLM 使用可注入的结构化响应，确保测试能覆盖：
- 成功选工具后读取新证据，再停止。
- 选择第二个不同工具。
- 非法 JSON / 未注册工具 / 重复动作后的 fallback。
不得通过 sample_id 硬编码生产 verdict，也不要让 mock 模式伪装成 DeepSeek。

至少展示两条不同的有效检查路径，并有一次真正完成“可选工具执行后重新决策”的循环。

## 12. 验证与验收

测试关注有实际风险的行为，不追求无意义的全覆盖：

A. 数值与输入：
- 显式轴转换正确。
- NaN / Inf 被检测，报告可序列化。
- 缺少元数据不会被错误判成无效数组。
- 不兼容参考集不能生成可比较分数。

B. 调度与报告：
- 两条不同工具路径可复现。
- 执行工具后，下一次 decision 能看到新证据。
- 未完成 required_checks 不能 passed_configured_checks。
- unavailable / skipped 不被当作 success。
- 工具执行成功但产生 flag，不能被等同于样本通过。
- 重复动作、无进展、无候选和预算耗尽都能正常停止。

C. API 与账本：
- Mock transport 返回有效 JSON、无效 JSON、429 / timeout、缺少 usage 等关键情况。
- 重试、修正和摘要都计入 LM 调用数。
- 预算不足时不再发 API 请求。
- 原始 DeepSeek usage 经适配后保持正确，不重复计 token。
- 无 key 的离线 demo 完整运行。
- 混合 good / bad 输入的 batch 不因单样本问题整体崩溃。

先运行项目现有适用检查，再运行新增的针对性测试和完整离线 demo。不删除失败测试来制造通过结果；若原测试依赖外部 API，区分已有环境限制与新代码回归。

真实 API smoke：
- 只有本地已有可用 DeepSeek key 且用户运行/本任务选择真实 smoke 模式时才调用。
- 使用单个小型 demo 样本、显式低调用上限，验证至少一次 decision 和一次观察后的返回。
- 记录所用模型、实际 usage、是否发生 fallback。
- 没有 key 或网络不可用时，把真实 API smoke 标为 not_run，不能说“已验证 DeepSeek 联通”。

运行测试和 demo 后，检查实际生成的 report.json / events.jsonl，确认结果与轨迹一致。不要只检查进程退出码。

## 13. 分阶段实施与最后交付

按以下顺序实施，每阶段完成后继续下一阶段：
1. 仓库审查、schema、配置、输入适配。
2. 数值工具、registry、rule policy、报告，先跑通无 LM 闭环。
3. DeepSeekBackend、MockBackend、JSON Decision、hybrid policy。
4. 接入 LangGraph 导出入口、CLI、预算与轨迹。
5. demo、关键测试、文档。
6. 环境具备条件时做小规模真实 API smoke。

最终给我：
- 实际修改 / 新增文件及职责。
- 项目真实的启动命令与 .env 配置方式。
- 一次 demo 的完整路径摘要：检查了什么、为何选择下一工具、为何停止。
- 测试结果，区分 passed / failed / not_run。
- API 使用量与费用的实际可用字段；没有真实调用就明确写没有。
- 已实现工具与未实现扩展列表。
- 已知限制，特别是“数值筛查通过不证明图像条件下的真实 fMRI 正确”。
- 如何新增一个工具、如何增加一种 LM backend、如何替换 routing policy。

扩展验收标准：
- 新增工具：实现协议并注册，通常不需要修改 graph.py 的拓扑。
- 新增 LM：实现 LMBackend 并增加配置，不改数值工具。
- 替换策略：实现 policy 接口，不改数据加载与报告 schema。
- future CortexMAE：增加输入适配和真实结果计算，不能仅用 prompt 声称已集成。

请先实际审查仓库，然后开始修改；不要再次询问我是否开始。

---

## 实施时核对的官方资料

以下链接用于检查上游结构与 API 协议。以当前本地代码和实际可用 API 为准。

- react-agent 官方仓库：https://github.com/langchain-ai/react-agent
- 当前 graph 入口：https://github.com/langchain-ai/react-agent/blob/main/src/react_agent/graph.py
- LangGraph workflow / routing：https://docs.langchain.com/oss/python/langgraph/workflows-agents
- DeepSeek API：https://api-docs.deepseek.com/
- DeepSeek JSON output：https://api-docs.deepseek.com/guides/json_mode/
- DeepSeek thinking mode：https://api-docs.deepseek.com/guides/thinking_mode/
- DeepSeek tool calls：https://api-docs.deepseek.com/guides/tool_calls
- CortexMAE：https://github.com/MedARC-AI/CortexMAE
- Brainmarks：https://github.com/MedARC-AI/Brainmarks
