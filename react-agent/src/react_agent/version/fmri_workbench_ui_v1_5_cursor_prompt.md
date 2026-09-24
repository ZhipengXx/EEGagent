# fMRI Workbench UI V1.5：科研结果页、脑图浏览与指标解释

将本文完整交给 Cursor，在现有 react-agent 的 fMRI Web Workbench 上做增量改造。本文是任务书，不是已经修改完成的说明。

依据：用户提供的平台截图，以及 V1.3/V1.4 的 report、planned、memory 和分层诊断约定。必须先审阅实际前后端代码，不能假设截图背后已经使用 React。

目标仓库：/home/zxuff/data/EEGagent/react-agent。

## 1. 设计参考：Langfuse Trace Detail

主要模仿对象：

- [Langfuse Observability 官方介绍](https://langfuse.com/docs/observability/overview)
- [官方界面截图](https://langfuse.com/images/docs/tracing-overview.png)
- [公开示例项目入口](https://langfuse.com/docs/demo)

参考其白色背景、细分隔线、紧凑信息密度、运行列表、步骤树和证据详情的组织方式。使用你自己的标题、数据与组件，不照搬品牌标识。

本项目的对应关系：

| Langfuse 中的结构 | 本项目中对应的内容 |
| --- | --- |
| Trace / 一次运行 | 一个 run 下的具体 sample 检查 |
| Observation / 单步 | generate、planner、repair、tool、memory、stop |
| Trace tree | 按实际事件组织的 Agent 执行轨迹 |
| 单步详情 | 选择理由、输入摘要、结果、证据来源、成本 |
| Metadata | protocol、policy、backend、signal mode、版本 |

这些是界面和信息组织参考，不要求部署 Langfuse 或把研究数据发给其服务，也不要求复制它的所有筛选器和企业功能。

科研内容单独设计：脑图是主要分析对象，不把它塞进日志详情，也不把完整原始 JSON 当首页。

## 2. 当前截图中的问题和对应修改

1. 顶部新建检查表单占据首屏大部分高度。
   → 收进“新建检查”抽屉，结果页优先展示当前样本和结果。

2. 左侧直接展示长文件路径，缺少清晰的样本名、状态和选中态。
   → 改成可搜索的运行列表；路径放详情并支持复制。

3. 脑图按很多帧挤在一张缩略拼图中，单帧和文字都太小。
   → 默认显示可放大的当前帧，配时间轴；完整 montage 作为另一种视图。

4. true、answered、described、JSON 对象都被画成红绿渐变条。
   → 数字用数值/曲线；布尔用状态标签；类别用标签；结构化数据展开显示。

5. “未评估”处于红端，容易被读为低质量；描述完成处于绿端，容易被读为高质量。
   → 将运行状态、问题覆盖、数值 finding 和科学效度分开显示。

6. 页面内有明显的多层滚动容器。
   → 主内容使用一个纵向滚动区域；不要把完整 workflow.html 再嵌套成可滚动报告页。

7. 技术路径、YAML、policy/backend 和内部字段占据主要视觉位置。
   → 用可读名称表示模式，技术细节按需展开。

8. 指标没有解释；用户不知道“大/小”意味着什么。
   → 建立统一指标释义注册表，给出含义、公式、适用范围和判断限制。

不要只换颜色和圆角后宣称完成。以上信息组织和交互问题必须解决。

## 3. 先审阅代码，再选择最小改造路径

读取 AGENTS.md、启动入口、路由/API、模板、静态资源、生成 workflow.html 的代码、report schema、可视化产物和事件协议。

必须确认：

- 当前前端是 React/Vue，还是 Python 模板 + HTML/JS；
- 结果页是否使用 iframe；若有，数据是否已有结构化 endpoint；
- report.json、events.jsonl、plan_history、memory、图片 manifest 的真实字段；
- 脑图只有 montage，还是已有逐帧图片和原始 fMRI 数组；
- 图像/产物是否已有安全访问路由；
- UI 中同一时间一个任务的限制由哪里执行；
- 实际 required/optional question、screening_decision、coverage 的来源。

优先复用已有技术栈：

- 已有 React/TypeScript：模块化组件和 typed view model。
- 纯模板/HTML：使用模块化 JS、CSS、模板片段和统一 view model；不要为了样式重写为 Next.js。
- 图表优先复用现有依赖；若确需新增，可使用适合当前栈的 ECharts 或已有 React 图表库，只选一套。
- 图标复用现有库或少量本地 SVG。
- 字体和静态资源默认本地/系统可用，不让离线工作台依赖公网 CDN。

不要重写 TRIBE worker、检查图、planner、memory DB 或判决逻辑。本轮只改呈现与必要的只读数据适配；必要的单帧图片导出复用现有渲染器。

先给出实际文件映射，随后直接实现。不要只返回设计说明或一张静态 demo。

## 4. 视觉系统

设计方向：浅色、安静、清晰的科研工作台，信息密度接近 Langfuse。脑图颜色最突出，界面本身克制。

建议设计 token，可依据仓库已有风格做小幅调整：

~~~css
:root {
  --bg-canvas: #f6f8fb;
  --bg-surface: #ffffff;
  --bg-subtle: #f8fafc;
  --border-default: #e2e8f0;
  --text-primary: #17212f;
  --text-secondary: #526075;
  --text-muted: #64748b;
  --accent: #2563eb;
  --accent-subtle: #eff6ff;
  --status-success: #15803d;
  --status-warning: #b45309;
  --status-danger: #b91c1c;
  --radius-control: 6px;
  --radius-panel: 10px;
  --shadow-panel: 0 1px 2px rgb(15 23 42 / 4%);
}
~~~

规则：

- 顶部栏 56px，桌面左栏约 224–240px，内容 padding 24px，区块 gap 16–24px。
- 标题 24px/600；分区标题 16px/600；正文 14px；辅助文字最低 12px。
- 使用系统无衬线字体，兼容中文；数值使用 tabular-nums，代码/路径才用等宽字体。
- 按钮/输入框高约 36–40px，表格行约 44px；正常标签不过度堆成胶囊。
- 边框和留白优先，不叠加大量阴影。
- 不用彩虹渐变、玻璃背景、发光边框、巨型 KPI 字号或大面积深色侧栏。
- 产品名保持“fMRI Workbench”或“fMRI 检查工作台”，不要创造新的营销品牌。
- 不以进度条样式表达没有上限、没有方向性或没有校准的数值。
- 状态永远同时有文字/图标，不能只依赖颜色。

## 5. 页面骨架与导航

页面结构：

- 顶部：产品名 / 当前样本标题 / 运行状态 / 新建检查。
- 左栏：搜索、运行记录、样本名、简短时间、状态。
- 主区：样本信息与摘要，再进入五个标签页。

五个标签页：

1. 概览
2. 脑图与时序
3. 指标解释
4. Agent 轨迹
5. 产物与记忆

必须保留 selected_run_id + sample_id 的清晰边界；不要只用 sample_id 当全局唯一键。

切换记录时取消或忽略旧请求，禁止左侧选中 accordion 而右侧仍显示另一次 antenna 的指标/脑图。把加载状态与最终数据一起绑定到同一个 run/sample key。

左栏：

- 主行：sample_id，可选小缩略图；
- 次行：运行时间 + 简短模式；
- 右侧小状态图标；
- hover/详情中可复制完整 run_id/路径；
- 不把长路径按字符断成三四行作为标题；
- 用后端真实时间字段，不解析目录名伪造时间。

已有记录时默认显示明确选中的记录或最新可用记录；没有记录时才显示简洁空状态和“新建检查”入口。

## 6. 新建检查：移到抽屉

桌面从右侧打开约 480px 抽屉，小屏为全屏面板。

默认只显示：

- 刺激图像：服务器路径输入 + 文件名预览；支持什么就显示什么。
- 检查模式：可读名称 + 一句说明。
- 开始检查按钮。

模式显示示例，实际映射到真实配置：

- 快速数值检查：使用规则策略；不调用决策 LLM。
- 分层诊断：使用 planned + DeepSeek；按证据执行进一步检查。

不要因为换 UI 自动修改当前默认配置。提交前显示 resolved policy/backend/model 的紧凑摘要。

高级设置折叠：

- sample_id；
- 输出目录；
- config_path；
- policy/backend override。

文件交互：

- 如果后端只接受服务器路径，就明确写“服务器图片路径”。
- 浏览器选择文件不能自动获得服务器绝对路径；没有上传接口就不要摆一个假的上传区。
- 若已有上传功能则接入；不为了 UI 本轮另建复杂文件管理器。
- 关闭路径输入的拼写检查，长路径在输入内部横向滚动；提供复制完整值。
- 图片预览通过已有受控资源接口，不拼任意 file:// 地址。

运行时：

- 防止重复提交；
- 单任务限制由后端继续执行，前端展示“当前有任务运行中”并可进入该任务；
- 有真实取消 API 才显示“取消任务”，否则显示运行状态和日志入口；
- 缺 API key/路径不存在/配置错误，在具体输入附近展示明确错误；
- key 保留服务端，浏览器只知道是否配置，不展示或复制真实密钥。

## 7. 概览：首屏让用户知道发生了什么

顶部样本摘要：

- 刺激图像缩略图；
- sample_id；
- 协议：4s 灰屏 · 1s 图像 · 11s 灰屏；
- 实际 signal shape，例如 16 × 20,484；
- 当前运行状态与检查模式。

协议/shape 必须读取当前样本 metadata，旧 12s 样本不能显示成 16s。

下面最多四个紧凑摘要块：

1. 数值检查结论：配置内通过 / 发现需关注项 / 阻塞 / 暂无法判断。
2. 必需问题覆盖：满足数/必需数，点击查看问题清单。
3. 实际检查路径：例如 L0 → L1 → L2；没有阶段数据就显示实际工具数。
4. 执行成本：工具执行次数、LLM 调用次数；费用未知明确“未提供”，不写 $0。

摘要文案使用确定性模板，根据真实结果生成，不在每次打开页面时调用 LLM：

示例：
“当前配置要求的数值检查已完成。完成了对照、时序与空间描述；尚未评估生物学有效性。”

只有实际覆盖了这三个部分，才能出现这句话；旧报告仅跑四个工具时应按实际生成。

首屏放两个主要区域：

- 当前帧脑图预览 + 紧凑时序预览；
- 检查范围与待跟进问题，点击进入完整页面。

1440×900 下，首屏应看到样本身份、结论、范围和一个能看清的脑图预览；不要被新建表单挤走。

限制说明用一条简短、持久可见的文字：
“检查范围：生成 fMRI 的数值一致性；不代表真实脑响应已验证。”

详细科学限制放可展开的“解释与限制”，避免用大段免责声明挤满首屏。

## 8. 运行状态、覆盖状态和判决分开

为每种状态配置固定 label、icon、tone、description：

| 原始状态 | 默认中文 | 显示语义 |
| --- | --- | --- |
| running | 运行中 | 进度，不代表质量 |
| completed | 运行完成 | 程序执行完成 |
| pass_configured | 配置内数值检查通过 | 限定范围内结论 |
| flagged | 发现需关注项 | 展示实际 finding 和依据 |
| blocked | 检查被阻塞 | 展示阻塞原因，不能直接说样本坏 |
| abstain | 暂无法判断 | 证据/资源/预算不足等实际原因 |
| answered | 已回答 | 问题状态 |
| described | 已描述 | 有描述性证据 |
| unassessed | 未评估 | 中性，不是零分 |
| unavailable | 资源不可用 | 资源状态 |
| not_applicable | 不适用 | 不作为未完成或错误 |
| skipped | 未执行 | 显示真实跳过原因 |

unassessed 通常灰色；若它属于实际 required 且阻止完成，可在问题行显示“阻塞完成”提示，但不把它画成红色低质量分。

described 用蓝/灰标签，不能用“大绿色满分条”。

必须读取每个问题的真实 required 标记，不把 reference_quality 硬编码为“可选”或“必答”。旧记录缺该字段时显示“要求未记录”。

覆盖比例必须使用后端满足证据合同的结果。不能前端简单 count(status!=unassessed)，更不能把所有 described 一律当 answered。

## 9. 脑图查看器

默认展示单帧或少量真实解剖视图，宽度充分；不要把 16 帧 × 多视图挤进首屏。

控件：

- 信号模式：contrast / raw / gray，仅显示真实可用项；
- 帧滑块与前后帧按钮；
- 播放/暂停，明确是浏览播放，不是生成过程；
- 当前绝对时间和相对刺激 onset 时间；
- 原尺寸/适应窗口/全屏查看；
- 单帧 / 全部帧 montage 切换。

单帧上的视图名称必须来自实际导出，例如 LH lateral、LH medial、RH lateral、RH medial；若现有渲染器是另三种视角，就按真实名称显示，不擅自重命名。

16s 示例：

- 绝对 t=4s 对应刺激相对 t=0s；
- slider 的 index 与时间由 segments 映射；
- 不做额外 HRF 平移，不通过 UI 插值创造更多帧。

色阶：

- 必须显示信号来源、单位或“模型输出单位”、色条范围和 0 的位置；
- signed raw/contrast 使用以 0 为中心的发散色阶；正/负表示该数值的符号，不表示好/坏；
- RMS 等非负数使用顺序色阶；
- 同一信号模式跨帧默认固定色阶；
- raw/contrast 并排比较时，若单位可比，提供共享色阶；
- 若用各模式独立色阶增强对比，明确标注“色阶不同，不直接比较颜色强弱”；
- 未知 vmin/vmax 时不能凭图片颜色编出精确色条，显示 metadata 缺失；
- 所有缩放只改变显示，不改原始数组或 report。

产物适配：

1. 已有逐帧产物 → 直接用 manifest。
2. 只有 montage，但有原数组和现有 renderer → 增加可缓存的逐帧导出，复用已有 renderer，不重跑 TRIBE。
3. 无法取得逐帧数据 → 提供可缩放 montage，并明确“当前记录只有拼图”；不能放一个滑块却总显示相同图。

图片 manifest 至少包含 frame_index、absolute_time_s、stimulus_relative_time_s、
signal_mode、view_id、image_url、color_limits、units、source_artifact_id。
缺值为 null，不能补造。缓存键包含输入/模式/视角/色阶/渲染版本。

浏览器懒加载相邻帧，避免一次加载所有大图或把完整 [T,V] 数组发送给前端。

## 10. 时序与空间图表

时序主图：

- x：实际绝对时间；可切换刺激相对时间，改变标签不改变数据；
- y：明确指标名，例如 contrast RMS；
- 背景标注灰屏/图像/灰屏窗口；
- 显示真实观测点，使用直线连接；不做默认样条平滑；
- 点击时间点联动脑图帧；
- hover 显示值、时间、信号来源；
- 无序列只给摘要时，不根据 peak 和 RMS 摘要伪造整条曲线。

R(t)、相邻 RMS 幅度变化 A(t)、全顶点模式变化 S(t) 分图或用明确切换，不直接画成同一“响应”。

transition 指标显示 from→to，例如 3s→4s，不能偷偷当某一单点时间。

空间分析：

- 有 ROI 数据则显示可排序 top-K 表/条形图；
- 标注计算量：signed mean / RMS / transition contribution；
- 展示 hemisphere:label_id 与可读名称；
- 不按“视觉区域应该第一”着色判断好坏；
- 有未分配 atlas 顶点，显示覆盖或 unassigned；
- 非同单位、非同归一化的指标不可放在一条公共数值轴上。

没有数据时展示具体缺失原因和已有产物入口；不画随机示例数据。

## 11. 建立统一指标解释注册表

新增 metricDefinitions 或等价模块，避免解释文本散落在组件里。定义文件可以是 Python/JSON/TS，选择当前栈中最容易保持单一来源的形式。

每项至少：

~~~json
{
  "id": "gray_control_contrast.overall_delta_rms",
  "label_zh": "图像相对灰屏的整体差异",
  "value_kind": "number",
  "unit_kind": "model_output",
  "meaning": "概括图像预测与匹配灰屏预测的整体数值差异。",
  "formula_source": "must_match_tool_implementation",
  "direction": "no_quality_direction",
  "threshold_kind": "none",
  "limitations": [
    "不是信噪比",
    "不是统计显著性",
    "不能单独证明真实脑响应正确"
  ],
  "explanation_version": "ui.metric.v1"
}
~~~

示例 formula_source 是占位要求；实现时必须读工具代码并写入真实定义，不能给生产用户显示 must_match_tool_implementation。

指标显示分四层：

1. 行内：名称、当前值、单位/状态、简短解释。
2. info 按钮：一句“这是什么”，支持 hover 和键盘 focus。
3. 点击行/“查看解释”：右侧详情面板。
4. 详情面板底部：来源 execution_id、signal_mode、原始字段、公式和 raw JSON。

详情面板固定结构：

- 这是什么；
- 怎么计算；
- 当前结果可以说明什么；
- 不能说明什么；
- 是否有阈值，以及阈值来源；
- 来自哪个工具/输入。

解释默认确定性生成，不新增解释指标的 LLM API 调用。

## 12. 首批指标解释必须覆盖

先读取实际实现核对公式，尤其 overall_delta_rms 的聚合轴/权重、lag1 的平均方式和 ratio 的 epsilon。下表给出解释要求，不授权改变计算。

| 字段 | 显示名称/解释要点 | 必须避免 |
| --- | --- | --- |
| shape/T/V | 时间点数 × 皮层顶点数；不是体积图像尺寸 | 将 1s 输出间隔说成真实扫描仪 TR |
| finite_ratio | 有限数值占比；100% 表示没有 NaN/Inf | 100% 质量或准确率 |
| valid_for_numeric_checks | 满足当前数值分析输入合同 | 生理有效 |
| degenerate_signal | 是否触发当前退化/近常量判据，展示判据来源 | false 等于信号正常 |
| mean/std/min/max/quantiles | 模型输出的分布摘要；单位依据 metadata | 均值负数就是错误、BOLD 百分比 |
| comparable | image 与 gray 满足已实现的比较条件 | 对照具有真实生理噪声估计能力 |
| overall_delta_rms | 图像与匹配 gray 的整体 RMS 差异；公式核对代码 | SNR、质量分 |
| normalized_delta_over_control_rms | 差异 RMS 相对 gray RMS 的比值，显示分母 | 脑激活提高多少百分比 |
| peak_frame/peak_time_s | 指定统计量峰值的位置；显示 raw/contrast 和定义 | 生理 HRF 峰、通用异常时间 |
| temporal_change_ratio | 全局空间均值变化相对其波动尺度；按真实公式解释 | 局部变化已被完整检查 |
| max_frame_diff_ratio/max_diff_ratio | 最大相邻变化相对中位变化，可能受小分母影响 | 超阈值自动判坏 |
| max_step_over_median | 明确是哪条序列的 step，及其分母/稳定性 | 和完整顶点模式变化混为一谈 |
| lag1_corr_mean | 相邻时间点相关性摘要，说明短序列限制 | 越高越健康 |
| ROI contribution | 所声明 transition 能量在 ROI 内的贡献占比 | 激活概率、统计显著性 |
| cortex_mae.embedding_available | 表征提取成功状态 | 质量概率 |
| reference_score_assessed | 是否实际使用兼容参考进行相应评估 | 无参考也显示通过 |
| coverage/answered/described | 当前问题的覆盖与回答类型 | 连续 0–100% 科学效度 |
| evidence_confidence | 如无值则“未提供/未校准”，原因以 report 为准 | 从工具数自动推算置信度 |
| recommended_training_weight | 无值显示“未提供”，当前不自动赋权 | null 转 0 或 1 |
| llm_calls/repairs/accepted_plan_count | 调用、修复、接受计划分别显示 | HTTP 成功次数等于计划数 |
| input_tokens/output_tokens | 实际 provider usage；缺失标未知 | null 计 0 |
| api_usd | 有值显示来源/货币；无值显示费用未提供 | 没价格表也显示免费 |
| new_tribe_predictions | 本轮新增预测数量 | 0 必然等于 cache hit |

示例解释，仅当实际值和定义匹配时使用：

- finite_ratio=1：“全部输入元素为有限值，可以继续数值分析；这不评估脑响应真实性。”
- normalized_delta_over_control_rms≈0.296：“差异 RMS 约为灰屏预测 RMS 的 0.296 倍；这是模型输出的相对差异。”
- max_step_over_median≈6.69：“最大相邻变化约为中位相邻变化的 6.69 倍。该量可能触发进一步查看，不能单独判定异常。”
- described：“已生成这个维度的描述性证据，尚不等于证明该维度正常。”
- unassessed：“这个维度尚未评估；查看资源或配置原因。”

0.296、6.69 来自此前 accordion 示例，不是当前截图 antenna 的真实数值。只用于开发 fixture，并标注来源；生产页面完全以当前 report 为准。

原始 ratio 默认显示 × 或比值，不随意转换成百分比；finite_ratio 等真正的比例可显示百分比，并保留原始值。

## 13. Agent 轨迹页面

参考 Langfuse 的左侧步骤树 + 右侧详情：

- 左边约 280–340px：真实事件形成的时间线或树；
- 右边：选中步骤的解释与数据；
- 小屏上下排列，不压缩成不可读多栏。

步骤类型：

生成 / 输入检查 / 初始规划 / 格式修复 / 工具执行 / 重规划 / 记忆检索 / 记忆记录 / 停止。

每步显示类型、状态、耗时（若有）、简短工具/角色名。

右侧默认显示：

- 这一步做了什么；
- 为什么执行：真实 brief_reason 或明确“未记录选择理由”；
- 输入信号、问题和证据引用；
- 结果摘要、后续问题；
- 模型调用时的 token usage 和 requested/response model；
- 原始输入/输出按需展开。

不要展示或要求模型提供隐含思维链；展示已有可审计理由、工具参数和结果即可。

父子关系只来自实际 parent_call_id/plan/execution 引用。旧事件没有层级时用平面时间线，不编造树结构。

repair 和 revise 必须视觉上区别：

- repair：格式/字段修复；
- revise：因新证据更新计划。

旧报告仅有两次成功调用和一次 accepted plan，就按这些事实展示，不能补造“二次推理成功”节点。

## 14. 产物与记忆

产物按类型分组：

- report JSON/Markdown；
- 图像与时序数据；
- plan/history；
- events/usage；
- 其他真实可用文件。

下载走现有受控路由，以 run/artifact id 访问。浏览器不能把任意绝对路径作为自由读取接口参数。

Memory：

- 显示命中的 memory 数量及其 scope、verification_status；
- 点开可看来自哪个历史 run、为什么适用、影响了什么选择；
- 未记录影响则显示“未记录对路径的影响”，不宣称它提升了准确率；
- episode_written 与 curator_called 分开；
- unverified 不得呈现为“已成功筛除案例”；
- UI 不直接修改数据库中的信任级别；已有 feedback 功能则保留明确的来源与证据字段。

## 15. 前端数据适配层与组件封装

建立 normalizeRunReport / buildRunViewModel 或等价适配层，集中处理 V1.1–V1.4 字段差异。

最低 view model：

RunIdentity、SampleMetadata、RunStatus、DecisionSummary、CoverageItem、
MetricView、BrainFrameManifest、TemporalSeries、TraceEvent、ArtifactLink、MemorySummary。

规则：

- unknown/null/false/0 必须分别处理；
- 不使用 value || fallback 误吞合法 0/false；
- 每个 metric 有明确 value_kind(number/boolean/status/object)；
- 保留 provenance 和 execution_id；
- 多次执行同一工具按 execution_id 展示，不能按 tool_id 覆盖；
- 不修改原 report，不前端重算或放宽 verdict；
- 老报告缺信息则显示缺失，不能用最新样本 metadata 填充。

推荐组件边界：

- WorkbenchShell / RunSidebar / RunHeader；
- NewCheckDrawer；
- DecisionSummary / CoverageTable；
- BrainViewer / FrameTimeline / TemporalChart；
- MetricTable / MetricExplanationPanel；
- TraceExplorer / StepDetail；
- ArtifactList / MemoryPanel；
- StatusBadge / EmptyState / LoadingState / ErrorState。

模板项目使用对应的模块和片段即可，不必照搬 React 组件命名。

## 16. 加载、失败与实时进度

必须实现：

- 无记录；
- 正在加载记录；
- 生成中；
- 检查中；
- 部分结果已可用；
- 完成；
- 任务失败；
- 资源缺失；
- 历史 schema 信息不足。

运行过程沿用已有 SSE/WebSocket/polling；没有实时接口时可有界轮询状态，勿为 UI 强行搭新消息系统。

- SSE 重连/轮询应避免重复事件；
- 浏览旧结果不受当前任务影响；
- 浏览器刷新后通过真实 job/run id 恢复状态；
- 只显示已知进度：“已完成 4 项检查”；动态计划没有固定总数时不伪造 80%；
- 新事件自动跟随只在用户位于日志底部时启用，避免打断手动查看。

## 17. 响应式与可访问性

- 1440–1920px：完整左栏 + 主区；分析页按空间允许布局双栏。
- 1024–1280px：可折叠左栏；指标解释抽屉覆盖主区，不挤成四五栏。
- <768px：导航抽屉，卡片单列；表格允许局部横向滚动，页面本身不横向溢出。
- 抽屉/弹窗有焦点管理、Esc 关闭、关闭后返回触发按钮。
- tooltip 也能键盘打开；所有图标按钮有 accessible name。
- 时间滑块支持方向键；播放状态有文字。
- 图表提供数值表/下载入口；图片有描述性 alt。
- 不用色彩作为唯一状态通道；不使用过浅小字。
- 减少动画；尊重 prefers-reduced-motion。

主页面只保留一个主要纵向滚动区域。sidebar 列表和明确的详情抽屉可独立滚动；不要让脑图、整份报告和外层内容同时有无提示的纵向滚动条。

## 18. 数据与安全边界

本轮不触发额外 DeepSeek 调用、不重跑 TRIBE、不改变科学阈值。
用户主动提交新任务沿用原流程，和普通浏览行为明确区分。

输出 HTML/Markdown 时转义路径、sample_id、finding、tool result 等文本，避免把报告内容当可执行 HTML。
图表只接收允许的数值/标签，不执行工具结果中的任意脚本。

不向第三方 demo 上传研究数据。参考网站只用于看设计。

## 19. 必须完成的验收

功能验收：

1. 现有真实 run 能打开；当前样本身份、脑图、指标和轨迹一致。
2. 新建检查通过原后端真实提交；不是只有按钮动画。
3. 新建抽屉、高级配置、模式说明、错误状态可用。
4. 存在逐帧产物时 slider 真正改变图片并和时间标签一致。
5. 只有 montage 时提供真实可用的缩放体验，明确能力边界。
6. 布尔/类别/JSON 不再画红绿渐变条。
7. described、unassessed、不适用、资源缺失和 failed 可清楚区分。
8. 指标解释覆盖第 12 节，公式与工具实现核对；无阈值时不补正常范围。
9. 两次 HTTP 成功、一份 accepted plan 的例子不会显示成两份计划。
10. null 的费用、置信度、训练权重保持未知/未提供；0 和 false 显示正常。
11. 切换 run 时慢请求不会覆盖新记录；同工具不同 execution 不互相覆盖。
12. 无不必要的额外 LLM/生成调用，静态资源无外网 CDN 依赖。

视觉验收：

- 至少在 1440×900、1280×800 和 390px 宽浏览器中查看截图。
- 检查首屏可读性、脑图大小、中文换行、路径溢出和滚动区域。
- 用实际数据截图，mock 数据必须标明 demo，不得冒充真实结果。
- 打开“概览”“脑图与时序”“指标解释”“Agent 轨迹”以及“新建检查”抽屉分别检查。
- 有浏览器自动化则补少量关键交互测试；没有则明确记录人工检查步骤。
- 不写大量样式快照测试来替代真实浏览器查看。

## 20. 交付和实施顺序

按以下顺序完成，避免停留在半成品皮肤：

1. 核对数据结构与参考界面，建立 view model/设计 token。
2. 主骨架、运行列表、新建抽屉、概览与状态语义。
3. 脑图查看器、真实时间轴、时序与空间数据展示。
4. 统一指标解释注册表和详情面板。
5. Agent 轨迹、产物与 memory。
6. 响应式、空/错/加载状态、真实运行验证与截图。

写 docs/fmri_workbench_ui_v1_5.md：

- 实际修改文件及启动命令；
- 原 UI 到新 UI 的变化；
- 字段/公式与来源映射；
- 哪些图片交互依赖已有产物，哪些真实验收通过；
- 浏览器验收截图和未完成部分；
- 明确本版本只改 UI/只读适配及必要渲染导出，没有改变筛查判决。

提交最终说明时给出实际页面入口、启动方式、已验证交互和剩余限制。不要仅回复“已完成现代化设计”。
