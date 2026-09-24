# Cursor 增量实施 Prompt：V1.2 — 将 TRIBE v2 作为图像到 fMRI 的生成工具

请在当前 react-agent V1.1 项目上直接实施本任务。目标是让用户输入图像后，系统自动构造 4 秒灰色 + 1 秒图像 + 11 秒灰色的视频，调用真实 TRIBE v2 生成 fMRI，然后进入已有检查闭环。

本文件是增量实施要求，不是已经完成的代码。先审查当前代码，再实施、验证并报告实际结果。不要只回复设计方案；缺少某项资源时完成其余可验证部分，并准确标记阻塞项。

## 1. 当前环境与必须复用的能力

根据用户提供的 V1.1：
- agent 仓库：/home/zxuff/data/EEGagent/react-agent。
- TRIBE 仓库：/home/zxuff/data/tribev2。
- 权重、atlas、CLIP 缓存：/home/zxuff/data/EEGagent/assets。
- fmri_check 已有 registry、LoopRuntime、SampleSpec、rule/hybrid、DeepSeek JSON Action、预算、缓存和报告。
- gray_control_contrast、stimulus_temporal_profile、surface_roi_profile、cross_image_specificity 已在真实 12 点数据上运行。
- 真实 Destrieux、CBIG fsaverage5 Schaefer-400 已接通。
- CortexMAE-P wrapper 已存在，目前拒绝 T=12，未用 padding。
- semantic_consistency 已使用独立 CLIP 缓存。
- 旧生成协议为 4 秒灰 + 1 秒图像 + 7 秒灰，共 12 秒。

请先核实：
1. 当前 graph.py、loop.py、registry、CLI 和 runtime 的真实接口。
2. 当前 TRIBE 静态图像生成脚本、视频构造函数、missing-modality 处理、实际推理配置与导出代码。
3. TRIBE 的可运行 Python 环境、checkpoint、视觉 backbone 缓存和 GPU 配置。
4. 当前 CortexMAE-P 的输入变换、归一化、维度顺序和 forward 路径。
5. 哪些值是实际配置，哪些只是文档中的声明。

保留现有 DeepSeek 配置和全部旧报告。不要覆盖旧 1s7s 生成目录、参考集、cohort index 和版本文档。不要无条件升级两个项目的依赖树。

本轮明确授权实现并小规模运行 TRIBE 生成工具；旧 prompt 中“不重跑 TRIBE”的约束不适用于本轮明确要求的生成任务。先做单图像 + 一个匹配灰色对照，不自动重生成一万张图像。

## 2. 固定的新刺激协议

新增一个独立、不可变的 generation profile，例如：

```yaml
profile_id: static_gray4_image1_gray11_tribev2_v1
pre_gray_seconds: 4
image_seconds: 1
post_gray_seconds: 11
total_seconds: 16
expected_fmri_interval_seconds: 1.0
expected_fmri_timepoints: 16
```

视频内容必须是：

| 时间区间 | 内容 |
| --- | --- |
| [0, 4) 秒 | 均匀灰色 |
| [4, 5) 秒 | 同一张输入图像，静态展示 |
| [5, 16) 秒 | 均匀灰色 |

匹配 control 是整段 16 秒均匀灰色。除目标视觉内容外，两者的编码、尺寸、帧率、生成模型、缺失模态策略和输出映射保持可比。

这是重新构造刺激并真实推理。严禁：
- 给旧 12 点 fMRI 追加 4 个零。
- 复制最后 4 个时间点。
- 将 12 点插值成 16 点后声称重新生成。
- 对旧灰色对照 padding 后复用。
- 把改变刺激协议说成单纯的数组尺寸变换。

延长视频会改变模型可见的时间上下文，不要求新序列前 12 点与旧结果完全相同。

## 3. 图像到视频模块

实现或封装已有 StimulusVideoBuilder，与 TRIBE 推理和 LM 解耦。

接口职责：
- 输入：已注册图像资源、固定 profile、视频编码参数。
- 输出：视频路径、实际视频元数据、刺激事件、内容 fingerprint。
- 同时支持构造该 profile 对应的全灰 control 视频。

参数选择：
- 优先继承旧静态图像脚本已使用的灰度值、尺寸、图像缩放/裁剪、FPS、codec、pixel format，尽量只改变 post_gray_seconds。
- 若旧参数无法确定，使用明确记录的可配置默认，例如 RGB(128,128,128)、24 FPS、512×512 画布、保持比例缩放后灰色补边；这只是工程默认，不代表 TRIBE 作者要求。
- EXIF 方向、RGB 转换、透明通道背景合成必须明确。
- 不把拉伸、中心裁剪、灰色补边混用而不记录。
- 保持等间隔视频帧时间戳。
- 视频 FPS 与 fMRI 采样率分别记录；24 FPS 的 16 秒视频有 384 帧，不等于 fMRI 有 384 个时间点。

当 FPS=24 时：
- pre-gray：96 帧。
- image：24 帧。
- post-gray：264 帧。
- 总计：384 帧。
使用整数帧数量构造，避免 4 秒和 5 秒边界的 off-by-one，不在末尾追加额外一帧，不添加转场。

验证视频：
- 用 ffprobe / 解码器核实真实时长、帧数、FPS、尺寸、音轨和时间戳。
- 抽查边界前后帧，确认图像仅在 [4,5) 秒出现。
- 有损编码后允许明确的像素误差容限，不能只验证源帧而不看导出视频。
- 视频写入临时文件，验证后原子发布。失败的半文件不得成为缓存命中。
- 同一协议的灰色 control 独立缓存，与任意输入图像的名字无关。

默认图像任务没有文本描述和声音。不要给图像自动添加 caption、TTS 或语音识别产生的内容。

## 4. 真正的 TRIBE 生成工具

新增工具 tribev2_generate_fmri，kind=generator。

对外是一个有资源与成本记录的工具；内部可以分为 video_builder、event_adapter、predictor、exporter。不要让 CLI 临时拼 shell 命令成为唯一实现。

输入示例：

```json
{
  "image_asset_id": "image:apple_01b",
  "profile_id": "static_gray4_image1_gray11_tribev2_v1",
  "ensure_matching_control": true
}
```

LM 能看到工具定义，但不能任意改变 checkpoint、duration、GPU 或资源路径。当前用户明确指定图像用此协议，所以 image → TRIBE 属于必要前置动作，可由代码确定性调用，不必付费让 LM 再猜一次。

产出 GeneratedFmriBundle：
- sample：已有 SampleSpec 可消费的真实生成 fMRI。
- control：匹配控制资源，或明确的不可用原因。
- stimulus_video / control_video artifact。
- 原始 preds / segments 的持久化引用。
- sidecar / provenance / compatibility 报告。
- 实际调用数、时间、设备、峰值显存（可测时）、缓存命中情况。
- generation_status：success / partial / error。

生成工具只生成数据，不能给样本“质量通过”的 verdict。

### 4.1 模型与环境封装

定义 TribePredictor backend，提供：
- capabilities / availability：检查路径与环境，不在此阶段加载大模型。
- prepare / load：按需加载真实模型。
- predict：接收固定视频或已核实的 events，返回 preds + segments + run metadata。
- close / release：释放资源。

优先复用已能运行的本地 TRIBE 环境、权重和缓存：
- 不在每个样本中重新下载 checkpoint 或 backbone。
- 不在 import 时加载 GPU 模型。
- 顺序处理共享模型、灰色 control 和已计算视觉特征。
- TRIBE 与 CortexMAE 共用 GPU 时，有明确的释放/串行策略，不让两套模型无界常驻导致 OOM。
- device 使用显式配置，正确处理 CUDA_VISIBLE_DEVICES 后的逻辑编号；不擅自占用全部 GPU。

如果两个项目依赖兼容，可以用 in-process backend。
若 Python / PyTorch / dependency 冲突，优先采用本地 worker backend：
- 使用明确配置的 TRIBE Python 可执行路径。
- 使用结构化 JSON request / result 文件交换路径与元数据。
- subprocess 使用参数列表，shell=False；stdout/stderr 与机器结果分离。
- 独立 worker 仍是本地模型工具，不是 LM API。
- 不为此搭 HTTP 服务、容器平台或另一套 Agent。
- 本轮选择一个能真实工作的 backend 实现；接口允许未来增加另一个，不要求两个都实现。

mock backend 仅用于测试，必须在所有产物中标记 synthetic/mock，绝不可伪装成真实生成。

### 4.2 无音频/无文本的事件构造

先复用当前已经跑通静态图像的 missing-modality 实现，再核对官方/本地模型契约。

注意：
- 上游高层视频接口可能进入提音频与转写流程，不能假定无声 mp4 在所有环境中可直接按期望处理。
- 本轮必须确认没有意外 ASR / TTS / 自动 caption。
- 优先走已验证的 visual-only events 或模型支持的缺失模态逻辑。
- 不凭空补造文本，不猜测音频/文本特征维度，不用随意构造的零向量欺骗 shape 检查。
- 如果实际可运行流程依赖显式静音音轨或特定 null 表示，记录该行为及其语义，样本与 control 使用相同策略。
- 记录最终模态、event 内容摘要和 fallback 策略。

不得为本轮大改 TRIBE 模型架构、参数或训练配置。必要的导出/事件适配留在薄 wrapper，保留可追溯的本地实现。

## 5. 确保“16 秒视频”对应真实 16 个输出时间点

目标输出：
- fMRI 原始导出为 [16, 20484]，前提是当前 checkpoint 确认输出完整 fsaverage5 两半球。
- 采样间隔为 1 秒，并由真实运行配置与 segments 证明。
- 保留 signed 值，不 clamp、abs 或 min-max 处理来“修正”输出。

不能只检查 shape：
- preds 每一行必须对应实际返回的 segment。
- 记录 segment 的真实时间含义、起止/偏移、时长、对应视频区间和选择规则。
- 核实 remove_empty_segments、视频 events 覆盖范围和 chunk/context 行为。
- 灰色画面仍然是有效视频内容，不能因为刺激图像只出现 1 秒就把其他 15 秒标成“没有事件”。
- 不盲目用 remove_empty_segments=False 掩盖错误事件，也不把底层上下文 padding 当成真实刺激。
- 若底层确实产生额外上下文时间点，只允许通过已核实的 segment 映射选出目标 [0,16) 内容对应的 16 行；保留完整 raw preds、原 segments 和选择索引。
- 严禁 preds[:16] 这种没有时间依据的截取。
- 少于 16 个真实有效时间点时，不补齐；返回明确的 temporal_contract_error，并定位事件/导出问题。

此前 sidecar 中“preds[k] 对齐视频时间、不再额外平移 5 秒”的行为，需在当前本地生成路径中再次核实。不要新增一次 HRF shift，不把序号自动解释为生理采集时间。

sidecar 至少记录：
- 新 profile_id 与完整视频参数。
- source image fingerprint、视频 fingerprint。
- checkpoint / config / exporter version。
- 模态与缺失模态策略。
- 原始 shape、实际采样间隔、segment 映射、选中索引。
- LH/RH 顺序、顶点数、空间/有效 mask 身份及核实来源。
- 导出缩放/标准化信息；raw signed 不等于未经训练目标标准化。
- prediction_type=generated_average_subject 或实际配置对应类型。
- temporal_padding_applied=false。
- 独立的 stimulus/control 身份与绑定关系。
- 生成时间和设备信息。
未知项保持 unknown，不能用默认值填成“已核实”。

## 6. 新控制、缓存与协议隔离

控制生成：
- 首个需要此配置的请求生成全灰 16 秒视频及其 fMRI。
- 后续相同配置复用，不能每张图重复生成。
- 若画布、FPS、codec、灰度、模型、模态、时间映射或相关 preprocessing 不同，control 缓存不能共用。
- 与旧 12 秒 control 明确隔离。
- control 失败但图像 fMRI 成功时，可以保存 sample，继续不依赖 control 的检查；对应 evidence coverage 必须 partial。

缓存：
- image generation key = 实际图像/变换 fingerprint + 视频参数 + profile + checkpoint/config + feature/event preprocessing version + 导出规则 + 影响结果的精度/采样参数。
- control key 与协议/模型等绑定，独立于具体图像文件名。
- 缓存命中前核验数据文件、sidecar、shape、segment 契约和 fingerprint，不只看文件存在。
- 图像内容改变但文件名相同，必须失效。
- 编码/归一化或生成 profile 改变，相关结果必须失效。
- 并发时使用简单锁与原子发布，避免重复生成同一 control；默认串行。
- 不完整或失败结果不能成为成功缓存。
- 缓存命中不计新的 GPU 推理，但记录读取验证耗时与 cache_hit。

旧资源兼容：
- 旧 fMRI 输入仍支持 V1.1，不强制重新生成。
- 旧 atlas 若空间与顶点身份相同可复用。
- CLIP 图像缓存若图像处理和模型版本相同可复用。
- 旧 12 点 fMRI cohort、CortexMAE embedding、冻结参考统计不能默认用于新协议。
- 即使某种时间均值 feature 维度相同，也不能忽略协议差异。
- 为 16 点样本构建独立 cohort / reference 身份；资源不足时跳过相关检查并保留限制。
- 初次只有一张图像与 control 时，不能声称已完成跨图像或语义 RSA 检查。

输出新目录，例如：
- runs/tribe_image16_v1_2/...
- 生成 artifact 使用 profile/hash 命名或项目现有 ArtifactStore。
旧 runs/tribe_1s7s_* 与旧版本说明保持不变。

## 7. 将图像请求接入已有检查闭环

当前 validate_input 需要 fMRI 数组，图像不能直接送进去后报“维度错误”。

新增入口请求，使用 discriminated union：
- ImageRequest：input_kind=image、image_path/image_asset_id、sample_id、固定生成 profile。
- FmriRequest：input_kind=fmri、已有 SampleSpec 或 sample 文件。
保留旧 CLI 的兼容方式。明确拒绝含糊输入，不通过扩展名乱猜任务。

生成工具与检查工具的输入不同：
- 在现有 registry/catalog 加入 kind=generator 与类型化 generation entry。
- 已有 CheckTool.run(sample, ...) 接口尽量保持。
- 共用 ToolResult 的必要字段、executor、资源存储、预算、事件记录。
- 不要求所有数值工具学会处理图像。
- 不为了支持生成重新建立一套检查 registry。

推荐实现：
1. 统一 PipelineRunner / input materialization 层接收 ImageRequest 或 FmriRequest。
2. image 且没有有效缓存：通过受预算控制的 executor 调用 tribev2_generate_fmri。
3. 注册生成的 sample、control、视频与 sidecar artifact。
4. 将得到的 SampleSpec 交给现有 fmri_check。
5. existing fMRI 直接进入 fmri_check。
6. 检查结果按当前 rule / hybrid policy 继续选择工具并总结。

保留内部 fmri_check graph.py 拓扑。若 Studio 需要直接接受 image，在外层新增很薄的 fmri_pipeline graph，调用同一个 materialization 层和原检查图，并注册 langgraph.json。CLI / Studio 不要各写一套生成逻辑。

要求：
- tribev2_generate_fmri 必须是真正注册、可由统一 executor 调用和追踪的工具。
- 单独 generate-from-image 命令与 end-to-end 检查调用同一实现。
- 必要生成使用 dispatcher，不能因 LM 提前 stop 被跳过。
- 新 fMRI 注册后才进入需要数组的检查。
- 一旦检查开始，默认不再把生成器当成“试到通过”的候选。
- 已有相同输入的有效结果不重复生成。
- 原聊天图无需合并，不要求 Tavily key。

新增 pipeline_status，例如：
- preparing_input
- generated
- checking
- completed
- generation_failed

GPU OOM / 环境缺失 / 推理失败属于工具执行失败，不能解释为 fMRI 生理异常。没有生成 fMRI 时，保留 pipeline 失败报告，check_verdict=null；不要填成样本“检查不通过”。

## 8. CortexMAE-P 的 T=16 实际验证

复用 V1.1 已实现的 CortexMAE-P wrapper 和 CBIG Schaefer-400 资源。

本轮必须尝试验证：
1. 实际 TRIBE 输出 [16,20484]。
2. 空间聚合后 [16,400]，LH/RH 与 parcel 顺序正确。
3. 按当前 checkpoint 的真实契约转换模型 batch/channel/time 布局。
4. 原始生成数据经已核实 preprocessing 后执行 encoder.forward_embedding。
5. 返回真实、非空且有限的 embedding。
6. padded=false，保留模型身份、输入 shape、输出 shape 和变换版本。

重点检查：
- 直接调用 encoder.forward_embedding 是否绕过了必须的数据归一化、mask 或布局变换。
- 使用该 checkpoint 官方/现有 transform 的真实定义，不因为 shape 匹配就认为 preprocessing 正确。
- 数值统计与 gray contrast 工具继续使用原始生成值；模型专用变换使用独立副本与 artifact。
- 默认对模型进行检查的输入是完整生成序列；不要因为上一个工具生成了 contrast 就静默把减灰结果当作模型预训练分布中的输入。
- 若显式支持 contrast 模式，单独命名并标为实验性表示，不能与 raw 模式共享 reference。
- 不通过修改 positional embedding、关闭断言或绕过错误来假装原模型支持。
- 如果仍失败，报告真实错误和输入布局，不补零凑数。

为 real smoke 提供显式的 required-check cortex_mae 选项，确保模型不会刚跑完 gray contrast 就 stop，让本轮长度兼容性验收落空。这个选项只用于明确要求的请求/验收，不把 CortexMAE 强制设为所有样本的常规检查。

区分：
- encoder_forward_validated：真实运行成功。
- embedding_available：特征已提取。
- reference_score_assessed：只有匹配且可用参考时成立。
- biological_validity：本轮不声称已验证。

T=16 满足长度只是模型输入兼容的一部分，不能将 forward 成功写成伪标签真实可靠。

## 9. 预算、可恢复执行与资源生命周期

扩展现有预算字段：
- max_generation_requests：生成工具请求数。
- max_new_tribe_predictions：实际新增预测次数，image 和 control 分别计数。
- 单样本/全批 generation time budget。
- 已有 max_tool_calls、max_lm_calls、总时间。
- 可选 GPU 使用限制。

首个单图 smoke 预期至少可能需要 2 次新的 TRIBE 预测：image + gray control。不要因为只调用了一次复合工具就把内部推理记为一次。

记录：
- 视频构造、模型加载、特征提取、预测、导出各阶段时间；能够拆分才拆，不能编造。
- 新 control 的共享 setup 成本。
- 每图实际 incremental 成本。
- 后续缓存命中的费用。
- DeepSeek 调用量仍单独记录；TRIBE 本地推理不记成 API token。

遇到超时/OOM：
- 保留已经完成且验证通过的 artifacts。
- 对 worker 设置可终止的执行边界，并清理其进程资源。
- 不反复重试同一 OOM，也不悄悄改 precision / device / checkpoint 导致结果改变。
- 允许显式配置的恢复策略，记录最终生效设置。
- 在新生成、重试和 control 生成前预检预算。
- 在模型转换阶段释放上一个大模型，或按测得显存选择安全的串行策略。
- 报告中保留失败阶段，不让检查器读取半写入数组。

## 10. 新增/修改模块建议

按实际项目布局调整，不重复已有功能：

| 模块 | 职责 |
| --- | --- |
| generation/schemas.py | ImageRequest、StimulusProfile、GeneratedFmriBundle |
| generation/video.py | 图像/灰色视频构造与导出验证 |
| generation/tribe_backend.py | 模型生命周期、事件、推理 backend |
| generation/worker.py | 仅在需要独立环境时的 worker 入口 |
| generation/export.py | preds/segments 校验、sidecar、SampleSpec |
| generation/tool.py | tribev2_generate_fmri 及 registry entry |
| pipeline.py | 图像或 fMRI 请求的统一处理 |
| runtime/resources/budget | 复用并扩展 artifact、control 和预算 |
| cli.py | 新图像入口、生成命令与旧命令兼容 |
| configs/fmri_check_tribe_image16.yaml | 16 秒 profile 与检查设置 |
| docs/tribe_image16_v1_2.md | 使用说明、版本与验收结果 |

可以合并较小模块。不要把视频编码、模型初始化、policy 和报告全部放入一个函数。

配置包含：
- repo_path、backend、python_executable（worker 模式）
- checkpoint_path、cache_dir、device、precision
- video profile 与 inherited encoding 参数
- output directory
- budget
- cortex_mae check requirements

真实凭据仅来自已有环境。命令一律使用 uv run；worker 内使用配置的 TRIBE Python，而不是误用系统 Python 3.10。

## 11. CLI 与验收方式

请实现与下列语义等效、实际可运行的命令，参数名可按现有风格调整并准确写入文档。

```bash
# 只构造视频，检查时间与帧数，不加载 TRIBE
uv run python -m react_agent.fmri.cli prepare-image-video --image /path/to/apple.jpg --profile static_gray4_image1_gray11_tribev2_v1 --out runs/tribe_image16_video_smoke

# 真实生成 image + 匹配 control，不进入检查
uv run python -m react_agent.fmri.cli generate-from-image --image /path/to/apple.jpg --sample-id apple_01b --config configs/fmri_check_tribe_image16.yaml --out runs/tribe_image16_generate_smoke

# 端到端：图像 → TRIBE → 现有检查，明确要求 CortexMAE forward
uv run python -m react_agent.fmri.cli check-image --image /path/to/apple.jpg --sample-id apple_01b --config configs/fmri_check_tribe_image16.yaml --backend none --policy rule --require-check cortex_mae --out runs/tribe_image16_rule_smoke

# DeepSeek 负责后续检查工具选择，生成步骤仍使用相同注册工具
uv run python -m react_agent.fmri.cli check-image --image /path/to/apple.jpg --sample-id apple_01b --config configs/fmri_check_tribe_image16.yaml --backend deepseek --policy hybrid --out runs/tribe_image16_hybrid_smoke
```

CLI 中 backend=deepseek/none 继续表示决策 LM，生成 backend 使用单独配置，避免同一个字段混指 TRIBE 和 LM。

第一次使用真实已存在图像替换示例路径，从本地旧 sidecar/source metadata 解析；无法唯一找到时明确列出缺失路径，不随机挑选别的图像。

不要默认重跑整个旧 8 条面板。单图验证通过后，再提供 batch-images 命令按显式 manifest 和预算生成 16 秒新面板，重建 index；未执行的批量迁移标为 not_run。

## 12. 有意义的测试与真实验收

A. 视频：
- 16 秒总长、三个分段、边界帧数正确。
- 图像转换/透明背景可复现。
- control 全灰且编码配置一致。
- 导出后解码验证，非仅检查构造参数。

B. 生成与注册：
- mock TRIBE 返回 preds + segments 可被正确注册到旧 SampleSpec。
- image 与 fmri 两种入口走正确分支。
- 缺失、重复、无序或不连续 segments 不被自动补齐。
- 16 秒视频但实际仅输出 12/15 点时正确报告契约错误。
- checkpoint/profile/control 不匹配时拒绝比较。
- generation failure 不被标成数据质量异常。

C. 缓存与预算：
- 同图同配置二次请求不再真实推理。
- 图像内容变化、post-gray 从 7 改为 11、checkpoint 或缺失模态策略变化使缓存失效。
- 多样本共享 control，只产生一次匹配 control 推理。
- cache hit 不重复计新增 prediction。
- 控制生成也计预算，不能隐形超支。
- 失败半文件不能作为成功缓存。
- 旧 12 点数据不被覆盖或误用为新输入。

D. CortexMAE：
- [16,V] → [16,400] → 实际所需 tensor layout。
- 真实 encoder forward、输出有限、padded=false。
- 不将 reference 不足混同 encoder 不可运行。
- 12 点旧输入仍按原规则拒绝，不能为了兼容新入口而撤销保护。

E. 真实 smoke：
- 先复用用户现有权重与环境，跑 1 张真实图 + 16 秒灰色 control。
- 检查视频元信息、preds shape、segments 与 sidecar。
- 在该结果上实际运行 gray contrast、一个时间/空间工具及 CortexMAE-P。
- 可用 DeepSeek key 和预算允许时，再运行小规模 hybrid，记录非 fallback 的真实选择及完整轨迹。
- 没有 GPU/权重/环境时完成 mock 和工具代码，真实验收标 not_run；不能伪造输出。
- 只要某一步实际失败，就具体报告失败阶段与错误，不能仅因为顶层程序未崩溃而写 real_data_passed。

测试只覆盖上述关键行为与已有回归风险；不要添加纯粹重复实现的测试来堆数量。

## 13. 最后给我的交付

完成后说明：
1. 真实修改的文件与复用的 V1.1 模块。
2. 使用的 TRIBE 环境、checkpoint 和视频参数。
3. 一次真实生成的 video metadata、preds shape、segments 摘要。
4. 匹配 16 秒 control 的生成与缓存情况。
5. 一个端到端轨迹：image → registered generator → fMRI artifacts → checks → report。
6. CortexMAE 实际 input/output shape、是否执行成功、是否 padding、预处理来源。
7. 实际时间、prediction 次数、GPU/CPU 资源与 API usage。
8. 新生成目录与旧 12 秒资源的隔离情况。
9. 测试结果以及 implemented / synthetic_passed / real_data_passed / not_run。
10. 实际可复制的单图与 batch-images 命令。

生成与检查必须通过代码接通，不能停在新增文件或工具占位；若资源不足，准确列出剩余阻塞，不把未运行写成通过。

## 官方核对资料

- TRIBE v2：https://github.com/facebookresearch/tribev2
- TRIBE 事件、predict 与 segments：https://github.com/facebookresearch/tribev2/blob/main/tribev2/demo_utils.py
- CortexMAE：https://github.com/MedARC-AI/CortexMAE
- CortexMAE 论文（16 帧 / 16 秒输入、parcel 表示与归一化）：https://arxiv.org/html/2510.13768v2

这里的 4+1+11 是用户指定的生成协议；模型输入长度兼容与生成信号的生理合理性需要分别记录。

