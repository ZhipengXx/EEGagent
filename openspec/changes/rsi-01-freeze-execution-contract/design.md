# Design

## Context

见 proposal.md 的 Why 与已证实问题。代码级研究今天在创建时按用户 Design 写评价合同，worker 启动时却固定构造另一套任务。评价合同的指纹不含种子和图像身份。结果接收只核对来源文件哈希和分数是否存在。

约束：复用现有 Design 与 JSON 文件，不新增 agent 框架；不改 fMRI 链；不改 `policy` 不是 `agentic` 的旧 EEG 路径；不调用 DeepSeek，不启动 GPU 训练。`final_test_enabled` 保持 false。

## Goals / Non-Goals

**Goals:**

- 在 campaign 目录持久化一份执行协议，作为启动命令和接收结果的唯一任务来源。
- 让协议指纹覆盖种子和验证样本身份，并和评价合同、训练命令、job 元数据互相引用。
- 对当前单进程合训执行器表达不了的 UI 选择，在创建前拒绝并返回可见原因。
- 错协议 job 不能标成评价有效，也不能进入可比较 evidence。

**Non-Goals:**

- 不重写研究循环的假设、实现、审查或记忆。
- 不实现每被试一个模型，不实现真实 loss / augmentation 插件。
- 不把 pilot 的 3 epoch 改回用户完整 epoch。该例外写入协议，而不是取消。
- 不扫描或迁移 `runs/` 里的历史 campaign。

## Decisions

### 1. 新增不可变 `execution_protocol.json`

创建代码级研究 campaign 时，与 `evaluation_contract.json` 同一次写入 `execution_protocol.json`。写完后 worker、恢复和结果接收都只读它。缺失或无法解析时，campaign 进入 blocked，不得用 EEG / inter-subject / all 填一套新任务。

协议正文至少包含：

- `dataset`、`exp_setting`、`subject`、`data_root`、`training_strategy`
- `seed`、`full_epochs`、`batch_size`、`lr`、`gpu`
- `val_mode`
- holdout 类划分的 `train_image_ids` 与 `validation_image_ids`；非 holdout 划分则记录实际验证文件路径
- `final_test_enabled: false`
- `fidelity_overrides`：pilot 的 epoch 为 3、stop 为 `single_full`；完整训练的 stop 为 `single_early`，epoch 为 `full_epochs`
- `fingerprint`：对上述身份字段（含种子和样本身份，不含 pid 等运行时字段）做稳定哈希

备选：把这些字段塞进现有 `evaluation_contract.json`。不采用。评价合同今天描述文件角色和指标，执行协议描述“这次进程必须怎么跑、验证的是哪些样本”。分开后，合同仍可按现有公开视图对规划器隐藏最终测试路径，协议则只给启动和核对使用。

### 2. 协议、评价合同、命令、结果的关系

- 评价合同继续由同一次用户 Design 的文件角色生成，并增加与执行协议相同的指纹，或保存 `execution_fingerprint` 字段指向协议。两者不一致则创建失败。
- `goal.json` 与 `resolved_goal.json` 的 `research_scope` 改为这次 Design 解析出的范围，不再固定写 pooled subject retrieval。两份文件仍一起写，避免一个更新、一个过期。
- 训练命令是协议加上所选 fidelity 例外的投影。`job.json` 保存 `execution_fingerprint`、`fidelity` 和实际 `command`。
- `metrics.json` 增加验证样本身份摘要（图像 ID 的稳定哈希，或非 holdout 时的验证文件列表）。不把最终测试分数写入研究循环会采纳的字段。
- `source_binding.json` 仍只证明加载了哪份候选代码。来源匹配不是协议匹配。
- 接收结果时同时要求：来源绑定接受、命令身份字段等于协议、metrics 中的样本身份摘要等于协议、指纹等于协议指纹。任一失败则 `evaluation_valid` 为 false，原因使用稳定代码（如 `protocol_mismatch`）。`comparable` 只比较双方都有效、full、且协议指纹一致的结果。不得把 campaign 状态里的旧指纹抄到未核对的结果上。

备选：只在 worker 里把用户 Design 存进 `campaign_state.json`。不采用。状态文件会被循环改写，恢复时容易再次丢失字段。协议文件单独写一次。

### 3. 样本身份在创建时算好，而不是等训练进程私下决定

holdout 类 `val_mode` 在创建协议时用与训练器相同的留出规则和种子算出训练/验证图像 ID，并写入协议。训练命令必须带上该种子和数据根目录，训练器按同一规则复现。结果里的身份摘要与协议比较。

非 holdout（验证文件本身就是留出被试的 test 文件，且角色不是研究循环可用的最终测试）把验证文件路径纳入指纹。这种划分不声称种子会改变图像集合。

图像 ID 若在创建时读不到数据文件，创建失败并说明缺文件，不写一个空身份的协议。现有 probe 已经会因缺文件成为 blocker；代码级研究只在没有 blocker 时创建，因此身份可以在那次通过的数据根目录上计算。计算只读图像 ID，不加载模型、不占用 GPU。

备选：指纹只加入种子，不记录图像 ID。不采用。种子相同但数据文件内容或顺序规则变化时，仍可能训到另一批样本。

### 4. 支持矩阵放在创建门禁，不放在 worker 替换

代码级研究提交时先判断执行器能力，再写协议：

- 支持：`dataset` 为 eeg 或 meg；`exp_setting` 为 intra-subject 或 inter-subject；`subject` 为 all 或显式列表；`training_strategy` 为 pooled_subjects；数据根目录、种子、完整 epoch、批大小、学习率、GPU 能进入训练命令。成对出现的自定义训练/测试目录也支持，因为现有命令已经会传递它们，且验证身份仍按 holdout 规则冻结。
- 拒绝：`training_strategy` 为 per_subject。原因说明当前执行器只有一个合训过程，不会为每个被试单独训练。
- 拒绝：用户显式填写的泛化目标或留出被试与“只靠模态、设置和被试就能由训练器推导出的划分”不一致。原因说明这些覆盖不会进入训练命令。空的覆盖表示沿用推导结果，可以接受。

拒绝通过现有提交响应的 blockers 和 log 返回，HTTP 仍使用现有错误形态，不新增前端框架。worker 不再拥有构造默认 Design 的路径。

备选：把 per_subject 和泛化覆盖补进命令行，让训练器执行它们。不采用。那是扩大执行器能力，不属于“任务合同与实际执行一致”。

### 5. 最终测试保持关闭

协议和评价合同都写 `final_test_enabled: false`。子进程继续设置不读取最终测试集的环境开关。协议中的验证样本不得包含合同里角色为最终测试的文件。研究循环的 evidence 不读取最终测试分数。

### 6. 旧路径不动

`launch_design` 在 policy 不是 agentic 时的探测、试运行和训练保持现状。fMRI 的启动、状态和视图不读取执行协议。共享的表单解析仍可读取全部字段；门禁只包在代码级研究创建之前。

## Risks / Trade-offs

- [创建时枚举图像 ID 需要打开数据文件] → 只读图像 ID 列表，不跑训练；缺文件则沿用现有 blocker，不创建 campaign。
- [历史 campaign 没有协议文件] → 不迁移。恢复时若缺少协议则 blocked，避免再次合成默认任务。不在本 change 扫描 `runs/`。
- [pilot epoch 与用户 epoch 不同，容易被误判为不匹配] → 例外写入 `fidelity_overrides`，核对时按 fidelity 允许这一项，其他身份字段仍必须一致。
- [MEG 数据是否在本机齐全尚未实测] → 不因此拒绝 MEG。缺文件仍由现有文件检查拒绝。
- [goal 的 research_scope 改为跟随 Design，可能让依赖固定 pooled 文案的测试失败] → 实现时更新对应测试预期；非 agentic 路径不写这份 goal。

## Migration Plan

没有线上迁移。新 campaign 从创建起带协议文件。回滚时停止使用新的核对逻辑即可；已写的协议文件可留在目录中，不参与旧路径。

## Open Questions

无。支持矩阵、指纹覆盖范围和 pilot 例外已在本设计确定。历史 `runs/` 是否已有错配 job 不影响本 change 的行为。
