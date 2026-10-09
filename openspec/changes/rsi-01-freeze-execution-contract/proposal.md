# Proposal

## Why

用户在 Workbench 里选定的 EEG/MEG 研究任务，和 worker 真正启动的训练，不是同一份协议。评价合同按这次点击冻结，训练命令却在 worker 里被重新固定成 EEG、被试间、全部被试。这样写出的分数可以被标成有效并可比较，但无法追溯到用户选择的样本。

## What Changes

- 新增持久化执行协议：创建 campaign 时写入，worker 只读取，不得在进程内悄悄重建成另一套任务。
- 启动训练前核对该协议与即将执行的命令；当前执行器表达不了的 UI 选择必须拒绝并显示原因，禁止静默替换成 EEG / inter-subject / all。
- 接收结果时核对该协议与 job 的实际命令和验证样本身份。不一致的 job 不得标记为有效，也不得进入可比较 evidence。
- 合同指纹必须能区分 seed 以及 holdout 模式下的训练/验证图像身份。现有指纹只覆盖文件角色，不够。
- 保留 `final_test_enabled=false`。研究循环不读取、不使用最终测试集。
- 不改 fMRI 链，不改 `policy` 不是 `agentic` 的旧 EEG 路径。不引入新的 agent 框架。pilot 的 3 个 epoch 和对应 stop 改写保留为协议里写明的 fidelity 例外。

## Capabilities

### New Capabilities

- `execution-protocol`: 把用户选择的任务、冻结的评价合同、实际训练命令和接收的结果绑到同一份可追溯协议上，并在不匹配时拒绝启动或拒绝把结果当成有效可比证据。

### Modified Capabilities

无。仓库里还没有已发布的 spec。

## Impact

- 代码级研究创建与启动：`react-agent/src/react_agent/fmri/workbench.py` 的 `submit_retrieval`，`react-agent/src/react_agent/eeg_training/launch.py` 的 `parse_design` / `launch_design`，`react-agent/src/react_agent/eeg_research/agentic/cli.py` 的 `open_agentic_run` / `goal`。
- 协议与合同：`react-agent/src/react_agent/eeg_research/agentic/contract.py` 的 `freeze_contract`；新增 campaign 级执行协议文件，与 `evaluation_contract.json`、`goal.json`、`resolved_goal.json`、`campaign_state.json` 并列。
- 训练启动：`react-agent/src/react_agent/eeg_research/agentic/worker.py` 的 `_design`，`react-agent/src/react_agent/eeg_research/agentic/jobs.py` 的 `start_job`，`react-agent/src/react_agent/eeg_training/protocol.py` 的 `train_command` / `holdout_image_ids`，`react-agent/src/react_agent/eeg_training/train_entry.py`。
- 结果接收：`react-agent/src/react_agent/eeg_research/agentic/runner.py` 的 `accept_job` / `comparable`，`react-agent/src/react_agent/eeg_research/agentic/binding.py` 的 `binding_accepts`，`react-agent/src/react_agent/eeg_research/agentic/loop.py` 的 `_record_job`。`job.json`、`source_binding.json`、`metrics.json` 仍是 evidence 的来源，但有效性要额外通过协议核对。
- 测试：`react-agent/tests/eeg_research/` 与 `react-agent/tests/eeg_training/` 增加不启动 GPU、不调用 DeepSeek 的行为测试。

## 已证实问题

证据均来自当前 checkout，不是旧问题清单的假定。

1. UI 冻结所选 Design，worker 重新固定构造任务。`submit_retrieval` 用表单调用 `parse_design`，再 `launch_design`；仅当动作为 run、policy 为 agentic 且没有 blockers 时调用 `open_agentic_run`。`open_agentic_run` 用这次点击的 Design 调用 `freeze_contract`，并写入 `goal.json`、`resolved_goal.json`、`evaluation_contract.json`、`campaign_state.json`。`campaign_state` 只额外保存 `gpu`。`worker._design` 不读这些文件，固定返回 EEG、inter-subject、all。`cli` 的 create 命令也使用同一组默认值。因此 MEG、被试内或指定被试会在训练时被换成这组默认值。
2. 设置从创建到训练没有保持一致。`data_root`、`subject`、用户 epochs、`seed`、`training_strategy` 存在于 `parse_design` 读出的 Design，但 `_design` 不携带它们。worker Design 的 `data_root` 为空，`start_job` 把 `data_root()`（环境变量或内置路径）传给 `train_command`。`seed` 落回 Design 默认值 0。`training_strategy`、`generalization_target`、`held_out_subjects` 不在 `train_command` 的参数里；`train_entry.main` 从命令行重建 Design 时也不读取它们，策略因此总是默认的 pooled。`goal()` 把 `research_scope` 固定写成 pooled subject retrieval，可以和按 UI Design 算出的 contract scope 不一致；`create_campaign` 把同一份 goal 同时写入 `goal.json` 和 `resolved_goal.json`。
3. 当前 contract fingerprint 不能确定真实训练与验证样本。`freeze_contract` 哈希的是任务类型、scope、文件角色、`val_mode`、特征缓存路径和标签，不含 seed，也不含图像 ID。`holdout_image_ids` 用 seed 打乱训练图像并切出验证集；`train_entry.build_loaders` 在 holdout 类 `val_mode` 下调用它。同一批文件、不同 seed，指纹可以相同，图像集合不同。`metrics.json` 的 `validation_image_count` 只是数量。`comparable` 用 gallery 数量比较，不能区分身份。被试间且存在留出被试时，`val_mode` 为 `other_subjects_test`，这条路径不调用 `holdout_image_ids`；seed 依赖只在 holdout 类划分上成立。
4. 训练设置与合同不符时，结果仍可被标成有效。`accept_job` 只检查 `metrics.json` 是否有数值分数，以及 `source_binding.json` 的模块名和文件哈希是否与 manifest 一致。它不比较 `job.json` 里的命令和评价合同。`_record_job` 把 campaign 上的 `contract_fingerprint` 抄到结果上。因此错协议的 job 可以 `evaluation_valid` 为 true，并在 full fidelity 下进入可比较 evidence。
5. 最终测试集当前对 agentic 子进程是关闭的，本 change 必须保住这一点。`freeze_contract` 写 `final_test_enabled: false`。`goal()` 同样写 false。`start_job` 和 `research_env` 设置 `EEG_FINAL_TEST=0`。`train_entry.main` 在该环境变量为 0 时不调用留出测试打分，`test_result` 保持空。

## 尚未证实的风险

- 没有扫描 `runs/`。磁盘上是否已经存在“合同是 UI 选择、命令是 EEG / inter-subject / all”的历史 job，本次未知。
- 没有检查本机 MEG 数据目录是否齐全。命令行接受 `meg`，baseline 编码器使用调用方传入的通道数，没有把通道数写死成 EEG。因此不把 MEG 判成当前执行器不支持。
- `per_subject` 只进入 `protocol_label` 和 `split_manifest` 的策略字段。`split_plan` 不按它为每个被试启动单独模型。当前 agentic 执行器实际只有一个 pooled 的 `train_entry`。这是源码行为，不是一次真实训练的实测。

## 后续独立 changes

以下只记录依赖，本 change 不创建它们的目录，也不实现它们。

- 配对 seed 与 replicate：依赖本协议先把单次运行的 seed 和验证图像身份冻结。没有这份身份，多次重复无法判断是在重复同一划分。
- 真实可执行的 loss 与 augmentation 插件：依赖训练命令已经遵守协议。插件不能在协议之外改样本或目标。
- memory 与 planning：依赖 evidence 只收录协议匹配的有效结果。规划若读到错协议分数，后续假设会建在不可比的证据上。
- review gate：依赖启动前已经拒绝执行器不支持的任务。审查不能代替“静默换成另一套任务”的缺口。
