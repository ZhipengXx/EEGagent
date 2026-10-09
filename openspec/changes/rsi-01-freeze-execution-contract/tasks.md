# Tasks

## 1. 协议持久化

- [x] 1.1 在 `react-agent/src/react_agent/eeg_research/agentic/execution_protocol.py` 增加由用户 Design 和只读图像身份生成 `execution_protocol.json` 的函数，指纹覆盖种子与验证样本身份；在 `cli.open_agentic_run` 与 `loop.create_campaign` 同一次创建中写入该文件，并让 `evaluation_contract.json` 记录同一指纹。验收：用临时目录和伪造图像 ID 调用创建函数，协议字段与输入一致，缺数据文件时不写出空身份协议。
- [x] 1.2 在 `cli.goal` 中让 `goal.json` 与 `resolved_goal.json` 的 `research_scope` 跟随这次 Design，并保持 `final_test_enabled` 为 false。验收：对非 pooled 的支持任务，两份 goal 的范围与评价合同一致；`react-agent/tests/eeg_research/test_v1_8.py` 里 inter-subject / all 的现有范围断言仍然成立。
- [x] 1.3 让 `worker._design` 改为只读 `execution_protocol.json`。文件缺失或无法解析时把 campaign 标为 blocked，不得构造 EEG / inter-subject / all。验收：准备好协议的临时 campaign 被恢复读取后任务字段不变；删掉协议文件后再进入 worker 准备路径得到 blocked，且没有默认任务命令。

## 2. 启动前核对

- [x] 2.1 在 `workbench.submit_retrieval` 进入 `open_agentic_run` 之前拒绝执行器不支持的代码级研究选择：`per_subject`，以及训练命令表达不了的显式泛化目标或留出被试覆盖。原因写入返回的 blockers 与 log，不创建 campaign。EEG/MEG、被试内/被试间、显式被试或 all、pooled、成对自定义目录保持可创建。验收：对上述拒绝与接受各发一次表单解析，拒绝响应含可见原因且 `runs` 下无新 campaign，接受响应不把任务改成另一组默认值。fMRI 与非 agentic 的 `launch_design` 路径不增加该门禁。
- [x] 2.2 在 `jobs.start_job` 用协议投影 `train_command`，并把 `execution_fingerprint`、fidelity 和实际命令写入 `job.json`。完整训练使用协议中的 epoch；试跑只允许协议 `fidelity_overrides` 里的 3 epoch 与 `single_full`，完整训练的 stop 例外为 `single_early`。命令与协议身份不符时不启动子进程。验收：不启动真实训练，只检查将要执行的参数列表；数据根目录、被试、种子和完整 epoch 与协议一致，试跑仅 epoch 与 stop 按例外变化。

## 3. 结果接收核对

- [x] 3.1 在 `train_entry` 写出的 `metrics.json` 中增加验证样本身份摘要，且 `EEG_FINAL_TEST=0` 时 `test_result` 仍为空。身份摘要用与协议相同的规则计算，本任务不调用拟合循环。验收：对 holdout 规则的纯函数比较摘要；环境开关为关闭时结果字典不含最终测试分数。
- [x] 3.2 在 `runner.accept_job`、`runner.comparable` 与 `loop._record_job` 中同时要求来源绑定、命令身份、样本身份摘要和协议指纹一致，才可 `evaluation_valid`。不一致时原因为 `protocol_mismatch`，不得把 campaign 上的指纹抄到该结果上，也不得进入可比较 evidence 或计算相对对照增益。验收：构造含 `job.json`、`source_binding.json`、`metrics.json` 的临时 job，错命令与错身份两种都得到无效，且 `comparable` 为假；四项都匹配的 full 结果仍可比较。

## 4. 行为测试

- [x] 4.1 在 `react-agent/tests/eeg_research/test_execution_protocol.py` 覆盖不同 UI 任务选择：MEG 或被试内或指定被试被原样冻结；`per_subject` 与不会进入训练的覆盖被拒绝并带原因，且不会变成 EEG / inter-subject / all。验收：`pytest react-agent/tests/eeg_research/test_execution_protocol.py -q -k "selection or reject"` 通过，且测试不调用 DeepSeek、不启动 GPU 训练。
- [x] 4.2 在同一文件覆盖错误合同：命令或验证身份与协议不符的 job 不能 `evaluation_valid`，也不能进入可比较 evidence。验收：`pytest react-agent/tests/eeg_research/test_execution_protocol.py -q -k mismatch` 通过。
- [x] 4.3 在同一文件覆盖 worker 恢复：已有协议时再次准备训练仍使用该协议；协议缺失时 blocked，且不合成默认任务。验收：`pytest react-agent/tests/eeg_research/test_execution_protocol.py -q -k recovery` 通过。
- [x] 4.4 在同一文件覆盖验证划分身份：相同文件、不同种子的 holdout 协议指纹不同，验证图像身份不同；只改数量、不改身份不得视为匹配。验收：`pytest react-agent/tests/eeg_research/test_execution_protocol.py -q -k identity` 通过。不修改 fMRI 测试预期，除非某条旧 EEG agentic 测试因 goal 范围跟随 Design 而需要更新预期。
