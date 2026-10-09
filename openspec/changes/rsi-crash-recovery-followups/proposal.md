# Proposal

## Why

上一份 worker 恢复只覆盖「coder 第一次调用就失败」和「只有 spec、没有 coder 日志」的候选。检查已经通过之后的审查、规划器决策落盘前后、以及 `coder_log.jsonl` 写到一半时进程退出，resume 仍会重新向 planner 要下一条决策，或另开一个候选编号。

## What Changes

- 审查阶段的 LLM 失败记为可恢复；程序错误保留真实异常类型。resume 复用原候选、已写入的代码、检查和已落盘的审查结果，不重跑 coder，不把未完成审查当成否决。
- 没有有效决策时，resume 可以重新请求 planner。有效决策已写入但动作未完成时，沿用该决策，不生成下一条来跳过它。账本上的调用次数是预算依据。无法确认训练子进程是否已启动时阻塞，不静默再开一个。
- 已有 coder 日志的候选仍用原编号恢复。完整日志里的补丁、检查和 `finish_patch` 不重复执行。日志尾行不完整，或磁盘产物对不上日志时阻塞。

## Capabilities

### New Capabilities

- `crash-recovery-followups`: 审查、规划器决策和 coder 日志三个中断点的恢复规则。

### Modified Capabilities

无。

## Impact

- `react-agent/src/react_agent/eeg_research/agentic/worker.py`
- `react-agent/src/react_agent/eeg_research/agentic/loop.py`
- `react-agent/src/react_agent/eeg_research/agentic/native_patch.py`
- `react-agent/src/react_agent/eeg_research/agentic/cli.py`
- `react-agent/tests/eeg_research/test_crash_recovery_followups.py`

不改 EEG 数据划分、评估协议和科学结果判定。测试使用假 LLM 和临时目录。
