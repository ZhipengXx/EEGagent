# Proposal

## Why

代码级研究 worker 在实现候选时如果遇到 LLM 返回非法 JSON，异常会逃出 tick 循环。进程退出后 campaign 状态仍停在 planning，界面继续显示规划中。已完成的决策、不完整的候选目录和账本对不齐，恢复时会重复决策并跳过中断的候选。

## What Changes

- LLM 格式错误或调用失败按有限次数在同一决策、同一候选目录重试；每次失败计入预算。耗尽后持久化 blocked、错误类型、阶段、是否可恢复，并写事件。界面显示原因，不再显示规划中。
- 意外程序错误同样落成 blocked，但标记为不可恢复，并留下诊断，不得伪装成 DeepSeek 解析失败。
- status 与界面识别僵尸进程、失效 PID 和过期心跳。不能只靠 `kill(pid, 0)` 判断存活，也不能仅因心跳过期就判定仍在运行的耗时步骤已死亡。
- resume 合并已落盘但未写入状态文件的决策，保留已有候选和证据，不重复决策、账本或训练作业。不完整候选沿用原编号重试，不新建下一个编号。已有 live job 只核对，不新开训练。
- 给现有 campaign 提供先备份、再检查、再恢复的步骤。默认不自动重跑真实 DeepSeek 调用或 GPU 训练。

## Capabilities

### New Capabilities

- `worker-failure-recovery`: worker 在 LLM 失败或进程失联后持久化可恢复状态，并在 resume 时从中断点继续，而不重复已提交的工作。

### Modified Capabilities

无。

## Impact

- `react-agent/src/react_agent/eeg_research/agentic/worker.py` 的 tick 循环与心跳。
- `react-agent/src/react_agent/eeg_research/agentic/loop.py` 的决策落盘与 `save_state` 时机。
- `react-agent/src/react_agent/eeg_research/agentic/llm.py` 的失败记账。
- `react-agent/src/react_agent/eeg_research/agentic/native_patch.py` 与候选目录创建。
- `react-agent/src/react_agent/eeg_research/agentic/cli.py` 的 resume。
- `react-agent/src/react_agent/eeg_research/agentic/view.py` 与工作台状态文案。
- 训练子进程存活判断仍以 `jobs.py` 的进程状态为准，不把 worker 失联误判成训练死亡。
- 失败复现测试：`react-agent/tests/eeg_research/test_worker_failure_recovery.py`。当前它断言 blocked，实现仍停在 planning，所以失败。

## 已核对的落盘顺序

对照一次真实运行目录，不把该 campaign 标识或 PID 写成代码常量：

- 决策事件和 `decisions/` 文件写在 implement 之前。
- `campaign_state.json` 要等 implement 返回后才保存。异常发生在 coder 第一次调用时，状态文件仍是上一次的 planning。
- 候选目录和 `spec.json` 在调用 coder 之前创建。该目录没有 coder 日志，也不在 `candidates[]` 或 evidence 里。
- 失败调用在抛出之前已经写入 `llm_calls.jsonl` 和 `cost.json`。状态文件里的 `llm_calls` 没有跟上。
- `training_jobs` 为 0，没有 live job。
- 界面只读状态文件里的 planning，不读 `worker.json`。进程已是僵尸。

## 后续独立 change

候选连续 `review_needs_fix` 的代码质量问题不在本次修复范围内。
