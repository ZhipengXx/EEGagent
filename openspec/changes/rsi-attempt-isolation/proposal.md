# Proposal

## Why

`c3` 已经 `ready_for_review` 且没有 `review.json`，恢复逻辑却因为同一 campaign 账本里 `c2` 的一次 reviewer 成功，把 `c3` 阻塞成 `review_result_missing`。费用账本被当成了业务阶段已经完成的证据。

## What Changes

- LLM 调用记录带上 campaign 内的 candidate、attempt、phase、operation 和输入哈希。`cost.json` 仍只汇总费用。
- `review.json`、`implementation.json`、`checks.json` 声明所属身份和输入哈希。读取时不匹配则视为没有这份结果。
- 没有匹配的审查文件时继续审查当前候选，不新建下一个候选，也不改写已提交候选。
- 训练 `job.json` 的 `candidate_id` 与本次请求不一致时阻塞，不启动第二个进程。

## Capabilities

### New Capabilities

- `attempt-isolation`: 一次 RSI 运行里的候选、尝试、调用和训练任务互相隔离，恢复只认身份匹配的结果文件。

### Modified Capabilities

无。

## Impact

- `react-agent/src/react_agent/eeg_research/agentic/worker.py`
- `react-agent/src/react_agent/eeg_research/agentic/llm.py`
- `react-agent/src/react_agent/eeg_research/agentic/native_patch.py`
- `react-agent/src/react_agent/eeg_research/agentic/loop.py`
- `react-agent/src/react_agent/eeg_research/agentic/identity.py`
- `react-agent/tests/eeg_research/test_attempt_isolation.py`

不改 EEG 数据划分、评估协议和科学结果判定。
