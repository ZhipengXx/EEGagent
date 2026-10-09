# Tasks

## 1. 审查恢复

- [x] 1.1 审查阶段的 LLM 失败记为可恢复的 `review_candidate`，程序错误保留异常类型。resume 复用原候选，不重跑 coder，不把未完成审查写成否决。验收：`pytest tests/eeg_research/test_crash_recovery_followups.py -q -k review`。

## 2. 规划器与未执行动作

- [x] 2.1 没有有效决策时允许 resume 后重新请求 planner；`executed: false` 的有效决策只重做该动作。schema 失败、解析失败和运行时异常分开记录，预算以 `cost.json` 为准。没有 `job.json` 的训练目录不得再次启动。验收：`pytest tests/eeg_research/test_crash_recovery_followups.py -q -k "planner or training or decision"`。

## 3. coder 日志

- [x] 3.1 有完整日志但没有 implementation evidence 的目录仍用原编号。已记录的补丁、检查和 `finish_patch` 不重复执行。尾行不完整或产物对不上时 blocked。恢复不清零 `llm_calls`。验收：`pytest tests/eeg_research/test_crash_recovery_followups.py -q -k coder`。

## 4. 回归

- [x] 4.1 假 LLM 下跑新测试，以及 `test_worker_failure_recovery.py`、`test_execution_protocol.py`、`test_v1_8.py`。不得请求真实 DeepSeek、GPU 或训练进程。
