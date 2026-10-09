# Design

## Context

`align_interrupt` 只把决策文件并进状态。`incomplete_candidate_id` 遇到 `coder_log.jsonl` 就放弃该目录。`do_implement` 不捕获 `review()` 的异常。`tick` 在 `decide()` 之前用状态里的计数自增 `llm_calls`。

## Goals / Non-Goals

**Goals:**

- 三条崩溃边界都能在假 LLM 测试里 resume，并且不重复候选、evidence、决策和训练。
- 对不上磁盘产物的窗口明确 blocked。

**Non-Goals:**

- 不自动补写已经发生但没记入日志的补丁或 `Popen`。
- 不改评估协议和数据划分。
- 不把旧决策里缺失的 `executed` 字段当成未执行。只有显式 `executed: false` 才重做动作，避免把这次改动之前已经做完的决策再跑一遍。

## Decisions

### 1. 决策游标

新的有效决策写入 `executed: false`。动作结果进入即将保存的状态后才改为 true。`tick` 见到最后一条有效决策显式为 false 时重做该动作，不调用 planner。schema 失败写入 `ok: false`，不执行动作。`LlmUnavailable` 不写有效决策。其他异常 `recoverable: false`，`cli resume` 不把这种 blocked 改回 created。

`llm_calls` 在 `cost.json` 有该字段时只采用账本。没有账本文件时保持状态里的原计数，避免把没有账本的旧测试预算清掉。

### 2. 审查与实现进度

`ready_for_review` 先写 `implementation.json`，再审查。合法 `review.json` 直接采用。账本里已有成功的 reviewer 调用但没有合法审查文件时 blocked，原因 `review_result_missing`。真正的 `needs_fix` / `blocked` 结论仍只记录一次候选和 `ev_impl_<id>`。

### 3. coder 日志

只重放以换行结束且每行都可解析的日志。从中恢复步骤、history 和检查。已记录的工具不执行。`finish_patch` 已成功则直接返回 `ready_for_review`。否则从下一步请求 coder。

以下情况抛出 `RecoveryBlocked`，worker 持久化为不可恢复 blocked，不重跑工具：

- `coder_log_tail_incomplete`
- `unlogged_code_change`
- `unlogged_finish`

### 4. 训练副作用

`job.json` 已在则采纳为 `live_job`，把 `training_jobs` 设为该作业序号，不再次 `Popen`。目录在但没有 `job.json` 时 detail 为 `training_start_unconfirmed`。假设文件已在则读回，不另写。

## Risks / Trade-offs

工具已经改写文件但对应日志行没有完整落盘，以及 `Popen` 已返回但 `job.json` 还没写完，都不能自动恢复。这两处保持 blocked。
