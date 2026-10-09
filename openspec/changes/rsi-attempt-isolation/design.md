# Design

## Context

`_reviewer_success_unpaired` 在整个 `llm_calls.jsonl` 里查找 `candidate_reviewer` 成功行。账本行没有 `candidate_id`。`review.json` 只检查 `status` 枚举。

## Goals / Non-Goals

**Goals:**

- 阶段是否完成只由身份匹配的结果文件决定。
- `c3` 缺审查文件时只审查 `c3`。
- 同一尝试在 worker 重启后保持 `attempt_id`。

**Non-Goals:**

- 不凭账本重建已经丢失的模型回复。
- 不自动补写未落盘的补丁或 `Popen`。
- 不改 12 位 campaign 目录哈希长度。
- 不把记忆检索扩大到别的 campaign 目录。

## Decisions

### 1. 身份与范围

`campaign_id` 是运行目录。`cost.json`、`llm_calls.jsonl`、`request_index.json` 和 `worker.lock` 都在这个目录内。`candidate_id` 是 `candidates/cN`。`attempt_id` 写在 `attempt.json`，重启继续同一次实现或审查时不变。`operation_id` 在该尝试的一个 phase 内不变。每次 LLM 调用新建 `call_id`。`job_id` 留在该 campaign 的 `jobs/` 下。

同一 `request_id` 仍返回已有 campaign。不同 `request_id` 使用各自的 `eeg_ui_` 目录。

### 2. 结果文件

`review.json` 必须同时具有 `candidate_id`、`attempt_id`、`phase=review_candidate` 和当前源码 `input_hash` 才可复用。缺少这些字段的旧文件仍可被界面读到，但恢复时不把它当成审查结论，而是重新审查当前候选。哈希或身份不一致同样重新审查，不用别的候选的文件或账本行代替。

写入使用临时文件替换。

### 3. 阶段

`pending`：还没有匹配结果。`running`：正在调用。`persisted`：结果文件已匹配。`committed`：候选和 `ev_impl_<id>` 已在 state 中。已提交则再进入时直接返回，不追加第二次，也不再调用模型。

账本里有成功调用但没有匹配结果文件时，不能编造回复，只能重新调用。预算不足则阻塞在 `review_candidate`，detail 为 `budget_exhausted`。

### 4. 训练

采纳已有 `job.json` 前，若文件里的 `candidate_id` 与本次请求不同，阻塞为 `job_identity_mismatch`。没有该字段的旧文件仍可采纳，以便已有 campaign 可读。

## Risks / Trade-offs

工具已改文件但日志行不完整、`Popen` 已返回但 `job.json` 未写完、模型已返回但回复未写入匹配结果文件，这三处仍不能自动恢复。
