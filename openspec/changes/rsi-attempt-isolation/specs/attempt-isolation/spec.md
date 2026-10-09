# Spec Delta

## Purpose

同一次 RSI 运行里的候选、尝试、模型调用和训练任务必须能对上号。恢复只接受身份和输入哈希都匹配的结果文件，不能用别的候选的费用记录证明当前阶段已经完成。

## ADDED Requirements

### Requirement: 账本不能代替候选结果

每一条新的 LLM 调用记录 MUST 带有新的 `call_id`，以及当时的 candidate、attempt、phase 和 operation。`cost.json` MAY 只汇总调用次数。系统 MUST NOT 因为账本里某个角色曾经成功，就认定另一个候选的审查或实现已经完成。

#### Scenario: c2 的审查成功不能阻塞 c3

- **WHEN** 同一 campaign 中 c2 有一次成功的 reviewer 调用，c3 已 `ready_for_review` 且没有属于 c3 的 `review.json`
- **THEN** resume 只为 c3 发起审查，不创建 c4，不改写 c1 或 c2 的候选和 evidence，也不以 `review_result_missing` 阻塞

### Requirement: 结果文件必须对上身份和输入

`review.json` 在复用前 MUST 匹配当前 `candidate_id`、`attempt_id`、`phase` 和源码哈希。不匹配或缺少这些字段时，系统 MUST 重新审查当前候选，而不是采纳该文件。结果已写入但 campaign state 尚未包含对应 evidence 时，resume MUST 只提交一次。重复 resume MUST NOT 再追加候选、evidence 或训练 job。

#### Scenario: 审查文件已在而 state 未提交

- **WHEN** `review.json` 的身份和输入哈希与当前候选一致，但 state 里还没有该候选的 implementation evidence
- **THEN** resume 提交该候选一次，且不再调用 reviewer

#### Scenario: 输入哈希不一致

- **WHEN** `review.json` 的 `input_hash` 与当前 `eeg_candidate.py` 不同
- **THEN** 系统不采纳该文件，并重新调用 reviewer

### Requirement: 尝试在重启后保持不变

worker 重启并继续同一次实现或审查时，`attempt_id` MUST 不变。同一次尝试里的两次实际模型调用 MUST 使用不同的 `call_id`。两个 campaign 的账本、候选目录和 job 目录 MUST 互不作为对方的完成证据。`job.json` 中的 `candidate_id` 与本次请求不一致时 MUST 阻塞且不启动第二个训练进程。

#### Scenario: 重启继续同一次审查

- **WHEN** 审查在写出 `review.json` 之前失败，随后再次 resume
- **THEN** `attempt_id` 与失败前相同，并且再次调用使用新的 `call_id`

#### Scenario: 训练身份不一致

- **WHEN** 将要采纳的 `job.json` 里 `candidate_id` 不是本次候选
- **THEN** campaign 阻塞，训练启动函数不被调用，`training_jobs` 不增加
