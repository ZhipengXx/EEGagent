# Spec Delta

## Purpose

worker 在审查、规划器决策和 coder 日志这三个边界崩溃后，resume 必须沿用已落盘的候选、决策和账本，不能另起一条决策或另一个候选来跳过未完成的工作。

## ADDED Requirements

### Requirement: 审查中断后复用原候选

检查已通过、审查尚未写入候选和 evidence 时，LLM 不可用 MUST 持久化为 `review_candidate` 且可恢复，detail 使用该异常类型。其他程序错误 MUST 使用真实异常类型，不得写成 JSON 解析失败。resume MUST 复用原 candidate id、已写入代码和检查结果。已有合法 `review.json` 时不得再次调用 reviewer。不得重跑 coder，不得追加第二次候选或 evidence，不得把未完成审查记成审查否决。reviewer 真正返回否决时，仍按原语义记录一次。

#### Scenario: 审查时 LLM 失败

- **WHEN** 候选代码和检查已经落盘，reviewer 调用抛出 LLM 不可用
- **THEN** 状态为 blocked，阶段为审查，可恢复为真，detail 含该错误类型，候选数和 evidence 数不增加，原候选目录仍在

#### Scenario: 已有审查结果时恢复

- **WHEN** `review.json` 已是合法审查结论，并且候选尚未写入状态
- **THEN** resume 采用该文件，不再调用 reviewer 或 coder，候选和 implementation evidence 只出现一次

### Requirement: 规划器失败与未执行决策分开

没有持久化有效决策时，resume MAY 重新请求 planner。有效决策已写入且 `executed` 为 false 时，resume MUST 重做该动作，不得再生成下一条决策。JSON 解析失败、schema 不符和运行时异常 MUST 分开记录。`cost.json` 的 `llm_calls` MUST 是预算计数来源。训练目录已有 `job.json` 时 MUST 采纳该作业。目录在但没有 `job.json` 时 MUST 阻塞且不得再次启动训练。不可恢复的运行时错误在 resume 时不得被改回 created 后静默再问 planner。

#### Scenario: 决策已落盘但动作未做

- **WHEN** 决策文件里有一条有效决策且 executed 为 false，状态里还没有该动作的结果
- **THEN** resume 后决策 id 不变，planner 调用次数为 0，该动作只执行一次

#### Scenario: 训练是否已启动无法确认

- **WHEN** 将要启动的训练目录已存在，但其中没有 `job.json`
- **THEN** campaign 为 blocked，detail 说明无法确认子进程，训练启动函数不被调用，training_jobs 不增加

### Requirement: coder 日志中断后继续原候选

存在完整 `coder_log.jsonl`、但还没有 implementation evidence 的目录 MUST 仍解析为原候选 id。resume MUST 从完整日志恢复步骤和最近检查，不得重做已成功的补丁、检查或 `finish_patch`。日志尾行不是完整 JSON、代码哈希与最后一条成功补丁不一致，或 `source_manifest.json` 已在而日志没有成功的 `finish_patch` 时，MUST 阻塞并给出原因。候选、implementation evidence 和对应事件最多一次。恢复不得把 `cost.json` 的 `llm_calls` 清零。

#### Scenario: 日志完整时从下一步继续

- **WHEN** coder 日志含一条成功补丁，文件哈希与该记录一致，并且最后一步不是 finish_patch
- **THEN** resume 使用同一候选 id，已记录的补丁不再执行，下一次 coder 请求的步骤号为上一条之后

#### Scenario: 日志尾行不完整

- **WHEN** `coder_log.jsonl` 的最后一行不是完整 JSON
- **THEN** campaign 为 blocked，detail 说明日志不完整，coder 不再被调用
