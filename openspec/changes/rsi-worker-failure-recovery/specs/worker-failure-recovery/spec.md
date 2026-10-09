# Spec Delta

## Purpose

让代码级研究 worker 在 LLM 失败或进程失联后留下可核对的阻塞状态，并在恢复时从已提交的决策和候选继续，而不是重复工作或把已退出的进程显示成仍在规划。

## ADDED Requirements

### Requirement: LLM 失败按有限次数重试后阻塞

系统在实现候选阶段遇到 LLM 格式错误或调用失败时，MUST 在同一条已做出的决策和同一个候选目录上重试，最多再试 2 次（含第一次共 3 次）。每一次失败调用 MUST 计入调用预算。次数耗尽后，系统 MUST 把 campaign 持久化为 blocked，并记录错误类型、阶段、可恢复标记和一条事件。界面 MUST 显示该原因，不得再把该 campaign 显示为规划中。

#### Scenario: 实现阶段非法 JSON

- **WHEN** 假 LLM 在实现候选时返回无法解析的 JSON，并且重试次数耗尽
- **THEN** campaign 状态为 blocked，detail 含解析错误类型，阶段为实现候选，标记可恢复，训练作业数仍为 0，失败调用出现在预算账本中

#### Scenario: 意外程序错误不被伪装

- **WHEN** 实现阶段抛出不是 LLM 调用失败的程序错误
- **THEN** campaign 持久化为 blocked，detail 含该异常类型和诊断信息，可恢复标记为假，且错误类型不是解析失败

### Requirement: 失联的 worker 不能显示为仍在运行

当 campaign 不是终态时，系统 MUST 用进程表判断 worker。僵尸、已退出或 PID 不存在时，界面 MUST 显示中断及原因。仅当信号检查成功不得视为存活。进程仍处于可运行、睡眠或不可中断睡眠时，心跳过期 MUST 只表示该步骤仍在进行，不得据此把步骤判为死亡。

#### Scenario: 僵尸进程

- **WHEN** worker 记录的 PID 在进程表中是僵尸或已不存在，且状态文件仍是 planning
- **THEN** 界面状态不是规划中，并说明 worker 已退出

#### Scenario: 心跳过期但进程仍在

- **WHEN** worker 进程仍在运行，但心跳时间早于一个耗时的实现步骤
- **THEN** 界面不把该步骤标成已死亡

### Requirement: resume 对齐未提交的中断点

resume MUST 把已经写入决策目录但尚未进入状态文件的决策合并进状态，并且不得再次向规划器请求同一决策。已提交的候选和证据 MUST 保持不变，训练作业数不得增加。只有目录和规格、没有实现证据、没有 coder 日志的候选 MUST 沿用原编号继续实现，不得分配下一个编号。存在未结束的训练作业时，resume MUST 只核对该作业，不得再启动一个训练。

#### Scenario: worker 中途退出后恢复

- **WHEN** 状态文件停在 planning，决策目录里多出一条尚未进入状态文件的实现决策，并且存在一个只有规格文件的候选目录
- **THEN** 一次 resume 后该决策出现在状态中且只出现一次，原有候选与证据条数不变，不完整候选编号不变，训练作业数仍为 0

#### Scenario: 重复 resume

- **WHEN** 一个 worker 锁仍被占用时再次 resume
- **THEN** 系统不启动第二个 worker，决策条数、候选数和训练作业数与第一次相同

#### Scenario: 已有训练作业的恢复

- **WHEN** 状态里有一个尚未结束的 live job，并且 resume 被调用
- **THEN** 系统只核对该作业，不增加训练作业数，也不新写一条训练命令
