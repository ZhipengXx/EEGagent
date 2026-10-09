# Design

## Context

见 proposal 的落盘顺序。`run_worker` 只在构造角色后端时捕获 LLM 不可用。`tick` 先把决策写入事件和 `decisions/`，再调用实现；`save_state` 在实现返回之后。实现函数在调用 coder 之前就创建候选目录和 `spec.json`。失败账本写在抛出之前。界面只读 `campaign_state.json` 的 status。

## Goals / Non-Goals

**Goals:**

- 把 LLM 失败和进程失联变成可持久化、可显示、可恢复的状态。
- resume 对齐“决策已落盘、状态未提交、候选目录已创建”的中断点。

**Non-Goals:**

- 不修复候选反复审查未通过的代码质量。
- 不自动重跑真实 DeepSeek 或 GPU 训练。
- 不把某次运行的 campaign 标识或 PID 写进代码。

## Decisions

### 1. 重试包在实现阶段，不重新规划

`LlmUnavailable` 在 `do_implement` / `tick` 内捕获。同一决策、同一候选目录最多 3 次 coder 调用。每次失败沿用现有账本写入。耗尽后 `save_state` 为 blocked：`detail` 为错误类型，`failure.phase` 为 `implement_candidate`，`failure.recoverable` 为 true，并 `event` 记录 `llm_unavailable`。其他异常同样 `save_state`，但 `recoverable` 为 false，detail 使用异常类型，不改写成解析错误。

备选：在 `llm.call` 里吞掉异常并返回空 JSON。不采用。那会把失败伪装成一次正常工具回复。

### 2. worker 存活与训练作业存活分开

界面读取 `worker.json` 的 pid，并用 `/proc/<pid>/stat` 的状态字段。`Z`、`X` 或进程不存在且 campaign 非终态时，视图状态为 `interrupted`，detail 说明 worker 已退出。`kill(pid, 0)` 只作为补充，不能单独通过。状态为 `R`、`S` 或 `D` 时忽略心跳新旧。训练作业继续用现有 `jobs._alive`，不因 worker 心跳过期去结算一个仍在跑的训练。

### 3. resume 的提交边界

恢复时若 `decisions/` 里有状态文件尚未包含的决策，先并入状态，不调用规划器。候选目录存在、状态 `candidates[]` 无对应行、且没有 coder 日志时，视为未完成实现，沿用该编号重试 coder。已有 c1/c2 证据行不改写。`training_jobs` 与 `live_job` 不新增。若 `live_job` 已存在，只走现有 reconcile。锁被占用时返回已有 worker，不第二次 `Popen`。

### 4. 现有 campaign 的人工恢复

操作步骤写在 tasks 中：复制运行目录，检查状态、决策条数、候选目录、账本和 worker 进程，再决定是否 resume。脚本或文档不得默认调用真实模型或启动训练。

## Risks / Trade-offs

- [重试 3 次会多记预算] → 这是验收要求；预算耗尽仍走现有 blocked 路径。
- [把 planning 显示成 interrupted 会改变界面文案] → 只在进程已死时改变，活进程保持原状态。
- [合并磁盘决策可能和内存计数不一致] → 以决策文件和事件为准补齐状态，不追加第二条同 id 决策。

## Migration Plan

不自动迁移历史运行。操作者按 tasks 里的备份、检查、恢复步骤处理。回滚时保留状态文件，不删除候选目录。

## Open Questions

无。重试次数、不完整候选的编号规则和失联判据已在本设计确定。
