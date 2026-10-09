# Tasks

## 1. 失败重试与 blocked

- [x] 1.1 在 `react-agent/src/react_agent/eeg_research/agentic/worker.py` 的实现路径捕获 LLM 不可用，同一决策和同一候选目录最多共 3 次；每次失败仍写入 `cost.json`。验收：`pytest react-agent/tests/eeg_research/test_worker_failure_recovery.py -q -k invalid_json` 在假 LLM 下得到 blocked，detail 含 `DeepSeekParseError`，阶段为 `implement_candidate`，`recoverable` 为 true，训练作业数为 0，且不调用真实 API。
- [x] 1.2 同一循环捕获非 LLM 的程序错误，持久化 blocked、异常类型和诊断，`recoverable` 为 false，不得写成解析失败。验收：假后端抛 `RuntimeError` 的测试断言 detail 类型不是 `DeepSeekParseError`。

## 2. 失联识别

- [x] 2.1 在 `view.py` 与工作台状态文案中根据 `/proc/<pid>/stat` 识别僵尸、退出和缺失 PID；`kill(pid, 0)` 不能单独算存活。进程为 `R`/`S`/`D` 时心跳过期仍显示步骤进行中。验收：构造僵尸或缺失 PID 且状态为 planning 的临时 campaign，视图状态不是规划中；构造仍在睡眠的 PID 与旧心跳，视图不显示步骤已死亡。

## 3. resume 中断点

- [x] 3.1 在 `cli.py` 的 resume 中合并已写入 `decisions/` 但未进入状态文件的决策，不再次调用规划器。验收：临时 campaign 含两条已提交候选证据和一条未进状态的实现决策，resume 后决策只增加那一条已落盘记录，候选数与证据数不变，训练作业数仍为 0。
- [x] 3.2 不完整候选（有目录和 `spec.json`、无 coder 日志、无 evidence）沿用原编号重试，不创建下一个编号。已有 `live_job` 时只 reconcile。锁占用时第二次 resume 不启动第二个进程。验收：对应三个假 LLM、无 GPU 测试分别检查候选编号、训练作业数和 worker 数量。

## 4. 行为测试

- [x] 4.1 扩展 `react-agent/tests/eeg_research/test_worker_failure_recovery.py`，覆盖实现阶段非法 JSON、worker 退出后恢复、重复 resume、已有 live job 的恢复。验收命令：`pytest react-agent/tests/eeg_research/test_worker_failure_recovery.py -q`。每个用例打印或断言状态文件 status、视图 status、候选数和 `training_jobs`。不得请求真实 DeepSeek，不得启动 GPU。

## 5. 现有 campaign 的人工恢复

- [x] 5.1 在 `react-agent/docs/eeg_research_v1_8.md` 写明先复制运行目录、再核对状态文件、决策条数、候选目录、`cost.json` 和 worker 进程，最后才 resume。文档中的命令不得默认执行真实模型调用或 GPU 训练。验收：文档包含备份、检查、恢复三步，且没有无条件的训练启动命令。
