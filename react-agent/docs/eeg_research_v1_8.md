# V1.8 入口、状态与恢复

代码在 `src/react_agent/eeg_research/agentic/`。campaign 写在 `runs/eeg_research_v18/`。旧的 `runs/eeg_research/` 不会被这里写入。

## 命令

以下命令都从 `react-agent` 目录执行（用 `.venv/bin/python`）。帮助来自真实 parser：`python -m react_agent.eeg_research.agentic.cli --help`。

```text
python -m react_agent.eeg_research.agentic.cli create --campaign eeg_retrieval_research_v1 --request-id <id> --gpu 0
python -m react_agent.eeg_research.agentic.cli start  --campaign eeg_retrieval_research_v1
python -m react_agent.eeg_research.agentic.cli status --campaign eeg_retrieval_research_v1
python -m react_agent.eeg_research.agentic.cli pause  --campaign eeg_retrieval_research_v1
python -m react_agent.eeg_research.agentic.cli resume --campaign eeg_retrieval_research_v1 [--reopen-reason "<原因>"]
python -m react_agent.eeg_research.agentic.cli stop   --campaign eeg_retrieval_research_v1
```

- `create` 从真实的 `split_plan` 冻结 `evaluation_contract.json`。同一个 `request-id` 不会重复创建。
- `start` 和 `resume` 在后台启动 worker（`cli run`），同一个 campaign 只允许一个 worker 持有 `worker.lock`。
- `pause` 和 `stop` 写入 `control.json`，正在运行的 worker 保存状态时不会覆盖它。`stop` 会结束当前作业的整个进程组。
- 已结束的 campaign 只有带 `--reopen-reason` 才能重开，原因会写进事件和证据。

工作台（`127.0.0.1:8765`）检索页有一张「代码级研究」卡片，读 `/api/agentic_status`：显示目标、当前问题、正在做什么、最好可比结果、剩余预算、实验表（不含 test）、候选源码、决策记录，以及「继续 / 本轮后暂停 / 停止当前作业」三个按钮。刷新页面不会触发任何动作。页面上选「代码级研究」再点开始研究，不会写旧 campaign。

## 一次循环

1. worker 先对账在跑的作业：核对 `job.json` 里的 PID、进程启动时间和心跳。
2. 如果有排队的对照，先启动对照，不调用规划器。
3. 规划器只能从运行时算出的可用动作里选。同一类本地动作在没有新实质证据时会被去掉。
4. `implement_candidate`：coder 用受控工具在 `candidates/<id>/extension/` 下写 `eeg_candidate.py`。每次写入后运行时自动在 `ubp` 子进程里检查，子进程不带 API key、不能联网调用。连续两次只读而不写会被拒绝。之后交给 reviewer，reviewer 只返回阻塞问题，不打分。
5. `run_pilot`：如果还没有同 fidelity 的 baseline 对照，就先排队跑对照。训练子进程设置 `EEG_CANDIDATE_MODULE` 和 `EEG_FINAL_TEST=0`，写出 `source_binding.json`。
6. 作业结束后，分数只有在 binding 与候选 manifest 的哈希一致时才进入证据。pilot 只和同 fidelity 的对照比较。

## 训练集统计量 hook

候选可以定义 `fit_statistics(self, encoder, stats)`。训练开始前，运行时只在训练文件上计算每个通道的 mean 和 std，调用这个 hook 一次，并把统计量写到 `train_statistics.json`。候选要把它们存成 buffer，不能是可训练参数。不提供被试 ID。

## 恢复

worker 异常退出后先备份，再检查，最后才由操作者决定是否 resume。这三步不启动训练，也不调用模型。

1. 备份：复制整个 campaign 目录，至少包含 `campaign_state.json`、`decisions/`、`candidates/`、`cost.json` 和 `worker.json`。
2. 检查：确认状态不是暂停或停止；对比 `decisions/` 与状态文件里的决策条数；找出只有 `spec.json`、没有 `coder_log.jsonl` 的候选目录；核对 `cost.json` 的 `llm_calls` 和 `llm_failures`；看 `worker.json` 里的 pid 是否还在，以及 `/proc/<pid>/stat` 是否为 `Z` 或 `X`。
3. 恢复：检查通过后，由操作者执行 `python -m react_agent.eeg_research.agentic.cli resume --campaign <id> --root <runs 目录>`。不要在这一步附带训练命令或模型请求。暂停和停止不会被这一检查改写。

已结束的历史 campaign 若要重开，仍然需要新的预算和 `resume --reopen-reason`。原因会写入事件。详见该次运行目录里的 `report.md`。

## 验收矩阵

| 项 | 状态 | 依据 |
| --- | --- | --- |
| implemented | 是 | `agentic/` 各模块；本次 `tests/eeg_research` 56 项通过 |
| synthetic_passed | 是 | `tests/eeg_research/test_v1_8.py` |
| real_api_passed | 是 | 138 次真实 DeepSeek 调用：planner 67、coder 68、reviewer 3，失败 0 |
| real_patch | 是 | c6、c7、c10 由 API 写出补丁，并在 `ubp` 子进程中通过检查 |
| real_source_load | baseline 与 c10 | j1、j2 的 `source_binding.json` 都与各自 manifest 哈希一致 |
| real_training_passed | baseline 与 c10 的 pilot | 都是 3 epoch、gallery 1654。j1 top1 0.00441354；j2 top1 0.00066505 |
| adaptive_replan_passed | 部分 | d64 读了 c10 的分数后提出新实验；该决定没有执行 |
| resume_passed | 部分 | 多次从持久化状态重开 worker；在跑作业的对账只有合成测试 |
| paired_confirmation_passed | 否 | 没有第二颗 seed |
| performance_improved | 否 | c10 相对 j1 低 0.3748 个百分点 |

## 已知限制

- 候选 checkpoint 不能用 `--evaluate-only` 重新评估：该路径只会构建 `EEGProjectLayer`。候选分数只来自训练过程中的固定候选集评估。
- 候选编码器如果没有 `logit_scale`，`LocalRetrieval` 会自带一个与基线相同初值的尺度。基线编码器仍使用自己的尺度。候选目录会以绝对路径放进子进程，避免相对路径按作业目录解析。
- memory curator、goal compiler、diagnostician、experiment designer、reporter 这几个角色没有实现。诊断和报告目前用确定性代码生成。
- 本机工作台只有路径检查和子进程环境隔离，没有 OS 级沙箱。
- 没有复制 ARIS 的源码，也没有记录上游 commit。
