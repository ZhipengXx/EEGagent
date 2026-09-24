# V1.7 清单

工作台「开始研究」以前进入 `eeg_training/launch.py` 的 `trial_steps`：基线结束后直接安排 weight decay，最多两次。CLI 的 `eeg_research/loop.py` 仍是另一条路，默认 mock，一次调用只录取一个试验。这两条路径现在分开：`policy=legacy_fixed` 保留原来的固定两试；`policy=adaptive` 走 `eeg_research/controller.py`，没有 `DEEPSEEK_API_KEY` 时停在无法继续规划，不退回固定两试。

可复用的位置：

- 预算：`eeg_research/executor.py` 的 `ResearchBudget`。GPU 时间按卡数乘持续时间累计。
- 记忆：`eeg_research/memory.py` 的 SQLite。控制器检索时丢掉带 test 成绩或不兼容的条目。
- 规划：`eeg_research/planner.py` 的 `OfflinePlanner` 和 `FailingBackend`。自适应提示在 `eeg_research/prompts/controller.txt`。
- 训练子进程：`eeg_research/adapters/ubp_retrieval.py`，解释器是 ubp，源码通过 `PYTHONPATH`。
- 评价：`eeg_training/fixed_bank.py` 是跨试验选择指标。within-batch top1 只留在训练诊断里。旧的 `runs/eeg_research/eeg_inter_subject_all_s0/metrics.json` 不改写。

核对：

1. 固定两试是 `launch.py` 里的循环，不是 HTTP handler 里直接写两个 subprocess。handler 调用 `launch_design`。
2. 旧 inter-subject 且指定一个被试时，该被试的 test 文件充当选择用验证。文件名 test.pt 不等于最终测试。全部被试合训时是图像留出，界面写「多被试合训 · 图像留出验证」。
3. 旧 top1 在每个 batch 内按对角线计算。新的 fixed-bank 按图像 ID 在同一候选集合上排序，不丢 query。
4. 表单上的 GPU 秒数同时是预算上限和单次进程超时。新的记账把已用时间留下，下一次不能重新拿到全部预算。未选卡时不会占用全部卡：可以自动绑定一张空闲卡，或勾选指定的卡。
5. 打开检索页不再把全局最近一次 campaign 的曲线填进新表单。
6. 脑图容器原先固定高度。现改为最大高度内完整显示，不靠裁切。逐帧表按帧号各一行，并默认折叠。

模块 A 的 `claim_scope` 仍是 `numeric_consistency_only`。这次没有重跑 TRIBE，没有覆盖旧 run。
