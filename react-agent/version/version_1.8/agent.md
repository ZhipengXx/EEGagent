# version 1.8：代码级研究闭环

这是仓库到 **version 1.8** 的实现快照，只写已经发生的事。不替代 [`version/version_1.7/agent.md`](../version_1.7/agent.md)。
fMRI 的 `claim_scope` 仍是 `numeric_consistency_only`，EEG 研究使用自己的 `research_scope`。检索分数不改写筛查判决。

## 已完成

- **评估合同。**从真实的 `split_plan` 生成。全部被试合训标为 `pooled_subject_retrieval`（图像留出验证，不是未见被试）。`final_test` 关闭，新 campaign 不计算测试分。
- **候选插件与源码绑定。**`EEGCandidate` 包住现有 `EEGProjectLayer`。候选只能写在自己的 `extension/` 下。训练时写出 `source_binding.json`，哈希与 manifest 不一致的分数不能进入比较。
- **隔离检查。**候选检查在 `ubp` 子进程里跑，使用真实的 17×250 输入，不带 API key。检查内容：输出形状、数值有限、有梯度的参数、checkpoint 往返，以及可选的训练集统计量 hook。
- **coder。**NativePatch 多轮工具循环：读、写、自动检查、修复。连续两次只读而不写会被拒绝；每一步都查预算。
- **reviewer。**输入包含运行时保证；只返回阻塞问题，不打分；回复格式不合格时修复一次。
- **后台 worker。**训练结束后自动再规划。`pause` 和 `stop` 写入单独的控制文件。已结束的 campaign 只能带原因重开。
- **工作台。**检索页的「代码级研究」卡片与控制按钮。选「代码级研究」时不写旧 campaign；「固定两试（对照）」选项已恢复。
- **修掉的回归。**V1.8 初版新加的 `_load_encoder` 与已有的 checkpoint 加载函数同名，会让 `evaluate_checkpoint` 和 `score_held_out` 用上没有权重的编码器。已改名为 `_build_encoder`。改名前没有任何评估走过这两条路径。

## 真实运行

`runs/eeg_research_v18/eeg_retrieval_research_v1`：

- baseline pilot：3 epoch，验证固定候选集 top1 0.00441354，gallery 1654，GPU 约 270 秒。
- c10 由 API 写成卷积主干加时间注意力，检查通过。审查里属于运行时的阻塞意见被程序降级后，跑了同样 3 epoch 的 pilot。记录分 0.00066505，相对 baseline 低 0.3748 个百分点。这是负结果。
- c10 替换的是整个展平线性编码器。baseline 没有均匀时域平均，所以这个分数不能写成「只改了池化」。
- c1–c9 没有进入训练。逐项原因写在 `report.md`。
- API 调用 138 次，预算 140（用户批准从 100 追加 40）。`api_usd` 未记录。d64 提出了下一个实验，没有执行。

## 还不能做

- c10 的 pilot 已经跑完，但 `performance_improved` 仍不成立。没有第二颗 seed，不能写成已确认改进。
- 不能把 0.363 批内分写成固定候选集基线。
- pilot 分数不能进入 full 排名；3 epoch 的 baseline 只用于排错和同 fidelity 对照。
- 配对 seed 确认还没有真实作业，未确认的结果保持 provisional。
