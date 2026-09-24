# version 1.6：检索训练和一次自进化链

这是仓库到 **version 1.6** 的实现快照。  
不是改代码的任务说明。不替代 [`version/version_1.5/agent.md`](../version_1.5/agent.md)。  
`claim_scope` 仍是 `numeric_consistency_only`。检索页上的 validation 或 test 分数不进入 fMRI 通过卡片，也不写成最优 EEG 模型。

仓库根目录：`/home/zxuff/data/EEGagent/react-agent`。下文路径相对该根目录。

---

## 1. 现在是什么

1.1 到 1.5 的筛查链还在，没有被这次检索训练替换。

- **1.1–1.4**：数值工具、image16 协议、planned 策略、诊断合同和 StopGate。
- **1.5 / 1.5.1**：本机工作台。数值检查只读展示，不重跑 TRIBE，不改判决。
- **1.6**：同一工作台上的检索实验。EEG / MEG，被试内或被试间，勾选的被试合成一个模型。自进化只在获准的两个设置之间接一次。

工作台仍在 http://127.0.0.1:8765 ，只绑定本机。数值检查和检索实验是两个页面。`api_usd` 保持空。

---

## 2. 检索实验怎么走

1. 打开页面不扫描被试，也不读 GPU。先点「检索」。
2. 检索之后才能多选被试和 GPU，也可以选全部被试。没勾选 GPU 时不设置 `CUDA_VISIBLE_DEVICES`。
3. 被试内：选中被试的 `train.pt` 合成训练集，验证集是训练图像的 10% 留出。每个选中被试的 `test.pt` 不进入 fit。
4. 被试间：选中被试的 `train.pt` 训练，他们的 `test.pt` 做验证；没选中、但磁盘上有的被试整段留出。若选了全部、没有人留在外面，验证改成训练图像的 10% 留出，全部 `test.pt` 仍然不进入 fit。
5. 预检和试运行不写分数。缺文件时不写 `metrics.json`。
6. 「开始训练」按表单上的结束条件往下走。比较和下一次是否开始只看 validation top1。

GPU 秒数上限默认 28800，表单优先于环境变量 `EEG_GPU_SECONDS`。这个数既是预算，也是单次子进程的超时。训练进程使用 `ubp` 环境的 Python，并把本仓库的 `src` 放进 `PYTHONPATH`。GPU 编号按 `nvidia-smi` 的 PCI 顺序，对应 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。

多卡时 batch 会拆开，within-batch top1 的对照规模变小。

---

## 3. 这次加上的界面

- 状态卡片有进度。检查文件时是不确定进度；进入训练后按 `epoch / 总 epoch` 前进。
- 两条本地曲线：训练 loss，以及 validation 的 top1 和 top5。横坐标是 epoch，纵坐标是对应数值。没有 epoch 时不画点。
- 曲线来自该次 campaign 的 `history.jsonl`。刷新检索页会读最近一次有记录的 campaign，不重新训练。
- 每次训练结束之后，只用留出的 `test.pt` 和 test 特征缓存再评一次 within-batch top1 / top5。结果单独显示，并标明未用于选择。缓存或 checkpoint 缺失时，`test_result` 保持空。

结束条件在表单里选，默认是「自动两试，验证集早停」：

| 选项 | 行为 |
| --- | --- |
| 单次跑满 epoch | 只跑基线，跑完填写的 epoch |
| 单次验证集早停 | 只跑基线；连续 5 个 epoch 的 validation top1 没有比最好结果再高 0.001 就停 |
| 自动两试，每次跑满 epoch | 基线结束后自动开始第二次，两次都跑满 epoch |
| 自动两试，验证集早停 | 基线结束后自动开始第二次，每一次都按上面的验证集规则停 |

获准的设置只有两个：`profile_baseline`（weight decay `1e-4`）和 `profile_weight_decay`（`0.01`）。不调用 DeepSeek。单次提升仍记为 provisional。

---

## 4. 已经跑过的一次

`runs/eeg_research/eeg_inter_subject_all_s0` 是一次真实的 EEG 被试间、全部被试基线。

- 状态是 `finished`，停在第 23 个 epoch，计划 50 个。停的原因是当时还没有可选的结束条件，验证集连续 5 次没有超过最好结果。
- `metrics.json` 已写入。validation top1 约 0.363，top5 约 0.772。训练图像 14886，验证图像 1654，两者没有重叠。
- `test_result` 仍是空。这次没有第二次 weight decay 试验。

这次结果只说明这一次基线跑完了。它不是 `multi_trial_real_passed`，也不是选出的模型。

---

## 5. 还不能做

- 不能把这一次 validation 写成可迁移规律，也不能写成最优 EEG 模型。
- 不能把 test 分数用于早停、选下一次，或写进 fMRI 的 `verdict`。
- DeepSeek 在线规划没有跑过。`real_api_passed` 仍是未跑。
- 配置里的 mock 闭环 `bounded_research` 仍是关的，`max_trials` 仍是 1。工作台这条链是单独的两次上限，不把 mock 分数当成训练结果。
- `evidence_confidence` 和 `recommended_training_weight` 保持空。`claim_scope` 保持 `numeric_consistency_only`。
