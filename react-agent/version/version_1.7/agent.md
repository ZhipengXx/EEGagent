# version 1.7：按轨迹规划的自适应研究

这是仓库到 **version 1.7** 的实现快照。  
不是改代码的任务说明。不替代 [`version/version_1.6/agent.md`](../version_1.6/agent.md)。  
`claim_scope` 仍是 `numeric_consistency_only`。`evidence_confidence` 和 `recommended_training_weight` 保持空。`api_usd` 保持空。检索页上的 validation 或 test 分数不进入 fMRI 通过卡片，也不写成最优 EEG 模型。

仓库根目录：`/home/zxuff/data/EEGagent/react-agent`。下文路径相对该根目录。

---

## 1. 现在是什么

1.1 到 1.6 的链还在，没有被这次自适应研究替换。

- **1.1–1.4**：数值工具、image16 协议、planned 策略、诊断合同和 StopGate。
- **1.5 / 1.5.1**：本机工作台。数值检查只读展示，不重跑 TRIBE，不改判决。
- **1.6**：同一工作台上的检索实验。自进化只在获准的两个设置之间接一次，不调用 DeepSeek，选择指标是 validation 批内 top1。
- **1.7**：检索实验改为按磁盘轨迹规划。每次点击读取已有试验，DeepSeek 只选下一步。选择指标改为验证固定候选集 top1。

工作台仍在 http://127.0.0.1:8765 ，只绑定本机。三个面是首页、筛查、检索。

---

## 2. 相对 1.6 的修正

1.6 的自进化是获准的两试：基线 weight decay `1e-4`，然后 `0.01`。1.7 换成按磁盘轨迹规划的自适应研究。下面按出现顺序记。

- 工作台启动时读取仓库根目录 `.env`（`override=False`），再绑定端口。规划才能看到 `DEEPSEEK_API_KEY`。密钥不写入这份说明。
- 规划回复必须是一个 JSON：`action`、一句简体中文 `reason`、`changes`。中文动作名、`decision.action`、`next_action` 会映射到 9 个英文动作。非法动作停在 `planning_blocked`，保留截断后的原文，不退回固定两试。
- 控制台只显示一次「下一步」和一句理由。`/`、`app.js`、`app.css` 带 `Cache-Control: no-store`。
- 查看数据、检索经验、分析曲线、分析检索错误、提出实验、验证集评价只记入 `research_trace.jsonl`，然后立刻再规划，不启动 GPU。同一完成试验数下，做过的本地动作会从合法列表拿掉。模型再选一次已做过的动作就重问，第二次才停。
- 相同 seed、学习率、权重衰减不再重训，直接复用。新目录编号是已有最大编号加一。`replicate_candidate` 才换下一个没用过的 seed。
- 状态不再放在这一次点击的内存里。每次点击和每次再规划都读 `trials/tN`。没有 `trial_config.json` 且 checkpoint 相同的旧试验算同一个设置。根目录那次 epoch 23/50、批内 top1 `0.3630694088088461` 仍是历史批内记录，不计入试验，文件不改。
- 选择指标改为验证固定候选集 top1。每个验证批次用该批 EEG 对整组冻结验证图像排序，写入 `history.jsonl`。早停和存 checkpoint 都用它：连续 5 轮没有比最好再高 `0.001` 就停。批内 top1 只作诊断。测试固定候选集只在该次训练结束后算一次，写入该试验的 `test_result`。它不参与早停，不送给规划器，也不进记忆。已完成但缺验证分的试验，下次点击用 `--evaluate-only` 把分数写入 `eval_scores.json`，不改 `metrics.json`、`history.jsonl`、`last.ckpt`。
- 试验轨迹表离开高度固定 180px 的 `.chart`，改用按内容增高的 `.trial-trace`。学习率和权重衰减按有效数字显示，`0.0002` 不再显示成 `0.000`。
- 训练决策收成单因素搜索。一次只改学习率或权重衰减，相对当前验证最好的已完成设置乘 2 或除 2。学习率允许 `1e-6` 到 `1e-2`，权重衰减允许 `0` 到 `1e-2`。已有试验状态为训练中时，这一轮不启动新训练。两个以上已完成分数里，最好的没有比第二好至少高 `0.005`，提示词要求停止。这个差距规则在提示词里。运行时硬性拦住的是「已有训练中」和「同一设置不重训」。剩余预算不是训练理由。未记录设置按协议默认值交给模型：跨被试学习率 `1e-5`，权重衰减 `1e-4`。

---

## 3. Check 工作流

数值检查链仍是 1.1–1.5 那条。1.7 没有替换它。筛查走 `POST /run`。图在 `src/react_agent/fmri/graph.py`。

一次检查的顺序是固定的。模型不能改阈值或判决。

1. 输入是服务器上的图片路径和 `configs/fmri_check*.yaml`。没有浏览器上传。
2. 生成或复用 TRIBE 预测和配对灰屏。打开已有结果不重跑 TRIBE，不改判决。缓存命中时可以不产生新的预测。
3. `ingest` 读入样本。
4. `run_required_checks` 先跑配置里的必做检查。标准深度做 L0（`validate_input`、`basic_statistics`）和 L1（灰屏对照、刺激时序）。时序问题只接受 contrast 上的 `ras_v1`。raw 时序不能把这题标成已回答。
5. `update_evidence` 把工具结果写成证据。
6. `select_action` 选下一步。快速检查用 rule。分层数值检查用 planned：DeepSeek 只提议下一步，覆盖和 `screening_decision` 由合同与 StopGate 写。
7. `validate_action` 拒绝不在允许列表里的工具。通过后 `execute_tool` 只执行一个工具，再回到证据更新。
8. 没有可做的下一步时进入 `finalize`。
9. 深入筛查额外把 ROI、特异性、校准参考、image–fMRI RSA 收进必做问题。参考样本量小于 30 时，参考分布不把判决翻成通过。没有兼容参考时，参考质量保持未评估。
10. 报告同时留下四件事，互不替代：
    - `verdict`：配置内数值检查
    - `screening_decision`：`pass_configured` / `abstain` / `blocked` / `flagged`
    - `stop_reason`
    - 探索性 ticket。`max_step_over_median` 相对 5.0 只开 ticket，不把质量标成通过。

`passed_configured_checks` 不是每个工具都跑过，也不是真实脑响应正确。

---

## 4. Auto-research 工作流

检索实验在同一工作台，`POST /api/retrieval`。`policy=adaptive` 时进入 `src/react_agent/eeg_training/launch.py` 的 `run_adaptive_loop`。

上限是 6 个不同设置、12 次语言模型调用、4 次执行尝试。GPU 秒数默认 28800，表单优先于环境变量 `EEG_GPU_SECONDS`。这个数既是预算，也是单次子进程的超时。出错不退回 1.6 的固定两试。子进程用 `ubp` 环境的 Python，并把本仓库的 `src` 放在 `PYTHONPATH` 最前。GPU 编号按 `nvidia-smi` 的 PCI 顺序。

划分仍按 1.6：

- 被试内：选中被试的 `train.pt` 合成训练集，验证集是训练图像的 10% 留出。每个选中被试的 `test.pt` 不进入 fit。
- 被试间：选中被试的 `train.pt` 训练，他们的 `test.pt` 做验证。没选中、但磁盘上有的被试整段留出。若选了全部、没有人留在外面，验证改成训练图像的 10% 留出，全部 `test.pt` 仍然不进入 fit。

一次点击内部按这个顺序循环：

1. `read_trace` 扫描 `trials/tN`。持有 `.lock` 为训练中。有 `metrics.json` 为已完成。有 `error.txt` 或 phase 为 failed 为失败。phase 停在 loading 或 training、又没有指标，为中断。
2. 已完成、有 checkpoint、没有验证固定候选集分、没有记过补算错误、又不是重复 checkpoint 的试验，先补算。同一轮补算失败就写入错误，避免死循环。补算只写 `eval_scores.json`。
3. 用不同设置的个数更新预算。语言模型次数用完则停。
4. 观察含 `base_setting`、每个非重复试验的数值设置、`tried_settings`。表上有学习率、权重衰减、状态、验证 top1、验证候选数，以及当前最好验证分。批内 top1 和任何带 test 的字段都剥掉。
5. DeepSeek 按 `src/react_agent/eeg_research/prompts/controller.txt` 返回一个动作。缺密钥、非 JSON、动作不在列表里，都是 `planning_blocked`。
6. `stop_research` 结束这次点击。本地动作写入 `research_trace.jsonl` 后回到第 1 步。
7. 训练或重复 seed：若已有训练中的试验，结束这次点击。设置已存在则复用，并回到第 1 步。设置已满 6 个则停。否则把 `trial_config.json` 写到 `t{最大编号+1}`，启动一次训练。这次训练在子进程里跑完才返回，返回后回到第 1 步再规划。

训练时，每个验证批次用该批 EEG 对整组冻结验证图像算固定候选集。训练结束后，用最好的 checkpoint 对测试候选集算一次。测试分只出现在页面「未用于选择」一列。

九个动作是：查看数据、检索经验、分析曲线、分析检索错误、提出实验、训练、验证集评价、重复 seed、停止。`changes` 只允许 `learning_rate` 和 `weight_decay`。

页面每秒读 `/api/train_status`。试验表列出设置、状态、验证固定候选集 top1、测试 top1。编号最大的试验作为当前曲线。旧曲线没有逐轮固定候选集分时，仍标成「validation 批内（仅诊断）」。

---

## 5. 这次 campaign 的磁盘快照

`runs/eeg_research/eeg_inter_subject_all_s0` 是一次真实的 EEG 被试间、全部被试搜索。验证候选 1654 张，测试候选 200 张。两个 top1 不能直接比较。

`cost.json`：不同设置 6/6，语言模型调用 4/12，`api_usd` 空。根目录旧指标不改。那次仍是 epoch 23/50，批内 top1 `0.3630694088088461`，`test_result` 空。

- t1–t4：没有 `trial_config.json`，checkpoint 相同，按默认 `lr 1e-5`、`wd 1e-4` 计一次。t1 验证 top1 0.0414，测试 0.270。t2–t4 标为同 t1。
- t5：`lr 0.001`，`wd 1e-4`。验证 0.0417，测试 0.275。
- t6：`lr 0.0005`，`wd 1e-4`。验证 0.0405，测试 0.271。
- t7：`lr 0.002`，`wd 1e-4`。验证 0.0426，测试 0.246。
- t8：`lr 0.0002`，`wd 1e-4`。已完成，epoch 18/50。验证 0.0384，测试 0.268。
- t9：`lr 0.002`，`wd 0.0005`。已完成，epoch 11/50。验证 0.0420，测试 0.245。

这些分数只说明这一次搜索。单种子提升仍是 provisional。不是 `multi_trial_real_passed`，也不是选出的模型。测试分没有用来选择。

---

## 6. 还不能做

- 不能把 0.363 或任何批内 top1 写成固定候选集分。
- 不能把测试分用于早停、选下一设置、送给规划器，或写进 fMRI 的 `verdict`。
- 提示词里的乘 2 / 除 2 和 `0.005` 差距是给模型的规则。运行时没有拒绝「一次改两个旋钮」。t9 就是在这条规则加上之前，一次改了两个旋钮。
- `real_api_passed` 不因这次规划调用而改写。mock 不是 `real_training_passed`。
- 不能把这一次验证分写成可迁移规律，也不能写成最优 EEG 模型。
