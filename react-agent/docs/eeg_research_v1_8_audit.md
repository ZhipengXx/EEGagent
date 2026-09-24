# V1.8 审计

仓库：`/home/zxuff/data/EEGagent/react-agent`。没有 `AGENTS.md`。训练解释器：`/home/zxuff/miniconda3/envs/ubp/bin/python`（该环境 `torch.cuda.is_available()` 为真）。数据根：`protocol.py` 的 `DATA_ROOT`，可用 `EEG_DATA_ROOT` 覆盖。

## 划分

`split_plan` 在被试间且选中全部被试时使用 `all_subjects_holdout`。训练集是各被试 `train.pt`，验证是训练图像 10% 留出，全部 `test.pt` 只是禁止进入 fit 的文件。`split_manifest` 把它们标成 `held_out_unused`，不是最终测试。

因此 `eeg_inter_subject_all_s0` 和这次新合同都是 pooled-subject / 未见图像，不是未见被试泛化。新合同写在 `runs/eeg_research_v18/eeg_retrieval_research_v1/evaluation_contract.json`，`research_scope` 为 `pooled_subject_retrieval`，`final_test_enabled` 为 false。指纹 `4fdfbbf22d12ee0f2356084953ddd3ad39ed7b8995b634082f4f2337e8febec8`。

## 训练与评估

现役模型是本仓库 `EEGProjectLayer` 加对称 InfoNCE。选择指标是验证固定候选集 top1。批内 top1 只诊断。根目录批内 `0.3630694088088461` 标为 `legacy_unverified`，不能当固定候选集对照。1654 与 200 两种 gallery 的 top1 不能相减。

`train_entry` 只有在 `EEG_CANDIDATE_MODULE` 存在时才加载候选，并写 `source_binding.json`。`EEG_FINAL_TEST=0` 时不计算测试分。

## 旧动作

`run_adaptive_loop` 仍只允许改学习率和权重衰减。`analyze_retrieval_errors` 与 `propose_experiment` 只写 trace。新循环不使用「最好两次相差 0.005 就停」。旧 `controller.txt` 仍保留这句话，供 1.7 路径使用。

## 未复制 ARIS

没有复制 ARIS 源码。上游仓库 https://github.com/wanshuiyin/auto-claude-code-research-in-sleep 。本次未把某个 commit 检入依赖。
