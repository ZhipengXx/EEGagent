# EEG auto-research v1.6

模块 B 是独立的实验闭环。它不改 fMRI 筛查的 verdict、screening_decision 或 `claim_scope`。

## 状态

| 标记 | 状态 |
| --- | --- |
| implemented | 协议、注册表、mock 闭环、基线 EEG/MEG 训练入口、独立 SQLite、CLI、检索实验页 |
| mock_passed | `tests/eeg_research/test_gates.py`、`tests/eeg_training/test_protocol.py` |
| real_api_passed | 未跑。没有把真实 DeepSeek 轨迹写成通过。 |
| real_training_passed | 未跑。入口已接上，本轮没有真实训练日志。 |
| multi_trial_real_passed | 未跑。`bounded_research` 默认关闭。 |

基线入口是 `python -m react_agent.eeg_training.cli`。可选 EEG / MEG，以及被试内 / 被试间。被试内从训练图像划出验证集，测试文件不进入 fit。预处理数据和 RN50 缓存位于 Uncertainty-aware-Blur-Prior 的 data 目录。解释器默认是本机 `ubp` 环境。没有 `EEG_GPU_SECONDS` 时，`run` 拒绝启动，不写 `metrics.json`。`unavailable_retrieval` 仍表示没有批准的真实分数。

## 命令

在仓库根目录：

```bash
uv run python -m react_agent.eeg_research.cli probe --config configs/eeg_research_v1_6.yaml --out runs/eeg_research/probe
uv run python -m react_agent.eeg_research.cli plan --dry-run --config configs/eeg_research_v1_6.yaml --out runs/eeg_research/plan
uv run python -m react_agent.eeg_research.cli run --config configs/eeg_research_v1_6.yaml --out runs/eeg_research/mock_once
uv run python -m react_agent.eeg_research.cli inspect --out runs/eeg_research/mock_once
uv run python -m react_agent.eeg_research.cli resume --config configs/eeg_research_v1_6.yaml --out runs/eeg_research/mock_once
uv run python -m react_agent.eeg_training.cli probe --dataset eeg --exp-setting intra-subject --subject sub-01 --out runs/eeg_research/eeg_probe
uv run python -m react_agent.eeg_training.cli run --dry-run --dataset meg --exp-setting inter-subject --subject sub-01 --out runs/eeg_research/meg_probe
```

`probe` 和 `plan` 是 dry-run，不把 mock 分数当成训练结果。`run` 在当前配置里只跑 mock profile，`max_trials=1`。`api_usd` 保持 null。

工作台 `GET /api/research?campaign=` 只读 `runs/eeg_research/` 下的 campaign。成绩不进入 fMRI 概览卡片。

## 还不能做

- 不能声明选出了最优 EEG 模型。mock 的 availability 是 unverified。
- 不能把一次 validation 变化写成可迁移规律。
- 不能用 fMRI 数值通过给 EEG 样本加权。那要另立对照实验，本轮不跑。
