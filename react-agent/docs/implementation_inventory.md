# V1.6 implementation inventory

核对范围是 `/home/zxuff/data/EEGagent/react-agent`。没有为找模型去扫无关私有目录。

## 可复用

| 组件 | 路径 | 用法 |
| --- | --- | --- |
| DeepSeek JSON 调用 | `src/react_agent/fmri/llm/deepseek.py` | 模块 B 另建调用，注入 backend。无 key 时失败，不退回 rule。 |
| 费用空值 | `src/react_agent/fmri/budget.py` | `api_usd` 未知保持 null。B 使用自己的调用计数，不写 A 的账本。 |
| SQLite 写法 | `src/react_agent/fmri/memory/repository.py` | 只复用连接方式。库文件和表与 fMRI memory 分开。 |
| 事件与报告习惯 | `src/react_agent/fmri/loop.py` | campaign 目录写 json/jsonl。不改 A 的 report 字段。 |
| CLI 风格 | `src/react_agent/fmri/cli.py` | argparse 子命令。B 使用 `python -m react_agent.eeg_research.cli`。 |

## EEG 任务与模型

基线检索入口在 `src/react_agent/eeg_training/`。它从 Uncertainty-aware-Blur-Prior 抽出 `EEGProjectLayer`、对称对比损失，以及 EEG/MEG 的预处理读取。图网络、SNN、SAM 和四视图没有带进这个入口。

数据目录默认是 `/home/zxuff/data/Uncertainty-aware-Blur-Prior/data`，里面已有 THINGS-EEG / THINGS-MEG 预处理文件和 RN50 特征缓存。训练解释器默认使用本机 `ubp` 环境，也可以用 `EEG_TRAIN_PYTHON` 覆盖。没有 `EEG_GPU_SECONDS` 时只允许试运行。本轮没有启动训练，`real_training_passed` 仍是未跑。

- `ubp_retrieval`：四种设计，`eeg`/`meg` × `intra-subject`/`inter-subject`。被试内验证集来自训练图像。测试文件不进入 fit。
- `unavailable_retrieval`：仍拒绝，不填分数。
- `mock_retrieval`：只用于契约测试。`availability=unverified`。mock 通过不算 `real_training_passed`。

## 本轮变更

新增 `src/react_agent/eeg_training/`、`adapters/ubp_retrieval.py`、`tests/eeg_training/`。工作台增加「检索实验」页。不改 A 的阈值、六个问题合同、verdict 或历史 runs。
