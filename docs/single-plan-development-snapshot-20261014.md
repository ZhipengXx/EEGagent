# Single-plan 开发快照（2026-10-14）

本次按用户要求保存当前代码。基于 `99aec93295c51f7c97240a83d998c60a4d68042b`，属于实施中快照，不能视为完整验收通过或真实研究性能提升。此次保存没有启动真实 API、训练或 GPU 作业，也没有恢复已取消的研究。

## 已落盘的实现

- 新任务协议：Planner 提出多个研究任务，选中一个 SelectedPlan；内部设计、实现、有限修复及反馈保持任务身份，完成后再进入外层规划。
- Owner 预算：请求发送前分配和记账，角色、测量、工程 GPU 检查及下一轮规划有独立额度；接入共享 provider 出口和原生 job/probe 路径。
- ExperimentObservation：原生事实生成卡片，SQLite 检索和来源读取支持小型 Planner 输入；工程失败与科学测量分开记录。
- 新版短 mission 与条件 case；新协议默认关闭付费维护，已有付费结果仅在原任务、来源和发布条件可证明时恢复发布。

新行为通过 `goal.research_execution.version = "single_plan_v1"` 显式启用。旧版 16 个 prompt 文件与基线逐字节一致；此快照不自动迁移旧 CAM 或更新旧运行环境。

## 验证与限制

本次快照的 29 个 Python 文件语法解析和 `git diff --check` 已通过。提交前对最终工作区运行以下离线测试：**69 项通过，1 项失败，用时 66.14 秒**。本地报告为 `/tmp/eegagent-git-snapshot-20261014.xml`。

```bash
env PYTHONPATH=/tmp/eegagent-compaction-stage14-20261010-denskuz0/offline_hooks:src \
  EEG_EXTERNAL_API_DISABLED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/eegagent-minilm-validation-venv/bin/python -m pytest -q \
  tests/eeg_research/test_selected_plan_execution.py \
  tests/eeg_research/test_owner_budget.py \
  tests/eeg_research/context/test_owner_role_send.py \
  tests/eeg_research/test_experiment_observations_v1.py \
  tests/eeg_research/test_multigpu_check_lifecycle.py \
  --junitxml=/tmp/eegagent-git-snapshot-20261014.xml
```

以上命令在 `react-agent` 目录执行，使用离线网络拦截、MockTransport 和模拟训练进程；不能代表真实模型或 GPU 实验结果。测试和本地报告仍遵循现有 `.gitignore` 策略保留在工作区。

尚未完成：

- 确认阶段下一轮显式选择后的完整 dispatch/reuse 集成验收，以及 50 轮完整回放。
- 科学任务预算不足及恢复的全部集成边界。
- `planning_evaluation.py` 是未验收的 planning-only 入口草稿；owner 上下文与跨场景生命周期尚需校验，不应直接用作真实 API 验证入口。
- 最终全项目回归与真实 planning 验证。

本次失败项为 `test_native_tick_recovers_successful_settlement_without_new_probe`。其 fixture 缺少 `method_search_manifest.json`，原生恢复路径返回 `method_recovery_invalid`；尚未完成处理。不得将更早批次的通过数当作当前全量验证结果。

Git 快照包含源代码、runtime_v1 prompt 和这份说明；密钥、运行目录、模型缓存与无关 `git_file/` 目录不纳入提交。
