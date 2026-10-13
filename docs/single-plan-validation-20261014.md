# EEGagent 单任务改造验收记录

实施合同、状态机、预算接口、启用配置、create/start 模板与旧 runtime 只读预览见 [协议说明](single-plan-protocol-20261014.md)。本次使用新协议和临时 CAM 验证，没有恢复原研究、没有真实 GPU 训练。此前用户要求保存的 `1703738` 是实施中快照；本版包含其后的修正和验收。

## 原生执行证据

- 一个被选新方法经历两次真实 CPU shape check 失败（输出维度 3、7）后，在原 plan/candidate/owner/科学意图内修复；通过真实源码检查和对应 revision 的 Reviewer，再完成 matched baseline 两折＋candidate 两折、一次 Analyst、阴性卡及下一轮 K。新增 admission 为 1，未选项执行为 0。
- 筛查后重新比较并明确选择 confirmation，沿原 candidate/source/spec 复用已完成两折，再补齐剩余 folds。两项任务合计 20 个唯一模拟 jobs，确认新增 admission 为 0；卡片分别保留两折和十折覆盖。
- 55 轮使用原生 loop、ContextCompiler、真实 SDK MockTransport 和失败 export fixtures：41 个诊断＋14 个诊断验证任务，8 种机制、55 组不同 source/spec 字节身份、7 次重启、6 次重复结果采用和 5 次有界草稿纠正。模拟主链调用 60、维护 0；不是 55 次真实方法实验。
- 长期诊断按任务 refs/真实 scope 筛选后优先最新相关 source。最后一轮 `offline_export_54` 位于首条，真实归档 source/spec SHA 可核验；55 个原生来源中 19 个在本任务 scope，交付 12、省略 7，partial history 和未知身份如实标记。诊断 annotation 没有被伪装成科学测量。

55 轮最终 Prepared 大小为 40,319–65,436 Unicode chars，实际两段 schema 为 3,607 chars。每轮 calls/tokens/费用 unknown、repair、admission、完成状态、输入组成和重复操作均在 `native_55_round_replay.json`。诊断任务 repair=0；方法任务两次 repair 的证据另存，不把两者混为同一组实验。

## 输入与维护开销

| 项目 | 原版 | 当前 | 证据范围 |
| --- | ---: | ---: | --- |
| Planner 完整 system | 22,027 | 6,535 | 当前 role/schema；不是只量 mission |
| Designer 完整 system | 16,491 | 5,495 | schema 另有分量统计 |
| Coder 完整 system | 15,620 | 2,768 | user 工具契约另 956 |
| 20 个已准入候选的成本表 | 23,468 | 3,921 | 原生批准/准入 fixtures，未训练 |
| 20 个已准入候选的 Prepared | 72,401 | 65,351 | 必要身份/权限/cost facts 保留，原整表可读回 |

Shared mission 832、Planner mission 1242、Designer mission 1135、Coder mission 1133 chars。其他角色 system/user/schema/tools 的实际分量、schema/system SHA、规则到 native validator/conditional hint 的映射在 `runtime_input_sizes.json`。原 16 个 legacy prompt 文件保持原字节。

新协议 `paid_maintenance=off` 时，卡片、source、SQLite/MiniLM 检索仍工作，没有新维护 transport；原已付费有效 proposal 能按原证明以零新增 API 恢复发布。emergency 的准备 child 与 Analyst/Continuation 隔离，batch 只用 unreserved auxiliary。已证明无发送的预算拒绝不消耗本次 generation，paid/unknown 不退款。旧 native pending/unknown/unpriced 与 SQL claims 的观测按 call_id 合并，未重复增加占用。

## 验证命令和结果

环境使用本地 Python 3.12 验证 venv，离线 hooks 拦截网络，CUDA 隐藏、各 CPU 线程上限为 2；SDK HTTP 和训练 transport 使用 fixtures。典型前缀为：

```bash
env PYTHONPATH=/tmp/eegagent-compaction-stage14-20261010-denskuz0/offline_hooks:src \
  EEG_EXTERNAL_API_DISABLED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/eegagent-minilm-validation-venv/bin/python -m pytest -q ...
```

所有命令在 `react-agent` 执行，以下 XML 位于 `version/single_plan_20261013/`。测试组存在重叠，不把通过数相加：

| 实际范围 | 结果 | XML |
| --- | --- | --- |
| 完整 `tests/eeg_research tests/eeg_training` | 1751 passed、69 failed、17 skipped，1935.82s | `final_offline.xml` |
| 同69旧失败节点，隔离 `99aec93` source | 69 全部失败；归一临时路径/随机ID后错误与断言一致 | `baseline_comparison/baseline_failed_nodes.xml` |
| 当前 context＋memory 原兼容组 | 554 passed | `memory_context_compat_final.xml` |
| 最新诊断卡/相邻 suite diagnostics/semantic memory | 181 passed | `memory_card_current_source_fix.xml` |
| 原生 subject scope 卡 | 1 passed | `memory_card_subject_scope_fix.xml` |
| 55回放＋source选择＋阴性/确认/两次repair后测量 | 6 passed，315.86s | `current_diagnostic_and_native_measurement.xml` |
| 最新 source/archive 复核 | 3 passed，73.67s | `diagnostic_latest_source_archive_final.xml` |
| 旧 pending/unknown/public owner 观测及 SDK/legacy 组 | 113 passed | `eegagent-native-legacy-owner-observability-final.xml` |
| JSON 尾兼容与相邻训练诊断 | 29 passed | `legal_terminal_json_final.xml` |
| 最终 planning/coding 入口、JSON、诊断选择、维护、owner/SDK | 63 passed，18.84s | `final_entry_and_admission.xml` |

完整回归中的 69 个失败可在基线同节点重现，不能声称全项目或基线全绿。该宽批早于最后几项诊断/观测修正；后续改动用表中对应受影响组复核。17 个 skipped 没有视为通过。完整原失败 XML、日志和逐项对照保留，没有改断言或削弱科学 gate 来得到绿。

当前 JSON 兼容修正只允许训练诊断读取完整合法但没有尾换行的 JSON 行；非法尾仍被记录为 partial，不生成零指标；UI/账本 reader 默认不变。新科学 manifest 同时冻结其 `jsonl_read.py` 依赖。相关故障修前 XML 与修后 XML 都保留。

## 真实 API 与原产物

真实 DeepSeek 验证限 planning-only、合成诊断输入，两批共11次：首批6次调用覆盖的3个场景均未通过，并暴露缺失 item schema；修正后5次，3场景中2场景产生有效池，1场景有界纠正后拒绝。训练、GPU 和维护调用均为0，没有真实 Coder 能力或 EEG 提升结论。API input/output/reasoning usage 和 unknown USD 详见 `provider_evaluation_summary.json`；reasoning 已包含在 output，未重复计价。

`planning_evaluation.py` 和 `coding_evaluation.py` 都需显式真实 API flag、空新目录与有限 calls/steps/repairs；编码入口只做原生受限工具与 CPU 接口检查，尚未真实调用。

已取消原 CAM 的11个关键文件、旧 runtime 的29个冻结依赖再次逐字节核对通过，status=cancelled、control=stop。当前主源码6个旧科学依赖变化只用于新 runtime/新 CAM；旧 manifest/hash 没有刷新。实际只读预览：冻结 runtime 全匹配，当前主源码6处 mismatch，二者均不授予 resume。

最终必要原生证据按 case 存在 `native_evidence/`，文件字节及原始 ref 索引按 SHA 核验，原产物未改写。逐项 25 条验收在 `acceptance_matrix.json`，文件变更摘要在 `git_diff_summary.json`。源代码和示例配置随本完成版本保存；测试、运行目录与原始验证证据遵循现有 `.gitignore` 策略保留在本地。
