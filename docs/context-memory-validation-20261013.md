# Context/memory 离线验收

验收使用临时 campaign、SQLite 和 registry，并只读真实 fixture。全部模型回复来自 mock HTTP transport 或受控 generator；真实外部 API 调用 0，GPU 训练 0，真实 LLM 摘要质量未验证。原有 CAM 没有被本次测试重启或写入。

## 真实超限样例

fixture 为 `react-agent/runs/eeg_research_v18/eeg_screen_real_20261012_r1/context_diagnostics/ctx_d5d5beb736c8.json`，SHA256 保持 `f7675db17a85eed922d51ab1c602fca109591cab297ae39bc928ff1c9a0404b7`。

旧最终 system/user 实际合计 164617 chars，旧内部报告 164424 少计 193。该样例 `memory=[]`，retrieved memory 4926 chars，不能归因于巨大长期 memory。

新路径使用原生 role 装配、最终 renderer 和真实 SDK 的 mock transport，实际发送的是同一冻结 PreparedRequest。最终输入 **118,966 chars**（system 21,569 + user 97,397），减少 27.7%。它保留必要精确材料，未达到 112000 target；报告明确标记 `required_exact_material` fallback，仍低于 144000 ceiling。独立 provider 策略检查通过，但 token 精度为 estimate。机器结果保存在 `react-agent/version/context_fix_20261013/native_replay_report.json`。

两折事实保留：fold07/10、subject07/10、seed0、two_fold_development、coverage2、required2/full-required10；baseline top1/top5 为 .095/.3275，c1 为 .0125/.0475；paired delta -8.5/-8.0 pp，mean -8.25 pp，threshold .5，not_promoted，statistical_significance not_established。没有升级成十折确认。

原递归舍入会改变 23 处值。新交付中对应 22 个实际需要的值精确保留，全部 23 个原值可以从完整来源恢复；未交付的可选值不伪装成已读。source/ref/hash、scope、时间戳、资源、成本和 gate 通过 native/source 驱动测试检查。

| 语义检查 | 离线结论 |
| --- | --- |
| 对应旧舍入的已交付值 | 22/22 精确；原始来源 23/23 可恢复 |
| 两折范围、coverage、baseline/paired delta | 固定 source 驱动断言通过，无十折/确认升级 |
| 来源、pointer 与 read scope | Ready 验证闭包/hash；delivery summary 不声称 native read receipt |
| 关键失败、pending 和未知必要信息 | 原生事实/状态保留或有限 NeedsContext/Blocked |
| 条件化经验、反例、来源撤销 | Phase A/B 与迁移测试通过 |
| 重复提出已做任务 | 原生 task/输入缓存、unchanged no-op 验证通过；真实模型重复建议率未测 |

## 多角色与增长回放

八个角色分别检查 normal、schema repair 和 HTTP retry；repair 是新编译请求，HTTP retry digest/messages 相同。额外覆盖 summary/extraction/consolidation、原生 task/intent/API 计账、全维护两次/turn 限额、usage 分组校准及 profile 改变后不发送。

100 轮使用不同 candidate、seed、协议、评价结果、错误及来源身份，包含两折/十折、partial/failed、晚到结果、方法 revision、撤销和 repair/retry。报告保存每轮原 observation 大小、实际输入、tokens 精度、digest、API attempts、错误和运行时间。 本次原始历史最大 **771,767 chars**，实际请求最大 **64,868 chars**；100 轮无错误，116 次 mock attempts（110 个 JSON 回复和 6 次 HTTP 错误）全部进入原生账本，新增摘要调用 0。增长测试中的摘要 generator 可以返回 no_signal；这不能用来证明真实模型压缩质量。成功 delta 的覆盖替换和收益由独立服务集成测试验证。

## 故障和恢复

测试覆盖巨大第一条日志/代码、目标函数和中段读回、未知必要错误、seed/evaluator/held-out identity 差异、hash/revision/撤销、数字错误/伪造 ref/空或膨胀摘要、failed/partial 和 pending、CAS 期间新事件/资源变化、draft/artifact/DB 崩溃、重复提交、lease 到期/restart、registry 页预算及 hash mismatch、pointer 闭包/环和数组重排、巨型 system/schema、小 provider 窗口、费用耗尽/退避、no_signal、来源支持撤销、向量失效及幂等迁移。

旧单独 `prepare_context` 触发 Curator 摘要的测试迁到统一出口；旧 helper 的直接离线兼容测试仍保留。已有检索测试更新为最终 8000 chars cap、显式 tokenizer 分块和 query vector restart 复用。没有以提高预算或恢复全局浮点舍入使测试通过。

## 最终测试结果

| 批次 | 通过 | 时间 |
| --- | ---: | ---: |
| combined-final | 638 | 176.00s |
| all-final | 209 | 123.30s |

按测试身份去重，共 **694 项通过，0 项失败**。组合批次与完整 context 批次存在重叠，不能直接相加。完整 context 批次包含最终代码的 100 轮回放；另存核心、维护恢复和服务集成 XML。唯一提示是 SentenceTransformers 接口改名的 FutureWarning，没有测试失败。

机器汇总为 `react-agent/version/context_fix_20261013/validation_summary.json`。

## 复现与产物

从 `react-agent/` 运行，使用项目依赖完整的 Python 3 环境。本机验证解释器为 `/tmp/eegagent-minilm-validation-venv/bin/python`。离线 hooks 阻止 socket 和 dotenv 外部配置；`EEG_EXTERNAL_API_DISABLED=0` 只让 mock 原生适配器进入计账路径。

```bash
env PYTHONPATH=/tmp/eegagent-compaction-stage14-20261010-denskuz0/offline_hooks:src \
  EEG_EXTERNAL_API_DISABLED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/eegagent-minilm-validation-venv/bin/python -m pytest -q \
  tests/eeg_research/context --junitxml=/tmp/eegagent-context-all-final.xml
```

协议模块集中结果、原生回放及兼容测试 XML 和最终机器汇总已保存到 `react-agent/version/context_fix_20261013/`。源码编译和 `git diff --check` 同时检查。最终汇总不宣称整个仓库所有历史测试已全量通过。

任务开始时保留 `initial_status.txt`、`initial_tracked.patch`、源码 hash 和 `initial_local_source.tar.gz`；完整旧 prompt 保存到同目录。最终实现补丁、修改文件清单、规则映射及调用点扫描也在该目录。`version/` 和 tests 在仓库中被忽略，需随交付备份一起保留；本次没有 commit、push 或 PR。

## 实际验证边界

mock usage、价格和 elapsed 仅解释离线账本行为，不是实际 DeepSeek 费用或真实研究质量。provider token 计数在没有官方 tokenizer 时仍为标记明确的保守估计。真实 LLM summary 的遗漏/漂移率和实际 latency 尚未评估；本次没有 GPU 或线上 CAM 验证。后续真实实验仍需遵守原预算、权限、API 未知费用恢复和训练审批 gate。

100 轮记录的 2200 input tokens、880 output tokens 和 mock 已知费用小计是 fixture 注入 usage，仅用于费用恢复断言；不能据此估算真实调用价格。报告同时保留每个请求的模型 token 估计及精度标签。真实授权额度在本轮离线测试中没有消耗。
