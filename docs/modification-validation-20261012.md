# 2026-10-12 修改与验证记录

实施依据：`react-agent/docs/EEGagent_Codex_Modification_List_20261012 (1).md`。起始提交为 `b1e60233362d34798cb799a8e68033a6299fb6d3`。修改留在工作区，没有启动真实 API、GPU 训练或修改既有 campaign 的产物。

## 交付内容

| 范围 | 实际实现与入口 | 验证文件 |
| --- | --- | --- |
| Planner 与执行身份 | 逐项严格校验、独立行动 payload、有效池、有限修复、确定性选择、原始/规范化双 hash、执行前重检 | `test_independent_action_planning.py` |
| 候选生命周期 | 审批时占用名额、跨 campaign 继承、失败不退款、原生 detach、进程/API/probe 对账、exhaustion receipt | `test_independent_action_planning.py`、`test_method_budget_handoff.py` |
| Coding | 语义错误指纹、候选总步骤/检查额度、一次 Pro 升级及崩溃幂等恢复、CPU/GPU 分流、延迟检查 | `test_coding_recovery_v2.py`、`test_coder_reliability.py` |
| API | 八角色独立配置与真实 SDK body、thinking/effort、最终 content/stop、墙钟 deadline、持久实际请求上限、unknown 费用处置 | `test_role_api_config.py`、`test_role_context_recovery.py`、`test_api_intent_recovery.py` |
| 实验阶段 | 固定两被试、每折九训一评、同一完整十折 suite 的 2+8、两侧预算预留、launch 前实时对账和预算/控制重检 | `test_screen_then_confirm.py`、`test_phase_launch_reconciliation.py` |
| 消费者与记忆 | scoped 两折证据、原生分析/curation/Planner/Auditor、工程经验隔离、十折推翻筛查后的追加来源关系 | `test_screening_consumers.py`、`test_engineering_memory.py`、`test_method_memory_consumers.py` |
| UI 与运维 | 有效池/拒绝原因、实际角色模型、阶段进度/成本/复用折、配置预览、实际 campaign 缓存 dry-run、旧 runtime SHA 预览 | `test_agentic_view.py`、上述新协议集成与 CLI 测试 |

所有测试路径相对 `react-agent/tests/eeg_research/`。完整新 goal：[`eeg_screen_then_confirm.yaml`](../react-agent/configs/goals/eeg_screen_then_confirm.yaml)。使用命令、模型优先级、暂停/恢复、历史兼容边界见 [执行指南](screen-then-confirm.md)；详细修改文件和验收分组见 [JSON 记录](modification-validation-20261012.json)。

## 已发现并修复的集成问题

- 筛查和完整确认共用 episode 去重 ID，吞掉十折记忆；现保留同一科学 suite 身份，分别绑定 episode 阶段，追加更新/矛盾来源。
- 两折证据被 observation 过滤，Planner 无法引用；现保留严格验证的筛查证据及完整 scope。
- 升级后崩溃可能丢失 attempt/operation；现幂等完成升级意图并继承原步骤与请求额度。
- 最后工具步骤遇到 GPU busy 后无法完成既有检查；现完成原检查，不新增模型请求或源码修改。
- 可选 Curator timeout 可能让确认训练继续；现原生对账 unknown intent，明确等待处置。
- 已准入 receipt 未按预算变动复检；现重新检查整份剩余候选/对照工作及 GPU/API 预留。
- 角色恢复 config 的旧字符串导致未处理异常；现安全拒绝缺模型配置证明的旧缓存。
- 预算投影丢失 fold 身份或改变核心数值；现精确保留并共享重复字段。

## 验证状态

最终相关验收覆盖 **577 个唯一测试案例，均有通过证据**。大批次 568 passed / 1 failed 的末项是测试收集后仍运行旧断言；固定当前版本单独重跑该完整原生案例通过（247.81 秒）。另外补充 gradient 和八项兼容差异回归均通过。原始 XML 和精确各批次计数保存在 JSON 记录，不将相互重叠的批次简单相加。

初次广泛回归为 1218 passed / 55 failed / 17 skipped。对这 55 项使用固定起始提交和隔离测试目录比较，47 项在旧提交上也失败；其余配置/契约/对账差异已逐项处理。旧失败主要使用不完整审批、source binding、manifest 或停止条件的历史 fixture；没有降低真实执行门槛以让这些 fixture 通过。

离线测试使用 socket 禁止联网的 hook、删除真实 key、隐藏 CUDA，并限制 CPU 线程。真实 CPU 小模块检查覆盖 import、shape、gradient、前向/反向及 checkpoint 重建。HTTP body 捕获使用 SDK MockTransport，原生角色链使用 mocked provider；它们不构成 DeepSeek 实测、真实 EEG 收益或 Pro 的效果优势。

现有 ignore 规则使测试和验证输出留在本地；本任务没有执行 Git 提交或推送。

## 兼容与限制

旧 full-only 证据不改标签、不降为两折，不覆盖其 frozen hash。新协议需正常创建并原生继承真实资源和已用科研名额。历史 source manifest 含绝对路径，旧 checkout 换到其他路径并不等于可恢复；提供只读 SHA 匹配预览和原路径容器映射说明，恢复仍需要旧 runtime 的数据/环境/进程检查。

跨旧协议的结果 adopter 未自动启用。缺少完整训练、评价、来源及费用证明时拒绝导入；同一新协议内的有效两折复用已验证。两折属于固定 gallery 的开发反馈；十折确认仍保留筛选偏差，不能宣称独立 final test、多 seed 复现或统计显著性。
