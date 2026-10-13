# EEG 模型出口与旧规则映射

实际调用点机器清单保存在 `react-agent/version/context_fix_20261013/callsite_inventory.json`。扫描覆盖 EEG controller、原生 `agentic/llm.py`、ContextService 和共享 DeepSeek backend；仅共享 backend 的 `_complete` 调用 SDK。

| 入口 | 编译与计账 |
| --- | --- |
| research_planner | `role_backend` → ContextService → Ready → `_send_compiled`；原生 operation/intent/API ledger |
| experiment_designer | 同一出口；现有 task envelope 和设计 validator 保留 |
| candidate_coder | 同一出口；代码读取义务、task/source binding 和审核 gate 保留 |
| candidate_reviewer | 同一出口；实际源码/checks 与 read receipt 独立验证 |
| result_analyst | 同一出口；评价身份、scope、coverage 和原生开发证据保留 |
| memory_curator | 同一出口；输出仍是现有条件化建议，不是研究证据或批准 |
| result_auditor | 同一出口；原生报告 hash、依赖和审计规则保留 |
| research_librarian | 同一出口；过滤后的检索有界交付 |
| 工作摘要 / episode 提取 / consolidation | `_ledgered_memory_maintenance` → native proposal recovery/cache → `_invoke_compaction` → 同一 compiler；禁止语义维护再入；原生 task/intent/API ledger |
| schema repair | 当前错误与相关回复形成新 CompileRequest；通过同一预算检查 |
| HTTP 重试 | 同一冻结 PreparedRequest，保留已有有限重试和未知 outcome barrier |
| 保留的 handoff helper | 自身确定性/兼容接口保留；原生调用 `_invoke_compaction` 也经过 prepared 出口 |
| 旧 ResearchController | `request_deepseek` 创建 campaign 内隔离 delivery store，Ready 后发送；其原有业务和费用流程保留 |

EEG backend 显式开启 `enable_context_protocol()`。开启后直接 `complete_json` 不传 PreparedRequest 会在 SDK 前拒绝；传入冻结内容必须符合 digest、messages、预算和 profile。fMRI 未开启该要求，现有调用兼容。测试注入的 lightweight backend 可以接收完全相同的 measured system/user；这不提供生产绕过入口。

`backend.prepare_context` 保留业务 payload，供 native task/input digest 冻结；最终交付在真实 transport 边界编译。它不再单独付费调用旧 Curator flat-schema 摘要器。外部禁用、pause/stop、未结算 API、调用上限仍由原生预算检查阻止发送。

独立候选池继续使用 raw `PlannerProposal.options` 加 `INDEPENDENT_OPTION_SCHEMA`。provider schema 只去 JSON-schema 节点的 title/description 等注释，不删除 business property/default/enum/const；runtime `PlannerOption` 和每个候选独立校验保持原样。schema repair 可恢复有效 plan，一个坏 sibling 不能使其他有效候选全部失效。

旧 shared/planner prompt 按语义重写，完整旧文本另存。映射文件 `react-agent/version/context_fix_20261013/prompt_rule_mapping.json` 包含 43 个旧规则组及 130 个 native validator/dispatch 位置，记录旧文本 hash、新规则位置和实际代码 hash。模型仍需遵守解释性政策；native validator 的存在不代表模型 prose 已被自动证明正确。

启用前后均保留 screen_then_confirm：两折完成只说明两折开发筛查完成，`fidelity=full` 仍不是十折覆盖；筛查不通过不自动启动确认。Context summary、delivery source handle 或检索相似度均不能满足 development evidence、execution approval 或真实 artifact read gate。
