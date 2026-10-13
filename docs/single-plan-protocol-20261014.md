# EEGagent 单任务研究协议与验证（2026-10-14）

本次实现以 `99aec93` 为检查基线。`1703738` 是此前按用户要求推送的实施中快照；本文记录后续完成的修正与验收。本协议用新 goal 的 `research_execution.version: single_plan_v1` 启用，旧研究不会自动迁移。

## 任务和反馈

`ResearchRoundDraft` 是一次 Planner 请求生成的紧凑草稿：2–4 个竞争任务、独立排序、选中项及理由。每项说明问题、科学意图、做法、评价范围、预期 observation、完成条件和成本依据。发送的 schema 展示候选和评价的完整字段；接收端仍逐项校验，单个坏项不会使正确项失效。至多一次有预算的纠正；不足 K 如实记录，单项继续必须显式开启 `allow_degraded_singleton`。

完整权威 observation 用于原生校验，卡片 view 用于模型交付。程序计算已知 jobs/folds/对照复用成本；实现、API 和耗时预测独立保留为预测或 unknown。首选合法且可承受时保持模型选择；首选不可行时依据有效排序 fallback，并记录原因。只有选中项分配资源，未选项不准入、不编码、不训练。

`SelectedPlan` 冻结 originating round/proposal、科学意图、candidate/spec 绑定、budget owner 和 recovery limits；内部 action 以 `origin_plan_id` 和追加 receipts 关联。主要状态为 selected → preparing → implementing/checking/repairing → executing_measurement → collecting_observation → observation_ready，另有 paused/blocked/failed/cancelled/needs_replan。

一个新方法任务包含必需 matched baseline、事先声明的 folds、有限工程修复和一次必要 Analyst。baseline 原生事实不再被伪装为第二份候选 Analyst 解释。工程修复保留 plan/candidate/owner/科学意图；核心干预变化要求重新规划。工程失败生成独立卡，不能解释为科学负结果。paused/blocked 保持可恢复状态，不宣称测量完成。

筛查通过只产生 confirmation eligibility。确认或额外消融必须进入下一轮比较并被选中；确认复用原候选和严格验证的已完成 folds，不自动创建新方法。旧版 action/auto-confirm 路径继续按旧协议读取和恢复。

## 预算和真实发送

`owner_budget.py` 用持久 SQLite 事务和原生 journal 管理 Campaign → SelectedPlan → role/phase/必要准备子分配，同时保护一次下一轮 Planner。parent 和 child 不重复增加全局授权。Auxiliary 只使用未被活跃主链占用的余额。

每个真实 HTTP 出口执行 claim → durable intent → inflight → actual usage settlement；共享 DeepSeek backend 也检查直连、重试、模型升级、pause/stop 和 unknown。Calls、input/output/total tokens、USD、training jobs、GPU-seconds 分别比较。已有 intent/cost 按 call_id 去重；旧 spent 保留，unknown 不因超时、lease 过期或任务失败而退款。

实际 usage 证明的未用差额可以释放；明确发送前失败允许恢复该未发送 claim。已付费回复和已完成任务 adoption 不新增 API。Continuation 绑定 terminal observation digest，只能用于该反馈的一次有限规划，不能反复借用。训练和工程多卡检查共享相应 owner 与全局 GPU 边界，实际超额如实记账并阻止后续入场。

默认的新方法 API 封套按有限工具步骤与最多两次修复计算，是可配置的上界，不能当作预计每个任务必花的调用数。诊断、确认和验证使用各自必要阶段。显式 token/USD cap 才构成相应硬限制；没有可靠价格时费用为 unpriced/unknown，不能把抽象维护单位或 reasoning tokens 再加为美元。

## Memory 和 prompt

原生 `ExperimentObservation` 保存真实身份、scope/seed/folds、精确 paired metrics/delta、缺失及 promotion 状态；已有 Analyst artifact 独立绑定为解释。原文和来源继续存储，SQLite/MiniLM 增加 `experiment_observation` 类型，短查询检索最近与相关卡，并受数量、字符和 token cap 约束。source/record 更新或撤销使相应卡和向量失效。

Planner 使用当前问题、小型权限/资源/合法目标 kernel、必要近期及相关卡和可读历史导航。交付 view 不替代原生科学校验。Coder/Reviewer 保留真实源码、接口、检查和 actual-read 义务。未知必要控制不能静默丢失。

`paid_maintenance` 独立于 `memory.enabled`：off 默认不发新的付费维护，capture/卡片/索引/检索继续；已有付费 proposal 只在原任务、来源、CAS 和回复证明有效时以零新增 API 恢复发布。emergency 仅用本角色准备子预算，必要材料已经可交付时不额外压缩。batch 只用 auxiliary，未变化 no-op。预算拒绝且可证明没有发送时标 deferred；付费失败和 unknown 不重置费用或失败额度。可选维护不能阻塞新协议反馈闭环。

新版 mission 位于 `prompts/runtime_v1/`。旧 16 个 prompt 文件保留原字节。InfoNCE、tied CE、objective-off、pooling、capacity、checkpoint 等只按当前任务条件加载；规则映射由 `runtime_prompts.prompt_rule_mapping()` 生成。mission、完整 system、schema、tools、user 分别统计，不能用 mission 大小替代整个请求大小。新 schema/system/model/config 改变生成新 identity，旧 paid proof 不被新 system 静默覆盖。

## 配置和兼容

可验证的完整示例位于 `react-agent/examples/single_plan_goal.example.json`。示例预算不是新的运行授权；`validate-goal` 不创建 worker：

```bash
PYTHONPATH=src python -m react_agent.eeg_research.agentic.cli validate-goal \
  --goal examples/single_plan_goal.example.json
```

以上命令在 `react-agent` 目录执行，使用装有项目依赖的 Python。LOSO 新 CAM 创建仍需要原生 verified budget history 和明确 allocation；多卡、数据、fold selection、baseline 及科学 gate 继续由原生 manifest 冻结。不能通过修改旧 manifest/hash 把新代码强行放进旧科学运行。

下面是未来另获研究运行授权时的新 CAM 命令模板，本次没有执行。先将 `/path/to/` 路径、研究名称、唯一 request-id 和 GPU 编号换成真实配置。history source 必须是原生可验证、已静止、没有 unknown API/未结算进程的历史研究；`--fresh-budget-authorization` 明确创建 goal 中的一份新额度，历史费用作为估计依据保留，不把旧额度自动恢复为新授权。

```bash
PYTHONPATH=src python -m react_agent.eeg_research.agentic.cli create \
  --root /path/to/new_research_root --campaign new_single_plan_research \
  --request-id unique_new_research_authorization_id \
  --goal examples/single_plan_goal.example.json \
  --fresh-budget-authorization --budget-history-source /path/to/quiescent_native_history \
  --data-root /path/to/eeg_data --subject all --gpu 0,1

PYTHONPATH=src python -m react_agent.eeg_research.agentic.cli start \
  --root /path/to/new_research_root --campaign new_single_plan_research --max-ticks 200
```

GPU 编号应按实际可用多卡选择；共享已占用卡需要明确的 `--allow-shared-gpus` 授权与原生资源检查。也可用 `--inherit-study /path/to/quiescent_native_history` 继承原生剩余授权，不能同时使用 fresh allocation。增加预算必须使用已有明确授权入口，不能只改 counter。`start` 会启动真实 worker，和 `validate-goal` 的只读校验不同。

只读保护检查确认已取消 CAM 的 11 项关键文件和旧 runtime 的 29 项依赖保持原字节，control=stop、status=cancelled。当前主源码有 6 个旧科学依赖发生变化：diagnostics、jobs、method_suite、multigpu_check、screening、budget。它们只适用于新 runtime/新 CAM；旧冻结 runtime 没有部署更新。

新建 manifest 另捕获诊断使用的 `jsonl_read.py` 依赖。训练诊断支持完整合法 JSON 末行没有换行的历史格式，非法 partial tail 仍标记缺失；UI/账本 reader 默认尾部规则保持原语义。相关训练/诊断 29 项验证通过。

实际只读兼容预览保存在 `historical_runtime_preview.json`：原冻结 runtime 的 29 文件全部匹配；当前主源码有 6 处 mismatch。两种预览都返回 `native_resume_authorized=false`。下面只预览旧冻结副本，不恢复已取消 CAM：

```bash
PYTHONPATH=src python -m react_agent.eeg_research.agentic.cli historical-runtime-preview \
  --root runs/eeg_research_v18 --campaign eeg_context_real_20261013_r1 \
  --runtime-checkout version/real_eeg_context_20261013_r1/runtime_project \
  --original-project-root version/real_eeg_context_20261013_r1/runtime_project
```

## 当前全量回归记录

完整 `tests/eeg_research tests/eeg_training` 回归实际执行 1837 项：1751 passed、69 failed、17 skipped，用时 1935.82 秒。隔离 `99aec93` 源码对同 69 个未改旧用例逐项复核：69 项全部失败，归一临时路径与随机 ID 后错误与断言一致。这证明这些本次观察到的失败可在基线重现；不代表整个项目或基线测试全绿。

完整原结果在 `final_offline.xml`，对照原始日志、XML 和逐项 comparison 在 `baseline_comparison/`。后续最新诊断 source 选择与归档修正已通过独立的6项原生回归及最终3项复核；这份宽回归早于这些修正，最终源码的验证范围见验收报告。

## 有界真实 planning 验证

`planning_evaluation.py` 提供显式 `--real-api` 入口。每个独立场景创建自己的隔离 CAM/bootstrap owner；不派发 SelectedPlan，不准入候选、不训练、不使用 GPU，维护 off。整个入口最多 12 次调用，默认 6，每场景最多两次，目录非空时拒绝覆盖。

```bash
PYTHONPATH=src python -m react_agent.eeg_research.agentic.planning_evaluation \
  --root version/single_plan_20261013/new_planning_evaluation \
  --real-api --max-api-calls 6
```

入口使用既有环境或项目 `.env` 的 API 配置。当前已实际使用 DeepSeek 做两批合成诊断 planning，共 **11 次 API**、维护 0、训练/GPU 0。首批 6 次暴露 provider item schema 缺失，3 个场景均未通过；修正后 5 次中 2/3 场景生成有效竞争池，另一个在有界纠正后仍拒绝。原始请求/回复与失败记录全部留存，不通过改中间产物得到通过结论。

首批 input/output/reasoning 为 32,338 / 63,775 / 43,036 tokens；修后为 25,438 / 39,805 / 26,247。Reasoning 已包含在 output，不能重复计费。没有配置价格来源，USD 仍为 unknown。样本较小，且输入明确为合成诊断，不能据此宣称 EEG 性能或编码能力提升。

另提供 `coding_evaluation.py`，复用原生 NativePatch 工具、源码和 CPU 接口检查，默认不执行真实 API；显式调用时总请求最多 16、工具步骤最多 8、repair 最多 2，训练/GPU 为 0。不创建科学 Approval/SelectedPlan/suite，也不宣称科学测量。该入口只经过离线 SDK 验证，真实 Coder 未调用：

```bash
PYTHONPATH=src python -m react_agent.eeg_research.agentic.coding_evaluation \
  --root version/single_plan_20261013/new_coding_evaluation \
  --task 'Implement the declared encoder interface and check its output shape.' \
  --real-api --max-api-calls 8 --max-steps 4 --max-repairs 1
```

完整证据、逐项验收和最终测试结果见 `react-agent/version/single_plan_20261013/` 及后续验证报告。参考 [Thinking Before Thinking](https://arxiv.org/pdf/2609.38147) §3.3、Appendix B 的提案、比较、派发及反馈结构；论文允许并行 recommendations/workers，本项目明确只选一个任务，未固定复制成四次付费调用。
