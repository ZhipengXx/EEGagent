# EEGagent canonical 研究闭环增量验收

基线：`e9f9165245cc666a9d201b9727faf32a45334f09`。  
这不是“找到了最好的 EEG 模型”的声明。一次 CPU toy 分数变化不能写成最优模型。`real_training_passed` 不得由 mock 或手写指标标绿。

离线套件（不含 live API）：`tests/eeg_research tests/eeg_training --ignore=tests/eeg_research/test_live_roles.py` → **253 passed**（含 T01–T14 与 Phase 7 CPU 闭环）。

命令：

```text
cd react-agent && .venv/bin/pytest tests/eeg_research tests/eeg_training --ignore=tests/eeg_research/test_live_roles.py -q
```

## Schema / migration

- `ApprovalRecord` 不再是 boolean。新记录为 `eeg_research.approval_record.v1` 对象，绑定 `spec_hash`、`context_hash`、`validator_version`、producer task/attempt、artifact ref、capabilities、parent/control、policy version。
- 旧 `approval_record: true` 只读、不能作为执行授权；不自动升级历史 runs。
- 配置别名 `model_config` / `objective_config` / `transform_config` 只在 `canonicalize_experiment_spec` 入口迁移；两套字段冲突则拒绝。hash 在 canonical 化之后计算。
- `FrozenRunSpec`：`eeg_research.frozen_run_spec.v1`。`intervention_config_hash` 不含 seed；`run_config_hash` 含 seed/fidelity。`spec_hash` 保持非空。
- `ConfirmationPolicy`：`eeg_research.confirmation_policy.v1`，在 `create_campaign` 时冻结。
- `candidate_binding.v2`、`hook_config.v2`、`candidate_pack.v2`（含 content hashes）。
- LessonProposal 别名只做显式 `canonicalize_lesson_proposal` 迁移；`requested_evidence_level` 与 runtime `evidence_level` 分存。
- 诊断包增加 `hook_consumption`：对比 `hook_config.json` 与训练入口写入的 `hook_consumed.json`。`execution_status=not_applied` 表示批准配置未到达 hook。
- `selected_checkpoint.json` 记录 `best_epoch` / `last_epoch` / 两端分数。诊断从 `last.ckpt`（最佳轮）重载，不用最后一轮内存 embedding。
- 未知旧记录只读/未验证，不自动升级。未扩大模型族/损失/预处理 capability registry。

## 验收矩阵

| 项 | 状态 | 说明 |
| --- | --- | --- |
| Helper（canonical / resolver / policy） | offline_passed | T01–T14 经 action/train/settlement/export/memory/CLI 消费者触发。 |
| 消费者 CPU | offline_passed | 公共 campaign/worker/job：baseline + `drop_proj` candidate，真实 AdamW、fixed-bank、checkpoint、diagnostics、comparison。指标由模型计算。 |
| 恢复 | offline_passed | T06 journal/state 间隙；Phase 7 在真实 job 产物上清空外层 evidence/memory 后 `rebuild_campaign_projection`，成本/episode 不重复。未对运行中 GPU 子进程做 SIGKILL。 |
| 导出 | offline_passed | 删除原 campaign 后，不同 cwd、清空 `EEG_CANDIDATE_*`，`evaluate-pack --execute` 在 CPU 上重建评价成功；缺 source 明确失败，不回退 baseline。 |
| live roles | real_api_not_run | 本轮未再跑 `test_live_roles.py`。 |
| GPU | real_training_not_run | 未启动新的 GPU campaign。生产默认仍是 CUDA。 |
| confirmation | offline_passed | 预声明 pairs 可 `confirmed`；缺 full/valid/未声明 seed 保持 provisional。配对确认的真实 GPU 重复 **not_run**。 |

诊断分流（offline agent-loop，不是 live LM reasoning）：配置未到达 hook → `repair_candidate`；执行有效但机制未区分 → `replicate`/`run_full`/`collect_diagnostics`，不默认改学习率。

## T01–T14

| ID | 消费者 | 离线 | live/API/GPU | 剩余缺口 |
| --- | --- | --- | --- | --- |
| T01 | canonicalize + `write_hook_config` + 真实 `fit`/`hook_consumed` | passed | not_run | 无 |
| T02 | implement `tick` | passed | not_run | 无 |
| T03 | repair resolver | passed | not_run | reviewer 重绑的 live 路径未跑 |
| T04 | train resolver + `resolve_run_context` | passed | not_run | 无 |
| T05 | `design_experiment` + designer | passed | not_run | 无 |
| T06 | `_record_job` / `rebuild_campaign_projection` | passed | not_run | 运行中训练子进程 SIGKILL **not_run** |
| T07 | settlement replay / hash conflict | passed | not_run | 无 |
| T08 | `promotion_decision` + policy | passed | not_run | 注册 ConfirmationRecord 的 GPU 对未跑 |
| T09 | `refresh_audit_freshness` | passed | not_run | 无 |
| T10 | `EpisodeStore.accept_lessons` | passed | not_run | 无 |
| T11 | pack hash + 搬迁后 `evaluate-pack --execute` | passed | not_run | 无 |
| T12 | `score_query` / diagnostics | passed | not_run | 无 |
| T13 | 真实 `fit()` 最佳轮≠最后轮，报告/ckpt/诊断一致 | passed | not_run | 无 |
| T14 | CLI `create` + policy/planner/executor seed | passed | not_run | 生产 THINGS 数据根上的 CLI create **not_run** |

## 角色 prompts

`shared_contract.txt` 只维护一份身份/范围/权限合同。各 role 去掉重复通用尾段，补入阶段 5 指定的专属段落。`research_librarian` 区分方法证据与 registered capability。

## 保留

fMRI 模块 A / `numeric_consistency_only`；空 `evidence_confidence`/`training_weight`；未知 `api_usd=null`；冻结 evaluation identities；baseline 隔离；训练目标与本地验证分离；参数去重；Studio 注册；旧兼容只读。B 的研究结果不进入 A 的 verdict。

## 未做 / 不标绿

- 未恢复 `eeg_ui_44f7c763011b` / 未自动续跑 UI campaign
- 未启动 GPU 训练，未把一次分数改善写成最优模型
- 未跑 live DeepSeek
- 未扩大模型族/损失/预处理 capability registry
- 未对每个阶段边界做运行中子进程 kill；恢复验收是真实 job 落盘后的外层 state 间隙重建
