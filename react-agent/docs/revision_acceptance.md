# EEGagent e2a13a2 增量修订验收

本文件记录对 `EEGagent_e2a13a2_Audit_and_Cursor_Delta_Prompt.md` 第五部分的落地结果。  
这不是“找到了最好的 EEG 模型”的声明。`real_training_passed` 不得由 mock 或手写指标标绿。

离线套件（不含 live API）：`tests/eeg_research tests/eeg_training --ignore=tests/eeg_research/test_live_roles.py` → **231 passed**。

## 交付矩阵

| 项 | 状态 | 说明 |
| --- | --- | --- |
| P0-1 批准绑定 / `resolve_approved_experiment` | implemented / offline_passed | 注册 artifact 为真源；state 仅缓存。自报 `status=approved` 不能 implement/train。 |
| P0-1 context 中 disabled hook | implemented / offline_passed | 不再把 `KNOWN_HOOKS` 与 disabled 项并集视为可用。 |
| P0-2 去掉 `already_started` 目录豁免 | implemented / offline_passed | 恢复只复用已批准 spec；夹具改为 `approve_experiment()`。 |
| P0-1/D03 baseline 配置隔离 | implemented / offline_passed | `resolve_run_context` 给 baseline 冻结空 hook spec，不写入当前候选 dropout/noise。 |
| P0-3 训练目标 vs 本地验证 | implemented / offline_passed | 本地验证走 raw encoder + `within_batch_accuracy`；优化器参数按对象身份去重。 |
| P0-4 PairRecord / 确认 | implemented / offline_passed | 确认要求完整身份字段 + goal/protocol 预声明 seeds。 |
| P1-1 可恢复结算 | implemented / offline_passed | `settlement.json` 分阶段；已有 evidence 但缺 diagnostics 会补写。 |
| P1-2 诊断 producer | implemented / offline_passed | `write_validation_artifacts` 写出 `retrieval_queries.jsonl` / `embeddings.json`。CPU toy 用真实打分，不是手写指标。 |
| P1-3 TaskContext digest | implemented / offline_passed | designer request 含 budget / capabilities / parent / control / contract。 |
| P1-4 memory 字段 | implemented / offline_passed | 保留 `statement` / `uncertainty`；空 `{}` comparison 拒绝。 |
| P1-5 审计新鲜度 | implemented / offline_passed | `report_dependency_hash`；stop / view 复查；新不可比行使审计 stale。 |
| P1-6 可搬迁 pack | implemented / offline_passed | binding 用相对路径；`job.command` 改写补 `--checkpoint <pack>/last.ckpt`。 |
| Studio / fresh-clone | implemented / offline_passed | `langgraph.json` 注册 `eeg_research`；UI 新 campaign 用 `tmp_path`，历史 run 不存在时不读。 |
| CPU toy 闭环 | offline_passed | 真实 retrieval/representation 计算。无 CUDA 全量 `fit()`。 |
| 真实 API 角色 | real_api_not_run | 本轮未再跑 live DeepSeek。此前授权的 live 角色不在本套件内。 |
| 真实 GPU 训练 / 配对确认 | real_training_not_run | 未启动新的 GPU campaign；未知 `api_usd` 保持 `null`。 |
| 自适应研究闭环 | blocked_prerequisite | 依赖真实训练与已授权预算，本轮不做。 |

## 保留的既有修复

R01–R08、blocked draft 拒绝、空 support lesson 拒绝、SVD rank-one=1、`job_id` 扣费/episode 去重、planner “Do not stop because the best two scores are close”、混杂 JSON 抢救不恢复。

## 未做

- 未恢复 `eeg_ui_44f7c763011b`
- 未把 mock 分数标成 `real_training_passed`
- 未提交本增量（等待明确要求）
