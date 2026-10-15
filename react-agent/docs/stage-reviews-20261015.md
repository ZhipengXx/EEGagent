# 分阶段验收

Designer 在同一份 v2 design brief 中提出三组要求，各组独立验收。一次完整迭代仍按 Planner → Designer → Coder → Implementation → 训练 → Measurement → Analysis → memory 执行。

| 阶段 | 执行者 | 输入与验收 |
| --- | --- | --- |
| Implementation | 原 candidate_reviewer 模型 | 只验 implementation_requirements、已选择的接口、当前源码与原生检查。通过表示可以训练。 |
| Measurement | 原生确定性 subagent | 复用 suite/screening/job 验收，不调用模型；核验冻结身份、配对覆盖、指标。 |
| Analysis | 原 result_analyst 模型 | 接收 analysis_requirements、研究问题和已核验事实；解释结果及范围，完成后进入原 memory 流程。 |

正常路径没有增加模型调用、训练或代码检查。三个阶段复用现有角色和核验流程，不新增通用 Reviewer。API 前的 selected_research_task 注入也遵守当前阶段，避免把整项任务的完成条件重新传入 Implementation。

Designer 应显式输出 measurement_requirements 和 analysis_requirements，即使为空。条款 ID 在三组中唯一。Measurement 的 verification 仅支持 run_identity、paired_coverage、metric_integrity，语义来自冻结协议；Analysis 支持 evidence_grounding、paired_comparison、scope_limits、hypothesis_assessment。Implementation 继续使用原代码条款和 source/check 引用。

可选 hook 的可用性不构成设计义务。Implementation 输入标出 selected / existing / unselected optional 的适用性；保留选定与已有 hook 的检查上下文。参数量仍使用原生 scalar census。

负实验可以完成 Analysis。analysis_requirement_coverage 的 implemented 表示已完成要求的判断；假设本身可为 weakened。核心分析条款缺失或引用其他实验时，保留分析产物与未完成状态，后续迭代不能把它当作已闭环。

Measurement 和 Analysis 的验收记录原生写入 CAM 的 stage_reviews/，并绑定源代码、spec、测量收据和分析 artifact；相同输入重复执行复用原记录。这里不修改旧设计、源码、运行收据或审核结果。

兼容性：没有显式新字段的旧 v1/v2 spec 保持原哈希和原路径；旧 Analyst 的 schema 与 mission 也保留。新条款进入批准、科学摘要和冻结执行身份。已经批准的旧条款不会被自动迁移至新阶段。

本次实现位于隔离工作树 `/home/zxuff/data/EEGagent-stage-reviews` 的 `codex/stage-reviews-20261015` 分支，修改前本地快照为 `924ddc4`。旧研究位于原工作树，其 c3 记录保持原状。当前验证使用原生合成 CPU 训练及 SDK MockTransport，不代表真实 EEG 或在线模型性能验证。

离线回归命令（在 react-agent/ 下）：

```bash
PYTHONPATH=src EEG_EXTERNAL_API_DISABLED=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/eeg_research/version_audit -q
```
