# 工作台改版说明

按 `Cursor_UI_Revision_Prompt.md` 实施。概念图只作布局参考，数字均为合成示例。审查基线 `c26f68f`。

## 已完成

- 最好完整结果：`0` pp 高于 `-2` pp；None / NaN / Inf / Pilot / baseline / 无效行不参与排序；全为非正增益时标注「尚未优于对照」，不显示研究成功。
- `GET /api/agentic_status` 列表只给摘要；详情、timeline、job history、source 按需读取。
- timeline 分页 `cursor/next_cursor/has_more`，超过 8 条可翻页。
- 训练曲线跟 `jobs/<job_id>/history.jsonl`；`live_job` 清空后仍可读；0 是有效点，缺失断线。
- `ui_events.jsonl` 在模型调用前写入开始事件，不增加 `cost.json` 的 `llm_calls`。
- 工作台主入口为「自主研究」；创建表单独立；刷新用 `#research/<id>`；轮询不再抢焦点。
- 明确 demo：`POST /api/agentic_demo` + `?demo=1`，根目录 `runs/eeg_research_ui_demo/`。
- 桌面三栏；1024/768 收起研究列表与详情抽屉，避免挤成三列。

## 测试结果

在 `react-agent` 下运行（本机有 torch，未出现「缺 torch 无法执行」）：

```
PYTHONPATH=src python -m pytest \
  tests/eeg_research/test_ui_workspace.py \
  tests/eeg_research/test_agentic_view.py \
  tests/eeg_research/test_worker_failure_recovery.py \
  tests/eeg_research/test_e2a13a2_delta.py \
  tests/eeg_research/test_v1_8.py \
  tests/eeg_training/test_protocol.py -q
```

**78 passed**，2 条 `os.fork` DeprecationWarning。这不是全仓库测试。

## 浏览器验收

工作台：`http://127.0.0.1:8766/?demo=1#research/demo_ui_workspace`（仅绑定本机）。

真实页面截图（Firefox headless，等待 `.workspace-rank` 后再拍，不是概念 PNG）：

- `docs/figures/ui-workspace/ui-workspace-1440.png`
- `docs/figures/ui-workspace/ui-workspace-1024.png`
- `docs/figures/ui-workspace/ui-workspace-768.png`
- `docs/figures/ui-workspace/ui-workspace-1440-experiments.png`

1440 可见完整标题、`c01 0 pp · 尚未优于对照`、左栏选中 demo、右栏步骤详情。1024/768 出现「研究列表」，侧栏与详情默认收起。

## 未完成 / 边界

- 未迁移 SSE/WebSocket。
- 未改 Pilot 默认 3 epoch、调度器或确认协议。
- demo 根下仍有 `other_campaign`，只用于跨 campaign 拒绝测试，会出现在研究列表。
- 时间线摘要仍保留部分原始 `event_type` 英文（如 `llm_call_started`）。
- 没有付费 LLM，没有 GPU 训练。
