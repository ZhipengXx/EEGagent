"""Short metric explanations for the workbench. No network call."""

from __future__ import annotations

METRICS = {
    "fixed_bank_top1": {
        "label": "fixed-bank Top-1",
        "short": "固定候选集合中正确图像排第一的 query 比例。",
        "not": "不跨候选集合比较。",
    },
    "fixed_bank_top5": {
        "label": "fixed-bank Top-5",
        "short": "正确图像进入前五的 query 比例。",
        "not": "不是置信度。",
    },
    "train_loss": {
        "label": "训练损失",
        "short": "当前训练目标的优化值。",
        "not": "不同目标或设置未必可比。",
    },
    "gpu_hours": {
        "label": "GPU 小时",
        "short": "本任务分配 GPU 的累计时长。",
        "not": "不是美元费用或实际利用率。",
    },
    "within_batch_top1": {
        "label": "历史批内指标",
        "short": "一个训练 batch 内部的 top1。",
        "not": "不是固定候选集合上的选择指标。",
    },
}
