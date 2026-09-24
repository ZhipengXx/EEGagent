"""Read-only view model for one finished sample directory."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.metric_defs import definition

RUNS = Path(__file__).resolve().parents[3] / "runs"
_DONE = {"answered", "described", "not_applicable"}
_STATUS_LABEL = {
    "answered": ("已回答", "info"),
    "described": ("已描述", "info"),
    "unassessed": ("未评估", "neutral"),
    "blocked": ("检查被阻塞", "warning"),
    "unavailable": ("资源不可用", "warning"),
    "not_applicable": ("不适用", "neutral"),
    "skipped": ("未执行", "neutral"),
}
_VERDICT = {
    "passed_configured_checks": ("配置内数值检查通过", "配置内通过", "success"),
    "flagged": ("发现需关注项", "需关注", "warning"),
    "invalid_input": ("检查被阻塞", "阻塞", "danger"),
    "inconclusive": ("暂无法判断", "暂无法判断", "neutral"),
}
_QUESTION_LABEL = {
    "input_contract": "输入格式",
    "gray_comparability": "灰屏对照",
    "stimulus_temporal_description": "刺激时序",
    "coarse_spatial_description": "皮层空间分布",
    "triggered_followups": "后续检查",
    "reference_quality": "参考质量评估",
    "roi": "ROI 分区描述",
    "specificity": "跨图特异性",
    "calibrated_reference": "参考分布比较",
    "image_fmri_rsa": "图文一致性",
}
_DEEP_QUESTIONS = {"roi", "specificity", "calibrated_reference", "image_fmri_rsa"}
_METRIC_GROUP = {
    "shape_tv": "输入与格式",
    "finite_ratio": "输入与格式",
    "valid_for_numeric_checks": "输入与格式",
    "degenerate_signal": "输入与格式",
    "mean": "分布统计",
    "std": "分布统计",
    "min": "分布统计",
    "max": "分布统计",
    "quantiles": "分布统计",
    "comparable": "灰屏对照",
    "overall_delta_rms": "灰屏对照",
    "normalized_delta_over_control_rms": "灰屏对照",
    "peak_frame": "灰屏对照",
    "peak_time_s": "灰屏对照",
    "temporal_change_ratio": "时序特征",
    "max_frame_diff_ratio": "时序特征",
    "max_step_over_median": "时序特征",
    "lag1_corr_mean": "时序特征",
    "satisfies_contrast_contract": "时序特征",
    "roi_contribution": "空间特征",
    "embedding_available": "模型和参考评估",
    "reference_score_assessed": "模型和参考评估",
    "evidence_confidence": "运行成本",
    "recommended_training_weight": "运行成本",
    "lm_calls": "运行成本",
    "accepted_plan_count": "运行成本",
    "input_tokens": "运行成本",
    "output_tokens": "运行成本",
    "api_usd": "运行成本",
    "new_tribe_predictions": "运行成本",
}
_ARTIFACTS = (
    ("report.json", "报告", "report"),
    ("report.md", "报告", "report"),
    ("brain_tstrip.png", "图像", "image"),
    ("response_rms.json", "时序", "series"),
    ("plan.json", "计划", "plan"),
    ("plan_history.jsonl", "计划", "plan"),
    ("events.jsonl", "事件", "events"),
    ("llm_usage.json", "用量", "events"),
    ("memory_retrieval.json", "记忆", "memory"),
    ("episode.json", "记忆", "memory"),
    ("roi_profile.json", "空间", "series"),
    ("workflow.json", "页面元数据", "other"),
)


def list_runs(runs_root: Path | None = None) -> list[dict[str, Any]]:
    root = (runs_root or RUNS).resolve()
    if not root.is_dir():
        return []
    rows = []
    for report_path in root.glob("**/report.json"):
        if not report_path.is_file():
            continue
        sample_dir = report_path.parent
        try:
            rel = sample_dir.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        report = _read_json(report_path) or {}
        mtime = report_path.stat().st_mtime
        decision = _decision(report)
        rows.append(
            {
                "run_id": rel,
                "sample_id": report.get("sample_id"),
                "file_mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "file_time_note": "文件修改时间",
                "file_mtime_unix": mtime,
                "mode_label": _mode_label(report),
                "program_label": "运行完成",
                "decision_label": decision["short_label"],
                "decision_title": decision["verdict_label"],
                "decision_tone": decision["verdict_tone"],
            }
        )
    rows.sort(key=lambda row: row["file_mtime_unix"], reverse=True)
    for row in rows:
        row.pop("file_mtime_unix", None)
    return rows


def build_run_view(run_id: str, runs_root: Path | None = None) -> dict[str, Any] | None:
    sample_dir = resolve_sample_dir(run_id, runs_root)
    if sample_dir is None:
        return None
    report_path = sample_dir / "report.json"
    if not report_path.is_file():
        return None
    report = _read_json(report_path) or {}
    workflow = _read_json(sample_dir / "workflow.json") or {}
    usage = _read_json(sample_dir / "llm_usage.json") or {}
    memory = _read_json(sample_dir / "memory_retrieval.json") or {}
    pipeline = _read_json(sample_dir.parent / "pipeline.json") or {}
    events = _read_events(sample_dir / "events.jsonl")
    metrics_table = report.get("metrics_table") or {}
    validate = metrics_table.get("validate_input") or {}
    shape = validate.get("shape") if isinstance(validate.get("shape"), list) else None
    time_n = validate.get("T")
    protocol = _protocol(report, workflow, time_n)
    coverage = _coverage(report)
    required = [row for row in coverage if row["required"] is True]
    satisfied = sum(1 for row in required if row["status"] in _DONE)
    ledger = usage.get("ledger") if isinstance(usage.get("ledger"), dict) else {}
    inputs = workflow.get("inputs") if isinstance(workflow.get("inputs"), dict) else {}
    if not inputs:
        inputs = {
            "preds_path": validate.get("resolved_path"),
            "image_path": None,
            "gray_control_path": (pipeline.get("generation") or {}).get("control_preds_path"),
        }
    series = _series(metrics_table, sample_dir, protocol)
    return {
        "run_id": sample_dir.resolve().relative_to((runs_root or RUNS).resolve()).as_posix(),
        "sample_id": report.get("sample_id"),
        "file_mtime": datetime.fromtimestamp(report_path.stat().st_mtime).isoformat(timespec="seconds"),
        "file_time_note": "文件修改时间",
        "headline": protocol["headline"],
        "windows": protocol["windows"],
        "mode_label": _mode_label(report),
        "program": {"label": "运行完成", "tone": "neutral"},
        "decision": _decision(report),
        "claim_scope": report.get("claim_scope"),
        "analysis_goal": report.get("analysis_goal"),
        "summary": _summary(report, coverage),
        "scope_note": _scope_note(report, satisfied, len(required)),
        "followup_note": _followup_note(report),
        "biological_validity": report.get("biological_validity"),
        "sample": {
            "image_path": inputs.get("image_path"),
            "stimulus_rel": _stimulus_rel(sample_dir, inputs.get("image_path")),
            "preds_path": inputs.get("preds_path"),
            "gray_control_path": inputs.get("gray_control_path"),
            "shape": shape,
            "protocol_kind": protocol["kind"],
            "protocol_text": protocol["text"],
            "onset_s": protocol["onset_s"],
            "dt_s": protocol["dt_s"],
        },
        "coverage": coverage,
        "coverage_satisfied": satisfied,
        "coverage_required": len(required),
        "depth_reached": report.get("depth_reached"),
        "screen_depth": "deep" if any(row["id"] in _DEEP_QUESTIONS for row in coverage) else "standard",
        "tool_count": sum(1 for row in report.get("executions") or [] if row.get("status") == "success"),
        "llm": _llm(ledger),
        "new_tribe_predictions": _nested(pipeline, "generation", "new_tribe_predictions"),
        "metrics": _metrics(report, ledger, pipeline),
        "series": series,
        "roi": _roi(metrics_table),
        "trace": _trace(events),
        "memory": _memory(memory),
        "artifacts": _artifacts(sample_dir),
        "brain": _brain(sample_dir, inputs, protocol, time_n),
        "evidence_confidence": report.get("evidence_confidence", None),
        "recommended_training_weight": report.get("recommended_training_weight", None),
    }


def resolve_sample_dir(run_id: str, runs_root: Path | None = None) -> Path | None:
    root = (runs_root or RUNS).resolve()
    rel = (run_id or "").strip().strip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _decision(report: dict[str, Any]) -> dict[str, Any]:
    verdict = report.get("verdict")
    label, short, tone = _VERDICT.get(verdict, ("判决未记录", "未记录", "neutral"))
    stop = report.get("stop_reason")
    stop_label = None
    if stop == "abstain":
        stop_label = "暂无法判断"
    elif isinstance(stop, str) and stop.startswith("generation_failed"):
        stop_label = "任务失败"
        tone = "danger"
    return {
        "verdict": verdict,
        "verdict_label": label,
        "short_label": short,
        "verdict_tone": tone,
        "stop_reason": stop,
        "stop_label": stop_label,
        "screening_decision": report.get("screening_decision"),
    }


def _mode_label(report: dict[str, Any]) -> str:
    goal = report.get("analysis_goal") or report.get("check_profile")
    policy = report.get("resolved_policy") or report.get("requested_policy")
    if goal == "image16_numeric_diagnostic" or policy == "planned":
        return "分层诊断"
    if policy == "rule" or goal in {None, "tribe_image16"}:
        return "快速数值检查" if policy == "rule" else (str(goal) if goal else "模式未记录")
    return str(goal or policy or "模式未记录")


def _protocol(report: dict[str, Any], workflow: dict[str, Any], time_n: Any) -> dict[str, Any]:
    text = workflow.get("protocol")
    goal = str(report.get("analysis_goal") or "")
    image16 = time_n == 16 and ("image16" in goal or (isinstance(text, str) and "T=16" in text))
    if image16:
        return {
            "kind": "image16",
            "headline": "16s · 4s 灰 / 1s 图 / 后续灰 · fsaverage5",
            "text": text or "4 秒灰屏，1 秒图像，后续灰屏；T=16，t_stim = t − 4。",
            "onset_s": 4.0,
            "dt_s": 1.0,
            "windows": [
                {"label": "灰屏", "start_s": 0.0, "end_s": 4.0},
                {"label": "图像", "start_s": 4.0, "end_s": 5.0},
                {"label": "灰屏", "start_s": 5.0, "end_s": 16.0},
            ],
        }
    recorded = text if isinstance(text, str) and text.strip() else None
    return {
        "kind": "unrecorded" if not recorded else "recorded_text",
        "headline": recorded or "协议未记录",
        "text": recorded or "协议未记录，不把这次样本写成 16 秒。",
        "onset_s": None,
        "dt_s": None,
        "windows": [],
    }


def _coverage(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("coverage_by_dimension") or {}
    rows = []
    if not isinstance(raw, dict):
        return rows
    for name, item in raw.items():
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        required = item.get("required", None)
        label, tone = _STATUS_LABEL.get(status, ("状态未记录", "neutral"))
        rows.append(
            {
                "id": name,
                "label_zh": _QUESTION_LABEL.get(name, name),
                "status": status,
                "status_label": label,
                "tone": tone,
                "required": required if isinstance(required, bool) else None,
                "required_label": (
                    "必答"
                    if required is True
                    else "非必答"
                    if required is False
                    else "要求未记录"
                ),
                "blocks_completion": required is True and status not in _DONE and status is not None,
            }
        )
    return rows


def _summary(report: dict[str, Any], coverage: list[dict[str, Any]]) -> str:
    by_id = {row["id"]: row for row in coverage}
    bits = []
    decision = _decision(report)
    bits.append(decision["verdict_label"] + "。")
    if decision["stop_label"]:
        bits.append("停止原因是" + decision["stop_label"] + "。")
    named = (
        ("gray_comparability", "灰屏对照"),
        ("stimulus_temporal_description", "时序描述"),
        ("coarse_spatial_description", "空间描述"),
    )
    covered = []
    for key, title in named:
        row = by_id.get(key)
        if row is None:
            continue
        covered.append(f"{title}{row['status_label']}")
    if covered:
        bits.append("，".join(covered) + "。")
    bio = report.get("biological_validity")
    if bio in {None, "not_assessed"}:
        bits.append("生物学有效性未评估。")
    return "".join(bits)


def _llm(ledger: dict[str, Any]) -> dict[str, Any]:
    def pick(key: str) -> Any:
        return ledger.get(key, None) if key in ledger else None

    return {
        "lm_calls": pick("lm_calls"),
        "planner_calls_succeeded": pick("planner_calls_succeeded"),
        "accepted_plan_count": pick("accepted_plan_count"),
        "repair_attempts": pick("repair_attempts"),
        "revise_plan_attempts": pick("revise_plan_attempts"),
        "input_tokens": pick("input_tokens"),
        "output_tokens": pick("output_tokens"),
        "api_usd": pick("api_usd"),
        "api_usd_label": "未提供" if pick("api_usd") is None else None,
        "curator_calls": pick("curator_calls"),
    }


def _metric_row(
    metric_id: str,
    value: Any,
    *,
    execution_id: str | None = None,
    signal_mode: str | None = None,
) -> dict[str, Any]:
    spec = definition(metric_id) or {
        "id": metric_id,
        "label_zh": metric_id,
        "value_kind": _kind(value),
        "meaning": "报告里有这个字段，解释尚未登记。",
        "formula": "未登记",
        "limitations": [],
        "threshold_kind": "none",
        "direction": "no_quality_direction",
    }
    return {
        "id": metric_id,
        "label": spec["label_zh"],
        "value": value,
        "value_kind": spec["value_kind"],
        "group": _METRIC_GROUP.get(metric_id, "其他"),
        "display": _display(value, spec),
        "execution_id": execution_id,
        "signal_mode": signal_mode,
        "definition": spec,
    }


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, dict):
        return "object"
    return "status"


def _display(value: Any, spec: dict[str, Any] | str) -> str:
    kind = spec.get("value_kind") if isinstance(spec, dict) else spec
    unit = spec.get("unit_kind") if isinstance(spec, dict) else None
    metric_id = spec.get("id") if isinstance(spec, dict) else None
    if value is None:
        return "未提供"
    if isinstance(value, bool) or kind == "boolean":
        return "是" if value is True else "否"
    if metric_id == "shape_tv" and isinstance(value, dict):
        time_n = value.get("T")
        vertices = value.get("V")
        if isinstance(time_n, int) and isinstance(vertices, int):
            return f"{time_n} 个时间点 × {vertices:,} 个顶点"
    if metric_id == "quantiles" and isinstance(value, dict):
        return _quantile_text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return "0"
        if unit == "proportion":
            return _percent(float(value))
        if unit == "count" and float(value).is_integer():
            return f"{int(value):,}"
        if unit == "usd":
            return f"${_sig(float(value))}"
        return _sig(float(value))
    if kind == "object":
        return "查看原始字段"
    return str(value)


def _percent(value: float) -> str:
    percent = value * 100
    if percent == 0:
        return "0%"
    text = f"{percent:.4g}"
    return text + "%"


def _sig(value: float) -> str:
    return f"{value:.4g}"


def _quantile_text(value: dict[str, Any]) -> str:
    keys = (
        (("0.05", "p05", "P5"), "P5"),
        (("0.5", "p50", "median"), "中位数"),
        (("0.95", "p95", "P95"), "P95"),
    )
    parts = []
    for names, label in keys:
        found = next((value[name] for name in names if name in value), None)
        parts.append(f"{label} {_display(found, {'value_kind': 'number', 'unit_kind': 'model_output'})}")
    return " / ".join(parts)


def _scope_note(report: dict[str, Any], satisfied: int, required_n: int) -> str:
    scope = report.get("claim_scope") or "claim_scope 未记录"
    if required_n:
        cover = f"必需问题 {satisfied}/{required_n}。"
    else:
        cover = "必需问题数未记录。"
    bio = report.get("biological_validity")
    bio_bit = "真实脑响应未评估。" if bio in {None, "not_assessed"} else ""
    return f"检查范围：{scope}。{cover}{bio_bit}"


def _followup_note(report: dict[str, Any]) -> str:
    items = report.get("followups", None)
    if items is None:
        items = report.get("tickets", None)
    if isinstance(items, list) and not items:
        return "无待跟进问题"
    if isinstance(items, list):
        return f"{len(items)} 个待跟进问题"
    return "跟进记录未提供"


def _stimulus_rel(sample_dir: Path, image_path: Any) -> str | None:
    if not isinstance(image_path, str) or not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        rel = path.resolve().relative_to(sample_dir.resolve())
    except ValueError:
        return None
    if rel.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    return rel.as_posix()


def _metrics(report: dict[str, Any], ledger: dict[str, Any], pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    table = report.get("metrics_table") or {}
    rows: list[dict[str, Any]] = []
    validate = table.get("validate_input") or {}
    basic = table.get("basic_statistics") or {}
    gray = table.get("gray_control_contrast") or {}
    temporal = table.get("stimulus_temporal_profile") or {}
    mae = table.get("cortex_mae") or {}
    if validate:
        shape = validate.get("shape")
        rows.append(_metric_row("shape_tv", {"T": validate.get("T"), "V": validate.get("V"), "shape": shape}))
        for key in ("finite_ratio", "valid_for_numeric_checks", "degenerate_signal"):
            if key in validate:
                rows.append(_metric_row(key, validate.get(key)))
    for key in (
        "degenerate_signal",
        "mean",
        "std",
        "min",
        "max",
        "quantiles",
        "temporal_change_ratio",
        "max_frame_diff_ratio",
    ):
        if key in basic:
            rows.append(_metric_row(key, basic.get(key)))
    if "lag1_corr_mean" in basic or "lag1_corr_mean" in temporal:
        source = temporal if "lag1_corr_mean" in temporal else basic
        rows.append(_metric_row("lag1_corr_mean", source.get("lag1_corr_mean")))
    for key in (
        "comparable",
        "overall_delta_rms",
        "normalized_delta_over_control_rms",
        "peak_frame",
        "peak_time_s",
        "max_step_over_median",
    ):
        if key in gray:
            rows.append(
                _metric_row(key, gray.get(key), signal_mode=gray.get("signal_mode"))
            )
    if temporal:
        rows.append(
            _metric_row(
                "satisfies_contrast_contract",
                temporal.get("satisfies_contrast_contract"),
                signal_mode=temporal.get("signal_mode"),
            )
        )
    for key in ("embedding_available", "reference_score_assessed"):
        if key in mae:
            rows.append(_metric_row(key, mae.get(key)))
    rows.append(_metric_row("evidence_confidence", report.get("evidence_confidence", None)))
    rows.append(
        _metric_row("recommended_training_weight", report.get("recommended_training_weight", None))
    )
    for key in ("lm_calls", "accepted_plan_count", "input_tokens", "output_tokens", "api_usd"):
        if key in ledger:
            rows.append(_metric_row(key, ledger.get(key)))
    gen = pipeline.get("generation") if isinstance(pipeline.get("generation"), dict) else {}
    if "new_tribe_predictions" in gen:
        rows.append(_metric_row("new_tribe_predictions", gen.get("new_tribe_predictions")))
    return rows


def _series(metrics: dict[str, Any], sample_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    gray = metrics.get("gray_control_contrast") or {}
    ras = gray.get("ras_v1") if isinstance(gray.get("ras_v1"), dict) else None
    rms_file = _read_json(sample_dir / "response_rms.json")
    if not ras and not (isinstance(rms_file, dict) and rms_file.get("response_rms")):
        return {"available": False, "reason": "这份报告没有整条 RMS 或 R/A/S 序列。"}
    times = _times(protocol, len((ras or {}).get("R") or (rms_file or {}).get("response_rms") or []))
    return {
        "available": True,
        "signal_mode": (ras or {}).get("signal_mode") or gray.get("signal_mode"),
        "times": times,
        "R": (ras or {}).get("R"),
        "A": (ras or {}).get("A"),
        "S": (ras or {}).get("S"),
        "response_rms": (rms_file or {}).get("response_rms") if isinstance(rms_file, dict) else None,
        "note": "R、A、S 分开画。A 和 S 的点位于相邻帧之间，不是单个时间点。",
    }


def _times(protocol: dict[str, Any], n: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(n):
        if protocol["kind"] == "image16" and protocol["dt_s"] is not None:
            absolute = index * float(protocol["dt_s"])
            relative = absolute - float(protocol["onset_s"])
        else:
            absolute = None
            relative = None
        rows.append(
            {
                "index": index,
                "absolute_time_s": absolute,
                "stimulus_relative_time_s": relative,
            }
        )
    return rows


def _roi(metrics: dict[str, Any]) -> dict[str, Any]:
    roi = metrics.get("surface_roi_profile") or {}
    top = roi.get("top_k")
    if not isinstance(top, list) or not top:
        return {"available": False, "reason": roi.get("skip_reason") or "没有 ROI 表。"}
    rows = []
    for item in top[:8]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "roi_key": item.get("roi_key"),
                "name": item.get("name"),
                "hemisphere": item.get("hemisphere"),
                "label_id": item.get("label_id"),
                "n_vertices": item.get("n_vertices"),
                "signed_mean": item.get("signed_mean"),
                "rms": item.get("rms"),
            }
        )
    return {
        "available": True,
        "signal_mode": roi.get("mode"),
        "excluded_vertex_fraction": roi.get("excluded_vertex_fraction"),
        "rows": rows,
        "note": "signed mean 和 RMS 单位不同，不放在同一条轴上。",
    }


def _trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, event in enumerate(events):
        kind = str(event.get("type") or "event")
        label, group = _trace_label(kind, event)
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        reason = decision.get("reason") or event.get("message")
        rows.append(
            {
                "index": index,
                "type": kind,
                "group": group,
                "label": label,
                "status": event.get("status") or decision.get("action"),
                "tool_name": event.get("tool_name") or decision.get("tool_name") or event.get("role"),
                "reason": reason if reason else None,
                "reason_missing": reason in {None, ""},
                "result_id": event.get("result_id") or event.get("call_id") or event.get("plan_id"),
                "parent_call_id": event.get("parent_call_id"),
            }
        )
    return rows


def _trace_label(kind: str, event: dict[str, Any]) -> tuple[str, str]:
    role = str(event.get("role") or "")
    if kind == "generate":
        return "生成", "generate"
    if kind == "tool" and event.get("tool_name") in {"validate_input", "basic_statistics"}:
        return "输入检查", "check"
    if kind == "tool":
        return "工具执行", "tool"
    if "repair" in kind:
        return "格式修复", "repair"
    if "revise" in kind or role == "replanner":
        return "重规划", "revise"
    if kind.startswith("plan.") or kind.startswith("api.") or kind == "decision":
        if role == "replanner":
            return "重规划", "revise"
        return "初始规划", "plan"
    if kind.startswith("memory.retrieved"):
        return "记忆检索", "memory"
    if kind.startswith("memory."):
        return "记忆记录", "memory"
    if kind == "stop":
        return "停止", "stop"
    return kind, "other"


def _memory(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"recorded": False, "hit_count": None, "note": "没有记忆检索记录。"}
    hits = payload.get("hit_count", None)
    return {
        "recorded": True,
        "hit_count": hits,
        "path_effect": payload.get("path_effect"),
        "path_changed": payload.get("path_changed", None),
        "note": (
            "未记录对路径的影响。"
            if not payload.get("effects")
            else "检索记录里写了影响。"
        ),
        "items": payload.get("items") or [],
    }


def _artifacts(sample_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name, group, kind in _ARTIFACTS:
        path = sample_dir / name
        if path.is_file():
            rows.append({"file": name, "group": group, "kind": kind})
    return rows


def _brain(
    sample_dir: Path,
    inputs: dict[str, Any],
    protocol: dict[str, Any],
    time_n: Any,
) -> dict[str, Any]:
    modes = []
    preds = inputs.get("preds_path")
    gray = inputs.get("gray_control_path")
    n_frames = int(time_n) if isinstance(time_n, int) else None
    limits: dict[str, Any] = {}
    if isinstance(preds, str) and Path(preds).is_file():
        modes.append("raw")
        limits["raw"] = _limits(preds)
        if n_frames is None:
            n_frames = _frame_count(preds)
    if (
        "raw" in modes
        and isinstance(gray, str)
        and Path(gray).is_file()
        and _frame_count(gray) == n_frames
    ):
        modes.append("contrast")
        limits["contrast"] = _contrast_limits(preds, gray)
    montage = sample_dir / "brain_tstrip.png"
    return {
        "modes": modes,
        "views": ["left", "right", "posterior"],
        "n_frames": n_frames,
        "protocol_kind": protocol["kind"],
        "onset_s": protocol["onset_s"],
        "dt_s": protocol["dt_s"],
        "montage": montage.is_file(),
        "units": "模型输出单位",
        "color_limits": limits,
        "scale_note": "同一信号模式跨帧使用固定色阶。raw 与 contrast 色阶不同，不直接比较颜色强弱。正负只表示符号。",
    }


def _frame_count(path: str) -> int | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if getattr(arr, "ndim", 0) != 2:
        return None
    return int(arr.shape[0])


def _limits(path: str) -> dict[str, float] | None:
    try:
        arr = np.load(path, mmap_mode="r")
        finite = np.asarray(arr, dtype=np.float64)
    except (OSError, ValueError):
        return None
    if finite.size == 0:
        return None
    vmax = max(float(np.nanpercentile(np.abs(finite), 99)), 1e-6)
    return {"vmin": -vmax, "vmax": vmax, "source": "nanpercentile_abs_99"}


def _contrast_limits(preds: str, gray: str) -> dict[str, float] | None:
    try:
        delta = np.asarray(np.load(preds), dtype=np.float64) - np.asarray(np.load(gray), dtype=np.float64)
    except (OSError, ValueError):
        return None
    vmax = max(float(np.nanpercentile(np.abs(delta), 99)), 1e-6)
    return {"vmin": -vmax, "vmax": vmax, "source": "nanpercentile_abs_99"}


FRAME_VERSION = "frame_v2"
_VIEWS = ("left", "right", "posterior")
_frame_lock = threading.Lock()
_plot_lock = threading.Lock()
_limit_cache: dict[str, dict[str, Any]] = {}
_render_jobs: dict[tuple[str, str, int], threading.Event] = {}
_render_notes: dict[tuple[str, str, int], str] = {}


def _triplet_ready(out_dir: Path) -> bool:
    return all((out_dir / f"{name}.png").is_file() for name in _VIEWS)


def _frame_context(sample_dir: Path, run_id: str, runs_root: Path | None) -> dict[str, Any] | None:
    key = str(sample_dir.resolve())
    cached = _limit_cache.get(key)
    if cached is not None:
        return cached
    run_view = build_run_view(run_id, runs_root)
    if run_view is None:
        return None
    brain = run_view["brain"]
    sample = run_view["sample"]
    ctx = {
        "modes": set(brain.get("modes") or []),
        "n_frames": int(brain.get("n_frames") or 0),
        "limits": brain.get("color_limits") or {},
        "preds": sample.get("preds_path"),
        "gray": sample.get("gray_control_path"),
    }
    _limit_cache.setdefault(key, ctx)
    return _limit_cache[key]


def _render_missing_frame(sample_dir: Path, ctx: dict[str, Any], mode: str, index: int) -> str:
    limits = (ctx.get("limits") or {}).get(mode)
    if not limits:
        return "color limits missing"
    out_dir = sample_dir / "frames" / FRAME_VERSION / mode / f"{index:02d}"
    if _triplet_ready(out_dir):
        return ""
    preds = ctx.get("preds")
    gray = ctx.get("gray")
    try:
        pred = np.load(preds, mmap_mode="r")
        vector = np.asarray(pred[index], dtype=np.float64)
        if mode == "contrast":
            control = np.load(gray, mmap_mode="r")
            vector = vector - np.asarray(control[index], dtype=np.float64)
    except (OSError, ValueError, IndexError, TypeError) as exc:
        return f"array unreadable: {exc}"
    from react_agent.fmri._brain_strip import render_frame_views

    with _plot_lock:
        if _triplet_ready(out_dir):
            return ""
        ok, note = render_frame_views(
            vector, out_dir, vmin=float(limits["vmin"]), vmax=float(limits["vmax"])
        )
    if not ok or not _triplet_ready(out_dir):
        return note or "frame missing"
    return ""


def ensure_frame(
    run_id: str,
    mode: str,
    index: int,
    runs_root: Path | None = None,
    view_name: str = "left",
) -> tuple[Path | None, str]:
    """Return one cached view. A cache hit does not rebuild the report."""
    sample_dir = resolve_sample_dir(run_id, runs_root)
    if sample_dir is None:
        return None, "run not found"
    if mode not in {"raw", "contrast"}:
        return None, "unknown signal mode"
    if view_name not in _VIEWS:
        return None, "unknown view"
    out_dir = sample_dir / "frames" / FRAME_VERSION / mode / f"{index:02d}"
    target = out_dir / f"{view_name}.png"
    if _triplet_ready(out_dir):
        return target, ""
    job_key = (str(sample_dir.resolve()), mode, index)
    with _frame_lock:
        if _triplet_ready(out_dir):
            return target, ""
        event = _render_jobs.get(job_key)
        owner = event is None
        if owner:
            event = threading.Event()
            _render_jobs[job_key] = event
    if not owner:
        assert event is not None
        event.wait(timeout=180)
        if target.is_file() and _triplet_ready(out_dir):
            return target, ""
        return None, _render_notes.get(job_key) or "frame missing"
    note = ""
    try:
        ctx = _frame_context(sample_dir, run_id, runs_root)
        if ctx is None:
            note = "report missing"
        elif mode not in ctx["modes"]:
            note = "signal mode unavailable"
        elif index < 0 or index >= int(ctx["n_frames"]):
            note = "frame out of range"
        else:
            note = _render_missing_frame(sample_dir, ctx, mode, index)
    finally:
        _render_notes[job_key] = note
        event.set()
        with _frame_lock:
            _render_jobs.pop(job_key, None)
    if note or not target.is_file():
        return None, note or "frame missing"
    return target, ""


def _nested(payload: dict[str, Any], key: str, child: str) -> Any:
    block = payload.get(key)
    if isinstance(block, dict) and child in block:
        return block.get(child)
    return None
