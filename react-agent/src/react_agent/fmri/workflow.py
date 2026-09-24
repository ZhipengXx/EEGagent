"""Per-run workflow page: inputs, protocol note, TRIBE strip, contract bars.

Bars score numeric-contract completeness. They leave evidence_confidence empty
and do not paint max_step_over_median as a confidence.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from react_agent.fmri._brain_strip import render_brain_strip

import numpy as np

MESH = "fsaverage5"
N_VERTICES = 20484
APPLE_CACHE = Path(
    "/home/zxuff/data/tribev2/exp/things_static_1s_7s/vis/apple_signed_tstrip.png"
)
_DONE = {"answered", "described"}
_LOW = {"unassessed", "blocked"}


def contract_bars(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a few report fields onto 0–1 display scores. Green is high."""
    metrics = report.get("metrics_table") or {}
    validate = metrics.get("validate_input") or {}
    temporal = metrics.get("stimulus_temporal_profile") or {}
    gray = metrics.get("gray_control_contrast") or {}
    spatial = metrics.get("surface_spatial_sanity") or {}
    bars: list[dict[str, Any]] = []

    finite = validate.get("finite_ratio")
    if finite is None:
        finite = (metrics.get("basic_statistics") or {}).get("finite_ratio")
    bars.append(_numeric_bar("finite_ratio", "输入有限值比例", finite))

    valid = validate.get("valid_for_numeric_checks")
    bars.append(
        {
            "id": "valid_for_numeric_checks",
            "label": "输入可用于数值检查",
            "score": 1.0 if valid is True else 0.0,
            "raw": valid,
        }
    )

    comparable = gray.get("comparable")
    bars.append(
        {
            "id": "gray_control_contrast.comparable",
            "label": "灰屏可对照",
            "score": 1.0 if comparable is True else 0.0,
            "raw": comparable,
        }
    )

    contrast_ok = temporal.get("satisfies_contrast_contract")
    bars.append(
        {
            "id": "satisfies_contrast_contract",
            "label": "contrast 时序合同",
            "score": 1.0 if contrast_ok is True else 0.0,
            "raw": contrast_ok if contrast_ok is not None else temporal.get("signal_mode"),
        }
    )

    spatial_ok = _spatial_contrast_done(report, spatial)
    bars.append(
        {
            "id": "surface_spatial_sanity",
            "label": "粗空间扫描（contrast）",
            "score": 1.0 if spatial_ok else 0.0,
            "raw": {
                "signal_mode": spatial.get("signal_mode"),
                "mode": spatial.get("mode"),
            },
        }
    )

    coverage = report.get("coverage_by_dimension") or {}
    for name, row in coverage.items():
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status in _DONE:
            score: float | None = 1.0
        elif status in _LOW:
            score = 0.0
        else:
            score = None
        bars.append(
            {
                "id": f"coverage.{name}",
                "label": f"必答 {name}",
                "score": score,
                "raw": status,
            }
        )
    return bars


def descriptive_numbers(report: dict[str, Any]) -> dict[str, Any]:
    """Numbers shown as text. They are not painted as confidence."""
    gray = (report.get("metrics_table") or {}).get("gray_control_contrast") or {}
    return {
        "claim_scope": report.get("claim_scope") or "numeric_consistency_only",
        "analysis_goal": report.get("analysis_goal"),
        "evidence_confidence": report.get("evidence_confidence"),
        "recommended_training_weight": report.get("recommended_training_weight"),
        "max_step_over_median": gray.get("max_step_over_median"),
    }


def publish_workflow(
    sample_dir: Path,
    report: dict[str, Any],
    *,
    sample_spec: dict[str, Any] | None = None,
    gray_path: str | None = None,
    image_path: str | None = None,
    preds_path: str | None = None,
) -> dict[str, Any]:
    """Write brain_tstrip.png, workflow.html, and a small score sidecar."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    spec = sample_spec or {}
    preds = preds_path or _preds_path(report, spec)
    image = image_path or spec.get("image_path") or _image_from_sidecar(preds)
    gray = gray_path or _gray_from_pipeline(sample_dir)
    sample_id = str(report.get("sample_id") or spec.get("sample_id") or "")
    bars = contract_bars(report)
    notes = descriptive_numbers(report)
    strip_path = sample_dir / "brain_tstrip.png"
    image_source = "missing"
    plot_note = ""
    try:
        items = _load_items(preds, gray)
        ok, plot_note = render_brain_strip(items, strip_path)
        if ok:
            image_source = "rendered"
        elif _is_apple_01b(sample_id, image) and APPLE_CACHE.is_file():
            shutil.copyfile(APPLE_CACHE, strip_path)
            image_source = "cache"
            plot_note = (
                "本次没有画出新图。下面是 apple_01b 的缓存条带 "
                f"({APPLE_CACHE})。原因：{plot_note}"
            )
        else:
            image_source = "missing"
    except Exception as exc:  # noqa: BLE001
        image_source = "missing"
        plot_note = f"plot failed: {exc}"

    page = {
        "sample_id": sample_id,
        "html_path": str((sample_dir / "workflow.html").resolve()),
        "brain_tstrip": str(strip_path.resolve()) if strip_path.is_file() else None,
        "image_source": image_source,
        "plot_note": plot_note,
        "inputs": {
            "image_path": image,
            "preds_path": preds,
            "gray_control_path": gray,
        },
        "protocol": _protocol_text(notes.get("analysis_goal")),
        "claim_scope": notes["claim_scope"],
        "analysis_goal": notes.get("analysis_goal"),
        "metric_scores": {bar["id"]: bar["score"] for bar in bars},
        "max_step_over_median": notes.get("max_step_over_median"),
        "evidence_confidence": None,
        "recommended_training_weight": None,
    }
    _write_html(sample_dir / "workflow.html", page, bars, notes, strip_path.is_file())
    (sample_dir / "workflow.json").write_text(
        json.dumps(page, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return page


def publish_existing_run(sample_dir: Path) -> dict[str, Any]:
    """Rebuild the page from a finished run. Does not call TRIBE or an LM."""
    sample_dir = Path(sample_dir)
    report_path = sample_dir / "report.json"
    if not report_path.is_file():
        nested = sorted(sample_dir.glob("*/report.json"))
        if len(nested) == 1:
            sample_dir = nested[0].parent
            report_path = nested[0]
        else:
            raise FileNotFoundError(f"no report.json under {sample_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pipeline = _find_pipeline(sample_dir)
    gen = (pipeline or {}).get("generation") or {}
    return publish_workflow(
        sample_dir,
        report,
        gray_path=gen.get("control_preds_path"),
        preds_path=gen.get("preds_path")
        or ((report.get("metrics_table") or {}).get("validate_input") or {}).get(
            "resolved_path"
        ),
        image_path=_image_from_sidecar(gen.get("preds_path")),
    )


def _numeric_bar(bar_id: str, label: str, value: Any) -> dict[str, Any]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
        value = None
    score = max(0.0, min(1.0, score))
    return {"id": bar_id, "label": label, "score": score, "raw": value}


def _spatial_contrast_done(report: dict[str, Any], spatial: dict[str, Any]) -> bool:
    if spatial.get("signal_mode") != "contrast":
        return False
    if not spatial:
        return False
    for row in report.get("executions") or []:
        if row.get("tool_name") == "surface_spatial_sanity":
            return row.get("status") == "success"
    return True


def _preds_path(report: dict[str, Any], spec: dict[str, Any]) -> str | None:
    metrics = (report.get("metrics_table") or {}).get("validate_input") or {}
    return metrics.get("resolved_path") or spec.get("fmri_path")


def _image_from_sidecar(preds: str | None) -> str | None:
    if not preds:
        return None
    side = Path(preds).with_suffix(".json")
    if not side.is_file():
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    path = payload.get("path") or payload.get("image_path")
    return str(path) if path else None


def _gray_from_pipeline(sample_dir: Path) -> str | None:
    pipeline = _find_pipeline(sample_dir)
    if not pipeline:
        return None
    gen = pipeline.get("generation") or {}
    return gen.get("control_preds_path")


def _find_pipeline(sample_dir: Path) -> dict[str, Any] | None:
    for path in (sample_dir / "pipeline.json", sample_dir.parent / "pipeline.json"):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _load_items(preds: str | None, gray: str | None) -> list[tuple[str, np.ndarray]]:
    if not preds or not Path(preds).is_file():
        return []
    arr = np.load(preds)
    if getattr(arr, "ndim", 0) != 2:
        return []
    items = [("pred", arr)]
    if gray and Path(gray).is_file():
        control = np.load(gray)
        if getattr(control, "shape", None) == arr.shape:
            items.append(("contrast", arr.astype(np.float64) - control.astype(np.float64)))
    return items


def _is_apple_01b(sample_id: str, image: str | None) -> bool:
    if sample_id == "apple_01b":
        return True
    return bool(image) and Path(image).stem == "apple_01b"


def _protocol_text(analysis_goal: Any) -> str:
    goal = analysis_goal or "（报告未写 analysis_goal）"
    return (
        "协议：10 fps，4 秒灰屏 + 1 秒图片 + 后续灰屏；导出窗 [0, 16) 秒，"
        f"按 1 秒取样得到 T=16。网格 {MESH}，顶点 LH||RH 共 {N_VERTICES}。"
        "t_stim = t_video − 4，图片起点为 t_stim=0。预测没有再加 5 秒，也没有做 HRF。"
        f"这次 analysis_goal = {goal}。"
    )


def refresh_workflow_html(sample_dir: Path) -> Path:
    """Rewrite workflow.html from the saved report. Does not draw a new strip."""
    sample_dir = Path(sample_dir)
    report = json.loads((sample_dir / "report.json").read_text(encoding="utf-8"))
    meta_path = sample_dir / "workflow.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    notes = descriptive_numbers(report)
    page = {
        "sample_id": report.get("sample_id") or meta.get("sample_id"),
        "image_source": meta.get("image_source") or (
            "rendered" if (sample_dir / "brain_tstrip.png").is_file() else "missing"
        ),
        "plot_note": meta.get("plot_note") or "",
        "inputs": meta.get("inputs") or {},
        "protocol": meta.get("protocol") or _protocol_text(notes.get("analysis_goal")),
    }
    html_path = sample_dir / "workflow.html"
    _write_html(
        html_path,
        page,
        contract_bars(report),
        notes,
        (sample_dir / "brain_tstrip.png").is_file(),
    )
    return html_path


def _write_html(
    path: Path,
    page: dict[str, Any],
    bars: list[dict[str, Any]],
    notes: dict[str, Any],
    has_image: bool,
) -> None:
    inputs = page.get("inputs") or {}
    rows = "\n".join(_bar_row(bar) for bar in bars)
    if has_image and page.get("image_source") == "cache":
        figure = (
            f"<p class='warn'>{html.escape(page.get('plot_note') or '')}</p>"
            "<img src='brain_tstrip.png' alt='cached apple strip'>"
        )
    elif has_image:
        figure = (
            f"<p class='note'>{html.escape(page.get('plot_note') or '')}</p>"
            "<img src='brain_tstrip.png' alt='TRIBE signed brain strip'>"
        )
    else:
        figure = (
            "<p class='warn'>这次没有脑图。"
            f"{html.escape(page.get('plot_note') or '未写出原因')}</p>"
        )
    step = notes.get("max_step_over_median")
    step_txt = "空" if step is None else html.escape(str(step))
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>工作流 {html.escape(str(page.get('sample_id') or ''))}</title>
<style>
:root {{
  --bg: #F4F5F7;
  --surface: #FFFFFF;
  --ink: #1C1F24;
  --muted: #6B7078;
  --line: #E6E8EC;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif; }}
.wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 40px;
  display: flex; flex-direction: column; gap: 16px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 24px; }}
h1 {{ font-size: 20px; margin: 0 0 8px; font-weight: 650; }}
h2 {{ font-size: 15px; margin: 0 0 16px; font-weight: 650; }}
.note {{ color: var(--muted); font-size: 13px; line-height: 1.55; margin: 0; }}
.warn {{ color: #8A4B3A; font-size: 13px; }}
.paths {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
.kicker {{ color: var(--muted); font-size: 12px; margin: 0 0 8px; }}
.path {{ font-family: ui-monospace, monospace; font-size: 12px; word-break: break-all; margin: 0; }}
.figure img {{ width: 100%; height: auto; display: block; background: var(--surface); }}
.bar-row {{ display: grid; grid-template-columns: 220px 1fr 200px; gap: 16px;
  align-items: center; margin: 8px 0; }}
.track {{ position: relative; height: 10px; border-radius: 10px;
  background: linear-gradient(to right, #d73027, #fee08b, #1a9850); }}
.mark {{ position: absolute; top: -4px; width: 3px; height: 18px; background: #1C1F24; border-radius: 2px; }}
.raw {{ font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); }}
@media (max-width: 860px) {{
  .paths, .bar-row {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header class="card">
<h1>工作流 · {html.escape(str(page.get('sample_id') or ''))}</h1>
<p class="note">色条表示数值合同完整度，左红右绿。claim_scope = {html.escape(str(notes.get('claim_scope')))}。
evidence_confidence 与 recommended_training_weight 保持为空。
max_step_over_median = {step_txt}，只作描述。</p>
</header>
<section class="card">
<h2>输入</h2>
<div class="paths">
<article><p class="kicker">图片</p><p class="path">{html.escape(str(inputs.get('image_path') or '（无）'))}</p></article>
<article><p class="kicker">预测 npy</p><p class="path">{html.escape(str(inputs.get('preds_path') or '（无）'))}</p></article>
<article><p class="kicker">灰屏对照</p><p class="path">{html.escape(str(inputs.get('gray_control_path') or '（无）'))}</p></article>
</div>
</section>
<section class="card">
<h2>说明</h2>
<p class="note">{html.escape(page.get('protocol') or '')}</p>
</section>
<section class="card figure">
<h2>TRIBE 脑图</h2>
{figure}
</section>
<section class="card">
<h2>数值合同</h2>
{rows}
</section>
</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _bar_row(bar: dict[str, Any]) -> str:
    score = bar.get("score")
    label = html.escape(str(bar.get("label") or bar.get("id")))
    raw = html.escape(json.dumps(bar.get("raw"), ensure_ascii=False))
    if score is None:
        return (
            f"<div class='bar-row'><div>{label}</div>"
            "<div class='track' style='background:#d0d0d0'></div>"
            f"<div class='raw'>{raw}</div></div>"
        )
    pct = max(0.0, min(1.0, float(score))) * 100.0
    return (
        f"<div class='bar-row'><div>{label}</div>"
        f"<div class='track'><div class='mark' style='left:{pct:.2f}%'></div></div>"
        f"<div class='raw'>{raw}</div></div>"
    )


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Write workflow.html for an existing run")
    parser.add_argument("sample_dir", type=Path)
    args = parser.parse_args()
    page = publish_existing_run(args.sample_dir)
    print(page["html_path"])
    if page.get("image_source") == "missing":
        print(page.get("plot_note") or "no brain strip", file=sys.stderr)


if __name__ == "__main__":
    main()
