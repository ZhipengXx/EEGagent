"""Local fMRI workbench. Binds to 127.0.0.1 and serves a read-only run view."""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

_REPO = Path(__file__).resolve().parents[3]
RUNS = (_REPO / "runs").resolve()
RESEARCH = (RUNS / "eeg_research").resolve()
CONFIGS = (_REPO / "configs").resolve()
DEFAULT_CONFIG = "fmri_check_tribe_image16.yaml"
DEFAULT_OUT = "workbench"
_ALLOWED_SUFFIXES = {".html", ".png", ".jpg", ".jpeg", ".webp"}
_POLICIES = {"", "rule", "hybrid", "planned"}
_BACKENDS = {"", "none", "mock", "deepseek"}

_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",
    "message": "空闲",
    "view": None,
    "error": None,
}


def config_choices() -> list[Path]:
    files = sorted(CONFIGS.glob("fmri_check*.yaml"))
    return [path for path in files if path.is_file()]


def list_workflows() -> list[Path]:
    if not RUNS.is_dir():
        return []
    pages = [path for path in RUNS.glob("**/workflow.html") if path.is_file()]
    pages.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return pages[:40]


def resolve_run_file(runs_root: Path, url_path: str) -> Path | None:
    """Map /runs/... to a file inside runs_root. Reject traversal and other types."""
    parsed = urlparse(url_path)
    raw = unquote(parsed.path)
    if not raw.startswith("/runs/"):
        return None
    rel = raw[len("/runs/") :]
    if not rel or rel.endswith("/"):
        return None
    root = runs_root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    if not candidate.is_file():
        return None
    return candidate


def runs_url(path: Path) -> str:
    rel = path.resolve().relative_to(RUNS)
    return "/runs/" + rel.as_posix()


def _public_job() -> dict[str, Any]:
    with _lock:
        return {
            "status": _job["status"],
            "message": _job["message"],
            "view": _job["view"],
            "error": _job["error"],
        }


def _set_job(**fields: Any) -> None:
    with _lock:
        _job.update(fields)


def _validate_form(form: dict[str, str]) -> dict[str, Any]:
    image = (form.get("image_path") or "").strip()
    if not image:
        raise ValueError("请填写图片路径")
    config_name = Path((form.get("config") or DEFAULT_CONFIG).strip()).name
    config_path = (CONFIGS / config_name).resolve()
    allowed = {path.resolve() for path in config_choices()}
    if config_path not in allowed:
        raise ValueError("配置必须是 configs 里的 fmri_check 文件")
    screen_depth = (form.get("screen_depth") or "standard").strip()
    if screen_depth not in {"standard", "deep"}:
        raise ValueError("筛查深度只能是标准筛查或深层筛查")
    out_name = (form.get("out_dir") or DEFAULT_OUT).strip().strip("/")
    if not out_name:
        out_name = DEFAULT_OUT
    if screen_depth == "deep" and out_name == DEFAULT_OUT:
        out_name = DEFAULT_OUT + "_deep"
    out_dir = (RUNS / out_name).resolve()
    try:
        out_dir.relative_to(RUNS)
    except ValueError as exc:
        raise ValueError("输出目录必须在 runs/ 下面") from exc
    policy = (form.get("policy") or "").strip()
    backend = (form.get("backend") or "").strip()
    if policy not in _POLICIES or backend not in _BACKENDS:
        raise ValueError("policy 或 backend 不在允许列表里")
    sample_id = (form.get("sample_id") or "").strip() or None
    return {
        "image_path": image,
        "config_path": config_path,
        "out_dir": out_dir,
        "policy": policy or None,
        "backend": backend or None,
        "sample_id": sample_id,
        "screen_depth": screen_depth,
    }


def load_job_config(fields: dict[str, Any]) -> Any:
    """Load the chosen YAML and apply the screening depth without editing the file."""
    from react_agent.fmri.config import load_config

    depth = fields.get("screen_depth") or "standard"
    cfg = load_config(fields["config_path"], cli_overrides={"diagnostic": {"screen_depth": depth}})
    if depth == "deep" and not cfg.diagnostic.enabled:
        raise ValueError("深层筛查需要选择分层数值检查")
    return cfg


def _execute(fields: dict[str, Any]) -> None:
    from dotenv import load_dotenv

    from react_agent.fmri.config import resolve_policy_backend
    from react_agent.fmri.pipeline import run_image_pipeline

    try:
        load_dotenv()
        cfg = load_job_config(fields)
        policy, backend = resolve_policy_backend(
            cfg, policy=fields["policy"], backend=fields["backend"]
        )
        report = asyncio.run(
            run_image_pipeline(
                fields["image_path"],
                config=cfg,
                out_dir=fields["out_dir"],
                sample_id=fields["sample_id"],
                backend=backend,
                policy=policy,
            )
        )
    except Exception as exc:  # noqa: BLE001
        _set_job(status="error", message="运行失败", view=None, error=str(exc))
        return
    if report.get("pipeline_status") != "ok":
        reason = report.get("stop_reason") or report.get("pipeline_status") or "failed"
        detail = report.get("error") or reason
        _set_job(status="error", message=str(reason), view=None, error=str(detail))
        return
    safe = report.get("safe_sample_id")
    run_dir = (Path(fields["out_dir"]) / safe) if safe else None
    if run_dir is None or not run_dir.is_dir():
        _set_job(status="error", message="检查结束，但没有样本目录", view=None, error=None)
        return
    try:
        view = run_dir.resolve().relative_to(RUNS).as_posix()
    except ValueError:
        _set_job(status="error", message="结果不在 runs/ 里", view=None, error=str(run_dir))
        return
    _set_job(status="done", message=f"完成 {report.get('sample_id') or ''}".strip(), view=view, error=None)


def start_job(form: dict[str, str]) -> tuple[int, dict[str, Any]]:
    try:
        fields = _validate_form(form)
    except ValueError as exc:
        return 400, {"status": "error", "message": str(exc), "view": None, "error": str(exc)}
    with _lock:
        if _job["status"] == "running":
            return 409, {
                "status": _job["status"],
                "message": "已有任务在跑",
                "view": _job["view"],
                "error": _job["error"],
            }
        _job.update(
            {
                "status": "running",
                "message": "运行中。缓存未命中时会跑 TRIBE；planned + deepseek 会调用接口。",
                "view": None,
                "error": None,
            }
        )
        accepted = {
            "status": _job["status"],
            "message": _job["message"],
            "view": _job["view"],
            "error": _job["error"],
        }
    threading.Thread(target=_execute, args=(fields,), daemon=True).start()
    return 202, accepted


def _form_value(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key) or [""]
    return values[0]


STATIC = Path(__file__).resolve().parent / "workbench_static"
_ARTIFACTS = {
    "report.json",
    "report.md",
    "brain_tstrip.png",
    "response_rms.json",
    "plan.json",
    "plan_history.jsonl",
    "events.jsonl",
    "llm_usage.json",
    "memory_retrieval.json",
    "episode.json",
    "roi_profile.json",
    "workflow.json",
}
_MODES = {
    "quick": (
        "fmri_check_tribe_image16.yaml",
        "快速数值检查",
        "使用规则策略，不调用决策 LLM。",
        "policy=rule，backend=none",
    ),
    "diagnostic": (
        "fmri_check_tribe_image16_diagnostic_v1_4.yaml",
        "分层诊断",
        "使用 planned + DeepSeek，按证据做进一步检查。",
        "policy=planned，backend=deepseek，会调用接口",
    ),
}


def _query(path: str) -> dict[str, str]:
    parsed = parse_qs(urlparse(path).query, keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items() if values}


def _json_bytes(payload: Any, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _research_campaigns() -> list[dict[str, str]]:
    if not RESEARCH.is_dir():
        return []
    rows = []
    for path in sorted(RESEARCH.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if (path / "task_card.json").is_file():
            rows.append({"campaign_id": path.name})
    return rows[:12]


def submit_retrieval(form: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """Start a probe, dry-run, or capped retrieval trial."""
    from react_agent.eeg_training.launch import launch_design, parse_design

    action = form.get("action") or "probe"
    if action not in {"discover", "probe", "dry_run", "run"}:
        return 400, {"ok": False, "started": False, "blockers": ["unknown_action"]}
    try:
        design = parse_design(form)
    except ValueError as exc:
        return 400, {"ok": False, "started": False, "discovered": False, "subjects": [], "gpus": [], "blockers": [str(exc)]}
    if action == "discover":
        from react_agent.eeg_training.launch import discover_design
        from react_agent.eeg_training.protocol import SplitError

        try:
            payload = discover_design(design)
        except SplitError as exc:
            return 400, {"ok": False, "started": False, "discovered": False, "subjects": [], "gpus": [], "blockers": [str(exc)]}
        payload["campaigns"] = _research_campaigns()
        return 200, payload
    payload = launch_design(design, RESEARCH / design.campaign_id(), action)
    if action == "run" and design.policy == "agentic":
        from react_agent.eeg_research.agentic.cli import DEFAULT_ROOT, open_agentic_run
        from react_agent.eeg_research.agentic.execution_protocol import unsupported_agentic_reason

        reason = unsupported_agentic_reason(design)
        if reason:
            payload["ok"] = False
            payload["started"] = False
            payload["blockers"] = list(payload.get("blockers") or []) + [reason]
            payload["log"] = list(payload.get("log") or []) + [reason]
        elif not payload.get("blockers"):
            payload = open_agentic_run(design, payload, request_id=(form.get("request_id") or "").strip(), root=DEFAULT_ROOT)
    payload["campaigns"] = _research_campaigns()
    return 200, payload


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "fmri-workbench/1.5"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._file(STATIC / "index.html", "text/html; charset=utf-8", store=False)
            return
        if path.startswith("/static/"):
            name = Path(path).name
            target = STATIC / name
            if name not in {"app.css", "app.js"} or not target.is_file():
                self._send(404, "text/plain; charset=utf-8", b"not found")
                return
            mime = "text/css" if name.endswith(".css") else "text/javascript; charset=utf-8"
            self._file(target, mime, store=False)
            return
        if path == "/status":
            status, mime, body = _json_bytes(_public_job())
            self._send(status, mime, body)
            return
        if path == "/api/runs":
            from react_agent.fmri.workbench_view import list_runs

            status, mime, body = _json_bytes({"runs": list_runs(RUNS)})
            self._send(status, mime, body)
            return
        if path == "/api/view":
            from react_agent.fmri.workbench_view import build_run_view

            view = build_run_view(_query(self.path).get("run", ""), RUNS)
            if view is None:
                self._send(404, "application/json; charset=utf-8", b'{"error":"run not found"}')
                return
            status, mime, body = _json_bytes(view)
            self._send(status, mime, body)
            return
        if path == "/api/configs":
            rows = [
                {"id": key, "config": item[0], "label": item[1], "note": item[2], "resolved": item[3]}
                for key, item in _MODES.items()
            ]
            status, mime, body = _json_bytes({"modes": rows, "default": "quick"})
            self._send(status, mime, body)
            return
        if path == "/api/frame":
            self._frame()
            return
        if path == "/api/artifact":
            self._artifact()
            return
        if path == "/api/research":
            from react_agent.eeg_research.report import read_campaign

            payload = read_campaign(RESEARCH, _query(self.path).get("campaign", ""))
            if payload is None:
                self._send(404, "application/json; charset=utf-8", b'{"error":"campaign not found"}')
                return
            status, mime, body = _json_bytes(payload)
            self._send(status, mime, body)
            return
        if path == "/api/agentic_status":
            from react_agent.eeg_research.agentic.view import agentic_status

            payload = agentic_status(campaign=_query(self.path).get("campaign", ""))
            status, mime, body = _json_bytes(payload)
            self._send(status, mime, body, store=False)
            return
        if path == "/api/train_status":
            from react_agent.eeg_training.train_entry import read_train_status

            payload = read_train_status(RESEARCH, _query(self.path).get("campaign", ""))
            if payload is None:
                self._send(404, "application/json; charset=utf-8", b'{"error":"campaign not found"}')
                return
            status, mime, body = _json_bytes(payload)
            self._send(status, mime, body)
            return
        if path == "/api/retrieval":
            from react_agent.eeg_training.protocol import data_root, gpu_seconds, options
            from react_agent.eeg_training.train_entry import latest_train_status

            payload = {
                "ok": True,
                "started": False,
                "discovered": False,
                "subjects": [],
                "gpus": [],
                "options": options(),
                "campaigns": _research_campaigns(),
                "data_root": str(data_root()),
                "split": None,
                "blockers": [],
                "test_result": None,
                "api_usd": None,
                "gpu_seconds": gpu_seconds(),
                "latest": latest_train_status(RESEARCH),
            }
            status, mime, body = _json_bytes(payload)
            self._send(status, mime, body)
            return
        target = resolve_run_file(RUNS, self.path)
        if target is None:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._file(target, mime)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/retrieval":
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            parsed = parse_qs(raw, keep_blank_values=True)
            form = {key: _form_value(parsed, key) for key in parsed}
            code, payload = submit_retrieval(form)
            status, mime, body = _json_bytes(payload, code)
            self._send(status, mime, body)
            return
        if path == "/api/agentic_control":
            from react_agent.eeg_research.agentic.view import control

            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            parsed = parse_qs(raw, keep_blank_values=True)
            form = {key: _form_value(parsed, key) for key in parsed}
            payload = control(form.get("action", ""), form.get("campaign", ""))
            status, mime, body = _json_bytes(payload, 200 if payload.get("ok") else 400)
            self._send(status, mime, body)
            return
        if path != "/run":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        parsed = parse_qs(raw, keep_blank_values=True)
        form = {key: _form_value(parsed, key) for key in parsed}
        mode = _MODES.get(form.get("check_mode") or "")
        if mode and not form.get("config"):
            form["config"] = mode[0]
        status, payload = start_job(form)
        code, mime, body = _json_bytes(payload, status)
        self._send(code, mime, body)

    def _frame(self) -> None:
        from react_agent.fmri.workbench_view import ensure_frame

        query = _query(self.path)
        try:
            index = int(query.get("index", "0"))
        except ValueError:
            self._send(400, "text/plain; charset=utf-8", b"bad index")
            return
        path, note = ensure_frame(
            query.get("run", ""),
            query.get("mode", ""),
            index,
            RUNS,
            view_name=query.get("view") or "left",
        )
        if path is None:
            code, mime, body = _json_bytes({"error": note or "frame unavailable"}, 404)
            self._send(code, mime, body)
            return
        self._file(path, "image/png")

    def _artifact(self) -> None:
        from react_agent.fmri.workbench_view import resolve_sample_dir

        query = _query(self.path)
        name = Path(query.get("file", "")).name
        if name not in _ARTIFACTS:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        sample = resolve_sample_dir(query.get("run", ""), RUNS)
        if sample is None:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        target = sample / name
        if not target.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._file(target, mime)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _file(self, path: Path, content_type: str, *, store: bool = True) -> None:
        if not path.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        self._send(200, content_type, path.read_bytes(), store=store)

    def _send(self, status: int, content_type: str, body: bytes, *, store: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fMRI workflow workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("workbench binds to 127.0.0.1 only")
    from dotenv import load_dotenv

    load_dotenv(_REPO / ".env", override=False)
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()

