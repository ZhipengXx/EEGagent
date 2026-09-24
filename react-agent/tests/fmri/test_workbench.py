"""Path guard for the local workbench. Does not start a pipeline."""

from __future__ import annotations

import io
from pathlib import Path

from react_agent.fmri.workbench import WorkbenchHandler, resolve_run_file


def test_resolve_run_file_allows_workflow_page(tmp_path: Path) -> None:
    page = tmp_path / "sample" / "workflow.html"
    page.parent.mkdir(parents=True)
    page.write_text("<p>ok</p>", encoding="utf-8")
    png = page.parent / "brain_tstrip.png"
    png.write_bytes(b"\x89PNG")
    assert resolve_run_file(tmp_path, "/runs/sample/workflow.html") == page.resolve()
    assert resolve_run_file(tmp_path, "/runs/sample/brain_tstrip.png") == png.resolve()


def test_resolve_run_file_rejects_escape_and_other_types(tmp_path: Path) -> None:
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    nested = tmp_path / "sample" / "report.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="utf-8")
    assert resolve_run_file(tmp_path, "/runs/../secret.txt") is None
    assert resolve_run_file(tmp_path, "/runs/sample/report.json") is None
    assert resolve_run_file(tmp_path, "/etc/passwd") is None


def test_page_assets_are_not_stored(tmp_path: Path) -> None:
    handler = WorkbenchHandler.__new__(WorkbenchHandler)
    headers: list[tuple[str, object]] = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: headers.append(("status", status))
    handler.send_header = lambda key, value: headers.append((key, value))
    handler.end_headers = lambda: None
    page = tmp_path / "index.html"
    page.write_text("<p>ok</p>", encoding="utf-8")
    handler._file(page, "text/html; charset=utf-8", store=False)
    assert ("Cache-Control", "no-store") in headers
    headers.clear()
    handler.wfile = io.BytesIO()
    handler._file(page, "image/png")
    assert ("Cache-Control", "no-store") not in headers
