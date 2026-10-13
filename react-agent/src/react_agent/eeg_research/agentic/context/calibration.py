"""Calibrate estimates only upward from the same role/model/mode usage.

Provider usage is evidence about a completed request, not an exact counting API
for a new request. Small unrelated roles cannot lower the current estimate.
"""
from __future__ import annotations
import math
import sqlite3
from pathlib import Path


def _connection(camp: Path):
    camp.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(camp / "context_sources.sqlite")
    conn.execute("CREATE TABLE IF NOT EXISTS context_token_calibration (call_id TEXT UNIQUE NOT NULL, role TEXT NOT NULL, model TEXT NOT NULL, mode TEXT NOT NULL, estimated INTEGER NOT NULL, observed INTEGER NOT NULL, created INTEGER PRIMARY KEY AUTOINCREMENT)")
    return conn


def record_usage(camp: Path, *, call_id: str, role: str, model: str, mode: str, estimated: int, observed: int | None):
    if not isinstance(observed, int) or observed < 0 or estimated <= 0:
        return
    # Explicit order rather than timestamps avoids out-of-order wall clocks.
    conn = _connection(camp)
    try:
        with conn:
            conn.execute("INSERT OR IGNORE INTO context_token_calibration(call_id,role,model,mode,estimated,observed) VALUES(?,?,?,?,?,?)",
                         (call_id, role, model, mode, estimated, observed))
    finally:
        conn.close()


def tokenizer_for(camp: Path, *, role: str, model: str, mode: str):
    from .budget import token_count
    path = camp / "context_sources.sqlite"
    scale = 1.0
    call_ids = []
    if path.is_file():
        conn = sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT estimated,observed,call_id FROM context_token_calibration WHERE role=? AND model=? AND mode=? ORDER BY created DESC LIMIT 8", (role,model,mode)).fetchall()
            call_ids = [row[2] for row in rows]
            scale = max([1.0] + [actual / expected for expected,actual,_ in rows if expected > 0])
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()
    def estimate(text):
        base, _, version = token_count(text)
        return {"tokens":math.ceil(base * scale), "precision":"estimate_usage_calibrated",
                "version":version + ":same_role_model_mode_raise_only.v1"}
    estimate.calibration_metadata = {"role":role,"model":model,"mode":mode,
        "source_call_ids":call_ids,"bounded_sample_count":len(call_ids),
        "scale":scale,"precision":"estimate", "policy":"raise_only"}
    return estimate
