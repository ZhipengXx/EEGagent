"""TRIBE worker launcher and standalone entry.

The standalone ``__main__`` path must not import ``react_agent``: it runs under
the trib ev2 Python whose torch matches the local NVIDIA driver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class WorkerError(RuntimeError):
    """Subprocess worker failed."""


def worker_script() -> Path:
    return Path(__file__).resolve()


def launch(
    python_executable: str,
    request_path: Path,
    response_path: Path,
    *,
    timeout_s: float | None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Run the standalone worker with a JSON request file. ``shell=False``."""
    cmd = [
        str(python_executable),
        str(worker_script()),
        "--request",
        str(Path(request_path).resolve()),
        "--response",
        str(Path(response_path).resolve()),
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    log_path = Path(response_path).with_suffix(".log")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        shell=False,
        env=env,
    )
    log_path.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""), encoding="utf-8")
    if Path(response_path).is_file():
        payload = json.loads(Path(response_path).read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            return payload
        if proc.returncode != 0 or payload.get("status") == "error":
            raise WorkerError(str(payload.get("error") or payload))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise WorkerError(f"worker exit {proc.returncode}: {tail}")
    raise WorkerError("worker produced no response")


def launch_write_video(
    *,
    python_executable: str,
    repo_path: str | None,
    frames_dir: Path | None,
    image_path: Path | None,
    size_hw: tuple[int, int],
    video_path: Path,
    profile: Any,
    gray_only: bool,
    timeout_s: float = 180.0,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Encode a protocol video in the trib ev2 environment (moviepy)."""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    req = video_path.with_suffix(".write_req.json")
    resp = video_path.with_suffix(".write_resp.json")
    proto = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
    req.write_text(
        json.dumps(
            {
                "mode": "write_video",
                "repo_path": repo_path or "/home/zxuff/data/tribev2",
                "image_path": str(image_path) if image_path else None,
                "size_hw": list(size_hw),
                "video_path": str(video_path),
                "gray_only": bool(gray_only),
                "protocol": proto,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return launch(
        python_executable,
        req,
        resp,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )


def launch_predict(
    *,
    python_executable: str,
    repo_path: str,
    video_path: Path,
    duration_s: float,
    onset_s: float,
    checkpoint: str,
    cache_folder: Path,
    device: str,
    out_npy: Path,
    out_meta: Path,
    timeout_s: float | None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Run TribeModel.predict in the trib ev2 environment."""
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    req = out_npy.with_suffix(".predict_req.json")
    resp = out_npy.with_suffix(".predict_resp.json")
    req.write_text(
        json.dumps(
            {
                "mode": "predict",
                "repo_path": repo_path,
                "video_path": str(Path(video_path).resolve()),
                "duration_s": float(duration_s),
                "onset_s": float(onset_s),
                "checkpoint": checkpoint,
                "cache_folder": str(Path(cache_folder)),
                "device": device,
                "out_npy": str(out_npy),
                "out_meta": str(Path(out_meta)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return launch(
        python_executable,
        req,
        resp,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )


def _standalone_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="tribe-worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = _dispatch(request)
        response_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}),
            encoding="utf-8",
        )
        raise SystemExit(1)


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode") or "predict"
    repo = Path(request.get("repo_path") or "/home/zxuff/data/tribev2")
    scripts = repo / "scripts"
    for path in (str(repo), str(scripts)):
        if path not in sys.path:
            sys.path.insert(0, path)
    if mode == "write_video":
        return _write_video(request)
    if mode == "predict":
        return _predict(request)
    raise ValueError(f"unknown worker mode: {mode}")


def _stim_cfg(protocol: dict[str, Any]):
    from stim_protocol import StimConfig

    return StimConfig(
        protocol=str(protocol.get("protocol") or "static_gray4_image1_gray11_tribev2_v1"),
        fps=int(protocol.get("fps") or 10),
        pre_gray_s=float(protocol.get("pre_gray_s") or 4.0),
        image_s=float(protocol.get("image_s") or 1.0),
        post_gray_s=float(protocol.get("post_gray_s") or 11.0),
        gray_rgb=tuple(protocol.get("gray_rgb") or (128, 128, 128)),
        fixation=bool(protocol.get("fixation") or False),
        letterbox=bool(protocol.get("letterbox") or False),
    )


def _write_video(request: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path as P

    from stim_protocol import build_frames, load_rgb, write_protocol_video

    cfg = _stim_cfg(request.get("protocol") or {})
    video_path = P(request["video_path"])
    size_hw = tuple(int(x) for x in request["size_hw"])
    image = None
    if request.get("image_path") and not request.get("gray_only"):
        image = load_rgb(P(request["image_path"]))
        size_hw = (int(image.shape[0]), int(image.shape[1]))
    frames = build_frames(image, cfg, size_hw=size_hw)
    write_protocol_video(frames, video_path, cfg)
    return {
        "status": "ok",
        "mode": "write_video",
        "video_path": str(video_path),
        "n_frames": len(frames),
        "size_hw": list(size_hw),
    }


def _predict(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    from tribev2 import TribeModel

    video_path = Path(request["video_path"])
    out_npy = Path(request["out_npy"])
    out_meta = Path(request["out_meta"])
    cache_folder = Path(request["cache_folder"])
    cache_folder.mkdir(parents=True, exist_ok=True)
    requested = str(request.get("device") or "cuda")
    device = requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = TribeModel.from_pretrained(
        str(request.get("checkpoint") or "facebook/tribev2"),
        cache_folder=str(cache_folder),
        device=device,
    )
    inner = model._model
    predictor = getattr(inner, "predictor", None)
    if predictor is not None:
        n_rows = int(predictor.weights.shape[0])
        if not getattr(predictor, "average_subjects", False):
            predictor.average_subjects = True
        if getattr(predictor, "subject_dropout", None) in (None, 0, 0.0):
            predictor.subject_dropout = 0.1
        predictor.n_subjects = 0 if n_rows == 1 else n_rows - 1
    duration_s = float(request["duration_s"])
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": str(video_path),
                "start": 0.0,
                "duration": duration_s,
                "timeline": "video",
                "subject": "default",
            }
        ]
    )
    preds, segments = model.predict(events, verbose=False)
    array = np.asarray(preds)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, array)
    seg_rows = []
    for item in segments:
        seg_rows.append(
            {
                "start": float(getattr(item, "start", item.get("start") if isinstance(item, dict) else 0.0)),
                "duration": float(
                    getattr(item, "duration", item.get("duration") if isinstance(item, dict) else 1.0)
                ),
            }
        )
    meta = {
        "status": "ok",
        "preds_path": str(out_npy),
        "shape": list(array.shape),
        "segments": seg_rows,
        "device": device,
        "requested_device": requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "video_path": str(video_path),
        "duration_s": duration_s,
        "onset_s": float(request.get("onset_s") or 4.0),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["status"] = "ok"
    return meta


if __name__ == "__main__":
    _standalone_main()
