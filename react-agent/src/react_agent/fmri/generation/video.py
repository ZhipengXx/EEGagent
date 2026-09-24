"""4s gray + 1s image + 11s gray at FPS=10. Inherits local stim_protocol parameters."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from react_agent.fmri.generation.schemas import (
    CODEC,
    GRAY_RGB,
    GenerationError,
    StimulusProfile,
)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB. Native size; no letterbox."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise GenerationError("pillow_missing", "Pillow is required to load the stimulus image") from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def gray_frame(size_hw: tuple[int, int], rgb: tuple[int, int, int] = GRAY_RGB) -> np.ndarray:
    """Solid gray canvas matching the image size."""
    height, width = size_hw
    return np.full((height, width, 3), rgb, dtype=np.uint8)


def build_frames(
    image: np.ndarray | None,
    profile: StimulusProfile,
    size_hw: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Build the 160-frame protocol (40 + 10 + 110 at FPS=10)."""
    if image is not None:
        size_hw = (int(image.shape[0]), int(image.shape[1]))
    if size_hw is None:
        raise GenerationError("size_required", "size_hw is required when image is None")
    gray = gray_frame(size_hw, profile.gray_rgb)
    frames: list[np.ndarray] = [gray] * profile.n_pre
    if image is None:
        frames.extend([gray] * profile.n_img)
    else:
        if image.shape[:2] != size_hw:
            raise GenerationError("size_mismatch", f"image size {image.shape[:2]} != {size_hw}")
        frames.extend([image] * profile.n_img)
    frames.extend([gray] * profile.n_post)
    if len(frames) != profile.n_frames:
        raise GenerationError("frame_count", f"frame count {len(frames)} != {profile.n_frames}")
    return frames


def frame_boundaries(profile: StimulusProfile) -> dict[str, Any]:
    """Inclusive frame indices for protocol segments."""
    return {
        "n_frames": profile.n_frames,
        "pre": [0, profile.n_pre],
        "image": [profile.n_pre, profile.n_pre + profile.n_img],
        "post": [profile.n_pre + profile.n_img, profile.n_frames],
        "last_pre": profile.n_pre - 1,
        "first_image": profile.n_pre,
        "last_image": profile.n_pre + profile.n_img - 1,
        "first_post": profile.n_pre + profile.n_img,
    }


def classify_frames(
    frames: list[np.ndarray],
    profile: StimulusProfile,
    *,
    expect_image: bool = True,
    tol: float = 12.0,
) -> dict[str, Any]:
    """Check gray/image boundaries on an in-memory frame list."""
    target = np.asarray(profile.gray_rgb, dtype=float)
    labels: list[str] = []
    for frame in frames:
        mean = frame.reshape(-1, 3).mean(axis=0)
        gray = float(np.max(np.abs(mean - target))) <= tol
        labels.append("gray" if gray else "image")
    n_pre = sum(1 for lab in labels[: profile.n_pre] if lab == "gray")
    n_img = sum(1 for lab in labels[profile.n_pre : profile.n_pre + profile.n_img] if lab == "image")
    n_post = sum(1 for lab in labels[profile.n_pre + profile.n_img :] if lab == "gray")
    leaked = [
        i
        for i, lab in enumerate(labels)
        if lab == "image"
        and not (profile.n_pre <= i < profile.n_pre + profile.n_img)
    ]
    if expect_image:
        ok = (
            len(frames) == profile.n_frames
            and n_pre == profile.n_pre
            and n_img == profile.n_img
            and n_post == profile.n_post
            and not leaked
        )
    else:
        ok = len(frames) == profile.n_frames and all(lab == "gray" for lab in labels)
    return {
        "n_frames": len(frames),
        "n_pre_gray_ok": n_pre,
        "n_image_ok": n_img,
        "n_post_gray_ok": n_post,
        "leaked_image_frame_indices": leaked,
        "protocol_ok": bool(ok),
        "boundaries": frame_boundaries(profile),
    }


def control_cache_key(
    size_hw: tuple[int, int],
    profile: StimulusProfile,
    *,
    checkpoint: str,
    missing_audio: str,
    missing_text: str,
    export_rule_id: str,
) -> str:
    """Control identity is independent of the image filename."""
    payload = {
        "size_hw": list(size_hw),
        "fps": profile.fps,
        "gray_rgb": list(profile.gray_rgb),
        "codec": profile.codec or CODEC,
        "profile_id": profile.profile_id,
        "pre_gray_s": profile.pre_gray_s,
        "image_s": profile.image_s,
        "post_gray_s": profile.post_gray_s,
        "checkpoint": checkpoint,
        "missing_audio": missing_audio,
        "missing_text": missing_text,
        "export_rule_id": export_rule_id,
        "letterbox": profile.letterbox,
        "fixation": profile.fixation,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def write_protocol_video(
    frames: list[np.ndarray],
    video_path: Path,
    profile: StimulusProfile,
    *,
    python_executable: str | None = None,
    repo_path: str | None = None,
) -> Path:
    """Encode libx264, no audio, tmp then atomic replace."""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = video_path.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    if shutil.which("ffmpeg"):
        _write_ffmpeg(frames, tmp, profile.fps)
    elif python_executable:
        from react_agent.fmri.generation.worker import launch_write_video

        launch_write_video(
            python_executable=python_executable,
            repo_path=repo_path,
            frames_dir=None,
            image_path=None,
            size_hw=(int(frames[0].shape[0]), int(frames[0].shape[1])),
            video_path=tmp,
            profile=profile,
            gray_only=all(
                float(np.max(np.abs(f.reshape(-1, 3).mean(0) - profile.gray_rgb))) <= 1.0
                for f in (frames[0], frames[profile.n_pre])
            )
            and len(frames) == profile.n_frames,
        )
        if not tmp.is_file():
            raise GenerationError("encode_failed", "worker write_video produced no file")
    else:
        raise GenerationError("encoder_missing", "ffmpeg or TRIBE worker python is required")
    tmp.replace(video_path)
    return video_path


def write_image_video(
    image_path: Path,
    video_path: Path,
    profile: StimulusProfile,
    *,
    python_executable: str | None = None,
    repo_path: str | None = None,
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    """Load the still, build frames, encode, and classify boundaries."""
    image = load_rgb(image_path)
    size_hw = (int(image.shape[0]), int(image.shape[1]))
    frames = build_frames(image, profile, size_hw=size_hw)
    report = classify_frames(frames, profile, expect_image=True)
    if not report["protocol_ok"]:
        raise GenerationError("protocol_frames", f"in-memory protocol failed: {report}")
    if shutil.which("ffmpeg"):
        write_protocol_video(frames, video_path, profile)
    elif python_executable:
        from react_agent.fmri.generation.worker import launch_write_video

        launch_write_video(
            python_executable=python_executable,
            repo_path=repo_path,
            frames_dir=None,
            image_path=image_path,
            size_hw=size_hw,
            video_path=video_path,
            profile=profile,
            gray_only=False,
        )
    else:
        write_protocol_video(frames, video_path, profile)
    return video_path, size_hw, report


def write_gray_video(
    size_hw: tuple[int, int],
    video_path: Path,
    profile: StimulusProfile,
    *,
    python_executable: str | None = None,
    repo_path: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Matching 16s gray control video for a canvas size."""
    frames = build_frames(None, profile, size_hw=size_hw)
    report = classify_frames(frames, profile, expect_image=False)
    if not report["protocol_ok"]:
        raise GenerationError("protocol_frames", f"gray protocol failed: {report}")
    if shutil.which("ffmpeg"):
        write_protocol_video(frames, video_path, profile)
    elif python_executable:
        from react_agent.fmri.generation.worker import launch_write_video

        launch_write_video(
            python_executable=python_executable,
            repo_path=repo_path,
            frames_dir=None,
            image_path=None,
            size_hw=size_hw,
            video_path=video_path,
            profile=profile,
            gray_only=True,
        )
    else:
        write_protocol_video(frames, video_path, profile)
    return video_path, report


def probe_video(video_path: Path, profile: StimulusProfile) -> dict[str, Any]:
    """Decode duration / frame count via ffprobe when available."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise GenerationError("video_missing", f"video missing: {video_path}")
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration,avg_frame_rate,width,height",
            "-of",
            "json",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if proc.returncode != 0:
            raise GenerationError("ffprobe_failed", proc.stderr[-400:])
        payload = json.loads(proc.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        rate = stream.get("avg_frame_rate") or "0/1"
        num, den = (rate.split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den) else 0.0
        duration = float(stream.get("duration") or 0.0)
        n_frames = int(stream.get("nb_frames") or round(duration * fps))
        return {
            "duration": duration,
            "fps": fps,
            "n_frames": n_frames,
            "size": [int(stream.get("width") or 0), int(stream.get("height") or 0)],
            "expected_n_frames": profile.n_frames,
            "expected_duration": profile.total_s,
            "duration_ok": abs(duration - profile.total_s) < 0.15,
            "frames_ok": n_frames == profile.n_frames,
        }
    stat = video_path.stat()
    return {
        "duration": None,
        "fps": profile.fps,
        "n_frames": None,
        "size": None,
        "expected_n_frames": profile.n_frames,
        "expected_duration": profile.total_s,
        "duration_ok": stat.st_size > 0,
        "frames_ok": stat.st_size > 0,
        "probe": "size_only",
    }


def _write_ffmpeg(frames: list[np.ndarray], tmp: Path, fps: int) -> None:
    height, width = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
    except BrokenPipeError as exc:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")
        raise GenerationError("ffmpeg_pipe", stderr[-400:]) from exc
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise GenerationError("ffmpeg_failed", (stderr or b"").decode("utf-8", errors="replace")[-400:])
    _ = stdout
