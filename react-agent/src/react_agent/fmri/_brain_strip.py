"""Signed left/right/posterior strips. This file does not import the agent package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

MESH = "fsaverage5"
N_VERTICES = 20484
VIEWS = ("left", "right", "posterior")
TRIBE_ROOT = Path("/home/zxuff/data/tribev2")
TRIBE_PYTHON = Path("/home/zxuff/miniconda3/envs/tribev2/bin/python")


def render_brain_strip(
    items: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    onset_s: float = 4.0,
    dt: float = 1.0,
) -> tuple[bool, str]:
    """Draw the current sample. Falls back to the TRIBE interpreter if needed."""
    if not items:
        return False, "no arrays to plot"
    try:
        preds0 = np.asarray(items[0][1])
    except Exception as exc:  # noqa: BLE001
        return False, f"array unreadable: {exc}"
    if preds0.ndim != 2 or int(preds0.shape[1]) != N_VERTICES:
        return False, (
            f"expected [T,{N_VERTICES}] for mesh {MESH}, got {tuple(preds0.shape)}"
        )
    if os.environ.get("FMRI_BRAIN_PLOT_CHILD") == "1" or _plotbrain_importable():
        return _render_inprocess(items, out_path, onset_s=onset_s, dt=dt)
    return _render_subprocess(items, out_path, onset_s=onset_s, dt=dt)


def _plotbrain_importable() -> bool:
    try:
        from tribev2.plotting import PlotBrain  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _render_inprocess(
    items: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    onset_s: float,
    dt: float,
) -> tuple[bool, str]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tribev2.plotting import PlotBrain
    except Exception as exc:  # noqa: BLE001
        return False, f"plot backend unavailable: {exc}"
    try:
        preds0 = np.asarray(items[0][1])
        plotter = PlotBrain(mesh=MESH)
        vmax = 1e-6
        for _, preds in items:
            arr = np.asarray(preds, dtype=np.float64)
            vmax = max(vmax, float(np.nanpercentile(np.abs(arr), 99)))
        vmin = -vmax
        n_tr = int(preds0.shape[0])
        n_views = len(VIEWS)
        n_items = len(items)
        fig, axes = plt.subplots(
            n_items * n_views,
            n_tr,
            figsize=(1.45 * n_tr, 1.55 * n_items * n_views),
            gridspec_kw={"wspace": 0.02, "hspace": 0.08},
        )
        axes = np.array(axes).reshape(n_items * n_views, n_tr)
        for r_item, (title, preds) in enumerate(items):
            arr = np.asarray(preds)
            for r_view, view in enumerate(VIEWS):
                r = r_item * n_views + r_view
                for c in range(n_tr):
                    ax = axes[r, c]
                    plotter.plot_surf(
                        arr[c],
                        axes=ax,
                        views=view,
                        cmap="coolwarm",
                        vmin=vmin,
                        vmax=vmax,
                        symmetric_cbar=True,
                    )
                    ts = c * dt - onset_s
                    mark = ""
                    if abs(ts - 0.0) < 0.51:
                        mark = " onset"
                    elif abs(ts - 1.0) < 0.51:
                        mark = " offset"
                    if r_view == 0:
                        ax.set_title(f"t_stim={ts:.0f}s{mark}", fontsize=7)
                    if c == 0:
                        ax.set_ylabel(f"{title}\n{view}", fontsize=7)
        fig.suptitle(
            f"image16 signed  cmap=coolwarm  shared [{vmin:.3g},{vmax:.3g}]  "
            f"onset t_stim=0 (t_vid={onset_s:g}s)  mesh={MESH}  no extra +5 shift",
            fontsize=10,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        return False, f"plot failed: {exc}"
    return True, f"shared scale ±{vmax:.4g}"


def _render_subprocess(
    items: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    onset_s: float,
    dt: float,
) -> tuple[bool, str]:
    if not TRIBE_PYTHON.is_file():
        return False, "plot backend unavailable: tribev2 python missing"
    bundle = out_path.with_suffix(".plot.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bundle, **{title: np.asarray(arr) for title, arr in items})
    env = os.environ.copy()
    env["FMRI_BRAIN_PLOT_CHILD"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYVISTA_OFF_SCREEN"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TRIBE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    try:
        proc = subprocess.run(
            [
                str(TRIBE_PYTHON),
                str(Path(__file__).resolve()),
                str(bundle),
                str(out_path),
                str(onset_s),
                str(dt),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"plot backend unavailable: {exc}"
    finally:
        bundle.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        return False, f"plot failed: {detail}"
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return False, "plot failed: empty plotter output"
    try:
        payload = json.loads(line[-1])
    except json.JSONDecodeError:
        return False, "plot failed: plotter output was not json"
    return bool(payload.get("ok")), str(payload.get("note") or "")


def render_frame_views(
    vector: np.ndarray,
    out_dir: Path,
    *,
    vmin: float,
    vmax: float,
) -> tuple[bool, str]:
    """Draw left, right, and posterior into separate files. No English scale title."""
    arr = np.asarray(vector)
    if arr.ndim != 1 or int(arr.shape[0]) != N_VERTICES:
        return False, f"expected [{N_VERTICES}], got {tuple(arr.shape)}"
    if os.environ.get("FMRI_BRAIN_PLOT_CHILD") == "1" or _plotbrain_importable():
        return _render_frame_inprocess(arr, out_dir, vmin=vmin, vmax=vmax)
    return _render_frame_subprocess(arr, out_dir, vmin=vmin, vmax=vmax)


def _render_frame_inprocess(
    vector: np.ndarray,
    out_dir: Path,
    *,
    vmin: float,
    vmax: float,
) -> tuple[bool, str]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tribev2.plotting import PlotBrain
    except Exception as exc:  # noqa: BLE001
        return False, f"plot backend unavailable: {exc}"
    try:
        plotter = PlotBrain(mesh=MESH)
        out_dir.mkdir(parents=True, exist_ok=True)
        for view in VIEWS:
            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))
            plotter.plot_surf(
                vector,
                axes=ax,
                views=view,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=True,
            )
            ax.set_title(view, fontsize=11)
            fig.savefig(out_dir / f"{view}.png", dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        return False, f"plot failed: {exc}"
    return True, f"shared scale ±{vmax:.4g}"


def _render_frame_subprocess(
    vector: np.ndarray,
    out_dir: Path,
    *,
    vmin: float,
    vmax: float,
) -> tuple[bool, str]:
    if not TRIBE_PYTHON.is_file():
        return False, "plot backend unavailable: tribev2 python missing"
    bundle = out_dir.with_suffix(".frame.npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(bundle, np.asarray(vector))
    env = os.environ.copy()
    env["FMRI_BRAIN_PLOT_CHILD"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYVISTA_OFF_SCREEN"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TRIBE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    try:
        proc = subprocess.run(
            [
                str(TRIBE_PYTHON),
                str(Path(__file__).resolve()),
                "--frame",
                str(bundle),
                str(out_dir),
                str(vmin),
                str(vmax),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"plot backend unavailable: {exc}"
    finally:
        bundle.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        return False, f"plot failed: {detail}"
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return False, "plot failed: empty plotter output"
    try:
        payload = json.loads(line[-1])
    except json.JSONDecodeError:
        return False, "plot failed: plotter output was not json"
    return bool(payload.get("ok")), str(payload.get("note") or "")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--frame":
        _, _, bundle, out_path, vmin, vmax = sys.argv[:6]
        ok, note = render_frame_views(
            np.load(bundle),
            Path(out_path),
            vmin=float(vmin),
            vmax=float(vmax),
        )
        print(json.dumps({"ok": ok, "note": note}))
        if not ok:
            raise SystemExit(1)
        return
    bundle, out_path, onset_s, dt = sys.argv[1:5]
    data = np.load(bundle)
    items = [(name, data[name]) for name in data.files]
    ok, note = render_brain_strip(
        items, Path(out_path), onset_s=float(onset_s), dt=float(dt)
    )
    print(json.dumps({"ok": ok, "note": note}))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
