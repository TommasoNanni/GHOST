"""Render the FUSED body of one person over a whole clip, on a flat background.

This is the pose GHOST actually reports: every camera's SAM3D body pose is
combined by the geodesic median (`median_fuse`), and the root translation and
global orientation come from the placer's multi-camera DLT + Procrustes.  The
per-camera stills rendered by visualize_for_presentation.py are the INPUTS to
that; this is the output.

The body is drawn root-centred at a constant on-screen size — the translation
that the placer solved is removed each frame, the global orientation is kept —
so the clip shows the pose and the turning of the body rather than the person
walking out of frame.

The fusion and placement come from visualize/render_flyover.py and the
rasteriser from visualize/visualize_for_presentation.py.  Both are imported
rather than copied: they are several hundred lines of arithmetic that the paper
numbers depend on, and a second transcription of them would be free to drift.

MP4 cannot carry transparency — H.264 has no alpha channel, and this ffmpeg has
no alpha-capable H.26x encoder — so the background is a flat colour.  Pass
--bg to change it.

Usage:
    pixi run python visualize/render_fused_pose.py \\
        --scene-dir /.../egohumans_new/02_lego/001_legoassemble \\
        --pid 1 --out fused.mp4
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from visualize.render_flyover import predict_vertices                  # noqa: E402
from visualize.visualize_for_presentation import (                     # noqa: E402
    body_faces_only, render_mesh_zbuffer,
)

logger = logging.getLogger("fused_pose")


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World -> camera rotation for a yaw/pitch orbit position."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rx @ Ry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    ap.add_argument("--smplx-model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--size", type=int, default=1080, help="output square size")
    ap.add_argument("--ss", type=int, default=2, help="supersample factor")
    ap.add_argument("--fill", type=float, default=0.82,
                    help="fraction of the frame the tallest pose occupies")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--colour", default="#2e9e46")
    ap.add_argument("--bg", default="#ffffff")
    ap.add_argument("--root-smooth", type=int, default=0,
                    help="placer Savitzky-Golay window; 0 is the reported protocol")
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cv2.setNumThreads(1)

    # frame span: everything this person is tracked over, unless asked otherwise
    body_dirs = sorted(args.scene_dir.glob(f"*/body_data_clean/person_{args.pid}.npz"))
    if not body_dirs:
        sys.exit(f"no body_data_clean tracks for person {args.pid} under {args.scene_dir}")
    lo, hi = None, None
    for f in body_dirs:
        fi = np.load(str(f), allow_pickle=False)["frame_indices"].astype(int)
        lo = int(fi.min()) if lo is None else min(lo, int(fi.min()))
        hi = int(fi.max()) if hi is None else max(hi, int(fi.max()))
    if args.frames:
        lo, hi = args.frames
    frames = list(range(lo, hi + 1))
    logger.info(f"person {args.pid}: frames {lo}..{hi}")

    verts_by_pid, faces_all, ma_scale = predict_vertices(
        args.scene_dir, frames, [args.pid], args.device, str(args.smplx_model),
        smooth_window=args.root_smooth, min_views=args.min_views)
    placed = verts_by_pid.get(args.pid, {})
    if not placed:
        sys.exit("no frame of this person survived placement")
    faces = body_faces_only(faces_all.astype(np.int32))       # SMPL-X minus eyeballs
    logger.info(f"placed {len(placed)}/{len(frames)} frames, scale {ma_scale:.4f} m/unit")

    R_view = view_rotation(args.yaw, args.pitch)
    bgr = tuple(int(args.colour.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))
    bg = np.array([int(args.bg.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0)], np.float64)

    # One camera for the whole clip.  The focal and principal point are solved
    # from the PROJECTED bounding box over every frame, not from a world-space
    # radius: under perspective those differ, and sizing by the radius leaves the
    # body small and off-centre.  Fixed for the clip, so the pose never rescales.
    keys = sorted(placed)
    cams = {}
    reach = 0.0
    for gfr in keys:
        v = (R_view @ (placed[gfr] - placed[gfr].mean(0)).T).T
        cams[gfr] = v
        reach = max(reach, float(np.abs(v).max()))
    depth = 4.0 * reach + 1e-6

    f0 = float(args.size)
    u0 = u1 = v0 = v1 = None
    for gfr in keys:
        v = cams[gfr].copy()
        v[:, 2] += depth
        z = np.maximum(v[:, 2], 1e-6)
        uu, vv = v[:, 0] / z * f0, v[:, 1] / z * f0
        u0 = uu.min() if u0 is None else min(u0, uu.min())
        u1 = uu.max() if u1 is None else max(u1, uu.max())
        v0 = vv.min() if v0 is None else min(v0, vv.min())
        v1 = vv.max() if v1 is None else max(v1, vv.max())
    k = args.fill * args.size / max(u1 - u0, v1 - v0, 1e-9)
    focal = f0 * k
    cx = args.size / 2.0 - k * (u0 + u1) / 2.0
    cy = args.size / 2.0 - k * (v0 + v1) / 2.0

    tmp = Path(tempfile.mkdtemp(prefix="fused_pose_"))
    try:
        for i, gfr in enumerate(keys):
            v = cams[gfr].copy()
            v[:, 2] += depth
            rgba = render_mesh_zbuffer([(v, faces, bgr)], focal, cx, cy,
                                       args.size, args.size, args.ss)
            a = rgba[:, :, 3:4] / 255.0
            frame = (rgba[:, :, :3] * a + bg[None, None] * (1 - a)).astype(np.uint8)
            cv2.imwrite(str(tmp / f"{i:05d}.png"), frame)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i + 1}/{len(keys)} frames")
        encode(tmp, args.out, args.fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def encode(png_dir: Path, mp4: Path, fps: int) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-y", "-framerate", str(fps), "-i", str(png_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)],
                   check=True, capture_output=True)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
