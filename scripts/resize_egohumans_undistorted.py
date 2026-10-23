#!/usr/bin/env python3
"""Resize freshly-fetched EgoHumans undistorted frames to the pipeline's max_side.

The ghost pipeline never ran VGGT on full-resolution EgoHumans frames: Video is
constructed with ``max_side=CONFIG.data.egohumans_max_side`` (1440), which writes
aspect-preserving copies into an ``images_undistorted/frames/`` subdirectory and
deletes the originals. Every ``pred_keypoints_2d`` in body_data_clean is therefore
in 1440-wide pixel space, and each scene's ``vggt_cameras_centered.npz`` records
``original_size = [1440, 810]``.

Scenes downloaded later by scripts/fetch_egohumans_inexact.py have no ``frames/``
subdirectory, only 3840x2160 images sitting directly in ``images_undistorted/``.
Re-running VGGT on those produces intrinsics and ``original_size`` for a 3840-wide
image while the 2D keypoints are still 1440-wide, so BodyPlacer maps them with a
2.67x wrong scale -- measured as W-dagger 2564 mm against a 104 mm control on
02_lego/002_legoassemble, with the reconstruction otherwise identical (extrinsics
agreed to 0.045, depth to 1.005x, MapAnything scale to 1.00x).

This script brings those scenes to the same state the pipeline would have left
them in. The resize is copied exactly from data/video_dataset.py::_maybe_resize --
same scale rule, same rounding, same cv2 default interpolation, same in-place
delete of the originals -- because the keypoints were computed against ITS output;
any difference in rounding re-introduces the same class of error at a smaller
magnitude.

Idempotent: a camera that already has a populated ``frames/`` is skipped, so the
14 scenes staged by earlier work are untouched.

Usage
-----
    pixi run python scripts/resize_egohumans_undistorted.py \\
        --data_root /iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted \\
        --workers 16
    # report only:
    ... --dry_run
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _resize_camera(args: tuple[Path, int]) -> str:
    """Resize one camera's images_undistorted/*.jpg into images_undistorted/frames/."""
    und, max_side = args
    src_frames = sorted(p for p in und.iterdir()
                        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    if not src_frames:
        return f"SKIP {und.parent.name}: no loose images"

    dst = und / "frames"
    dst.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in _IMAGE_EXTS for p in dst.iterdir()):
        return f"SKIP {und.parent.name}: frames/ already populated"

    first = cv2.imread(str(src_frames[0]))
    if first is None:
        return f"FAIL {und.parent.name}: unreadable first frame"
    if max(first.shape[:2]) <= max_side:
        # Already small enough: the pipeline would not have resized either, so the
        # loose images ARE the frames. Hard-link them rather than re-encoding (a
        # re-encode would change the pixels the keypoints were computed on).
        for p in src_frames:
            (dst / p.name).hardlink_to(p)
        for p in src_frames:
            p.unlink()
        return f"LINK {und.parent.name}: {len(src_frames)} frames already <= {max_side}"

    n = 0
    for p in src_frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)))
        cv2.imwrite(str(dst / p.name), img)
        n += 1
    for p in src_frames:                      # in-place resize: drop the originals
        if p.exists():
            p.unlink()
    return f"OK   {und.parent.name}: {n} frames -> max_side {max_side}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, type=Path)
    ap.add_argument("--max_side", type=int, default=None,
                    help="Default: CONFIG.data.egohumans_max_side (the value the pipeline used).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    max_side = args.max_side
    if max_side is None:
        from configuration import CONFIG
        max_side = int(getattr(CONFIG.data, "egohumans_max_side", 1440))

    todo: list[tuple[Path, int]] = []
    already = 0
    for und in sorted(args.data_root.glob(f"*/{INNER}/*/*/exo/*/images_undistorted")):
        frames = und / "frames"
        has_frames = frames.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in frames.iterdir())
        loose = any(p.is_file() and p.suffix.lower() in _IMAGE_EXTS for p in und.iterdir())
        if has_frames:
            already += 1
        elif loose:
            todo.append((und, max_side))

    print(f"max_side={max_side}   cameras needing resize: {len(todo)}   already done: {already}")
    if args.dry_run:
        for und, _ in todo[:20]:
            print(f"  {und.parent.parent.parent.name}/{und.parent.name}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_resize_camera, todo), 1):
            if i % 20 == 0 or i == len(todo) or not res.startswith("OK"):
                print(f"[{i}/{len(todo)}] {res}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
