"""Render a frame-numbered track-debug video for manual within-view ReID.

The pipeline's ``*_segmentation_reid.mp4`` shows per-id overlays but no frame
numbers, so you can *see* an id-steal but can't read the frame it happens on —
which is exactly what utilities/within_reid_operations.py (merge/split/swap)
needs. This tool renders the **current on-disk state** (ids read live from
json_data/ + mask_data.npz) with:

  * a per-id consistent colour + "P<id>" label on each bbox,
  * a large frame number burned top-left,
  * optional id filter and frame range to zoom in on one crossing.

Because it reads the live files, the loop is: render → read swap frames →
apply a swap op → re-render the same window → confirm. Fast: bboxes come from
json_data (no mask decode) unless --mask is passed.

Usage
-----
    pixi run python -m utilities.render_track_debug \\
        --cam_dir  /.../007_tagging/cam04 \\
        --frames   /.../007_tagging/exo/cam04/images_undistorted/frames \\
        --start 400 --end 620 --ids 2,5,9

Output defaults to <cam_dir>/<cam>_tracks_debug.mp4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _color(pid: int) -> tuple[int, int, int]:
    """Deterministic bright BGR colour per id."""
    rng = np.random.default_rng(pid * 9973 + 12345)
    hsv = np.uint8([[[int(rng.integers(0, 180)), 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _frame_path(frames_dir: Path, idx: int) -> Path | None:
    for width in (5, 6, 4):
        for ext in (".jpg", ".jpeg", ".png"):
            p = frames_dir / f"{idx:0{width}d}{ext}"
            if p.exists():
                return p
    hits = list(frames_dir.glob(f"*{idx}*"))
    return hits[0] if hits else None


def render(
    cam_dir: Path,
    frames_dir: Path,
    out_path: Path,
    start: int | None,
    end: int | None,
    ids: set[int] | None,
    fps: int,
    draw_mask: bool,
) -> None:
    json_dir = cam_dir / "json_data"
    npz_path = cam_dir / "mask_data.npz"
    json_files = sorted(json_dir.glob("*.json"), key=lambda p: int(p.stem.replace("mask_", "").split("_")[0]))

    mask_zip = None
    if draw_mask and npz_path.exists():
        import zipfile
        mask_zip = zipfile.ZipFile(str(npz_path))

    writer = None
    n_written = 0
    for jp in json_files:
        idx = int(jp.stem.replace("mask_", "").split("_")[0])
        if start is not None and idx < start:
            continue
        if end is not None and idx > end:
            break
        fp = _frame_path(frames_dir, idx)
        if fp is None:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(jp) as f:
            labels = json.load(f).get("labels", {})

        if draw_mask and mask_zip is not None:
            key = f"{jp.stem}.npy"
            if key in mask_zip.namelist():
                import io
                mask = np.load(io.BytesIO(mask_zip.read(key)))
                overlay = img.copy()
                for str_id in labels:
                    pid = int(str_id)
                    if ids is not None and pid not in ids:
                        continue
                    overlay[mask == pid] = _color(pid)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        for str_id, info in labels.items():
            pid = int(str_id)
            if ids is not None and pid not in ids:
                continue
            col = _color(pid)
            x1, y1, x2, y2 = info.get("x1", 0), info.get("y1", 0), info.get("x2", 0), info.get("y2", 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            cv2.putText(img, f"P{pid}", (x1 + 4, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)

        # big frame number, black outline + white fill for legibility
        txt = f"frame {idx}"
        cv2.putText(img, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(img, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
        n_written += 1

    if writer is not None:
        writer.release()
    if mask_zip is not None:
        mask_zip.close()
    print(f"wrote {n_written} frames -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam_dir", type=Path, required=True, help="camera output dir (has json_data/, mask_data.npz)")
    ap.add_argument("--frames", type=Path, required=True, help="dir of source frames (00001.jpg ...)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--start", type=int, default=None, help="first frame idx (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame idx (inclusive)")
    ap.add_argument("--ids", type=str, default=None, help="comma-separated ids to show (default: all)")
    ap.add_argument("--fps", type=int, default=30, help="output fps")
    ap.add_argument("--mask", action="store_true", help="also tint segmentation masks (slower)")
    args = ap.parse_args()

    out = args.out or args.cam_dir / f"{args.cam_dir.name}_tracks_debug.mp4"
    ids = {int(x) for x in args.ids.split(",")} if args.ids else None
    render(args.cam_dir, args.frames, out, args.start, args.end, ids, args.fps, args.mask)


if __name__ == "__main__":
    main()
