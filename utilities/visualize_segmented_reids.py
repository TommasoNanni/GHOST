"""Visualise re-identified person tracks as an annotated mp4.

Boxes and IDs are read **directly from ``body_data/person_<id>.npz``** — the
canonical, post-reid tracks that the evaluation consumes — NOT from
``json_data/`` or ``mask_data.npz``.

Why: ``json_data``/``mask_data`` are remapped after reid with a single *global*
id→id map, while ``body_data`` is reassigned *per-frame* by the appearance reid.
Around reid transitions (crossings, gap-severs) these two granularities cannot
agree, so ``json``/``mask`` ids can point to a different person than
``person_<id>.npz`` for a handful of frames.  Drawing straight from
``person_<id>.npz`` guarantees the label in the video is the same identity the
eval uses — coherent by construction — so the video is safe to read off for
manual cross-view reid.

Each ``person_<id>.npz`` stores ``frame_indices`` and ``bbox`` (the original
detection bbox, passed through from segmentation); we draw one coloured box +
``P<id>`` label per person per frame.  If ``reid_id_mapping.json`` is present,
the merged SAM2 ids are appended to the label, e.g. ``P1 [SAM2: 7, 12]``.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# BGR colour palette – one entry per canonical person ID (cycles if needed).
_PALETTE: list[tuple[int, int, int]] = [
    ( 60,  80, 220),   # red
    ( 60, 200,  60),   # green
    (220,  80,  60),   # blue
    ( 40, 210, 210),   # yellow
    (210,  60, 210),   # magenta
    (210, 210,  40),   # cyan
    ( 40, 140, 220),   # orange
    (160,  60, 160),   # purple
]


def _color(person_id: int) -> tuple[int, int, int]:
    return _PALETTE[person_id % len(_PALETTE)]


def _frame_index(frame_path: Path) -> int | None:
    """Frame number from a filename, across every naming scheme in this repo.

    ``00001_00.jpg``   RICH image sequences (frame_camera)  -> 1
    ``000001.jpg``     Video._extract_frames ``{idx:06d}``  -> 1
    ``00001.jpg``      EgoHumans ``images_undistorted/``    -> 1
    ``frame_001426.jpg``  EgoHumans ``frames_root/``        -> 1426

    Plain ``int(stem)`` is wrong for the RICH layout: Python reads ``_`` as a
    digit separator, so ``int("00001_00")`` returns **100** instead of raising —
    every bbox lands on the wrong frame and most fall past the end of the clip.
    So: leading digit-run when the name starts with a digit, trailing one when a
    prefix like ``frame_`` comes first.
    """
    stem = frame_path.stem
    runs = re.findall(r"\d+", stem)
    if not runs:
        return None
    return int(runs[0] if stem[0].isdigit() else runs[-1])


def visualize_reid(
    video_dir: Path,
    fps: int = 30,
    frames_dir: Path | None = None,
) -> Path:
    """Render an mp4 of the canonical person tracks (boxes + id labels).

    Parameters
    ----------
    video_dir : Path
        Root output directory for one video.  Must contain ``body_data/`` with
        ``person_<id>.npz`` files.
    fps : int
        Frame rate for the output mp4.
    frames_dir : Path | None
        Directory containing the extracted frames (e.g.
        ``data/<scene>/<video_id>/frames/``).  Falls back to
        ``video_dir/frames/`` when not provided.  Frame files must be named by
        their frame index (e.g. ``00042.jpg``) so they line up with the
        ``frame_indices`` stored in ``person_<id>.npz``.

    Returns
    -------
    Path
        Path to the written mp4 file.
    """
    frame_dir = frames_dir if frames_dir is not None else video_dir / "frames"
    body_dir  = video_dir / "body_data"

    # ── Load canonical per-person tracks: {pid: {frame_idx: (x1,y1,x2,y2)}} ──
    tracks: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for npz_path in sorted(body_dir.glob("person_*.npz")):
        try:
            pid = int(npz_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        with np.load(npz_path) as d:
            if "frame_indices" not in d.files or "bbox" not in d.files:
                continue
            fi = d["frame_indices"]
            bb = d["bbox"]
        tracks[pid] = {
            int(f): tuple(int(round(float(v))) for v in bb[i])
            for i, f in enumerate(fi)
        }
    if not tracks:
        raise FileNotFoundError(
            f"No person_*.npz with 'bbox' found in {body_dir}"
        )

    # Optional: SAM2 ids merged into each canonical id, for the label text.
    merged_from: dict[int, list[int]] = {}
    for map_filename in ("reid_id_mapping.json", "cross_view_id_mapping.json"):
        map_path = body_dir / map_filename
        if map_path.exists():
            with open(map_path) as f:
                for k, v in json.load(f).items():
                    merged_from.setdefault(int(v), []).append(int(k))

    # ── Frame list + dimensions ────────────────────────────────────────────
    _FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    frame_files = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in _FRAME_EXTS
    )
    if not frame_files:
        raise FileNotFoundError(f"No readable frames found in {frame_dir}")
    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        raise FileNotFoundError(f"Could not read frame {frame_files[0]}")
    H, W = sample.shape[:2]

    out_path = video_dir / f"{video_dir.name}_segmentation_reid.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H)
    )

    for frame_path in tqdm(frame_files, desc=f"Rendering {video_dir.name}", leave=False):
        frame_idx = _frame_index(frame_path)
        if frame_idx is None:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        for pid, per_frame in tracks.items():
            bbox = per_frame.get(frame_idx)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = _color(pid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"P{pid}"
            merged = merged_from.get(pid)
            if merged:
                label += f"  [SAM2: {', '.join(str(m) for m in merged)}]"

            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(
                frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, cv2.FILLED
            )
            cv2.putText(
                frame, label, (x1 + 3, ty - 1),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

        writer.write(frame)

    writer.release()
    print(f"  Re-ID visualisation saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise re-identified person tracks as an mp4 (drawn from body_data)."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        help="Root output directory for one video (contains body_data/, frames/, ...)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Output video frame rate (default: 30)"
    )
    parser.add_argument(
        "--frames_dir", type=Path, default=None,
        help="Directory containing extracted frames (overrides default video_dir/frames/)",
    )
    args = parser.parse_args()
    visualize_reid(args.video_dir, fps=args.fps, frames_dir=args.frames_dir)


if __name__ == "__main__":
    main()
