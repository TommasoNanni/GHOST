"""Visualise re-identified segmentation tracks as an annotated mp4.

Reads all required data from a single video output directory:
  - mask_data.npz     per-frame uint16 masks (pixel value = canonical person ID)
  - json_data/        per-frame bbox + label metadata
  - body_data/
      reid_id_mapping.json   raw SAM2 id → canonical id (written by BodyParameterEstimator)

Original frames are read from the frames_dir argument (co-located with the
source data at data/<scene>/<video_id>/frames/).  Pass ``frames_dir`` explicitly;
if omitted the function falls back to ``video_dir/frames/`` for backward
compatibility.

Renders one mp4 with:
  - a coloured translucent mask per person
  - a bounding box and ID label per person
  - for re-identified persons: the original SAM2 ID(s) that were merged
    are shown in the label, e.g. "P1  [SAM2: 7, 12]"
"""
import argparse
import io
import json
import zipfile
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


def visualize_reid(
    video_dir: Path,
    fps: int = 30,
    frames_dir: Path | None = None,
) -> Path:
    """Render an mp4 showing re-identified persons with coloured masks and labels.

    Parameters
    ----------
    video_dir : Path
        Root output directory for one video, as produced by PersonSegmenter +
        BodyParameterEstimator.  Must contain mask_data.npz, json_data/, and
        body_data/.
    fps : int
        Frame rate for the output mp4.
    frames_dir : Path | None
        Directory containing the extracted JPEG frames.  Should point to the
        canonical co-located path (e.g. ``data/<scene>/<video_id>/frames/``).
        Falls back to ``video_dir/frames/`` when not provided.

    Returns
    -------
    Path
        Path to the written mp4 file.
    """
    frame_dir = frames_dir if frames_dir is not None else video_dir / "frames"
    npz_path  = video_dir / "mask_data.npz"
    json_dir  = video_dir / "json_data"
    body_dir  = video_dir / "body_data"

    # ── Load re-ID mapping ────────────────────────────────────────────────────
    # reid_map : raw SAM2 id → canonical id
    # merged_from : canonical id → [raw SAM2 ids that were merged into it]
    reid_map: dict[int, int] = {}
    merged_from: dict[int, list[int]] = {}

    reid_map_path = body_dir / "reid_id_mapping.json"
    if reid_map_path.exists():
        with open(reid_map_path) as f:
            reid_map = {int(k): int(v) for k, v in json.load(f).items()}
        for raw_id, canon_id in reid_map.items():
            merged_from.setdefault(canon_id, []).append(raw_id)
    else:
        print(f"  No reid_id_mapping.json found in {body_dir} — rendering without re-ID labels")

    # ── Collect sorted frame list ─────────────────────────────────────────────
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON metadata files found in {json_dir}")

    if not npz_path.exists():
        raise FileNotFoundError(f"mask_data.npz not found at {npz_path}")

    # ── Determine output video dimensions from the first readable frame ───────
    H, W = 0, 0
    for jf in json_files:
        fi_str = jf.stem.replace("mask_", "")
        p = frame_dir / f"{fi_str}.jpg"
        if p.exists():
            sample = cv2.imread(str(p))
            if sample is not None:
                H, W = sample.shape[:2]
                break
    if H == 0:
        raise FileNotFoundError(f"No readable JPEG frames found in {frame_dir}")

    out_path = video_dir / "segmentation_reid.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H),
    )

    with zipfile.ZipFile(str(npz_path), "r") as zf:
        npz_keys = set(zf.namelist())

        for json_path in tqdm(json_files, desc=f"Rendering {video_dir.name}", leave=False):
            fi_str = json_path.stem.replace("mask_", "")
            frame_path = frame_dir / f"{fi_str}.jpg"
            if not frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            with open(json_path) as f:
                labels: dict = json.load(f).get("labels", {})

            # Load the per-frame mask (already remapped to canonical IDs).
            mask_key = json_path.stem + ".npy"    # e.g. "mask_000042.npy"
            mask_img: np.ndarray | None = None
            if mask_key in npz_keys:
                with zf.open(mask_key) as mf:
                    mask_img = np.load(io.BytesIO(mf.read()))

            # 1. Coloured mask overlay (drawn before boxes so boxes stay sharp).
            if mask_img is not None:
                overlay = frame.copy()
                for str_id in labels:
                    canon_id = int(str_id)
                    person_mask = mask_img == canon_id
                    if not person_mask.any():
                        continue
                    color = np.array(_color(canon_id), dtype=np.float32)
                    overlay[person_mask] = (
                        0.5 * overlay[person_mask] + 0.5 * color
                    ).astype(np.uint8)
                frame = overlay

            # 2. Bounding boxes and ID labels.
            for str_id, info in labels.items():
                canon_id = int(str_id)
                color    = _color(canon_id)
                x1, y1   = int(info["x1"]), int(info["y1"])
                x2, y2   = int(info["x2"]), int(info["y2"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label: canonical ID + any SAM2 IDs that were merged into it.
                label = f"P{canon_id}"
                merged = merged_from.get(canon_id)
                if merged:
                    label += f"  [SAM2: {', '.join(str(m) for m in merged)}]"

                # Solid background chip for legibility.
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness  = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                ty = max(y1 - 6, th + 4)
                cv2.rectangle(
                    frame,
                    (x1, ty - th - 4), (x1 + tw + 6, ty + 2),
                    color, cv2.FILLED,
                )
                cv2.putText(
                    frame, label,
                    (x1 + 3, ty - 1),
                    font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA,
                )

            writer.write(frame)

    writer.release()
    print(f"  Re-ID visualisation saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise re-identified segmentation tracks as an mp4."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("/cluster/project/cvg/students/tnanni/ghost/test_outputs/segmentation_test/cmu_bike01_2/cam02"),
        help="Root output directory for one video (contains frames/, mask_data.npz, etc.)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Output video frame rate (default: 30)"
    )
    args = parser.parse_args()
    visualize_reid(args.video_dir, fps=args.fps)


if __name__ == "__main__":
    main()
