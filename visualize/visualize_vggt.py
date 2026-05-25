"""
Visualize VGGT depth maps side-by-side with input frames.

For each camera and frame, produces:
    <output_dir>/<scene_name>/<camera_name>/<frame_idx:06d>.jpg

Left half  : original RGB frame (resized to 518 height).
Right half : depth map coloured with the 'plasma' colormap.
             Depth is read from vggt_depth.npz (uint16 mm → float32 m),
             cropped to the original image region (removing padding),
             and normalised per-camera over the whole sequence for
             consistent colour mapping across frames.

Usage
-----
    pixi run python -m visualize.visualize_vggt \\
        --scene-dir /path/to/scene/output_dir \\
        --frames-dir /path/to/scene/data_dir \\
        [--output-dir renders/vggt_outputs] \\
        [--cameras cam_01 cam_03] \\
        [--max-frames 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_frame(frames_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        p = frames_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None



def visualize_scene(
    scene_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    cameras: list[str] | None = None,
    max_frames: int | None = None,
) -> None:
    scene_name = scene_dir.name

    # Load depth and camera metadata.
    depth_data = np.load(scene_dir / "vggt_depth.npz", allow_pickle=False)
    cam_data   = np.load(scene_dir / "vggt_cameras.npz", allow_pickle=False)

    depth_mm      = depth_data["depth"]          # (T, K, 518, 518) uint16
    depth_valid   = depth_data["depth_valid"]    # (T, K) bool
    original_coords = cam_data["original_coords"] # (T, K, 4) [x1,y1,x2,y2] in 518-space
    camera_names  = [n.decode() for n in cam_data["camera_names"]]  # (K,)

    T, K = depth_mm.shape[:2]

    # Filter cameras.
    cam_indices = list(range(K))
    if cameras:
        cam_indices = [i for i, n in enumerate(camera_names) if n in cameras]

    for k in cam_indices:
        cam_name = camera_names[k]

        # Find frames directory for this camera: prefer frames/ subdir, fall back to root.
        actual_frames_dir: Path | None = None
        for candidate in (frames_dir / cam_name / "frames", frames_dir / cam_name):
            if candidate.is_dir() and any(
                p.suffix.lower() in _IMAGE_EXTS for p in candidate.iterdir()
            ):
                actual_frames_dir = candidate
                break

        if actual_frames_dir is None:
            print(f"  [{cam_name}] no frames directory found under {frames_dir / cam_name}, skipping.")
            continue

        frame_stems = sorted(
            p.stem for p in actual_frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )
        if not frame_stems:
            print(f"  [{cam_name}] no images found, skipping.")
            continue

        n_frames = min(T, len(frame_stems), max_frames or T)

        # Compute per-camera depth range for consistent colour normalisation.
        valid_depths = depth_mm[:n_frames, k][depth_valid[:n_frames, k]].astype(np.float32) / 1000.0
        if valid_depths.size == 0:
            d_min, d_max = 0.0, 10.0
        else:
            d_min = float(np.percentile(valid_depths[valid_depths > 0], 2))
            d_max = float(np.percentile(valid_depths[valid_depths > 0], 98))
        if d_max <= d_min:
            d_max = d_min + 1.0

        out_dir = output_dir / scene_name / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{cam_name}] {n_frames} frames  depth [{d_min:.2f}m – {d_max:.2f}m]  → {out_dir}")

        for t in range(n_frames):
            stem = frame_stems[t]
            frame_path = _find_frame(actual_frames_dir, stem)
            if frame_path is None:
                continue

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue

            # VGGT-Omega: no square padding, image fills full depth map.
            # original_coords = [0, 0, W_vggt, H_vggt], so the crop is the whole map.
            depth_f = depth_mm[t, k].astype(np.float32) / 1000.0  # metres
            depth_crop = depth_f

            # Normalise to [0, 255] using per-camera range.
            depth_norm = np.clip((depth_crop - d_min) / (d_max - d_min), 0, 1)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # Zero-depth pixels (invalid) → 0 in normalised map → black in plasma.
            # Mark them explicitly as 0 so they're visually distinct.
            depth_uint8[depth_crop == 0] = 0

            depth_colour = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)

            # ── Resize both to the same height ───────────────────────────
            H = frame_bgr.shape[0]
            depth_colour = cv2.resize(depth_colour, (H, H), interpolation=cv2.INTER_LINEAR)

            canvas = np.concatenate([frame_bgr, depth_colour], axis=1)
            out_path = out_dir / f"{t:06d}.jpg"
            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"    done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise VGGT depth maps alongside input frames.")
    parser.add_argument("--scene-dir",   required=True, type=Path,
                        help="Scene output directory containing vggt_depth.npz and vggt_cameras.npz.")
    parser.add_argument("--frames-dir",  required=True, type=Path,
                        help="Scene data directory with camera subdirs containing frames.")
    parser.add_argument("--output-dir",  type=Path, default=Path("renders/vggt_outputs"),
                        help="Root output directory (default: renders/vggt_outputs).")
    parser.add_argument("--cameras",     nargs="+", default=None,
                        help="Camera names to process (default: all).")
    parser.add_argument("--max-frames",  type=int, default=None,
                        help="Maximum number of frames per camera (default: all).")
    args = parser.parse_args()

    print(f"Scene:     {args.scene_dir}")
    print(f"Frames:    {args.frames_dir}")
    print(f"Output:    {args.output_dir}")

    visualize_scene(
        scene_dir  = args.scene_dir,
        frames_dir = args.frames_dir,
        output_dir = args.output_dir,
        cameras    = args.cameras,
        max_frames = args.max_frames,
    )


if __name__ == "__main__":
    main()
