"""Re-run only the VGGT camera+depth estimation step for scenes with stale outputs.

Stale = vggt_depth.npz exists but depth_valid is all-False (caused by the
_get_vggt_hw aspect-ratio bug). Skips scenes that already have valid depth.

Usage:
    pixi run python scripts/rerun_vggt_only.py [--scenes SCENE1 SCENE2 ...]
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import RichDataset
from preprocessing.run_vggt import VGGTPreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}



def _build_frame_paths(scene, video_dirs: dict[str, Path]):
    sorted_videos = sorted(scene.videos, key=lambda v: v.video_id)
    camera_names  = [v.video_id for v in sorted_videos]
    per_cam_frames = []
    for video in sorted_videos:
        fdir = video.frames_home
        if fdir is None or not fdir.is_dir():
            logger.warning(f"  No frames dir for {video.video_id}")
            per_cam_frames.append([])
            continue
        frames = sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        per_cam_frames.append(frames)

    T = min((len(f) for f in per_cam_frames), default=0)
    if T == 0:
        return [], camera_names

    frame_paths = [[cam[t] if cam else None for cam in per_cam_frames] for t in range(T)]
    return frame_paths, camera_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Scene names to process. If omitted, processes all stale scenes.")
    parser.add_argument("--vggt-weights", type=str,
                        default="checkpoints/vggt_omega/vggt_omega_1b_512.pt",
                        help="Path to VGGT-Omega weights .pt file.")
    parser.add_argument("--vggt-devices", nargs="+", default=None,
                        help="CUDA device strings, e.g. cuda:0 cuda:1.")
    args = parser.parse_args()

    if args.vggt_devices:
        vggt_devices = args.vggt_devices
    elif torch.cuda.is_available():
        vggt_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        vggt_devices = ["cpu"]
    logger.info(f"VGGT devices: {vggt_devices}")

    rich_data_root = CONFIG.data.rich_data_root
    output_dir     = Path(CONFIG.data.output_directory)

    ds = RichDataset(
        data_root=rich_data_root,
        slice=getattr(CONFIG.data, "slice", None),
        max_side=getattr(CONFIG.data, "rich_max_side", None),
        frames_base_dir=getattr(CONFIG.data, "rich_frames_base_dir", None),
    )

    scenes = [s for s in ds.scenes
              if args.scenes is None or s.scene_id in args.scenes]
    logger.info(f"Found {len(scenes)} scene(s) to consider.")

    for scene in scenes:
        scene_out = output_dir / scene.scene_id
        cam_path  = scene_out / "vggt_cameras.npz"
        dep_path  = scene_out / "vggt_depth.npz"

        if cam_path.exists() and dep_path.exists():
            logger.info(f"[{scene.scene_id}] already done — skipping.")
            continue

        logger.info(f"\n=== {scene.scene_id} ===")

        video_dirs = {v.video_id: scene_out / v.video_id for v in scene.videos}
        result = _build_frame_paths(scene, video_dirs)
        if not result[0]:
            logger.warning(f"  No frames found — skipping.")
            continue
        frame_paths, camera_names = result

        logger.info(f"  {len(frame_paths)} frames × {len(camera_names)} cameras")
        logger.info(f"  Devices: {vggt_devices}")

        try:
            preprocessor = VGGTPreprocessor(weights=args.vggt_weights, device=vggt_devices[0])
            preprocessor.process_scene(
                frame_paths=frame_paths,
                camera_names=camera_names,
                output_dir=scene_out,
                devices=vggt_devices,
            )
            del preprocessor
            torch.cuda.empty_cache()
            logger.info(f"  Saved → {cam_path.name}, {dep_path.name}")
        except Exception as e:
            logger.error(f"  VGGT failed: {e}", exc_info=True)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
