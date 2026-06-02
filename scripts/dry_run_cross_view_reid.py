"""Dry-run cross-view ReID over an output directory.

Walks output_dir, finds scenes with body_data, runs the new constrained
R+t+λ ReID algorithm, and prints what remappings would be applied —
without writing anything to disk.

Usage:
    pixi run python -m scripts.dry_run_cross_view_reid [--output_dir DIR] [--scene SCENE_ID]
                                                       [--data_root DIR]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration import CONFIG
from data.video_dataset import Scene, Video
from preprocessing.cross_view_reid_v2 import CrossVideoReidentifierV2


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)


def _build_scene(
    scene_id: str,
    scene_dir: Path,
    data_root: Path | None,
) -> tuple[Scene, dict[str, Path], dict[str, Path]] | None:
    """Build a minimal Scene from an output scene directory.

    A valid camera subdir has a body_data/ folder with at least one person_*.npz.
    Returns (scene, video_dirs, frames_dirs).  frames_dirs maps cam_id to the
    actual frames directory (data_root/<scene>/<cam>/frames/) when data_root is
    provided, so that DROID-SLAM can find images.
    """
    video_dirs: dict[str, Path] = {}
    for cam_dir in sorted(scene_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        body_dir = cam_dir / "body_data"
        if not body_dir.exists():
            continue
        if not any(body_dir.glob("person_*.npz")):
            continue
        video_dirs[cam_dir.name] = cam_dir

    if len(video_dirs) < 2:
        return None

    # Resolve frames dirs from data_root/<scene>/<cam>/ (RICH stores images directly
    # in the cam dir, no frames/ subdir).  Also try frames/ subdir as fallback.
    frames_dirs: dict[str, Path] = {}
    for cam_id, cam_out_dir in video_dirs.items():
        if data_root is not None:
            for candidate in [
                data_root / scene_id / cam_id,
                data_root / scene_id / cam_id / "frames",
            ]:
                if candidate.is_dir() and any(
                    p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    for p in candidate.iterdir()
                ):
                    frames_dirs[cam_id] = candidate
                    break

    videos = [Video(frames_dir=vdir) for vdir in video_dirs.values()]
    scene = Scene(scene_id=scene_id, videos=videos)
    return scene, video_dirs, frames_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=getattr(CONFIG.data, "output_dir", None),
        help="Root output directory (default: CONFIG.data.output_dir)",
    )
    parser.add_argument(
        "--data_root",
        default=getattr(CONFIG.data, "data_root", None),
        help="Raw data root for locating frames/ dirs needed by DROID-SLAM "
             "(default: CONFIG.data.data_root)",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Run only this scene ID (default: all scenes)",
    )
    parser.add_argument(
        "--no_slam",
        action="store_true",
        help="Skip DROID-SLAM entirely and treat all cameras as static.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        print("ERROR: --output_dir required (or set CONFIG.data.output_dir)")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"ERROR: output_dir {output_dir} does not exist")
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else None

    reidentifier = CrossVideoReidentifierV2(
        threshold=getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
        appearance_weight=getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
        shape_weight=getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
        pose_weight=getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
        droid_weights=None if args.no_slam else getattr(CONFIG.data, "droid_weights", None),
        slam_cams=getattr(CONFIG.parameters_extraction, "slam_cams", None),
    )

    scene_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (args.scene is None or d.name == args.scene)
    )

    if not scene_dirs:
        print(f"No scene directories found in {output_dir}")
        return

    for scene_dir in scene_dirs:
        result = _build_scene(scene_dir.name, scene_dir, data_root)
        if result is None:
            print(f"\nSkipping {scene_dir.name}: fewer than 2 cameras with body_data")
            continue

        scene, video_dirs, frames_dirs = result
        print(f"\n{'='*60}")
        print(f"Scene: {scene_dir.name}  ({len(video_dirs)} cameras: {list(video_dirs)})")
        if frames_dirs:
            print(f"  frames_dirs resolved for: {list(frames_dirs)}")
        else:
            print(f"  WARNING: no frames_dirs resolved — SLAM will be skipped")

        reidentifier.match_across_views(
            scene=scene,
            video_dirs=video_dirs,
            frames_dirs=frames_dirs if frames_dirs else None,
            dry_run=True,
        )


if __name__ == "__main__":
    main()
