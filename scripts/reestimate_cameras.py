"""Re-estimate camera alignment for already-processed RICH scenes.

Runs only Step 6 (camera alignment) from rich_pipeline.py on scenes that
already have body_data/ and cross_view_reid.json on disk.  Overwrites the
existing camera_alignment.npz in each scene directory.

Scenes and cameras are enumerated directly from the output directory —
no access to the raw data root (capstor) is needed.

Usage
-----
    pixi run python scripts/reestimate_cameras.py
    pixi run python scripts/reestimate_cameras.py --scene-start 0 --scene-end 10
    pixi run python scripts/reestimate_cameras.py --min-correspondences 30
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from configuration import CONFIG
from preprocessing.camera_alignment import CameraAlignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def enumerate_scenes(output_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    """Return [(scene_id, {video_id: video_dir})] for every scene that has body data."""
    scenes = []
    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        video_dirs = {
            d.name: d
            for d in sorted(scene_dir.iterdir())
            if d.is_dir() and any((d / "body_data").glob("person_*.npz"))
        }
        if video_dirs:
            scenes.append((scene_dir.name, video_dirs))
    return scenes


def reestimate_scene(scene_id: str, video_dirs: dict[str, Path], min_correspondences: int) -> bool:
    scene_dir = next(iter(video_dirs.values())).parent

    missing = [vid_id for vid_id, vd in video_dirs.items()
               if not any((vd / "body_data").glob("person_*.npz"))]
    if missing:
        logging.warning(f"{scene_id}: cameras missing body_data: {missing}")

    logging.info(f"\n=== {scene_id} ({len(video_dirs)} cameras) ===")

    alignment = CameraAlignment().estimate(
        video_dirs=video_dirs,
        min_correspondences=min_correspondences,
        scene_dir=scene_dir,
    )

    if not alignment:
        logging.warning(f"{scene_id}: no camera pairs aligned — camera_alignment.npz NOT written")
        return False

    align_path = CameraAlignment.save(alignment, scene_dir)
    logging.info(f"{scene_id}: {len(alignment)} pair(s) saved → {align_path}")

    for (vid_a, vid_b), (R_arr, t_arr) in alignment.items():
        logging.info(
            f"  {vid_a} ↔ {vid_b}: |t| = {np.linalg.norm(t_arr):.3f} m"
        )

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-start", type=int, default=None)
    parser.add_argument("--scene-end", type=int, default=None)
    parser.add_argument("--min-correspondences", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(CONFIG.data.output_directory)
    scenes = enumerate_scenes(output_dir)

    if args.scene_start is not None or args.scene_end is not None:
        scenes = scenes[args.scene_start:args.scene_end]

    logging.info(f"Found {len(scenes)} scene(s) with body data in {output_dir}")

    ok, skipped = 0, 0
    for scene_id, video_dirs in scenes:
        success = reestimate_scene(scene_id, video_dirs, args.min_correspondences)
        if success:
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} scene(s) re-estimated, {skipped} skipped.")


if __name__ == "__main__":
    main()
