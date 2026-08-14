#!/usr/bin/env python
"""Sweep RICH scenes computing the baseline-ratio MapAnything scale.

Writes <scene>/mapanything_scale_baseline.npy (per-frame, images-only MA camera
baselines / vggt baselines). Legacy mapanything_scale_centered.npy untouched.
Resumable: scenes with an existing output are skipped.

  pixi run python scripts/run_ma_baseline_rich.py --split train --img_root <mnt>
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/users/tnanni/ghost")
from preprocessing.run_mapanything import MapAnythingScaleEstimator  # noqa

GHOST = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--img_root", required=True,
                    help="mounted centered_<split>.sqsh root (scene/cam_XX/*.jpg)")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    root = GHOST / f"rich_{args.split}"
    img_root = Path(args.img_root)
    est = MapAnythingScaleEstimator(batch_size=args.batch_size)

    n_done = n_skip = n_fail = 0
    for scene_dir in sorted(root.iterdir()):
        if not (scene_dir / "vggt_cameras_centered.npz").exists():
            continue
        if (scene_dir / "mapanything_scale_baseline.npy").exists():
            n_skip += 1
            continue
        scene_img = img_root / scene_dir.name
        if not scene_img.is_dir():
            logging.warning(f"{scene_dir.name}: no images at {scene_img} — skip")
            n_fail += 1
            continue
        try:
            r = est.process_scene(scene_dir=scene_dir, img_root=scene_img)
            if r is None:
                n_fail += 1
            else:
                n_done += 1
        except Exception as e:
            logging.warning(f"{scene_dir.name}: FAILED — {e}")
            n_fail += 1
    print(f"SWEEP_DONE split={args.split} done={n_done} skipped={n_skip} failed={n_fail}",
          flush=True)


if __name__ == "__main__":
    main()
