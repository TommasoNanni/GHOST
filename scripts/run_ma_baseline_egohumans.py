#!/usr/bin/env python
"""Sweep all EgoHumans scenes computing the baseline-ratio MapAnything scale.

Writes <scene>/mapanything_scale_baseline.npy (per-frame, images-only MA camera
baselines / vggt baselines). Legacy mapanything_scale_centered.npy untouched.
Resumable: scenes with an existing output are skipped — relaunch until done.

  pixi run python scripts/run_ma_baseline_egohumans.py [--activity 06_badminton]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/users/tnanni/ghost")
from preprocessing.run_mapanything import MapAnythingScaleEstimator  # noqa

OUT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
TEMP = Path("/iopsstor/scratch/cscs/tnanni/temp_egohumans")
INNER = "media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default=None, help="restrict to one activity dir")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    est = MapAnythingScaleEstimator(batch_size=args.batch_size)

    n_done = n_skip = n_fail = 0
    for act_dir in sorted(OUT.iterdir()):
        if not act_dir.is_dir():
            continue
        if args.activity and act_dir.name != args.activity:
            continue
        for scene_dir in sorted(act_dir.iterdir()):
            if not (scene_dir / "vggt_cameras_centered.npz").exists():
                continue
            if (scene_dir / "mapanything_scale_baseline.npy").exists():
                n_skip += 1
                continue
            img_root = TEMP / act_dir.name / INNER / act_dir.name / scene_dir.name / "exo"
            if not img_root.is_dir():
                logging.warning(f"{scene_dir.name}: no img_root {img_root} — skip")
                n_fail += 1
                continue
            try:
                r = est.process_scene(scene_dir=scene_dir, img_root=img_root)
                if r is None:
                    n_fail += 1
                else:
                    n_done += 1
            except Exception as e:
                logging.warning(f"{scene_dir.name}: FAILED — {e}")
                n_fail += 1
    print(f"SWEEP_DONE done={n_done} skipped={n_skip} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
