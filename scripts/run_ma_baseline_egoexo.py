#!/usr/bin/env python
"""Sweep all EgoExo4D takes computing the baseline-ratio MapAnything scale.

Writes <take>/mapanything_scale_baseline.npy (per-frame, images-only MapAnything
camera baselines / vggt camera baselines).  Legacy mapanything_scale_centered.npy
is left untouched.  Resumable: takes with an existing output are skipped.

Why this exists
---------------
``fusion/placer.py::load_mapanything_scale`` defaults to the *baseline* file, but
EgoExo4D never generated one — only RICH and EgoHumans got a sweep script.  The
placer therefore returned None and the eval silently fell back to scale 1.0 while
the camera centres were scaled by ~12-25, putting every subject at a fraction of
its true distance (W-MPJPE 2235 mm vs 389 mm with oracle scale, PA unaffected).

The baseline estimator is also the *correct* one here: the legacy "depth" method
is biased on wide-FOV rigs (MapAnything has no wide-angle training data), and on
these fisheye GoPro field scenes it measured 0.71-0.75 of true scale, versus
~1.02 on small indoor rooms.

    pixi run python scripts/run_ma_baseline_egoexo.py [--take cmu_soccer06_3]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/users/tnanni/ghost")
from preprocessing.run_mapanything import MapAnythingScaleEstimator  # noqa: E402

OUT    = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d")
FRAMES = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default=None, help="restrict to a single take")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--force", action="store_true", default=False,
                    help="recompute even when the baseline file already exists")
    args = ap.parse_args()

    est = MapAnythingScaleEstimator(batch_size=args.batch_size,
                                    force=args.force)

    n_done = n_skip = n_fail = 0
    for scene_dir in sorted(OUT.iterdir()):
        if not scene_dir.is_dir():
            continue
        if args.take and scene_dir.name != args.take:
            continue
        if not (scene_dir / "vggt_cameras_centered.npz").exists():
            continue
        if (scene_dir / "mapanything_scale_baseline.npy").exists() and not args.force:
            n_skip += 1
            continue
        # EgoExo4D layout: frames/<take>/<cam>/frames/frame_XXXXXX.jpg.  process_scene
        # descends into per-camera sub-directories, so the take dir is the right root.
        img_root = FRAMES / scene_dir.name
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
