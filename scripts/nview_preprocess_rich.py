#!/usr/bin/env python
"""N-view RICH ablation: re-run VGGT + MapAnything on the first N cameras only.

Builds a SHADOW output root that mirrors a production ghost scene root but with
camera geometry estimated from a reduced camera set:

    <out_root>/<scene>/vggt_cameras_centered.npz    (N views)
    <out_root>/<scene>/vggt_depth_centered.npz      (N views, deleted after MA)
    <out_root>/<scene>/mapanything_scale_baseline.npy
    <out_root>/<scene>/<cam>/body_data/person_*.npz (COPIED from src_root)
    <out_root>/<scene>/nview_manifest.json          (provenance)

Nothing is ever written inside --src_root; it is opened read-only. The shadow
root is laid out so that evaluation/evaluate_rich_median.py runs against it
UNMODIFIED (it discovers scenes by vggt_cameras_centered.npz and cameras by
<cam>/body_data), which keeps the M10 protocol, the MapAnything-baseline scale
and the always-median scale smoothing bit-identical to the 4-view numbers. The
only thing that differs is how many cameras VGGT/MapAnything/the fusion saw.

Why body_data is reused as-is: SAM3D fits each camera crop independently, so
per-camera body parameters do not depend on how many cameras exist. The ablation
therefore isolates (a) VGGT camera geometry from fewer views, (b) the
MapAnything baseline-ratio scale from fewer baselines, and (c) fusing fewer
views — which is exactly the intended question.

Camera choice: the first N names of the production camera list, sorted. Sorting
is what both the production pipeline and the evaluator use, so camera 0 (the
VGGT world origin and the evaluator's reference camera) is unchanged, and the
world frame stays comparable across N.

T is capped to the production T. Dropping a camera can only *raise* the
min-frames-across-cameras count, and a longer T than the one body_data was
written against would desync the frame axis.

Scenes whose production camera count is already <= N are copied over verbatim
(no recompute) so that N=3 reproduces the 4-view result exactly on 3-camera
scenes instead of re-rolling VGGT on the same inputs.

Resumable: a scene is skipped when its cameras npz, scale npy and body_data are
all present in the shadow root.

Usage (images live inside centered_<split>.sqsh, so mount it first):

    pixi run python scripts/nview_preprocess_rich.py \
        --n_views 2 \
        --img_root /tmp/centered_test_mnt \
        --src_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \
        --out_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test_nview2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configuration import CONFIG                                     # noqa: E402
from preprocessing.run_mapanything import MapAnythingScaleEstimator  # noqa: E402
from preprocessing.run_vggt import VGGTPreprocessor                  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nview")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CAM_NPZ   = "vggt_cameras_centered.npz"
DEPTH_NPZ = "vggt_depth_centered.npz"
SCALE_NPY = "mapanything_scale_baseline.npy"
MANIFEST  = "nview_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_images(cam_dir: Path) -> list[Path]:
    """Sorted image files for one camera.

    Mirrors the directory search in MapAnythingScaleEstimator.process_scene
    (images directly in cam_dir, else one or two levels deeper) so that VGGT and
    MapAnything are guaranteed to consume the same files in the same order.
    """
    if not cam_dir.is_dir():
        return []
    files = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if files:
        return files
    for sub in sorted(p for p in cam_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS)
        if files:
            return files
        for sub2 in sorted(p for p in sub.iterdir() if p.is_dir()):
            files = sorted(p for p in sub2.iterdir() if p.suffix.lower() in IMG_EXTS)
            if files:
                return files
    return []


def prod_camera_names(cam_npz_path: Path) -> tuple[list[str], int]:
    """(sorted production camera names, production T) from a cameras npz."""
    z = np.load(cam_npz_path, mmap_mode="r")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["camera_names"]]
    return sorted(names), int(z["extrinsics"].shape[0])


def copy_body_data(src_scene: Path, out_scene: Path, cams: list[str],
                   link: bool) -> None:
    """Materialise <cam>/body_data for the selected cameras in the shadow root.

    Copied by default so the shadow root is self-contained and the production
    root is never referenced at evaluation time. json_data / mask_data.npz are
    intentionally not brought over: neither the evaluator nor BodyPlacer reads
    them.
    """
    for cam in cams:
        src_bd = src_scene / cam / "body_data"
        dst_bd = out_scene / cam / "body_data"
        if dst_bd.exists() or dst_bd.is_symlink():
            continue
        if not src_bd.is_dir():
            logger.warning(f"  {cam}: no body_data in source — evaluator will drop it")
            continue
        dst_bd.parent.mkdir(parents=True, exist_ok=True)
        if link:
            dst_bd.symlink_to(src_bd.resolve())
        else:
            shutil.copytree(src_bd, dst_bd)


def scene_done(out_scene: Path, cams: list[str]) -> bool:
    return (
        (out_scene / CAM_NPZ).exists()
        and (out_scene / SCALE_NPY).exists()
        and all((out_scene / c / "body_data").is_dir() for c in cams)
    )


# ---------------------------------------------------------------------------
# Per-scene work
# ---------------------------------------------------------------------------

def process_scene(
    scene:      str,
    src_root:   Path,
    out_root:   Path,
    img_root:   Path,
    n_views:    int,
    vggt:       VGGTPreprocessor,
    ma:         MapAnythingScaleEstimator,
    devices:    list[str],
    keep_depth: bool,
    link_body:  bool,
) -> str:
    """Returns one of: 'done', 'reused', 'skipped', 'failed'."""
    src_scene = src_root / scene
    out_scene = out_root / scene
    src_cam_npz = src_scene / CAM_NPZ
    if not src_cam_npz.exists():
        logger.warning(f"{scene}: no production {CAM_NPZ} — skip")
        return "skipped"

    all_cams, T_prod = prod_camera_names(src_cam_npz)
    cams = all_cams[:n_views]
    out_scene.mkdir(parents=True, exist_ok=True)

    if scene_done(out_scene, cams):
        logger.info(f"{scene}: already complete — skip")
        return "skipped"

    # ── Production camera count already <= N: copy, do not recompute ──────────
    if len(all_cams) <= n_views:
        logger.info(f"{scene}: production has {len(all_cams)} cams <= N={n_views} "
                    f"— copying production artefacts verbatim")
        shutil.copy2(src_cam_npz, out_scene / CAM_NPZ)
        src_scale = src_scene / SCALE_NPY
        if not src_scale.exists():
            logger.warning(f"{scene}: production {SCALE_NPY} missing — cannot reuse")
            return "failed"
        shutil.copy2(src_scale, out_scene / SCALE_NPY)
        copy_body_data(src_scene, out_scene, cams, link_body)
        (out_scene / MANIFEST).write_text(json.dumps({
            "scene": scene, "n_views_requested": n_views,
            "cameras": cams, "production_cameras": all_cams,
            "T": T_prod, "mode": "reused_production",
            "src_root": str(src_root),
        }, indent=2))
        return "reused"

    # ── Recompute VGGT on the first N cameras ─────────────────────────────────
    scene_img = img_root / scene
    if not scene_img.is_dir():
        logger.warning(f"{scene}: no images at {scene_img} — skip")
        return "failed"

    per_cam_files = {c: list_images(scene_img / c) for c in cams}
    missing = [c for c, f in per_cam_files.items() if not f]
    if missing:
        logger.warning(f"{scene}: no images for {missing} — skip")
        return "failed"

    T = min([T_prod] + [len(f) for f in per_cam_files.values()])
    if T < 2:
        logger.warning(f"{scene}: T={T} — skip")
        return "failed"
    if T != T_prod:
        logger.warning(f"{scene}: T={T} < production T={T_prod} (short camera)")

    frame_paths: list[list[Path | None]] = [
        [per_cam_files[c][t] for c in cams] for t in range(T)
    ]

    # Depth is only an input to the MapAnything step, and it is deleted once that
    # step has produced the scale. So a missing depth npz is only a reason to
    # re-run VGGT when the scale is missing too — otherwise a job killed after
    # the depth deletion (e.g. the 1:30 debug limit landing mid-scene) would
    # redo the whole VGGT pass for a scene that already has cameras and scale.
    if not (out_scene / CAM_NPZ).exists() or (
        not (out_scene / DEPTH_NPZ).exists() and not (out_scene / SCALE_NPY).exists()
    ):
        logger.info(f"{scene}: VGGT on {len(cams)} cams {cams}  T={T}  devices={devices}")
        t0 = time.perf_counter()
        vggt.process_scene(
            frame_paths  = frame_paths,
            camera_names = cams,
            output_dir   = out_scene,
            devices      = devices,
        )
        torch.cuda.empty_cache()
        logger.info(f"{scene}: VGGT done in {(time.perf_counter() - t0) / 60:.1f} min")
    else:
        logger.info(f"{scene}: VGGT outputs present — reusing")

    # ── MapAnything baseline-ratio scale over the same N cameras ──────────────
    # The estimator takes its camera list from the npz we just wrote, so it is
    # automatically restricted to the N views.
    if not (out_scene / SCALE_NPY).exists():
        t0 = time.perf_counter()
        res = ma.process_scene(scene_dir=out_scene, img_root=scene_img)
        torch.cuda.empty_cache()
        if res is None:
            logger.warning(f"{scene}: MapAnything scale failed")
            return "failed"
        logger.info(f"{scene}: MA scale median={float(np.median(res)):.4f} "
                    f"in {(time.perf_counter() - t0) / 60:.1f} min")

    if not keep_depth:
        dp = out_scene / DEPTH_NPZ
        if dp.exists():
            dp.unlink()   # only MapAnything needs it; the evaluator never reads it

    copy_body_data(src_scene, out_scene, cams, link_body)
    (out_scene / MANIFEST).write_text(json.dumps({
        "scene": scene, "n_views_requested": n_views,
        "cameras": cams, "production_cameras": all_cams,
        "T": T, "T_production": T_prod, "mode": "recomputed",
        "src_root": str(src_root), "img_root": str(img_root),
    }, indent=2))
    return "done"


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_views",  type=int, required=True,
                   help="Number of cameras to keep (first N, sorted by name).")
    p.add_argument("--img_root", type=Path, required=True,
                   help="Mounted centered_<split>.sqsh root: <scene>/<cam>/*.jpg")
    p.add_argument("--src_root", type=Path, required=True,
                   help="Production ghost output root (READ-ONLY here).")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Shadow output root for this N. Must not be inside src_root.")
    p.add_argument("--scenes",       default="", help="Comma-separated subset (default: all).")
    p.add_argument("--scene_start",  type=int, default=None)
    p.add_argument("--scene_end",    type=int, default=None)
    p.add_argument("--devices",      nargs="+", default=None,
                   help="CUDA devices for VGGT, e.g. cuda:0 cuda:1. Default: all visible.")
    p.add_argument("--batch_size",   type=int, default=8, help="MapAnything frames per call.")
    p.add_argument("--vggt_weights", default=CONFIG.data.vggt_omega_checkpoint)
    p.add_argument("--keep_depth",   action="store_true",
                   help="Keep vggt_depth_centered.npz (~1 GB/scene). Default: delete "
                        "after the MapAnything scale is computed.")
    p.add_argument("--link_body_data", action="store_true",
                   help="Symlink body_data instead of copying it (saves ~2.5 GB, but "
                        "the shadow root then points back into src_root).")
    args = p.parse_args()

    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    if out_root == src_root or src_root in out_root.parents:
        raise SystemExit(f"REFUSING: out_root {out_root} is inside/equal to src_root {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.devices:
        devices = args.devices
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]

    only = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = sorted(d.name for d in src_root.iterdir()
                    if d.is_dir() and (d / CAM_NPZ).exists()
                    and (not only or d.name in only))
    if args.scene_start is not None or args.scene_end is not None:
        scenes = scenes[args.scene_start:args.scene_end]

    logger.info(f"N={args.n_views}  scenes={len(scenes)}  devices={devices}")
    logger.info(f"src (read-only): {src_root}")
    logger.info(f"out            : {out_root}")

    vggt = VGGTPreprocessor(weights=args.vggt_weights, device=devices[0])
    ma   = MapAnythingScaleEstimator(device=devices[0], batch_size=args.batch_size,
                                     scale_from="baselines")

    tally = {"done": 0, "reused": 0, "skipped": 0, "failed": 0}
    t_all = time.perf_counter()
    for i, scene in enumerate(scenes, 1):
        logger.info(f"─── [{i}/{len(scenes)}] {scene} ───")
        try:
            status = process_scene(
                scene=scene, src_root=src_root, out_root=out_root,
                img_root=args.img_root, n_views=args.n_views,
                vggt=vggt, ma=ma, devices=devices,
                keep_depth=args.keep_depth, link_body=args.link_body_data,
            )
        except Exception as e:
            logger.exception(f"{scene}: FAILED — {e}")
            status = "failed"
        tally[status] += 1

    print(f"NVIEW_SWEEP_DONE n_views={args.n_views} "
          f"done={tally['done']} reused={tally['reused']} "
          f"skipped={tally['skipped']} failed={tally['failed']} "
          f"time={(time.perf_counter() - t_all) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
