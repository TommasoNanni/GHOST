"""STEP 1 of the through-sync EgoHumans evaluation: rebuild the INEXACT scenes.

Reads the manifest from evaluation/sync_dryrun_egohumans.py and, for every trial
whose sync estimate was NOT exact, materialises a self-contained trial directory
built on the ESTIMATED alignment:

    <sync_output_root>/<activity>/<scene>/trial<k>/
        <cam>/body_data_clean/person_*.npz   (windowed + relabeled, NOT re-fitted)
        vggt_cameras_centered.npz            (rerun on the estimated alignment)
        vggt_depth_centered.npz
        mapanything_scale_baseline.npy       (rerun on the same alignment)
        sync_meta.json                       (trial entry copied from the manifest)

Body parameters are never re-estimated. SAM3D ran per camera on frames the desync
does not change -- a camera's own footage is the same footage whatever the offset
between cameras is -- so the existing person_*.npz are still valid and only their
frame windowing changes. What genuinely changes is the MULTI-VIEW reconstruction:
with a wrong offset, camera c contributes the image at a different real instant,
so VGGT sees an inconsistent set of views and MapAnything a different scene. That
is exactly the error this evaluation is trying to measure, so those two do rerun.

EXACT trials are deliberately not built. When shift_hat == true_shift the read
index collapses to max_s + T0 + t in every camera, i.e. a contiguous slice of the
original synchronised stream, so the scene's existing VGGT/MapAnything outputs are
already valid on that window and STEP 2 reads them in place. Building a trial dir
for them would only copy bytes.

Images come from the undistorted exo tree that the pipeline itself ran on
(<undist_root>/<activity>/<INNER>/<activity>/<scene>/exo/<cam>/images_undistorted),
fetched by scripts/fetch_egohumans_inexact.py. Frame LABELS (1-based, matching
00001.jpg and the GT's 00001.npy) are converted to positions in that sorted list
by subtracting frame_label_min, the same convention evaluate_egohumans_median.py
uses when it does `t = gfr - fmin`.

Usage
-----
    pixi run python evaluation/sync_build_egohumans.py \\
        --manifest         eval_egohumans/sync_manifest_s75.json \\
        --undist_root      /iopsstor/scratch/cscs/tnanni/sync_egohumans_undistorted \\
        --sync_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_sync_s75 \\
        --vggt_weights     checkpoints/vggt_omega/vggt_omega_1b_512.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.run_vggt import VGGTPreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_camera_frames(cam_dir: Path) -> list[Path]:
    """Sorted undistorted images for one exo camera.

    Prefers images_undistorted/frames/ (created when an earlier pipeline run
    resized in place) and falls back to images_undistorted/ itself -- the same
    two-case lookup scripts/egohumans_pipeline.py does.
    """
    und = cam_dir / "images_undistorted"
    if not und.is_dir():
        return []
    sub = und / "frames"
    d = sub if (sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in sub.iterdir())) else und
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def window_and_relabel(data: dict[str, np.ndarray], s_hat: int, T_hat: int,
                       base: int) -> dict[str, np.ndarray] | None:
    """Keep frames whose LABEL falls in [s_hat, s_hat+T_hat), relabel to start at `base`.

    Every per-frame array (leading dim == len(frame_indices)) is sliced by the same
    boolean mask; anything else is copied through. Relabeling keeps the original
    label base so the rebuilt tracks stay in the same convention as the untouched
    scenes (1-based), which is what STEP 2's `gfr - fmin` arithmetic expects.
    Returns None if no frame of this person survives the window.
    """
    if "frame_indices" not in data:
        return None
    fi = data["frame_indices"].astype(int)
    mask = (fi >= s_hat) & (fi < s_hat + T_hat)
    if not mask.any():
        return None
    n = len(fi)
    out: dict[str, np.ndarray] = {}
    for k, v in data.items():
        out[k] = v[mask] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n) else v
    out["frame_indices"] = (fi[mask] - s_hat + base).astype(data["frame_indices"].dtype)
    return out


def build_trial(scene_ghost_dir: Path, seq_img_dir: Path, trial_dir: Path,
                trial: dict, base: int, preprocessor: VGGTPreprocessor,
                vggt_devices: list[str],
                ma_estimator: MapAnythingScaleEstimator | None) -> None:
    """Window body_data_clean and rerun VGGT (+ MapAnything) for one inexact trial.

    cam_ids come from the manifest in VGGT camera_names order, and every camera
    directory is created even if a camera ends up with no surviving person --
    BodyPlacer and MapAnythingScaleEstimator positionally align their own sorted
    directory listings against the VGGT camera axis, so a missing directory would
    shift that correspondence.
    """
    cam_ids = trial["cam_ids"]
    s_hat   = trial["s_hat"]
    T_hat   = int(trial["T_hat"])
    trial_dir.mkdir(parents=True, exist_ok=True)

    # frame_paths[t][k]: label s_hat[c] + t -> position (label - base) in the sorted list.
    per_cam_frames = {c: list_camera_frames(seq_img_dir / "exo" / c) for c in cam_ids}
    for c in cam_ids:
        need = s_hat[c] - base + T_hat
        if len(per_cam_frames[c]) < need:
            raise FileNotFoundError(
                f"{seq_img_dir.name}/{c}: {len(per_cam_frames[c])} undistorted image(s), "
                f"need {need} (labels {s_hat[c]}..{s_hat[c] + T_hat - 1}). Fetch it with "
                f"scripts/fetch_egohumans_inexact.py before building.")
    frame_paths = [[per_cam_frames[c][s_hat[c] - base + t] for c in cam_ids] for t in range(T_hat)]

    for c in cam_ids:
        out_dir = trial_dir / c / "body_data_clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        src_dir = scene_ghost_dir / c / "body_data_clean"
        for npz_path in sorted(src_dir.glob("person_*.npz")):
            data = dict(np.load(npz_path, allow_pickle=False))
            windowed = window_and_relabel(data, s_hat[c], T_hat, base)
            if windowed is not None:
                np.savez_compressed(out_dir / npz_path.name, **windowed)

    preprocessor.process_scene(frame_paths=frame_paths, camera_names=cam_ids,
                               output_dir=trial_dir, devices=vggt_devices)

    if ma_estimator is not None:
        # MapAnythingScaleEstimator.process_scene needs a real img_root/<cam>/ tree
        # (sorted, positionally matching the VGGT T axis) and cannot take an explicit
        # frame list, so materialise the same selection as symlinks named by local
        # index -- sort order then equals VGGT frame order whatever the source naming.
        images_dir = trial_dir / "_ma_images"
        for ci, c in enumerate(cam_ids):
            cam_img_dir = images_dir / c
            cam_img_dir.mkdir(parents=True, exist_ok=True)
            for t in range(T_hat):
                src = frame_paths[t][ci]
                dst = cam_img_dir / f"{t:06d}{src.suffix.lower()}"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
        ma_estimator.process_scene(scene_dir=trial_dir, img_root=images_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--undist_root", required=True, type=Path,
                   help="Root holding <activity>/<INNER>/<activity>/<scene>/exo/<cam>/images_undistorted.")
    p.add_argument("--sync_output_root", required=True, type=Path)
    p.add_argument("--ghost_root", type=Path, default=None,
                   help="Override the manifest's ghost_root (source of body_data_clean).")
    p.add_argument("--vggt_weights", default=None)
    p.add_argument("--vggt_devices", nargs="+", default=None)
    p.add_argument("--skip_mapanything", action="store_true")
    p.add_argument("--mapanything_device", default=None)
    p.add_argument("--mapanything_batch_size", type=int, default=8)
    p.add_argument("--scenes", default="", help="Comma-separated '<activity>/<scene>' filter.")
    p.add_argument("--min_error", type=int, default=1,
                   help="Only build trials whose max per-camera sync error is >= this.")
    p.add_argument("--max_scenes", type=int, default=None)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    ghost_root = args.ghost_root or Path(manifest["config"]["ghost_root"])
    only = {s.strip() for s in args.scenes.split(",") if s.strip()}

    todo: list[tuple[str, dict]] = []
    for key, entry in manifest["scenes"].items():
        if only and key not in only:
            continue
        for trial in entry["trials"]:
            if not trial["exact"] and trial["max_sync_error"] >= args.min_error:
                todo.append((key, trial))
    todo.sort(key=lambda kt: -kt[1]["max_sync_error"])
    if args.max_scenes:
        todo = todo[:args.max_scenes]
    logger.info(f"{len(todo)} inexact trial(s) to build "
                f"(of {sum(len(e['trials']) for e in manifest['scenes'].values())} total)")

    devices = args.vggt_devices or (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"])
    preprocessor = VGGTPreprocessor(weights=args.vggt_weights, device=devices[0])
    ma_estimator = None
    if not args.skip_mapanything:
        ma_estimator = MapAnythingScaleEstimator(
            device=args.mapanything_device or devices[0],
            batch_size=args.mapanything_batch_size)

    n_built = n_skip = n_fail = 0
    for i, (key, trial) in enumerate(todo, 1):
        activity, scene = key.split("/", 1)
        k = trial["trial"]
        trial_dir = args.sync_output_root / activity / scene / f"trial{k}"
        head = f"[{i}/{len(todo)}] {key} trial{k} (max_err={trial['max_sync_error']})"

        if (trial_dir / "sync_meta.json").exists():
            logger.info(f"{head}: already built -- skip")
            n_skip += 1
            continue

        base = int(trial.get("frame_label_min", manifest["scenes"][key].get("frame_label_min", 0)))
        seq_img_dir = args.undist_root / activity / INNER / activity / scene
        logger.info(f"\n{'='*66}\n{head}\n{'='*66}")
        try:
            build_trial(ghost_root / activity / scene, seq_img_dir, trial_dir, trial,
                        base, preprocessor, devices, ma_estimator)
        except Exception as e:
            logger.error(f"{head}: FAILED -- {e}", exc_info=True)
            n_fail += 1
            continue

        meta = dict(trial)
        meta.update({"scene": key, "frame_label_min": base,
                     "ghost_scene_dir": str(ghost_root / activity / scene),
                     "max_shift": manifest["config"]["max_shift"]})
        with open(trial_dir / "sync_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        n_built += 1
        logger.info(f"{head}: built -> {trial_dir}")

    logger.info(f"\n{'='*66}\nDone. built={n_built} skipped={n_skip} failed={n_fail} "
                f"-> {args.sync_output_root}\n{'='*66}")


if __name__ == "__main__":
    main()
