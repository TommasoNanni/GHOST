"""
EgoExo4D PRODUCTION pipeline — SAM3 segmentation + **GT-driven ReID** + SAM3D body
(GT subject only) + VGGT + MapAnything.

Same stages as the old ``egoexo4d_pipeline.py`` (now in ``backup/superseded/``) with
cross-view ReID (``CrossViewReidentifierV4``) replaced by GT projection, and the GT
ReID run **before** SAM3D so SAM3D only ever sees the annotated subject.

Steps: 1 SAM3 -> 2 GT ReID -> 3 SAM3D -> 4 body check -> 5 VGGT -> 6 MapAnything.

EgoExo4D specifics
------------------
* One annotated frame per take: ``<gt-root>/<take>/keypoints_gt.json`` holds a single
  ``{frame_idx: {joint_name: {x, y, z, num_views_for_3d}}}`` entry — triangulated
  COCO-17 joints of ONE subject in the GT world frame. The frame extractor
  (``utilities/extract_egoexo4d_frames.py``) pulls only that frame per camera, so a
  take is 1 frame x N GoPros.
* Cameras: ``<gt-root>/<take>/gopro_calibs.csv`` (Kannala-Brandt fisheye). The frames
  were undistorted in place by ``utilities/undistort_egoexo4d.py`` with
  ``balance=0``, so the pinhole camera of the frames is
  ``Knew = fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), I, 0)``
  — identical to ``evaluation/evaluate_egoexo_median.py:_gopro_cameras``.

Matching
--------
Per camera, on the annotated frame, every SAM3 track is scored against the projected
GT skeleton with

    score = 0.5 * (fraction of projected GT joints inside the track mask)
          + 0.5 * IoU(track bbox, padded bbox of the projected GT joints)

The best track wins if ``score >= --min-score`` AND it beats the runner-up by
``--min-margin`` (a near-tie, e.g. a dance partner, is refused, never guessed — same
spirit as the eval's ``_MATCH_RATIO`` guard). The winner becomes ``person_1``
(``GT_PERSON_PID`` in the eval); every other track is erased from ``mask_data.npz``
and ``json_data/``. A refused camera is emptied and logged in ``gt_reid.json``.

Pixel space
-----------
``Knew`` is scaled from the calibration size (``image_width`` x ``image_height`` in the
csv) to the json ``mask_width`` x ``mask_height`` — exactly the frames SAM3 segmented
(1440x810 with ``egoexo4d_max_side: 1440``).

    pixi run python scripts/egoexo4d_pipeline_production.py \\
        --frames-root /capstor/scratch/cscs/tnanni/datasets/egoexo4d/frames \\
        --gt-root     /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \\
        --output-dir  /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \\
        --take cmu_soccer06_3
"""

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "MHR" / "tools" / "mhr_smpl_conversion"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

import csv
import gc
import io
import json
import zipfile

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

from configuration import CONFIG
from data.video_dataset import EgoExo4DSceneDataset
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction_v2 import ParametersExtractor
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.run_vggt import VGGTPreprocessor


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Within-video ReID merges on combined similarity >= threshold; similarities are <= 1,
# so 2.0 disables merging. Ids are GT identities already and must never be merged.
_NO_MERGE_REID_THRESHOLD = 2.0

# The single annotated subject is person 1 (evaluate_egoexo_median.py GT_PERSON_PID).
GT_PERSON_PID = 1

# keypoints_gt.json joint names used for projection (COCO-17).
GT_JOINTS = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
)


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def frame_index(stem: str) -> int:
    """First purely-numeric segment of a frame / json stem: 'mask_frame_001426' -> 1426.

    Same rule the body estimator uses to parse json names.
    """
    return int(next(p for p in stem.replace("mask_", "").split("_") if p.isdigit()))


def load_gt_joints(gt_take_dir: Path) -> tuple[int, np.ndarray] | None:
    """(annotated frame index, (J, 3) world joints with 3D support)."""
    kp_path = gt_take_dir / "keypoints_gt.json"
    if not kp_path.exists():
        return None
    frame_str, joints_raw = next(iter(json.loads(kp_path.read_text()).items()))
    X = [[v["x"], v["y"], v["z"]] for n, v in joints_raw.items()
         if n in GT_JOINTS and v.get("num_views_for_3d", 0) > 0]
    if not X:
        return None
    return int(frame_str), np.asarray(X, dtype=np.float64)


def load_cameras(gt_take_dir: Path) -> dict[str, dict]:
    """Undistorted-pinhole GoPro cameras at calibration resolution, keyed by cam_uid."""
    cams: dict[str, dict] = {}
    with open(gt_take_dir / "gopro_calibs.csv") as f:
        for row in csv.DictReader(f):
            K = np.array([[float(row["intrinsics_0"]), 0, float(row["intrinsics_2"])],
                          [0, float(row["intrinsics_1"]), float(row["intrinsics_3"])],
                          [0, 0, 1]])
            D = np.array([[float(row[f"intrinsics_{i}"])] for i in range(4, 8)])
            W, H = int(row["image_width"]), int(row["image_height"])
            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (W, H), np.eye(3), balance=0.0)
            cams[row["cam_uid"]] = {
                "Knew": Knew, "W": W, "H": H,
                "t": np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"]),
                "Rwc": SciR.from_quat(
                    [float(row[f"q{a}_world_cam"]) for a in "xyzw"]).as_matrix(),
            }
    return cams


def project(cam: dict, X: np.ndarray, work_w: int, work_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Project (J, 3) world points into the SAM3 frame. Returns (uv (J, 2), in_front (J,))."""
    Xc = (X - cam["t"]) @ cam["Rwc"]            # == (Rwc.T @ (X - t).T).T
    z = Xc[:, 2]
    zs = np.where(z > 0.05, z, 1.0)
    sx, sy = work_w / cam["W"], work_h / cam["H"]
    u = (cam["Knew"][0, 0] * Xc[:, 0] / zs + cam["Knew"][0, 2]) * sx
    v = (cam["Knew"][1, 1] * Xc[:, 1] / zs + cam["Knew"][1, 2]) * sy
    return np.stack([u, v], 1), z > 0.05


# ---------------------------------------------------------------------------
# Per-camera GT matching
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_camera(
    cam_dir: Path,
    cam: dict,
    gt_frame: int,
    gt_X: np.ndarray,
    min_score: float,
    min_margin: float,
    min_joints: int,
) -> tuple[dict[str, dict[int, int]], dict]:
    """``assign[json_stem][sam3_tid] = GT_PERSON_PID`` for the annotated frame only."""
    stats: dict = {"gt_frame": gt_frame}
    jp = next((p for p in sorted((cam_dir / "json_data").glob("mask_*.json"))
               if frame_index(p.stem) == gt_frame), None)
    if jp is None:
        stats["error"] = "annotated frame not segmented"
        return {}, stats
    meta = json.loads(jp.read_text())
    labels = meta.get("labels", {})
    W, H = meta["mask_width"], meta["mask_height"]
    stats["sam3_size"] = [W, H]
    stats["calib_size"] = [cam["W"], cam["H"]]
    stats["sam3_tracks"] = sorted(int(k) for k in labels)
    if not labels:
        stats["error"] = "no SAM3 detections"
        return {}, stats

    uv, front = project(cam, gt_X, W, H)
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    stats["gt_joints_in_frame"] = int(ok.sum())
    if ok.sum() < min_joints:
        stats["error"] = f"GT subject not visible ({int(ok.sum())} joints in frame)"
        return {}, stats
    u = np.round(uv[ok]).astype(int)
    lo, hi = uv[ok].min(0), uv[ok].max(0)
    pad = 0.1 * (hi - lo)
    gt_box = (lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[1])

    with zipfile.ZipFile(str(cam_dir / "mask_data.npz"), "r") as zf:
        with zf.open(f"{jp.stem}.npy") as f:
            mask = np.load(io.BytesIO(f.read()))

    scores = []
    for str_id, info in labels.items():
        tid = int(str_id)
        inside = float(np.mean(mask[u[:, 1], u[:, 0]] == tid))
        iou = _iou((info["x1"], info["y1"], info["x2"], info["y2"]), gt_box)
        scores.append((0.5 * inside + 0.5 * iou, tid))
    scores.sort(reverse=True)
    stats["scores"] = {str(t): round(s, 3) for s, t in scores}

    best, tid = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    if best < min_score:
        stats["error"] = f"best score {best:.2f} < {min_score}"
        return {}, stats
    if best - second < min_margin:
        stats["error"] = f"ambiguous: track {tid} {best:.2f} vs runner-up {second:.2f}"
        return {}, stats
    stats["matched_track"] = tid
    return {jp.stem: {tid: GT_PERSON_PID}}, stats


def apply_assignment(cam_dir: Path, assign: dict[str, dict[int, int]]) -> None:
    """Rewrite mask_data.npz and json_data/ with the GT id; erase everything unmatched.

    The kept track's pixels become ``GT_PERSON_PID``; every other id becomes 0
    (background) and its json label is removed. Frames absent from ``assign`` are
    emptied entirely.
    """
    npz_path = cam_dir / "mask_data.npz"
    tmp_path = npz_path.with_suffix(".tmp.npz")
    with (
        zipfile.ZipFile(str(npz_path), "r") as zf_in,
        zipfile.ZipFile(str(tmp_path), "w",
                        compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf_out,
    ):
        for name in sorted(zf_in.namelist()):
            with zf_in.open(name) as f:
                mask = np.load(io.BytesIO(f.read()))
            new_mask = np.zeros_like(mask)
            for tid, pid in assign.get(Path(name).stem, {}).items():
                new_mask[mask == tid] = pid
            buf = io.BytesIO()
            np.save(buf, new_mask)
            zf_out.writestr(name, buf.getvalue())
    tmp_path.replace(npz_path)

    for jp in sorted((cam_dir / "json_data").glob("mask_*.json")):
        meta = json.loads(jp.read_text())
        tmap = assign.get(jp.stem, {})
        new_labels = {}
        for str_id, info in meta.get("labels", {}).items():
            tid = int(str_id)
            if tid not in tmap:
                continue
            info["instance_id"] = tmap[tid]
            new_labels[str(tmap[tid])] = info
        meta["labels"] = new_labels
        jp.write_text(json.dumps(meta))


def gt_reid_scene(scene, video_dirs: dict[str, Path], gt_take_dir: Path, args) -> dict | None:
    """Relabel every camera of a take from the projected GT. Returns the gt_reid.json plan."""
    gt = load_gt_joints(gt_take_dir)
    if gt is None:
        return None
    gt_frame, gt_X = gt
    cams = load_cameras(gt_take_dir)
    plan: dict = {
        "scene": scene.scene_id,
        "gt_frame": gt_frame,
        "gt_joints": int(len(gt_X)),
        "person_id": GT_PERSON_PID,
        "params": {"min_score": args.min_score, "min_margin": args.min_margin,
                   "min_joints": args.min_joints},
        "cameras": {},
    }
    for video in scene.videos:
        cam_id = video.video_id
        cam_dir = Path(video_dirs[cam_id])
        if cam_id not in cams:
            logging.warning(f"  {cam_id}: not in gopro_calibs.csv — camera emptied")
            apply_assignment(cam_dir, {})
            plan["cameras"][cam_id] = {"error": "no calibration"}
            continue
        assign, stats = match_camera(cam_dir, cams[cam_id], gt_frame, gt_X,
                                     args.min_score, args.min_margin, args.min_joints)
        plan["cameras"][cam_id] = stats
        if "error" in stats:
            logging.warning(f"  {cam_id}: {stats['error']} — camera emptied")
        else:
            logging.info(f"  {cam_id}: SAM3 track {stats['matched_track']} -> "
                         f"person_{GT_PERSON_PID}  scores {stats['scores']}")
        apply_assignment(cam_dir, assign)
    plan["cameras_matched"] = sum("error" not in s for s in plan["cameras"].values())
    return plan


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def _build_vggt_frame_paths(scene) -> tuple[list[list[Path | None]], list[str]]:
    """frame_paths[t][k] from each camera's frames_home (the frames SAM3 segmented)."""
    sorted_videos = sorted(scene.videos, key=lambda v: v.video_id)
    camera_names = [v.video_id for v in sorted_videos]
    per_cam = []
    for video in sorted_videos:
        fdir = video.frames_home
        per_cam.append(sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
                       if fdir is not None and fdir.is_dir() else [])
    T = min((len(f) for f in per_cam), default=0)
    return [[f[t] for f in per_cam] for t in range(T)], camera_names


def _vggt_valid(cam_path: Path, dep_path: Path) -> bool:
    """Loadable, not just present: a job killed mid-write leaves a truncated npz."""
    try:
        np.load(cam_path)["extrinsics"]
        return bool(np.load(dep_path, mmap_mode="r")["depth_valid"].any())
    except Exception:
        return False


def _npy_valid(path: Path) -> bool:
    try:
        return np.load(path).size > 0
    except Exception:
        return False


def process_scene(scene, args, ma: MapAnythingScaleEstimator | None) -> None:
    print(f"\n=== Take: {scene.scene_id} ({len(scene)} cameras) ===")
    for v in scene:
        print(f"  {v}")

    scene_dir = args.output_dir / scene.scene_id
    gt_reid_path = scene_dir / "gt_reid.json"
    gt_take_dir = args.gt_root / scene.scene_id
    video_dirs = {v.video_id: scene_dir / v.video_id for v in scene.videos}

    if not (gt_take_dir / "keypoints_gt.json").exists():
        logging.error(f"  no GT at {gt_take_dir} — refusing to run SAM3D on unfiltered tracks")
        return

    # Step 1: Segmentation.
    if gt_reid_path.exists():
        print(f"\n--- Step 1: Segmentation (skipped — gt_reid.json exists) ---")
    else:
        print(f"\n--- Step 1: Segmentation ---")
        segmenter = PersonSegmenter(
            checkpoint_path=CONFIG.segmentation.checkpoint_path,
            text_prompt=CONFIG.segmentation.text_prompt,
            redetect_interval=CONFIG.segmentation.redetect_interval,
            new_det_thresh=CONFIG.segmentation.new_det_thresh,
            score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
            single_frame_mode=True,
        )
        video_dirs = segmenter.segment_scene(scene=scene, output_dir=str(args.output_dir), vis=False)
        del segmenter
        torch.cuda.empty_cache()

    # Step 2: GT-driven ReID (must run before SAM3D).
    if gt_reid_path.exists():
        print(f"\n--- Step 2: GT ReID (skipped — gt_reid.json exists) ---")
    else:
        print(f"\n--- Step 2: GT ReID ---")
        plan = gt_reid_scene(scene, video_dirs, gt_take_dir, args)
        if plan is None:
            logging.error(f"  keypoints_gt.json has no usable joints — stopping this take")
            return
        gt_reid_path.write_text(json.dumps(plan, indent=2))
        print(f"  matched {plan['cameras_matched']}/{len(scene.videos)} cameras")

    # Step 3: SAM3D on the GT subject only.
    print(f"\n--- Step 3: Body parameter estimation (GT subject only) ---")
    estimator = ParametersExtractor(
        sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
        sam3d_step=CONFIG.parameters_extraction.sam3d_step,
        bbox_padding=CONFIG.parameters_extraction.bbox_padding,
        smplx_model_path=CONFIG.data.smplx_model_path,
        mhr_model_path=CONFIG.data.mhr_model_path,
        reid_threshold=_NO_MERGE_REID_THRESHOLD,
        reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
        rich_data_root=None,
    )
    estimator.estimate_scene(scene=scene, video_dirs={k: str(v) for k, v in video_dirs.items()})
    del estimator
    torch.cuda.empty_cache()

    # Step 4: Verify body data + MHR -> SMPL-X conversion.
    print(f"\n--- Step 4: Verifying body data ---")
    for cam, vdir in video_dirs.items():
        files = sorted((Path(vdir) / "body_data").glob("person_*.npz"))
        if not files:
            print(f"  {cam}: no body data (camera refused in GT ReID or no detection)")
            continue
        has_smplx = all(any(k.startswith("smplx_") for k in np.load(p).files) for p in files)
        print(f"  {cam}: {[p.stem for p in files]}"
              + ("" if has_smplx else "  WARNING: missing smplx_* fields"))

    # Step 5: VGGT on the frames SAM3 segmented -> vggt_{cameras,depth}_centered.npz.
    # (EgoExo has no PP-centering step; the _centered names are what the eval reads.)
    cam_path = scene_dir / "vggt_cameras_centered.npz"
    dep_path = scene_dir / "vggt_depth_centered.npz"
    if _vggt_valid(cam_path, dep_path):
        print(f"\n--- Step 5: VGGT — already done, skipping ---")
    else:
        print(f"\n--- Step 5: VGGT ---")
        cam_path.unlink(missing_ok=True)
        dep_path.unlink(missing_ok=True)
        frame_paths, camera_names = _build_vggt_frame_paths(scene)
        if not frame_paths:
            logging.error("  no frames — skipping VGGT")
            return
        print(f"  {len(frame_paths)} frames x {len(camera_names)} cameras  devices {args.vggt_devices}")
        preprocessor = VGGTPreprocessor(weights=args.vggt_weights, device=args.vggt_devices[0])
        preprocessor.process_scene(frame_paths=frame_paths, camera_names=camera_names,
                                   output_dir=scene_dir, devices=args.vggt_devices)
        del preprocessor
        torch.cuda.empty_cache()

    # Step 6: MapAnything metric scale (same img_root as the old driver: the take dir).
    scale_path = scene_dir / "mapanything_scale_baseline.npy"
    if ma is None:
        print(f"\n--- Step 6: MapAnything skipped (--skip-mapanything) ---")
    elif _npy_valid(scale_path):
        print(f"\n--- Step 6: MapAnything — already done, skipping ---")
    else:
        print(f"\n--- Step 6: MapAnything scale ---")
        scale_path.unlink(missing_ok=True)
        ma.process_scene(scene_dir=scene_dir, img_root=args.frames_root / scene.scene_id)


def main():
    import argparse
    p = argparse.ArgumentParser(description="EgoExo4D production pipeline (GT-driven ReID)")
    p.add_argument("--frames-root", type=Path,
                   default=getattr(CONFIG.data, "egoexo4d_frames_root", None),
                   help="Undistorted frames: <root>/<take>/<cam>/frame_XXXXXX.jpg")
    p.add_argument("--gt-root", type=Path, required=True,
                   help="<root>/<take>/{keypoints_gt.json, gopro_calibs.csv}")
    p.add_argument("--output-dir", type=Path,
                   default=getattr(CONFIG.data, "egoexo4d_output_dir", None))
    p.add_argument("--take", default=None, help="Process only takes whose name contains this.")
    p.add_argument("--scene-start", type=int, default=None)
    p.add_argument("--scene-end", type=int, default=None)
    p.add_argument("--min-score", type=float, default=0.3,
                   help="Minimum track/GT match score to accept the subject.")
    p.add_argument("--min-margin", type=float, default=0.15,
                   help="Best score must beat the runner-up by this much, else the camera is refused.")
    p.add_argument("--min-joints", type=int, default=4,
                   help="Minimum in-frame projected GT joints for the subject to count as visible.")
    p.add_argument("--vggt-weights", default=CONFIG.data.vggt_omega_checkpoint)
    p.add_argument("--vggt-devices", nargs="+", default=None,
                   help="CUDA devices for VGGT/MapAnything; defaults to all visible GPUs.")
    p.add_argument("--skip-mapanything", action="store_true")
    p.add_argument("--mapanything-batch-size", type=int, default=8)
    args = p.parse_args()

    if args.frames_root is None or not Path(args.frames_root).is_dir():
        raise SystemExit(f"--frames-root {args.frames_root} does not exist")
    args.frames_root = Path(args.frames_root)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.vggt_devices is None:
        args.vggt_devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                             if torch.cuda.is_available() else ["cpu"])

    ds = EgoExo4DSceneDataset(frames_root=args.frames_root,
                              max_side=getattr(CONFIG.data, "egoexo4d_max_side", None))
    if args.take is not None:
        ds.scenes = [s for s in ds.scenes if args.take in s.scene_id]
    if args.scene_start is not None or args.scene_end is not None:
        ds.scenes = ds.scenes[args.scene_start:args.scene_end]
    print(f"{len(ds.scenes)} take(s): {[s.scene_id for s in ds.scenes]}")

    ma = None if args.skip_mapanything else MapAnythingScaleEstimator(
        device=args.vggt_devices[0], batch_size=args.mapanything_batch_size)

    failed: list[str] = []
    for scene in ds.scenes:
        try:
            process_scene(scene, args, ma)
        except Exception as e:
            logging.error(f"Take {scene.scene_id} failed: {e}", exc_info=True)
            failed.append(scene.scene_id)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    for scene in ds.scenes:
        sdir = args.output_dir / scene.scene_id
        reid_path = sdir / "gt_reid.json"
        matched = json.loads(reid_path.read_text()).get("cameras_matched") if reid_path.exists() else None
        vggt = _vggt_valid(sdir / "vggt_cameras_centered.npz", sdir / "vggt_depth_centered.npz")
        scale = _npy_valid(sdir / "mapanything_scale_baseline.npy")
        status = "FAILED" if scene.scene_id in failed else "ok"
        print(f"  {scene.scene_id}: {status}  gt-matched cams={matched if matched is not None else 'NO'}"
              f"/{len(scene.videos)}  vggt={'yes' if vggt else 'NO'}  scale={'yes' if scale else 'NO'}")


if __name__ == "__main__":
    main()
