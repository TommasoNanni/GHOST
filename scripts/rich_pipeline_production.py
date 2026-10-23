"""
RICH PRODUCTION pipeline — SAM3 segmentation + **GT-driven ReID** + SAM3D body (GT people only)
+ VGGT + MapAnything on the PP-centered frames.

Steps: 1 SAM3 -> 2 GT ReID -> 3 SAM3D -> 4 body check -> 5 VGGT -> 6 MapAnything.
Steps 5-6 run only with ``--centered-root`` (output of ``utilities/center_images.py``)
and write ``vggt_cameras_centered.npz``, ``vggt_depth_centered.npz`` and
``mapanything_scale_baseline.npy`` — the files ``fusion/placer.py`` loads.

Same segmentation / body stages as ``rich_pipeline_v3.py`` with two differences:

1. Cross-view ReID (``CrossViewReidentifierV4``) is replaced by GT projection. RICH GT
   SMPL-X bodies are posed, projected into every camera and matched to the SAM3 tracks
   frame by frame, so every kept mask carries the identity of a GT subject and ids are
   globally consistent across cameras *by construction*.
2. The GT ReID runs **before** SAM3D. Every SAM3 detection that does not match a GT
   subject (bystanders, false positives, duplicate tracks) is erased from
   ``mask_data.npz`` and ``json_data/``. SAM3D reads its person list from the json
   labels, so it only ever runs on GT subjects.

Matching
--------
Per frame and camera, every (SAM3 track, GT subject) pair is scored by

    score = 0.5 * (fraction of projected GT joints that land inside the track mask)
          + 0.5 * IoU(track bbox, padded bbox of the projected GT joints)

and solved with Hungarian assignment, so each GT subject gets at most one mask and each
mask at most one subject. A pair is kept if the mask touches the projected skeleton at
all (score > 0). Every frame is decided independently — nothing depends on SAM3 keeping
a track id, so lost / fragmented tracks and id switches cannot propagate. Frames with no
GT body get no subject and are emptied (they are never evaluated). A GT subject only
counts as visible in a frame with >= ``--min-joints`` projected joints inside the image.
SAM3D gets no GT intrinsics, as before.

Pixel space
-----------
The projection happens in exactly the pixel space SAM3 worked in: the target size is
the ``mask_width`` x ``mask_height`` stored in that camera's json (== mask_data.npz
frame shape). K from the XML is scaled from the full calibration resolution to that
size. The calibration resolution is read from a full-resolution frame in ``--raw-root``
when given, otherwise RICH's 4112x3008 (3008x4112 for portrait frames) is assumed.
Lens distortion coefficients are resolution-independent and are applied as-is.

Identity naming
---------------
RICH subject ids start at 0, which is the background value in the masks, so

    GT subject g  ->  person_{g + 1}

The scene-level ``gt_reid.json`` records the mapping, per-camera match statistics and
the dropped SAM3 tracks. It is also the "step done" marker.

Layout
------
    --data-root   <root>/<scene>/cam_XX/*.jpg   frames SAM3 runs on (e.g. test_840)
    --gt-root     <root>/<split>_body/<scene>/<frame>/<subject>.pkl
                  <root>/scan_calibration/<Location>/calibration/NNN.xml
    --raw-root    optional, <root>/<scene>/cam_XX/*.jpg full-resolution frames
    --centered-root optional, <root>/<scene>/cam_XX/*.jpg PP-centered frames (+ crop_meta.json)

    pixi run python scripts/rich_pipeline_production.py \\
        --data-root  /iopsstor/scratch/cscs/tnanni/datasets/rich/test_840 \\
        --gt-root    /iopsstor/scratch/cscs/tnanni/datasets/rich \\
        --raw-root   /iopsstor/scratch/cscs/tnanni/datasets/rich/test_raw \\
        --centered-root /iopsstor/scratch/cscs/tnanni/datasets/rich/centered_test \\
        --output-dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --scene Gym_010_cooking1
"""

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "MHR" / "tools" / "mhr_smpl_conversion"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

import io
import json
import pickle
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

import cv2
import numpy as np
import smplx
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from configuration import CONFIG
from data.video_dataset import RichDataset
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction_v2 import ParametersExtractor
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.run_vggt import VGGTPreprocessor


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# RICH full-resolution sensor size (landscape). Used only when --raw-root is absent.
_RICH_CALIB_W, _RICH_CALIB_H = 4112, 3008

# Within-video ReID merges on combined similarity >= threshold; similarities are <= 1,
# so 2.0 disables merging. Ids are GT identities already and must never be merged.
_NO_MERGE_REID_THRESHOLD = 2.0

# SMPL-X body joints (pelvis .. right wrist) used for projection.
_N_BODY_JOINTS = 22


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def scene_location(scene_name: str) -> str:
    """'Gym_010_cooking1' -> 'Gym', 'LectureHall_009_021_reparingprojector1' -> 'LectureHall'."""
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def frame_index(stem: str) -> int:
    """First purely-numeric segment of a frame / json stem: '00005_01' -> 5.

    Same rule the body estimator uses to parse json names
    (parameters_extraction_v2.py ``_process_video_core``).
    """
    return int(next(p for p in stem.replace("mask_", "").split("_") if p.isdigit()))


def load_calibration(gt_root: Path, scene_name: str, cam_id: str) -> dict | None:
    """Return {'K', 'R', 't', 'dist'} from ``scan_calibration/<loc>/calibration/NNN.xml``.

    K is in full calibration resolution; [R|t] is camera-from-world in the calibrated
    multi-camera frame, which is the frame the GT SMPL-X bodies live in.
    """
    cam_idx = int(cam_id.split("_")[-1])
    xml_path = (gt_root / "scan_calibration" / scene_location(scene_name)
                / "calibration" / f"{cam_idx:03d}.xml")
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()

    def _node(tag: str) -> np.ndarray | None:
        n = root.find(tag)
        if n is None or n.find("data") is None:
            return None
        return np.array(list(map(float, n.find("data").text.split())), dtype=np.float64)

    K, Rt, dist = _node("Intrinsics"), _node("CameraMatrix"), _node("Distortion")
    if K is None or Rt is None:
        return None
    Rt = Rt.reshape(3, 4)
    return {
        "K": K.reshape(3, 3),
        "R": Rt[:, :3],
        "t": Rt[:, 3],
        "dist": dist if dist is not None else np.zeros(5),
    }


def load_gt_joints(
    gt_root: Path,
    split: str,
    scene_name: str,
    body_models_dir: Path,
    gender_json: Path,
) -> dict[int, dict[int, np.ndarray]]:
    """Pose every GT subject with its gendered SMPL-X model.

    Returns ``joints[gt_pid][frame_idx] = (22, 3)`` body joints in the multi-camera frame.
    """
    scene_dir = gt_root / f"{split}_body" / scene_name
    if not scene_dir.is_dir():
        return {}

    params: dict[int, dict[int, dict]] = {}
    for frame_dir in sorted(scene_dir.iterdir()):
        if not frame_dir.is_dir() or not frame_dir.name.isdigit():
            continue
        fidx = int(frame_dir.name)
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            params.setdefault(int(pkl_path.stem), {})[fidx] = data

    genders = json.loads(gender_json.read_text()) if gender_json.exists() else {}

    joints: dict[int, dict[int, np.ndarray]] = {}
    for pid, frames in params.items():
        gender = genders.get(str(pid), "neutral")
        model_path = body_models_dir / f"SMPLX_{gender.upper()}.pkl"
        if not model_path.exists():
            logging.warning(f"  GT subject {pid}: {model_path.name} missing — using neutral")
            model_path = body_models_dir / "SMPLX_NEUTRAL.pkl"

        fidxs = sorted(frames)
        cat = lambda key, dim: torch.tensor(np.stack([
            np.asarray(frames[i].get(key, np.zeros(dim)), dtype=np.float32).reshape(-1)[:dim]
            for i in fidxs
        ]))
        # ext must match the file: smplx defaults to "npz" and np.load cannot read
        # the (python-2) SMPLX_*.pkl files in body_models/.
        model = smplx.create(
            str(model_path), model_type="smplx", gender=gender, ext=model_path.suffix[1:],
            num_betas=10, use_pca=False, batch_size=len(fidxs),
        )
        with torch.no_grad():
            out = model(
                betas=cat("betas", 10),
                global_orient=cat("global_orient", 3),
                body_pose=cat("body_pose", 63),
                transl=cat("transl", 3),
            )
        J = out.joints[:, :_N_BODY_JOINTS].numpy().astype(np.float64)
        joints[pid] = {fidx: J[i] for i, fidx in enumerate(fidxs)}
        logging.info(f"  GT subject {pid} ({gender}): {len(fidxs)} frames")
    return joints


# ---------------------------------------------------------------------------
# Projection into SAM3 pixel space
# ---------------------------------------------------------------------------

def calibration_size(raw_root: Path | None, scene_name: str, cam_id: str,
                     work_w: int, work_h: int) -> tuple[int, int]:
    """(W, H) at which the XML intrinsics are expressed."""
    if raw_root is not None:
        cam_dir = raw_root / scene_name / cam_id
        if cam_dir.is_dir():
            first = next((p for p in sorted(cam_dir.iterdir())
                          if p.suffix.lower() in _IMAGE_EXTS), None)
            if first is not None:
                with Image.open(first) as im:
                    return im.size
        logging.warning(f"  {cam_id}: no raw frame under {cam_dir} — assuming "
                        f"{_RICH_CALIB_W}x{_RICH_CALIB_H}")
    if work_h > work_w:
        return _RICH_CALIB_H, _RICH_CALIB_W
    return _RICH_CALIB_W, _RICH_CALIB_H


def scale_K(K: np.ndarray, calib_w: int, calib_h: int, work_w: int, work_h: int) -> np.ndarray:
    K = K.copy()
    K[0] *= work_w / calib_w      # fx, cx
    K[1] *= work_h / calib_h      # fy, cy
    return K


def project(X: np.ndarray, calib: dict, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (J, 3) world points. Returns (uv (J, 2), in_front (J,) bool)."""
    z = (X @ calib["R"].T + calib["t"])[:, 2]
    rvec, _ = cv2.Rodrigues(calib["R"])
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, calib["t"], K, calib["dist"])
    return uv.reshape(-1, 2), z > 0.1


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
    calib: dict,
    K: np.ndarray,
    gt_joints: dict[int, dict[int, np.ndarray]],
    min_joints: int,
) -> tuple[dict[str, dict[int, int]], dict]:
    """Per-frame assignment ``assign[json_stem][sam3_tid] = gt_pid`` for one camera.

    Every frame is decided on its own — nothing depends on SAM3 keeping a track id.
    Hungarian on the match score between the frame's SAM3 masks and the GT subjects
    visible in it; a pair is kept if the mask touches the projected skeleton at all
    (score > 0). Frames with no GT body get no assignment.
    """
    json_paths = sorted((cam_dir / "json_data").glob("mask_*.json"))
    zf = zipfile.ZipFile(str(cam_dir / "mask_data.npz"), "r")
    names = set(zf.namelist())

    assign: dict[str, dict[int, int]] = {}
    tids_seen: set[int] = set()
    no_gt_frames = 0
    scores_kept: list[float] = []
    for jp in json_paths:
        stem = jp.stem
        meta = json.loads(jp.read_text())
        labels = meta.get("labels", {})
        if not labels:
            continue
        tids_seen.update(int(k) for k in labels)
        fidx = frame_index(stem)
        W, H = meta["mask_width"], meta["mask_height"]

        gts, gt_uv, gt_box = [], [], []
        for pid, frames in gt_joints.items():
            if fidx not in frames:
                continue
            uv, front = project(frames[fidx], calib, K)
            ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if ok.sum() < min_joints:
                continue
            u = uv[ok]
            lo, hi = u.min(0), u.max(0)
            pad = 0.1 * (hi - lo)
            gts.append(pid)
            gt_uv.append(np.round(u).astype(int))
            gt_box.append((lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[1]))
        if not gts:
            no_gt_frames += 1
            continue

        npy = f"{stem}.npy"
        if npy not in names:
            continue
        with zf.open(npy) as f:
            mask = np.load(io.BytesIO(f.read()))

        tids = [int(k) for k in labels]
        score = np.zeros((len(tids), len(gts)))
        for a, tid in enumerate(tids):
            info = labels[str(tid)]
            tbox = (info["x1"], info["y1"], info["x2"], info["y2"])
            for b in range(len(gts)):
                u = gt_uv[b]
                inside = float(np.mean(mask[u[:, 1], u[:, 0]] == tid))
                score[a, b] = 0.5 * inside + 0.5 * _iou(tbox, gt_box[b])

        ri, ci = linear_sum_assignment(-score)
        pairs = [(a, b) for a, b in zip(ri, ci) if score[a, b] > 0]
        if pairs:
            assign[stem] = {tids[a]: gts[b] for a, b in pairs}
            scores_kept.extend(float(score[a, b]) for a, b in pairs)
    zf.close()

    kept_tids = {tid for m in assign.values() for tid in m}
    frames_per_gt = Counter(g for m in assign.values() for g in m.values())
    stats = {
        "frames_with_detections": len(json_paths),
        "frames_without_gt": no_gt_frames,
        "frames_matched": len(assign),
        "frames_per_gt": {str(g): n for g, n in sorted(frames_per_gt.items())},
        "median_score": float(np.median(scores_kept)) if scores_kept else None,
        "min_score": float(np.min(scores_kept)) if scores_kept else None,
        "sam3_tracks_used": sorted(kept_tids),
        "sam3_tracks_never_used": sorted(tids_seen - kept_tids),
    }
    return assign, stats


def apply_assignment(cam_dir: Path, assign: dict[str, dict[int, int]]) -> None:
    """Rewrite mask_data.npz and json_data/ with GT ids; erase everything unmatched.

    Each kept track's pixels become ``gt_pid + 1``; every other id becomes 0
    (background) and its json label is removed. Frames absent from ``assign``
    (no GT body / no match) are emptied entirely.
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
            for tid, g in assign.get(Path(name).stem, {}).items():
                new_mask[mask == tid] = g + 1
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
            new_id = tmap[tid] + 1
            info["instance_id"] = new_id
            info["gt_pid"] = tmap[tid]
            new_labels[str(new_id)] = info
        meta["labels"] = new_labels
        jp.write_text(json.dumps(meta))


def gt_reid_scene(
    scene,
    video_dirs: dict[str, str],
    gt_joints: dict[int, dict[int, np.ndarray]],
    gt_root: Path,
    raw_root: Path | None,
    min_joints: int,
) -> dict:
    """Relabel every camera of a scene from projected GT. Returns the gt_reid.json plan."""
    plan: dict = {
        "scene": scene.scene_id,
        "id_rule": "person_{gt_pid + 1}",
        "gt_subjects": sorted(gt_joints),
        "params": {"min_joints": min_joints, "rule": "per-frame Hungarian, keep score > 0"},
        "cameras": {},
    }
    for video in scene.videos:
        cam = video.video_id
        cam_dir = Path(video_dirs[cam])
        calib = load_calibration(gt_root, scene.scene_id, cam)
        if calib is None:
            logging.warning(f"  {cam}: no calibration XML — camera emptied")
            apply_assignment(cam_dir, {})
            plan["cameras"][cam] = {"error": "no calibration"}
            continue

        first_json = next(iter(sorted((cam_dir / "json_data").glob("mask_*.json"))), None)
        if first_json is None:
            logging.warning(f"  {cam}: no json_data — skipped")
            continue
        meta = json.loads(first_json.read_text())
        work_w, work_h = meta["mask_width"], meta["mask_height"]
        calib_w, calib_h = calibration_size(raw_root, scene.scene_id, cam, work_w, work_h)
        if abs(calib_w / calib_h - work_w / work_h) > 0.01:
            logging.warning(f"  {cam}: aspect mismatch calib {calib_w}x{calib_h} vs "
                            f"SAM3 {work_w}x{work_h}")
        K = scale_K(calib["K"], calib_w, calib_h, work_w, work_h)
        cx, cy = K[0, 2] / work_w, K[1, 2] / work_h
        if not (0.4 < cx < 0.6 and 0.4 < cy < 0.6):
            logging.warning(f"  {cam}: scaled PP at ({cx:.2f}, {cy:.2f}) of the frame — "
                            f"calibration/image mismatch?")

        assign, stats = match_camera(cam_dir, calib, K, gt_joints, min_joints)
        stats.update({"sam3_size": [work_w, work_h], "calib_size": [calib_w, calib_h]})
        plan["cameras"][cam] = stats
        logging.info(
            f"  {cam}: matched {stats['frames_matched']}/{stats['frames_with_detections']} "
            f"frames  per-GT {stats['frames_per_gt']}  median score "
            f"{stats['median_score'] if stats['median_score'] is None else round(stats['median_score'], 3)}  "
            f"SAM3 tracks never used {stats['sam3_tracks_never_used']}"
        )
        apply_assignment(cam_dir, assign)
    return plan


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def process_scene(scene, args, output_dir: Path) -> None:
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    scene_dir = output_dir / scene.scene_id
    gt_reid_path = scene_dir / "gt_reid.json"
    video_dirs = {v.video_id: scene_dir / v.video_id for v in scene.videos}

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
        )
        video_dirs = segmenter.segment_scene(scene=scene, output_dir=str(output_dir), vis=False)
        del segmenter
        torch.cuda.empty_cache()

    # Step 2: GT-driven ReID (must run before SAM3D).
    if gt_reid_path.exists():
        print(f"\n--- Step 2: GT ReID (skipped — gt_reid.json exists) ---")
    else:
        print(f"\n--- Step 2: GT ReID ---")
        gt_joints = load_gt_joints(args.gt_root, args.split, scene.scene_id,
                                   args.body_models_dir, args.gender_json)
        if not gt_joints:
            logging.error(f"  no GT bodies under {args.gt_root / f'{args.split}_body' / scene.scene_id} "
                          f"— refusing to run SAM3D on unfiltered tracks")
            return
        plan = gt_reid_scene(scene, video_dirs, gt_joints, args.gt_root, args.raw_root,
                             args.min_joints)
        gt_reid_path.write_text(json.dumps(plan, indent=2))

    # Step 3: SAM3D on the GT subjects only.
    print(f"\n--- Step 3: Body parameter estimation (GT subjects only) ---")
    estimator = ParametersExtractor(
        sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
        sam3d_step=CONFIG.parameters_extraction.sam3d_step,
        bbox_padding=CONFIG.parameters_extraction.bbox_padding,
        smplx_model_path=CONFIG.data.smplx_model_path,
        mhr_model_path=CONFIG.data.mhr_model_path,
        reid_threshold=_NO_MERGE_REID_THRESHOLD,
        reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
        # None = SAM3D uses its own default focal, as in every earlier RICH run (their
        # rich_data_root held only scene folders, never scan_calibration/). Pointing it
        # at the GT root would feed GT intrinsics into SAM3D and change its outputs.
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
            print(f"  WARNING: {cam}: no body data")
            continue
        has_smplx = all(any(k.startswith("smplx_") for k in np.load(p).files) for p in files)
        print(f"  {cam}: {[p.stem for p in files]}"
              + ("" if has_smplx else "  WARNING: missing smplx_* fields"))

    if args.centered_root is None:
        print(f"\n--- Steps 5-6: VGGT / MapAnything skipped (no --centered-root) ---")
        return
    centered_scene = args.centered_root / scene.scene_id
    if not centered_scene.is_dir():
        logging.error(f"  {centered_scene} missing — run utilities/center_images.py first")
        return

    # Step 5: VGGT on the PP-centered frames -> vggt_{cameras,depth}_centered.npz.
    cam_path = scene_dir / "vggt_cameras_centered.npz"
    dep_path = scene_dir / "vggt_depth_centered.npz"
    if _vggt_valid(cam_path, dep_path):
        print(f"\n--- Step 5: VGGT (centered) — already done, skipping ---")
    else:
        print(f"\n--- Step 5: VGGT (centered) ---")
        cam_path.unlink(missing_ok=True)
        dep_path.unlink(missing_ok=True)
        frame_paths, camera_names = _centered_frame_paths(centered_scene, scene)
        if not frame_paths:
            logging.error(f"  no centered frames under {centered_scene} — skipping VGGT")
            return
        print(f"  {len(frame_paths)} frames x {len(camera_names)} cameras  devices {args.vggt_devices}")
        preprocessor = VGGTPreprocessor(weights=args.vggt_weights, device=args.vggt_devices[0])
        preprocessor.process_scene(frame_paths=frame_paths, camera_names=camera_names,
                                   output_dir=scene_dir, devices=args.vggt_devices)
        del preprocessor
        torch.cuda.empty_cache()

    # Step 6: MapAnything metric scale (reads the centered VGGT outputs + centered frames).
    scale_path = scene_dir / "mapanything_scale_baseline.npy"
    if args.skip_mapanything:
        print(f"\n--- Step 6: MapAnything skipped (--skip-mapanything) ---")
    elif _npy_valid(scale_path):
        print(f"\n--- Step 6: MapAnything — already done, skipping ---")
    else:
        print(f"\n--- Step 6: MapAnything scale ---")
        scale_path.unlink(missing_ok=True)
        ma = MapAnythingScaleEstimator(device=args.vggt_devices[0],
                                       batch_size=args.mapanything_batch_size)
        ma.process_scene(scene_dir=scene_dir, img_root=centered_scene)
        del ma
        torch.cuda.empty_cache()


def _centered_frame_paths(centered_scene: Path, scene) -> tuple[list[list[Path | None]], list[str]]:
    """frame_paths[t][k] from <centered_root>/<scene>/<cam>/, cameras sorted by id."""
    camera_names = sorted(v.video_id for v in scene.videos)
    per_cam = []
    for cam in camera_names:
        d = centered_scene / cam
        per_cam.append(sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
                       if d.is_dir() else [])
    T = min((len(f) for f in per_cam), default=0)
    if T and max(len(f) for f in per_cam) != T:
        logging.warning(f"  centered frame counts differ per camera "
                        f"{[len(f) for f in per_cam]} — using first {T}")
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True,
                   help="Frames SAM3 runs on: <root>/<scene>/cam_XX/*.jpg (e.g. test_840).")
    p.add_argument("--gt-root", type=Path, required=True,
                   help="RICH root with <split>_body/ and scan_calibration/.")
    p.add_argument("--raw-root", type=Path, default=None,
                   help="Full-resolution frames, used to read the calibration resolution.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--scene", default=None,
                   help="Process only scenes whose ID contains this string.")
    p.add_argument("--scene-start", type=int, default=None)
    p.add_argument("--scene-end", type=int, default=None)
    p.add_argument("--body-models-dir", type=Path, default=_REPO_ROOT / "body_models")
    p.add_argument("--gender-json", type=Path, default=_REPO_ROOT / "resource" / "rich_gender.json")
    p.add_argument("--min-joints", type=int, default=4,
                   help="Minimum in-image projected joints for a GT subject to count as visible.")
    p.add_argument("--centered-root", type=Path, default=None,
                   help="PP-centered frames from utilities/center_images.py "
                        "(<root>/<scene>/cam_XX/*.jpg). Enables Steps 5-6; omit to stop after SAM3D.")
    p.add_argument("--vggt-weights", default=CONFIG.data.vggt_omega_checkpoint)
    p.add_argument("--vggt-devices", nargs="+", default=None,
                   help="CUDA devices for VGGT/MapAnything; defaults to all visible GPUs.")
    p.add_argument("--skip-mapanything", action="store_true")
    p.add_argument("--mapanything-batch-size", type=int, default=8)
    args = p.parse_args()
    if args.vggt_devices is None:
        args.vggt_devices = ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                             if torch.cuda.is_available() else ["cpu"])

    ds = RichDataset(data_root=str(args.data_root), max_side=None, frames_base_dir=None)
    if args.scene_start is not None or args.scene_end is not None:
        ds.scenes = ds.scenes[args.scene_start:args.scene_end]
    if args.scene is not None:
        ds.scenes = [s for s in ds.scenes if args.scene in s.scene_id]
    print(f"{len(ds.scenes)} scene(s): {[s.scene_id for s in ds.scenes]}")

    import gc
    failed: list[str] = []
    for scene in ds.scenes:
        try:
            process_scene(scene, args, args.output_dir)
        except Exception as e:
            logging.error(f"Scene {scene.scene_id} failed: {e}", exc_info=True)
            failed.append(scene.scene_id)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    for scene in ds.scenes:
        sdir = args.output_dir / scene.scene_id
        reid = (sdir / "gt_reid.json").exists()
        body = [v.video_id for v in scene.videos
                if not (sdir / v.video_id / "body_data" / ".done").exists()]
        status = "FAILED" if scene.scene_id in failed else "ok"
        vggt = _vggt_valid(sdir / "vggt_cameras_centered.npz", sdir / "vggt_depth_centered.npz")
        scale = _npy_valid(sdir / "mapanything_scale_baseline.npy")
        print(f"  {scene.scene_id}: {status}  gt_reid={'yes' if reid else 'NO'}  "
              f"body missing={body or '-'}  vggt={'yes' if vggt else 'NO'}  "
              f"scale={'yes' if scale else 'NO'}")


if __name__ == "__main__":
    main()
