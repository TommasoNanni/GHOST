"""
EgoHumans PRODUCTION pipeline — segmentation + body estimation + **GT-driven ReID**
+ VGGT + MapAnything.

Same stages as ``egohumans_pipeline.py`` with one difference: Step 3 does not run
``CrossViewReidentifierV4``. Instead every SAM3 track is matched to the dataset's own
per-camera 2D ground truth and inherits that subject's identity, so the person ids are
globally consistent across cameras *by construction* and every non-GT person (referee,
bystander, second court) is dropped.

Why GT 2D and not GT 3D reprojection
------------------------------------
EgoHumans ships ``processed_data/poses2d/<cam>/rgb/<frame>.npy`` — an object array of
per-person dicts::

    {'bbox': [x1,y1,x2,y2,score], 'human_name': 'aria01', 'human_id': 0,
     'keypoints': (17,3) COCO [x,y,conf], 'is_valid': bool, ...}

already in the raw fisheye pixel space of that camera. Matching there is 2D-to-2D in
the same image, which avoids the ``colmap_from_aria["aria01"]`` anchor whose quality
varies per scene (basketball 001 lands ~25 px off at 1440 px wide, tagging 007 is
exact). A 3D-reprojection matcher would inherit that error; this one cannot.

Identity naming
---------------
The GT calls subjects ``aria01``..``aria04``. Person files stay numeric because every
consumer parses them with ``int(stem.split("_")[1])`` (``data/fusion_dataset.py``,
``evaluation/evaluate_egoexo.py``, ...), so the mapping is fixed and global::

    aria01 -> person_1    aria02 -> person_2    aria03 -> person_3    aria04 -> person_4

The scene-level ``gt_reid.json`` records the mapping, the per-camera vote margins and
every dropped track, so the relabelling is auditable and reversible.

Coordinate handling
-------------------
GT keypoints are in RAW fisheye pixels (3840x2160). The pipeline runs on balance=0
undistorted frames resized to ``--max-side`` (1440). GT points are carried into that
space with ``cv2.fisheye.undistortPoints(..., P=K_new_r)`` using the *same* ``K_new_r``
that ``scripts/prep_undistort_egohumans.py:build_map`` computes, so the two spaces
agree exactly.

Layout
------
``--data-root`` is the undistorted/resized tree written by prep_undistort (images
only). ``--gt-root`` is the raw tree (squashfuse mount or download) that carries
``processed_data/`` and ``colmap/``. They are different trees; both are required.

    pixi run python scripts/egohumans_pipeline_production.py \
        --data-root /iopsstor/scratch/cscs/tnanni/temp_egohumans \
        --gt-root   /iopsstor/scratch/cscs/tnanni/egohumans_raw \
        --output-dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_prod \
        --activity 03_fencing --seq 001_fencing
"""

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

import gc
import io
import json
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from configuration import CONFIG
from data.video_dataset import Video, EgoHumansScene
from preprocessing.run_vggt import VGGTPreprocessor
from preprocessing.run_mapanything import MapAnythingScaleEstimator
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction_v2 import ParametersExtractor
from utilities.visualize_segmented_reids import visualize_reid

VGGT_WEIGHTS = CONFIG.data.vggt_omega_checkpoint

# Deep inner path baked into every EgoHumans activity folder.
_INNER = Path("media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready")

# Exo cameras to use per activity (others were deleted during preprocessing).
KEEP_CAMS: dict[str, list[str]] = {
    "01_tagging":    ["cam01", "cam04", "cam06", "cam08"],
    "02_lego":       ["cam02", "cam03", "cam04", "cam06"],
    "03_fencing":    ["cam04", "cam05", "cam10", "cam13"],
    "04_basketball": ["cam01", "cam03", "cam04", "cam08"],
    "05_volleyball": ["cam02", "cam04", "cam08", "cam11"],
    "06_badminton":  ["cam01", "cam02", "cam05", "cam07"],
    "07_tennis":     ["cam04", "cam09", "cam12", "cam20"],
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# GT subject name -> person id written to disk. Global across every camera and scene.
ARIA_TO_PID = {f"aria{i:02d}": i for i in range(1, 9)}


# ---------------------------------------------------------------------------
# Scene construction (copied from egohumans_pipeline.py — this file is standalone)
# ---------------------------------------------------------------------------

def _build_scenes(
    data_root: Path,
    activities: list[str] | None = None,
    seq_filter: str | None = None,
) -> list[EgoHumansScene]:
    """Walk data_root and build EgoHumansScene objects for all sequences."""
    all_activities = sorted(KEEP_CAMS.keys())
    if activities:
        all_activities = [a for a in all_activities if a in activities]

    scenes: list[EgoHumansScene] = []
    for activity in all_activities:
        cam_ready = data_root / activity / _INNER / activity
        if not cam_ready.is_dir():
            logging.warning(f"Activity dir not found: {cam_ready}")
            continue
        keep = KEEP_CAMS[activity]
        for seq_dir in sorted(cam_ready.iterdir()):
            if not seq_dir.is_dir():
                continue
            if seq_filter and seq_filter not in seq_dir.name:
                continue
            videos: list[Video] = []
            for cam_name in keep:
                undist_dir = seq_dir / "exo" / cam_name / "images_undistorted"
                if not undist_dir.is_dir():
                    logging.warning(f"Missing {undist_dir.relative_to(data_root)}")
                    continue
                _has_images = any(p.suffix.lower() in _IMAGE_EXTS for p in undist_dir.iterdir())
                if not _has_images:
                    # After resizing, originals are deleted and only frames/ subdir remains.
                    _frames_sub = undist_dir / "frames"
                    _has_images = _frames_sub.is_dir() and any(
                        p.suffix.lower() in _IMAGE_EXTS for p in _frames_sub.iterdir())
                if not _has_images:
                    logging.warning(f"Empty {undist_dir.relative_to(data_root)}")
                    continue
                _frames_sub = undist_dir / "frames"
                _effective_dir = _frames_sub if (
                    _frames_sub.is_dir()
                    and any(p.suffix.lower() in _IMAGE_EXTS for p in _frames_sub.iterdir())
                ) else undist_dir
                video = Video(
                    frames_dir=_effective_dir,
                    max_side=getattr(CONFIG.data, "egohumans_max_side", None)
                    if _effective_dir == undist_dir else None,
                )
                video.video_id = cam_name
                videos.append(video)
            if not videos:
                logging.warning(f"No valid cameras for {activity}/{seq_dir.name}, skipping.")
                continue
            scenes.append(EgoHumansScene(
                scene_id=f"{activity}/{seq_dir.name}",
                videos=videos,
                seq_dir=seq_dir,
            ))
    return scenes


def _build_vggt_frame_paths(
    scene: EgoHumansScene,
    video_dirs: dict[str, str],
) -> tuple[list[list[Path | None]], list[str]]:
    sorted_videos = sorted(scene.videos, key=lambda v: v.video_id)
    camera_names = [v.video_id for v in sorted_videos]

    per_cam_frames: list[list[Path]] = []
    for video in sorted_videos:
        fdir = video.frames_home
        if fdir is None or not fdir.is_dir():
            logging.warning(f"VGGT: no frames dir for {video.video_id}")
            per_cam_frames.append([])
            continue
        per_cam_frames.append(sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS))

    T = min((len(f) for f in per_cam_frames), default=0)
    if T == 0:
        logging.warning("VGGT: no frames found — skipping.")
        return [], camera_names

    frame_paths = [
        [cam_frames[t] if cam_frames else None for cam_frames in per_cam_frames]
        for t in range(T)
    ]
    return frame_paths, camera_names


# ---------------------------------------------------------------------------
# GT-driven ReID
# ---------------------------------------------------------------------------

def parse_colmap_calibration(seq_dir: Path) -> dict[str, dict]:
    """{cam_name: {K, D, W, H}} from COLMAP OPENCV_FISHEYE cameras.txt + images.txt.

    Copied from scripts/prep_undistort_egohumans.py so the two agree bit-for-bit on
    the intrinsics used to build the undistort map.
    """
    cameras_txt = seq_dir / "colmap" / "workplace" / "cameras.txt"
    images_txt = seq_dir / "colmap" / "workplace" / "images.txt"
    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError(f"{seq_dir}: colmap/workplace/{{cameras,images}}.txt missing")

    cam_params: dict[int, dict] = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if parts[1] != "OPENCV_FISHEYE":
                raise ValueError(f"{seq_dir.name}: cam {parts[0]} model {parts[1]} != OPENCV_FISHEYE")
            cid = int(parts[0])
            W, H = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = (float(x) for x in parts[4:8])
            k1, k2, k3, k4 = (float(x) for x in parts[8:12])
            cam_params[cid] = {
                "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
                "D": np.array([[k1], [k2], [k3], [k4]], dtype=np.float64),
                "W": W, "H": H,
            }

    name_to_cid: dict[str, int] = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            cam = parts[9].split("/")[0]
            if cam.startswith("cam") and cam not in name_to_cid:
                name_to_cid[cam] = int(parts[8])

    return {n: cam_params[c] for n, c in name_to_cid.items() if c in cam_params}


def undistorted_intrinsics(calib: dict, max_side: int) -> tuple[np.ndarray, int, int]:
    """balance=0 K_new scaled by the resize factor — mirrors prep_undistort's build_map."""
    K, D, W, H = calib["K"], calib["D"], calib["W"], calib["H"]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=0.0
    )
    s = max_side / max(W, H) if max_side else 1.0
    W_r, H_r = int(round(W * s)), int(round(H * s))
    K_new_r = np.diag([s, s, 1.0]) @ K_new
    return K_new_r, W_r, H_r


def gt_points_to_undistorted(pts: np.ndarray, calib: dict, K_new_r: np.ndarray) -> np.ndarray:
    """Map raw-fisheye pixel points into the balance=0 undistorted+resized image space.

    Exact inverse of the remap prep_undistort applies to the pixels, so a GT keypoint
    lands on the same feature in the undistorted frame that it marked in the raw one.
    """
    # Contiguous copy is required: callers pass strided views (kps[..., :2]) and
    # np.asarray does not copy those when the dtype already matches. OpenCV then
    # misreads the buffer — only the first point is mapped right, the rest is garbage,
    # which scattered every GT centroid and broke the matching.
    p = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.fisheye.undistortPoints(p, calib["K"], calib["D"], R=np.eye(3), P=K_new_r)
    return out.reshape(-1, 2)


def load_gt_people(poses2d_dir: Path, stem: int) -> list[dict]:
    """Valid person dicts for one frame of one camera, or [] when the file is absent."""
    npy_path = poses2d_dir / f"{stem:05d}.npy"
    if not npy_path.exists():
        # Some activities write unpadded stems.
        npy_path = poses2d_dir / f"{stem}.npy"
    if not npy_path.exists():
        return []
    try:
        arr = np.load(str(npy_path), allow_pickle=True)
    except Exception as exc:
        logging.warning(f"GT poses2d unreadable {npy_path}: {exc}")
        return []
    out = []
    for person in arr:
        if not isinstance(person, dict):
            continue
        if not person.get("is_valid", True):
            continue
        if "keypoints" not in person or "human_name" not in person:
            continue
        out.append(person)
    return out


def track_centroids(cam_dir: Path) -> dict[int, dict]:
    """{pid: {'fi': frame_indices, 'c': centroid_xy, 's': track scale}} from body_data."""
    out: dict[int, dict] = {}
    for p in sorted((cam_dir / "body_data").glob("person_*.npz")):
        try:
            pid = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        d = np.load(p, allow_pickle=False)
        if "pred_keypoints_2d" not in d.files or "frame_indices" not in d.files:
            continue
        kp = d["pred_keypoints_2d"]              # (T, J, 2+)
        fi = d["frame_indices"].astype(int)      # (T,)
        cent = np.full((len(fi), 2), np.nan)
        size = np.full(len(fi), np.nan)
        for i in range(len(fi)):
            k = np.asarray(kp[i], np.float64)[..., :2]
            k = k[np.isfinite(k).all(-1)]
            if not k.size:
                continue
            cent[i] = k.mean(0)
            size[i] = float(np.hypot(*(k.max(0) - k.min(0))))
        out[pid] = {"fi": fi, "c": cent, "s": size}
    return out


def match_camera_to_gt(
    cam_dir: Path,
    frame_stems: list[int],
    poses2d_dir: Path,
    calib: dict,
    K_new_r: np.ndarray,
    n_frames: int,
    max_dist: float,
) -> tuple[dict[int, str], dict[str, float], dict[str, int]]:
    """Assign each SAM3 track in one camera to a GT subject.

    Hungarian assignment per frame between track centroids and GT keypoint centroids
    (distance normalised by track extent so it is scale-free), then a majority vote
    over frames. Returns (old_pid -> aria_name, per-name median distance, vote counts).
    """
    tracks = track_centroids(cam_dir)
    if not tracks:
        return {}, {}, {}

    stems = frame_stems
    if n_frames and len(stems) > n_frames:
        step = max(1, len(stems) // n_frames)
        stems = stems[::step][:n_frames]

    votes: dict[int, Counter] = defaultdict(Counter)
    dists: dict[tuple[int, str], list[float]] = defaultdict(list)

    pids = sorted(tracks)
    for stem in stems:
        # body_data frame_indices are the IMAGE STEMS: segmentation names each mask
        # `mask_<frame_name_without_ext>.json` (segmentation.py:401) and the body
        # estimator parses that digit back out. So GT stem and track index compare
        # directly — no position lookup.
        people = load_gt_people(poses2d_dir, stem)
        if not people:
            continue

        gt_names, gt_cent = [], []
        for person in people:
            kps = np.asarray(person["keypoints"], np.float64)
            vis = kps[kps[:, 2] > 0.3][:, :2] if kps.shape[1] > 2 else kps[:, :2]
            if not vis.size:
                continue
            gt_names.append(str(person["human_name"]))
            gt_cent.append(gt_points_to_undistorted(vis, calib, K_new_r).mean(0))
        if not gt_names:
            continue

        rows, tr_cent, tr_size = [], [], []
        for pid in pids:
            tr = tracks[pid]
            i = np.where(tr["fi"] == stem)[0]
            if not i.size:
                continue
            c, s = tr["c"][int(i[0])], tr["s"][int(i[0])]
            if not np.isfinite(c).all() or not np.isfinite(s) or s <= 0:
                continue
            rows.append(pid)
            tr_cent.append(c)
            tr_size.append(s)
        if not rows:
            continue

        cost = np.zeros((len(rows), len(gt_names)))
        for a, (c, s) in enumerate(zip(tr_cent, tr_size)):
            for b, g in enumerate(gt_cent):
                cost[a, b] = float(np.hypot(*(np.asarray(c) - g)) / s)

        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] > max_dist:
                continue
            votes[rows[a]][gt_names[b]] += 1
            dists[(rows[a], gt_names[b])].append(float(cost[a, b]))

    # One GT subject may only win one track per camera: resolve by vote count.
    claims: list[tuple[int, int, str]] = []
    for pid, counter in votes.items():
        name, n = counter.most_common(1)[0]
        claims.append((n, pid, name))
    claims.sort(reverse=True)

    assign: dict[int, str] = {}
    taken: set[str] = set()
    counts: dict[str, int] = {}
    quality: dict[str, float] = {}
    for n, pid, name in claims:
        if name in taken:
            continue
        assign[pid] = name
        taken.add(name)
        counts[name] = n
        quality[name] = float(np.median(dists[(pid, name)])) if dists[(pid, name)] else float("nan")
    return assign, quality, counts


def apply_gt_remap(video_dir: Path, id_remap: dict[int, int], drop: set[int]) -> None:
    """Renumber person ids in body_data, mask_data.npz and json_data; delete dropped tracks.

    ``id_remap`` is old_pid -> new_pid for the GT subjects; ``drop`` are tracks that
    matched no GT subject. Dropped ids become 0 (background) in the masks, their labels
    are removed from the json, and their npz files are deleted.
    """
    body_dir = video_dir / "body_data"

    # --- body_data: move aside first so a permutation cannot clobber itself ---
    staged: dict[int, Path] = {}
    for old in list(id_remap) + sorted(drop):
        src = body_dir / f"person_{old}.npz"
        if src.exists():
            tmp = body_dir / f".tmp_person_{old}.npz"
            src.rename(tmp)
            staged[old] = tmp
    for old, tmp in staged.items():
        if old in drop:
            tmp.unlink()
        else:
            tmp.rename(body_dir / f"person_{id_remap[old]}.npz")

    # --- mask_data.npz ---
    npz_path = video_dir / "mask_data.npz"
    if npz_path.exists():
        tmp_path = npz_path.with_suffix(".tmp.npz")
        with (
            zipfile.ZipFile(str(npz_path), "r") as zf_in,
            zipfile.ZipFile(str(tmp_path), "w",
                            compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf_out,
        ):
            for name in sorted(zf_in.namelist()):
                with zf_in.open(name) as f:
                    mask_img = np.load(io.BytesIO(f.read()))
                new_mask = np.zeros_like(mask_img)
                for old_id, new_id in id_remap.items():
                    new_mask[mask_img == old_id] = new_id
                # ids in `drop` and any id we never saw both fall through as 0.
                buf = io.BytesIO()
                np.save(buf, new_mask)
                zf_out.writestr(name, buf.getvalue())
        tmp_path.replace(npz_path)

    # --- json_data ---
    json_dir = video_dir / "json_data"
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        if "labels" in data:
            new_labels = {}
            for str_id, info in data["labels"].items():
                old_id = int(str_id)
                if old_id not in id_remap:
                    continue          # dropped: not a GT subject
                new_id = id_remap[old_id]
                info["instance_id"] = new_id
                new_labels[str(new_id)] = info
            data["labels"] = new_labels
        with open(json_path, "w") as f:
            json.dump(data, f)


def gt_reid_scene(
    scene: EgoHumansScene,
    video_dirs: dict[str, str],
    gt_seq_dir: Path,
    max_side: int,
    n_frames: int,
    max_dist: float,
    dry_run: bool = False,
) -> dict:
    """Relabel every camera of a scene from the dataset's own 2D GT.

    Returns the plan dict that is written to ``gt_reid.json``.
    """
    calibs = parse_colmap_calibration(gt_seq_dir)
    plan: dict = {"scene": scene.scene_id, "cameras": {}, "names": {}, "dropped": {}}

    for video in scene.videos:
        cam = video.video_id
        cam_dir = Path(video_dirs[cam])
        poses2d_dir = gt_seq_dir / "processed_data" / "poses2d" / cam / "rgb"
        if not poses2d_dir.is_dir():
            logging.warning(f"  {cam}: no GT poses2d ({poses2d_dir}) — left untouched")
            continue
        if cam not in calibs:
            logging.warning(f"  {cam}: absent from colmap cameras.txt — left untouched")
            continue

        frames_dir = video.frames_home
        frame_stems = [int(p.stem) for p in
                       sorted(frames_dir.iterdir()) if p.suffix.lower() in _IMAGE_EXTS]
        if not frame_stems:
            logging.warning(f"  {cam}: no frames — left untouched")
            continue

        K_new_r, _, _ = undistorted_intrinsics(calibs[cam], max_side)
        assign, quality, counts = match_camera_to_gt(
            cam_dir, frame_stems, poses2d_dir, calibs[cam], K_new_r, n_frames, max_dist,
        )
        all_pids = {int(p.stem.split("_")[1])
                    for p in (cam_dir / "body_data").glob("person_*.npz")}
        if not assign:
            logging.warning(f"  {cam}: no track matched GT — left untouched")
            continue

        id_remap = {pid: ARIA_TO_PID[name] for pid, name in assign.items()
                    if name in ARIA_TO_PID}
        dropped = sorted(all_pids - set(id_remap))
        plan["cameras"][cam] = {str(k): v for k, v in id_remap.items()}
        plan["names"][cam] = {name: {"pid": ARIA_TO_PID[name],
                                     "votes": counts.get(name, 0),
                                     "median_dist": quality.get(name)}
                              for name in assign.values() if name in ARIA_TO_PID}
        plan["dropped"][cam] = dropped

        kept = ", ".join(f"{n}->person_{ARIA_TO_PID[n]}" for n in sorted(assign.values())
                         if n in ARIA_TO_PID)
        logging.info(f"  {cam}: {kept}   dropped {len(dropped)} non-GT track(s)")
        if not dry_run:
            apply_gt_remap(cam_dir, id_remap, set(dropped))

    return plan


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def process_scene(
    scene: EgoHumansScene,
    segmenter: PersonSegmenter,
    estimator: ParametersExtractor,
    output_dir: str | Path,
    gt_root: Path,
    vggt_weights: str,
    vggt_devices: list[str],
    ma_estimator: MapAnythingScaleEstimator | None = None,
    skip_cams: set[str] | None = None,
    max_side: int = 1440,
    gt_frames: int = 60,
    gt_max_dist: float = 1.0,
):
    """Run the production pipeline on a single EgoHumans scene."""
    skip_cams = skip_cams or set()
    if skip_cams:
        removed = [v for v in scene.videos if v.video_id in skip_cams]
        if removed:
            print(f"  Skipping cameras: {[v.video_id for v in removed]}")
            scene.videos = [v for v in scene.videos if v.video_id not in skip_cams]

    print(f"\n=== Scene: {scene.scene_id} ({len(scene.videos)} cameras) ===")
    for v in scene.videos:
        print(f"  {v}")

    activity, seq_name = scene.scene_id.split("/", 1)
    gt_seq_dir = gt_root / activity / _INNER / activity / seq_name
    if not gt_seq_dir.is_dir():
        # The GT tree may be rooted directly at camera_ready/<activity>/.
        alt = gt_root / _INNER / activity / seq_name
        gt_seq_dir = alt if alt.is_dir() else gt_seq_dir

    _seg_output_dir = Path(output_dir) / scene.scene_id
    _reid_already_done = (_seg_output_dir / "gt_reid.json").exists()
    _body_already_done = all(
        any((_seg_output_dir / v.video_id / "body_data").glob("person_*.npz"))
        for v in scene.videos
    )

    # Step 1: Segmentation.
    if _body_already_done:
        print(f"\n--- Step 1: Segmentation (skipped — body already done) ---")
        video_dirs = {v.video_id: str(_seg_output_dir / v.video_id) for v in scene.videos}
    else:
        print(f"\n--- Step 1: Segmentation ---")
        video_dirs = segmenter.segment_scene(scene=scene, output_dir=output_dir, vis=False)
        segmenter._predictor = None
        segmenter._models_ready = False
        torch.cuda.empty_cache()

    # After segmentation, originals may have been deleted and replaced by frames/ subdir.
    for video in scene.videos:
        if video.frames_dir is not None:
            frames_sub = video.frames_dir / "frames"
            if frames_sub.is_dir() and any(p.suffix.lower() in _IMAGE_EXTS for p in frames_sub.iterdir()):
                video.frames_dir = frames_sub

    scene_output_dir = Path(next(iter(video_dirs.values()))).parent

    # Step 2: Body parameter estimation.
    if _body_already_done:
        print(f"\n--- Step 2: Body parameter estimation (skipped — already done) ---")
    else:
        print(f"\n--- Step 2: Body parameter estimation ---")
        estimator.estimate_scene(scene=scene, video_dirs=video_dirs)

    missing_body = [
        vid_id for vid_id, vid_dir in video_dirs.items()
        if not any((Path(vid_dir) / "body_data").glob("person_*.npz"))
    ]
    if missing_body:
        print(f"  WARNING: body estimation incomplete for {missing_body} — skipping scene.")
        return

    # Step 3: GT-driven ReID (replaces cross-view ReID).
    print(f"\n--- Step 3: GT-driven ReID ---")
    if _reid_already_done:
        print(f"  Already done, skipping.")
    elif not gt_seq_dir.is_dir():
        print(f"  WARNING: GT scene dir not found ({gt_seq_dir}) — ids left as segmented.")
    else:
        plan = gt_reid_scene(
            scene, video_dirs, gt_seq_dir,
            max_side=max_side, n_frames=gt_frames, max_dist=gt_max_dist,
        )
        (scene_output_dir / "gt_reid.json").write_text(json.dumps(plan, indent=2))
        _reid_already_done = False

    # Step 4: Verify MHR → SMPLX conversion.
    print(f"\n--- Step 4: Verifying MHR → SMPLX conversion ---")
    smplx_fields_found = {}
    for video_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            data = dict(np.load(str(npz_path), allow_pickle=False))
            smplx_keys = [k for k in data if k.startswith("smplx_")]
            if smplx_keys:
                smplx_fields_found.setdefault(video_id, {})[npz_path.name] = smplx_keys
    if not smplx_fields_found:
        print(
            "  WARNING: No smplx_* fields found. "
            "Check that smplx_model_path and mhr_model_path are set in CONFIG."
        )

    # Step 5: VGGT camera + depth estimation.
    vggt_cameras_path = scene_output_dir / "vggt_cameras_centered.npz"
    vggt_depth_path = scene_output_dir / "vggt_depth_centered.npz"

    def _depth_valid(path: Path) -> bool:
        try:
            return bool(np.load(path, mmap_mode="r")["depth_valid"].any())
        except Exception:
            return False

    _vggt_done = (
        vggt_cameras_path.exists()
        and vggt_depth_path.exists()
        and _depth_valid(vggt_depth_path)
    )

    if _vggt_done:
        print(f"\n--- Step 5: VGGT (already done, skipping) ---")
    else:
        if vggt_cameras_path.exists() and vggt_depth_path.exists():
            print(f"\n--- Step 5: VGGT depth stale/invalid — recomputing ---")
            vggt_cameras_path.unlink()
            vggt_depth_path.unlink()
        else:
            print(f"\n--- Step 5: VGGT camera + depth estimation ---")
        frame_paths, camera_names = _build_vggt_frame_paths(scene, video_dirs)
        if frame_paths:
            print(f"  {len(frame_paths)} frames × {len(camera_names)} cameras")
            print(f"  Devices: {vggt_devices}")
            preprocessor = VGGTPreprocessor(weights=vggt_weights, device=vggt_devices[0])
            preprocessor.process_scene(
                frame_paths=frame_paths,
                camera_names=camera_names,
                output_dir=scene_output_dir,
                devices=vggt_devices,
            )
            del preprocessor
            torch.cuda.empty_cache()
        else:
            print("  WARNING: no frames — skipping VGGT.")

    # Step 6: MapAnything metric scale estimation.
    print(f"\n--- Step 6: MapAnything scale estimation ---")
    if ma_estimator is None:
        print("  Skipped (--skip-mapanything).")
    elif not (vggt_cameras_path.exists() and vggt_depth_path.exists()):
        print("  Skipped — VGGT outputs missing.")
    else:
        ma_estimator.process_scene(
            scene_dir=scene_output_dir,
            img_root=scene.seq_dir / "exo",
        )

    # Step 7: Visualise re-ID (only if ReID ran this session).
    if not _reid_already_done:
        print(f"\n--- Step 7: Visualising re-ID corrected segmentation ---")
        for video in scene.videos:
            if video.video_id not in video_dirs:
                continue
            try:
                visualize_reid(
                    video_dir=Path(video_dirs[video.video_id]),
                    fps=int(video.fps),
                    frames_dir=video.frames_home,
                )
            except FileNotFoundError as e:
                print(f"  WARNING: skipping visualisation — {e}")
    else:
        print(f"\n--- Step 7: Skipping re-ID visualisation (already done) ---")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, required=True,
                        help="Undistorted/resized EgoHumans root (prep_undistort output)")
    parser.add_argument("--gt-root", type=str, required=True,
                        help="Raw EgoHumans root carrying processed_data/ and colmap/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Pipeline output directory")
    parser.add_argument("--activity", type=str, nargs="+", default=None,
                        help="Restrict to these activities (e.g. 01_tagging 02_lego)")
    parser.add_argument("--seq", type=str, default=None,
                        help="Only process sequences whose name contains this string")
    parser.add_argument("--scene-start", type=int, default=None)
    parser.add_argument("--scene-end", type=int, default=None)
    parser.add_argument("--vggt-weights", type=str, default=VGGT_WEIGHTS)
    parser.add_argument("--vggt-devices", type=str, nargs="+", default=None)
    parser.add_argument("--skip-mapanything", action="store_true", default=False)
    parser.add_argument("--mapanything-device", type=str, default=None,
                        help="CUDA device for MapAnything (defaults to first VGGT device).")
    parser.add_argument("--mapanything-batch-size", type=int, default=8)
    parser.add_argument("--skip-cameras", type=str, nargs="+", default=[],
                        metavar="SCENE_ID:CAM_ID")
    parser.add_argument("--max-side", type=int,
                        default=getattr(CONFIG.data, "egohumans_max_side", 1440),
                        help="Must equal the --max-side used by prep_undistort_egohumans.py")
    parser.add_argument("--gt-frames", type=int, default=60,
                        help="Frames sampled per camera for the GT vote (0 = all)")
    parser.add_argument("--gt-max-dist", type=float, default=1.0,
                        help="Reject a track/GT pair beyond this many track-extents")
    args = parser.parse_args()

    skip_cams_map: dict[str, set[str]] = {}
    for entry in args.skip_cameras:
        if ":" not in entry:
            parser.error(f"--skip-cameras: expected 'scene_id:cam_id', got {entry!r}")
        scene_id, cam_id = entry.split(":", 1)
        skip_cams_map.setdefault(scene_id, set()).add(cam_id)

    if args.vggt_devices:
        vggt_devices = args.vggt_devices
    elif torch.cuda.is_available():
        vggt_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        vggt_devices = ["cpu"]
    print(f"VGGT devices: {vggt_devices}")

    if args.skip_mapanything:
        ma_estimator = None
        print("MapAnything: skipped (--skip-mapanything).")
    else:
        ma_device = args.mapanything_device or vggt_devices[0]
        ma_estimator = MapAnythingScaleEstimator(
            device=ma_device,
            batch_size=args.mapanything_batch_size,
        )
        print(f"MapAnything estimator: device={ma_device}  batch_size={args.mapanything_batch_size}")

    data_root = Path(args.data_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes(data_root, activities=args.activity, seq_filter=args.seq)
    if args.scene_start is not None or args.scene_end is not None:
        scenes = scenes[args.scene_start:args.scene_end]
    print(f"Processing {len(scenes)} scenes")

    failed: dict[str, list[str]] = {}
    needs_reid: list[str] = []

    for scene in scenes:
        scene_dir = output_dir / scene.scene_id
        if (
            (scene_dir / "gt_reid.json").exists()
            and (scene_dir / "vggt_cameras_centered.npz").exists()
            and (ma_estimator is None or (scene_dir / "mapanything_scale_baseline.npy").exists())
        ):
            print(f"Scene {scene.scene_id}: already done, skipping.")
            continue

        segmenter = PersonSegmenter(
            checkpoint_path=CONFIG.segmentation.checkpoint_path,
            text_prompt=CONFIG.segmentation.text_prompt,
            redetect_interval=CONFIG.segmentation.redetect_interval,
            new_det_thresh=CONFIG.segmentation.new_det_thresh,
            score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
        )
        estimator = ParametersExtractor(
            sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
            sam3d_step=CONFIG.parameters_extraction.sam3d_step,
            bbox_padding=CONFIG.parameters_extraction.bbox_padding,
            smplx_model_path=CONFIG.data.smplx_model_path,
            mhr_model_path=CONFIG.data.mhr_model_path,
            reid_threshold=CONFIG.parameters_extraction.reid_threshold,
            reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
            rich_data_root=None,
        )
        try:
            process_scene(
                scene, segmenter, estimator,
                output_dir=output_dir,
                gt_root=gt_root,
                vggt_weights=args.vggt_weights,
                vggt_devices=vggt_devices,
                ma_estimator=ma_estimator,
                skip_cams=skip_cams_map.get(scene.scene_id, set()),
                max_side=args.max_side,
                gt_frames=args.gt_frames,
                gt_max_dist=args.gt_max_dist,
            )
        except Exception as e:
            logging.error(f"Scene {scene.scene_id} failed: {e}", exc_info=True)
        finally:
            del segmenter, estimator
            gc.collect()
            torch.cuda.empty_cache()

        scene_dir = output_dir / scene.scene_id
        missing = [
            v.video_id for v in scene.videos
            if not any((scene_dir / v.video_id / "body_data").glob("person_*.npz"))
        ]
        if missing:
            failed[scene.scene_id] = missing
        if len(missing) < len(scene.videos) and not (scene_dir / "gt_reid.json").exists():
            needs_reid.append(scene.scene_id)

    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    if failed:
        print(f"\nScenes with missing body data ({len(failed)}):")
        for sid, cams in failed.items():
            print(f"  {sid}: {cams}")
    else:
        print("\nBody estimation complete for all scenes.")
    if needs_reid:
        print(f"\nScenes missing GT ReID ({len(needs_reid)}):")
        for sid in needs_reid:
            print(f"  {sid}")
    else:
        print("\nGT ReID complete for all scenes.")
    print("=" * 64)


if __name__ == "__main__":
    main()
