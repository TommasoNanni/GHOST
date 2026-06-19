"""Evaluate BodyPlacer: scale, root translation, and global orientation.

Translation methods compared:
  - Hip midpoint   : back-project left/right hip (MHR70 9/10) through VGGT depth → midpoint
  - PnP            : align FK body (zero global_orient) to 2D keypoints via RANSAC PnP

Orientation:
  - Triangulate hip/shoulder keypoints across cameras → body coordinate frame

GT comparison is against the true pelvis world position:
    pelvis_world = GT_transl + joint_0_canonical(GT_betas)
  (joint_0_canonical ≈ [0.003, −0.354, 0.012] — shape-dependent, rotation-invariant)

Usage:
    pixi run python scripts/eval_placer_trans.py \\
        --scene_root /path/to/ghost_outputs/BBQ_001_guitar \\
        --rich_root  /path/to/rich \\
        --smplx_model /path/to/SMPLX_NEUTRAL.pkl \\
        [--use_gt_intrinsics]
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer, _SMPLX_TO_MHR70

_EVAL_ROOT = Path(__file__).resolve().parent.parent / "evaluation"
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
from evaluate_on_rich_test import (
    metric_wa_mpjpe, metric_w_mpjpe, metric_ga_mpjpe, metric_pa_mpjpe, metric_rte,
    load_gt_body_data as _load_gt_body_data_chromm,
    _BODY_JOINT_IDX,
)

_6D_TO_AA_IMPORT = None  # lazily set to _6d_to_aa from evaluate_on_rich_test

SCENE_ROOT:       Path
RICH_ROOT:        Path
SMPLX_MODEL:      Path
USE_GT_INTRINSICS: bool = False
MAX_SCENES:       int | None = None
CAMERAS_ONLY:     bool = False
FUSION_MODEL  = None   # loaded FusionWithBetas, set by argparse
FUSION_DEVICE = None   # torch.device

# RICH cameras resolution (full-res, before max_side=1440 resize).
# The scan_calibration XMLs contain intrinsics in this space.
# RICH uses 4112×3008 cameras. Change via --rich_orig_w / --rich_orig_h.
RICH_ORIG_W: int = 4112
RICH_ORIG_H: int = 3008


# ---------------------------------------------------------------------------
# GT intrinsics (RICH calibration XMLs)
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    """'BBQ_001_guitar' → 'BBQ', 'LectureHall_018_...' → 'LectureHall'."""
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def load_gt_intrinsics(scene_name: str) -> list[np.ndarray] | None:
    """Return list of (3,3) K matrices in full-res pixels, one per camera.

    Reads ``scan_calibration/<Location>/calibration/{000..}.xml``.
    Returns None if the calibration directory does not exist.
    """
    location = _scene_to_location(scene_name)
    calib_dir = RICH_ROOT / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        print(f"  [WARN] GT calibration not found at {calib_dir}")
        return None

    ks: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        intr_node = root.find("Intrinsics")
        if intr_node is None:
            continue
        vals = list(map(float, intr_node.find("data").text.split()))
        K = np.array(vals, dtype=np.float32).reshape(3, 3)
        ks.append(K)
    return ks if ks else None


# ---------------------------------------------------------------------------
# GT extrinsics (RICH calibration XMLs)
# ---------------------------------------------------------------------------

def load_gt_extrinsics(scene_name: str) -> list[np.ndarray] | None:
    """Return list of (3,4) [R|t] matrices per camera (camera-from-world, t in metres).

    Reads ``scan_calibration/<Location>/calibration/{000..}.xml``.
    """
    location = _scene_to_location(scene_name)
    calib_dir = RICH_ROOT / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    exts: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root_node = tree.getroot()
        cam_node = root_node.find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        M = np.array(vals, dtype=np.float64).reshape(3, 4)
        exts.append(M)
    return exts if exts else None


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt_trans(scene_name: str) -> dict[int, dict[int, np.ndarray]]:
    """gt[gt_pid][frame_idx] = transl (3,) float32 in RICH world space."""
    gt: dict[int, dict[int, np.ndarray]] = {}
    gt_root = RICH_ROOT / "train_body" / scene_name
    if not gt_root.is_dir():
        return gt
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            t = np.asarray(data["transl"], dtype=np.float64).squeeze()
            gt.setdefault(pid, {})[frame_idx] = t.astype(np.float32)
    return gt


def correct_gt_pelvis(
    gt_trans_raw: dict[int, dict[int, np.ndarray]],
    gt_betas_map: dict[int, np.ndarray],
    placer: "BodyPlacer",
) -> dict[int, dict[int, np.ndarray]]:
    """Return true pelvis world position per pid/frame.

    In SMPL-X, joint[0] (pelvis) is the root of the kinematic tree.
    global_orient rotates the body *around* the pelvis, so joint[0] is
    invariant to global_orient.  The true world position is therefore:

        pelvis_world = GT_transl + joint_0_canonical(betas)

    where joint_0_canonical ≈ [0.003, -0.354, 0.012] for a neutral body.
    """
    zero_pose   = np.zeros((1, 63), dtype=np.float32)
    zero_orient = np.zeros((1, 3),  dtype=np.float32)
    corrected: dict[int, dict[int, np.ndarray]] = {}
    for gt_pid, frames in gt_trans_raw.items():
        betas = gt_betas_map.get(gt_pid, np.zeros(10, dtype=np.float32))
        fk = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)  # (1, 55, 3)
        j0 = fk[0, 0].astype(np.float32)  # canonical pelvis offset (3,)
        corrected[gt_pid] = {frame: (transl + j0) for frame, transl in frames.items()}
    return corrected


def load_gt_global_orient(scene_name: str) -> dict[int, dict[int, np.ndarray]]:
    """gt_orient[gt_pid][frame_idx] = R (3,3) float32 in RICH world space."""
    gt: dict[int, dict[int, np.ndarray]] = {}
    gt_root = RICH_ROOT / "train_body" / scene_name
    if not gt_root.is_dir():
        return gt
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            aa = data.get("global_orient")
            if aa is None:
                continue
            R = SciR.from_rotvec(np.asarray(aa, dtype=np.float64).squeeze()).as_matrix()
            gt.setdefault(pid, {})[frame_idx] = R.astype(np.float32)
    return gt


def _geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle in degrees between two (3,3) rotation matrices."""
    cos = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def eval_global_orient(
    pred_orient: dict[int, tuple[np.ndarray, np.ndarray]],
    gt_orient: dict[int, dict[int, np.ndarray]],
    all_pids: set[int],
) -> None:
    """Print geodesic rotation errors for estimate_global_orient output."""
    gt_pids = sorted(gt_orient.keys())
    all_errs: list[float] = []

    for pid in sorted(all_pids):
        if pid not in pred_orient:
            print(f"    pid {pid}: no estimates")
            continue
        fi, R_pred = pred_orient[pid]
        gt_pid = min(gt_pids, key=lambda p: abs(p - pid))
        gt_frames = gt_orient.get(gt_pid, {})
        errs = [
            _geodesic_deg(R_pred[i], gt_frames[int(fi[i])])
            for i in range(len(fi))
            if int(fi[i]) in gt_frames
        ]
        if not errs:
            print(f"    pid {pid}: no matched frames")
            continue
        e = np.array(errs)
        all_errs.extend(errs)
        print(
            f"    pid {pid}: N={len(e):4d}  "
            f"median={np.median(e):.1f}°  mean={np.mean(e):.1f}°  "
            f"<30°={(e < 30).mean()*100:5.1f}%  <60°={(e < 60).mean()*100:5.1f}%"
        )
    if all_errs:
        e = np.array(all_errs)
        print(
            f"    TOTAL: N={len(e):4d}  "
            f"median={np.median(e):.1f}°  mean={np.mean(e):.1f}°  "
            f"<30°={(e < 30).mean()*100:5.1f}%  <60°={(e < 60).mean()*100:5.1f}%"
        )
    print()


def load_gt_body_pose(scene_name: str) -> dict[int, dict[int, np.ndarray]]:
    """gt_body_pose[gt_pid][frame_idx] = body_pose (63,) float32."""
    gt: dict[int, dict[int, np.ndarray]] = {}
    gt_root = RICH_ROOT / "train_body" / scene_name
    if not gt_root.is_dir():
        return gt
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            bp = data.get("body_pose")
            if bp is None:
                continue
            gt.setdefault(pid, {})[frame_idx] = np.asarray(bp, dtype=np.float32).reshape(63)
    return gt


def load_gt_betas(scene_name: str) -> dict[int, np.ndarray]:
    """gt_betas[gt_pid] = betas (10,) — takes first valid frame per pid."""
    gt: dict[int, np.ndarray] = {}
    gt_root = RICH_ROOT / "train_body" / scene_name
    if not gt_root.is_dir():
        return gt
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            pid = int(pkl_path.stem)
            if pid in gt:
                continue
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            raw = data.get("betas")
            if raw is None:
                raw = data.get("smplx_betas")
            if raw is not None:
                gt[pid] = np.asarray(raw, dtype=np.float32).reshape(-1)[:10]
    return gt


# ---------------------------------------------------------------------------
# Predicted betas: average over frames and cameras per ghost pid
# ---------------------------------------------------------------------------

def load_pred_betas(cam_dirs: list[Path]) -> dict[int, np.ndarray]:
    """pred_betas[ghost_pid] = mean betas (10,) from raw SAM3D body_data files."""
    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    for cam_dir in cam_dirs:
        for f in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(f.stem.split("_")[1])
            data = np.load(f, allow_pickle=False)
            if "smplx_betas" in data.files:
                accum[pid].append(data["smplx_betas"].mean(axis=0))  # (10,)
    return {pid: np.mean(np.stack(v, axis=0), axis=0) for pid, v in accum.items()}


# ---------------------------------------------------------------------------
# Translation evaluation
# ---------------------------------------------------------------------------



def _match_pids_by_distance(
    preds: dict[int, dict[int, np.ndarray]],
    gt_trans: dict[int, dict[int, np.ndarray]],
    foreground_pids: set[int],
) -> dict[int, int]:
    """Match ghost pids to GT pids by minimum mean 3-D translation distance.

    Same logic as RICHFusionDatapoint._match_persons_to_gt(): for each GT
    person pick the ghost person whose predicted translation is closest on
    average over common frames.  Only foreground ghost pids (visible in ≥2
    cameras) are considered; background detections are excluded.

    Returns {ghost_pid: gt_pid}.
    """
    ghost_pids = sorted(foreground_pids)
    gt_pids    = sorted(gt_trans.keys())
    gt_to_ghost: dict[int, int] = {}
    for gt_pid in gt_pids:
        gt_frames = gt_trans[gt_pid]
        best_gpid, best_dist = None, float("inf")
        for gpid in ghost_pids:
            pred_frames = preds.get(gpid, {})
            common = set(pred_frames) & set(gt_frames)
            if not common:
                continue
            mean_dist = float(np.mean([
                np.linalg.norm(pred_frames[f] - gt_frames[f]) for f in common
            ]))
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_gpid = gpid
        if best_gpid is not None:
            gt_to_ghost[gt_pid] = best_gpid
    return {ghost: gt for gt, ghost in gt_to_ghost.items()}


def eval_translations(
    preds: dict[int, dict[int, np.ndarray]],
    gt_trans: dict[int, dict[int, np.ndarray]],
    foreground_pids: set[int],
) -> tuple[dict[int, list[float]], list[np.ndarray], dict[int, int]]:
    """Compare fused predictions against GT.

    Returns:
        errors_per_pid : {ghost_pid: [L2 error in metres per matched frame]}
        signed_vecs    : list of (pred - gt) (3,) vectors for axis-bias analysis
        pid_match      : {ghost_pid: gt_pid} used for matching (for orient eval)
    """
    pid_match = _match_pids_by_distance(preds, gt_trans, foreground_pids)
    errors_per_pid: dict[int, list[float]] = defaultdict(list)
    signed_vecs: list[np.ndarray] = []

    for pid, gt_pid in pid_match.items():
        pred_frames = preds.get(pid, {})
        gt_frames   = gt_trans[gt_pid]
        for global_t, t_pred in sorted(pred_frames.items()):
            if global_t in gt_frames:
                diff = t_pred - gt_frames[global_t]
                errors_per_pid[pid].append(float(np.linalg.norm(diff)))
                signed_vecs.append(diff)

    return errors_per_pid, signed_vecs, pid_match


# ---------------------------------------------------------------------------
# PnP-based translation (prototype)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Procrustes-based translation + orientation
# ---------------------------------------------------------------------------

def predict_translations_procrustes(
    placer: BodyPlacer,
    scale: float,
    cam_dirs: list[Path],
    all_pids: set[int],
    pred_betas_by_pid: dict[int, np.ndarray],
    gt_intrinsics: list[np.ndarray] | None = None,
    min_cams: int = 2,
    min_joints: int = 4,
    gt_body_pose_map: dict[int, dict[int, np.ndarray]] | None = None,
    gt_cameras: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
    gt_betas_by_pid: dict[int, np.ndarray] | None = None,
    gt_transl_cameras: list[np.ndarray | None] | None = None,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
    """Estimate root translation + global orient via multi-camera DLT + Procrustes.


    For each (person, frame):
      1. For each SMPL-X joint j (with MHR70 mapping), collect 2D observations
         from all cameras and DLT-triangulate → J_world[j] in metric world space.
      2. Run SMPL-X FK with predicted (or GT) body_pose + betas, zero global_orient,
         zero transl → J_can[j] for the same joints.
      3. Procrustes: minimize ||R @ J_can[visible] + t − J_world[visible]||²
         → global_orient R and (implicit) pelvis translation t.

    Returns:
        translations : {ghost_pid: {global_frame_idx: pelvis_world (3,)}}
        orientations : {ghost_pid: {global_frame_idx: R (3,3)}}
    """
    _JOINTS = sorted(_SMPLX_TO_MHR70.keys())

    # Pre-load body data once per (cam, pid) to avoid repeated file reads.
    cam_data_all: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_map: dict[int, dict] = {}
        for pid in sorted(all_pids):
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if not bf.exists():
                continue
            d = np.load(bf, allow_pickle=False)
            assert {"pred_keypoints_2d", "frame_indices", "smplx_body_pose"}.issubset(d.files), (
                f"Missing required keys in {bf}"
            )
            fi = d["frame_indices"].astype(int)
            cam_map[pid] = {
                "local_t": {int(g): int(l) for l, g in enumerate(fi)},
                "kp2d":       d["pred_keypoints_2d"],
                "body_pose":  d["smplx_body_pose"],
            }
        cam_data_all.append(cam_map)

    translations: dict[int, dict[int, np.ndarray]] = {}
    orientations: dict[int, dict[int, np.ndarray]] = {}

    for pid in sorted(all_pids):
        if gt_betas_by_pid is not None and pid in gt_betas_by_pid:
            betas = gt_betas_by_pid[pid]
        else:
            betas = pred_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))

        all_frames: set[int] = set()
        for cm in cam_data_all:
            if pid in cm:
                all_frames.update(cm[pid]["local_t"].keys())

        trans_out: dict[int, np.ndarray] = {}
        orient_out: dict[int, np.ndarray] = {}

        for global_t in sorted(all_frames):
            if global_t >= placer.T:
                continue

            # ── Step 1: DLT-triangulate each joint across cameras ─────────────
            joint_world: dict[int, np.ndarray] = {}
            for smplx_j in _JOINTS:
                mhr70_j = _SMPLX_TO_MHR70[smplx_j]
                obs:   list[tuple[float, float]] = []
                pmats: list[np.ndarray] = []

                for k, cm in enumerate(cam_data_all):
                    if pid not in cm:
                        continue
                    if global_t not in cm[pid]["local_t"]:
                        continue
                    if not placer.cam_valid[global_t, k]:
                        continue

                    local_t = cm[pid]["local_t"][global_t]
                    kp2d  = cm[pid]["kp2d"]
                    if mhr70_j >= kp2d.shape[1]:
                        continue

                    oc  = placer.original_coords[global_t, k]
                    os_ = placer.original_size[global_t, k]
                    W_orig, H_orig = float(os_[0]), float(os_[1])

                    use_gt_cam = (
                        gt_cameras is not None
                        and k < len(gt_cameras)
                        and gt_cameras[k] is not None
                    )
                    use_gt_transl = (
                        not use_gt_cam
                        and gt_transl_cameras is not None
                        and k < len(gt_transl_cameras)
                        and gt_transl_cameras[k] is not None
                    )

                    if use_gt_cam:
                        # GT cameras: observations in resized-image-space pixels,
                        # projection matrix in RICH world frame (metric, no VGGT scale).
                        K_gt_resized, E_gt_metric = gt_cameras[k]
                        u = float(kp2d[local_t, mhr70_j][0])
                        v = float(kp2d[local_t, mhr70_j][1])
                        if not (0.0 <= u < W_orig and 0.0 <= v < H_orig):
                            continue
                        P = K_gt_resized.astype(np.float64) @ E_gt_metric.astype(np.float64)
                    elif use_gt_transl:
                        # Hybrid: VGGT predicted rotation + GT translation (metric, no scaling).
                        # Observations are in VGGT output space.
                        u, v = placer._orig_to_vggt(kp2d[local_t, mhr70_j], oc, W_orig, H_orig)
                        if not placer._in_bounds(u, v, oc[2], oc[3]):
                            continue
                        intr = placer.intrinsics[global_t, k].astype(np.float64)
                        ext = placer.extrinsics[global_t, k].astype(np.float64).copy()
                        ext[:3, 3] = gt_transl_cameras[k].astype(np.float64)
                        P = intr @ ext
                    else:
                        # VGGT cameras: map keypoint to VGGT output space.
                        u, v = placer._orig_to_vggt(kp2d[local_t, mhr70_j], oc, W_orig, H_orig)
                        if not placer._in_bounds(u, v, oc[2], oc[3]):
                            continue
                        intr = placer.intrinsics[global_t, k].astype(np.float64)
                        if gt_intrinsics and k < len(gt_intrinsics):
                            gt_K = gt_intrinsics[k]
                            intr = intr.copy()
                            intr[0, 2] = float(oc[0]) + float(gt_K[0, 2]) * intr[0, 0] / float(gt_K[0, 0])
                            intr[1, 2] = float(oc[1]) + float(gt_K[1, 2]) * intr[1, 1] / float(gt_K[1, 1])
                        ext = placer.extrinsics[global_t, k].astype(np.float64).copy()
                        ext[:3, 3] *= float(scale[global_t]) if isinstance(scale, np.ndarray) else scale
                        P = intr @ ext

                    pmats.append(P)
                    obs.append((u, v))

                if len(obs) >= min_cams:
                    joint_world[smplx_j] = placer._triangulate_dlt(obs, pmats)

            if len(joint_world) < min_joints:
                continue

            # ── Step 2: FK canonical joint positions ──────────────────────────
            if gt_body_pose_map is not None and pid in gt_body_pose_map:
                body_pose_frame = gt_body_pose_map[pid].get(global_t)
            else:
                body_pose_frame = None
                for cm in cam_data_all:
                    if pid in cm and global_t in cm[pid]["local_t"]:
                        lt = cm[pid]["local_t"][global_t]
                        body_pose_frame = cm[pid]["body_pose"][lt]
                        break
            if body_pose_frame is None:
                continue  # frame not annotated / detection gap

            fk = placer._smplx_fk(
                betas[np.newaxis],
                body_pose_frame[np.newaxis],
                np.zeros((1, 3), dtype=np.float32),
            )  # (1, 55, 3)
            J_can = fk[0]  # (55, 3) in metres

            # ── Step 3: Procrustes — find R, t s.t. R @ J_can + t ≈ J_world ──
            vis = sorted(joint_world.keys())
            A = np.stack([joint_world[j] for j in vis], axis=0).astype(np.float64)  # world (M,3)
            B = np.stack([J_can[j]       for j in vis], axis=0).astype(np.float64)  # canonical (M,3)

            A_mean = A.mean(0)
            B_mean = B.mean(0)
            # H = B_c.T @ A_c  →  R s.t. R @ B ≈ A
            H = (B - B_mean).T @ (A - A_mean)
            U, _, Vt = np.linalg.svd(H)
            d_sign = np.linalg.det(Vt.T @ U.T)
            R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
            t = (A_mean - R.astype(np.float64) @ B_mean).astype(np.float32)

            # Pelvis world position: R @ J_can[pelvis] + t
            # (SMPL-X global_orient rotates around the pelvis, so pelvis_world = R@j0 + t)
            pelvis_world = (R.astype(np.float64) @ J_can[0].astype(np.float64) + t.astype(np.float64)).astype(np.float32)

            trans_out[global_t] = pelvis_world
            orient_out[global_t] = R

        translations[pid] = trans_out
        orientations[pid] = orient_out

    return translations, orientations


# ---------------------------------------------------------------------------
# GT scale computation
# ---------------------------------------------------------------------------

def compute_gt_scale(
    placer: "BodyPlacer",
    gt_extrinsics: list[np.ndarray],
    cam_dirs: list[Path],
) -> float | None:
    """Compute GT depth scale as median(baseline_gt / baseline_vggt) over non-reference cameras.

    baseline = physical distance between reference camera and camera k.
    Using camera centers C = -R^T @ t avoids sensitivity to the choice of world origin
    and handles any reference camera correctly (not just cam_00).
    """
    # Reference camera: first cam_dir → look up its GT extrinsic
    _m_ref = re.search(r"\d+", cam_dirs[0].name) if cam_dirs else None
    ref_xml_idx = int(_m_ref.group()) if _m_ref else 0
    if ref_xml_idx >= len(gt_extrinsics):
        return None
    E_ref = gt_extrinsics[ref_xml_idx].astype(np.float64)
    C_ref = -E_ref[:3, :3].T @ E_ref[:3, 3]   # camera center in GT world (metres)

    samples: list[float] = []
    for k in range(1, placer.K):
        _m = re.search(r"\d+", cam_dirs[k].name)
        gt_idx = int(_m.group()) if _m else k
        if gt_idx >= len(gt_extrinsics):
            continue
        E_k  = gt_extrinsics[gt_idx].astype(np.float64)
        C_k  = -E_k[:3, :3].T @ E_k[:3, 3]
        baseline_gt = float(np.linalg.norm(C_k - C_ref))
        if baseline_gt < 1e-6:
            continue
        valid_frames = np.where(placer.cam_valid[:, k])[0]
        for global_t in valid_frames:
            t_vggt = placer.extrinsics[global_t, k, :3, 3]
            norm_vggt = float(np.linalg.norm(t_vggt))
            if norm_vggt > 1e-6:
                samples.append(baseline_gt / norm_vggt)
    return float(np.median(samples)) if samples else None


# ---------------------------------------------------------------------------
# Camera comparison table
# ---------------------------------------------------------------------------

_BONE_NAMES = {
    (16, 18): "L humerus",
    (17, 19): "R humerus",
    (1,  4):  "L femur  ",
    (2,  5):  "R femur  ",
    (4,  7):  "L tibia  ",
    (5,  8):  "R tibia  ",
}


def eval_chromm_metrics(
    label: str,
    placer: "BodyPlacer",
    trans_dict: dict[int, dict[int, np.ndarray]],
    orient_dict: dict[int, dict[int, np.ndarray]],
    pid_match: dict[int, int],
    gt_body_data: "dict[int, dict[int, dict]]",
    fused_pose_arr: np.ndarray,        # (T_scene, P, 54, 6) from _run_fusion_fwd
    pid_to_slot: dict[int, int],       # ghost_pid → slot in fused_pose_arr
    betas_by_pid: dict[int, np.ndarray],
    frame_start: int,
    agg: dict[str, list[float]],
) -> None:
    """Compute WA-MPJPE, W-MPJPE, GA-MPJPE, PA-MPJPE, RTE and print + accumulate."""
    from fusion.placer import _6d_to_aa_batch  # noqa: PLC0415

    if not pid_match:
        return

    all_frames: set[int] = set()
    for ghost_pid in pid_match:
        for frames in trans_dict.get(ghost_pid, {}).keys():
            all_frames.add(frames)
    if not all_frames:
        return

    frame_start_t = min(all_frames)
    frame_end_t   = max(all_frames)
    T = frame_end_t - frame_start_t + 1
    P = len(pid_match)
    J = len(_BODY_JOINT_IDX)

    pred_joints = np.full((T, P, J, 3), np.nan, dtype=np.float32)
    pred_roots  = np.full((T, P, 3),    np.nan, dtype=np.float32)
    gt_joints   = np.full((T, P, J, 3), np.nan, dtype=np.float32)
    gt_roots    = np.full((T, P, 3),    np.nan, dtype=np.float32)

    for slot, (ghost_pid, gt_pid) in enumerate(sorted(pid_match.items())):
        betas = betas_by_pid.get(ghost_pid, np.zeros(10, dtype=np.float32))
        p_slot = pid_to_slot.get(ghost_pid)

        # ── Predicted world joints ────────────────────────────────────────────
        for global_t, pelvis_world in trans_dict.get(ghost_pid, {}).items():
            R_mat = orient_dict.get(ghost_pid, {}).get(global_t)
            if R_mat is None or p_slot is None:
                continue
            t_rel = int(global_t) - frame_start_t
            if not (0 <= t_rel < T):
                continue
            t_fused = int(global_t) - frame_start
            if not (0 <= t_fused < fused_pose_arr.shape[0]):
                continue
            bp_6d = fused_pose_arr[t_fused, p_slot, :21]   # (21, 6)
            bp_aa = _6d_to_aa_batch(bp_6d).reshape(63)
            J_can = placer._smplx_fk(
                betas[np.newaxis],
                bp_aa[np.newaxis],
                np.zeros((1, 3), dtype=np.float32),
            )[0]  # (55, 3)
            J_world = (R_mat @ (J_can - J_can[0]).T).T + pelvis_world
            pred_joints[t_rel, slot] = J_world[_BODY_JOINT_IDX]
            pred_roots[t_rel,  slot] = J_world[0]

        # ── GT world joints ───────────────────────────────────────────────────
        for frame_idx, params in gt_body_data.get(gt_pid, {}).items():
            t_rel = int(frame_idx) - frame_start_t
            if not (0 <= t_rel < T):
                continue
            J_gt = placer._smplx_fk(
                params["betas"][np.newaxis],
                params["body_pose"][np.newaxis],
                params["global_orient"][np.newaxis],
            )[0] + params["transl"]
            gt_joints[t_rel, slot] = J_gt[_BODY_JOINT_IDX]
            gt_roots[t_rel,  slot] = J_gt[0]

    valid = (
        np.isfinite(pred_joints).all((-2, -1)) &
        np.isfinite(gt_joints).all((-2, -1))
    )
    if not valid.any():
        print(f"  [{label}] CHROMM: no valid pairs")
        return

    wa  = metric_wa_mpjpe(pred_joints, gt_joints, valid)
    w   = metric_w_mpjpe( pred_joints, gt_joints, valid)
    ga  = metric_ga_mpjpe(pred_joints, gt_joints, valid)
    pa  = metric_pa_mpjpe(pred_joints, gt_joints, valid)
    rte = metric_rte(pred_roots, gt_roots)

    print(f"  [{label}] WA={wa:.1f}mm  W={w:.1f}mm  GA={ga:.1f}mm  PA={pa:.1f}mm  RTE={rte:.2f}%")
    for k, v in [("wa", wa), ("w", w), ("ga", ga), ("pa", pa), ("rte", rte)]:
        if np.isfinite(v):
            agg[f"{label}/{k}"].append(v)


def eval_bone_lengths(
    placer: "BodyPlacer",
    pid_match: dict[int, int],
    gt_betas_map: dict[int, np.ndarray],
    pred_betas_map: dict[int, np.ndarray],
) -> None:
    """Print per-bone length error between pred-betas FK and GT-betas FK."""
    from fusion.placer import _LONG_BONES

    # Single-pass: FK once per pid, collect lengths and errors together.
    # bone_data[bone] = list of (L_pred, L_gt) per matched pid
    bone_data: dict[tuple, list[tuple[float, float]]] = {b: [] for b in _LONG_BONES}

    zero_bp = np.zeros((1, 63), dtype=np.float32)
    zero_go = np.zeros((1, 3),  dtype=np.float32)

    for ghost_pid, gt_pid in pid_match.items():
        pred_b = pred_betas_map.get(ghost_pid)
        gt_b   = gt_betas_map.get(gt_pid)
        if pred_b is None or gt_b is None:
            continue
        pred_fk = placer._smplx_fk(pred_b[np.newaxis], zero_bp, zero_go)[0]  # (55, 3)
        gt_fk   = placer._smplx_fk(gt_b[np.newaxis],   zero_bp, zero_go)[0]
        for j_a, j_b in _LONG_BONES:
            L_pred = float(np.linalg.norm(pred_fk[j_b] - pred_fk[j_a]))
            L_gt   = float(np.linalg.norm(gt_fk[j_b]   - gt_fk[j_a]))
            if L_gt > 1e-4:
                bone_data[(j_a, j_b)].append((L_pred, L_gt))

    if not any(bone_data.values()):
        return

    print("  [Bone length error: pred betas vs GT betas]")
    print(f"  {'bone':<12}  {'L_pred(m)':>9}  {'L_gt(m)':>9}  {'err%':>7}")
    print("  " + "─" * 42)
    all_errs: list[float] = []
    for bone, pairs in bone_data.items():
        if not pairs:
            continue
        pred_lens = [p for p, _ in pairs]
        gt_lens   = [g for _, g in pairs]
        errs      = [(p - g) / g * 100.0 for p, g in pairs]
        name      = _BONE_NAMES.get(bone, str(bone))
        all_errs.extend(errs)
        print(f"  {name:<12}  {np.mean(pred_lens):>9.4f}  {np.mean(gt_lens):>9.4f}  {np.mean(errs):>+7.1f}%")
    print(f"  {'ALL':<12}  {'':>9}  {'':>9}  {float(np.mean(all_errs)):>+7.1f}%")
    print()


def eval_cameras(
    placer: "BodyPlacer",
    scale: float,
    scene_name: str,
    cam_dirs: list[Path],
) -> None:
    """Print per-camera table: VGGT vs GT rotation, translation, and intrinsics.

    Rotation error: geodesic angle (°) between VGGT and GT rotation matrices.
    Translation error: ||t_vggt * scale − t_gt_rel|| in metres, where t_gt_rel is
    the GT extrinsic re-expressed relative to the first available camera (= VGGT reference).
    Intrinsics: converted from VGGT output space to original pixel space.
        Δfx (%) = (fx_vggt_orig − fx_gt) / fx_gt × 100
        Δcx, Δcy = principal point offset in original pixels
    """
    gt_exts = load_gt_extrinsics(scene_name)
    gt_intr = load_gt_intrinsics(scene_name)

    # GT reference extrinsic — same reference camera VGGT uses (first cam_dir)
    _m_ref = re.search(r"\d+", cam_dirs[0].name) if cam_dirs else None
    ref_xml_idx = int(_m_ref.group()) if _m_ref else 0
    if gt_exts and ref_xml_idx < len(gt_exts):
        E_ref  = gt_exts[ref_xml_idx].astype(np.float64)
        R_ref  = E_ref[:3, :3]
        t_ref  = E_ref[:3, 3]
    else:
        R_ref = np.eye(3, dtype=np.float64)
        t_ref = np.zeros(3, dtype=np.float64)

    K = placer.extrinsics.shape[1]
    n_cams = K

    print("  [Camera diagnostics: VGGT vs GT]")
    hdr = (
        f"  {'cam':>4}  {'rot_med(°)':>10}  {'rot_mean(°)':>11}"
        f"  {'t_med(m)':>8}  {'t_mean(m)':>9}"
        f"  {'Δfx(%)':>8}  {'Δcx(px)':>9}  {'Δcy(px)':>9}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for k in range(n_cams):
        cam_name = cam_dirs[k].name if k < len(cam_dirs) else str(k)
        _m = re.search(r"\d+", cam_name)
        gt_idx = int(_m.group()) if _m else k

        valid_frames = np.where(placer.cam_valid[:, k])[0]
        if len(valid_frames) == 0:
            print(f"  {cam_name:>6}  {'(no valid frames)':>54}")
            continue

        # VGGT extrinsics over all valid frames
        vggt_R_all = placer.extrinsics[valid_frames, k, :3, :3]  # (N, 3, 3)
        vggt_t_all = placer.extrinsics[valid_frames, k, :3, 3]   # (N, 3)

        # VGGT intrinsics in VGGT output space (use first valid frame — static camera)
        vggt_intr = placer.intrinsics[valid_frames[0], k]          # (3, 3)
        vggt_oc   = placer.original_coords[valid_frames[0], k]     # [x1,y1,x2,y2]
        vggt_os   = placer.original_size[valid_frames[0], k]       # [W_orig, H_orig]

        W_orig, H_orig = float(vggt_os[0]), float(vggt_os[1])
        x1, y1, x2, y2 = (float(vggt_oc[i]) for i in range(4))
        crop_w, crop_h = x2 - x1, y2 - y1

        # Convert focal length and principal point to original pixel space
        fx_vggt = float(vggt_intr[0, 0]) * W_orig / crop_w
        cx_vggt = (float(vggt_intr[0, 2]) - x1) * W_orig / crop_w
        cy_vggt = (float(vggt_intr[1, 2]) - y1) * H_orig / crop_h

        # ── Intrinsics vs GT ────────────────────────────────────────────────
        # GT calibration is in full-res (RICH_ORIG_W × RICH_ORIG_H).
        # Scale to the resized (1440-space) image that VGGT processed.
        if gt_intr and gt_idx < len(gt_intr):
            gt_K = gt_intr[gt_idx]
            s_x = W_orig / RICH_ORIG_W
            s_y = H_orig / RICH_ORIG_H
            fx_gt_scaled = float(gt_K[0, 0]) * s_x
            cx_gt_scaled = float(gt_K[0, 2]) * s_x
            cy_gt_scaled = float(gt_K[1, 2]) * s_y
            dfx = (fx_vggt - fx_gt_scaled) / fx_gt_scaled * 100.0
            dcx = cx_vggt - cx_gt_scaled
            dcy = cy_vggt - cy_gt_scaled
            intr_str = f"  {dfx:>+8.1f}%  {dcx:>+9.0f}  {dcy:>+9.0f}"
        else:
            intr_str = f"  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}"

        # ── Extrinsics vs GT ────────────────────────────────────────────────
        # Re-express GT extrinsic in VGGT reference frame (first cam_dir).
        if gt_exts and gt_idx < len(gt_exts):
            E_k   = gt_exts[gt_idx].astype(np.float64)
            R_k   = E_k[:3, :3]
            t_k   = E_k[:3, 3]
            R_k_rel = R_k @ R_ref.T
            t_k_rel = t_k - R_k_rel @ t_ref   # GT translation in VGGT reference frame

            rot_errs = [_geodesic_deg(vggt_R_all[i], R_k_rel) for i in range(len(valid_frames))]
            rot_med  = float(np.median(rot_errs))
            rot_mean = float(np.mean(rot_errs))

            t_scaled = vggt_t_all * scale
            t_errs   = np.linalg.norm(t_scaled - t_k_rel[np.newaxis], axis=1)
            t_med    = float(np.median(t_errs))
            t_mean   = float(np.mean(t_errs))

            ext_str = f"  {rot_med:>10.2f}  {rot_mean:>11.2f}  {t_med:>8.4f}  {t_mean:>9.4f}"
        else:
            ext_str = f"  {'N/A':>10}  {'N/A':>11}  {'N/A':>8}  {'N/A':>9}"

        print(f"  {cam_name:>6}{ext_str}{intr_str}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_results(
    label: str,
    errors: dict[int, list[float]],
    signed_vecs: list[np.ndarray],
) -> None:
    print(f"  [{label}]")
    all_errs: list[float] = []
    for pid, errs_list in sorted(errors.items()):
        e = np.array(errs_list)
        all_errs.extend(errs_list)
        print(
            f"    pid {pid}: N={len(e):4d}  "
            f"median={np.median(e):.3f}m  mean={np.mean(e):.3f}m  "
            f"<0.5m={(e < 0.5).mean()*100:5.1f}%"
        )
    if all_errs:
        e = np.array(all_errs)
        print(
            f"    TOTAL: N={len(e):4d}  "
            f"median={np.median(e):.3f}m  mean={np.mean(e):.3f}m  "
            f"<0.5m={(e < 0.5).mean()*100:5.1f}%  <1.0m={(e < 1.0).mean()*100:5.1f}%"
        )
    else:
        print("    TOTAL: no valid estimates")
    if signed_vecs:
        ae = np.array(signed_vecs)
        print(
            f"    Axis bias (pred−GT):  "
            f"x={ae[:, 0].mean():+.3f}m  y={ae[:, 1].mean():+.3f}m  z={ae[:, 2].mean():+.3f}m"
        )
    print()


def _to_orient_fmt(
    frame_Rs: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not frame_Rs:
        return None
    frames = np.array(sorted(frame_Rs.keys()), dtype=np.int32)
    Rs = np.stack([frame_Rs[f] for f in frames], axis=0)
    return frames, Rs


# ---------------------------------------------------------------------------
# Fusion model helpers
# ---------------------------------------------------------------------------

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) → 6-D rotation (..., 6) (first two rows of R).

    Matches the convention used in fusion_dataset.py (training data encoding).
    """
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()  # (N, 3, 3)
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)  # rows → (N, 6)
    return sixd.reshape(shape + (6,)).astype(np.float32)


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """6-D rotation (..., 6) → axis-angle (..., 3).

    Inverse of _aa_to_6d: interprets the 6 values as the first two rows of R,
    applies Gram-Schmidt, then converts to axis-angle.
    """
    shape = sixd.shape[:-1]
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)   # (N, 3, 3) — rows are b1, b2, b3
    return SciR.from_matrix(R).as_rotvec().reshape(shape + (3,)).astype(np.float32)


def _load_fusion_model(checkpoint: Path, device: "torch.device"):
    """Load FusionWithBetas from a checkpoint, infer hyper-params from state dict."""
    import torch
    from fusion.fusion_module_v2 import FusionWithBetas, PoseFusionModule, BetasAggregator
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    embedding_dim    = state["pose_module.joint_id_embedding.weight"].shape[1]
    num_joints       = state["pose_module.joint_id_embedding.weight"].shape[0]
    num_layers       = sum(1 for k in state
                           if k.startswith("pose_module.layers.") and k.endswith(".ff.norm.weight"))
    max_temporal_len = state["pose_module.temporal_pe.pe"].shape[0]
    num_inducing     = state["betas_aggregator.inducing_points"].shape[0]
    betas_emb_dim    = state["betas_aggregator.inducing_points"].shape[1]
    pose_module = PoseFusionModule(
        embedding_dim=embedding_dim, num_layers=num_layers,
        num_joints=num_joints, max_temporal_len=max_temporal_len,
    )
    betas_agg = BetasAggregator(n_betas=10, embedding_dim=betas_emb_dim,
                                num_inducing=num_inducing, dropout=0.1)
    model = FusionWithBetas(pose_module, betas_agg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  [fusion] loaded checkpoint: embedding_dim={embedding_dim} "
          f"num_layers={num_layers} num_joints={num_joints}")
    return model


def _run_fusion_fwd(
    cam_dirs: list[Path],
    model: "Any",
    device: "Any",
) -> "tuple[np.ndarray, np.ndarray | None, list[int], int] | None":
    """Build input tensors from cam_dirs, run model, return (fused_pose, all_pids, frame_start)."""
    import torch

    J = model.pose_module.joint_id_embedding.weight.shape[0]  # root-excluded (e.g. 54)
    num_joints = J + 1  # total joints including root, for padding (e.g. 55)

    # Load body_data per camera
    raw: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_map: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            d = np.load(npz_path, allow_pickle=False)
            cam_map[pid] = {k: d[k] for k in d.files}
        raw.append(cam_map)

    all_pids   = sorted({pid for cam in raw for pid in cam})
    all_frames = sorted({int(fi) for cam in raw
                         for pdata in cam.values() for fi in pdata["frame_indices"]})
    if not all_pids or not all_frames:
        return None

    frame_start = all_frames[0]
    T = all_frames[-1] - frame_start + 1
    K = len(cam_dirs)
    P = len(all_pids)
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    pose_arr  = np.zeros((T, K, P, J, 6),  dtype=np.float32)
    mask_arr  = np.zeros((T, K, P),         dtype=np.float32)
    shape_arr = np.zeros((T, K, P, 10),     dtype=np.float32)

    for k, cam in enumerate(raw):
        for pid, pdata in cam.items():
            p  = pid_to_slot[pid]
            fi = pdata["frame_indices"].astype(int)
            go    = pdata["smplx_global_orient"]         # (T_local, 3)
            bp    = pdata["smplx_body_pose"]             # (T_local, 63)
            betas = pdata.get("smplx_betas")             # (T_local, 10)
            lh    = pdata.get("smplx_left_hand_pose")    # (T_local, 45) or None
            rh    = pdata.get("smplx_right_hand_pose")   # (T_local, 45) or None
            for local_t, global_t in enumerate(fi):
                t = int(global_t) - frame_start
                if not (0 <= t < T):
                    continue
                parts = [go[local_t].reshape(1, 3), bp[local_t].reshape(21, 3)]
                if lh is not None:
                    parts.append(lh[local_t].reshape(15, 3))
                if rh is not None:
                    parts.append(rh[local_t].reshape(15, 3))
                aa = np.concatenate(parts, axis=0)  # (22 or 52, 3)
                if aa.shape[0] < num_joints:
                    aa = np.concatenate([aa, np.zeros((num_joints - aa.shape[0], 3),
                                                      dtype=np.float32)])
                sixd = _aa_to_6d(aa)          # (num_joints, 6)
                pose_arr[t, k, p]  = sixd[1:]  # root excluded
                mask_arr[t, k, p]  = 1.0
                if betas is not None:
                    shape_arr[t, k, p] = betas[local_t, :10]

    pose_t  = torch.from_numpy(pose_arr).unsqueeze(0).to(device)
    mask_t  = torch.from_numpy(mask_arr).unsqueeze(0).to(device)
    shape_t = torch.from_numpy(shape_arr).unsqueeze(0).to(device)

    with torch.no_grad():
        pose_aggr, betas_out = model(pose_t, mask_t, shape=shape_t)

    fused_pose  = pose_aggr[0].cpu().numpy()   # (T, P, J, 6)
    return fused_pose, all_pids, frame_start


def run() -> None:
    if (SCENE_ROOT / "vggt_cameras_centered.npz").exists():
        scenes = [SCENE_ROOT]
    else:
        scenes = sorted(
            d for d in SCENE_ROOT.iterdir()
            if d.is_dir() and (d / "vggt_cameras_centered.npz").exists()
        )

    if MAX_SCENES is not None:
        scenes = scenes[:MAX_SCENES]

    # Cross-scene error accumulators {label: [L2 errors]}
    agg_trans:  dict[str, list[float]] = defaultdict(list)
    agg_orient: dict[str, list[float]] = defaultdict(list)
    agg_scale:  list[tuple[str, float, float, float | None, float | None]] = []  # (scene, raw, fused, gt_betas, gt_cam)
    agg_chromm: dict[str, list[float]] = defaultdict(list)  # "{label}/{metric}" → values

    for scene_dir in scenes:
        scene_name = scene_dir.name
        print(f"\n{'='*65}")
        print(f"Scene: {scene_name}")
        print(f"{'='*65}")

        assert (scene_dir / "vggt_depth.npz").exists(), f"Missing vggt_depth.npz in {scene_dir}"

        placer = BodyPlacer(scene_dir, SMPLX_MODEL)
        cam_dirs = placer._cam_dirs

        if CAMERAS_ONLY:
            _scale_cameras = 1.0  # default: show unscaled VGGT translations
            if FUSION_MODEL is not None:
                fwd_c = _run_fusion_fwd(cam_dirs, FUSION_MODEL, FUSION_DEVICE)
                if fwd_c is not None:
                    _fa, _mp, _fs = fwd_c
                    _pts = {pid: _fa[:, i] for i, pid in enumerate(_mp)}
                    _sam3d_b = load_pred_betas(cam_dirs)
                    _bm = {
                        cam_dir / "body_data" / f"person_{pid}.npz": _sam3d_b[pid]
                        for pid in _mp
                        for cam_dir in placer._cam_dirs
                        if pid in _sam3d_b
                        and (cam_dir / "body_data" / f"person_{pid}.npz").exists()
                    }
                    _scale_c = placer.estimate_scale_triangulated(
                        fused_betas_map=_bm, fused_pose_by_pid=_pts, frame_start=_fs,
                    )
                    _scale_cameras = float(np.median(_scale_c))
                else:
                    print("  [fusion] forward pass failed — using scale=1.0")
            else:
                print("  [cameras_only] no fusion model — translations shown at raw VGGT scale")
            eval_cameras(placer, _scale_cameras, scene_name, cam_dirs)
            continue

        gt_trans_raw = load_gt_trans(scene_name)
        gt_betas_map = load_gt_betas(scene_name)
        assert gt_trans_raw, f"No GT translations found for scene {scene_name}"
        assert gt_betas_map, f"No GT betas found for scene {scene_name}"
        gt_body_data_chromm = _load_gt_body_data_chromm(scene_name, RICH_ROOT, split="train")

        gt_trans = correct_gt_pelvis(gt_trans_raw, gt_betas_map, placer)
        K = len(cam_dirs)
        all_pids: set[int] = set()
        pid_cam_count: dict[int, int] = defaultdict(int)
        for cam_dir in cam_dirs:
            for f in (cam_dir / "body_data").glob("person_*.npz"):
                pid = int(f.stem.split("_")[1])
                all_pids.add(pid)
                pid_cam_count[pid] += 1
        # Foreground: visible in ≥ K-1 cameras; background are spurious detections.
        foreground_pids: set[int] = {p for p, c in pid_cam_count.items() if c >= max(1, K - 1)}

        assert all_pids, f"No body_data found under {scene_dir}"

        pred_betas_map_by_pid = load_pred_betas(cam_dirs)
        fused_betas_file_map: dict[Path, np.ndarray] = {
            cam_dir / "body_data" / f"person_{pid}.npz": pred_betas_map_by_pid[pid]
            for cam_dir in cam_dirs
            for pid in all_pids
            if pid in pred_betas_map_by_pid
            and (cam_dir / "body_data" / f"person_{pid}.npz").exists()
        }

        # Optional GT intrinsics (upper-bound check, not used at test time).
        gt_intrinsics = load_gt_intrinsics(scene_name) if USE_GT_INTRINSICS else None
        if USE_GT_INTRINSICS:
            status = f"{len(gt_intrinsics)} cameras" if gt_intrinsics else "NOT FOUND"
            print(f"  GT intrinsics: {status}")

        # ── Fusion model forward pass (must run before scale estimation) ──────
        assert FUSION_MODEL is not None, "A --checkpoint is required (fused pose needed for scale)"
        fused_body_pose_by_ghost: dict[int, dict[int, np.ndarray]] | None = None
        fused_pose_by_pid_arr: dict[int, np.ndarray] = {}
        fwd_frame_start = 0
        fwd = _run_fusion_fwd(cam_dirs, FUSION_MODEL, FUSION_DEVICE)
        if fwd is not None:
            fused_pose_arr, model_pids, fwd_frame_start = fwd
            pid_to_slot = {pid: i for i, pid in enumerate(model_pids)}
            T_fused = fused_pose_arr.shape[0]
            fused_body_pose_by_ghost = {}
            for ghost_pid in all_pids:
                if ghost_pid not in pid_to_slot:
                    continue
                slot = pid_to_slot[ghost_pid]
                fused_pose_by_pid_arr[ghost_pid] = fused_pose_arr[:, slot]  # (T, 54, 6)
                frames_dict: dict[int, np.ndarray] = {}
                for t_local in range(T_fused):
                    global_t = fwd_frame_start + t_local
                    aa = _6d_to_aa(fused_pose_arr[t_local, slot, :21])
                    frames_dict[global_t] = aa.reshape(63).astype(np.float32)
                fused_body_pose_by_ghost[ghost_pid] = frames_dict
        else:
            print("  [fusion] forward pass failed — skipping scene")
            continue

        # ── Scale estimation: MapAnything preferred, triangulated as fallback ──
        scale = placer.load_mapanything_scale()
        if scale is not None:
            print(f"  [scale] using MapAnything  median={float(np.median(scale)):.4f}")
        else:
            try:
                scale = placer.estimate_scale_triangulated(
                    fused_betas_map=fused_betas_file_map,
                    fused_pose_by_pid=fused_pose_by_pid_arr,
                    frame_start=fwd_frame_start,
                )  # (T,) per-frame scale
                print(f"  [scale] using triangulated  median={float(np.median(scale)):.4f}")
            except RuntimeError as e:
                print(f"  [scale] failed: {e} — skipping scene")
                continue
        scale_median = float(np.median(scale))  # scalar for display / eval_cameras

        # GT scale from camera baseline ratios (isolates scale error).
        gt_exts_list = load_gt_extrinsics(scene_name)
        gt_intr_list = load_gt_intrinsics(scene_name)
        gt_scale = compute_gt_scale(placer, gt_exts_list, cam_dirs) if gt_exts_list else None

        gt_scale_str = f"{gt_scale:.4f}" if gt_scale is not None else "N/A"

        # ── World-to-VGGT-reference transform ────────────────────────────────
        # VGGT always uses the first cam_dir as its reference (identity extrinsic).
        # GT data (transl, global_orient) is in RICH world = cam_00 frame.
        # When cam_00 is absent (e.g. only cam_02/03/04 available), we must
        # re-express GT into the first available camera's frame — same logic as
        # RICHFusionDatapoint._transform_to_world_frame().
        _m_ref = re.search(r"\d+", cam_dirs[0].name) if cam_dirs else None
        _ref_gt_idx = int(_m_ref.group()) if _m_ref else 0
        if gt_exts_list and _ref_gt_idx < len(gt_exts_list):
            E_ref = gt_exts_list[_ref_gt_idx].astype(np.float64)  # camera-from-RICH-world
            R_w2ref = E_ref[:3, :3]   # rotation: RICH world → VGGT ref
            t_w2ref = E_ref[:3, 3]    # translation
        else:
            R_w2ref = np.eye(3, dtype=np.float64)
            t_w2ref = np.zeros(3, dtype=np.float64)
        _is_cam00 = np.allclose(R_w2ref, np.eye(3)) and np.allclose(t_w2ref, 0.0)
        if not _is_cam00:
            print(f"  NOTE: first cam is {cam_dirs[0].name} — re-expressing GT in its frame")

        # Transform GT pelvis positions to VGGT reference frame.
        # correct_gt_pelvis returns positions in RICH world; apply R_w2ref, t_w2ref.
        if not _is_cam00:
            gt_trans = {
                gt_pid: {
                    frame: (R_w2ref @ p.astype(np.float64) + t_w2ref).astype(np.float32)
                    for frame, p in frames.items()
                }
                for gt_pid, frames in gt_trans.items()
            }

        # ── GT camera list for oracle DLT (K scaled to resized image space) ──
        # camera_names[k] = b'cam_02' etc. — extract integer to index GT XMLs,
        # then re-express extrinsic relative to the VGGT reference frame so the
        # DLT result is in the same frame as all other predictions.
        gt_cameras_oracle: list[tuple[np.ndarray, np.ndarray] | None] = []
        gt_transl_cameras_list: list[np.ndarray | None] = []
        for k, cam_dir in enumerate(cam_dirs):
            _m = re.search(r"\d+", cam_dir.name)
            gt_idx = int(_m.group()) if _m else k
            if (gt_exts_list and gt_idx < len(gt_exts_list)
                    and gt_intr_list and gt_idx < len(gt_intr_list)):
                valid_frames = np.where(placer.cam_valid[:, k])[0]
                if len(valid_frames) > 0:
                    os_ = placer.original_size[valid_frames[0], k]
                    W_orig, H_orig = float(os_[0]), float(os_[1])
                    K_full = gt_intr_list[gt_idx].astype(np.float64)
                    sx, sy = W_orig / RICH_ORIG_W, H_orig / RICH_ORIG_H
                    K_r = K_full.copy()
                    K_r[0, 0] *= sx; K_r[0, 2] *= sx
                    K_r[1, 1] *= sy; K_r[1, 2] *= sy
                    # Re-express extrinsic relative to VGGT reference frame:
                    # E_i_rel = [R_i @ R_ref.T | t_i - (R_i @ R_ref.T) @ t_ref]
                    E_i = gt_exts_list[gt_idx].astype(np.float64)
                    R_i_rel = E_i[:3, :3] @ R_w2ref.T
                    t_i_rel = E_i[:3, 3] - R_i_rel @ t_w2ref
                    E_i_rel = np.concatenate([R_i_rel, t_i_rel[:, np.newaxis]], axis=1)
                    gt_cameras_oracle.append((K_r, E_i_rel))
                    gt_transl_cameras_list.append(t_i_rel.astype(np.float32))
                else:
                    gt_cameras_oracle.append(None)
                    gt_transl_cameras_list.append(None)
            else:
                gt_cameras_oracle.append(None)
                gt_transl_cameras_list.append(None)
        has_gt_cams = any(c is not None for c in gt_cameras_oracle)
        has_gt_transl = any(t is not None for t in gt_transl_cameras_list)

        # ── GT body_pose map keyed by ghost pid ───────────────────────────────
        gt_body_pose_raw = load_gt_body_pose(scene_name)
        gt_pids_list = sorted(gt_body_pose_raw.keys())
        gt_body_pose_by_ghost: dict[int, dict[int, np.ndarray]] = {}
        if gt_body_pose_raw:
            for ghost_pid in sorted(all_pids):
                matched_gt = min(gt_pids_list, key=lambda p: abs(p - ghost_pid))
                gt_body_pose_by_ghost[ghost_pid] = gt_body_pose_raw[matched_gt]
        has_gt_pose = bool(gt_body_pose_by_ghost)

        # ── GT betas map keyed by ghost pid ──────────────────────────────────
        gt_betas_by_ghost: dict[int, np.ndarray] = {}
        if gt_betas_map and gt_pids_list:
            for ghost_pid in sorted(all_pids):
                matched_gt = min(gt_pids_list, key=lambda p: abs(p - ghost_pid))
                gt_betas_by_ghost[ghost_pid] = gt_betas_map[matched_gt]

        # Scale with GT betas + fused pose (oracle upper bound for bone-length scale).
        scale_gt_betas: np.ndarray | None = None
        scale_gt_betas_median: float | None = None
        if gt_betas_by_ghost:
            _gt_b_file_map: dict[Path, np.ndarray] = {
                _cam_dir / "body_data" / f"person_{_ghost_pid}.npz": gt_betas_by_ghost[_ghost_pid]
                for _cam_dir in cam_dirs
                for _ghost_pid in all_pids
                if (_cam_dir / "body_data" / f"person_{_ghost_pid}.npz").exists()
                and _ghost_pid in gt_betas_by_ghost
            }
            try:
                scale_gt_betas = placer.estimate_scale_triangulated(
                    fused_betas_map=_gt_b_file_map,
                    fused_pose_by_pid=fused_pose_by_pid_arr,
                    frame_start=fwd_frame_start,
                )  # (T,) per-frame
                scale_gt_betas_median = float(np.median(scale_gt_betas))
            except RuntimeError:
                pass

        # Record scale variants (use medians for display).
        if gt_scale is not None:
            agg_scale.append((scene_name, scale_median, scale_gt_betas_median, gt_scale))

        # Helper: run eval + accumulate cross-scene errors.
        # Returns the ghost→gt pid match so orient eval can reuse it.
        def _run(label: str, preds: dict, ref: dict) -> dict[int, int]:
            errs, vecs, pm = eval_translations(preds, ref, foreground_pids)
            _print_results(label, errs, vecs)
            for errs_list in errs.values():
                agg_trans[label].extend(errs_list)
            return pm

        def _run_orient(label: str, orient_map: dict[int, np.ndarray],
                        gt_or: dict, pid_match: dict[int, int]) -> None:
            fmt = {pid: _to_orient_fmt(frs) for pid, frs in orient_map.items()
                   if _to_orient_fmt(frs) is not None}
            print(f"  [{label}]")
            errs_all: list[float] = []
            for pid, gt_pid in pid_match.items():
                if pid not in fmt:
                    continue
                fi_p, Rs_p = fmt[pid]
                gt_fr = gt_or.get(gt_pid, {})
                errs = [_geodesic_deg(Rs_p[i], gt_fr[int(fi_p[i])])
                        for i in range(len(fi_p)) if int(fi_p[i]) in gt_fr]
                if errs:
                    e = np.array(errs)
                    errs_all.extend(errs)
                    print(f"    pid {pid}: N={len(e):4d}  median={np.median(e):.1f}°  "
                          f"mean={np.mean(e):.1f}°  <30°={(e<30).mean()*100:5.1f}%")
            if errs_all:
                e = np.array(errs_all)
                agg_orient[label].extend(errs_all)
                print(f"    TOTAL: N={len(e):4d}  median={np.median(e):.1f}°  "
                      f"mean={np.mean(e):.1f}°  <30°={(e<30).mean()*100:5.1f}%")
            print()

        # ── Procrustes DLT all-cams, pred scale ──────────────────────────────
        preds_dlt, ors_dlt = predict_translations_procrustes(
            placer, scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
            gt_body_pose_map=fused_body_pose_by_ghost,
        )
        pm_dlt = _run("Proc DLT | pred scale | pred pose", preds_dlt, gt_trans)
        eval_chromm_metrics(
            "Proc DLT | pred scale | pred pose",
            placer, preds_dlt, ors_dlt, pm_dlt, gt_body_data_chromm,
            fused_pose_arr, pid_to_slot, pred_betas_map_by_pid,
            fwd_frame_start, agg_chromm,
        )

        pm_dlt_pgb = pm_dlt
        if gt_betas_by_ghost:
            _scale_pgb = scale_gt_betas if scale_gt_betas is not None else scale  # (T,) array
            preds_dlt_pgb, ors_dlt_pgb = predict_translations_procrustes(
                placer, _scale_pgb, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                gt_body_pose_map=fused_body_pose_by_ghost,
                gt_betas_by_pid=gt_betas_by_ghost,
            )
            pm_dlt_pgb = _run("Proc DLT | pred scale | pred pose | GT betas", preds_dlt_pgb, gt_trans)

        pm_dlt_gp = pm_dlt
        if has_gt_pose:
            preds_dlt_gp, ors_dlt_gp = predict_translations_procrustes(
                placer, scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                gt_body_pose_map=gt_body_pose_by_ghost,
                gt_betas_by_pid=gt_betas_by_ghost,
            )
            pm_dlt_gp = _run("Proc DLT | pred scale | GT pose+betas", preds_dlt_gp, gt_trans)

        # ── Procrustes DLT all-cams, GT scale ────────────────────────────────
        pm_dlt_gs = pm_dlt
        pm_dlt_gs_pgb = pm_dlt_pgb
        pm_dlt_gs_gp = pm_dlt_gp
        if gt_scale is not None:
            preds_dlt_gs, ors_dlt_gs = predict_translations_procrustes(
                placer, gt_scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                gt_body_pose_map=fused_body_pose_by_ghost,
            )
            pm_dlt_gs = _run("Proc DLT | GT scale  | pred pose", preds_dlt_gs, gt_trans)
            eval_chromm_metrics(
                "Proc DLT | GT scale  | pred pose",
                placer, preds_dlt_gs, ors_dlt_gs, pm_dlt_gs, gt_body_data_chromm,
                fused_pose_arr, pid_to_slot, pred_betas_map_by_pid,
                fwd_frame_start, agg_chromm,
            )

            if gt_betas_by_ghost:
                preds_dlt_gs_pgb, ors_dlt_gs_pgb = predict_translations_procrustes(
                    placer, gt_scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                    gt_body_pose_map=fused_body_pose_by_ghost,
                    gt_betas_by_pid=gt_betas_by_ghost,
                )
                pm_dlt_gs_pgb = _run("Proc DLT | GT scale  | pred pose | GT betas",  preds_dlt_gs_pgb, gt_trans)

            if has_gt_pose:
                preds_dlt_gs_gp, ors_dlt_gs_gp = predict_translations_procrustes(
                    placer, gt_scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                    gt_body_pose_map=gt_body_pose_by_ghost,
                    gt_betas_by_pid=gt_betas_by_ghost,
                )
                pm_dlt_gs_gp = _run("Proc DLT | GT scale  | GT pose+betas", preds_dlt_gs_gp, gt_trans)

        # ── Hybrid: VGGT rotation + GT translation (no scale needed) ─────────────
        pm_dlt_gt_ps = pm_dlt
        if has_gt_transl:
            preds_dlt_gt_ps, ors_dlt_gt_ps = predict_translations_procrustes(
                placer, scale, cam_dirs, all_pids, pred_betas_map_by_pid, gt_intrinsics,
                gt_transl_cameras=gt_transl_cameras_list,
                gt_body_pose_map=fused_body_pose_by_ghost,
            )
            pm_dlt_gt_ps = _run("Proc DLT | GT transl | pred pose", preds_dlt_gt_ps, gt_trans)
            eval_chromm_metrics(
                "Proc DLT | GT transl | pred pose",
                placer, preds_dlt_gt_ps, ors_dlt_gt_ps, pm_dlt_gt_ps, gt_body_data_chromm,
                fused_pose_arr, pid_to_slot, pred_betas_map_by_pid,
                fwd_frame_start, agg_chromm,
            )

        # ── Oracle: GT cameras ────────────────────────────────────────────────
        pm_ora_pp_ps = pm_dlt
        pm_ora_pp_gs = pm_dlt_gs
        pm_ora_pgb   = pm_dlt_pgb
        pm_ora_gp    = pm_dlt_gp
        if has_gt_cams:
            preds_ora_pp_ps, ors_ora_pp_ps = predict_translations_procrustes(
                placer, scale, cam_dirs, all_pids, pred_betas_map_by_pid,
                gt_intrinsics=None,
                gt_cameras=gt_cameras_oracle,
                gt_body_pose_map=fused_body_pose_by_ghost,
            )
            pm_ora_pp_ps = _run("GT cams | pred pose | pred scale", preds_ora_pp_ps, gt_trans)
            eval_chromm_metrics(
                "GT cams | pred pose | pred scale",
                placer, preds_ora_pp_ps, ors_ora_pp_ps, pm_ora_pp_ps, gt_body_data_chromm,
                fused_pose_arr, pid_to_slot, pred_betas_map_by_pid,
                fwd_frame_start, agg_chromm,
            )

            if gt_betas_by_ghost:
                preds_ora_pgb, ors_ora_pgb = predict_translations_procrustes(
                    placer, _scale_pgb, cam_dirs, all_pids, pred_betas_map_by_pid,
                    gt_intrinsics=None,
                    gt_cameras=gt_cameras_oracle,
                    gt_body_pose_map=fused_body_pose_by_ghost,
                    gt_betas_by_pid=gt_betas_by_ghost,
                )
                pm_ora_pgb = _run("GT cams | pred pose | pred scale | GT betas", preds_ora_pgb, gt_trans)

            if gt_scale is not None:
                preds_ora_pp_gs, ors_ora_pp_gs = predict_translations_procrustes(
                    placer, gt_scale, cam_dirs, all_pids, pred_betas_map_by_pid,
                    gt_intrinsics=None,
                    gt_cameras=gt_cameras_oracle,
                    gt_body_pose_map=fused_body_pose_by_ghost,
                )
                pm_ora_pp_gs = _run("GT cams | pred pose | GT scale", preds_ora_pp_gs, gt_trans)

            if has_gt_pose:
                preds_ora_gp, ors_ora_gp = predict_translations_procrustes(
                    placer, scale, cam_dirs, all_pids, pred_betas_map_by_pid,
                    gt_intrinsics=None,
                    gt_body_pose_map=gt_body_pose_by_ghost,
                    gt_cameras=gt_cameras_oracle,
                    gt_betas_by_pid=gt_betas_by_ghost,
                )
                pm_ora_gp = _run("GT cams | GT pose+betas | pred scale", preds_ora_gp, gt_trans)

                if gt_scale is not None:
                    preds_ora_gp_gs, ors_ora_gp_gs = predict_translations_procrustes(
                        placer, gt_scale, cam_dirs, all_pids, pred_betas_map_by_pid,
                        gt_intrinsics=None,
                        gt_body_pose_map=gt_body_pose_by_ghost,
                        gt_cameras=gt_cameras_oracle,
                        gt_betas_by_pid=gt_betas_by_ghost,
                    )
                    pm_ora_gp_gs = _run("GT cams | GT pose+betas | GT scale", preds_ora_gp_gs, gt_trans)

        # ── Global orientation ────────────────────────────────────────────────
        gt_orient = load_gt_global_orient(scene_name)
        # Re-express GT rotations in VGGT reference frame: R_ref = R_w2ref @ R_gt
        if gt_orient and not _is_cam00:
            gt_orient = {
                gt_pid: {
                    frame: (R_w2ref @ R.astype(np.float64)).astype(np.float32)
                    for frame, R in frames.items()
                }
                for gt_pid, frames in gt_orient.items()
            }
        if gt_orient:
            _run_orient("Orient — Proc DLT | pred scale | pred pose", ors_dlt,       gt_orient, pm_dlt)
            if gt_betas_by_ghost:
                _run_orient("Orient — Proc DLT | pred scale | pred pose | GT betas", ors_dlt_pgb, gt_orient, pm_dlt_pgb)
            if has_gt_pose:
                _run_orient("Orient — Proc DLT | pred scale | GT pose+betas",  ors_dlt_gp,    gt_orient, pm_dlt_gp)
            if gt_scale is not None:
                _run_orient("Orient — Proc DLT | GT scale  | pred pose", ors_dlt_gs,   gt_orient, pm_dlt_gs)
                if gt_betas_by_ghost:
                    _run_orient("Orient — Proc DLT | GT scale  | pred pose | GT betas", ors_dlt_gs_pgb, gt_orient, pm_dlt_gs_pgb)
                if has_gt_pose:
                    _run_orient("Orient — Proc DLT | GT scale  | GT pose+betas", ors_dlt_gs_gp, gt_orient, pm_dlt_gs_gp)
            if has_gt_transl:
                _run_orient("Orient — Proc DLT | GT transl | pred pose", ors_dlt_gt_ps, gt_orient, pm_dlt_gt_ps)
            if has_gt_cams:
                _run_orient("Orient — GT cams | pred pose | pred scale", ors_ora_pp_ps, gt_orient, pm_ora_pp_ps)
                if gt_betas_by_ghost:
                    _run_orient("Orient — GT cams | pred pose | pred scale | GT betas", ors_ora_pgb, gt_orient, pm_ora_pgb)
                if gt_scale is not None:
                    _run_orient("Orient — GT cams | pred pose | GT scale", ors_ora_pp_gs, gt_orient, pm_ora_pp_gs)
                if has_gt_pose:
                    _run_orient("Orient — GT cams | GT pose+betas | pred scale", ors_ora_gp,    gt_orient, pm_ora_gp)
                    if gt_scale is not None:
                        _run_orient("Orient — GT cams | GT pose+betas | GT scale",  ors_ora_gp_gs, gt_orient, pm_ora_gp_gs)
        else:
            print("  [Global orient] no GT data\n")

        # ── Bone length error (pred betas vs GT betas) ───────────────────────
        eval_bone_lengths(placer, pm_dlt, gt_betas_map, pred_betas_map_by_pid)

        # ── Per-camera diagnostics ───────────────────────────────────────────
        eval_cameras(placer, scale_median, scene_name, cam_dirs)

    # ── Aggregate summary across all scenes ──────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AGGREGATE over {len(scenes)} scenes")
    print(f"{'='*65}")
    print(f"  {'Method':<48}  {'N':>5}  {'median':>7}  {'mean':>7}  {'<0.5m':>6}")
    print(f"  {'-'*48}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}")
    for label, errs in agg_trans.items():
        if not errs:
            continue
        e = np.array(errs)
        print(f"  {label:<48}  {len(e):>5}  {np.median(e):>7.3f}m  {np.mean(e):>7.3f}m  "
              f"{(e<0.5).mean()*100:>5.1f}%")
    if agg_orient:
        print(f"\n  {'Method':<48}  {'N':>5}  {'median':>7}  {'<30°':>6}")
        print(f"  {'-'*48}  {'-'*5}  {'-'*7}  {'-'*6}")
        for label, errs in agg_orient.items():
            if not errs:
                continue
            e = np.array(errs)
            print(f"  {label:<48}  {len(e):>5}  {np.median(e):>6.1f}°  "
                  f"{(e<30).mean()*100:>5.1f}%")
    if agg_scale:
        def _epct(pred, gt): return (pred - gt) / gt * 100.0 if pred is not None else float("nan")
        print(f"\n  {'Scene':<40}  {'fused':>7}  {'gt_b':>6}  {'gt_cam':>7}  "
              f"{'err_fused':>9}  {'err_gtb':>8}")
        print(f"  {'-'*40}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*8}")
        fused_errs, gtb_errs = [], []
        for scene, s_fused, s_gtb, s_gt in agg_scale:
            ef = _epct(s_fused, s_gt)
            eg = _epct(s_gtb,   s_gt)
            if not np.isnan(ef): fused_errs.append(ef)
            if not np.isnan(eg): gtb_errs.append(eg)
            sg  = f"{s_gtb:.4f}"   if s_gtb   is not None else "  N/A"
            eef = f"{ef:+.1f}%"    if not np.isnan(ef)   else "  N/A  "
            eeg = f"{eg:+.1f}%"    if not np.isnan(eg)   else "  N/A  "
            print(f"  {scene:<40}  {s_fused:>7.4f}  {sg:>6}  {s_gt:>7.4f}  "
                  f"{eef:>9}  {eeg:>8}")
        if fused_errs:
            fa = np.array(fused_errs)
            print(f"\n  AGGREGATE fused : median={np.nanmedian(fa):+.1f}%  mean={np.nanmean(fa):+.1f}%")
        if gtb_errs:
            ga = np.array(gtb_errs)
            print(f"  AGGREGATE gt_b  : median={np.nanmedian(ga):+.1f}%  mean={np.nanmean(ga):+.1f}%")

    if agg_chromm:
        # Collect unique labels (preserve insertion order via dict key order)
        _seen: dict[str, None] = {}
        for k in agg_chromm:
            label = k.rsplit("/", 1)[0]
            _seen[label] = None
        chromm_labels = list(_seen)
        metrics = ["wa", "w", "ga", "pa", "rte"]
        print(f"\n  CHROMM metrics (mm / %, aggregate means across scenes)")
        header = f"  {'Method':<48}  {'WA':>7}  {'W':>7}  {'GA':>7}  {'PA':>7}  {'RTE':>7}"
        print(header)
        print(f"  {'-'*48}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for label in chromm_labels:
            row_vals = []
            for m in metrics:
                vals = agg_chromm.get(f"{label}/{m}", [])
                row_vals.append(np.mean(vals) if vals else float("nan"))
            wa, w, ga, pa, rte = row_vals
            unit = lambda v, is_rte: f"{v:.2f}%" if is_rte else f"{v:.1f}mm"
            print(f"  {label:<48}  {unit(wa,False):>7}  {unit(w,False):>7}  "
                  f"{unit(ga,False):>7}  {unit(pa,False):>7}  {unit(rte,True):>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: evaluate BodyPlacer scale + translation with pred vs GT betas/pose."
    )
    parser.add_argument("--scene_root",  required=True, type=Path,
                        help="Scene dir (or root of many scenes).")
    parser.add_argument("--rich_root",   required=True, type=Path,
                        help="RICH dataset root (must have train_body/<scene_name>/).")
    parser.add_argument("--smplx_model", required=True, type=Path,
                        help="Path to SMPLX_NEUTRAL.pkl.")
    parser.add_argument("--use_gt_intrinsics", action="store_true",
                        help="Override VGGT cx/cy with RICH GT calibration intrinsics.")
    parser.add_argument("--cameras_only", action="store_true",
                        help="Skip body evaluation; only print per-camera rotation/translation/intrinsics diagnostics.")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit evaluation to the first N scenes.")
    parser.add_argument("--rich_orig_w", type=int, default=4112,
                        help="Full-res image width that RICH scan_calibration XMLs were made for (default: 4112).")
    parser.add_argument("--rich_orig_h", type=int, default=3008,
                        help="Full-res image height that RICH scan_calibration XMLs were made for (default: 3008).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to fusion model checkpoint (.pt). When provided, fused pose "
                             "replaces raw SAM3D body_pose for all 'pred pose' calls.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for fusion model inference (default: cpu).")
    args = parser.parse_args()
    SCENE_ROOT        = args.scene_root
    RICH_ROOT         = args.rich_root
    SMPLX_MODEL       = args.smplx_model
    USE_GT_INTRINSICS = args.use_gt_intrinsics
    CAMERAS_ONLY      = args.cameras_only
    MAX_SCENES        = args.max_scenes
    RICH_ORIG_W       = args.rich_orig_w  # noqa: F841
    RICH_ORIG_H       = args.rich_orig_h
    if args.checkpoint is not None:
        import torch
        FUSION_DEVICE = torch.device(args.device)
        FUSION_MODEL  = _load_fusion_model(args.checkpoint, FUSION_DEVICE)
    run()
