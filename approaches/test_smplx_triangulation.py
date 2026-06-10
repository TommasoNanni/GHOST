"""Test SMPL-X joint triangulation for body placement.

Instead of triangulating noisy MHR70 2D keypoints, this script:
  1. For each camera k: runs SMPL-X FK with SAM3D's per-camera estimate
     (body_pose_k, betas_k, global_orient_k, smplx_transl_k) → 3D joints in camera space.
  2. Projects those 3D joints to 2D using SAM3D's own focal_length
     (consistent: projection uses the same camera model SAM3D fitted against).
  3. Weighted DLT triangulates the 2D observations across cameras,
     using pred_joint_confidence (T, 55) to downweight uncertain joints.
  4. Procrustes fits the triangulated world joints → global_orient + pelvis_world.
  5. Prints frame-by-frame GT vs predicted translation and orientation.

Usage:
    pixi run python approaches/test_smplx_triangulation.py \\
        --scene_dir /path/to/ghost_outputs/BBQ_001_guitar \\
        --rich_root  /path/to/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        [--split train_body] [--max_frames 20] [--min_conf 0.3]
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
import torch
from scipy.spatial.transform import Rotation as SciR
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models


# ---------------------------------------------------------------------------
# SMPL-X kinematic tree for joints 0-21 (child → parent index)
# ---------------------------------------------------------------------------

_SMPLX_PARENTS_21: dict[int, int] = {
    1: 0,  2: 0,  3: 0,        # L_hip, R_hip, spine1  ← pelvis
    4: 1,  5: 2,  6: 3,        # L_knee, R_knee, spine2
    7: 4,  8: 5,  9: 6,        # L_ankle, R_ankle, spine3
    10: 7, 11: 8, 12: 9,       # L_foot, R_foot, neck
    13: 12, 14: 12, 15: 12,    # L_collar, R_collar, head
    16: 13, 17: 14,             # L_shoulder, R_shoulder
    18: 16, 19: 17,             # L_elbow, R_elbow
    20: 18, 21: 19,             # L_wrist, R_wrist
}

_SMPLX_JOINT_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1",
    "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3",
    "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow",
    "L_wrist", "R_wrist",
]


# ---------------------------------------------------------------------------
# 1. load_cam_body_data
# ---------------------------------------------------------------------------

def load_cam_body_data(cam_dir: Path, pid: int) -> dict | None:
    """Load all per-camera SAM3D body data for one person.

    Reads body_data/person_{pid}.npz and returns a dict keyed by global frame
    index with the per-frame SMPL-X parameters and metadata.

    Keys returned per frame:
      body_pose      (63,)  — SMPL-X body pose axis-angle
      global_orient  (3,)   — global orientation axis-angle in camera frame
      betas          (10,)  — shape coefficients
      smplx_transl   (3,)   — SMPL-X translation in camera space (metres)
      focal_length   float  — SAM3D's focal length (pixels) for this frame
      confidence     (55,)  — per-joint confidence for 55 SMPL-X joints
    """
    path = cam_dir / "body_data" / f"person_{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    required = {"smplx_body_pose", "smplx_global_orient", "smplx_betas",
                "smplx_transl", "focal_length", "pred_joint_confidence", "frame_indices"}
    if not required.issubset(d.files):
        return None

    frames = d["frame_indices"].astype(int)
    result: dict[int, dict] = {}
    for local_t, global_t in enumerate(frames):
        result[int(global_t)] = {
            "body_pose":     d["smplx_body_pose"][local_t],        # (63,)
            "global_orient": d["smplx_global_orient"][local_t],    # (3,)
            "betas":         d["smplx_betas"][local_t],            # (10,)
            "smplx_transl":  d["smplx_transl"][local_t],           # (3,)
            "focal_length":  float(d["focal_length"][local_t]),
            "confidence":    d["pred_joint_confidence"][local_t],  # (55,)
        }
    return result


# ---------------------------------------------------------------------------
# 2. build_sam3d_K
# ---------------------------------------------------------------------------

def build_sam3d_K(focal_length: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3×3 intrinsic matrix from SAM3D's camera model.

    SAM3D fits the body using a pinhole model with a single focal length
    (fx = fy = focal_length) and principal point at image centre (cx, cy).
    This K is used both to project FK joints to 2D and to build the DLT
    projection matrices, ensuring full consistency.

    Args:
        focal_length: SAM3D's focal length in pixels (stored per frame in body_data).
        cx, cy:       Principal point in pixels — taken from VGGT intrinsics
                      (≈ image_W/2, image_H/2).
    Returns:
        K  (3, 3) float64.
    """
    return np.array([
        [focal_length, 0.0,          cx],
        [0.0,          focal_length, cy],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. fk_camera_space
# ---------------------------------------------------------------------------

def fk_camera_space(
    placer:        BodyPlacer,
    body_pose:     np.ndarray,   # (63,)
    global_orient: np.ndarray,   # (3,)
    betas:         np.ndarray,   # (10,)
    smplx_transl:  np.ndarray,   # (3,)
) -> np.ndarray:
    """Run SMPL-X FK with SAM3D's per-camera estimate.

    Feeds all four parameters — body_pose, global_orient, betas, smplx_transl —
    directly into the SMPL-X model. global_orient and smplx_transl are in the
    camera's coordinate frame (as SAM3D estimated them), so the returned joints
    are also in camera space (in metres, same scale as smplx_transl).

    Unlike the Procrustes step in placer.py (which uses zero global_orient and
    zero transl), here we use SAM3D's actual per-camera estimates so that the
    projected joints reflect what SAM3D "saw" in each camera.

    Returns:
        J_cam  (55, 3) float32 — SMPL-X joint positions in camera frame.
    """
    fk = placer._smplx_fk(
        betas[np.newaxis],
        body_pose[np.newaxis],
        global_orient[np.newaxis],
    )  # (1, 55, 3) — joints with global_orient applied, zero transl

    # Add smplx_transl to shift joints from body origin to camera frame.
    # In SMPL-X: J_world[j] = R_global @ (J_can[j] - J_can[0]) + J_can[0] + transl
    # _smplx_fk already applies global_orient; transl is additive.
    J_cam = fk[0] + smplx_transl.astype(np.float32)   # (55, 3)
    return J_cam


# ---------------------------------------------------------------------------
# 3b. precompute_cam_fk
# ---------------------------------------------------------------------------

def precompute_cam_fk(
    placer:    BodyPlacer,
    cam_data:  dict[int, dict],   # frame → per-frame data for one (cam, pid)
) -> dict[int, np.ndarray]:
    """Batch FK across all frames for one (camera, pid) pair.

    Instead of calling _smplx_fk once per frame (T separate calls), this
    stacks all frames into a single batch and calls _smplx_fk once.
    Returns {global_frame: J_cam (55, 3)} where J_cam includes smplx_transl.

    This is the main fix for the per-frame FK OOM: reduces 3056 torch calls
    (382 frames × 8 cameras) to 8 batched calls.
    """
    if not cam_data:
        return {}

    frames = sorted(cam_data.keys())
    N = len(frames)

    body_poses     = np.stack([cam_data[f]["body_pose"]     for f in frames])  # (N, 63)
    global_orients = np.stack([cam_data[f]["global_orient"] for f in frames])  # (N,  3)
    betas_arr      = np.stack([cam_data[f]["betas"]         for f in frames])  # (N, 10)
    translsarr     = np.stack([cam_data[f]["smplx_transl"]  for f in frames])  # (N,  3)

    # Batched FK: (N, 55, 3) with global_orient applied, zero transl
    J_batch = placer._smplx_fk(betas_arr, body_poses, global_orients)   # (N, 55, 3)

    # Add smplx_transl to place joints in camera space
    J_batch = J_batch + translsarr[:, np.newaxis, :]   # (N, 55, 3)

    return {f: J_batch[i] for i, f in enumerate(frames)}


# ---------------------------------------------------------------------------
# 4. project_joints_2d
# ---------------------------------------------------------------------------

def project_joints_2d(J_cam: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Project 3D camera-space joints to 2D pixel coordinates.

    Applies standard perspective projection: (u, v) = K @ x_cam / x_cam[2].
    Joints behind the camera (z ≤ 0) are returned as NaN.

    Args:
        J_cam  (55, 3) — joints in camera space.
        K      (3, 3)  — SAM3D intrinsic matrix from build_sam3d_K.
    Returns:
        pts2d  (55, 2) float64 — pixel coordinates (u, v), NaN for z ≤ 0.
    """
    pts2d = np.full((J_cam.shape[0], 2), np.nan, dtype=np.float64)
    for j, x in enumerate(J_cam.astype(np.float64)):
        if x[2] > 1e-4:
            uv = K @ x
            pts2d[j] = uv[:2] / uv[2]
    return pts2d


# ---------------------------------------------------------------------------
# 5. weighted_dlt_single_joint
# ---------------------------------------------------------------------------

def weighted_dlt_single_joint(
    pts2d_per_cam: list[np.ndarray],   # list of (2,) — one per camera
    P_mats:        list[np.ndarray],   # list of (3, 4)
    confs:         list[float],        # confidence per camera for this joint
    min_conf:      float = 0.3,
) -> np.ndarray | None:
    """Triangulate one joint from multi-camera 2D observations using weighted DLT.

    For each camera k with confidence c_k >= min_conf, the standard DLT builds
    two linear equations from (u_k, v_k) and P_k:
        (p1_k - u_k * p3_k) @ X = 0
        (p2_k - v_k * p3_k) @ X = 0
    Rows are scaled by sqrt(c_k), giving higher-confidence cameras more weight
    in the least-squares solution. The system A @ X = 0 is solved via SVD;
    the solution is the right singular vector corresponding to the smallest
    singular value.

    Args:
        pts2d_per_cam: 2D pixel coordinates per camera.
        P_mats:        Projection matrices P = K_sam3d @ [R | t*scale].
        confs:         Per-camera confidence for this joint (0–1).
        min_conf:      Minimum confidence to include a camera.
    Returns:
        X  (3,) float32 world-space 3D position, or None if < 2 valid cameras.
    """
    rows = []
    for (u, v), P, c in zip(pts2d_per_cam, P_mats, confs):
        if c < min_conf or np.isnan(u) or np.isnan(v):
            continue
        w = float(np.sqrt(c))
        rows.append(w * (P[0] - u * P[2]))
        rows.append(w * (P[1] - v * P[2]))

    if len(rows) < 4:   # need at least 2 cameras → 4 rows
        return None

    A = np.stack(rows, axis=0)   # (2K, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]                   # homogeneous solution
    if abs(X[3]) < 1e-10:
        return None
    return (X[:3] / X[3]).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. triangulate_all_joints
# ---------------------------------------------------------------------------

def triangulate_all_joints(
    placer:        BodyPlacer,
    cam_data_all:  list[dict[int, dict]],   # cam_data_all[k][pid][frame] = per-frame data
    scale:         np.ndarray | float,
    pid:           int,
    min_conf:      float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Triangulate all 22 body SMPL-X joints (0-21) for one person across all frames.

    Step 1 — Precompute FK per camera (batched):
      For each camera k that has data for this pid, stack all frames into one
      batch and call _smplx_fk once. Result: {k: {frame: J_cam (55, 3)}}.

    Step 2 — Per-frame DLT:
      For each joint j in 0-21: collect per-camera (u,v) projections and
      confidences, call weighted_dlt_single_joint, record mean confidence.

    Returns:
        {global_frame: (J_world (22, 3), joint_conf (22,))}
        J_world has NaN where triangulation failed; joint_conf is 0 for those joints.
    """
    # ── Step 1: batched FK per camera ─────────────────────────────────────────
    fk_by_cam: dict[int, dict[int, np.ndarray]] = {}
    for k, cam_data in enumerate(tqdm(cam_data_all, desc="  FK per camera", leave=False)):
        if pid not in cam_data:
            continue
        fk_by_cam[k] = precompute_cam_fk(placer, cam_data[pid])

    # Collect all frames this pid appears in.
    all_frames: set[int] = set()
    for frames in fk_by_cam.values():
        all_frames.update(frames.keys())

    # ── Step 2: per-frame DLT ─────────────────────────────────────────────────
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for global_t in tqdm(sorted(all_frames), desc="  DLT frames", leave=False):
        vggt_t = global_t
        if vggt_t < 0 or vggt_t >= placer.T:
            continue

        s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

        # Gather per-camera observations for this frame.
        cam_obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # Each entry: (pts2d (55,2), P (3,4), confidence (55,))

        for k, fk_frames in fk_by_cam.items():
            if not placer.cam_valid[vggt_t, k]:
                continue
            J_cam = fk_frames.get(global_t)
            if J_cam is None:
                continue

            conf = cam_data_all[k][pid][global_t]["confidence"]   # (55,)

            K_vggt = placer.intrinsics[vggt_t, k].astype(np.float64)
            pts2d  = project_joints_2d(J_cam, K_vggt)             # (55, 2)

            E = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
            E[:3, 3] *= s
            P = K_vggt @ E   # (3, 4)

            cam_obs.append((pts2d, P, conf))

        if len(cam_obs) < 2:
            continue

        J_world    = np.full((22, 3), np.nan, dtype=np.float32)
        joint_conf = np.zeros(22, dtype=np.float32)   # mean confidence per joint
        for j in range(22):
            pts_j  = [o[0][j]        for o in cam_obs]
            P_j    = [o[1]           for o in cam_obs]
            conf_j = [float(o[2][j]) for o in cam_obs]
            X = weighted_dlt_single_joint(pts_j, P_j, conf_j, min_conf)
            if X is not None:
                J_world[j] = X
                valid_c = [c for c in conf_j if c >= min_conf]
                joint_conf[j] = float(np.mean(valid_c)) if valid_c else 0.0

        result[global_t] = (J_world, joint_conf)

    return result


# ---------------------------------------------------------------------------
# 7. procrustes_fit
# ---------------------------------------------------------------------------

def procrustes_fit(
    J_world:    np.ndarray,          # (22, 3) triangulated, may have NaN
    J_can:      np.ndarray,          # (55, 3) canonical FK (zero orient, zero transl, pred betas)
    weights:    np.ndarray | None = None,  # (22,) per-joint confidence weights; None = uniform
    min_joints: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Confidence-weighted Kabsch alignment of canonical skeleton to triangulated joints.

    Valid joints (non-NaN in J_world) are weighted by `weights` in the cross-covariance
    matrix, so high-confidence joints drive the rotation more than uncertain ones.
    With weights=None this reduces to the standard unweighted Procrustes.

    Returns:
        (R (3,3), t (3,), pelvis_world (3,)) or None if < min_joints survive.
    """
    n = J_world.shape[0]
    valid = ~np.isnan(J_world[:, 0])
    if valid.sum() < min_joints:
        return None

    A = J_world[valid].astype(np.float64)    # world points
    B = J_can[:n][valid].astype(np.float64)  # canonical points

    # Per-joint weights: clamp to (0, 1], fall back to uniform if not provided.
    if weights is not None:
        w = np.maximum(weights[:n][valid].astype(np.float64), 1e-6)
    else:
        w = np.ones(len(A))
    w /= w.sum()   # normalise so they sum to 1

    # Weighted centroids and cross-covariance.
    A_mean = w @ A                           # (3,)
    B_mean = w @ B
    H = (B - B_mean).T @ (w[:, None] * (A - A_mean))   # (3, 3)

    U, _, Vt = np.linalg.svd(H)
    d_sign = np.linalg.det(Vt.T @ U.T)
    R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
    t = (A_mean - R.astype(np.float64) @ B_mean).astype(np.float32)

    pelvis_world = (R.astype(np.float64) @ J_can[0].astype(np.float64) + t.astype(np.float64)).astype(np.float32)
    return R, t, pelvis_world


# ---------------------------------------------------------------------------
# 7b. anatomical_bone_guard
# ---------------------------------------------------------------------------

def anatomical_bone_guard(
    J_world:   np.ndarray,   # (22, 3) — triangulated, may contain NaN
    J_can:     np.ndarray,   # (55, 3) — canonical FK (zero orient, pred betas)
    abs_floor: float = 0.03, # minimum absolute tolerance per bone (metres)
    rel_frac:  float = 0.25, # tolerance as fraction of FK bone length
    vote_thr:  int   = 2,    # votes needed from violated bones to flag a joint
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], list[int]]:
    """Vote-based anatomical guard: flag joints implicated in ≥ vote_thr violated bones.

    Per-bone tolerance:
        tau = max(abs_floor, rel_frac * fk_len)
        A bone is violated if |tri_len - fk_len| > tau.

    For every violated bone both its endpoints receive +1 vote. A joint is flagged
    (set to NaN) only when its vote count reaches vote_thr. This disambiguates the
    shared joint: a mislocated L_hip violates both (pelvis→L_hip) and (L_hip→L_knee),
    so L_hip accumulates 2 votes and is flagged; pelvis and L_knee each get 1 → kept.

    Leaf joints (wrist, ankle, head, feet) participate in only one bone and cannot
    reach vote_thr=2 — they are never flagged, which is acceptable since they matter
    less for orientation than the central skeleton.

    Returns:
        J_filtered      (22, 3) copy with flagged joints set to NaN.
        violated_bones  list of (child, parent, tri_len, fk_len) for every violation.
        flagged_joints  list of joint indices that were set to NaN.
    """
    Jc    = J_can[:22].astype(np.float64)
    votes = np.zeros(22, dtype=int)
    violated_bones: list[tuple[int, int, float, float]] = []

    for child, parent in _SMPLX_PARENTS_21.items():
        if np.isnan(J_world[child]).any() or np.isnan(J_world[parent]).any():
            continue

        tri_len = float(np.linalg.norm(J_world[child].astype(np.float64) - J_world[parent].astype(np.float64)))
        fk_len  = float(np.linalg.norm(Jc[child] - Jc[parent]))
        if fk_len < 1e-4:
            continue

        tau = max(abs_floor, rel_frac * fk_len)
        if abs(tri_len - fk_len) > tau:
            votes[child]  += 1
            votes[parent] += 1
            violated_bones.append((child, parent, tri_len, fk_len))

    J = J_world.copy()
    flagged = [j for j in range(22) if votes[j] >= vote_thr and not np.isnan(J_world[j]).any()]
    for j in flagged:
        J[j] = np.nan

    return J, violated_bones, flagged


# ---------------------------------------------------------------------------
# GT helpers (reused from debug_translation.py)
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


_RICH_ORIG_W = 4112
_RICH_ORIG_H = 3008


def _load_gt_extrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    calib_dir = rich_root / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    if not calib_dir.is_dir():
        return None
    exts: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        cam_node = ET.parse(xml_path).getroot().find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts if exts else None


def _load_gt_intrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    calib_dir = rich_root / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    if not calib_dir.is_dir():
        return None
    ks: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        intr_node = ET.parse(xml_path).getroot().find("Intrinsics")
        if intr_node is None:
            continue
        vals = list(map(float, intr_node.find("data").text.split()))
        ks.append(np.array(vals, dtype=np.float64).reshape(3, 3))
    return ks if ks else None


def _make_gt_cam_arrays(
    placer: BodyPlacer, gt_exts: list[np.ndarray],
    gt_intrs: list[np.ndarray], cam_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, K = placer.T, placer.K
    vggt_K0 = placer.intrinsics[0, 0]
    s_x = float(vggt_K0[0, 2]) * 2.0 / _RICH_ORIG_W
    s_y = float(vggt_K0[1, 2]) * 2.0 / _RICH_ORIG_H
    m_ref = re.search(r"\d+", cam_dirs[0].name)
    ref_idx = int(m_ref.group()) if m_ref else 0
    E_ref4 = np.eye(4); E_ref4[:3] = gt_exts[ref_idx]
    extrinsics_gt = np.zeros((T, K, 3, 4), dtype=np.float32)
    intrinsics_gt = np.zeros((T, K, 3, 3), dtype=np.float32)
    cam_valid_gt  = np.zeros((T, K), dtype=bool)
    for k, cam_dir in enumerate(cam_dirs):
        m = re.search(r"\d+", cam_dir.name)
        gt_idx = int(m.group()) if m else k
        if gt_idx >= len(gt_exts) or gt_idx >= len(gt_intrs):
            continue
        E_k4 = np.eye(4); E_k4[:3] = gt_exts[gt_idx]
        E_k_rel = (E_k4 @ np.linalg.inv(E_ref4))[:3]
        K_gt = gt_intrs[gt_idx].copy()
        K_gt[0, 0] *= s_x; K_gt[0, 2] *= s_x
        K_gt[1, 1] *= s_y; K_gt[1, 2] *= s_y
        extrinsics_gt[:, k] = E_k_rel.astype(np.float32)
        intrinsics_gt[:, k] = K_gt.astype(np.float32)
        cam_valid_gt[:, k]  = True
    return extrinsics_gt, intrinsics_gt, cam_valid_gt


def _load_gt_raw(scene_name: str, rich_root: Path, split: str) -> dict[int, dict[int, dict]]:
    gt: dict[int, dict[int, dict]] = {}
    gt_root = rich_root / split / scene_name
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT not found: {gt_root}")
    for frame_dir in sorted(gt_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue
        for pkl_path in sorted(frame_dir.glob("*.pkl")):
            gt_pid = int(pkl_path.stem)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            transl = np.asarray(data["transl"], dtype=np.float32).squeeze()
            orient = np.asarray(data["global_orient"], dtype=np.float32).squeeze()
            raw_betas = data.get("betas") if data.get("betas") is not None else data.get("smplx_betas")
            betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)[:10] if raw_betas is not None else np.zeros(10, dtype=np.float32)
            gt.setdefault(gt_pid, {})[frame_idx] = {
                "transl": transl, "global_orient": orient, "betas": betas,
            }
    return gt


def _gt_pelvis_world(gt_raw: dict, placer: BodyPlacer) -> dict[int, dict[int, np.ndarray]]:
    zero_pose   = np.zeros((1, 63), dtype=np.float32)
    zero_orient = np.zeros((1, 3),  dtype=np.float32)
    result = {}
    for gt_pid, frames in gt_raw.items():
        betas = next(iter(frames.values()))["betas"]
        fk = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)
        j0 = fk[0, 0].astype(np.float32)
        result[gt_pid] = {fi: v["transl"] + j0 for fi, v in frames.items()}
    return result


def _match_pids(trans_dict, gt_pelvis, foreground_pids):
    gt_pids   = list(gt_pelvis.keys())
    ghost_pids = [p for p in trans_dict if p in foreground_pids]
    used: set[int] = set()
    mapping: dict[int, int] = {}
    for gt_pid in gt_pids:
        gt_frames = gt_pelvis[gt_pid]
        best, best_d = None, float("inf")
        for gp in ghost_pids:
            if gp in used:
                continue
            common = set(gt_frames) & set(trans_dict[gp])
            if not common:
                continue
            d = float(np.mean([np.linalg.norm(trans_dict[gp][f] - gt_frames[f]) for f in common]))
            if d < best_d:
                best_d, best = d, gp
        if best is not None:
            mapping[gt_pid] = best
            used.add(best)
    return mapping



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   required=True)
    p.add_argument("--rich_root",   required=True)
    p.add_argument("--smplx_model", required=True)
    p.add_argument("--split",       default="train_body")
    p.add_argument("--max_frames",  type=int, default=None)
    p.add_argument("--min_conf",    type=float, default=0.3,
                   help="Minimum pred_joint_confidence to include a camera observation")
    p.add_argument("--use_gt_cams", action="store_true",
                   help="Replace VGGT cameras with GT cameras from RICH calibration XMLs (scale=1.0)")
    p.add_argument("--use_gt_scale", action="store_true",
                   help="Use GT metric scale (from XML calibration baselines) with pred VGGT cameras")
    p.add_argument("--abs_floor", type=float, default=0.03,
                   help="Bone guard: minimum absolute tolerance per bone in metres (default 3cm)")
    p.add_argument("--rel_frac",  type=float, default=0.25,
                   help="Bone guard: tolerance as fraction of FK bone length (default 25%%)")
    p.add_argument("--vote_thr",  type=int,   default=2,
                   help="Bone guard: votes needed from violated bones to flag a joint (default 2)")
    p.add_argument("--min_joints", type=int,  default=5,
                   help="Minimum surviving joints required to run Procrustes (default 5)")
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    # ── Placer (loads VGGT cameras + depth) ───────────────────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer   = BodyPlacer(scene_dir, _smplx_arg)
    cam_dirs = placer._cam_dirs
    K        = len(cam_dirs)

    # ── Foreground pids ───────────────────────────────────────────────────────
    pid_cam_count: dict[int, int] = defaultdict(int)
    for cam_dir in cam_dirs:
        for f in (cam_dir / "body_data").glob("person_*.npz"):
            pid_cam_count[int(f.stem.split("_")[1])] += 1
    foreground_pids = {p for p, c in pid_cam_count.items() if c >= max(1, K - 1)}

    print(f"Scene      : {scene_name}")
    print(f"Cams       : {K}  |  Foreground pids: {sorted(foreground_pids)}")

    # ── GT cameras (optional) ─────────────────────────────────────────────────
    orig_extrinsics = orig_intrinsics = orig_cam_valid = None
    if args.use_gt_cams:
        gt_exts  = _load_gt_extrinsics(scene_name, rich_root)
        gt_intrs = _load_gt_intrinsics(scene_name, rich_root)
        if gt_exts is None or gt_intrs is None:
            print("ERROR: GT calibration XMLs not found"); return
        gt_exts_arr, gt_intrs_arr, gt_valid_arr = _make_gt_cam_arrays(
            placer, gt_exts, gt_intrs, cam_dirs
        )
        orig_extrinsics, orig_intrinsics, orig_cam_valid = (
            placer.extrinsics, placer.intrinsics, placer.cam_valid
        )
        placer.extrinsics = gt_exts_arr
        placer.intrinsics = gt_intrs_arr
        placer.cam_valid  = gt_valid_arr
        scale = np.ones(placer.T, dtype=np.float32)
        print("Scale      : 1.0 (GT cameras are metric)")
        print("Cams       : GT (from RICH calibration XMLs)")
    else:
        # ── Scale ─────────────────────────────────────────────────────────────
        scale = placer.load_mapanything_scale()
        if scale is not None:
            print(f"Scale      : MapAnything  median={float(np.median(scale)):.4f} m/VGGT-unit")
        else:
            print("Scale      : triangulation fallback")
            pred_betas = {
                int(f.stem.split("_")[1]): np.load(f)["smplx_betas"].mean(0)
                for cam_dir in cam_dirs
                for f in (cam_dir / "body_data").glob("person_*.npz")
                if "smplx_betas" in np.load(f, allow_pickle=False).files
            }
            scale = placer.estimate_scale_triangulated(pred_betas)

        # ── GT scale (for comparison) ─────────────────────────────────────────
        # GT scale = ratio of GT camera-center baselines to VGGT baselines.
        # Tells us how much MapAnything/triangulation scale is off from ground truth.
        _gt_exts_scale = _load_gt_extrinsics(scene_name, rich_root)
        if _gt_exts_scale is not None:
            _ref_m = re.search(r"\d+", cam_dirs[0].name)
            _ref_idx = int(_ref_m.group()) if _ref_m else 0
            _E_ref = _gt_exts_scale[_ref_idx]
            _C_ref = -_E_ref[:3, :3].T @ _E_ref[:3, 3]
            _gt_scales = []
            for _k, _cd in enumerate(cam_dirs):
                _m = re.search(r"\d+", _cd.name)
                _gi = int(_m.group()) if _m else _k
                if _gi >= len(_gt_exts_scale) or _gi == _ref_idx:
                    continue
                _E_k = _gt_exts_scale[_gi]
                _C_k = -_E_k[:3, :3].T @ _E_k[:3, 3]
                _t_vggt = float(np.linalg.norm(placer.extrinsics[0, _k, :, 3]))
                _gt_baseline = float(np.linalg.norm(_C_k - _C_ref))
                if _t_vggt > 1e-4:
                    _gt_scales.append(_gt_baseline / _t_vggt)
            if _gt_scales:
                _gt_scale_med = float(np.median(_gt_scales))
                _pred_scale_med = float(np.median(scale))
                _scale_err_pct = (_pred_scale_med - _gt_scale_med) / _gt_scale_med * 100
                print(f"GT scale   : {_gt_scale_med:.4f}  pred={_pred_scale_med:.4f}  "
                      f"err={_scale_err_pct:+.1f}%  "
                      f"({'over' if _scale_err_pct > 0 else 'under'}estimated)")
                if args.use_gt_scale:
                    scale = np.full(placer.T, _gt_scale_med, dtype=np.float32)
                    print(f"Scale      : GT override → {_gt_scale_med:.4f} (constant over all frames)")

    # ── Image principal point from VGGT intrinsics ────────────────────────────
    # SAM3D assumes cx = W/2, cy = H/2. We infer W, H from the depth map shape.
    if placer.depth_mm is not None:
        _, _, H_img, W_img = placer.depth_mm.shape
    else:
        # Fallback: estimate from VGGT principal point.
        K0 = placer.intrinsics[0, 0]
        W_img = int(round(K0[0, 2] * 2))
        H_img = int(round(K0[1, 2] * 2))
    cx, cy = W_img / 2.0, H_img / 2.0
    print(f"Image size : {W_img}×{H_img}  →  cx={cx:.1f}  cy={cy:.1f}")

    # ── Load per-camera body data ─────────────────────────────────────────────
    # cam_data_all[k] = {global_t: {body_pose, global_orient, betas, smplx_transl,
    #                                focal_length, confidence}}
    cam_data_all: list[dict[int, dict[int, dict]]] = []
    for cam_dir in cam_dirs:
        cam_map: dict[int, dict[int, dict]] = {}
        for pid in sorted(foreground_pids):
            data = load_cam_body_data(cam_dir, pid)
            if data is not None:
                cam_map[pid] = data
        cam_data_all.append(cam_map)

    # ── Triangulate + Procrustes per pid ──────────────────────────────────────
    # pred_betas for canonical FK (mean over cameras and frames).
    pred_betas_by_pid: dict[int, np.ndarray] = defaultdict(list)
    for cam_dir in cam_dirs:
        for pid in foreground_pids:
            f = cam_dir / "body_data" / f"person_{pid}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=False)
                if "smplx_betas" in d.files:
                    pred_betas_by_pid[pid].append(d["smplx_betas"].mean(0))
    pred_betas_by_pid = {p: np.mean(np.stack(v), 0) for p, v in pred_betas_by_pid.items()}

    zero_pose   = np.zeros((1, 63), dtype=np.float32)
    zero_orient = np.zeros((1, 3),  dtype=np.float32)

    trans_dict:   dict[int, dict[int, np.ndarray]] = {}
    orient_dict:  dict[int, dict[int, np.ndarray]] = {}
    raw_tri_dict: dict[int, dict[int, np.ndarray]] = {}  # J_world[0] before Procrustes

    for pid in tqdm(sorted(foreground_pids), desc="Persons"):
        betas = pred_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))

        # Canonical skeleton for Procrustes (zero orient, zero transl, pred betas).
        J_can = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)[0]  # (55, 3)

        # Triangulate SMPL-X joints in world space.
        J_world_by_frame = triangulate_all_joints(
            placer, cam_data_all, scale, pid, args.min_conf,
        )

        t_out:   dict[int, np.ndarray] = {}
        R_out:   dict[int, np.ndarray] = {}
        raw_tri: dict[int, np.ndarray] = {}  # raw triangulated pelvis before Procrustes
        # Accumulators for the end-of-pid summary.
        joint_flag_count: dict[int, int] = defaultdict(int)   # joint → frames flagged
        bone_viol_count:  dict[tuple, int] = defaultdict(int) # (child,par) → frames violated
        n_skipped = 0   # frames skipped because too few joints survived the guard

        for global_t, (J_world, joint_conf) in J_world_by_frame.items():
            # Store raw triangulated pelvis before any guard/Procrustes.
            if not np.isnan(J_world[0]).any():
                raw_tri[global_t] = J_world[0].copy()

            J_filtered, violated_bones, flagged_joints = anatomical_bone_guard(
                J_world, J_can,
                abs_floor=args.abs_floor,
                rel_frac=args.rel_frac,
                vote_thr=args.vote_thr,
            )

            # Count surviving (non-NaN) joints after guard.
            n_surviving = int((~np.isnan(J_filtered[:, 0])).sum())

            # Log any activity this frame.
            if violated_bones or flagged_joints:
                viol_str = "  ".join(
                    f"{_SMPLX_JOINT_NAMES[c]}→{_SMPLX_JOINT_NAMES[p]}"
                    f"(tri={tl:.2f}m fk={fl:.2f}m)"
                    for c, p, tl, fl in violated_bones
                )
                flag_str = " ".join(_SMPLX_JOINT_NAMES[j] for j in flagged_joints) or "—"
                print(f"    [t={global_t:5d}] violated: {viol_str or '—'}  "
                      f"| flagged: {flag_str}  | survivors: {n_surviving}/22")

            # Accumulate stats.
            for j in flagged_joints:
                joint_flag_count[j] += 1
            for c, p, _, __ in violated_bones:
                bone_viol_count[(c, p)] += 1

            # Skip frame if too few joints survive for a reliable Procrustes.
            if n_surviving < args.min_joints:
                n_skipped += 1
                continue

            out = procrustes_fit(J_filtered, J_can, weights=joint_conf,
                                 min_joints=args.min_joints)
            if out is None:
                n_skipped += 1
                continue
            R, _, pelvis_world = out
            t_out[global_t] = pelvis_world
            R_out[global_t] = R

        trans_dict[pid]   = t_out
        orient_dict[pid]  = R_out
        raw_tri_dict[pid] = raw_tri

        n_total = len(J_world_by_frame)
        print(f"  pid {pid}: {len(t_out)}/{n_total} frames valid  "
              f"| {n_skipped} skipped (< {args.min_joints} joints)  "
              f"| guard abs={args.abs_floor}m rel={args.rel_frac} vote≥{args.vote_thr}")
        if joint_flag_count:
            print("  flagged joints:")
            for jidx, cnt in sorted(joint_flag_count.items(), key=lambda x: -x[1]):
                print(f"    {_SMPLX_JOINT_NAMES[jidx]:>12}: {cnt:4d}/{n_total} frames flagged")
        if bone_viol_count:
            print("  violated bones:")
            for (c, p), cnt in sorted(bone_viol_count.items(), key=lambda x: -x[1]):
                print(f"    {_SMPLX_JOINT_NAMES[p]:>12}→{_SMPLX_JOINT_NAMES[c]:<12}: "
                      f"{cnt:4d}/{n_total} frames")
        if not joint_flag_count and not bone_viol_count:
            print("    (no violations)")

    # ── Restore placer cameras ────────────────────────────────────────────────
    if orig_extrinsics is not None:
        placer.extrinsics = orig_extrinsics
        placer.intrinsics = orig_intrinsics
        placer.cam_valid  = orig_cam_valid

    # ── GT loading + matching ─────────────────────────────────────────────────
    gt_raw    = _load_gt_raw(scene_name, rich_root, args.split)
    gt_pelvis = _gt_pelvis_world(gt_raw, placer)
    mapping   = _match_pids(trans_dict, gt_pelvis, foreground_pids)
    print(f"Match      : {mapping}\n")

    # ── Per-frame printout ────────────────────────────────────────────────────
    for gt_pid, ghost_pid in sorted(mapping.items()):
        betas   = pred_betas_by_pid.get(ghost_pid, np.zeros(10, dtype=np.float32))
        J_can_0 = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)[0, 0]

        # Betas sanity check: compare J_can[0] from pred vs GT betas.
        # A difference in J_can[0].y explains a systematic Δy bias because GT pelvis
        # is computed as gt_transl + J_can_0_GT while pred uses R @ J_can_0_pred + t.
        _betas_gt  = next(iter(gt_raw[gt_pid].values()))["betas"]
        _j0_gt     = placer._smplx_fk(_betas_gt[np.newaxis], zero_pose, zero_orient)[0, 0]
        _j0_diff   = J_can_0 - _j0_gt
        print(f"  J_can[0] pred={J_can_0}  gt={_j0_gt}  diff={_j0_diff}")

        pred_frames   = trans_dict.get(ghost_pid, {})
        orient_frames = orient_dict.get(ghost_pid, {})
        gt_frames     = gt_raw.get(gt_pid, {})
        common = sorted(set(pred_frames) & set(gt_frames))
        if args.max_frames:
            common = common[:args.max_frames]

        # Coordinate frame note:
        # pred_transl is in VGGT world frame (= cam_00 camera frame when cam_00
        # is the reference).  Camera convention: x=right, y=down, z=into-scene.
        # gt_transl is in the RICH multi-cam world frame; for scenes where cam_00
        # is the GT reference camera (e.g. BBQ) these two frames coincide.
        # Δx<0 = pred is to the left,  Δy<0 = pred is higher,  Δz<0 = pred is closer.
        # ωx/ωy/ωz are rotvec components of R_gt.T @ R_pred in world frame (degrees).
        W = 152
        print(f"{'─'*W}")
        _scale_mode = "GT" if (args.use_gt_cams or args.use_gt_scale) else "MapAnything"
        print(f"gt_pid={gt_pid}  ghost_pid={ghost_pid}  common_frames={len(common)}  "
              f"cameras={'GT' if args.use_gt_cams else 'VGGT-pred'}  scale={_scale_mode}")
        print(
            f"{'frame':>6}  {'GT_transl (x,y,z)':>28}  {'pred_transl (x,y,z)':>28}  "
            f"{'err_m':>6}  {'Δx':>7}  {'Δy':>7}  {'Δz':>7}  "
            f"{'orient°':>7}  {'ωx°':>7}  {'ωy°':>7}  {'ωz°':>7}"
        )
        print(f"{'─'*W}")

        transl_errs, orient_errs = [], []
        diffs   = []   # (Δx, Δy, Δz) for bias analysis
        rotvecs = []   # (ωx, ωy, ωz) for orient axis analysis
        for frame in common:
            pelvis_world = pred_frames[frame]
            R_pred       = orient_frames.get(frame)
            pred_transl  = pelvis_world - J_can_0   # SMPL-X: transl = pelvis_world - J_can[0]
            gt_entry     = gt_frames[frame]
            gt_transl    = gt_entry["transl"]
            gt_aa        = gt_entry["global_orient"]

            diff = (pred_transl - gt_transl).astype(np.float64)   # (3,) signed error
            diffs.append(diff)
            transl_err = float(np.linalg.norm(diff))
            transl_errs.append(transl_err)

            orient_err = np.nan
            rv = np.full(3, np.nan)
            if R_pred is not None:
                R_gt = SciR.from_rotvec(gt_aa.astype(np.float64)).as_matrix()
                R_rel = R_gt.T @ R_pred.astype(np.float64)   # rotation error in GT frame
                rv = SciR.from_matrix(R_rel).as_rotvec() * (180.0 / np.pi)
                orient_err = float(np.linalg.norm(rv))
                orient_errs.append(orient_err)
                rotvecs.append(rv)

            def _f3(v): return f"[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"
            def _s(v):  return f"{v:+.3f}" if not np.isnan(v) else "    nan"
            print(f"{frame:>6}  {_f3(gt_transl):>28}  {_f3(pred_transl):>28}  "
                  f"{transl_err:>6.3f}  {_s(diff[0]):>7}  {_s(diff[1]):>7}  {_s(diff[2]):>7}  "
                  f"{orient_err:>7.2f}  {_s(rv[0]):>7}  {_s(rv[1]):>7}  {_s(rv[2]):>7}")

        print(f"{'─'*W}")
        if transl_errs:
            te = np.array(transl_errs)
            D  = np.array(diffs)
            bias   = D.mean(0)
            spread = D.std(0)
            print(f"  transl  median={np.median(te):.3f}m  mean={np.mean(te):.3f}m  "
                  f"<0.5m={100*(te<0.5).mean():.0f}%")
            print(f"  bias    Δx={bias[0]:+.3f}m  Δy={bias[1]:+.3f}m  Δz={bias[2]:+.3f}m  "
                  f"  (pred−GT; cam frame: x=right y=down z=into-scene)")
            print(f"  spread  σx={spread[0]:.3f}m  σy={spread[1]:.3f}m  σz={spread[2]:.3f}m")
        if orient_errs:
            oe  = np.array(orient_errs)
            RV  = np.array(rotvecs)          # (N, 3)
            rv_bias   = RV.mean(0)
            rv_spread = RV.std(0)
            print(f"  orient  median={np.median(oe):.2f}°  mean={np.mean(oe):.2f}°  "
                  f"<30°={100*(oe<30).mean():.0f}%")
            print(f"  rv bias ωx={rv_bias[0]:+.2f}°  ωy={rv_bias[1]:+.2f}°  ωz={rv_bias[2]:+.2f}°"
                  f"  (rotvec of R_gt.T@R_pred in GT frame)")
            print(f"  rv σ    ωx={rv_spread[0]:.2f}°   ωy={rv_spread[1]:.2f}°   ωz={rv_spread[2]:.2f}°")

        # ── Naive modality: J_world[0] translation + Procrustes orientation ──
        # Compares using the raw triangulated pelvis (average of per-camera
        # smplx_transl unprojected to world) combined with the Procrustes rotation.
        raw_frames  = raw_tri_dict.get(ghost_pid, {})
        proc_orient = orient_dict.get(ghost_pid, {})
        naive_common = sorted(set(raw_frames) & set(gt_frames) & set(proc_orient))
        if args.max_frames:
            naive_common = naive_common[:args.max_frames]
        if naive_common:
            naive_te, naive_oe = [], []
            naive_diffs, naive_rvs = [], []
            for frame in naive_common:
                gt_pelv = gt_pelvis[gt_pid].get(frame)
                R_pred  = proc_orient.get(frame)
                if gt_pelv is None or R_pred is None:
                    continue
                diff = raw_frames[frame].astype(np.float64) - gt_pelv.astype(np.float64)
                naive_diffs.append(diff)
                naive_te.append(float(np.linalg.norm(diff)))
                gt_aa = gt_frames[frame]["global_orient"]
                R_gt  = SciR.from_rotvec(gt_aa.astype(np.float64)).as_matrix()
                rv    = SciR.from_matrix(R_gt.T @ R_pred.astype(np.float64)).as_rotvec() * (180.0 / np.pi)
                naive_oe.append(float(np.linalg.norm(rv)))
                naive_rvs.append(rv)
            if naive_te:
                nte  = np.array(naive_te)
                ND   = np.array(naive_diffs)
                bias = ND.mean(0)
                print(f"  naive   transl median={np.median(nte):.3f}m  mean={np.mean(nte):.3f}m  "
                      f"bias Δx={bias[0]:+.3f}m  Δy={bias[1]:+.3f}m  Δz={bias[2]:+.3f}m"
                      f"  (J_world[0] + Procrustes R)")
            if naive_oe:
                noe  = np.array(naive_oe)
                NRV  = np.array(naive_rvs)
                rv_b = NRV.mean(0)
                print(f"  naive   orient median={np.median(noe):.2f}°  mean={np.mean(noe):.2f}°  "
                      f"rv bias ωx={rv_b[0]:+.2f}°  ωy={rv_b[1]:+.2f}°  ωz={rv_b[2]:+.2f}°")
        print()


if __name__ == "__main__":
    main()
