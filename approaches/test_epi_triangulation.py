"""Test SMPL-X joint triangulation with epipolar-weighted DLT.

Same pipeline as test_smplx_triangulation.py, but replaces pure confidence-
weighted DLT with epipolar-consistency-weighted DLT.

For each (camera, joint) pair the Sampson epipolar error across all other
cameras is computed and combined with SAM3D's pred_joint_confidence as a
multiplicative weight:

    w_k = conf_k * exp(-mean_sampson_error_k / epi_sigma)

Intuition: a camera/joint observation that is geometrically inconsistent with
the rest of the camera rig (high Sampson error) is down-weighted, even if
SAM3D reports high internal confidence.  Epipolar errors are scale-invariant
(F matrix scales out), so they require no MapAnything correction.

Usage:
    pixi run python approaches/test_epi_triangulation.py \\
        --scene_dir /path/to/ghost_outputs/BBQ_001_guitar \\
        --rich_root  /path/to/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        [--epi_sigma 5.0] [--split train_body] [--max_frames 20] [--min_conf 0.3]
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
            "body_pose":     d["smplx_body_pose"][local_t],
            "global_orient": d["smplx_global_orient"][local_t],
            "betas":         d["smplx_betas"][local_t],
            "smplx_transl":  d["smplx_transl"][local_t],
            "focal_length":  float(d["focal_length"][local_t]),
            "confidence":    d["pred_joint_confidence"][local_t],
        }
    return result


# ---------------------------------------------------------------------------
# 2. build_sam3d_K  (unused directly but kept for reference)
# ---------------------------------------------------------------------------

def build_sam3d_K(focal_length: float, cx: float, cy: float) -> np.ndarray:
    return np.array([
        [focal_length, 0.0,          cx],
        [0.0,          focal_length, cy],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. precompute_cam_fk
# ---------------------------------------------------------------------------

def precompute_cam_fk(
    placer:    BodyPlacer,
    cam_data:  dict[int, dict],
) -> dict[int, np.ndarray]:
    if not cam_data:
        return {}

    frames = sorted(cam_data.keys())
    body_poses     = np.stack([cam_data[f]["body_pose"]     for f in frames])
    global_orients = np.stack([cam_data[f]["global_orient"] for f in frames])
    betas_arr      = np.stack([cam_data[f]["betas"]         for f in frames])
    translsarr     = np.stack([cam_data[f]["smplx_transl"]  for f in frames])

    J_batch = placer._smplx_fk(betas_arr, body_poses, global_orients)
    J_batch = J_batch + translsarr[:, np.newaxis, :]

    return {f: J_batch[i] for i, f in enumerate(frames)}


# ---------------------------------------------------------------------------
# 4. project_joints_2d
# ---------------------------------------------------------------------------

def project_joints_2d(J_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts2d = np.full((J_cam.shape[0], 2), np.nan, dtype=np.float64)
    for j, x in enumerate(J_cam.astype(np.float64)):
        if x[2] > 1e-4:
            uv = K @ x
            pts2d[j] = uv[:2] / uv[2]
    return pts2d


# ---------------------------------------------------------------------------
# 5. weighted_dlt_single_joint  (unchanged)
# ---------------------------------------------------------------------------

def weighted_dlt_single_joint(
    pts2d_per_cam: list[np.ndarray],
    P_mats:        list[np.ndarray],
    confs:         list[float],
    min_conf:      float = 0.3,
) -> np.ndarray | None:
    rows = []
    for (u, v), P, c in zip(pts2d_per_cam, P_mats, confs):
        if c < min_conf or np.isnan(u) or np.isnan(v):
            continue
        w = float(np.sqrt(max(c, 0.0)))
        rows.append(w * (P[0] - u * P[2]))
        rows.append(w * (P[1] - v * P[2]))

    if len(rows) < 4:
        return None

    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return (X[:3] / X[3]).astype(np.float32)


# ---------------------------------------------------------------------------
# 5b. Epipolar helpers  ← NEW
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that _skew(v) @ w == v × w."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=np.float64)


def compute_F_matrix(
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> np.ndarray:
    """Fundamental matrix F s.t. x2^T @ F @ x1 = 0 for true correspondences.

    Cameras given as extrinsics [R | t] (X_cam = R @ X_world + t).
    Translations need not be metric — epipolar errors are scale-invariant.
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R2 @ R1.T @ t1
    E = _skew(t_rel) @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def sampson_error(x1: np.ndarray, x2: np.ndarray, F: np.ndarray) -> float:
    """Symmetric Sampson epipolar error in pixels.

    Returns sqrt( |x2^T F x1|^2 / (||(Fx1)_xy||^2 + ||(F^T x2)_xy||^2) ).
    Scale-invariant in F and symmetric in (x1, F, x2) ↔ (x2, F^T, x1).
    Returns np.nan if either point is NaN.
    """
    if np.isnan(x1).any() or np.isnan(x2).any():
        return float("nan")
    x1h  = np.array([x1[0], x1[1], 1.0], dtype=np.float64)
    x2h  = np.array([x2[0], x2[1], 1.0], dtype=np.float64)
    Fx1  = F   @ x1h
    FTx2 = F.T @ x2h
    num   = float(x2h @ Fx1) ** 2
    denom = Fx1[0]**2 + Fx1[1]**2 + FTx2[0]**2 + FTx2[1]**2 + 1e-10
    return float(np.sqrt(num / denom))


def compute_epi_scores(
    cam_obs:  list,   # entries: (pts2d (55,2), P (3,4), conf (55,), K (3,3), R (3,3), t (3,))
    n_joints: int = 22,
) -> np.ndarray:
    """Mean Sampson epipolar error per (camera, joint) across all camera pairs.

    For camera i and joint j: average Sampson error with every other camera.
    Lower = more geometrically consistent with the rest of the rig.

    The error is attributed symmetrically to both cameras in each pair so
    that a single bad camera accumulates high error across all its joints.

    Returns:
        epi  (n_cams, n_joints) float64 — np.inf where no valid pairs exist.
    """
    n     = len(cam_obs)
    epi   = np.zeros((n, n_joints), dtype=np.float64)
    count = np.zeros((n, n_joints), dtype=np.int32)

    for i in range(n):
        K_i, R_i, t_i = cam_obs[i][3], cam_obs[i][4], cam_obs[i][5]
        pts_i = cam_obs[i][0]
        for j in range(i + 1, n):
            K_j, R_j, t_j = cam_obs[j][3], cam_obs[j][4], cam_obs[j][5]
            pts_j = cam_obs[j][0]
            F = compute_F_matrix(K_i, R_i, t_i, K_j, R_j, t_j)
            for jnt in range(n_joints):
                err = sampson_error(pts_i[jnt], pts_j[jnt], F)
                if not np.isnan(err):
                    epi[i, jnt]   += err
                    epi[j, jnt]   += err   # symmetric: both cameras share the error
                    count[i, jnt] += 1
                    count[j, jnt] += 1

    good        = count > 0
    epi[good]  /= count[good]
    epi[~good]  = np.inf
    return epi


# ---------------------------------------------------------------------------
# 6. triangulate_all_joints  ← modified: epipolar-weighted DLT
# ---------------------------------------------------------------------------

def triangulate_all_joints(
    placer:        BodyPlacer,
    cam_data_all:  list[dict[int, dict]],
    scale:         np.ndarray | float,
    pid:           int,
    min_conf:      float,
    epi_sigma:     float = 5.0,
    log_epi:       bool  = False,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Triangulate 22 SMPL-X joints with epipolar-weighted DLT.

    Same as test_smplx_triangulation.py but the per-camera weight is:
        w_k = conf_k * exp(-mean_sampson_error_k / epi_sigma)

    epi_sigma is in pixels; larger values reduce the epipolar penalty.

    Returns:
        {global_frame: (J_world (22, 3), joint_conf (22,))}
    """
    # ── Step 1: batched FK per camera ─────────────────────────────────────────
    fk_by_cam: dict[int, dict[int, np.ndarray]] = {}
    for k, cam_data in enumerate(tqdm(cam_data_all, desc="  FK per camera", leave=False)):
        if pid not in cam_data:
            continue
        fk_by_cam[k] = precompute_cam_fk(placer, cam_data[pid])

    all_frames: set[int] = set()
    for frames in fk_by_cam.values():
        all_frames.update(frames.keys())

    # ── Step 2: per-frame epipolar-weighted DLT ───────────────────────────────
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    _epi_all: list[np.ndarray] = []

    for global_t in tqdm(sorted(all_frames), desc="  DLT frames", leave=False):
        vggt_t = global_t
        if vggt_t < 0 or vggt_t >= placer.T:
            continue

        s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

        # Build cam_obs — now includes (pts2d, P, conf, K, R_raw, t_raw)
        # R_raw / t_raw are UNSCALED extrinsics (epipolar error is scale-invariant).
        cam_obs: list[tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]] = []

        for k, fk_frames in fk_by_cam.items():
            if not placer.cam_valid[vggt_t, k]:
                continue
            J_cam = fk_frames.get(global_t)
            if J_cam is None:
                continue

            conf     = cam_data_all[k][pid][global_t]["confidence"]
            K_vggt   = placer.intrinsics[vggt_t, k].astype(np.float64)
            pts2d    = project_joints_2d(J_cam, K_vggt)

            E_raw    = placer.extrinsics[vggt_t, k].astype(np.float64)   # unscaled
            R_raw    = E_raw[:3, :3]
            t_raw    = E_raw[:3, 3]

            E_scaled         = E_raw.copy()
            E_scaled[:3, 3] *= s
            P = K_vggt @ E_scaled

            cam_obs.append((pts2d, P, conf, K_vggt, R_raw, t_raw))

        if len(cam_obs) < 2:
            continue

        # ── Epipolar scores: (n_cams, 22) ─────────────────────────────────────
        epi = compute_epi_scores(cam_obs, n_joints=22)   # lower = more consistent

        valid_epi = epi[np.isfinite(epi)]
        _epi_all.append(valid_epi)
        if log_epi and valid_epi.size:
            epi_w = np.exp(-valid_epi / epi_sigma)
            print(f"  t={global_t:4d}  Sampson px  mean={valid_epi.mean():.2f}  "
                  f"p50={np.median(valid_epi):.2f}  max={valid_epi.max():.2f}  "
                  f"| epi_w  mean={epi_w.mean():.3f}  min={epi_w.min():.3f}")

        # ── Per-joint weighted DLT ─────────────────────────────────────────────
        J_world    = np.full((22, 3), np.nan, dtype=np.float32)
        joint_conf = np.zeros(22, dtype=np.float32)

        for j in range(22):
            pts_j  = [o[0][j]        for o in cam_obs]
            P_j    = [o[1]           for o in cam_obs]
            conf_j = [float(o[2][j]) for o in cam_obs]

            # Combined weight: SAM3D confidence × epipolar geometric consistency.
            combined_j = [
                c * float(np.exp(-epi[i, j] / epi_sigma))
                if np.isfinite(epi[i, j]) else c
                for i, c in enumerate(conf_j)
            ]

            X = weighted_dlt_single_joint(pts_j, P_j, combined_j, min_conf)
            if X is not None:
                J_world[j] = X
                valid_c = [c for c in combined_j if c >= min_conf]
                joint_conf[j] = float(np.mean(valid_c)) if valid_c else 0.0

        result[global_t] = (J_world, joint_conf)

    # ── Epipolar weight summary ────────────────────────────────────────────────
    if _epi_all:
        all_epi = np.concatenate(_epi_all)
        all_w   = np.exp(-all_epi / epi_sigma)
        print(f"\n[epi summary over {len(_epi_all)} frames, {all_epi.size} (cam,joint) pairs]")
        print(f"  Sampson error (px): mean={all_epi.mean():.3f}  "
              f"p25={np.percentile(all_epi,25):.3f}  p50={np.median(all_epi):.3f}  "
              f"p75={np.percentile(all_epi,75):.3f}  p95={np.percentile(all_epi,95):.3f}  "
              f"max={all_epi.max():.3f}")
        print(f"  epi weight exp(-epi/{epi_sigma}): "
              f"mean={all_w.mean():.4f}  "
              f"p5={np.percentile(all_w,5):.4f}  "
              f"min={all_w.min():.4f}  max(theoretical)=1.0000")

    return result


# ---------------------------------------------------------------------------
# 7. procrustes_fit
# ---------------------------------------------------------------------------

def procrustes_fit(
    J_world:    np.ndarray,
    J_can:      np.ndarray,
    weights:    np.ndarray | None = None,
    min_joints: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n     = J_world.shape[0]
    valid = ~np.isnan(J_world[:, 0])
    if valid.sum() < min_joints:
        return None

    A = J_world[valid].astype(np.float64)
    B = J_can[:n][valid].astype(np.float64)

    if weights is not None:
        w = np.maximum(weights[:n][valid].astype(np.float64), 1e-6)
    else:
        w = np.ones(len(A))
    w /= w.sum()

    A_mean = w @ A
    B_mean = w @ B
    H = (B - B_mean).T @ (w[:, None] * (A - A_mean))

    U, _, Vt = np.linalg.svd(H)
    d_sign   = np.linalg.det(Vt.T @ U.T)
    R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
    t = (A_mean - R.astype(np.float64) @ B_mean).astype(np.float32)

    pelvis_world = (R.astype(np.float64) @ J_can[0].astype(np.float64)
                    + t.astype(np.float64)).astype(np.float32)
    return R, t, pelvis_world


# ---------------------------------------------------------------------------
# 7b. anatomical_bone_guard
# ---------------------------------------------------------------------------

def anatomical_bone_guard(
    J_world:   np.ndarray,
    J_can:     np.ndarray,
    abs_floor: float = 0.03,
    rel_frac:  float = 0.25,
    vote_thr:  int   = 2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], list[int]]:
    Jc    = J_can[:22].astype(np.float64)
    votes = np.zeros(22, dtype=int)
    violated_bones: list[tuple[int, int, float, float]] = []

    for child, parent in _SMPLX_PARENTS_21.items():
        if np.isnan(J_world[child]).any() or np.isnan(J_world[parent]).any():
            continue
        tri_len = float(np.linalg.norm(
            J_world[child].astype(np.float64) - J_world[parent].astype(np.float64)))
        fk_len  = float(np.linalg.norm(Jc[child] - Jc[parent]))
        if fk_len < 1e-4:
            continue
        tau = max(abs_floor, rel_frac * fk_len)
        if abs(tri_len - fk_len) > tau:
            votes[child]  += 1
            votes[parent] += 1
            violated_bones.append((child, parent, tri_len, fk_len))

    J       = J_world.copy()
    flagged = [j for j in range(22) if votes[j] >= vote_thr and not np.isnan(J_world[j]).any()]
    for j in flagged:
        J[j] = np.nan

    return J, violated_bones, flagged


# ---------------------------------------------------------------------------
# GT helpers
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
    gt_pids    = list(gt_pelvis.keys())
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
    p.add_argument("--min_conf",    type=float, default=0.3)
    p.add_argument("--epi_sigma",   type=float, default=5.0,
                   help="Sampson error scale (pixels) for epipolar weight exp(-err/sigma). "
                        "Larger = softer penalty. (default 5.0)")
    p.add_argument("--use_gt_cams", action="store_true",
                   help="Replace VGGT cameras with GT cameras from RICH calibration XMLs (scale=1.0)")
    p.add_argument("--use_gt_scale", action="store_true",
                   help="Use GT metric scale (from XML calibration baselines) with pred VGGT cameras")
    p.add_argument("--abs_floor", type=float, default=0.03)
    p.add_argument("--rel_frac",  type=float, default=0.25)
    p.add_argument("--vote_thr",  type=int,   default=2)
    p.add_argument("--min_joints", type=int,  default=5)
    p.add_argument("--log_epi",   action="store_true",
                   help="Print per-frame Sampson error stats and a per-person summary.")
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer   = BodyPlacer(scene_dir, _smplx_arg)
    cam_dirs = placer._cam_dirs
    K        = len(cam_dirs)

    pid_cam_count: dict[int, int] = defaultdict(int)
    for cam_dir in cam_dirs:
        for f in (cam_dir / "body_data").glob("person_*.npz"):
            pid_cam_count[int(f.stem.split("_")[1])] += 1
    foreground_pids = {p for p, c in pid_cam_count.items() if c >= max(1, K - 1)}

    print(f"Scene      : {scene_name}")
    print(f"Cams       : {K}  |  Foreground pids: {sorted(foreground_pids)}")
    print(f"epi_sigma  : {args.epi_sigma} px")

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
                _gt_scale_med  = float(np.median(_gt_scales))
                _pred_scale_med = float(np.median(scale))
                _scale_err_pct = (_pred_scale_med - _gt_scale_med) / _gt_scale_med * 100
                print(f"GT scale   : {_gt_scale_med:.4f}  pred={_pred_scale_med:.4f}  "
                      f"err={_scale_err_pct:+.1f}%  "
                      f"({'over' if _scale_err_pct > 0 else 'under'}estimated)")
                if args.use_gt_scale:
                    scale = np.full(placer.T, _gt_scale_med, dtype=np.float32)
                    print(f"Scale      : GT override → {_gt_scale_med:.4f} (constant over all frames)")

    if placer.depth_mm is not None:
        _, _, H_img, W_img = placer.depth_mm.shape
    else:
        K0 = placer.intrinsics[0, 0]
        W_img = int(round(K0[0, 2] * 2))
        H_img = int(round(K0[1, 2] * 2))
    cx, cy = W_img / 2.0, H_img / 2.0
    print(f"Image size : {W_img}×{H_img}  →  cx={cx:.1f}  cy={cy:.1f}")

    cam_data_all: list[dict[int, dict[int, dict]]] = []
    for cam_dir in cam_dirs:
        cam_map: dict[int, dict[int, dict]] = {}
        for pid in sorted(foreground_pids):
            data = load_cam_body_data(cam_dir, pid)
            if data is not None:
                cam_map[pid] = data
        cam_data_all.append(cam_map)

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

    trans_dict:        dict[int, dict[int, np.ndarray]] = {}
    orient_dict:       dict[int, dict[int, np.ndarray]] = {}
    raw_tri_dict:      dict[int, dict[int, np.ndarray]] = {}
    naive_orient_dict: dict[int, dict[int, np.ndarray]] = {}

    for pid in tqdm(sorted(foreground_pids), desc="Persons"):
        betas = pred_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))
        J_can = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)[0]

        J_world_by_frame = triangulate_all_joints(
            placer, cam_data_all, scale, pid, args.min_conf,
            epi_sigma=args.epi_sigma,
            log_epi=args.log_epi,
        )

        t_out:   dict[int, np.ndarray] = {}
        R_out:   dict[int, np.ndarray] = {}
        raw_tri: dict[int, np.ndarray] = {}
        joint_flag_count: dict[int, int] = defaultdict(int)
        bone_viol_count:  dict[tuple, int] = defaultdict(int)
        n_skipped = 0

        for global_t, (J_world, joint_conf) in J_world_by_frame.items():
            if not np.isnan(J_world[0]).any():
                raw_tri[global_t] = J_world[0].copy()

            J_filtered, violated_bones, flagged_joints = anatomical_bone_guard(
                J_world, J_can,
                abs_floor=args.abs_floor,
                rel_frac=args.rel_frac,
                vote_thr=args.vote_thr,
            )

            n_surviving = int((~np.isnan(J_filtered[:, 0])).sum())

            if violated_bones or flagged_joints:
                viol_str = "  ".join(
                    f"{_SMPLX_JOINT_NAMES[c]}→{_SMPLX_JOINT_NAMES[p]}"
                    f"(tri={tl:.2f}m fk={fl:.2f}m)"
                    for c, p, tl, fl in violated_bones
                )
                flag_str = " ".join(_SMPLX_JOINT_NAMES[j] for j in flagged_joints) or "—"
                print(f"    [t={global_t:5d}] violated: {viol_str or '—'}  "
                      f"| flagged: {flag_str}  | survivors: {n_surviving}/22")

            for j in flagged_joints:
                joint_flag_count[j] += 1
            for c, par, _, __ in violated_bones:
                bone_viol_count[(c, par)] += 1

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
            for (c, par), cnt in sorted(bone_viol_count.items(), key=lambda x: -x[1]):
                print(f"    {_SMPLX_JOINT_NAMES[par]:>12}→{_SMPLX_JOINT_NAMES[c]:<12}: "
                      f"{cnt:4d}/{n_total} frames")
        if not joint_flag_count and not bone_viol_count:
            print("    (no violations)")

    if orig_extrinsics is not None:
        placer.extrinsics = orig_extrinsics
        placer.intrinsics = orig_intrinsics
        placer.cam_valid  = orig_cam_valid

    gt_raw    = _load_gt_raw(scene_name, rich_root, args.split)
    gt_pelvis = _gt_pelvis_world(gt_raw, placer)
    mapping   = _match_pids(trans_dict, gt_pelvis, foreground_pids)
    print(f"Match      : {mapping}\n")

    for gt_pid, ghost_pid in sorted(mapping.items()):
        betas   = pred_betas_by_pid.get(ghost_pid, np.zeros(10, dtype=np.float32))
        J_can_0 = placer._smplx_fk(betas[np.newaxis], zero_pose, zero_orient)[0, 0]

        _betas_gt = next(iter(gt_raw[gt_pid].values()))["betas"]
        _j0_gt    = placer._smplx_fk(_betas_gt[np.newaxis], zero_pose, zero_orient)[0, 0]
        print(f"  J_can[0] pred={J_can_0}  gt={_j0_gt}  diff={J_can_0 - _j0_gt}")

        pred_frames   = trans_dict.get(ghost_pid, {})
        orient_frames = orient_dict.get(ghost_pid, {})
        gt_frames     = gt_raw.get(gt_pid, {})
        common = sorted(set(pred_frames) & set(gt_frames))
        if args.max_frames:
            common = common[:args.max_frames]

        W = 152
        print(f"{'─'*W}")
        _scale_mode = "GT" if (args.use_gt_cams or args.use_gt_scale) else "MapAnything"
        print(f"gt_pid={gt_pid}  ghost_pid={ghost_pid}  common_frames={len(common)}  "
              f"cameras={'GT' if args.use_gt_cams else 'VGGT-pred'}  scale={_scale_mode}  "
              f"epi_sigma={args.epi_sigma}px")
        print(
            f"{'frame':>6}  {'GT_transl (x,y,z)':>28}  {'pred_transl (x,y,z)':>28}  "
            f"{'err_m':>6}  {'Δx':>7}  {'Δy':>7}  {'Δz':>7}  "
            f"{'orient°':>7}  {'ωx°':>7}  {'ωy°':>7}  {'ωz°':>7}"
        )
        print(f"{'─'*W}")

        transl_errs, orient_errs = [], []
        diffs, rotvecs = [], []
        for frame in common:
            pelvis_world = pred_frames[frame]
            R_pred       = orient_frames.get(frame)
            pred_transl  = pelvis_world - J_can_0
            gt_entry     = gt_frames[frame]
            gt_transl    = gt_entry["transl"]
            gt_aa        = gt_entry["global_orient"]

            diff = (pred_transl - gt_transl).astype(np.float64)
            diffs.append(diff)
            transl_err = float(np.linalg.norm(diff))
            transl_errs.append(transl_err)

            orient_err = np.nan
            rv = np.full(3, np.nan)
            if R_pred is not None:
                R_gt = SciR.from_rotvec(gt_aa.astype(np.float64)).as_matrix()
                R_rel = R_gt.T @ R_pred.astype(np.float64)
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
            bias, spread = D.mean(0), D.std(0)
            print(f"  transl  median={np.median(te):.3f}m  mean={np.mean(te):.3f}m  "
                  f"<0.5m={100*(te<0.5).mean():.0f}%")
            print(f"  bias    Δx={bias[0]:+.3f}m  Δy={bias[1]:+.3f}m  Δz={bias[2]:+.3f}m  "
                  f"  (pred−GT; cam frame: x=right y=down z=into-scene)")
            print(f"  spread  σx={spread[0]:.3f}m  σy={spread[1]:.3f}m  σz={spread[2]:.3f}m")
        if orient_errs:
            oe       = np.array(orient_errs)
            RV       = np.array(rotvecs)
            rv_bias  = RV.mean(0)
            rv_spread = RV.std(0)
            print(f"  orient  median={np.median(oe):.2f}°  mean={np.mean(oe):.2f}°  "
                  f"<30°={100*(oe<30).mean():.0f}%")
            print(f"  rv bias ωx={rv_bias[0]:+.2f}°  ωy={rv_bias[1]:+.2f}°  ωz={rv_bias[2]:+.2f}°"
                  f"  (rotvec of R_gt.T@R_pred in GT frame)")
            print(f"  rv σ    ωx={rv_spread[0]:.2f}°   ωy={rv_spread[1]:.2f}°   ωz={rv_spread[2]:.2f}°")

        # ── Naive modality: J_world[0] + Procrustes R ─────────────────────────
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
