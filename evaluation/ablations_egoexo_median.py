"""EgoExo4D progressive oracle ablations — error attribution for the ghost
pipeline, GEODESIC-MEDIAN FUSION (no checkpoint).

Copy of evaluation/ablations_egoexo.py with the learned PoseFusionModule
swapped for the geodesic median of the per-camera SAM3D poses (median_fuse,
identical to evaluation/evaluate_egoexo_median.py): the L1 estimator on SO(3),
solved by Weiszfeld/IRLS from a chordal-mean seed. This is the shipped fusion
rule, so this ladder is the one that matches the reported production numbers.
No temporal padding is needed here (unlike the v2-checkpoint version): median
fusion has no cross-frame coupling, so running it directly at T=1 is provably
identical to padding first — see evaluate_egoexo_median.py's note on this.

Each rung substitutes one more component with ground truth, so total error can
be attributed to a stage:

  M2: pred-cam + GT-scale + pred-pose   (removes scale error)
  M3: GT-cam   + GT-scale + pred-pose   (removes camera error too)

Reading the table: M2 - prod(median) = scale error, M3 - M2 = camera error.
The remaining M3 error is placement-given-perfect-cameras + pose error; the
pose part is read directly off PA-MPJPE, which is placement-invariant.

There is deliberately NO M4 (GT pose). EgoExo4D ships triangulated COCO-17
keypoints, not SMPL/SMPL-X parameters, so a GT body pose does not exist and
cannot be substituted into the placer. PA-MPJPE already isolates pose error.
Note PA is identical across M2/M3 by construction (same fused pose, and PA is
invariant to the Sim(3) that placement changes) — only W-MPJPE† moves.

GT cameras: gopro_calibs.csv gives cam->world (q,t) per GoPro. They are turned
into world->cam extrinsics re-rooted to camera_names[0] (the placer's reference
frame) and patched onto BodyPlacer.extrinsics; metric scale is then exactly 1.0.
GT *intrinsics* are NOT substituted: the placer re-maps pred_keypoints_2d into
VGGT image space (_orig_to_vggt), so VGGT intrinsics are the ones consistent
with the observations; a GT K would need the same crop+resize composition
applied to it. Same choice as the RICH ladder.

For the honest production number (no oracle substitutions) see
evaluation/evaluate_egoexo_median.py.

Usage
-----
    pixi run python evaluation/ablations_egoexo_median.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \\
        --gt_root    /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \\
        [--modalities 2,3] [--max_scenes N] [--scene SCENE_NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scipy.spatial.transform import Rotation as SciR

from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_NUM_JOINTS = 55   # SMPL-X joints fed to the fusion model (root + 54)

# Oracle ladder run by this script. See the module docstring; M4 (GT pose) is
# impossible on EgoExo4D — there are no GT body-model parameters.
_MODALITIES = (2, 3)
_MODALITY_LABELS = {
    2: "M2: pred-cam + GT-scale + pred-pose",
    3: "M3: GT-cam   + GT-scale + pred-pose",
}

# ---------------------------------------------------------------------------
# Joint mapping: GT keypoints_gt.json name → SMPL-X body joint index (0-21)
# ---------------------------------------------------------------------------
# GT joints are COCO-17 (ViTPose-triangulated).  Mesh-based competitors (HSfM,
# CHROMM via Multi-HMR/HMR2) report COCO-17 joints *regressed from the body mesh*,
# NOT SMPL-X FK joint centres.  SMPL-X FK hips are femoral-head joints ~87 mm
# narrower than the COCO hip keypoints; Procrustes cannot absorb a per-joint
# convention offset -> systematic PA-MPJPE inflation.  We therefore regress
# COCO-17 from the posed SMPL-X mesh and index GT by COCO order.
GT_TO_COCO: dict[str, int] = {
    # All 17 COCO joints — the EgoExo4D body-pose benchmark definition. GT
    # annotates all 17 (verified 2026-08-04). Matching is unaffected: it gates
    # on GT_TO_MHR70 (12 limbs only), so this is a metric-only change.
    "nose":            0,
    "left-eye":        1,
    "right-eye":       2,
    "left-ear":        3,
    "right-ear":       4,
    "left-shoulder":   5,
    "right-shoulder":  6,
    "left-elbow":      7,
    "right-elbow":     8,
    "left-wrist":      9,
    "right-wrist":    10,
    "left-hip":       11,
    "right-hip":      12,
    "left-knee":      13,
    "right-knee":     14,
    "left-ankle":     15,
    "right-ankle":    16,
}

# manual_reid convention: the single annotated GT subject is labelled group 1.
GT_PERSON_PID = 1

# Takes excluded for broken ground truth / camera calibration, verified by
# reprojecting the GT skeleton into every camera (2026-07-20).  These are data
# defects, not model failures, and must be reported alongside the results.
EXCLUDED_TAKES: dict[str, str] = {
    "cmu_soccer16_2":       "cam02 + cam05 miscalibrated; GT joints project behind the camera",
    "uniandes_dance_002_2":  "GT lower body mistriangulated; reprojection invalid in all 5 cameras",
    "uniandes_dance_002_11": "same scene and defect as uniandes_dance_002_2, different frame",
}

# GT joint name -> MHR70 index in SAM3D ``pred_keypoints_2d`` (mirrors
# fusion/placer.py::_SMPLX_TO_MHR70).  Used to auto-identify the GT subject.
GT_TO_MHR70: dict[str, int] = {
    "left-shoulder":  5, "right-shoulder":  6,
    "left-elbow":     7, "right-elbow":     8,
    "left-wrist":    62, "right-wrist":    41,
    "left-hip":       9, "right-hip":      10,
    "left-knee":     11, "right-knee":     12,
    "left-ankle":    13, "right-ankle":    14,
}

# Auto-match acceptance gates (see auto_match_gt_subject).
_MATCH_RATIO  = 0.60   # best candidate must beat runner-up by this factor
_MATCH_RES_PX = 12.0   # cross-camera triangulation residual = "same person"
_MATCH_OFF_CM = 20.0   # triangulated subject must land this close to GT

_COCO_REG: np.ndarray | None = None


def coco_regressor() -> np.ndarray:
    """COCO-17 joint regressor for SMPL-X vertices: (17, 10475).

    Chains J_regressor_coco (SMPL, 17x6890) with smplx2smpl (6890x10475) so a
    posed SMPL-X mesh maps directly to the 17 COCO joints in COCO order, matching
    the convention used by mesh-based multi-view methods.
    """
    global _COCO_REG
    if _COCO_REG is None:
        import pickle
        bm = Path(__file__).resolve().parent.parent / "body_models"
        jr = np.load(bm / "coco" / "J_regressor_coco.npy")             # (17, 6890)
        mtx = pickle.load(open(bm / "smplx2smpl.pkl", "rb"))["matrix"]  # (6890, 10475)
        _COCO_REG = (jr @ mtx).astype(np.float64)                       # (17, 10475)
    return _COCO_REG

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def se3_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch SE(3): find R (3×3), t (3,) minimising ||R @ src_i + t - dst_i||.

    src, dst : (N, 3) matched point sets.  N ≥ 2.
    Returns (R, t) such that R @ src[i] + t ≈ dst[i].
    """
    src_c = src.mean(0)
    dst_c = dst.mean(0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_c - R @ src_c
    return R, t


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Sim(3) alignment of *pred* to *gt*; returns aligned pred.

    pred, gt : (J, 3).
    """
    pred_c, gt_c = pred.mean(0), gt.mean(0)
    pred0, gt0   = pred - pred_c, gt - gt_c
    var_pred = (pred0 ** 2).sum()
    scale    = np.sqrt((gt0 ** 2).sum() / (var_pred + 1e-8))
    H        = pred0.T @ gt0
    U, _, Vt = np.linalg.svd(H)
    d        = np.linalg.det(Vt.T @ U.T)
    R        = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return scale * pred0 @ R.T + gt_c


# ---------------------------------------------------------------------------
# Rotation utilities (6D <-> axis-angle, matching fusion_dataset.py convention)
# ---------------------------------------------------------------------------

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> 6D (..., 6) using the first two rows of R."""
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)
    return sixd.reshape(shape + (6,)).astype(np.float32)


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """6D (..., 6) -> axis-angle (..., 3) via Gram-Schmidt then matrix->rotvec."""
    shape = sixd.shape[:-1]
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    aa = SciR.from_matrix(R).as_rotvec()
    return aa.reshape(shape + (3,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------

def _sixd_to_matrix(sixd: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). Rows are b1, b2, b3 — same convention as `_6d_to_aa`."""
    r0, r1 = sixd[..., :3], sixd[..., 3:]
    b1 = r0 / (r0.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def median_fuse(pose_t: torch.Tensor, mask_t: torch.Tensor,
                iters: int = 5, eps: float = 1e-3) -> torch.Tensor:
    """Fuse per-camera poses by the GEODESIC MEDIAN over the visible cameras.

    Drop-in replacement for `PoseFusionModule.forward`. Identical to
    `median_fuse` in evaluation/evaluate_egoexo_median.py.

    The chordal mean minimises sum_k ||R - R_k||_F^2 (L2), so a single badly
    wrong camera drags the estimate. The geodesic median minimises L1 on SO(3),

        R_bar = argmin_R  sum_k  w_k * d_geo(R, R_k),

    solved by Weiszfeld/IRLS: seed with the chordal mean, then repeatedly re-take
    a chordal mean with weights w_k / (theta_k + eps). Each step is the same
    closed-form SVD projection — no training, no parameters, no checkpoint.

    Selected on RICH's 12-scene train pool (-0.8 mm RR-MPJPE vs the chordal
    mean, beating every trained fusion module); applied here unchanged, so
    EgoExo4D stays a zero-shot report with no per-dataset tuning.

    Args:
        pose_t: (B, T, K, P, J, 6) per-camera SAM3D poses in 6D rotation form.
        mask_t: (B, T, K, P) 1.0 where camera k has a detection for person p at t.
        iters:  IRLS steps. 3-5 is converged; 5 used for every reported number.
        eps:    radians, guards 1/theta when a camera sits on the current estimate.

    Returns:
        (B, T, P, J, 6) — rows 0 and 1 of R_bar. R_bar is already orthonormal, so
        the Gram-Schmidt inside `_6d_to_aa` is a no-op on it.
    """
    R = _sixd_to_matrix(pose_t)                             # (B,T,K,P,J,3,3)
    J = R.shape[4]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)

    # No camera saw this (t, p): the mean matrix is all-zero and its SVD is
    # arbitrary. Seed the identity so the decomposition stays well-conditioned;
    # these slots carry no placed prediction downstream.
    empty = (mask_t.sum(dim=2) == 0)[..., None]             # (B,T,P,1) -> over J

    def _chordal(w: torch.Tensor) -> torch.Tensor:
        """Weighted chordal mean. w: (B,T,K,P,J) -> (B,T,P,J,3,3)."""
        ww = w[..., None, None]
        M = (R * ww).sum(dim=2) / ww.sum(dim=2).clamp_min(1e-8)
        if bool(empty.any()):
            M = torch.where(empty[..., None, None], eye.expand_as(M), M)
        U, _, Vh = torch.linalg.svd(M)
        d = torch.linalg.det(U @ Vh)                        # (B,T,P,J)  ±1
        D = eye.expand(*d.shape, 3, 3).clone()
        D[..., 2, 2] = d
        return U @ D @ Vh

    w0 = mask_t[..., None].expand(*mask_t.shape, J)         # (B,T,K,P,J)
    R_bar = _chordal(w0)                                    # chordal-mean seed
    for _ in range(iters):
        rel = R @ R_bar[:, :, None].transpose(-1, -2)       # (B,T,K,P,J,3,3)
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        theta = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))  # (B,T,K,P,J) rad
        R_bar = _chordal(w0 / (theta + eps))

    return torch.cat([R_bar[..., 0, :], R_bar[..., 1, :]], dim=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt(gt_scene_dir: Path):
    """Load GT body joints and camera positions for one scene.

    Returns
    -------
    frame_idx : int
        The annotated frame index (key of keypoints_gt.json).
    gt_joints : dict[str, np.ndarray]
        Joint name → (3,) xyz in GT world frame (metres).  Only joints with
        num_views_for_3d > 0 and that appear in GT_TO_COCO are included.
    cam_pos_gt : dict[str, np.ndarray]
        Camera ID → (3,) camera centre in GT world frame (metres).
    """
    kp_path  = gt_scene_dir / "keypoints_gt.json"
    cal_path = gt_scene_dir / "gopro_calibs.csv"

    with open(kp_path) as f:
        kp_raw = json.load(f)

    frame_idx_str, joints_raw = next(iter(kp_raw.items()))
    frame_idx = int(frame_idx_str)

    gt_joints: dict[str, np.ndarray] = {}
    for name, val in joints_raw.items():
        if name in GT_TO_COCO and val.get("num_views_for_3d", 0) > 0:
            gt_joints[name] = np.array([val["x"], val["y"], val["z"]], dtype=np.float64)

    cam_pos_gt: dict[str, np.ndarray] = {}
    with open(cal_path) as f:
        for row in csv.DictReader(f):
            cam_pos_gt[row["cam_uid"]] = np.array(
                [float(row["tx_world_cam"]),
                 float(row["ty_world_cam"]),
                 float(row["tz_world_cam"])],
                dtype=np.float64,
            )

    return frame_idx, gt_joints, cam_pos_gt


def _parse_reid_tokens(tokens: list[str]) -> dict[str, int]:
    """Parse [cXpY, ...] into {cam_name: ghost_pid}, where cX → 'cam{X:02d}'."""
    out: dict[str, int] = {}
    for tok in tokens:
        m = re.match(r"c(\d+)p(\d+)", tok)
        if m:
            out[f"cam{int(m.group(1)):02d}"] = int(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Automatic GT-subject identification (replaces manual ReID)
#
# Manual ReID was only ever needed because the extracted frames were the wrong
# ones (a decoder bug returned the preceding keyframe), so the GT skeleton did
# not land on the annotated subject and could not be matched automatically.  With
# the correct frame the GT reprojects onto the subject, so the correspondence can
# be recovered — and, crucially, *verified* — without human labelling.
# ---------------------------------------------------------------------------

def _gopro_cameras(gt_scene_dir: Path) -> dict[str, dict]:
    """Undistorted-pinhole gopro cameras, scaled to the 1440-wide frame space
    that ``pred_keypoints_2d`` live in."""
    cams: dict[str, dict] = {}
    with open(gt_scene_dir / "gopro_calibs.csv") as f:
        for row in csv.DictReader(f):
            K = np.array([[float(row["intrinsics_0"]), 0, float(row["intrinsics_2"])],
                          [0, float(row["intrinsics_1"]), float(row["intrinsics_3"])],
                          [0, 0, 1]])
            D = np.array([[float(row[f"intrinsics_{i}"])] for i in range(4, 8)])
            W, H = int(row["image_width"]), int(row["image_height"])
            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (W, H), np.eye(3), balance=0.0)
            t   = np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"])
            Rwc = SciR.from_quat(
                [float(row[f"q{a}_world_cam"]) for a in ["x", "y", "z", "w"]]).as_matrix()
            sx, sy = 1440 / W, 810 / H
            cams[row["cam_uid"]] = {
                "Knew": Knew, "Rwc": Rwc, "t": t, "sx": sx, "sy": sy,
                "P": (np.diag([sx, sy, 1.0]) @ Knew)
                     @ np.hstack([Rwc.T, (-Rwc.T @ t).reshape(3, 1)]),
            }
    return cams


def _project(cam: dict, X: np.ndarray) -> np.ndarray | None:
    """Project a world point; None if it falls behind the camera."""
    Xc = cam["Rwc"].T @ (X - cam["t"])
    if Xc[2] <= 0.05:
        return None
    return np.array([(cam["Knew"][0, 0] * Xc[0] / Xc[2] + cam["Knew"][0, 2]) * cam["sx"],
                     (cam["Knew"][1, 1] * Xc[1] / Xc[2] + cam["Knew"][1, 2]) * cam["sy"]])


def _dlt(cams: list[dict], uvs: list[np.ndarray]) -> np.ndarray:
    A: list[np.ndarray] = []
    for cam, uv in zip(cams, uvs):
        P = cam["P"]
        A += [uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def auto_match_gt_subject(
    ghost_scene_dir: Path,
    gt_scene_dir: Path,
    frame_idx: int,
    gt_joints: dict[str, np.ndarray],
) -> tuple[dict[str, int] | None, str]:
    """Identify the annotated GT subject in every camera, or refuse.

    Reprojects the GT skeleton into each gopro camera and takes the nearest
    detected person, accepting only when the choice is provably unambiguous:

    1. the best candidate must beat the runner-up by ``_MATCH_RATIO`` — a
       near-tie (e.g. a dance partner) is refused, never guessed;
    2. the accepted cameras must triangulate to a *single* 3D point (residual
       <= ``_MATCH_RES_PX``); while inconsistent the worst camera is dropped,
       which removes miscalibrated views automatically;
    3. that point must land within ``_MATCH_OFF_CM`` of the GT skeleton.

    Returns ({cam_name: disk_pid}, info) on success, else (None, reason).
    """
    cams = _gopro_cameras(gt_scene_dir)
    picks: dict[str, tuple[float, int, dict[str, np.ndarray]]] = {}
    notes: list[str] = []

    for cam_id, cam in cams.items():
        body_dir = ghost_scene_dir / cam_id / "body_data"
        if not body_dir.exists():
            continue
        gt_uv = {n: _project(cam, gt_joints[n]) for n in gt_joints if n in GT_TO_MHR70}
        use = [n for n, uv in gt_uv.items() if uv is not None]
        if not use:
            continue
        cands: list[tuple[float, int, dict[str, np.ndarray]]] = []
        for npz in sorted(body_dir.glob("person_*.npz")):
            d = np.load(str(npz), allow_pickle=False)
            idx = np.where(d["frame_indices"].astype(int) == frame_idx)[0]
            if not idx.size:
                continue
            k = d["pred_keypoints_2d"][idx[0]]
            err = float(np.mean([np.linalg.norm(gt_uv[n] - k[GT_TO_MHR70[n]]) for n in use]))
            cands.append((err, int(npz.stem.split("_")[1]), {n: k[GT_TO_MHR70[n]] for n in use}))
        if not cands:
            continue
        cands.sort(key=lambda c: c[0])
        if len(cands) == 1 or cands[0][0] < _MATCH_RATIO * cands[1][0]:
            picks[cam_id] = cands[0]
        else:
            notes.append(f"{cam_id}:tie(p{cands[0][1]}@{cands[0][0]:.0f} "
                         f"vs p{cands[1][1]}@{cands[1][0]:.0f})")

    if len(picks) < 2:
        return None, "; ".join(notes) or "<2 unambiguous cameras"

    active = list(picks)
    while len(active) >= 2:
        shared = set.intersection(*[set(picks[c][2]) for c in active])
        if not shared:
            return None, "no shared joints across cameras"
        res: list[float] = []
        off: list[float] = []
        for n in shared:
            X = _dlt([cams[c] for c in active], [picks[c][2][n] for c in active])
            rr = [np.linalg.norm(p - picks[c][2][n])
                  for c in active if (p := _project(cams[c], X)) is not None]
            if rr:
                res.append(float(np.mean(rr)))
            off.append(float(np.linalg.norm(X - gt_joints[n]) * 100))
        if not res:
            return None, "triangulation degenerate (behind camera)"
        m_res, m_off = float(np.mean(res)), float(np.mean(off))
        if m_res <= _MATCH_RES_PX and m_off <= _MATCH_OFF_CM:
            info = f"cams={len(active)} res={m_res:.0f}px off={m_off:.0f}cm"
            if notes:
                info += " (" + "; ".join(notes) + ")"
            return {c: picks[c][1] for c in active}, info
        if len(active) == 2:
            return None, (f"cams=2 res={m_res:.0f}px off={m_off:.0f}cm "
                          + "; ".join(notes)).strip()
        worst = max(active, key=lambda c: picks[c][0])
        active.remove(worst)
        notes.append(f"dropped {worst}")
    return None, "; ".join(notes) or "exhausted cameras"


# ---------------------------------------------------------------------------
# Inference: fusion module -> Procrustes DLT placement -> SMPL-X FK
#
# Mirrors evaluate_on_rich_test.py's pipeline (fuse multi-view body pose, then
# estimate metric root translation + global orient via BodyPlacer), specialised
# for EgoExo4D's single annotated GT frame. Person identity across cameras comes
# from manual_reid (cross-view ReID is unreliable here), fed to BodyPlacer via
# its cam_pid_remap argument so triangulation reads the right per-camera person.
# ---------------------------------------------------------------------------

def _load_remapped_raw(
    ghost_scene_dir: Path,
    cam_names: list[str],
    valid_mask: np.ndarray,
    global_groups: dict[int, dict[str, int]],
    gt_frame_idx: int,
) -> tuple[list[dict[int, dict]], dict[int, np.ndarray]]:
    """Per (valid) camera, load the GT-frame SMPL-X params for each global pid.

    Returns (raw, betas_by_pid):
      raw          : list over cameras of {global_pid: {"go": (3,), "bp": (63,)}}
      betas_by_pid : {global_pid: (10,)} mean SAM3D betas across cameras.
    """
    raw: list[dict[int, dict]] = []
    betas_acc: dict[int, list[np.ndarray]] = {pid: [] for pid in global_groups}
    for k, cam_id in enumerate(cam_names):
        cam_map: dict[int, dict] = {}
        if valid_mask[k]:
            body_dir = ghost_scene_dir / cam_id / "body_data"
            for gpid, cam_disk in global_groups.items():
                disk_pid = cam_disk.get(cam_id)
                if disk_pid is None:
                    continue
                f = body_dir / f"person_{disk_pid}.npz"
                if not f.exists():
                    continue
                d  = np.load(str(f), allow_pickle=False)
                fi = d["frame_indices"].astype(int)
                idx = np.where(fi == gt_frame_idx)[0]
                if not idx.size:
                    continue
                t = int(idx[0])
                entry = {
                    "go": d["smplx_global_orient"][t].reshape(3),
                    "bp": d["smplx_body_pose"][t].reshape(63),
                }
                # Hand poses must be included: the fusion model was trained with
                # all 54 joints (body+hands) populated, so omitting hands feeds it
                # out-of-distribution zeros and corrupts the body-joint output.
                if "smplx_left_hand_pose" in d.files:
                    entry["lh"] = d["smplx_left_hand_pose"][t].reshape(-1, 3)   # (15, 3)
                if "smplx_right_hand_pose" in d.files:
                    entry["rh"] = d["smplx_right_hand_pose"][t].reshape(-1, 3)  # (15, 3)
                cam_map[gpid] = entry
                if "smplx_betas" in d.files:
                    betas_acc[gpid].append(d["smplx_betas"][t][:10])
        raw.append(cam_map)
    betas_by_pid = {
        pid: (np.mean(v, 0).astype(np.float32) if v else np.zeros(10, np.float32))
        for pid, v in betas_acc.items()
    }
    return raw, betas_by_pid


def _build_single_frame_tensors(
    raw: list[dict[int, dict]],
    all_pids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Assemble (1, 1, K, P, 54, 6) pose + (1, 1, K, P) mask for the fusion model.

    Single frame (T=1): EgoExo4D body GT is one annotated frame per scene. Pose
    is 6D, root excluded, matching build_fusion_tensors in the RICH eval.
    """
    K = len(raw)
    P = len(all_pids)
    J = _NUM_JOINTS - 1   # 54, root excluded
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    pose = np.zeros((1, K, P, J, 6), dtype=np.float32)
    mask = np.zeros((1, K, P),       dtype=np.float32)
    for k, cam_map in enumerate(raw):
        for pid, pd in cam_map.items():
            p  = pid_to_slot[pid]
            # Pack joints in the SAME order as the RICH training pipeline
            # (build_fusion_tensors): [global_orient, body(21), lhand(15), rhand(15)],
            # then pad to 55 and drop the root. Hands are required (see _load_remapped_raw).
            parts = [pd["go"].reshape(1, 3), pd["bp"].reshape(21, 3)]
            if "lh" in pd:
                parts.append(pd["lh"].reshape(15, 3))
            if "rh" in pd:
                parts.append(pd["rh"].reshape(15, 3))
            aa = np.concatenate(parts, 0)
            if aa.shape[0] < _NUM_JOINTS:
                aa = np.concatenate(
                    [aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3), dtype=np.float32)], 0
                )
            pose[0, k, p] = _aa_to_6d(aa)[1:]   # (54, 6), root excluded
            mask[0, k, p] = 1.0
    return (
        torch.from_numpy(pose).unsqueeze(0),   # (1, T=1, K, P, 54, 6)
        torch.from_numpy(mask).unsqueeze(0),   # (1, T=1, K, P)
        pid_to_slot,
    )


def _gt_scale(placer, cam_pos_gt: dict[str, np.ndarray]) -> float | None:
    """Oracle metric scale (metres per VGGT unit): Sim(3) scale aligning the
    *unscaled* VGGT camera centres to the GT camera centres."""
    names = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    pred, gt = [], []
    for k, cam in enumerate(names):
        if cam not in cam_pos_gt or not placer.cam_valid[:, k].any():
            continue
        E = placer.extrinsics[0, k]
        R, t = E[:3, :3], E[:3, 3]
        pred.append(-R.T @ t)              # VGGT camera centre (unscaled units)
        gt.append(cam_pos_gt[cam])
    if len(pred) < 2:
        return None
    pred = np.stack(pred).astype(np.float64)
    gt   = np.stack(gt).astype(np.float64)
    p0, g0 = pred - pred.mean(0), gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    return float((S * [1, 1, d]).sum() / ((p0 ** 2).sum() + 1e-12))


def _gt_camera_extrinsics(
    gt_scene_dir: Path,
    camera_names,
    T: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """GT GoPro extrinsics as (T, K, 3, 4), re-rooted to ``camera_names[0]``.

    gopro_calibs.csv stores cam->world (q_world_cam, t_world_cam), so the
    world->cam extrinsic of camera k is  E_k = [R_k^T | -R_k^T t_k].  BodyPlacer
    expects camera 0 to be the reference ([I|0]), which is how the VGGT cameras
    are stored, so every camera is re-rooted through camera 0:

        R_rel = R_k @ R_0^T ,   t_rel = t_k - R_k @ R_0^T @ t_0

    Returns (extrinsics, filled) where ``filled`` (K,) marks cameras that have a
    GT calibration; unfilled cameras stay all-zero and MUST be masked out of
    cam_valid or the DLT builds a degenerate projection matrix from them.
    """
    names = [n.decode() if isinstance(n, bytes) else n for n in camera_names]
    w2c: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(gt_scene_dir / "gopro_calibs.csv") as f:
        for row in csv.DictReader(f):
            R_c2w = SciR.from_quat(
                [float(row[f"q{a}_world_cam"]) for a in ["x", "y", "z", "w"]]
            ).as_matrix()
            t_c2w = np.array([float(row[f"t{a}_world_cam"]) for a in "xyz"])
            w2c[row["cam_uid"]] = (R_c2w.T, -R_c2w.T @ t_c2w)

    if not names or names[0] not in w2c:
        return None, None
    R0, t0 = w2c[names[0]]

    K_ = len(names)
    exts = np.zeros((T, K_, 3, 4), dtype=np.float64)
    filled = np.zeros(K_, dtype=bool)
    for k, cn in enumerate(names):
        if cn not in w2c:
            continue
        Rk, tk = w2c[cn]
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        exts[:, k] = np.hstack([R_rel, t_rel[:, None]])[None]
        filled[k] = True
    return exts.astype(np.float32), filled


def run_fusion_placer(
    ghost_scene_dir: Path,
    gt_scene_dir: Path,
    gt_frame_idx: int,
    global_groups: dict[int, dict[str, int]],
    device: torch.device,
    smplx_arg,
    cam_pos_gt: dict[str, np.ndarray],
    modalities: list[int],
) -> dict[int, dict]:
    """Fuse multi-view pose once, then place it under each oracle modality.

    Returns {modality: {"persons": [(global_pid, coco_world (17, 3))],
                        "scale": float, "extrinsics": (K, 3, 4)}}.
    ``extrinsics``/``scale`` are the ones the placer actually used, so the
    caller's SE(3) camera alignment can be built from the same rig.
    """
    vggt       = np.load(ghost_scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names  = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    valid_mask = vggt["valid"][0]

    raw, betas_by_pid = _load_remapped_raw(
        ghost_scene_dir, cam_names, valid_mask, global_groups, gt_frame_idx
    )
    if not any(raw):
        return {}
    all_pids = sorted(global_groups)

    # ── Fusion: geodesic median over K cameras for the single frame ───────────
    # No temporal padding: median_fuse has no cross-frame coupling, so running
    # it directly at T=1 is provably identical to padding first (each (t,p,j)
    # is computed independently over cameras — see evaluate_egoexo_median.py).
    pose_t, mask_t, pid_to_slot = _build_single_frame_tensors(raw, all_pids)
    with torch.no_grad():
        fused_t = median_fuse(pose_t.to(device), mask_t.to(device))
    fused = fused_t[0, :1].cpu().numpy()   # (1, P, 54, 6) — the annotated frame

    # ── Procrustes DLT placement (translation + global orient) ────────────────
    cam_pid_remap: dict[str, dict[int, int]] = {}
    for pid, cam_disk in global_groups.items():
        for cam, disk in cam_disk.items():
            cam_pid_remap.setdefault(cam, {})[pid] = disk

    placer = BodyPlacer(
        ghost_scene_dir, smplx_arg, crop_meta_path=None, cam_pid_remap=cam_pid_remap
    )
    fused_pose_by_pid = {pid: fused[:, pid_to_slot[pid]] for pid in all_pids}  # {pid: (1, 54, 6)}

    # ── Oracle inputs, shared by every rung ──────────────────────────────────
    # GT scale: the Sim(3) scale that maps the *unscaled* VGGT camera centres
    # onto the GT GoPro centres — the scale the pipeline's MapAnything estimator
    # is trying to predict. Every rung uses it (that is what makes M2 the
    # scale-free rung), so without it there is no ladder at all.
    s_gt = _gt_scale(placer, cam_pos_gt)
    if not s_gt:
        logger.error(f"{ghost_scene_dir.name}: oracle scale unavailable "
                     "(<2 cameras matched to GT) — skipping")
        return {}
    gt_ext, gt_filled = _gt_camera_extrinsics(gt_scene_dir, placer.camera_names, placer.T)

    # BodyPlacer loads its own extrinsics at construction, so the oracle
    # substitution has to be patched onto THIS instance and restored afterwards.
    orig_ext   = placer.extrinsics
    orig_valid = placer.cam_valid.copy()

    out: dict[int, dict] = {}
    for modality in modalities:
        use_gt_cams = modality == 3
        if use_gt_cams:
            if gt_ext is None:
                logger.warning(f"{ghost_scene_dir.name}: [M{modality}] no GT calibration for "
                               "the reference camera — skipping this rung")
                continue
            placer.extrinsics = gt_ext
            # GT extrinsics are already metric, so the subject scale is exactly 1.0.
            scale_used = 1.0
            # Cameras without GT calibration are all-zero: mask them out so the
            # DLT never builds a projection matrix from a zero extrinsic.
            placer.cam_valid = orig_valid & gt_filled[np.newaxis, :]
        else:
            placer.extrinsics = orig_ext
            placer.cam_valid  = orig_valid
            scale_used = float(s_gt)

        # The SAME scalar must scale the camera rig in eval_scene; W-MPJPE aligns
        # with SE(3) (no scale), so any mismatch between the subject's scale and
        # the rig's displaces the body by their ratio and corrupts the metric.
        scale = np.full(placer.T, scale_used, dtype=np.float32)
        try:
            trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
                scale=scale,
                all_pids=set(all_pids),
                pred_betas_by_pid=betas_by_pid,
                fused_pose_by_pid=fused_pose_by_pid,
                frame_start=gt_frame_idx,
            )
        except Exception as e:
            logger.warning(f"{ghost_scene_dir.name}: [M{modality}] placer failed — {e}")
            continue
        finally:
            placer.extrinsics = orig_ext      # always restore
            placer.cam_valid  = orig_valid

        # ── SMPL-X FK: pivot canonical joints about the SMPL-X pelvis, place ──
        persons: list[tuple[int, np.ndarray]] = []
        for pid in all_pids:
            slot         = pid_to_slot[pid]
            pelvis_world = trans_dict.get(pid, {}).get(gt_frame_idx)
            R_mat        = orient_dict.get(pid, {}).get(gt_frame_idx)
            if pelvis_world is None or R_mat is None:
                continue
            bp_aa   = _6d_to_aa(fused[0, slot, :21]).reshape(63)
            betas_p = betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))
            J55, verts = placer._smplx_fk(
                betas_p[np.newaxis], bp_aa[np.newaxis],
                np.zeros((1, 3), dtype=np.float32), return_verts=True,
            )
            J_can      = J55[0].astype(np.float64)         # (55, 3) canonical, root-pivoted
            pelvis     = J_can[0]                           # SMPL-X root = placement pivot
            coco_can   = coco_regressor() @ verts[0].astype(np.float64)    # (17, 3) canonical
            coco_world = (R_mat @ (coco_can - pelvis).T).T + pelvis_world  # (17, 3) metric cam-0
            persons.append((pid, coco_world))

        if persons:
            out[modality] = {
                "persons":    persons,
                "scale":      scale_used,
                "extrinsics": (gt_ext[0] if use_gt_cams else orig_ext[0]),
            }
    return out


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(
    ghost_scene_dir: Path,
    gt_scene_dir: Path,
    device: torch.device,
    smplx_arg,
    reid_groups: dict[str, list[str]] | None = None,
    modalities: list[int] | None = None,
) -> dict | None:
    """Evaluate one scene through the oracle ladder (fusion + Procrustes DLT).

    reid_groups : {global_pid_str: [cXpY, ...]} from manual_reid.json — the
        per-camera identities for every annotated global person in the scene.
        Required: the placer triangulates per global pid, and cross-view ReID is
        unreliable on EgoExo4D, so manual identities are mandatory here.

    Returns dict with keys scene, n_joints and, per rung, m<N>_w_mpjpe (mm),
    m<N>_pa_mpjpe (mm), m<N>_pa_per_joint.  None if the scene is skipped.
    """
    if modalities is None:
        modalities = list(_MODALITIES)
    for fname in ("vggt_cameras_centered.npz", "mapanything_scale_baseline.npy"):
        if not (ghost_scene_dir / fname).exists():
            logger.debug(f"{ghost_scene_dir.name}: missing {fname}, skip")
            return None
    if not gt_scene_dir.exists():
        logger.debug(f"{ghost_scene_dir.name}: no GT dir, skip")
        return None

    try:
        frame_idx, gt_joints, cam_pos_gt = load_gt(gt_scene_dir)
    except Exception as e:
        logger.warning(f"{ghost_scene_dir.name}: GT load error — {e}")
        return None

    if not gt_joints:
        logger.info(f"{ghost_scene_dir.name}: hand-only GT, skipping")
        return None

    # --- Resolve per-camera person identities -------------------------------
    # Default: automatic, self-verifying identification of the GT subject.
    # manual_reid.json, when supplied for a scene, overrides it.
    global_groups: dict[int, dict[str, int]] = {}
    if reid_groups:
        for gpid_str, tokens in reid_groups.items():
            cam_disk = _parse_reid_tokens(tokens)
            if cam_disk:
                global_groups[int(gpid_str)] = cam_disk
        if not global_groups:
            logger.warning(f"{ghost_scene_dir.name}: manual reid groups yielded empty map "
                           "— skipping")
            return None
        logger.debug(f"{ghost_scene_dir.name}: using manual reid override")
    else:
        picks, info = auto_match_gt_subject(
            ghost_scene_dir, gt_scene_dir, frame_idx, gt_joints
        )
        if picks is None:
            logger.warning(f"{ghost_scene_dir.name}: auto-match could not verify the GT "
                           f"subject ({info}) — skipping")
            return None
        global_groups = {GT_PERSON_PID: picks}
        logger.info(f"{ghost_scene_dir.name}: auto-matched GT subject — {info}")

    # --- Inference: fusion -> Procrustes DLT placement -> FK, per rung -------
    per_mod = run_fusion_placer(
        ghost_scene_dir, gt_scene_dir, frame_idx, global_groups, device,
        smplx_arg, cam_pos_gt, modalities,
    )
    if not per_mod:
        logger.warning(f"{ghost_scene_dir.name}: no placed predictions for frame {frame_idx}")
        return None

    cam_names = [n.decode() if isinstance(n, bytes) else n
                 for n in np.load(ghost_scene_dir / "vggt_cameras_centered.npz",
                                  allow_pickle=False)["camera_names"]]

    # --- Build GT joint array for evaluation subset -------------------------
    joint_names   = sorted(gt_joints.keys())
    coco_indices  = [GT_TO_COCO[n] for n in joint_names]
    gt_arr        = np.stack([gt_joints[n] for n in joint_names])  # (J, 3)

    res: dict = {"scene": ghost_scene_dir.name, "n_joints": len(joint_names)}

    for modality in sorted(per_mod):
        mod        = per_mod[modality]
        extrinsics = mod["extrinsics"]     # (K, 3, 4) — the rig the placer used
        scale_used = mod["scale"]

        # --- SE(3) alignment from the placer's cameras to the GT cameras -----
        # The rig MUST be scaled by the SAME scalar the placer used for the
        # subject.  W-MPJPE aligns with SE(3) (no scale), so a mismatch displaces
        # the body by the ratio of the two scales and corrupts the metric.  For
        # the GT-camera rungs the rig IS the GT rig (scale 1.0), so this SE(3)
        # simply recovers the reference-camera pose in the GT world frame.
        pred_centers, gt_centers = [], []
        for k, cam_id in enumerate(cam_names):
            if cam_id not in cam_pos_gt or not np.any(extrinsics[k]):
                continue
            R_k = extrinsics[k, :, :3]
            t_k = extrinsics[k, :, 3]
            pred_centers.append(((-R_k.T @ t_k) * scale_used).astype(np.float64))
            gt_centers.append(cam_pos_gt[cam_id])
        if len(pred_centers) < 2:
            logger.warning(f"{ghost_scene_dir.name}: [M{modality}] <2 alignable cameras, skip")
            continue
        R_align, t_align = se3_align(np.stack(pred_centers), np.stack(gt_centers))

        # --- Score the GT subject only --------------------------------------
        # The EgoExo body GT is ONE annotated person, labelled group
        # GT_PERSON_PID in manual_reid.  Score exactly that person — never an
        # argmin over all placed people, which would be an oracle that picks the
        # closest match.
        match = next((jw for pid, jw in mod["persons"] if pid == GT_PERSON_PID), None)
        if match is None:
            logger.warning(f"{ghost_scene_dir.name}: [M{modality}] GT subject (group "
                           f"{GT_PERSON_PID}) not placed "
                           f"(have {[p for p, _ in mod['persons']]}) — skipping")
            continue

        pred_j     = match[coco_indices].astype(np.float64)          # (J, 3) metric cam-0
        # W-MPJPE†: SE(3) camera-pose alignment (no scale), then measure error
        pred_world = pred_j @ R_align.T + t_align                    # (J, 3)
        w_mpjpe    = float(np.linalg.norm(pred_world - gt_arr, axis=-1).mean()) * 1000
        # PA-MPJPE: per-person Procrustes (Sim3) alignment
        pred_proc  = procrustes_align(pred_j, gt_arr)
        pa_per     = np.linalg.norm(pred_proc - gt_arr, axis=-1) * 1000   # (J,) mm per joint

        pfx = f"m{modality}_"
        res[pfx + "w_mpjpe"]      = w_mpjpe
        res[pfx + "pa_mpjpe"]     = float(pa_per.mean())
        res[pfx + "pa_per_joint"] = dict(zip(joint_names, pa_per.tolist()))

    return res if any(k.startswith("m") for k in res) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EgoExo4D progressive oracle ablations (M2, M3) for the ghost pipeline.")
    parser.add_argument("--ghost_root",  required=True,  help="ghost output root (egoexo4d/)")
    parser.add_argument("--gt_root",     required=True,  help="EgoExo4D GT root (contains per-scene dirs)")
    parser.add_argument("--smplx_model", default=str(_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl"),
                        help="Path to SMPLX_NEUTRAL.pkl (or folder containing it)")
    parser.add_argument("--modalities", default=",".join(str(m) for m in _MODALITIES),
                        help="Comma-separated subset of the oracle ladder to run: 2,3 "
                             "(default: all). There is no M4 — EgoExo4D has no GT body "
                             "parameters; see the module docstring.")
    parser.add_argument("--max_scenes",  type=int, default=None)
    parser.add_argument("--scene",       default=None, help="Evaluate a single scene by name")
    parser.add_argument("--reid_map",    default=str(_REPO_ROOT / "manual_reid.json"),
                        help="Path to manual_reid.json (only used with --use_manual_reid)")
    parser.add_argument("--use_manual_reid", action="store_true", default=False,
                        help="Override automatic GT-subject identification with "
                             "manual_reid.json. NOTE: those person ids are tied to a "
                             "specific pipeline run and go stale whenever body_data is "
                             "regenerated; the automatic matcher is self-verifying and "
                             "is the default.")
    args = parser.parse_args()

    args.modalities = [int(x.strip()) for x in args.modalities.split(",") if x.strip()]
    bad = [m for m in args.modalities if m not in _MODALITIES]
    if bad:
        parser.error(f"--modalities: {bad} not in {list(_MODALITIES)}")
    if not args.modalities:
        parser.error("--modalities: empty")

    ghost_root = Path(args.ghost_root)
    gt_root    = Path(args.gt_root)

    # Load manual reid mapping (egoexo4d section): {scene: {global_pid_str: [cXpY,...]}}.
    # All annotated global persons are kept (not just person 1) so the placer has
    # the full multi-person context and GT can match the best person.
    reid_map_egoexo: dict[str, dict[str, list[str]]] = {}
    if args.use_manual_reid:
        reid_map_path = Path(args.reid_map)
        if reid_map_path.exists():
            with open(reid_map_path) as f:
                reid_raw = json.load(f)
            for scene_name, entry in reid_raw.get("egoexo4d", {}).items():
                groups = entry.get("groups", {})
                if groups:
                    reid_map_egoexo[scene_name] = groups
            logger.warning(
                f"--use_manual_reid: overriding auto-match for {len(reid_map_egoexo)} scenes. "
                "These person ids are only valid for the pipeline run they were made on."
            )
        else:
            logger.warning(f"manual_reid.json not found at {reid_map_path} — "
                           "falling back to automatic matching")
    else:
        logger.info("GT subject identified automatically (self-verifying matcher); "
                    "pass --use_manual_reid to override with manual_reid.json")

    # Fusion: geodesic median over cameras (no checkpoint). The placer builds
    # its own SMPL-X model from the path.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx_arg = args.smplx_model
    logger.info(f"Device: {device}")
    logger.info("Fusion: geodesic median over cameras, L1/Weiszfeld (no checkpoint)")

    if args.scene:
        scene_dirs = [ghost_root / args.scene]
    else:
        scene_dirs = sorted(ghost_root.iterdir())
        if args.max_scenes:
            scene_dirs = scene_dirs[: args.max_scenes]

    results = []
    skipped_hand = 0
    skipped_missing = 0

    # Broken-GT takes are still evaluated; they are split out in the summary so
    # both the full-set and the cleaned number are reported from a single run.
    for ghost_scene_dir in scene_dirs:
        if not ghost_scene_dir.is_dir():
            continue
        gt_scene_dir = gt_root / ghost_scene_dir.name
        reid_groups = reid_map_egoexo.get(ghost_scene_dir.name)
        res = eval_scene(ghost_scene_dir, gt_scene_dir, device, smplx_arg,
                         reid_groups, args.modalities)
        if res is None:
            if not gt_scene_dir.exists():
                skipped_missing += 1
            elif not (ghost_scene_dir / "vggt_cameras_centered.npz").exists():
                skipped_missing += 1
            else:
                skipped_hand += 1
            continue
        results.append(res)
        cells = "  ".join(
            f"M{m}: W={res[f'm{m}_w_mpjpe']:6.1f} PA={res[f'm{m}_pa_mpjpe']:5.1f}"
            for m in sorted(args.modalities) if f"m{m}_w_mpjpe" in res
        )
        logger.info(f"  {res['scene']:45s}  {cells}  ({res['n_joints']}j)")

    if not results:
        logger.error("No scenes evaluated.")
        return

    kept    = [r for r in results if r["scene"] not in EXCLUDED_TAKES]
    dropped = [r for r in results if r["scene"] in EXCLUDED_TAKES]

    mods_run = sorted(args.modalities)

    def _report(label: str, rows: list[dict]) -> None:
        if not rows:
            print(f"  {label}: no scenes")
            return
        print(f"  {label}  (n={len(rows)})")
        # MEAN is the headline: CHROMM / HSfM report means, so it is the
        # comparable figure. Median is kept alongside as a robustness check.
        # Scenes are only counted where that rung actually produced a number, so
        # n can differ per rung; it is printed to keep the columns honest.
        for m in mods_run:
            w  = [r[f"m{m}_w_mpjpe"]  for r in rows if f"m{m}_w_mpjpe"  in r]
            pa = [r[f"m{m}_pa_mpjpe"] for r in rows if f"m{m}_pa_mpjpe" in r]
            if not w:
                print(f"    {_MODALITY_LABELS[m]:<40s}  — no scenes")
                continue
            print(f"    {_MODALITY_LABELS[m]:<40s}  (n={len(w)})")
            print(f"      W-MPJPE†  MEAN: {np.mean(w):7.1f} mm   (median {np.median(w):6.1f})")
            print(f"      PA-MPJPE  MEAN: {np.mean(pa):7.1f} mm   (median {np.median(pa):6.1f})")

    print("\n" + "=" * 60)
    print(f"EgoExo4D ORACLE ABLATIONS — {len(results)} body-GT scenes  "
          f"[modalities={','.join(str(m) for m in mods_run)}]")
    print("  Oracle numbers — NOT comparable to CHROMM/HSfM. Production number:")
    print("  evaluation/evaluate_egoexo.py")
    print(f"  Skipped (hand-only GT): {skipped_hand}")
    print(f"  Skipped (missing files): {skipped_missing}")
    print("-" * 60)
    _report("ALL evaluated scenes", results)
    print()
    _report("EXCLUDING broken GT/calibration", kept)
    if dropped:
        print(f"\n  Excluded takes ({len(dropped)}) — broken GT/calibration, "
              f"reported for transparency:")
        for r in sorted(dropped, key=lambda x: x["scene"]):
            cells = "  ".join(
                f"M{m}: W={r[f'm{m}_w_mpjpe']:7.1f} PA={r[f'm{m}_pa_mpjpe']:6.1f}"
                for m in mods_run if f"m{m}_w_mpjpe" in r
            )
            print(f"    {r['scene']:24s} {cells}  — {EXCLUDED_TAKES[r['scene']]}")
    missing_excluded = [t for t in EXCLUDED_TAKES if t not in {r["scene"] for r in results}]
    if missing_excluded:
        print(f"  (excluded takes not present in results: {missing_excluded})")
    print("-" * 60)
    # Per-joint diagnostic uses the cleaned set: the excluded takes fail in the
    # legs specifically (mistriangulated knees/ankles, joints behind the camera),
    # which would distort exactly the per-joint signal this table is read for.
    # PA is placement-invariant and the fused pose is shared by every rung, so
    # this table is identical across M2/M3 — it is printed once, from the lowest
    # rung that produced numbers.
    m_ref = next((m for m in mods_run if any(f"m{m}_pa_per_joint" in r for r in kept)), None)
    if m_ref is not None:
        print(f"  PA-MPJPE per joint (mean over scenes, EXCLUDING broken GT/calibration, "
              f"M{m_ref}):")
        per_joint: dict[str, list[float]] = {}
        for r in kept:
            for n, e in r.get(f"m{m_ref}_pa_per_joint", {}).items():
                per_joint.setdefault(n, []).append(e)
        for n in sorted(per_joint, key=lambda k: -np.mean(per_joint[k])):
            print(f"    {n:16s} {np.mean(per_joint[n]):6.1f} mm  (n={len(per_joint[n])})")
    print("=" * 60)

    print("\nReading the ladder:")
    print("  M2 - production  = metric-scale error (MapAnything)")
    print("  M3 - M2          = camera-pose error (VGGT)")
    print("  M3               = placement-given-perfect-cameras + pose error;")
    print("                     the pose part is PA-MPJPE (placement-invariant)")
    print("  Production (no oracle): evaluation/evaluate_egoexo.py")


if __name__ == "__main__":
    main()
