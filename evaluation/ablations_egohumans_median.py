#!/usr/bin/env python
"""EgoHumans progressive oracle ablations — error attribution for the ghost
pipeline, GEODESIC-MEDIAN FUSION (no checkpoint).

Copy of evaluation/ablations_egohumans.py with the learned PoseFusionModule
swapped for the geodesic median of the per-camera SAM3D poses (median_fuse,
identical to evaluation/evaluate_egohumans_median.py): the L1 estimator on
SO(3), solved by Weiszfeld/IRLS from a chordal-mean seed. This is the shipped
fusion rule, so this ladder is the one that matches the reported production
numbers. The median is independent across frames, so unlike the learned
module there is no temporal window, no OOD behaviour at temporal length 1,
and hence no `--temporal` flag and no TEMPORAL_PAD replication trick — those
existed only to work around the checkpoint being trained with a 128-frame
window.

Companion of evaluation/ablations.py (RICH). Each rung substitutes one more
component with ground truth, so total error can be attributed to a stage:

  M2: pred-cam + GT-scale + pred-pose   (removes scale error)
  M3: GT-cam   + GT-scale + pred-pose   (removes camera error too)
  M4: GT-cam   + GT-scale + GT-pose     (full oracle — placement/fusion ceiling)

Reading the table: M2 - prod = scale error, M3 - M2 = camera error,
M4 - M3 = pose/fusion error.

Oracle sources
--------------
GT scale : Sim(3) scale mapping the *unscaled* VGGT camera centres onto the GT
           exo camera centres (aria world) — exactly what MapAnything predicts.
GT cams  : colmap/workplace/images.txt (world->cam per exo camera), lifted into
           the aria01 world via colmap_from_aria_transforms.pkl, then re-rooted
           to camera_names[0] (the placer's reference).  Metric scale is then 1.0.
GT pose  : processed_data/smpl/<frame>.npy ``body_pose`` (69 = 23 SMPL joints).
           SMPL-X body_pose is the first 21 of those joints, i.e. body_pose[:63]
           (SMPL joints 22/23 are the hands, which SMPL-X models separately).
           GT ``betas`` are NOT substituted: they live in the SMPL shape space
           and would be wrong as SMPL-X betas, so M4 is a pure POSE oracle with
           predicted shape.
GT intrinsics are NOT substituted (same choice as the RICH ladder): the placer
re-maps pred_keypoints_2d into VGGT image space, so VGGT intrinsics are the ones
consistent with the observations.

Scenes without colmap GT cameras (no images.txt / colmap_from_aria pkl) are
skipped entirely — every rung needs GT camera centres, M2 included, because the
GT scale is derived from them.  As of 2026-07-24 that is 110/133 scenes
(missing: 040-061_badminton, 010_basketball).

For the honest production number see evaluation/evaluate_egohumans_median.py.

Metrics (metres, per the CHROMM paper "single-frame setting" on EgoHumans):
  W-MPJPE†  — world MPJPE after aligning ONLY the camera poses with SE(3)
              (no scale). Per-frame.  Needs GT cameras -> world frame.
  GA-MPJPE  — a single Sim(3) over ALL persons jointly, per frame
              (group-aligned; keeps inter-person geometry).
  PA-MPJPE  — per-person Procrustes (Sim3) per frame (local pose only).

Every frame is treated as a standalone scene: NO temporal attention in the
fusion module (frames go in the batch dim, temporal length 1).

Predictions come from ``<cam>/body_data_clean/person_<GLOBAL>.npz`` — already
within-view-ops'd and globally re-id'd, so global person id N == GT aria0N.
SMPL-X body pose -> SMPL-X FK verts -> SMPL verts (smplx2smpl) -> COCO-17
(J_regressor_coco), matching the GT ``poses3d`` COCO-17 joints.

World frame = aria01 SLAM world (GT ``poses3d`` is native there; the GT exo
cameras from ``images.txt`` (colmap world) are transformed into it via
``colmap_from_aria['aria01']^-1``).

Two stages so the 1.5 h walltime cannot lose work:
  Stage A (GPU): per scene, dump ``<dump_dir>/<scene>.npz`` =
      {pred_m2 / pred_m3 / pred_m4 (T,P,24,3) in aria world, gt (T,P,24,3), ...}.
      Scenes with an existing dump are skipped -> relaunch until all done.
  Stage B (CPU): ``--metrics_only`` loads every dump and prints the table.

NOTE the GT root must be the one that actually carries processed_data/smpl —
/iopsstor/scratch/cscs/tnanni/egohumans_gt_full/<activity>.  The activity sqsh
images and capstor/.../datasets/egohumans have EMPTY smpl dirs, which silently
costs the M4 rung.

Usage
-----
  # single scene (prints metrics inline; needs GT smpl + colmap)
  pixi run python evaluation/ablations_egohumans_median.py \
      --ghost_root /iopsstor/.../egohumans_new/06_badminton \
      --gt_root    /iopsstor/.../egohumans_gt_full/06_badminton \
      --scene 031_badminton --dump_dir eval_ablations_egohumans/dumps_median

  # full run (resumable), then aggregate
  pixi run python evaluation/ablations_egohumans_median.py --ghost_root ... --gt_root ... \
      --dump_dir eval_ablations_egohumans/dumps_median
  pixi run python evaluation/ablations_egohumans_median.py --metrics_only \
      --dump_dir eval_ablations_egohumans/dumps_median
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_egohumans")

_NUM_JOINTS = 55                       # SMPL-X joints fed to fusion (root + 54)

# Oracle ladder run by this script. See the module docstring.
_MODALITIES = (2, 3, 4)
_MODALITY_LABELS = {
    2: "M2: pred-cam + GT-scale + pred-pose",
    3: "M3: GT-cam   + GT-scale + pred-pose",
    4: "M4: GT-cam   + GT-scale + GT-pose  (oracle)",
}

# COCO-17: 0 nose,1/2 eyes,3/4 ears,5/6 shoulders,7/8 elbows,9/10 wrists,
#          11/12 hips,13/14 knees,15/16 ankles. Score the 12 limb joints only
#          (face joints have no clean SMPL-X correspondence).
_LIMB_COCO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
_SMPL_EVAL = list(range(24))          # all 24 SMPL joints (GT is SMPL; see smpl24_regressor)
_COCO_REG = None


def coco_regressor() -> np.ndarray:
    """(17, 10475): SMPL-X verts -> COCO-17 via smplx2smpl then J_regressor_coco."""
    global _COCO_REG
    if _COCO_REG is None:
        bm = _REPO_ROOT / "body_models"
        jr = np.load(bm / "coco" / "J_regressor_coco.npy")                 # (17, 6890)
        mtx = pickle.load(open(bm / "smplx2smpl.pkl", "rb"))["matrix"]      # (6890, 10475)
        _COCO_REG = (jr @ mtx).astype(np.float64)
    return _COCO_REG


_SMPL_REG = None
def smpl24_regressor() -> np.ndarray:
    """(24, 10475): SMPL-X verts -> SMPL-24 joints via smplx2smpl then SMPL J_regressor.

    GT is SMPL (EgoHumans processed_data/smpl), so we score in SMPL-24 joint space:
    our SMPL-X FK verts -> SMPL verts (smplx2smpl) -> 24 SMPL joints (SMPL J_regressor),
    matching the GT smpl ``joints[:24]``.
    """
    global _SMPL_REG
    if _SMPL_REG is None:
        import scipy.sparse as _sp
        bm = _REPO_ROOT / "body_models"
        jr = pickle.load(open(bm / "smpl" / "SMPL_NEUTRAL.pkl", "rb"),
                         encoding="latin1")["J_regressor"]                  # (24, 6890) sparse
        jr = jr.toarray() if _sp.issparse(jr) else np.asarray(jr)
        mtx = pickle.load(open(bm / "smplx2smpl.pkl", "rb"))["matrix"]      # (6890, 10475)
        _SMPL_REG = (jr @ mtx).astype(np.float64)
    return _SMPL_REG


# ── geometry helpers (copied — this is a standalone eval) ──────────────────
def se3_align(src: np.ndarray, dst: np.ndarray):
    """Kabsch SE(3) (no scale): R,t minimising ||R@src+t-dst||. src,dst (N,3)."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dc - R @ sc


def sim3_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return pred aligned to gt by Sim(3). pred,gt (N,3)."""
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    s = np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12))
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return s * p0 @ R.T + gc


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """(J,3) axis-angle -> (J,6) = first two ROWS of R (training convention)."""
    from scipy.spatial.transform import Rotation as SciR
    Rm = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()      # (J,3,3)
    return Rm[:, :2, :].reshape(-1, 6)                        # first two rows


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """(J,6) -> (J,3) axis-angle; 6 = first two ROWS of R (Gram-Schmidt)."""
    from scipy.spatial.transform import Rotation as SciR
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    Rm = np.stack([b1, b2, b3], axis=1)                      # rows are b1,b2,b3
    return SciR.from_matrix(Rm).as_rotvec()


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
    `median_fuse` in evaluation/evaluate_egohumans_median.py.

    The chordal mean minimises sum_k ||R - R_k||_F^2 (L2), so a single badly
    wrong camera drags the estimate. The geodesic median minimises L1 on SO(3),

        R_bar = argmin_R  sum_k  w_k * d_geo(R, R_k),

    solved by Weiszfeld/IRLS: seed with the chordal mean, then repeatedly re-take
    a chordal mean with weights w_k / (theta_k + eps). Each step is the same
    closed-form SVD projection — no training, no parameters, no checkpoint.

    Selected on RICH's 12-scene train pool (-0.8 mm RR-MPJPE vs the chordal
    mean, beating every trained fusion module); applied here unchanged, so
    EgoHumans stays a zero-shot report with no per-dataset tuning.

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


# ── GT loading (EgoHumans) ─────────────────────────────────────────────────
def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _colmap_to_aria(gt_scene: Path):
    """4x4 mapping colmap-world -> aria01-world, or None if pkl missing."""
    p = gt_scene / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    if not p.exists():
        return None
    d = _load_pkl(p)
    T_aria_to_colmap = np.asarray(d["aria01"], dtype=np.float64)   # aria01 -> colmap
    return np.linalg.inv(T_aria_to_colmap)


def _gt_exo_cameras_aria(gt_scene: Path, T_c2a):
    """Exo cameras from images.txt, expressed in the aria01 world frame.

    Returns (centres, world_to_cam):
      centres     : {cam_name: (3,) camera centre}
      world_to_cam: {cam_name: (R (3,3), t (3,))} with  X_cam = R @ X_aria + t

    colmap stores world(colmap)->cam as (R, t), centre C_col = -R^T t.
    T_c2a is a SIMILARITY, not a rigid transform: the colmap reconstruction is
    up to scale and the pkl carries that scale (~10x here), so its 3x3 block is
    s*R_a and must be split before use — treating it as a rotation puts the
    scale into the extrinsic and silently destroys the rig.  Centres take the
    full similarity, orientations take the rotation only:

        C_aria = (s R_a) C_col + t_a ,  R_new = R @ R_a^T ,  t_new = -R_new C_aria

    The centres are therefore identical to the previous centre-only code path.
    """
    imgs = gt_scene / "colmap" / "workplace" / "images.txt"
    if not imgs.exists():
        return {}, {}
    centres: dict[str, np.ndarray] = {}
    w2c: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    from scipy.spatial.transform import Rotation as SciR
    if T_c2a is not None:
        A_c2a = np.asarray(T_c2a[:3, :3], dtype=np.float64)
        s_c2a = float(np.cbrt(abs(np.linalg.det(A_c2a))))          # similarity scale
        R_c2a = A_c2a / s_c2a                                       # rotation only
        t_c2a = np.asarray(T_c2a[:3, 3], dtype=np.float64)
    with open(imgs) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:      # image lines only (skip POINTS2D lines)
                continue
            try:
                qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
            except ValueError:
                continue
            name = parts[9]
            cam = name.split("/")[0]
            if not cam.startswith("cam") or cam in centres:
                continue
            R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()   # world(colmap)->cam
            t = np.array([tx, ty, tz], dtype=np.float64)
            C = -R.T @ t                                       # centre, colmap world
            if T_c2a is not None:
                C = A_c2a @ C + t_c2a                          # -> aria world (full similarity)
                R = R @ R_c2a.T                                # rotation only
                t = -R @ C
            centres[cam] = C
            w2c[cam] = (R, t)
    return centres, w2c


def build_gt_extrinsics(camera_names, cam_w2c, T: int):
    """GT exo extrinsics as (T, K, 3, 4), re-rooted to ``camera_names[0]``.

    BodyPlacer expects camera 0 to be the reference ([I|0]) — that is how the
    VGGT cameras are stored — so every camera is re-rooted through camera 0:

        R_rel = R_k @ R_0^T ,   t_rel = t_k - R_rel @ t_0

    Returns (extrinsics, filled); ``filled`` (K,) marks cameras that have a GT
    calibration.  Unfilled cameras stay all-zero and MUST be masked out of
    cam_valid, or the DLT builds a degenerate projection matrix from them.
    """
    names = [n.decode() if isinstance(n, bytes) else n for n in camera_names]
    if not names or names[0] not in cam_w2c:
        return None, None
    R0, t0 = cam_w2c[names[0]]
    K_ = len(names)
    exts = np.zeros((T, K_, 3, 4), dtype=np.float64)
    filled = np.zeros(K_, dtype=bool)
    for k, cn in enumerate(names):
        if cn not in cam_w2c:
            continue
        Rk, tk = cam_w2c[cn]
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        exts[:, k] = np.hstack([R_rel, t_rel[:, None]])[None]
        filled[k] = True
    return exts.astype(np.float32), filled


def gt_scale_from_cameras(placer, cam_pos_aria: dict[str, np.ndarray]) -> float | None:
    """Oracle metric scale = Sim(3) scale from unscaled VGGT centres to GT centres.

    VGGT solves the cameras per frame, so each camera contributes the median of
    its per-frame centres (the same robust summary the production W† alignment
    uses) instead of an arbitrary single frame.
    """
    names = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    pred, gt = [], []
    for k, cn in enumerate(names):
        if cn not in cam_pos_aria:
            continue
        vt = np.where(placer.cam_valid[:, k])[0]
        if not vt.size:
            continue
        ctrs = [-(placer.extrinsics[t, k, :3, :3].T @ placer.extrinsics[t, k, :3, 3])
                for t in vt]
        pred.append(np.median(np.stack(ctrs), 0).astype(np.float64))
        gt.append(np.asarray(cam_pos_aria[cn], dtype=np.float64))
    if len(pred) < 2:
        return None
    pred, gt = np.stack(pred), np.stack(gt)
    p0, g0 = pred - pred.mean(0), gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    return float((S * [1, 1, d]).sum() / ((p0 ** 2).sum() + 1e-12))


def load_gt_scene(gt_scene: Path):
    """Return (frames, gt_by_frame, gt_pose_by_frame, cam_pos_aria, cam_w2c, have_world).

    frames          : sorted list of GT frame indices (1-based, from smpl files)
    gt_by_frame     : {frame: {aria_id: (24,3) SMPL joints, aria world}}
    gt_pose_by_frame: {frame: {aria_id: (69,) SMPL body_pose}} — the M4 oracle
    cam_pos_aria    : {cam_name: (3,) centre in aria world}  (empty if no pkl)
    cam_w2c         : {cam_name: (R, t)} world(aria)->cam    (empty if no pkl)
    have_world      : True if the colmap_from_aria pkl was found (=> W-MPJPE† ok)
    """
    smpl_dir = gt_scene / "processed_data" / "smpl"
    if not smpl_dir.is_dir():
        raise FileNotFoundError(f"no processed_data/smpl in {gt_scene}")
    T_c2a = _colmap_to_aria(gt_scene)
    cam_pos, cam_w2c = (_gt_exo_cameras_aria(gt_scene, T_c2a)
                        if T_c2a is not None else ({}, {}))
    frames, gt_by_frame, gt_pose_by_frame = [], {}, {}
    for f in sorted(smpl_dir.glob("*.npy")):
        try:
            fi = int(f.stem)
        except ValueError:
            continue
        arr = np.load(str(f), allow_pickle=True)
        d = arr.item() if arr.dtype == object and arr.shape == () else arr
        if not isinstance(d, dict):
            continue
        people, poses = {}, {}
        for aid, params in d.items():
            if not isinstance(params, dict) or "joints" not in params:
                continue
            j = np.asarray(params["joints"], dtype=np.float64)   # (45,3) SMPL joints, aria world
            if j.shape[0] >= 24 and j.shape[1] >= 3:
                people[aid] = j[:24, :3]
                if "body_pose" in params:
                    poses[aid] = np.asarray(params["body_pose"], dtype=np.float64).reshape(-1)
        if people:
            frames.append(fi)
            gt_by_frame[fi] = people
            gt_pose_by_frame[fi] = poses
    return sorted(frames), gt_by_frame, gt_pose_by_frame, cam_pos, cam_w2c, (T_c2a is not None)


# ── prediction: fusion (per-frame) -> placer -> FK -> COCO ─────────────────
def _load_clean_tracks(ghost_scene: Path, cam_names, valid_mask, pids):
    """{cam_idx: {pid: {frame: entry}}} from body_data_clean."""
    per_cam = {}
    for k, cam in enumerate(cam_names):
        if not valid_mask[k]:
            per_cam[k] = {}
            continue
        bd = ghost_scene / cam / "body_data_clean"
        cam_map = {}
        for pid in pids:
            f = bd / f"person_{pid}.npz"
            if not f.exists():
                continue
            d = np.load(str(f), allow_pickle=False)
            fi = d["frame_indices"].astype(int)
            frames_map = {}
            for t, gfr in enumerate(fi):
                e = {"go": d["smplx_global_orient"][t].reshape(3),
                     "bp": d["smplx_body_pose"][t].reshape(63)}
                if "smplx_left_hand_pose" in d.files:
                    e["lh"] = d["smplx_left_hand_pose"][t].reshape(-1, 3)
                if "smplx_right_hand_pose" in d.files:
                    e["rh"] = d["smplx_right_hand_pose"][t].reshape(-1, 3)
                if "smplx_betas" in d.files:
                    e["betas"] = d["smplx_betas"][t][:10]
                frames_map[int(gfr)] = e
            if frames_map:
                cam_map[pid] = frames_map
        per_cam[k] = cam_map
    return per_cam


def _pack_pose(entry) -> np.ndarray:
    parts = [entry["go"].reshape(1, 3), entry["bp"].reshape(21, 3)]
    parts.append(entry.get("lh", np.zeros((15, 3))).reshape(15, 3))
    parts.append(entry.get("rh", np.zeros((15, 3))).reshape(15, 3))
    aa = np.concatenate(parts, 0).astype(np.float64)
    if aa.shape[0] < _NUM_JOINTS:
        aa = np.concatenate([aa, np.zeros((_NUM_JOINTS - aa.shape[0], 3))], 0)
    return _aa_to_6d(aa)[1:]            # (54,6) root excluded


def predict_scene(ghost_scene: Path, frames, pids, device, smplx_arg,
                  cam_pos_aria, cam_w2c, gt_pose_by_frame, modalities):
    """Fuse once, then place under each oracle rung.

    Returns (per_mod, raw_coco, cam_names) where per_mod is
        {modality: {"pred": {pid: {frame: (24,3) SMPL joints, ghost-metric}},
                    "extrinsics": (T,K,3,4), "valid": (T,K), "scale": float}}
    with the extrinsics/scale the placer actually used, so the caller's SE(3)
    camera alignment can be built from the same rig.
    """
    vggt = np.load(ghost_scene / "vggt_cameras_centered.npz", allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    valid_mask = vggt["valid"][0]
    extrinsics = vggt["extrinsics"][0]
    valid_full = vggt["valid"]            # (T,K) per-frame camera validity
    extrinsics_full = vggt["extrinsics"]  # (T,K,3,4) per-frame extrinsics

    per_cam = _load_clean_tracks(ghost_scene, cam_names, valid_mask, pids)
    frames_set = set(int(f) for f in frames)
    K, P = len(cam_names), len(pids)
    pid_slot = {p: i for i, p in enumerate(pids)}
    fmin, fmax = frames[0], frames[-1]
    T = fmax - fmin + 1

    # dense per-frame tensors (T,1,K,P,54,6) — temporal length 1 => no temporal attn
    pose = np.zeros((T, 1, K, P, 54, 6), dtype=np.float32)
    mask = np.zeros((T, 1, K, P), dtype=np.float32)
    betas_acc = {p: [] for p in pids}
    for k, cam_map in per_cam.items():
        for pid, fm in cam_map.items():
            s = pid_slot[pid]
            for gfr, e in fm.items():
                if fmin <= gfr <= fmax:
                    t = gfr - fmin
                    pose[t, 0, k, s] = _pack_pose(e)
                    mask[t, 0, k, s] = 1.0
                    if "betas" in e:
                        betas_acc[pid].append(e["betas"])
    betas_by_pid = {p: (np.mean(v, 0).astype(np.float32) if v else np.zeros(10, np.float32))
                    for p, v in betas_acc.items()}

    with torch.no_grad():
        # The geodesic median is independent across frames, so the frames
        # already sit in the batch dim and each is fused on its own. No
        # temporal padding: that trick only existed because the learned
        # module was OOD at temporal length 1 (trained with a 128-frame window).
        chunks = []
        for t0 in range(0, T, 256):
            pt = torch.from_numpy(pose[t0:t0 + 256]).to(device)   # (b,1,K,P,54,6)
            mt = torch.from_numpy(mask[t0:t0 + 256]).to(device)   # (b,1,K,P)
            chunks.append(median_fuse(pt, mt)[:, 0].cpu().numpy())
        fused = np.concatenate(chunks, 0)              # (T,P,54,6)

    # GT pose array for the M4 rung: (T,P,54,6), identity everywhere the GT has
    # no entry (6D identity = [1,0,0, 0,1,0]) so the placer never sees garbage;
    # those frames are dropped by the valid mask anyway (GT joints are NaN there).
    # SMPL body_pose is 69 = 23 joints; SMPL-X takes the first 21 (SMPL 22/23 are
    # the hands, modelled separately in SMPL-X).
    gt_pose_arr = np.zeros((T, P, 54, 6), dtype=np.float32)
    gt_pose_arr[..., 0] = 1.0
    gt_pose_arr[..., 4] = 1.0
    n_gt_pose = 0
    for gfr, people in (gt_pose_by_frame or {}).items():
        if not (fmin <= gfr <= fmax):
            continue
        t = gfr - fmin
        for aid, bp in people.items():
            try:
                pid = int(str(aid).replace("aria", ""))
            except ValueError:
                continue
            if pid not in pid_slot or bp.size < 63:
                continue
            gt_pose_arr[t, pid_slot[pid], :21] = _aa_to_6d(bp[:63].reshape(21, 3))
            n_gt_pose += 1

    # BodyPlacer hard-codes cam/body_data; give it a symlinked view whose
    # body_data points at body_data_clean (global ids, ops applied). Identity
    # remap since clean already uses global ids.
    view = _clean_scene_view(ghost_scene, cam_names)
    per_mod: dict[int, dict] = {}
    raw_coco = {p: {} for p in pids}          # {pid: {frame: {cam_idx: (24,3)}}}
    try:
        placer = BodyPlacer(view, smplx_arg, crop_meta_path=None)
        reg = smpl24_regressor()

        # ── Oracle inputs, shared by every rung ──────────────────────────────
        s_gt = gt_scale_from_cameras(placer, cam_pos_aria)
        if not s_gt:
            raise RuntimeError("oracle scale unavailable (<2 cameras matched to GT)")
        gt_ext, gt_filled = build_gt_extrinsics(placer.camera_names, cam_w2c, placer.T)
        if 4 in modalities and n_gt_pose == 0:
            logger.warning(f"{ghost_scene.name}: no GT body_pose in the smpl npys "
                           "— M4 would be a no-op, dropping that rung")
            modalities = [m for m in modalities if m != 4]

        # BodyPlacer loads its own extrinsics at construction, so the oracle
        # substitution must be patched onto THIS instance and restored after.
        orig_ext   = placer.extrinsics
        orig_valid = placer.cam_valid.copy()

        for modality in modalities:
            use_gt_cams = modality in (3, 4)
            use_gt_pose = modality == 4
            if use_gt_cams:
                if gt_ext is None:
                    logger.warning(f"{ghost_scene.name}: [M{modality}] no GT calibration for "
                                   "the reference camera — skipping this rung")
                    continue
                placer.extrinsics = gt_ext
                # GT extrinsics are already metric, so subject scale is exactly 1.0.
                scale_used = 1.0
                # Cameras without GT calibration are all-zero: mask them out so
                # the DLT never builds a projection matrix from a zero extrinsic.
                placer.cam_valid = orig_valid & gt_filled[np.newaxis, :]
            else:
                placer.extrinsics = orig_ext
                placer.cam_valid  = orig_valid
                scale_used = float(s_gt)

            pose_arr = gt_pose_arr if use_gt_pose else fused
            fused_pose_by_pid = {p: pose_arr[:, pid_slot[p]] for p in pids}
            # Shape stays PREDICTED even at M4: GT betas are SMPL shape space and
            # would be wrong fed to SMPL-X FK. M4 is a pure pose oracle.
            scale = np.full(placer.T, scale_used, dtype=np.float32)

            try:
                trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
                    scale=scale, all_pids=set(pids), pred_betas_by_pid=betas_by_pid,
                    fused_pose_by_pid=fused_pose_by_pid, frame_start=fmin)
            except Exception as e:
                logger.warning(f"{ghost_scene.name}: [M{modality}] placer failed — {e}")
                continue
            finally:
                ext_used   = placer.extrinsics
                valid_used = placer.cam_valid
                placer.extrinsics = orig_ext      # always restore
                placer.cam_valid  = orig_valid

            pred_joints = {p: {} for p in pids}
            for pid in pids:
                s = pid_slot[pid]
                betas_p = betas_by_pid[pid][np.newaxis]
                for gfr in frames:
                    pw = trans_dict.get(pid, {}).get(gfr)
                    R_m = orient_dict.get(pid, {}).get(gfr)
                    if pw is None or R_m is None:
                        continue
                    t = gfr - fmin
                    bp_aa = _6d_to_aa(pose_arr[t, s, :21]).reshape(63)
                    J55, verts = placer._smplx_fk(betas_p, bp_aa[np.newaxis],
                                                  np.zeros((1, 3), np.float32), return_verts=True)
                    pelvis = J55[0].astype(np.float64)[0]
                    smpl_can = reg @ verts[0].astype(np.float64)     # (24,3) canonical SMPL joints
                    pred_joints[pid][gfr] = (R_m @ (smpl_can - pelvis).T).T + pw   # ghost-metric

            per_mod[modality] = {"pred": pred_joints, "scale": scale_used,
                                 "extrinsics": np.asarray(ext_used),
                                 "valid": np.asarray(valid_used)}

        # Raw single-view baseline: canonical SMPL-24 from each cam's own SMPL-X
        # pose (no fusion, no placement — PA-comparable only, rung-independent).
        for k, cam_map in per_cam.items():
            for pid, fm in cam_map.items():
                betas_p = betas_by_pid[pid][np.newaxis]
                for gfr, e in fm.items():
                    if gfr not in frames_set:
                        continue
                    _, verts = placer._smplx_fk(betas_p, e["bp"].reshape(1, 63),
                                                np.zeros((1, 3), np.float32), return_verts=True)
                    raw_coco[pid].setdefault(gfr, {})[k] = reg @ verts[0].astype(np.float64)
    finally:
        import shutil
        shutil.rmtree(view, ignore_errors=True)
    return per_mod, raw_coco, cam_names


def _clean_scene_view(ghost_scene: Path, cam_names) -> Path:
    """Temp scene dir: root vggt/scale files symlinked; each cam/body_data ->
    the real cam/body_data_clean. Lets BodyPlacer read the clean tracks."""
    import tempfile, os
    view = Path(tempfile.mkdtemp(prefix="cleanview_"))
    for f in ghost_scene.iterdir():
        if f.is_file():
            os.symlink(f, view / f.name)
    for cam in cam_names:
        clean = ghost_scene / cam / "body_data_clean"
        if clean.is_dir():
            (view / cam).mkdir(parents=True, exist_ok=True)
            os.symlink(clean, view / cam / "body_data")
    return view


# ── per-scene orchestration ────────────────────────────────────────────────
def eval_scene(ghost_scene: Path, gt_scene: Path, device, smplx_arg,
               modalities=None):
    """Return dump dict {pred_m<N> (T,P,24,3) aria, gt, raw, have_world} or None."""
    if modalities is None:
        modalities = list(_MODALITIES)
    if not (ghost_scene / "vggt_cameras_centered.npz").exists():
        logger.warning(f"{ghost_scene.name}: no vggt cameras, skip"); return None
    (frames, gt_by_frame, gt_pose_by_frame,
     cam_pos_aria, cam_w2c, have_world) = load_gt_scene(gt_scene)
    if not frames:
        logger.warning(f"{ghost_scene.name}: no GT frames, skip"); return None
    # Every rung needs GT camera centres — M2 included, since the GT scale is
    # derived from them. Without colmap there is no ladder, so skip loudly
    # instead of silently reporting a rung that is not the rung it claims to be.
    if not have_world or not cam_pos_aria:
        logger.warning(f"{ghost_scene.name}: no colmap GT cameras — the whole ladder "
                       "needs them (GT scale included), skip"); return None

    aria_ids = sorted({a for fr in gt_by_frame.values() for a in fr})   # e.g. aria01..04
    pid_of_aria = {a: int(a.replace("aria", "")) for a in aria_ids}     # aria0N -> N
    pids = sorted(pid_of_aria.values())

    per_mod, raw_coco, cam_names = predict_scene(
        ghost_scene, frames, pids, device, smplx_arg,
        cam_pos_aria, cam_w2c, gt_pose_by_frame, modalities)
    if not per_mod:
        logger.warning(f"{ghost_scene.name}: no rung produced predictions, skip"); return None

    fmin = frames[0]
    P = len(pids); F = len(frames)
    fidx = {fr: i for i, fr in enumerate(frames)}

    out: dict = {"have_world": have_world, "pids": np.array(pids),
                 "frames": np.array(frames), "modalities": np.array(sorted(per_mod))}

    for modality, mod in sorted(per_mod.items()):
        extrinsics_full = mod["extrinsics"]      # (T,K,3,4) — the rig the placer used
        valid_full      = mod["valid"]           # (T,K)
        ma_scale        = mod["scale"]
        Textr           = extrinsics_full.shape[0]

        # Per-frame SE(3): placement-frame cameras (this frame) -> aria-world GT
        # cameras. VGGT cameras are estimated per frame, so a single global SE(3)
        # would dump per-frame camera jitter into W†. CHROMM's single-frame
        # protocol aligns the camera poses per frame (GA/PA are per-frame); we fit
        # SE(3) per frame for W† too (still no scaling). Frames with <2 valid cams
        # fall back to a global (median-camera) SE(3). For the GT-camera rungs the
        # rig IS the GT rig at scale 1.0, so this recovers the reference-camera
        # pose in the aria world and is constant across frames.
        def _frame_se3(t):
            pc, gc = [], []
            for k, cam in enumerate(cam_names):
                if 0 <= t < Textr and valid_full[t, k] and cam in cam_pos_aria:
                    Rk, tk = extrinsics_full[t, k, :, :3], extrinsics_full[t, k, :, 3]
                    pc.append((-Rk.T @ tk) * ma_scale)
                    gc.append(cam_pos_aria[cam])
            return se3_align(np.stack(pc), np.stack(gc)) if len(pc) >= 2 else None

        R_glob = t_glob = None
        pcg, gcg = [], []
        for k, cam in enumerate(cam_names):
            if cam not in cam_pos_aria:
                continue
            ctrs = [-(extrinsics_full[tt, k, :, :3].T @ extrinsics_full[tt, k, :, 3]) * ma_scale
                    for tt in range(Textr) if valid_full[tt, k]]
            if ctrs:
                pcg.append(np.median(ctrs, 0)); gcg.append(cam_pos_aria[cam])
        if len(pcg) >= 2:
            R_glob, t_glob = se3_align(np.stack(pcg), np.stack(gcg))

        pred = np.full((F, P, 24, 3), np.nan)
        for fr in frames:
            i = fidx[fr]
            RA = _frame_se3(fr - fmin)
            if RA is None and R_glob is not None:
                RA = (R_glob, t_glob)
            for si, pid in enumerate(pids):
                j = mod["pred"][pid].get(fr)
                if j is None:
                    continue
                pred[i, si] = (j @ RA[0].T + RA[1]) if RA is not None else j
        out[f"pred_m{modality}"] = pred.astype(np.float32)

    # GT joints (rung-independent)
    gt = np.full((F, P, 24, 3), np.nan)
    for fr in frames:
        for si, pid in enumerate(pids):
            aid = f"aria{pid:02d}"
            if aid in gt_by_frame[fr]:
                gt[fidx[fr], si] = gt_by_frame[fr][aid]
    out["gt"] = gt.astype(np.float32)

    # raw single-view canonical poses: (T,P,K,24,3), NaN where absent
    Kn = len(cam_names)
    raw = np.full((F, P, Kn, 24, 3), np.nan)
    for si, pid in enumerate(pids):
        for fr, camj in raw_coco[pid].items():
            for k, j in camj.items():
                raw[fidx[fr], si, k] = j
    out["raw"] = raw.astype(np.float32)
    return out


# ── metrics over a dump ────────────────────────────────────────────────────
def scene_metrics(pred, gt, have_world):
    """W†/GA/PA in metres over the 12 limb joints; per-frame single-frame protocol."""
    L = _SMPL_EVAL
    T, P = pred.shape[:2]
    valid = np.isfinite(pred[..., L, :]).all((-1, -2)) & np.isfinite(gt[..., L, :]).all((-1, -2))
    w, ga, pa = [], [], []
    for t in range(T):
        ps = [p for p in range(P) if valid[t, p]]
        if not ps:
            continue
        pr = pred[t][ps][:, L]      # (n,12,3)
        gtt = gt[t][ps][:, L]
        if have_world:
            w.append(np.linalg.norm(pr - gtt, axis=-1).mean())          # already SE3-cam-aligned
        # GA: single Sim3 over all persons jointly, this frame
        a = sim3_align(pr.reshape(-1, 3), gtt.reshape(-1, 3)).reshape(pr.shape)
        ga.append(np.linalg.norm(a - gtt, axis=-1).mean())
        # PA: per person
        for i in range(len(ps)):
            a = sim3_align(pr[i], gtt[i])
            pa.append(np.linalg.norm(a - gtt[i], axis=-1).mean())
    return (float(np.mean(w)) if w else float("nan"),
            float(np.mean(ga)) if ga else float("nan"),
            float(np.mean(pa)) if pa else float("nan"))


def raw_pa(raw, gt):
    """Single-view PA baselines, same protocol as fused PA (mean over frames/persons).
    Returns (best_view_pa, median_view_pa): per (frame,person), Procrustes-align each
    camera's raw pose to GT, then take best / median across cameras."""
    L = _SMPL_EVAL
    T, P, K = raw.shape[:3]
    best, med = [], []
    for t in range(T):
        for p in range(P):
            if not np.isfinite(gt[t, p, L]).all():
                continue
            errs = []
            for k in range(K):
                if not np.isfinite(raw[t, p, k, L]).all():
                    continue
                a = sim3_align(raw[t, p, k, L], gt[t, p, L])
                errs.append(np.linalg.norm(a - gt[t, p, L], axis=-1).mean())
            if errs:
                best.append(min(errs)); med.append(float(np.median(errs)))
    return (float(np.mean(best)) if best else float("nan"),
            float(np.mean(med)) if med else float("nan"))


def aggregate(dump_dir: Path):
    """Print per-scene rows and the per-rung aggregate table over all dumps.

    Accepts either a flat directory of ``<scene>.npz`` or a parent holding
    rung-keyed subdirs (``m2/``, ``m3/``, ``m4/``) as written by the driver —
    one job per activity and rung.  Dumps are merged per scene, so a scene that
    appears under several rungs contributes one row with all of them.
    """
    # {scene: {modality: (w, ga, pa)}} and {scene: (best, median) single-view PA}
    per_scene: dict[str, dict[int, tuple[float, float, float]]] = {}
    raw_by_scene: dict[str, tuple[float, float]] = {}

    files = sorted(dump_dir.glob("*.npz")) or sorted(dump_dir.glob("m*/*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=False)
        mods = [int(k.split("_m")[1]) for k in d.files if k.startswith("pred_m")]
        if not mods:
            print(f"  {f.stem:24s}  (no rungs in dump — stale format?)")
            continue
        slot = per_scene.setdefault(f.stem, {})
        for m in sorted(mods):
            w, ga, pa = scene_metrics(d[f"pred_m{m}"], d["gt"], bool(d["have_world"]))
            slot[m] = (w, ga, pa)
        if "raw" in d.files and f.stem not in raw_by_scene:
            raw_by_scene[f.stem] = raw_pa(d["raw"], d["gt"])

    if not per_scene:
        print("no dumps"); return

    all_mods = sorted({m for slot in per_scene.values() for m in slot})
    for scene in sorted(per_scene):
        cells = [f"M{m}: W†={w*1000:7.1f} GA={ga*1000:6.1f} PA={pa*1000:5.1f}"
                 for m, (w, ga, pa) in sorted(per_scene[scene].items())]
        extra = ""
        if scene in raw_by_scene:
            b, mm = raw_by_scene[scene]
            extra = f"   | single-view PA: best={b*1000:.1f} median={mm*1000:.1f}"
        print(f"  {scene:24s}  " + "  ".join(cells) + f" mm{extra}")

    print("\n=== AGGREGATE — EgoHumans ORACLE ABLATIONS (mm) ===")
    print("  Oracle numbers — NOT comparable to CHROMM (W†510 GA150 PA50).")
    print("  Production number: evaluation/evaluate_egohumans.py")
    print(f"  {'rung':<40s} {'n':>4s} {'W-MPJPE†':>10s} {'GA-MPJPE':>10s} {'PA-MPJPE':>10s}")
    for m in all_mods:
        vals = [v[m] for v in per_scene.values() if m in v]
        W = np.array([v[0] for v in vals]); G = np.array([v[1] for v in vals])
        A = np.array([v[2] for v in vals])
        print(f"  {_MODALITY_LABELS[m]:<40s} {len(vals):>4d} "
              f"{np.nanmean(W)*1000:>10.1f} {np.nanmean(G)*1000:>10.1f} {np.nanmean(A)*1000:>10.1f}")

    # Rungs are only comparable on scenes where BOTH finished. If the per-rung
    # scene sets differ, the gaps above mix a rung difference with a scene-set
    # difference, so the common-subset table is printed too.
    common = sorted(s for s, v in per_scene.items() if all(m in v for m in all_mods))
    if len(all_mods) > 1 and len(common) != len(per_scene):
        print(f"\n  --- common subset ({len(common)} scenes with every rung) ---")
        for m in all_mods:
            vals = [per_scene[s][m] for s in common]
            W = np.array([v[0] for v in vals]); G = np.array([v[1] for v in vals])
            A = np.array([v[2] for v in vals])
            print(f"  {_MODALITY_LABELS[m]:<40s} {len(vals):>4d} "
                  f"{np.nanmean(W)*1000:>10.1f} {np.nanmean(G)*1000:>10.1f} {np.nanmean(A)*1000:>10.1f}")

    print("\n  Reading the ladder:  M2 - production = scale error;  "
          "M3 - M2 = camera error;  M4 - M3 = pose/fusion error.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost_root")
    ap.add_argument("--gt_root", help="camera_ready/<activity> dir with per-scene GT. Must be "
                                      "the copy carrying processed_data/smpl (egohumans_gt_full)")
    ap.add_argument("--smplx_model", default=str(_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl"))
    ap.add_argument("--modalities", default=",".join(str(m) for m in _MODALITIES),
                    help="Comma-separated subset of the oracle ladder: 2,3,4 (default: all). "
                         "All rungs share one fusion pass, so running them together is cheap.")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--dump_dir", default="eval_ablations_egohumans/dumps_median",
                    help="Separate from the v2-module dumps: mixing the two in one dir "
                         "would silently combine results from different fusion methods "
                         "via the skip-if-exists resume logic.")
    ap.add_argument("--metrics_only", action="store_true", help="Stage B: aggregate dumps")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.modalities = [int(x.strip()) for x in args.modalities.split(",") if x.strip()]
    bad = [m for m in args.modalities if m not in _MODALITIES]
    if bad:
        ap.error(f"--modalities: {bad} not in {list(_MODALITIES)}")
    if not args.modalities:
        ap.error("--modalities: empty")

    dump_dir = Path(args.dump_dir); dump_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_only:
        aggregate(dump_dir); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Fusion: geodesic median over cameras, L1/Weiszfeld (no checkpoint)")
    smplx_arg = args.smplx_model
    ghost_root, gt_root = Path(args.ghost_root), Path(args.gt_root)

    scenes = [args.scene] if args.scene else sorted(
        p.name for p in ghost_root.iterdir()
        if p.is_dir() and any(c.is_dir() and c.name.startswith("cam") for c in p.iterdir()))
    scenes = [s for s in scenes if (gt_root / s).exists()]

    for scene in scenes:
        out = dump_dir / f"{scene}.npz"
        if out.exists() and not args.overwrite:
            logger.info(f"{scene}: dump exists, skip"); continue
        try:
            d = eval_scene(ghost_root / scene, gt_root / scene, device,
                           smplx_arg, args.modalities)
        except Exception as e:
            logger.warning(f"{scene}: FAILED — {e}"); continue
        if d is None:
            continue
        np.savez_compressed(out, **d)
        cells = []
        for m in sorted(int(k.split("_m")[1]) for k in d if k.startswith("pred_m")):
            w, ga, pa = scene_metrics(d[f"pred_m{m}"], d["gt"], d["have_world"])
            cells.append(f"M{m}: W†={w*1000:.1f} GA={ga*1000:.1f} PA={pa*1000:.1f}")
        logger.info(f"{scene}: dumped  " + "  ".join(cells) + " mm")

    if args.scene:
        print("\n--- single-scene metrics ---")
        aggregate(dump_dir)


if __name__ == "__main__":
    main()
