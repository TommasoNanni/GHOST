"""Evaluate ghost pipeline on RICH test set — production config (M10: neutral pred vs gendered GT).

Single honest number: pred camera + pred scale + pred pose, with the
prediction run through the NEUTRAL SMPL-X body model (production inference
never knows subject gender) evaluated against RICH's ground truth AS IT
ACTUALLY EXISTS — gendered SMPL-X params, no artificial correction. This is
the real deployment comparison: you don't get to neutralize GT in practice,
the dataset just has gendered GT.

(M9 — neutral pred AND neutral GT — over-corrects: it modifies GT to match
production's blind spot, which isn't a real scenario. M10 is the honest one.)

For the gender-control comparison (M1 gender-oracle, M9 fully-neutral),
see evaluation/gender_ablations.py. For oracle ablations (GT-scale / GT-camera
/ GT-pose), see evaluation/ablations.py. For the placing-approach comparison
(FK-DLT vs Sapiens-DLT), see evaluation/placing_approaches.py. For the full
M1-M10 modality zoo, see evaluation/evaluate_on_rich_test.py.

Metrics (all in millimetres unless noted):
  WA-MPJPE     — World-Aligned MPJPE: single Sim(3) over ALL frames jointly.
  W-MPJPE      — World MPJPE: Sim(3) from first 2 frames applied to all.
  WA-MPJPE-100 — WHAM/CHROMM protocol: sequence split into 100-frame segments,
                 one Sim(3) per segment (all persons jointly), errors pooled.
  W-MPJPE-100  — per 100-frame segment, Sim(3) from its first 2 valid frames.
  GA-MPJPE     — Group-Aligned MPJPE: per-frame Sim(3) over all persons jointly.
  PA-MPJPE     — Procrustes-Aligned MPJPE: per-person per-frame Sim(3).
  RTE          — Root Translation Error (%): SE(3) aligned, normalised by displacement.

CHROMM (ours-multi) on RICH: WA=53.1 mm  W=79.0 mm  RTE=1.4%
CHROMM evaluates on 100-frame segments, so compare against the *-100 rows.

Usage
-----
    python evaluation/evaluate_rich.py \\
        --ghost_output_root /path/to/ghost_outputs/rich_test \\
        --rich_root         /path/to/rich \\
        --checkpoint        /path/to/fusion_checkpoint.pt \\
        --smplx_model       /path/to/SMPLX_NEUTRAL.pkl \\
        [--device cuda] [--max_scenes N] [--gt_split test]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule
from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Evaluation joint convention: the 24 SMPL joints (NOT the SMPL-X body joints), to
# match the WHAM/DuoMo/CHROMM RICH protocol. SMPL-X mesh vertices are mapped to SMPL
# vertices via the barycentric `smplx2smpl` matrix (6890 x 10475), then to 24 joints
# via the SMPL `J_regressor` (24 x 6890). We precompute the combined, gender-agnostic
# operator J24 = J_regressor @ smplx2smpl  (24 x 10475) and apply it to SMPL-X FK verts.
_N_SMPL_JOINTS = 24
_J24_OPERATOR: np.ndarray | None = None


def _verts_to_smpl24(verts: np.ndarray) -> np.ndarray:
    """Map SMPL-X mesh vertices (10475, 3) -> 24 SMPL joints (24, 3) in the same frame."""
    global _J24_OPERATOR
    if _J24_OPERATOR is None:
        import joblib
        import scipy.sparse as sp
        smplx2smpl = np.asarray(
            joblib.load(_REPO_ROOT / "body_models" / "smplx2smpl.pkl")["matrix"]
        )  # (6890, 10475), barycentric
        with open(_REPO_ROOT / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl", "rb") as f:
            Jr = pickle.load(f)["J_regressor"]  # (24, 6890)
        Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr)
        _J24_OPERATOR = (Jr @ smplx2smpl).astype(np.float32)  # (24, 10475)
    return (_J24_OPERATOR @ verts).astype(np.float32)

# RICH full-resolution (before max_side=1440 resize), needed for calibration scaling.
_RICH_ORIG_W = 4112
_RICH_ORIG_H = 3008


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (..., 6) → axis-angle (..., 3).

    Interprets the 6 values as the first two rows of R (matching the training
    convention in fusion_dataset.py). Applies Gram-Schmidt then converts to axis-angle.
    """
    shape = sixd.shape[:-1]
    s = sixd.reshape(-1, 6)
    r0, r1 = s[:, :3], s[:, 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)         # (N, 3, 3) — rows are b1, b2, b3
    aa = SciR.from_matrix(R).as_rotvec()
    return aa.reshape(shape + (3,)).astype(np.float32)


def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) → 6D (..., 6).

    Uses the first two rows of the rotation matrix — matches the training
    convention in fusion_dataset.py.
    """
    shape = aa.shape[:-1]
    try:
        mats = SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix()
    except Exception:
        return np.zeros(shape + (6,), dtype=np.float32)
    sixd = np.concatenate([mats[:, 0, :], mats[:, 1, :]], axis=1)  # rows → (N, 6)
    return sixd.reshape(shape + (6,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _sim3_align(
    pred: np.ndarray,
    gt:   np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Sim(3) alignment: find s, R, t minimising ||s·R·pred + t − gt||².

    Args:
        pred, gt: (N, 3) matched point sets.

    Returns:
        aligned_pred (N, 3), scale s, rotation R (3,3), translation t (3,).
    """
    assert pred.shape == gt.shape and pred.ndim == 2
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    pred_c = pred - mu_p
    gt_c   = gt   - mu_g

    sigma2 = float((pred_c ** 2).sum()) / len(pred)
    if sigma2 < 1e-12:
        t = mu_g - mu_p
        return (pred + t).astype(np.float32), 1.0, np.eye(3, dtype=np.float32), t.astype(np.float32)

    H = pred_c.T @ gt_c / len(pred)
    U, S_sv, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)
    s = float(np.sum(S_sv * np.diag(D))) / sigma2
    t = (mu_g - s * (R @ mu_p)).astype(np.float32)

    aligned = (s * (pred @ R.T) + t).astype(np.float32)
    return aligned, s, R, t


def _se3_align(
    pred: np.ndarray,
    gt:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SE(3) alignment (no scale): find R, t minimising ||R·pred + t − gt||².

    Returns:
        aligned_pred (N, 3), rotation R (3,3), translation t (3,).
    """
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    H = (pred - mu_p).T @ (gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)
    t = (mu_g - R @ mu_p).astype(np.float32)
    return (pred @ R.T + t).astype(np.float32), R, t


# ---------------------------------------------------------------------------
# CHROMM metrics
# ---------------------------------------------------------------------------

def metric_wa_mpjpe(
    pred: np.ndarray,   # (T, P, J, 3) metres
    gt:   np.ndarray,   # (T, P, J, 3) metres
    valid: np.ndarray,  # (T, P) bool
) -> float:
    """World-Aligned MPJPE (mm): single Sim(3) over all valid frames, then MPJPE."""
    t_idx, p_idx = np.where(valid)
    if len(t_idx) < 2:
        return float("nan")
    pred_flat = pred[t_idx, p_idx].reshape(-1, 3)
    gt_flat   = gt[t_idx,   p_idx].reshape(-1, 3)
    aligned, _, _, _ = _sim3_align(pred_flat, gt_flat)
    return float(np.linalg.norm(aligned - gt_flat, axis=-1).mean()) * 1000.0


def metric_w_mpjpe(
    pred: np.ndarray,
    gt:   np.ndarray,
    valid: np.ndarray,
    n_align_frames: int = 2,
) -> float:
    """W-MPJPE (mm): Sim(3) from first N valid frames, applied to all."""
    t_idx, _ = np.where(valid)
    if len(t_idx) == 0:
        return float("nan")
    first_frames = sorted(set(t_idx.tolist()))[:n_align_frames]
    align_mask = np.zeros_like(valid)
    for tf in first_frames:
        align_mask[tf] = valid[tf]
    ta, pa = np.where(align_mask)
    if len(ta) < 2:
        return float("nan")
    _, s, R, t = _sim3_align(
        pred[ta, pa].reshape(-1, 3),
        gt[ta,   pa].reshape(-1, 3),
    )
    te, pe = np.where(valid)
    aligned_all = (s * (pred[te, pe].reshape(-1, 3) @ R.T) + t)
    gt_all      = gt[te, pe].reshape(-1, 3)
    return float(np.linalg.norm(aligned_all - gt_all, axis=-1).mean()) * 1000.0


def _iter_segments(valid: np.ndarray, segment_len: int):
    """Yield (t0, t1) bounds of consecutive ``segment_len``-frame windows.

    Windows are cut on absolute frame index (CHROMM/WHAM protocol), not on
    valid-frame count, so a window may contain fewer valid frames.
    """
    T = valid.shape[0]
    for t0 in range(0, T, segment_len):
        yield t0, min(t0 + segment_len, T)


def metric_wa_mpjpe_100(
    pred: np.ndarray,   # (T, P, J, 3) metres
    gt:   np.ndarray,   # (T, P, J, 3) metres
    valid: np.ndarray,  # (T, P) bool
    segment_len: int = 100,
) -> float:
    """WA-MPJPE-100 (mm): one Sim(3) per 100-frame segment (all persons jointly).

    This is the protocol CHROMM reports on RICH. Per-joint errors from all
    segments are pooled before averaging, so segments with more valid frames
    weigh proportionally more.
    """
    errs: list[np.ndarray] = []
    for t0, t1 in _iter_segments(valid, segment_len):
        t_idx, p_idx = np.where(valid[t0:t1])
        if len(t_idx) < 2:
            continue
        pred_flat = pred[t0:t1][t_idx, p_idx].reshape(-1, 3)
        gt_flat   = gt[t0:t1][t_idx,   p_idx].reshape(-1, 3)
        aligned, _, _, _ = _sim3_align(pred_flat, gt_flat)
        errs.append(np.linalg.norm(aligned - gt_flat, axis=-1))
    if not errs:
        return float("nan")
    return float(np.concatenate(errs).mean()) * 1000.0


def metric_w_mpjpe_100(
    pred: np.ndarray,
    gt:   np.ndarray,
    valid: np.ndarray,
    segment_len: int = 100,
    n_align_frames: int = 2,
) -> float:
    """W-MPJPE-100 (mm): per 100-frame segment, Sim(3) from its first 2 valid frames."""
    errs: list[np.ndarray] = []
    for t0, t1 in _iter_segments(valid, segment_len):
        valid_seg = valid[t0:t1]
        t_idx, _ = np.where(valid_seg)
        if len(t_idx) == 0:
            continue
        first_frames = sorted(set(t_idx.tolist()))[:n_align_frames]
        align_mask = np.zeros_like(valid_seg)
        for tf in first_frames:
            align_mask[tf] = valid_seg[tf]
        ta, pa = np.where(align_mask)
        if len(ta) < 2:
            continue
        _, s, R, t = _sim3_align(
            pred[t0:t1][ta, pa].reshape(-1, 3),
            gt[t0:t1][ta,   pa].reshape(-1, 3),
        )
        te, pe = np.where(valid_seg)
        aligned_all = s * (pred[t0:t1][te, pe].reshape(-1, 3) @ R.T) + t
        gt_all      = gt[t0:t1][te, pe].reshape(-1, 3)
        errs.append(np.linalg.norm(aligned_all - gt_all, axis=-1))
    if not errs:
        return float("nan")
    return float(np.concatenate(errs).mean()) * 1000.0


def metric_ga_mpjpe(
    pred: np.ndarray,
    gt:   np.ndarray,
    valid: np.ndarray,
) -> float:
    """GA-MPJPE (mm): per-frame Sim(3) over all persons jointly."""
    T = pred.shape[0]
    errs: list[float] = []
    for t in range(T):
        p_valid = np.where(valid[t])[0]
        if len(p_valid) == 0:
            continue
        pred_t = pred[t, p_valid].reshape(-1, 3)
        gt_t   = gt[t,   p_valid].reshape(-1, 3)
        aligned, _, _, _ = _sim3_align(pred_t, gt_t)
        errs.append(float(np.linalg.norm(aligned - gt_t, axis=-1).mean()))
    return float(np.mean(errs)) * 1000.0 if errs else float("nan")


def metric_pa_mpjpe(
    pred: np.ndarray,
    gt:   np.ndarray,
    valid: np.ndarray,
) -> float:
    """PA-MPJPE (mm): per-person per-frame Procrustes alignment."""
    T, P = pred.shape[:2]
    errs: list[float] = []
    for t in range(T):
        for p in range(P):
            if not valid[t, p]:
                continue
            aligned, _, _, _ = _sim3_align(pred[t, p], gt[t, p])
            errs.append(float(np.linalg.norm(aligned - gt[t, p], axis=-1).mean()))
    return float(np.mean(errs)) * 1000.0 if errs else float("nan")


def metric_rte(
    pred_roots: np.ndarray,  # (T, P, 3)
    gt_roots:   np.ndarray,  # (T, P, 3)
) -> float:
    """Root Translation Error (%) averaged over persons.

    Per person:  SE(3)-align the root trajectory, then
                 RTE = 100 × Σ||aligned_pred - gt|| / total_GT_displacement
    """
    P = pred_roots.shape[1]
    rtes: list[float] = []
    for p in range(P):
        pred_p = pred_roots[:, p]   # (T, 3)
        gt_p   = gt_roots[:, p]
        valid  = np.isfinite(pred_p).all(-1) & np.isfinite(gt_p).all(-1)
        if valid.sum() < 2:
            continue
        pred_v = pred_p[valid]
        gt_v   = gt_p[valid]
        aligned, _, _ = _se3_align(pred_v, gt_v)
        errors = np.linalg.norm(aligned - gt_v, axis=-1)
        # Total displacement: sum of consecutive-frame steps in GT
        valid_frames = np.where(valid)[0]
        disp = 0.0
        for i in range(1, len(valid_frames)):
            if valid_frames[i] == valid_frames[i - 1] + 1:
                disp += float(np.linalg.norm(gt_v[i] - gt_v[i - 1]))
        if disp < 1e-6:
            continue
        rtes.append(float(errors.mean()) / disp * 100.0)
    return float(np.mean(rtes)) if rtes else float("nan")


# ---------------------------------------------------------------------------
# GT loading  (rich/<gt_split>_body/<scene>/<frame>/<pid>.pkl)
# ---------------------------------------------------------------------------

def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def load_gt_body_data(
    scene_name: str,
    rich_root:  Path,
    split:      str = "test",
) -> dict[int, dict[int, dict]]:
    """Return gt[gt_pid][frame_idx] = {transl, global_orient, body_pose, betas}."""
    gt_root = rich_root / f"{split}_body" / scene_name
    if not gt_root.is_dir():
        return {}
    gt: dict[int, dict[int, dict]] = {}
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
            gt.setdefault(pid, {})[frame_idx] = {
                "transl":        np.asarray(data["transl"],        dtype=np.float32).reshape(3),
                "global_orient": np.asarray(data["global_orient"], dtype=np.float32).reshape(3),
                "body_pose":     np.asarray(
                    data.get("body_pose", np.zeros(63)), dtype=np.float32
                ).reshape(63),
                "betas": np.asarray(
                    data.get("betas", data.get("smplx_betas", np.zeros(10))), dtype=np.float32
                ).reshape(-1)[:10],
            }
    return gt


def load_gt_extrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,4) [R|t] GT extrinsics per camera (camera-from-world)."""
    location  = _scene_to_location(scene_name)
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    exts: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        cam_node = tree.getroot().find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts if exts else None


def load_gt_intrinsics(scene_name: str, rich_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,3) GT intrinsic matrices per camera in original full-res image space."""
    location  = _scene_to_location(scene_name)
    calib_dir = rich_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    intrs: list[np.ndarray] = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        intr_node = tree.getroot().find("Intrinsics")
        if intr_node is None:
            continue
        vals = list(map(float, intr_node.find("data").text.split()))
        intrs.append(np.array(vals, dtype=np.float64).reshape(3, 3))
    return intrs if intrs else None


# ---------------------------------------------------------------------------
# Person ID matching
# ---------------------------------------------------------------------------

def match_ghost_to_gt(
    trans_dict:      dict[int, dict[int, np.ndarray]],  # ghost_pid → {frame → t (VGGT frame)}
    gt_body_data:    dict[int, dict[int, dict]],          # gt_pid   → {frame → params}
    foreground_pids: set[int],
    R_w2ref:         np.ndarray,   # (3,3) RICH-world → VGGT-ref rotation
    t_w2ref:         np.ndarray,   # (3,)  RICH-world → VGGT-ref translation
) -> dict[int, int]:
    """Match ghost person IDs to GT person IDs by minimum mean translation distance.

    GT translations (in RICH world frame) are first transformed to the VGGT
    reference frame (first camera frame) so they can be compared directly with
    the Procrustes DLT predictions.

    Returns {ghost_pid: gt_pid}.
    """
    # Build GT pelvis translations in VGGT reference frame.
    # GT transl is the SMPL-X root offset (pelvis ≈ transl + canonical_joint0).
    # We approximate pelvis ≈ transl for matching purposes (canonical_joint0 is small ~3 cm).
    gt_trans_vggt: dict[int, dict[int, np.ndarray]] = {}
    for gt_pid, frames in gt_body_data.items():
        gt_trans_vggt[gt_pid] = {
            fi: (R_w2ref @ p["transl"].astype(np.float64) + t_w2ref).astype(np.float32)
            for fi, p in frames.items()
        }

    ghost_pids = sorted(foreground_pids)
    gt_pids    = sorted(gt_trans_vggt.keys())
    gt_to_ghost: dict[int, int] = {}

    for gt_pid in gt_pids:
        gt_frames  = gt_trans_vggt[gt_pid]
        best_gpid, best_dist = None, float("inf")
        for gpid in ghost_pids:
            pred_frames = trans_dict.get(gpid, {})
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


# ---------------------------------------------------------------------------
# Inference pipeline helpers
# ---------------------------------------------------------------------------

def load_scene_body_data(
    scene_dir: Path,
    exclude_cameras: list[str] | None = None,
) -> tuple[list[Path], list[dict[int, dict]]]:
    exclude_cameras = set(exclude_cameras or [])
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir()
                      and d.name not in exclude_cameras)
    raw: list[dict[int, dict]] = []
    for cam_dir in cam_dirs:
        cam_persons: dict[int, dict] = {}
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            data = np.load(npz_path, allow_pickle=False)
            cam_persons[pid] = {k: data[k] for k in data.files}
        raw.append(cam_persons)
    return cam_dirs, raw


def build_fusion_tensors(
    raw: list[dict[int, dict]],
    num_joints: int = 55,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    """Assemble (1, T, K, P, J-1, 6), mask, shape tensors for the fusion model."""
    all_pids   = sorted({pid for cam in raw for pid in cam})
    all_frames = sorted({int(fi) for cam in raw for pd in cam.values()
                         for fi in pd["frame_indices"]})
    if not all_pids or not all_frames:
        raise RuntimeError("No person data found.")

    frame_start = all_frames[0]
    T = all_frames[-1] + 1 - frame_start
    K, P = len(raw), len(all_pids)
    J = num_joints - 1  # root excluded
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    pose_arr  = np.zeros((T, K, P, J, 6),  dtype=np.float32)
    mask_arr  = np.zeros((T, K, P),        dtype=np.float32)
    shape_arr = np.zeros((T, K, P, 10),    dtype=np.float32)

    for k, cam in enumerate(raw):
        for pid, pdata in cam.items():
            p  = pid_to_slot[pid]
            fi = pdata["frame_indices"].astype(int)
            go = pdata.get("smplx_global_orient")
            bp = pdata.get("smplx_body_pose")
            if go is None or bp is None:
                continue
            lh    = pdata.get("smplx_left_hand_pose")
            rh    = pdata.get("smplx_right_hand_pose")
            betas = pdata.get("smplx_betas")

            for local_t, global_t in enumerate(fi):
                t = int(global_t) - frame_start
                if t < 0 or t >= T:
                    continue
                parts = [go[local_t].reshape(1, 3), bp[local_t].reshape(21, 3)]
                if lh is not None:
                    parts.append(lh[local_t].reshape(15, 3))
                if rh is not None:
                    parts.append(rh[local_t].reshape(15, 3))
                aa = np.concatenate(parts, axis=0)
                if aa.shape[0] < num_joints:
                    aa = np.concatenate(
                        [aa, np.zeros((num_joints - aa.shape[0], 3), dtype=np.float32)], 0
                    )
                pose_arr[t, k, p]  = _aa_to_6d(aa)[1:]   # root excluded
                mask_arr[t, k, p]  = 1.0
                if betas is not None:
                    shape_arr[t, k, p] = betas[local_t, :10]

    return (
        torch.from_numpy(pose_arr).unsqueeze(0),
        torch.from_numpy(mask_arr).unsqueeze(0),
        torch.from_numpy(shape_arr).unsqueeze(0),
        all_pids,
        frame_start,
    )


def load_fusion_model(checkpoint: Path, device: torch.device) -> PoseFusionModule:
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    emb_dim  = state["joint_id_embedding.weight"].shape[1]
    n_joints = state["joint_id_embedding.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".ff.norm.weight"))
    max_tlen = state["temporal_pe.pe"].shape[0]

    model = PoseFusionModule(
        embedding_dim=emb_dim, num_layers=n_layers,
        num_joints=n_joints, max_temporal_len=max_tlen,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint: emb={emb_dim} layers={n_layers} joints={n_joints}")
    return model


# ---------------------------------------------------------------------------
# Per-scene evaluation (M10: pred-cam + pred-scale + pred-pose, neutral pred vs gendered GT)
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene_dir:        Path,
    scene_name:       str,
    rich_root:        Path,
    fusion_model:     PoseFusionModule,
    device:           torch.device,
    smplx_model_path: Path,
    gt_split:         str = "test",
    exclude_cameras:  list[str] | None = None,
    scale_mode:       str = "centered",
    scale_smooth:     str = "none",
) -> dict[str, float] | None:
    """Run full inference + evaluation for one scene. Returns metric dict or None."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Scene: {scene_name}")

    if not (scene_dir / "vggt_cameras_centered.npz").exists():
        logger.warning("  Missing vggt_cameras_centered.npz — skipping")
        return None

    # ── 1. Load body data ────────────────────────────────────────────────────
    cam_dirs, raw = load_scene_body_data(scene_dir, exclude_cameras=exclude_cameras)
    if not cam_dirs or all(len(c) == 0 for c in raw):
        logger.warning("  No body data — skipping")
        return None

    # ── 2. Fusion model ───────────────────────────────────────────────────────
    try:
        pose_t, mask_t, shape_t, all_pids, frame_start = build_fusion_tensors(raw)
    except RuntimeError as e:
        logger.warning(f"  {e} — skipping")
        return None

    T = pose_t.shape[1]
    P = len(all_pids)
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    with torch.no_grad():
        fused_pose_t = fusion_model(
            pose_t.to(device), mask_t.to(device),
        )
    fused_pose  = fused_pose_t[0].cpu().numpy()                                  # (T, P, 54, 6)

    # Betas per pid: mean SAM3D betas across all cameras and frames.
    _betas_lists: dict[int, list[np.ndarray]] = {}
    for cam_dir in cam_dirs:
        for pid in all_pids:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if bf.exists():
                d = np.load(bf, allow_pickle=False)
                if "smplx_betas" in d.files:
                    _betas_lists.setdefault(pid, []).append(d["smplx_betas"].mean(0))
    betas_by_pid: dict[int, np.ndarray] = {
        pid: np.mean(v, axis=0).astype(np.float32)
        for pid, v in _betas_lists.items()
    }
    for pid in all_pids:
        betas_by_pid.setdefault(pid, np.zeros(10, dtype=np.float32))

    # ── 3. Procrustes DLT ────────────────────────────────────────────────────
    # `placer` (gendered, resolved per-scene) is only used for ghost↔GT pid
    # matching below — every actual metric uses `placer_neutral`.
    try:
        _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
        _smplx_arg = (
            resolve_smplx_models(scene_name, smplx_model_path.parent, _gender_json)
            if _gender_json.exists() else smplx_model_path
        )
        # SAM3D kp2d are in uncropped source pixels; VGGT cameras live in
        # centered-crop space. crop_meta.json (written by center_images.py next
        # to the centered images) supplies the per-camera offset to reconcile them.
        crop_meta_path = rich_root / f"centered_{gt_split}" / scene_name / "crop_meta.json"
        placer = BodyPlacer(scene_dir, _smplx_arg, crop_meta_path=crop_meta_path)
    except Exception as e:
        logger.warning(f"  BodyPlacer init failed: {e} — skipping")
        return None

    fused_pose_by_pid: dict[int, np.ndarray] | None = None
    if fused_pose is not None:
        fused_pose_by_pid = {
            pid: fused_pose[:, pid_to_slot[pid]]   # (T, 54, 6)
            for pid in all_pids
        }

    # ── 3a. GT extrinsics/intrinsics (for camera diagnostics + world frame) ──
    gt_exts_for_scale  = load_gt_extrinsics(scene_name, rich_root)
    gt_intrs_for_scale = load_gt_intrinsics(scene_name, rich_root)
    _ref_name_idx      = int(re.search(r"\d+", cam_dirs[0].name).group()) if cam_dirs else 0

    # Pred scale — triangulation-based (more robust than depth-based).
    try:
        # Build mean SAM3D betas per pid across all cameras and frames
        sam3d_mean_betas: dict[Path, np.ndarray] = {}
        for cam_dir in cam_dirs:
            for pid in all_pids:
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                if not bf.exists():
                    continue
                d = np.load(bf, allow_pickle=False)
                if "smplx_betas" in d.files:
                    sam3d_mean_betas.setdefault(pid, []).append(d["smplx_betas"].mean(0))
        # Per-pid SAM3D average betas (used for scale estimation AND FK reference)
        sam3d_betas_by_pid: dict[int, np.ndarray] = {
            pid: np.mean(v, axis=0).astype(np.float32)
            for pid, v in sam3d_mean_betas.items()
        }
        sam3d_betas_map: dict[Path, np.ndarray] = {}
        for cam_dir in cam_dirs:
            for pid in all_pids:
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                if not bf.exists():
                    continue
                if pid in sam3d_betas_by_pid:
                    sam3d_betas_map[bf] = sam3d_betas_by_pid[pid]

        if scale_mode == "human":
            pred_scale_pf = placer.estimate_scale_human_reference(frame_start=frame_start)
            if scale_smooth == "median":
                _valid = pred_scale_pf[pred_scale_pf > 0]
                if _valid.size:
                    pred_scale_pf = np.full_like(pred_scale_pf, float(np.median(_valid)))
            logger.info(f"  [scale] using human-reference (smooth={scale_smooth})  median={float(np.median(pred_scale_pf)):.4f}")
        else:
            pred_scale_pf = placer.load_mapanything_scale(scale_mode=scale_mode, smooth=scale_smooth)
            if pred_scale_pf is not None:
                logger.info(f"  [scale] using MapAnything ({scale_mode}, smooth={scale_smooth})  median={float(np.median(pred_scale_pf)):.4f}")
            else:
                pred_scale_pf = placer.estimate_scale_triangulated(
                    fused_pose_by_pid=fused_pose_by_pid,
                    pred_betas_map=sam3d_betas_map,
                    frame_start=frame_start,
                )
                logger.info(f"  [scale] using triangulated  median={float(np.median(pred_scale_pf)):.4f}")
    except Exception as e:
        logger.warning(f"  Scale estimation failed: {e} — skipping")
        return None

    # ── 4. GT body data + world-to-ref transform ──────────────────────────
    gt_body_data = load_gt_body_data(scene_name, rich_root, split=gt_split)
    if not gt_body_data:
        logger.warning(f"  No GT found in {gt_split}_body/ — skipping")
        return None

    gt_exts  = gt_exts_for_scale
    _ref_idx = _ref_name_idx
    if gt_exts and _ref_idx < len(gt_exts):
        E_ref   = gt_exts[_ref_idx].astype(np.float64)
        R_w2ref = E_ref[:3, :3]
        t_w2ref = E_ref[:3, 3]
    else:
        R_w2ref = np.eye(3, dtype=np.float64)
        t_w2ref = np.zeros(3, dtype=np.float64)

    # ── 4b. Camera diagnostics (pred vs GT) ──────────────────────────────
    cam_rot_err  = float("nan")
    cam_t_cos    = float("nan")
    cam_t_err_cm = float("nan")
    gt_scale_val = float("nan")
    pred_scale_val_diag = float("nan")
    scale_err_pct = float("nan")

    if gt_exts:
        E0 = gt_exts[_ref_idx].astype(np.float64)
        R0_gt, t0_gt = E0[:3, :3], E0[:3, 3]
        vggt_names = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
        rot_errs, t_coses, gt_scale_vals, t_err_cms_list = [], [], [], []
        _cam_t_data: list[tuple[np.ndarray, np.ndarray]] = []
        for ki, cam_name in enumerate(vggt_names):
            m = re.search(r"\d+", cam_name)
            if not m:
                continue
            gt_idx = int(m.group())
            if gt_idx >= len(gt_exts):
                continue
            Ek    = gt_exts[gt_idx].astype(np.float64)
            Rk_gt = Ek[:3, :3] @ R0_gt.T
            tk_gt = Ek[:3, 3] - Ek[:3, :3] @ R0_gt.T @ t0_gt
            vmask = placer.cam_valid[:, ki]
            if not vmask.any():
                continue
            ext_k = placer.extrinsics[vmask, ki]
            R_med = np.median(ext_k[:, :3, :3], axis=0)
            t_med = np.median(ext_k[:, :3,  3], axis=0)
            U, _, Vt = np.linalg.svd(R_med)
            R_med = U @ Vt
            if np.linalg.det(R_med) < 0:
                U[:, -1] *= -1; R_med = U @ Vt
            if ki == 0:
                continue
            R_err_m = R_med @ Rk_gt.T
            rot_errs.append(float(np.degrees(np.arccos(np.clip((np.trace(R_err_m) - 1) / 2, -1, 1)))))
            pn, gn = np.linalg.norm(t_med), np.linalg.norm(tk_gt)
            if pn > 1e-6 and gn > 1e-6:
                t_coses.append(float(np.dot(t_med / pn, tk_gt / gn)))
            if pn > 1e-6:
                gt_scale_vals.append(gn / pn)
            _cam_t_data.append((t_med, tk_gt))
        if rot_errs:
            cam_rot_err = float(np.mean(rot_errs))
        if t_coses:
            cam_t_cos = float(np.mean(t_coses))
        if gt_scale_vals:
            gt_scale_val = float(np.median(gt_scale_vals))
        valid_spf = pred_scale_pf[pred_scale_pf > 0]
        if valid_spf.size:
            pred_scale_val_diag = float(np.mean(valid_spf))
        if gt_scale_val > 0 and np.isfinite(pred_scale_val_diag):
            scale_err_pct = (pred_scale_val_diag - gt_scale_val) / gt_scale_val * 100.0
        if _cam_t_data and np.isfinite(pred_scale_val_diag):
            t_err_cms_list = [float(np.linalg.norm(tm * pred_scale_val_diag - tg)) * 100.0
                              for tm, tg in _cam_t_data]
            cam_t_err_cm = float(np.mean(t_err_cms_list))

    logger.info(
        f"  Cam rot err = {cam_rot_err:.2f}°  cam_t_cos = {cam_t_cos:.4f}  "
        f"cam_t_err(pred_sc) = {cam_t_err_cm:.1f}cm  "
        f"pred_scale = {pred_scale_val_diag:.4f}  gt_scale = {gt_scale_val:.4f}  "
        f"scale_err = {scale_err_pct:+.1f}%"
    )

    # ── 5. Ghost↔GT pid matching (via gendered placer, canonical scale) ──────
    # Canonical MA-baseline scale is used (independent of --scale) so matching
    # is identical across scale-mode runs — the tested pred_scale_pf still
    # drives every metric below.
    K_cams = len(cam_dirs)
    pid_cam_count: dict[int, int] = defaultdict(int)
    for cam_dir in cam_dirs:
        for f in (cam_dir / "body_data").glob("person_*.npz"):
            pid_cam_count[int(f.stem.split("_")[1])] += 1
    foreground_pids: set[int] = {
        pid for pid, cnt in pid_cam_count.items() if cnt >= max(1, K_cams - 1)
    }

    match_scale = placer.load_mapanything_scale(
        filename="mapanything_scale_baseline.npy", smooth="median")
    if match_scale is None:
        match_scale = pred_scale_pf
    try:
        trans_dict_match, _ = placer.estimate_procrustes_dlt_mhr(
            scale=match_scale,
            all_pids=set(all_pids),
            pred_betas_by_pid=betas_by_pid,
            fused_pose_by_pid=fused_pose_by_pid,
            frame_start=frame_start,
        )
    except Exception as e:
        logger.warning(f"  Placer (matching) failed: {e} — skipping")
        return None

    pid_match = match_ghost_to_gt(
        trans_dict_match, gt_body_data, foreground_pids, R_w2ref, t_w2ref
    )
    if not pid_match:
        logger.warning("  No ghost↔GT pid matches found — skipping")
        return None
    n_matched = len(pid_match)

    # ── 6. GT joints — GENDERED model (M10: RICH GT as it actually exists; ──
    # pred FK below stays neutral, matching production).
    J_body = _N_SMPL_JOINTS
    neutral_path = smplx_model_path.parent / "SMPLX_NEUTRAL.pkl"
    placer_neutral = BodyPlacer(scene_dir, neutral_path, crop_meta_path=crop_meta_path)

    def _build_gt_joints(_plc):
        """GT 24-SMPL joints + roots (T, n_matched, ...) FK'd with the given placer's model."""
        gj = np.full((T, n_matched, J_body, 3), np.nan, dtype=np.float32)
        gr = np.full((T, n_matched, 3),          np.nan, dtype=np.float32)
        for slot, (ghost_pid, gt_pid) in enumerate(sorted(pid_match.items())):
            for frame_idx, params in gt_body_data[gt_pid].items():
                t_rel = frame_idx - frame_start
                if not (0 <= t_rel < T):
                    continue
                _, V_gt = _plc._smplx_fk(
                    params["betas"][np.newaxis],
                    params["body_pose"][np.newaxis],
                    params["global_orient"][np.newaxis],
                    return_verts=True,
                )
                J_gt = _verts_to_smpl24(V_gt[0]) + params["transl"]   # (24, 3) world frame
                gj[t_rel, slot] = J_gt
                gr[t_rel,  slot] = J_gt[0]
        return gj, gr

    gt_joints, gt_roots = _build_gt_joints(placer)

    # ── 7. Placement — pred cam + pred scale + pred pose, neutral model ─────
    fused_pose_by_pid_mod: dict[int, np.ndarray] = {
        pid: fused_pose[:, pid_to_slot[pid]] for pid in all_pids
    }
    try:
        trans_dict, orient_dict = placer_neutral.estimate_procrustes_dlt_mhr(
            scale=pred_scale_pf,
            all_pids=set(all_pids),
            pred_betas_by_pid=sam3d_betas_by_pid,
            fused_pose_by_pid=fused_pose_by_pid_mod,
            frame_start=frame_start,
        )
    except Exception as e:
        logger.warning(f"  Placer failed: {e} — skipping")
        return None

    pred_joints = np.full((T, P, J_body, 3), np.nan, dtype=np.float32)
    pred_roots  = np.full((T, P, 3),          np.nan, dtype=np.float32)

    for pid, frames_t in trans_dict.items():
        if pid not in pid_to_slot:
            continue
        p_slot  = pid_to_slot[pid]
        betas_p = sam3d_betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))
        for global_t, pelvis_world in sorted(frames_t.items()):
            t_rel = int(global_t) - frame_start
            R_mat = orient_dict.get(pid, {}).get(global_t)
            if not (0 <= t_rel < T) or R_mat is None:
                continue
            body_pose_aa = _6d_to_aa(fused_pose[t_rel, p_slot, :21])
            J_can_smplx, V_can = placer_neutral._smplx_fk(
                betas_p[np.newaxis],
                body_pose_aa.reshape(63)[np.newaxis],
                np.zeros((1, 3), dtype=np.float32),
                return_verts=True,
            )
            J_can  = _verts_to_smpl24(V_can[0])    # (24, 3) canonical SMPL joints
            # Pivot about the SMPL-X pelvis: the placer's R_mat/pelvis_world are
            # defined in the SMPL-X convention, so we reconstruct the placed SMPL-X
            # mesh (equivalently its J24 joints) about the SMPL-X root, NOT J_can[0].
            pelvis_smplx = J_can_smplx[0, 0]       # SMPL-X canonical pelvis
            J_world = (R_mat @ (J_can - pelvis_smplx).T).T + pelvis_world
            pred_joints[t_rel, p_slot] = J_world
            pred_roots[t_rel,  p_slot] = J_world[0]

    # Gather matched-person arrays
    pred_joints_m = np.full((T, n_matched, J_body, 3), np.nan, dtype=np.float32)
    pred_roots_m  = np.full((T, n_matched, 3),          np.nan, dtype=np.float32)
    for slot, (ghost_pid, _) in enumerate(sorted(pid_match.items())):
        g_slot = pid_to_slot[ghost_pid]
        pred_joints_m[:, slot] = pred_joints[:, g_slot]
        pred_roots_m[:, slot]  = pred_roots[:,  g_slot]

    valid = (
        np.isfinite(pred_joints_m).all((-2, -1)) &
        np.isfinite(gt_joints).all((-2, -1))
    )
    n_valid = int(valid.sum())

    wa    = metric_wa_mpjpe(pred_joints_m, gt_joints, valid)
    w     = metric_w_mpjpe( pred_joints_m, gt_joints, valid)
    wa100 = metric_wa_mpjpe_100(pred_joints_m, gt_joints, valid)
    w100  = metric_w_mpjpe_100( pred_joints_m, gt_joints, valid)
    ga  = metric_ga_mpjpe(pred_joints_m, gt_joints, valid)
    pa  = metric_pa_mpjpe(pred_joints_m, gt_joints, valid)
    rte = metric_rte(pred_roots_m, gt_roots)

    # Raw root error (pred in VGGT/GT-cam frame, GT in RICH world)
    raw_errs = []
    orient_errs = []
    for slot, (ghost_pid, gt_pid) in enumerate(sorted(pid_match.items())):
        for frame_idx, params in gt_body_data[gt_pid].items():
            t_rel = frame_idx - frame_start
            if not (0 <= t_rel < T):
                continue
            if np.isfinite(pred_roots_m[t_rel, slot]).all():
                # GT pelvis = J[0] + transl (not just transl — canonical J[0] is ~33cm offset)
                _, V_gt_body = placer_neutral._smplx_fk(
                    params["betas"][np.newaxis],
                    params["body_pose"][np.newaxis],
                    params["global_orient"][np.newaxis],
                    return_verts=True,
                )
                gt_pelvis_world = _verts_to_smpl24(V_gt_body[0])[0] + params["transl"].astype(np.float64)
                gt_root_ref = R_w2ref @ gt_pelvis_world + t_w2ref
                raw_errs.append(float(np.linalg.norm(pred_roots_m[t_rel, slot] - gt_root_ref)))
            R_pred_mat = orient_dict.get(ghost_pid, {}).get(frame_idx)
            if R_pred_mat is not None:
                R_gt_w = SciR.from_rotvec(params["global_orient"].astype(np.float64)).as_matrix()
                R_gt_ref = R_w2ref @ R_gt_w
                R_err = R_pred_mat.astype(np.float64) @ R_gt_ref.T
                orient_errs.append(float(np.degrees(
                    np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                )))

    raw_root_err_cm     = float(np.median(raw_errs))     * 100.0 if raw_errs     else float("nan")
    root_orient_err_deg = float(np.median(orient_errs))          if orient_errs  else float("nan")

    logger.info(
        f"  WA={wa:6.1f}mm  W={w:6.1f}mm  WA100={wa100:6.1f}mm  W100={w100:6.1f}mm  "
        f"GA={ga:6.1f}mm  PA={pa:6.1f}mm  "
        f"RTE={rte:5.2f}%  raw_root={raw_root_err_cm:.1f}cm  orient={root_orient_err_deg:.1f}°"
    )

    return {
        "n_valid": n_valid,
        "cam_rot_err": cam_rot_err, "cam_t_cos": cam_t_cos,
        "cam_t_err_cm": cam_t_err_cm,
        "pred_scale": pred_scale_val_diag, "gt_scale": gt_scale_val,
        "scale_err_pct": scale_err_pct,
        "wa_mpjpe": wa, "w_mpjpe": w, "wa_mpjpe_100": wa100, "w_mpjpe_100": w100,
        "ga_mpjpe": ga, "pa_mpjpe": pa, "rte": rte,
        "raw_root_err_cm": raw_root_err_cm, "root_orient_err_deg": root_orient_err_deg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ghost on RICH test set — production config (neutral pred vs gendered GT, M10)."
    )
    parser.add_argument("--ghost_output_root", required=True, type=Path,
                        help="Root dir of ghost test outputs (contains scene subdirs).")
    parser.add_argument("--rich_root",         required=True, type=Path,
                        help="RICH dataset root (must have <gt_split>_body/ and scan_calibration/).")
    parser.add_argument("--checkpoint",        required=True, type=Path,
                        help="PoseFusionModule checkpoint (.pt).")
    parser.add_argument("--smplx_model",       required=True, type=Path,
                        help="Path to SMPLX_NEUTRAL.pkl.")
    parser.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_scenes",        type=int, default=None,
                        help="Limit evaluation to first N scenes (for debugging).")
    parser.add_argument("--gt_split",          default="test",
                        help="GT body split: 'test' or 'train'")
    parser.add_argument("--scenes",            default="",
                        help="Comma-separated scene names to evaluate (default: all)")
    parser.add_argument("--skip_scenes",       default="",
                        help="Comma-separated scene names to skip")
    parser.add_argument("--skip_cameras",      default="",
                        help="Per-scene camera exclusions: 'scene:cam1,cam2;scene2:cam3'")
    parser.add_argument("--scale",              default="centered",
                        choices=["centered", "baseline", "human"],
                        help="MapAnything scale variant "
                             "(centered=legacy depth-ratio, baseline=images-only camera baselines, "
                             "human=MA-free self-consistent human-reference triangulation).")
    parser.add_argument("--scale_smooth",       default="none",
                        choices=["none", "median"],
                        help="Temporal denoise of per-frame scale: median=one robust scalar "
                             "per scene (kills VGGT per-frame jitter).")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.info("Loading fusion model ...")
    fusion_model = load_fusion_model(args.checkpoint, device)

    skip_scenes: set[str] = {s.strip() for s in args.skip_scenes.split(",") if s.strip()}

    # Parse --skip_cameras "scene:cam1,cam2;scene2:cam3" → dict[scene_name, list[cam]]
    skip_cameras: dict[str, list[str]] = {}
    for entry in args.skip_cameras.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        scene_part, _, cams_part = entry.partition(":")
        skip_cameras[scene_part.strip()] = [c.strip() for c in cams_part.split(",") if c.strip()]

    only_scenes: set[str] = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = sorted(
        d for d in args.ghost_output_root.iterdir()
        if d.is_dir() and (d / "vggt_cameras_centered.npz").exists()
        and d.name not in skip_scenes
        and (not only_scenes or d.name in only_scenes)
    )
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    logger.info(f"Found {len(scenes)} scene(s) (skipping: {skip_scenes or 'none'}).")

    all_results: list[dict] = []
    for scene_dir in scenes:
        result = evaluate_scene(
            scene_dir=scene_dir,
            scene_name=scene_dir.name,
            rich_root=args.rich_root,
            fusion_model=fusion_model,
            device=device,
            smplx_model_path=args.smplx_model,
            gt_split=args.gt_split,
            exclude_cameras=skip_cameras.get(scene_dir.name),
            scale_mode=args.scale,
            scale_smooth=args.scale_smooth,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No scenes evaluated successfully.")
        return

    def agg(key: str) -> float:
        vals = [r[key] for r in all_results if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def fmt(v: float, unit: str = "") -> str:
        if np.isnan(v):
            return f"{'—':>14}"
        if unit == "mm":
            return f"{v:>12.1f}mm"
        if unit == "%":
            return f"{v:>13.2f}%"
        if unit == "deg":
            return f"{v:>13.1f}°"
        if unit == "cm":
            return f"{v:>12.1f}cm"
        return f"{v:>14.4f}"

    print(f"\n{'='*65}")
    print(f"AGGREGATE  ({len(all_results)} scenes evaluated)  —  M10 neutral-pred vs gendered-GT")
    print(f"{'='*65}")

    chromm = {"wa_mpjpe_100": "53.1 mm", "w_mpjpe_100": "79.0 mm", "rte": "1.4 %"}

    def mrow(label, key, unit):
        v = agg(key)
        row = f"  {label:<26}  {fmt(v, unit):>14}"
        if key in chromm:
            row += f"   (CHROMM multi: {chromm[key]})"
        print(row)

    mrow("WA-MPJPE",      "wa_mpjpe",      "mm")
    mrow("W-MPJPE",       "w_mpjpe",       "mm")
    mrow("WA-MPJPE-100",  "wa_mpjpe_100",  "mm")
    mrow("W-MPJPE-100",   "w_mpjpe_100",   "mm")
    mrow("GA-MPJPE",      "ga_mpjpe",      "mm")
    mrow("PA-MPJPE",      "pa_mpjpe",      "mm")
    mrow("RTE",           "rte",           "%")
    print()
    print("  --- Diagnostics ---")
    print(f"  {'Cam rot err (°)':<26}  {agg('cam_rot_err'):>14.2f}")
    print(f"  {'Cam t_cos':<26}  {agg('cam_t_cos'):>14.4f}")
    print(f"  {'Cam t_err pred-sc (cm)':<26}  {agg('cam_t_err_cm'):>14.1f}")
    print(f"  {'Pred scale':<26}  {agg('pred_scale'):>14.4f}")
    print(f"  {'GT scale':<26}  {agg('gt_scale'):>14.4f}")
    print(f"  {'Scale err (%)':<26}  {agg('scale_err_pct'):>13.1f}%")
    print(f"  {'Raw root err (cm)':<26}  {agg('raw_root_err_cm'):>13.1f}cm")
    print(f"  {'Root orient err (°)':<26}  {agg('root_orient_err_deg'):>13.1f}°")
    print()


if __name__ == "__main__":
    main()
