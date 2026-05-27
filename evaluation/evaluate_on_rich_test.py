"""Evaluate ghost pipeline on RICH test set using CHROMM paper metrics.

Metrics (all in millimetres unless noted):
  WA-MPJPE  — World-Aligned MPJPE: single Sim(3) over ALL frames jointly.
  W-MPJPE   — World MPJPE: Sim(3) from first 2 frames applied to all.
  GA-MPJPE  — Group-Aligned MPJPE: per-frame Sim(3) over all persons jointly.
  PA-MPJPE  — Procrustes-Aligned MPJPE: per-person per-frame Sim(3).
  RTE       — Root Translation Error (%): SE(3) aligned, normalised by displacement.

CHROMM (ours-multi) on RICH: WA=53.1 mm  W=79.0 mm  RTE=1.4%

Usage
-----
    python evaluation/evaluate_on_rich_test.py \\
        --ghost_output_root /path/to/ghost_outputs/rich_test \\
        --rich_root         /path/to/rich \\
        --checkpoint        /path/to/fusion_checkpoint.pt \\
        --smplx_model       /path/to/SMPLX_NEUTRAL.pkl \\
        [--device cuda] [--max_scenes N] [--gt_split test]

The script processes every scene subdirectory in ``ghost_output_root`` that
contains a ``vggt_cameras.npz`` file (i.e. scenes that completed the VGGT
preprocessing step).  For each scene it:
  1. Loads SAM3D body estimates from body_data/ directories.
  2. Runs the FusionWithBetas model to refine body pose and shape.
  3. Estimates metric translation + global orientation via Procrustes DLT.
  4. Runs SMPL-X FK to obtain world-frame 3D joints for predictions.
  5. Loads GT from ``<rich_root>/<gt_split>_body/<scene_name>/``.
  6. Runs SMPL-X FK on GT parameters.
  7. Matches ghost person IDs to GT person IDs by translation proximity.
  8. Computes the five CHROMM metrics and prints per-scene + aggregate results.
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

from fusion.fusion_module_v2 import FusionWithBetas, PoseFusionModule, BetasAggregator
from fusion.placer import BodyPlacer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# SMPL-X body joints used for MPJPE (pelvis + 21 body joints; no hands, face).
# This matches the standard 22-joint body evaluation used for RICH/SMPL-X papers.
_BODY_JOINT_IDX = list(range(22))

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


def load_scene_body_data(scene_dir: Path) -> tuple[list[Path], list[dict[int, dict]]]:
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and (d / "body_data").is_dir())
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


def load_fusion_model(checkpoint: Path, device: torch.device) -> FusionWithBetas:
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    emb_dim   = state["pose_module.joint_id_embedding.weight"].shape[1]
    n_joints  = state["pose_module.joint_id_embedding.weight"].shape[0]
    n_layers  = sum(1 for k in state
                    if k.startswith("pose_module.layers.") and k.endswith(".ff.norm.weight"))
    max_tlen  = state["pose_module.temporal_pe.pe"].shape[0]

    pose_module = PoseFusionModule(
        embedding_dim=emb_dim, num_layers=n_layers,
        num_joints=n_joints, max_temporal_len=max_tlen,
    )
    betas_agg = BetasAggregator(n_betas=10, embedding_dim=64, num_inducing=4, num_heads=4, dropout=0.3, input_noise_std=0.5)
    model = FusionWithBetas(pose_module, betas_agg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint: emb={emb_dim} layers={n_layers} joints={n_joints}")
    return model


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene_dir:      Path,
    scene_name:     str,
    rich_root:      Path,
    fusion_model:   FusionWithBetas,
    device:         torch.device,
    smplx_model_path: Path,
    gt_split:       str = "test",
) -> dict[str, float] | None:
    """Run full inference + evaluation for one scene. Returns metric dict or None."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"Scene: {scene_name}")

    if not (scene_dir / "vggt_cameras.npz").exists():
        logger.warning("  Missing vggt_cameras.npz — skipping")
        return None

    # ── 1. Load body data ────────────────────────────────────────────────────
    cam_dirs, raw = load_scene_body_data(scene_dir)
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
        fused_pose_t, betas_out = fusion_model(
            pose_t.to(device), mask_t.to(device), shape=shape_t.to(device)
        )
    fused_pose  = fused_pose_t[0].cpu().numpy()                                  # (T, P, 54, 6)
    fused_betas = betas_out[0].cpu().numpy() if betas_out is not None else None  # (P, 10) or None

    # Betas per pid: prefer fused model output, fall back to mean SAM3D betas.
    if fused_betas is not None:
        betas_by_pid = {pid: fused_betas[i] for i, pid in enumerate(all_pids)}
    else:
        betas_by_pid: dict[int, np.ndarray] = {}
        for cam_dir in cam_dirs:
            for pid in all_pids:
                if pid in betas_by_pid:
                    continue
                bf = cam_dir / "body_data" / f"person_{pid}.npz"
                if bf.exists():
                    d = np.load(bf, allow_pickle=False)
                    if "smplx_betas" in d.files:
                        betas_by_pid[pid] = d["smplx_betas"].mean(axis=0)
        for pid in all_pids:
            betas_by_pid.setdefault(pid, np.zeros(10, dtype=np.float32))

    # ── 3. Procrustes DLT ────────────────────────────────────────────────────
    try:
        placer = BodyPlacer(scene_dir, smplx_model_path)
    except Exception as e:
        logger.warning(f"  BodyPlacer init failed: {e} — skipping")
        return None

    fused_betas_map = {
        cam_dir / "body_data" / f"person_{pid}.npz": betas_by_pid[pid]
        for cam_dir in cam_dirs
        for pid in all_pids
        if (cam_dir / "body_data" / f"person_{pid}.npz").exists()
    }
    fused_pose_by_pid: dict[int, np.ndarray] | None = None
    if fused_pose is not None:
        fused_pose_by_pid = {
            pid: fused_pose[:, pid_to_slot[pid]]   # (T, 54, 6)
            for pid in all_pids
        }

    try:
        scale_pf = placer.estimate_scale_per_frame(fused_betas_map=fused_betas_map)
        trans_dict, orient_dict = placer.estimate_procrustes_dlt(
            scale=scale_pf,
            all_pids=set(all_pids),
            pred_betas_by_pid=betas_by_pid,
            fused_pose_by_pid=fused_pose_by_pid,
            frame_start=frame_start,
        )
    except Exception as e:
        logger.warning(f"  Placer failed: {e} — skipping")
        return None

    # ── 4. Predicted world-frame joints (in VGGT reference frame) ────────────
    J_body = len(_BODY_JOINT_IDX)
    pred_joints = np.full((T, P, J_body, 3), np.nan, dtype=np.float32)
    pred_roots  = np.full((T, P, 3),          np.nan, dtype=np.float32)

    for pid, frames_t in trans_dict.items():
        if pid not in pid_to_slot:
            continue
        p_slot  = pid_to_slot[pid]
        betas_p = betas_by_pid.get(pid, np.zeros(10, dtype=np.float32))

        for global_t, pelvis_world in sorted(frames_t.items()):
            t_rel  = int(global_t) - frame_start
            R_mat  = orient_dict.get(pid, {}).get(global_t)
            if not (0 <= t_rel < T) or R_mat is None:
                continue

            # Fused body_pose: joints 1-21 in 6D → axis-angle → (63,)
            body_pose_aa = _6d_to_aa(fused_pose[t_rel, p_slot, :21])  # (21, 3)
            body_pose    = body_pose_aa.reshape(63)

            # FK with zero global_orient and zero transl → canonical joints
            J_can = placer._smplx_fk(
                betas_p[np.newaxis],
                body_pose[np.newaxis],
                np.zeros((1, 3), dtype=np.float32),
            )[0]  # (55, 3)

            # Apply Procrustes rotation + place pelvis at pelvis_world.
            # J_world[j] = R @ (J_can[j] - J_can[0]) + pelvis_world
            J_world = (R_mat @ (J_can - J_can[0]).T).T + pelvis_world   # (55, 3)

            pred_joints[t_rel, p_slot] = J_world[_BODY_JOINT_IDX]
            pred_roots[t_rel,  p_slot] = J_world[0]  # pelvis

    # ── 5. GT loading and world-to-VGGT-reference transform ─────────────────
    gt_body_data = load_gt_body_data(scene_name, rich_root, split=gt_split)
    if not gt_body_data:
        logger.warning(f"  No GT found in {gt_split}_body/ — skipping")
        return None

    # Build R_w2ref, t_w2ref: RICH world frame → VGGT reference frame.
    # The VGGT reference frame is the first cam_dir (first valid camera).
    gt_exts = load_gt_extrinsics(scene_name, rich_root)
    _m_ref  = re.search(r"\d+", cam_dirs[0].name) if cam_dirs else None
    _ref_idx = int(_m_ref.group()) if _m_ref else 0
    if gt_exts and _ref_idx < len(gt_exts):
        E_ref    = gt_exts[_ref_idx].astype(np.float64)
        R_w2ref  = E_ref[:3, :3]
        t_w2ref  = E_ref[:3, 3]
    else:
        R_w2ref = np.eye(3, dtype=np.float64)
        t_w2ref = np.zeros(3, dtype=np.float64)

    # ── 5b. Camera + scale diagnostics ──────────────────────────────────────
    cam_rot_err   = float("nan")
    cam_t_cos     = float("nan")
    gt_scale_val  = float("nan")
    pred_scale_val = float("nan")
    scale_err_pct  = float("nan")

    if gt_exts:
        E0     = gt_exts[_ref_idx].astype(np.float64)
        R0_gt  = E0[:3, :3]
        t0_gt  = E0[:3,  3]

        vggt_names = [n.decode() if isinstance(n, bytes) else n
                      for n in placer.camera_names]
        rot_errs, t_coses, gt_scale_vals = [], [], []

        for ki, cam_name in enumerate(vggt_names):
            m = re.search(r"\d+", cam_name)
            if not m:
                continue
            gt_idx = int(m.group())
            if gt_idx >= len(gt_exts):
                continue

            # GT extrinsic re-rooted to first available camera
            Ek    = gt_exts[gt_idx].astype(np.float64)
            Rk_gt = Ek[:3, :3] @ R0_gt.T
            tk_gt = Ek[:3,  3] - Ek[:3, :3] @ R0_gt.T @ t0_gt

            # Predicted extrinsic: median over valid frames, re-orthogonalised
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

            if ki == 0:   # reference camera — skip (both [I|0] by construction)
                continue

            # Rotation error
            R_err = R_med @ Rk_gt.T
            angle = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
            rot_errs.append(angle)

            # Translation direction cosine
            pn, gn = np.linalg.norm(t_med), np.linalg.norm(tk_gt)
            if pn > 1e-6 and gn > 1e-6:
                t_coses.append(float(np.dot(t_med / pn, tk_gt / gn)))

            # GT scale: ||t_gt_baseline|| / ||t_pred_baseline||
            if pn > 1e-6:
                gt_scale_vals.append(gn / pn)

        if rot_errs:
            cam_rot_err = float(np.mean(rot_errs))
        if t_coses:
            cam_t_cos = float(np.mean(t_coses))
        if gt_scale_vals:
            gt_scale_val = float(np.median(gt_scale_vals))

        valid_scale = scale_pf[scale_pf > 0]
        if valid_scale.size:
            pred_scale_val = float(np.mean(valid_scale))
        if gt_scale_val > 0 and np.isfinite(pred_scale_val):
            scale_err_pct = (pred_scale_val - gt_scale_val) / gt_scale_val * 100.0

    logger.info(
        f"  Cam rot err = {cam_rot_err:.2f}°  |  Cam t_cos = {cam_t_cos:.4f}  |  "
        f"pred_scale = {pred_scale_val:.4f}  gt_scale = {gt_scale_val:.4f}  "
        f"scale_err = {scale_err_pct:+.1f}%"
    )

    # ── 6. Ghost↔GT pid matching ─────────────────────────────────────────────
    K = len(cam_dirs)
    pid_cam_count: dict[int, int] = defaultdict(int)
    for cam_dir in cam_dirs:
        for f in (cam_dir / "body_data").glob("person_*.npz"):
            pid_cam_count[int(f.stem.split("_")[1])] += 1
    foreground_pids: set[int] = {
        pid for pid, cnt in pid_cam_count.items() if cnt >= max(1, K - 1)
    }

    pid_match = match_ghost_to_gt(
        trans_dict, gt_body_data, foreground_pids, R_w2ref, t_w2ref
    )
    if not pid_match:
        logger.warning("  No ghost↔GT pid matches found — skipping")
        return None

    n_matched = len(pid_match)

    # ── 7. GT world-frame joints (in RICH world frame, then kept as-is) ──────
    # Metrics use Sim(3)/SE(3) alignment, which absorbs the VGGT↔RICH coordinate
    # difference; no explicit frame transform on GT joints is needed.
    gt_joints  = np.full((T, n_matched, J_body, 3), np.nan, dtype=np.float32)
    gt_roots   = np.full((T, n_matched, 3),          np.nan, dtype=np.float32)
    pred_joints_m = np.full_like(gt_joints,  np.nan)
    pred_roots_m  = np.full_like(gt_roots,   np.nan)

    for slot, (ghost_pid, gt_pid) in enumerate(sorted(pid_match.items())):
        # Copy matched pred slots
        g_slot = pid_to_slot[ghost_pid]
        pred_joints_m[:, slot] = pred_joints[:, g_slot]
        pred_roots_m[:, slot]  = pred_roots[:,  g_slot]

        # GT joints: run FK with GT params (in RICH world frame)
        gt_pdata = gt_body_data[gt_pid]
        for frame_idx, params in gt_pdata.items():
            t_rel = frame_idx - frame_start
            if not (0 <= t_rel < T):
                continue
            J_gt_zero_transl = placer._smplx_fk(
                params["betas"][np.newaxis],
                params["body_pose"][np.newaxis],
                params["global_orient"][np.newaxis],
            )[0]   # (55, 3), zero transl → add gt_transl below
            J_gt_world = J_gt_zero_transl + params["transl"]

            gt_joints[t_rel, slot] = J_gt_world[_BODY_JOINT_IDX]
            gt_roots[t_rel,  slot] = J_gt_world[0]

    # ── 7b. Raw root translation error (no alignment) ────────────────────────
    # Pred roots are in VGGT reference frame; GT roots are in RICH world frame.
    # Apply R_w2ref / t_w2ref to GT to bring both into the same frame.
    raw_root_err_cm = float("nan")
    raw_errs = []
    for slot, (ghost_pid, gt_pid) in enumerate(sorted(pid_match.items())):
        gt_pdata = gt_body_data[gt_pid]
        for frame_idx, params in gt_pdata.items():
            t_rel = frame_idx - frame_start
            if not (0 <= t_rel < T):
                continue
            if not np.isfinite(pred_roots_m[t_rel, slot]).all():
                continue
            gt_root_vggt = R_w2ref @ params["transl"].astype(np.float64) + t_w2ref
            raw_errs.append(float(np.linalg.norm(pred_roots_m[t_rel, slot] - gt_root_vggt)))
    if raw_errs:
        raw_root_err_cm = float(np.median(raw_errs)) * 100.0

    logger.info(f"  Raw root error (median, no alignment) = {raw_root_err_cm:.1f} cm")

    # ── 8. Validity mask ──────────────────────────────────────────────────────
    valid = (
        np.isfinite(pred_joints_m).all((-2, -1)) &
        np.isfinite(gt_joints).all((-2, -1))
    )  # (T, n_matched)
    n_valid = int(valid.sum())
    if n_valid == 0:
        logger.warning("  No valid (pred, GT) frame-person pairs — skipping")
        return None
    logger.info(f"  Matched persons: {n_matched}  |  valid frames×persons: {n_valid}/{T * n_matched}")

    # ── 9. Compute CHROMM metrics ─────────────────────────────────────────────
    wa  = metric_wa_mpjpe(pred_joints_m, gt_joints, valid)
    w   = metric_w_mpjpe( pred_joints_m, gt_joints, valid)
    ga  = metric_ga_mpjpe(pred_joints_m, gt_joints, valid)
    pa  = metric_pa_mpjpe(pred_joints_m, gt_joints, valid)
    rte = metric_rte(pred_roots_m, gt_roots)

    logger.info(
        f"  WA-MPJPE = {wa:6.1f} mm   W-MPJPE = {w:6.1f} mm   "
        f"GA-MPJPE = {ga:6.1f} mm   PA-MPJPE = {pa:6.1f} mm   RTE = {rte:5.2f}%"
    )
    return {
        "wa_mpjpe": wa, "w_mpjpe": w, "ga_mpjpe": ga, "pa_mpjpe": pa, "rte": rte,
        "n_valid": n_valid,
        "cam_rot_err": cam_rot_err, "cam_t_cos": cam_t_cos,
        "pred_scale": pred_scale_val, "gt_scale": gt_scale_val,
        "scale_err_pct": scale_err_pct,
        "raw_root_err_cm": raw_root_err_cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ghost on RICH test set (CHROMM paper metrics)."
    )
    parser.add_argument("--ghost_output_root", required=True, type=Path,
                        help="Root dir of ghost test outputs (contains scene subdirs).")
    parser.add_argument("--rich_root",         required=True, type=Path,
                        help="RICH dataset root (must have <gt_split>_body/ and scan_calibration/).")
    parser.add_argument("--checkpoint",        required=True, type=Path,
                        help="FusionWithBetas checkpoint (.pt).")
    parser.add_argument("--smplx_model",       required=True, type=Path,
                        help="Path to SMPLX_NEUTRAL.pkl.")
    parser.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_scenes",        type=int, default=None,
                        help="Limit evaluation to first N scenes (for debugging).")
    parser.add_argument("--gt_split",          default="test",
                        help="GT split to use: 'test' or 'train' (default: test).")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.info("Loading fusion model ...")
    fusion_model = load_fusion_model(args.checkpoint, device)

    scenes = sorted(
        d for d in args.ghost_output_root.iterdir()
        if d.is_dir() and (d / "vggt_cameras.npz").exists()
    )
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    logger.info(f"Found {len(scenes)} scene(s).")

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
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No scenes evaluated successfully.")
        return

    def agg(key: str) -> float:
        vals = [r[key] for r in all_results if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'='*65}")
    print(f"AGGREGATE  ({len(all_results)} scenes evaluated)")
    print(f"{'='*65}")
    print(f"  {'Metric':<26}  {'Ghost (ours)':>14}  {'CHROMM multi':>14}")
    print(f"  {'-'*26}  {'-'*14}  {'-'*14}")
    print(f"  {'WA-MPJPE':<26}  {agg('wa_mpjpe'):>12.1f}mm  {'53.1 mm':>14}")
    print(f"  {'W-MPJPE':<26}  {agg('w_mpjpe'):>12.1f}mm  {'79.0 mm':>14}")
    print(f"  {'GA-MPJPE':<26}  {agg('ga_mpjpe'):>12.1f}mm  {'—':>14}")
    print(f"  {'PA-MPJPE':<26}  {agg('pa_mpjpe'):>12.1f}mm  {'—':>14}")
    print(f"  {'RTE':<26}  {agg('rte'):>13.2f}%  {'1.4 %':>14}")
    print()
    print(f"  --- Diagnostics ---")
    print(f"  {'Cam rot err (°)':<26}  {agg('cam_rot_err'):>14.2f}")
    print(f"  {'Cam t_cos':<26}  {agg('cam_t_cos'):>14.4f}")
    print(f"  {'Pred scale':<26}  {agg('pred_scale'):>14.4f}")
    print(f"  {'GT scale':<26}  {agg('gt_scale'):>14.4f}")
    print(f"  {'Scale err (%)':<26}  {agg('scale_err_pct'):>13.1f}%")
    print(f"  {'Raw root err (cm)':<26}  {agg('raw_root_err_cm'):>14.1f}")
    print()


if __name__ == "__main__":
    main()
