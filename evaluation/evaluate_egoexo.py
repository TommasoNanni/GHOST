"""Evaluate ghost pipeline on EgoExo4D validation set.

Metrics (millimetres):
  W-MPJPE†  — world MPJPE after SE(3) camera-pose alignment (no scale), per
               the CHROMM / HSfM single-frame protocol.
  PA-MPJPE  — per-person Procrustes-Aligned MPJPE (Sim3, scale allowed).

GT source: keypoints_gt.json (triangulated COCO-17 body joints in world frame,
metres) + gopro_calibs.csv (GT GoPro camera positions in world frame).

Only the 12 limb joints shared between COCO-17 and SMPL-X body joints are used:
shoulders, elbows, wrists, hips, knees, ankles.  Face joints (nose, eyes, ears)
are excluded because they do not correspond to standard SMPL-X body joints.

Scenes with hand-only GT (bike, covid tasks) are automatically skipped.
108 of the 182 validation scenes have body GT and are evaluated.

Usage
-----
    pixi run python evaluation/evaluate_egoexo.py \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d \\
        --gt_root    /capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt \\
        [--smplx_model body_models/SMPLX_NEUTRAL.pkl] \\
        [--max_scenes N] [--scene SCENE_NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint mapping: GT keypoints_gt.json name → SMPL-X body joint index (0-21)
# ---------------------------------------------------------------------------
GT_TO_SMPLX: dict[str, int] = {
    "left-shoulder":  16,
    "right-shoulder": 17,
    "left-elbow":     18,
    "right-elbow":    19,
    "left-wrist":     20,
    "right-wrist":    21,
    "left-hip":        1,
    "right-hip":       2,
    "left-knee":       4,
    "right-knee":      5,
    "left-ankle":      7,
    "right-ankle":     8,
}

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
        num_views_for_3d > 0 and that appear in GT_TO_SMPLX are included.
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
        if name in GT_TO_SMPLX and val.get("num_views_for_3d", 0) > 0:
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


def _smplx_forward(params: dict[str, np.ndarray], model) -> np.ndarray:
    """Run SMPL-X FK and return the 22 body joints (J=22, 3) in camera frame."""
    def _t(key):
        return torch.tensor(params[key], dtype=torch.float32)
    with torch.no_grad():
        out = model(
            global_orient=_t("smplx_global_orient"),
            body_pose=_t("smplx_body_pose"),
            betas=_t("smplx_betas"),
            transl=_t("smplx_transl"),
        )
    return out.joints.cpu().numpy()[0, :22]   # (22, 3) in camera frame


def load_predictions(
    ghost_scene_dir: Path,
    gt_frame_idx: int,
    smplx_model,
) -> list[tuple[str, int, np.ndarray]]:
    """Load per-camera SMPL-X predictions and transform to metric cam0 frame.

    Returns list of (cam_id, global_pid, joints_cam0_metric) tuples.

    Coordinate transform:
        SAM3D produces joints in camera-k frame (metric depth via MoGe2).
        VGGT gives [R_k | t_k] mapping world(cam0) → cam_k (up-to-scale).
        MapAnything scale s converts VGGT baseline to metric.
        Inverse: joints_cam0 = (joints_camk - s*t_k) @ R_k
    """
    vggt     = np.load(ghost_scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    ma_scale = float(np.load(ghost_scene_dir / "mapanything_scale_centered.npy")[0])

    cam_names  = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    extrinsics = vggt["extrinsics"][0]   # (K, 3, 4)  [R|t] cam-from-world
    valid_mask = vggt["valid"][0]         # (K,) bool

    results: list[tuple[str, int, np.ndarray]] = []

    for k, cam_id in enumerate(cam_names):
        if not valid_mask[k]:
            continue
        body_dir = ghost_scene_dir / cam_id / "body_data"
        if not body_dir.exists():
            continue

        R_k = extrinsics[k, :, :3]  # (3, 3) — rotation is scale-free
        t_k = extrinsics[k, :, 3]   # (3,)   — up-to-scale

        for npz_path in sorted(body_dir.glob("person_*.npz")):
            data = np.load(str(npz_path), allow_pickle=False)
            matches = np.where(data["frame_indices"] == gt_frame_idx)[0]
            if not matches.size:
                continue
            t_idx = int(matches[0])

            params = {
                key: data[key][[t_idx]]
                for key in ("smplx_global_orient", "smplx_body_pose",
                            "smplx_betas", "smplx_transl")
            }
            joints_camk = _smplx_forward(params, smplx_model).astype(np.float64)

            # cam_k → cam0 (metric cam0 = VGGT cam0 * ma_scale):
            # p_world_metric = R_k^T @ (p_camk_metric - t_k_metric)
            #                = R_k^T @ (p_camk - s * t_k_vggt)
            # Written for row-vector batches: diff @ R_k  (equivalent)
            joints_cam0 = (joints_camk - ma_scale * t_k) @ R_k  # (22, 3)

            pid = int(npz_path.stem.split("_")[1])
            results.append((cam_id, pid, joints_cam0))

    return results


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(
    ghost_scene_dir: Path,
    gt_scene_dir: Path,
    smplx_model,
) -> dict | None:
    """Evaluate one scene.

    Returns dict with keys: w_mpjpe (mm), pa_mpjpe (mm), n_joints, scene.
    Returns None if scene should be skipped.
    """
    for fname in ("vggt_cameras_centered.npz", "mapanything_scale_centered.npy"):
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

    # --- Build SE(3) alignment from VGGT metric cameras to GT cameras --------
    vggt      = np.load(ghost_scene_dir / "vggt_cameras_centered.npz", allow_pickle=False)
    ma_scale  = float(np.load(ghost_scene_dir / "mapanything_scale_centered.npy")[0])
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vggt["camera_names"]]
    extrinsics = vggt["extrinsics"][0]   # (K, 3, 4)
    valid_mask = vggt["valid"][0]

    pred_centers, gt_centers = [], []
    for k, cam_id in enumerate(cam_names):
        if not valid_mask[k] or cam_id not in cam_pos_gt:
            continue
        R_k = extrinsics[k, :, :3]
        t_k = extrinsics[k, :, 3]
        # Camera centre in VGGT world = -R_k^T @ t_k, scaled to metric
        center_metric = (-R_k.T @ t_k) * ma_scale
        pred_centers.append(center_metric.astype(np.float64))
        gt_centers.append(cam_pos_gt[cam_id])

    if len(pred_centers) < 2:
        logger.warning(f"{ghost_scene_dir.name}: <2 valid cameras, skip")
        return None

    R_align, t_align = se3_align(
        np.stack(pred_centers),
        np.stack(gt_centers),
    )

    # --- Load predictions ---------------------------------------------------
    persons = load_predictions(ghost_scene_dir, frame_idx, smplx_model)
    if not persons:
        logger.warning(f"{ghost_scene_dir.name}: no predictions for frame {frame_idx}")
        return None

    # --- Build GT joint array for evaluation subset -------------------------
    joint_names   = sorted(gt_joints.keys())
    smplx_indices = [GT_TO_SMPLX[n] for n in joint_names]
    gt_arr        = np.stack([gt_joints[n] for n in joint_names])  # (J, 3)

    # --- Match predicted person to GT (pick best W-MPJPE) -------------------
    best_w   = float("inf")
    best_pa  = float("inf")

    for cam_id, pid, joints_cam0 in persons:
        pred_j    = joints_cam0[smplx_indices].astype(np.float64)    # (J, 3)
        # W-MPJPE†: apply SE(3) camera alignment then measure error
        pred_world = pred_j @ R_align.T + t_align                    # (J, 3)
        w_err  = float(np.linalg.norm(pred_world - gt_arr, axis=-1).mean()) * 1000

        # PA-MPJPE: Procrustes (Sim3) per-person alignment
        pred_proc = procrustes_align(pred_j, gt_arr)
        pa_err = float(np.linalg.norm(pred_proc - gt_arr, axis=-1).mean()) * 1000

        if w_err < best_w:
            best_w  = w_err
            best_pa = pa_err

    return {
        "scene":    ghost_scene_dir.name,
        "w_mpjpe":  best_w,
        "pa_mpjpe": best_pa,
        "n_joints": len(joint_names),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ghost pipeline on EgoExo4D val set")
    parser.add_argument("--ghost_root",  required=True,  help="ghost output root (egoexo4d/)")
    parser.add_argument("--gt_root",     required=True,  help="EgoExo4D GT root (contains per-scene dirs)")
    parser.add_argument("--smplx_model", default=str(_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl"),
                        help="Path to SMPLX_NEUTRAL.pkl (or folder containing it)")
    parser.add_argument("--max_scenes",  type=int, default=None)
    parser.add_argument("--scene",       default=None, help="Evaluate a single scene by name")
    args = parser.parse_args()

    ghost_root = Path(args.ghost_root)
    gt_root    = Path(args.gt_root)

    # Load SMPL-X model once
    import smplx
    smplx_path = Path(args.smplx_model)
    model_dir  = smplx_path.parent if smplx_path.is_file() else smplx_path
    smplx_model = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        batch_size=1,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
    ).eval()
    logger.info("SMPL-X model loaded")

    if args.scene:
        scene_dirs = [ghost_root / args.scene]
    else:
        scene_dirs = sorted(ghost_root.iterdir())
        if args.max_scenes:
            scene_dirs = scene_dirs[: args.max_scenes]

    results = []
    skipped_hand = 0
    skipped_missing = 0

    for ghost_scene_dir in scene_dirs:
        if not ghost_scene_dir.is_dir():
            continue
        gt_scene_dir = gt_root / ghost_scene_dir.name
        res = eval_scene(ghost_scene_dir, gt_scene_dir, smplx_model)
        if res is None:
            if not gt_scene_dir.exists():
                skipped_missing += 1
            elif not (ghost_scene_dir / "vggt_cameras_centered.npz").exists():
                skipped_missing += 1
            else:
                skipped_hand += 1
            continue
        results.append(res)
        logger.info(
            f"  {res['scene']:45s}  W={res['w_mpjpe']:6.1f}mm  PA={res['pa_mpjpe']:6.1f}mm"
            f"  ({res['n_joints']}j)"
        )

    if not results:
        logger.error("No scenes evaluated.")
        return

    w_vals  = [r["w_mpjpe"]  for r in results]
    pa_vals = [r["pa_mpjpe"] for r in results]

    print("\n" + "=" * 60)
    print(f"EgoExo4D evaluation — {len(results)} body-GT scenes")
    print(f"  Skipped (hand-only GT): {skipped_hand}")
    print(f"  Skipped (missing files): {skipped_missing}")
    print("-" * 60)
    print(f"  W-MPJPE†   mean: {np.mean(w_vals):6.1f} mm   median: {np.median(w_vals):6.1f} mm")
    print(f"  PA-MPJPE   mean: {np.mean(pa_vals):6.1f} mm   median: {np.median(pa_vals):6.1f} mm")
    print("=" * 60)

    # CHROMM reference (single-frame protocol, reported in metres → convert)
    print("\nCHROMM reference (Table 2):")
    print("  W-MPJPE†:  260 mm")
    print("  PA-MPJPE:   60 mm")
    print("\nHSfM reference:")
    print("  W-MPJPE:   500 mm")


if __name__ == "__main__":
    main()
