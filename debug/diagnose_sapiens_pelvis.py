"""Diagnose the definitional offset between Sapiens/Goliath pelvis and SMPL-X joint-0.

Strategy: use GT cameras (perfect geometry) to triangulate Sapiens l_hip + r_hip,
take midpoint as "Sapiens pelvis", compare to GT SMPL-X joint-0.
Also runs full Procrustes with GT cameras to see residual error.

Usage:
    pixi run python debug/diagnose_sapiens_pelvis.py \
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \
        --checkpoint /capstor/scratch/cscs/tnanni/ghost_checkpoints/fusion_module_latest/best.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion.placer import BodyPlacer, _6d_to_aa_batch
from utilities.rich_gender_plugin import resolve_smplx_models
from test_sapiens_dlt import (
    _build_gt_cameras, _load_gt, _load_sapiens,
    _GOLIATH_SMPLX_ALIGN, run_sapiens_procrustes,
)
from eval_placer_trans import (
    _load_fusion_model, _run_fusion_fwd, _6d_to_aa,
    load_pred_betas as _load_pred_betas,
)

# Goliath indices for hips (used to approximate Sapiens pelvis)
_G_L_HIP = 9
_G_R_HIP = 10

def _triangulate_joint(
    goliath_j: int,
    sapiens_data: list[dict],
    pid: int,
    global_t: int,
    vggt_t: int,
    placer: BodyPlacer,
    scale: float,
    conf_thr: float = 0.3,
) -> np.ndarray | None:
    obs, pmats, weights = [], [], []
    for k, cm in enumerate(sapiens_data):
        if pid not in cm or global_t not in cm[pid]["local_t"]:
            continue
        if not placer.cam_valid[vggt_t, k]:
            continue
        lt  = cm[pid]["local_t"][global_t]
        kps = cm[pid]["kps"][lt]
        x, y, conf = float(kps[goliath_j, 0]), float(kps[goliath_j, 1]), float(kps[goliath_j, 2])
        if conf < conf_thr:
            continue
        oc = placer.original_coords[vggt_t, k]
        os_ = placer.original_size[vggt_t, k]
        u, v = placer._orig_to_vggt(np.array([x, y]), oc, float(os_[0]), float(os_[1]))
        if not placer._in_bounds(u, v, oc[2], oc[3]):
            continue
        intr = placer.intrinsics[vggt_t, k].astype(np.float64)
        ext  = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
        ext[:3, 3] *= scale
        pmats.append(intr @ ext); obs.append((u, v)); weights.append(conf)
    if len(obs) < 2:
        return None
    return placer._triangulate_dlt(obs, pmats, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--checkpoint",  required=True, type=Path)
    ap.add_argument("--body_split",  default="train_body")
    ap.add_argument("--conf_thr",    type=float, default=0.3)
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    import torch
    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(str(scene_dir), _smplx_arg)
    cam_dirs = placer._cam_dirs
    print(f"Scene: {scene_name}  T={placer.T}  cams={[d.name for d in cam_dirs]}")

    # Use VGGT cameras (same as M1 — known to work correctly)
    scale = placer.load_mapanything_scale()
    if scale is None:
        from eval_placer_trans import load_pred_betas as _lpb
        scale = placer.estimate_scale_triangulated(set(pred_betas), pred_betas)
    print(f"Scale: median={float(np.median(scale)):.4f} m/VGGT-unit")

    # Data
    sapiens_data = _load_sapiens(list(cam_dirs))
    gt_body_data, gt_betas_map = _load_gt(scene_name, args.rich_root, args.body_split)
    pred_betas   = _load_pred_betas(list(cam_dirs))

    # Fusion model for Procrustes body_pose
    fusion_model = _load_fusion_model(args.checkpoint, torch.device(args.device))
    fwd = _run_fusion_fwd(list(cam_dirs), fusion_model, torch.device(args.device))
    fused_pose_arr, _, fwd_pids, frame_start = fwd
    fused_pose_by_pid = {pid: fused_pose_arr[:, i] for i, pid in enumerate(fwd_pids)}

    zero_orient = np.zeros((1, 3), dtype=np.float32)

    # ── Diagnostic ───────────────────────────────────────────────────────────
    pid = 1  # single-person scene
    gt_pid = 1
    betas = pred_betas.get(pid, np.zeros(10, dtype=np.float32))
    gt_betas = gt_betas_map.get(gt_pid, np.zeros(10, dtype=np.float32))

    offsets_midpoint = []   # Sapiens midpoint(l_hip,r_hip) − GT joint-0
    offsets_procrustes = [] # Procrustes-estimated joint-0 − GT joint-0
    cam0_axis = np.array([0.0, 0.0, 1.0])  # depth axis in cam-0 frame (=world for BBQ)
    debug_printed = 0

    for global_t, params in sorted(gt_body_data.get(gt_pid, {}).items()):
        vggt_t = global_t - frame_start
        if vggt_t < 0 or vggt_t >= placer.T:
            continue

        # GT pelvis in world frame
        J_gt = placer._smplx_fk(
            params["betas"][np.newaxis],
            params["body_pose"][np.newaxis],
            params["global_orient"][np.newaxis],
        )[0] + params["transl"]
        gt_pelvis = J_gt[0].astype(np.float64)

        # --- Sapiens midpoint pelvis ---
        s = float(scale[vggt_t])
        lh = _triangulate_joint(_G_L_HIP, sapiens_data, pid, global_t, vggt_t, placer, s, args.conf_thr)
        rh = _triangulate_joint(_G_R_HIP, sapiens_data, pid, global_t, vggt_t, placer, s, args.conf_thr)
        if lh is not None and rh is not None:
            mid = (lh + rh) / 2.0
            offsets_midpoint.append(mid - gt_pelvis)
            if debug_printed < 3:
                print(f"  [frame {global_t}] GT pelvis (m): {gt_pelvis}")
                print(f"  [frame {global_t}] Sap midpoint (m): {mid}")
                print(f"  [frame {global_t}] GT transl: {params['transl']}  L_hip_world: {lh}")
                debug_printed += 1

        # --- Procrustes with GT cameras ---
        if pid not in fused_pose_by_pid:
            continue
        t_local = global_t - frame_start
        if not (0 <= t_local < len(fused_pose_by_pid[pid])):
            continue
        body_pose_aa = _6d_to_aa(fused_pose_by_pid[pid][t_local, :21]).reshape(63)
        J_can = placer._smplx_fk(betas[np.newaxis], body_pose_aa[np.newaxis], zero_orient)[0]

        joint_world = {}
        for smplx_j, goliath_j in _GOLIATH_SMPLX_ALIGN.items():
            pt = _triangulate_joint(goliath_j, sapiens_data, pid, global_t, vggt_t, placer, s, args.conf_thr)
            if pt is not None:
                joint_world[smplx_j] = pt

        if len(joint_world) < 3:
            continue

        vis   = sorted(joint_world)
        A     = np.stack([joint_world[j] for j in vis]).astype(np.float64)
        B     = np.stack([J_can[j]        for j in vis]).astype(np.float64)
        Am, Bm = A.mean(0), B.mean(0)
        H = (B - Bm).T @ (A - Am)
        U, _, Vt = np.linalg.svd(H)
        R = (Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T)
        t = Am - R @ Bm
        pelvis_proc = R @ J_can[0].astype(np.float64) + t
        offsets_procrustes.append(pelvis_proc - gt_pelvis)

    # ── Report ────────────────────────────────────────────────────────────────
    def _report(name, offs):
        if not offs:
            print(f"\n{name}: no data"); return
        O = np.stack(offs)                        # (N, 3)
        norm = np.linalg.norm(O, axis=1)
        depth = O @ cam0_axis                     # component along cam-0 Z
        print(f"\n{name}  (N={len(O)})")
        print(f"  3D offset norm : mean={norm.mean()*100:.1f}cm  median={np.median(norm)*100:.1f}cm  std={norm.std()*100:.1f}cm")
        print(f"  Depth (cam0-Z) : mean={depth.mean()*100:.1f}cm  median={np.median(depth)*100:.1f}cm  std={depth.std()*100:.1f}cm")
        print(f"  XYZ mean (cm)  : {O.mean(0)*100}")
        print(f"  XYZ std  (cm)  : {O.std(0)*100}")

    _report("Sapiens midpoint(l_hip,r_hip) − GT joint-0", offsets_midpoint)
    _report("Procrustes(GT cams) joint-0  − GT joint-0", offsets_procrustes)


if __name__ == "__main__":
    main()
