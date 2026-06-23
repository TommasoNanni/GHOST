"""Cam0 ray-based root placement — single camera, metric body size.

For each frame, casts rays from cam0 (world origin by construction)
through each observed Sapiens 2D keypoint using the full-resolution RICH
calibration K (correct principal point, NOT pp forced to centre).

Solves rigid (R, t) minimising the point-to-ray perpendicular distance:

    min_{R ∈ SO(3), t ∈ R³}  Σ_j w_j · huber( ‖(I − d_j dⱼᵀ)(R J[j]+t)‖ )

Scale is NOT solved — depth is pinned by the metric SMPL-X body size
(betas + FK).  Hips are downweighted (Goliath surface landmark ≠ SMPL-X
joint centre).

Reports side-by-side vs. multi-cam DLT + Procrustes with a depth (z) /
lateral (xy) breakdown to answer: does z-from-body-size beat
z-from-triangulation?

Usage:
    pixi run python approaches/test_cam0_ray_placement.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --checkpoint /users/tnanni/ghost/checkpoints/fusion_module/best.pt \\
        --body_split train_body
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS   = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models
from eval_placer_trans import (
    _load_fusion_model,
    _run_fusion_fwd,
    _6d_to_aa,
    load_pred_betas as _load_pred_betas,
)
from test_sapiens_dlt import (
    _GOLIATH_SMPLX_ALIGN,
    _build_gt_cameras,
    _load_gt,
    _load_sapiens,
    build_eval_arrays,
    score_from_arrays,
    run_sapiens_procrustes,
)

_HIP_SMPLX  = {1, 2}
_KNEE_SMPLX = {4, 5}


# ---------------------------------------------------------------------------
# Cam0 ray-based placement
# ---------------------------------------------------------------------------

def run_cam0_ray_placement(
    placer:            BodyPlacer,
    cam0_intrs:        np.ndarray,            # (T, 3, 3) GT K in VGGT crop space
    sapiens_data:      list[dict[int, dict]],
    pred_betas:        dict[int, np.ndarray],
    fused_pose_by_pid: dict[int, np.ndarray],
    frame_start:       int,
    conf_thr:          float = 0.3,
    joint_map:         dict[int, int] | None = None,
    huber_px:          float = 20.0,          # Huber threshold in pixels
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]], dict[int, dict[int, float]]]:
    """Cam0-only 2D reprojection loss body placement.

    Objective per frame:
        min_{R,t}  Σ_j w_j · huber_px( ‖π_{K0}(R J_can[j]+t) − kp2d_j‖ )

    π_{K0} is the pinhole projection with the GT K (correct principal point).
    Reprojection error explicitly constrains depth: a body at wrong depth
    projects joints too large/small in the image, unlike point-to-ray which
    has zero gradient along the ray.

    Init: try R=I and R=Ry(π) (facing cam), pick lower-loss result for first
    frame; warm-start from previous frame thereafter.
    """
    if joint_map is None:
        joint_map = _GOLIATH_SMPLX_ALIGN
    smplx_joints = sorted(joint_map)
    zero_orient  = np.zeros((1, 3), dtype=np.float32)
    delta        = huber_px
    cam0_data    = sapiens_data[0]
    rv_ry180     = SciR.from_euler('y', np.pi).as_rotvec()

    translations: dict[int, dict[int, np.ndarray]] = {}
    orientations: dict[int, dict[int, np.ndarray]] = {}
    proc_scales:  dict[int, dict[int, float]]      = {}

    for pid in sorted(cam0_data):
        betas     = pred_betas.get(pid, np.zeros(10, dtype=np.float32))
        fused_arr = fused_pose_by_pid.get(pid)
        if fused_arr is None:
            continue

        trans_out:  dict[int, np.ndarray] = {}
        orient_out: dict[int, np.ndarray] = {}
        scale_out:  dict[int, float]      = {}
        x_prev: np.ndarray | None = None

        for global_t in sorted(cam0_data[pid]["local_t"]):
            vggt_t  = global_t - frame_start
            t_local = int(global_t) - frame_start
            if vggt_t < 0 or vggt_t >= placer.T:
                continue
            if not placer.cam_valid[vggt_t, 0]:
                continue
            if not (0 <= t_local < len(fused_arr)):
                continue

            K0  = cam0_intrs[vggt_t].astype(np.float64)
            fx, fy = K0[0, 0], K0[1, 1]
            cx, cy = K0[0, 2], K0[1, 2]
            lt  = cam0_data[pid]["local_t"][global_t]
            kps = cam0_data[pid]["kps"][lt]   # (308, 3)

            u_obs:    list[float] = []
            v_obs:    list[float] = []
            smplx_js: list[int]   = []
            weights:  list[float] = []

            for smplx_j in smplx_joints:
                goliath_j = joint_map[smplx_j]
                x, y, conf = float(kps[goliath_j, 0]), float(kps[goliath_j, 1]), float(kps[goliath_j, 2])
                if conf < conf_thr:
                    continue
                oc  = placer.original_coords[vggt_t, 0]
                os_ = placer.original_size[vggt_t, 0]
                u, v = placer._orig_to_vggt(np.array([x, y]), oc, float(os_[0]), float(os_[1]))
                if not placer._in_bounds(u, v, oc[2], oc[3]):
                    continue
                w = float(conf)
                if smplx_j in _HIP_SMPLX:   w *= 0.15
                elif smplx_j in _KNEE_SMPLX: w *= 0.50
                u_obs.append(u); v_obs.append(v)
                smplx_js.append(smplx_j); weights.append(w)

            if len(smplx_js) < 3:
                continue

            u_arr = np.array(u_obs)
            v_arr = np.array(v_obs)
            w_arr = np.array(weights, dtype=np.float64)
            w_arr /= w_arr.sum()

            bp_aa = _6d_to_aa(fused_arr[t_local, :21]).reshape(63)
            J_can = placer._smplx_fk(betas[np.newaxis], bp_aa[np.newaxis], zero_orient)[0]
            J_sel = J_can[[j for j in smplx_js]].astype(np.float64)
            J0    = J_can[0].astype(np.float64)

            def loss_fn(x: np.ndarray) -> float:
                R   = SciR.from_rotvec(x[:3]).as_matrix()
                P   = (R @ J_sel.T).T + x[3:]            # (N, 3) world
                Z   = np.maximum(P[:, 2], 1e-2)           # guard negative depth
                u_p = fx * P[:, 0] / Z + cx
                v_p = fy * P[:, 1] / Z + cy
                r   = np.sqrt((u_p - u_arr)**2 + (v_p - v_arr)**2 + 1e-12)
                h   = np.where(r < delta, 0.5 * r**2, delta * r - 0.5 * delta**2)
                return float(w_arr @ h)

            # Try both facing directions; pick lower loss.
            # With reprojection loss, a body at wrong depth projects too small/large
            # → loss correctly penalises depth error and front/back flip.
            if x_prev is not None:
                res    = sp_minimize(loss_fn, x_prev, method='L-BFGS-B',
                                     options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7})
                best_x = res.x
            else:
                candidates = [
                    np.array([0., 0., 0., 0., 0., 3.0]),
                    np.concatenate([rv_ry180, [0., 0., 3.0]]),
                ]
                best_x, best_val = None, float('inf')
                for x0c in candidates:
                    rc = sp_minimize(loss_fn, x0c, method='L-BFGS-B',
                                     options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7})
                    if rc.fun < best_val:
                        best_val, best_x = rc.fun, rc.x
            x_prev = best_x.copy()

            R_opt = SciR.from_rotvec(best_x[:3]).as_matrix().astype(np.float32)
            pelvis_world = (R_opt.astype(np.float64) @ J0 + best_x[3:]).astype(np.float32)
            trans_out[global_t]  = pelvis_world
            orient_out[global_t] = R_opt
            scale_out[global_t]  = 1.0

        if trans_out:
            translations[pid] = trans_out
            orientations[pid] = orient_out
            proc_scales[pid]  = scale_out

    return translations, orientations, proc_scales


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depth_lateral_split(pred_roots: np.ndarray, gt_roots: np.ndarray):
    """Mean depth (z) and lateral (xy) root errors over valid frames.

    cam0 = world origin → cam0 z-axis = world z-axis = optical / depth axis.
    """
    valid = np.isfinite(pred_roots).all(-1) & np.isfinite(gt_roots).all(-1)
    if not valid.any():
        return float('nan'), float('nan')
    d = (pred_roots - gt_roots)[valid]           # (N, 3)
    depth   = float(np.abs(d[:, 2]).mean())
    lateral = float(np.linalg.norm(d[:, :2], axis=-1).mean())
    return depth, lateral


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--checkpoint",  required=True, type=Path)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--conf_thr",    type=float, default=0.3)
    ap.add_argument("--min_cams",    type=int,   default=2)
    ap.add_argument("--min_joints",  type=int,   default=3)
    ap.add_argument("--body_split",  default="train_body")
    args = ap.parse_args()

    import torch
    device     = torch.device(args.device)
    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    rich_root  = args.rich_root.resolve()

    print(f"Scene: {scene_name}")

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer   = BodyPlacer(str(scene_dir), _smplx_arg)
    cam_dirs = placer._cam_dirs
    print(f"Cameras: {[d.name for d in cam_dirs]}  T={placer.T}")

    pred_betas = _load_pred_betas(list(cam_dirs))
    scale = placer.load_mapanything_scale()
    if scale is None:
        scale = placer.estimate_scale_triangulated(set(pred_betas), pred_betas)
    print(f"Scale: median={float(np.median(scale)):.4f} m/VGGT-unit")

    print("Loading fusion model …")
    fusion_model = _load_fusion_model(args.checkpoint, device)
    fwd = _run_fusion_fwd(list(cam_dirs), fusion_model, device)
    if fwd is None:
        print("ERROR: fusion forward pass failed"); return
    fused_pose_arr, _, fwd_pids, frame_start = fwd
    fused_pose_by_pid = {pid: fused_pose_arr[:, slot] for slot, pid in enumerate(fwd_pids)}
    print(f"Fused pose: {len(fused_pose_by_pid)} pids  frame_start={frame_start}")

    sapiens_data = _load_sapiens(list(cam_dirs))
    gt_body_data, gt_betas = _load_gt(scene_name, rich_root, args.body_split)
    print(f"GT: {len(gt_body_data)} persons")

    # GT cam0 intrinsics — correct principal point from RICH XML calibration
    _, gt_cam_intrs = _build_gt_cameras(placer, scene_name, rich_root)
    if gt_cam_intrs is None:
        print("ERROR: GT camera calibration not found"); return
    cam0_intrs = gt_cam_intrs[:, 0]   # (T, 3, 3)
    print("GT cam0 K loaded (correct PP)\n")

    common_eval = dict(
        gt_body_data      = gt_body_data,
        gt_betas          = gt_betas,
        fused_pose_by_pid = fused_pose_by_pid,
        frame_start       = frame_start,
        pred_betas        = pred_betas,
        placer            = placer,
    )

    # ── Method A: multi-cam DLT + SE(3) Procrustes ───────────────────────────
    print("=" * 62)
    print("  Method A: multi-cam DLT + SE(3) Procrustes")
    pt_A, po_A, ps_A = run_sapiens_procrustes(
        placer=placer, scale=scale, sapiens_data=sapiens_data,
        pred_betas=pred_betas, fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start, conf_thr=args.conf_thr,
        min_cams=args.min_cams, min_joints=args.min_joints,
    )
    arr_A = build_eval_arrays(pred_trans=pt_A, pred_orient=po_A, proc_scales=ps_A, **common_eval)
    if arr_A is None:
        print("  No valid predictions for method A"); return
    pjr_A, pr_A, gtj_A, gtr_A, _, _, _, oerrs_A = arr_A
    mA = score_from_arrays(pjr_A, pr_A, gtj_A, gtr_A, oerrs_A, label="DLT+Procrustes")
    d_A, lat_A = _depth_lateral_split(pr_A, gtr_A)
    print(f"    depth(z)={d_A*100:.1f}cm  lateral(xy)={lat_A*100:.1f}cm")

    # ── Method B: cam0 ray, single camera, metric body size ──────────────────
    print("\n" + "=" * 62)
    print("  Method B: cam0 ray (1 cam, metric body size, GT K)")
    pt_B, po_B, ps_B = run_cam0_ray_placement(
        placer=placer, cam0_intrs=cam0_intrs,
        sapiens_data=sapiens_data, pred_betas=pred_betas,
        fused_pose_by_pid=fused_pose_by_pid, frame_start=frame_start,
        conf_thr=args.conf_thr,
    )
    arr_B = build_eval_arrays(pred_trans=pt_B, pred_orient=po_B, proc_scales=ps_B, **common_eval)
    if arr_B is None:
        print("  No valid predictions for method B"); return
    pjr_B, pr_B, gtj_B, gtr_B, _, _, _, oerrs_B = arr_B
    mB = score_from_arrays(pjr_B, pr_B, gtj_B, gtr_B, oerrs_B, label="cam0 ray")
    d_B, lat_B = _depth_lateral_split(pr_B, gtr_B)
    print(f"    depth(z)={d_B*100:.1f}cm  lateral(xy)={lat_B*100:.1f}cm")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    hdr = f"  {'Method':<28}  {'root':>7}  {'depth-z':>8}  {'lateral':>8}  {'WA':>7}  {'W':>7}  {'RTE':>7}"
    sep = f"  {'─'*28}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}"
    print(hdr); print(sep)

    def row(name, m, d, lat):
        print(f"  {name:<28}  {m['root_mean']*100:>7.1f}  {d*100:>8.1f}  {lat*100:>8.1f}  "
              f"{m['WA']:>7.1f}  {m['W']:>7.1f}  {m['RTE']*100:>7.2f}")

    row("DLT+Procrustes (multi-cam)", mA, d_A, lat_A)
    row("Cam0 ray (1 cam, GT K)",     mB, d_B, lat_B)

    winner = "cam0-ray" if d_B < d_A else "DLT"
    print(f"\n  Depth: DLT={d_A*100:.1f}cm  cam0-ray={d_B*100:.1f}cm  → {winner} wins")


if __name__ == "__main__":
    main()
