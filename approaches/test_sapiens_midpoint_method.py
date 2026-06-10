"""Generalised midpoint triangulation for body placement using Sapiens keypoints.

Replaces DLT (algebraic error) with the generalised midpoint method, which
minimises the sum of squared perpendicular distances from a 3D point X to each
camera ray:

    min_X  sum_k || (I - d_k d_k^T)(X - C_k) ||^2

Closed-form solution:  [sum_k (I - d_k d_k^T)] X = sum_k (I - d_k d_k^T) C_k

where C_k = -R_k^T t_k  (optical centre in world)  and
      d_k = normalised( R_k^T K_k^{-1} [u, v, 1]^T )  (ray direction in world).

Everything else (Sapiens loading, scale, FK, Procrustes, GT, evaluation) is
identical to test_sapiens_dlt.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer, _6d_to_aa_batch
from utilities.rich_gender_plugin import resolve_smplx_models

_EVAL_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_SCRIPTS))
from eval_placer_trans import (
    _load_fusion_model,
    _run_fusion_fwd,
    _6d_to_aa,
    load_pred_betas as _load_pred_betas,
)

sys.path.insert(0, str(_REPO_ROOT / "evaluation"))
from evaluate_on_rich_test import (
    _sim3_align,
    metric_wa_mpjpe,
    metric_w_mpjpe,
    metric_ga_mpjpe,
    metric_pa_mpjpe,
    metric_rte,
    _BODY_JOINT_IDX,
)

# Import shared helpers from test_sapiens_dlt to avoid duplication
from test_sapiens_dlt import (
    _GOLIATH_SMPLX_ALIGN,
    _GOLIATH_SMPLX_DISTAL,
    _build_gt_cameras,
    _load_gt,
    _load_sapiens,
    build_eval_arrays,
    score_from_arrays,
    smooth_translations_sg,
    _geodesic_deg,
)

_BODY_J = len(_BODY_JOINT_IDX)


# ---------------------------------------------------------------------------
# Generalised midpoint triangulation
# ---------------------------------------------------------------------------

def _triangulate_midpoint(
    observations:  list[tuple[float, float]],
    intrinsics:    list[np.ndarray],   # each (3, 3)
    extrinsics:    list[np.ndarray],   # each (3, 4) camera-from-world [R|t]
    weights:       list[float] | None = None,
) -> np.ndarray:
    """Generalised midpoint triangulation from N >= 2 views.

    Solves:  [sum_k w_k (I - d_k d_k^T)] X = sum_k w_k (I - d_k d_k^T) C_k

    Returns:
        (3,) world-space point.
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3,      dtype=np.float64)

    for i, ((u, v), K, E) in enumerate(zip(observations, intrinsics, extrinsics)):
        w  = float(weights[i]) if weights is not None else 1.0
        R  = E[:3, :3]
        t  = E[:3, 3]

        # Ray in camera frame (back-project through K)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

        # Ray direction in world frame
        d = R.T @ ray_cam
        d /= np.linalg.norm(d)

        # Optical centre in world frame
        C = -(R.T @ t)

        I_ddt = np.eye(3) - np.outer(d, d)
        A += w * I_ddt
        b += w * (I_ddt @ C)

    X, *_ = np.linalg.lstsq(A, b, rcond=None)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Procrustes + midpoint
# ---------------------------------------------------------------------------

def run_sapiens_procrustes_midpoint(
    placer:            BodyPlacer,
    scale:             float | np.ndarray,
    sapiens_data:      list[dict[int, dict]],
    pred_betas:        dict[int, np.ndarray],
    fused_pose_by_pid: dict[int, np.ndarray],
    frame_start:       int,
    conf_thr:          float = 0.3,
    min_cams:          int   = 2,
    min_joints:        int   = 3,
    joint_map:         dict[int, int] | None = None,
    sin_weight:        bool  = False,
) -> tuple[dict, dict, dict]:
    """Midpoint-triangulate Sapiens joints, then SE(3) Procrustes.

    Interface identical to run_sapiens_procrustes in test_sapiens_dlt.py.
    """
    if joint_map is None:
        joint_map = _GOLIATH_SMPLX_ALIGN
    smplx_joints = sorted(joint_map)
    zero_orient  = np.zeros((1, 3), dtype=np.float32)

    all_pids: set[int] = set()
    for cm in sapiens_data:
        all_pids.update(cm.keys())

    translations: dict[int, dict[int, np.ndarray]] = {}
    orientations: dict[int, dict[int, np.ndarray]] = {}
    proc_scales:  dict[int, dict[int, float]]      = {}

    for pid in sorted(all_pids):
        betas = pred_betas.get(pid, np.zeros(10, dtype=np.float32))

        all_frames: set[int] = set()
        for cm in sapiens_data:
            if pid in cm:
                all_frames.update(cm[pid]["local_t"].keys())

        trans_out:  dict[int, np.ndarray] = {}
        orient_out: dict[int, np.ndarray] = {}
        scale_out:  dict[int, float]      = {}

        for global_t in sorted(all_frames):
            vggt_t = global_t - frame_start
            if vggt_t < 0 or vggt_t >= placer.T:
                continue

            s = float(scale[vggt_t]) if isinstance(scale, np.ndarray) else float(scale)

            # ── Step 1: midpoint-triangulate each joint ───────────────────
            joint_world: dict[int, np.ndarray] = {}
            for smplx_j in smplx_joints:
                goliath_j = joint_map[smplx_j]
                obs:   list[tuple[float, float]] = []
                Ks:    list[np.ndarray]          = []
                Es:    list[np.ndarray]          = []
                ws:    list[float]               = []

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

                    oc     = placer.original_coords[vggt_t, k]
                    os_    = placer.original_size[vggt_t, k]
                    u, v   = placer._orig_to_vggt(np.array([x, y]), oc, float(os_[0]), float(os_[1]))
                    if not placer._in_bounds(u, v, oc[2], oc[3]):
                        continue

                    intr = placer.intrinsics[vggt_t, k].astype(np.float64)
                    ext  = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
                    ext[:3, 3] *= s   # scale translation to metric

                    if sin_weight and k > 0:
                        cos_a = float(np.clip(ext[2, 2], -1.0, 1.0))
                        ang_w = float(np.sqrt(max(1.0 - cos_a ** 2, 0.0))) ** 10
                    else:
                        ang_w = 1.0

                    obs.append((u, v))
                    Ks.append(intr)
                    Es.append(ext)
                    ws.append(conf * ang_w)

                if len(obs) >= min_cams:
                    joint_world[smplx_j] = _triangulate_midpoint(obs, Ks, Es, ws)

            if len(joint_world) < min_joints:
                continue

            # ── Step 2: FK with fused body_pose ──────────────────────────
            if pid not in fused_pose_by_pid:
                continue
            fused_arr = fused_pose_by_pid[pid]
            t_local   = global_t - frame_start
            if not (0 <= t_local < len(fused_arr)):
                continue
            body_pose_frame = _6d_to_aa(fused_arr[t_local, :21]).reshape(63)
            J_can = placer._smplx_fk(
                betas[np.newaxis], body_pose_frame[np.newaxis], zero_orient
            )[0]  # (55, 3)

            # ── Step 3: SE(3) Procrustes ──────────────────────────────────
            vis = sorted(joint_world)
            A   = np.stack([joint_world[j] for j in vis]).astype(np.float64)
            B   = np.stack([J_can[j]        for j in vis]).astype(np.float64)

            A_m, B_m = A.mean(0), B.mean(0)
            H = (B - B_m).T @ (A - A_m)
            U, _, Vt = np.linalg.svd(H)
            d_sign = np.linalg.det(Vt.T @ U.T)
            R = (Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T).astype(np.float32)
            t_vec = (A_m - R.astype(np.float64) @ B_m).astype(np.float32)

            pelvis_world = (R.astype(np.float64) @ J_can[0].astype(np.float64) + t_vec).astype(np.float32)
            trans_out[global_t]  = pelvis_world
            orient_out[global_t] = R
            scale_out[global_t]  = 1.0

        if trans_out:
            translations[pid] = trans_out
            orientations[pid] = orient_out
            proc_scales[pid]  = scale_out

    return translations, orientations, proc_scales


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
    ap.add_argument("--gt_cams",     action="store_true")
    ap.add_argument("--hybrid_pp",   action="store_true")
    ap.add_argument("--distal_only", action="store_true")
    ap.add_argument("--sin_weight",  action="store_true")
    ap.add_argument("--gt_pid",      type=int, default=0)
    ap.add_argument("--exclude_cams", nargs="+", default=[])
    args = ap.parse_args()

    import torch
    device     = torch.device(args.device)
    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    rich_root  = args.rich_root.resolve()

    print(f"Scene: {scene_name}  [midpoint triangulation]")

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(
        scene_output_dir = str(scene_dir),
        smplx_model_path = _smplx_arg,
    )
    cam_dirs = placer._cam_dirs
    print(f"Cameras: {[d.name for d in cam_dirs]}  T={placer.T}")

    pred_betas = _load_pred_betas(list(cam_dirs))
    scale = placer.load_mapanything_scale()
    if scale is not None:
        print(f"Scale: MapAnything  median={float(np.median(scale)):.4f}")
    else:
        scale = placer.estimate_scale_triangulated(
            all_pids          = set(pred_betas),
            pred_betas_by_pid = pred_betas,
        )
        print(f"Scale: triangulated  median={float(np.median(scale)):.4f}")

    print("Loading fusion model …")
    fusion_model = _load_fusion_model(args.checkpoint, device)
    fwd = _run_fusion_fwd(list(cam_dirs), fusion_model, device)
    if fwd is None:
        print("ERROR: fusion forward pass failed"); return
    fused_pose_arr, _, fwd_pids, frame_start = fwd
    fused_pose_by_pid = {pid: fused_pose_arr[:, i] for i, pid in enumerate(fwd_pids)}
    print(f"Fused pose: {len(fused_pose_by_pid)} pids, frame_start={frame_start}")

    sapiens_data = _load_sapiens(list(cam_dirs))
    print(f"Sapiens kps: {sum(len(cm) for cm in sapiens_data)} (cam, pid) pairs")

    # ── Camera overrides ──────────────────────────────────────────────────────
    orig_extrinsics = placer.extrinsics.copy()
    orig_intrinsics = placer.intrinsics.copy()
    orig_cam_valid  = placer.cam_valid.copy()

    if args.gt_cams:
        gt_exts, gt_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_exts is None:
            print("ERROR: GT cameras not found"); return
        placer.extrinsics = gt_exts
        placer.intrinsics = gt_intrs
        filled = np.any(gt_exts != 0, axis=(0, 2, 3))
        placer.cam_valid = orig_cam_valid & filled[np.newaxis, :]
        scale = np.ones(placer.T, dtype=np.float32)
        print("GT cameras loaded")
    elif args.hybrid_pp:
        _, gt_intrs = _build_gt_cameras(placer, scene_name, rich_root)
        if gt_intrs is None:
            print("ERROR: GT cameras not found"); return
        hybrid = orig_intrinsics.copy()
        hybrid[:, :, 0, 2] = gt_intrs[:, :, 0, 2]
        hybrid[:, :, 1, 2] = gt_intrs[:, :, 1, 2]
        placer.intrinsics = hybrid
        dcx = float(np.abs(hybrid[:,:,0,2] - orig_intrinsics[:,:,0,2]).mean())
        print(f"Hybrid PP: |Δcx|={dcx:.1f}px")

    _vnames = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    for cn in args.exclude_cams:
        for ki, name in enumerate(_vnames):
            if name == cn:
                placer.cam_valid[:, ki] = False
                print(f"Excluded: {cn}")

    gt_body_data, gt_betas = _load_gt(scene_name, rich_root, args.body_split)
    print(f"GT: {len(gt_body_data)} persons\n")

    joint_map = _GOLIATH_SMPLX_DISTAL if args.distal_only else _GOLIATH_SMPLX_ALIGN

    pred_trans, pred_orient, proc_scales = run_sapiens_procrustes_midpoint(
        placer             = placer,
        scale              = scale,
        sapiens_data       = sapiens_data,
        pred_betas         = pred_betas,
        fused_pose_by_pid  = fused_pose_by_pid,
        frame_start        = frame_start,
        conf_thr           = args.conf_thr,
        min_cams           = args.min_cams,
        min_joints         = args.min_joints,
        joint_map          = joint_map,
        sin_weight         = args.sin_weight,
    )
    print(f"  {len(pred_trans)} pids, {sum(len(v) for v in pred_trans.values())} frames")

    pid_map = {1: args.gt_pid} if args.gt_pid else None
    arrays = build_eval_arrays(
        pred_trans        = pred_trans,
        pred_orient       = pred_orient,
        gt_body_data      = gt_body_data,
        gt_betas          = gt_betas,
        fused_pose_by_pid = fused_pose_by_pid,
        frame_start       = frame_start,
        pred_betas        = pred_betas,
        placer            = placer,
        proc_scales       = proc_scales,
        pid_map           = pid_map,
    )
    if arrays is None:
        print("ERROR: no predictions"); return
    pred_joints_rel, pred_roots, gt_joints, gt_roots, frame_start_t, T, P, all_oerrs = arrays

    m0 = score_from_arrays(pred_joints_rel, pred_roots, gt_joints, gt_roots,
                           all_oerrs, label="midpoint / no smoothing")

    valid_r = np.isfinite(pred_roots).all(-1) & np.isfinite(gt_roots).all(-1)
    if valid_r.any():
        d = (pred_roots - gt_roots)[valid_r]
        print(f"    x={np.abs(d[:,0]).mean()*100:.1f}cm  y={np.abs(d[:,1]).mean()*100:.1f}cm"
              f"  depth(z)={np.abs(d[:,2]).mean()*100:.1f}cm"
              f"  lateral(xy)={np.linalg.norm(d[:,:2],axis=-1).mean()*100:.1f}cm")

    pid_slots = {pid: slot for slot, pid in enumerate(sorted(pred_trans))}
    print(f"\n  {'Window':>8}  {'WA(mm)':>8}  {'W(mm)':>8}  "
          f"{'RTE(cm)':>8}  {'root(cm)':>9}  {'ΔRTE':>7}  {'ΔW':>7}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}")
    for win in [5, 7, 9, 11, 13, 15]:
        pt_s    = smooth_translations_sg(pred_trans, window=win, polyorder=2)
        pr_s    = np.full_like(pred_roots, np.nan)
        for ghost_pid, slot in pid_slots.items():
            for gf, root in pt_s.get(ghost_pid, {}).items():
                t_rel = int(gf) - frame_start_t
                if 0 <= t_rel < T:
                    pr_s[t_rel, slot] = root
        m = score_from_arrays(pred_joints_rel, pr_s, gt_joints, gt_roots,
                              all_oerrs, label=f"win={win}", verbose=False)
        print(f"  {win:>8}  {m['WA']:>8.1f}  {m['W']:>8.1f}  "
              f"{m['RTE']*100:>8.2f}  {m['root_mean']*100:>9.1f}  "
              f"{(m['RTE']-m0['RTE'])*100:>+7.2f}  {m['W']-m0['W']:>+7.1f}")


if __name__ == "__main__":
    main()
