"""visualize/visualize_trajectories.py

3-panel diagnostic visualisation for a single scene:

  Panel A  — 3D scene view
               · Camera optical centres over time (one colour per camera,
                 scatter so temporal drift is visible)
               · Camera optical-axis arrows (sampled every --arrow_stride frames)
               · GT pelvis trajectory  (green)
               · Predicted pelvis trajectory  (red, coloured by time)
               · Pelvis orientation arrows for both GT and pred

  Panel B  — Camera optical-centre stability (x/y/z vs frame index per camera)
               Flat lines = stable VGGT estimate. Wiggly = per-frame noise.

  Panel C  — Camera optical-axis (z) stability (same layout as B)

  Panel D  — Root translation error per frame (total + x/y/z breakdown)

  Panel E  — Orientation error per frame (geodesic degrees)

Usage:
    pixi run python visualize/visualize_trajectories.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --checkpoint checkpoints/fusion_module_latest/best.pt \\
        --out figures/BBQ_001_guitar_trajectories.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))

from fusion.placer import BodyPlacer
from eval_placer_trans import (
    _load_fusion_model,
    _run_fusion_fwd,
    load_pred_betas as _load_pred_betas,
)
from test_sapiens_dlt import (
    _GOLIATH_SMPLX_ALIGN,
    _load_gt,
    _load_sapiens,
    run_sapiens_procrustes,
    smooth_extrinsics_sg,
)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    """(T, K, 3, 4) extrinsics → (T, K, 3) optical centres in world frame.

    C_k = -R_k^T t_k
    """
    R = extrinsics[:, :, :3, :3]
    t = extrinsics[:, :, :3, 3]
    return -np.einsum('tkji,tkj->tki', R, t)


def _camera_zaxis(extrinsics: np.ndarray) -> np.ndarray:
    """(T, K, 3, 4) → (T, K, 3) optical-axis direction in world frame.

    Camera z in world = last row of R (row index 2).
    """
    return extrinsics[:, :, 2, :3].copy()


def _geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",      required=True, type=Path)
    ap.add_argument("--rich_root",      required=True, type=Path)
    ap.add_argument("--smplx_model",    required=True, type=Path)
    ap.add_argument("--checkpoint",     required=True, type=Path)
    ap.add_argument("--out",            default="trajectories.png", type=Path)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--body_split",     default="train_body")
    ap.add_argument("--conf_thr",       type=float, default=0.3)
    ap.add_argument("--smooth_cameras", type=int,   default=0,
                    help="SG window for extrinsic smoothing (0=off, must be odd)")
    ap.add_argument("--arrow_stride",   type=int,   default=20,
                    help="Draw orientation arrow every N frames")
    ap.add_argument("--arrow_len",      type=float, default=0.3,
                    help="Arrow length in metres")
    ap.add_argument("--gt_pid",         type=int,   default=0)
    ap.add_argument("--sin_weight",     action="store_true")
    args = ap.parse_args()

    import torch
    device     = torch.device(args.device)
    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    rich_root  = args.rich_root.resolve()
    print(f"Scene: {scene_name}")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    placer   = BodyPlacer(str(scene_dir), str(args.smplx_model))
    cam_dirs = placer._cam_dirs
    vnames   = [n.decode() if isinstance(n, bytes) else n for n in placer.camera_names]
    K        = len(vnames)
    print(f"Cameras: {vnames}  T={placer.T}")

    if args.smooth_cameras > 0:
        w = args.smooth_cameras if args.smooth_cameras % 2 == 1 else args.smooth_cameras + 1
        placer.extrinsics = smooth_extrinsics_sg(placer.extrinsics, window=w)
        print(f"Extrinsics smoothed (window={w})")

    pred_betas = _load_pred_betas(list(cam_dirs))
    scale = placer.load_mapanything_scale()
    if scale is None:
        scale = placer.estimate_scale_triangulated(
            all_pids=set(pred_betas), pred_betas_by_pid=pred_betas,
        )

    print("Loading fusion model …")
    fusion_model = _load_fusion_model(args.checkpoint, device)
    fwd = _run_fusion_fwd(list(cam_dirs), fusion_model, device)
    if fwd is None:
        print("ERROR: fusion failed"); return
    fused_pose_arr, _, fwd_pids, frame_start = fwd
    fused_pose_by_pid = {pid: fused_pose_arr[:, i] for i, pid in enumerate(fwd_pids)}

    sapiens_data = _load_sapiens(list(cam_dirs))

    pred_trans, pred_orient, _ = run_sapiens_procrustes(
        placer=placer, scale=scale, sapiens_data=sapiens_data,
        pred_betas=pred_betas, fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start, conf_thr=args.conf_thr,
        min_cams=2, min_joints=3, joint_map=_GOLIATH_SMPLX_ALIGN,
        sin_weight=args.sin_weight,
    )
    gt_body_data, _ = _load_gt(scene_name, rich_root, args.body_split)
    gt_pids = sorted(gt_body_data)

    # ── Camera geometry (metric extrinsics) ───────────────────────────────────
    ext_metric = placer.extrinsics.copy()
    s_arr = scale[:, None, None] if isinstance(scale, np.ndarray) else float(scale)
    ext_metric[:, :, :3, 3] *= s_arr
    centres = _camera_centres(ext_metric)       # (T, K, 3)
    z_axes  = _camera_zaxis(placer.extrinsics)  # (T, K, 3) — unit dir, no scale

    # ── Build per-frame GT / pred arrays ──────────────────────────────────────
    ghost_pids  = sorted(pred_trans)
    all_frames  = sorted({f for pid in ghost_pids for f in pred_trans[pid]})
    frame_arr   = np.array(all_frames)
    T_plot      = len(frame_arr)
    f2i         = {f: i for i, f in enumerate(frame_arr)}

    pred_pelvis = np.full((T_plot, 3), np.nan)
    gt_pelvis   = np.full((T_plot, 3), np.nan)
    pred_R_arr  = [None] * T_plot
    gt_R_arr    = [None] * T_plot

    for ghost_pid in ghost_pids:
        gt_pid = (args.gt_pid if args.gt_pid
                  else min(gt_pids, key=lambda p: abs(p - ghost_pid)))

        for global_t, pelvis_w in pred_trans[ghost_pid].items():
            i = f2i.get(global_t)
            if i is None:
                continue
            pred_pelvis[i] = pelvis_w
            R = pred_orient.get(ghost_pid, {}).get(global_t)
            if R is not None:
                pred_R_arr[i] = R

        betas_gt = gt_body_data.get(gt_pid, {})
        for frame_idx, params in betas_gt.items():
            i = f2i.get(frame_idx)
            if i is None:
                continue
            J_gt = placer._smplx_fk(
                params["betas"][np.newaxis],
                params["body_pose"][np.newaxis],
                params["global_orient"][np.newaxis],
            )[0] + params["transl"]
            gt_pelvis[i] = J_gt[0]
            gt_R_arr[i]  = SciR.from_rotvec(
                params["global_orient"].astype(np.float64)
            ).as_matrix().astype(np.float32)

    # ── Per-frame errors ──────────────────────────────────────────────────────
    valid      = np.isfinite(pred_pelvis).all(-1) & np.isfinite(gt_pelvis).all(-1)
    root_err   = np.full(T_plot, np.nan)
    err_xyz    = np.full((T_plot, 3), np.nan)
    orient_err = np.full(T_plot, np.nan)

    for i in range(T_plot):
        if valid[i]:
            d          = pred_pelvis[i] - gt_pelvis[i]
            root_err[i]   = np.linalg.norm(d)
            err_xyz[i]    = np.abs(d)
        if pred_R_arr[i] is not None and gt_R_arr[i] is not None:
            orient_err[i] = _geodesic_deg(pred_R_arr[i], gt_R_arr[i])

    print(f"root  mean={np.nanmean(root_err)*100:.1f}cm  "
          f"depth(z)={np.nanmean(err_xyz[:,2])*100:.1f}cm  "
          f"orient  mean={np.nanmean(orient_err):.1f}°")

    # ── Figure ────────────────────────────────────────────────────────────────
    cam_colours = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
    t_norm      = np.linspace(0, 1, T_plot)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(scene_name, fontsize=13, fontweight='bold')

    ax3d  = fig.add_subplot(2, 3, (1, 4), projection='3d')
    ax_cc = fig.add_subplot(2, 3, 2)
    ax_cz = fig.add_subplot(2, 3, 3)
    ax_re = fig.add_subplot(2, 3, 5)
    ax_oe = fig.add_subplot(2, 3, 6)

    # ── Panel A: 3D scene ─────────────────────────────────────────────────────
    ax3d.set_title("3D scene — cameras + pelvis trajectories", fontsize=10)

    for k in range(K):
        valid_t = placer.cam_valid[:, k]
        C_k = centres[valid_t, k]
        if len(C_k) == 0:
            continue
        t_c = np.linspace(0, 1, len(C_k))
        ax3d.scatter(C_k[:, 0], C_k[:, 1], C_k[:, 2],
                     c=t_c, cmap='plasma', s=5, alpha=0.35,
                     label=f"{vnames[k]}")

        # Optical-axis arrows
        all_t = np.where(valid_t)[0]
        for ti in all_t[::args.arrow_stride]:
            c = centres[ti, k]
            d = z_axes[ti, k]
            d = d / (np.linalg.norm(d) + 1e-8) * args.arrow_len
            ax3d.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                        color=cam_colours[k], alpha=0.5, linewidth=0.8)

    # GT pelvis
    gt_v = np.isfinite(gt_pelvis).all(-1)
    if gt_v.any():
        gp = gt_pelvis[gt_v]
        ax3d.plot(gp[:, 0], gp[:, 1], gp[:, 2],
                  'g-', lw=2, label='GT pelvis', zorder=5)
        for i in np.where(gt_v)[0][::args.arrow_stride]:
            if gt_R_arr[i] is None:
                continue
            fwd = gt_R_arr[i] @ np.array([0., 0., 1.]) * args.arrow_len
            p   = gt_pelvis[i]
            ax3d.quiver(p[0], p[1], p[2], fwd[0], fwd[1], fwd[2],
                        color='green', alpha=0.7, linewidth=1.2)

    # Pred pelvis (coloured by time)
    pv = np.isfinite(pred_pelvis).all(-1)
    if pv.any():
        pp = pred_pelvis[pv]
        tc = t_norm[pv]
        segs = np.concatenate([pp[:-1, np.newaxis], pp[1:, np.newaxis]], axis=1)
        lc   = Line3DCollection(segs, cmap='autumn', linewidth=1.8, zorder=4)
        lc.set_array(tc[:-1])
        ax3d.add_collection3d(lc)
        ax3d.scatter(*pp[0], c='red', s=40, zorder=6, label='pred (start→end)')
        ax3d.scatter(*pp[-1], c='darkred', s=40, marker='*', zorder=6)

        for i in np.where(pv)[0][::args.arrow_stride]:
            if pred_R_arr[i] is None:
                continue
            fwd = pred_R_arr[i] @ np.array([0., 0., 1.]) * args.arrow_len
            p   = pred_pelvis[i]
            ax3d.quiver(p[0], p[1], p[2], fwd[0], fwd[1], fwd[2],
                        color='red', alpha=0.55, linewidth=0.8)

    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend(fontsize=7, loc='upper left', markerscale=1.5)

    # ── Panel B: Camera centre stability ──────────────────────────────────────
    ax_cc.set_title("Camera optical-centre stability (flat = stable)", fontsize=9)
    for k in range(K):
        valid_t = placer.cam_valid[:, k]
        C_k     = centres[valid_t, k]
        t_idx   = np.where(valid_t)[0]
        for dim, ls, lbl in zip(range(3), ['-', '--', ':'], ['x', 'y', 'z']):
            ax_cc.plot(t_idx, C_k[:, dim], color=cam_colours[k],
                       ls=ls, lw=0.8, alpha=0.85,
                       label=f"{vnames[k]}-{lbl}" if k < 3 else None)
    ax_cc.set_xlabel("Frame"); ax_cc.set_ylabel("Position (m)")
    ax_cc.legend(fontsize=6, ncol=2)

    # ── Panel C: Camera z-axis stability ──────────────────────────────────────
    ax_cz.set_title("Camera optical-axis (z) stability", fontsize=9)
    for k in range(K):
        valid_t = placer.cam_valid[:, k]
        z_k     = z_axes[valid_t, k]
        t_idx   = np.where(valid_t)[0]
        for dim, ls in zip(range(3), ['-', '--', ':']):
            ax_cz.plot(t_idx, z_k[:, dim], color=cam_colours[k],
                       ls=ls, lw=0.8, alpha=0.85)
    ax_cz.set_xlabel("Frame"); ax_cz.set_ylabel("Direction component")

    # ── Panel D: Root error time series ───────────────────────────────────────
    ax_re.set_title("Root translation error per frame", fontsize=9)
    ax_re.plot(frame_arr, root_err * 100,       'k-',           lw=1.2, label='total')
    ax_re.plot(frame_arr, err_xyz[:, 0] * 100,  color='tab:red',   lw=0.9, alpha=0.7, label='|x|')
    ax_re.plot(frame_arr, err_xyz[:, 1] * 100,  color='tab:green', lw=0.9, alpha=0.7, label='|y|')
    ax_re.plot(frame_arr, err_xyz[:, 2] * 100,  color='tab:blue',  lw=0.9, alpha=0.7, label='|z| depth')
    ax_re.axhline(np.nanmean(root_err) * 100, color='k', ls='--', lw=0.7, alpha=0.5)
    ax_re.set_xlabel("Frame"); ax_re.set_ylabel("Error (cm)")
    ax_re.legend(fontsize=8)

    # ── Panel E: Orientation error time series ────────────────────────────────
    ax_oe.set_title("Orientation error per frame", fontsize=9)
    ax_oe.plot(frame_arr, orient_err, color='tab:purple', lw=1.0)
    ax_oe.axhline(np.nanmean(orient_err), color='tab:purple', ls='--', lw=0.8, alpha=0.6,
                  label=f"mean={np.nanmean(orient_err):.1f}°")
    ax_oe.set_xlabel("Frame"); ax_oe.set_ylabel("Geodesic error (°)")
    ax_oe.legend(fontsize=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
