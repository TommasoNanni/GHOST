"""Visualize placer's sapiens-DLT placement vs GT SMPL-X joints for one frame.

4-panel figure: front (XY), side (ZY), top (XZ), 3-D.
Blue = BodyPlacer.estimate_procrustes_dlt_sapiens FK skeleton.
Red  = GT SMPL-X from RICH pkl.
Thin black line between matched joints shows per-joint offset.

Usage:
    pixi run python visualize/vis_triangulated_joints.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        [--frame 5] [--pid 1] [--gt_pid 1] [--body_split train_body] \\
        [--out vis_tri_joints.png]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion.placer import BodyPlacer, _6d_to_aa_batch
from utilities.rich_gender_plugin import resolve_smplx_models
from test_sapiens_dlt import _load_gt

# ------------------------------------------------------------------
# Joint metadata
# ------------------------------------------------------------------
_SMPLX_NAMES = {
    0:"pelvis", 1:"l_hip", 2:"r_hip", 3:"spine1",
    4:"l_knee", 5:"r_knee", 6:"spine2", 7:"l_ankle",
    8:"r_ankle", 9:"spine3", 10:"l_foot", 11:"r_foot",
    12:"neck", 13:"l_collar", 14:"r_collar", 15:"head",
    16:"l_shoulder", 17:"r_shoulder", 18:"l_elbow",
    19:"r_elbow", 20:"l_wrist", 21:"r_wrist",
}

# SMPL-X kinematic tree (child → parent) for first 22 joints
_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

_GT_COLOR   = "#e05252"   # red
_TRI_COLOR  = "#3a7fd5"   # blue
_SKEL_COLOR = "#cccccc"   # light gray for GT skeleton lines


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------
def _draw_skeleton_2d(ax, J, parents, color, lw=1.0, alpha=0.5):
    for j, p in enumerate(parents):
        if p < 0 or j >= len(J) or p >= len(J):
            continue
        ax.plot([J[p, 0], J[j, 0]], [J[p, 1], J[j, 1]],
                color=color, lw=lw, alpha=alpha, zorder=1)


def _draw_skeleton_3d(ax, J, parents, color, lw=1.0, alpha=0.5):
    for j, p in enumerate(parents):
        if p < 0 or j >= len(J) or p >= len(J):
            continue
        ax.plot([J[p, 0], J[j, 0]],
                [J[p, 1], J[j, 1]],
                [J[p, 2], J[j, 2]],
                color=color, lw=lw, alpha=alpha)


def _scatter2d(ax, pts, color, s=60, zorder=3, label=None):
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=s, zorder=zorder,
               edgecolors="white", linewidths=0.4, label=label)


def _label_joints(ax, pts, names, fontsize=6, color="black"):
    for (x, y), name in zip(pts, names):
        ax.text(x, y, f" {name}", fontsize=fontsize, color=color,
                va="center", zorder=5)


def _connect_pairs_2d(ax, tri_pts, gt_pts, names):
    for (tx, ty), (gx, gy), name in zip(tri_pts, gt_pts, names):
        ax.plot([tx, gx], [ty, gy], color="black", lw=0.6, alpha=0.5, zorder=2)


def _connect_pairs_3d(ax, tri_pts, gt_pts):
    for t, g in zip(tri_pts, gt_pts):
        ax.plot([t[0], g[0]], [t[1], g[1]], [t[2], g[2]],
                color="black", lw=0.6, alpha=0.5)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--frame",       type=int, default=5, help="global_t frame index")
    ap.add_argument("--pid",         type=int, default=1, help="Ghost person ID")
    ap.add_argument("--gt_pid",      type=int, default=1, help="RICH GT person ID")
    ap.add_argument("--body_split",  default="train_body")
    ap.add_argument("--conf_thr",    type=float, default=0.3)
    ap.add_argument("--fusion_output", type=Path, default=None,
                    help="Path to fusion output .npz (default: fusion_outputs/<scene>.npz). "
                         "If not found, falls back to SAM3D body_data.")
    ap.add_argument("--pid_slot",    type=int, default=0,
                    help="Slot index of this pid in fused pose tensor (default 0).")
    ap.add_argument("--frame_start", type=int, default=0,
                    help="Global frame index of fused_pose[0] (default 0).")
    ap.add_argument("--out",         type=Path, default=Path("renders/debug/vis_tri_joints.png"))
    args = ap.parse_args()

    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    global_t   = args.frame
    pid        = args.pid
    gt_pid     = args.gt_pid

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(str(scene_dir), _smplx_arg)
    print(f"Scene: {scene_name}  T={placer.T}  cams={len(placer._cam_dirs)}")

    scale = placer.load_mapanything_scale()
    if scale is None:
        print("ERROR: no mapanything_scale.npy"); return
    print(f"Scale median={float(np.median(scale)):.4f}")

    # ── Load fusion output (betas + body_pose), fall back to SAM3D ─
    fusion_path = args.fusion_output or (_REPO_ROOT / "fusion_outputs" / f"{scene_name}.npz")
    betas           = np.zeros(10, np.float32)
    body_pose_frame = None
    fused_pose_by_pid: dict | None = None

    if fusion_path.exists():
        fo = np.load(fusion_path, allow_pickle=False)
        fused_pose_full = fo["pose"]     # (T, P, 55, 6) — joint 0=root, 1-21=body
        betas = fo["shape"][args.pid_slot].astype(np.float32)[:10]
        t_local = global_t - args.frame_start
        if 0 <= t_local < fused_pose_full.shape[0]:
            # joints 1:22 → body_pose (21 joints × 6D → 63 AA)
            slot_pose = fused_pose_full[t_local, args.pid_slot, 1:22]  # (21, 6)
            body_pose_frame = _6d_to_aa_batch(slot_pose).reshape(63).astype(np.float32)
            # placer expects (T, ≥21, 6); pass joints 1:22 for all frames
            fused_pose_by_pid = {pid: fused_pose_full[:, args.pid_slot, 1:22]}  # (T, 21, 6)
            print(f"Using fusion body_pose + betas from {fusion_path.name}")
        else:
            print(f"WARN: frame {global_t} (t_local={t_local}) out of range — falling back to SAM3D")

    if body_pose_frame is None:
        for cam_dir in placer._cam_dirs:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if not bf.exists():
                continue
            d  = np.load(bf, allow_pickle=False)
            fi = d["frame_indices"].astype(int)
            idx = np.where(fi == global_t)[0]
            if not len(idx):
                continue
            lt = int(idx[0])
            if "smplx_betas" in d.files:
                betas = d["smplx_betas"][lt].astype(np.float32)[:10]
            if "smplx_body_pose" in d.files:
                body_pose_frame = d["smplx_body_pose"][lt].astype(np.float32)
            break
        print("Using SAM3D body_pose (fusion output not found or frame out of range)")

    if body_pose_frame is None:
        print(f"ERROR: no body_pose found for pid {pid} frame {global_t}"); return

    # ── Run placer's sapiens DLT ────────────────────────────────────
    translations, orientations = placer.estimate_procrustes_dlt_sapiens(
        scale, {pid}, {pid: betas}, conf_thr=args.conf_thr,
        fused_pose_by_pid=fused_pose_by_pid,
        frame_start=args.frame_start,
    )
    if pid not in translations or global_t not in translations[pid]:
        print(f"ERROR: estimate_procrustes_dlt_sapiens returned no result "
              f"for pid {pid} frame {global_t}"); return

    transl = translations[pid][global_t]   # (3,) pelvis world
    R      = orientations[pid][global_t]   # (3,3)

    # Canonical joints with zero global orient — same formula as eval_placer_trans.py
    zero_orient = np.zeros((1, 3), np.float32)
    J_can = placer._smplx_fk(betas[np.newaxis], body_pose_frame[np.newaxis], zero_orient)[0]
    J_pred = (R.astype(np.float64) @ (J_can - J_can[0]).T).T.astype(np.float32) + transl
    print(f"Pelvis pred: {transl}")

    # ── GT SMPL-X skeleton ──────────────────────────────────────────
    gt_body_data, _ = _load_gt(scene_name, args.rich_root, args.body_split)
    params = gt_body_data.get(gt_pid, {}).get(global_t)
    if params is None:
        print(f"ERROR: GT frame {global_t} pid {gt_pid} not found"); return

    J_gt = placer._smplx_fk(
        params["betas"][np.newaxis],
        params["body_pose"][np.newaxis],
        params["global_orient"][np.newaxis],
    )[0] + params["transl"]   # (55, 3) world frame
    print(f"Pelvis GT:   {J_gt[0]}")

    # ── First 22 body joints ────────────────────────────────────────
    pred_arr = J_pred[:22]   # (22, 3)
    gt_arr   = J_gt[:22]     # (22, 3)
    names    = [_SMPLX_NAMES[j] for j in range(22)]

    errs = np.linalg.norm(pred_arr - gt_arr, axis=1)
    print(f"\n{'Joint':<15} {'error(cm)':>10}")
    print("-" * 30)
    for name, e in zip(names, errs):
        print(f"  {name:<13} {e*100:>10.1f}")
    print("-" * 30)
    print(f"  {'mean':<13} {errs.mean()*100:>10.1f}")

    # ── Figure: 2×2 grid ────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"{scene_name} | frame {global_t} | sapiens DLT placement",
                 fontsize=12, fontweight="bold")

    views = [
        # (subplot_idx, xlabel, ylabel, xi, yi, flip_y, title)
        (1, "X (left-right)", "Y (up-down)", 0, 1, True,  "Front  (X–Y)"),
        (2, "Z (depth)",      "Y (up-down)", 2, 1, True,  "Side   (Z–Y)"),
        (3, "X (left-right)", "Z (depth)",   0, 2, False, "Top    (X–Z)"),
    ]

    axes_2d = []
    for subplot_idx, xlabel, ylabel, xi, yi, flip_y, title in views:
        ax = fig.add_subplot(2, 2, subplot_idx)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.5)

        # Collect 2d coords and apply flip before ANY drawing
        gt2d   = gt_arr[:, [xi, yi]]
        pred2d = pred_arr[:, [xi, yi]]
        if flip_y:
            gt2d   = gt2d   * [1, -1]
            pred2d = pred2d * [1, -1]

        _draw_skeleton_2d(ax, gt2d,   _PARENTS, _GT_COLOR,  lw=1.2, alpha=0.5)
        _draw_skeleton_2d(ax, pred2d, _PARENTS, _TRI_COLOR, lw=1.2, alpha=0.5)

        _scatter2d(ax, gt2d,   _GT_COLOR,  s=60, label="GT SMPL-X")
        _scatter2d(ax, pred2d, _TRI_COLOR, s=40, label="Sapiens DLT")
        _label_joints(ax, gt2d,   names, fontsize=6, color=_GT_COLOR)
        _label_joints(ax, pred2d, names, fontsize=6, color=_TRI_COLOR)

        axes_2d.append(ax)

    # ── 3-D subplot ────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.set_title("3-D view", fontsize=10)
    ax3.set_xlabel("X", fontsize=8)
    ax3.set_ylabel("Z", fontsize=8)   # matplotlib 3d: y axis = depth
    ax3.set_zlabel("-Y (up)", fontsize=8)

    def _to_plot3(pts):
        # X=right, Y_plot=depth(Z), Z_plot=-Y so person stands upright
        return pts[:, 0], pts[:, 2], -pts[:, 1]

    _draw_skeleton_3d(ax3,
                      np.stack([gt_arr[:, 0], gt_arr[:, 2], -gt_arr[:, 1]], axis=1),
                      _PARENTS, _GT_COLOR, lw=1.0, alpha=0.4)
    _draw_skeleton_3d(ax3,
                      np.stack([pred_arr[:, 0], pred_arr[:, 2], -pred_arr[:, 1]], axis=1),
                      _PARENTS, _TRI_COLOR, lw=1.0, alpha=0.4)

    gx, gz, gy = _to_plot3(gt_arr)
    tx, tz, ty = _to_plot3(pred_arr)
    ax3.scatter(gx, gz, gy, c=_GT_COLOR,  s=50, depthshade=False, label="GT")
    ax3.scatter(tx, tz, ty, c=_TRI_COLOR, s=40, depthshade=False, label="Sapiens DLT")

    for xi, zi, yi, name in zip(gx, gz, gy, names):
        ax3.text(xi, zi, yi, f" {name}", fontsize=5, color=_GT_COLOR)

    ax3.view_init(elev=15, azim=-60)

    # ── Legend ──────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GT_COLOR,
               markersize=8, label="GT SMPL-X"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TRI_COLOR,
               markersize=8, label="Sapiens DLT (VGGT+MA scale)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    # ── Error table inset ─────────────────────────────────────────
    err_text = f"{'Joint':<13} {'err(cm)':>8}\n"
    for name, e in zip(names, errs):
        err_text += f"{name:<13} {e * 100:>8.1f}\n"
    err_text += f"{'mean':<13} {errs.mean() * 100:>8.1f}"
    fig.text(0.76, 0.52, err_text, fontsize=7, family="monospace",
             va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = args.out
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
