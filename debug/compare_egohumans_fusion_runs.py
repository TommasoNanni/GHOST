"""Per-scene comparison of two EgoHumans dump dirs (v2 module vs geodesic median).

For every scene present in BOTH dirs, computes on the 24 SMPL joints:
  W†   — world MPJPE (pred already SE3-cam-aligned in the dump)
  PA   — per-frame per-person Sim3 MPJPE
  root — pelvis-only world error (joint 0)
  RR   — root-relative MPJPE (subtract pelvis from pred and gt, no rotation align)
  OR   — residual global-orient angle: best rigid R between root-centred pred and gt

Decomposition: W ≈ root placement + global orientation + articulated pose.
PA removes root+orient+scale; RR removes root only; OR isolates orientation.

Usage:
  pixi run python debug/compare_egohumans_fusion_runs.py \
      --a eval_egohumans/dumps_smpl24 --b eval_egohumans/dumps_smpl24_median
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

L = list(range(24))


def sim3_align(pred, gt):
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    s = np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12))
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return s * p0 @ R.T + gc


def rigid_angle(pred, gt):
    """Angle (deg) of the best rotation aligning root-centred pred to gt."""
    p0 = pred - pred[0]
    g0 = gt - gt[0]
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def scene_stats(pred, gt):
    T, P = pred.shape[:2]
    valid = np.isfinite(pred[..., L, :]).all((-1, -2)) & np.isfinite(gt[..., L, :]).all((-1, -2))
    w, pa, root, rr, orient = [], [], [], [], []
    for t in range(T):
        for p in range(P):
            if not valid[t, p]:
                continue
            pr, gtt = pred[t, p, L], gt[t, p, L]
            w.append(np.linalg.norm(pr - gtt, axis=-1).mean())
            a = sim3_align(pr, gtt)
            pa.append(np.linalg.norm(a - gtt, axis=-1).mean())
            root.append(np.linalg.norm(pr[0] - gtt[0]))
            rrp = (pr - pr[0]) - (gtt - gtt[0])
            rr.append(np.linalg.norm(rrp, axis=-1).mean())
            orient.append(rigid_angle(pr, gtt))
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return f(w), f(pa), f(root), f(rr), f(orient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="baseline dump dir (v2)")
    ap.add_argument("--b", type=Path, required=True, help="comparison dump dir (median)")
    args = ap.parse_args()

    files_a = {f.stem: f for f in args.a.glob("*/*.npz")}
    files_b = {f.stem: f for f in args.b.glob("*/*.npz")}
    common = sorted(set(files_a) & set(files_b))
    print(f"scenes: a={len(files_a)} b={len(files_b)} common={len(common)}")

    rows = []
    for name in common:
        da, db = np.load(files_a[name]), np.load(files_b[name])
        sa = scene_stats(da["pred"], da["gt"])
        sb = scene_stats(db["pred"], db["gt"])
        rows.append((name, sa, sb))

    print(f"\n{'scene':26s} {'W_a':>7} {'W_b':>7} {'dW':>7} | {'PA_a':>6} {'PA_b':>6} "
          f"| {'rt_a':>6} {'rt_b':>6} | {'RR_a':>6} {'RR_b':>6} | {'OR_a':>6} {'OR_b':>6}")
    for name, sa, sb in sorted(rows, key=lambda r: (r[2][0] - r[1][0]), reverse=True):
        mm = lambda v: v * 1000
        print(f"{name:26s} {mm(sa[0]):7.1f} {mm(sb[0]):7.1f} {mm(sb[0]-sa[0]):+7.1f} | "
              f"{mm(sa[1]):6.1f} {mm(sb[1]):6.1f} | {mm(sa[2]):6.0f} {mm(sb[2]):6.0f} | "
              f"{mm(sa[3]):6.1f} {mm(sb[3]):6.1f} | {sa[4]:6.1f} {sb[4]:6.1f}")

    A = np.array([r[1] for r in rows]); B = np.array([r[2] for r in rows])
    lab = ["W", "PA", "root", "RR", "OR(deg)"]
    print("\n=== POOLED (mean over scenes) ===")
    for i, l in enumerate(lab):
        s = 1000 if l != "OR(deg)" else 1
        print(f"  {l:8s} a={np.nanmean(A[:, i])*s:8.1f}  b={np.nanmean(B[:, i])*s:8.1f}  "
              f"d={(np.nanmean(B[:, i])-np.nanmean(A[:, i]))*s:+8.1f}")


if __name__ == "__main__":
    main()
