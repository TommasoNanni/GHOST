"""Per-take metric-scale error on EgoExo4D — the same quantity
debug/w_error_decomposition_egohumans.py measures for EgoHumans.

  s_ma   median of mapanything_scale_baseline.npy (the scale the placer uses)
  s_opt  Sim(3) scale aligning the UNSCALED VGGT camera centres to the GT
         gopro centres  (identical to evaluate_egoexo_median._gt_scale)
  err    s_ma / s_opt - 1

W-MPJPE† applies one SE(3) fitted from camera centres, so err lands on the
bodies multiplied by their distance from the camera centroid.

Usage: pixi run python debug/scale_error_egoexo.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egoexo4d")
GT_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/egoexo4d/gt")


def gopro_centres(gt_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    with open(gt_dir / "gopro_calibs.csv") as f:
        for row in csv.DictReader(f):
            out[row["cam_uid"]] = np.array(
                [float(row[f"t{a}_world_cam"]) for a in "xyz"], dtype=np.float64)
    return out


def umeyama_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    p0, g0 = pred - pred.mean(0), gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    return float((S * [1, 1, d]).sum() / ((p0 ** 2).sum() + 1e-12))


def sim3_resid(pred: np.ndarray, gt: np.ndarray) -> float:
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12))
    return float(np.linalg.norm(s * p0 @ R.T + gc - gt, axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_explainability/scale_err_egoexo.json")
    a = ap.parse_args()

    rows = []
    for scene in sorted(GHOST_ROOT.iterdir()):
        npz = scene / "vggt_cameras_centered.npz"
        sp = scene / "mapanything_scale_baseline.npy"
        gt_dir = GT_ROOT / scene.name
        if not (npz.exists() and sp.exists() and (gt_dir / "gopro_calibs.csv").exists()):
            continue
        v = np.load(npz, allow_pickle=False)
        names = [n.decode() if isinstance(n, bytes) else n for n in v["camera_names"]]
        extr, valid = v["extrinsics"], v["valid"]
        gtc = gopro_centres(gt_dir)
        P, G = [], []
        for k, cam in enumerate(names):
            if cam not in gtc or not valid[:, k].any():
                continue
            t0 = int(np.argmax(valid[:, k]))
            R, t = extr[t0, k, :, :3], extr[t0, k, :, 3]
            P.append(-R.T @ t)
            G.append(gtc[cam])
        if len(P) < 3:
            continue
        P, G = np.stack(P), np.stack(G)
        s_ma = float(np.median(np.asarray(np.load(sp), dtype=np.float64).reshape(-1)))
        s_opt = umeyama_scale(P, G)
        ctr = G.mean(0)
        rows.append({
            "scene": scene.name, "s_ma": s_ma, "s_opt": s_opt,
            "scale_err": s_ma / s_opt - 1.0,
            "rig_resid_mm": sim3_resid(P, G) * 1000.0,
            "ext_cam_m": float(np.linalg.norm(G - ctr, axis=1).mean()),
            "n_cams": len(P),
        })
        r = rows[-1]
        print(f"{scene.name:34s} s_err={r['scale_err']*100:+7.1f}%  "
              f"rig={r['rig_resid_mm']:7.1f}mm  ext={r['ext_cam_m']:5.1f}m")

    if not rows:
        print("no rows"); return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

    se = np.array([r["scale_err"] for r in rows])
    ext = np.array([r["ext_cam_m"] for r in rows])
    rig = np.array([r["rig_resid_mm"] / 1000.0 / r["ext_cam_m"] for r in rows])
    print(f"\n{len(rows)} takes")
    print(f"  scale_err  mean {se.mean()*100:+.1f}%  median {np.median(se)*100:+.1f}%  "
          f"|mean| {np.abs(se).mean()*100:.1f}%  |median| {np.median(np.abs(se))*100:.1f}%  "
          f"[{se.min()*100:+.0f}, {se.max()*100:+.0f}]")
    print(f"  rig distortion (resid/extent)  mean {rig.mean()*100:.1f}%  "
          f"median {np.median(rig)*100:.1f}%")
    print(f"  camera extent  mean {ext.mean():.1f} m  median {np.median(ext):.1f} m")
    print(f"  corr(|scale_err|, extent) = {np.corrcoef(np.abs(se), ext)[0,1]:+.3f}")

    print("\nby venue prefix:")
    ven = {}
    for r in rows:
        ven.setdefault(r["scene"].split("_")[0], []).append(r)
    for k in sorted(ven):
        q = ven[k]
        s = np.array([x["scale_err"] for x in q])
        print(f"  {k:16s} n={len(q):3d}  s_err {s.mean()*100:+6.1f}%  "
              f"|s| {np.abs(s).mean()*100:5.1f}%  ext {np.mean([x['ext_cam_m'] for x in q]):5.1f}m")


if __name__ == "__main__":
    main()
