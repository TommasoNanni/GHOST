"""Per-scene decomposition of EgoHumans W-MPJPE† into its placement causes.

W† applies ONE SE(3) (no scale) fitted from camera centres, so every metric
error of the camera rig lands directly on the bodies. This probe measures, per
scene and WITHOUT running the pipeline:

  s_ma        the MapAnything scale actually used by the placer/eval
  s_opt       the scale a Sim(3) to the GT colmap cameras would have chosen
  scale_err   s_ma / s_opt - 1                      (pure scale error)
  cam_used    mean ||SE3(s_ma * C_pred) - C_gt||    (residual eval really has)
  cam_shape   mean ||Sim3(C_pred)   - C_gt||        (scale-free rig error)
  ext_cam     RMS radius of the GT camera centres about their centroid
  lever       mean ||GT pelvis - GT camera centroid||  (scale-error lever arm)

and pairs them with the per-scene W†/GA/PA read from an existing dump dir.

Everything is copied from evaluation/evaluate_egohumans_median.py so the
numbers line up with the published eval by construction.

Usage:
  pixi run python debug/w_error_decomposition_egohumans.py \
      --dump_dir eval_egohumans/dumps_smpl24_median \
      --out eval_explainability/w_decomp_egohumans.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

GHOST_ROOT = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/egohumans_new")
GT_ROOT = Path("/iopsstor/scratch/cscs/tnanni/egohumans_gt_full")

# same 12 limb joints the eval scores
_SMPL_EVAL = list(range(24))


# ── copied verbatim from evaluate_egohumans_median.py ──────────────────────
def se3_align(src: np.ndarray, dst: np.ndarray):
    """Kabsch SE(3) (no scale): R,t minimising ||R@src+t-dst||. src,dst (N,3)."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, dc - R @ sc


def sim3_align_full(pred: np.ndarray, gt: np.ndarray):
    """Umeyama with scale. Returns (s, R, t) mapping pred -> gt."""
    pc, gc = pred.mean(0), gt.mean(0)
    p0, g0 = pred - pc, gt - gc
    U, _, Vt = np.linalg.svd(p0.T @ g0)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float(np.sqrt((g0 ** 2).sum() / ((p0 ** 2).sum() + 1e-12)))
    return s, R, gc - s * R @ pc


def _colmap_to_aria(gt_scene: Path):
    p = gt_scene / "colmap" / "workplace" / "colmap_from_aria_transforms.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return np.linalg.inv(np.asarray(d["aria01"], dtype=np.float64))


def _gt_exo_cameras_aria(gt_scene: Path, T_c2a):
    from scipy.spatial.transform import Rotation as SciR
    imgs = gt_scene / "colmap" / "workplace" / "images.txt"
    out: dict[str, np.ndarray] = {}
    with open(imgs) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
            except ValueError:
                continue
            cam = parts[9].split("/")[0]
            if not cam.startswith("cam") or cam in out:
                continue
            R = SciR.from_quat([qx, qy, qz, qw]).as_matrix()
            C = -R.T @ np.array([tx, ty, tz])
            if T_c2a is not None:
                C = (T_c2a[:3, :3] @ C) + T_c2a[:3, 3]
            out[cam] = C
    return out


def _median_pred_centres(ghost_scene: Path):
    """{cam_name: median centre over frames} in raw ghost/VGGT units, + scale."""
    npz = ghost_scene / "vggt_cameras_centered.npz"
    if not npz.exists():
        return None, None
    v = np.load(npz, allow_pickle=False)
    names = [n.decode() if isinstance(n, bytes) else n for n in v["camera_names"]]
    extr, valid = v["extrinsics"], v["valid"]           # (T,K,3,4), (T,K)
    centres = {}
    for k, cam in enumerate(names):
        c = [-(extr[t, k, :, :3].T @ extr[t, k, :, 3])
             for t in range(extr.shape[0]) if valid[t, k]]
        if c:
            centres[cam] = np.median(np.stack(c), 0)
    sp = ghost_scene / "mapanything_scale_baseline.npy"
    if not sp.exists():
        return centres, None
    s = np.asarray(np.load(sp), dtype=np.float64).reshape(-1)
    return centres, float(np.median(s))


def scene_row(scene: str, activity: str, dump: Path):
    d = np.load(dump, allow_pickle=False)
    pred, gt = d["pred"].astype(np.float64), d["gt"].astype(np.float64)
    have_world = bool(d["have_world"])

    gt_scene = GT_ROOT / activity / scene
    ghost_scene = GHOST_ROOT / activity / scene
    T_c2a = _colmap_to_aria(gt_scene)
    gt_cams = _gt_exo_cameras_aria(gt_scene, T_c2a) if T_c2a is not None else {}
    pred_cams, s_ma = _median_pred_centres(ghost_scene)
    if not gt_cams or not pred_cams or s_ma is None:
        return None

    shared = [c for c in pred_cams if c in gt_cams]
    if len(shared) < 3:
        return None
    P = np.stack([pred_cams[c] for c in shared])       # raw ghost units
    G = np.stack([gt_cams[c] for c in shared])         # aria world, metres

    # scale the placer used vs the scale the cameras themselves imply
    s_opt, _, _ = sim3_align_full(P, G)
    scale_err = s_ma / s_opt - 1.0

    # residual the eval actually carries: fixed scale s_ma, then SE(3)
    R, t = se3_align(P * s_ma, G)
    cam_used = float(np.linalg.norm((P * s_ma) @ R.T + t - G, axis=1).mean())
    # scale-free rig error (rotation/topology only)
    s2, R2, t2 = sim3_align_full(P, G)
    cam_shape = float(np.linalg.norm(s2 * P @ R2.T + t2 - G, axis=1).mean())

    # scene geometry
    ctr = G.mean(0)
    ext_cam = float(np.linalg.norm(G - ctr, axis=1).mean())
    base_max = float(max(np.linalg.norm(G[i] - G[j])
                         for i in range(len(G)) for j in range(i + 1, len(G))))

    # lever arm: how far the bodies sit from the camera centroid
    pelvis = gt[..., 0, :].reshape(-1, 3)
    pelvis = pelvis[np.isfinite(pelvis).all(1)]
    lever = float(np.linalg.norm(pelvis - ctr, axis=1).mean()) if len(pelvis) else float("nan")

    # per-scene metrics, same code path as the eval aggregate
    L = _SMPL_EVAL
    valid = (np.isfinite(pred[..., L, :]).all((-1, -2))
             & np.isfinite(gt[..., L, :]).all((-1, -2)))
    w, root = [], []
    for tt in range(pred.shape[0]):
        ps = [p for p in range(pred.shape[1]) if valid[tt, p]]
        if not ps:
            continue
        pr, gg = pred[tt][ps][:, L], gt[tt][ps][:, L]
        if have_world:
            w.append(np.linalg.norm(pr - gg, axis=-1).mean())
            root.append(np.linalg.norm(pr.mean(1) - gg.mean(1), axis=-1).mean())
    if not w:
        return None

    return {
        "scene": scene, "activity": activity,
        "W": float(np.mean(w)) * 1000.0,
        "root": float(np.mean(root)) * 1000.0,
        "s_ma": s_ma, "s_opt": s_opt, "scale_err": scale_err,
        "cam_used_mm": cam_used * 1000.0, "cam_shape_mm": cam_shape * 1000.0,
        "ext_cam_m": ext_cam, "baseline_max_m": base_max, "lever_m": lever,
        "n_cams": len(shared),
        "pred_scale_err_mm": abs(scale_err) * lever * 1000.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", default="eval_egohumans/dumps_smpl24_median")
    ap.add_argument("--out", default="eval_explainability/w_decomp_egohumans.json")
    a = ap.parse_args()

    dumps = sorted(Path(a.dump_dir).rglob("*.npz"))
    act_of = {}
    for act in sorted(GT_ROOT.iterdir()):
        for sc in act.iterdir():
            act_of[sc.name] = act.name

    rows = []
    for dp in dumps:
        scene = dp.stem
        act = act_of.get(scene)
        if act is None:
            continue
        try:
            r = scene_row(scene, act, dp)
        except Exception as e:  # noqa: BLE001
            print(f"{scene}: {type(e).__name__}: {e}")
            continue
        if r:
            rows.append(r)
            print(f"{scene:20s} W={r['W']:7.1f} root={r['root']:7.1f} "
                  f"s_err={r['scale_err']*100:+6.1f}% cam_used={r['cam_used_mm']:7.1f} "
                  f"cam_shape={r['cam_shape_mm']:7.1f} ext={r['ext_cam_m']:5.1f}m "
                  f"lever={r['lever_m']:5.1f}m")

    if not rows:
        print("no rows"); return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

    W = np.array([r["W"] for r in rows])
    print(f"\n{len(rows)} scenes   W mean {W.mean():.1f}  median {np.median(W):.1f}")
    for key in ("scale_err", "cam_used_mm", "cam_shape_mm", "ext_cam_m",
                "lever_m", "baseline_max_m", "pred_scale_err_mm"):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        if key == "scale_err":
            v = np.abs(v)
        c = np.corrcoef(v, W)[0, 1]
        print(f"  {key:20s} mean {v.mean():9.3f}  median {np.median(v):9.3f}  corr(.,W) {c:+.3f}")

    print("\nper activity:")
    for act in sorted({r["activity"] for r in rows}):
        rs = [r for r in rows if r["activity"] == act]
        f = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        print(f"  {act:15s} n={len(rs):3d} W={f('W'):7.1f} root={f('root'):7.1f} "
              f"|s_err|={np.mean([abs(r['scale_err']) for r in rs])*100:5.1f}% "
              f"cam_used={f('cam_used_mm'):7.1f} cam_shape={f('cam_shape_mm'):6.1f} "
              f"ext={f('ext_cam_m'):5.1f}m lever={f('lever_m'):5.1f}m")


if __name__ == "__main__":
    main()
