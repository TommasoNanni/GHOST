"""Bone-distance scale diagnostic.

For each frame, triangulate pairs of Goliath joints (shoulders, hips) using VGGT
cameras + MapAnything scale, compute their 3D distance, and compare to the same
distance in the GT SMPL-X mesh.

  s_residual = smplx_dist / triangulated_dist

If MapAnything scale is accurate, s_residual ≈ 1.0. A systematic bias (e.g. 0.9)
means the triangulated distances are inflated, i.e. scale is under-estimated.

Joint pairs used (rigid, well-separated):
  l_shoulder ↔ r_shoulder   (SMPL-X 16,17 / Goliath 5,6)
  l_hip      ↔ r_hip        (SMPL-X 1,2   / Goliath 9,10)
  l_shoulder ↔ l_hip        (SMPL-X 16,1  / Goliath 5,9)
  r_shoulder ↔ r_hip        (SMPL-X 17,2  / Goliath 6,10)

Usage:
    pixi run python debug/diagnose_bone_scale.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        [--body_split train_body] [--conf_thr 0.3]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models
from test_sapiens_dlt import _load_sapiens, _load_gt

# (smplx_j_A, goliath_j_A), (smplx_j_B, goliath_j_B), label
_PAIRS = [
    ((16, 5),  (17, 6),  "l_shoulder ↔ r_shoulder"),
    ((1,  9),  (2,  10), "l_hip      ↔ r_hip"),
    ((16, 5),  (1,  9),  "l_shoulder ↔ l_hip"),
    ((17, 6),  (2,  10), "r_shoulder ↔ r_hip"),
]


def _triangulate(
    goliath_j: int,
    sapiens_data: list[dict],
    pid: int,
    global_t: int,
    vggt_t: int,
    placer: BodyPlacer,
    scale: float,
    conf_thr: float,
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
        oc      = placer.original_coords[vggt_t, k]
        os_     = placer.original_size[vggt_t, k]
        u, v    = placer._orig_to_vggt(np.array([x, y]), oc, float(os_[0]), float(os_[1]))
        if not placer._in_bounds(u, v, oc[2], oc[3]):
            continue
        intr = placer.intrinsics[vggt_t, k].astype(np.float64)
        ext  = placer.extrinsics[vggt_t, k].astype(np.float64).copy()
        ext[:3, 3] *= scale
        pmats.append(intr @ ext)
        obs.append((u, v))
        weights.append(conf)
    if len(obs) < 2:
        return None
    return placer._triangulate_dlt(obs, pmats, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--body_split",  default="train_body")
    ap.add_argument("--conf_thr",    type=float, default=0.3)
    ap.add_argument("--pid",         type=int,   default=1, help="Ghost person ID in Sapiens data")
    ap.add_argument("--gt_pid",      type=int,   default=1, help="RICH person ID in GT pkl files")
    args = ap.parse_args()

    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(str(scene_dir), _smplx_arg)
    cam_dirs = placer._cam_dirs
    print(f"Scene: {scene_name}  T={placer.T}  cams={len(cam_dirs)}")

    scale = placer.load_mapanything_scale()
    if scale is None:
        print("ERROR: no mapanything_scale.npy found"); return
    print(f"Scale: median={float(np.median(scale)):.4f} m/VGGT-unit")

    sapiens_data = _load_sapiens(list(cam_dirs))
    gt_body_data, _ = _load_gt(scene_name, args.rich_root, args.body_split)

    zero_orient = np.zeros((1, 3), dtype=np.float32)
    pid    = args.pid
    gt_pid = args.gt_pid

    # per-pair residuals: list of s = smplx_dist / tri_dist
    pair_residuals: list[list[float]] = [[] for _ in _PAIRS]
    pair_tri:   list[list[float]] = [[] for _ in _PAIRS]
    pair_smplx: list[list[float]] = [[] for _ in _PAIRS]

    for global_t, params in sorted(gt_body_data.get(gt_pid, {}).items()):
        # need to know frame_start — derive from placer's valid range
        # VGGT frame 0 = first frame in sapiens data; we need to find offset
        # Use the same logic as eval scripts: frame_start stored in body_data npz
        vggt_t = None
        for cm in sapiens_data:
            if pid in cm and global_t in cm[pid]["local_t"]:
                vggt_t = cm[pid]["local_t"][global_t]
                break
        if vggt_t is None or vggt_t < 0 or vggt_t >= placer.T:
            continue

        s = float(scale[vggt_t])

        # GT SMPL-X joint positions (world frame, metric)
        J_gt = placer._smplx_fk(
            params["betas"][np.newaxis],
            params["body_pose"][np.newaxis],
            params["global_orient"][np.newaxis],
        )[0] + params["transl"]   # (55, 3)

        for pi, ((sj_a, gj_a), (sj_b, gj_b), _) in enumerate(_PAIRS):
            pt_a = _triangulate(gj_a, sapiens_data, pid, global_t, vggt_t, placer, s, args.conf_thr)
            pt_b = _triangulate(gj_b, sapiens_data, pid, global_t, vggt_t, placer, s, args.conf_thr)
            if pt_a is None or pt_b is None:
                continue

            tri_dist   = float(np.linalg.norm(pt_a - pt_b))
            smplx_dist = float(np.linalg.norm(J_gt[sj_a] - J_gt[sj_b]))

            if tri_dist < 1e-4:
                continue

            pair_residuals[pi].append(smplx_dist / tri_dist)
            pair_tri[pi].append(tri_dist)
            pair_smplx[pi].append(smplx_dist)

    print(f"\n{'Pair':<30}  {'N':>5}  {'s_res mean':>10}  {'s_res std':>9}  {'tri_dist(cm)':>13}  {'smplx_dist(cm)':>15}")
    print("-" * 90)
    all_res = []
    for pi, (_, _, label) in enumerate(_PAIRS):
        res = pair_residuals[pi]
        if not res:
            print(f"{label:<30}  {'—':>5}")
            continue
        r  = np.array(res)
        td = np.array(pair_tri[pi])
        sd = np.array(pair_smplx[pi])
        all_res.extend(res)
        print(f"{label:<30}  {len(r):>5}  {r.mean():>10.4f}  {r.std():>9.4f}"
              f"  {td.mean()*100:>13.1f}  {sd.mean()*100:>15.1f}")

    if all_res:
        a = np.array(all_res)
        print(f"\nAll pairs combined: mean={a.mean():.4f}  std={a.std():.4f}  "
              f"bias={(a.mean()-1)*100:+.1f}%")
        print(f"  → MapAnything scale is {(1/a.mean()-1)*100:+.1f}% off "
              f"(s_res<1 means triangulated dist too large = scale underestimated)")


if __name__ == "__main__":
    main()
