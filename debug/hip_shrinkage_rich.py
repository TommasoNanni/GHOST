"""Per-joint shrinkage analysis of the fusion module — RICH test.

QUESTION
--------
The fusion module degrades the HIPS relative to a chordal mean (bias ratio
1.19 / 1.33 in the error decomposition) while improving distal joints
(feet 0.25 / 0.37). The explainability study established the module acts as
a per-joint denoiser that shrinks the multi-view average toward an implicit
prior. Hypothesis:

    hip degradation = over-shrinkage of a joint whose per-view errors are
    CORRELATED across cameras (occlusion, SAM3D depth-from-proportions), so
    the chordal mean is already biased; shrinkage toward a mean pose then
    adds bias instead of removing variance, and the penalty concentrates on
    poses far from the prior (sitting, crouching, bent over).

WHAT IS MEASURED
----------------
Per body joint j (SMPL-X 1..21 == packed slots 0..20), pooling all valid
(t, p) over the cached scenes:

1. SHRINKAGE SLOPE a_j — regress the module against the chordal mean in a
   common tangent space at M_j (chordal mean of the GT rotations of j):

       log(M_j^T R_ghost)  ~  a_j * log(M_j^T R_chordal) + b_j

   a_j < 1 means the module compresses deviations from the reference:
   shrinkage = 1 - a_j.  R^2 reports how well the linear model fits.

2. ERROR vs ATYPICALITY — atypicality_t = geodesic(R_gt[t], M_j) and
   delta_t = err_ghost[t] - err_chordal[t] (geodesic to GT).  Pearson and
   Spearman corr(delta, atypicality): positive = the module loses ground
   exactly on atypical poses, as the shrinkage story predicts.

CONFIRMS the hypothesis: hips show above-average shrinkage (1 - a_j) AND a
clearly positive corr, while feet show a weaker corr (their gain being
unconditional denoising).
REFUTES: hips shrink no more than average, or delta uncorrelated with
atypicality — the hip degradation then needs another explanation.

CAVEAT — M_j is estimated from the TEST GT of the cached scenes, a proxy
for the training-set mean pose the module actually internalised.

Usage
-----
    pixi run python evaluation/hip_shrinkage_rich.py \
        --cache_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# SMPL-X joints 1..21 == packed slots 0..20 (root excluded from the cache).
_BODY_JOINT_NAMES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
_N_BODY = len(_BODY_JOINT_NAMES)

_GROUPS = {
    "hips": [0, 1], "knees": [3, 4], "ankles": [6, 7], "feet": [9, 10],
    "spine": [2, 5, 8], "neck/head": [11, 14], "collars": [12, 13],
    "shoulders": [15, 16], "elbows": [17, 18], "wrists": [19, 20],
}


def sixd_to_matrix(sixd: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3). The 6 values are the FIRST TWO ROWS of R
    (same convention as fusion/placer.py:_6d_to_aa_batch)."""
    r0, r1 = sixd[..., :3], sixd[..., 3:]
    b1 = r0 / (np.linalg.norm(r0, axis=-1, keepdims=True) + 1e-8)
    b2 = r1 - (b1 * r1).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def chordal_mean(R: np.ndarray) -> np.ndarray:
    """Project the Euclidean mean of (N, 3, 3) rotations onto SO(3)."""
    M = R.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def log_map(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 3) axis-angle via the matrix log."""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)                                      # (N,)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=-1)           # (N, 3)
    s = 2.0 * np.sin(theta)
    scale = np.where(theta < 1e-6, 0.5, theta / np.where(s < 1e-12, 1.0, s))
    return w * scale[:, None]


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(np.swapaxes(Ra, -1, -2) @ Rb,
                           axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/hip_shrinkage_rich.json"))
    args = ap.parse_args()

    files = sorted(args.cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no cache files in {args.cache_dir}")

    # Pool per-joint rotations over all scenes.
    pool: dict[str, list[np.ndarray]] = {"chordal": [], "ghost": [], "gt": []}
    for f in files:
        z = np.load(f)
        valid = z["valid"].astype(bool)                        # (T, P)
        t_idx, p_idx = np.nonzero(valid)
        for key, name in (("pose_chordal", "chordal"),
                          ("pose_ghost", "ghost"),
                          ("pose_gt", "gt")):
            R = sixd_to_matrix(z[key][t_idx, p_idx][:, :_N_BODY])  # (N, 21, 3, 3)
            pool[name].append(R.astype(np.float64))
        print(f"{f.stem}: {len(t_idx)} valid person-frames")

    Rc = np.concatenate(pool["chordal"])   # (N, 21, 3, 3)
    Rg = np.concatenate(pool["ghost"])
    Rt = np.concatenate(pool["gt"])
    n = Rc.shape[0]
    print(f"\npooled person-frames: {n}\n")

    rows = []
    for j in range(_N_BODY):
        M = chordal_mean(Rt[:, j])
        x = log_map(M.T @ Rc[:, j])        # chordal deviation from reference
        y = log_map(M.T @ Rg[:, j])        # ghost deviation from reference

        xc = x - x.mean(axis=0)
        yc = y - y.mean(axis=0)
        denom = (xc * xc).sum()
        a = float((xc * yc).sum() / denom) if denom > 0 else np.nan
        resid = yc - a * xc
        r2 = float(1.0 - (resid ** 2).sum() / (yc ** 2).sum()) if (yc ** 2).sum() > 0 else np.nan

        err_c = geodesic_deg(Rc[:, j], Rt[:, j])
        err_g = geodesic_deg(Rg[:, j], Rt[:, j])
        atyp = geodesic_deg(Rt[:, j], np.broadcast_to(M, Rt[:, j].shape))
        delta = err_g - err_c

        pear = float(np.corrcoef(delta, atyp)[0, 1])
        rows.append({
            "joint": _BODY_JOINT_NAMES[j],
            "slope_a": a, "shrinkage": 1.0 - a, "fit_r2": r2,
            "err_chordal_deg": float(err_c.mean()),
            "err_ghost_deg": float(err_g.mean()),
            "delta_deg": float(delta.mean()),
            "atypicality_deg": float(atyp.mean()),
            "pearson_delta_vs_atyp": pear,
            "spearman_delta_vs_atyp": spearman(delta, atyp),
        })

    hdr = (f"{'joint':<15} {'slope':>6} {'shrink':>7} {'R2':>5} "
           f"{'err_c':>6} {'err_g':>6} {'delta':>6} {'atyp':>6} {'pear':>6} {'spear':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['joint']:<15} {r['slope_a']:>6.3f} {r['shrinkage']:>7.3f} "
              f"{r['fit_r2']:>5.2f} {r['err_chordal_deg']:>6.2f} "
              f"{r['err_ghost_deg']:>6.2f} {r['delta_deg']:>+6.2f} "
              f"{r['atypicality_deg']:>6.2f} {r['pearson_delta_vs_atyp']:>+6.3f} "
              f"{r['spearman_delta_vs_atyp']:>+6.3f}")

    print("\nby group (means over member joints):")
    for gname, idxs in _GROUPS.items():
        g = [rows[i] for i in idxs]
        print(f"  {gname:<11} shrink={np.mean([r['shrinkage'] for r in g]):+.3f}  "
              f"delta={np.mean([r['delta_deg'] for r in g]):+.2f} deg  "
              f"spear={np.mean([r['spearman_delta_vs_atyp'] for r in g]):+.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"n_person_frames": int(n), "per_joint": rows}, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
