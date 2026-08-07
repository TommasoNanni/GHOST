"""Bias / residual / drift decomposition of pose error — RICH test.

QUESTION
--------
The fusion module is WORSE than a chordal mean on per-frame pose accuracy
(root-relative 55-joint: 68.06 vs 62.39 mm) yet BETTER on W-MPJPE-100
(70.4 vs 84.0 mm). W-MPJPE-100 fits a Sim(3) per 100-frame segment, which
absorbs a CONSTANT error but not a FLUCTUATING one. So the hypothesis is:

    the module's error is larger but temporally stable (a systematic bias),
    the mean's error is smaller but fluctuates.

DECOMPOSITION
-------------
Per (person, joint) and per 100-frame segment, with E_t = R_t^pred^T @ R_t^gt:

    total     mean_t angle(E_t)
    bias      angle(Ebar),  Ebar = chordal mean of {E_t} over the segment
    residual  mean_t angle(Ebar^T @ E_t)      error after removing the constant
    drift     mean_t angle(E_t^T @ E_{t+1})   frame-to-frame wobble of the error

CONFIRMS the hypothesis: B has higher total but a higher bias/total ratio and
lower residual and drift than A — the module's error is a fixed per-segment
offset that the alignment removes.

REFUTES it: similar bias/total ratios, or B with higher residual/drift. Then the
W-MPJPE-100 advantage does not come from body pose at all and the next suspect is
the placer (smoothness/drift of the root trajectory).

INPUT
-----
The cached fused poses written by evaluation/temporal_smoothness_rich.py — no
inference pass. Each <scene>.npz holds pose_chordal / pose_ghost / pose_gt as
(T, P, 54, 6) plus the shared `valid` (T, P) mask.

SCOPE
-----
Body joints only: SMPL-X 1..21 == packed slots 0..20, global orientation
excluded — the same subset as the smoothness experiment. Segments are taken from
`_iter_segments` in evaluation/evaluate_rich.py, IMPORTED rather than
reimplemented so the windowing cannot drift from the evaluator's.

Usage
-----
    pixi run python evaluation/error_decomposition_rich.py \\
        --cache_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Segmentation comes from the evaluator itself, not a local copy.
from evaluation.evaluate_rich import _iter_segments

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_N_BODY = 21
_VARIANTS = ("chordal", "ghost")
_QUANTITIES = ("total", "bias", "residual", "drift")

_JOINT_NAMES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def sixd_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3). Gram-Schmidt on the first two rows."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def angle_deg(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of (...,3,3) rotations, degrees. arccos argument clamped."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def chordal_mean(R: np.ndarray) -> np.ndarray:
    """Chordal mean of (N, J, 3, 3) over N -> (J, 3, 3). SVD projection onto SO(3)."""
    M = R.astype(np.float64).mean(axis=0)                  # (J,3,3)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros_like(M)
    D[..., 0, 0] = D[..., 1, 1] = 1.0
    D[..., 2, 2] = d
    out = U @ D @ Vt
    dets = np.linalg.det(out)
    assert np.allclose(dets, 1.0, atol=1e-6), f"chordal mean det != +1: {dets.min()},{dets.max()}"
    return out


def contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a 1-D bool array."""
    if not valid.any():
        return []
    padded = np.concatenate(([False], valid.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def process_scene(npz_path: Path, segment_len: int, min_valid: int) -> dict | None:
    d = np.load(npz_path, allow_pickle=False)
    valid = d["valid"].astype(bool)                        # (T,P)
    T, P = valid.shape

    R = {}
    for v in _VARIANTS + ("gt",):
        key = f"pose_{v}"
        R[v] = sixd_to_matrix(d[key][:, :, :_N_BODY])      # (T,P,21,3,3)
        if R[v].shape != (T, P, _N_BODY, 3, 3):
            raise RuntimeError(f"{npz_path.name}:{key} bad shape {R[v].shape}")

    acc = {v: {q: [] for q in _QUANTITIES} for v in _VARIANTS}
    per_joint = {v: {"bias": np.zeros(_N_BODY), "residual": np.zeros(_N_BODY),
                     "n": 0} for v in _VARIANTS}
    n_seg = 0

    for t0, t1 in _iter_segments(valid, segment_len):
        for p in range(P):
            vseg = valid[t0:t1, p]
            if int(vseg.sum()) < min_valid:
                continue
            idx = t0 + np.flatnonzero(vseg)                # absolute frame indices
            n_seg += 1
            # Runs of CONSECUTIVE valid frames inside the segment, for drift only.
            runs = [(t0 + s, t0 + e) for s, e in contiguous_runs(vseg)]

            for v in _VARIANTS:
                # E_t = R_pred^T @ R_gt        (T_seg, 21, 3, 3)
                E = np.einsum("njab,njac->njbc", R[v][idx, p], R["gt"][idx, p])

                total = angle_deg(E).mean(axis=0)                       # (21,)
                Ebar  = chordal_mean(E)                                 # (21,3,3)
                bias  = angle_deg(Ebar)                                 # (21,)
                res   = angle_deg(np.einsum("jab,njac->njbc", Ebar, E)).mean(axis=0)

                drift_parts = []
                for s, e in runs:
                    if e - s < 2:
                        continue
                    Er = np.einsum("njab,njac->njbc",
                                   R[v][s:e, p], R["gt"][s:e, p])       # (L,21,3,3)
                    drift_parts.append(
                        angle_deg(np.einsum("njab,njac->njbc", Er[:-1], Er[1:])))
                drift = (np.concatenate(drift_parts, axis=0).mean(axis=0)
                         if drift_parts else np.full(_N_BODY, np.nan))

                acc[v]["total"].append(total)
                acc[v]["bias"].append(bias)
                acc[v]["residual"].append(res)
                acc[v]["drift"].append(drift)
                per_joint[v]["bias"]     += bias
                per_joint[v]["residual"] += res
                per_joint[v]["n"]        += 1

    if n_seg == 0:
        return None

    # A and B must cover exactly the same (segment, person, joint) set.
    counts = {v: len(acc[v]["total"]) for v in _VARIANTS}
    assert len(set(counts.values())) == 1, f"segment count mismatch: {counts}"

    return {"scene": npz_path.stem, "T": T, "P": P,
            "n_segments": counts["chordal"], "acc": acc, "per_joint": per_joint}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bias/residual/drift decomposition of pose error (RICH, cached poses)")
    ap.add_argument("--cache_dir", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/"
                                 "fused_cache/rich_test"))
    ap.add_argument("--segment_len", type=int, default=100)
    ap.add_argument("--min_valid",   type=int, default=20,
                    help="drop segments with fewer valid frames than this")
    ap.add_argument("--scenes",      default="")
    ap.add_argument("--out", type=Path,
                    default=Path("eval_explainability/error_decomposition_rich.json"))
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    files = sorted(args.cache_dir.glob("*.npz"))
    if wanted:
        files = [f for f in files if f.stem in wanted]
    if not files:
        logger.error(f"no cached scenes in {args.cache_dir}")
        return
    logger.info(f"{len(files)} cached scene(s) from {args.cache_dir}")

    rows = []
    for f in files:
        try:
            r = process_scene(f, args.segment_len, args.min_valid)
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"{f.stem}: skipped ({type(e).__name__}: {e})")
            continue
        if r is None:
            logger.warning(f"{f.stem}: no usable segments")
            continue
        bt = {v: np.concatenate(r["acc"][v]["bias"]).mean() /
                 np.concatenate(r["acc"][v]["total"]).mean() for v in _VARIANTS}
        logger.info(f"{f.stem:<40} segs={r['n_segments']:3d}  "
                    f"bias/total  A={bt['chordal']:.3f}  B={bt['ghost']:.3f}")
        rows.append(r)

    if not rows:
        logger.error("no scenes processed")
        return

    agg, per_joint = {}, {}
    for v in _VARIANTS:
        agg[v] = {}
        for q in _QUANTITIES:
            vals = np.concatenate([np.concatenate(r["acc"][v][q]) for r in rows])
            vals = vals[np.isfinite(vals)]
            agg[v][q] = {"mean": float(vals.mean()), "median": float(np.median(vals))}
        agg[v]["bias_over_total"] = agg[v]["bias"]["mean"] / agg[v]["total"]["mean"]
        n = sum(r["per_joint"][v]["n"] for r in rows)
        per_joint[v] = {
            "bias":     (sum(r["per_joint"][v]["bias"]     for r in rows) / n).tolist(),
            "residual": (sum(r["per_joint"][v]["residual"] for r in rows) / n).tolist(),
        }

    n_triplets = int(sum(r["n_segments"] for r in rows)) * _N_BODY

    print(f"\n{'='*78}")
    print("ERROR DECOMPOSITION — RICH test, body joints 1..21, global orient excluded")
    print(f"  scenes={len(rows)}  segments={sum(r['n_segments'] for r in rows)}  "
          f"(segment,person,joint) triplets={n_triplets:,} (identical for A and B)")
    print(f"  segment_len={args.segment_len}  min_valid={args.min_valid}")
    print(f"{'='*78}")
    print(f"{'quantity':<14}{'A chordal mean':>16}{'A median':>11}"
          f"{'B ghost mean':>15}{'B median':>11}   (deg)")
    print("-" * 78)
    for q in _QUANTITIES:
        a, b = agg["chordal"][q], agg["ghost"][q]
        print(f"{q:<14}{a['mean']:>16.3f}{a['median']:>11.3f}"
              f"{b['mean']:>15.3f}{b['median']:>11.3f}")
    print("-" * 78)
    print(f"{'bias / total':<14}{agg['chordal']['bias_over_total']:>16.3f}"
          f"{'':>11}{agg['ghost']['bias_over_total']:>15.3f}")
    print()

    ba, bb = agg["chordal"], agg["ghost"]
    confirms = (bb["total"]["mean"] > ba["total"]["mean"]
                and bb["bias_over_total"] > ba["bias_over_total"]
                and bb["residual"]["mean"] < ba["residual"]["mean"]
                and bb["drift"]["mean"] < ba["drift"]["mean"])
    print("  VERDICT: " + (
        "CONFIRMS — B's error is more of a fixed per-segment offset, which the "
        "W-MPJPE-100 Sim(3) removes."
        if confirms else
        "DOES NOT CONFIRM — B is not more bias-dominated on every criterion. "
        "See the rows above; next suspect is the placer root trajectory."))
    print()
    print("  per-joint bias / residual (degrees)")
    print(f"    {'joint':<16}{'A bias':>9}{'B bias':>9}{'A resid':>10}{'B resid':>10}"
          f"{'B/A bias':>10}")
    for j, name in enumerate(_JOINT_NAMES):
        ab, bb_ = per_joint["chordal"]["bias"][j], per_joint["ghost"]["bias"][j]
        ar, br  = per_joint["chordal"]["residual"][j], per_joint["ghost"]["residual"][j]
        print(f"    {name:<16}{ab:>9.3f}{bb_:>9.3f}{ar:>10.3f}{br:>10.3f}"
              f"{(bb_ / ab if ab else float('nan')):>10.2f}")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {"cache_dir": str(args.cache_dir), "segment_len": args.segment_len,
                       "min_valid": args.min_valid, "n_scenes": len(rows),
                       "n_segments": int(sum(r["n_segments"] for r in rows)),
                       "n_triplets": n_triplets,
                       "joint_set": "SMPL-X body 1..21 (packed 0..20), global orient excluded"},
            "variants": agg,
            "per_joint": {v: {"joints": _JOINT_NAMES, **per_joint[v]} for v in _VARIANTS},
            "hypothesis_confirmed": bool(confirms),
            "per_scene": [{"scene": r["scene"], "n_segments": r["n_segments"]} for r in rows],
        }, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
