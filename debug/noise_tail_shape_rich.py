"""What SHAPE is the view-specific noise, and is the geodesic median optimal? — RICH.

QUESTION
--------
debug/bias_variance_decomp_rich.py split the per-view error into a shared bias b
(untouchable by fusion) and view-specific noise n_k. The estimator's ONLY job is
recovering the common component from K ~ 4 noisy samples, so its quality depends
entirely on the SHAPE of the n_k distribution.

We already know the shape is not Gaussian, and the proof needs no new experiment:
under Gaussian noise the median is asymptotically 2/pi ~ 0.64 as efficient as the
mean, so switching to it would COST ~0.3-0.4 mm. Measured: it GAINS 0.8 mm.
Therefore the tails are heavier than Gaussian.

What is NOT known is how MUCH heavier, and hence whether the geodesic median is
near-optimal or merely the best of the five estimators tried
([[fusion-saturated-ship-geodesic-median]]: mean 48.5, Karcher 48.6, median 47.8,
biweight 47.9, trim 48.1). This script answers that.

METHOD — non-parametric bootstrap on the REAL residuals
--------------------------------------------------------
No distribution is fitted and no parametric family assumed. The empirical
residuals themselves are resampled:

  1. xi_k = log(R_gt^T R_k) for every (frame, person, joint) with K >= min_cams
  2. n_k  = xi_k - mean_k(xi_k), pooled over all slots
  3. rescale by sqrt(K/(K-1)): subtracting the sample mean shrinks residuals by
     exactly that factor, so raw residuals understate the true noise
  4. draw K samples from the pool, estimate the location, record the error.
     Repeat many times.

Because the estimators compared are translation-equivariant and the spread is
small (median inter-camera 7.3 deg, comfortably in the tangent-space regime where
chordal and intrinsic averaging agree to 0.0 mm), this runs in R^3 on the residual
vectors with the true location at the origin: the error is just ||estimate||.

ESTIMATORS COMPARED
  mean                 optimal under Gaussian; the previous default
  geometric median     L1 / Weiszfeld — what ships
  Huber                tuned over a grid of thresholds
  Tukey biweight       tuned over a grid
  trimmed              drop the single worst sample
  ORACLE-TUNED         the best score achievable over every tuning grid point,
                       i.e. an upper bound on this whole estimator family

The gap between "geometric median" and "ORACLE-TUNED" is the answer: how much
is left on the table by shipping the median. Tail statistics (excess kurtosis,
quantile ratios, outlier fractions) are reported alongside to characterise WHY.

SCOPE — this is a regime measurement, not a theorem. It describes SAM3D on RICH
with K ~ 4 calibrated views. K enters directly (dropping 1 of 4 samples costs
sqrt(4/3) = +15.5% standard error; at K=10 it would cost only +5.4%), so the
"gentle beats hard" ordering is specific to sparse rigs.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/noise_tail_shape_rich.py \
        --split test --max_scenes 52 --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "human_body_prior"))

from fusion.fusion_module_v3 import sixd_to_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (th / (2 * torch.sin(th).clamp(min=1e-7)))


# ── estimators, all on (N, K, 3) batches of samples ─────────────────────────
def est_mean(X):
    return X.mean(1)


def est_geomedian(X, iters=8, eps=1e-9):
    """Weiszfeld: the R^3 analogue of the geodesic median that ships."""
    m = X.mean(1)
    for _ in range(iters):
        d = np.linalg.norm(X - m[:, None], axis=2)
        w = 1.0 / np.maximum(d, eps)
        m = (X * w[..., None]).sum(1) / w.sum(1)[:, None]
    return m


def est_huber(X, c, iters=8, eps=1e-9):
    m = X.mean(1)
    s = np.median(np.linalg.norm(X - m[:, None], axis=2), axis=1)[:, None]
    s = np.maximum(s, eps)
    for _ in range(iters):
        d = np.linalg.norm(X - m[:, None], axis=2) / s
        w = np.where(d <= c, 1.0, c / np.maximum(d, eps))
        m = (X * w[..., None]).sum(1) / w.sum(1)[:, None]
    return m


def est_biweight(X, c, iters=8, eps=1e-9):
    m = X.mean(1)
    s = np.median(np.linalg.norm(X - m[:, None], axis=2), axis=1)[:, None]
    s = np.maximum(s, eps)
    for _ in range(iters):
        d = np.linalg.norm(X - m[:, None], axis=2) / s
        w = np.where(d <= c, (1 - (d / c) ** 2) ** 2, 0.0)
        sw = w.sum(1)[:, None]
        upd = (X * w[..., None]).sum(1) / np.maximum(sw, eps)
        m = np.where(sw > eps, upd, m)
    return m


def est_trimmed(X):
    """Drop the single sample furthest from the current mean."""
    m = X.mean(1)
    d = np.linalg.norm(X - m[:, None], axis=2)
    worst = d.argmax(1)
    mask = np.ones(X.shape[:2], dtype=bool)
    mask[np.arange(X.shape[0]), worst] = False
    return (X * mask[..., None]).sum(1) / mask.sum(1)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=52)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--ghost_output_root", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"))
    ap.add_argument("--rich_data_root", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich/centered_test"))
    ap.add_argument("--rich_gt_dir", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich"))
    ap.add_argument("--body_split", default="test_body")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--min_cams", type=int, default=4,
                    help="fix K so the bootstrap matches the deployed rig size")
    ap.add_argument("--n_boot", type=int, default=200_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/noise_tail_shape_rich.json"))
    args = ap.parse_args()

    if args.split == "test":
        from data.fusion_dataset import RICHFusionDatapoint
        dps = []
        for sd in sorted(d for d in args.ghost_output_root.iterdir() if d.is_dir()):
            if len(dps) >= args.max_scenes:
                break
            try:
                dp = RICHFusionDatapoint(
                    scene_dir=sd, rich_data_root=args.rich_data_root,
                    rich_gt_dir=args.rich_gt_dir, body_split=args.body_split,
                    restrict_to_gt_persons=True)
                if dp.num_frames and dp.has_gt:
                    dps.append(dp)
            except Exception as e:
                logger.warning(f"  skip {sd.name}: {e}")
    else:
        scenes = [s for s in sorted(_tr.SCENES_ROOT.iterdir())
                  if s.is_dir() and s.name not in _tr.SKIP_SCENES]
        train_scenes, val_scenes = _tr._split_by_location(scenes, _tr.NUM_VAL_SCENES)
        dps = _tr.load_datapoints((val_scenes + train_scenes)[:args.max_scenes])
    logger.info(f"{len(dps)} scenes ({args.split})")

    pool = []          # residual vectors, rad
    spread = []        # per-slot inter-camera spread, deg — regime check

    for dp in dps:
        inputs, targets = dp._build_sample()
        T = min(args.max_frames, inputs["pose"].shape[0])
        pose = inputs["pose"][:T].float()[None]
        pmask = inputs["person_mask"][:T].float()[None]
        gt = targets["pose"][:T].float()[None]
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:T][None] if gt_valid is not None else None

        with torch.no_grad():
            R_k = sixd_to_matrix(pose[..., 1:, :]).permute(0, 1, 3, 4, 2, 5, 6)
            R_gt = sixd_to_matrix(gt[..., 1:, :])
            Kc, J = R_k.shape[4], R_k.shape[3]
            vis = pmask.permute(0, 1, 3, 2)[..., None, :].expand(
                1, T, pmask.shape[3], J, Kc)
            xi = so3_log(R_gt[..., None, :, :].transpose(-1, -2) @ R_k)  # (1,T,P,J,K,3)

            m = vis > 0
            if gt_valid is not None:
                m = m & gt_valid[..., None, None].expand_as(m)
            # Fix K exactly: mixing K would blend different shrinkage factors.
            full = m.all(-1) & (m.sum(-1) == args.min_cams)
            if not bool(full.any()):
                continue
            X = xi[full]                                   # (n, K, 3)
            n_res = X - X.mean(1, keepdim=True)
            # Undo the shrinkage from subtracting the sample mean.
            n_res = n_res * np.sqrt(args.min_cams / (args.min_cams - 1))
            pool.append(n_res.reshape(-1, 3).numpy())

            # regime check: mean pairwise angle between cameras, in degrees
            d = torch.linalg.norm(X[:, :, None] - X[:, None, :], dim=-1)
            iu = torch.triu_indices(args.min_cams, args.min_cams, offset=1)
            spread.append(torch.rad2deg(d[:, iu[0], iu[1]]).flatten().numpy())

        logger.info(f"{dp.scene_dir.name}: pool {sum(len(p) for p in pool):,}")

    if not pool:
        raise SystemExit("no full-K slots found")
    P = np.concatenate(pool).astype(np.float64)            # (M, 3) rad
    S = np.concatenate(spread)
    M = len(P)

    # ── tail statistics ─────────────────────────────────────────────────────
    comp = P.reshape(-1)
    comp = comp / comp.std()
    kurt = float((comp ** 4).mean() - 3.0)                 # excess kurtosis
    r = np.linalg.norm(P, axis=1)
    r_deg = np.degrees(r)
    q = {p: float(np.percentile(r_deg, p)) for p in (50, 90, 95, 99, 99.9)}
    sd = float(r_deg.std())
    frac3 = float((r_deg > 3 * r_deg.std()).mean())

    # ── bootstrap ───────────────────────────────────────────────────────────
    K = args.min_cams
    rng = np.random.default_rng(0)
    idx = rng.integers(0, M, size=(args.n_boot, K))
    X = P[idx]                                             # (n_boot, K, 3)

    HUBER_C = [0.5, 0.8, 1.0, 1.345, 1.8, 2.5]
    BIWT_C = [1.5, 2.0, 2.5, 3.0, 4.685, 6.0]
    results: dict[str, float] = {}

    def score(name, est):
        e = np.degrees(np.linalg.norm(est, axis=1))
        results[name] = float(np.sqrt((e ** 2).mean()))

    score("mean", est_mean(X))
    score("geometric median", est_geomedian(X))
    score("trimmed (drop worst)", est_trimmed(X))
    for c in HUBER_C:
        score(f"huber c={c}", est_huber(X, c))
    for c in BIWT_C:
        score(f"biweight c={c}", est_biweight(X, c))

    base = results["mean"]
    med = results["geometric median"]
    best_name = min(results, key=results.get)
    best = results[best_name]

    print(f"\n{'='*74}\nSHAPE OF THE VIEW-SPECIFIC NOISE, AND IS THE MEDIAN OPTIMAL?"
          f"\nRICH {args.split}, {len(dps)} scenes, K={K} exactly, "
          f"{M:,} residual vectors\n{'='*74}")

    print(f"\nREGIME CHECK  inter-camera spread (deg): "
          f"median {np.median(S):.2f}  p90 {np.percentile(S,90):.2f}  "
          f"p99 {np.percentile(S,99):.2f}")
    print(f"   (small-angle regime is what makes chordal ~ intrinsic; this is a "
          f"property of THIS rig, not a theorem)")

    print(f"\nTAIL SHAPE")
    print(f"   excess kurtosis of components   {kurt:8.2f}   (0 = Gaussian)")
    print(f"   ||n|| median / p90 / p99 (deg)  {q[50]:.2f} / {q[90]:.2f} / {q[99]:.2f}")
    print(f"   p99.9 (deg)                     {q[99.9]:.2f}")
    print(f"   p99 / median ratio              {q[99]/max(q[50],1e-9):8.2f}   "
          f"(Gaussian 3D would be ~2.4)")
    print(f"   fraction beyond 3 sd            {100*frac3:8.2f}%   (Gaussian ~0.3%)")

    print(f"\nESTIMATOR COMPARISON — RMS error recovering the location, K={K}")
    print(f"   {'estimator':<24} {'deg':>8} {'vs mean':>9}")
    for nm in sorted(results, key=results.get):
        print(f"   {nm:<24} {results[nm]:8.4f} {100*(results[nm]/base-1):+8.2f}%")

    print(f"\nVERDICT")
    print(f"   mean               {base:.4f} deg")
    print(f"   geometric median   {med:.4f} deg   ({100*(med/base-1):+.2f}% vs mean)")
    print(f"   best of family     {best:.4f} deg   ({best_name})")
    print(f"   LEFT ON THE TABLE by shipping the median: "
          f"{100*(med/best-1):+.2f}% of the noise term")
    print(f"   -> the noise term contributes ~3.3 mm of the 38.9 mm RR-MPJPE, so "
          f"this is worth ~{3.3*(med/best-1):+.2f} mm end-to-end")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split, "n_scenes": len(dps), "K": K, "n_residuals": M,
        "spread_deg": {"median": float(np.median(S)),
                       "p90": float(np.percentile(S, 90)),
                       "p99": float(np.percentile(S, 99))},
        "excess_kurtosis": kurt,
        "norm_quantiles_deg": q,
        "frac_beyond_3sd": frac3,
        "estimators_deg": results,
        "median_vs_best_pct": 100 * (med / best - 1),
        "best_estimator": best_name,
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
