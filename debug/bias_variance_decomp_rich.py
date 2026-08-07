"""How much of the per-view pose error is SHARED across cameras? — RICH.

QUESTION
--------
Fusion averages over the camera index. Write each camera's error in the tangent
space at ground truth,

    xi_k = log(R_gt^T R_k) = b + n_k,

where b is the component IDENTICAL in every view (the same network, with the same
training prior, applied to K images of the same person in the same pose — when
SAM3D under-rotates a hip it under-rotates it in every view at once) and n_k is
view-specific with variance sigma^2. Then for any aggregation rule

    E||mean_k xi_k||^2 = ||b||^2 + sigma^2 / K   ->   ||b||^2   as K -> infinity.

**||b|| is the exact fusion floor.** Not an oracle bound — the literal asymptote.
No aggregation rule, learned or not, reaches below it, because b carries no camera
index to average over. Fusion's entire budget is the sigma^2/K term.

This closes the one quantitative gap in the argument. What we had before was
"87% of the error survives an ORACLE that picks the best camera per joint"
(debug/view_weighting_ceiling_rich.py), which is suggestive but (a) phrased
through a hypothetical and (b) an oracle over SELECTIONS does not bound a method
that uses a PRIOR. This measures the floor directly instead.

METHOD — one-way random-effects decomposition (ANOVA) per slot
--------------------------------------------------------------
For every (frame, person, joint) seen by K >= 2 cameras:

    b_hat = (1/K) sum_k xi_k                              between-camera mean
    s2    = (1/(K-1)) sum_k ||xi_k - b_hat||^2            within-camera scatter,
                                                          unbiased for sigma^2
    ||b||^2_hat = ||b_hat||^2 - s2/K                      remove the noise that
                                                          leaks into b_hat

Per-slot ||b||^2 estimates are noisy and individually can go negative; they are
UNBIASED, so they are averaged over ~1M slots before any square root is taken.
Never clamp per slot — that would bias the result upward.

WHAT WOULD FALSIFY THE THESIS
-----------------------------
    ||b|| >> sigma   error is bias-dominated -> fusion is structurally capped,
                     the geodesic median is near-optimal, a learned module has
                     nothing to learn. This is the hypothesis, NOT yet measured.
    ||b|| ~ sigma    real room remains -> our negative signal results are the
                     failure, not the problem.
    ||b|| << sigma   fusion should work well -> the thesis is WRONG.

Consistency check reported alongside: E||xi_k||^2 must equal ||b||^2 + sigma^2.

KNOWN CONFOUND — read before quoting the number
-----------------------------------------------
Ground-truth error is VIEW-INDEPENDENT (one GT per frame, shared by all K
cameras), so it enters every xi_k identically — which is exactly the signature of
b. The measurement cannot separate them:

    ||b||_measured = ||b_SAM3D|| + ||eps_GT||

so this is an UPPER BOUND on the true estimator bias. That is why this is run on
RICH, whose GT is high-quality gendered SMPL-X fits. Do NOT run it on EgoExo4D to
support a bias-dominated claim: that GT is 2-view triangulated pseudo-GT and the
contamination would be large enough to manufacture the conclusion.

IDENTIFIABILITY LIMIT (stated, not tested — it cannot be tested)
----------------------------------------------------------------
With K samples per slot and one shared component, a positive correlation among
the n_k is mathematically INDISTINGUISHABLE from a larger b: both make the views
agree with each other more than independent noise would. So this decomposition
cannot separate "SAM3D has a shared bias" from "neighbouring viewpoints fail
alike". (An earlier version of this script "checked" it by correlating the
mean-subtracted residuals — that is vacuous: sum_k n_k = 0 by construction, so
the off-diagonal correlation is identically -1/(K-1) whatever the data does.)

This is CONSERVATIVE for the claim being made. If the n_k are positively
correlated, averaging reduces them by less than 1/K, so the true fusion floor is
HIGHER than the ||b|| reported here — the conclusion "fusion is structurally
capped" only gets stronger. The number is therefore a lower bound on the floor
and an upper bound on what fusion can win.

HEADLINE IS BODY-ONLY
---------------------
Aggregates are reported over the root + 21 body joints, matching RR-MPJPE and
every other number in this project. The all-54-joint figure is also printed but
is dominated by the 30 hand joints, which no metric here scores.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/bias_variance_decomp_rich.py \
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

_GROUPS = {"body": slice(0, 21), "hands": slice(21, 51), "face": slice(51, 54)}
_BODY_NAMES = [
    "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
    "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar",
    "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
    "r_wrist",
]


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,3) rotation vector, numerically safe near 0 and pi."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (th / (2 * torch.sin(th).clamp(min=1e-7)))


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
    ap.add_argument("--min_cams", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/bias_variance_decomp_rich.json"))
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

    J = 54
    # Sufficient statistics, accumulated per joint over all slots.
    n_slots = np.zeros(J)
    sum_b2 = np.zeros(J)      # ||b_hat||^2
    sum_s2 = np.zeros(J)      # s^2  (unbiased sigma^2 per slot)
    sum_s2_over_K = np.zeros(J)
    sum_xi2 = np.zeros(J)     # mean_k ||xi_k||^2   (single-view second moment)
    K_hist: dict[int, int] = {}

    for dp in dps:
        scene = dp.scene_dir.name
        inputs, targets = dp._build_sample()
        T = min(args.max_frames, inputs["pose"].shape[0])
        pose = inputs["pose"][:T].float()[None]           # (1,T,K,P,55,6)
        pmask = inputs["person_mask"][:T].float()[None]   # (1,T,K,P)
        gt = targets["pose"][:T].float()[None]
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:T][None] if gt_valid is not None else None

        with torch.no_grad():
            # Drop the root: placement is the placer's job, not fusion's.
            R_k = sixd_to_matrix(pose[..., 1:, :]).permute(0, 1, 3, 4, 2, 5, 6)
            R_gt = sixd_to_matrix(gt[..., 1:, :])                  # (1,T,P,J,3,3)
            Kc = R_k.shape[4]
            vis = pmask.permute(0, 1, 3, 2)[..., None, :].expand(
                1, T, pmask.shape[3], J, Kc)                       # (1,T,P,J,K)

            # xi_k = log(R_gt^T R_k), the per-view error in the tangent space at GT
            xi = so3_log(R_gt[..., None, :, :].transpose(-1, -2) @ R_k)  # (1,T,P,J,K,3)
            m = (vis > 0)
            if gt_valid is not None:
                m = m & gt_valid[..., None, None].expand_as(m)
            Kn = m.sum(-1)                                          # (1,T,P,J)
            ok = Kn >= args.min_cams
            if not bool(ok.any()):
                continue

            w = m.float()[..., None]                                # (1,T,P,J,K,1)
            Kf = Kn.clamp(min=1).float()[..., None]
            b_hat = (xi * w).sum(-2) / Kf                           # (1,T,P,J,3)
            resid = (xi - b_hat[..., None, :]) * w                  # (1,T,P,J,K,3)
            s2 = resid.pow(2).sum(-1).sum(-1) / (Kn - 1).clamp(min=1).float()
            xi2 = (xi.pow(2).sum(-1) * m.float()).sum(-1) / Kf[..., 0]

            for j in range(J):
                sel = ok[..., j]
                if not bool(sel.any()):
                    continue
                n_slots[j] += int(sel.sum())
                sum_b2[j] += float(b_hat[..., j, :].pow(2).sum(-1)[sel].sum())
                sum_s2[j] += float(s2[..., j][sel].sum())
                sum_s2_over_K[j] += float((s2[..., j] / Kf[..., j, 0])[sel].sum())
                sum_xi2[j] += float(xi2[..., j][sel].sum())
            for k, c in zip(*np.unique(Kn[ok].numpy(), return_counts=True)):
                K_hist[int(k)] = K_hist.get(int(k), 0) + int(c)

        logger.info(f"{scene}: {int(n_slots.sum()):,} slots so far")

    if n_slots.sum() == 0:
        raise SystemExit("no valid slots")

    deg = np.degrees(1.0)
    per_j = {}
    for j in range(J):
        if n_slots[j] == 0:
            continue
        mb2 = sum_b2[j] / n_slots[j]
        ms2 = sum_s2[j] / n_slots[j]
        mb2_corr = mb2 - sum_s2_over_K[j] / n_slots[j]     # unbiased ||b||^2
        per_j[j] = (mb2_corr, ms2, sum_xi2[j] / n_slots[j])

    def agg(js):
        n = n_slots[list(js)].sum()
        if n == 0:
            return float("nan"), float("nan"), float("nan"), 0.0
        b2 = float(sum(sum_b2[j] - sum_s2_over_K[j] for j in js) / n)
        s2 = float(sum(sum_s2[j] for j in js) / n)
        x2 = float(sum(sum_xi2[j] for j in js) / n)
        return b2, s2, x2, n

    BODY = range(21)
    B2, S2, X2, tot_n = agg(BODY)               # headline: what every metric scores
    B2a, S2a, _, _ = agg(range(J))              # all 54, hand-dominated
    b_deg = np.degrees(np.sqrt(max(B2, 0.0)))
    s_deg = np.degrees(np.sqrt(max(S2, 0.0)))
    share = B2 / (B2 + S2) if (B2 + S2) > 0 else float("nan")
    Kbar = sum(k * c for k, c in K_hist.items()) / max(sum(K_hist.values()), 1)

    print(f"\n{'='*74}\nHOW MUCH OF THE PER-VIEW ERROR IS SHARED ACROSS CAMERAS?"
          f"\nRICH {args.split}, {len(dps)} scenes, {int(tot_n):,} body-joint slots"
          f"\n(root + 21 body joints, matching RR-MPJPE)\n{'='*74}")
    print(f"\ncameras per slot: mean {Kbar:.2f}   "
          + "  ".join(f"K={k}:{100*c/sum(K_hist.values()):.0f}%"
                      for k, c in sorted(K_hist.items())))
    print(f"\n  ||b||  (shared bias)      {b_deg:7.3f} deg")
    print(f"  sigma  (view-specific)    {s_deg:7.3f} deg")
    print(f"  ratio  ||b|| / sigma      {b_deg/max(s_deg,1e-9):7.3f}")
    print(f"  SHARED FRACTION of per-view error variance   {100*share:5.1f}%")
    print(f"\n  consistency  E||xi_k||^2 = ||b||^2 + sigma^2 ?")
    print(f"     measured {np.degrees(np.sqrt(X2)):.3f} deg   "
          f"reconstructed {np.degrees(np.sqrt(B2+S2)):.3f} deg   "
          f"(rel. err {abs(X2-(B2+S2))/max(X2,1e-12):.2%})")
    print(f"\n  (all 54 joints incl. hands, for reference: ||b||="
          f"{np.degrees(np.sqrt(max(B2a,0))):.2f} deg  sigma="
          f"{np.degrees(np.sqrt(max(S2a,0))):.2f} deg)")
    print("\n  NOTE: correlated n_k are indistinguishable from a larger b, so this"
          "\n  ||b|| is a LOWER bound on the fusion floor — conservative for the claim.")

    print(f"\n  FUSION FLOOR: averaging K={Kbar:.1f} views gives "
          f"sqrt(||b||^2 + sigma^2/K) = "
          f"{np.degrees(np.sqrt(B2 + S2/Kbar)):.3f} deg;")
    print(f"  K -> infinity gives {b_deg:.3f} deg. Everything a fusion rule can "
          f"ever remove is the gap between them:")
    print(f"     {np.degrees(np.sqrt(B2 + S2/Kbar)) - b_deg:+.3f} deg "
          f"({100*(1 - b_deg/np.degrees(np.sqrt(B2+S2/Kbar))):.1f}% of the fused error)")

    print(f"\n  per body joint (deg):")
    print(f"    {'joint':<12} {'||b||':>8} {'sigma':>8} {'shared%':>8}")
    for j in range(21):
        if j not in per_j:
            continue
        mb2, ms2, _ = per_j[j]
        bj, sj = np.degrees(np.sqrt(max(mb2, 0))), np.degrees(np.sqrt(max(ms2, 0)))
        print(f"    {_BODY_NAMES[j]:<12} {bj:8.2f} {sj:8.2f} "
              f"{100*mb2/max(mb2+ms2,1e-12):7.1f}%")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split, "n_scenes": len(dps), "n_slots": int(tot_n),
        "mean_K": Kbar,
        "b_deg": b_deg, "sigma_deg": s_deg, "shared_fraction": share,
        "consistency_rel_err": abs(X2 - (B2 + S2)) / max(X2, 1e-12),
        "b_deg_all54": float(np.degrees(np.sqrt(max(B2a, 0.0)))),
        "fused_at_meanK_deg": float(np.degrees(np.sqrt(B2 + S2 / Kbar))),
        "per_joint": {_BODY_NAMES[j]: {
            "b_deg": float(np.degrees(np.sqrt(max(per_j[j][0], 0)))),
            "sigma_deg": float(np.degrees(np.sqrt(max(per_j[j][1], 0)))),
        } for j in range(21) if j in per_j},
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
