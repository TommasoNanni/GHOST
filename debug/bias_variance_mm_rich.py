"""How many MILLIMETRES of the per-view error are shared across cameras? — RICH.

Position-space twin of debug/bias_variance_decomp_rich.py.

WHY A SECOND VERSION
--------------------
The rotation-space decomposition answered "how much of the per-view error is
shared" in DEGREES, averaging the 21 body joints equally. Our metric is
RR-MPJPE — MILLIMETRES of joint POSITION — and joints do not contribute equally
to it:

  * a hip rotating 1 deg swings the knee, ankle and foot: large mm effect;
  * a wrist rotating 1 deg moves only the hand, and hand joints are NOT among the
    22 scored: ~0 mm effect.

Worse, the two run OPPOSITE ways. Measured per-joint fusion budgets are largest
exactly where the metric weights most (hips 7.4%, knee 6.1%) and smallest where
it weights least (wrist 2.0%, spine2 0.6%). So the equal-weight angular average
(2.5%) UNDERSTATES the millimetre budget.

Rather than convert degrees to mm through lever arms — which needs a
linearisation — this runs the identical decomposition directly in position space:

    FK each camera's OWN pose -> joint positions p_k
    e_k = p_k - p_gt          (root-relative, per joint, in mm)
    e_k = B + N_k             B shared across cameras, N_k view-specific

The kinematic chain does the weighting itself, because FK is exactly what turns
rotations into the positions the metric scores. No approximation.

    fused error at K views = sqrt(||B||^2 + sigma^2/K)  ->  ||B||  as K -> inf

so ||B|| is the fusion floor IN MILLIMETRES, directly comparable to the 38.9 mm
uniform-mean RR-MPJPE and to the 0.8 mm the geodesic median wins over it.

PROTOCOL — identical to every other number in this project: GT root and GT betas
are supplied so ONLY the body pose differs between cameras.

ESTIMATOR — one-way random-effects decomposition per (frame, person, joint):

    B_hat = (1/K) sum_k e_k
    s2    = (1/(K-1)) sum_k ||e_k - B_hat||^2          unbiased for sigma^2
    ||B||^2_hat = ||B_hat||^2 - s2/K                   remove noise leaking into B_hat

Per-slot ||B||^2 estimates are noisy and can go negative; they are UNBIASED, so
they are averaged over all slots before any square root. Never clamp per slot.

CAVEATS (both conservative, same as the rotation version)
  * GT error is view-independent, so it is indistinguishable from B: ||B|| is an
    UPPER bound on SAM3D's own shared error. Fine on RICH's SMPL-X GT; do not run
    this on EgoExo4D pseudo-GT to support a bias-dominated claim.
  * Correlated N_k is likewise indistinguishable from a larger B, so the true
    floor can only be HIGHER. Both push the conclusion the same way.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/bias_variance_mm_rich.py \
        --split test --max_scenes 52 --device cuda --fk_stride 4
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

from fusion.fusion_module_v3 import matrix_to_sixd, sixd_to_matrix
from utilities.smplx_utilities import get_smplx_joints

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

# The 22 scored joints: root + 21 body, exactly RR-MPJPE's set.
_N_SCORED = 22
_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2",
    "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar",
    "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
]


def fk_positions(pose6d: torch.Tensor, betas: torch.Tensor,
                 device, chunk: int = 16) -> torch.Tensor:
    """(B,T,P,55,6) + (B,T,P,10) -> (B,T,P,22,3) root-relative joint positions."""
    outs = []
    for t0 in range(0, pose6d.shape[1], chunk):
        t1 = min(t0 + chunk, pose6d.shape[1])
        j = get_smplx_joints(pose6d[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :_N_SCORED, :]
        outs.append((j - j[..., :1, :]).detach().cpu())
    return torch.cat(outs, dim=1)


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
    ap.add_argument("--fk_stride", type=int, default=4,
                    help="FK runs once PER CAMERA, so this is ~K times the cost "
                         "of the usual probes; 4 keeps it to tens of minutes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/bias_variance_mm_rich.json"))
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

    n_slots = np.zeros(_N_SCORED)
    sum_B2 = np.zeros(_N_SCORED)        # ||B_hat||^2
    sum_s2 = np.zeros(_N_SCORED)        # unbiased sigma^2
    sum_s2_over_K = np.zeros(_N_SCORED)
    sum_e2 = np.zeros(_N_SCORED)        # mean_k ||e_k||^2, single-view second moment
    sum_fused2 = np.zeros(_N_SCORED)    # ||mean_k e_k||^2 = the actual fused error
    K_hist: dict[int, int] = {}

    for dp in dps:
        scene = dp.scene_dir.name
        inputs, targets = dp._build_sample()
        T0 = min(args.max_frames, inputs["pose"].shape[0])
        t_idx = torch.arange(0, T0, args.fk_stride)
        T = len(t_idx)

        pose = inputs["pose"][:T0][t_idx].float()[None]        # (1,T,K,P,55,6)
        pmask = inputs["person_mask"][:T0][t_idx].float()[None]
        gt = targets["pose"][:T0][t_idx].float()[None]         # (1,T,P,55,6)
        betas = targets["shape"][:T0][t_idx].float()[None]     # (1,T,P,10)
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:T0][t_idx][None] if gt_valid is not None else None
        Kc, P_n = pose.shape[2], pose.shape[3]

        with torch.no_grad():
            # GT positions: GT root + GT body pose + GT betas.
            p_gt = fk_positions(gt, betas, args.device)         # (1,T,P,22,3)

            # Per-camera positions: GT root + GT betas, camera k's BODY pose only,
            # so nothing but the fused quantity differs between views.
            # All K cameras are folded into the person axis and FK'd in ONE pass —
            # a per-camera loop is ~K times slower for identical output.
            root = gt[..., :1, :][:, :, None].expand(1, T, Kc, P_n, 1, 6)
            pk = torch.cat([root, pose[..., 1:, :]], dim=4)      # (1,T,K,P,55,6)
            pk = pk.reshape(1, T, Kc * P_n, 55, 6)
            bk = betas[:, :, None].expand(1, T, Kc, P_n, 10).reshape(1, T, Kc * P_n, 10)
            p_k = fk_positions(pk, bk, args.device)              # (1,T,K*P,22,3)
            p_k = p_k.reshape(1, T, Kc, P_n, _N_SCORED, 3)
            e = (p_k - p_gt[:, :, None]).permute(0, 1, 3, 2, 4, 5) * 1000.0  # (1,T,P,K,22,3)

            vis = pmask.permute(0, 1, 3, 2)                     # (1,T,P,K)
            m = vis > 0
            if gt_valid is not None:
                m = m & gt_valid[..., None].expand_as(m)
            Kn = m.sum(-1)                                      # (1,T,P)
            ok = Kn >= args.min_cams
            if not bool(ok.any()):
                continue

            w = m.float()[..., None, None]                      # (1,T,P,K,1,1)
            Kf = Kn.clamp(min=1).float()[..., None, None]       # (1,T,P,1,1)
            B_hat = (e * w).sum(3) / Kf                         # (1,T,P,22,3)
            resid = (e - B_hat[:, :, :, None]) * w
            s2 = resid.pow(2).sum(-1).sum(3) / (Kn - 1).clamp(min=1).float()[..., None]
            e2 = (e.pow(2).sum(-1) * m.float()[..., None]).sum(3) / Kf[..., 0]

            B2 = B_hat.pow(2).sum(-1)                           # (1,T,P,22)
            for j in range(_N_SCORED):
                sel = ok
                if not bool(sel.any()):
                    continue
                n_slots[j] += int(sel.sum())
                sum_B2[j] += float(B2[..., j][sel].sum())
                sum_s2[j] += float(s2[..., j][sel].sum())
                sum_s2_over_K[j] += float((s2[..., j] / Kf[..., 0, 0])[sel].sum())
                sum_e2[j] += float(e2[..., j][sel].sum())
                sum_fused2[j] += float(B2[..., j][sel].sum())
            for k, c in zip(*np.unique(Kn[ok].numpy(), return_counts=True)):
                K_hist[int(k)] = K_hist.get(int(k), 0) + int(c)

        logger.info(f"{scene}: {int(n_slots.max()):,} slots so far")

    if n_slots.sum() == 0:
        raise SystemExit("no valid slots")

    # Per joint, then averaged over the 22 scored joints — which is exactly how
    # RR-MPJPE pools (mean over joints of the per-joint distance).
    per_j = {}
    for j in range(_N_SCORED):
        if n_slots[j] == 0:
            continue
        B2j = sum_B2[j] / n_slots[j] - sum_s2_over_K[j] / n_slots[j]
        s2j = sum_s2[j] / n_slots[j]
        e2j = sum_e2[j] / n_slots[j]
        fusedj = sum_fused2[j] / n_slots[j]
        per_j[j] = (max(B2j, 0.0), s2j, e2j, fusedj)

    Kbar = sum(k * c for k, c in K_hist.items()) / max(sum(K_hist.values()), 1)
    # RR-MPJPE pools as a mean of DISTANCES, so average the per-joint RMS values.
    floor_mm = float(np.mean([np.sqrt(per_j[j][0]) for j in per_j]))
    fused_mm = float(np.mean([np.sqrt(per_j[j][3]) for j in per_j]))
    single_mm = float(np.mean([np.sqrt(per_j[j][2]) for j in per_j]))
    sigma_mm = float(np.mean([np.sqrt(per_j[j][1]) for j in per_j]))
    # Variance-WEIGHTED shared fraction: sum the variances, then divide. An
    # unweighted mean of per-joint ratios is meaningless here, because pelvis /
    # hips / spine1 have exactly zero positional error under this protocol (GT
    # root + GT betas fix them regardless of body pose), so their ratio is 0/0
    # and flips on floating-point noise. Weighting by error also matches what the
    # metric cares about.
    _sb = float(sum(per_j[j][0] for j in per_j))
    _ss = float(sum(per_j[j][1] for j in per_j))
    shared = _sb / max(_sb + _ss, 1e-12)

    print(f"\n{'='*74}\nHOW MANY MILLIMETRES OF THE ERROR ARE SHARED ACROSS CAMERAS?"
          f"\nRICH {args.split}, {len(dps)} scenes, {int(n_slots.max()):,} slots, "
          f"root + 21 body joints\n{'='*74}")
    print(f"\ncameras per slot: mean {Kbar:.2f}   "
          + "  ".join(f"K={k}:{100*c/sum(K_hist.values()):.0f}%"
                      for k, c in sorted(K_hist.items())))
    print(f"\n  single view                       {single_mm:7.2f} mm")
    print(f"  fused over K={Kbar:.1f} (measured)      {fused_mm:7.2f} mm")
    print(f"  ||B||  FUSION FLOOR (K -> inf)    {floor_mm:7.2f} mm")
    print(f"  sigma  view-specific               {sigma_mm:7.2f} mm")
    print(f"  SHARED FRACTION of error variance  {100*shared:5.1f}%")
    # CAREFUL WITH THIS NUMBER. sqrt(||B||^2 + sigma^2/K) at the measured K is
    # what the PLAIN MEAN already achieves, so the gap below is what MORE CAMERAS
    # would buy — NOT what a smarter rule can win at fixed K. The mean is optimal
    # under Gaussian noise; a better rule only gains from heavy tails, which is
    # what the geodesic median exploits (~0.8 mm). And oracle per-joint weighting
    # (~4.5 mm) is a different thing again: it does not estimate B, it selects
    # views whose noise happens to cancel part of it, which measurably cannot be
    # predicted from any observable signal.
    print(f"\n  >>> gap from K={Kbar:.1f} to K=inf: {fused_mm - floor_mm:.2f} mm "
          f"({100*(1-floor_mm/max(fused_mm,1e-9)):.1f}% of the fused error)")
    print(f"      = what MORE CAMERAS buy, not what a better rule buys.")
    print(f"  >>> 1 view -> K={Kbar:.1f} views: {single_mm - fused_mm:.2f} mm "
          f"({100*(1-fused_mm/max(single_mm,1e-9)):.1f}%)  <- fusion doing its job")
    print(f"      choosing the best RULE instead of the mean is worth ~0.8 mm.")

    print(f"\n  per joint (mm):")
    print(f"    {'joint':<12} {'||B||':>8} {'sigma':>8} {'fused':>8} "
          f"{'budget':>8} {'shared%':>8}")
    for j in range(_N_SCORED):
        if j not in per_j:
            continue
        B2j, s2j, _, fj = per_j[j]
        b, s, f = np.sqrt(B2j), np.sqrt(s2j), np.sqrt(fj)
        print(f"    {_NAMES[j]:<12} {b:8.2f} {s:8.2f} {f:8.2f} {f-b:8.2f} "
              f"{100*B2j/max(B2j+s2j,1e-12):7.1f}%")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split, "n_scenes": len(dps), "mean_K": Kbar,
        "single_view_mm": single_mm, "fused_mm": fused_mm,
        "floor_mm": floor_mm, "sigma_mm": sigma_mm,
        "shared_fraction": shared,
        "budget_mm": fused_mm - floor_mm,
        "budget_frac_of_fused": 1 - floor_mm / max(fused_mm, 1e-9),
        "per_joint": {_NAMES[j]: {
            "B_mm": float(np.sqrt(per_j[j][0])),
            "sigma_mm": float(np.sqrt(per_j[j][1])),
            "fused_mm": float(np.sqrt(per_j[j][3])),
        } for j in per_j},
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
