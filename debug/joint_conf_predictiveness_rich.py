"""Does per-joint confidence predict per-camera error? — RICH.

QUESTION
--------
The oracle ceiling experiment (debug/view_weighting_ceiling_rich.py) established:

    uniform chordal mean          48.5 mm RR-MPJPE   (what the pipeline ships)
    oracle PER-FRAME weighting    48.5 mm  (-0.0)    <- per-frame features are USELESS
    oracle PER-JOINT weighting    42.8 mm  (-5.7)    <- all the headroom is here
    oracle PER-JOINT selection    39.4 mm  (-9.1)

So a 5.7 mm win exists, but ONLY for a signal that varies per (camera, joint) —
a camera is not uniformly good, it is occluded for one limb while being the best
view of another. Exactly one such signal exists on disk: SAM3D's
`pred_joint_confidence`, which the dataset loads into `joint_mask` (T,K,P,J).

Its previous verdict — fed once, hurt, switched off — is suspect: at that time
hand confidences were misaligned by three slots (canonical vs packed layout) and
the model being evaluated had been trained under a train/eval asymmetry.

WHAT IS MEASURED
----------------
For every (frame, person, joint) slot seen by >= min_cams cameras, over the K
observing cameras:

  1. Spearman correlation between confidence and geodesic error to GT, computed
     WITHIN the slot (across cameras) and then averaged. This is the quantity
     that matters: weighting only needs to rank cameras correctly for THIS joint
     at THIS frame, not to predict absolute error.
  2. The same correlation pooled globally, and per joint group.
  3. Top-1 agreement: how often the highest-confidence camera is also the
     lowest-error camera, against the 1/K chance rate.
  4. END-TO-END: RR-MPJPE in mm of a chordal mean weighted by confidence
     (w = c^p for several exponents), against uniform and against the oracles.
     This is the actual answer — correlation is diagnostic, mm is the verdict.

CONFIRMS a route to the headroom: positive correlation, above-chance top-1, and
confidence-weighted mm below uniform.
REFUTES it: correlation ~0 and confidence-weighted mm >= uniform. The headroom
is then real but unreachable with the information available, and the chordal
mean is the right fusion rule — the question closes for good.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/joint_conf_predictiveness_rich.py \
        --max_scenes 12 --device cuda
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

# The dataset only materialises `joint_mask` when this flag is on (it is off in
# production so training stays symmetric with evaluation). It is read at call
# time, so flipping it here turns the channel on for THIS diagnostic only —
# nothing on disk or in the repo changes.
from configuration import CONFIG as _CONFIG
_CONFIG.fusion.use_joint_confidence = True

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

_GROUPS = {"body": slice(0, 21), "hands": slice(21, 51), "face": slice(51, 54)}


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    rel = Ra @ Rb.transpose(-1, -2)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.arccos(cos))


def weighted_chordal(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M = (R * w[..., None, None]).sum(dim=-3) / w.sum(dim=-1).clamp(min=1e-8)[..., None, None]
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


def _fk_chunks(pose_full, betas, device, chunk: int = 32):
    outs = []
    for t0 in range(0, pose_full.shape[1], chunk):
        t1 = min(t0 + chunk, pose_full.shape[1])
        j = get_smplx_joints(pose_full[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :55, :]
        outs.append(j.detach().cpu())
    return torch.cat(outs, dim=1)


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Ranks along the last axis (ties broken by position — fine for Spearman)."""
    order = x.argsort(dim=-1)
    r = torch.empty_like(order)
    r.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device).expand_as(order))
    return r.float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=12)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="'test' evaluates the 52-scene RICH TEST split, the same "
                         "scenes every published number uses. 'train' uses the "
                         "training pool's val+train scenes.")
    ap.add_argument("--ghost_output_root", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"))
    ap.add_argument("--rich_data_root", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich/centered_test"))
    ap.add_argument("--rich_gt_dir", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich"))
    ap.add_argument("--body_split", default="test_body")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--min_cams", type=int, default=3,
                    help="within-slot ranking needs at least 3 cameras to mean anything")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fk_stride", type=int, default=2)
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/joint_conf_predictiveness_rich.json"))
    args = ap.parse_args()

    if args.split == "test":
        # Same construction as evaluation/temporal_smoothness_rich.py, so these
        # are exactly the scenes behind every published RICH number.
        from data.fusion_dataset import RICHFusionDatapoint
        scene_dirs = sorted(d for d in args.ghost_output_root.iterdir() if d.is_dir())
        scene_dirs = scene_dirs[:args.max_scenes]
        dps = []
        for sd in scene_dirs:
            try:
                dp = RICHFusionDatapoint(
                    scene_dir=sd, rich_data_root=args.rich_data_root,
                    rich_gt_dir=args.rich_gt_dir, body_split=args.body_split,
                    restrict_to_gt_persons=True)
                if dp.num_frames == 0 or not dp.has_gt:
                    logger.warning(f"  skip {sd.name}: no frames / no GT")
                    continue
                dps.append(dp)
                logger.info(f"  loaded {sd.name}")
            except Exception as e:
                logger.warning(f"  skip {sd.name}: {e}")
        ds = _tr.RICHFusionDataset(dps, augment=False)
        logger.info(f"TEST split: {len(dps)} scenes")
    else:
        scenes = [s for s in sorted(_tr.SCENES_ROOT.iterdir())
                  if s.is_dir() and s.name not in _tr.SKIP_SCENES]
        train_scenes, val_scenes = _tr._split_by_location(scenes, _tr.NUM_VAL_SCENES)
        dps = _tr.load_datapoints((val_scenes + train_scenes)[:args.max_scenes])
        ds = _tr.RICHFusionDataset(dps, augment=False)

    from torch.utils.data import DataLoader

    EXPONENTS = [1.0, 2.0, 4.0]
    MM = ["uniform", "oracle_inv"] + [f"conf^{p:g}" for p in EXPONENTS]
    acc_mm: dict[str, list[np.ndarray]] = {m: [] for m in MM}
    rho_group: dict[str, list[float]] = {g: [] for g in _GROUPS}
    rho_all, top1_hit, top1_n, conf_stats = [], 0, 0, []

    for inputs, targets in DataLoader(ds, batch_size=1):
        scene = targets.get("scene_name", ["?"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene
        if "joint_mask" not in inputs:
            logger.warning(f"{scene}: no joint_mask in batch — skipping")
            continue

        T = min(args.max_frames, inputs["pose"].shape[1])
        pose = inputs["pose"][:, :T].float()
        pmask = inputs["person_mask"][:, :T].float()
        jconf = inputs["joint_mask"][:, :T].float()          # (B,T,K,P,J) J=55
        gt = targets["pose"][:, :T].float()
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:, :T] if gt_valid is not None else None

        # Drop the root everywhere: it is the placer's job, not fusion's.
        pose, gt = pose[..., 1:, :], gt[..., 1:, :]
        jconf = jconf[..., 1:] if jconf.shape[-1] == 55 else jconf

        with torch.no_grad():
            R_k = sixd_to_matrix(pose).permute(0, 1, 3, 4, 2, 5, 6)   # (B,T,P,J,K,3,3)
            R_gt = sixd_to_matrix(gt)                                  # (B,T,P,J,3,3)
            vis = pmask.permute(0, 1, 3, 2)[..., None, :].expand(
                pmask.shape[0], T, pmask.shape[3], R_k.shape[3], pmask.shape[2])
            conf = jconf.permute(0, 1, 3, 4, 2)                        # (B,T,P,J,K)

            err = geodesic_deg(R_k, R_gt[..., None, :, :])             # (B,T,P,J,K)
            n_seen = (vis > 0).sum(-1)
            ok = n_seen >= args.min_cams
            if gt_valid is not None:
                ok = ok & gt_valid[..., None].expand_as(ok)
            if not bool(ok.any()):
                continue

            conf_stats.append(conf[vis > 0].numpy())

            # ── 1/2. Within-slot Spearman(conf, err) across cameras ──────────
            big = torch.tensor(1e6)
            e_r = _rank(torch.where(vis > 0, err, big))
            c_r = _rank(torch.where(vis > 0, conf, -big))
            m = (vis > 0).float()
            n = m.sum(-1, keepdim=True).clamp(min=1)
            ec = (e_r - (e_r * m).sum(-1, keepdim=True) / n) * m
            cc = (c_r - (c_r * m).sum(-1, keepdim=True) / n) * m
            denom = (ec.pow(2).sum(-1) * cc.pow(2).sum(-1)).sqrt()
            rho = torch.where(denom > 0, (ec * cc).sum(-1) / denom.clamp(min=1e-8),
                              torch.zeros_like(denom))                 # (B,T,P,J)

            rho_all.append(rho[ok].numpy())
            for g, sl in _GROUPS.items():
                v = rho[..., sl][ok[..., sl]]
                if v.numel():
                    rho_group[g].append(float(v.mean()))

            # ── 3. Top-1 agreement (body joints only) ────────────────────────
            body_ok = ok[..., _GROUPS["body"]]
            if bool(body_ok.any()):
                ci = torch.where(vis > 0, conf, -big)[..., _GROUPS["body"], :].argmax(-1)
                ei = torch.where(vis > 0, err, big)[..., _GROUPS["body"], :].argmin(-1)
                top1_hit += int((ci[body_ok] == ei[body_ok]).sum())
                top1_n += int(body_ok.sum())

            # ── 4. END-TO-END mm ─────────────────────────────────────────────
            t_idx = torch.arange(0, T, args.fk_stride)
            ok_s = ok[:, t_idx][..., 0]
            if not bool(ok_s.any()):
                continue
            gt_root = targets["pose"][:, :T][:, t_idx, :, :1, :].float()
            betas = targets["shape"][:, :T][:, t_idx].float()
            J_gt = _fk_chunks(torch.cat([gt_root, gt[:, t_idx]], dim=3), betas, args.device)

            weights = {"uniform": vis,
                       "oracle_inv": vis / (err ** 2 + 1.0)}
            for p in EXPONENTS:
                weights[f"conf^{p:g}"] = vis * conf.clamp(min=1e-4).pow(p)

            for name, w in weights.items():
                R_bar = weighted_chordal(R_k, w)
                p_full = torch.cat([gt_root, matrix_to_sixd(R_bar[:, t_idx])], dim=3)
                J_pred = _fk_chunks(p_full, betas, args.device)
                d = torch.linalg.norm(
                    (J_pred - J_pred[..., :1, :]) - (J_gt - J_gt[..., :1, :]), dim=-1)
                acc_mm[name].append((d[..., :22][ok_s] * 1000.0).mean(-1).numpy())

        logger.info(f"{scene}: rho={float(np.mean(rho_all[-1])):+.3f}  "
                    f"uniform={np.mean(acc_mm['uniform'][-1]):.1f}mm  "
                    f"conf^2={np.mean(acc_mm['conf^2'][-1]):.1f}mm")

    if not rho_all:
        raise SystemExit("no valid slots")

    rho_cat = np.concatenate(rho_all)
    cstat = np.concatenate(conf_stats)
    mm = {m: float(np.concatenate(acc_mm[m]).mean()) for m in MM if acc_mm[m]}

    print(f"\n{'='*70}\nDOES pred_joint_confidence PREDICT PER-CAMERA ERROR?  "
          f"({len(rho_cat):,} slots)\n{'='*70}")
    print(f"\nconfidence distribution: mean {cstat.mean():.3f}  median "
          f"{np.median(cstat):.3f}  exactly-zero {100*(cstat==0).mean():.1f}%  "
          f"exactly-one {100*(cstat==1).mean():.1f}%")
    print(f"\n1. Within-slot Spearman(confidence, error), averaged over slots")
    print(f"   NEGATIVE is what we want: high confidence <-> low error.")
    print(f"   all joints : {rho_cat.mean():+.4f}")
    for g in _GROUPS:
        if rho_group[g]:
            print(f"   {g:<11}: {np.mean(rho_group[g]):+.4f}")
    if top1_n:
        print(f"\n2. Top-1 agreement (body): highest-confidence camera is also the "
              f"lowest-error one\n   {100*top1_hit/top1_n:.1f}% of {top1_n:,} slots")
    print(f"\n3. END-TO-END root-relative MPJPE (root + 21 body joints) — THE VERDICT")
    u = mm.get("uniform", float('nan'))
    for m in MM:
        if m in mm:
            print(f"   {m:<12} {mm[m]:>7.1f} mm   {mm[m]-u:>+6.1f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "n_slots": int(len(rho_cat)),
        "spearman_all": float(rho_cat.mean()),
        "spearman_by_group": {g: float(np.mean(v)) for g, v in rho_group.items() if v},
        "top1_agreement": (top1_hit / top1_n) if top1_n else None,
        "rr_mpjpe_mm": mm,
        "conf_zero_frac": float((cstat == 0).mean()),
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
