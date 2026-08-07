"""Is per-view noise SCALE heterogeneous, i.e. is there anything to weight? — RICH.

QUESTION
--------
Three per-joint quality signals have now measured at chance (confidence, 2D
reprojection residual, mesh occlusion — [[per-joint-view-signals-exhausted]]).
The explanation so far has been that the oracle's headroom comes from noise
REALISATIONS (which view's n_k happens to point against the shared bias b), and
that is unobservable because b cancels out of every inter-view comparison.

But that argument quietly skips a separate question. Every one of those signals
is really a proxy for the noise MAGNITUDE ||n_k||. If the per-view noise SCALE
sigma_k genuinely differs between cameras — some views reliably worse for a given
joint — then inverse-variance weighting beats uniform, and that IS reachable,
because sigma_k is a PROPERTY (predictable from geometry) rather than a
REALISATION (predictable by nobody).

    homogeneous sigma_k  -> views are exchangeable, uniform weighting is already
                            optimal, and NO magnitude-based signal can ever help.
                            The whole per-joint weighting line closes with a
                            mechanism instead of three null results.
    heterogeneous sigma_k -> real structure exists and our three signals were
                            merely too weak. The search reopens.

WHAT IS MEASURED
----------------
An ORACLE inverse-variance weighting — the ceiling for every magnitude-based
signal, present or future:

    sigma_k,j estimated per (scene, camera, joint) on EVEN frames
    w_k,j = 1 / sigma_k,j^2   applied on ODD frames

Even/odd rather than a temporal split: the subject moves, so first-half /
second-half would introduce distribution shift on top of the effect being
measured. Estimating on held-out frames keeps it honest — using the same frames
would let each camera's weight absorb its own noise realisation, which is exactly
the unreachable thing this experiment is trying to exclude.

Contrast with the OTHER oracle, deliberately reported alongside:

    oracle A (noise luck)      picks argmin_k ||xi_k|| per slot, needs GT.
                               UNREACHABLE by construction.
    oracle B (inverse variance) needs only a per-camera-per-joint scale.
                               REACHABLE in principle.

If B ~ 0 while A is large, then every magnitude signal is chasing something that
does not exist, and the failures of the three probes are fully explained.

Heterogeneity is also reported directly: the coefficient of variation of
sigma_k,j across cameras, against a PERMUTATION baseline (camera labels shuffled
across slots). Finite samples make sigma_hat vary even when the truth is common,
so the shuffled baseline — not zero — is what "no heterogeneity" looks like.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/inverse_variance_ceiling_rich.py \
        --split test --max_scenes 52 --device cuda
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


def so3_log(R: torch.Tensor) -> torch.Tensor:
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (th / (2 * torch.sin(th).clamp(min=1e-7)))


def weighted_chordal(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M = (R * w[..., None, None]).sum(dim=-3) / w.sum(dim=-1).clamp(min=1e-8)[..., None, None]
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


def geodesic_median(R_k, w0, iters: int = 5, eps: float = 1e-3):
    M = weighted_chordal(R_k, w0)
    for _ in range(iters):
        d = so3_log(M[..., None, :, :].transpose(-1, -2) @ R_k).norm(dim=-1)
        M = weighted_chordal(R_k, w0 / (d + eps))
    return M


def _fk(pose_full, betas, device, chunk: int = 32):
    outs = []
    for t0 in range(0, pose_full.shape[1], chunk):
        t1 = min(t0 + chunk, pose_full.shape[1])
        j = get_smplx_joints(pose_full[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :55, :]
        outs.append(j.detach().cpu())
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
    ap.add_argument("--min_cams", type=int, default=3)
    ap.add_argument("--fk_stride", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/inverse_variance_ceiling_rich.json"))
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

    METHODS = ["uniform", "geo_median", "invvar", "invvar_soft", "invvar_clip4",
               "invvar x median", "oracle_A_noise_luck"]
    acc: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    cv_obs, cv_perm = [], []
    rng = np.random.default_rng(0)

    for dp in dps:
        scene = dp.scene_dir.name
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
            J, Kc, P_n = R_k.shape[3], R_k.shape[4], R_k.shape[2]
            vis = pmask.permute(0, 1, 3, 2)[..., None, :].expand(1, T, P_n, J, Kc)
            m = vis > 0
            if gt_valid is not None:
                m = m & gt_valid[..., None, None].expand_as(m)
            ok = m.sum(-1) >= args.min_cams
            if not bool(ok.any()):
                continue

            xi = so3_log(R_gt[..., None, :, :].transpose(-1, -2) @ R_k)  # (1,T,P,J,K,3)
            n_res = xi - (xi * m[..., None].float()).sum(-2, keepdim=True) / \
                m.sum(-1).clamp(min=1)[..., None, None].float()
            sq = (n_res.pow(2).sum(-1) * m.float())                      # (1,T,P,J,K)

            # ── sigma_k,j estimated on EVEN frames only ─────────────────────
            ev = torch.zeros(T, dtype=torch.bool); ev[0::2] = True
            od = ~ev
            # (1,T_ev,P,J,K) summed over time and person -> (J,K): one variance
            # per (camera, joint) for this scene.
            cnt_e = (m[:, ev].float() * ok[:, ev][..., None].float()).sum((1, 2))[0]
            s2 = (sq[:, ev] * ok[:, ev][..., None].float()).sum((1, 2))[0] / cnt_e.clamp(min=1)
            s2 = torch.where(cnt_e >= 8, s2, s2.mean(-1, keepdim=True).expand_as(s2))
            s2 = s2.clamp(min=1e-8)                                      # (J,K)

            # heterogeneity: CV of sigma across cameras, per joint, vs a
            # permutation baseline (camera labels shuffled across slots).
            sg = s2.sqrt().numpy()
            cv_obs.append((sg.std(1) / np.maximum(sg.mean(1), 1e-9)))
            sh = sq[:, ev].reshape(-1, Kc).numpy()
            mh = m[:, ev].reshape(-1, Kc).numpy()
            keep = mh.all(1)
            if keep.sum() > 16:
                A = sh[keep]
                for r in range(A.shape[0]):
                    A[r] = A[r][rng.permutation(Kc)]
                sg_p = np.sqrt(A.mean(0))
                cv_perm.append(np.array([sg_p.std() / max(sg_p.mean(), 1e-9)]))

            # ── evaluate on ODD frames ─────────────────────────────────────
            # Odd frames FIRST, then stride — striding first would select even
            # indices and the odd-frame filter would empty it.
            t_idx = torch.arange(T)[od][::args.fk_stride]
            if len(t_idx) < 4:
                continue
            ok_s = ok[:, t_idx][..., 0]
            if not bool(ok_s.any()):
                continue

            w_uni = m[:, t_idx].float()
            w_iv = w_uni / s2[None, None, None]          # (J,K) broadcasts over (1,T,P,J,K)
            # Two softer variants, to separate "inverse-variance is the wrong
            # instrument" from "the weights are merely too extreme". 1/sigma^2 with
            # sigma estimated from finite data can hand one camera an enormous
            # weight; if that alone were the problem, these would recover.
            s2m = s2.median(dim=-1, keepdim=True).values
            w_soft = w_uni / (s2 + s2m)[None, None, None]
            r = (s2m / s2).clamp(max=4.0, min=0.25)      # cap the weight ratio at 4:1
            w_clip = w_uni * r[None, None, None]
            err = so3_log(R_gt[:, t_idx][..., None, :, :].transpose(-1, -2)
                          @ R_k[:, t_idx]).norm(dim=-1)
            w_or = w_uni / (err ** 2 + 1e-4)

            gt_root = targets["pose"][:T][None][:, t_idx, :, :1, :].float()
            betas = targets["shape"][:T][None][:, t_idx].float()
            J_gt = _fk(torch.cat([gt_root, gt[..., 1:, :][:, t_idx]], dim=3),
                       betas, args.device)

            fused = {
                "uniform": weighted_chordal(R_k[:, t_idx], w_uni),
                "geo_median": geodesic_median(R_k[:, t_idx], w_uni),
                "invvar": weighted_chordal(R_k[:, t_idx], w_iv),
                "invvar_soft": weighted_chordal(R_k[:, t_idx], w_soft),
                "invvar_clip4": weighted_chordal(R_k[:, t_idx], w_clip),
                "invvar x median": geodesic_median(R_k[:, t_idx], w_iv),
                "oracle_A_noise_luck": weighted_chordal(R_k[:, t_idx], w_or),
            }
            for nm, R_bar in fused.items():
                p_full = torch.cat([gt_root, matrix_to_sixd(R_bar)], dim=3)
                Jp = _fk(p_full, betas, args.device)
                d = torch.linalg.norm((Jp - Jp[..., :1, :]) - (J_gt - J_gt[..., :1, :]), dim=-1)
                acc[nm].append((d[..., :22][ok_s] * 1000.0).mean(-1).numpy())

        logger.info(f"{scene}: uni={np.mean(acc['uniform'][-1]):.1f} "
                    f"med={np.mean(acc['geo_median'][-1]):.1f} "
                    f"iv={np.mean(acc['invvar'][-1]):.1f} mm")

    if not acc["uniform"]:
        raise SystemExit("no valid slots")

    mm = {k: float(np.concatenate(v).mean()) for k, v in acc.items() if v}
    u = mm["uniform"]
    CVo = float(np.concatenate(cv_obs).mean())
    CVp = float(np.concatenate(cv_perm).mean()) if cv_perm else float("nan")

    print(f"\n{'='*74}\nIS PER-VIEW NOISE SCALE HETEROGENEOUS? — RICH {args.split}, "
          f"{len(acc['uniform'])} scenes\n{'='*74}")
    print(f"\nHETEROGENEITY of sigma_k across cameras (per joint)")
    print(f"   observed CV      {CVo:.4f}")
    print(f"   permuted CV      {CVp:.4f}   <- what 'no heterogeneity' looks like")
    print(f"   excess           {CVo - CVp:+.4f}")

    print(f"\nEND-TO-END RR-MPJPE on HELD-OUT (odd) frames")
    for k in METHODS:
        if k in mm:
            print(f"   {k:<22} {mm[k]:7.2f} mm  {mm[k]-u:+6.2f}")

    iv = mm.get("invvar", float("nan"))
    print(f"\nVERDICT")
    print(f"   oracle B (inverse variance, REACHABLE)  {iv-u:+.2f} mm")
    print(f"   oracle A (noise luck, UNREACHABLE)      "
          f"{mm['oracle_A_noise_luck']-u:+.2f} mm")
    print(f"   -> if B ~ 0 while A is large, views are exchangeable in SCALE and "
          f"no magnitude-based\n      signal can ever help; the three at-chance "
          f"probes are then fully explained.")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split, "n_scenes": len(acc["uniform"]),
        "rr_mpjpe_mm": mm, "cv_observed": CVo, "cv_permuted": CVp,
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
