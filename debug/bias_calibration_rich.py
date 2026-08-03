"""Can the SHARED bias be calibrated away? — RICH. (Step 5)

QUESTION
--------
Steps 1-4 bound FUSION: any rule that combines the K per-camera estimates is
capped, and the geodesic median is within ~0.07 mm of that cap
([[inverse-variance-ceiling]]). But all of that concerns the sigma term. The
SHARED bias b — 55% of the error variance, a 42.5 mm floor — is untouched,
because b carries no camera index to average over.

A learned model with a PRIOR is not bound by that cap: it can emit rotations
outside the span of its inputs, so in principle it can correct b itself. That is
not fusion, it is CALIBRATION, and it is the only idea that attacks the 42.5 mm
rather than the ~4 mm. It is also the one thing v2/R2/v3 could have learned and
evidently did not — v3 started bit-exact at the chordal mean and trained AWAY
from it ([[fusion-saturated-ship-geodesic-median]]).

Now that b has actually been measured per joint (8.8-31 deg,
[[fusion-bias-variance-decomposition]]), the question is finally testable, and it
reduces to: **is b a stable offset, or pose-dependent noise that merely has a
non-zero average?**

METHOD
------
xi_k = log(R_gt^T R_k) so R_k ~ R_gt exp(xi_k), and the fused estimate satisfies
R_bar ~ R_gt exp(b). Therefore

    R_gt ~ R_bar exp(-b)          ->    R_corrected = R_bar exp(-alpha * b_j)

which needs only the fused rotation at inference — no ground truth. b_j lives in
the joint's own body frame (it comes from R_gt^T R_k), which is the frame a
systematic estimator bias would be constant in.

  * b_j estimated on TRAIN scenes, applied to TEST. 54 joints x 3 = 162 numbers.
  * alpha in {1.0, 0.5} — a partial correction guards against overshoot if the
    estimate is noisy.
  * ORACLE variant: b_j estimated on TEST itself. This separates "the calibration
    does not transfer" from "a constant per-joint offset is inherently too weak a
    model". Without it a null result is uninterpretable.

THE DECISIVE DIAGNOSTIC is cheaper than the end-to-end number: correlate
b_j(train) against b_j(test). If they agree, the bias is stable and calibration
must work; if they are unrelated, b is pose-dependent and the 42.5 mm floor is
genuinely SAM3D's.

PRIOR EXPECTATION: it will not transfer, because three trained models with far
more capacity than 162 parameters failed to find it. Worth running anyway — it is
cheap, it is the last untested item, and a 162-parameter calibration beating a
1.1M-parameter transformer would be a strong result if it did work.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/bias_calibration_rich.py \
        --max_train 20 --max_scenes 52 --device cuda
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

_BODY = [
    "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
    "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar",
    "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
]


def so3_log(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (th / (2 * torch.sin(th).clamp(min=1e-7)))


def _skew(v):
    O = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    O[..., 0, 1], O[..., 0, 2] = -v[..., 2], v[..., 1]
    O[..., 1, 0], O[..., 1, 2] = v[..., 2], -v[..., 0]
    O[..., 2, 0], O[..., 2, 1] = -v[..., 1], v[..., 0]
    return O


def so3_exp(v):
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    K = _skew(v / th)
    th = th[..., None]
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def weighted_chordal(R, w):
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


def _load(split, args):
    if split == "test":
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
        return dps
    scenes = [s for s in sorted(_tr.SCENES_ROOT.iterdir())
              if s.is_dir() and s.name not in _tr.SKIP_SCENES]
    train_scenes, val_scenes = _tr._split_by_location(scenes, _tr.NUM_VAL_SCENES)
    return _tr.load_datapoints((train_scenes + val_scenes)[:args.max_train])


def accumulate_bias(dps, max_frames, min_cams, J=54):
    """Mean of b_hat = mean_k log(R_gt^T R_k) per joint, over all slots."""
    tot = torch.zeros(J, 3, dtype=torch.float64)
    n = torch.zeros(J, dtype=torch.float64)
    for dp in dps:
        inputs, targets = dp._build_sample()
        T = min(max_frames, inputs["pose"].shape[0])
        pose = inputs["pose"][:T].float()[None]
        pmask = inputs["person_mask"][:T].float()[None]
        gt = targets["pose"][:T].float()[None]
        gv = targets.get("gt_valid")
        gv = gv[:T][None] if gv is not None else None
        with torch.no_grad():
            R_k = sixd_to_matrix(pose[..., 1:, :]).permute(0, 1, 3, 4, 2, 5, 6)
            R_gt = sixd_to_matrix(gt[..., 1:, :])
            Kc, P_n = R_k.shape[4], R_k.shape[2]
            m = pmask.permute(0, 1, 3, 2)[..., None, :].expand(1, T, P_n, J, Kc) > 0
            if gv is not None:
                m = m & gv[..., None, None].expand_as(m)
            ok = m.sum(-1) >= min_cams
            if not bool(ok.any()):
                continue
            xi = so3_log(R_gt[..., None, :, :].transpose(-1, -2) @ R_k)
            b = (xi * m[..., None].float()).sum(-2) / m.sum(-1).clamp(min=1)[..., None].float()
            for j in range(J):
                sel = ok[..., j]
                if bool(sel.any()):
                    tot[j] += b[..., j, :][sel].sum(0).double()
                    n[j] += int(sel.sum())
        logger.info(f"  bias from {dp.scene_dir.name}")
    return (tot / n.clamp(min=1)[:, None]).float(), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=52)
    ap.add_argument("--max_train", type=int, default=20)
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
                    default=Path("eval_explainability/bias_calibration_rich.json"))
    args = ap.parse_args()

    logger.info("estimating b on TRAIN scenes...")
    tr_dps = _load("train", args)
    b_train, n_tr = accumulate_bias(tr_dps, args.max_frames, args.min_cams)
    del tr_dps

    logger.info("loading TEST scenes...")
    te_dps = _load("test", args)
    b_test, _ = accumulate_bias(te_dps, args.max_frames, args.min_cams)

    # ── the decisive diagnostic ─────────────────────────────────────────────
    bt, be = b_train.numpy(), b_test.numpy()
    nt, ne = np.linalg.norm(bt, axis=1), np.linalg.norm(be, axis=1)
    cos = (bt * be).sum(1) / np.maximum(nt * ne, 1e-9)
    flat_c = float(np.corrcoef(bt.reshape(-1), be.reshape(-1))[0, 1])

    METHODS = ["uniform", "geo_median", "median+calib(train,a=1)",
               "median+calib(train,a=0.5)", "median+calib(ORACLE test)"]
    acc: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}

    for dp in te_dps:
        inputs, targets = dp._build_sample()
        T = min(args.max_frames, inputs["pose"].shape[0])
        pose = inputs["pose"][:T].float()[None]
        pmask = inputs["person_mask"][:T].float()[None]
        gt = targets["pose"][:T].float()[None]
        gv = targets.get("gt_valid")
        gv = gv[:T][None] if gv is not None else None
        with torch.no_grad():
            R_k = sixd_to_matrix(pose[..., 1:, :]).permute(0, 1, 3, 4, 2, 5, 6)
            J, Kc, P_n = R_k.shape[3], R_k.shape[4], R_k.shape[2]
            m = pmask.permute(0, 1, 3, 2)[..., None, :].expand(1, T, P_n, J, Kc) > 0
            if gv is not None:
                m = m & gv[..., None, None].expand_as(m)
            ok = m.sum(-1) >= args.min_cams
            if not bool(ok.any()):
                continue
            t_idx = torch.arange(0, T, args.fk_stride)
            ok_s = ok[:, t_idx][..., 0]
            if not bool(ok_s.any()):
                continue
            w = m[:, t_idx].float()
            R_med = geodesic_median(R_k[:, t_idx], w)
            R_uni = weighted_chordal(R_k[:, t_idx], w)

            gt_root = targets["pose"][:T][None][:, t_idx, :, :1, :].float()
            betas = targets["shape"][:T][None][:, t_idx].float()
            J_gt = _fk(torch.cat([gt_root, gt[..., 1:, :][:, t_idx]], dim=3),
                       betas, args.device)

            def corrected(bvec, a):
                # R_gt ~ R_bar exp(-b): right-multiply the fused rotation.
                E = so3_exp(-a * bvec)[None, None, None]      # (1,1,1,J,3,3)
                return R_med @ E

            fused = {
                "uniform": R_uni,
                "geo_median": R_med,
                "median+calib(train,a=1)": corrected(b_train, 1.0),
                "median+calib(train,a=0.5)": corrected(b_train, 0.5),
                "median+calib(ORACLE test)": corrected(b_test, 1.0),
            }
            for nm, R_bar in fused.items():
                p_full = torch.cat([gt_root, matrix_to_sixd(R_bar)], dim=3)
                Jp = _fk(p_full, betas, args.device)
                d = torch.linalg.norm((Jp - Jp[..., :1, :]) - (J_gt - J_gt[..., :1, :]), dim=-1)
                acc[nm].append((d[..., :22][ok_s] * 1000.0).mean(-1).numpy())
        logger.info(f"{dp.scene_dir.name}: med={np.mean(acc['geo_median'][-1]):.1f} "
                    f"calib={np.mean(acc['median+calib(train,a=1)'][-1]):.1f} mm")

    if not acc["uniform"]:
        raise SystemExit("no valid slots")
    mm = {k: float(np.concatenate(v).mean()) for k, v in acc.items() if v}
    u = mm["uniform"]

    print(f"\n{'='*74}\nCAN THE SHARED BIAS BE CALIBRATED AWAY?  RICH: "
          f"b fitted on <= {args.max_train} TRAIN scenes, applied to "
          f"{len(acc['uniform'])} TEST scenes\n{'='*74}")
    print(f"\nDOES b TRANSFER?  (the decisive diagnostic)")
    print(f"   correlation of b(train) vs b(test), all 162 numbers: {flat_c:+.4f}")
    print(f"   {'joint':<12} {'|b| train':>10} {'|b| test':>10} {'cos':>8}")
    for j in range(21):
        print(f"   {_BODY[j]:<12} {np.degrees(nt[j]):10.2f} {np.degrees(ne[j]):10.2f} "
              f"{cos[j]:8.3f}")
    print(f"   {'MEAN(body)':<12} {np.degrees(nt[:21]).mean():10.2f} "
          f"{np.degrees(ne[:21]).mean():10.2f} {cos[:21].mean():8.3f}")

    print(f"\nEND-TO-END RR-MPJPE (root + 21 body joints)")
    for k in METHODS:
        if k in mm:
            print(f"   {k:<28} {mm[k]:7.2f} mm  {mm[k]-u:+6.2f}")

    print(f"\nVERDICT")
    print(f"   a constant per-joint offset fitted on TEST itself is worth "
          f"{mm.get('median+calib(ORACLE test)', float('nan'))-mm['geo_median']:+.2f} mm "
          f"vs the median")
    print(f"   the same thing fitted on TRAIN transfers to "
          f"{mm.get('median+calib(train,a=1)', float('nan'))-mm['geo_median']:+.2f} mm")
    print(f"   -> if ORACLE ~ 0 the model is too weak; if ORACLE helps but TRAIN "
          f"does not, b is pose-dependent.")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "rr_mpjpe_mm": mm, "b_corr_flat": flat_c,
        "b_train_deg": {_BODY[j]: float(np.degrees(nt[j])) for j in range(21)},
        "b_test_deg": {_BODY[j]: float(np.degrees(ne[j])) for j in range(21)},
        "cos_per_joint": {_BODY[j]: float(cos[j]) for j in range(21)},
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
