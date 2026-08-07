"""Oracle ceiling for multi-view rotation weighting — RICH.

QUESTION
--------
Every trained fusion module is beaten by the parameter-free chordal mean of the
per-camera SAM3D estimates (RICH test: mean 47.6/67.7/26.5 vs v2 50.4/70.4/30.4
vs R2-converged 48.3/68.2/27.7). Before building anything that weights views by
predicted quality, answer the prior question:

    HOW MUCH IS ANY VIEW WEIGHTING WORTH AT ALL?

This uses ground truth to construct weights no real method could ever exceed, so
the answer is an upper bound on every possible weighting scheme — learned or
hand-designed, whatever features it consumes.

WHAT IS MEASURED
----------------
Per (frame, person, joint), over the K cameras that observe it, the geodesic
error to GT of:

  uniform    chordal mean, uniform weights            — what the pipeline ships
  oracle_best   the single camera closest to GT       — ceiling on hard selection
  oracle_inv    chordal mean, weights 1/(err^2+eps)   — ceiling on soft weighting
  oracle_topk   chordal mean over the best ceil(K/2) cameras
  perframe_inv  soft weighting CONSTRAINED to one weight per (frame, camera):
                the per-joint errors are averaged before inverting, so the
                weights cannot vary across joints. THIS is the ceiling any
                per-frame feature (bbox area, distance, truncation, detection
                score) could reach — those features have no per-joint resolution.
  perframe_best hard selection of one camera for the whole body, per frame

REALISABLE estimators — no ground truth, no new data. These use only the
per-camera DISAGREEMENT, i.e. exactly the information the fusion module already
consumes. They answer: can an outlier view be identified simply by the fact that
it disagrees with the others?

  geo_median    iteratively reweighted chordal mean (Weiszfeld on SO(3)):
                w_k = 1 / (geodesic distance to the current estimate)
  biweight      Tukey biweight M-estimator, c = 4.685 * MAD, hard-zeroing views
                beyond the cutoff — the standard robust redescending estimator
  trim1         drop the single view furthest from the chordal mean, average the
                rest (only when >= 3 views observe the joint)
  karcher       INTRINSIC (Riemannian) mean: argmin sum d_geo^2, by log/exp
                iteration. The chordal mean minimises the same objective in the
                EMBEDDING metric; they differ once the views are spread out.
  geo_median_i  INTRINSIC geodesic median: argmin sum d_geo, proper manifold
                Weiszfeld rather than reweighting a chordal solve.
  single_mean   mean over per-camera errors           — the "pick one at random"
                                                        reference, for context

plus the same table restricted to body joints (SMPL-X 1..21 == packed 0..20),
which is what every reported metric actually scores.

Each fused rotation set is then run through SMPL-X FK with GT betas and the GT
root orientation, and scored as ROOT-RELATIVE MPJPE in mm over root + 21 body
joints — the same quantity the trainer reports as RR-MPJPE and the space every
published number lives in. Degrees are reported too, but mm is the metric: a
degree costs very different position error at a hip than at a wrist.

INTERPRETATION
--------------
If oracle_inv is within ~1 deg of uniform, no feature set can rescue view
weighting and the mean is the right fusion rule — the question closes here.
If the gap is large, view weighting has real headroom and the next step is to
test whether any observable feature predicts per-camera error (Phase 2).

CAVEAT — rotation degrees, not mm. A degree at a proximal joint costs far more
position error than at a distal one, so the mm-equivalent of any gap depends on
which joints it sits on; the per-group breakdown is printed for that reason.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/view_weighting_ceiling_rich.py \
        --split train --max_scenes 12
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

_BODY = slice(0, 21)          # packed slots 0..20 == SMPL-X 1..21


def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Angle between two rotations, in degrees. Shapes broadcast."""
    rel = Ra @ Rb.transpose(-1, -2)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.arccos(cos))


def weighted_chordal(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Chordal mean of (..., K, 3, 3) with weights (..., K).

    SVD projection of the weighted arithmetic mean onto SO(3) — the same
    estimator the pipeline ships, with the weights left free.
    """
    M = (R * w[..., None, None]).sum(dim=-3) / w.sum(dim=-1).clamp(min=1e-8)[..., None, None]
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([        z, -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],         z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],  v[..., 0],         z], dim=-1)], dim=-2)


def so3_exp(v):
    """Rodrigues, safe at v = 0."""
    th2 = (v * v).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-12)
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2.clamp(min=1e-12)
    W = _skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(W.shape)
    return eye + a[..., None] * W + b[..., None] * (W @ W)


def so3_log(R):
    """Rotation -> axis-angle (radians)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    th = torch.arccos(((tr - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7))
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin = torch.sin(th)
    scale = torch.where(sin < 1e-6, torch.full_like(th, 0.5),
                        th / (2 * sin.clamp(min=1e-6)))
    return w * scale[..., None]


def intrinsic_estimator(R_k, vis, mode: str, iters: int = 5):
    """Riemannian (INTRINSIC) location estimators on SO(3), by log/exp iteration.

    mode="mean"   Karcher / Frechet mean: argmin sum_k d_geo(R, R_k)^2
    mode="median" geodesic median:        argmin sum_k d_geo(R, R_k)

    Both iterate  R <- R exp( weighted mean of log(R^T R_k) ), which is gradient
    descent on the respective objective. The median's 1/d weighting is Weiszfeld,
    done on the manifold rather than by reweighting a chordal solve.

    R_k: (..., K, 3, 3), vis: (..., K) -> (..., 3, 3)
    """
    R_cur = weighted_chordal(R_k, vis)           # chordal solution as the seed
    for _ in range(iters):
        v = so3_log(R_cur[..., None, :, :].transpose(-1, -2) @ R_k)   # (...,K,3)
        if mode == "median":
            d = v.norm(dim=-1)
            w = vis / (d + 1e-3)                 # 1e-3 rad ~ 0.06 deg
        else:
            w = vis
        v_bar = (v * w[..., None]).sum(-2) / w.sum(-1).clamp(min=1e-8)[..., None]
        R_cur = R_cur @ so3_exp(v_bar)
    return R_cur


def _fk_chunks(pose_full: torch.Tensor, betas: torch.Tensor,
               device: str, chunk: int = 32) -> torch.Tensor:
    """SMPL-X FK over time in chunks. (B,T,P,55,6) + (B,T,P,10) -> (B,T,P,55,3)."""
    outs = []
    for t0 in range(0, pose_full.shape[1], chunk):
        t1 = min(t0 + chunk, pose_full.shape[1])
        j = get_smplx_joints(pose_full[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :55, :]
        outs.append(j.detach().cpu())
    return torch.cat(outs, dim=1)


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
    ap.add_argument("--max_frames", type=int, default=400,
                    help="cap frames per scene (memory)")
    ap.add_argument("--min_cams", type=int, default=2,
                    help="skip (t,p) seen by fewer cameras — weighting is vacuous there")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="FK is the expensive part; a GPU makes it ~10x faster")
    ap.add_argument("--fk_stride", type=int, default=2,
                    help="score every Nth frame for the mm table (FK memory/time)")
    ap.add_argument("--models", nargs="*", default=[],
                    help="checkpoint paths to score alongside the estimators, as "
                         "name=path (e.g. R2=checkpoints/fusion_r2/best.pt). The v2 "
                         "vs v3 class is chosen from the checkpoint's model_config.")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/view_weighting_ceiling_rich.json"))
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
        # Val scenes first, so a small --max_scenes still says something about
        # data the model never trained on.
        dps = _tr.load_datapoints((val_scenes + train_scenes)[:args.max_scenes])
        ds = _tr.RICHFusionDataset(dps, augment=False)

    # ── Trained models, scored on exactly the same slots ─────────────────────
    models = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        st = ck.get("model_state_dict", ck.get("model", ck))
        cfg = ck.get("model_config") or {}
        kw = dict(embedding_dim=st["joint_id_embedding.weight"].shape[1],
                  num_joints=st["joint_id_embedding.weight"].shape[0],
                  num_layers=sum(1 for k in st if k.startswith("layers.")
                                 and k.endswith(".ff.norm.weight")),
                  max_temporal_len=st["temporal_pe.pe"].shape[0],
                  num_heads=cfg.get("num_heads", 8),
                  temporal_window=cfg.get("temporal_window", 128),
                  kintree_mask_k=cfg.get("kintree_mask_k"))
        # v3 shares v2's parameter shapes exactly; only model_config distinguishes.
        if "residual_head" in cfg or "centered_input" in cfg:
            from fusion.fusion_module_v3 import PoseFusionModuleV3
            m = PoseFusionModuleV3(residual_head=cfg.get("residual_head", True),
                                   centered_input=cfg.get("centered_input", True), **kw)
            kind = "V3-residual"
        else:
            from fusion.fusion_module_v2 import PoseFusionModule
            m = PoseFusionModule(**kw)
            kind = "V2-direct"
        m.load_state_dict(st, strict=True)
        models[name] = m.to(args.device).eval()
        logger.info(f"model {name}: {kind}  epoch {ck.get('epoch')}  from {path}")

    from torch.utils.data import DataLoader

    METHODS = ["uniform", "oracle_best", "oracle_inv", "oracle_topk",
               "perframe_inv", "perframe_best",
               "geo_median", "biweight", "trim1",
               "karcher", "geo_median_i", "single_mean"] + list(models)
    MM_METHODS = ["uniform", "oracle_best", "oracle_inv", "oracle_topk",
                  "perframe_inv", "perframe_best",
                  "geo_median", "biweight", "trim1",
                  "karcher", "geo_median_i"] + list(models)
    acc: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    acc_body: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    acc_mm: dict[str, list[np.ndarray]] = {m: [] for m in MM_METHODS}
    n_slots = 0
    per_scene = []

    for inputs, targets in DataLoader(ds, batch_size=1):
        scene = targets.get("scene_name", ["?"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene

        pose = inputs["pose"][:, :args.max_frames].float()          # (B,T,K,P,55,6)
        pmask = inputs["person_mask"][:, :args.max_frames].float()  # (B,T,K,P)
        gt = targets["pose"][:, :args.max_frames].float()           # (B,T,P,55,6)
        gt_valid = targets.get("gt_valid")
        if gt_valid is not None:
            gt_valid = gt_valid[:, :args.max_frames]

        # Root is produced by the placer, not by fusion — drop it everywhere.
        pose, gt = pose[..., 1:, :], gt[..., 1:, :]

        with torch.no_grad():
            R_k = sixd_to_matrix(pose)                              # (B,T,K,P,J,3,3)
            R_gt = sixd_to_matrix(gt)                               # (B,T,P,J,3,3)

            # Move K to the second-to-last axis so weighting broadcasts cleanly.
            R_k = R_k.permute(0, 1, 3, 4, 2, 5, 6)                  # (B,T,P,J,K,3,3)
            vis = pmask.permute(0, 1, 3, 2)                         # (B,T,P,K)
            vis = vis[..., None, :].expand(*vis.shape[:-1], R_k.shape[3], vis.shape[-1])
            # -> (B,T,P,J,K)

            err_k = geodesic_deg(R_k, R_gt[..., None, :, :])        # (B,T,P,J,K)
            err_k = torch.where(vis > 0, err_k, torch.full_like(err_k, float("nan")))

            # Slot validity: GT present and >= min_cams observing cameras.
            n_seen = (vis > 0).sum(-1)                              # (B,T,P,J)
            ok = n_seen >= args.min_cams
            if gt_valid is not None:
                ok = ok & gt_valid[..., None].expand_as(ok)
            if not bool(ok.any()):
                continue

            w_uniform = vis.clone()

            # ORACLE 1 — hard selection of the single closest camera.
            e_filled = torch.nan_to_num(err_k, nan=float("inf"))
            best = e_filled.argmin(dim=-1, keepdim=True)
            w_best = torch.zeros_like(vis).scatter_(-1, best, 1.0)

            # ORACLE 2 — soft inverse-error weighting.
            w_inv = vis / (torch.nan_to_num(err_k, nan=0.0) ** 2 + 1.0)

            # ORACLE 3 — uniform over the best half of the cameras.
            k_keep = torch.clamp((n_seen + 1) // 2, min=1)          # (B,T,P,J)
            order = e_filled.argsort(dim=-1)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(
                order.shape[-1], device=order.device).expand_as(order))
            w_topk = ((rank < k_keep[..., None]).float()) * vis

            # CONSTRAINED ORACLES — one weight per (frame, person, camera),
            # constant across joints. A per-frame feature cannot do better than
            # this, however perfectly it predicts camera quality.
            err_pf = torch.nanmean(
                torch.where(vis > 0, err_k, torch.full_like(err_k, float("nan"))),
                dim=3, keepdim=True)                                # (B,T,P,1,K)
            err_pf = torch.nan_to_num(err_pf, nan=0.0).expand_as(err_k)
            w_pf_inv = vis / (err_pf ** 2 + 1.0)
            pf_filled = torch.where(vis > 0, err_pf, torch.full_like(err_pf, float("inf")))
            pf_best = pf_filled.argmin(dim=-1, keepdim=True)
            w_pf_best = torch.zeros_like(vis).scatter_(-1, pf_best, 1.0) * vis

            # ── REALISABLE: robust estimators on disagreement alone ──────
            # Distance of each view to the current estimate, no GT involved.
            R_bar_u = weighted_chordal(R_k, w_uniform)
            d_u = geodesic_deg(R_k, R_bar_u[..., None, :, :])       # (B,T,P,J,K)
            d_u = torch.where(vis > 0, d_u, torch.full_like(d_u, float("nan")))

            # Weiszfeld / geodesic median: 3 IRLS steps from the chordal mean.
            R_cur = R_bar_u
            for _ in range(3):
                d_cur = geodesic_deg(R_k, R_cur[..., None, :, :])
                w_gm = vis / (torch.nan_to_num(d_cur, nan=0.0) + 1.0)
                R_cur = weighted_chordal(R_k, w_gm)
            w_geo = w_gm
            fused_geo = R_cur

            # Tukey biweight around the median distance (MAD scale).
            med = torch.nanmedian(d_u, dim=-1, keepdim=True).values
            mad = torch.nanmedian((d_u - med).abs(), dim=-1, keepdim=True).values
            scale = (1.4826 * mad).clamp(min=1.0)
            u = torch.nan_to_num((d_u - med) / (4.685 * scale), nan=0.0)
            w_bi = vis * (1.0 - u.clamp(-1, 1) ** 2) ** 2

            # Trim the single most deviant view (needs >= 3 observers).
            worst = torch.nan_to_num(d_u, nan=-1.0).argmax(dim=-1, keepdim=True)
            w_tr = vis.clone().scatter_(-1, worst, 0.0)
            w_tr = torch.where((n_seen >= 3)[..., None].expand_as(vis), w_tr, vis)

            R_karcher = intrinsic_estimator(R_k, vis, "mean")
            R_geomed_i = intrinsic_estimator(R_k, vis, "median")

            res = {}
            fused = {}
            for name, w in (("uniform", w_uniform), ("oracle_best", w_best),
                            ("oracle_inv", w_inv), ("oracle_topk", w_topk),
                            ("perframe_inv", w_pf_inv), ("perframe_best", w_pf_best),
                            ("geo_median", w_geo), ("biweight", w_bi), ("trim1", w_tr)):
                R_bar = weighted_chordal(R_k, w)
                res[name] = geodesic_deg(R_bar, R_gt)               # (B,T,P,J)
                fused[name] = R_bar
            # Intrinsic estimators are computed directly, not via a weight vector.
            for nm, Rb in (("karcher", R_karcher), ("geo_median_i", R_geomed_i)):
                res[nm] = geodesic_deg(Rb, R_gt)
                fused[nm] = Rb

            # Trained models: same slots, same GT, same metric as the estimators.
            for nm, m in models.items():
                out = m(pose.to(args.device), pmask.to(args.device)).cpu().float()
                Rm = sixd_to_matrix(out)                            # (B,T,P,J,3,3)
                res[nm] = geodesic_deg(Rm, R_gt)
                fused[nm] = Rm

            # Reference: expected error of a randomly chosen single camera.
            res["single_mean"] = torch.nanmean(err_k, dim=-1)

            # ── mm: FK each fused rotation set and score root-relative MPJPE ──
            # GT betas and GT root orientation are supplied to both sides, so the
            # only thing that differs between methods is the body pose — exactly
            # the quantity view weighting controls.
            t_idx = torch.arange(0, pose.shape[1], args.fk_stride)
            gt_root = targets["pose"][:, :args.max_frames][:, t_idx, :, :1, :].float()
            betas = targets["shape"][:, :args.max_frames][:, t_idx].float()
            ok_s = ok[:, t_idx][..., 0]                             # (B,T',P) joint-independent
            mm_res = {}
            if bool(ok_s.any()):
                gt_full = torch.cat([gt_root, gt[:, t_idx]], dim=3)
                J_gt = _fk_chunks(gt_full, betas, args.device)
                for name in MM_METHODS:
                    p_full = torch.cat(
                        [gt_root, matrix_to_sixd(fused[name][:, t_idx])], dim=3)
                    J_pred = _fk_chunks(p_full, betas, args.device)
                    d = torch.linalg.norm(
                        (J_pred - J_pred[..., :1, :]) - (J_gt - J_gt[..., :1, :]),
                        dim=-1)                                     # (B,T',P,55)
                    mm_res[name] = (d[..., :22][ok_s] * 1000.0).mean(dim=-1)

        for name in METHODS:
            v = res[name][ok]
            acc[name].append(v[torch.isfinite(v)].numpy())
            body_ok = ok[..., _BODY]
            vb = res[name][..., _BODY][body_ok]
            acc_body[name].append(vb[torch.isfinite(vb)].numpy())

        for name in MM_METHODS:
            if name in mm_res:
                acc_mm[name].append(mm_res[name].numpy())

        n_slots += int(ok.sum())
        per_scene.append({
            "scene": scene,
            "n_slots": int(ok.sum()),
            "uniform": float(np.nanmean(acc["uniform"][-1])),
            "oracle_inv": float(np.nanmean(acc["oracle_inv"][-1])),
            "oracle_best": float(np.nanmean(acc["oracle_best"][-1])),
        })
        logger.info(
            f"{scene}: slots={int(ok.sum())} "
            f"deg uniform={per_scene[-1]['uniform']:.2f} "
            f"oracle_inv={per_scene[-1]['oracle_inv']:.2f} | "
            f"mm uniform={np.mean(acc_mm['uniform'][-1]) if acc_mm['uniform'] else float('nan'):.1f} "
            f"oracle_inv={np.mean(acc_mm['oracle_inv'][-1]) if acc_mm['oracle_inv'] else float('nan'):.1f}")

    if n_slots == 0:
        raise SystemExit("no valid slots")

    def summarise(a: dict[str, list[np.ndarray]]) -> dict:
        return {m: {"mean": float(np.concatenate(a[m]).mean()),
                    "median": float(np.median(np.concatenate(a[m])))}
                for m in METHODS}

    all_j, body_j = summarise(acc), summarise(acc_body)

    print(f"\n{'='*66}\nORACLE VIEW-WEIGHTING CEILING — {len(per_scene)} scenes, "
          f"{n_slots:,} (frame,person,joint) slots\n{'='*66}")
    for tag, tbl in (("ALL 54 JOINTS", all_j), ("BODY JOINTS 1..21 (what metrics score)", body_j)):
        u = tbl["uniform"]["mean"]
        print(f"\n{tag}")
        print(f"  {'method':<14} {'mean err':>9} {'median':>9} {'vs uniform':>12}")
        for m in METHODS:
            print(f"  {m:<14} {tbl[m]['mean']:>8.2f}° {tbl[m]['median']:>8.2f}° "
                  f"{tbl[m]['mean'] - u:>+11.2f}°")

    mm = {m: {"mean": float(np.concatenate(acc_mm[m]).mean()),
              "median": float(np.median(np.concatenate(acc_mm[m])))}
          for m in MM_METHODS if acc_mm[m]}
    if mm:
        u = mm["uniform"]["mean"]
        print(f"\nROOT-RELATIVE MPJPE, root + 21 body joints — THE METRIC (mm)")
        print(f"  {'method':<14} {'mean':>9} {'median':>9} {'vs uniform':>12}")
        for m in MM_METHODS:
            if m in mm:
                print(f"  {m:<14} {mm[m]['mean']:>8.1f}  {mm[m]['median']:>8.1f}  "
                      f"{mm[m]['mean'] - u:>+11.1f}")

    print("\nREADING: oracle_inv is the best ANY weighting scheme could do with "
          "perfect knowledge\nof each camera's error. If it barely beats uniform, "
          "no feature set can help and\nthe chordal mean is the right fusion rule.")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"n_slots": n_slots, "all_joints": all_j, "body_joints": body_j,
         "rr_mpjpe_mm": mm, "per_scene": per_scene}, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
