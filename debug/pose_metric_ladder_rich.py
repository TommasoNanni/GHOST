"""Where does R2's body-pose advantage go? — RICH test metric ladder.

QUESTION
--------
Two measurements disagree about the same models on the SAME 52 test scenes:

    body pose only (RR-MPJPE, GT betas + GT root)
        uniform chordal mean 38.9    geodesic median 38.2    R2 37.1   <- R2 WINS by 1.9
    full pipeline (PA-MPJPE, predicted betas, Procrustes, SMPL-24)
        uniform chordal mean 26.5                            R2 27.7   <- R2 LOSES by 1.2

The placer is NOT the explanation: PA-MPJPE is Procrustes-aligned and therefore
invariant to the root, verified empirically (debug/root_swap_rich.py guard (a),
max gap 7.4e-6 mm across root variants). So the flip is created by one of the
three remaining differences between the two measurements:

    betas      GT                     vs   predicted (mean of the per-camera SAM3D betas)
    joint set  SMPL-X root + 21 body  vs   SMPL-24 (J_regressor @ smplx2smpl on FK verts)
    alignment  root-relative          vs   Procrustes Sim(3)

WHAT IS MEASURED
----------------
The full 2x2x2 factorial of those three, for each fusion rule, on identical slots.
Eight cells; the cell (predicted betas, SMPL-24, Procrustes) is the pipeline's
PA-MPJPE and must reproduce the published 26.5 / 27.7 — that is the experiment's
self-validation. Any cell where the mean-vs-R2 ordering flips names the factor
responsible.

INTERPRETATION
--------------
- flip at the BETAS axis   -> R2's pose is better but is being FK'd with a shape it
                              was not optimised for; fixable by better shape estimation.
- flip at the ALIGNMENT axis -> R2's gain lives in a component Sim(3) absorbs, i.e.
                              it is real but invisible to the metric the field uses.
- flip at the JOINT-SET axis -> the gain sits on SMPL-X joints that the SMPL-24
                              regressor does not see.
- no flip anywhere         -> the two measurements were never comparable and the
                              pipeline number is the one to trust.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/pose_metric_ladder_rich.py \
        --max_scenes 52 --device cuda --rich_data_root /tmp/centered_test \
        --models R2=checkpoints/fusion_r2/best.pt
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
from evaluation.evaluate_rich import _verts_to_smpl24, _sim3_align
from utilities.smplx_utilities import _get_smplx_model, _rot_matrix_to_axis_angle_safe
from pytorch3d.transforms import rotation_6d_to_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)


def weighted_chordal(R, w):
    M = (R * w[..., None, None]).sum(dim=-3) / w.sum(dim=-1).clamp(min=1e-8)[..., None, None]
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1)], dim=-2)


def so3_exp(v):
    th2 = (v * v).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-12)
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2.clamp(min=1e-12)
    W = _skew(v)
    return torch.eye(3, dtype=v.dtype, device=v.device).expand(W.shape) \
        + a[..., None] * W + b[..., None] * (W @ W)


def so3_log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    th = torch.arccos(((tr - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7))
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin = torch.sin(th)
    return w * torch.where(sin < 1e-6, torch.full_like(th, 0.5),
                           th / (2 * sin.clamp(min=1e-6)))[..., None]


def geodesic_median(R_k, vis, iters: int = 5):
    """Intrinsic Weiszfeld — the estimator selected on the train pool."""
    R_cur = weighted_chordal(R_k, vis)
    for _ in range(iters):
        v = so3_log(R_cur[..., None, :, :].transpose(-1, -2) @ R_k)
        w = vis / (v.norm(dim=-1) + 1e-3)
        v_bar = (v * w[..., None]).sum(-2) / w.sum(-1).clamp(min=1e-8)[..., None]
        R_cur = R_cur @ so3_exp(v_bar)
    return R_cur


def fk_verts_joints(pose_full: torch.Tensor, betas: torch.Tensor, device: str,
                    chunk: int = 8):
    """SMPL-X FK returning BOTH joints and vertices (needed for SMPL-24).

    `utilities.smplx_utilities.get_smplx_joints` hardcodes return_verts=False, so
    the model is called directly here with the same conventions.
    pose_full (B,T,P,55,6), betas (B,T,P,10) -> joints (B,T,P,55,3), verts (B,T,P,V,3)
    """
    B, T, P = pose_full.shape[:3]
    Js, Vs = [], []
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        p = pose_full[:, t0:t1].to(device)
        b = betas[:, t0:t1].to(device)
        n = p.shape[1]
        aa = _rot_matrix_to_axis_angle_safe(
            rotation_6d_to_matrix(p.reshape(-1, 6))).reshape(B * n * P, 55 * 3)
        out = _get_smplx_model(B * n * P, device, p.dtype)(
            global_orient=aa[:, :3], body_pose=aa[:, 3:66],
            left_hand_pose=aa[:, 66:111], right_hand_pose=aa[:, 111:156],
            jaw_pose=aa[:, 156:159], leye_pose=aa[:, 159:162],
            reye_pose=aa[:, 162:165], betas=b.reshape(-1, 10),
            return_verts=True)
        Js.append(out.joints[:, :55].reshape(B, n, P, 55, 3).detach().cpu())
        Vs.append(out.vertices.reshape(B, n, P, -1, 3).detach().cpu())
    return torch.cat(Js, 1), torch.cat(Vs, 1)


def score(pred_j: np.ndarray, gt_j: np.ndarray, alignment: str) -> float:
    """mm error for one (frame, person). pred/gt: (J, 3)."""
    if alignment == "root":
        d = (pred_j - pred_j[:1]) - (gt_j - gt_j[:1])
    else:
        aligned, _, _, _ = _sim3_align(pred_j, gt_j)
        d = aligned - gt_j
    return float(np.linalg.norm(d, axis=-1).mean()) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=52)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--fk_stride", type=int, default=8,
                    help="vertex FK is memory-heavy; stride the frames")
    ap.add_argument("--min_cams", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ghost_output_root", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"))
    ap.add_argument("--rich_data_root", type=Path,
                    default=Path("/tmp/centered_test"))
    ap.add_argument("--rich_gt_dir", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich"))
    ap.add_argument("--body_split", default="test_body")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/pose_metric_ladder_rich.json"))
    args = ap.parse_args()

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
        if "residual_head" in cfg or "centered_input" in cfg:
            from fusion.fusion_module_v3 import PoseFusionModuleV3
            m = PoseFusionModuleV3(residual_head=cfg.get("residual_head", True),
                                   centered_input=cfg.get("centered_input", True), **kw)
        else:
            from fusion.fusion_module_v2 import PoseFusionModule
            m = PoseFusionModule(**kw)
        m.load_state_dict(st, strict=True)
        models[name] = m.to(args.device).eval()
        logger.info(f"model {name}: epoch {ck.get('epoch')}  {path}")

    from data.fusion_dataset import RICHFusionDatapoint
    from torch.utils.data import DataLoader

    scene_dirs = sorted(d for d in args.ghost_output_root.iterdir() if d.is_dir())
    dps = []
    for sd in scene_dirs[:args.max_scenes]:
        try:
            dp = RICHFusionDatapoint(
                scene_dir=sd, rich_data_root=args.rich_data_root,
                rich_gt_dir=args.rich_gt_dir, body_split=args.body_split,
                restrict_to_gt_persons=True)
            if dp.num_frames == 0 or not dp.has_gt:
                continue
            dps.append(dp)
        except Exception as e:
            logger.warning(f"  skip {sd.name}: {e}")
    logger.info(f"{len(dps)} scenes")
    ds = _tr.RICHFusionDataset(dps, augment=False)

    METHODS = ["mean", "geo_median"] + list(models)
    CELLS = [(b, j, a) for b in ("gt", "pred") for j in ("body22", "smpl24")
             for a in ("root", "proc")]
    acc = {m: {c: [] for c in CELLS} for m in METHODS}

    for inputs, targets in DataLoader(ds, batch_size=1):
        scene = targets.get("scene_name", ["?"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene
        T = min(args.max_frames, inputs["pose"].shape[1])
        pose = inputs["pose"][:, :T].float()
        pmask = inputs["person_mask"][:, :T].float()
        gt = targets["pose"][:, :T].float()
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:, :T] if gt_valid is not None else None

        t_idx = torch.arange(0, T, args.fk_stride)
        with torch.no_grad():
            R_k = sixd_to_matrix(pose[..., 1:, :]).permute(0, 1, 3, 4, 2, 5, 6)
            vis = pmask.permute(0, 1, 3, 2)[..., None, :].expand(
                pmask.shape[0], T, pmask.shape[3], R_k.shape[3], pmask.shape[2])
            n_seen = (vis > 0).sum(-1)
            ok = (n_seen >= args.min_cams)[..., 0]           # (B,T,P)
            if gt_valid is not None:
                ok = ok & gt_valid
            ok_s = ok[:, t_idx]
            if not bool(ok_s.any()):
                continue

            fused = {"mean": weighted_chordal(R_k, vis),
                     "geo_median": geodesic_median(R_k, vis)}
            for nm, m in models.items():
                fused[nm] = sixd_to_matrix(
                    m(pose.to(args.device), pmask.to(args.device)).cpu().float())

            gt_root = gt[:, t_idx][..., :1, :]
            betas_gt = targets["shape"][:, :T][:, t_idx].float()
            # Predicted betas: mean of the per-camera SAM3D betas over VISIBLE
            # cameras — exactly what the pipeline feeds the placer.
            w = pmask[:, t_idx][..., None]
            betas_pred = ((inputs["shape"][:, :T][:, t_idx].float() * w).sum(2)
                          / w.sum(2).clamp(min=1e-8))

            J_gt = {}
            for bkey, bet in (("gt", betas_gt), ("pred", betas_pred)):
                j, v = fk_verts_joints(torch.cat([gt_root, gt[:, t_idx][..., 1:, :]], 3),
                                       bet, args.device)
                J_gt[bkey] = {"body22": j[..., :22, :].numpy(),
                              "smpl24": _verts_to_smpl24(v.numpy())}

            for name, R in fused.items():
                p6 = torch.cat([gt_root, matrix_to_sixd(R[:, t_idx])], dim=3)
                for bkey, bet in (("gt", betas_gt), ("pred", betas_pred)):
                    j, v = fk_verts_joints(p6, bet, args.device)
                    Jp = {"body22": j[..., :22, :].numpy(),
                          "smpl24": _verts_to_smpl24(v.numpy())}
                    idx = np.nonzero(ok_s.numpy())
                    for jkey in ("body22", "smpl24"):
                        for akey in ("root", "proc"):
                            errs = [score(Jp[jkey][b, t, p], J_gt[bkey][jkey][b, t, p], akey)
                                    for b, t, p in zip(*idx)]
                            acc[name][(bkey, jkey, akey)].extend(errs)
        logger.info(f"{scene}: {int(ok_s.sum())} person-frames")

    print(f"\n{'='*78}\nMETRIC LADDER — where the advantage is created and destroyed\n{'='*78}")
    hdr = f"{'cell (betas / joints / align)':<34}" + "".join(f"{m:>11}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    out = {}
    for c in CELLS:
        row = {m: float(np.mean(acc[m][c])) if acc[m][c] else float("nan") for m in METHODS}
        out["/".join(c)] = row
        tag = f"{c[0]:<5}/ {c[1]:<7}/ {c[2]:<5}"
        mark = "   <- pipeline PA-MPJPE" if c == ("pred", "smpl24", "proc") else ""
        print(f"{tag:<34}" + "".join(f"{row[m]:>11.1f}" for m in METHODS) + mark)

    print("\nDELTA vs mean (negative = better than the chordal mean):")
    for c in CELLS:
        r = out["/".join(c)]
        tag = f"{c[0]:<5}/ {c[1]:<7}/ {c[2]:<5}"
        print(f"{tag:<34}" + "".join(f"{r[m]-r['mean']:>+11.1f}" for m in METHODS))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"n_scenes": len(dps), "cells": out}, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
