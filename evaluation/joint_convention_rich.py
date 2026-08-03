"""Is the chordal mean's PA-MPJPE win a joint-convention artifact? — RICH.

QUESTION
--------
On RICH test the fusion module LOSES to a plain chordal rotation mean on
PA-MPJPE (30.4 vs 26.5 mm) while winning the temporally-integrated metrics.
The training objective (`JointPositionLoss`, fusion/loss_v2.py:152) and the
evaluation metric do NOT measure the same thing:

    JointPositionLoss :  55 SMPL-X joints, ROOT-RELATIVE,     GT betas
    PA-MPJPE          :  24 SMPL joints,   PROCRUSTES Sim(3), predicted betas

The 24 SMPL joints are not a subset of the 55 SMPL-X joints — they are
regressed from the mesh (J_regressor @ smplx2smpl @ verts), so a model that is
better in SMPL-X joint space can be worse in SMPL joint space.

Three things differ at once, so comparing the two numbers directly proves
nothing. This script holds betas fixed (GT, as the loss does) and sweeps the
other two factors independently:

                     |  55 SMPL-X joints  |  24 SMPL joints
    -----------------+--------------------+------------------
    root-relative    |   = the loss       |   joint set only
    Procrustes Sim3  |   alignment only   |   = PA-MPJPE

If B(ghost) beats A(chordal) in the top-left cell but loses in the bottom-right,
the deficit is a convention artifact and the column/row it flips on says whether
the joint set or the alignment is responsible.

CAVEAT ON ABSOLUTE VALUES
-------------------------
Both variants here use GT betas and GT global orientation (exactly what
JointPositionLoss does) so that POSE is the only thing under test. The published
PA-MPJPE additionally carries predicted betas and the Procrustes-DLT placement,
so the bottom-right cell will NOT equal 30.4 / 26.5 mm. What is comparable is
the SIGN and SIZE of the A-vs-B gap within each cell.

Usage
-----
    pixi run python evaluation/joint_convention_rich.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --rich_data_root    /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \\
        --rich_gt_dir       /capstor/scratch/cscs/tnanni/datasets/rich \\
        --checkpoint        checkpoints/fusion_module/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule
from utilities.smplx_utilities import get_smplx_joints, get_smplx_vertices

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_VARIANTS = ("chordal", "ghost")
_CELLS = ("rootrel_55", "rootrel_24", "pa_55", "pa_24")

_J24_OPERATOR: np.ndarray | None = None


def _j24_operator() -> np.ndarray:
    """J24 = J_regressor @ smplx2smpl  (24, 10475). Copied from evaluate_rich.py."""
    global _J24_OPERATOR
    if _J24_OPERATOR is None:
        import joblib
        import scipy.sparse as sp
        smplx2smpl = np.asarray(
            joblib.load(_REPO_ROOT / "body_models" / "smplx2smpl.pkl")["matrix"]
        )                                        # (6890, 10475), barycentric
        with open(_REPO_ROOT / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl", "rb") as f:
            Jr = pickle.load(f, encoding="latin1")["J_regressor"]   # (24, 6890)
        Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr)
        _J24_OPERATOR = (Jr @ smplx2smpl).astype(np.float32)        # (24, 10475)
    return _J24_OPERATOR


# ---------------------------------------------------------------------------
# Rotation / alignment helpers (copied from evaluate_rich.py)
# ---------------------------------------------------------------------------

def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_sixd(R: torch.Tensor) -> torch.Tensor:
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def chordal_mean(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Closed-form chordal (L2) rotation average. R (N,K,3,3), w (N,K) -> (N,3,3)."""
    M = (R.double() * w.double()[..., None, None]).sum(dim=1)
    U, _, Vt = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vt)
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return (U @ D @ Vt).to(R.dtype)


def sim3_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Umeyama Sim(3) of pred onto gt. Both (N, 3). Returns aligned pred."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p0, g0 = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(p0.T @ g0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var = (p0 ** 2).sum()
    s = (S * [1.0, 1.0, d]).sum() / var if var > 1e-12 else 1.0
    return (s * (R @ p0.T)).T + mu_g


# ---------------------------------------------------------------------------
# Model loading — copied from evaluation/evaluate_rich.py
# ---------------------------------------------------------------------------

def load_fusion_model(checkpoint: Path, device: torch.device) -> PoseFusionModule:
    ckpt  = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    emb_dim  = state["joint_id_embedding.weight"].shape[1]
    n_joints = state["joint_id_embedding.weight"].shape[0]
    n_layers = sum(1 for k in state if k.startswith("layers.") and k.endswith(".ff.norm.weight"))
    max_tlen = state["temporal_pe.pe"].shape[0]
    cfg = ckpt.get("model_config") or {}
    model = PoseFusionModule(
        embedding_dim=emb_dim, num_layers=n_layers,
        num_joints=n_joints, max_temporal_len=max_tlen,
        num_heads=cfg.get("num_heads", 8),
        temporal_window=cfg.get("temporal_window", 128),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded checkpoint: emb={emb_dim} layers={n_layers} joints={n_joints}")
    return model


# ---------------------------------------------------------------------------
# FK
# ---------------------------------------------------------------------------

@torch.no_grad()
def fk_joints_and_j24(
    pose_full: torch.Tensor,      # (1, T, P, 55, 6) — GT root prepended
    betas: torch.Tensor,          # (1, T, P, 10)
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (joints55 (T,P,55,3), joints24 (T,P,24,3)) in the SMPL-X local frame.

    Vertices are reduced to the 24 SMPL joints inside the chunk loop; the full
    (T, P, 10475, 3) vertex tensor is never held.
    """
    T = pose_full.shape[1]
    J24 = torch.from_numpy(_j24_operator()).to(pose_full.device, pose_full.dtype)
    j55_out, j24_out = [], []
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        p, b = pose_full[:, t0:t1], betas[:, t0:t1]
        j55 = get_smplx_joints(p, b)[..., :55, :]           # (1,dt,P,55,3)
        v   = get_smplx_vertices(p, b)                      # (1,dt,P,10475,3)
        j24 = torch.einsum("vn,btpnc->btpvc", J24, v)       # (1,dt,P,24,3)
        j55_out.append(j55[0].cpu())
        j24_out.append(j24[0].cpu())
        del v
    return (torch.cat(j55_out, 0).numpy().astype(np.float64),
            torch.cat(j24_out, 0).numpy().astype(np.float64))


# ---------------------------------------------------------------------------
# Per-scene
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_scene(dp, model, device, min_views: int, chunk: int, scene: str) -> dict | None:
    from data.fusion_dataset import RICHFusionDataset

    loader = torch.utils.data.DataLoader(RICHFusionDataset([dp]), batch_size=1)
    inputs, targets = next(iter(loader))

    pose     = inputs["pose"].to(device).float()          # (1,T,K,P,55,6)
    pmask    = inputs["person_mask"].to(device).float()   # (1,T,K,P)
    gt_pose  = targets["pose"].to(device).float()         # (1,T,P,55,6)
    betas    = targets["shape"].to(device).float()        # (1,T,P,10)
    gt_valid = targets["gt_valid"].to(device).bool()      # (1,T,P)

    _, T, K, P, _, _ = pose.shape

    # ── Variant B: fusion module (returns 54, root stripped) ─────────────────
    fused = model(pose, pmask)                            # (1,T,P,54,6)

    # ── Variant A: chordal mean over views, same inputs ──────────────────────
    Rin = sixd_to_matrix(pose[0][..., 1:, :])             # (T,K,P,54,3,3)
    Jd  = Rin.shape[-3]
    Rin = Rin.permute(0, 2, 3, 1, 4, 5).reshape(-1, K, 3, 3)
    w   = pmask[0].permute(0, 2, 1)[:, :, None, :].expand(T, P, Jd, K).reshape(-1, K)
    chordal = matrix_to_sixd(chordal_mean(Rin, w).reshape(T, P, Jd, 3, 3))[None]

    # ── Prepend GT global orient, exactly as JointPositionLoss does ──────────
    gt_root = gt_pose[..., :1, :]                         # (1,T,P,1,6)
    pose_by_variant = {
        "chordal": torch.cat([gt_root, chordal], dim=3),
        "ghost":   torch.cat([gt_root, fused],   dim=3),
    }

    valid = (gt_valid[0] & (pmask[0].sum(dim=1) >= min_views)).cpu().numpy()   # (T,P)
    if not valid.any():
        return None

    gt55, gt24 = fk_joints_and_j24(gt_pose, betas, chunk)
    res = {c: {v: [] for v in _VARIANTS} for c in _CELLS}

    for v in _VARIANTS:
        p55, p24 = fk_joints_and_j24(pose_by_variant[v], betas, chunk)
        for t in range(T):
            for p in range(P):
                if not valid[t, p]:
                    continue
                for tag, pr, g in (("55", p55[t, p], gt55[t, p]),
                                   ("24", p24[t, p], gt24[t, p])):
                    # root-relative: subtract each convention's own root joint
                    rr = np.linalg.norm((pr - pr[:1]) - (g - g[:1]), axis=-1).mean()
                    pa = np.linalg.norm(sim3_align(pr, g) - g, axis=-1).mean()
                    res[f"rootrel_{tag}"][v].append(rr * 1000.0)
                    res[f"pa_{tag}"][v].append(pa * 1000.0)

    n = len(res["pa_24"]["ghost"])
    for c in _CELLS:
        assert all(len(res[c][v]) == n for v in _VARIANTS), f"{c}: sample mismatch"
    return {"scene": scene, "T": T, "K": K, "P": P, "n": n,
            "cells": {c: {v: np.asarray(res[c][v], dtype=np.float64) for v in _VARIANTS}
                      for c in _CELLS}}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separate joint-set from alignment in the PA-MPJPE gap (RICH)")
    ap.add_argument("--ghost_output_root", required=True, type=Path)
    ap.add_argument("--rich_data_root",    required=True, type=Path)
    ap.add_argument("--rich_gt_dir",       required=True, type=Path)
    ap.add_argument("--checkpoint",        required=True, type=Path)
    ap.add_argument("--body_split",        default="test_body")
    ap.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scenes",            default="")
    ap.add_argument("--max_scenes",        type=int, default=None)
    ap.add_argument("--min_views",         type=int, default=1)
    ap.add_argument("--chunk",             type=int, default=32,
                    help="frames per SMPL-X FK call (vertex memory bound)")
    ap.add_argument("--out",               type=Path,
                    default=Path("eval_explainability/joint_convention_rich.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}  min_views={args.min_views}")
    model = load_fusion_model(args.checkpoint, device)

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scene_dirs = sorted(d for d in args.ghost_output_root.iterdir() if d.is_dir())
    if wanted:
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    logger.info(f"{len(scene_dirs)} scene(s)")

    from data.fusion_dataset import RICHFusionDatapoint

    rows, skipped = [], []
    for sd in scene_dirs:
        try:
            dp = RICHFusionDatapoint(
                scene_dir=sd, rich_data_root=args.rich_data_root,
                rich_gt_dir=args.rich_gt_dir, body_split=args.body_split,
                restrict_to_gt_persons=True)
            if dp.num_frames == 0 or not dp.has_gt:
                skipped.append((sd.name, "no frames / no GT"));  continue
            r = process_scene(dp, model, device, args.min_views, args.chunk, sd.name)
            if r is None:
                skipped.append((sd.name, "no valid frames"));    continue
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"{sd.name}: skipped ({type(e).__name__}: {e})")
            skipped.append((sd.name, f"{type(e).__name__}: {e}"));  continue
        c = r["cells"]
        logger.info(
            f"{sd.name:<26} n={r['n']:4d}  "
            f"loss55 A={c['rootrel_55']['chordal'].mean():6.2f} B={c['rootrel_55']['ghost'].mean():6.2f}  "
            f"pa24 A={c['pa_24']['chordal'].mean():6.2f} B={c['pa_24']['ghost'].mean():6.2f}")
        rows.append(r)

    if not rows:
        logger.error("no scenes processed");  return

    agg = {c: {v: float(np.concatenate([r["cells"][c][v] for r in rows]).mean())
               for v in _VARIANTS} for c in _CELLS}
    n_tot = int(sum(r["n"] for r in rows))

    # NOTE: the rootrel_55 cell uses the loss's joint set and alignment, but is
    # reported as mean L2 in mm; JointPositionLoss itself reduces with MSE.
    title = {"rootrel_55": "root-relative, 55 SMPL-X   (loss joint set)",
             "rootrel_24": "root-relative, 24 SMPL",
             "pa_55":      "Procrustes Sim3, 55 SMPL-X",
             "pa_24":      "Procrustes Sim3, 24 SMPL   (= PA-MPJPE)"}

    print(f"\n{'='*84}")
    print("JOINT CONVENTION vs ALIGNMENT — mean per-joint error in mm (GT betas, GT root)")
    print(f"  scenes={len(rows)}  skipped={len(skipped)}  person-frames={n_tot:,}")
    print(f"{'='*84}")
    print(f"{'cell':<46}{'A chordal':>11}{'B ghost':>11}{'B - A':>10}{'winner':>8}")
    print("-" * 84)
    for c in _CELLS:
        a, b = agg[c]["chordal"], agg[c]["ghost"]
        print(f"{title[c]:<46}{a:>11.2f}{b:>11.2f}{b - a:>+10.2f}"
              f"{('B' if b < a else 'A'):>8}")
    print("-" * 84)
    d_joint = ((agg["pa_24"]["ghost"] - agg["pa_24"]["chordal"])
               - (agg["pa_55"]["ghost"] - agg["pa_55"]["chordal"]))
    d_align = ((agg["pa_24"]["ghost"] - agg["pa_24"]["chordal"])
               - (agg["rootrel_24"]["ghost"] - agg["rootrel_24"]["chordal"]))
    print(f"  gap(B-A) attributable to the 55->24 joint set : {d_joint:+.2f} mm")
    print(f"  gap(B-A) attributable to rootrel->Procrustes  : {d_align:+.2f} mm")
    print("=" * 84)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {"checkpoint": str(args.checkpoint), "min_views": args.min_views,
                       "betas": "GT", "global_orient": "GT",
                       "n_scenes": len(rows), "n_person_frames": n_tot},
            "cells_mm": agg,
            "delta_from_joint_set_mm": d_joint,
            "delta_from_alignment_mm": d_align,
            "per_scene": [{"scene": r["scene"], "n": r["n"],
                           **{c: {v: float(r["cells"][c][v].mean()) for v in _VARIANTS}
                              for c in _CELLS}} for r in rows],
            "skipped": [{"scene": s, "reason": x} for s, x in skipped],
        }, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
