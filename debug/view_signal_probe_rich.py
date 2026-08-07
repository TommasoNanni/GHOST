"""Can a REPROJECTION RESIDUAL rank cameras per joint? — RICH.

QUESTION
--------
`debug/view_weighting_ceiling_rich.py` established the error budget for fusion:

    uniform chordal mean            48.5 mm RR-MPJPE   (12-scene pool)
    geodesic median                 47.8 mm  (-0.8)    <- best realisable so far
    oracle PER-JOINT weighting      42.8 mm  (-5.7)    <- the whole headroom
    oracle PER-FRAME weighting      48.5 mm  (-0.0)    <- structurally useless

Every signal tried so far reaches only ~14% of that headroom, and all of them
share one flaw: they are DISAGREEMENT-based (distance to the consensus) and
therefore self-referential.  When three views share SAM3D's bias, the honest
fourth view looks like the outlier and gets down-weighted.  `pred_joint_
confidence` failed for the same reason plus being near-uninformative.

This probe tests the first signal that is INDEPENDENT of the quantity being
fused.  Body pose fusion combines ROTATIONS; the signal here is built purely
from 2D landmarks and camera calibration, never from the rotations:

    for each joint, DLT-triangulate its 3D position from every view's kp2d,
    reproject into camera k, and measure the residual against camera k's own
    kp2d, normalised by that view's bbox diagonal.

A view whose 2D landmark for a joint is wrong — occluded limb, truncation,
left/right swap, wrong person — disagrees with the multi-view consensus in the
image plane and gets a large residual.  That is exactly the per-joint,
per-camera failure mode the oracle says the headroom consists of.

KNOWN BLIND SPOT (state it in the paper): the residual is small whenever all
views agree, even if they agree on something wrong.  The signal is independent
of the ROTATIONS but not of SAM3D itself, so correlated 2D bias is invisible to
it — the same weakness disagreement-weighting has.  It can only catch views that
fail *differently* from the others.

KINEMATIC PAIRING
-----------------
A subtlety that changes the answer.  In SMPL-X, joint j's rotation determines
the position of its CHILDREN, not its own position: the knee's rotation moves
the ankle.  So the observable for "is this view's knee rotation good?" is the
reprojection residual at the ANKLE, not at the knee.  Both pairings are scored:

    at_joint : weight rotation j by the residual measured at joint j   (naive)
    at_child : weight rotation j by the residual at j's child          (correct)

CALIBRATION
-----------
`kp2d` lives in the source-frame pixel space of the preprocessed images (e.g.
840x614 for the Gym scenes) while the RICH XML intrinsics are for the original
full-resolution capture (fx ~ 4649).  The full->source scale is not a constant:
different RICH scenes come from different rigs, the principal point is not at
the image centre, and the centred-crop step adds a small per-camera offset.
Rather than hardcode a factor -- the exact place a silent-fallback bug already
bit this codebase once -- a per-camera 2D similarity (s, du, dv) is FITTED:

    u_obs = s * (fx * X_c / Z_c + cx) + du            (linear in s, du, dv)

seeded by a coarse shared-s search and refined by alternating triangulation and
closed-form least squares.  The converged s is printed per scene: for a source
frame that is a 1/5 downscale of the original it must land near 0.20, so a
mis-calibrated scene announces itself instead of silently poisoning the signal.
Scenes whose final median reprojection error exceeds --max_calib_px are dropped
from the signal (and reported), never silently kept.

WHAT IS REPORTED
----------------
  1. within-slot Spearman(signal, per-joint geodesic error to GT) across cameras
  2. top-1 agreement: is the best-signal camera also the lowest-error one
  3. END-TO-END RR-MPJPE (root + 21 body joints, GT root, GT betas) for
     signal-weighted chordal means and for the geodesic median with its IRLS
     weights modulated by the signal -- the actual verdict
  4. the oracle RESTRICTED to the joints this signal can reach, i.e. the ceiling
     of this whole signal class, next to the unrestricted oracle

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/view_signal_probe_rich.py \
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

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

# ── Joint bookkeeping ────────────────────────────────────────────────────────
# SMPL-X joint index -> MHR70 landmark index, mirroring fusion/placer.py.
# Only these 13 SMPL-X joints have a 2D landmark, so only these can produce a
# reprojection residual.  Everything else keeps the baseline weight.
_SMPLX_TO_MHR70 = {
    1: 9,   2: 10,   # left/right hip
    4: 11,  5: 12,   # left/right knee
    7: 13,  8: 14,   # left/right ankle
    12: 69,          # neck
    16: 5,  17: 6,   # left/right shoulder
    18: 7,  19: 8,   # left/right elbow
    20: 62, 21: 41,  # left/right wrist
}

# rotation of joint j is observed through the position of child c.
# Only pairs where BOTH ends are mapped above are usable.
_ROT_TO_OBS_CHILD = {
    1: 4,   2: 5,     # hip rotation      -> knee position
    4: 7,   5: 8,     # knee rotation     -> ankle position
    16: 18, 17: 19,   # shoulder rotation -> elbow position
    18: 20, 19: 21,   # elbow rotation    -> wrist position
}

_GROUPS = {"body": slice(0, 21), "hands": slice(21, 51), "face": slice(51, 54)}


# ── SO(3) helpers ────────────────────────────────────────────────────────────
def geodesic_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    rel = Ra @ Rb.transpose(-1, -2)
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.arccos(cos))


def weighted_chordal(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted extrinsic (chordal) mean of rotations over the -3 axis."""
    M = (R * w[..., None, None]).sum(dim=-3) / w.sum(dim=-1).clamp(min=1e-8)[..., None, None]
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, dtype=M.dtype, device=M.device).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


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


def so3_log(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.arccos(cos)[..., None]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return v * (th / (2 * torch.sin(th).clamp(min=1e-7)))


def geodesic_median(R_k: torch.Tensor, w0: torch.Tensor, iters: int = 5,
                    eps: float = 1e-3) -> torch.Tensor:
    """Weiszfeld/IRLS L1 estimator on SO(3), seeded from the weighted chordal mean.

    ``w0`` is a fixed prior weight per view (visibility, optionally modulated by
    an external signal); the IRLS weight ``1/(d+eps)`` multiplies it each step.
    """
    M = weighted_chordal(R_k, w0)
    for _ in range(iters):
        d = geodesic_deg(R_k, M[..., None, :, :]).deg2rad()
        w = w0 / (d + eps)
        M = weighted_chordal(R_k, w)
    return M


def _fk_chunks(pose_full, betas, device, chunk: int = 32):
    outs = []
    for t0 in range(0, pose_full.shape[1], chunk):
        t1 = min(t0 + chunk, pose_full.shape[1])
        j = get_smplx_joints(pose_full[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :55, :]
        outs.append(j.detach().cpu())
    return torch.cat(outs, dim=1)


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort(dim=-1)
    r = torch.empty_like(order)
    r.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device).expand_as(order))
    return r.float()


# ── Triangulation + self-calibration ─────────────────────────────────────────
def triangulate_dlt(uv: torch.Tensor, P: torch.Tensor,
                    vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched linear triangulation.

    Args:
        uv:  (N, K, 2) observed pixels.
        P:   (K, 3, 4) projection matrices in the SAME pixel space as ``uv``.
        vis: (N, K) bool, which views observe this point.

    Returns:
        X:  (N, 3) triangulated positions (garbage where ``ok`` is False).
        ok: (N,) bool, at least 2 views and a non-degenerate solution.
    """
    N, K, _ = uv.shape
    u, v = uv[..., 0:1], uv[..., 1:2]                       # (N,K,1)
    P0, P1, P2 = P[None, :, 0], P[None, :, 1], P[None, :, 2]  # (1,K,4)
    rows = torch.stack([u * P2 - P0, v * P2 - P1], dim=2)   # (N,K,2,4)
    m = vis[:, :, None, None].to(rows.dtype)
    A = (rows * m).reshape(N, 2 * K, 4)
    # Row-normalise so no single view dominates the SVD purely through scale.
    A = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    _, _, Vh = torch.linalg.svd(A)
    X = Vh[:, -1, :]
    w = X[:, 3]
    ok = (vis.sum(-1) >= 2) & (w.abs() > 1e-9)
    X3 = X[:, :3] / torch.where(ok, w, torch.ones_like(w))[:, None]
    return X3, ok


def _project_unit(X: torch.Tensor, K_full: torch.Tensor,
                  E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project into full-resolution pixels (before the fitted similarity).

    Returns ``(ab, in_front)`` where ``ab`` is (N, 2) and ``in_front`` is (N,).
    """
    Xc = X @ E[:, :3].T + E[:, 3]                      # (N,3) camera frame
    z = Xc[:, 2]
    zc = torch.where(z.abs() > 1e-6, z, torch.ones_like(z))
    a = K_full[0, 0] * Xc[:, 0] / zc + K_full[0, 2]
    b = K_full[1, 1] * Xc[:, 1] / zc + K_full[1, 2]
    return torch.stack([a, b], dim=-1), z > 1e-6


def _fit_similarity(obs: torch.Tensor, ab: torch.Tensor,
                    m: torch.Tensor) -> tuple[float, float, float]:
    """Closed-form least squares for u = s*a + du, v = s*b + dv (shared s)."""
    if int(m.sum()) < 8:
        return float("nan"), 0.0, 0.0
    a, b = ab[m, 0], ab[m, 1]
    u, v = obs[m, 0], obs[m, 1]
    am, bm, um, vm = a.mean(), b.mean(), u.mean(), v.mean()
    num = ((a - am) * (u - um)).sum() + ((b - bm) * (v - vm)).sum()
    den = ((a - am) ** 2).sum() + ((b - bm) ** 2).sum()
    if float(den) < 1e-9:
        return float("nan"), 0.0, 0.0
    s = float(num / den)
    return s, float(um - s * am), float(vm - s * bm)


def calibrate_and_reproject(uv: torch.Tensor, vis: torch.Tensor,
                            K_full: torch.Tensor, E: torch.Tensor,
                            s_grid: torch.Tensor, iters: int = 4):
    """Fit per-camera (s, du, dv), triangulate, return reprojection residuals.

    Args:
        uv:  (N, K, 2) observed pixels for every (frame, person, joint) sample.
        vis: (N, K) bool.
        K_full, E: (K,3,3) and (K,3,4) full-resolution intrinsics / extrinsics.

    Returns:
        resid: (N, K) pixel residual, +inf where not observed / not triangulable.
        info:  dict with the fitted scales and the final median residual.
    """
    N, K, _ = uv.shape
    dev, dt = uv.device, uv.dtype

    def build_P(s: torch.Tensor, du: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        S = torch.zeros(K, 3, 3, dtype=dt, device=dev)
        S[:, 0, 0] = s
        S[:, 1, 1] = s
        S[:, 2, 2] = 1.0
        S[:, 0, 2] = du
        S[:, 1, 2] = dv
        return S @ K_full @ E

    # ── coarse search on a single shared scale, du = dv = 0 ──────────────────
    sub = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:min(N, 4000)]
    best_s, best_err = float(s_grid[0]), float("inf")
    for s0 in s_grid:
        P = build_P(torch.full((K,), float(s0), dtype=dt, device=dev),
                    torch.zeros(K, dtype=dt, device=dev),
                    torch.zeros(K, dtype=dt, device=dev))
        X, ok = triangulate_dlt(uv[sub], P, vis[sub])
        if not bool(ok.any()):
            continue
        e = []
        for k in range(K):
            ab, front = _project_unit(X, K_full[k], E[k])
            m = ok & vis[sub, k] & front
            if bool(m.any()):
                e.append((uv[sub, k][m] - float(s0) * ab[m]).norm(dim=-1))
        if not e:
            continue
        err = float(torch.cat(e).median())
        if err < best_err:
            best_err, best_s = err, float(s0)

    s = torch.full((K,), best_s, dtype=dt, device=dev)
    du = torch.zeros(K, dtype=dt, device=dev)
    dv = torch.zeros(K, dtype=dt, device=dev)

    # ── alternate: triangulate with current calibration, then refit it ───────
    for _ in range(iters):
        P = build_P(s, du, dv)
        X, ok = triangulate_dlt(uv, P, vis)
        for k in range(K):
            ab, front = _project_unit(X, K_full[k], E[k])
            m = ok & vis[:, k] & front
            s_k, du_k, dv_k = _fit_similarity(uv[:, k], ab, m)
            if np.isfinite(s_k) and s_k > 0:
                s[k], du[k], dv[k] = s_k, du_k, dv_k

    # ── final residuals under the converged calibration ─────────────────────
    P = build_P(s, du, dv)
    X, ok = triangulate_dlt(uv, P, vis)
    resid = torch.full((N, K), float("inf"), dtype=dt, device=dev)
    for k in range(K):
        ab, front = _project_unit(X, K_full[k], E[k])
        m = ok & vis[:, k] & front
        pred = s[k] * ab + torch.stack([du[k], dv[k]])
        resid[m, k] = (uv[m, k] - pred[m]).norm(dim=-1)

    finite = torch.isfinite(resid)
    med = float(resid[finite].median()) if bool(finite.any()) else float("nan")
    return resid, {"scale": [round(float(x), 5) for x in s],
                   "median_reproj_px": med,
                   "coarse_px": best_err,
                   "frac_triangulated": float(ok.float().mean())}


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=12)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="'train' = the 12-scene selection pool (10 val + 2 train) "
                         "used to choose the geodesic median. 'test' = the 52 "
                         "published scenes; only for a final confirmation run.")
    ap.add_argument("--ghost_output_root", type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test"))
    ap.add_argument("--rich_data_root", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich/centered_test"))
    ap.add_argument("--rich_gt_dir", type=Path,
                    default=Path("/capstor/scratch/cscs/tnanni/datasets/rich"))
    ap.add_argument("--body_split", default="test_body")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--min_cams", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fk_stride", type=int, default=2)
    ap.add_argument("--max_calib_px", type=float, default=15.0,
                    help="drop a scene's signal if the fitted calibration leaves a "
                         "larger median reprojection error than this (source pixels)")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/view_signal_probe_rich.json"))
    args = ap.parse_args()

    # ── scenes ───────────────────────────────────────────────────────────────
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
                if dp.num_frames == 0 or not dp.has_gt:
                    continue
                dps.append(dp)
            except Exception as e:
                logger.warning(f"  skip {sd.name}: {e}")
    else:
        scenes = [s for s in sorted(_tr.SCENES_ROOT.iterdir())
                  if s.is_dir() and s.name not in _tr.SKIP_SCENES]
        train_scenes, val_scenes = _tr._split_by_location(scenes, _tr.NUM_VAL_SCENES)
        dps = _tr.load_datapoints((val_scenes + train_scenes)[:args.max_scenes])
    logger.info(f"{len(dps)} scenes ({args.split} pool)")

    # Signal-carrying joint sets, in the root-dropped indexing used below
    # (array index i  <->  SMPL-X joint i+1).
    obs_idx = {j: m for j, m in _SMPLX_TO_MHR70.items() if j >= 1}
    at_joint_pairs = [(j - 1, obs_idx[j]) for j in sorted(obs_idx)]
    at_child_pairs = [(j - 1, obs_idx[c]) for j, c in sorted(_ROT_TO_OBS_CHILD.items())
                      if c in obs_idx]

    EXP = [1.0, 2.0]
    METHODS = (["uniform", "geo_median", "oracle_joint", "oracle_reachable"]
               + [f"{v}^{p:g}" for v in ("at_joint", "at_child",
                                         "at_joint_raw", "at_child_raw")
                  for p in EXP]
               + ["median x at_child", "median x at_child_raw"])
    acc_mm: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    rho_at_joint, rho_at_child = [], []
    top1 = {"at_joint": [0, 0, 0.0], "at_child": [0, 0, 0.0]}
    calib_log, skipped = {}, []

    s_grid = torch.linspace(0.05, 0.60, 45, dtype=torch.float64)

    for dp in dps:
        scene = dp.scene_dir.name
        inputs, targets = dp._build_sample()

        T = min(args.max_frames, inputs["pose"].shape[0])
        pose = inputs["pose"][:T].float()[None]              # (1,T,K,P,J,6)
        pmask = inputs["person_mask"][:T].float()[None]      # (1,T,K,P)
        kp2d = inputs["kp2d"][:T].double()                   # (T,K,P,70,2)
        gt = targets["pose"][:T].float()[None]
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:T][None] if gt_valid is not None else None

        Kc, P_n = pose.shape[2], pose.shape[3]

        # Cameras: XML intrinsics/extrinsics at full capture resolution.
        cams = getattr(dp, "_cameras", [])
        if len(cams) < Kc or any(c.get("intrinsics") is None or c.get("extrinsics") is None
                                 for c in cams[:Kc]):
            skipped.append((scene, "missing calibration"))
            logger.warning(f"{scene}: no XML calibration — skipped")
            continue
        K_full = torch.stack([torch.as_tensor(np.asarray(cams[k]["intrinsics"]),
                                              dtype=torch.float64) for k in range(Kc)])
        E = torch.stack([torch.as_tensor(np.asarray(cams[k]["extrinsics"]),
                                         dtype=torch.float64) for k in range(Kc)])

        # ── residuals for every mapped landmark ─────────────────────────────
        mhr_ids = sorted({m for _, m in at_joint_pairs})
        col = {m: i for i, m in enumerate(mhr_ids)}
        uv = kp2d[:, :, :, mhr_ids]                          # (T,K,P,M,2)
        M = len(mhr_ids)
        uv = uv.permute(0, 2, 3, 1, 4).reshape(-1, Kc, 2)    # (T*P*M, K, 2)
        vis = (pmask[0] > 0).permute(0, 2, 1)[..., None, :]  # (T,P,1,K)
        vis = vis.expand(T, P_n, M, Kc).reshape(-1, Kc)

        resid_flat, info = calibrate_and_reproject(uv, vis, K_full, E, s_grid)
        calib_log[scene] = info
        if not np.isfinite(info["median_reproj_px"]) or \
                info["median_reproj_px"] > args.max_calib_px:
            skipped.append((scene, f"calib {info['median_reproj_px']:.1f}px"))
            logger.warning(f"{scene}: calibration failed "
                           f"({info['median_reproj_px']:.1f}px) — skipped")
            continue

        # (T,P,M,K) -> normalise by apparent person size so a distant view's
        # smaller pixel errors are not mistaken for a better view.  The batch
        # carries no bbox, so the size is the kp2d bounding-box diagonal over the
        # mapped landmarks, taken as a MEDIAN OVER TIME per (camera, person): a
        # normaliser that moved with the per-frame failure would mask it.
        resid = resid_flat.reshape(T, P_n, M, Kc)
        lm = kp2d[:, :, :, mhr_ids].permute(0, 2, 3, 1, 4)        # (T,P,M,K,2)
        seen = (pmask[0] > 0).permute(0, 2, 1)[:, :, None, :, None]  # (T,P,1,K,1)
        lm = torch.where(seen.expand_as(lm), lm, torch.full_like(lm, float("nan")))
        span = (lm.nanquantile(0.95, dim=2) - lm.nanquantile(0.05, dim=2))  # (T,P,K,2)
        diag = span.pow(2).sum(-1).sqrt()                          # (T,P,K)
        diag = diag.nanmedian(dim=0, keepdim=True).values          # (1,P,K)
        diag = torch.nan_to_num(diag, nan=1.0).clamp(min=1.0)[:, :, None, :]
        resid = resid / diag
        # Per (person, landmark, camera) median over time: removes the systematic
        # MHR-landmark-vs-joint-centre offset and any residual calibration bias,
        # leaving the time-varying failures we actually want to detect.
        finite = torch.isfinite(resid)
        big = torch.where(finite, resid, torch.full_like(resid, float("nan")))
        med = big.nanmedian(dim=0, keepdim=True).values.clamp(min=1e-6)
        resid_rel = torch.where(finite, resid / med, torch.full_like(resid, float("inf")))
        # Ablation: keep the raw (person-size-normalised only) residual too. The
        # median-relative form above deliberately keeps only TRANSIENT failures,
        # but "camera 3 is always bad at the left ankle" is a legitimate static
        # per-(camera, joint) quality signal that the division removes.
        resid_abs = torch.where(finite, resid, torch.full_like(resid, float("inf")))

        with torch.no_grad():
            # Drop the root: placement is the placer's job, not fusion's.
            pose_b, gt_b = pose[..., 1:, :], gt[..., 1:, :]
            R_k = sixd_to_matrix(pose_b).permute(0, 1, 3, 4, 2, 5, 6)  # (1,T,P,J,K,3,3)
            R_gt = sixd_to_matrix(gt_b)                                # (1,T,P,J,3,3)
            J = R_k.shape[3]
            v_all = pmask.permute(0, 1, 3, 2)[..., None, :].expand(1, T, P_n, J, Kc)

            err = geodesic_deg(R_k, R_gt[..., None, :, :])             # (1,T,P,J,K)
            ok = (v_all > 0).sum(-1) >= args.min_cams
            if gt_valid is not None:
                ok = ok & gt_valid[..., None].expand_as(ok)
            if not bool(ok.any()):
                continue

            # Signal tensors: baseline 1.0 everywhere, replaced on mapped joints.
            def build_signal(pairs, R, eps) -> tuple[torch.Tensor, torch.Tensor]:
                sig = torch.ones(1, T, P_n, J, Kc, dtype=torch.float32)
                touched = torch.zeros(J, dtype=torch.bool)
                for j_idx, mhr in pairs:
                    r = R[:, :, col[mhr], :]                           # (T,P,K)
                    w = 1.0 / (r + eps)
                    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
                    sig[0, :, :, j_idx, :] = w.float()
                    touched[j_idx] = True
                return sig, touched

            # eps sets how hard the down-weighting bites; it is ~1 unit for the
            # median-relative form and ~1 median residual for the raw form.
            sig_j, touch_j = build_signal(at_joint_pairs, resid_rel, 0.1)
            sig_c, touch_c = build_signal(at_child_pairs, resid_rel, 0.1)
            r_med = float(resid_abs[torch.isfinite(resid_abs)].median())
            sig_jr, _ = build_signal(at_joint_pairs, resid_abs, r_med)
            sig_cr, _ = build_signal(at_child_pairs, resid_abs, r_med)

            # ── 1/2. within-slot Spearman + top-1, on touched joints only ───
            bigv = torch.tensor(1e6)
            for name, sig, touch, store in (("at_joint", sig_j, touch_j, rho_at_joint),
                                            ("at_child", sig_c, touch_c, rho_at_child)):
                sel = ok & touch[None, None, None, :]
                if not bool(sel.any()):
                    continue
                e_r = _rank(torch.where(v_all > 0, err, bigv))
                c_r = _rank(torch.where(v_all > 0, sig, -bigv))
                m = (v_all > 0).float()
                n = m.sum(-1, keepdim=True).clamp(min=1)
                ec = (e_r - (e_r * m).sum(-1, keepdim=True) / n) * m
                cc = (c_r - (c_r * m).sum(-1, keepdim=True) / n) * m
                den = (ec.pow(2).sum(-1) * cc.pow(2).sum(-1)).sqrt()
                rho = torch.where(den > 0, (ec * cc).sum(-1) / den.clamp(min=1e-8),
                                  torch.zeros_like(den))
                store.append(rho[sel].numpy())
                ci = torch.where(v_all > 0, sig, -bigv).argmax(-1)
                ei = torch.where(v_all > 0, err, bigv).argmin(-1)
                top1[name][0] += int((ci[sel] == ei[sel]).sum())
                top1[name][1] += int(sel.sum())
                # Chance is 1/(cameras observing THIS slot), which varies, so it
                # is accumulated per slot rather than assumed to be 1/K.
                top1[name][2] += float((1.0 / (v_all > 0).sum(-1).clamp(min=1))[sel].sum())

            # ── 3. end-to-end mm ────────────────────────────────────────────
            t_idx = torch.arange(0, T, args.fk_stride)
            ok_s = ok[:, t_idx][..., 0]
            if not bool(ok_s.any()):
                continue
            gt_root = targets["pose"][:T][None][:, t_idx, :, :1, :].float()
            betas = targets["shape"][:T][None][:, t_idx].float()
            J_gt = _fk_chunks(torch.cat([gt_root, gt_b[:, t_idx]], dim=3),
                              betas, args.device)

            # Oracle restricted to the joints this signal class can ever touch:
            # the ceiling of the whole idea, not just of this estimator.
            reach = (touch_j | touch_c)[None, None, None, :, None]
            w_or = v_all / (err ** 2 + 1.0)
            fused = {
                "uniform":          weighted_chordal(R_k, v_all),
                "geo_median":       geodesic_median(R_k, v_all),
                "oracle_joint":     weighted_chordal(R_k, w_or),
                "oracle_reachable": weighted_chordal(
                    R_k, torch.where(reach.expand_as(v_all), w_or, v_all)),
                "median x at_child": geodesic_median(R_k, v_all * sig_c),
            }
            fused["median x at_child_raw"] = geodesic_median(R_k, v_all * sig_cr)
            for nm, sig in (("at_joint", sig_j), ("at_child", sig_c),
                            ("at_joint_raw", sig_jr), ("at_child_raw", sig_cr)):
                for p in EXP:
                    fused[f"{nm}^{p:g}"] = weighted_chordal(
                        R_k, v_all * sig.clamp(min=1e-4).pow(p))

            for nm, R_bar in fused.items():
                p_full = torch.cat([gt_root, matrix_to_sixd(R_bar[:, t_idx])], dim=3)
                J_pred = _fk_chunks(p_full, betas, args.device)
                d = torch.linalg.norm(
                    (J_pred - J_pred[..., :1, :]) - (J_gt - J_gt[..., :1, :]), dim=-1)
                acc_mm[nm].append((d[..., :22][ok_s] * 1000.0).mean(-1).numpy())

        logger.info(
            f"{scene}: s={info['scale'][0]:.4f} reproj={info['median_reproj_px']:.1f}px "
            f"| uniform={np.mean(acc_mm['uniform'][-1]):.1f} "
            f"med={np.mean(acc_mm['geo_median'][-1]):.1f} "
            f"child^1={np.mean(acc_mm['at_child^1'][-1]):.1f} mm")

    if not acc_mm["uniform"]:
        raise SystemExit("no scene produced valid slots")

    mm = {m: float(np.concatenate(acc_mm[m]).mean()) for m in METHODS if acc_mm[m]}
    u = mm["uniform"]

    print(f"\n{'='*74}\nREPROJECTION RESIDUAL AS A PER-JOINT VIEW WEIGHT — RICH "
          f"({args.split} pool, {len(acc_mm['uniform'])} scenes)\n{'='*74}")

    print("\nCALIBRATION (fitted per camera; s should be ~0.20 for a 1/5 downscale)")
    for sc, inf in calib_log.items():
        print(f"   {sc:<28} s={inf['scale'][0]:.4f}  median reproj "
              f"{inf['median_reproj_px']:6.2f} px  tri {100*inf['frac_triangulated']:.0f}%")
    if skipped:
        print("   SKIPPED: " + ", ".join(f"{s} ({r})" for s, r in skipped))

    print("\n1. Within-slot Spearman(signal, error) across cameras — want NEGATIVE"
          "\n   (high weight <-> low error; signal is 1/residual so negative = good)")
    for nm, store in (("at_joint", rho_at_joint), ("at_child", rho_at_child)):
        if store:
            print(f"   {nm:<10}: {np.concatenate(store).mean():+.4f} "
                  f"({len(np.concatenate(store)):,} slots)")

    print("\n2. Top-1 agreement: best-signal camera is also the lowest-error one")
    for nm, (hit, n, ch) in top1.items():
        if n:
            print(f"   {nm:<10}: {100*hit/n:5.1f}%   chance {100*ch/n:5.1f}%   "
                  f"({n:,} slots)")

    print("\n3. END-TO-END root-relative MPJPE (root + 21 body joints) — THE VERDICT")
    for m in METHODS:
        if m in mm:
            print(f"   {m:<20} {mm[m]:>7.1f} mm   {mm[m]-u:>+6.1f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split,
        "n_scenes": len(acc_mm["uniform"]),
        "rr_mpjpe_mm": mm,
        "spearman": {
            "at_joint": float(np.concatenate(rho_at_joint).mean()) if rho_at_joint else None,
            "at_child": float(np.concatenate(rho_at_child).mean()) if rho_at_child else None,
        },
        "top1_agreement": {k: {"rate": v[0] / v[1], "chance": v[2] / v[1]}
                           for k, v in top1.items() if v[1]},
        "calibration": calib_log,
        "skipped": skipped,
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
