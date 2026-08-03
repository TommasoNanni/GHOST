"""Does MESH SELF-OCCLUSION rank cameras per joint? — RICH.

QUESTION
--------
The oracle bound says per-joint camera weighting is worth -5.7 mm RR-MPJPE on the
12-scene pool (48.5 uniform -> 42.8).  Two signals have now been measured against
it and both came back empty:

    pred_joint_confidence        top-1 at chance, 0.0 mm end-to-end
    reprojection residual (2D)   top-1 at chance, -0.1 mm on top of the median

`debug/view_signal_probe_rich.py` also established WHY the 2D signal failed: the
multi-view reprojection residual is only 1.3-5.5 px, i.e. the views agree in the
image plane almost perfectly, while their ROTATIONS disagree a lot.  Monocular
depth/orientation ambiguity lives in the null space of the 2D observation, so no
amount of 2D consistency can see it.

Self-occlusion is the one remaining cheap signal that does NOT go through 2D
agreement.  A camera looking at the back of a torso genuinely cannot see the
elbow in front of it; SAM3D will still emit a confident, 2D-consistent, plausibly
hallucinated arm.  That failure is invisible to every signal tried so far but is
exactly what an occlusion test detects.

Two structural advantages over the reprojection residual:

  * EVERY joint has vertices, so all 54 joints get a signal.  The 2D probe could
    only reach the 13 joints with an MHR70 landmark, capping its ceiling at
    -4.2 of the -5.7 mm.  Here the reachable ceiling IS the full oracle.
  * No calibration whatsoever is required (see below), so none of the crop-offset
    / resize-scale ambiguity that the 2D probe had to fit its way around.

METHOD
------
For each (frame, camera, person): FK that camera's OWN SMPL-X estimate, place the
mesh in that camera's frame with its own ``pred_cam_t``, and ask which of joint
j's surface vertices are visible.

Visibility is a pure depth-ordering question along rays from the camera centre,
so it depends only on ``(x/z, y/z)`` and ``z`` -- NOT on focal length or
principal point.  That is why this probe needs no calibration at all: a 96x96
grid is laid over the person's own projected extent, the minimum depth per cell
is taken (a point z-buffer via ``scatter_reduce``, far cheaper than rasterising
faces), and a vertex counts as visible when its depth is within ``--tol`` of the
frontmost surface in its cell.  Joint j's signal is the visible FRACTION of its
top-``--n_verts`` vertices by SMPL-X skinning weight.

As in the 2D probe both kinematic pairings are scored, since joint j's rotation
determines its CHILD's position, not its own:

    at_joint : weight rotation j by the visibility of joint j
    at_child : weight rotation j by the visibility of j's child

WHAT IT CANNOT SEE (state it in the paper)
------------------------------------------
Only SELF- and inter-person occlusion.  Environment occlusion -- the table in
`wipingtable`, the fence in `overfence` -- needs scene geometry that is not in
the fusion batch, and the segmentation masks that would proxy for it live in a
different pixel space (source frame vs SAM3D's centred crop).  So a limb hidden
behind furniture still scores as fully visible here.  If this probe shows signal,
the mask-based version is the obvious follow-up; if it shows none, that follow-up
is not worth the plumbing.

Usage
-----
    OMP_NUM_THREADS=8 pixi run python debug/occlusion_signal_probe_rich.py \
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
from utilities.smplx_utilities import (_get_smplx_model,
                                       _rot_matrix_to_axis_angle_safe,
                                       get_smplx_joints, rotation_6d_to_matrix)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "train_rich_v3", _REPO / "scripts" / "train_rich_v3.py")
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

# SMPL-X kinematic parent of each body joint, used for the at_child pairing:
# rotation of j is observed through the position/visibility of its child.
_CHILD_OF = {
    1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12,
    12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21,
}


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


def geodesic_median(R_k: torch.Tensor, w0: torch.Tensor, iters: int = 5,
                    eps: float = 1e-3) -> torch.Tensor:
    M = weighted_chordal(R_k, w0)
    for _ in range(iters):
        d = geodesic_deg(R_k, M[..., None, :, :]).deg2rad()
        M = weighted_chordal(R_k, w0 / (d + eps))
    return M


def _fk_chunks(pose_full, betas, device, chunk: int = 32):
    outs = []
    for t0 in range(0, pose_full.shape[1], chunk):
        t1 = min(t0 + chunk, pose_full.shape[1])
        j = get_smplx_joints(pose_full[:, t0:t1].to(device),
                             betas[:, t0:t1].to(device))[..., :55, :]
        outs.append(j.detach().cpu())
    return torch.cat(outs, dim=1)


def _rank_avg(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Average ranks along the last axis, ties shared, masked entries ignored.

    Position-based tie-breaking is NOT acceptable here: a quarter of visibility
    values are exactly 1.0, so ordering ties by array index would make the
    statistic partly measure camera index instead of visibility.
    """
    a, b = x[..., :, None], x[..., None, :]
    mj = m[..., None, :].float()
    less = ((b < a).float() * mj).sum(-1)
    eq = ((b == a).float() * mj).sum(-1)
    return less + 0.5 * (eq - 1.0)


def _top1_expected(sig: torch.Tensor, err: torch.Tensor,
                   m: torch.Tensor) -> torch.Tensor:
    """P(picking the lowest-error camera) when ties in ``sig`` break at random.

    Returns a per-slot probability rather than a hard hit, so that a signal which
    is constant across cameras scores exactly at chance instead of being credited
    for whichever camera happens to come first.
    """
    neg = torch.finfo(sig.dtype).min
    s = torch.where(m, sig, torch.full_like(sig, neg))
    best = s.amax(-1, keepdim=True)
    tied = m & (s >= best)
    e = torch.where(m, err, torch.full_like(err, torch.finfo(err.dtype).max))
    lo = e.amin(-1, keepdim=True)
    win = m & (e <= lo)
    return (tied & win).sum(-1).float() / tied.sum(-1).clamp(min=1).float()


def top_vertices_per_joint(n_verts: int, device) -> torch.Tensor:
    """(J, n_verts) indices of the highest-skinning-weight vertices per joint."""
    model = _get_smplx_model(1, device, torch.float32)
    W = model.lbs_weights                      # (V, J)
    return W.topk(n_verts, dim=0).indices.T.contiguous()


def fk_vertices(pose: torch.Tensor, betas: torch.Tensor,
                device) -> tuple[torch.Tensor, torch.Tensor]:
    """FK returning (vertices, joints), both with the pelvis at the origin.

    ``pose`` is (N, J, 6) and ``betas`` is (N, 10).  SMPL-X returns vertices with
    the pelvis at the shape-dependent rest position J[0]; subtracting it puts the
    pelvis at the origin so ``pred_cam_t`` can be added directly.
    """
    N, J, _ = pose.shape
    aa = _rot_matrix_to_axis_angle_safe(
        rotation_6d_to_matrix(pose.reshape(N * J, 6))).reshape(N, J * 3)
    out = _get_smplx_model(N, device, torch.float32)(
        global_orient=aa[:, :3], body_pose=aa[:, 3:66],
        left_hand_pose=aa[:, 66:111], right_hand_pose=aa[:, 111:156],
        jaw_pose=aa[:, 156:159], leye_pose=aa[:, 159:162],
        reye_pose=aa[:, 162:165], betas=betas, return_verts=True)
    root = out.joints[:, :1]
    return out.vertices - root, out.joints[:, :55] - root


def visible_fraction(verts: torch.Tensor, top_idx: torch.Tensor,
                     grid: int, tol: float) -> torch.Tensor:
    """Fraction of each joint's surface vertices visible from the camera origin.

    Args:
        verts:   (N, V, 3) vertices in CAMERA frame (camera at the origin,
                 looking down +z).
        top_idx: (J, n) vertex indices per joint.
        grid:    z-buffer resolution over the person's own projected extent.
        tol:     depth tolerance in metres for "at the front surface".

    Returns:
        (N, J) visible fraction in [0, 1].
    """
    N, V, _ = verts.shape
    J, n = top_idx.shape
    z = verts[..., 2]
    valid = z > 1e-3
    zc = torch.where(valid, z, torch.ones_like(z))
    uv = verts[..., :2] / zc[..., None]                      # (N,V,2) rays

    # Grid over each person's own projected extent -> no focal, no principal point.
    big = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan")))
    lo = big.nan_to_num(nan=float("inf")).amin(dim=1, keepdim=True)
    hi = big.nan_to_num(nan=float("-inf")).amax(dim=1, keepdim=True)
    span = (hi - lo).clamp(min=1e-6)
    cell = (((uv - lo) / span) * (grid - 1e-4)).floor().clamp(0, grid - 1).long()
    flat = (cell[..., 1] * grid + cell[..., 0]).clamp(0, grid * grid - 1)  # (N,V)

    # Point z-buffer: minimum depth per cell.
    zbuf = torch.full((N, grid * grid), float("inf"), device=verts.device,
                      dtype=verts.dtype)
    zbuf.scatter_reduce_(1, flat, torch.where(valid, z, torch.full_like(z, float("inf"))),
                         reduce="amin", include_self=True)
    front = zbuf.gather(1, flat)                              # (N,V)
    vis = valid & (z <= front + tol)

    sel = vis.gather(1, top_idx.reshape(1, -1).expand(N, J * n))
    return sel.reshape(N, J, n).float().mean(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--max_scenes", type=int, default=12)
    ap.add_argument("--split", choices=["train", "test"], default="train")
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
    ap.add_argument("--fk_stride", type=int, default=4,
                    help="frames are subsampled once and EVERY quantity (signal, "
                         "correlation, mm) is computed on the same subset")
    ap.add_argument("--n_verts", type=int, default=40)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--tol", type=float, default=0.03,
                    help="depth tolerance in metres for 'at the front surface'")
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/occlusion_signal_probe_rich.json"))
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
    logger.info(f"{len(dps)} scenes ({args.split} pool)")

    top_idx = top_vertices_per_joint(args.n_verts, args.device)

    EXP = [1.0, 2.0]
    METHODS = (["uniform", "geo_median", "oracle_joint"]
               + [f"{v}^{p:g}" for v in ("at_joint", "at_child") for p in EXP]
               + ["median x at_joint", "median x at_child"]
               + [f"{a}gate@{t:g}" for t in (0.3, 0.6) for a in ("", "median x ")])
    acc_mm: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    rho = {"at_joint": [], "at_child": []}
    top1 = {"at_joint": [0, 0, 0.0], "at_child": [0, 0, 0.0]}
    vis_stats = []

    for dp in dps:
        scene = dp.scene_dir.name
        inputs, targets = dp._build_sample()

        T0 = min(args.max_frames, inputs["pose"].shape[0])
        t_idx = torch.arange(0, T0, args.fk_stride)
        T = len(t_idx)
        pose = inputs["pose"][:T0][t_idx].float()[None]          # (1,T,K,P,J,6)
        shape_in = inputs["shape"][:T0][t_idx].float()           # (T,K,P,10)
        cam_t = inputs["body_transl_cam_in"][:T0][t_idx].float()  # (T,K,P,3)
        pmask = inputs["person_mask"][:T0][t_idx].float()[None]  # (1,T,K,P)
        gt = targets["pose"][:T0][t_idx].float()[None]
        gt_valid = targets.get("gt_valid")
        gt_valid = gt_valid[:T0][t_idx][None] if gt_valid is not None else None
        Kc, P_n = pose.shape[2], pose.shape[3]

        with torch.no_grad():
            pose_b, gt_b = pose[..., 1:, :], gt[..., 1:, :]
            R_k = sixd_to_matrix(pose_b).permute(0, 1, 3, 4, 2, 5, 6)  # (1,T,P,J,K,3,3)
            R_gt = sixd_to_matrix(gt_b)
            J = R_k.shape[3]
            v_all = pmask.permute(0, 1, 3, 2)[..., None, :].expand(1, T, P_n, J, Kc)

            err = geodesic_deg(R_k, R_gt[..., None, :, :])             # (1,T,P,J,K)
            ok = (v_all > 0).sum(-1) >= args.min_cams
            if gt_valid is not None:
                ok = ok & gt_valid[..., None].expand_as(ok)
            if not bool(ok.any()):
                continue

            # ── occlusion from the FUSED mesh, seen from each camera ────────
            # Deriving visibility from a camera's OWN estimate would contaminate
            # the signal with the very error it is meant to predict (a camera
            # that hallucinates a limb into free space scores it "visible").  So
            # the body configuration is the CONSENSUS one, shared by all views,
            # and only the viewpoint differs.  Since global orient just rotates
            # the pelvis-centred mesh, one canonical FK per (frame, person) is
            # enough: V_cam(k) = R_root(k) @ V_canon + cam_t(k).  That is also 8x
            # cheaper than meshing every camera separately.
            R_fused = geodesic_median(R_k, v_all)                      # (1,T,P,J,3,3)
            eye6 = torch.tensor([1., 0., 0., 0., 1., 0.]).expand(1, T, P_n, 1, 6)
            pose_canon = torch.cat([eye6, matrix_to_sixd(R_fused)], dim=3)
            # betas averaged over the observing views — never GT, this is a signal.
            wv = pmask[0].permute(1, 0, 2)[..., None]                  # (K,T,P,1)
            beta_avg = ((shape_in.permute(1, 0, 2, 3) * wv).sum(0)
                        / wv.sum(0).clamp(min=1e-6))                   # (T,P,10)

            occ = torch.zeros(T, Kc, P_n, 55)
            R_root = sixd_to_matrix(pose[0, :, :, :, 0])               # (T,K,P,3,3)
            chunk = 64
            for t0 in range(0, T, chunk):
                t1 = min(t0 + chunk, T)
                n = (t1 - t0) * P_n
                V, _ = fk_vertices(
                    pose_canon[0, t0:t1].reshape(n, 55, 6).to(args.device),
                    beta_avg[t0:t1].reshape(n, 10).to(args.device), args.device)
                V = V.reshape(t1 - t0, P_n, -1, 3)                     # (t,P,V,3)
                for k in range(Kc):
                    m = pmask[0, t0:t1, k] > 0                         # (t,P)
                    if not bool(m.any()):
                        continue
                    Rk = R_root[t0:t1, k].to(args.device)              # (t,P,3,3)
                    Vk = torch.einsum("tpij,tpvj->tpvi", Rk, V)
                    Vk = Vk + cam_t[t0:t1, k].to(args.device)[:, :, None]
                    f = visible_fraction(Vk[m], top_idx, args.grid, args.tol)
                    occ[t0:t1, k][m] = f.cpu()
            vis_stats.append(occ[(pmask[0] > 0)].numpy())

            # (T,K,P,55) -> (1,T,P,J,K), dropping the root to match R_k
            o = occ[..., 1:].permute(0, 2, 3, 1)[None]                 # (1,T,P,J,K)
            sig_j = o.clone()
            sig_c = torch.ones_like(o)
            for j, c in _CHILD_OF.items():
                if j - 1 < J and c - 1 < J:
                    sig_c[..., j - 1, :] = o[..., c - 1, :]

            vm = v_all > 0
            for name, sig in (("at_joint", sig_j), ("at_child", sig_c)):
                e_r = _rank_avg(err, vm)
                c_r = _rank_avg(sig, vm)
                m = vm.float()
                n = m.sum(-1, keepdim=True).clamp(min=1)
                ec = (e_r - (e_r * m).sum(-1, keepdim=True) / n) * m
                cc = (c_r - (c_r * m).sum(-1, keepdim=True) / n) * m
                den = (ec.pow(2).sum(-1) * cc.pow(2).sum(-1)).sqrt()
                r = torch.where(den > 0, (ec * cc).sum(-1) / den.clamp(min=1e-8),
                                torch.zeros_like(den))
                rho[name].append(r[ok].numpy())
                top1[name][0] += float(_top1_expected(sig, err, vm)[ok].sum())
                top1[name][1] += int(ok.sum())
                top1[name][2] += float((1.0 / vm.sum(-1).clamp(min=1))[ok].sum())

            # ── end-to-end mm on the same frames ────────────────────────────
            ok_s = ok[..., 0]
            gt_root = targets["pose"][:T0][t_idx][None][:, :, :, :1, :].float()
            betas = targets["shape"][:T0][t_idx][None].float()
            J_gt = _fk_chunks(torch.cat([gt_root, gt_b], dim=3), betas, args.device)

            fused = {
                "uniform":      weighted_chordal(R_k, v_all),
                "geo_median":   R_fused,   # already computed for the signal
                "oracle_joint": weighted_chordal(R_k, v_all / (err ** 2 + 1.0)),
                "median x at_joint": geodesic_median(R_k, v_all * sig_j.clamp(min=1e-3)),
                "median x at_child": geodesic_median(R_k, v_all * sig_c.clamp(min=1e-3)),
            }
            for nm, sig in (("at_joint", sig_j), ("at_child", sig_c)):
                for p in EXP:
                    fused[f"{nm}^{p:g}"] = weighted_chordal(
                        R_k, v_all * sig.clamp(min=1e-3).pow(p))
            # Hard gate: DROP a view when the joint is mostly hidden, instead of
            # gently down-weighting it. 28% of visibilities are exactly 1.0, so a
            # soft weight is nearly uniform among the visible views and cannot
            # express "this one is useless".  Falls back to uniform for slots
            # where every view would be gated out.
            for thr in (0.3, 0.6):
                g = v_all * (sig_j >= thr).float()
                g = torch.where(g.sum(-1, keepdim=True) > 0, g, v_all)
                fused[f"gate@{thr:g}"] = weighted_chordal(R_k, g)
                fused[f"median x gate@{thr:g}"] = geodesic_median(R_k, g)

            for nm, R_bar in fused.items():
                p_full = torch.cat([gt_root, matrix_to_sixd(R_bar)], dim=3)
                J_pred = _fk_chunks(p_full, betas, args.device)
                d = torch.linalg.norm(
                    (J_pred - J_pred[..., :1, :]) - (J_gt - J_gt[..., :1, :]), dim=-1)
                acc_mm[nm].append((d[..., :22][ok_s] * 1000.0).mean(-1).numpy())

        logger.info(f"{scene}: vis={vis_stats[-1].mean():.3f} "
                    f"| uniform={np.mean(acc_mm['uniform'][-1]):.1f} "
                    f"med={np.mean(acc_mm['geo_median'][-1]):.1f} "
                    f"joint^1={np.mean(acc_mm['at_joint^1'][-1]):.1f} mm")

    if not acc_mm["uniform"]:
        raise SystemExit("no scene produced valid slots")

    mm = {m: float(np.concatenate(acc_mm[m]).mean()) for m in METHODS if acc_mm[m]}
    u = mm["uniform"]
    vs = np.concatenate(vis_stats)

    print(f"\n{'='*74}\nMESH SELF-OCCLUSION AS A PER-JOINT VIEW WEIGHT — RICH "
          f"({args.split} pool, {len(acc_mm['uniform'])} scenes)\n{'='*74}")
    print(f"\nvisible fraction: mean {vs.mean():.3f}  median {np.median(vs):.3f}  "
          f"fully visible {100*(vs > 0.99).mean():.1f}%  "
          f"fully hidden {100*(vs < 0.01).mean():.1f}%")
    print("   (if almost everything is fully visible there is no signal to exploit)")

    print("\n1. Within-slot Spearman(visibility, error) — want NEGATIVE")
    for nm, v in rho.items():
        if v:
            print(f"   {nm:<10}: {np.concatenate(v).mean():+.4f} "
                  f"({len(np.concatenate(v)):,} slots)")
    print("\n2. Top-1 agreement: most-visible camera is also the lowest-error one")
    for nm, (hit, n, ch) in top1.items():
        if n:
            print(f"   {nm:<10}: {100*hit/n:5.1f}%   chance {100*ch/n:5.1f}%   ({n:,} slots)")
    print("\n3. END-TO-END root-relative MPJPE (root + 21 body joints) — THE VERDICT")
    for m in METHODS:
        if m in mm:
            print(f"   {m:<20} {mm[m]:>7.1f} mm   {mm[m]-u:>+6.1f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "split": args.split, "n_scenes": len(acc_mm["uniform"]),
        "rr_mpjpe_mm": mm,
        "visible_fraction": {"mean": float(vs.mean()),
                             "frac_fully_visible": float((vs > 0.99).mean()),
                             "frac_fully_hidden": float((vs < 0.01).mean())},
        "spearman": {k: float(np.concatenate(v).mean()) for k, v in rho.items() if v},
        "top1_agreement": {k: {"rate": v[0] / v[1], "chance": v[2] / v[1]}
                           for k, v in top1.items() if v[1]},
    }, indent=2))
    print(f"\nwritten: {args.out_json}")


if __name__ == "__main__":
    main()
