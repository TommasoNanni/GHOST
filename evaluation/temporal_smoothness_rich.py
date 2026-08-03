"""Temporal smoothness of fused pose sequences — RICH test split.

QUESTION
--------
The trained fusion module beats a chordal rotation mean on temporally
integrated metrics (RICH test: WA-MPJPE-100 50.4 vs 55.1 mm, W-MPJPE-100 70.4 vs
84.0) while LOSING on the per-frame metric (PA-MPJPE 30.4 vs 26.5). Is the
advantage temporal smoothing? If so the model's pose sequences should be
measurably smoother than the mean's — and the honest control is ground truth,
because smoother-than-GT is over-smoothing, not accuracy.

WHAT IS MEASURED
----------------
For every (person p, joint j) sequence of root-relative rotations R_t:

    angular velocity      w_t = arccos((trace(R_{t-1}^T R_t) - 1) / 2)
    angular acceleration  a_t = | w_{t+1} - w_t |

reported in degrees, aggregated over all valid (t, p, j) triplets, plus a
per-joint breakdown of mean acceleration.

VARIANTS (identical scenes, frames and joints)
    A  chordal — chordal rotation mean over the visible views, no fusion module
    B  ghost   — PoseFusionModule output
    C  gt      — RICH ground-truth SMPL-X pose

JOINT SET
---------
Body joints only: SMPL-X canonical joints 1..21, which are packed slots 0..20
once the root is dropped — the same subset as the Q1/Q2/Q3 explainability
experiments (`_N_BODY` in explainability_q1.py). Global orientation (joint 0) is
EXCLUDED: it is produced by the placing stage, not the fusion module, and its
magnitude would dominate the statistic.

PoseFusionModule.forward drops the root itself when handed all 55 joints
(fusion_module_v2.py:415), so variants A and C slice `[..., 1:, :]` explicitly to
land on exactly the same 54-joint layout the model emits. Verified: both
checkpoints carry a 54-row joint_id_embedding.

VALIDITY
--------
A (t, p) pair is valid when GT exists (`gt_valid`) and the person is seen by at
least --min_views cameras. Velocities are computed only on CONTIGUOUS runs of
valid frames — never across a gap — and runs shorter than --min_run frames are
discarded outright. All three variants consume ONE shared validity mask, so the
triplet sets are identical by construction; the script asserts this and reports
the count.

PERSISTENCE
-----------
Per scene, the fused sequences of all three variants are written to
<cache_dir>/<scene>.npz as 6D rotations plus the masks and frame/person
metadata, so later experiments can reuse them without re-running inference.

Usage
-----
    pixi run python evaluation/temporal_smoothness_rich.py \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --rich_data_root    /capstor/scratch/cscs/tnanni/datasets/rich/centered_test \\
        --rich_gt_dir       /capstor/scratch/cscs/tnanni/datasets/rich \\
        --checkpoint        checkpoints/fusion_module/best.pt \\
        --body_split        test_body
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion.fusion_module_v2 import PoseFusionModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_N_BODY = 21          # packed slots 0..20 == SMPL-X joints 1..21 (root dropped)

_JOINT_NAMES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
assert len(_JOINT_NAMES) == _N_BODY

_VARIANTS = ("chordal", "ghost", "gt")


# ---------------------------------------------------------------------------
# Rotation helpers (6-D convention = first two ROWS of the matrix, matching
# _aa_to_6d in evaluate_rich.py and convert_pose in data/fusion_dataset.py)
# ---------------------------------------------------------------------------

def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3). Gram-Schmidt on the two rows."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_sixd(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6). First two rows."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def chordal_mean(R: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Closed-form chordal (L2) rotation average.

    R : (N, K, 3, 3)   w : (N, K) visibility weights (0/1)  ->  (N, 3, 3)

    Identical estimator to `chordal_mean` in explainability_q1.py and to
    `mean_fuse` in evaluate_rich_mean.py (verified agreement 5e-08 deg in
    float64). Computed in double: the SVD of a near-degenerate mean matrix is
    the one numerically delicate step here.
    """
    M = (R.double() * w.double()[..., None, None]).sum(dim=1)      # (N,3,3)
    U, _, Vt = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vt)
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return (U @ D @ Vt).to(R.dtype)


def angular_velocity_deg(R: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame geodesic angle of a rotation sequence.

    R : (L, J, 3, 3)  ->  (L-1, J) degrees.

    w_t = arccos((trace(R_{t-1}^T R_t) - 1) / 2), with the arccos argument
    clamped to [-1, 1] so floating-point overshoot cannot produce NaN.
    """
    rel_trace = torch.einsum("ljab,ljab->lj", R[:-1], R[1:])        # trace(A^T B)
    cos = ((rel_trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


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
    # num_heads / temporal_window change no parameter shape, so strict=True cannot
    # catch a mismatch; the trainer persists them under "model_config".
    cfg = ckpt.get("model_config") or {}
    num_heads       = cfg.get("num_heads", 8)
    temporal_window = cfg.get("temporal_window", 128)
    # kintree_bias is a non-persistent buffer: absent from state_dict, invisible
    # to strict=True. Only model_config knows whether the model was masked.
    kintree_mask_k  = cfg.get("kintree_mask_k")
    model = PoseFusionModule(
        embedding_dim=emb_dim, num_layers=n_layers,
        num_joints=n_joints, max_temporal_len=max_tlen,
        num_heads=num_heads, temporal_window=temporal_window,
        kintree_mask_k=kintree_mask_k,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(
        f"Loaded checkpoint: emb={emb_dim} layers={n_layers} joints={n_joints} "
        f"heads={num_heads} twin={temporal_window}"
        f"{'' if cfg else '  (no model_config -> defaults)'}"
    )
    return model


# ---------------------------------------------------------------------------
# Contiguous-run segmentation
# ---------------------------------------------------------------------------

def contiguous_runs(valid: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a 1-D bool array, length >= min_run."""
    if not valid.any():
        return []
    padded = np.concatenate(([False], valid.astype(bool), [False]))
    edges  = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_run]


# ---------------------------------------------------------------------------
# Per-scene computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_scene(
    dp,
    model: PoseFusionModule,
    device: torch.device,
    min_views: int,
    min_run: int,
    cache_dir: Path | None,
    scene_name: str,
) -> dict | None:
    """Build the three variants, segment into runs, return per-scene raw values."""
    from data.fusion_dataset import RICHFusionDataset

    loader = torch.utils.data.DataLoader(RICHFusionDataset([dp]), batch_size=1)
    inputs, targets = next(iter(loader))

    pose     = inputs["pose"].to(device).float()          # (1,T,K,P,55,6)
    pmask    = inputs["person_mask"].to(device).float()   # (1,T,K,P)
    gt       = targets["pose"].to(device).float()         # (1,T,P,55,6)
    gt_valid = targets["gt_valid"].to(device).bool()      # (1,T,P)

    _, T, K, P, J55, _ = pose.shape
    if J55 != 55:
        raise RuntimeError(f"expected 55 packed joints from the dataset, got {J55}")

    # ── Variant B: the fusion module (drops the root internally) ─────────────
    fused = model(pose, pmask)                            # (1,T,P,54,6)
    if fused.shape[-2] != 54:
        raise RuntimeError(f"model returned {fused.shape[-2]} joints, expected 54")
    R_ghost = sixd_to_matrix(fused[0])                    # (T,P,54,3,3)

    # ── Variant A: chordal mean over the K views, same inputs as B ───────────
    Rin = sixd_to_matrix(pose[0][..., 1:, :])             # (T,K,P,54,3,3)
    Jd  = Rin.shape[-3]
    Rin = Rin.permute(0, 2, 3, 1, 4, 5).reshape(-1, K, 3, 3)
    w   = pmask[0].permute(0, 2, 1)[:, :, None, :].expand(T, P, Jd, K).reshape(-1, K)
    R_chordal = chordal_mean(Rin, w).reshape(T, P, Jd, 3, 3)

    # ── Variant C: ground truth ──────────────────────────────────────────────
    R_gt = sixd_to_matrix(gt[0])[:, :, 1:]                # (T,P,54,3,3)

    # ── Body joints only (packed slots 0..20 == SMPL-X 1..21) ────────────────
    R = {
        "chordal": R_chordal[:, :, :_N_BODY],
        "ghost":   R_ghost[:, :, :_N_BODY],
        "gt":      R_gt[:, :, :_N_BODY],
    }
    for name, r in R.items():
        if r.shape != (T, P, _N_BODY, 3, 3):
            raise RuntimeError(f"variant {name}: bad shape {tuple(r.shape)}")

    # ── One shared validity mask for all three variants ──────────────────────
    n_views = pmask[0].sum(dim=1)                         # (T,P)
    valid   = (gt_valid[0] & (n_views >= min_views)).cpu().numpy()   # (T,P)

    out = {v: {"w": [], "a": [], "a_by_joint_sum": np.zeros(_N_BODY, dtype=np.float64),
               "a_by_joint_n": 0} for v in _VARIANTS}
    run_mask = np.zeros((T, P), dtype=bool)
    n_w = n_a = 0
    n_runs = 0

    for p in range(P):
        for s, e in contiguous_runs(valid[:, p], min_run):
            n_runs += 1
            run_mask[s:e, p] = True
            for v in _VARIANTS:
                wv = angular_velocity_deg(R[v][s:e, p])   # (L-1, 21)
                av = (wv[1:] - wv[:-1]).abs()             # (L-2, 21)
                out[v]["w"].append(wv.reshape(-1).cpu().numpy().astype(np.float32))
                out[v]["a"].append(av.reshape(-1).cpu().numpy().astype(np.float32))
                out[v]["a_by_joint_sum"] += av.sum(dim=0).double().cpu().numpy()
                out[v]["a_by_joint_n"]   += av.shape[0]
            n_w += (e - s - 1) * _N_BODY
            n_a += (e - s - 2) * _N_BODY

    if n_runs == 0:
        return None

    # SANITY: all three variants must have consumed exactly the same triplets.
    counts_w = {v: sum(x.size for x in out[v]["w"]) for v in _VARIANTS}
    counts_a = {v: sum(x.size for x in out[v]["a"]) for v in _VARIANTS}
    assert len(set(counts_w.values())) == 1, f"velocity triplet mismatch: {counts_w}"
    assert len(set(counts_a.values())) == 1, f"acceleration triplet mismatch: {counts_a}"
    assert counts_w["gt"] == n_w, f"{counts_w['gt']} != {n_w}"
    assert counts_a["gt"] == n_a, f"{counts_a['gt']} != {n_a}"

    # ── Persist the fused sequences for later reuse ──────────────────────────
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame_start = int(getattr(dp, "_frame_start", 0))
        frame_end   = int(getattr(dp, "_frame_end", frame_start + T))
        rich_to_ghost = getattr(dp, "_rich_to_ghost", {}) or {}
        np.savez_compressed(
            cache_dir / f"{scene_name}.npz",
            # 6D rotations, all 54 packed joints (not just the body subset)
            pose_chordal=matrix_to_sixd(R_chordal).cpu().numpy().astype(np.float32),
            pose_ghost=matrix_to_sixd(R_ghost).cpu().numpy().astype(np.float32),
            pose_gt=matrix_to_sixd(R_gt).cpu().numpy().astype(np.float32),
            valid=valid,                       # (T,P) GT present & >= min_views
            run_mask=run_mask,                 # (T,P) survived the run-length filter
            n_views=n_views.cpu().numpy().astype(np.int16),
            frame_indices=np.arange(frame_start, frame_start + T, dtype=np.int32),
            person_ids=np.array(sorted(getattr(dp, "_gt_matched_ghost_pids", []) or []),
                                dtype=np.int32),
            rich_to_ghost=np.array(sorted(rich_to_ghost.items()), dtype=np.int32)
                            if rich_to_ghost else np.zeros((0, 2), dtype=np.int32),
            joint_layout=np.array(
                "packed 54 = SMPL-X 1..54; body = slots 0..20 (SMPL-X 1..21)"),
            min_views=np.array(min_views), min_run=np.array(min_run),
            frame_start=np.array(frame_start), frame_end=np.array(frame_end),
        )

    return {"scene": scene_name, "T": T, "K": K, "P": P, "n_runs": n_runs,
            "n_w": n_w, "n_a": n_a, "per_variant": out}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(rows: list[dict]) -> dict:
    agg = {}
    for v in _VARIANTS:
        w = np.concatenate([r["per_variant"][v]["w"][i]
                            for r in rows for i in range(len(r["per_variant"][v]["w"]))])
        a = np.concatenate([r["per_variant"][v]["a"][i]
                            for r in rows for i in range(len(r["per_variant"][v]["a"]))])
        js = np.sum([r["per_variant"][v]["a_by_joint_sum"] for r in rows], axis=0)
        jn = int(np.sum([r["per_variant"][v]["a_by_joint_n"] for r in rows]))
        agg[v] = {
            "mean_angular_velocity_deg":     float(w.mean()),
            "median_angular_velocity_deg":   float(np.median(w)),
            "mean_angular_acceleration_deg": float(a.mean()),
            "median_angular_acceleration_deg": float(np.median(a)),
            "per_joint_mean_acceleration_deg": {
                _JOINT_NAMES[j]: float(js[j] / jn) for j in range(_N_BODY)
            },
            "n_velocity_triplets":     int(w.size),
            "n_acceleration_triplets": int(a.size),
        }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Temporal smoothness of chordal mean vs fusion module vs GT (RICH test)")
    ap.add_argument("--ghost_output_root", required=True, type=Path)
    ap.add_argument("--rich_data_root",    required=True, type=Path,
                    help="RICH split root, e.g. .../rich/centered_test")
    ap.add_argument("--rich_gt_dir",       required=True, type=Path,
                    help="RICH root holding <body_split>/")
    ap.add_argument("--checkpoint",        required=True, type=Path)
    ap.add_argument("--body_split",        default="test_body")
    ap.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_scenes",        type=int, default=None)
    ap.add_argument("--scenes",            default="", help="comma-separated scene names")
    ap.add_argument("--min_views",         type=int, default=1,
                    help="minimum cameras seeing the person for a frame to count")
    ap.add_argument("--min_run",           type=int, default=5,
                    help="discard contiguous valid runs shorter than this")
    ap.add_argument("--out",               type=Path,
                    default=Path("eval_explainability/temporal_smoothness_rich.json"))
    ap.add_argument("--cache_dir",         type=Path,
                    default=Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test"),
                    help="where to persist the fused sequences ('none' to skip)")
    args = ap.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}  min_views={args.min_views}  min_run={args.min_run}")
    model = load_fusion_model(args.checkpoint, device)

    cache_dir = None if str(args.cache_dir).lower() == "none" else args.cache_dir
    if cache_dir:
        logger.info(f"Persisting fused sequences to {cache_dir}")

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
                restrict_to_gt_persons=True,
            )
            if dp.num_frames == 0 or not dp.has_gt:
                skipped.append((sd.name, "no frames / no GT"))
                continue
            r = process_scene(dp, model, device, args.min_views, args.min_run,
                              cache_dir, sd.name)
            if r is None:
                skipped.append((sd.name, "no valid runs"))
                continue
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"{sd.name}: skipped ({type(e).__name__}: {e})")
            skipped.append((sd.name, f"{type(e).__name__}: {e}"))
            continue
        gm = float(np.concatenate(r["per_variant"]["gt"]["a"]).mean())
        cm = float(np.concatenate(r["per_variant"]["chordal"]["a"]).mean())
        bm = float(np.concatenate(r["per_variant"]["ghost"]["a"]).mean())
        logger.info(f"{sd.name:<26} runs={r['n_runs']:3d} n_a={r['n_a']:7d}  "
                    f"mean|accel| chordal={cm:6.3f}  ghost={bm:6.3f}  gt={gm:6.3f}")
        rows.append(r)

    if not rows:
        logger.error("no scenes processed")
        return

    agg = summarise(rows)

    # SANITY: GT must actually move.
    g = agg["gt"]
    if g["mean_angular_velocity_deg"] < 1e-6 or g["mean_angular_acceleration_deg"] < 1e-6:
        logger.error("GROUND TRUTH IS STATIC — GT loading or the root-relative "
                     "conversion is wrong. Numbers below are not meaningful.")
    # SANITY: identical triplet counts across variants.
    nw = {v: agg[v]["n_velocity_triplets"] for v in _VARIANTS}
    na = {v: agg[v]["n_acceleration_triplets"] for v in _VARIANTS}
    assert len(set(nw.values())) == 1, f"velocity triplet mismatch across variants: {nw}"
    assert len(set(na.values())) == 1, f"accel triplet mismatch across variants: {na}"

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"TEMPORAL SMOOTHNESS — RICH test, body joints 1..21, global orient excluded")
    print(f"  scenes={len(rows)}  skipped={len(skipped)}  "
          f"min_views={args.min_views}  min_run={args.min_run}")
    print(f"  valid triplets: velocity={nw['gt']:,}  acceleration={na['gt']:,} "
          f"(identical across all three variants)")
    print(f"{'='*78}")
    print(f"{'variant':<12}{'mean |w|':>12}{'med |w|':>11}"
          f"{'mean |a|':>12}{'med |a|':>11}   (degrees)")
    print("-" * 78)
    label = {"chordal": "A chordal", "ghost": "B ghost", "gt": "C gt"}
    for v in _VARIANTS:
        s = agg[v]
        print(f"{label[v]:<12}"
              f"{s['mean_angular_velocity_deg']:>12.4f}"
              f"{s['median_angular_velocity_deg']:>11.4f}"
              f"{s['mean_angular_acceleration_deg']:>12.4f}"
              f"{s['median_angular_acceleration_deg']:>11.4f}")
    print("-" * 78)
    ga = agg["gt"]["mean_angular_acceleration_deg"]
    for v in ("chordal", "ghost"):
        va = agg[v]["mean_angular_acceleration_deg"]
        rel = "smoother than GT" if va < ga else "jerkier than GT"
        print(f"  {label[v]:<11} mean |a| / GT = {va / ga:5.2f}x   ({rel})")
    print()
    print("  per-joint mean |a| (degrees)")
    print(f"    {'joint':<16}{'A chordal':>11}{'B ghost':>11}{'C gt':>11}{'B/A':>8}")
    for j in _JOINT_NAMES:
        c = agg["chordal"]["per_joint_mean_acceleration_deg"][j]
        b = agg["ghost"]["per_joint_mean_acceleration_deg"][j]
        t = agg["gt"]["per_joint_mean_acceleration_deg"][j]
        print(f"    {j:<16}{c:>11.4f}{b:>11.4f}{t:>11.4f}{b / c if c else float('nan'):>8.2f}")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint), "body_split": args.body_split,
            "min_views": args.min_views, "min_run": args.min_run,
            "joint_set": "SMPL-X body joints 1..21 (packed slots 0..20), "
                         "global orientation excluded",
            "n_scenes": len(rows), "cache_dir": str(cache_dir) if cache_dir else None,
        },
        "variants": agg,
        "skipped": [{"scene": s, "reason": r} for s, r in skipped],
        "per_scene": [{"scene": r["scene"], "T": r["T"], "K": r["K"], "P": r["P"],
                       "n_runs": r["n_runs"], "n_w": r["n_w"], "n_a": r["n_a"]}
                      for r in rows],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
