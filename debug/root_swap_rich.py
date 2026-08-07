"""Root/pose component swap (2x2) on RICH test — where does ghost's world edge live?

QUESTION
--------
The trained fusion module ("ghost") has WORSE per-frame pose than a chordal
rotation mean (PA-MPJPE 30.4 vs 26.5 mm) yet BETTER temporally-integrated world
metrics (W-MPJPE-100 70.4 vs 84.0, WA-100 50.4 vs 55.1, RTE 1.00 vs 1.39 %).
Hypothesis: the advantage lives entirely in the ROOT trajectory the placer
derives from the fused skeleton (stable-but-biased skeleton -> stable Procrustes
root), not in the pose itself.

DESIGN
------
2x2 component swap: pose_X in {chordal, ghost} x root_Y in {chordal, ghost}.
root_Y = BodyPlacer.estimate_procrustes_dlt_mhr output when `fused_pose_by_pid`
is variant Y (the DLT half is pose-independent; only the FK'd canonical skeleton
fed to Procrustes changes). World joints are assembled from (pose_X FK, root_Y)
exactly as evaluation/evaluate_rich.py step 7 does (M10: neutral-pred FK vs
gendered-GT, SMPL-24 joints via smplx2smpl + J_regressor, x' = R@(x-J0_smplx)+t
placement convention), and all four combos are scored with the evaluator's own
metric functions (IMPORTED, not reimplemented) on ONE shared validity mask
(cache `valid` AND GT finite AND both root variants placed the frame).

CONFIRMS the hypothesis: (chordal pose + ghost root) ~ full ghost on
W-MPJPE-100 / RTE — swapping in the ghost root recovers (nearly) the whole
world-metric edge even under the better chordal pose.

REFUTES it: (chordal pose + ghost root) ~ full chordal — then the edge needs
the ghost POSE at placement-and-scoring time, not just its root trajectory.

INPUT
-----
Cached fused poses written by evaluation/temporal_smoothness_rich.py:
<cache_dir>/<scene>.npz with pose_chordal / pose_ghost / pose_gt (T, P, 54, 6),
`valid` (T, P), `person_ids` (P,) (= the P-axis order), `rich_to_ghost` (N, 2)
[rich_gt_pid, ghost_pid] (GT matching ALREADY done — reused, never re-matched),
`frame_start` / `frame_indices`. Plus the per-scene ghost outputs
(body_data, vggt_cameras_centered.npz, mapanything scale) and RICH GT.

DIAGNOSTICS
-----------
Per root variant (chordal, ghost) and for the GT root trajectory:
mean/median |second difference| of pelvis translation (jitter), angular
velocity + acceleration of R_root, and per-100-frame-segment drift of the root
translation ERROR (pred - GT-in-ref-frame; zero for GT by construction).
Computed on contiguous valid runs only. Segments come from
evaluate_rich._iter_segments (imported).

Usage
-----
    OMP_NUM_THREADS=8 pixi run python evaluation/root_swap_rich.py \\
        --cache_dir         /iopsstor/scratch/cscs/tnanni/ghost_outputs/fused_cache/rich_test \\
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --rich_root         /capstor/scratch/cscs/tnanni/datasets/rich \\
        --centered_root     /tmp/centered_test \\
        --device            cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Metrics + segmentation + GT loading come from the evaluator itself, not a copy.
from evaluation.evaluate_rich import (
    _6d_to_aa,
    _iter_segments,
    _N_SMPL_JOINTS,
    _verts_to_smpl24,
    load_gt_body_data,
    load_gt_extrinsics,
    metric_ga_mpjpe,
    metric_pa_mpjpe,
    metric_rte,
    metric_w_mpjpe,
    metric_w_mpjpe_100,
    metric_wa_mpjpe,
    metric_wa_mpjpe_100,
)
from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_POSES = ("chordal", "ghost")
_ROOTS = ("chordal", "ghost")
_METRIC_KEYS = ("wa", "w", "wa100", "w100", "ga", "pa", "rte")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fk_j24_batch(
    placer: BodyPlacer,
    betas: np.ndarray,        # (10,) or (N, 10)
    body_pose: np.ndarray,    # (N, 63) axis-angle
    global_orient: np.ndarray | None = None,   # (N, 3) or None -> zeros
    chunk: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched FK through the placer's SMPL-X model (same model `_smplx_fk` with
    no pid would pick, i.e. mirrors evaluate_rich's calls exactly).

    Returns (J55, J24): SMPL-X joints (N, 55, 3) and SMPL-24 joints (N, 24, 3),
    both in the model's canonical/world frame (transl NOT applied).
    """
    N = body_pose.shape[0]
    if betas.ndim == 1:
        betas = np.broadcast_to(betas, (N, 10)).copy()
    go = np.zeros((N, 3), dtype=np.float32) if global_orient is None else global_orient
    J55 = np.empty((N, 55, 3), dtype=np.float32)
    J24 = np.empty((N, _N_SMPL_JOINTS, 3), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        j, v = placer._smplx_fk(
            betas[s:e].astype(np.float32),
            body_pose[s:e].astype(np.float32),
            go[s:e].astype(np.float32),
            return_verts=True,
        )
        J55[s:e] = j
        for i in range(e - s):
            J24[s + i] = _verts_to_smpl24(v[i])
    return J55, J24


def _contiguous_runs(valid: np.ndarray, min_run: int = 2) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a 1-D bool array, length >= min_run."""
    if not valid.any():
        return []
    padded = np.concatenate(([False], valid.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_run]


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between (N,3,3) rotation pairs, degrees."""
    tr = np.einsum("nab,nab->n", Ra, Rb)          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _root_diagnostics(
    roots: np.ndarray,               # (T, P, 3) NaN outside valid
    rots: np.ndarray,                # (T, P, 3, 3) NaN outside valid
    err_vs_gt: np.ndarray | None,    # (T, P, 3) pred_root - gt_root_ref, or None
    valid: np.ndarray,               # (T, P)
    segment_len: int = 100,
) -> dict:
    """Jitter / angular velocity / angular acceleration / per-segment error drift.

    All quantities on CONTIGUOUS valid runs only, pooled over persons and runs.
    """
    T, P = valid.shape
    jit, angv, anga = [], [], []
    drifts, drift_rates = [], []
    for p in range(P):
        for s, e in _contiguous_runs(valid[:, p], min_run=2):
            x = roots[s:e, p].astype(np.float64)          # (L, 3)
            R = rots[s:e, p].astype(np.float64)           # (L, 3, 3)
            if e - s >= 3:
                d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
                jit.append(np.linalg.norm(d2, axis=-1) * 1000.0)      # mm
            w = _geodesic_deg(R[:-1], R[1:])                          # (L-1,)
            angv.append(w)
            if w.size >= 2:
                anga.append(np.abs(w[1:] - w[:-1]))
        if err_vs_gt is not None:
            for t0, t1 in _iter_segments(valid[:, p : p + 1], segment_len):
                for s, e in _contiguous_runs(valid[t0:t1, p], min_run=10):
                    seg_e = err_vs_gt[t0 + s : t0 + e, p].astype(np.float64)
                    d = float(np.linalg.norm(seg_e[-1] - seg_e[0])) * 1000.0  # mm
                    drifts.append(d)
                    drift_rates.append(d / (e - s - 1))
    def _mm(vals):
        v = np.concatenate(vals) if vals and isinstance(vals[0], np.ndarray) else np.asarray(vals)
        if v.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {"mean": float(v.mean()), "median": float(np.median(v)), "n": int(v.size)}
    out = {
        "transl_jitter_mm": _mm(jit),
        "root_ang_vel_deg": _mm(angv),
        "root_ang_acc_deg": _mm(anga),
    }
    if err_vs_gt is not None:
        out["seg_err_drift_mm"] = _mm(drifts)
        out["seg_err_drift_mm_per_frame"] = _mm(drift_rates)
    return out


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    scene_name: str,
    cache_path: Path,
    ghost_output_root: Path,
    rich_root: Path,
    centered_root: Path,
    smplx_model_path: Path,
    gt_split: str,
    scale_mode: str,
    scale_smooth: str,
) -> dict | None:
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Scene: {scene_name}")
    scene_dir = ghost_output_root / scene_name

    if not (scene_dir / "vggt_cameras_centered.npz").exists():
        logger.warning("  missing vggt_cameras_centered.npz — skipping")
        return None
    crop_meta_path = centered_root / scene_name / "crop_meta.json"
    if not crop_meta_path.exists():
        logger.warning(f"  missing {crop_meta_path} — skipping")
        return None

    # ── 1. Cache ────────────────────────────────────────────────────────────
    d = np.load(cache_path, allow_pickle=False)
    pose = {v: d[f"pose_{v}"].astype(np.float32) for v in _POSES}   # (T,P,54,6)
    cache_valid = d["valid"].astype(bool)                            # (T,P)
    person_ids = [int(p) for p in d["person_ids"]]                   # P-axis order
    frame_start = int(d["frame_start"])
    T, P = cache_valid.shape
    rich_to_ghost = {int(r): int(g) for r, g in d["rich_to_ghost"]}  # rich → ghost
    ghost_to_rich = {g: r for r, g in rich_to_ghost.items()}
    if sorted(ghost_to_rich) != person_ids:
        logger.warning(
            f"  person_ids {person_ids} != matched ghost pids "
            f"{sorted(ghost_to_rich)} — skipping"
        )
        return None
    pid_to_slot = {pid: i for i, pid in enumerate(person_ids)}

    # ── 2. Placers — mirrors evaluate_rich: gendered for GT FK, neutral for pred
    gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    smplx_arg = (
        resolve_smplx_models(scene_name, smplx_model_path.parent, gender_json)
        if gender_json.exists() else smplx_model_path
    )
    try:
        placer_gendered = BodyPlacer(scene_dir, smplx_arg, crop_meta_path=crop_meta_path)
        neutral_path = smplx_model_path.parent / "SMPLX_NEUTRAL.pkl"
        placer_neutral = BodyPlacer(scene_dir, neutral_path, crop_meta_path=crop_meta_path)
    except Exception as e:
        logger.warning(f"  BodyPlacer init failed: {e} — skipping")
        return None
    if placer_neutral.T != T:
        logger.info(
            f"  note: cache T={T} vs VGGT T={placer_neutral.T} "
            f"(placer skips frames beyond VGGT range)"
        )

    # ── 3. Per-pid mean SAM3D betas (mirrors evaluate_rich) ─────────────────
    betas_lists: dict[int, list[np.ndarray]] = {}
    for cam_dir in placer_neutral._cam_dirs:
        for pid in person_ids:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if bf.exists():
                bd = np.load(bf, allow_pickle=False)
                if "smplx_betas" in bd.files:
                    betas_lists.setdefault(pid, []).append(bd["smplx_betas"].mean(0))
    betas_by_pid = {
        pid: np.mean(v, axis=0).astype(np.float32) for pid, v in betas_lists.items()
    }
    for pid in person_ids:
        betas_by_pid.setdefault(pid, np.zeros(10, dtype=np.float32))

    # ── 4. Scale — same default path as evaluate_rich ───────────────────────
    fused_by_pid = {
        v: {pid: pose[v][:, pid_to_slot[pid]] for pid in person_ids} for v in _POSES
    }
    pred_scale = placer_neutral.load_mapanything_scale(
        scale_mode=scale_mode, smooth=scale_smooth
    )
    if pred_scale is not None:
        logger.info(
            f"  [scale] MapAnything ({scale_mode}, smooth={scale_smooth})  "
            f"median={float(np.median(pred_scale)):.4f}"
        )
    else:
        pred_scale = placer_neutral.estimate_scale_triangulated(
            fused_pose_by_pid=fused_by_pid["ghost"], frame_start=frame_start
        )
        logger.info(f"  [scale] triangulated  median={float(np.median(pred_scale)):.4f}")

    # ── 5. Placer TWICE — root_Y from fused pose variant Y ──────────────────
    trans, orient = {}, {}
    for v in _ROOTS:
        try:
            trans[v], orient[v] = placer_neutral.estimate_procrustes_dlt_mhr(
                scale=pred_scale,
                all_pids=set(person_ids),
                pred_betas_by_pid=betas_by_pid,
                fused_pose_by_pid=fused_by_pid[v],
                frame_start=frame_start,
            )
        except Exception as e:
            logger.warning(f"  placer({v}) failed: {e} — skipping")
            return None

    # ── 6. GT joints — gendered model, mirrors evaluate_rich._build_gt_joints
    gt_body_data = load_gt_body_data(scene_name, rich_root, split=gt_split)
    if not gt_body_data:
        logger.warning(f"  no GT in {gt_split}_body/ — skipping")
        return None
    J_body = _N_SMPL_JOINTS
    gt_joints = np.full((T, P, J_body, 3), np.nan, dtype=np.float32)
    gt_roots = np.full((T, P, 3), np.nan, dtype=np.float32)
    gt_rot_world = np.full((T, P, 3, 3), np.nan, dtype=np.float32)
    for pid in person_ids:
        slot = pid_to_slot[pid]
        gt_pid = ghost_to_rich[pid]
        frames = sorted(
            f for f in gt_body_data.get(gt_pid, {}) if 0 <= f - frame_start < T
        )
        if not frames:
            continue
        prm = [gt_body_data[gt_pid][f] for f in frames]
        _, J24 = _fk_j24_batch(
            placer_gendered,
            np.stack([p["betas"] for p in prm]),
            np.stack([p["body_pose"] for p in prm]),
            np.stack([p["global_orient"] for p in prm]),
        )
        for i, f in enumerate(frames):
            t = f - frame_start
            gt_joints[t, slot] = J24[i] + prm[i]["transl"]
            gt_roots[t, slot] = gt_joints[t, slot, 0]
            gt_rot_world[t, slot] = SciR.from_rotvec(prm[i]["global_orient"]).as_matrix()

    # RICH-world → VGGT-ref transform (for the raw root-error diagnostics only).
    gt_exts = load_gt_extrinsics(scene_name, rich_root)
    ref_idx = (
        int(re.search(r"\d+", placer_neutral._cam_dirs[0].name).group())
        if placer_neutral._cam_dirs else 0
    )
    if gt_exts and ref_idx < len(gt_exts):
        E0 = gt_exts[ref_idx].astype(np.float64)
        R_w2ref, t_w2ref = E0[:3, :3], E0[:3, 3]
    else:
        R_w2ref, t_w2ref = np.eye(3), np.zeros(3)

    # ── 7. FK each pose variant ONCE per pid (root-independent), then place ──
    # Frames worth FK-ing: union of the two root dicts per pid.
    pred_joints = {
        (px, ry): np.full((T, P, J_body, 3), np.nan, dtype=np.float32)
        for px in _POSES for ry in _ROOTS
    }
    pred_roots = {
        ry: np.full((T, P, 3), np.nan, dtype=np.float32) for ry in _ROOTS
    }
    root_rot = {
        ry: np.full((T, P, 3, 3), np.nan, dtype=np.float32) for ry in _ROOTS
    }
    for pid in person_ids:
        slot = pid_to_slot[pid]
        frames = sorted(
            set(trans["chordal"].get(pid, {})) | set(trans["ghost"].get(pid, {}))
        )
        frames = [f for f in frames if 0 <= f - frame_start < T]
        if not frames:
            continue
        t_rels = np.array([f - frame_start for f in frames], dtype=int)
        betas_p = betas_by_pid[pid]
        for px in _POSES:
            bp_aa = _6d_to_aa(pose[px][t_rels, slot, :21]).reshape(len(frames), 63)
            J55, J24 = _fk_j24_batch(placer_neutral, betas_p, bp_aa)
            pelvis_smplx = J55[:, 0]                       # (N, 3) SMPL-X canonical pelvis
            for ry in _ROOTS:
                tr_d, or_d = trans[ry].get(pid, {}), orient[ry].get(pid, {})
                for i, f in enumerate(frames):
                    pelvis_world = tr_d.get(f)
                    R_mat = or_d.get(f)
                    if pelvis_world is None or R_mat is None:
                        continue
                    Jw = (R_mat @ (J24[i] - pelvis_smplx[i]).T).T + pelvis_world
                    pred_joints[(px, ry)][t_rels[i], slot] = Jw
                    if px == "chordal":   # root arrays are pose-independent
                        pred_roots[ry][t_rels[i], slot] = Jw[0]
                        root_rot[ry][t_rels[i], slot] = R_mat

    # NOTE: the DIAGNOSTICS root arrays (pred_roots/root_rot) use the
    # chordal-pose placement for both root variants, so the root-axis
    # comparison is not polluted by the (mm-level) pose-dependent J24-pelvis
    # offset. RTE below instead uses each combo's own placed pelvis
    # (pred_joints[..., 0, :]), which reproduces evaluate_rich exactly on the
    # diagonal combos.

    # ── 8. ONE shared validity mask for all four combos ─────────────────────
    finite_gt = np.isfinite(gt_joints).all((-2, -1))
    finite_root = {
        ry: np.isfinite(pred_joints[("chordal", ry)]).all((-2, -1)) for ry in _ROOTS
    }
    valid = cache_valid & finite_gt & finite_root["chordal"] & finite_root["ghost"]
    n_valid = int(valid.sum())
    if n_valid < 4:
        logger.warning(f"  only {n_valid} valid (t,p) — skipping")
        return None

    # ── 9. Metrics per combo (shared mask; NaN-masked roots for RTE) ────────
    gt_roots_m = np.where(valid[..., None], gt_roots, np.nan)
    results: dict[str, dict[str, float]] = {}
    for px in _POSES:
        for ry in _ROOTS:
            pj = pred_joints[(px, ry)]
            pr = np.where(valid[..., None], pj[:, :, 0, :], np.nan)
            results[f"{px}+{ry}"] = {
                "wa":    metric_wa_mpjpe(pj, gt_joints, valid),
                "w":     metric_w_mpjpe(pj, gt_joints, valid),
                "wa100": metric_wa_mpjpe_100(pj, gt_joints, valid),
                "w100":  metric_w_mpjpe_100(pj, gt_joints, valid),
                "ga":    metric_ga_mpjpe(pj, gt_joints, valid),
                "pa":    metric_pa_mpjpe(pj, gt_joints, valid),
                "rte":   metric_rte(pr, gt_roots_m),
            }

    # ── 10. Root diagnostics (pred variants in ref frame, GT in world) ──────
    gt_root_ref = np.einsum("ab,tpb->tpa", R_w2ref, gt_roots.astype(np.float64)) + t_w2ref
    diagnostics = {}
    for ry in _ROOTS:
        err = pred_roots[ry].astype(np.float64) - gt_root_ref
        diagnostics[ry] = _root_diagnostics(pred_roots[ry], root_rot[ry], err, valid)
    gt_rot_ref = np.einsum(
        "ab,tpbc->tpac", R_w2ref, gt_rot_world.astype(np.float64)
    ).astype(np.float32)
    diagnostics["gt"] = _root_diagnostics(gt_roots, gt_rot_ref, None, valid)

    line = "  " + "  ".join(
        f"{c}: W100={m['w100']:6.1f} PA={m['pa']:5.1f} RTE={m['rte']:.2f}"
        for c, m in results.items()
    )
    logger.info(line)
    return {
        "scene": scene_name, "n_valid": n_valid, "T": T, "P": P,
        "combos": results, "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="2x2 pose/root component swap on RICH test (chordal vs ghost)."
    )
    ap.add_argument("--cache_dir", required=True, type=Path,
                    help="Dir of <scene>.npz fused-pose caches "
                         "(written by temporal_smoothness_rich.py).")
    ap.add_argument("--ghost_output_root", required=True, type=Path)
    ap.add_argument("--rich_root", required=True, type=Path)
    ap.add_argument("--centered_root", required=True, type=Path,
                    help="Dir holding <scene>/crop_meta.json (node-local mount).")
    ap.add_argument("--smplx_model", type=Path,
                    default=_REPO_ROOT / "body_models" / "SMPLX_NEUTRAL.pkl")
    ap.add_argument("--device", default="cpu",
                    help="Accepted for CLI parity; everything here runs on CPU.")
    ap.add_argument("--gt_split", default="test")
    ap.add_argument("--scale", default="baseline", choices=["centered", "baseline"])
    ap.add_argument("--scale_smooth", default="none", choices=["none", "median"])
    ap.add_argument("--scenes", default="", help="comma-separated scene names")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--out_json", type=Path,
                    default=Path("eval_explainability/root_swap_rich.json"))
    args = ap.parse_args()

    wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
    cache_files = sorted(args.cache_dir.glob("*.npz"))
    if wanted:
        cache_files = [f for f in cache_files if f.stem in wanted]
    if args.max_scenes:
        cache_files = cache_files[: args.max_scenes]
    logger.info(f"{len(cache_files)} cached scene(s) in {args.cache_dir}")

    rows, skipped = [], []
    for f in cache_files:
        try:
            r = evaluate_scene(
                scene_name=f.stem, cache_path=f,
                ghost_output_root=args.ghost_output_root,
                rich_root=args.rich_root, centered_root=args.centered_root,
                smplx_model_path=args.smplx_model, gt_split=args.gt_split,
                scale_mode=args.scale, scale_smooth=args.scale_smooth,
            )
        except Exception as e:                                    # noqa: BLE001
            logger.exception(f"{f.stem}: failed")
            r, e_str = None, f"{type(e).__name__}: {e}"
            skipped.append((f.stem, e_str))
            continue
        if r is None:
            skipped.append((f.stem, "skipped (see log)"))
        else:
            rows.append(r)

    if not rows:
        logger.error("no scenes evaluated")
        return

    # ── Aggregate: mean over scenes per combo (same as evaluate_rich's agg) ──
    combos = [f"{px}+{ry}" for px in _POSES for ry in _ROOTS]
    agg = {
        c: {
            k: float(np.mean([r["combos"][c][k] for r in rows
                              if np.isfinite(r["combos"][c][k])]))
            for k in _METRIC_KEYS
        }
        for c in combos
    }

    def _diag_agg(variant: str, quantity: str, stat: str) -> float:
        vals = [
            r["diagnostics"][variant][quantity][stat]
            for r in rows
            if quantity in r["diagnostics"][variant]
            and np.isfinite(r["diagnostics"][variant][quantity][stat])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    # ── Guards ──────────────────────────────────────────────────────────────
    guards = {}
    # (a) PA-MPJPE must be root-invariant for a fixed pose (tol 0.2 mm), per scene.
    max_pa_gap = max(
        abs(r["combos"][f"{px}+chordal"]["pa"] - r["combos"][f"{px}+ghost"]["pa"])
        for r in rows for px in _POSES
    )
    guards["a_pa_root_invariance"] = {
        "pass": bool(max_pa_gap <= 0.2), "max_gap_mm": float(max_pa_gap), "tol_mm": 0.2,
    }
    # (b) known ordering on the diagonal.
    pa_ok = agg["chordal+chordal"]["pa"] < agg["ghost+ghost"]["pa"]
    w100_ok = agg["ghost+ghost"]["w100"] < agg["chordal+chordal"]["w100"]
    guards["b_known_ordering"] = {
        "pass": bool(pa_ok and w100_ok),
        "pa_chordal_lt_ghost": bool(pa_ok),
        "w100_ghost_lt_chordal": bool(w100_ok),
    }
    # (c) segment windowing is the evaluator's own function.
    guards["c_segments_imported"] = {
        "pass": _iter_segments.__module__ == "evaluation.evaluate_rich",
        "module": _iter_segments.__module__,
    }

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"ROOT/POSE 2x2 SWAP — RICH test, {len(rows)} scene(s), shared validity mask, "
          f"M10 neutral-pred vs gendered-GT")
    print(f"{'=' * 100}")
    corner = "pose \\ root"
    hdr = f"{corner:<16}" + "".join(f"{ry:>42}" for ry in _ROOTS)
    print(hdr)
    print(f"{'':16}" + "".join(f"{'WA / W / WA100 / W100 / GA / PA / RTE':>42}" for _ in _ROOTS))
    print("-" * 100)
    for px in _POSES:
        cells = []
        for ry in _ROOTS:
            m = agg[f"{px}+{ry}"]
            cells.append(
                f"{m['wa']:6.1f} {m['w']:6.1f} {m['wa100']:6.1f} "
                f"{m['w100']:6.1f} {m['ga']:5.1f} {m['pa']:5.1f} {m['rte']:5.2f}%"
            )
        print(f"{px:<16}" + "".join(f"{c:>42}" for c in cells))
    print("-" * 100)

    print("\nROOT DIAGNOSTICS (mean over scenes of per-scene aggregates)")
    print(f"{'quantity':<38}{'chordal root':>16}{'ghost root':>16}{'GT root':>16}")
    for q, s in [
        ("transl_jitter_mm", "mean"), ("transl_jitter_mm", "median"),
        ("root_ang_vel_deg", "mean"), ("root_ang_vel_deg", "median"),
        ("root_ang_acc_deg", "mean"), ("root_ang_acc_deg", "median"),
        ("seg_err_drift_mm", "mean"), ("seg_err_drift_mm", "median"),
        ("seg_err_drift_mm_per_frame", "mean"),
    ]:
        row = f"{q + ' (' + s + ')':<38}"
        for v in ("chordal", "ghost", "gt"):
            row += f"{_diag_agg(v, q, s):>16.3f}"
        print(row)

    print("\nPER-SCENE W-MPJPE-100 (mm)")
    print(f"{'scene':<40}" + "".join(f"{c:>15}" for c in combos) + f"{'n_valid':>9}")
    for r in rows:
        print(f"{r['scene']:<40}"
              + "".join(f"{r['combos'][c]['w100']:>15.1f}" for c in combos)
              + f"{r['n_valid']:>9}")

    print("\nGUARDS")
    for name, g in guards.items():
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {name}: "
              f"{ {k: v for k, v in g.items() if k != 'pass'} }")
    if skipped:
        print("\nSKIPPED SCENES")
        for s, why in skipped:
            print(f"  {s}: {why}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "cache_dir": str(args.cache_dir),
            "ghost_output_root": str(args.ghost_output_root),
            "rich_root": str(args.rich_root),
            "centered_root": str(args.centered_root),
            "smplx_model": str(args.smplx_model),
            "gt_split": args.gt_split, "scale": args.scale,
            "scale_smooth": args.scale_smooth, "n_scenes": len(rows),
            "joint_set": "SMPL-24 via smplx2smpl + J_regressor (evaluate_rich convention)",
            "validity": "cache valid & GT finite & BOTH root variants placed (shared mask)",
        },
        "aggregate": agg,
        "diagnostics_aggregate": {
            v: {
                q: {s: _diag_agg(v, q, s) for s in ("mean", "median")}
                for q in ("transl_jitter_mm", "root_ang_vel_deg", "root_ang_acc_deg",
                          "seg_err_drift_mm", "seg_err_drift_mm_per_frame")
            }
            for v in ("chordal", "ghost", "gt")
        },
        "guards": guards,
        "per_scene": rows,
        "skipped": [{"scene": s, "reason": w} for s, w in skipped],
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
