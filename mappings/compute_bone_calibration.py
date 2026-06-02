"""
mappings/compute_bone_calibration.py

Compute per-bone calibration factors that correct the systematic bias in
estimate_scale_triangulated caused by the MHR70 2D detector firing at slightly
different positions than where the SMPL-X vertex regressor places the same
landmarks.

For each RICH training scene with a valid GT calibration:
  1. Compute gt_scale = median(|t_gt_k| / |t_pred_k|) over non-reference cameras.
  2. For each (person, frame, bone): triangulate bone endpoints with unscaled
     VGGT cameras → L_vggt_unscaled.
  3. Compute L_fk via mapper (mean SAM3D betas + per-camera SAM3D body_pose).
  4. Record correction_factor = L_fk / (L_vggt_unscaled × gt_scale).

Output: mappings/bone_scale_calibration.npy
  numpy object array containing a dict {(mhr_a, mhr_b): median_factor}.
  In estimate_scale_triangulated, dividing L_fk by correction_factor[bone]
  before computing s = L_fk / L_vggt removes the systematic bias.

Usage:
    pixi run python mappings/compute_bone_calibration.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from fusion.placer import BodyPlacer, _SCALE_BONE_CANDIDATES
from evaluation.evaluate_on_rich_test import load_gt_extrinsics

GHOST_ROOT  = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train")
RICH_ROOT   = Path("/capstor/scratch/cscs/tnanni/datasets/rich")
SMPLX_MODEL = Path("/users/tnanni/ghost/body_models/SMPLX_NEUTRAL.pkl")
OUT_FILE    = Path(__file__).parent / "bone_scale_calibration.npy"
MIN_CAMS    = 2

_BONE_NAMES = {
    ( 9, 13): "L hip→ankle",   (10, 14): "R hip→ankle",
    ( 9, 11): "L hip→knee",    (10, 12): "R hip→knee",
    (11, 13): "L knee→ankle",  (12, 14): "R knee→ankle",
    ( 7, 62): "L elbow→wrist", ( 8, 41): "R elbow→wrist",
}


def _compute_gt_scale(scene_name: str, cam_names: list[str],
                      pred_exts: np.ndarray, cam_valid: np.ndarray) -> float | None:
    gt_exts_xml = load_gt_extrinsics(scene_name, RICH_ROOT)
    if not gt_exts_xml:
        return None

    ref_idx = int(re.search(r"\d+", cam_names[0]).group())
    if ref_idx >= len(gt_exts_xml):
        return None
    E0 = gt_exts_xml[ref_idx]
    R0, t0 = E0[:, :3], E0[:, 3]

    K = len(cam_names)
    pred_t_med = np.zeros((K, 3), dtype=np.float64)
    for ki in range(K):
        vmask = cam_valid[:, ki].astype(bool)
        if vmask.any():
            pred_t_med[ki] = np.median(
                pred_exts[vmask, ki, :, 3].astype(np.float64), axis=0
            )

    ratios = []
    for ki in range(1, K):
        gidx = int(re.search(r"\d+", cam_names[ki]).group())
        if gidx >= len(gt_exts_xml):
            continue
        Ek = gt_exts_xml[gidx]
        tk_gt = Ek[:, 3] - Ek[:, :3] @ R0.T @ t0
        tp = np.linalg.norm(pred_t_med[ki])
        tg = np.linalg.norm(tk_gt)
        if tp > 1e-6:
            ratios.append(tg / tp)
    if not ratios:
        return None
    s = float(np.median(ratios))
    return s if 0.5 < s < 50.0 else None


def main() -> None:
    scene_dirs = sorted(
        d for d in GHOST_ROOT.iterdir()
        if d.is_dir() and (d / "vggt_cameras.npz").exists()
    )
    print(f"Found {len(scene_dirs)} scenes in {GHOST_ROOT}")

    bone_samples: dict[tuple, list[float]] = defaultdict(list)

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        cam_npz    = np.load(scene_dir / "vggt_cameras.npz", mmap_mode="r")
        pred_exts  = cam_npz["extrinsics"]    # (T, K, 3, 4)
        pred_intrs = cam_npz["intrinsics"]    # (T, K, 3, 3)
        cam_valid  = cam_npz["valid"]         # (T, K) bool
        cam_names  = [n.decode() if isinstance(n, bytes) else n
                      for n in cam_npz["camera_names"]]
        T, K = cam_valid.shape

        gt_scale = _compute_gt_scale(scene_name, cam_names, pred_exts, cam_valid)
        if gt_scale is None:
            print(f"  {scene_name}: no GT calibration — skip")
            continue

        try:
            placer = BodyPlacer(scene_dir, SMPLX_MODEL)
        except Exception as e:
            print(f"  {scene_name}: BodyPlacer error ({e}) — skip")
            continue

        valid_bones = [
            (a, b) for a, b in _SCALE_BONE_CANDIDATES
            if a in placer._mapper._index and b in placer._mapper._index
        ]

        # ── Mean SAM3D betas per pid ──────────────────────────────────────
        betas_sum: dict[int, np.ndarray] = {}
        betas_cnt: dict[int, int] = {}
        for cam_dir in placer._cam_dirs:
            for bf in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(bf.stem.split("_")[1])
                d = np.load(bf, allow_pickle=False)
                if "smplx_betas" not in d.files:
                    continue
                b = d["smplx_betas"].mean(0)
                betas_sum[pid] = betas_sum.get(pid, np.zeros(10, np.float32)) + b
                betas_cnt[pid] = betas_cnt.get(pid, 0) + 1
        mean_betas: dict[int, np.ndarray] = {
            pid: betas_sum[pid] / betas_cnt[pid] for pid in betas_sum
        }

        # ── Precompute batched FK per (cam, pid) ─────────────────────────
        # fk_cache[(k, pid)][local_t] = kp_smplx (N_kps, 3)
        fk_cache: dict[tuple, np.ndarray] = {}   # key → (T_local, N_kps, 3)
        cam_body: dict[int, dict[int, dict]] = {}  # cam_body[k][pid] = {gt_map, kp2d}

        # compute frame_start as min global frame index across all body files
        frame_start = None
        for cam_dir in placer._cam_dirs:
            for bf in sorted((cam_dir / "body_data").glob("person_*.npz")):
                d_fi = np.load(bf, allow_pickle=False)
                if "frame_indices" in d_fi.files and len(d_fi["frame_indices"]):
                    fi_min = int(d_fi["frame_indices"].min())
                    frame_start = fi_min if frame_start is None else min(frame_start, fi_min)
        if frame_start is None:
            frame_start = 0

        # indices into the mapper output that we actually need
        needed_kp_set = set()
        for a, b in valid_bones:
            needed_kp_set.add(placer._mapper.index(a))
            needed_kp_set.add(placer._mapper.index(b))
        needed_kp_idx = sorted(needed_kp_set)           # len ≤ 16
        # remap ia/ib to positions within needed_kp_idx
        needed_pos = {orig: pos for pos, orig in enumerate(needed_kp_idx)}

        for k, cam_dir in enumerate(placer._cam_dirs):
            cam_body[k] = {}
            for bf in sorted((cam_dir / "body_data").glob("person_*.npz")):
                pid = int(bf.stem.split("_")[1])
                d = np.load(bf, allow_pickle=False)
                req = {"pred_keypoints_2d", "frame_indices", "smplx_body_pose"}
                if not req.issubset(d.files):
                    continue
                fi    = d["frame_indices"].astype(int)
                betas = np.tile(
                    mean_betas.get(pid, np.zeros(10, np.float32))[np.newaxis],
                    (len(fi), 1)
                )
                go    = np.zeros((len(fi), 3), np.float32)
                _, verts = placer._smplx_fk(betas, d["smplx_body_pose"], go,
                                             return_verts=True)
                kps_full = placer._mapper.map(verts)    # (T_local, N_kps, 3)
                del verts
                fk_cache[(k, pid)] = kps_full[:, needed_kp_idx, :]  # (T_local, ≤16, 3)
                del kps_full
                cam_body[k][pid] = {
                    "gt_map": {int(g): lt for lt, g in enumerate(fi)},
                    "kp2d":   d["pred_keypoints_2d"],
                }

        # ── Collect calibration samples ───────────────────────────────────
        all_pids: set[int] = set()
        for k in cam_body:
            all_pids.update(cam_body[k].keys())

        frames_by_pid: dict[int, set[int]] = defaultdict(set)
        for k in cam_body:
            for pid, bd in cam_body[k].items():
                frames_by_pid[pid].update(bd["gt_map"].keys())

        # ia_pos/ib_pos: index into the compact fk_cache axis (≤16 kps)
        ia_pos = {bone: needed_pos[placer._mapper.index(bone[0])] for bone in valid_bones}
        ib_pos = {bone: needed_pos[placer._mapper.index(bone[1])] for bone in valid_bones}

        n_scene = 0
        n_frames = sum(len(v) for v in frames_by_pid.values())
        n_processed = 0
        last_pct = -1

        for pid in sorted(all_pids):
            for global_t in sorted(frames_by_pid[pid]):
                vggt_t = global_t - frame_start
                n_processed += 1
                pct = 10 * (n_processed * 10 // n_frames) if n_frames else 0
                if pct != last_pct:
                    print(f"    {scene_name}  {n_processed}/{n_frames} frames "
                          f"({100*n_processed//n_frames if n_frames else 0}%)  "
                          f"samples={n_scene}", flush=True)
                    last_pct = pct

                if vggt_t < 0 or vggt_t >= T:
                    continue

                for bone in valid_bones:
                    mhr_a, mhr_b = bone
                    pts_a, pts_b, Pmats, fk_lens = [], [], [], []

                    for k in range(K):
                        if not cam_valid[vggt_t, k]:
                            continue
                        bd = cam_body.get(k, {}).get(pid)
                        if bd is None or global_t not in bd["gt_map"]:
                            continue
                        lt = bd["gt_map"][global_t]

                        if (mhr_a >= bd["kp2d"].shape[1] or
                                mhr_b >= bd["kp2d"].shape[1]):
                            continue

                        oc = placer.original_coords[vggt_t, k]
                        Wo = float(placer.original_size[vggt_t, k, 0])
                        Ho = float(placer.original_size[vggt_t, k, 1])

                        ua, va = placer._orig_to_vggt(
                            bd["kp2d"][lt, mhr_a], oc, Wo, Ho)
                        ub, vb = placer._orig_to_vggt(
                            bd["kp2d"][lt, mhr_b], oc, Wo, Ho)
                        if not (placer._in_bounds(ua, va, oc[2], oc[3]) and
                                placer._in_bounds(ub, vb, oc[2], oc[3])):
                            continue

                        P = (pred_intrs[vggt_t, k].astype(np.float64) @
                             pred_exts[vggt_t, k].astype(np.float64))
                        Pmats.append(P)
                        pts_a.append((ua, va))
                        pts_b.append((ub, vb))

                        kp = fk_cache.get((k, pid))
                        if kp is not None:
                            ia = ia_pos[bone]
                            ib = ib_pos[bone]
                            L = float(np.linalg.norm(kp[lt, ib] - kp[lt, ia]))
                            if L > 0.01:
                                fk_lens.append(L)

                    if len(pts_a) < MIN_CAMS or not fk_lens:
                        continue

                    try:
                        Xa = placer._triangulate_dlt(pts_a, Pmats)
                        Xb = placer._triangulate_dlt(pts_b, Pmats)
                    except Exception:
                        continue

                    L_vggt = float(np.linalg.norm(Xb - Xa))
                    if L_vggt < 1e-4:
                        continue

                    L_fk = float(np.median(fk_lens))
                    factor = L_fk / (L_vggt * gt_scale)
                    if 0.3 < factor < 3.0:
                        bone_samples[bone].append(factor)
                        n_scene += 1

        # ── Per-bone running stats after this scene ───────────────────────
        print(f"  {scene_name}: gt_scale={gt_scale:.3f}  samples={n_scene}")
        header = f"  {'bone':<18}  {'factor':>7}  {'corr':>7}  {'std':>6}  {'N':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for bone in valid_bones:
            s = bone_samples.get(bone, [])
            if not s:
                print(f"  {_BONE_NAMES.get(bone, str(bone)):<18}  {'—':>7}")
                continue
            med = float(np.median(s))
            std = float(np.std(s))
            print(f"  {_BONE_NAMES.get(bone, str(bone)):<18}  {med:7.4f}  "
                  f"{1/med:7.4f}  {std:6.4f}  {len(s):6d}")
        print()

    # ── Per-bone calibration table ─────────────────────────────────────────
    print("\nPer-bone calibration factors (median L_fk / (L_vggt × gt_scale)):")
    calibration: dict[tuple, float] = {}
    for bone in _SCALE_BONE_CANDIDATES:
        samples = bone_samples.get(bone, [])
        if not samples:
            print(f"  {_BONE_NAMES.get(bone, str(bone)):<18}: NO DATA — defaulting to 1.0")
            calibration[bone] = 1.0
            continue
        med = float(np.median(samples))
        std = float(np.std(samples))
        calibration[bone] = med
        correction = 1.0 / med
        print(f"  {_BONE_NAMES.get(bone, str(bone)):<18}: factor={med:.4f}  "
              f"correction=×{correction:.4f}  std={std:.4f}  N={len(samples)}")

    np.save(OUT_FILE, calibration)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
