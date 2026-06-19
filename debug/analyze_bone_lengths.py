"""Analyze bone length estimation error for scale computation.

For each matched (ghost_pid, gt_pid) across frames:
  - Compute FK bone lengths from fused model output (pred)
  - Compute FK bone lengths from GT body params
  - Compare → tells us if overestimation comes from FK reference or from triangulation

Long bones used for scale estimation:
  (1,7)   L-hip → L-ankle
  (2,8)   R-hip → R-ankle
  (16,20) L-shoulder → L-wrist
  (17,21) R-shoulder → R-wrist

Usage:
    pixi run python debug/analyze_bone_lengths.py \\
        --scene Gym_010_lunge2 \\
        --ghost_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_test \\
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \\
        --checkpoint checkpoints/fusion_module_latest/best.pt \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.evaluate_on_rich_test import (
    load_fusion_model,
    load_gt_body_data,
    build_fusion_tensors,
    load_scene_body_data,
    _aa_to_6d,
    _6d_to_aa,
)
from fusion.placer import BodyPlacer

# Long bones: (smplx_j_a, smplx_j_b, name)
_LONG_BONES = [
    (1,  7,  "L hip→ankle"),
    (2,  8,  "R hip→ankle"),
    (16, 20, "L sho→wrist"),
    (17, 21, "R sho→wrist"),
]


def bone_len(J: np.ndarray, ja: int, jb: int) -> float:
    return float(np.linalg.norm(J[jb] - J[ja]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",       required=True)
    parser.add_argument("--ghost_root",  required=True, type=Path)
    parser.add_argument("--rich_root",   required=True, type=Path)
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--smplx_model", required=True, type=Path)
    parser.add_argument("--gt_split",    default="test")
    parser.add_argument("--n_frames",    type=int, default=20,
                        help="Number of equally-spaced frames to sample.")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scene_dir = args.ghost_root / args.scene
    device    = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    model  = load_fusion_model(args.checkpoint, device)
    placer = BodyPlacer(scene_dir, args.smplx_model)

    # ── Load body data + fusion tensors ───────────────────────────────────────
    cam_dirs, raw = load_scene_body_data(scene_dir)
    pose_t, mask_t, shape_t, all_pids, frame_start = build_fusion_tensors(raw)
    T   = pose_t.shape[1]
    pid_to_slot = {pid: i for i, pid in enumerate(all_pids)}

    with torch.no_grad():
        fused_pose_t, _ = model(
            pose_t.to(device), mask_t.to(device), shape=shape_t.to(device)
        )
    fused_pose  = fused_pose_t[0].cpu().numpy()   # (T, P, 54, 6)

    _betas_lists: dict[int, list[np.ndarray]] = {}
    for cam_dir in cam_dirs:
        for pid in all_pids:
            bf = cam_dir / "body_data" / f"person_{pid}.npz"
            if bf.exists():
                d = np.load(bf, allow_pickle=False)
                if "smplx_betas" in d.files:
                    _betas_lists.setdefault(pid, []).append(d["smplx_betas"].mean(0))
    betas_by_pid: dict[int, np.ndarray] = {
        pid: np.mean(v, axis=0).astype(np.float32)
        for pid, v in _betas_lists.items()
    }
    for pid in all_pids:
        betas_by_pid.setdefault(pid, np.zeros(10, dtype=np.float32))

    # ── GT body data ──────────────────────────────────────────────────────────
    gt_body_data = load_gt_body_data(args.scene, args.rich_root, split=args.gt_split)
    if not gt_body_data:
        print("No GT found."); return

    # Simple pid matching: ghost pid 1 → only gt pid (single person scenes)
    gt_pid = sorted(gt_body_data.keys())[0]
    ghost_pid = all_pids[0]  # assume single person
    print(f"Matching ghost_pid={ghost_pid} → gt_pid={gt_pid}\n")

    p_slot    = pid_to_slot[ghost_pid]
    betas_p   = betas_by_pid[ghost_pid]
    gt_frames = sorted(gt_body_data[gt_pid].keys())

    # Sample n_frames equally spaced
    idxs   = np.linspace(0, len(gt_frames) - 1, args.n_frames, dtype=int)
    frames = [gt_frames[i] for i in idxs]

    # ── Header ────────────────────────────────────────────────────────────────
    bone_names = [b[2] for b in _LONG_BONES]
    hdr = f"{'Frame':>6}  " + "  ".join(
        f"{'GT':>6} {'Pred':>6} {'Err%':>6}" for _ in bone_names
    )
    sep = "-" * len(hdr)
    print("  " + "  ".join(f"{n:>20}" for n in bone_names))
    print(f"{'Frame':>6}  " + "  ".join(f"{'GT(cm)':>6} {'Pr(cm)':>6} {'Err%':>6}" for _ in bone_names))
    print(sep)

    errs_pct: dict[str, list[float]] = {b[2]: [] for b in _LONG_BONES}

    for fi in frames:
        t_rel = fi - frame_start
        if not (0 <= t_rel < T):
            continue
        params = gt_body_data[gt_pid][fi]

        # GT FK
        J_gt = placer._smplx_fk(
            params["betas"][np.newaxis],
            params["body_pose"][np.newaxis],
            params["global_orient"][np.newaxis],
        )[0]  # (55,3) zero transl

        # Pred FK: fused body_pose (slots 0..20 = body joints 1..21)
        body_pose_aa = _6d_to_aa(fused_pose[t_rel, p_slot, :21])  # (21,3)
        J_pred = placer._smplx_fk(
            betas_p[np.newaxis],
            body_pose_aa.reshape(63)[np.newaxis],
            np.zeros((1, 3), dtype=np.float32),
        )[0]

        row = f"{fi:>6}  "
        for ja, jb, name in _LONG_BONES:
            l_gt   = bone_len(J_gt,   ja, jb) * 100  # cm
            l_pred = bone_len(J_pred, ja, jb) * 100
            err_pct = (l_pred - l_gt) / l_gt * 100
            errs_pct[name].append(err_pct)
            row += f"  {l_gt:6.1f} {l_pred:6.1f} {err_pct:+6.1f}%"
        print(row)

    print(sep)
    row = f"{'Mean':>6}  "
    for _, _, name in _LONG_BONES:
        v = errs_pct[name]
        row += f"  {'—':>6} {'—':>6} {np.mean(v):+6.1f}%" if v else f"  {'—':>20}"
    print(row)
    row = f"{'Std':>6}  "
    for _, _, name in _LONG_BONES:
        v = errs_pct[name]
        row += f"  {'—':>6} {'—':>6} {np.std(v):6.1f}% " if v else f"  {'—':>20}"
    print(row)


if __name__ == "__main__":
    main()
