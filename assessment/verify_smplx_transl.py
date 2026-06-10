"""One-shot script: verify which SMPL-X transl formula is correct.

Question: given GT pkl (transl, global_orient, betas, body_pose), which holds?
  (A) pelvis_world = J_can[0] + transl          (rotation-around-pelvis)
  (B) pelvis_world = R_gt @ J_can[0] + transl   (rotation-around-origin)

Method: call smplx with the ACTUAL transl (non-zero) and compare joint-0.

Usage:
    pixi run python assessment/verify_smplx_transl.py \
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \
        --smplx_model body_models/SMPLX_NEUTRAL.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smplx as smplx_lib
from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   required=True)
    p.add_argument("--rich_root",   required=True)
    p.add_argument("--smplx_model", required=True)
    p.add_argument("--split",       default="train_body")
    p.add_argument("--n_frames",    type=int, default=5)
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    # ── Load SMPL-X model (we need to call it WITH transl) ────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(scene_dir, _smplx_arg)
    model  = placer._smplx_model
    num_expr = model.num_expression_coeffs

    # ── Load a few GT frames ──────────────────────────────────────────────────
    gt_root = rich_root / args.split / scene_name
    frames_loaded = 0
    print(f"{'frame':>6}  {'formula':14}  {'err_mm':>8}  {'J_can[0]':30}  {'gt_transl':30}")
    print("─" * 100)

    for frame_dir in sorted(gt_root.iterdir()):
        if frames_loaded >= args.n_frames:
            break
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue

        pkl_path = next(frame_dir.glob("*.pkl"), None)
        if pkl_path is None:
            continue

        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        transl        = np.asarray(d["transl"],        dtype=np.float32).reshape(1, 3)
        global_orient = np.asarray(d["global_orient"], dtype=np.float32).reshape(1, 3)
        body_pose     = np.asarray(d["body_pose"],     dtype=np.float32).reshape(1, 63)
        raw_betas = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
        betas = np.asarray(raw_betas, dtype=np.float32).reshape(1, -1)[:, :10]

        # ── Ground truth: run smplx WITH actual transl ────────────────────────
        with torch.no_grad():
            out_full = model(
                betas=torch.tensor(betas),
                global_orient=torch.tensor(global_orient),
                body_pose=torch.tensor(body_pose),
                transl=torch.tensor(transl),
                jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
                expression=torch.zeros(1, num_expr),
                return_verts=False,
            )
        pelvis_gt = out_full.joints[0, 0].numpy()  # joint 0 WITH transl applied

        # ── J_can[0]: FK with zero orient, zero transl ────────────────────────
        J_can = placer._smplx_fk(betas, np.zeros((1, 63)), np.zeros((1, 3)))[0]  # (55,3)
        j_can0 = J_can[0]

        # ── FK with actual orient, zero transl ───────────────────────────────
        J_rotated = placer._smplx_fk(betas, body_pose, global_orient)[0]  # (55,3)
        j_rotated0 = J_rotated[0]  # joint 0 after global_orient, no transl

        R_gt = SciR.from_rotvec(global_orient[0].astype(np.float64)).as_matrix().astype(np.float32)

        # ── Test formula A: pelvis_world = J_can[0] + transl ─────────────────
        pred_A = j_can0 + transl[0]
        err_A  = np.linalg.norm(pred_A - pelvis_gt) * 1000  # mm

        # ── Test formula B: pelvis_world = R @ J_can[0] + transl ─────────────
        pred_B = R_gt @ j_can0 + transl[0]
        err_B  = np.linalg.norm(pred_B - pelvis_gt) * 1000  # mm

        # ── Test formula C: pelvis_world = J_rotated[0] + transl ────────────
        # (what _smplx_fk actually returns for joint 0, + transl)
        pred_C = j_rotated0 + transl[0]
        err_C  = np.linalg.norm(pred_C - pelvis_gt) * 1000  # mm

        def _f3(v): return f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]"
        print(f"{frame_idx:>6}  A (J_can+t)   {err_A:>8.3f}  J_can[0]={_f3(j_can0)}  transl={_f3(transl[0])}")
        print(f"        B (R@J_can+t) {err_B:>8.3f}  R@J_can[0]={_f3(R_gt @ j_can0)}")
        print(f"        C (fk[0]+t)  {err_C:>8.3f}  fk[0]={_f3(j_rotated0)}")
        print(f"        gt_pelvis   = {_f3(pelvis_gt)}")
        print()
        frames_loaded += 1


if __name__ == "__main__":
    main()
