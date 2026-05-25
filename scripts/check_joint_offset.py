"""Check per-joint offset between SMPL-X FK joints and MHR70 pred_keypoints_3d.

Both are root-centred in camera space, then averaged across all frames and
body files found under a scene output directory.  If the offset is close to
zero the current _SMPLX_TO_MHR70 mapping is fine; large offsets indicate a
systematic skeleton-definition mismatch.

Usage:
    pixi run python scripts/check_joint_offset.py \
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test/BBQ_001_guitar
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import smplx as smplx_lib
from scipy.spatial.transform import Rotation as SciR

_SMPLX_TO_MHR70 = {
    1: 9,   2: 10,   # left/right hip
    4: 11,  5: 12,   # left/right knee
    7: 13,  8: 14,   # left/right ankle
    12: 69,          # neck
    16: 5,  17: 6,   # left/right shoulder
    18: 7,  19: 8,   # left/right elbow
    20: 62, 21: 41,  # left/right wrist
}

_SMPLX_NAMES = {
    1: "left_hip",     2: "right_hip",
    4: "left_knee",    5: "right_knee",
    7: "left_ankle",   8: "right_ankle",
    12: "neck",
    16: "left_shoulder",  17: "right_shoulder",
    18: "left_elbow",     19: "right_elbow",
    20: "left_wrist",     21: "right_wrist",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="Scene output directory")
    parser.add_argument("--smplx_model_path", default="body_models/SMPLX_NEUTRAL.pkl")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    smplx_path = Path(args.smplx_model_path)
    create_kwargs = {"model_type": "smplx"}
    if smplx_path.is_file():
        create_kwargs["ext"] = smplx_path.suffix.lstrip(".")
    model = smplx_lib.create(
        str(smplx_path),
        **create_kwargs,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).eval()

    offsets = {j: [] for j in _SMPLX_TO_MHR70}
    n_frames = 0

    for body_file in sorted(scene_dir.glob("*/body_data/person_*.npz")):
        data = np.load(body_file, allow_pickle=False)
        required = {"pred_keypoints_3d", "smplx_body_pose", "smplx_betas",
                    "smplx_global_orient", "smplx_transl"}
        if not required.issubset(data.files):
            continue

        kp3d        = data["pred_keypoints_3d"]       # (T, 70, 3)  camera space
        body_pose   = data["smplx_body_pose"]          # (T, 63)
        betas       = data["smplx_betas"]              # (T, 10)
        global_orient = data["smplx_global_orient"]    # (T, 3)
        transl      = data["smplx_transl"]             # (T, 3)

        T = len(body_pose)
        num_expr = model.num_expression_coeffs

        # FK with actual global_orient + transl → SMPL-X joints in camera space
        with torch.no_grad():
            out = model(
                betas=torch.tensor(betas, dtype=torch.float32),
                body_pose=torch.tensor(body_pose, dtype=torch.float32),
                global_orient=torch.tensor(global_orient, dtype=torch.float32),
                transl=torch.tensor(transl, dtype=torch.float32),
                jaw_pose=torch.zeros(T, 3),
                leye_pose=torch.zeros(T, 3),
                reye_pose=torch.zeros(T, 3),
                left_hand_pose=torch.zeros(T, 45),
                right_hand_pose=torch.zeros(T, 45),
                expression=torch.zeros(T, num_expr),
                return_verts=False,
            )
        J_cam = out.joints[:, :55].cpu().numpy()  # (T, 55, 3)

        for t in range(T):
            # Root = midpoint of left/right hip — same anatomical anchor for both skeletons.
            # SMPL-X: joints 1=left_hip, 2=right_hip. MHR70: joints 9=left_hip, 10=right_hip.
            # (MHR70 joint 0 is a head landmark, not the pelvis.)
            pelvis_smplx = (J_cam[t, 1] + J_cam[t, 2]) / 2.0
            pelvis_mhr   = (kp3d[t, 9] + kp3d[t, 10]) / 2.0
            for smplx_j, mhr70_j in _SMPLX_TO_MHR70.items():
                if mhr70_j >= kp3d.shape[1]:
                    continue
                j_smplx = J_cam[t, smplx_j] - pelvis_smplx
                j_mhr   = kp3d[t, mhr70_j]  - pelvis_mhr
                delta = j_smplx - j_mhr
                offsets[smplx_j].append(delta)

        n_frames += T
        print(f"  {body_file.relative_to(scene_dir)}  ({T} frames)")

    print(f"\nTotal frames processed: {n_frames}\n")
    print(f"{'Joint':<20} {'dx (cm)':>10} {'dy (cm)':>10} {'dz (cm)':>10} {'|δ| (cm)':>10}")
    print("-" * 62)
    for smplx_j in sorted(offsets):
        if not offsets[smplx_j]:
            continue
        mean_d = np.mean(offsets[smplx_j], axis=0) * 100  # → cm
        norm   = np.linalg.norm(mean_d)
        name   = _SMPLX_NAMES.get(smplx_j, str(smplx_j))
        print(f"{name:<20} {mean_d[0]:>10.2f} {mean_d[1]:>10.2f} {mean_d[2]:>10.2f} {norm:>10.2f}")


if __name__ == "__main__":
    main()
