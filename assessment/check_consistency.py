"""Verify that Kabsch cameras and per-camera pred_cam_t are consistent by construction.

For each camera k, Kabsch was computed using absolute joints =
pred_keypoints_3d + pred_cam_t. By construction:
    R_0k @ joints_0 + t_0k ≈ joints_k
In particular, applying the Kabsch transform to the body root from cam0
should recover the body root in cam_k:
    R_0k @ cam_t_0 + t_0k ≈ cam_t_k
Equivalently, transforming cam_t_k back to cam0 frame should give cam_t_0:
    R_0k^T @ (cam_t_k - t_0k) ≈ cam_t_0

If this holds tightly, then body root, body translation and camera extrinsics
are NOT independent — they're determined by the same underlying data.
"""

import numpy as np
from pathlib import Path


SCENES = [
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test/BBQ_001_guitar",
    "/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test/LectureHall_018_wipingchairs1",
]


def load_kabsch(scene_dir: Path) -> dict:
    """Load Kabsch cameras: returns {(vid_a, vid_b): (R, t)}."""
    path = scene_dir / "camera_alignment.npz"
    if not path.exists():
        return {}
    results = {}
    with np.load(str(path)) as f:
        prefixes = {k[:-3] for k in f.files if k.endswith("__R")}
        for prefix in sorted(prefixes):
            parts = prefix.split("__to__")
            if len(parts) != 2:
                continue
            vid_a, vid_b = parts
            results[(vid_a, vid_b)] = (f[f"{prefix}__R"], f[f"{prefix}__t"])
    return results


def load_pred_cam_t(body_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pred_cam_t and pred_keypoints_3d for person_1."""
    path = body_dir / "person_1.npz"
    if not path.exists():
        return None
    with np.load(str(path)) as f:
        cam_t = f.get("pred_cam_t")        # (T, 3)
        kpts  = f.get("pred_keypoints_3d") # (T, J, 3) root-relative
        fidx  = f.get("frame_indices")     # (T,)
    if cam_t is None:
        return None
    return cam_t, kpts, fidx


def main():
    for scene_path in SCENES:
        scene_dir = Path(scene_path)
        print(f"\n{'='*60}")
        print(f"Scene: {scene_dir.name}")

        kabsch = load_kabsch(scene_dir)
        if not kabsch:
            print("  No camera_alignment.npz found, skipping.")
            continue

        cam_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")])
        cam_names = [d.name for d in cam_dirs]
        print(f"  Cameras: {cam_names}")

        # Load pred_cam_t for each camera
        cam_data = {}
        for cam_dir in cam_dirs:
            result = load_pred_cam_t(cam_dir / "body_data")
            if result is not None:
                cam_t, kpts, fidx = result
                cam_data[cam_dir.name] = {"cam_t": cam_t, "kpts": kpts, "fidx": fidx}

        if not cam_data:
            print("  No body data found.")
            continue

        # Reference camera: cam_00
        ref = "cam_00"
        if ref not in cam_data:
            ref = list(cam_data.keys())[0]
        print(f"  Reference camera: {ref}")

        ref_cam_t = cam_data[ref]["cam_t"]  # (T_ref, 3)
        ref_fidx  = cam_data[ref]["fidx"]   # (T_ref,)
        ref_frame_map = {int(fi): i for i, fi in enumerate(ref_fidx)}

        print(f"\n  {'Camera':<12} {'Pairs':<8} {'Mean err (m)':<16} {'Max err (m)':<14} {'RMSE kpts (m)'}")
        print(f"  {'-'*70}")

        for cam_name, data in cam_data.items():
            if cam_name == ref:
                continue

            # Find Kabsch for (ref -> cam_name) or (cam_name -> ref)
            key_fwd = (ref, cam_name)
            key_inv = (cam_name, ref)
            if key_fwd in kabsch:
                R, t = kabsch[key_fwd]  # maps ref → cam_name
            elif key_inv in kabsch:
                R_inv, t_inv = kabsch[key_inv]  # maps cam_name → ref
                # Invert: R^T, -R^T @ t
                R = R_inv.T
                t = -R_inv.T @ t_inv
            else:
                print(f"  {cam_name:<12} No Kabsch pair found")
                continue

            other_cam_t = data["cam_t"]   # (T_other, 3)
            other_fidx  = data["fidx"]    # (T_other,)
            other_frame_map = {int(fi): i for i, fi in enumerate(other_fidx)}

            # Find common frames
            common = sorted(set(ref_frame_map) & set(other_frame_map))
            if not common:
                print(f"  {cam_name:<12} No common frames")
                continue

            # For each common frame:
            # 1. Transform ref cam_t to cam_name frame: pred = R @ cam_t_ref + t
            # 2. Compare to actual cam_t in cam_name
            ref_pts = np.array([ref_cam_t[ref_frame_map[fi]] for fi in common])   # (N, 3)
            other_pts = np.array([other_cam_t[other_frame_map[fi]] for fi in common])  # (N, 3)

            pred_other = (R @ ref_pts.T).T + t  # (N, 3)
            errors = np.linalg.norm(pred_other - other_pts, axis=-1)  # (N,)

            # Also check full absolute joints if available
            ref_kpts = cam_data[ref].get("kpts")
            other_kpts = data.get("kpts")
            kpts_rmse = float("nan")
            if ref_kpts is not None and other_kpts is not None:
                ref_abs = np.array([
                    ref_kpts[ref_frame_map[fi]] + ref_cam_t[ref_frame_map[fi]][None]
                    for fi in common
                ])  # (N, J, 3)
                other_abs = np.array([
                    other_kpts[other_frame_map[fi]] + other_cam_t[other_frame_map[fi]][None]
                    for fi in common
                ])  # (N, J, 3)
                pred_other_kpts = (R @ ref_abs.reshape(-1, 3).T).T.reshape(len(common), -1, 3) + t
                kpts_rmse = float(np.sqrt(((pred_other_kpts - other_abs)**2).sum(-1).mean()))

            print(f"  {cam_name:<12} {len(common):<8} {errors.mean():<16.4f} {errors.max():<14.4f} {kpts_rmse:.4f}")

        # Now check: transforming cam_t_k back to ref should give cam_t_ref
        print(f"\n  Inverse check: transform cam_t_k → ref frame, compare to cam_t_ref")
        print(f"  {'Camera':<12} {'Mean err (m)':<16} {'Max err (m)'}")
        print(f"  {'-'*45}")
        for cam_name, data in cam_data.items():
            if cam_name == ref:
                continue
            key_fwd = (ref, cam_name)
            key_inv = (cam_name, ref)
            if key_fwd in kabsch:
                R, t = kabsch[key_fwd]
            elif key_inv in kabsch:
                R_inv, t_inv = kabsch[key_inv]
                R = R_inv.T
                t = -R_inv.T @ t_inv
            else:
                continue

            other_cam_t = data["cam_t"]
            other_fidx  = data["fidx"]
            other_frame_map = {int(fi): i for i, fi in enumerate(other_fidx)}
            common = sorted(set(ref_frame_map) & set(other_frame_map))
            if not common:
                continue

            ref_pts   = np.array([ref_cam_t[ref_frame_map[fi]] for fi in common])
            other_pts = np.array([other_cam_t[other_frame_map[fi]] for fi in common])

            # Recover ref from other: R^T @ (other - t)
            recovered_ref = (R.T @ (other_pts - t).T).T
            errors = np.linalg.norm(recovered_ref - ref_pts, axis=-1)
            print(f"  {cam_name:<12} {errors.mean():<16.4f} {errors.max():.4f}")


if __name__ == "__main__":
    main()
