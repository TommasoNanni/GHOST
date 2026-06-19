"""Test RePoGen-S keypoints on ParkingLot2_008_eating1:
  1. Generate kps for all frames → saved to /tmp/repogen_kps/
  2. Compare spread / conf stats on bad frames vs ViTPose-B
  3. Run estimate_scale_triangulated + estimate_procrustes_dlt_mhr using the new kps
     (SAM3D raw body_pose used as proxy for fused pose — good enough for a sanity check)
  4. Report scale, and pelvis translation for a few bad frames

Usage:
    pixi run python approaches/test_repogen_triangulate.py \
        --repogen-weights checkpoints/vitpose/VitPose-s_RePoGen.pth \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCENE      = Path("/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/ParkingLot2_008_eating1")
SMPLX_PATH = "body_models/SMPLX_NEUTRAL.pkl"
KPS_TMP    = Path("/tmp/repogen_kps/ParkingLot2_008_eating1")
OFFSET     = 180
BAD_FRAMES = [523, 531, 535, 592, 593, 594, 595, 807, 672, 694]


# ── helpers ───────────────────────────────────────────────────────────────────

def _aa_to_6d(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(aa.reshape(-1, 3)).as_matrix()   # (N,3,3)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=-1).reshape(aa.shape[:-1] + (6,))


def _6d_to_aa(sixd: np.ndarray) -> np.ndarray:
    s = sixd.reshape(-1, 6)
    b1 = s[:, :3] / (np.linalg.norm(s[:, :3], axis=1, keepdims=True) + 1e-8)
    b2 = s[:, 3:] - (b1 * s[:, 3:]).sum(1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    R  = np.stack([b1, b2, b3], axis=1)
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec().reshape(sixd.shape[:-1] + (3,)).astype(np.float32)


# ── Step 1: generate RePoGen kps and save to /tmp ────────────────────────────

def generate_kps(weights: str, device: str) -> None:
    import cv2
    from scripts.run_vitpose import _load_vitpose_model, _run_vitpose_on_bbox

    KPS_TMP.mkdir(parents=True, exist_ok=True)

    print(f"Loading RePoGen-S from {weights} …")
    model, target_size = _load_vitpose_model(weights, "s", device)
    print("Model loaded.\n")

    cam_dirs = sorted(d for d in SCENE.iterdir() if d.is_dir() and (d / "body_data").is_dir())

    for cam_dir in cam_dirs:
        cam_tmp = KPS_TMP / cam_dir.name
        cam_tmp.mkdir(parents=True, exist_ok=True)
        # copy body_data symlink / dir reference so the placer can load it
        body_dst = cam_tmp / "body_data"
        if not body_dst.exists():
            body_dst.symlink_to(cam_dir / "body_data")

        vid_path = cam_dir / f"{cam_dir.name}_segmentation_reid.mp4"
        if not vid_path.exists():
            continue

        cap = cv2.VideoCapture(str(vid_path))
        for npz_path in sorted((cam_dir / "body_data").glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            d   = np.load(npz_path)
            fi  = d["frame_indices"].astype(int)
            bbox = d["bbox"]

            kps_out = np.zeros((len(fi), 17, 3), dtype=np.float32)
            # Only run inference on bad frames + context window (full scene ~10 min/cam on CPU)
            target_gfs = set()
            for gf in BAD_FRAMES:
                for delta in range(-5, 6):
                    target_gfs.add(gf + delta)

            for lt, gf in enumerate(fi):
                if int(gf) not in target_gfs:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(gf) - OFFSET)
                ret, frame = cap.read()
                if not ret:
                    continue
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                kp = _run_vitpose_on_bbox(model, target_size, device, img_rgb, bbox[lt])
                if kp is not None:
                    kps_out[lt] = kp

            np.savez_compressed(str(cam_tmp / f"vitpose_kps_person_{pid}.npz"),
                                keypoints=kps_out)

        cap.release()
        print(f"  {cam_dir.name}: kps saved")


# ── Step 2: compare stats on bad frames ──────────────────────────────────────

def compare_stats() -> None:
    print("\n── Spread / conf on bad frames (B vs RePoGen-S) ──────────────────────")
    print(f"{'cam':>6} {'fr':>5}  {'B_ysp':>7} {'B_conf':>7}  {'S_ysp':>7} {'S_conf':>7}  note")

    cam_dirs = sorted(d for d in SCENE.iterdir() if d.is_dir() and (d / "body_data").is_dir())
    for cam_dir in cam_dirs:
        vp_b_path = cam_dir / "vitpose_kps_person_1.npz"
        vp_s_path = KPS_TMP / cam_dir.name / "vitpose_kps_person_1.npz"
        bd_path   = cam_dir / "body_data" / "person_1.npz"
        if not (vp_b_path.exists() and vp_s_path.exists() and bd_path.exists()):
            continue

        fi = np.load(bd_path)["frame_indices"].astype(int)
        vp_b = np.load(vp_b_path)["keypoints"]
        vp_s = np.load(vp_s_path)["keypoints"]
        fi_to_lt = {int(f): i for i, f in enumerate(fi)}

        for gf in BAD_FRAMES:
            if gf not in fi_to_lt:
                continue
            lt = fi_to_lt[gf]
            b_ysp  = vp_b[lt, :, 1].max() - vp_b[lt, :, 1].min()
            b_conf = vp_b[lt, :, 2].mean()
            s_ysp  = vp_s[lt, :, 1].max() - vp_s[lt, :, 1].min()
            s_conf = vp_s[lt, :, 2].mean()
            bbox_h = float(np.load(bd_path)["bbox"][lt, 3] - np.load(bd_path)["bbox"][lt, 1])
            if s_ysp > b_ysp + 50:
                note = "FIXED" if b_ysp < 200 else "better"
            elif s_ysp < b_ysp - 50:
                note = "WORSE"
            else:
                note = "same"
            print(f"{cam_dir.name:>6} {gf:>5}  {b_ysp:>7.1f} {b_conf:>7.2f}  {s_ysp:>7.1f} {s_conf:>7.2f}  {note}")


# ── Step 3: triangulation sanity check ───────────────────────────────────────

def triangulate_check() -> None:
    from fusion.placer import BodyPlacer

    print("\n── Scale estimation (triangulation) ──────────────────────────────────")

    # Build a fake scene dir that has the RePoGen kps alongside real body_data
    # We already symlinked body_data → original; placer reads vitpose_kps_person_*.npz
    # from cam_dir (= KPS_TMP/cam_XX), so this should work directly.

    # For fused_pose_by_pid: use SAM3D raw body_pose (axis-angle → 6D) as proxy.
    # This gives approximate FK, good enough for a sanity check on scale.
    all_pids: set[int] = set()
    all_global_frames: list[int] = []
    cam_dirs_tmp = sorted(d for d in KPS_TMP.iterdir() if d.is_dir())

    for cam_d in cam_dirs_tmp:
        for npz in sorted((cam_d / "body_data").glob("person_*.npz")):
            pid = int(npz.stem.split("_")[1])
            all_pids.add(pid)
            fi = np.load(npz)["frame_indices"].astype(int)
            all_global_frames.extend(fi.tolist())

    if not all_pids:
        print("  No person data found in tmp dir.")
        return

    frame_start = min(all_global_frames)
    frame_end   = max(all_global_frames) + 1
    T_scene     = frame_end - frame_start

    # Build fused_pose_by_pid from raw SAM3D body_pose (axis-angle, averaged across cameras)
    fused_pose_by_pid: dict[int, np.ndarray] = {}
    for pid in sorted(all_pids):
        # Collect per-frame body_pose across cameras; average at each frame
        frame_poses: dict[int, list[np.ndarray]] = {}
        for cam_d in cam_dirs_tmp:
            bd_path = cam_d / "body_data" / f"person_{pid}.npz"
            if not bd_path.exists():
                continue
            d  = np.load(bd_path)
            if "smplx_body_pose" not in d.files or "smplx_global_orient" not in d.files:
                continue
            fi = d["frame_indices"].astype(int)
            bp = d["smplx_body_pose"]   # (T_local, 63) axis-angle
            go = d["smplx_global_orient"]  # (T_local, 3)
            for lt, gf in enumerate(fi):
                # Full 22 joints (root + 21 body) as 6D, shape (22, 6)
                aa_full = np.concatenate([go[lt].reshape(1, 3), bp[lt].reshape(21, 3)], axis=0)
                sixd = _aa_to_6d(aa_full)  # (22, 6)
                frame_poses.setdefault(int(gf), []).append(sixd)

        arr = np.zeros((T_scene, 54, 6), dtype=np.float32)  # placer uses joints 1-54 (body_pose)
        for gf, poses in frame_poses.items():
            t = gf - frame_start
            if 0 <= t < T_scene:
                mean_pose = np.mean(poses, axis=0)  # (22, 6)
                arr[t, :21] = mean_pose[1:]  # joints 1-21 body, skip root
        fused_pose_by_pid[pid] = arr

    print(f"  frame_start={frame_start}, T_scene={T_scene}, pids={sorted(all_pids)}")

    # BodyPlacer needs vggt_cameras.npz and vggt_depth.npz in scene_dir
    for fname in ("vggt_cameras.npz", "vggt_depth.npz"):
        dst = KPS_TMP / fname
        if not dst.exists():
            dst.symlink_to(SCENE / fname)

    placer = BodyPlacer(KPS_TMP, SMPLX_PATH)

    scale_per_frame = placer.estimate_scale_triangulated(
        fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start,
    )
    med = float(np.nanmedian(scale_per_frame))
    p25, p75 = float(np.nanpercentile(scale_per_frame, 25)), float(np.nanpercentile(scale_per_frame, 75))
    print(f"  Scale: median={med:.4f}  p25={p25:.4f}  p75={p75:.4f}  (expected ~0.85–1.0)")

    # Procrustes DLT — translation check on bad frames
    print("\n── Procrustes DLT translation (bad frames subset) ────────────────────")
    pred_betas: dict[int, np.ndarray] = {}
    for pid in sorted(all_pids):
        betas_list = []
        for cam_d in cam_dirs_tmp:
            bd_path = cam_d / "body_data" / f"person_{pid}.npz"
            if not bd_path.exists():
                continue
            d = np.load(bd_path)
            if "smplx_betas" in d.files:
                betas_list.append(d["smplx_betas"].mean(axis=0))
        pred_betas[pid] = np.mean(betas_list, axis=0) if betas_list else np.zeros(10, dtype=np.float32)

    trans_dict, orient_dict = placer.estimate_procrustes_dlt_mhr(
        scale=scale_per_frame,
        all_pids=all_pids,
        pred_betas_by_pid=pred_betas,
        fused_pose_by_pid=fused_pose_by_pid,
        frame_start=frame_start,
    )

    all_trans = []
    print(f"  {'pid':>4} {'frame':>6}  {'tx':>8} {'ty':>8} {'tz':>8}  {'|t|':>8}")
    for pid in sorted(all_pids):
        frames = trans_dict.get(pid, {})
        for gf in BAD_FRAMES:
            if gf in frames:
                t = frames[gf]
                norm = float(np.linalg.norm(t))
                all_trans.append(norm)
                print(f"  {pid:>4} {gf:>6}  {t[0]:>8.3f} {t[1]:>8.3f} {t[2]:>8.3f}  {norm:>8.3f}m")

    # Also report coverage: what fraction of frames have a valid translation
    n_frames_with_trans = sum(len(v) for v in trans_dict.values())
    print(f"\n  Total frames with valid translation: {n_frames_with_trans}")
    if all_trans:
        print(f"  |t| across bad frames: mean={np.mean(all_trans):.3f}m  "
              f"min={np.min(all_trans):.3f}m  max={np.max(all_trans):.3f}m")
        print(f"  (Expect 1–10m depending on scene; NaN/inf or 100+ = still broken)")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repogen-weights", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip kps generation if /tmp already has them")
    args = parser.parse_args()

    if not args.skip_generate:
        generate_kps(args.repogen_weights, args.device)
    else:
        print("Skipping kps generation (--skip-generate set)")

    compare_stats()
    triangulate_check()


if __name__ == "__main__":
    main()
