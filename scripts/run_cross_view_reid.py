"""
Run cross-view person re-identification on pre-computed body_data, followed by
temporal synchronisation, camera alignment, and geometric post-ReID.

Copies body_data/ from SOURCE_DIR into OUTPUT_DIR at the start of each run,
then runs CrossVideoReidentifier on OUTPUT_DIR (in-place). SOURCE_DIR is never
modified, so experiments can be repeated cleanly.

mask_data.npz and json_data/ are also copied since match_across_views rewrites them in-place.
"""
import json
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

from configuration import CONFIG
from preprocessing.parameters_extraction import CrossVideoReidentifier
from preprocessing.camera_alignment import CameraAlignment
from preprocessing.geometric_reidentifier import GeometricReidentifier
from synchronize_videos.synchronizer import Synchronizer

# --- configuration -----------------------------------------------------------
SOURCE_DIR  = Path("/cluster/project/cvg/students/tnanni/ghost/preprocessing_outputs/backup_scenes")
OUTPUT_DIR  = Path("/cluster/project/cvg/students/tnanni/ghost/preprocessing_outputs/new_reid_test")
SCENE       = None      # set to a scene ID string to process only that scene, e.g. "scene_01"
FORCE       = True      # re-run even if cross_view_reid.json already exists
VISUALIZE   = True      # generate *_segmentation_reid.mp4 videos after ReID
DATA_ROOT   = "/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train"
# temporal sync evaluation
SYNC_MAX_SHIFT   = 50    # maximum absolute random shift in frames
SYNC_N_TRIALS    = 1     # random-shift trials per scene
SYNC_SEED        = 11    # RNG seed
SYNC_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SYNC_MIN_OVERLAP = 100   # min frames of overlap required for a valid offset estimate
# -----------------------------------------------------------------------------


def _load_body_data(
    video_dirs: dict[str, Path],
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Load smplx pose sequences for all cameras.

    Returns {cam_id: {person_id: (rotations T×51×3, conf T×51)}}.

    Uses frame_indices to build a dense sequence spanning first_frame..last_frame.
    Frames missing from the npz (detection gaps) are filled with zero pose and
    zero confidence so they do not contribute to the cross-correlation cost.
    """
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for cam_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            continue
        persons: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            pid = int(npz_path.stem.split("_")[1])
            with np.load(str(npz_path)) as d:
                required = {"smplx_body_pose", "smplx_left_hand_pose",
                            "smplx_right_hand_pose", "pred_joint_confidence"}
                if not required.issubset(d.files):
                    print(f"  WARNING: {npz_path.name} missing pose keys, skipping")
                    continue
                pose = np.concatenate([
                    d["smplx_body_pose"],       # (T, 63)
                    d["smplx_left_hand_pose"],  # (T, 45)
                    d["smplx_right_hand_pose"], # (T, 45)
                ], axis=1)                      # (T, 153)
                raw_conf = d["pred_joint_confidence"][:, 1:52].astype(np.float32)  # (T, 51)
                frame_indices = d["frame_indices"].astype(int) if "frame_indices" in d.files else None

            if frame_indices is not None and len(frame_indices) < (frame_indices[-1] - frame_indices[0] + 1):
                # Gaps exist: build a dense array and zero-out missing frames.
                T_full = int(frame_indices[-1] - frame_indices[0] + 1)
                dense_pose = np.zeros((T_full, 153), dtype=np.float32)
                dense_conf = np.zeros((T_full, 51),  dtype=np.float32)
                idx = frame_indices - frame_indices[0]
                dense_pose[idx] = pose
                dense_conf[idx] = raw_conf
                n_missing = T_full - len(frame_indices)
                print(f"  {cam_id}/person_{pid}: padded {n_missing} missing frame(s) with zero confidence")
                pose     = dense_pose
                raw_conf = dense_conf

            rotations = torch.from_numpy(pose).reshape(-1, 51, 3)
            conf      = torch.from_numpy(raw_conf)
            persons[pid] = (rotations, conf)
        if persons:
            cam_data[cam_id] = persons
    return cam_data


def _common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Person IDs present in every camera."""
    sets = [set(p.keys()) for p in cam_data.values()]
    return sorted(set.intersection(*sets))


def _apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    pids: list[int],
    min_overlap: int = SYNC_MIN_OVERLAP,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]] | None:
    """Slice sequences to simulate temporal offsets.

    Each camera starts at a different offset and keeps all its remaining
    frames — no common end time, no common length, fully independent.
    Returns None if any camera ends up with fewer than min_overlap frames
    (the overlap between any pair is at most the shorter sequence length).
    """
    cam_ids = list(shifts.keys())
    max_s   = max(shifts.values())
    spread  = max_s - min(shifts.values())
    print(f"  shift_spread={spread}")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []
    for cam_id in cam_ids:
        s = max_s - shifts[cam_id]
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            remaining = rotations.shape[0] - s
            if remaining < min_overlap:
                print(
                    f"  WARNING: {cam_id} has only {remaining} frames after shift "
                    f"(need ≥{min_overlap}) — skipping sync"
                )
                return None
            per_person_joints.append(rotations[s:].to(SYNC_DEVICE))
            per_person_confs .append(conf     [s:].to(SYNC_DEVICE))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)
    return joints_list, confs_list


def _load_gt_cameras_cam0(
    scene_id: str,
    cam_ids: list[str],
    rich_data_root: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load RICH GT cameras expressed in cam-0 frame.

    Parses scan_calibration/{location}/calibration/{cam_num:03d}.xml for each
    camera, extracts the world-to-camera extrinsic [R|t] (3x4), then computes
    the cam-0 → cam-i relative transform:
        R_rel = R_i @ R_0^T
        t_rel = t_i - R_rel @ t_0
    so that  X_cami = R_rel @ X_cam0 + t_rel.

    Returns {cam_id: (R_rel, t_rel)}.  cam-0 entry has identity R and zero t.
    """
    m = re.match(r'^(.+?)_\d{3}_', scene_id)
    location = m.group(1) if m else scene_id
    calib_dir = Path(rich_data_root) / "scan_calibration" / location / "calibration"

    extrinsics: dict[str, np.ndarray] = {}
    for cam_id in cam_ids:
        num_m = re.search(r"\d+", cam_id)
        cam_num = int(num_m.group()) if num_m else cam_ids.index(cam_id)
        xml_path = calib_dir / f"{cam_num:03d}.xml"
        if not xml_path.exists():
            logging.warning(f"GT calib: no XML for {cam_id} at {xml_path}")
            continue
        tree = ET.parse(str(xml_path))
        node = tree.getroot().find("CameraMatrix")
        if node is None:
            logging.warning(f"GT calib: no CameraMatrix in {xml_path}")
            continue
        rows = int(node.findtext("rows", default="3"))
        cols = int(node.findtext("cols", default="4"))
        data = list(map(float, node.findtext("data", default="").split()))
        extrinsics[cam_id] = np.array(data, dtype=np.float64).reshape(rows, cols)

    if not extrinsics:
        return {}
    cam0 = cam_ids[0]
    if cam0 not in extrinsics:
        logging.warning(f"GT calib: cam-0 ({cam0}) has no extrinsics — cannot relativise")
        return {}

    R0 = extrinsics[cam0][:3, :3]
    t0 = extrinsics[cam0][:3, 3]

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam_id, ext in extrinsics.items():
        R_i = ext[:3, :3]
        t_i = ext[:3, 3]
        R_rel = R_i @ R0.T
        t_rel = t_i - R_rel @ t0
        result[cam_id] = (R_rel, t_rel)
    return result


def scan_scenes(output_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    """Return (scene_id, {video_id: video_dir}) for every scene that has body_data."""
    scenes = []
    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        video_dirs: dict[str, Path] = {}
        for cam_dir in sorted(scene_dir.iterdir()):
            if cam_dir.is_dir() and (cam_dir / "body_data").exists():
                video_dirs[cam_dir.name] = cam_dir
        if video_dirs:
            scenes.append((scene_dir.name, video_dirs))
    return scenes


def copy_scene_data(source_dir: Path, output_dir: Path, scene_id: str) -> None:
    """Copy body_data/, mask_data.npz and json_data/ from source_dir into output_dir.

    mask_data.npz and json_data/ are also copied (not just body_data) because
    match_across_views rewrites them in-place via _apply_reid_remap.
    """
    src_scene = source_dir / scene_id
    dst_scene = output_dir / scene_id
    for cam_dir in sorted(src_scene.iterdir()):
        if not cam_dir.is_dir() or not (cam_dir / "body_data").exists():
            continue
        dst_cam = dst_scene / cam_dir.name

        for item in ("body_data", "json_data"):
            src = cam_dir / item
            dst = dst_cam / item
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        mask_src = cam_dir / "mask_data.npz"
        if mask_src.exists():
            dst_cam.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_src, dst_cam / "mask_data.npz")

        print(f"  Copied {cam_dir.name}")


def main() -> None:
    if not SOURCE_DIR.exists():
        print(f"ERROR: SOURCE_DIR does not exist: {SOURCE_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if VISUALIZE and not DATA_ROOT:
        print("ERROR: VISUALIZE=True requires DATA_ROOT to be set.")
        sys.exit(1)

    reidentifier = CrossVideoReidentifier(
        threshold=getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
        appearance_weight=getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
        shape_weight=getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
        pose_weight=getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
    )

    scenes = scan_scenes(SOURCE_DIR)
    if not scenes:
        print(f"No scenes with body_data found in {SOURCE_DIR}")
        sys.exit(0)

    if SCENE:
        scenes = [(sid, vd) for sid, vd in scenes if sid == SCENE]
        if not scenes:
            print(f"Scene '{SCENE}' not found or has no body_data in {SOURCE_DIR}")
            sys.exit(1)

    print(f"Found {len(scenes)} scene(s) with body_data.")

    for scene_id, _ in scenes:
        # Resolve video_dirs to OUTPUT_DIR (where ReID will write in-place)
        video_dirs = {
            cam_dir.name: OUTPUT_DIR / scene_id / cam_dir.name
            for cam_dir in sorted((SOURCE_DIR / scene_id).iterdir())
            if (cam_dir / "body_data").exists()
        }
        print(f"\n=== Scene: {scene_id} ({len(video_dirs)} cameras) ===")
        for vid_id in sorted(video_dirs):
            print(f"  {vid_id}")

        print(f"  Copying data from source...")
        copy_scene_data(SOURCE_DIR, OUTPUT_DIR, scene_id)

        scene_out_dir = OUTPUT_DIR / scene_id

        # Remove stale artefacts so reruns are clean.
        for stale in ("cross_view_reid.json", "temporal_offsets.json", "camera_alignment.npz"):
            stale_path = scene_out_dir / stale
            if FORCE and stale_path.exists():
                print(f"  FORCE: removing existing {stale_path.name}")
                stale_path.unlink()

        # match_across_views only reads scene.scene_id; no full Scene object needed.
        mock_scene = SimpleNamespace(scene_id=scene_id)
        reidentifier.match_across_views(scene=mock_scene, video_dirs=video_dirs)

        # ── Temporal synchronisation (random-shift evaluation) ────────────────
        print(f"\n--- Temporal synchronisation ---")
        cam_data = _load_body_data(video_dirs)
        if len(cam_data) < 2:
            print("  WARNING: fewer than 2 cameras with pose data — skipping sync")
        else:
            pids = _common_persons(cam_data)
            if not pids:
                print("  WARNING: no person ID common across all cameras — skipping sync")
            else:
                cam_ids = list(cam_data.keys())
                print(f"  Cameras: {cam_ids}")
                print(f"  Common persons: {pids}")
                sync = Synchronizer(method="cross_corr", device=SYNC_DEVICE, min_overlap=SYNC_MIN_OVERLAP)
                rng  = np.random.default_rng(SYNC_SEED)
                all_results = []
                for trial in range(SYNC_N_TRIALS):
                    raw_shifts  = [0] + rng.integers(
                        -SYNC_MAX_SHIFT, SYNC_MAX_SHIFT + 1, size=len(cam_ids) - 1
                    ).tolist()
                    true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
                    print(f"\n  Trial {trial + 1}/{SYNC_N_TRIALS}  true shifts: {true_shifts}")

                    result = _apply_shifts(cam_data, true_shifts, pids)
                    if result is None:
                        continue
                    joints_list, confs_list = result
                    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
                    weights    = sync.cycle_consistency_weights(offset_mat)
                    estimated  = sync.estimate_initial_times(offset_mat, weights)

                    # Normalise ground truth to the same reference (min=0)
                    true_t = torch.tensor(
                        [true_shifts[c] for c in cam_ids], dtype=torch.float32
                    )
                    true_t = true_t - true_t.min()
                    errors = (estimated.cpu() - true_t).abs()
                    mae    = errors.mean().item()

                    print(f"  {'Camera':<20} {'True':>6}  {'Estimated':>10}  {'Error':>6}")
                    print(f"  {'-'*20} {'-'*6}  {'-'*10}  {'-'*6}")
                    for cam_id, tt, est, err in zip(
                        cam_ids, true_t.tolist(), estimated.cpu().tolist(), errors.tolist()
                    ):
                        print(f"  {cam_id:<20} {tt:>+6.0f}  {est:>+10.1f}  {err:>6.1f}")
                    print(
                        f"  MAE={mae:.2f}  "
                        f"within-1={((errors <= 1).float().mean().item()) * 100:.0f}%  "
                        f"within-2={((errors <= 2).float().mean().item()) * 100:.0f}%"
                    )
                    all_results.append({
                        "mae": mae,
                        "within_1": (errors <= 1).float().mean().item(),
                        "within_2": (errors <= 2).float().mean().item(),
                    })

                    # Save the last trial's estimated offsets to disk.
                    if trial == SYNC_N_TRIALS - 1:
                        offsets_dict = {
                            c: int(round(t))
                            for c, t in zip(cam_ids, estimated.cpu().tolist())
                        }
                        with open(scene_out_dir / "temporal_offsets.json", "w") as _f:
                            json.dump(offsets_dict, _f, indent=2)
                        print(f"  Saved temporal_offsets.json")

                if SYNC_N_TRIALS > 1:
                    all_mae = [r["mae"] for r in all_results]
                    print(f"\n  SUMMARY over {SYNC_N_TRIALS} trials:")
                    print(
                        f"  MAE  mean={np.mean(all_mae):.2f}  "
                        f"median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}"
                    )
                    print(f"  Within 1fr  {np.mean([r['within_1'] for r in all_results]) * 100:.1f}%")
                    print(f"  Within 2fr  {np.mean([r['within_2'] for r in all_results]) * 100:.1f}%")

        # ── Camera alignment ──────────────────────────────────────────────────
        print(f"\n--- Camera alignment ---")
        alignment = CameraAlignment().estimate(
            video_dirs, min_correspondences=30, scene_dir=scene_out_dir
        )
        if alignment:
            align_path = CameraAlignment.save(alignment, scene_out_dir)
            print(f"  Estimated {len(alignment)} camera pair(s) → {align_path}")
            for (vid_a, vid_b), (R, t) in alignment.items():
                centre = CameraAlignment.camera_center_in_A(R, t)
                print(
                    f"  {vid_a} ← {vid_b}: "
                    f"|t|={np.linalg.norm(t):.3f} m, "
                    f"cam_B in A={centre.round(3).tolist()}"
                )
        else:
            print("  WARNING: no camera pairs aligned — check cross-view ReID found shared persons")

        # ── Compare estimated cameras against RICH GT ─────────────────────────
        cam_ids_ordered = list(video_dirs.keys())
        gt_cameras = _load_gt_cameras_cam0(scene_id, cam_ids_ordered, DATA_ROOT)
        if gt_cameras and alignment:
            cam0_name = cam_ids_ordered[0]
            print(f"\n  Camera estimation vs GT (world = {cam0_name}):")
            print(f"  {'Camera':<20}  {'RE (deg)':>10}  {'TE (m)':>8}")
            print(f"  {'-'*20}  {'-'*10}  {'-'*8}")
            for cam_i_name in cam_ids_ordered[1:]:
                if cam_i_name not in gt_cameras:
                    print(f"  {cam_i_name:<20}  {'no GT':>10}")
                    continue
                R_gt, t_gt = gt_cameras[cam_i_name]
                if (cam0_name, cam_i_name) in alignment:
                    R_est, t_est = alignment[(cam0_name, cam_i_name)]
                elif (cam_i_name, cam0_name) in alignment:
                    R, t = alignment[(cam_i_name, cam0_name)]
                    R_est, t_est = R.T, -R.T @ t
                else:
                    print(f"  {cam_i_name:<20}  {'no estimate':>10}")
                    continue
                cos_a = np.clip((np.trace(R_gt.T @ R_est) - 1.0) / 2.0, -1.0, 1.0)
                re_deg = float(np.degrees(np.arccos(cos_a)))
                te = float(np.linalg.norm(t_gt - t_est))
                print(f"  {cam_i_name:<20}  {re_deg:>10.2f}  {te:>8.3f}")
        elif not gt_cameras:
            print("  (GT camera comparison skipped — calibration XMLs not found)")

        # ── Geometric post-ReID ───────────────────────────────────────────────
        print(f"\n--- Geometric post-ReID ---")
        geo_reid = GeometricReidentifier(distance_threshold=0.5, min_overlap_frames=10)
        geo_reid.reidentify(scene=mock_scene, video_dirs=video_dirs)

        if VISUALIZE:
            from utilities.visualize_segmented_reids import visualize_reid
            print(f"\n  Generating ReID visualizations...")
            for vid_id, video_dir in sorted(video_dirs.items()):
                _cam_dir = Path(DATA_ROOT) / scene_id / vid_id
                frames_dir = _cam_dir / "frames" if (_cam_dir / "frames").exists() else _cam_dir if _cam_dir.exists() else None
                try:
                    out = visualize_reid(video_dir=video_dir, fps=30, frames_dir=frames_dir)
                    print(f"    {vid_id}: saved {out}")
                except FileNotFoundError as e:
                    print(f"    WARNING: skipping {vid_id} — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
