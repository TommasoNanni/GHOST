import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

import json
import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import RichDataset
from data.fusion_dataset import RICHFusionDatapoint, RICHFusionDataset
from preprocessing.camera_alignment import CameraAlignment
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction import ParametersExtractor, CrossVideoReidentifier
from synchronize_videos.synchronizer import Synchronizer
from utilities.visualize_segmented_reids import visualize_reid

# ── Temporal-sync evaluation constants (random-shift test) ─────────────────────
SYNC_MAX_SHIFT = 148   # maximum absolute shift in frames
SYNC_N_TRIALS  = 1     # random-shift trials per scene
SYNC_SEED      = 42    # RNG seed
SYNC_DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def _load_body_data(
    video_dirs: dict[str, str],
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Load smplx pose sequences for all cameras.

    Mirrors ``load_scene`` in evaluation/alignment_experiments.py:
    concatenates smplx_body_pose (T,63) + smplx_left_hand_pose (T,45) +
    smplx_right_hand_pose (T,45) → (T,51,3), and extracts confidence
    pred_joint_confidence[:,1:52] → (T,51).

    Returns
    -------
    cam_data : {cam_id: {person_id: (rotations T×51×3, conf T×51)}}
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
                    d["smplx_body_pose"],        # (T, 63) — 21 joints
                    d["smplx_left_hand_pose"],   # (T, 45) — 15 joints
                    d["smplx_right_hand_pose"],  # (T, 45) — 15 joints
                ], axis=1)                       # (T, 153)
                rotations = torch.from_numpy(pose.astype(np.float32)).reshape(-1, 51, 3)
                conf = torch.from_numpy(
                    d["pred_joint_confidence"][:, 1:52].astype(np.float32)
                )  # (T, 51)
            persons[pid] = (rotations, conf)
        if persons:
            cam_data[cam_id] = persons
    return cam_data


def _common_persons(cam_data: dict[str, dict[int, tuple]]) -> list[int]:
    """Return person IDs present in every camera (mirrors alignment_experiments.py)."""
    sets = [set(persons.keys()) for persons in cam_data.values()]
    return sorted(set.intersection(*sets))


def _apply_shifts(
    cam_data: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]],
    shifts: dict[str, int],
    pids: list[int],
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Slice each camera's sequence to simulate temporal offsets.

    Mirrors ``apply_shifts`` in evaluation/alignment_experiments.py exactly:
    - T_base = min sequence length across all (cam, person) pairs
    - max_s  = max(shifts)
    - camera c gets frames [max_s - shifts[c] : T_base], so the latest-starting
      camera (shift == max_s) gets s=0 and all T_base frames.
    """
    cam_ids = list(shifts.keys())
    T_base  = min(cam_data[c][p][0].shape[0] for c in cam_ids for p in pids)
    max_s   = max(shifts.values())
    print(f"  T_base={T_base}  shift_spread={max_s - min(shifts.values())}")

    joints_list: list[list[torch.Tensor]] = []
    confs_list:  list[list[torch.Tensor]] = []
    for cam_id in cam_ids:
        s = max_s - shifts[cam_id]   # latest camera → s=0 → all T_base frames
        per_person_joints, per_person_confs = [], []
        for pid in pids:
            rotations, conf = cam_data[cam_id][pid]
            per_person_joints.append(rotations[s : T_base].to(SYNC_DEVICE))
            per_person_confs .append(conf     [s : T_base].to(SYNC_DEVICE))
        joints_list.append(per_person_joints)
        confs_list .append(per_person_confs)
    return joints_list, confs_list

def _build_gt_intrinsics_map(scene_id: str, cam_ids: list[str], rich_data_root: str) -> dict[str, np.ndarray]:
    """Parse RICH calibration XMLs and return {video_id: K (3x3)} per camera."""
    stem = re.match(r'^(.+?)_\d{3}_', scene_id)
    stem = stem.group(1) if stem else scene_id
    calib_dir = Path(rich_data_root) / "scan_calibration" / stem / "calibration"
    intr_map: dict[str, np.ndarray] = {}
    for i, vid_id in enumerate(cam_ids):
        xml_path = calib_dir / f"{i:03d}.xml"
        if not xml_path.exists():
            logging.warning(f"No calibration XML for {vid_id} at {xml_path}")
            continue
        tree = ET.parse(str(xml_path))
        intr_node = tree.getroot().find("Intrinsics")
        if intr_node is None:
            continue
        data = list(map(float, intr_node.findtext("data", default="").split()))
        rows = int(intr_node.findtext("rows", default="3"))
        cols = int(intr_node.findtext("cols", default="3"))
        K = np.array(data, dtype=np.float32).reshape(rows, cols)
        intr_map[vid_id] = K
        logging.info(f"  GT intrinsics {vid_id}: fx={K[0,0]:.1f} px")
    return intr_map


def process_scene(scene, segmenter, estimator, reidentifier, output_dir):
    """Run the full pipeline on a single scene."""
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    from preprocessing.segmentation import PersonSegmenter as _PS
    _seg_output_dir = Path(output_dir) / scene.scene_id
    _all_segmented = all(
        _PS._is_segmented(_seg_output_dir / v.video_id)
        for v in scene.videos
    )
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=False,  # vis=not _all_segmented
    )
    print(f"\nSegmentation output dirs:")
    for video_id, vdir in video_dirs.items():
        print(f"  {video_id}: {vdir}")
    """
    # Step 2: Estimate body parameters from segmentation output.
    print(f"\n--- Running body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Step 3: Match person IDs across camera views
    print(f"\n--- Running cross-view person re-identification ---")
    scene_output_dir_pre = Path(next(iter(video_dirs.values()))).parent
    _reid_already_done = (scene_output_dir_pre / "cross_view_reid.json").exists()
    reidentifier.match_across_views(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Derive the shared scene output directory (parent of all video dirs)
    scene_output_dir = scene_output_dir_pre

    # Step 4: Verify SMPLX conversion
    # MHR → SMPLX conversion happens automatically inside estimate_scene when
    # smplx_model_path and mhr_model_path are configured. Here we verify the
    # resulting smplx_* fields are present in the saved npz files.
    print(f"\n--- Step 4: Verifying MHR → SMPLX conversion output ---")
    smplx_fields_found = {}
    for video_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue
        for npz_path in sorted(body_dir.glob("person_*.npz")):
            data = dict(np.load(str(npz_path), allow_pickle=False))
            smplx_keys = [k for k in data if k.startswith("smplx_")]
            if smplx_keys:
                smplx_fields_found.setdefault(video_id, {})[npz_path.name] = smplx_keys
    if smplx_fields_found:
        for vid, files in smplx_fields_found.items():
            for fname, keys in files.items():
                print(f"  {vid}/{fname}: {keys}")
    else:
        print(
            "  WARNING: No smplx_* fields found. "
            "Check that smplx_model_path and mhr_model_path are set in CONFIG."
        )

    # Step 5: Temporal synchronisation (optional — enabled via CONFIG.synchronization.enabled)
    # Applies random shifts to each camera's pose sequence, runs the Synchronizer to
    # recover those shifts via DTW, and logs an evaluation (MAE, within-1/2 frames).
    sync_cfg = getattr(CONFIG, "synchronization", None)
    if sync_cfg is not None and getattr(sync_cfg, "enabled", False):
        print(f"\n--- Step 5: Temporal synchronisation (random-shift evaluation) ---")
        cam_data = _load_body_data(video_dirs)
        if len(cam_data) < 2:
            print("  WARNING: fewer than 2 cameras with pose data — skipping sync eval")
        else:
            pids = _common_persons(cam_data)
            if not pids:
                print("  WARNING: no person ID common across all cameras — skipping sync eval")
            else:
                cam_ids = list(cam_data.keys())
                print(f"  Cameras: {cam_ids}")
                print(f"  Common persons: {pids}")
                sync = Synchronizer(device=SYNC_DEVICE)
                rng  = np.random.default_rng(SYNC_SEED)
                results = []
                for trial in range(SYNC_N_TRIALS):
                    raw_shifts  = [0] + rng.integers(-SYNC_MAX_SHIFT, SYNC_MAX_SHIFT + 1,
                                                      size=len(cam_ids) - 1).tolist()
                    true_shifts = {c: int(s) for c, s in zip(cam_ids, raw_shifts)}
                    print(f"  Trial {trial + 1}/{SYNC_N_TRIALS}  true shifts: {true_shifts}")

                    joints_list, confs_list = _apply_shifts(cam_data, true_shifts, pids)
                    offset_mat = sync.estimate_offset_matrix(joints_list, confs_list)
                    weights    = sync.cycle_consistency_weights(offset_mat)
                    estimated  = sync.estimate_initial_times(offset_mat, weights)

                    true_t = torch.tensor([true_shifts[c] for c in cam_ids], dtype=torch.float32)
                    true_t = true_t - true_t.min()
                    errors = (estimated.cpu() - true_t).abs()
                    mae    = errors.mean().item()

                    for cam_id, tt, est, err in zip(cam_ids, true_t.tolist(),
                                                    estimated.cpu().tolist(), errors.tolist()):
                        print(f"    {cam_id}: true={tt:+.0f}  estimated={est:+.1f}  error={err:.1f}")
                    print(f"  MAE={mae:.2f}  "
                          f"within-1={((errors <= 1).float().mean().item()) * 100:.0f}%  "
                          f"within-2={((errors <= 2).float().mean().item()) * 100:.0f}%")
                    results.append({"mae": mae,
                                    "within_1": (errors <= 1).float().mean().item(),
                                    "within_2": (errors <= 2).float().mean().item()})

                if len(results) > 1:
                    all_mae = [r["mae"] for r in results]
                    print(f"\n  SUMMARY over {SYNC_N_TRIALS} trials:")
                    print(f"  MAE  mean={np.mean(all_mae):.2f}  "
                          f"median={np.median(all_mae):.2f}  max={np.max(all_mae):.2f}")
                    print(f"  Within 1fr  {np.mean([r['within_1'] for r in results]) * 100:.1f}%")
                    print(f"  Within 2fr  {np.mean([r['within_2'] for r in results]) * 100:.1f}%")

    # Step 6: Camera alignment
    # Estimate pairwise relative camera poses from the cross-view body
    # correspondences produced by the estimation + ReID steps, then persist
    # the result as camera_alignment.npz in the scene directory.
    print(f"\n--- Step 6: Camera alignment ---")
    alignment = CameraAlignment().estimate(video_dirs, min_correspondences=30)
    if alignment:
        align_path = CameraAlignment.save(alignment, scene_output_dir)
        print(f"  Estimated {len(alignment)} camera pair(s) → saved to {align_path}")
        for (vid_a, vid_b), (R, t) in alignment.items():
            centre = CameraAlignment.camera_center_in_A(R, t)
            print(
                f"  {vid_a} ← {vid_b}: "
                f"|t|={np.linalg.norm(t):.3f} m, "
                f"cam_B in A={centre.round(3).tolist()}"
            )
    else:
        print(
            "  WARNING: No camera pairs could be aligned. "
            "Check that cross-view ReID found shared persons across videos."
        )

    # Step 7: FusionDatapoint compatibility check
    print(f"\n--- Step 7: FusionDatapoint compatibility check ---")
    try:
        fusion_dp = RICHFusionDatapoint(
            scene_dir=scene_output_dir,
            rich_data_root=CONFIG.data.rich_data_root,
        )
        ds = RICHFusionDataset([fusion_dp])
        inputs, targets = ds[0]
        # print("  Inputs:")
        # for k, v in inputs.items():
        #     print(f"    {k}: {tuple(v.shape)} dtype={v.dtype}")
        # print("  Targets:")
        # for k, v in targets.items():
        #     print(f"    {k}: {tuple(v.shape)} dtype={v.dtype}")
        print("  FusionDatapoint compatibility: OK")
    except Exception as e:
        print(f"  ERROR: FusionDatapoint failed to load: {e}")

    # Step 8: Inspect output format for each video
    print(f"\n=== Body parameter output format ===")
    for video_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue

        # Count unique SAM3 person IDs seen across all JSON frames.
        json_dir = Path(video_dir) / "json_data"
        sam3_ids: set[int] = set()
        for jp in sorted(json_dir.glob("*.json")):
            with open(jp) as f:
                meta = json.load(f)
            for sid in meta.get("labels", {}):
                sam3_ids.add(int(sid))

        npz_files = sorted(body_dir.glob("person_*.npz"))
        print(f"\n--- {video_id}: {len(npz_files)} person file(s) ---")

        # Re-ID summary: if fewer tracks than SAM3 IDs, merges happened.
        if sam3_ids:
            n_merged = len(sam3_ids) - len(npz_files)
            merge_str = f"  ({n_merged} SAM3 ID(s) merged by re-ID)" if n_merged > 0 else "  (no merges)"
            print(
                f"  SAM3 unique IDs across all frames: {sorted(sam3_ids)}\n"
                f"  Body tracks after re-ID:           {len(npz_files)}"
                + merge_str
            )

        for npz_path in npz_files:
            data = dict(np.load(str(npz_path), allow_pickle=False))
            print(f"  {npz_path.name}:")
            for key, arr in sorted(data.items()):
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

            # if "frame_indices" in data:
            #     print(f"    -> frame_indices (first 5): {data['frame_indices'][:5].tolist()}")
            # if "pred_keypoints_3d" in data:
            #     kp3d = data["pred_keypoints_3d"]
            #     print(f"    -> pred_keypoints_3d[0] (first joint): {kp3d[0, 0].tolist()}")
            # if "pred_cam_t" in data:
            #     print(f"    -> pred_cam_t[0]: {data['pred_cam_t'][0].tolist()}")
            # if "bbox" in data:
            #     print(f"    -> bbox[0]: {data['bbox'][0].tolist()}")

        summary_path = body_dir / "body_params_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            # print(f"\n  Summary JSON for {video_id}:")
            # print(f"  {json.dumps(summary, indent=4)}")
        else:
            print(f"  WARNING: summary JSON not found at {summary_path}")

    # Step 9: Visualise the re-ID corrected segmentation (only if ReID ran this session).
    if not _reid_already_done:
        print(f"\n--- Visualising re-ID corrected segmentation ---")
        for video in scene.videos:
            if video.video_id not in video_dirs:
                continue
            print(f"  {video.video_id}")
            try:
                visualize_reid(
                    video_dir=Path(video_dirs[video.video_id]),
                    fps=int(video.fps),
                    frames_dir=video.frames_home,
                )
            except FileNotFoundError as e:
                print(f"  WARNING: skipping visualisation — {e}")
    else:
        print(f"\n--- Skipping re-ID visualisation (cross-view ReID was already done) ---")
    """

def main():
    rich_data_root = CONFIG.data.rich_data_root
    output_dir = CONFIG.data.output_directory
    scenes_slice = CONFIG.data.slice

    ds = RichDataset(
        data_root=rich_data_root,
        slice=scenes_slice,
        max_side=getattr(CONFIG.data, "rich_max_side", None),
    )
    # ds.scenes = [s for s in ds.scenes if "tossball" in s.scene_id]

    for scene in ds.scenes:
        # Re-instantiate per scene so no Python-level instance state (gallery
        # EMA, cached model handles, etc.) leaks from one scene into the next.
        segmenter = PersonSegmenter(
            checkpoint_path=CONFIG.segmentation.checkpoint_path,
            text_prompt=CONFIG.segmentation.text_prompt,
            redetect_interval=CONFIG.segmentation.redetect_interval,
            new_det_thresh=CONFIG.segmentation.new_det_thresh,
            score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
        )
        
        estimator = ParametersExtractor(
            sam3d_hf_repo = CONFIG.parameters_extraction.sam3d_id,
            sam3d_step = CONFIG.parameters_extraction.sam3d_step,
            bbox_padding = CONFIG.parameters_extraction.bbox_padding,
            smplx_model_path = CONFIG.data.smplx_model_path,
            mhr_model_path  = CONFIG.data.mhr_model_path,
            reid_threshold = CONFIG.parameters_extraction.reid_threshold,
            gallery_ema_alpha = CONFIG.parameters_extraction.gallery_moving_average_alpha,
            reid_match_window = getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
        )
        reidentifier = CrossVideoReidentifier(
            threshold = getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
            appearance_weight = getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
            shape_weight = getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
            pose_weight = getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
        )
        process_scene(scene, segmenter, estimator, reidentifier, output_dir)
        del segmenter, estimator, reidentifier
        import gc as _gc; _gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
