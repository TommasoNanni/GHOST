import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import numpy as np

from configuration import CONFIG
from data.video_dataset import EgoExoSceneDataset, RichDataset
from data.fusion_dataset import RICHFusionDataset
from preprocessing.camera_alignment import CameraAlignment
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction import BodyParameterEstimator, CrossViewReidentifier
from utilities.visualize_segmented_reids import visualize_reid

def main():
    rich_data_root = CONFIG.data.rich_data_root
    output_dir = CONFIG.data.output_directory
    scenes_slice = CONFIG.data.slice

    ds = RichDataset(
        data_root=rich_data_root,
        slice=scenes_slice,
        max_side=getattr(CONFIG.data, "rich_max_side", None),
    )
    scene = ds[0]
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    segmenter = PersonSegmenter(
        checkpoint_path=CONFIG.segmentation.checkpoint_path,
        text_prompt=CONFIG.segmentation.text_prompt,
        redetect_interval=CONFIG.segmentation.redetect_interval,
        new_det_thresh=CONFIG.segmentation.new_det_thresh,
        score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
    )
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=True,
    )
    print(f"\nSegmentation output dirs:")
    for video_id, vdir in video_dirs.items():
        print(f"  {video_id}: {vdir}")

    # Step 2: Estimate body parameters from segmentation output
    estimator = BodyParameterEstimator(
        sam3d_hf_repo = CONFIG.parameters_extraction.sam3d_id,
        sam3d_step = CONFIG.parameters_extraction.sam3d_step,
        bbox_padding = CONFIG.parameters_extraction.bbox_padding,
        smplx_model_path = CONFIG.data.smplx_model_path,
        mhr_model_path  = CONFIG.data.mhr_model_path,
        reid_threshold = CONFIG.parameters_extraction.reid_threshold,
        gallery_ema_alpha = CONFIG.parameters_extraction.gallery_moving_average_alpha,
        reid_match_window = getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
    )
    reidentifier = CrossViewReidentifier(
        threshold = getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
        appearance_weight = getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.7),
        shape_weight = getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.3),
    )
    print(f"\n--- Running body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Step 3: Match person IDs across camera views
    print(f"\n--- Running cross-view person re-identification ---")
    reidentifier.match_across_views(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Derive the shared scene output directory (parent of all video dirs)
    scene_output_dir = Path(next(iter(video_dirs.values()))).parent

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

    # Step 5: Camera alignment
    # Estimate pairwise relative camera poses from the cross-view body
    # correspondences produced by the estimation + ReID steps, then persist
    # the result as camera_alignment.npz in the scene directory.
    print(f"\n--- Step 5: Camera alignment ---")
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

    # Step 6: FusionDataset compatibility check
    # Instantiate EgoExoFusionDataset on the scene output directory and verify
    # that the pipeline output is compatible with the SST fusion models' input.
    print(f"\n--- Step 6: FusionDataset compatibility check ---")
    try:
        # ds = EgoExoFusionDataset(
        #     scene_dir=scene_output_dir,
        #     window_size=32,
        #     window_stride=16,
        # )
        ds = RICHFusionDataset(
            scene_dir=scene_output_dir,
            window_size=32,
            window_stride=16,
        )
        print(f"  Dataset: {ds}")
        # Sample one batch and report tensor shapes
        if len(ds) > 0:
            inputs, targets = ds[0]
            print("  Inputs:")
            for k, v in inputs.items():
                print(f"    {k}: {tuple(v.shape)} dtype={v.dtype}")
            print("  Targets:")
            for k, v in targets.items():
                print(f"    {k}: {tuple(v.shape)} dtype={v.dtype}")
            print("  FusionDataset compatibility: OK")
        else:
            print("  WARNING: dataset is empty — check frame coverage.")
    except Exception as e:
        print(f"  ERROR: FusionDataset failed to load: {e}")

    # Step 7: Inspect output format for each video
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

            if "frame_indices" in data:
                print(f"    -> frame_indices (first 5): {data['frame_indices'][:5].tolist()}")
            if "pred_keypoints_3d" in data:
                kp3d = data["pred_keypoints_3d"]
                print(f"    -> pred_keypoints_3d[0] (first joint): {kp3d[0, 0].tolist()}")
            if "pred_cam_t" in data:
                print(f"    -> pred_cam_t[0]: {data['pred_cam_t'][0].tolist()}")
            if "bbox" in data:
                print(f"    -> bbox[0]: {data['bbox'][0].tolist()}")

        summary_path = body_dir / "body_params_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"\n  Summary JSON for {video_id}:")
            print(f"  {json.dumps(summary, indent=4)}")
        else:
            print(f"  WARNING: summary JSON not found at {summary_path}")

    # Step 8: Visualise the re-ID corrected segmentation.
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


if __name__ == "__main__":
    main()
