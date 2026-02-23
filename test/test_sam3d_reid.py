import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import numpy as np

from configuration import CONFIG
from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from data.parameters_extraction import BodyParameterEstimator
from utilities.visualize_segmented_reids import visualize_reid

def main():

    data_root = CONFIG.data.data_root
    output_dir = CONFIG.data.output_directory
    scenes_slice = CONFIG.data.slice
    exclude_ego = CONFIG.data.exclude_egocentric

    # Load dataset (1 scene for testing)
    ds = EgoExoSceneDataset(
        data_root = data_root, 
        slice=scenes_slice, 
        exclude_ego=exclude_ego,
    )
    scene = ds[0]
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    segmenter = PersonSegmenter(
        sam2_checkpoint=CONFIG.segmentation.sam2_checkpoint,
        model_cfg=CONFIG.segmentation.sam2_cfg,
        gdino_model_id=CONFIG.segmentation.gdino_id,
        box_threshold=CONFIG.segmentation.box_threshold,
        text_threshold=CONFIG.segmentation.text_threshold,
        detection_step=CONFIG.segmentation.detection_step,
        min_mask_area=CONFIG.segmentation.min_mask_area,
        max_no_detection_windows=CONFIG.segmentation.max_no_detection_windows,
    )
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=False,
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
    )
    print(f"\n--- Running body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Step 3: Inspect output format for each video
    print(f"\n=== Body parameter output format ===")
    for video_id, video_dir in video_dirs.items():
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.exists():
            print(f"  WARNING: {body_dir} does not exist")
            continue

        # Count unique SAM2 person IDs seen across all JSON frames.
        json_dir = Path(video_dir) / "json_data"
        sam2_ids: set[int] = set()
        for jp in sorted(json_dir.glob("*.json")):
            with open(jp) as f:
                meta = json.load(f)
            for sid in meta.get("labels", {}):
                sam2_ids.add(int(sid))

        npz_files = sorted(body_dir.glob("person_*.npz"))
        print(f"\n--- {video_id}: {len(npz_files)} person file(s) ---")

        # Re-ID summary: if fewer tracks than SAM2 IDs, merges happened.
        if sam2_ids:
            n_merged = len(sam2_ids) - len(npz_files)
            merge_str = f"  ({n_merged} SAM2 ID(s) merged by re-ID)" if n_merged > 0 else "  (no merges)"
            print(
                f"  SAM2 unique IDs across all frames: {sorted(sam2_ids)}\n"
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

    # Step 4: Visualise the re-ID corrected segmentation.
    print(f"\n--- Visualising re-ID corrected segmentation ---")
    for video in scene.videos:
        if video.video_id not in video_dirs:
            continue
        print(f"  {video.video_id}")
        visualize_reid(
            video_dir=Path(video_dirs[video.video_id]),
            fps=int(video.fps),
        )


if __name__ == "__main__":
    main()
