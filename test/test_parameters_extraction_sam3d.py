import sys
from pathlib import Path

# Required for MHR conversion imports (mirrors main.py setup)
sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import numpy as np

from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from data.parameters_extraction import BodyParameterEstimator


def main():
    data_root = "/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"
    output_dir = "/cluster/project/cvg/students/tnanni/ghost/test_outputs/segmentation_test"

    # Load dataset (1 scene for testing)
    ds = EgoExoSceneDataset(data_root, slice=1)
    scene = ds[0]
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    segmenter = PersonSegmenter()
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=False,
        match_across_videos=True,
    )
    print(f"\nSegmentation output dirs:")
    for video_id, vdir in video_dirs.items():
        print(f"  {video_id}: {vdir}")

    # Step 2: Estimate body parameters from segmentation output
    estimator = BodyParameterEstimator(smooth=False)
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

        npz_files = sorted(body_dir.glob("person_*.npz"))
        print(f"\n--- {video_id}: {len(npz_files)} person file(s) ---")
        for npz_path in npz_files:
            data = dict(np.load(str(npz_path), allow_pickle=False))
            print(f"  {npz_path.name}:")
            for key, arr in sorted(data.items()):
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

            # Print a sample from the first frame for key arrays
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


if __name__ == "__main__":
    main()
