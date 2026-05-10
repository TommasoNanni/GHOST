import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

import gc as _gc
import numpy as np
import torch

from configuration import CONFIG
from data.video_dataset import EgoHumansDataset
from data.fusion_dataset import EgoHumansFusionDatapoint
from preprocessing.camera_alignment import CameraAlignment
from preprocessing.geometric_reidentifier import GeometricReidentifier
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction import ParametersExtractor, CrossVideoReidentifier
from utilities.visualize_segmented_reids import visualize_reid


def process_scene(scene, segmenter, estimator, reidentifier, output_dir):
    """Run the full pipeline on a single EgoHumans scene."""
    print(f"\n=== Scene: {scene.scene_id} ({len(scene)} videos) ===")
    for v in scene:
        print(f"  {v}")

    # Step 1: Segment people in the scene
    print(f"\n--- Running segmentation on scene '{scene.scene_id}' ---")
    _seg_output_dir = Path(output_dir) / scene.scene_id
    _all_segmented = all(
        PersonSegmenter._is_segmented(_seg_output_dir / v.video_id)
        for v in scene.videos
    )
    video_dirs = segmenter.segment_scene(
        scene=scene,
        output_dir=output_dir,
        vis=False,
    )
    print(f"\nSegmentation output dirs:")
    for video_id, vdir in video_dirs.items():
        print(f"  {video_id}: {vdir}")

    # Step 2: Estimate body parameters from segmentation output
    print(f"\n--- Running body parameter estimation ---")
    estimator.estimate_scene(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Guard: skip ReID if any camera is missing body data
    missing_body = [
        vid_id for vid_id, vid_dir in video_dirs.items()
        if not any((Path(vid_dir) / "body_data").glob("person_*.npz"))
    ]
    if missing_body:
        print(
            f"  WARNING: body estimation incomplete for cameras {missing_body} — "
            f"skipping cross-view ReID and subsequent steps for this scene."
        )
        return

    # Step 3: Match person IDs across camera views
    print(f"\n--- Running cross-view person re-identification ---")
    scene_output_dir = Path(next(iter(video_dirs.values()))).parent
    reidentifier.match_across_views(
        scene=scene,
        video_dirs=video_dirs,
    )

    # Step 4: Verify SMPLX conversion output
    print(f"\n--- Verifying MHR → SMPLX conversion output ---")
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
    if not smplx_fields_found:
        print(
            "  WARNING: No smplx_* fields found. "
            "Check that smplx_model_path and mhr_model_path are set in CONFIG."
        )

    # Step 5: Camera alignment
    print(f"\n--- Step 5: Camera alignment ---")
    alignment = CameraAlignment().estimate(video_dirs, min_correspondences=30, scene_dir=scene_output_dir)
    if alignment:
        align_path = CameraAlignment.save(alignment, scene_output_dir)
        print(f"  Estimated {len(alignment)} camera pair(s) → saved to {align_path}")
    else:
        print(
            "  WARNING: No camera pairs could be aligned. "
            "Check that cross-view ReID found shared persons across videos."
        )

    # Step 5b: Geometric post-ReID
    geo_reid = GeometricReidentifier(
        distance_threshold=CONFIG.parameters_extraction.geo_reid_distance_threshold,
        min_overlap_frames=10,
        second_pass_threshold=CONFIG.parameters_extraction.geo_reid_second_pass_threshold,
    )
    geo_reid.reidentify(scene=scene, video_dirs=video_dirs)

    # Step 6: FusionDatapoint compatibility check
    # EgoHumansFusionDatapoint needs egohumans_seq_dir to load GT calibration
    # from the COLMAP output (cameras.txt / images.txt) and SMPL annotations.
    print(f"\n--- Step 6: FusionDatapoint compatibility check ---")
    try:
        fusion_dp = EgoHumansFusionDatapoint(
            scene_dir=scene_output_dir,
            egohumans_seq_dir=scene.seq_dir,
        )
        from data.fusion_dataset import FusionDataset
        ds = FusionDataset([fusion_dp])
        inputs, targets = ds[0]
        print("  FusionDatapoint compatibility: OK")
    except Exception as e:
        print(f"  ERROR: FusionDatapoint failed to load: {e}")

    # Step 7: Visualise the re-ID corrected segmentation for any video missing its mp4
    print(f"\n--- Visualising re-ID corrected segmentation ---")
    for video in scene.videos:
        if video.video_id not in video_dirs:
            continue
        vid_dir = Path(video_dirs[video.video_id])
        vis_path = vid_dir / f"{video.video_id}_segmentation_reid.mp4"
        if vis_path.exists():
            print(f"  {video.video_id}: already exists, skipping")
            continue
        print(f"  {video.video_id}")
        try:
            visualize_reid(
                video_dir=vid_dir,
                fps=int(video.fps),
                frames_dir=video.frames_home,
            )
        except FileNotFoundError as e:
            print(f"  WARNING: skipping visualisation — {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-start", type=int, default=None)
    parser.add_argument("--scene-end",   type=int, default=None)
    parser.add_argument("--activities",  nargs="+", default=None,
                        help="Limit to specific activity folders, e.g. 01_tagging 02_lego")
    args = parser.parse_args()

    egohumans_data_root = CONFIG.data.egohumans_data_root
    output_dir = CONFIG.data.egohumans_output_directory
    scenes_slice = CONFIG.data.slice

    ds = EgoHumansDataset(
        data_root=egohumans_data_root,
        slice=scenes_slice,
        activities=args.activities,
    )

    if args.scene_start is not None or args.scene_end is not None:
        ds.scenes = ds.scenes[args.scene_start:args.scene_end]
        print(f"Processing scenes [{args.scene_start}:{args.scene_end}] → {len(ds.scenes)} scenes")

    failed_videos_by_scene: dict[str, list[str]] = {}
    needs_reid: list[str] = []

    for scene in ds.scenes:
        segmenter = PersonSegmenter(
            checkpoint_path=CONFIG.segmentation.checkpoint_path,
            text_prompt=CONFIG.segmentation.text_prompt,
            redetect_interval=CONFIG.segmentation.redetect_interval,
            new_det_thresh=CONFIG.segmentation.new_det_thresh,
            score_threshold_detection=CONFIG.segmentation.score_threshold_detection,
        )
        estimator = ParametersExtractor(
            sam3d_hf_repo=CONFIG.parameters_extraction.sam3d_id,
            sam3d_step=CONFIG.parameters_extraction.sam3d_step,
            bbox_padding=CONFIG.parameters_extraction.bbox_padding,
            smplx_model_path=CONFIG.data.smplx_model_path,
            mhr_model_path=CONFIG.data.mhr_model_path,
            reid_threshold=CONFIG.parameters_extraction.reid_threshold,
            reid_match_window=getattr(CONFIG.parameters_extraction, "reid_match_window", 5),
        )
        reidentifier = CrossVideoReidentifier(
            threshold=getattr(CONFIG.parameters_extraction, "cross_view_reid_threshold", 0.4),
            appearance_weight=getattr(CONFIG.parameters_extraction, "cross_view_appearance_weight", 0.5),
            shape_weight=getattr(CONFIG.parameters_extraction, "cross_view_shape_weight", 0.2),
            pose_weight=getattr(CONFIG.parameters_extraction, "cross_view_pose_weight", 0.3),
        )
        try:
            process_scene(scene, segmenter, estimator, reidentifier, output_dir)
        except Exception as e:
            logging.error(
                f"Scene {scene.scene_id} raised an unexpected error: {e}", exc_info=True
            )
        finally:
            del segmenter, estimator, reidentifier
            _gc.collect()
            torch.cuda.empty_cache()

        scene_dir = Path(output_dir) / scene.scene_id
        missing = [
            v.video_id for v in scene.videos
            if not any((scene_dir / v.video_id / "body_data").glob("person_*.npz"))
        ]
        if missing:
            failed_videos_by_scene[scene.scene_id] = missing
        reid_done = (scene_dir / "cross_view_reid.json").exists()
        has_any_body = len(missing) < len(scene.videos)
        if has_any_body and not reid_done:
            needs_reid.append(scene.scene_id)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    if failed_videos_by_scene:
        print(f"\nScenes with missing body data ({len(failed_videos_by_scene)}):")
        for scene_id, cams in failed_videos_by_scene.items():
            print(f"  {scene_id}: missing cameras → {cams}")
    else:
        print("\nBody estimation completed for all cameras in all scenes.")
    if needs_reid:
        print(f"\nScenes with body data but no cross-view ReID ({len(needs_reid)}):")
        for scene_id in needs_reid:
            print(f"  {scene_id}")
    else:
        print("\nCross-view ReID completed for all scenes.")
    print("=" * 64)


if __name__ == "__main__":
    main()
