# Add the path for the mhr conversion
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from data.video_dataset import Video, Scene, EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from data.parameters_extraction import BodyParameterEstimator
from synchronize_videos.synchronizer import Synchronizer


def parse_args():
    parser = argparse.ArgumentParser(description="EgoExo4D person segmentation and body parameter estimation")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing scene folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write results")
    parser.add_argument("--slice", type=int, default=None, help="Only process the first N scenes")

    # Segmentation
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gdino_model_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--detection_step", type=int, default=50)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--min_mask_area", type=int, default=500,
                        help="Min foreground pixels for a tracked mask to be carried forward")
    parser.add_argument("--max_no_detection_windows", type=int, default=2,
                        help="Max consecutive GDINO windows without detection before dropping a track")

    # Body estimation
    parser.add_argument("--sam3d_hf_repo", type=str, default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--sam3d_step", type=int, default=1, help="Run SAM3D every N frames")
    parser.add_argument("--smooth", action="store_true", help="Apply temporal smoothing to body params")

    # General
    parser.add_argument("--vis", action="store_true", help="Render visualisation videos")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args()


def main(args):
    # Load the data
    dataset = EgoExoSceneDataset(
        data_root = args.data_root, 
        slice=args.slice,
    )

    # Instatiate the segmenter to detect people
    segmenter = PersonSegmenter(
        sam2_checkpoint=args.sam2_checkpoint,
        model_cfg=args.model_cfg,
        gdino_model_id=args.gdino_model_id,
        device=args.device,
        box_threshold=args.box_threshold,
        detection_step=args.detection_step,
        min_mask_area=args.min_mask_area,
        max_no_detection_windows=args.max_no_detection_windows,
    )

    # Detect people in the dataset
    scene_directories = defaultdict()
    for scene in tqdm(dataset.scenes, desc="Segmenting scenes"):
        video_dir = segmenter.segment_scene(
            scene = scene,
            output_dir = args.output_dir,
            vis = args.vis,
        )
        scene_directories[scene.scene_id] = video_dir

    # Estimate the HMR parameters using SAM-3D-Body
    parameters_extractor = BodyParameterEstimator(
        sam3d_hf_repo=args.sam3d_hf_repo,
        sam3d_step=args.sam3d_step,
        smooth=args.smooth,
    )

    # Extract people parameters
    for scene in tqdm(dataset.scenes, desc="Extracting Body Parameters from scenes"):
        video_dir_dict = scene_directories[scene.scene_id]
        parameters_extractor.estimate_scene(
            scene = scene,
            video_dirs = video_dir_dict,
        )


    # Temporally align the videos
    synchronizer = Synchronizer(device=args.device)
    for scene in tqdm(dataset.scenes, desc="Synchronizing scenes"):
        video_dir_dict = scene_directories[scene.scene_id]

        # Load per-video joints and confidences for all people
        body_joints_list, confidences_list = load_body_data(
            scene, video_dir_dict, device=args.device,
        )
        if len(body_joints_list) < 2:
            logging.warning(f"Scene {scene.scene_id}: fewer than 2 videos with body data, skipping")
            continue

        offset_matrix = synchronizer.estimate_offset_matrix(body_joints_list, confidences_list)
        initial_times = synchronizer.estimate_initial_times(offset_matrix)

        logging.info(f"Scene {scene.scene_id} offsets (frames): {initial_times.cpu().tolist()}")

        # Apply the estimated offsets to the videos
        for video, t0 in zip(scene.videos, initial_times.cpu().tolist()):
            video.estimated_start = int(round(t0))

    return



if __name__ == "__main__":
    args = parse_args()
    main(args)
