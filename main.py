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
from preprocessing.segmentation import PersonSegmenter
from preprocessing.parameters_extraction import BodyParameterEstimator, CrossViewReidentifier
from synchronize_videos.synchronizer import Synchronizer


def load_body_data(
    scene,
    video_dir_dict: dict,
    device: str = "cuda",
) -> tuple[list, list]:
    """Load body joints and confidences from pipeline output for synchronization.

    Parameters
    ----------
    scene : Scene
        Scene object with video metadata.
    video_dir_dict : dict
        Maps video_id -> Path to that video's output directory.
    device : str
        Torch device string.

    Returns
    -------
    body_joints_list : list[list[Tensor]]
        K videos × P persons, each tensor shape (T, J, 3).
    confidences_list : list[list[Tensor]]
        K videos × P persons, each tensor shape (T, J).
    """
    body_joints_list = []
    confidences_list = []

    for video in scene.videos:
        video_dir = video_dir_dict.get(video.video_id)
        if video_dir is None:
            continue
        body_dir = Path(video_dir) / "body_data"
        if not body_dir.is_dir():
            continue

        per_person_joints: list[torch.Tensor] = []
        per_person_confs: list[torch.Tensor] = []

        for npz_path in sorted(body_dir.glob("person_*.npz")):
            data = np.load(str(npz_path), allow_pickle=False)
            if "pred_keypoints_3d" not in data:
                continue
            kpts = data["pred_keypoints_3d"]  # (T, J, 3)
            if len(kpts) == 0:
                continue
            joints = torch.from_numpy(kpts).float().to(device)  # (T, J, 3)
            T, J = joints.shape[:2]
            conf = torch.ones(T, J, device=device)
            per_person_joints.append(joints)
            per_person_confs.append(conf)

        if per_person_joints:
            body_joints_list.append(per_person_joints)
            confidences_list.append(per_person_confs)

    return body_joints_list, confidences_list


def parse_args():
    parser = argparse.ArgumentParser(description="EgoExo4D person segmentation and body parameter estimation")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing scene folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write results")
    parser.add_argument("--slice", type=int, default=None, help="Only process the first N scenes")

    # Segmentation
    parser.add_argument("--sam3_checkpoint", type=str, default=None,
                        help="Path to SAM3 checkpoint; None = auto-download from HuggingFace")
    parser.add_argument("--text_prompt", type=str, default="person")
    parser.add_argument("--redetect_interval", type=int, default=5,
                        help="Add text prompt every N frames so late arrivals are detected")
    parser.add_argument("--new_det_thresh", type=float, default=0.4)
    parser.add_argument("--score_threshold_detection", type=float, default=0.3)

    # Body estimation
    parser.add_argument("--sam3d_hf_repo", type=str, default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--sam3d_step", type=int, default=1, help="Run SAM3D every N frames")
    parser.add_argument("--smplx_model_path", type=str, default=None, help="Path to SMPLX_NEUTRAL.npz")
    parser.add_argument("--mhr_model_path", type=str, default=None, help="Path to MHR assets folder")
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
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        text_prompt=args.text_prompt,
        redetect_interval=args.redetect_interval,
        new_det_thresh=args.new_det_thresh,
        score_threshold_detection=args.score_threshold_detection,
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
        smplx_model_path=args.smplx_model_path,
        mhr_model_path=args.mhr_model_path,
    )
    reidentifier = CrossViewReidentifier()

    # Extract people parameters
    for scene in tqdm(dataset.scenes, desc="Extracting Body Parameters from scenes"):
        video_dir_dict = scene_directories[scene.scene_id]
        parameters_extractor.estimate_scene(
            scene = scene,
            video_dirs = video_dir_dict,
        )

    # Match person identities across camera views so that body_data/,
    # mask_data.npz, and json_data/ use consistent IDs throughout every scene.
    for scene in tqdm(dataset.scenes, desc="Cross-view person re-identification"):
        video_dir_dict = scene_directories[scene.scene_id]
        reidentifier.match_across_views(
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
