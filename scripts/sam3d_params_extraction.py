"""
Test only for checking that the parameters are extracted correctly and SAM3D loads well
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'MHR' / 'tools' / 'mhr_smpl_conversion'))

import json
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn

from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from data.parameters_extraction import BodyParameterEstimator
from configuration import CONFIG

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


if __name__ == "__main__":
    main()
