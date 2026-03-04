"""
Test for segmentation of one random scene for testing if the segmenter works
"""

from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter
from configuration import CONFIG

def main():
    data_root = CONFIG.data.data_root
    video_dataset = EgoExoSceneDataset(data_root, slice = 3)
    segmenter = PersonSegmenter(
        sam2_checkpoint=CONFIG.segmentation.sam2_checkpoint,
        model_cfg=CONFIG.segmentation.sam2_cfg,
        gdino_model_id=CONFIG.segmentation.gdino_id,
        box_threshold=CONFIG.segmentation.box_threshold,
        text_threshold=CONFIG.segmentation.text_threshold,
        detection_step=CONFIG.segmentation.detection_step,
    )
    first_scene = video_dataset.scenes[0]
    segmenter.segment_scene(
        first_scene, 
        "/cluster/project/cvg/students/tnanni/Thesis/test_outputs/segmentation_test", 
        vis=True
    )

if __name__ == "__main__":
    main()