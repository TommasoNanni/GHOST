from data.video_dataset import EgoExoSceneDataset
from data.segmentation import PersonSegmenter

def main():
    data_root = "/cluster/project/cvg/data/EgoExo_georgiatech/raw/takes"
    video_dataset = EgoExoSceneDataset(data_root, slice = 3)
    segmenter = PersonSegmenter()
    first_scene = video_dataset.scenes[0]
    segmenter.segment_scene(first_scene, "/cluster/project/cvg/students/tnanni/Thesis/test_outputs/segmentation_test", match_across_videos=True, vis=True)

if __name__ == "__main__":
    main()