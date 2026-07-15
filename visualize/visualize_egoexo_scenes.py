"""Generate segmentation-ReID visualisation videos for specific EgoExo4D scenes."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configuration import CONFIG
from data.video_dataset import EgoExo4DSceneDataset
from utilities.visualize_segmented_reids import visualize_reid

FRAMES_ROOT = Path(CONFIG.data.egoexo4d_frames_root)
OUTPUT_ROOT = Path(CONFIG.data.egoexo4d_output_dir)

SCENES = [
    "cmu_soccer12_2",
    "georgiatech_bike_06_12",
    "georgiatech_bike_06_2",
    "georgiatech_bike_06_6",
    "georgiatech_bike_06_8",
    "georgiatech_bike_07_10",
    "georgiatech_bike_07_12",
    "georgiatech_bike_07_2",
    "georgiatech_bike_07_4",
    "georgiatech_bike_07_6",
    "georgiatech_bike_07_8",
    "georgiatech_bike_14_12",
    "georgiatech_bike_14_2",
    "georgiatech_bike_14_6",
    "georgiatech_bike_14_8",
    "georgiatech_bike_15_2",
    "georgiatech_bike_15_4",
    "georgiatech_bike_15_6",
    "georgiatech_bike_15_8",
    "georgiatech_bike_16_2",
    "georgiatech_bike_16_6",
    "georgiatech_bike_16_8",
    "georgiatech_covid_02_10",
    "georgiatech_covid_02_12",
    "georgiatech_covid_02_14",
    "georgiatech_covid_02_2",
    "georgiatech_covid_02_4",
    "georgiatech_covid_04_10",
    "georgiatech_covid_04_12",
    "georgiatech_covid_04_4",
    "georgiatech_covid_04_6",
    "georgiatech_covid_06_2",
    "georgiatech_covid_06_4",
    "georgiatech_covid_18_10",
    "georgiatech_covid_18_12",
    "georgiatech_covid_18_2",
    "georgiatech_covid_18_4",
    "georgiatech_covid_18_6",
    "georgiatech_covid_18_8",
    "iiith_cooking_59_2",
    "iiith_cooking_64_2",
    "iiith_cooking_89_6",
    "iiith_cooking_90_4",
]

dataset = EgoExo4DSceneDataset(frames_root=FRAMES_ROOT, max_side=1440)
scene_map = {s.scene_id: s for s in dataset.scenes}

for scene_name in SCENES:
    scene = scene_map.get(scene_name)
    if scene is None:
        print(f"[WARN] {scene_name} not found in dataset")
        continue
    print(f"\n=== {scene_name} ===")
    for video in scene.videos:
        video_dir = OUTPUT_ROOT / scene_name / video.video_id
        if not video_dir.exists():
            print(f"  {video.video_id}: no output dir, skipping")
            continue
        print(f"  {video.video_id}")
        try:
            visualize_reid(
                video_dir=video_dir,
                fps=int(video.fps),
                frames_dir=video.frames_home,
            )
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

print("\nDone.")
