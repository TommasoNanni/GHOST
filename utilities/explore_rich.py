from pathlib import Path
import json
import pickle

rich_path = Path("/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train")
rich_camera = rich_path / "multicam2world" / "BBQ_multicam2world.json"
rich_data = rich_path / "train_body" / "BBQ_001_guitar" / "00005" / "001.pkl"

# with open(rich_camera) as f:
#     cam_data = json.load(f)

# print(json.dumps(cam_data, indent=4))

with open(rich_data , "rb") as f:
    body_summary = pickle.load(f)

print(body_summary.keys())