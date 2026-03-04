"""
Minimal single-frame SAM-3D Body test.

Picks the first segmented frame that has person detections in the existing
segmentation output, runs SAM-3D Body on it, and saves a visualization.

Assumes segmentation has already been run (test_parameters_extraction_sam3d.py).
"""

import sys
import json
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sam-3d-body"))

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

SEGMENTATION_DIR = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs/first_sam3_segmentation_test_15"
)
OUTPUT_DIR = Path(
    "/cluster/project/cvg/students/tnanni/ghost/test_outputs/sam3d_single_frame"
)
HF_REPO = "facebook/sam-3d-body-dinov3"


def find_first_frame_with_people(seg_dir: Path, index: int = 0):
    """Walk the segmentation output and return (frame_path, bbox_array) for
    the nth frame (0-based) that has at least one person detection."""
    count = 0
    for scene_dir in sorted(seg_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for video_dir in sorted(scene_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            json_dir = video_dir / "json_data"
            frame_dir = video_dir / "frames"
            if not json_dir.exists() or not frame_dir.exists():
                continue
            for json_path in sorted(json_dir.glob("*.json")):
                with open(json_path) as f:
                    meta = json.load(f)
                labels = meta.get("labels", {})
                if not labels:
                    continue
                frame_idx_str = json_path.stem.replace("mask_", "")
                frame_path = frame_dir / f"{frame_idx_str}.jpg"
                if not frame_path.exists():
                    continue
                bboxes = np.array(
                    [
                        [info["x1"], info["y1"], info["x2"], info["y2"]]
                        for info in labels.values()
                        if (info["x2"] - info["x1"]) >= 10
                        and (info["y2"] - info["y1"]) >= 10
                    ],
                    dtype=np.float32,
                )
                if len(bboxes) == 0:
                    continue
                if count < index:
                    count += 1
                    continue
                return frame_path, bboxes, json_path
    return None, None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Find a frame to test on ---
    frame_path, bboxes, json_path = find_first_frame_with_people(SEGMENTATION_DIR, index=99)
    if frame_path is None:
        raise RuntimeError(
            f"No segmented frames with people found under {SEGMENTATION_DIR}. "
            "Run test_parameters_extraction_sam3d.py first."
        )

    print(f"Frame : {frame_path}")
    print(f"JSON  : {json_path}")
    print(f"Bboxes: {bboxes}")

    # --- Load image ---
    img_bgr = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Load SAM-3D Body ---
    try:
        estimator = setup_sam_3d_body(hf_repo_id=HF_REPO)
    except Exception as e:
        print(traceback.format_exc())

    # --- Run inference ---
    print("bjqsbfjwebfwe")
    outputs = estimator.process_one_image(img_rgb, bboxes=bboxes)
    print("nwkebfiwekhfiwer")

    if not outputs:
        print("SAM-3D Body returned no outputs for this frame.")
        return

    print(f"\n{len(outputs)} person(s) detected:")
    for i, out in enumerate(outputs):
        print(f"    Keys:  {list(out.keys())}")
        print(f"    person {i}: bbox={out['bbox'].tolist()}")
        print(f"    pred_cam_t      : {out['pred_cam_t'].tolist()}")
        print(f"    pred_keypoints_3d shape: {out['pred_keypoints_3d'].shape}")
        print(f"    pred_vertices   shape: {out['pred_vertices'].shape}")
        print(f"    focal_length    : {out['focal_length']:.2f}")

    # --- Visualize ---
    # visualize_sample_together produces a 4-panel strip:
    # [original | 2D keypoints | 3D mesh overlay | side view]
    vis = visualize_sample_together(img_bgr, outputs, estimator.faces)

    out_path = OUTPUT_DIR / f"{frame_path.stem}_sam3d.jpg"
    cv2.imwrite(str(out_path), vis.astype(np.uint8))
    print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    main()