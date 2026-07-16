"""
Analyze SAM3D body estimates vs GT for a RICH scene:
  - Translation (pred_cam_t) vs GT body position in camera frame
  - Global orientation (smplx_global_orient) vs GT orientation in camera frame

Example call (copy-paste and change --scene):
    pixi run python debug/analyze_cameras.py \\
        --scene BBQ_001_guitar \\
        --pred_root /cluster/project/cvg/students/tnanni/ghost/test_outputs/focal_single_segmentation_test \\
        --rich_root /cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train \\
        --num_cams 8
"""

import argparse
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def parse_xml(xml_path: Path) -> dict:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    def _parse(tag):
        node = root.find(tag)
        if node is None:
            return None
        rows = int(node.findtext('rows', default='1'))
        cols = int(node.findtext('cols', default='1'))
        data = list(map(float, node.findtext('data', default='').split()))
        return np.array(data, dtype=np.float64).reshape(rows, cols)
    return {
        'extrinsics': _parse('CameraMatrix'),  # (3,4) world→cam
        'intrinsics': _parse('Intrinsics'),    # (3,3)
    }

def rot_error_deg(R_gt: np.ndarray, R_pred: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    R_diff = R_gt.T @ R_pred
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))

def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    return SciR.from_rotvec(rv.reshape(3)).as_matrix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze predicted SAM3D body estimates vs GT for a RICH scene.")
    parser.add_argument("--scene",     required=True, help="Scene name, e.g. BBQ_001_guitar")
    parser.add_argument("--pred_root", required=True, type=Path,
                        help="Parent output directory that contains the scene folder")
    parser.add_argument("--rich_root", required=True, type=Path,
                        help="Root of the RICH dataset (train split)")
    parser.add_argument("--num_cams",  type=int, default=8, help="Number of cameras (default: 8)")
    args = parser.parse_args()

    scene    = args.scene
    pred_dir = args.pred_root / scene
    rich_root = args.rich_root
    num_cams = args.num_cams

    # Derive calibration and GT body paths from scene name
    scene_stem = scene.split("_")[0]  # e.g. "BBQ_001_guitar" -> "BBQ"
    calib_dir  = rich_root / "scan_calibration" / scene_stem / "calibration"
    gt_body    = rich_root / "train_body" / scene

    # ──────────────────────────────────────────────
    # Load GT camera calibration (used to re-express GT body pose per camera)
    # ──────────────────────────────────────────────
    gt_cams = []
    for i in range(num_cams):
        calib = parse_xml(calib_dir / f"{i:03d}.xml")
        gt_cams.append(calib)
        intr = calib['intrinsics']
        f_gt = float(intr[0, 0]) if intr is not None else float('nan')
        print(f"  cam_{i:02d}  GT focal = {f_gt:.1f} px")

    # ──────────────────────────────────────────────
    # Analyze SAM3D body estimates vs GT body
    # ──────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("SAM3D BODY ESTIMATES vs GT (per camera)")
    print("=" * 60)

    gt_frames: dict[int, dict] = {}
    if gt_body.is_dir():
        for frame_dir in sorted(gt_body.iterdir()):
            if not frame_dir.is_dir():
                continue
            try:
                fidx = int(frame_dir.name)
            except ValueError:
                continue
            pkl_files = sorted(frame_dir.glob("*.pkl"))
            if not pkl_files:
                continue
            with open(pkl_files[0], 'rb') as f:
                params = pickle.load(f)
            gt_frames[fidx] = {
                'transl':         np.array(params['transl']).reshape(3),
                'global_orient':  np.array(params['global_orient']).reshape(3),
            }

    print(f"  Loaded GT body for {len(gt_frames)} frames\n")

    print("  cam   |  frames  |  trans err med (m)  |  trans err mean (m)  |  rot err med (°)  |  rot err mean (°)")
    print("  ------|----------|---------------------|----------------------|-------------------|------------------")

    for i in range(num_cams):
        cam_name = f"cam_{i:02d}"
        body_dir = pred_dir / cam_name / "body_data"
        npz_files = sorted(body_dir.glob("person_*.npz"))
        if not npz_files:
            print(f"  {cam_name} | NO BODY DATA")
            continue

        data = np.load(npz_files[0])
        frame_indices = data['frame_indices'].astype(int)
        pred_cam_t = data['pred_cam_t']                 # (T, 3) in camera frame
        pred_go    = data['smplx_global_orient']         # (T, 3) axis-angle in camera frame

        ext_i = gt_cams[i]['extrinsics']
        R_cam, t_cam = ext_i[:3, :3], ext_i[:3, 3]

        trans_errs = []
        rot_errs   = []

        for t_idx, fidx in enumerate(frame_indices):
            if fidx not in gt_frames:
                continue
            gt = gt_frames[fidx]

            t_world = gt['transl']
            t_cam_gt = R_cam @ t_world + t_cam

            R_world_gt = rotvec_to_mat(gt['global_orient'])
            R_cam_gt   = R_cam @ R_world_gt

            t_pred_cam = pred_cam_t[t_idx]
            R_pred_cam = rotvec_to_mat(pred_go[t_idx])

            trans_errs.append(float(np.linalg.norm(t_pred_cam - t_cam_gt)))
            rot_errs.append(rot_error_deg(R_cam_gt, R_pred_cam))

        if not trans_errs:
            print(f"  {cam_name} | no matching GT frames")
            continue

        te_med  = float(np.median(trans_errs))
        te_mean = float(np.mean(trans_errs))
        re_med  = float(np.median(rot_errs))
        re_mean = float(np.mean(rot_errs))
        print(f"  {cam_name} |  {len(trans_errs):6d}  |  {te_med:18.3f}   |  {te_mean:19.3f}   |  {re_med:16.1f}   |  {re_mean:.1f}")

    print()


if __name__ == "__main__":
    main()
