"""
Analyze camera quality for BBQ_001_guitar:

1. Compare Kabsch-predicted cameras (from camera_alignment.npz, in cam0 frame)
   against GT cameras (from scan_calibration XMLs, re-expressed in cam0 frame).

2. Analyze SAM3D body estimates per camera:
   - Translation (pred_cam_t) vs GT body position in camera frame
   - Global orientation (smplx_global_orient) vs GT orientation in camera frame
"""

import sys
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCENE = "BBQ_001_guitar"
PRED_DIR   = Path("/cluster/project/cvg/students/tnanni/ghost/test_outputs/focal_single_segmentation_test") / SCENE
RICH_ROOT  = Path("/cluster/project/cvg/data/rich/ps/project/multi-ioi/rich_release/train")
CALIB_DIR  = RICH_ROOT / "scan_calibration" / "BBQ" / "calibration"
GT_BODY    = RICH_ROOT / "train_body" / SCENE
NUM_CAMS   = 8

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

# ──────────────────────────────────────────────
# 1. Load GT calibration and compute cam0-relative poses
# ──────────────────────────────────────────────
print("=" * 60)
print("PART 1 — PREDICTED vs GT CAMERAS (cam0 frame)")
print("=" * 60)

gt_cams = []
for i in range(NUM_CAMS):
    calib = parse_xml(CALIB_DIR / f"{i:03d}.xml")
    gt_cams.append(calib)

ext0 = gt_cams[0]['extrinsics']   # (3,4) world→cam0
R0, t0 = ext0[:3, :3], ext0[:3, 3]

# GT relative poses in cam0 frame: x_i = R_rel @ x_0 + t_rel
gt_rel = []
for i, calib in enumerate(gt_cams):
    ext_i = calib['extrinsics']
    R_i, t_i = ext_i[:3, :3], ext_i[:3, 3]
    R_rel = R_i @ R0.T
    t_rel = t_i - R_rel @ t0
    gt_rel.append((R_rel, t_rel))
    intr = calib['intrinsics']
    f_gt = float(intr[0, 0]) if intr is not None else float('nan')
    print(f"  cam_{i:02d}  GT focal = {f_gt:.1f} px")

# ──────────────────────────────────────────────
# 2. Load predicted poses from camera_alignment.npz
# ──────────────────────────────────────────────
align = np.load(PRED_DIR / "camera_alignment.npz")

print("\n  cam   |  Rot err (°)  |  Trans err (m)  |  pred |t| (m)  |  GT |t| (m)")
print("  ------|---------------|-----------------|----------------|---------------")

for i in range(1, NUM_CAMS):
    cam_name = f"cam_{i:02d}"
    key_R = f"cam_00__to__{cam_name}__R"
    key_t = f"cam_00__to__{cam_name}__t"
    if key_R not in align:
        print(f"  cam_{i:02d} | NOT FOUND in alignment")
        continue

    R_pred = align[key_R]   # (3,3)
    t_pred = align[key_t]   # (3,)
    R_gt, t_gt = gt_rel[i]

    re = rot_error_deg(R_gt, R_pred)
    te = float(np.linalg.norm(t_pred - t_gt))
    print(f"  cam_{i:02d} |  {re:10.2f}   |  {te:12.3f}   |  {np.linalg.norm(t_pred):11.3f}   |  {np.linalg.norm(t_gt):.3f}")

# ──────────────────────────────────────────────
# 3. Analyze SAM3D body estimates vs GT body
# ──────────────────────────────────────────────
print("\n")
print("=" * 60)
print("PART 2 — SAM3D BODY ESTIMATES vs GT (per camera)")
print("=" * 60)

# Load GT body data (world frame) — use the common frame indices
gt_frames: dict[int, dict] = {}
if GT_BODY.is_dir():
    for frame_dir in sorted(GT_BODY.iterdir()):
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

for i in range(NUM_CAMS):
    cam_name = f"cam_{i:02d}"
    body_dir = PRED_DIR / cam_name / "body_data"
    npz_files = sorted(body_dir.glob("person_*.npz"))
    if not npz_files:
        print(f"  {cam_name} | NO BODY DATA")
        continue

    # Use first (only) person
    data = np.load(npz_files[0])
    frame_indices = data['frame_indices'].astype(int)
    pred_cam_t = data['pred_cam_t']                 # (T, 3) in camera frame
    pred_go    = data['smplx_global_orient']         # (T, 3) axis-angle in camera frame

    # GT camera extrinsic
    ext_i = gt_cams[i]['extrinsics']
    R_cam, t_cam = ext_i[:3, :3], ext_i[:3, 3]

    trans_errs = []
    rot_errs   = []

    for t_idx, fidx in enumerate(frame_indices):
        if fidx not in gt_frames:
            continue
        gt = gt_frames[fidx]

        # GT body translation in camera frame
        t_world = gt['transl']
        t_cam_gt = R_cam @ t_world + t_cam

        # GT global orient in camera frame: R_cam_frame = R_cam @ R_world
        R_world_gt = rotvec_to_mat(gt['global_orient'])
        R_cam_gt   = R_cam @ R_world_gt

        # Predicted translation and orientation
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
