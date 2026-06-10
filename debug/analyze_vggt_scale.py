"""
Compare per-frame GT scale vs placer-predicted scale for a RICH scene.

GT scale: median of ||C_gt_k|| / ||C_vggt_k|| across valid non-reference cameras.
Pred scale: BodyPlacer.estimate_scale_per_frame() (bone-length based).
"""

import numpy as np
import xml.etree.ElementTree as ET
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCENE = "BBQ_001_juggle"
SCENE_DIR = Path(f"/iopsstor/scratch/cscs/tnanni/ghost_outputs/rich11_segmentation_test/{SCENE}")
CALIB_DIR = "/capstor/scratch/cscs/tnanni/datasets/rich/scan_calibration/BBQ/calibration"
SMPLX_MODEL = "body_models/SMPLX_NEUTRAL.pkl"


def parse_gt_extrinsic(calib_dir, cam_name):
    cam_idx = int(cam_name.split("_")[1])
    tree = ET.parse(os.path.join(calib_dir, f"{cam_idx:03d}.xml"))
    vals = list(map(float, tree.getroot().find("CameraMatrix/data").text.split()))
    return np.array(vals, dtype=np.float64).reshape(3, 4)


def camera_center(ext):
    R, t = ext[:, :3], ext[:, 3]
    return -R.T @ t


def compute_gt_scale_per_frame(vggt_ext, cam_valid, cam_names, calib_dir):
    """Median of ||C_gt_k|| / ||C_vggt_k|| across valid non-ref cameras, per frame."""
    T = vggt_ext.shape[0]
    other_cams = cam_names[1:]

    gt_dists = {}
    for name in other_cams:
        ext = parse_gt_extrinsic(calib_dir, name)
        gt_dists[name] = np.linalg.norm(camera_center(ext))

    scales = np.full(T, np.nan)
    for t in range(T):
        ratios = []
        for name in other_cams:
            k = cam_names.index(name)
            if not cam_valid[t, k]:
                continue
            C_v = camera_center(vggt_ext[t, k].astype(np.float64))
            d_v = np.linalg.norm(C_v)
            if d_v < 1e-6:
                continue
            ratios.append(gt_dists[name] / d_v)
        if ratios:
            scales[t] = np.median(ratios)
    return scales


def main():
    cam_npz = np.load(SCENE_DIR / "vggt_cameras.npz")
    vggt_ext  = cam_npz["extrinsics"]   # (T, K, 3, 4)
    cam_valid = cam_npz["valid"]        # (T, K)
    cam_names = [n.decode() for n in cam_npz["camera_names"]]
    T = vggt_ext.shape[0]

    print(f"Scene: {SCENE}, T={T}")

    # --- GT scale per frame ---
    print("Computing GT scale per frame...")
    gt_scale = compute_gt_scale_per_frame(vggt_ext, cam_valid, cam_names, CALIB_DIR)

    # --- Predicted scale per frame ---
    print("Running BodyPlacer.estimate_scale_per_frame()...")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from fusion.placer import BodyPlacer
    placer = BodyPlacer(SCENE_DIR, SMPLX_MODEL)
    pred_scale = placer.estimate_scale_per_frame()

    # --- Stats ---
    valid_gt = gt_scale[~np.isnan(gt_scale)]
    valid_pred = pred_scale[~np.isnan(pred_scale)]
    print(f"\nGT scale   — mean={valid_gt.mean():.4f}  std={valid_gt.std():.4f}  "
          f"median={np.median(valid_gt):.4f}  p5={np.percentile(valid_gt,5):.4f}  p95={np.percentile(valid_gt,95):.4f}")
    print(f"Pred scale — mean={valid_pred.mean():.4f}  std={valid_pred.std():.4f}  "
          f"median={np.median(valid_pred):.4f}  p5={np.percentile(valid_pred,5):.4f}  p95={np.percentile(valid_pred,95):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(gt_scale, lw=0.9, color='steelblue', label='GT scale')
    axes[0].axhline(np.nanmedian(gt_scale), color='steelblue', lw=1.2, ls='--',
                    label=f'median={np.nanmedian(gt_scale):.3f}')
    axes[0].set_ylabel('scale (m / VGGT-unit)')
    axes[0].set_title('GT scale per frame')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, max(15, np.nanpercentile(gt_scale, 99) * 1.2))

    axes[1].plot(pred_scale, lw=0.9, color='darkorange', label='Pred scale (bone-length)')
    axes[1].axhline(np.nanmedian(pred_scale), color='darkorange', lw=1.2, ls='--',
                    label=f'median={np.nanmedian(pred_scale):.3f}')
    axes[1].set_ylabel('scale (m / VGGT-unit)')
    axes[1].set_title('Predicted scale per frame (BodyPlacer)')
    axes[1].set_xlabel('frame')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, max(15, np.nanpercentile(pred_scale, 99) * 1.2))

    fig.suptitle(f'VGGT scale — {SCENE}', fontsize=11)
    plt.tight_layout()
    out_path = f"results/vggt_scale_{SCENE}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
