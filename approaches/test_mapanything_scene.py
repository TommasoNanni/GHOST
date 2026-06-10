"""
Diagnostic script: run MapAnything on a single RICH scene frame and compare
predicted cameras / depth / metric scale against VGGT outputs and GT.

Usage:
    pixi run python approaches/test_mapanything_scene.py \
        --scene BBQ_001_guitar \
        --ghost_output_root /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train \
        --rich_root         /tmp/rich_train \
        --rich_gt_root      /capstor/scratch/cscs/tnanni/datasets/rich \
        --frame_idx         100
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--ghost_output_root", required=True)
    p.add_argument("--rich_root", required=True,
                   help="Mounted squash root with {scene}/{cam}/NNNNN_00.jpeg")
    p.add_argument("--rich_gt_root", required=True,
                   help="RICH dataset root containing scan_calibration/")
    p.add_argument("--frame_idx", type=int, default=-1,
                   help="Start frame (-1 = middle of sequence)")
    p.add_argument("--n_frames", type=int, default=1,
                   help="Number of consecutive frames to batch together (B dimension)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_geo", action="store_true",
                   help="Pass images only — no depth, intrinsics or camera poses")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GT calibration helpers (mirrors fusion_dataset.RICHFusionDatapoint)
# ---------------------------------------------------------------------------

def _scene_stem(scene_name: str) -> str:
    m = re.match(r'^(.+?)_\d{3}_', scene_name)
    return m.group(1) if m else scene_name


def _parse_calib_xml(xml_path: Path) -> dict:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    def _parse(tag):
        node = root.find(tag)
        if node is None:
            return None
        rows = int(node.findtext('rows', '1'))
        cols = int(node.findtext('cols', '1'))
        data = list(map(float, node.findtext('data', '').split()))
        return np.array(data, dtype=np.float64).reshape(rows, cols)
    return {'extrinsics': _parse('CameraMatrix'), 'intrinsics': _parse('Intrinsics')}


def load_gt_cameras(rich_gt_root: Path, scene_name: str,
                    cam_names: list[str]) -> dict[str, dict]:
    """Return {cam_name: {extrinsics (3,4), intrinsics (3,3)}} for each cam."""
    stem = _scene_stem(scene_name)
    calib_dir = rich_gt_root / "scan_calibration" / stem / "calibration"
    result = {}
    for cam in cam_names:
        m = re.search(r'\d+', cam)
        num = int(m.group()) if m else 0
        xml = calib_dir / f"{num:03d}.xml"
        if xml.exists():
            result[cam] = _parse_calib_xml(xml)
        else:
            print(f"  [WARN] No GT calibration for {cam} (expected {xml})")
    return result


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD  = [0.229, 0.224, 0.225]


def load_and_normalize_image(jpeg_path: Path, H_out: int, W_out: int) -> torch.Tensor:
    """Load JPEG, resize to (H_out, W_out), return (1, 3, H, W) DINOv2-normalized."""
    img = Image.open(jpeg_path).convert("RGB")
    img = img.resize((W_out, H_out), Image.BILINEAR)
    t = TF.to_tensor(img)                         # (3, H, W), [0, 1]
    t = TF.normalize(t, DINOV2_MEAN, DINOV2_STD)  # DINOv2 norm
    return t.unsqueeze(0)                          # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def world2cam_to_cam2world(ext: np.ndarray) -> np.ndarray:
    """Convert [R|t] (3,4) world2cam to 4×4 cam2world."""
    R = ext[:3, :3]
    t = ext[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3]  = -R.T @ t
    return c2w


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle between two rotation matrices in degrees."""
    trace = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_error_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Angle between two translation vectors in degrees."""
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float('nan')
    cos = np.clip(np.dot(t1, t2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scene  = args.scene
    out_dir = Path(args.ghost_output_root) / scene

    # --- Load VGGT outputs ---------------------------------------------------
    print(f"\n=== Scene: {scene} ===")
    cam_npz   = np.load(out_dir / "vggt_cameras.npz", mmap_mode="r")
    depth_npz = np.load(out_dir / "vggt_depth.npz",   mmap_mode="r")

    cam_names  = [n.decode() if isinstance(n, bytes) else n
                  for n in cam_npz["camera_names"]]  # (K,)
    vggt_exts  = cam_npz["extrinsics"].astype(np.float64)   # (T, K, 3, 4)
    vggt_intrs = cam_npz["intrinsics"].astype(np.float64)   # (T, K, 3, 3)
    vggt_valid = cam_npz["valid"]                            # (T, K) bool
    depth_mm   = depth_npz["depth"]                         # (T, K, H, W) uint16
    depth_conf = depth_npz["depth_conf"]                    # (T, K, H, W) float16

    T, K, H_vggt, W_vggt = depth_mm.shape
    frame_start = args.frame_idx if args.frame_idx >= 0 else T // 2
    frame_end   = min(frame_start + args.n_frames, T)
    B           = frame_end - frame_start
    frames      = list(range(frame_start, frame_end))

    # MapAnything requires H and W divisible by patch_size=14; round down
    PATCH = 14
    H_ma = (H_vggt // PATCH) * PATCH
    W_ma = (W_vggt // PATCH) * PATCH
    print(f"  T={T}  K={K}  VGGT={H_vggt}×{W_vggt}  MA={H_ma}×{W_ma}  "
          f"frames={frame_start}..{frame_end-1}  B={B}")

    # Use cameras valid across ALL batched frames
    valid_cams = [k for k in range(K) if vggt_valid[frames, k].all()]
    print(f"  Cameras valid across all {B} frames: {[cam_names[k] for k in valid_cams]}")

    # --- GT cameras ----------------------------------------------------------
    print("\nLoading GT calibration...")
    gt_cams = load_gt_cameras(
        Path(args.rich_gt_root), scene, [cam_names[k] for k in valid_cams]
    )
    valid_cams_gt = [k for k in valid_cams if cam_names[k] in gt_cams]
    if len(valid_cams_gt) < 2:
        print("[ERROR] Need at least 2 cameras with GT calibration.")
        sys.exit(1)

    # --- GT scale estimation (use first frame) --------------------------------
    gt_scales = []
    t_vggt_0 = vggt_exts[frame_start, valid_cams_gt[0], :3, 3]
    gt_ext0  = gt_cams[cam_names[valid_cams_gt[0]]]["extrinsics"]
    t_gt_0   = gt_ext0[:3, 3] if gt_ext0 is not None else np.zeros(3)

    for k in valid_cams_gt[1:]:
        t_vggt_i = vggt_exts[frame_start, k, :3, 3]
        gt_ext_i = gt_cams[cam_names[k]]["extrinsics"]
        if gt_ext_i is None:
            continue
        t_gt_i = gt_ext_i[:3, 3]
        d_vggt  = np.linalg.norm(t_vggt_i - t_vggt_0)
        d_gt    = np.linalg.norm(t_gt_i   - t_gt_0)
        if d_vggt > 1e-4:
            gt_scales.append(d_gt / d_vggt)

    gt_scale = float(np.median(gt_scales)) if gt_scales else 1.0
    print(f"\n  GT scale (from camera baselines): {gt_scale:.4f}  "
          f"(n={len(gt_scales)}  std={np.std(gt_scales):.4f})")

    # --- Load MapAnything model ----------------------------------------------
    print("\nLoading MapAnything model...")
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
    print("  Model loaded.")

    # --- Build views for MapAnything — stack B frames per camera --------------
    print(f"\nBuilding input views (B={B} frames per camera)...")
    views = []
    used_cams = valid_cams_gt

    for idx, k in enumerate(used_cams):
        cam = cam_names[k]
        m = re.search(r'\d+', cam)
        cam_num = int(m.group()) if m else 0

        imgs, intrs, depths, poses = [], [], [], []
        skip = False
        for fi in frames:
            jpeg_path = Path(args.rich_root) / scene / cam / f"{fi:05d}_{cam_num:02d}.jpeg"
            if not jpeg_path.exists():
                print(f"  [SKIP] {cam} frame {fi}: image not found"); skip = True; break
            imgs.append(load_and_normalize_image(jpeg_path, H_ma, W_ma))  # (1,3,H,W)

            intr_np = vggt_intrs[fi, k].copy()
            intr_np[0, 0] *= W_ma / W_vggt; intr_np[0, 2] *= W_ma / W_vggt
            intr_np[1, 1] *= H_ma / H_vggt; intr_np[1, 2] *= H_ma / H_vggt
            intrs.append(torch.from_numpy(intr_np).float())  # (3,3)

            d_m = depth_mm[fi, k].astype(np.float32) / 1000.0
            d_t = torch.nn.functional.interpolate(
                torch.from_numpy(d_m).unsqueeze(0).unsqueeze(0),
                size=(H_ma, W_ma), mode="nearest").squeeze()  # (H,W)
            depths.append(d_t)

            c2w = world2cam_to_cam2world(vggt_exts[fi, k])
            poses.append(torch.from_numpy(c2w).float())  # (4,4)

        if skip:
            continue

        # Stack: (B, 3, H, W), (B, 3, 3), (B, H, W), (B, 4, 4)
        img_b   = torch.cat(imgs, dim=0).to(device)
        intr_b  = torch.stack(intrs, dim=0).to(device)
        depth_b = torch.stack(depths, dim=0).to(device)
        pose_b  = torch.stack(poses, dim=0).to(device)
        depth_b_np = depth_b.cpu().numpy()  # (B, H, W) for comparison

        view = {
            "img":            img_b,
            "data_norm_type": ["dinov2"] * B,
            "_cam_name":      cam,
            "_k":             k,
            "_depth_ma":      depth_b_np,
        }
        if not args.no_geo:
            view["intrinsics"]       = intr_b
            view["depth_z"]          = depth_b
            view["camera_poses"]     = pose_b
            view["is_metric_scale"]  = torch.zeros(B, dtype=torch.bool, device=device)
        views.append(view)
        d_flat = depth_b_np[depth_b_np > 0]
        print(f"  {cam}: loaded {B} frames, depth range [{d_flat.min():.2f}, {d_flat.max():.2f}] m")

    if len(views) < 2:
        print("[ERROR] Need at least 2 views.")
        sys.exit(1)

    # Strip private keys before passing to MapAnything
    cam_name_order = [v.pop("_cam_name") for v in views]
    k_order        = [v.pop("_k")        for v in views]
    depth_ma_list  = [v.pop("_depth_ma") for v in views]  # (H_ma, W_ma) per cam

    # Normalize poses: per batch element, cam_0 → [I|0].
    # For static RICH cameras with VGGT cam_00≈[I|0] this is a no-op,
    # but we do it correctly for generality.
    if not args.no_geo:
        ref_poses = views[0]["camera_poses"].cpu().double().numpy()  # (B, 4, 4)
        for v in views:
            poses_i = v["camera_poses"].cpu().double().numpy()  # (B, 4, 4)
            normalized = np.stack([
                np.linalg.inv(ref_poses[b]) @ poses_i[b] for b in range(B)
            ])  # (B, 4, 4)
            v["camera_poses"] = torch.from_numpy(normalized).float().to(device)

    # --- Run inference -------------------------------------------------------
    print(f"\nRunning MapAnything on {len(views)} views...")
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=False,        # keep more valid pixels for comparison
            ignore_depth_scale_inputs=True,
            ignore_pose_scale_inputs=True,
        )
    print("  Inference done.")

    # --- Compare results — aggregate per frame across cameras ----------------
    # all_frame_scales[b] = list of per-camera depth_scale at frame b
    all_frame_scales: list[list[float]] = [[] for _ in range(B)]

    print("\n" + "=" * 72)
    print(f"{'Camera':<12}  {'mean_scale':>10}  {'std':>6}  "
          f"{'err%':>6}  {'rot_err°':>8}  {'trans_err°':>10}")
    print("-" * 72)

    for i, (pred, cam, k, vggt_depth_b) in enumerate(
            zip(preds, cam_name_order, k_order, depth_ma_list)):

        cam_scales = []
        for b in range(B):
            pred_depth = pred["depth_z"][b].squeeze(-1).cpu().numpy()  # (H, W)
            vggt_d     = vggt_depth_b[b]                               # (H, W)
            mask = (pred_depth > 0) & (vggt_d > 0)
            if mask.sum() > 100:
                s = float(np.median(pred_depth[mask] / vggt_d[mask]))
                cam_scales.append(s)
                all_frame_scales[b].append(s)

        mean_s = float(np.mean(cam_scales)) if cam_scales else float('nan')

        # Camera geometry error (use first batch element vs VGGT)
        if i == 0:
            rot_err_deg, trans_err_deg = 0.0, 0.0
        else:
            ref_pred   = preds[0]["camera_poses"][0].cpu().numpy()
            pred_pose  = pred["camera_poses"][0].cpu().numpy()
            ref_c2w    = world2cam_to_cam2world(vggt_exts[frame_start, k_order[0]])
            cam_c2w    = world2cam_to_cam2world(vggt_exts[frame_start, k])
            R_vggt_rel = ref_c2w[:3,:3].T @ cam_c2w[:3,:3]
            R_pred_rel = ref_pred[:3,:3].T @ pred_pose[:3,:3]
            rot_err_deg   = rotation_error_deg(R_vggt_rel, R_pred_rel)
            trans_err_deg = translation_error_deg(
                cam_c2w[:3,3] - ref_c2w[:3,3],
                pred_pose[:3,3] - ref_pred[:3,3])

        print(f"{cam:<12}  {mean_s:>10.4f}  "
              f"{np.std(cam_scales):>6.4f}  "
              f"{(mean_s-gt_scale)/gt_scale*100:>+6.1f}%  "
              f"{rot_err_deg:>8.2f}  {trans_err_deg:>10.2f}")

    # Per-frame summary
    print("=" * 72)
    per_frame_scales = [float(np.median(s)) for s in all_frame_scales if s]
    print(f"\n  Per-frame scale (median over cameras):")
    for b, s in enumerate(per_frame_scales):
        print(f"    frame {frames[b]:5d}:  {s:.4f}  err={( s-gt_scale)/gt_scale*100:+.1f}%")
    print(f"\n  Overall median: {np.median(per_frame_scales):.4f}  "
          f"std: {np.std(per_frame_scales):.4f}  "
          f"err: {(np.median(per_frame_scales)-gt_scale)/gt_scale*100:+.1f}%")
    print(f"  GT scale:       {gt_scale:.4f}")
    print()


if __name__ == "__main__":
    main()
