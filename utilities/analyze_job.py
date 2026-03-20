"""
Analyze a single-point overfitting job.

Usage:
    pixi run python -m scripts.analyze_job.py <job_id>

Reads:
    fusion_outputs/<job_id>_predictions.npz
    logs/*_<job_id>.err

Prints:
    - Prediction statistics (variance, MSE, correlation vs GT, camera collapse)
    - Per-epoch metric curves parsed from the log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_predictions(job_id: str) -> dict[str, np.ndarray]:
    path = ROOT / "fusion_outputs" / f"{job_id}_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"No predictions file: {path}")
    return dict(np.load(path))


def _find_log(job_id: str) -> Path:
    candidates = list((ROOT / "logs").glob(f"*_{job_id}.err"))
    if not candidates:
        raise FileNotFoundError(f"No .err log for job {job_id} in logs/")
    return candidates[0]


def _parse_log(log_path: Path) -> dict[str, list[float]]:
    """Parse INFO lines → {metric_name: [value_epoch0, value_epoch1, ...]}"""
    pattern = re.compile(
        r"INFO\s+\[Epoch\s+\d+\]\s+(.*)"
    )
    kv_re = re.compile(r"[\w/@\-\.]+=([\d\.\-eE+]+)")
    name_re = re.compile(r"([\w/@\-\.]+)=[\d\.\-eE+]+")

    curves: dict[str, list[float]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        payload = m.group(1)
        names  = name_re.findall(payload)
        values = kv_re.findall(payload)
        for name, val in zip(names, values):
            # strip train/ prefix for brevity
            key = name.replace("train/", "")
            curves.setdefault(key, []).append(float(val))
    return curves


def analyze_predictions(d: dict[str, np.ndarray]) -> None:
    pose      = d["pose"]       # (T, P, J, 6)
    gt_pose   = d["gt_pose"]
    camera    = d["camera"]     # (T, K, 8)
    gt_camera = d["gt_camera"]
    shape     = d["shape"]      # (T, P, 10)
    gt_shape  = d["gt_shape"]

    T, P, J, _ = pose.shape
    K = camera.shape[1]

    print("=" * 60)
    print("PREDICTION FILE ANALYSIS")
    print("=" * 60)
    print(f"T={T}  P={P}  J={J}  K={K}\n")

    # ── POSE ─────────────────────────────────────────────────────────────────
    print("── POSE (6D rotation) ──────────────────────────────────")
    pred_std = pose.std(axis=0).mean()
    gt_std   = gt_pose.std(axis=0).mean()
    mse      = ((pose - gt_pose) ** 2).mean()

    # Per-frame variance (how much does the prediction vary over T?)
    pred_var_per_frame = pose.var(axis=0).mean()   # var over T at each position
    gt_var_per_frame   = gt_pose.var(axis=0).mean()

    # Correlation: flatten T dimension, correlate per output element
    p_flat  = pose.reshape(T, -1)
    g_flat  = gt_pose.reshape(T, -1)
    # Mean pearson-r across all output dimensions
    p_c = p_flat - p_flat.mean(0, keepdims=True)
    g_c = g_flat - g_flat.mean(0, keepdims=True)
    denom = (np.linalg.norm(p_c, axis=0) * np.linalg.norm(g_c, axis=0)) + 1e-8
    corr  = (p_c * g_c).sum(0) / denom
    mean_corr = float(corr.mean())

    # Collapse check: std across time — if near-zero, pred is constant
    pred_temporal_std = pose.std(axis=0)   # (P, J, 6)
    frac_frozen = float((pred_temporal_std < 1e-3).mean())

    print(f"  pred std over T :  {pred_std:.6f}   (GT: {gt_std:.6f})")
    print(f"  pred var over T :  {pred_var_per_frame:.6f}   (GT: {gt_var_per_frame:.6f})")
    print(f"  MSE(pred, GT)   :  {mse:.6f}")
    print(f"  Mean Pearson r  :  {mean_corr:.4f}")
    print(f"  Frac frozen dims:  {frac_frozen:.2%}  (std<1e-3 over T)")

    # per-joint MSE
    joint_mse = ((pose - gt_pose) ** 2).mean(axis=(0, 2, 3))  # (P,)
    for p in range(P):
        print(f"  Person {p} joint MSE: {joint_mse[p]:.6f}")

    # ── SHAPE ────────────────────────────────────────────────────────────────
    print("\n── SHAPE (betas) ───────────────────────────────────────")
    shape_mse  = ((shape - gt_shape) ** 2).mean()
    shape_std  = shape.std(axis=0).mean()
    gt_shape_std = gt_shape.std(axis=0).mean()
    shape_frozen = float((shape.std(axis=0) < 1e-4).mean())
    print(f"  pred std over T :  {shape_std:.6f}   (GT: {gt_shape_std:.6f})")
    print(f"  MSE(pred, GT)   :  {shape_mse:.6f}")
    print(f"  Frac frozen dims:  {shape_frozen:.2%}")

    # ── CAMERA ───────────────────────────────────────────────────────────────
    print("\n── CAMERA ───────────────────────────────────────────────")

    try:
        from scipy.spatial.transform import Rotation as R

        def quat_to_mat(q):  # (N, 4) wxyz
            xyzw = np.concatenate([q[:, 1:], q[:, :1]], axis=-1)
            return R.from_quat(xyzw).as_matrix()  # (N, 3, 3)

        pred_R = quat_to_mat(camera[:, :, :4].reshape(-1, 4)).reshape(T, K, 3, 3)
        pred_t = camera[:, :, 4:7]
        gt_R   = quat_to_mat(gt_camera[:, :, :4].reshape(-1, 4)).reshape(T, K, 3, 3)
        gt_t   = gt_camera[:, :, 4:7]
        pred_f = camera[:, :, 7]
        gt_f   = gt_camera[:, :, 7]

        # Camera centres: C = -R^T t
        pred_C = -np.einsum("...ji,...j->...i", pred_R, pred_t)  # (T, K, 3)
        gt_C   = -np.einsum("...ji,...j->...i", gt_R,   gt_t)

        # ── Translation ─────────────────────────────────────────────────────
        te_per_cam = np.linalg.norm(pred_C - gt_C, axis=-1)  # (T, K)
        print(f"  Translation error (camera centre, metres):")
        for k in range(K):
            print(f"    Camera {k}: mean={te_per_cam[:, k].mean():.4f}m  "
                  f"std={te_per_cam[:, k].std():.4f}m")
        print(f"  Mean over all cams: {te_per_cam.mean():.4f}m")

        # ── Rotation ────────────────────────────────────────────────────────
        # Geodesic angle error: theta = arccos(clip((trace(R_rel)-1)/2, -1, 1))
        R_rel = np.einsum("...ji,...jk->...ik", gt_R, pred_R)  # R_gt^T R_pred
        trace = np.trace(R_rel, axis1=-2, axis2=-1)            # (T, K)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_err_deg = np.degrees(np.arccos(cos_theta))       # (T, K)
        print(f"\n  Rotation geodesic error (degrees):")
        for k in range(K):
            print(f"    Camera {k}: mean={angle_err_deg[:, k].mean():.3f}°  "
                  f"std={angle_err_deg[:, k].std():.3f}°")
        print(f"  Mean over all cams: {angle_err_deg.mean():.3f}°")

        # ── Focal ───────────────────────────────────────────────────────────
        focal_mae = np.abs(pred_f - gt_f).mean()
        log_focal_mse = ((np.log(np.clip(pred_f, 1, None)) -
                          np.log(np.clip(gt_f,   1, None))) ** 2).mean()
        print(f"\n  Focal MAE: {focal_mae:.2f} px  |  log-MSE: {log_focal_mse:.6f}")
        print(f"  Pred focal range: [{pred_f.min():.1f}, {pred_f.max():.1f}]  "
              f"GT: [{gt_f.min():.1f}, {gt_f.max():.1f}]")

    except ImportError:
        print("  (scipy not available — skipping camera decomposition)")


def analyze_log(curves: dict[str, list[float]]) -> None:
    if not curves:
        print("\n(No epoch metrics found in log.)")
        return

    n_epochs = max(len(v) for v in curves.values())
    print(f"\n{'=' * 60}")
    print(f"LOG METRICS  ({n_epochs} epochs)")
    print("=" * 60)

    # Key metrics to highlight
    highlights = [
        "pose", "shape", "camera_mse", "vposer", "total",
        "W-MPJPE", "GA-MPJPE", "PA-MPJPE",
        "W-MPJRE", "PA-MPJRE",
        "TE", "s-TE", "AE", "FocalMAE",
        "RRA@15", "CCA@15", "s-CCA@15",
    ]

    def _summary(vals: list[float]) -> str:
        a = np.array(vals)
        return (f"start={a[0]:.4f}  "
                f"end={a[-1]:.4f}  "
                f"min={a.min():.4f}@ep{int(a.argmin())}  "
                f"max={a.max():.4f}@ep{int(a.argmax())}")

    for key in highlights:
        if key in curves:
            print(f"  {key:<14} {_summary(curves[key])}")

    # Print any extra metrics present in log but not in highlights
    extra = [k for k in curves if k not in highlights and k != "lr"]
    if extra:
        print("\n  Other metrics:")
        for key in extra:
            print(f"  {key:<14} {_summary(curves[key])}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_job.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]

    d = _load_predictions(job_id)
    analyze_predictions(d)

    log_path = _find_log(job_id)
    print(f"\nLog: {log_path.name}")
    curves = _parse_log(log_path)
    analyze_log(curves)


if __name__ == "__main__":
    main()
