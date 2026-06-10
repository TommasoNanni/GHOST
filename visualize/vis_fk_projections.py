"""Visualize FK-projected SMPL-X joints vs pred_keypoints_2d (MHR70) for one frame.

For each camera that has body data at the requested frame, draws the actual
RICH frame as background and overlays:
  - Green dots  : FK-projected SMPL-X joints 0-21 using FUSED body_pose
                  (from --fusion_npz) or raw smplx_body_pose if not provided
  - Blue crosses: pred_keypoints_2d MHR70 joints (M1 pipeline)

The script mounts the RICH train squashfs automatically if --sqsh is given
and rich_root is not accessible.

Usage:
    # With fused pose (recommended):
    pixi run python visualize/vis_fk_projections.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --fusion_npz fusion_outputs/BBQ_001_guitar.npz \\
        --pid 1 --frame 0

    # Without fused pose (uses raw smplx_body_pose from body_data):
    pixi run python visualize/vis_fk_projections.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --pid 1 --frame 0
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.transform import Rotation as SciR

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer, _SMPLX_TO_MHR70, _6d_to_aa_batch
from utilities.rich_gender_plugin import resolve_smplx_models

_FK_JOINTS = list(range(22))

_SMPLX_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

_FRAME_EXTS = (".jpg", ".jpeg", ".bmp", ".png")

# RICH cameras are calibrated for this native resolution.
_RICH_ORIG_W = 4112
_RICH_ORIG_H = 3008


def _orig_to_vggt(kp: np.ndarray, oc: np.ndarray, W: float, H: float):
    x1, y1, x2, y2 = oc
    u = x1 + float(kp[0]) * (x2 - x1) / W
    v = y1 + float(kp[1]) * (y2 - y1) / H
    return u, v


def _ensure_mount(sqsh: str | None, rich_root: str) -> None:
    """Mount the squashfs if rich_root is not accessible."""
    import os
    rp = Path(rich_root)
    # Already a mount point (FUSE/squashfs or otherwise) → trust it.
    if rp.is_dir() and os.path.ismount(str(rp)):
        return
    try:
        next(rp.iterdir())
        return  # accessible and non-empty
    except StopIteration:
        return  # accessible but empty — don't remount
    except OSError:
        pass    # not accessible → try mounting below

    if sqsh is None:
        print(f"[warn] {rich_root} not accessible and --sqsh not given; frames will be blank")
        return

    print(f"Mounting {sqsh} → {rich_root} ...")
    rp.mkdir(parents=True, exist_ok=True)
    ret = subprocess.run(["squashfuse", sqsh, rich_root])
    if ret.returncode != 0:
        print("[warn] squashfuse failed; frames will be blank")


def _load_frame(rich_root: str, scene_name: str, cam_name: str, frame_idx: int) -> np.ndarray | None:
    """Load a RICH frame. Returns HxWx3 uint8 or None.

    RICH naming: {frame_idx:05d}_{cam_suffix}{ext}
    where cam_suffix is the numeric part of cam_name (e.g. cam_03 → 03).
    """
    cam_dir = Path(rich_root) / scene_name / cam_name
    # extract numeric suffix: "cam_03" → "03"
    cam_suffix = cam_name.split("_")[-1]
    for ext in _FRAME_EXTS:
        p = cam_dir / f"{frame_idx:05d}_{cam_suffix}{ext}"
        if p.exists():
            from PIL import Image
            return np.array(Image.open(p).convert("RGB"))
    return None


def _load_gt_betas_from_pkl(
    rich_calib: Path,
    scene_name: str,
    frame_idx: int,
    gt_transl_cam0: np.ndarray | None = None,
    body_split: str = "train_body",
) -> np.ndarray | None:
    """Load GT SMPL-X betas directly from RICH train_body pkl files.

    Picks the pkl whose transl best matches gt_transl_cam0 (world frame for BBQ
    where cam-0 = world; for multi-person scenes this disambiguates the person).
    Returns (10,) float32 or None if not found.
    """
    frame_dir = rich_calib / body_split / scene_name / f"{frame_idx:05d}"
    if not frame_dir.is_dir():
        return None
    pkls = sorted(frame_dir.glob("*.pkl"))
    if not pkls:
        return None
    if len(pkls) == 1 or gt_transl_cam0 is None:
        with open(pkls[0], "rb") as f:
            d = pickle.load(f, encoding="latin1")
        b = np.asarray(d["betas"], dtype=np.float32).reshape(-1)[:10]
        return b if not np.all(b == 0) else None
    # Multi-person: pick by transl proximity (pkl transl is in RICH world = cam-0 for BBQ)
    best_b, best_d = None, float("inf")
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        tr = np.asarray(d["transl"], dtype=np.float32).reshape(3)
        dist = float(np.linalg.norm(tr - gt_transl_cam0))
        if dist < best_d:
            best_d = dist
            best_b = np.asarray(d["betas"], dtype=np.float32).reshape(-1)[:10]
    return best_b if best_b is not None and not np.all(best_b == 0) else None


def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def _load_rich_extrinsics(scene_name: str, calib_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,4) world-to-cam [R|t] matrices from RICH XML calibration.

    Sorted alphabetically by XML filename (cam_00 → index 0, cam_01 → index 1, ...).
    Returns None if calibration directory is not found.
    """
    location = _scene_to_location(scene_name)
    calib_dir = calib_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    exts = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cam_node = root.find("CameraMatrix")
        if cam_node is None:
            continue
        vals = list(map(float, cam_node.find("data").text.split()))
        exts.append(np.array(vals, dtype=np.float64).reshape(3, 4))
    return exts if exts else None


def _load_rich_intrinsics(scene_name: str, calib_root: Path) -> list[np.ndarray] | None:
    """Return list of (3,3) K matrices from RICH XML calibration (sorted by XML filename)."""
    location = _scene_to_location(scene_name)
    calib_dir = calib_root / "scan_calibration" / location / "calibration"
    if not calib_dir.is_dir():
        return None
    Ks = []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        intr = root.find("Intrinsics")
        if intr is None:
            continue
        vals = list(map(float, intr.find("data").text.split()))
        Ks.append(np.array(vals, dtype=np.float64).reshape(3, 3))
    return Ks if Ks else None


def _load_gt_smplx_all(
    rich_calib: Path, body_split: str, scene_name: str, frame_idx: int
) -> list[dict]:
    """Load GT SMPL-X params for ALL persons at frame_idx from RICH pkl files.

    Returns a list of dicts with keys transl, global_orient, body_pose, betas
    (all float64 numpy arrays in RICH world frame).
    """
    frame_dir = rich_calib / body_split / scene_name / f"{frame_idx:05d}"
    if not frame_dir.is_dir():
        return []
    results = []
    for pkl_path in sorted(frame_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        raw_betas = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
        results.append({
            "transl":        np.array(d["transl"],        dtype=np.float64).reshape(3),
            "global_orient": np.array(d["global_orient"], dtype=np.float64).reshape(3),
            "body_pose":     np.array(d["body_pose"],     dtype=np.float64).reshape(63),
            "betas":         np.array(raw_betas,          dtype=np.float64).reshape(-1)[:10],
        })
    return results


def _project_gt_rich_native(
    J_can_gt: np.ndarray,   # (55, 3) canonical joints (zero orient, zero transl)
    R_gt:     np.ndarray,   # (3, 3) GT global orient
    transl_gt: np.ndarray,  # (3,) GT SMPL-X transl in RICH world frame
    Kk:       np.ndarray,   # (3, 3) RICH intrinsic matrix for cam k
    extk:     np.ndarray,   # (3, 4) RICH world-to-cam-k [R|t]
) -> list[tuple[float, float] | None]:
    """Project GT SMPL-X joints to cam k using RICH's native cameras (no relay through cam-0).

    GT params and RICH cameras share the same world frame, so the transform is
    direct:  x_camk = Rk @ x_world + tk.
    """
    Rk = extk[:, :3].astype(np.float64)
    tk = extk[:, 3].astype(np.float64)
    pelvis = J_can_gt[0].astype(np.float64)
    pts = []
    for j in _FK_JOINTS:
        j_world = R_gt @ (J_can_gt[j].astype(np.float64) - pelvis) + pelvis + transl_gt
        j_camk  = Rk @ j_world + tk
        if j_camk[2] <= 0.0:
            pts.append(None)
            continue
        u = Kk[0, 0] * j_camk[0] / j_camk[2] + Kk[0, 2]
        v = Kk[1, 1] * j_camk[1] / j_camk[2] + Kk[1, 2]
        pts.append((float(u), float(v)))
    return pts


def _project_body_cam0(
    J_can:            np.ndarray,   # (55, 3) FK joints, zero global_orient
    R_global_cam0:    np.ndarray,   # (3, 3) global orient in cam-00 frame
    smplx_transl_cam0: np.ndarray,  # (3,) raw SMPL-X transl in cam-00 frame
    ext0:             np.ndarray,   # (3, 4) XML world-to-cam-00
    extk:             np.ndarray,   # (3, 4) XML world-to-cam-k
    K:                np.ndarray,   # (3, 3) intrinsic matrix (already in target pixel space)
) -> list[tuple[float, float] | None]:
    """Project SMPL-X FK joints to camera k, given GT params in cam-00 frame.

    GT from the fusion dataset is expressed in cam-00 frame (after
    _transform_to_world_frame), NOT in true RICH world frame.  To get to cam-k:
        x_camk = R_rel @ x_cam0 + t_rel
        R_rel  = Rk @ R0.T
        t_rel  = tk - Rk @ R0.T @ t0
    """
    R0, t0 = ext0[:, :3].astype(np.float64), ext0[:, 3].astype(np.float64)
    Rk, tk = extk[:, :3].astype(np.float64), extk[:, 3].astype(np.float64)
    R_rel = Rk @ R0.T
    t_rel = tk - R_rel @ t0

    pelvis0 = J_can[0].astype(np.float64) + smplx_transl_cam0.astype(np.float64)

    pts = []
    for j in _FK_JOINTS:
        j_cam0 = R_global_cam0 @ (J_can[j].astype(np.float64) - J_can[0].astype(np.float64)) + pelvis0
        j_camk = R_rel @ j_cam0 + t_rel
        if j_camk[2] <= 0.0:
            pts.append(None)
            continue
        u = K[0, 0] * j_camk[0] / j_camk[2] + K[0, 2]
        v = K[1, 1] * j_camk[1] / j_camk[2] + K[1, 2]
        pts.append((u, v))
    return pts


def _draw_skeleton(ax, pts, color, lw=1.0, ms=20, alpha=0.8, zorder=4, label_joints=False):
    for pa, ch in _SMPLX_EDGES:
        if pa < len(pts) and ch < len(pts) and pts[pa] and pts[ch]:
            ax.plot([pts[pa][0], pts[ch][0]], [pts[pa][1], pts[ch][1]],
                    color=color, lw=lw, alpha=alpha, zorder=zorder)
    for j, pt in enumerate(pts):
        if pt is None:
            continue
        ax.scatter(*pt, c=color, s=ms, zorder=zorder + 1, marker="o")
        if label_joints:
            ax.text(pt[0] + 3, pt[1], str(j), fontsize=5, color=color, zorder=zorder + 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",  required=True)
    ap.add_argument("--smplx_model", default="body_models/SMPLX_NEUTRAL.pkl")
    ap.add_argument("--pid",   type=int, default=1)
    ap.add_argument("--frame", type=int, default=0, help="Global frame index")
    ap.add_argument("--rich_root", default="/tmp/rich_train")
    ap.add_argument("--sqsh",
        default="/capstor/scratch/cscs/tnanni/datasets/rich/train_dataset.sqsh")
    ap.add_argument("--fusion_npz", default=None,
        help="Path to fusion_outputs/<scene>.npz from infer_scene.py. "
             "If given, uses fused body_pose (joints 1-21 in 6D) instead of raw smplx_body_pose.")
    ap.add_argument("--frame_start", type=int, default=0,
        help="Frame offset: fusion_npz index = frame - frame_start.")
    ap.add_argument("--rich_calib",
        default="/capstor/scratch/cscs/tnanni/datasets/rich",
        help="RICH dataset root containing scan_calibration/ (for GT extrinsics).")
    ap.add_argument("--body_split", default="train_body",
        help="RICH body split directory (train_body or test_body).")
    ap.add_argument("--out", default=None,
        help="Output PNG path. Default: renders/comparison/<scene>_pid<pid>_f<frame>.png")
    args = ap.parse_args()

    scene_dir  = Path(args.scene_dir)
    scene_name = scene_dir.name
    pid        = args.pid
    global_t   = args.frame

    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "renders" / "comparison" /
        f"{scene_name}_pid{pid}_f{global_t:05d}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_mount(args.sqsh, args.rich_root)

    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(scene_dir, _smplx_arg)

    # ── fused body_pose + GT data (optional, from fusion_npz) ─────────────
    # fusion_npz["pose"]              (T, P, 55, 6) — predicted, joints 0=root, 1-21=body
    # fusion_npz["gt_body_pose"]      (T, P, 55, 6) — GT, same layout
    # fusion_npz["gt_body_transl_world"] (T, P, 3)  — GT SMPL-X transl in world frame
    # fusion_npz["gt_valid"]          (T, P)        — 1 where GT exists
    fused_body_pose_aa:    np.ndarray | None = None   # (63,) AA
    gt_body_pose_aa:       np.ndarray | None = None   # (63,) AA
    gt_global_orient_R:    np.ndarray | None = None   # (3,3) in cam-00 frame
    gt_smplx_transl_cam0:  np.ndarray | None = None   # (3,) raw SMPL-X transl in cam-00 frame
    gt_betas:              np.ndarray | None = None   # (10,)

    if args.fusion_npz:
        fnpz = np.load(args.fusion_npz, allow_pickle=False)
        fused_pose_55 = fnpz["pose"]   # (T, P, 55, 6)
        # Determine all pids (same sorted order as dataset)
        all_pids_sorted = sorted(set(
            int(p.stem.split("_")[1])
            for cam_dir in placer._cam_dirs
            for p in (cam_dir / "body_data").glob("person_*.npz")
        ))
        if pid not in all_pids_sorted:
            print(f"[warn] pid={pid} not in body_data; ignoring --fusion_npz")
        else:
            p_idx  = all_pids_sorted.index(pid)
            t_fuse = global_t - args.frame_start
            if 0 <= t_fuse < fused_pose_55.shape[0] and p_idx < fused_pose_55.shape[1]:
                body_6d = fused_pose_55[t_fuse, p_idx, 1:22, :]  # (21, 6)
                fused_body_pose_aa = _6d_to_aa_batch(body_6d).reshape(63).astype(np.float32)
                print(f"Loaded fused body_pose  pid={pid} frame={global_t}")

                # GT body pose + camera params
                has_gt = ("gt_body_pose" in fnpz.files and
                          "gt_body_transl_world" in fnpz.files and
                          "gt_valid" in fnpz.files)
                if has_gt:
                    gt_valid = fnpz["gt_valid"][t_fuse, p_idx]
                    if gt_valid:
                        gt_pose_55 = fnpz["gt_body_pose"]  # (T, P, 55, 6)
                        # GT global orient (joint 0, world frame, 6D → R)
                        gt_go_aa = _6d_to_aa_batch(
                            gt_pose_55[t_fuse, p_idx, 0:1, :]
                        ).squeeze(0)  # (3,)
                        gt_global_orient_R = SciR.from_rotvec(gt_go_aa).as_matrix()
                        # GT body pose (joints 1-21, world frame, 6D → AA)
                        gt_bp_6d = gt_pose_55[t_fuse, p_idx, 1:22, :]  # (21, 6)
                        gt_body_pose_aa = _6d_to_aa_batch(gt_bp_6d).reshape(63).astype(np.float32)
                        gt_smplx_transl_cam0 = fnpz["gt_body_transl_world"][t_fuse, p_idx]
                        # GT betas: load directly from RICH pkl (training uses mean
                        # SAM3D betas, so fnpz["gt_body_shape"] is always zeros).
                        gt_betas = _load_gt_betas_from_pkl(
                            Path(args.rich_calib), scene_name, global_t,
                            gt_transl_cam0=gt_smplx_transl_cam0,
                            body_split=args.body_split,
                        )
                        if gt_betas is not None:
                            print(f"Loaded GT betas from pkl: {gt_betas[:4]}")
                        print(f"Loaded GT body_pose    pid={pid} frame={global_t}")
                    else:
                        print(f"[info] GT not valid for pid={pid} frame={global_t} — skipping orange/red")
            else:
                print(f"[warn] t_fuse={t_fuse} or p_idx={p_idx} out of range")

    # ── GT camera extrinsics + intrinsics from RICH XML calibration ───────────
    gt_extrinsics: list[np.ndarray] | None = _load_rich_extrinsics(
        scene_name, Path(args.rich_calib)
    )
    gt_intrinsics: list[np.ndarray] | None = _load_rich_intrinsics(
        scene_name, Path(args.rich_calib)
    )
    if gt_extrinsics is None:
        print(f"[warn] RICH calibration not found under {args.rich_calib}/scan_calibration/"
              f"{_scene_to_location(scene_name)}/ — orange/red/pink overlays disabled")

    # ── GT SMPL-X params from RICH pkl (for pink: all persons at this frame) ─
    gt_smplx_all: list[dict] = _load_gt_smplx_all(
        Path(args.rich_calib), args.body_split, scene_name, global_t
    )
    if gt_smplx_all:
        print(f"Loaded {len(gt_smplx_all)} GT person(s) from pkl for frame {global_t}")
    else:
        print(f"[warn] No GT pkl found for frame {global_t} — pink overlay disabled")

    # ── load body data per camera ───────────────────────────────────────────
    cam_data = []  # list of (cam_name, dict | None)
    for cam_dir in placer._cam_dirs:
        bf = cam_dir / "body_data" / f"person_{pid}.npz"
        if not bf.exists():
            cam_data.append((cam_dir.name, None))
            continue
        d = np.load(bf, allow_pickle=False)
        required = {"smplx_transl", "smplx_global_orient", "smplx_body_pose",
                    "frame_indices", "focal_length", "pred_keypoints_2d"}
        if not required.issubset(d.files):
            cam_data.append((cam_dir.name, None))
            continue
        fi = d["frame_indices"].astype(int)
        local_t_map = {int(g): int(l) for l, g in enumerate(fi)}
        if global_t not in local_t_map:
            cam_data.append((cam_dir.name, None))
            continue
        lt = local_t_map[global_t]
        cam_data.append((cam_dir.name, {
            "transl":    d["smplx_transl"][lt].astype(np.float64),
            "orient":    d["smplx_global_orient"][lt].astype(np.float64),
            "body_pose": d["smplx_body_pose"][lt],
            "fl":        float(d["focal_length"][lt]),
            "kps2d":     d["pred_keypoints_2d"][lt].astype(np.float64),  # (70, 3) [u,v,conf]
            "betas":     d["smplx_betas"][lt].astype(np.float32),
        }))

    vggt_t = global_t  # frame_start=0

    if not any(d is not None for _, d in cam_data):
        print(f"No data for pid={pid} frame={global_t}")
        sys.exit(1)

    # Mean betas across all cameras (used by green modality).
    all_betas = [d["betas"] for _, d in cam_data if d is not None]
    mean_betas = np.stack(all_betas, axis=0).mean(axis=0).astype(np.float32)

    if fused_body_pose_aa is None:
        print("[warn] --fusion_npz not provided; green will use raw per-camera SAM3D body_pose")

    # ── layout ─────────────────────────────────────────────────────────────
    active = [(cname, data) for cname, data in cam_data if data is not None]
    if not active:
        print("No cameras have data for this frame/pid.")
        sys.exit(1)

    ncols = min(4, len(active))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, (cam_name, data) in enumerate(active):
        ax = axes_flat[ax_idx]

        # camera index in the VGGT K dimension
        k = next((i for i, (n, _) in enumerate(cam_data) if n == cam_name), None)

        if k is None or vggt_t >= placer.T or not placer.cam_valid[vggt_t, k]:
            ax.set_title(f"{cam_name}\n(invalid cam)")
            ax.axis("off")
            continue

        oc     = placer.original_coords[vggt_t, k]
        os_    = placer.original_size[vggt_t, k]
        W_orig = float(os_[0])
        H_orig = float(os_[1])
        W_vggt = float(oc[2])
        H_vggt = float(oc[3])

        # ── background frame ───────────────────────────────────────────────
        frame_img = _load_frame(args.rich_root, scene_name, cam_name, global_t)
        if frame_img is not None:
            H_img, W_img = frame_img.shape[:2]
            # Display image in its native pixel space.
            # W_orig/H_orig (from npz) may differ from W_img/H_img if the frames
            # on disk are at a different resolution than what VGGT was run on.
            # We scale FK and kps2d coords by (W_img/W_orig, H_img/H_orig).
            sx = W_img / W_orig
            sy = H_img / H_orig
            ax.imshow(frame_img, origin="upper", alpha=0.85)
            ax.set_xlim(0, W_img)
            ax.set_ylim(H_img, 0)
        else:
            # Fallback: blank canvas in W_orig space so all overlays align.
            sx, sy = 1.0, 1.0
            ax.set_xlim(-5, W_orig + 5)
            ax.set_ylim(H_orig + 5, -5)
            ax.add_patch(mpatches.Rectangle(
                (0, 0), W_orig, H_orig,
                linewidth=1, edgecolor="gray", facecolor="#f0f0f0"))

        def to_plot(u_o, v_o):
            """Scale from W_orig space to display (PIL image) space."""
            return float(u_o) * sx, float(v_o) * sy

        fl = data["fl"]

        # VGGT intrinsics (VGGT pixel space). Scale by vggt_sx/sy before to_plot.
        K_vggt = placer.intrinsics[vggt_t, k].astype(np.float64)
        vggt_sx = W_orig / W_vggt   # VGGT pixel → W_orig
        vggt_sy = H_orig / H_vggt

        # RICH XML intrinsics scaled to W_orig space (used by orange, red, pink).
        # Results from projection with K_rich_scaled go directly to to_plot.
        K_rich_scaled = None
        if gt_intrinsics is not None and k < len(gt_intrinsics):
            Krs = gt_intrinsics[k].astype(np.float64).copy()
            Krs[0, 0] *= W_orig / _RICH_ORIG_W   # fx
            Krs[0, 2] *= W_orig / _RICH_ORIG_W   # cx
            Krs[1, 1] *= H_orig / _RICH_ORIG_H   # fy
            Krs[1, 2] *= H_orig / _RICH_ORIG_H   # cy
            K_rich_scaled = Krs

        # ── green: fused (or raw) pose + VGGT cam ────────────────────────
        body_pose_for_fk = fused_body_pose_aa if fused_body_pose_aa is not None else data["body_pose"]
        J_can_green = placer._smplx_fk(
            mean_betas[np.newaxis],
            body_pose_for_fk[np.newaxis],
            np.zeros((1, 3), dtype=np.float32),
        )[0]  # (55, 3)
        R_o    = SciR.from_rotvec(data["orient"]).as_matrix()
        pelvis = J_can_green[0].astype(np.float64)
        green_pts = []
        for j in _FK_JOINTS:
            j_cam = R_o @ (J_can_green[j].astype(np.float64) - pelvis) + pelvis + data["transl"]
            if j_cam[2] <= 0.0:
                green_pts.append(None)
                continue
            u = K_vggt[0, 0] * j_cam[0] / j_cam[2] + K_vggt[0, 2]
            v = K_vggt[1, 1] * j_cam[1] / j_cam[2] + K_vggt[1, 2]
            # Scale from VGGT pixel space (W_vggt) → W_orig space for to_plot.
            green_pts.append(to_plot(u * vggt_sx, v * vggt_sy))
        _draw_skeleton(ax, green_pts, "green", lw=1.0, ms=20, zorder=3, label_joints=True)

        # ── blue: pred_keypoints_2d MHR70 ────────────────────────────────
        kps = data["kps2d"]
        for mhr_idx in sorted(_SMPLX_TO_MHR70.values()):
            if mhr_idx < len(kps):
                ax.scatter(*to_plot(kps[mhr_idx, 0], kps[mhr_idx, 1]),
                           c="blue", s=40, zorder=6, marker="x", linewidths=1.5)

        # ── orange + red: GT camera (RICH XML extrinsics) ─────────────────
        # GT params are in cam-00 frame (fusion dataset _transform_to_world_frame).
        # Use relative transforms ext0→extk, NOT raw world-to-cam extk directly.
        can_gt = (gt_extrinsics is not None and k < len(gt_extrinsics) and
                  gt_global_orient_R is not None and gt_smplx_transl_cam0 is not None
                  and K_rich_scaled is not None)
        if can_gt:
            ext0 = gt_extrinsics[0]   # cam-00 XML extrinsic (world-to-cam-00)
            extk = gt_extrinsics[k]   # cam-k  XML extrinsic (world-to-cam-k)

            # orange: fused/raw pose + GT rotation + GT translation
            orange_pts_raw = _project_body_cam0(
                J_can_green, gt_global_orient_R,
                gt_smplx_transl_cam0, ext0, extk, K_rich_scaled)
            orange_pts = [to_plot(*p) if p else None for p in orange_pts_raw]
            _draw_skeleton(ax, orange_pts, "orange", lw=1.0, ms=20, zorder=4)

            # red: GT pose + GT rotation + GT translation + GT betas
            if gt_body_pose_aa is not None:
                betas_red = gt_betas if gt_betas is not None else data["betas"]
                J_can_red = placer._smplx_fk(
                    betas_red[np.newaxis],
                    gt_body_pose_aa[np.newaxis],
                    np.zeros((1, 3), dtype=np.float32),
                )[0]
                red_pts_raw = _project_body_cam0(
                    J_can_red, gt_global_orient_R,
                    gt_smplx_transl_cam0, ext0, extk, K_rich_scaled)
                red_pts = [to_plot(*p) if p else None for p in red_pts_raw]
                _draw_skeleton(ax, red_pts, "red", lw=1.0, ms=20, zorder=5)

        # ── pink: GT SMPL-X + RICH native cameras (direct world→camk) ────────
        # GT pkl transl/orient are in the RICH world frame; RICH XML cameras share
        # that same frame, so no relay through cam-0 is needed.
        can_pink = (K_rich_scaled is not None and gt_extrinsics is not None
                    and k < len(gt_extrinsics) and len(gt_smplx_all) > 0)
        if can_pink:
            extk_rich = gt_extrinsics[k]
            for gt_params in gt_smplx_all:
                R_gt_pink = SciR.from_rotvec(gt_params["global_orient"]).as_matrix()
                J_can_pink = placer._smplx_fk(
                    gt_params["betas"][np.newaxis].astype(np.float32),
                    gt_params["body_pose"][np.newaxis].astype(np.float32),
                    np.zeros((1, 3), dtype=np.float32),
                )[0]
                # K_rich_scaled is already in W_orig space → to_plot directly.
                pink_pts_raw = _project_gt_rich_native(
                    J_can_pink, R_gt_pink, gt_params["transl"],
                    K_rich_scaled, extk_rich,
                )
                pink_pts = [to_plot(*p) if p else None for p in pink_pts_raw]
                _draw_skeleton(ax, pink_pts, "deeppink", lw=1.5, ms=25, zorder=6)

        # ── white: GT transl+orient (fusion NPZ, cam-00 frame) + RICH XML extrinsics + VGGT K ──
        # Same as orange but swaps RICH K → VGGT K. Isolates intrinsic difference.
        can_white = (can_gt and gt_global_orient_R is not None and gt_smplx_transl_cam0 is not None)
        if can_white:
            white_pts_raw = _project_body_cam0(
                J_can_green, gt_global_orient_R,
                gt_smplx_transl_cam0, ext0, extk, K_vggt)
            white_pts = [to_plot(p[0] * vggt_sx, p[1] * vggt_sy) if p else None for p in white_pts_raw]
            _draw_skeleton(ax, white_pts, "white", lw=1.5, ms=20, zorder=7)

        n_vis = sum(1 for p in green_pts if p is not None)
        ax.set_title(
            f"{cam_name}  |  green FK: {n_vis}/{len(_FK_JOINTS)}\n"
            f"fl={fl:.0f}px  W={W_orig:.0f}  H={H_orig:.0f}",
            fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes_flat[len(active):]:
        ax.axis("off")

    pose_label = "fused pose + VGGT cam" if fused_body_pose_aa is not None \
                 else "raw smplx_body_pose + VGGT cam"
    handles = [
        mpatches.Patch(color="green",    label=f"Green  : {pose_label} | VGGT ext | VGGT K"),
        mpatches.Patch(color="white",    label="White  : GT transl+orient (NPZ) | RICH XML ext | VGGT K"),
        mpatches.Patch(color="orange",   label="Orange : GT transl+orient (NPZ) | RICH XML ext | RICH K"),
        mpatches.Patch(color="red",      label="Red    : GT pose+transl+orient (NPZ) | RICH XML ext | RICH K"),
        mpatches.Patch(color="deeppink", label="Pink   : GT all (pkl, RICH world) | RICH XML ext | RICH K"),
        mpatches.Patch(color="blue",     label="Blue   : pred_keypoints_2d MHR70"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"scene={scene_name}  pid={pid}  frame={global_t}",
        fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
