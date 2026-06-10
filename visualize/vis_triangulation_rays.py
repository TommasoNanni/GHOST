"""Compare VGGT-predicted vs GT cameras for a scene.

For each of 4 joints (l/r shoulder, l/r hip):
  - Draws rays from cam_a and cam_b using BOTH predicted (VGGT+MA scale) and
    GT (RICH XML) cameras in a common frame (GT aligned to VGGT via Procrustes).
  - Triangulates from each pair and marks result vs actual GT 3D joint.

Also prints per-camera rotation-angle error and translation error.

Usage:
    pixi run python visualize/vis_triangulation_rays.py \\
        --scene_dir /path/to/ghost_outputs/rich_train/ParkingLot2_008_overfence1 \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \\
        --frame 171 --pid 1 --gt_pid 8 \\
        [--cam_a 0] [--cam_b 1] [--ray_len 3.0] [--out vis_rays.png]
"""
from __future__ import annotations
import argparse, re, sys, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models
from test_sapiens_dlt import _load_gt

RICH_ROOT = Path("/capstor/scratch/cscs/tnanni/datasets/rich")

# COCO-17 body keypoint indices for the 4 joints
_JOINTS = {
    "l_shoulder": {"goliath": 5,  "smplx": 16},
    "r_shoulder": {"goliath": 6,  "smplx": 17},
    "l_hip":      {"goliath": 11, "smplx": 1},
    "r_hip":      {"goliath": 12, "smplx": 2},
}


# ── Camera helpers ─────────────────────────────────────────────────────────

def cam_center(ext: np.ndarray) -> np.ndarray:
    R, t = ext[:3, :3], ext[:3, 3]
    return (-R.T @ t).astype(np.float64)


def pixel_ray_world(u: float, v: float, K: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Unit ray from camera centre through pixel (u,v) in world frame."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d = np.array([(u-cx)/fx, (v-cy)/fy, 1.0])
    d /= np.linalg.norm(d)
    return (ext[:3,:3].T @ d) / np.linalg.norm(ext[:3,:3].T @ d)


def dlt_2(obs, pmats):
    rows = []
    for (u, v), P in zip(obs, pmats):
        rows += [u*P[2]-P[0], v*P[2]-P[1]]
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]; return (X[:3]/X[3]).astype(np.float64)


def rot_angle_deg(Ra, Rb):
    cos = np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


# ── GT camera loading ──────────────────────────────────────────────────────

def _scene_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def load_gt_cameras(scene_name: str):
    """Return (Ks, Exts) lists of (3,3) and (3,4) per camera from XML, or (None,None)."""
    loc = _scene_location(scene_name)
    calib_dir = RICH_ROOT / "scan_calibration" / loc / "calibration"
    if not calib_dir.is_dir():
        print(f"[WARN] GT calib not found: {calib_dir}"); return None, None
    Ks, Exts = [], []
    for xml_path in sorted(calib_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        K_node = root.find("Intrinsics")
        E_node = root.find("CameraMatrix")
        if K_node is None or E_node is None: continue
        K = np.array(list(map(float, K_node.find("data").text.split()))).reshape(3,3)
        E = np.array(list(map(float, E_node.find("data").text.split()))).reshape(3,4)
        Ks.append(K); Exts.append(E)
    return (Ks, Exts) if Ks else (None, None)


# ── Procrustes alignment: map GT camera centres to VGGT frame ─────────────

def align_gt_to_vggt(C_pred: np.ndarray, C_gt: np.ndarray):
    """Rigid Procrustes (no scale): find R_T, t_T s.t. R_T @ C_gt + t_T ≈ C_pred.
    Returns R_T (3,3), t_T (3,).
    """
    A, B = C_pred, C_gt          # (K,3) each
    A_m, B_m = A.mean(0), B.mean(0)
    H = (B - B_m).T @ (A - A_m)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_T = (Vt.T @ np.diag([1,1,d]) @ U.T)
    t_T = A_m - R_T @ B_m
    return R_T, t_T


def transform_ext_to_vggt(ext_rich: np.ndarray, R_T: np.ndarray, t_T: np.ndarray) -> np.ndarray:
    """Convert [R|t] from RICH frame to VGGT frame.
    X_vggt = R_T @ X_rich + t_T  =>  R_new = R_rich @ R_T^T,  t_new = t_rich - R_rich @ R_T^T @ t_T
    """
    R, t = ext_rich[:3,:3], ext_rich[:3,3]
    R_new = R @ R_T.T
    t_new = t - R_new @ t_T
    ext_new = np.eye(4)
    ext_new[:3,:3] = R_new; ext_new[:3,3] = t_new
    return ext_new


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_ray(ax, C, endpoint, color, ls="-", lw=1.2, label=None, alpha=0.8):
    """Draw a ray from C to endpoint."""
    ax.plot([C[0],endpoint[0]], [C[1],endpoint[1]], [C[2],endpoint[2]],
            color=color, ls=ls, lw=lw, label=label, alpha=alpha)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",   required=True, type=Path)
    ap.add_argument("--rich_root",   required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--frame",       type=int, default=5)
    ap.add_argument("--pid",         type=int, default=1)
    ap.add_argument("--gt_pid",      type=int, default=1)
    ap.add_argument("--body_split",  default="train_body")
    ap.add_argument("--cam_a",       type=int, default=0)
    ap.add_argument("--cam_b",       type=int, default=1)
    ap.add_argument("--out",         type=Path, default=Path("renders/debug/vis_rays.png"))
    args = ap.parse_args()

    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    global_t   = args.frame

    # ── VGGT predicted cameras ────────────────────────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(str(scene_dir), _smplx_arg)
    scale  = float(np.median(placer.load_mapanything_scale()))
    vggt_t = 0
    K_pred = placer._cam_dirs  # just for count
    n_cams = len(placer._cam_dirs)

    def pred_cam(k):
        K  = placer.intrinsics[vggt_t, k].astype(np.float64)
        E4 = np.eye(4)
        E4[:3] = placer.extrinsics[vggt_t, k].astype(np.float64)
        E4[:3, 3] *= scale
        return K, E4

    # ── GT cameras (XML) ──────────────────────────────────────────────
    gt_Ks, gt_Exts = load_gt_cameras(scene_name)
    has_gt_cam = gt_Ks is not None and len(gt_Ks) >= n_cams

    # ── Align GT cameras to VGGT frame ───────────────────────────────
    R_T = t_T = None
    if has_gt_cam:
        C_pred_all = np.stack([cam_center(pred_cam(k)[1]) for k in range(n_cams)])
        C_gt_all   = np.stack([-gt_Exts[k][:3,:3].T @ gt_Exts[k][:3,3] for k in range(n_cams)])
        R_T, t_T   = align_gt_to_vggt(C_pred_all, C_gt_all)

        # Residual after alignment
        C_gt_aligned = (R_T @ C_gt_all.T).T + t_T
        align_res = np.linalg.norm(C_pred_all - C_gt_aligned, axis=1)
        print(f"\nProcrustes alignment residual (per cam): "
              f"{np.array2string(align_res*100, precision=1)} cm  "
              f"mean={align_res.mean()*100:.1f} cm")

        # Per-camera errors
        print(f"\n{'Cam':<8} {'Rot err (°)':>12} {'Trans err (m)':>14}")
        print("-" * 38)
        for k in range(n_cams):
            K_p, E_p = pred_cam(k)
            E_gt_vggt = transform_ext_to_vggt(
                np.hstack([gt_Exts[k], np.zeros((3,1))]), R_T, t_T
            )
            angle_err = rot_angle_deg(E_p[:3,:3], E_gt_vggt[:3,:3])
            C_p = cam_center(E_p)
            C_g = cam_center(E_gt_vggt)
            trans_err = np.linalg.norm(C_p - C_g)
            print(f"  cam_{k:02d}   {angle_err:>10.2f}°   {trans_err:>12.3f} m")

    # ── Sapiens kps ───────────────────────────────────────────────────
    def load_kps(k):
        f = placer._cam_dirs[k] / f"sapiens_centered_kps_person_{args.pid}.npz"
        if not f.exists(): return None
        d = np.load(f); fi = d["frame_indices"]
        idx = np.where(fi == global_t)[0]
        return d["keypoints"][idx[0]] if len(idx) else None

    kps_a = load_kps(args.cam_a)
    kps_b = load_kps(args.cam_b)
    if kps_a is None or kps_b is None:
        print("ERROR: sapiens kps missing"); return

    oc_a = placer.original_coords[vggt_t, args.cam_a]
    os_a = placer.original_size[vggt_t, args.cam_a]
    oc_b = placer.original_coords[vggt_t, args.cam_b]
    os_b = placer.original_size[vggt_t, args.cam_b]

    def to_vggt(xy, oc, os_):
        u, v = placer._orig_to_vggt(xy, oc, float(os_[0]), float(os_[1]))
        return float(u), float(v)

    # ── GT 3D joints ─────────────────────────────────────────────────
    gt_body_data, _ = _load_gt(scene_name, args.rich_root, args.body_split)
    params = gt_body_data.get(args.gt_pid, {}).get(global_t)
    if params is None:
        print(f"ERROR: GT frame {global_t} pid {args.gt_pid} not found"); return
    J_gt_rich = (placer._smplx_fk(params["betas"][None], params["body_pose"][None],
                                   params["global_orient"][None])[0]
                 + params["transl"])  # (55,3) in RICH frame
    # GT 3D joints in VGGT frame (if alignment available)
    if R_T is not None:
        J_gt_vggt = (R_T @ J_gt_rich.T).T + t_T
    else:
        J_gt_vggt = None

    # ── Figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"{scene_name} | frame {global_t} | "
                 f"cam_{args.cam_a:02d} × cam_{args.cam_b:02d}  "
                 f"[blue=VGGT pred, green=GT aligned]",
                 fontsize=11, fontweight="bold")

    print(f"\n{'Joint':<13} {'pred tri err':>14} {'gt tri err':>12}")
    print("-" * 42)

    for idx, (jname, jinfo) in enumerate(_JOINTS.items()):
        gj = jinfo["goliath"]; sj = jinfo["smplx"]
        ax = fig.add_subplot(2, 2, idx+1, projection="3d")
        ax.set_title(jname, fontsize=10)

        K_a, E_a = pred_cam(args.cam_a)
        K_b, E_b = pred_cam(args.cam_b)
        C_a, C_b = cam_center(E_a), cam_center(E_b)

        # VGGT pixel coords for sapiens kps
        u_a, v_a = to_vggt(kps_a[gj,:2], oc_a, os_a)
        u_b, v_b = to_vggt(kps_b[gj,:2], oc_b, os_b)

        # Predicted rays + triangulation
        d_pred_a = pixel_ray_world(u_a, v_a, K_a, E_a)
        d_pred_b = pixel_ray_world(u_b, v_b, K_b, E_b)
        tri_pred  = dlt_2([(u_a,v_a),(u_b,v_b)], [K_a@E_a[:3], K_b@E_b[:3]])

        draw_ray(ax, C_a, tri_pred, "#3a7fd5", "-",  1.5, f"pred cam_{args.cam_a:02d}")
        draw_ray(ax, C_b, tri_pred, "#5aafff", "-",  1.5, f"pred cam_{args.cam_b:02d}")
        ax.scatter(*C_a, marker="^", s=120, c="#3a7fd5", zorder=5)
        ax.scatter(*C_b, marker="^", s=120, c="#5aafff", zorder=5)

        # GT rays + triangulation (in VGGT frame)
        gt_tri_err_str = "N/A"
        if has_gt_cam and R_T is not None:
            E_gt_a = transform_ext_to_vggt(
                np.hstack([gt_Exts[args.cam_a], np.zeros((3,1))]), R_T, t_T)
            E_gt_b = transform_ext_to_vggt(
                np.hstack([gt_Exts[args.cam_b], np.zeros((3,1))]), R_T, t_T)
            # Use VGGT intrinsics + VGGT pixel coords with GT extrinsics
            # (GT intrinsics are full-res; sapiens kps are in centered_train space)
            C_gt_a = cam_center(E_gt_a); C_gt_b = cam_center(E_gt_b)
            d_gt_a = pixel_ray_world(u_a, v_a, K_a, E_gt_a)
            d_gt_b = pixel_ray_world(u_b, v_b, K_b, E_gt_b)
            tri_gt = dlt_2([(u_a,v_a),(u_b,v_b)],
                           [K_a@E_gt_a[:3], K_b@E_gt_b[:3]])
            # GT rays reach the GT 3D point
            gt_endpoint = J_gt_vggt[sj] if J_gt_vggt is not None else tri_gt
            draw_ray(ax, C_gt_a, gt_endpoint, "#27ae60", "-", 1.5, f"GT cam_{args.cam_a:02d}")
            draw_ray(ax, C_gt_b, gt_endpoint, "#82e0aa", "-", 1.5, f"GT cam_{args.cam_b:02d}")
            ax.scatter(*C_gt_a, marker="s", s=100, c="#27ae60", zorder=5)
            ax.scatter(*C_gt_b, marker="s", s=100, c="#82e0aa", zorder=5)
            ax.scatter(*tri_gt, marker="*", s=200, c="#27ae60", zorder=6,
                       label=f"GT-cam tri")
            if J_gt_vggt is not None:
                gt_pt = J_gt_vggt[sj]
                gt_tri_err = np.linalg.norm(tri_gt - gt_pt)*100
                gt_tri_err_str = f"{gt_tri_err:.1f}cm"

        # GT 3D point
        if J_gt_vggt is not None:
            gt_pt = J_gt_vggt[sj]
            pred_tri_err = np.linalg.norm(tri_pred - gt_pt)*100
            ax.scatter(*gt_pt, marker="*", s=300, c="#e05252", zorder=7, label="GT 3D")
        else:
            pred_tri_err = float("nan")

        ax.scatter(*tri_pred, marker="*", s=200, c="#3a7fd5", zorder=6,
                   label=f"pred-cam tri  ({pred_tri_err:.1f}cm)")

        print(f"  {jname:<11}  pred={pred_tri_err:>6.1f}cm   gt-cam={gt_tri_err_str:>8}")

        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Z", fontsize=7)
        ax.set_zlabel("-Y", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.view_init(elev=10, azim=-60)

    plt.tight_layout()
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
