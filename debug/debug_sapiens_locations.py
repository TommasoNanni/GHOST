"""Debug script: compare Sapiens 2D keypoints vs GT/fused SMPL-X joint projections.

Panel 1: image + 4 Sapiens kps (green) + GT shoulder/hip joint projections (red X)
Panel 2: image + full SMPL-X mesh projected (red) — GT pose or fused pose

Modes:
  Default (centered_train images):
    - K is back-calculated from XML + crop formula
    - Sapiens kps displayed

  --full_size (full-resolution BMP images):
    - K_xml used directly (scaled by W_full/4012 for sensor-width mismatch)
    - No Sapiens kps (they are in centered-image coordinates)

  --fusion_output path/to/BBQ_001_guitar.npz:
    - Panel 2 shows fused pose mesh instead of GT PKL pose
    - Panel 1 still uses GT PKL joints for reference

Usage:
    pixi run python debug/debug_sapiens_locations.py \\
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \\
        --rich_root /capstor/scratch/cscs/tnanni/datasets/rich \\
        --smplx_model body_models/SMPLX_MALE.pkl \\
        --frame 5 --pid 1 --gt_pid 1 --cam 1 \\
        [--full_size] [--fusion_output fusion_outputs/BBQ_001_guitar.npz] \\
        [--out renders/debug/debug.png]
"""
from __future__ import annotations
import argparse, pickle, re, sys, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fusion.placer import BodyPlacer, _6d_to_aa_batch
from utilities.rich_gender_plugin import resolve_smplx_models

_HIGHLIGHT = {
    "l_shoulder": {"goliath": 5,  "smplx": 16},
    "r_shoulder": {"goliath": 6,  "smplx": 17},
    "l_hip":      {"goliath": 9,  "smplx": 1},
    "r_hip":      {"goliath": 10, "smplx": 2},
}

# Resolution at which RICH GT intrinsics were calibrated.
_CALIB_W, _CALIB_H = 4012, 3008
_VGGT_W,  _VGGT_H  = 672,  448


def _scene_location(scene_name):
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def load_gt_cameras(rich_root, scene_name, n_cams):
    """Return (Ks, Exts): Ks[i] at _CALIB_W×_CALIB_H; Exts[i] (3,4) metres."""
    loc = _scene_location(scene_name)
    calib_dir = Path(rich_root) / "scan_calibration" / loc / "calibration"
    Ks, Exts = [], []
    for xml_path in sorted(calib_dir.glob("*.xml"))[:n_cams]:
        root = ET.parse(xml_path).getroot()
        K = np.array(list(map(float, root.find("Intrinsics").find("data").text.split()))).reshape(3,3)
        E = np.array(list(map(float, root.find("CameraMatrix").find("data").text.split()))).reshape(3,4)
        Ks.append(K); Exts.append(E)
    return Ks, Exts


def project_point(X_world, K, ext):
    X_cam = ext[:3,:3] @ X_world + ext[:3,3]
    if X_cam[2] <= 0:
        return None
    uv = K @ X_cam
    return float(uv[0]/uv[2]), float(uv[1]/uv[2])


def load_frame(rich_root, scene_name, cam_name, frame_idx, full_size=False):
    cam_suffix = cam_name.split("_")[-1]
    subdir = "full_size_sample" if full_size else "centered_train"
    cam_dir = Path(rich_root) / subdir / scene_name / cam_name
    for ext in (".bmp", ".jpg", ".jpeg", ".png"):
        p = cam_dir / f"{frame_idx:05d}_{cam_suffix}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    return None


def load_gt_body(rich_root, scene_name, body_split, gt_pid, global_t):
    """Load body params from GT PKL. Returns (betas, body_pose, orient, transl)."""
    pkl_path = (Path(rich_root) / body_split / scene_name
                / f"{global_t:05d}" / f"{gt_pid:03d}.pkl")
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    raw_betas = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
    betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)[:10] \
            if raw_betas is not None else np.zeros(10, dtype=np.float32)
    body_pose = np.asarray(d.get("body_pose", np.zeros(63)), dtype=np.float32).reshape(63)
    orient    = np.asarray(d.get("global_orient", np.zeros(3)), dtype=np.float32).squeeze()
    transl    = np.asarray(d["transl"], dtype=np.float32).squeeze()
    return betas, body_pose, orient, transl


def load_fused_body(fusion_npz, pid_slot, global_t):
    """Load fused pose from fusion_outputs NPZ.

    pose: (T, P, 55, 6)  — 6D rotations; joint 0 = global_orient, 1-21 = body,
                            22-36 = left hand, 37-51 = right hand
    body_transl_world: (T, P, 3)  — in VGGT cam_00 frame
    shape: (P, 10)

    Returns (betas, body_pose, orient, transl, lhand, rhand).
    """
    d = np.load(fusion_npz, allow_pickle=False)
    pose_6d = d["pose"][global_t, pid_slot]                  # (55, 6)
    transl  = d["body_transl_world"][global_t, pid_slot]     # (3,) — VGGT cam_00 frame
    betas   = d["shape"][pid_slot]                           # (10,)

    orient    = _6d_to_aa_batch(pose_6d[0:1]).reshape(3)      # (3,)
    body_pose = _6d_to_aa_batch(pose_6d[1:22]).reshape(63)    # (63,)
    lhand     = _6d_to_aa_batch(pose_6d[22:37]).reshape(45)   # (45,)
    rhand     = _6d_to_aa_batch(pose_6d[37:52]).reshape(45)   # (45,)
    return betas, body_pose, orient, transl, lhand, rhand


def load_vggt_camera(scene_dir, cam_name, global_t, W_img, H_img):
    """Return (K_scaled, E_metric) for fused-pose projection (VGGT cam_00 world frame).

    The VGGT translation column is in raw VGGT units; the mapanything scale converts
    it to metres so it is consistent with body_transl_world in the fusion NPZ.
    Intrinsics are scaled from VGGT output space to the original centered image size.
    """
    npz_path = scene_dir / "vggt_cameras_centered.npz"
    if not npz_path.exists():
        npz_path = scene_dir / "vggt_cameras.npz"
    vd = np.load(npz_path, allow_pickle=False)
    cam_names = [n.decode() if isinstance(n, bytes) else n for n in vd["camera_names"]]
    ci = cam_names.index(cam_name)

    E = vd["extrinsics"][global_t, ci].astype(np.float64).copy()   # (3, 4)
    K_vggt = vd["intrinsics"][global_t, ci].astype(np.float64)     # (3, 3) in VGGT output space

    # Apply metric scale to the translation column.
    scale_path = scene_dir / "mapanything_scale_centered.npy"
    if not scale_path.exists():
        scale_path = scene_dir / "mapanything_scale.npy"
    scale = float(np.load(scale_path)[global_t])
    E[:3, 3] *= scale

    # Scale intrinsics from VGGT output space to actual image resolution.
    # cx/cy from VGGT are at the VGGT output center; override with image center.
    vggt_out_w = float(K_vggt[0, 2] * 2)   # cx*2
    vggt_out_h = float(K_vggt[1, 2] * 2)   # cy*2
    K = K_vggt.copy()
    K[0, 0] *= W_img / vggt_out_w
    K[1, 1] *= H_img / vggt_out_h
    K[0, 2]  = W_img / 2.0
    K[1, 2]  = H_img / 2.0
    return K, E


def render_mesh(ax, placer, betas, body_pose, orient, transl, K, E,
                lhand=None, rhand=None):
    """Project full SMPL-X mesh and add as PolyCollection to ax."""
    _, verts_batch = placer._smplx_fk(
        betas[np.newaxis], body_pose[np.newaxis], orient[np.newaxis],
        left_hand_pose  = lhand[np.newaxis] if lhand is not None else None,
        right_hand_pose = rhand[np.newaxis] if rhand is not None else None,
        return_verts=True,
    )
    verts = verts_batch[0] + transl                              # (V, 3)
    faces = placer._smplx_model.faces.astype(np.int32)          # (F, 3)

    V_cam = (E[:3,:3] @ verts.T + E[:3,3:]).T                   # (V, 3)
    valid = V_cam[:, 2] > 0
    denom = np.where(V_cam[:,2] > 0, V_cam[:,2], 1.0)
    u = np.where(valid, K[0,0]*V_cam[:,0]/denom + K[0,2], -1)
    v = np.where(valid, K[1,1]*V_cam[:,1]/denom + K[1,2], -1)
    verts_2d = np.stack([u, v], axis=1)

    f_valid = valid[faces].all(axis=1)
    faces_ok = faces[f_valid]
    order    = np.argsort(-V_cam[faces_ok, 2].mean(axis=1))
    tris     = verts_2d[faces_ok[order]]

    coll = mc.PolyCollection(tris, facecolor=(0.87, 0.32, 0.32, 0.55),
                              edgecolor="none", zorder=3)
    ax.add_collection(coll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir",      required=True, type=Path)
    ap.add_argument("--rich_root",      required=True, type=Path)
    ap.add_argument("--smplx_model",    required=True, type=Path)
    ap.add_argument("--frame",          type=int, default=5)
    ap.add_argument("--pid",            type=int, default=1, help="Ghost person ID (Sapiens kps)")
    ap.add_argument("--gt_pid",         type=int, default=1, help="RICH GT person ID")
    ap.add_argument("--body_split",     default="train_body")
    ap.add_argument("--cam",            type=int, default=1)
    ap.add_argument("--intrinsics",     choices=["gt", "vggt"], default="gt")
    ap.add_argument("--full_size",      action="store_true",
                    help="Use full-resolution BMP images (full_size_sample/). K_xml used directly.")
    ap.add_argument("--fusion_output",  type=Path, default=None,
                    help="fusion_outputs/<scene>.npz — if given, Panel 2 shows fused pose mesh")
    ap.add_argument("--pid_slot",       type=int, default=0,
                    help="Person slot index in fusion NPZ (default 0 for single-person scenes)")
    ap.add_argument("--out",            type=Path, default=Path("renders/debug/debug_sapiens.png"))
    args = ap.parse_args()

    scene_dir  = args.scene_dir.resolve()
    scene_name = scene_dir.name
    global_t   = args.frame
    k          = args.cam

    # ── Placer ───────────────────────────────────────────────────────────
    _gender_json = _REPO_ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer   = BodyPlacer(str(scene_dir), _smplx_arg)
    n_cams   = len(placer._cam_dirs)
    cam_name = placer._cam_dirs[k].name

    # ── Load image ───────────────────────────────────────────────────────
    img = load_frame(str(args.rich_root), scene_name, cam_name, global_t,
                     full_size=args.full_size)
    if img is None:
        print(f"ERROR: image not found for {scene_name}/{cam_name} frame {global_t}"); return
    H_img, W_img = img.shape[:2]
    print(f"Image: {W_img}×{H_img}  ({'full_size' if args.full_size else 'centered_train'})")

    # ── GT cameras ───────────────────────────────────────────────────────
    Ks_xml, Exts_gt = load_gt_cameras(str(args.rich_root), scene_name, n_cams)
    K_xml = Ks_xml[k]
    E_gt  = Exts_gt[k]

    # ── Intrinsics matrix for projection ─────────────────────────────────
    if args.full_size:
        # K_xml was calibrated at _CALIB_W×_CALIB_H; BMPs may be slightly wider.
        # Scale fx/cx to match actual image width; fy/cy for height.
        K = K_xml.copy()
        K[0] *= W_img / _CALIB_W
        K[1] *= H_img / _CALIB_H
        print(f"[Full-size K] fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    elif args.intrinsics == "vggt":
        vd = np.load(scene_dir / "vggt_cameras_centered.npz")
        cam_names = [n.decode() for n in vd["camera_names"]]
        ci = cam_names.index(cam_name)
        K_vggt = vd["intrinsics"][global_t, ci].copy()
        K = K_vggt.copy()
        K[0,0] *= W_img / _VGGT_W
        K[1,1] *= H_img / _VGGT_H
        K[0,2] = W_img / 2.0
        K[1,2] = H_img / 2.0
        print(f"[VGGT K] fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    else:
        cx_xml, cy_xml = K_xml[0,2], K_xml[1,2]
        src_w = _CALIB_W*(W_img/2) / (_CALIB_W - cx_xml) if cx_xml > _CALIB_W/2 \
                else _CALIB_W*(W_img/2) / cx_xml
        src_h = _CALIB_H*(H_img/2) / cy_xml if cy_xml < _CALIB_H/2 \
                else _CALIB_H*(H_img/2) / (_CALIB_H - cy_xml)
        K = K_xml.copy()
        K[0] *= src_w / _CALIB_W
        K[1] *= src_h / _CALIB_H
        K[0,2] = W_img / 2.0
        K[1,2] = H_img / 2.0
        print(f"[GT centered K] fx={K[0,0]:.1f} fy={K[1,1]:.1f}  src={src_w:.0f}×{src_h:.0f}")

    # ── GT body (for Panel 1 joint projections) ───────────────────────────
    gt_body = load_gt_body(str(args.rich_root), scene_name, args.body_split,
                           args.gt_pid, global_t)
    if gt_body is None:
        print(f"ERROR: GT pkl not found"); return
    gt_betas, gt_body_pose, gt_orient, gt_transl = gt_body

    J_gt, _ = placer._smplx_fk(
        gt_betas[np.newaxis], gt_body_pose[np.newaxis], gt_orient[np.newaxis],
        return_verts=True,
    )
    J_gt = J_gt[0] + gt_transl   # (55, 3) in RICH world frame

    # ── Fused body (for Panel 2 mesh) ─────────────────────────────────────
    # Fused transl is in VGGT cam_00 frame; use VGGT extrinsics for projection.
    p2_lhand, p2_rhand = None, None
    if args.fusion_output is not None:
        fused_betas, fused_body_pose, fused_orient, fused_transl, p2_lhand, p2_rhand = \
            load_fused_body(str(args.fusion_output), args.pid_slot, global_t)
        K_fused, E_fused = load_vggt_camera(scene_dir, cam_name, global_t, W_img, H_img)
        p2_label = "Fused SMPL-X mesh (VGGT cams)"
        p2_betas, p2_body_pose, p2_orient, p2_transl = fused_betas, fused_body_pose, fused_orient, fused_transl
        p2_K, p2_E = K_fused, E_fused
    else:
        p2_label = "GT SMPL-X mesh (GT cams)"
        p2_betas, p2_body_pose, p2_orient, p2_transl = gt_betas, gt_body_pose, gt_orient, gt_transl
        p2_K, p2_E = K, E_gt

    # ── Sapiens kps (centered_train mode only) ────────────────────────────
    kps = None
    if not args.full_size:
        sap_path = placer._cam_dirs[k] / f"sapiens_centered_kps_person_{args.pid}.npz"
        if sap_path.exists():
            sd  = np.load(sap_path)
            fi  = sd["frame_indices"]
            idx = np.where(fi == global_t)[0]
            if len(idx):
                kps = sd["keypoints"][idx[0]]   # (308, 3)

    def proj(X):
        return project_point(X, K, E_gt)

    # ── Figure ────────────────────────────────────────────────────────────
    mode_tag = "full-size" if args.full_size else f"centered/{args.intrinsics}-K"
    pose_tag = "fused" if args.fusion_output else "GT"
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"{scene_name} | frame {global_t} | {cam_name} | {mode_tag} | pose={pose_tag}",
                 fontsize=11, fontweight="bold")
    for ax in axes:
        ax.imshow(img); ax.set_axis_off()

    # Panel 1 — GT joint projections + (optionally) Sapiens kps
    ax1 = axes[0]
    ax1.set_title("GT joints (✕) " + ("+ Sapiens kps (●)" if kps is not None else "(no Sapiens in full-size mode)"),
                  fontsize=9)
    for jname, jinfo in _HIGHLIGHT.items():
        sj = jinfo["smplx"]
        pt = proj(J_gt[sj])
        if pt is not None:
            ax1.plot(pt[0], pt[1], "x", color="#e05252", ms=12, mew=2.5, zorder=4)
            ax1.text(pt[0]+8, pt[1]-12, jname, fontsize=6, color="#e05252", zorder=5)

        if kps is not None:
            gj = jinfo["goliath"]
            sx, sy, sc = float(kps[gj,0]), float(kps[gj,1]), float(kps[gj,2])
            ax1.plot(sx, sy, "o", color="#27ae60", ms=8, mew=1.5,
                     markeredgecolor="white", zorder=4)
            ax1.text(sx+8, sy, f"{sc:.2f}", fontsize=6, color="#27ae60", zorder=5)
            if pt is not None:
                ax1.plot([sx, pt[0]], [sy, pt[1]], "-", color="yellow", lw=0.8, alpha=0.6, zorder=3)

    # Panel 2 — mesh
    ax2 = axes[1]
    ax2.set_title(p2_label, fontsize=9)
    render_mesh(ax2, placer, p2_betas, p2_body_pose, p2_orient, p2_transl, p2_K, p2_E,
                lhand=p2_lhand, rhand=p2_rhand)

    plt.tight_layout()
    plt.savefig(str(args.out), dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
