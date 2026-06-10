"""
Silhouette-based body fitting to cam_00 (VGGT reference camera, [I|0]).

For a single frame (or all frames), optimises global_orient + transl so that
the convex hull of the projected SMPL-X mesh overlaps the SAM3 mask in cam_00.

cam_00 has zero extrinsic error (it is the VGGT reference), so any remaining
error comes only from intrinsics and depth ambiguity.

Usage:
    pixi run python approaches/test_silhouette_fitting.py \
        --scene_dir /iopsstor/scratch/cscs/tnanni/ghost_outputs/rich_train/BBQ_001_guitar \
        --rich_root  /capstor/scratch/cscs/tnanni/datasets/rich \
        --smplx_model body_models/SMPLX_NEUTRAL.pkl \
        [--frame_idx 10] [--person_id 1] [--max_iters 300] [--all_frames]
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SciR

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fusion.placer import BodyPlacer
from utilities.rich_gender_plugin import resolve_smplx_models

_RICH_ORIG_W = 4112
_RICH_ORIG_H = 3008


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_cam_body_data(cam_dir: Path, pid: int) -> dict | None:
    path = cam_dir / "body_data" / f"person_{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    required = {"smplx_body_pose", "smplx_global_orient", "smplx_betas",
                "smplx_transl", "focal_length", "frame_indices"}
    if not required.issubset(d.files):
        return None
    frames = d["frame_indices"].astype(int)
    result: dict[int, dict] = {}
    for local_t, global_t in enumerate(frames):
        result[int(global_t)] = {
            "body_pose":     d["smplx_body_pose"][local_t],
            "global_orient": d["smplx_global_orient"][local_t],
            "betas":         d["smplx_betas"][local_t],
            "smplx_transl":  d["smplx_transl"][local_t],
            "focal_length":  float(d["focal_length"][local_t]),
        }
    return result


def load_mask(cam_dir: Path, frame_idx: int, person_id: int) -> np.ndarray | None:
    mask_path = cam_dir / "mask_data.npz"
    if not mask_path.exists():
        return None
    d = np.load(mask_path, allow_pickle=True)
    for pid_key in [person_id - 1, person_id]:
        key = f"mask_{frame_idx:05d}_{pid_key:02d}"
        if key in d.files:
            arr = np.asarray(d[key])
            if arr.max() > 1:
                return (arr == person_id).astype(np.uint8)
            return arr.astype(np.uint8)
    print(f"Key not found for frame={frame_idx} person={person_id}. "
          f"Unexpected mask_data keys: {list(d.files)}")
    return None


def _scene_to_location(scene_name: str) -> str:
    m = re.match(r"^(.+?)_\d{3}_", scene_name)
    return m.group(1) if m else scene_name


def load_gt_intrinsics_for_cam(scene_name: str, rich_root: Path,
                                cam_name: str) -> np.ndarray | None:
    calib_dir = rich_root / "scan_calibration" / _scene_to_location(scene_name) / "calibration"
    if not calib_dir.is_dir():
        return None
    m_ref = re.search(r"\d+", cam_name)
    ref_idx = int(m_ref.group()) if m_ref else 0
    xml_paths = sorted(calib_dir.glob("*.xml"))
    if ref_idx >= len(xml_paths):
        return None
    intr_node = ET.parse(xml_paths[ref_idx]).getroot().find("Intrinsics")
    if intr_node is None:
        return None
    vals = list(map(float, intr_node.find("data").text.split()))
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def load_gt_for_frame(rich_root: Path, split: str, scene_name: str,
                      frame_idx: int, placer: BodyPlacer) -> dict | None:
    """Return dict with 'pelvis' (3,) and 'global_orient' (3,), or None."""
    gt_dir = rich_root / split / scene_name
    if not gt_dir.is_dir():
        return None
    for frame_dir in gt_dir.iterdir():
        try:
            if int(frame_dir.name) != frame_idx:
                continue
            pkl_path = next(frame_dir.glob("*.pkl"), None)
            if pkl_path is None:
                return None
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            transl = np.array(d["transl"], dtype=np.float32).reshape(3)
            raw = d.get("betas") if d.get("betas") is not None else d.get("smplx_betas")
            betas_gt = np.array(raw, dtype=np.float32).reshape(1, -1)[:, :10]
            J0_gt = placer._smplx_fk(betas_gt, np.zeros((1, 63)), np.zeros((1, 3)))[0, 0]
            pelvis_gt = (transl + J0_gt).astype(np.float32)
            gt_orient = np.array(d["global_orient"], dtype=np.float32).reshape(3)
            return {"pelvis": pelvis_gt, "global_orient": gt_orient}
        except (ValueError, StopIteration):
            continue
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def apply_rigid(V_can: np.ndarray, J0: np.ndarray,
                rotvec: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """V_world = R @ (V_can - J0) + J0 + transl  (SMPL-X convention)."""
    R = SciR.from_rotvec(rotvec).as_matrix()
    return (R @ (V_can - J0).T).T + J0 + transl


def project_to_image(V_world: np.ndarray, R_cam: np.ndarray,
                     t_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,2) pixel coords and (N,) boolean front-visible mask."""
    V_cam = (R_cam @ V_world.T).T + t_cam
    front = V_cam[:, 2] > 0.1
    uv = np.full((len(V_world), 2), np.nan)
    x = V_cam[front, 0] / V_cam[front, 2]
    y = V_cam[front, 1] / V_cam[front, 2]
    uv[front, 0] = x * K[0, 0] + K[0, 2]
    uv[front, 1] = y * K[1, 1] + K[1, 2]
    return uv, front


def silhouette_iou_convexhull(uv: np.ndarray, front: np.ndarray,
                               gt_mask: np.ndarray) -> float:
    """Convex hull of projected vertices as silhouette proxy. Returns IoU."""
    pts = uv[front]
    H, W = gt_mask.shape
    in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    pts = pts[in_bounds].astype(np.int32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    pred = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(pred, hull, 1)
    inter = int((pred & gt_mask).sum())
    union = int((pred | gt_mask).sum())
    return inter / (union + 1e-6)


def orient_error_deg(rotvec_pred: np.ndarray, rotvec_gt: np.ndarray) -> float:
    R_pred = SciR.from_rotvec(rotvec_pred.astype(np.float64)).as_matrix()
    R_gt   = SciR.from_rotvec(rotvec_gt.astype(np.float64)).as_matrix()
    cos_val = np.clip(0.5 * (np.trace(R_pred.T @ R_gt) - 1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ---------------------------------------------------------------------------
# Per-frame fitting
# ---------------------------------------------------------------------------

def fit_frame(frame_idx: int, fd: dict, gt_mask: np.ndarray,
              V_can: np.ndarray, J0: np.ndarray,
              R_cam: np.ndarray, t_cam: np.ndarray, K: np.ndarray,
              max_iters: int, img_H: int, img_W: int) -> dict:
    """Run Powell silhouette optimisation for one frame. Returns result dict."""
    g_orient_init = fd["global_orient"].astype(np.float64)
    transl_init   = fd["smplx_transl"].astype(np.float64)

    if gt_mask.shape != (img_H, img_W):
        gt_mask = cv2.resize(gt_mask, (img_W, img_H), interpolation=cv2.INTER_NEAREST)

    def objective(params):
        rotvec = params[:3]
        transl = params[3:6]
        V_world = apply_rigid(V_can, J0, rotvec, transl)
        uv, front = project_to_image(V_world, R_cam, t_cam, K)
        return 1.0 - silhouette_iou_convexhull(uv, front, gt_mask)

    x0 = np.concatenate([g_orient_init, transl_init])
    iou_init = 1.0 - objective(x0)

    t0 = time.time()
    res = minimize(objective, x0, method="Powell",
                   options={"maxiter": max_iters, "disp": False})
    elapsed = time.time() - t0

    orient_opt = res.x[:3]
    transl_opt  = res.x[3:6]
    iou_opt     = 1.0 - res.fun

    return {
        "iou_init":    iou_init,
        "iou_opt":     iou_opt,
        "orient_init": g_orient_init,
        "transl_init": transl_init,
        "orient_opt":  orient_opt,
        "transl_opt":  transl_opt,
        "elapsed":     elapsed,
        "nfev":        res.nfev,
        "converged":   res.success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   required=True)
    p.add_argument("--rich_root",   required=True)
    p.add_argument("--smplx_model", required=True)
    p.add_argument("--split",       default="train_body")
    p.add_argument("--frame_idx",   type=int, default=10)
    p.add_argument("--person_id",   type=int, default=1)
    p.add_argument("--max_iters",   type=int, default=300)
    p.add_argument("--all_frames",  action="store_true",
                   help="Run on all available frames and print summary table")
    args = p.parse_args()

    scene_dir  = Path(args.scene_dir)
    rich_root  = Path(args.rich_root)
    scene_name = scene_dir.name

    _gender_json = _ROOT / "resource" / "rich_gender.json"
    _smplx_arg = (
        resolve_smplx_models(scene_dir.name, Path(args.smplx_model).parent, _gender_json)
        if _gender_json.exists() else args.smplx_model
    )
    placer = BodyPlacer(scene_dir, _smplx_arg)

    # ── Reference camera (cam_00, VGGT [I|0]) ──────────────────────────────
    cam_dirs = sorted(d for d in scene_dir.iterdir()
                      if d.is_dir() and re.search(r"cam", d.name, re.I))
    if not cam_dirs:
        print("No camera directories found"); return
    cam0_dir  = cam_dirs[0]
    cam0_name = cam0_dir.name
    print(f"Reference camera dir: {cam0_name}")

    # ── Camera intrinsics (VGGT) ────────────────────────────────────────────
    K_cam0 = placer.intrinsics[0, 0].astype(np.float64)  # (3,3) at t=0
    img_W  = int(round(float(K_cam0[0, 2]) * 2))
    img_H  = int(round(float(K_cam0[1, 2]) * 2))
    print(f"Image: {img_W}×{img_H}  K diag: fx={K_cam0[0,0]:.1f} fy={K_cam0[1,1]:.1f} "
          f"cx={K_cam0[0,2]:.1f} cy={K_cam0[1,2]:.1f}")

    # cam_00 is the VGGT reference → identity extrinsics
    R_cam = np.eye(3, dtype=np.float64)
    t_cam = np.zeros(3, dtype=np.float64)

    # ── Load body data ───────────────────────────────────────────────────────
    body_data = load_cam_body_data(cam0_dir, args.person_id)
    if body_data is None:
        print(f"No body data for person {args.person_id} in {cam0_name}"); return

    # ── Load SMPL-X model (vertices computed per-frame since body_pose varies) ─
    smplx_model = smplx.create(
        args.smplx_model, model_type="smplx", ext="pkl",
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )
    smplx_model.eval()

    def build_canonical(betas_np: np.ndarray, body_pose_np: np.ndarray):
        """V_can, J0 for given betas + body_pose (zero orient, zero transl)."""
        with torch.no_grad():
            out = smplx_model(
                betas=torch.tensor(betas_np[None].astype(np.float32)),
                body_pose=torch.tensor(body_pose_np[None].astype(np.float32)),
                global_orient=torch.zeros(1, 3),
                transl=torch.zeros(1, 3),
            )
        V = out.vertices[0].numpy().astype(np.float64)
        J = out.joints[0, 0].numpy().astype(np.float64)
        return V, J

    # Print vertex count once using first frame
    _fd0 = next(iter(body_data.values()))
    _V0, _J0 = build_canonical(_fd0["betas"], _fd0["body_pose"])
    print(f"Mesh: {len(_V0)} vertices")

    # ── Select frames to process ─────────────────────────────────────────────
    if args.all_frames:
        frame_list = sorted(body_data.keys())
    else:
        fi = args.frame_idx
        if fi not in body_data:
            fi = sorted(body_data.keys())[10]
            print(f"frame {args.frame_idx} not found → using {fi}")
        frame_list = [fi]

    # ── Per-frame loop ───────────────────────────────────────────────────────
    rows_table = []

    for frame_idx in frame_list:
        fd = body_data[frame_idx]

        gt_mask = load_mask(cam0_dir, frame_idx, args.person_id)
        if gt_mask is None:
            continue

        V_can, J0 = build_canonical(fd["betas"], fd["body_pose"])
        result = fit_frame(
            frame_idx, fd, gt_mask, V_can, J0,
            R_cam, t_cam, K_cam0, args.max_iters, img_H, img_W,
        )

        gt = load_gt_for_frame(rich_root, args.split, scene_name, frame_idx, placer)

        if not args.all_frames:
            # ── Verbose single-frame output ──────────────────────────────────
            def _f3(v): return f"[{float(v[0]):+.3f} {float(v[1]):+.3f} {float(v[2]):+.3f}]"

            print(f"\n{'─'*60}")
            print(f"Initial  IoU : {result['iou_init']:.4f}  (loss={1-result['iou_init']:.4f})")
            print(f"  orient : {result['orient_init']}")
            print(f"  transl : {result['transl_init']}")
            print(f"\nRunning Powell ({args.max_iters} max iters) …")
            print(f"Optimised IoU: {result['iou_opt']:.4f}  (loss={1-result['iou_opt']:.4f})")
            print(f"  orient : {result['orient_opt']}")
            print(f"  transl : {result['transl_opt']}")
            print(f"  evals={result['nfev']}  time={result['elapsed']:.1f}s  "
                  f"converged={result['converged']}")

            if gt is not None:
                p_init = (result["transl_init"] + J0).astype(np.float32)
                p_opt  = (result["transl_opt"]  + J0).astype(np.float32)
                err_i  = float(np.linalg.norm(p_init - gt["pelvis"]))
                err_o  = float(np.linalg.norm(p_opt  - gt["pelvis"]))

                print(f"\n{'─'*60}")
                print(f"GT   pelvis : {_f3(gt['pelvis'])}")
                print(f"Init pelvis : {_f3(p_init)}  err={err_i*100:.1f} cm")
                print(f"Opt  pelvis : {_f3(p_opt)}   err={err_o*100:.1f} cm")
                print(f"Delta err   : {(err_o - err_i)*100:+.1f} cm  "
                      f"({'better' if err_o < err_i else 'worse'})")

                print(f"\nPer-axis |error| (cm):")
                for ax, ei, eo in zip("XYZ",
                                       np.abs(p_init - gt["pelvis"]) * 100,
                                       np.abs(p_opt  - gt["pelvis"]) * 100):
                    print(f"  {ax}: init={ei:.1f}  opt={eo:.1f}  delta={eo-ei:+.1f}")

                o_err = orient_error_deg(result["orient_opt"], gt["global_orient"])
                o_err_init = orient_error_deg(result["orient_init"], gt["global_orient"])
                print(f"\nOrientation error:  init={o_err_init:.2f}°  opt={o_err:.2f}°")
            print(f"{'─'*60}\n")

        else:
            # ── Collect row for summary table ────────────────────────────────
            if gt is not None:
                p_opt = (result["transl_opt"] + J0).astype(np.float32)
                err_m = float(np.linalg.norm(p_opt - gt["pelvis"]))
                o_err = orient_error_deg(result["orient_opt"], gt["global_orient"])
            else:
                err_m = float("nan")
                o_err = float("nan")
            rows_table.append((frame_idx, err_m, o_err))

    if args.all_frames and rows_table:
        w = 60
        hdr = f"{'frame':>8}  {'err_m':>8}  {'orient°':>8}"
        print(f"\n{'─'*w}\n{hdr}\n{'─'*w}")
        for fi, em, oe in rows_table:
            print(f"{fi:>8}  {em:>8.4f}  {oe:>8.2f}")
        errs   = [r[1] for r in rows_table if not np.isnan(r[1])]
        orients = [r[2] for r in rows_table if not np.isnan(r[2])]
        if errs:
            print(f"{'─'*w}")
            print(f"transl: median={np.median(errs):.4f}m  mean={np.mean(errs):.4f}m")
        if orients:
            print(f"orient: median={np.median(orients):.2f}°  mean={np.mean(orients):.2f}°")


if __name__ == "__main__":
    main()
